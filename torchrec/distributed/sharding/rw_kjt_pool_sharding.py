#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torchrec.distributed.comm import (
    get_local_rank,
    get_local_size,
    intra_and_cross_node_pg,
)
from torchrec.distributed.dist_data import JaggedTensorAllToAll
from torchrec.distributed.sharding.rw_pool_sharding import (
    InferRwObjectPoolInputDist,
    RwObjectPoolIDsDist,
)
from torchrec.distributed.tensor_sharding import (
    InferObjectPoolSharding,
    ObjectPoolReplicatedRwShardingContext,
    ObjectPoolRwShardingContext,
    ObjectPoolSharding,
)
from torchrec.distributed.types import Awaitable, ShardingEnv
from torchrec.modules.object_pool_lookups import KeyedJaggedTensorPoolLookup
from torchrec.modules.utils import jagged_index_select_with_empty
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class RwKeyedJaggedTensorPoolLookupValuesDist(torch.nn.Module):
    """
    Module to distribute KeyedJaggedTensor to all ranks after local pool lookup

    Args:
        num_features (int): number of features in KeyedJaggedTensor to be distributed
        env (ShardingEnv): Sharding environment with info such as rank and world size

    Example:
        dist = RwKeyedJaggedTensorPoolLookupValuesDist(num_features=2, env=env)

        # on rank 0, sends 1 and receives 2 batches
        ctx = ObjectPoolRwShardingContext(num_ids_each_rank_to_send=1, num_ids_each_rank_to_receive=2)
        jt = JaggedTensor(values=[2,3,2], lengths=[2,1])

        # on rank 1, sends 2 and receives 1 batches
        ctx = ObjectPoolRwShardingContext(num_ids_each_rank_to_send=2, num_ids_each_rank_to_receive=1)
        jt = JaggedTensor(values=[1,1,5,2,8], lengths=[2,2,1])

        rank0_out = dist(ctx, jt).wait()

        # rank0_out is
        # JaggedTensor(values=[2,3,1,1,5,2], lengths=[2,2,2])

        rank1_out = dist(ctx, jt).wait()
        # rank1_out is
        # JaggedTensor(values=[2,8], lengths=[1,1])

    """

    def __init__(
        self,
        num_features: int,
        env: ShardingEnv,
    ) -> None:
        super().__init__()
        self._sharding_env = env
        self._num_features = num_features

    def forward(
        self,
        ctx: ObjectPoolRwShardingContext,
        jagged_tensor: JaggedTensor,
    ) -> Awaitable[JaggedTensor]:
        """
        Sends JaggedTensor to relevant `ProcessGroup` ranks.

        Args:
            ctx (ObjectPoolRwShardingContext): Context for RW sharding, containing
                number of items to send and receive from each rank.
            jagged_tensor (JaggedTensor): JaggedTensor to distribute. This JT is
                constructed from flattening a KeyedJaggedTensor.

        Returns:
            Awaitable[JaggedTensor]: awaitable of `JaggedTensor`
        """
        return JaggedTensorAllToAll(
            jt=jagged_tensor,
            # pyre-ignore
            num_items_to_send=ctx.num_ids_each_rank_to_send * self._num_features,
            # pyre-ignore
            num_items_to_receive=ctx.num_ids_each_rank_to_receive * self._num_features,
            # pyre-ignore
            pg=self._sharding_env.process_group,
        )


class RwKeyedJaggedTensorPoolUpdateValuesDist(torch.nn.Module):
    """
    Module to distribute updated KeyedJaggedTensor to all ranks after local pool update

    Args:
        num_features (int): number of features in KeyedJaggedTensor to be distributed
        env (ShardingEnv): Sharding environment with info such as rank and world size
        device (torch.device): Device on which to allocate tensors
        num_replicas (int): number of times KJT should be replicated across ranks in case
            of replicated row-wise sharding. Defaults to 1 (no replica).

    Example:
        keys=['A','B']
        dist = RwKeyedJaggedTensorPoolUpdateValuesDist(num_features=len(keys), env=env)
        ctx = ObjectPoolRwShardingContext(
            num_ids_each_rank_to_send=1,
            num_ids_each_rank_to_receive=1,
        )
        awaitable = dist(rank0_input, ctx)

        # where:
        # rank0_input is KeyedJaggedTensor holding

        #         0           1
        # 'A'    [A.V0]       None
        # 'B'    None         [B.V0]

        # rank1_input is KeyedJaggedTensor holding

        #         0           1
        # 'A'     [A.V3]      [A.V4]
        # 'B'     None        [B.V2]

        rank0_output = awaitable.wait()

        # where:
        # rank0_output is JaggedTensor holding

        values = [A.V0, A.V3]
        lengths = [1,0,1,0]

        # rank1_output is JaggedTensor holding

        values = [B.V0, A.V4, B.V2]
        lengths = [0,1,1,1]
    """

    def __init__(
        self,
        num_features: int,
        env: ShardingEnv,
        device: torch.device,
        num_replicas: int = 1,
    ) -> None:
        super().__init__()
        self._env = env
        self._num_features = num_features
        self._num_replicas = num_replicas
        self._device = device

    def forward(
        self,
        values: KeyedJaggedTensor,
        ctx: ObjectPoolRwShardingContext,
    ) -> Awaitable[JaggedTensor]:
        """
        Sends tensor to relevant `ProcessGroup` ranks.

        Args:
            values (KeyedJaggedTensor): KJT to distribute
            ctx (ObjectPoolRwShardingContext): Context for RW sharding, containing
                indices along batch dimension to permute KJT before A2A, as well as
                number of items to send and receive from each rank.

        Returns:
            Awaitable[JaggedTensor]: awaitable of `JaggedTensor` from which KJT can be
            reconstructed.

        """

        kjt = values
        permute_idx = ctx.unbucketize_permute

        # Below code lets us select values out from a KJT in a row manner format for example
        # KJT
        # f1 [0,1] [2,3]
        # f2 [3,4,5] [6]
        # the values come in as 0,1,2,3,4,5,6, however, we need it in feature order e.g.
        # 0,1,3,4,5,2,3,6 to more efficiently to the all to alls
        # we can use jagged index select to these, e.g. we need the indices to come in order of
        # 0,2,1,3.
        # this can be done by first chunking viewing as [[0,1][2,3]],
        # then taking a transpose and flatten => [0,2,1,3]

        arange_idx = torch.arange(
            kjt.stride() * self._num_features, device=self._device
        )
        jagged_idx = arange_idx.view(self._num_features, -1).t()
        jt_lengths_in_order_for_a2a = jagged_idx[permute_idx].flatten()

        lengths_to_send = kjt.lengths()[jt_lengths_in_order_for_a2a]
        kjt_values_to_send_offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
            lengths_to_send
        )
        kjt_values_to_send = jagged_index_select_with_empty(
            kjt.values().unsqueeze(-1),
            jt_lengths_in_order_for_a2a,
            kjt.offsets()[1:],
            kjt_values_to_send_offsets,
        )
        kjt_values_to_send = kjt_values_to_send.flatten()

        kjt_weights_to_send = None
        if kjt.weights_or_none() is not None:
            kjt_weights_to_send = jagged_index_select_with_empty(
                kjt.weights().unsqueeze(-1),
                jt_lengths_in_order_for_a2a,
                kjt.offsets()[1:],
                kjt_values_to_send_offsets,
            )

        if self._num_replicas > 1:
            kjt_values_to_send = kjt_values_to_send.repeat(self._num_replicas)
            lengths_to_send = lengths_to_send.flatten().repeat(self._num_replicas)
            if kjt_weights_to_send is not None:
                kjt_weights_to_send = kjt_weights_to_send.repeat(self._num_replicas)

        jt_all_to_all = JaggedTensorAllToAll(
            JaggedTensor(
                values=kjt_values_to_send,
                lengths=lengths_to_send,
                weights=kjt_weights_to_send,
            ),
            # pyre-ignore
            num_items_to_send=ctx.num_ids_each_rank_to_send * self._num_features,
            # pyre-ignore
            num_items_to_receive=ctx.num_ids_each_rank_to_receive * self._num_features,
            # pyre-ignore
            pg=self._env.process_group,
        )

        return jt_all_to_all


class KeyedJaggedTensorPoolRwSharding(ObjectPoolSharding):
    def __init__(
        self,
        env: ShardingEnv,
        device: torch.device,
        pool_size: int,
        num_features: int,
    ) -> None:
        self._env = env
        # pyre-ignore
        self._pg: dist.ProcessGroup = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        self._device = device
        self._pool_size = pool_size

        self._block_size: int = (
            pool_size + self._env.world_size - 1
        ) // self._env.world_size

        self.local_pool_size: int = (
            self._block_size
            if self._env.rank != self._env.world_size - 1
            else pool_size - self._block_size * (self._env.world_size - 1)
        )

        self._block_size_t: torch.Tensor = torch.tensor(
            [
                self._block_size,
            ],
            dtype=torch.long,
            device=self._device,
        )
        self._num_features = num_features

    def create_update_ids_dist(
        self,
    ) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(self._pg, is_update=True)

    def create_update_values_dist(
        self,
    ) -> RwKeyedJaggedTensorPoolUpdateValuesDist:
        return RwKeyedJaggedTensorPoolUpdateValuesDist(
            num_features=self._num_features,
            env=self._env,
            device=self._device,
        )

    def create_lookup_ids_dist(
        self,
    ) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(self._pg, is_update=False)

    def create_lookup_values_dist(self) -> RwKeyedJaggedTensorPoolLookupValuesDist:
        return RwKeyedJaggedTensorPoolLookupValuesDist(
            num_features=self._num_features, env=self._env
        )

    def get_sharded_states_to_register(
        self, lookup: KeyedJaggedTensorPoolLookup
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for fqn, tensor in lookup.states_to_register():
            yield fqn, ShardedTensor._init_from_local_shards(
                [
                    Shard(
                        tensor=tensor,
                        metadata=ShardMetadata(
                            shard_offsets=[
                                self._env.rank * self._block_size,
                                0,
                            ],
                            shard_sizes=[
                                tensor.shape[0],
                                tensor.shape[1],
                            ],
                            placement=f"rank:{self._env.rank}/{str(tensor.device)}",
                        ),
                    )
                ],
                torch.Size([self._pool_size, tensor.shape[1]]),
                process_group=self._env.process_group,
            )

    def create_context(self) -> ObjectPoolRwShardingContext:
        return ObjectPoolRwShardingContext(block_size=self._block_size_t)


@torch.fx.wrap
def _cat_if_multiple(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    if len(tensor_list) == 1:
        return tensor_list[0].flatten()
    else:
        return torch.cat([x.flatten() for x in tensor_list])


class InferRwKeyedJaggedTensorPoolOutputDist(torch.nn.Module):
    """
    Redistributes jaggd tensors in RW fashion with an AlltoOne operation.

    Inference assumes that this is called on a single rank, but jagged tensors are placed
    on different devices.

    Args:
        env (ShardingEnv): Sharding environment with info such as rank and world size
        device (torch.device): device on which the tensors will be communicated to.

    Example:
        device_cpu = torch.device("cpu")
        dist = InferRwKeyedJaggedTensorPoolOutputDist(env, device_cpu)
        jagged_tensors = [
            JaggedTensor(values=torch.tensor([1,2,3]), lengths=torch.tensor([1,1,1]), device=torch.device("rank:0/cuda:0")),
            JaggedTensor(values=torch.tensor([5,5,5]), lengths=torch.tensor([2,1]), device=torch.device("rank:1/cuda:0")),
        ]
        jt = dist(jagged_tensors)

        # jt has values [1,2,3,5,5,5] and lengths [1,1,1,2,1]
    """

    def __init__(
        self,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._sharding_env = env
        self._device = device

    def forward(
        self,
        jagged_tensors: List[JaggedTensor],
    ) -> JaggedTensor:
        """
        Performs AlltoOne operation on list of jagged tensors placed on different
        devices and returns merged jagged tensor.

        Args:
            jagged_tensors (List[JaggedTensor]): List of jagged tensors placed on
             different ranks

        Returns: JaggedTensor
        """
        lengths = [jt.lengths() for jt in jagged_tensors]
        values = [jt.values() for jt in jagged_tensors]
        values = _cat_if_multiple(
            torch.ops.fbgemm.all_to_one_device(
                [v.reshape(-1, v.shape[0]) for v in values], self._device
            )
        )
        lengths = _cat_if_multiple(
            torch.ops.fbgemm.all_to_one_device(
                [x.reshape(-1, x.shape[0]) for x in lengths],
                self._device,
            )
        )

        return JaggedTensor(values=values, lengths=lengths)


class InferRwKeyedJaggedTensorPoolSharding(InferObjectPoolSharding):
    def __init__(
        self,
        pool_size: int,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__(pool_size, env, device)

    def create_lookup_ids_dist(self) -> InferRwObjectPoolInputDist:
        return InferRwObjectPoolInputDist(
            self._env, device=self._device, block_size=self._block_size_t
        )

    def create_lookup_values_dist(self) -> InferRwKeyedJaggedTensorPoolOutputDist:
        return InferRwKeyedJaggedTensorPoolOutputDist(
            env=self._env, device=self._device
        )


class KeyedJaggedTensorPoolRwReplicatedSharding(ObjectPoolSharding):
    def __init__(
        self,
        env: ShardingEnv,
        device: torch.device,
        pool_size: int,
        num_features: int,
    ) -> None:
        self._env = env
        # pyre-ignore
        self._pg: dist.ProcessGroup = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        self._device = device
        self._local_world_size: int = get_local_size(self._world_size)

        self._pool_size = pool_size

        self._num_features = num_features

        intra_pg, _cross_pg = intra_and_cross_node_pg(
            device, backend=dist.get_backend(self._pg)
        )

        # pyre-ignore
        self._intra_pg: dist.ProcessGroup = intra_pg

        self._local_rank: int = get_local_rank(self._world_size)

        self._block_size: int = (
            pool_size + self._local_world_size - 1
        ) // self._local_world_size

        self.local_pool_size: int = (
            self._block_size
            if self._local_rank != self._local_world_size - 1
            else pool_size - self._block_size * (self._local_world_size - 1)
        )

        self._block_size_t: torch.Tensor = torch.tensor(
            [
                self._block_size,
            ],
            dtype=torch.long,
            device=self._device,
        )

        self._local_env = ShardingEnv(
            world_size=dist.get_world_size(self._intra_pg),
            rank=dist.get_rank(self._intra_pg),
            pg=self._intra_pg,
        )

        self._num_replicas: int = self._world_size // self._local_world_size

    def create_update_ids_dist(
        self,
    ) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(
            self._pg,
            is_update=True,
            bucketize_world_size=self._intra_pg.size(),
            num_replicas=self._num_replicas,
        )

    def create_update_values_dist(
        self,
    ) -> RwKeyedJaggedTensorPoolUpdateValuesDist:
        return RwKeyedJaggedTensorPoolUpdateValuesDist(
            num_features=self._num_features,
            env=self._env,
            num_replicas=self._num_replicas,
            device=self._device,
        )

    def create_lookup_ids_dist(
        self,
    ) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(self._intra_pg, is_update=False)

    def create_lookup_values_dist(
        self,
    ) -> RwKeyedJaggedTensorPoolLookupValuesDist:
        return RwKeyedJaggedTensorPoolLookupValuesDist(
            num_features=self._num_features, env=self._local_env
        )

    def get_sharded_states_to_register(
        self, lookup: KeyedJaggedTensorPoolLookup
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        for fqn, tensor in lookup.states_to_register():
            yield fqn, ShardedTensor._init_from_local_shards(
                [
                    Shard(
                        tensor=tensor,
                        metadata=ShardMetadata(
                            shard_offsets=[
                                self._local_env.rank * self._block_size,
                                0,
                            ],
                            shard_sizes=[
                                tensor.shape[0],
                                tensor.shape[1],
                            ],
                            placement=f"rank:{self._local_env.rank}/{str(tensor.device)}",
                        ),
                    )
                ],
                torch.Size([self._pool_size, tensor.shape[1]]),
                process_group=self._local_env.process_group,
            )

    def create_context(self) -> ObjectPoolReplicatedRwShardingContext:
        return ObjectPoolReplicatedRwShardingContext(block_size=self._block_size_t)
