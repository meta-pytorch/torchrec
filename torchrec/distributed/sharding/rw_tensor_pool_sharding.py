#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard, ShardMetadata
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torchrec.distributed.dist_data import TensorValuesAllToAll
from torchrec.distributed.sharding.rw_pool_sharding import (
    InferRwObjectPoolInputDist,
    RwObjectPoolIDsDist,
)
from torchrec.distributed.tensor_sharding import (
    InferObjectPoolSharding,
    ObjectPoolRwShardingContext,
    ObjectPoolSharding,
    TensorPoolRwShardingContext,
)
from torchrec.distributed.types import LazyAwaitable, ShardingEnv
from torchrec.modules.object_pool_lookups import TensorPoolLookup


class RwTensorPoolValuesDist(torch.nn.Module):
    """
    Module to distribute torch.Tensor to all ranks after local pool lookup

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        is_update (bool): Boolean indicating whether this is an update or not.

    Example:
        dist = RwTensorPoolLookupValuesDist(pg)
        # rank 0
        rank0_ctx = TensorPoolRwShardingContext(
            num_ids_each_rank_to_send=2,
            num_ids_each_rank_to_receive=3,
        )
        rank0_values = torch.tensor([2,3,4,5])

        # rank 1
        rank0_ctx = TensorPoolRwShardingContext(
            num_ids_each_rank_to_send=3,
            num_ids_each_rank_to_receive=2,
        )
        rank1_values = torch.tensor([1,1,1,3,4])

        rank0_out = dist(rank0_ctx, rank0_values).wait()
        # rank0_out has values [2,3,1,1,1]

        rank1_out = dist(rank1_ctx, rank1_values).wait()
        # rank1_out has values [4,5,3,4]
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        is_update: bool,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._dist = TensorValuesAllToAll(pg=pg)
        self._is_update = is_update

    def forward(
        self,
        ctx: TensorPoolRwShardingContext,
        values: torch.Tensor,
    ) -> LazyAwaitable[torch.Tensor]:
        """
        Redistributes local tensor values after tensor pool lookup.
        Will only permute values when updating.

        Args:
            ctx (TensorPoolRwShardingContext): Context for RW sharding, containing
                number of items to send and receive from each rank.
            values (torch.Tensor): tensor to distribute.

        Returns:
            LazyAwaitable[torch.Tensor]: Lazy awaitable of tensor
        """

        if self._is_update:
            with torch.no_grad():
                assert hasattr(ctx, "unbucketize_permute")
                bucketize_permute = torch.ops.fbgemm.invert_permute(
                    ctx.unbucketize_permute
                )
                values = values[bucketize_permute]

        return self._dist(
            input=values,
            input_splits=ctx.num_ids_each_rank_to_send,
            output_splits=ctx.num_ids_each_rank_to_receive,
        )


class TensorPoolRwSharding(ObjectPoolSharding):
    def __init__(
        self,
        pool_size: int,
        dim: int,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        self._env = env
        # pyre-ignore
        self._pg: dist.ProcessGroup = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        self._device = device
        self._pool_size = pool_size
        self._dim = dim

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

    def create_update_ids_dist(
        self,
    ) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(self._pg, is_update=True)

    def create_update_values_dist(
        self,
    ) -> RwTensorPoolValuesDist:
        """
        used in embedding A2A in update()
        """
        return RwTensorPoolValuesDist(self._pg, is_update=True)

    def create_lookup_ids_dist(self) -> RwObjectPoolIDsDist:
        return RwObjectPoolIDsDist(self._pg, is_update=False)

    def create_lookup_values_dist(
        self,
    ) -> RwTensorPoolValuesDist:
        """
        used in embedding A2A in lookup()
        """
        return RwTensorPoolValuesDist(self._pg, is_update=False)

    # pyre-fixme[14]: `get_sharded_states_to_register` overrides method defined in
    #  `ObjectPoolSharding` inconsistently.
    def get_sharded_states_to_register(
        self, lookup: TensorPoolLookup
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


class InferRwTensorPoolOutputDist(torch.nn.Module):
    """
    Collects local tensor values after tensor pool lookup
    to one device during inference.

    Args:
        env (ShardingEnv): Sharding environment
        device (torch.device): device to collect onto

    Example:
        device = torch.device("cpu")
        dist = InferRwTensorPoolOutputDist(env, device)
        lookups = [
            torch.Tensor([1,2,3], device="rank0:cuda:0"),
            torch.Tensor([4,5,6], device="rank1:cuda:0"),
        ]
        vals = dist(lookups)
        # tensors merged and on CPU
        vals = torch.Tensor([1,2,3,4,5,6], device=device)
    """

    __annotations__ = {"_device": Optional[torch.device]}

    def __init__(
        self,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._device: Optional[torch.device] = device
        self._world_size: int = env.world_size
        self._cat_dim = 0
        self._placeholder: torch.Tensor = torch.ones(1, device=device)

    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self._device = torch.device(device_str)
        self._placeholder = torch.ones(1, device=self._device)

    def forward(
        self,
        lookups: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge lookup values tensor on different devices onto a single device and rank

        Args:
            lookups (List[torch.Tensor]): List of tensors placed on possibly different
            devices / ranks.

        Returns:
            torch.Tensor: Merged tensor on the requested device
        """
        torch._assert(len(lookups) == self._world_size, "lookups size not world size")

        non_cat_size = lookups[0].size(1 - self._cat_dim)
        return torch.ops.fbgemm.merge_pooled_embeddings(
            lookups,
            non_cat_size,
            # syntax for torchscript
            self._placeholder.device,
            self._cat_dim,
        )


class InferRwTensorPoolSharding(InferObjectPoolSharding):
    def __init__(
        self,
        pool_size: int,
        env: ShardingEnv,
        device: torch.device,
        memory_capacity_per_rank: Optional[list[int]] = None,
    ) -> None:
        super().__init__(pool_size, env, device, memory_capacity_per_rank)

    def create_lookup_ids_dist(self) -> InferRwObjectPoolInputDist:
        return InferRwObjectPoolInputDist(
            self._env,
            device=self._device,
            block_size=self._block_size_t,
            block_bucketize_row_pos=self._block_bucketize_row_pos,
        )

    def create_lookup_values_dist(
        self,
    ) -> InferRwTensorPoolOutputDist:
        return InferRwTensorPoolOutputDist(env=self._env, device=self._device)
