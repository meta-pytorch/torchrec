#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import TensorAllToAll
from torchrec.distributed.tensor_sharding import ObjectPoolRwShardingContext
from torchrec.distributed.types import Awaitable, ShardingEnv

NUM_THREADS_BUCKETIZE = 32


class RwObjectPoolIDsDist(torch.nn.Module):
    """
    Redistribute torch.Tensor values containing IDs for sharded object pools

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        is_update (bool): Boolean indicating whether this is an update or not. Defaults
            to False.

            During update, number of values to send to each rank is determined by the
            number of IDs in each bucket, while no. of values to receive from each rank is
            determined by the no. of IDs for the current rank to be sent by all
            other ranks.

            During lookup, we first receive IDs to lookup from other ranks before
            distributing the looked up values, so the no. of values to send to each rank
            is collected from other ranks after IDs All2All. Conversely, no. of values
            to receive from each rank is determined by the number of IDs in each bucket.
            This is opposite of what happens during an update, so the code is shared.

        bucketize_world_size (Optional[int]): Number of buckets to bucketize IDs into.
            Defaults to `None` in which case the world size of the ProcessGroup is used.

        num_replicas (int): number of replicas of objects (tensor/KJT) to keep across
        ranks in case of replicated RW sharding. Defaults to 1.

    Example:
        dist = RwObjectPoolIDsDist(pg=pg, is_update=True, bucketize_world_size=2)
        ids = torch.Tensor([0,2,1,4,5])
        out = dist(ctx,ids).wait().wait()

        # values 2 and 1 need to be swapped
        ctx.unbucketize_permute == torch.tensor([0,2,1,3,4])
        ctx.num_ids_each_rank_to_send = torch.tensor([2,3])
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        is_update: bool = True,
        bucketize_world_size: Optional[int] = None,
        num_replicas: int = 1,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._dist = TensorAllToAll(pg=pg)
        self._is_update: bool = is_update

        self._num_replicas = num_replicas
        self._bucketize_world_size: int = bucketize_world_size or pg.size()

    def forward(
        self,
        ctx: ObjectPoolRwShardingContext,
        ids: torch.Tensor,
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        """
        Bucketizes IDs into `world_size` buckets and distributes them to other ranks.

        Args:
            ctx (Optional[EmbeddingShardingContext]): shared context from
                RW sharding operation. Number of ids to receive and send per rank
                is stored in this context
            ids (torch.Tensor): 1D tensor containing ids to be distributed

        Returns:
            Awaitable[Awaitable[torch.Tensor]]: awaitable of tensor awaitable.

        """

        num_ids = ids.shape[0]
        num_threads = NUM_THREADS_BUCKETIZE
        quot, rem = divmod(num_ids, num_threads)
        lengths = [quot] * num_threads
        for i in range(rem):
            lengths[i] += 1
        lengths = torch.tensor(lengths, device=ids.device, dtype=torch.int)

        (
            bucketized_lengths,
            bucketized_indices,
            _bucketized_weights,
            _bucketize_permute,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths=lengths,
            indices=ids,
            bucketize_pos=False,
            sequence=True,
            # pyre-ignore
            block_sizes=ctx.block_size.to(ids.dtype),
            my_size=self._bucketize_world_size,
            weights=None,
        )

        bucketized_lengths = (
            bucketized_lengths.reshape(self._bucketize_world_size, -1).sum(dim=1).int()
        )

        ctx.ids_before_input_dist = ids
        ctx.unbucketize_permute = unbucketize_permute
        # not needed, see if we can remove
        ctx.bucketize_permute = None

        if self._num_replicas > 1:
            bucketized_indices = bucketized_indices.repeat(self._num_replicas)
            bucketized_lengths = bucketized_lengths.repeat(self._num_replicas)

        await_dist_ids = self._dist(
            input=bucketized_indices,
            splits=bucketized_lengths,
        )

        if self._is_update:
            ctx.num_ids_each_rank_to_send = bucketized_lengths
            ctx.num_ids_each_rank_to_receive = await_dist_ids._output_splits
        else:
            ctx.num_ids_each_rank_to_send = await_dist_ids._output_splits
            ctx.num_ids_each_rank_to_receive = bucketized_lengths

        return await_dist_ids


@torch.fx.wrap
def _get_bucketize_shape(ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.tensor([ids.size(dim=0)], device=device, dtype=torch.long)


@torch.fx.wrap
def _get_unbucketize_permute_index(
    unbucketize_permute: Optional[torch.Tensor],
) -> torch.Tensor:
    assert unbucketize_permute is not None, "unbucketize permute must not be None"
    _, index = unbucketize_permute.sort()
    return index


class InferRwObjectPoolInputDist(torch.nn.Module):
    """
    Redistribute torch.Tensor values containing IDs for sharded object pools for inference

    Args:
        env (ShardingEnv): Sharding environment containing rank, world size, etc
        device (torch.device): device on which the tensors will be communicated to during
            lookup and update
        block_size (torch.Tensor): tensor containing block sizes for each rank.
            e.g. if block_size=torch.tensor(100), then IDs 0-99 will be assigned to rank
            0, 100-199 to rank 1, and so on.
        block_bucketize_row_pos (torch.Tensor]): tensor containing shard/row offsets for each
            rank in case of uneven sharding of the tensor pool across ranks. If not provided,
            then block_size will be used to permute the IDs across ranks.

    Example:
        device = torch.device("cpu")
        dist = InferRwObjectPoolInputDist(env=env, device=device, block_size=torch.tensor(100))
        ids = torch.Tensor([0,99,100,111])
        list_ids, permute = dist.lookup(ids)

        # list_ids == [torch.Tensor([0,99], device="cpu"), torch.Tensor([100,111], device="cpu)])]
    """

    _world_size: int
    _device: torch.device
    _block_size: torch.Tensor
    _block_bucketize_row_pos: list[torch.Tensor]

    def __init__(
        self,
        env: ShardingEnv,
        device: torch.device,
        block_size: torch.Tensor,
        block_bucketize_row_pos: Optional[list[torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self._world_size = env.world_size
        self._device = device
        self._block_size = block_size
        self._block_bucketize_row_pos: list[torch.Tensor] = (
            [] if block_bucketize_row_pos is None else block_bucketize_row_pos
        )

    def forward(
        self,
        ids: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bucketizes ids tensor into a list of tensors each containing ids
        for the corresponding rank. Places each tensor on the appropriate device.

        Args:
            ids (torch.Tensor): Tensor with ids

        Returns:
           Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
           Tuple containing
           1. list of ids tensors for each rank given the bucket sizes
           2. the tensor containing indices to permute the ids to get the original order before bucketization.
           3. the tensor containing the bucket mapping for each id
           4. the tensor containing the bucketized lengths
        """
        (
            bucketized_lengths,
            bucketized_indices,
            _,  # bucketized_weights
            _,  # _bucketize_permute
            unbucketize_permute,
            bucket_mapping,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths=_get_bucketize_shape(ids, ids.device),
            indices=ids.long(),
            bucketize_pos=False,
            sequence=True,
            block_sizes=self._block_size.long(),
            my_size=self._world_size,
            weights=None,
            block_bucketize_pos=(
                self._block_bucketize_row_pos
                if len(self._block_bucketize_row_pos) > 0
                else None
            ),
            return_bucket_mapping=True,
        )

        id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(bucketized_lengths)
        dist_ids = []
        for rank in range(self._world_size):
            offset = id_offsets[rank]
            next_offset = id_offsets[rank + 1]
            ids_for_rank = bucketized_indices[offset:next_offset]
            dist_ids.append(
                ids_for_rank
                if self._device == torch.device("cpu")
                else ids_for_rank.to(torch.device(f"cuda:{rank}"), non_blocking=True)
            )

        assert unbucketize_permute is not None, "unbucketize permute must not be None"
        assert bucket_mapping is not None, "bucket mapping must not be None"
        return (
            dist_ids,
            unbucketize_permute,
            bucket_mapping,
            bucketized_lengths,
        )

    def update(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Split the values into same buckets are IDs and place on the appropriate device
        for inference.

        Args:
            ids (torch.Tensor): Tensor with ids
            values (torch.Tensor): Tensor with values

        Returns:
           Tuple[List[torch.Tensor], List[torch.Tensor] torch.Tensor]: Tuple containing
            list of ids tensors, list of values tensors, and a tensor containing indices
            to permute the ids to get the original order before bucketization.
        """
        (
            bucketized_lengths,
            bucketized_indices,
            _bucketized_weights,
            _bucketize_permute,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            _get_bucketize_shape(ids, ids.device),
            ids.long(),
            bucketize_pos=False,
            sequence=True,
            block_sizes=self._block_size.long(),
            my_size=self._world_size,
            weights=None,
            block_bucketize_pos=(
                self._block_bucketize_row_pos
                if len(self._block_bucketize_row_pos) > 0
                else None
            ),
        )

        id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(bucketized_lengths)

        index = _get_unbucketize_permute_index(unbucketize_permute)
        unbucketize_values = values[index]
        dist_ids = []
        dist_values = []
        for rank in range(self._world_size):
            offset = id_offsets[rank]
            next_offset = id_offsets[rank + 1]
            ids_for_rank = bucketized_indices[offset:next_offset]
            values_for_rank = unbucketize_values[offset:next_offset]
            dist_ids.append(
                ids_for_rank
                if self._device == torch.device("cpu")
                else ids_for_rank.to(torch.device(f"cuda:{rank}"), non_blocking=True)
            )
            dist_values.append(
                values_for_rank
                if self._device == torch.device("cpu")
                else values_for_rank.to(torch.device(f"cuda:{rank}"), non_blocking=True)
            )

        return dist_ids, dist_values, unbucketize_permute
