#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple, Type, Union

import torch
from torchrec.distributed.object_pool import ShardedObjectPool
from torchrec.distributed.sharding.rw_pool_sharding import (
    InferRwObjectPoolInputDist,
    RwObjectPoolIDsDist,
)
from torchrec.distributed.sharding.rw_tensor_pool_sharding import (
    InferRwTensorPoolOutputDist,
    InferRwTensorPoolSharding,
    RwTensorPoolValuesDist,
    TensorPoolRwSharding,
)
from torchrec.distributed.tensor_sharding import ObjectPoolShardingContext
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ModuleSharder,
    ObjectPoolShardingPlan,
    ObjectPoolShardingType,
    ShardingEnv,
)
from torchrec.modules.object_pool_lookups import TensorLookup, TensorPoolLookup
from torchrec.modules.tensor_pool import TensorPool
from torchrec.modules.utils import (
    _get_batching_hinted_output,
    _get_unbucketize_tensor_via_length_alignment,
    deterministic_dedup,
)

torch.fx.wrap("_get_unbucketize_tensor_via_length_alignment")
torch.fx.wrap("_get_batching_hinted_output")


@torch.fx.wrap
def index_select_view(
    output: torch.Tensor,
    unbucketize_permute: Optional[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    return output[unbucketize_permute].view(-1, dim)


@torch.fx.wrap
def _fx_item_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def _get_id_length_sharded_tensor_pool(ids: torch.Tensor) -> torch.Tensor:
    return torch.tensor([ids.size(dim=0)], device=ids.device, dtype=torch.long)


class TensorPoolAwaitable(LazyAwaitable[torch.Tensor]):
    def __init__(
        self,
        awaitable: Awaitable[torch.Tensor],
        unbucketize_permute: torch.Tensor,
    ) -> None:
        super().__init__()
        self._awaitable = awaitable
        self._unbucketize_permute = unbucketize_permute

    def _wait_impl(self) -> torch.Tensor:
        tensor = self._awaitable.wait()

        return tensor[self._unbucketize_permute]


class ShardedTensorPool(
    ShardedObjectPool[torch.Tensor, torch.Tensor, ObjectPoolShardingContext]
):
    """
    Sharded implementation of `TensorPool`

    When dealing with large pool of tensors that cannot fit in a single device memory
    (i.e. HBM / UVM / CPU etc), this module handles sharding the pool row-wise, including
    orchestrating the communication between ranks for distributed lookup and update.

    Args:
        env (ShardingEnv): sharding environment (e.g. world_size, ranks, etc)
        pool_size (int): total number of rows of tensors in the pool
        dim (int): dimension that each tensor in the pool
        dtype (torch.dtype): dtype of the tensors in the pool
        sharding_plan (ObjectPoolShardingPlan): info about sharding strategy
        device (Optional[torch.device]): default device
        enable_uvm (bool): if set to true, the pool will be allocated on UVM

    Example::
        # Example on 2 GPUs

        # rank 0
        sharded_keyed_jagged_tensor_pool.update(
            ids=torch.Tensor([2,0],dtype=torch.int,device="cuda:0")
            values=torch.Tensor([
                [1,2,3],
                [4,5,6],
            ],dtype=torch.int,device="cuda:0")
        )

        # on rank 1
        sharded_keyed_jagged_tensor_pool.update(
            ids=torch.Tensor([1,3],dtype=torch.int,device="cuda:1")
            values=torch.Tensor([
                [7,8,9],
                [10,11,12],
            ],dtype=torch.int,device="cuda:1")
        )

        # At this point the global state is:
        # ids   tensor
        # 0     [1,2,3]         <- rank 0
        # 1     [7,8,9]         <- rank 1
        # 2     [4,5,6]         <- rank 0
        # 3     [10,11,12]      <- rank 1

    """

    def __init__(
        self,
        env: ShardingEnv,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        sharding_plan: ObjectPoolShardingPlan,
        device: Optional[torch.device] = None,
        enable_uvm: bool = False,
    ) -> None:
        super().__init__()

        self._world_size = env.world_size
        self._rank = env.rank
        self._pool_size = pool_size
        self._sharding_env = env
        self._dim: int = dim
        self._device = device if device is not None else torch.device("meta")
        self._dtype = dtype
        self._sharding_plan = sharding_plan
        self._enable_uvm = enable_uvm

        if sharding_plan.sharding_type == ObjectPoolShardingType.ROW_WISE:
            self._sharding: TensorPoolRwSharding = TensorPoolRwSharding(
                env=self._sharding_env,
                device=self._device,
                pool_size=self._pool_size,
                dim=dim,
            )
        else:
            raise NotImplementedError(
                f"Sharding type {self._sharding_plan.sharding_type} is not implemented"
            )

        self._lookup: TensorPoolLookup = TensorLookup(
            self._sharding.local_pool_size,
            self._dim,
            self._dtype,
            self._device,
            self._enable_uvm,
        )

        self._lookup_ids_dist_impl: RwObjectPoolIDsDist = (
            self._sharding.create_lookup_ids_dist()
        )
        self._lookup_values_dist_impl: RwTensorPoolValuesDist = (
            self._sharding.create_lookup_values_dist()
        )

        self._update_ids_dist_impl: RwObjectPoolIDsDist = (
            self._sharding.create_update_ids_dist()
        )
        self._update_values_dist_impl: RwTensorPoolValuesDist = (
            self._sharding.create_update_values_dist()
        )

        self._initialize_torch_state()

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        torch._assert(self._device is not None, "self._device should already be set")
        return self._device

    def _update_preproc(self, values: torch.Tensor) -> torch.Tensor:
        assert values.dtype == self.dtype
        assert values.size(1) == self._dim
        assert values.device.type == self._device.type
        return values

    def _update_ids_dist(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        return self._update_ids_dist_impl(ctx=ctx, ids=ids)

    def _update_values_dist(
        self, ctx: ObjectPoolShardingContext, values: torch.Tensor
    ) -> LazyAwaitable[torch.Tensor]:
        return self._update_values_dist_impl(ctx=ctx, values=values)

    def _update_local(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        deduped_ids, dedup_permutation = deterministic_dedup(ids)

        self._lookup.update(
            deduped_ids,
            values[dedup_permutation],
        )

    def _lookup_ids_dist(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        return self._lookup_ids_dist_impl(ctx=ctx, ids=ids)

    def _lookup_local(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> torch.Tensor:
        return self._lookup.lookup(ids)

    def _lookup_values_dist(
        self,
        ctx: ObjectPoolShardingContext,
        values: torch.Tensor,
    ) -> LazyAwaitable[torch.Tensor]:
        return TensorPoolAwaitable(
            awaitable=self._lookup_values_dist_impl(ctx, values),
            # pyrefly: ignore[bad-argument-type]
            unbucketize_permute=ctx.unbucketize_permute,
        )

    def create_context(self) -> ObjectPoolShardingContext:
        return self._sharding.create_context()

    def _initialize_torch_state(self) -> None:
        for fqn, tensor in self._sharding.get_sharded_states_to_register(self._lookup):
            self.register_buffer(fqn, tensor)


@torch.fx.wrap
def update(
    shard: torch.nn.Parameter, rank_ids: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    if values.device != shard.device:
        values = values.to(shard.device)
    shard[rank_ids] = values
    return torch.empty(0)


class LocalShardPool(torch.nn.Module):
    """
    Module containing a single shard of a tensor pool as a parameter.

    Used to lookup and update the pool during inference.

    Args:
        shard (torch.Tensor): Subset of the tensor pool.

    Example:
        # shard containing 2 rows from tensor pool with dim=3
        shard = torch.tensor([
            [1,2,3],
            [4,5,6],
        ])
        pool = LocalShardPool(shard)
        out = pool(torch.tensor([0]))
        # out is tensor([1,2,3]) i.e. first row of the shard
    """

    current_device: torch.device

    def __init__(
        self,
        shard: torch.Tensor,
    ) -> None:
        super().__init__()
        self._shard: torch.nn.Parameter = torch.nn.Parameter(
            shard,
            requires_grad=False,
        )
        self.current_device = self._shard.device

    @torch.jit.export
    def set_device(self, device_str: str) -> None:
        self.current_device = torch.device(device_str)

    def forward(self, rank_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the rows in the shard corresponding to the given rank ids.

        Args:
            rank_ids (torch.Tensor): Tensor of rank ids to lookup.

        Returns:
            torch.Tensor: Tensor of values corresponding to the given rank ids.
        """
        return self._shard[rank_ids.to(self.current_device)]

    def update(self, rank_ids: torch.Tensor, values: torch.Tensor) -> None:
        _ = update(self._shard, rank_ids, values)


class ShardedInferenceTensorPool(
    ShardedObjectPool[torch.Tensor, List[torch.Tensor], ObjectPoolShardingContext],
):
    _local_shard_pools: torch.nn.ModuleList
    _world_size: int
    _device: torch.device
    _rank: int

    def __init__(
        self,
        env: ShardingEnv,
        pool_size: int,
        dim: int,
        dtype: torch.dtype,
        plan: ObjectPoolShardingPlan,
        module: TensorPool,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self._pool_size = pool_size
        self._dtype = dtype
        self._sharding_env = env
        self._world_size = env.world_size
        self._device = device or torch.device("cuda")
        self._sharding_plan = plan

        self._rank = env.rank
        self._dim = dim

        torch._assert(
            self._sharding_plan.inference, "Plan needs to have inference enabled"
        )

        if self._sharding_plan.sharding_type == ObjectPoolShardingType.ROW_WISE:
            self._sharding = InferRwTensorPoolSharding(
                env=self._sharding_env,
                device=self._device,
                pool_size=self._pool_size,
                memory_capacity_per_rank=(
                    self._sharding_plan.memory_capacity_per_rank
                    if self._sharding_plan.memory_capacity_per_rank is not None
                    else None
                ),
            )
        else:
            raise NotImplementedError(
                f"Sharding type {self._sharding_plan.sharding_type} is not implemented"
            )

        self._local_shard_pools: torch.nn.ModuleList = torch.nn.ModuleList()
        offset = 0
        for rank, this_rank_size in zip(
            range(
                self._world_size,
            ),
            self._sharding.local_pool_size_per_rank,
        ):
            shard_device = (
                torch.device("cpu")
                if device == torch.device("cpu")
                else torch.device("cuda", rank)
            )

            self._local_shard_pools.append(
                LocalShardPool(
                    torch.empty(
                        (
                            this_rank_size,
                            self._dim,
                        ),
                        dtype=self._dtype,
                        device=shard_device,
                        requires_grad=False,
                    ),
                )
            )

            if module._pool.device != torch.device("meta"):
                local_shard = module._pool[offset : offset + this_rank_size]
                # pyrefly: ignore[not-callable]
                self._local_shard_pools[rank]._shard.copy_(local_shard)

            offset += this_rank_size

        self._lookup_ids_dist_impl: InferRwObjectPoolInputDist = (
            self._sharding.create_lookup_ids_dist()
        )
        self._lookup_values_dist_impl: InferRwTensorPoolOutputDist = (
            self._sharding.create_lookup_values_dist()
        )

    # TODO use DTensor that works with Inference Publishing. Right now ShardedTensor doesn't fit this shoe.

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        torch._assert(self._device is not None, "self._device should already be set")
        return self._device

    def create_context(self) -> ObjectPoolShardingContext:
        raise NotImplementedError("create_context() is not implemented")

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _lookup_ids_dist(
        self,
        ids: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._lookup_ids_dist_impl(ids)

    # pyrefly: ignore[bad-param-name-override]
    def _update_ids_dist(
        self,
        ids: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        return self._lookup_ids_dist_impl.update(ids, values)

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _lookup_local(
        self,
        dist_input: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        ret = []
        for i, shard in enumerate(self._local_shard_pools):
            ret.append(shard(dist_input[i]))
        return ret

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _lookup_values_dist(
        self,
        lookups: List[torch.Tensor],
    ) -> torch.Tensor:
        return self._lookup_values_dist_impl(lookups)

    # pyrefly: ignore[bad-override]
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        dist_input, unbucketize_permute, bucket_mapping, bucketized_lengths = (
            self._lookup_ids_dist(ids)
        )
        unbucketize_permute_non_opt = unbucketize_permute

        lookup = self._lookup_local(dist_input)

        # Here we are playing a trick to workaround a fx tracing issue,
        # as proxy is not iteratable.
        lookup_list = []
        # In case of non-heterogenous even sharding keeping the behavior
        # consistent with existing logic to ensure that additional fx wrappers
        # do not impact the model split logic during inference in anyway
        if self._sharding_plan.memory_capacity_per_rank is None:
            for i in range(self._world_size):
                lookup_list.append(lookup[i])
        else:
            # Adding fx wrappers in case of uneven heterogenous sharding to
            # make it compatible with model split boundaries during inference
            for i in range(self._world_size):
                lookup_list.append(
                    _get_batching_hinted_output(
                        _get_id_length_sharded_tensor_pool(dist_input[i]), lookup[i]
                    )
                )

            features_before_input_dist_length = _get_id_length_sharded_tensor_pool(ids)
            bucketized_lengths_col_view = bucketized_lengths.view(self._world_size, -1)
            unbucketize_permute_non_opt = _fx_item_unwrap_optional_tensor(
                unbucketize_permute
            )
            bucket_mapping_non_opt = _fx_item_unwrap_optional_tensor(bucket_mapping)
            unbucketize_permute_non_opt = _get_unbucketize_tensor_via_length_alignment(
                features_before_input_dist_length,
                bucketized_lengths_col_view,
                unbucketize_permute_non_opt,
                bucket_mapping_non_opt,
            )

        output = self._lookup_values_dist(lookup_list)
        # When memory_capacity_per_rank is added then boundary split for the
        # model is different. Handling device movement accordingly
        if self._sharding_plan.memory_capacity_per_rank is None:
            return index_select_view(
                output,
                unbucketize_permute_non_opt,
                self._dim,
            )
        else:
            return index_select_view(
                output,
                unbucketize_permute_non_opt.to(device=output.device),
                self._dim,
            )

    def _update_values_dist(self, ctx: ObjectPoolShardingContext, values: torch.Tensor):
        raise NotImplementedError("Inference does not support update")

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _update_local(
        self,
        dist_input: List[torch.Tensor],
        dist_values: List[torch.Tensor],
    ) -> None:
        for i, shard in enumerate(self._local_shard_pools):
            ids = dist_input[i]
            values = dist_values[i]
            deduped_ids, dedup_permutation = deterministic_dedup(ids)
            # pyrefly: ignore[not-callable]
            shard.update(deduped_ids, values[dedup_permutation])

    # pyrefly: ignore[bad-return]
    def _update_preproc(self, values: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, ids: torch.Tensor, values: torch.Tensor) -> None:
        dist_input, dist_values, unbucketize_permute = self._update_ids_dist(
            ids, values
        )
        self._update_local(dist_input, dist_values)


class TensorPoolSharder(ModuleSharder[TensorPool]):
    def __init__(self) -> None:
        super().__init__()

    # pyrefly: ignore[bad-param-name-override]
    def shard(
        self,
        module: TensorPool,
        plan: ObjectPoolShardingPlan,
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> Union[ShardedTensorPool, ShardedInferenceTensorPool]:
        if plan.inference:
            return ShardedInferenceTensorPool(
                env=env,
                pool_size=module.pool_size,
                dim=module.dim,
                dtype=module.dtype,
                plan=plan,
                device=device,
                module=module,
            )
        return ShardedTensorPool(
            env=env,
            pool_size=module.pool_size,
            dim=module.dim,
            dtype=module.dtype,
            sharding_plan=plan,
            device=device,
            enable_uvm=module._enable_uvm,
        )

    @property
    def module_type(self) -> Type[TensorPool]:
        return TensorPool
