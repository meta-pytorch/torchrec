#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import cast, Dict, List, Optional, Tuple, Type, Union

import torch
from torchrec.distributed.object_pool import ShardedObjectPool
from torchrec.distributed.sharding.rw_kjt_pool_sharding import (
    InferRwKeyedJaggedTensorPoolOutputDist,
    InferRwKeyedJaggedTensorPoolSharding,
    KeyedJaggedTensorPoolRwReplicatedSharding,
    KeyedJaggedTensorPoolRwSharding,
)
from torchrec.distributed.sharding.rw_pool_sharding import InferRwObjectPoolInputDist
from torchrec.distributed.tensor_sharding import ObjectPoolShardingContext
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ModuleSharder,
    ObjectPoolShardingPlan,
    ObjectPoolShardingType,
    ShardingEnv,
)
from torchrec.modules.keyed_jagged_tensor_pool import KeyedJaggedTensorPool
from torchrec.modules.object_pool_lookups import (
    KeyedJaggedTensorPoolLookup,
    TensorJaggedIndexSelectLookup,
    UVMCachingInt32Lookup,
    UVMCachingInt64Lookup,
)
from torchrec.modules.utils import deterministic_dedup, jagged_index_select_with_empty
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class KeyedJaggedTensorPoolAwaitable(LazyAwaitable[KeyedJaggedTensor]):
    def __init__(
        self,
        awaitable: Awaitable[JaggedTensor],
        keys: List[str],
        device: torch.device,
        unbucketize_permute: torch.Tensor,
    ) -> None:
        super().__init__()
        self._awaitable = awaitable
        self._unbucketize_permute = unbucketize_permute
        self._keys = keys
        self._device = device

    def _wait_impl(self) -> KeyedJaggedTensor:
        # we could un-permute, but perhaps unnecessary for caching use case.
        jt = self._awaitable.wait()

        # if we have an empty found, we should not perform any lookups and just
        # return an empty KJT.
        if jt.lengths().size()[0] == 0:
            return KeyedJaggedTensor.empty(
                is_weighted=jt.weights_or_none() is not None,
                device=self._device,
                values_dtype=jt.values().dtype,
                lengths_dtype=jt.lengths().dtype,
                weights_dtype=getattr(jt.weights_or_none(), "dtype", None),
            )

        """
        We need to permute the row order KJT based on the unbucketize permute tensor
        to respect the original order that it came in.
        """

        unbucketize_id_permute = (
            torch.arange(jt.lengths().shape[0], device=self._device)
            .view(-1, len(self._keys))[self._unbucketize_permute]
            .flatten()
        )

        """
        Since the all to all will return to us in a row manner format, we need to regroup
        using jaggeed index_select to key order.
        For example, we will receive 0,2,3,4,5,6,7. But we need it to be in [0,2,5,6,3,4,7] order.

        F1      F2
        [0,2] . [3,4]
        [5,6]   [7]

        Can remove if we can write efficient kernel that can return in feature order. This would also
        require splits to be transposed and flattened, to be put in feature order.
        """

        row_major_to_feature_major_permute = (
            torch.arange(jt.lengths().shape[0], device=self._device)
            .view(-1, len(self._keys))
            .t()
            .flatten()
        )
        """
            The below is equivalent to doing
            reorder_v = jagged_index_select(values, unbucketize_id_permute)
            reorder_v = jagged_index_select(reorder_v, row_major_to_feature_major_permute)
        """

        indices = unbucketize_id_permute[row_major_to_feature_major_permute]
        reorder_l = jt.lengths()[indices]
        reorder_o = torch.ops.fbgemm.asynchronous_inclusive_cumsum(reorder_l)
        reorder_v = jagged_index_select_with_empty(
            jt.values().unsqueeze(-1), indices, jt.offsets()[1:], reorder_o
        )

        reorder_w = (
            jagged_index_select_with_empty(
                jt.weights().unsqueeze(-1),
                indices,
                jt.offsets()[1:],
                reorder_o,
            )
            if jt.weights_or_none() is not None
            else None
        )

        return KeyedJaggedTensor(
            keys=self._keys,
            values=reorder_v.flatten(),
            weights=reorder_w.flatten() if reorder_w is not None else None,
            lengths=reorder_l,
        )


class ShardedKeyedJaggedTensorPool(
    ShardedObjectPool[
        KeyedJaggedTensor,  # Out
        JaggedTensor,  # DistOut
        ObjectPoolShardingContext,  # Ctx
    ]
):
    """
    Sharded implementation of `KeyedJaggedTensorPool`

    When dealing with a large pool that cannot fit in a single device memory
    (i.e. HBM / UVM / CPU etc), this module handles sharding the pool row-wise, including
    orchestrating the communication between ranks for distributed lookup and update.

    Args:
        pool_size (int): total number of batches that can be stored in the pool
        values_dtype (torch.dtype): dtype of the KJT values in the pool
        feature_max_lengths (Dict[str,int]): Mapping from feature name in KJT
            to the maximum size of the jagged slices for the feature.
        is_weighted (bool): whether KJT values have weights that need to be stored.
        sharding_env (ShardingEnv): sharding environment (e.g. world_size, ranks, etc)
        sharding_plan (ObjectPoolShardingPlan): info about sharding strategy
        device (Optional[torch.device]): default device
        enable_uvm (bool): if set to true, the pool will be allocated on UVM

    Example::
        # Example on 2 GPUs
        # on rank 0, update ids [2,0] with values
        # ids   f1       f2
        # 2     [1]      [2, 3]
        # 0     [4,5]    [6]
        sharded_keyed_jagged_tensor_pool.update(
            ids=torch.Tensor([2,0],dtype=torch.int,device="cuda:0")
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1","f2"],
                values=torch.Tensor([1,2,3,4,5,6],device="cuda:0"),
                lengths=torch.Tensor([1,2,2,1],device="cuda:0")
            )
        )

        # on rank 1, update ids [1,3] with values
        # ids   f1           f2
        # 1     [7,8]        []
        # 3     [9,10,11]    [12]
        sharded_keyed_jagged_tensor_pool.update(
            ids=torch.Tensor([1,3],dtype=torch.int,device="cuda:1")
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1","f2"],
                values=torch.Tensor([7,8,9,10,11,12],device="cuda:1"),
                lengths=torch.Tensor([2,0,3,1],device="cuda:1")
            )
        )

        # At this point the global state is:
        # ids   f1      f2
        # 0    [2,3]    [6]         <- rank 0
        # 1    [7,8]    [9,10,11]   <- rank 0
        # 2    [1]      [4,5]       <- rank 1
        # 3    []       [12]        <- rank 1

    """

    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_env: ShardingEnv,
        sharding_plan: ObjectPoolShardingPlan,
        device: Optional[torch.device] = None,
        # TODO add quantized comms codec registry
        enable_uvm: bool = False,
    ) -> None:

        super().__init__()
        self._pool_size = pool_size
        self._values_dtype = values_dtype
        self._sharding_env = sharding_env
        self._device: torch.device = device or torch.device("cuda")
        self._is_weighted = is_weighted
        self._sharding_plan = sharding_plan
        self._feature_max_lengths = feature_max_lengths
        self.register_buffer(
            "_feature_max_lengths_t",
            torch.tensor(
                list(feature_max_lengths.values()),
                dtype=torch.int32,
                device=self._device,
            ),
            persistent=False,
        )
        self._features: List[str] = list(feature_max_lengths.keys())
        self._enable_uvm = enable_uvm

        self._permute_feature = None
        if sharding_plan.sharding_type == ObjectPoolShardingType.ROW_WISE:
            self._sharding: KeyedJaggedTensorPoolRwSharding = (
                KeyedJaggedTensorPoolRwSharding(
                    env=self._sharding_env,
                    device=self._device,
                    pool_size=self._pool_size,
                    num_features=len(feature_max_lengths),
                )
            )
        elif sharding_plan.sharding_type == ObjectPoolShardingType.REPLICATED_ROW_WISE:
            self._sharding: KeyedJaggedTensorPoolRwReplicatedSharding = (
                # pyrefly: ignore[bad-assignment]
                KeyedJaggedTensorPoolRwReplicatedSharding(
                    env=self._sharding_env,
                    device=self._device,
                    pool_size=self._pool_size,
                    num_features=len(feature_max_lengths),
                )
            )

        else:
            raise NotImplementedError(
                f"Sharding type {self._sharding_plan.sharding_type} is not implemented"
            )

        # pyrefly: ignore[bad-assignment]
        self._lookup: KeyedJaggedTensorPoolLookup = None
        if self._enable_uvm:
            if values_dtype == torch.int64:
                self._lookup = UVMCachingInt64Lookup(
                    self._sharding.local_pool_size,
                    feature_max_lengths,
                    is_weighted,
                    self._device,
                )
            if values_dtype == torch.int32:
                self._lookup = UVMCachingInt32Lookup(
                    self._sharding.local_pool_size,
                    feature_max_lengths,
                    is_weighted,
                    self._device,
                )
        else:
            self._lookup = TensorJaggedIndexSelectLookup(
                self._sharding.local_pool_size,
                values_dtype,
                feature_max_lengths,
                is_weighted,
                self._device,
            )
        if self._lookup is None:
            raise ValueError(
                f"Cannot create lookup for {self._enable_uvm=} {self._values_dtype}"
            )

        for fqn, tensor in self._lookup.states_to_register():
            self.register_buffer(
                fqn,
                tensor,
            )

        self._lookup_ids_dist_impl = self._sharding.create_lookup_ids_dist()
        self._lookup_values_dist_impl = self._sharding.create_lookup_values_dist()
        self._update_ids_dist_impl = self._sharding.create_update_ids_dist()
        self._update_values_dist_impl = self._sharding.create_update_values_dist()

        self._initialize_torch_state(self._lookup, sharding_plan.sharding_type)

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def feature_max_lengths(self) -> Dict[str, int]:
        return self._feature_max_lengths

    @property
    def values_dtype(self) -> torch.dtype:
        return self._values_dtype

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted

    @property
    def device(self) -> torch.device:
        torch._assert(self._device is not None, "self._device should already be set")
        return self._device

    def _initialize_torch_state(
        self, lookup: KeyedJaggedTensorPoolLookup, sharding_type: ObjectPoolShardingType
    ) -> None:
        for fqn, tensor in self._sharding.get_sharded_states_to_register(self._lookup):
            self.register_buffer(fqn, tensor)
        # somewhat hacky. ideally, we should be able to invoke this method on
        # any update to the lookup's key_lengths field.
        lengths, offsets = lookup._infer_jagged_lengths_inclusive_offsets()
        lookup._lengths = lengths
        lookup._offsets = offsets

    def _lookup_ids_dist(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        return self._lookup_ids_dist_impl(ctx=ctx, ids=ids)

    def _update_preproc(self, values: KeyedJaggedTensor) -> KeyedJaggedTensor:
        """
        1. Permute/filter KJT keys to be the same as in feature_max_lengths
        2. Ensure the max_lengths of input is within the feature_max_lengths
        """
        if self._permute_feature is None:
            self._permute_feature = []
            for feature in self._feature_max_lengths.keys():
                for j, kjt_feature in enumerate(values.keys()):
                    if feature == kjt_feature:
                        self._permute_feature.append(j)

        valid_input = values.permute(self._permute_feature)
        # can disable below check if expensive
        max_elements, _max_indices = (
            valid_input.lengths()
            .reshape(len(self._feature_max_lengths.keys()), -1)
            .max(dim=1)
        )

        # pyrefly: ignore[no-matching-overload]
        assert torch.all(
            max_elements <= self._feature_max_lengths_t
        ).item(), "input KJT has a feature that exceeds specified max lengths"

        return valid_input

    def _update_local(
        self,
        ctx: ObjectPoolShardingContext,
        ids: torch.Tensor,
        values: JaggedTensor,
    ) -> None:
        if ids.size(0) == 0:
            return
        jt = values
        deduped_ids, dedup_permutation = deterministic_dedup(ids)

        device = ids.device
        arange_idx = torch.arange(len(jt.lengths()), device=device)
        value_dedup_permute = (arange_idx.view(-1, len(self._feature_max_lengths)))[
            dedup_permutation, :
        ].flatten()

        deduped_lengths = jt.lengths()[value_dedup_permute]
        deduped_offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(
            deduped_lengths
        )
        deduped_values = jagged_index_select_with_empty(
            jt.values().unsqueeze(-1),
            value_dedup_permute,
            jt.offsets()[1:],
            deduped_offsets,
        )

        deduped_values, deduped_lengths = (
            deduped_values.flatten(),
            deduped_lengths.flatten(),
        )

        deduped_weights = None
        if jt.weights_or_none() is not None:
            deduped_weights = jagged_index_select_with_empty(
                jt.weights().unsqueeze(-1),
                value_dedup_permute,
                jt.offsets()[1:],
                deduped_offsets,
            )
            deduped_weights = deduped_weights.flatten()

        self._lookup.update(
            deduped_ids,
            JaggedTensor(
                values=deduped_values,
                lengths=deduped_lengths,
                weights=deduped_weights,
            ),
        )

    def _lookup_local(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> JaggedTensor:
        return self._lookup.lookup(ids)

    def _lookup_values_dist(
        self,
        ctx: ObjectPoolShardingContext,
        values: JaggedTensor,
    ) -> LazyAwaitable[KeyedJaggedTensor]:
        return KeyedJaggedTensorPoolAwaitable(
            awaitable=self._lookup_values_dist_impl(ctx, values),
            # pyrefly: ignore[bad-argument-type]
            unbucketize_permute=ctx.unbucketize_permute,
            keys=self._features,
            device=self._device,
        )

    def _update_ids_dist(
        self, ctx: ObjectPoolShardingContext, ids: torch.Tensor
    ) -> Awaitable[Awaitable[torch.Tensor]]:
        return self._update_ids_dist_impl(ctx=ctx, ids=ids)

    def _update_values_dist(
        self, ctx: ObjectPoolShardingContext, values: KeyedJaggedTensor
    ) -> Awaitable[JaggedTensor]:
        return self._update_values_dist_impl(values, ctx)

    def create_context(self) -> ObjectPoolShardingContext:
        return cast(ObjectPoolShardingContext, self._sharding.create_context())


@torch.fx.wrap
def _get_reorder_values_lengths_weights(
    keys: List[str],
    jt: JaggedTensor,
    # not actually optional, just making torchscript type happy.
    unbucketize_permute: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    unbucketize_id_permute = (
        torch.arange(jt.lengths().shape[0], device=device)
        .view(-1, len(keys))[unbucketize_permute]
        .flatten()
    )
    row_major_to_feature_major_permute = (
        torch.arange(jt.lengths().shape[0], device=device)
        .view(-1, len(keys))
        .t()
        .flatten()
    )
    indices = unbucketize_id_permute[row_major_to_feature_major_permute]
    reorder_l = jt.lengths()[indices]
    reorder_o = torch.ops.fbgemm.asynchronous_inclusive_cumsum(reorder_l)
    reorder_v = jagged_index_select_with_empty(
        jt.values().unsqueeze(-1), indices, jt.offsets()[1:], reorder_o
    )
    reorder_w = (
        jagged_index_select_with_empty(
            jt.weights().unsqueeze(-1),
            indices,
            jt.offsets()[1:],
            reorder_o,
        ).flatten()
        if jt.weights_or_none() is not None
        else None
    )

    return (reorder_v.flatten(), reorder_l.flatten(), reorder_w)


class ShardedInferenceKeyedJaggedTensorPool(
    ShardedObjectPool[KeyedJaggedTensor, List[torch.Tensor], ObjectPoolShardingContext],
):
    _local_kjt_pool_shards: torch.nn.ModuleList
    _world_size: int
    _device: torch.device

    def __init__(
        self,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_env: ShardingEnv,
        sharding_plan: ObjectPoolShardingPlan,
        module: KeyedJaggedTensorPool,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self._pool_size = pool_size
        self._values_dtype = values_dtype
        self._sharding_env = sharding_env
        self._world_size = self._sharding_env.world_size
        self._device = device or torch.device("cuda")
        self._sharding_plan = sharding_plan

        self._is_weighted = is_weighted
        self._feature_max_lengths = feature_max_lengths

        torch._assert(
            self._sharding_plan.inference, "Plan needs to have inference enabled"
        )

        if self._sharding_plan.sharding_type == ObjectPoolShardingType.ROW_WISE:
            self._sharding = InferRwKeyedJaggedTensorPoolSharding(
                env=self._sharding_env,
                device=self._device,
                pool_size=self._pool_size,
            )
        else:
            raise NotImplementedError(
                f"Sharding type {self._sharding_plan.sharding_type} is not implemented"
            )

        self._local_kjt_pool_shards = torch.nn.ModuleList()
        offset = 0
        for rank, this_rank_size in zip(
            range(self._world_size), self._sharding.local_pool_size_per_rank
        ):
            shard_device = (
                torch.device("cpu")
                if device == torch.device("cpu")
                else torch.device("cuda", rank)
            )
            self._local_kjt_pool_shards.append(
                TensorJaggedIndexSelectLookup(
                    this_rank_size,
                    self._values_dtype,
                    feature_max_lengths,
                    self._is_weighted,
                    shard_device,
                )
            )
            if module._device != torch.device("meta"):
                # pyrefly: ignore[not-callable]
                self._local_kjt_pool_shards[rank]._values.copy_(
                    # pyrefly: ignore[bad-index]
                    module.values[offset : offset + this_rank_size]
                )
                # pyrefly: ignore[not-callable]
                self._local_kjt_pool_shards[rank]._key_lengths.copy_(
                    # pyrefly: ignore[bad-index]
                    module.key_lengths[offset : offset + this_rank_size]
                )
                # pyrefly: ignore[not-callable]
                jagged_lengths, jagged_offsets = self._local_kjt_pool_shards[
                    rank
                ]._infer_jagged_lengths_inclusive_offsets()
                self._local_kjt_pool_shards[rank]._jagged_lengths = jagged_lengths
                self._local_kjt_pool_shards[rank]._jagged_offsets = jagged_offsets
            offset += this_rank_size

        # TODO: move these to class type declarations
        #   this can be somewhat tricky w/ torchscript since these are
        #   abstract classes.
        self._lookup_ids_dist_impl: InferRwObjectPoolInputDist = torch.jit.annotate(
            InferRwObjectPoolInputDist,
            self._sharding.create_lookup_ids_dist(),
        )

        self._lookup_values_dist_impl: InferRwKeyedJaggedTensorPoolOutputDist = (
            torch.jit.annotate(
                InferRwKeyedJaggedTensorPoolOutputDist,
                self._sharding.create_lookup_values_dist(),
            )
        )

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def dim(self) -> int:
        # pyrefly: ignore[bad-return]
        return self._dim

    @property
    def dtype(self) -> torch.dtype:
        return self._values_dtype

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

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _lookup_local(
        self,
        dist_input: List[torch.Tensor],
    ) -> List[JaggedTensor]:
        ret = torch.jit.annotate(List[JaggedTensor], [])
        for i, shard in enumerate(self._local_kjt_pool_shards):
            ret.append(shard(dist_input[i]))
        return ret

    # pyrefly: ignore [bad-override, bad-param-name-override]
    def _lookup_values_dist(
        self,
        lookups: List[JaggedTensor],
    ) -> JaggedTensor:
        return self._lookup_values_dist_impl(lookups)

    # pyrefly: ignore[bad-override]
    def forward(self, ids: torch.Tensor) -> KeyedJaggedTensor:
        dist_input, unbucketize_permute, _, _ = self._lookup_ids_dist(ids)
        lookup = self._lookup_local(dist_input)
        # Here we are playing a trick to workaround a fx tracing issue,
        # as proxy is not iteratable.
        lookup_list = []
        for i in range(self._world_size):
            lookup_list.append(lookup[i])

        jt = self._lookup_values_dist(lookup_list)
        keys = list(self._feature_max_lengths.keys())
        reorder_v, reorder_l, reorder_w = _get_reorder_values_lengths_weights(
            keys, jt, unbucketize_permute, self._device
        )

        ret = KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=reorder_v,
            weights=reorder_w,
            lengths=reorder_l,
        )
        return ret

    # pyrefly: ignore[bad-override]
    def _update_ids_dist(
        self,
        ctx: ObjectPoolShardingContext,
        ids: torch.Tensor,
    ) -> None:
        raise NotImplementedError("Inference does not currently support update")

    # pyrefly: ignore[bad-override]
    def _update_values_dist(self, ctx: ObjectPoolShardingContext, values: torch.Tensor):
        raise NotImplementedError("Inference does not currently support update")

    def _update_local(
        self,
        ctx: ObjectPoolShardingContext,
        ids: torch.Tensor,
        values: List[torch.Tensor],
    ) -> None:
        raise NotImplementedError("Inference does not support update")

    #  `None`.
    # pyrefly: ignore[bad-return]
    def _update_preproc(self, values: KeyedJaggedTensor) -> KeyedJaggedTensor:
        pass


class KeyedJaggedTensorPoolSharder(ModuleSharder[KeyedJaggedTensorPool]):
    def __init__(self) -> None:
        super().__init__()

    # pyrefly: ignore[bad-param-name-override]
    def shard(
        self,
        module: KeyedJaggedTensorPool,
        plan: ObjectPoolShardingPlan,
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> Union[ShardedKeyedJaggedTensorPool, ShardedInferenceKeyedJaggedTensorPool]:
        if plan.inference:
            return ShardedInferenceKeyedJaggedTensorPool(
                pool_size=module.pool_size,
                feature_max_lengths=module.feature_max_lengths,
                values_dtype=module.values_dtype,
                is_weighted=module.is_weighted,
                sharding_env=env,
                sharding_plan=plan,
                module=module,
                device=device,
            )
        return ShardedKeyedJaggedTensorPool(
            module.pool_size,
            module.feature_max_lengths,
            module.values_dtype,
            module.is_weighted,
            sharding_plan=plan,
            sharding_env=env,
            device=device,
            enable_uvm=module._enable_uvm,
        )

    @property
    def module_type(self) -> Type[KeyedJaggedTensorPool]:
        return KeyedJaggedTensorPool
