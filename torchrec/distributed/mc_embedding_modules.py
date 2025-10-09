#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding import (
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)

from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    _create_mean_pooling_divisor,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.mc_modules import (
    ManagedCollisionCollectionSharder,
    ShardedManagedCollisionCollection,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    Multistreamable,
    NoWait,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    BaseManagedCollisionEmbeddingCollection,
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


logger: logging.Logger = logging.getLogger(__name__)


ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


class BaseShardedManagedCollisionEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        Tuple[LazyAwaitable[KeyedTensor], LazyAwaitable[Optional[KeyedJaggedTensor]]],
        ShrdCtx,
    ]
):
    def __init__(
        self,
        module: Union[
            ManagedCollisionEmbeddingBagCollection, ManagedCollisionEmbeddingCollection
        ],
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        e_sharder: Union[EmbeddingBagCollectionSharder, EmbeddingCollectionSharder],
        mc_sharder: ManagedCollisionCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env

        if isinstance(module, ManagedCollisionEmbeddingBagCollection):
            assert isinstance(e_sharder, EmbeddingBagCollectionSharder)
            assert isinstance(module._embedding_module, EmbeddingBagCollection)
            self.bagged: bool = True

            self._embedding_module: ShardedEmbeddingBagCollection = e_sharder.shard(
                module._embedding_module,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
            )
        else:
            assert isinstance(e_sharder, EmbeddingCollectionSharder)
            assert isinstance(module._embedding_module, EmbeddingCollection)
            self.bagged: bool = False

            self._embedding_module: ShardedEmbeddingCollection = e_sharder.shard(
                module._embedding_module,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
            )
        # TODO: This is a hack since _embedding_module doesn't need input
        # dist, so eliminating it so all fused a2a will ignore it.
        # self._embedding_module._has_uninitialized_input_dist = False
        embedding_shardings = (
            self._embedding_module._embedding_shardings
            if isinstance(self._embedding_module, ShardedEmbeddingBagCollection)
            else list(self._embedding_module._sharding_type_to_sharding.values())
        )
        self._managed_collision_collection: ShardedManagedCollisionCollection = (
            mc_sharder.shard(
                module._managed_collision_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
                embedding_shardings=embedding_shardings,
                use_index_dedup=(
                    e_sharder._use_index_dedup
                    if isinstance(e_sharder, EmbeddingCollectionSharder)
                    else False
                ),
            )
        )
        self._return_remapped_features: bool = module._return_remapped_features
        self._allow_in_place_embed_weight_update: bool = (
            module._allow_in_place_embed_weight_update
        )

        # pyre-ignore
        self._table_to_tbe_and_index = {}
        for lookup in self._embedding_module._lookups:
            # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Tensor, Module]` is not
            #  a function.
            for emb_module in lookup._emb_modules:
                for table_idx, table in enumerate(emb_module._config.embedding_tables):
                    self._table_to_tbe_and_index[table.name] = (
                        emb_module._emb_module,
                        torch.tensor([table_idx], dtype=torch.int, device=self._device),
                    )
        self._buffer_ids: torch.Tensor = torch.tensor(
            [0], device=self._device, dtype=torch.int
        )

    # pyre-ignore
    def input_dist(
        self,
        ctx: ShrdCtx,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        # TODO: resolve incompatiblity with different contexts
        if self._embedding_module._has_uninitialized_input_dist:
            if isinstance(self._embedding_module, ShardedEmbeddingBagCollection):
                self._features_order = []
                # disable feature permutation in mc, because we should
                # permute features in mc-ebc before mean pooling callback.
                if self._managed_collision_collection._has_uninitialized_input_dist:
                    self._managed_collision_collection._create_input_dists(
                        input_feature_names=features.keys()
                    )
                    self._managed_collision_collection._has_uninitialized_input_dist = (
                        False
                    )
                    if self._managed_collision_collection._features_order:
                        self._features_order = (
                            self._managed_collision_collection._features_order
                        )
                        self._managed_collision_collection._features_order = []
                if self._embedding_module._has_mean_pooling_callback:
                    self._embedding_module._init_mean_pooling_callback(
                        features.keys(),
                        # pyre-ignore [16]
                        ctx.inverse_indices,
                    )
            self._embedding_module._has_uninitialized_input_dist = False
        if isinstance(self._embedding_module, ShardedEmbeddingBagCollection):
            with torch.no_grad():
                if self._features_order:
                    features = features.permute(
                        self._features_order,
                        self._managed_collision_collection._features_order_tensor,
                    )
                if self._embedding_module._has_mean_pooling_callback:
                    ctx.divisor = _create_mean_pooling_divisor(
                        lengths=features.lengths(),
                        stride=features.stride(),
                        keys=features.keys(),
                        offsets=features.offsets(),
                        pooling_type_to_rs_features=self._embedding_module._pooling_type_to_rs_features,
                        stride_per_key=features.stride_per_key(),
                        dim_per_key=self._embedding_module._dim_per_key,
                        embedding_names=self._embedding_module._embedding_names,
                        embedding_dims=self._embedding_module._embedding_dims,
                        # pyre-ignore [16]
                        variable_batch_per_feature=ctx.variable_batch_per_feature,
                        kjt_inverse_order=self._embedding_module._kjt_inverse_order,
                        kjt_key_indices=self._embedding_module._kjt_key_indices,
                        kt_key_ordering=self._embedding_module._kt_key_ordering,
                        inverse_indices=ctx.inverse_indices,
                        weights=features.weights_or_none(),
                    )
        return self._managed_collision_collection.input_dist(
            # pyre-fixme [6]
            ctx,
            features,
        )

    def _evict(self, evictions_per_table: Dict[str, Optional[torch.Tensor]]) -> None:
        open_slots = None
        for table, evictions_indices_for_table in evictions_per_table.items():
            if evictions_indices_for_table is not None:
                (tbe, logical_table_ids) = self._table_to_tbe_and_index[table]
                pruned_indices_offsets = torch.tensor(
                    [0, evictions_indices_for_table.shape[0]],
                    dtype=torch.long,
                    device=self._device,
                )
                if open_slots is None:
                    open_slots = self._managed_collision_collection.open_slots()
                logger.info(
                    f"Table {table}: inserting {evictions_indices_for_table.numel()} ids with {open_slots[table].item()} open slots"
                )
                with torch.no_grad():
                    # embeddings, and optimizer state will be reset
                    tbe.reset_embedding_weight_momentum(
                        pruned_indices=evictions_indices_for_table.long(),
                        pruned_indices_offsets=pruned_indices_offsets,
                        logical_table_ids=logical_table_ids,
                        buffer_ids=self._buffer_ids,
                    )

                    if self.bagged:
                        table_weight_param = (
                            # pyre-fixme[16]: Item `Tensor` of `Tensor | ModuleDict
                            #  | Module` has no attribute `get_parameter`.
                            self._embedding_module.embedding_bags.get_parameter(
                                f"{table}.weight"
                            )
                        )
                    else:
                        table_weight_param = (
                            # pyre-fixme[16]: Item `Tensor` of `Tensor | ModuleDict
                            #  | Module` has no attribute `get_parameter`.
                            self._embedding_module.embeddings.get_parameter(
                                f"{table}.weight"
                            )
                        )

                    init_fn = self._embedding_module._table_name_to_config[
                        table
                    ].init_fn
                    # Set evicted indices to original init_fn instead of all zeros
                    if self._allow_in_place_embed_weight_update:
                        # In-place update with .data to bypass PyTorch's autograd tracking.
                        # This is required for model training with multiple forward passes where the autograd graph
                        # is already created. Direct tensor modification would trigger PyTorch's in-place operation
                        # checks and invalidate gradients, while .data allows safe reinitialization of evicted
                        # embeddings without affecting the computational graph.
                        # pyre-ignore [29]
                        table_weight_param.data[evictions_indices_for_table] = init_fn(
                            table_weight_param[evictions_indices_for_table]
                        )
                    else:
                        # pyre-ignore [29]
                        table_weight_param[evictions_indices_for_table] = init_fn(
                            table_weight_param[evictions_indices_for_table]
                        )

    def compute(
        self,
        ctx: ShrdCtx,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        with record_function("## compute:mcc ##"):
            remapped_kjt = self._managed_collision_collection.compute(
                # pyre-fixme [6]
                ctx,
                dist_input,
            )
            evictions_per_table = self._managed_collision_collection.evict()

            self._evict(evictions_per_table)
            ctx.remapped_kjt = remapped_kjt
            ctx.evictions_per_table = evictions_per_table

            # pyre-ignore
            return self._embedding_module.compute(ctx, remapped_kjt)

    # pyre-ignore
    def output_dist(
        self,
        ctx: ShrdCtx,
        output: List[torch.Tensor],
    ) -> Tuple[LazyAwaitable[KeyedTensor], LazyAwaitable[Optional[KeyedJaggedTensor]]]:

        # pyre-ignore [6]
        ebc_awaitable = self._embedding_module.output_dist(ctx, output)

        if self._return_remapped_features:
            kjt_awaitable = self._managed_collision_collection.output_dist(
                # pyre-fixme [6]
                ctx,
                # pyre-ignore [16]
                ctx.remapped_kjt,
            )
        else:
            kjt_awaitable = NoWait(None)

        # pyre-ignore
        return ebc_awaitable, kjt_awaitable

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            yield append_prefix(prefix, fqn)
        for fqn, _ in self.named_buffers():
            yield append_prefix(prefix, fqn)


M = TypeVar("M", bound=BaseManagedCollisionEmbeddingCollection)


class BaseManagedCollisionEmbeddingCollectionSharder(BaseEmbeddingSharder[M]):
    def __init__(
        self,
        e_sharder: Union[EmbeddingBagCollectionSharder, EmbeddingCollectionSharder],
        mc_sharder: ManagedCollisionCollectionSharder,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._e_sharder: Union[
            EmbeddingBagCollectionSharder, EmbeddingCollectionSharder
        ] = e_sharder
        self._mc_sharder: ManagedCollisionCollectionSharder = mc_sharder

    def shardable_parameters(
        self, module: BaseManagedCollisionEmbeddingCollection
    ) -> Dict[str, torch.nn.Parameter]:
        # pyre-ignore
        return self._e_sharder.shardable_parameters(module._embedding_module)

    def compute_kernels(
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.FUSED.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.KEY_VALUE.value,
        ]

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return list(
            set.intersection(
                set(self._e_sharder.sharding_types(compute_device_type)),
                set(self._mc_sharder.sharding_types(compute_device_type)),
            )
        )

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        # TODO: to be deprecate after planner get cache_load_factor from ParameterConstraints
        return self._e_sharder.fused_params
