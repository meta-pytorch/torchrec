#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import torch
from torch import nn

from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import append_prefix, init_parameters
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.modules.fp_embedding_modules import (
    apply_feature_processors_to_kjt,
    FeatureProcessedEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def param_dp_sync(kt: KeyedTensor, no_op_tensor: torch.Tensor) -> KeyedTensor:
    kt._values.add_(no_op_tensor)
    return kt


class ShardedFeatureProcessedEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList, List[torch.Tensor], KeyedTensor, EmbeddingBagCollectionContext
    ]
):
    def __init__(
        self,
        module: FeatureProcessedEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env

        self._embedding_bag_collection: ShardedEmbeddingBagCollection = (
            ebc_sharder.shard(
                module._embedding_bag_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
                module_fqn=module_fqn,
            )
        )

        self._lookups: List[nn.Module] = self._embedding_bag_collection._lookups

        self._is_collection: bool = False
        self._feature_processors: Union[nn.ModuleDict, FeatureProcessorsCollection]
        if isinstance(module._feature_processors, FeatureProcessorsCollection):
            self._feature_processors = module._feature_processors
            self._is_collection = True
        else:
            self._feature_processors = torch.nn.ModuleDict(
                # pyre-fixme[29]: `Union[(self: ModuleDict) -> ItemsView[str,
                #  Module], Module, Tensor]` is not a function.
                module._feature_processors.items()
            )
            self._is_collection = False

        init_parameters(self._feature_processors, device)
        self._no_op_zero: torch.Tensor = torch.zeros((1,), device=self._device)

    # pyre-ignore
    def input_dist(
        self, ctx: EmbeddingBagCollectionContext, features: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._embedding_bag_collection.input_dist(ctx, features)

    def apply_feature_processors_to_kjt_list(self, dist_input: KJTList) -> KJTList:
        kjt_list = []
        for features in dist_input:
            if self._is_collection:
                kjt_list.append(self._feature_processors(features))
            else:
                kjt_list.append(
                    apply_feature_processors_to_kjt(
                        features,
                        self._feature_processors,
                    )
                )
        return KJTList(kjt_list)

    def compute(
        self,
        ctx: EmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:

        fp_features = self.apply_feature_processors_to_kjt_list(dist_input)
        return self._embedding_bag_collection.compute(ctx, fp_features)

    def output_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        lazy_awaitable_kt = self._embedding_bag_collection.output_dist(ctx, output)
        return self.add_fp_params_grad_sync_callback(lazy_awaitable_kt)

    def compute_and_output_dist(
        self, ctx: EmbeddingBagCollectionContext, input: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        fp_features = self.apply_feature_processors_to_kjt_list(input)
        lazy_awaitable_kt = self._embedding_bag_collection.compute_and_output_dist(
            ctx, fp_features
        )
        return self.add_fp_params_grad_sync_callback(lazy_awaitable_kt)

    def add_fp_params_grad_sync_callback(
        self, lazy_awaitable_kt: LazyAwaitable[KeyedTensor]
    ) -> LazyAwaitable[KeyedTensor]:
        # This will ensure that all feature processor parameters participate in the
        # autograd graph across all ranks. This will protect from mismatched collective
        # calls order when using DistributedDataParallel over feature processors.
        no_op_tensor = (
            self._no_op_zero
            * torch.cat(
                [x.flatten() for x in self._feature_processors.parameters()]
            ).sum()
        )
        lazy_awaitable_kt.callbacks.append(
            partial(param_dp_sync, no_op_tensor=no_op_tensor)
        )
        return lazy_awaitable_kt

    def create_context(self) -> EmbeddingBagCollectionContext:
        return self._embedding_bag_collection.create_context()

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            if "_embedding_bag_collection" in fqn:
                yield append_prefix(prefix, fqn)

    def _initialize_torch_state(self, skip_registering: bool = False) -> None:  # noqa
        self._embedding_bag_collection._initialize_torch_state(skip_registering)


class FeatureProcessedEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[FeatureProcessedEmbeddingBagCollection]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder
            or EmbeddingBagCollectionSharder(
                qcomm_codecs_registry=self.qcomm_codecs_registry
            )
        )

    def shard(
        self,
        module: FeatureProcessedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedFeatureProcessedEmbeddingBagCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedFeatureProcessedEmbeddingBagCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            env=env,
            device=device,
            module_fqn=module_fqn,
        )

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        # TODO: to be deprecate after planner get cache_load_factor from ParameterConstraints
        return self._ebc_sharder.fused_params

    def shardable_parameters(
        self, module: FeatureProcessedEmbeddingBagCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_bag_collection)

    @property
    def module_type(self) -> Type[FeatureProcessedEmbeddingBagCollection]:
        return FeatureProcessedEmbeddingBagCollection

    @property
    def sharded_module_type(
        self,
    ) -> Type[ShardedFeatureProcessedEmbeddingBagCollection]:
        return ShardedFeatureProcessedEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        if compute_device_type in {"mtia"}:
            return [ShardingType.TABLE_WISE.value, ShardingType.COLUMN_WISE.value]

        # No row wise because position weighted FP and RW don't play well together.
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]

        return types
