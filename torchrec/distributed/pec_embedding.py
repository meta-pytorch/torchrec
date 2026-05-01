#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.pec_embedding_modules import (
    OverlappingCheckerType,
    PECEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class PECEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Per-module context for ShardedPECEmbeddingCollection.

    Inherits from EmbeddingCollectionContext so it can be passed directly
    to the inner ShardedEmbeddingCollection's compute/output_dist methods.
    PEC-specific fields will be added in later diffs.
    """

    pass


class ShardedPECEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,  # CompIn
        List[torch.Tensor],  # DistOut
        Dict[str, JaggedTensor],  # Out
        PECEmbeddingCollectionContext,  # ShrdCtx
    ],
):
    """Sharded PEC Embedding Collection. Wraps ShardedEmbeddingCollection via composition."""

    def __init__(
        self,
        module: PECEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ec_sharder: EmbeddingCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

        # Shard the inner EmbeddingCollection via composition
        self._embedding_collection: ShardedEmbeddingCollection = ec_sharder.shard(
            module._embedding_collection,
            table_name_to_parameter_sharding,
            env=env,
            device=device,
        )

        # PEC config
        self._checker_type: OverlappingCheckerType = module._checker_type

    def create_context(self) -> PECEmbeddingCollectionContext:
        return PECEmbeddingCollectionContext()

    def input_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._embedding_collection.input_dist(ctx, features)

    def compute(
        self,
        ctx: PECEmbeddingCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        return self._embedding_collection.compute(ctx, dist_input)

    def output_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        return self._embedding_collection.output_dist(ctx, output)

    def compute_and_output_dist(
        self,
        ctx: PECEmbeddingCollectionContext,
        input: KJTList,
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        return self._embedding_collection.compute_and_output_dist(ctx, input)


class PECEmbeddingCollectionSharder(BaseEmbeddingSharder[PECEmbeddingCollection]):
    """Sharder for PECEmbeddingCollection. Enforces RW-only sharding."""

    def __init__(
        self,
        ec_sharder: Optional[EmbeddingCollectionSharder] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry)
        self._embedding_collection_sharder: EmbeddingCollectionSharder = (
            ec_sharder
            or EmbeddingCollectionSharder(
                fused_params=fused_params,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    @property
    def module_type(self) -> Type[PECEmbeddingCollection]:
        return PECEmbeddingCollection

    def shardable_parameters(
        self, module: PECEmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return self._embedding_collection_sharder.shardable_parameters(
            module._embedding_collection
        )

    def shard(
        self,
        module: PECEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> "ShardedPECEmbeddingCollection":
        if device is None:
            device = torch.device("cuda")
        return ShardedPECEmbeddingCollection(
            module,
            params,
            ec_sharder=self._embedding_collection_sharder,
            env=env,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
