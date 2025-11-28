#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional

import torch
from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingsLookup,
    InferGroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    InputDistOutputs,
)
from torchrec.distributed.sharding.cw_sharding import BaseCwEmbeddingSharding
from torchrec.distributed.sharding.sequence_sharding import (
    InferSequenceShardingContext,
    SequenceShardingContext,
)
from torchrec.distributed.sharding.tw_sequence_sharding import TwSequenceEmbeddingDist
from torchrec.distributed.sharding.tw_sharding import (
    InferTwSparseFeaturesDist,
    TwSparseFeaturesDist,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class CwSequenceEmbeddingSharding(
    BaseCwEmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards sequence (unpooled) embeddings column-wise, i.e.. a given embedding is
    partitioned along its columns and placed on specified ranks.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        assert self._pg is not None
        return TwSparseFeaturesDist(
            self._pg,
            self.features_per_rank(),
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        assert feature_processor is None
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            env=self._env,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]:
        assert self._pg is not None
        return TwSequenceEmbeddingDist(
            self._pg,
            self.features_per_rank(),
            device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )


class InferCwSequenceEmbeddingSharding(
    BaseCwEmbeddingSharding[
        InferSequenceShardingContext,
        InputDistOutputs,
        List[torch.Tensor],
        List[torch.Tensor],
    ]
):
    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[InputDistOutputs]:
        return InferTwSparseFeaturesDist(
            features_per_rank=self.features_per_rank(),
            world_size=self._world_size,
            device=device if device is not None else self._device,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[InputDistOutputs, List[torch.Tensor]]:
        return InferGroupedEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]:
        device = device if device is not None else self._device
        assert device is not None

        dist_out = InferCwSequenceEmbeddingDist(
            device,
            self._world_size,
        )
        return dist_out


class InferCwSequenceEmbeddingDist(
    BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]
):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: SeqEmbeddingsAllToOne = SeqEmbeddingsAllToOne(
            device=device, world_size=world_size
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[InferSequenceShardingContext] = None,
    ) -> List[torch.Tensor]:
        return self._dist(local_embs)
