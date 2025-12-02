#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Optional

import torch
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
)
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.distributed.sharding.dp_sharding import (
    BaseDpEmbeddingSharding,
    DpSparseFeaturesDist,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable, NoWait
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class DpSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Distributes sequence embeddings to be data-parallel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        No-op as sequence embeddings are already distributed in data-parallel fashion.

        Args:
            local_embs (torch.Tensor): output sequence embeddings.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """

        return NoWait(local_embs)


class DpSequenceEmbeddingSharding(
    BaseDpEmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards sequence (unpooled) embedding data-parallel, with no table sharding i.e.. a
    given embedding table is replicated across all ranks.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        return DpSparseFeaturesDist()

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        assert feature_processor is None
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._env.process_group,
            device=device if device is not None else self._device,
            env=self._env,
        )

    def create_output_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]:
        return DpSequenceEmbeddingDist()
