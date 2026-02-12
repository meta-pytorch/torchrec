#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    SeqEmbeddingsAllToOne,
    SequenceEmbeddingsAllToAll,
)
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
from torchrec.distributed.sharding.sequence_sharding import (
    InferSequenceShardingContext,
    SequenceShardingContext,
)
from torchrec.distributed.sharding.tw_sharding import (
    BaseTwEmbeddingSharding,
    InferTwSparseFeaturesDist,
    TwSparseFeaturesDist,
)
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs
from torchrec.modules.utils import _fx_trec_get_feature_length
from torchrec.quant.embedding_modules import _get_batching_hinted_output
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

torch.fx.wrap("_get_batching_hinted_output")
torch.fx.wrap("_fx_trec_get_feature_length")


class TwSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes sequence embedding tensor in TW fashion with an AlltoAll operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        features_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding for each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingsAllToAll(
            pg,
            features_per_rank,
            device,
            codecs=(
                qcomm_codecs_registry.get(
                    CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name, None
                )
                if qcomm_codecs_registry
                else None
            ),
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            sharding_ctx (SequenceShardingContext): shared context from KJTAllToAll
                operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """

        assert sharding_ctx is not None
        return self._dist(
            local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            batch_size_per_rank=sharding_ctx.batch_size_per_rank,
            sparse_features_recat=sharding_ctx.sparse_features_recat,
            unbucketize_permute_tensor=None,
        )


class TwSequenceEmbeddingSharding(
    BaseTwEmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards sequence (unpooled) embedding table-wise, i.e.. a given embedding table is
    placed entirely on a selected rank.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        return TwSparseFeaturesDist(
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
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


class InferTwSequenceEmbeddingDist(
    BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]
):
    """
    Redistributes sequence embedding tensor in hierarchical fashion with an AlltoOne
    operation.

    Args:
        device (torch.device): device on which the tensors will be communicated to.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        storage_device_type_from_sharding_infos: Optional[str] = None,
        embedding_names_per_rank: Optional[List[List[str]]] = None,
    ) -> None:
        super().__init__()
        self._adjusted_world_size: int = (
            world_size
            if embedding_names_per_rank is None
            else sum(1 for sublist in embedding_names_per_rank if len(sublist) > 0)
        )
        self._dist: SeqEmbeddingsAllToOne = SeqEmbeddingsAllToOne(
            device, self._adjusted_world_size
        )
        self._storage_device_type_from_sharding_infos: Optional[str] = (
            storage_device_type_from_sharding_infos
        )
        self._embedding_names_per_rank = embedding_names_per_rank

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[InferSequenceShardingContext] = None,
    ) -> List[torch.Tensor]:
        """
        Performs AlltoOne operation on sequence embeddings tensor.

        Args:
            local_embs (List[orch.Tensor]): tensor of values to distribute.
            sharding_ctx (InferSequenceShardingContext): shared context from KJTAllToOne
                operation.


        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """
        assert (
            sharding_ctx is not None
        ), "sharding ctx should not be None for InferTwSequenceEmbeddingDist"
        if (
            self._storage_device_type_from_sharding_infos is None
            or self._storage_device_type_from_sharding_infos not in ["cpu", "ssd"]
        ):
            local_embs = [
                _get_batching_hinted_output(
                    _fx_trec_get_feature_length(
                        sharding_ctx.features[i],
                        # pyrefly: ignore[unsupported-operation]
                        sharding_ctx.embedding_names_per_rank[i],
                    ),
                    local_emb,
                )
                for i, local_emb in enumerate(local_embs)
                if self._embedding_names_per_rank is not None
                and len(self._embedding_names_per_rank[i]) > 0
            ]
            return self._dist(local_embs)
        else:
            return local_embs


class InferTwSequenceEmbeddingSharding(
    BaseTwEmbeddingSharding[
        InferSequenceShardingContext,
        InputDistOutputs,
        List[torch.Tensor],
        List[torch.Tensor],
    ]
):
    """
    Shards sequence (unpooled) embedding table-wise, i.e.. a given embedding table is
    placed entirely on a selected rank, for inference.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[InputDistOutputs]:
        return InferTwSparseFeaturesDist(
            features_per_rank=self.features_per_rank(),
            world_size=self._world_size,
            device=device,
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
            device=device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]:
        device = device if device is not None else self._device
        return InferTwSequenceEmbeddingDist(
            # pyrefly: ignore[bad-argument-type]
            device,
            self._world_size,
            self._storage_device_type_from_sharding_infos,
            self.embedding_names_per_rank(),
        )
