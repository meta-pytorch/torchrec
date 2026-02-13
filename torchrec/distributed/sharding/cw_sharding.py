#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import torch
import torch.distributed as dist  # noqa
from fbgemm_gpu.permute_pooled_embedding_modules_split import (
    PermutePooledEmbeddingsSplit,
)
from torch.distributed._tensor import Replicate, Shard
from torchrec.distributed.dist_data import EmbeddingsAllToOne
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    DTensorMetadata,
    EmbeddingComputeKernel,
    InputDistOutputs,
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding.tw_sharding import (
    BaseTwEmbeddingSharding,
    InferTwSparseFeaturesDist,
    TwPooledEmbeddingDist,
    TwSparseFeaturesDist,
)
from torchrec.distributed.types import (
    NullShardingContext,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class BaseCwEmbeddingSharding(BaseTwEmbeddingSharding[C, F, T, W]):
    """
    Base class for column-wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            sharding_infos,
            env,
            device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
        self._permute_embeddings = permute_embeddings
        if self._permute_embeddings:
            self._init_combined_embeddings()

    def _init_combined_embeddings(self) -> None:
        """
        Grabs the embedding names and dims from TwEmbeddingSharder.

        NOTE:
            This could have duplications if there are multiple shards from the same
            table on a rank. Later on we process these to combine shards together.
        """

        embedding_names: List[str] = super().embedding_names()
        embedding_dims: List[int] = super().embedding_dims()

        embedding_shard_metadata: List[Optional[ShardMetadata]] = (
            super().embedding_shard_metadata()
        )

        embedding_name_to_index_offset_tuples: Dict[str, List[Tuple[int, int]]] = {}
        for i, (name, metadata) in enumerate(
            zip(embedding_names, embedding_shard_metadata)
        ):
            if name not in embedding_name_to_index_offset_tuples:
                embedding_name_to_index_offset_tuples[name] = []
            embedding_name_to_index_offset_tuples[name].append(
                (i, metadata.shard_offsets[1] if metadata is not None else 0)
            )

        embedding_name_to_index: Dict[str, List[int]] = {}
        for name, index_offset_tuples in embedding_name_to_index_offset_tuples.items():
            embedding_name_to_index[name] = [
                idx_off_tuple[0]
                for idx_off_tuple in sorted(
                    index_offset_tuples,
                    key=lambda idx_off_tuple: idx_off_tuple[1],
                )
            ]

        combined_embedding_names: List[str] = []
        seen_embedding_names: Set[str] = set()

        for name in embedding_names:
            if name not in seen_embedding_names:
                combined_embedding_names.append(name)
                seen_embedding_names.add(name)

        combined_embedding_dims: List[int] = []

        embedding_order: List[int] = []
        for name in combined_embedding_names:
            combined_embedding_dims.append(
                sum([embedding_dims[idx] for idx in embedding_name_to_index[name]])
            )
            embedding_order.extend(embedding_name_to_index[name])

        self._embedding_names: List[str] = embedding_names
        self._embedding_dims: List[int] = embedding_dims
        self._embedding_order: List[int] = embedding_order

        self._combined_embedding_names: List[str] = combined_embedding_names
        self._combined_embedding_dims: List[int] = combined_embedding_dims

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size: int = self._world_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for _ in range(world_size)
        ]
        for info in sharding_infos:
            # pyrefly: ignore[missing-attribute]
            shards: List[ShardMetadata] = info.param_sharding.sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size(
                    [
                        (
                            info.embedding_config.num_embeddings_post_pruning
                            if info.embedding_config.num_embeddings_post_pruning
                            is not None
                            else info.embedding_config.num_embeddings
                        ),
                        info.embedding_config.embedding_dim,
                    ]
                ),
            )

            dtensor_metadata = None
            if self._env.output_dtensor:
                dtensor_metadata = DTensorMetadata(
                    mesh=self._env.device_mesh,
                    placements=(
                        (Replicate(), Shard(1)) if self._is_2D_parallel else (Shard(1),)
                    ),
                    size=(
                        (
                            info.embedding_config.num_embeddings_post_pruning
                            if info.embedding_config.num_embeddings_post_pruning
                            is not None
                            else info.embedding_config.num_embeddings
                        ),
                        info.embedding_config.embedding_dim,
                    ),
                    stride=info.param.stride(),
                )

            # pyrefly: ignore[bad-argument-type]
            for i, rank in enumerate(info.param_sharding.ranks):
                # Remap rank by number of replica groups if 2D parallelism is enabled
                rank = (
                    # pyrefly: ignore[missing-attribute]
                    self._env.remap_rank(
                        rank,
                        ShardingType.COLUMN_WISE,
                    )
                    if self._is_2D_parallel
                    else rank
                )
                tables_per_rank[rank].append(
                    ShardedEmbeddingTable(
                        num_embeddings=info.embedding_config.num_embeddings,
                        embedding_dim=info.embedding_config.embedding_dim,
                        name=info.embedding_config.name,
                        embedding_names=info.embedding_config.embedding_names,
                        data_type=info.embedding_config.data_type,
                        feature_names=info.embedding_config.feature_names,
                        pooling=info.embedding_config.pooling,
                        is_weighted=info.embedding_config.is_weighted,
                        has_feature_processor=info.embedding_config.has_feature_processor,
                        local_rows=(
                            none_throws(
                                info.embedding_config.num_embeddings_post_pruning
                            )
                            if info.embedding_config.num_embeddings_post_pruning
                            is not None
                            else info.embedding_config.num_embeddings
                        ),
                        local_cols=shards[i].shard_sizes[1],
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[i],
                        global_metadata=global_metadata,
                        dtensor_metadata=dtensor_metadata,
                        fused_params=info.fused_params,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                        num_embeddings_post_pruning=info.embedding_config.num_embeddings_post_pruning,
                    )
                )

        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        return (
            self._combined_embedding_dims
            if self._permute_embeddings
            else self.uncombined_embedding_dims()
        )

    def embedding_names(self) -> List[str]:
        return (
            self._combined_embedding_names
            if self._permute_embeddings
            else self.uncombined_embedding_names()
        )

    def uncombined_embedding_dims(self) -> List[int]:
        return super().embedding_dims()

    def uncombined_embedding_names(self) -> List[str]:
        return super().embedding_names()


class CwPooledEmbeddingSharding(
    BaseCwEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags column-wise, i.e.. a given embedding table is partitioned
    along its columns and placed on specified ranks.
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
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            env=self._env,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]:
        device = device if device is not None else self._device
        embedding_permute_op: Optional[PermutePooledEmbeddingsSplit] = None
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
        if self._permute_embeddings and self._embedding_order != list(
            range(len(self._embedding_order))
        ):
            assert len(self._embedding_order) == len(self._embedding_dims)
            embedding_permute_op = PermutePooledEmbeddingsSplit(
                self._embedding_dims, self._embedding_order, device=device
            )
            callbacks = [embedding_permute_op]
        assert self._pg is not None
        return TwPooledEmbeddingDist(
            pg=self._pg,
            dim_sum_per_rank=self._dim_sum_per_rank(),
            emb_dim_per_rank_per_feature=self._emb_dim_per_rank_per_feature(),
            device=device,
            callbacks=callbacks,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )


class InferCwPooledEmbeddingSharding(
    BaseCwEmbeddingSharding[
        NullShardingContext, InputDistOutputs, List[torch.Tensor], torch.Tensor
    ]
):
    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[InputDistOutputs]:
        return InferTwSparseFeaturesDist(
            self.features_per_rank(),
            self._world_size,
            device if device is not None else self._device,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[InputDistOutputs, List[torch.Tensor]]:
        return InferGroupedPooledEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]:
        device = device if device is not None else self._device
        assert device is not None

        dist_out = InferCwPooledEmbeddingDist(
            device,
            self._world_size,
        )

        if self._permute_embeddings and self._embedding_order != list(
            range(len(self._embedding_order))
        ):
            return InferCwPooledEmbeddingDistWithPermute(
                device, self._world_size, self._embedding_dims, self._embedding_order
            )

        return dist_out


class InferCwPooledEmbeddingDist(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOne = EmbeddingsAllToOne(
            device=device, world_size=world_size, cat_dim=1
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> torch.Tensor:
        return self._dist.forward(
            local_embs,
        )


@torch.fx.wrap
def _fx_wrap_permute(
    permute_module: PermutePooledEmbeddingsSplit, input: torch.Tensor
) -> torch.Tensor:
    return permute_module.forward(input)


class InferCwPooledEmbeddingDistWithPermute(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        embedding_dims: List[int],
        permute: List[int],
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOne = EmbeddingsAllToOne(
            device=device, world_size=world_size, cat_dim=1
        )
        self._permute: PermutePooledEmbeddingsSplit = PermutePooledEmbeddingsSplit(
            embs_dims=embedding_dims,
            permute=permute,
            device=device,
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> torch.Tensor:
        return self._permute(self._dist(local_embs))
