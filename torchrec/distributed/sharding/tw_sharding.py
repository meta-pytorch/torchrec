#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, cast, Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed._tensor.placement_types import Replicate
from torchrec.distributed.dist_data import (
    EmbeddingsAllToOne,
    KJTAllToAll,
    KJTOneToAll,
    PooledEmbeddingsAllToAll,
    VariableBatchPooledEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    group_tables,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    DTensorMetadata,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    InputDistOutputs,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import (
    Awaitable,
    CommOp,
    NullShardingContext,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardingEnv2D,
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


class BaseTwEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for table wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        storage_device_type_from_sharding_infos: Optional[str] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._env: ShardingEnv = env
        self._device: Optional[torch.device] = device
        self._is_2D_parallel: bool = isinstance(env, ShardingEnv2D)
        self._pg: Optional[dist.ProcessGroup] = (
            self._env.sharding_pg  # pyre-ignore[16]
            if self._is_2D_parallel
            else self._env.process_group
        )
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank

        sharded_tables_per_rank = self._shard(sharding_infos)

        self._sharded_tables_per_rank: List[List[ShardedEmbeddingTable]] = (
            sharded_tables_per_rank
        )

        self._grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]] = (
            []
        )
        self._grouped_embedding_configs_per_rank = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs: List[GroupedEmbeddingConfig] = (
            self._grouped_embedding_configs_per_rank[self._rank]
        )
        self._storage_device_type_from_sharding_infos: Optional[str] = (
            storage_device_type_from_sharding_infos
        )

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._world_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for _ in range(world_size)
        ]
        for info in sharding_infos:
            # pyre-fixme [16]
            shards = info.param_sharding.sharding_spec.shards
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
                    placements=(Replicate(),)
                    * (self._env.device_mesh.ndim),  # pyre-ignore[16]
                    size=(
                        info.embedding_config.num_embeddings,
                        info.embedding_config.embedding_dim,
                    ),
                    stride=info.param.stride(),
                )

            rank = (
                # pyre-ignore [16]
                self._env.remap_rank(
                    info.param_sharding.ranks[0],  # pyre-ignore[16]
                    ShardingType.TABLE_WISE,
                )
                if self._is_2D_parallel
                else info.param_sharding.ranks[0]
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
                        none_throws(info.embedding_config.num_embeddings_post_pruning)
                        if info.embedding_config.num_embeddings_post_pruning is not None
                        else info.embedding_config.num_embeddings
                    ),
                    local_cols=info.embedding_config.embedding_dim,
                    compute_kernel=EmbeddingComputeKernel(
                        info.param_sharding.compute_kernel
                    ),
                    local_metadata=shards[0],
                    global_metadata=global_metadata,
                    dtensor_metadata=dtensor_metadata,
                    weight_init_max=info.embedding_config.weight_init_max,
                    weight_init_min=info.embedding_config.weight_init_min,
                    fused_params=info.fused_params,
                    num_embeddings_post_pruning=info.embedding_config.num_embeddings_post_pruning,
                    use_virtual_table=info.embedding_config.use_virtual_table,
                )
            )
        return tables_per_rank

    def _dim_sum_per_rank(self) -> List[int]:
        dim_sum_per_rank = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            dim_sum = 0
            for grouped_config in grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            dim_sum_per_rank.append(dim_sum)
        return dim_sum_per_rank

    def _emb_dim_per_rank_per_feature(self) -> List[List[int]]:
        emb_dim_per_rank_per_feature = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            emb_dim_per_feature = []
            for grouped_config in grouped_embedding_configs:
                emb_dim_per_feature += grouped_config.embedding_dims()
            emb_dim_per_rank_per_feature.append(emb_dim_per_feature)
        return emb_dim_per_rank_per_feature

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())
        return embedding_names

    def embedding_names_per_rank(self) -> List[List[str]]:
        embedding_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            embedding_names_per_rank = []
            for grouped_config in grouped_embedding_configs:
                embedding_names_per_rank.extend(grouped_config.embedding_names())
            embedding_names.append(embedding_names_per_rank)
        return embedding_names

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                embedding_shard_metadata.extend(
                    grouped_config.embedding_shard_metadata()
                )
        return embedding_shard_metadata

    def feature_names(self) -> List[str]:
        feature_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                feature_names.extend(grouped_config.feature_names())
        return feature_names

    def embedding_tables(self) -> List[ShardedEmbeddingTable]:
        embedding_tables = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                embedding_tables.extend(grouped_config.embedding_tables)
        return embedding_tables

    def feature_names_per_rank(self) -> List[List[str]]:
        feature_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            feature_names_per_rank = []
            for grouped_config in grouped_embedding_configs:
                feature_names_per_rank.extend(grouped_config.feature_names())
            feature_names.append(feature_names_per_rank)
        return feature_names

    def features_per_rank(self) -> List[int]:
        features_per_rank = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            num_features = 0
            for grouped_config in grouped_embedding_configs:
                num_features += grouped_config.num_features()
            features_per_rank.append(num_features)
        return features_per_rank


class TwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Redistributes sparse features with an AlltoAll collective operation for table wise
    sharding.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        features_per_rank (List[int]): number of features to send to each rank.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
    ) -> None:
        super().__init__()
        self._dist = KJTAllToAll(
            pg=pg,
            splits=features_per_rank,
        )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Performs AlltoAll operation on sparse features.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        return self._dist(sparse_features)


class TwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor with an AlltoAll collective operation for
    table wise sharding.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        emb_dim_per_rank_per_feature (List[List[int]]): embedding dimension per rank per
            feature, used for variable batch per feature.
        device (Optional[torch.device]): device on which buffers will be allocated.
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]):
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]):
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        emb_dim_per_rank_per_feature: List[List[int]],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._dim_sum_per_rank = dim_sum_per_rank
        self._device = device
        self._callbacks = callbacks
        self._codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name, None)
            if qcomm_codecs_registry
            else None
        )
        self._emb_dim_per_rank_per_feature = emb_dim_per_rank_per_feature
        self._dist: Optional[PooledEmbeddingsAllToAll] = None
        self._variable_dist: Optional[VariableBatchPooledEmbeddingsAllToAll] = None

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            sharding_ctx (Optional[EmbeddingShardingContext]): shared context from
                KJTAllToAll operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings.
        """
        if self._dist is None:
            self._create_output_dist_module(sharding_ctx)

        if sharding_ctx is None:
            return cast(PooledEmbeddingsAllToAll, self._dist)(local_embs)
        elif sharding_ctx.variable_batch_per_feature:
            assert (
                self._variable_dist is not None
            ), "variable batch dist is not initialized!"
            return self._variable_dist(
                local_embs,
                batch_size_per_rank_per_feature=sharding_ctx.batch_size_per_rank_per_feature,
                batch_size_per_feature_pre_a2a=sharding_ctx.batch_size_per_feature_pre_a2a,
            )
        else:
            return cast(PooledEmbeddingsAllToAll, self._dist)(
                local_embs,
                batch_size_per_rank=sharding_ctx.batch_size_per_rank,
            )

    def _create_output_dist_module(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._variable_dist = VariableBatchPooledEmbeddingsAllToAll(
                pg=self._pg,
                emb_dim_per_rank_per_feature=self._emb_dim_per_rank_per_feature,
                device=self._device,
                callbacks=None,
                codecs=self._codecs,
            )
        self._dist = PooledEmbeddingsAllToAll(
            pg=self._pg,
            dim_sum_per_rank=self._dim_sum_per_rank,
            device=self._device,
            callbacks=self._callbacks,
            codecs=self._codecs,
        )


class TwPooledEmbeddingSharding(
    BaseTwEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise, i.e.. a given embedding table is entirely placed
    on a selected rank.
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
        assert self._pg is not None
        return TwPooledEmbeddingDist(
            pg=self._pg,
            dim_sum_per_rank=self._dim_sum_per_rank(),
            emb_dim_per_rank_per_feature=self._emb_dim_per_rank_per_feature(),
            device=device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )


class InferTwSparseFeaturesDist(BaseSparseFeaturesDist[InputDistOutputs]):
    """
    Redistributes sparse features to all devices for inference.

    Args:
        features_per_rank (List[int]): number of features to send to each rank.
        world_size (int): number of devices in the topology.
        fused_params (Dict[str, Any]): fused parameters of the model.
    """

    def __init__(
        self,
        features_per_rank: List[int],
        world_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = KJTOneToAll(
            splits=features_per_rank,
            world_size=world_size,
            device=device,
        )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> InputDistOutputs:
        """
        Performs OnetoAll operation on sparse features.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """
        return InputDistOutputs(features=self._dist.forward(sparse_features))


class InferTwPooledEmbeddingDist(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    """
    Merges pooled embedding tensor from each device for inference.

    Args:
        device (Optional[torch.device]): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOne = EmbeddingsAllToOne(device, world_size, 1)

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> torch.Tensor:
        """
        Performs AlltoOne operation on pooled embedding tensors.

        Args:
            local_embs (List[torch.Tensor]): pooled embedding tensors with
                `len(local_embs) == world_size`.

        Returns:
            torch.Tensor: merged pooled embedding tensor.
        """

        return self._dist(local_embs)


class InferTwEmbeddingSharding(
    BaseTwEmbeddingSharding[
        NullShardingContext, InputDistOutputs, List[torch.Tensor], torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise for inference
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
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
        return InferGroupedPooledEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
            device=device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]:
        device = device if device is not None else self._device
        assert device is not None
        return InferTwPooledEmbeddingDist(
            device,
            self._world_size,
        )
