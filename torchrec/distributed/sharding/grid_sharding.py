#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from fbgemm_gpu.permute_pooled_embedding_modules_split import (
    PermutePooledEmbeddingsSplit,
)
from torch.distributed._tensor import Replicate, Shard
from torchrec.distributed.comm import (
    get_local_size,
    intra_and_cross_node_pg,
    intra_and_cross_node_pg_2D,
)
from torchrec.distributed.dist_data import (
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
    VariableBatchPooledEmbeddingsAllToAll,
    VariableBatchPooledEmbeddingsReduceScatter,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
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
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding.twrw_sharding import TwRwSparseFeaturesDist
from torchrec.distributed.types import (
    Awaitable,
    CommOp,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardingEnv2D,
    ShardingType,
    ShardMetadata,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class BaseGridEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for grid sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._env: ShardingEnv = env
        self._is_2D_parallel: bool = isinstance(env, ShardingEnv2D)
        self._pg: Optional[dist.ProcessGroup] = (
            self._env.sharding_pg if self._is_2D_parallel else self._env.process_group
        )
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        self._device = device
        self._need_pos = need_pos
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._embedding_order: List[int] = []

        self._combined_embedding_names: List[str] = []
        self._combined_embedding_dims: List[int] = []

        if self._is_2D_parallel:
            intra_pg, cross_pg = intra_and_cross_node_pg_2D(
                self._env,
                device=device,
            )
        else:
            intra_pg, cross_pg = intra_and_cross_node_pg(
                device, backend=dist.get_backend(self._pg)
            )
        self._intra_pg: Optional[dist.ProcessGroup] = intra_pg
        self._cross_pg: Optional[dist.ProcessGroup] = cross_pg
        self._local_size: int = (
            intra_pg.size() if intra_pg else get_local_size(self._world_size)
        )

        sharded_tables_per_rank = self._shard(sharding_infos)
        self._grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]] = (
            []
        )
        self._grouped_embedding_configs_per_node: List[List[GroupedEmbeddingConfig]] = (
            []
        )
        self._grouped_embedding_configs_per_rank = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs_per_node = [
            self._grouped_embedding_configs_per_rank[rank]
            for rank in range(self._world_size)
            if rank % self._local_size == 0
        ]
        self._has_feature_processor: bool = False
        for group_config in self._grouped_embedding_configs_per_rank[
            self._rank // self._local_size
        ]:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

        self._init_combined_embeddings()

    def _init_combined_embeddings(self) -> None:
        """
        Initializes combined embeddings, similar to the CW sharding implementation,
        but in this case the CW shard is treated on a per node basis and not per rank.
        """
        embedding_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_node:
            for grouped_config in grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())

        embedding_dims = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_node:
            for grouped_config in grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())

        embedding_shard_metadata = self.embedding_shard_metadata()

        embedding_name_to_index_offset_tuples: Dict[str, List[Tuple[int, int]]] = {}
        for i, (name, metadata) in enumerate(
            zip(embedding_names, embedding_shard_metadata)
        ):
            if name not in embedding_name_to_index_offset_tuples:
                embedding_name_to_index_offset_tuples[name] = []
            # find index of each of the offset by column (CW sharding so only col dim changes)
            embedding_name_to_index_offset_tuples[name].append(
                (i, metadata.shard_offsets[1] if metadata is not None else 0)
            )

        # sort the index offset tuples by offset and then grab the associated index
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
        """
        Shards the embedding tables.
        This method takes the sharding infos and returns a list of lists of
        sharded embedding tables, where each inner list represents the tables
        for a specific rank.

        Args:
            sharding_infos (List[EmbeddingShardingInfo]): The sharding infos.
        Returns:
            List[List[ShardedEmbeddingTable]]: The sharded embedding tables.
        """
        world_size = self._world_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for _ in range(world_size)
        ]
        for info in sharding_infos:
            shards = info.param_sharding.sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size(
                    [
                        info.embedding_config.num_embeddings,
                        info.embedding_config.embedding_dim,
                    ]
                ),
            )

            dtensor_metadata = None
            if self._env.output_dtensor:
                placements = (
                    (Replicate(), Shard(1)) if self._is_2D_parallel else (Shard(1),)
                )
                dtensor_metadata = DTensorMetadata(
                    mesh=self._env.device_mesh,
                    placements=placements,
                    size=(
                        info.embedding_config.num_embeddings,
                        info.embedding_config.embedding_dim,
                    ),
                    stride=info.param.stride(),
                )

            # Expectation is planner CW shards across a node, so each CW shard will have local_size number of row shards
            for i, rank in enumerate(info.param_sharding.ranks):
                rank = (
                    self._env.remap_rank(rank, ShardingType.GRID_SHARD)
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
                        local_rows=shards[i].shard_sizes[0],
                        local_cols=shards[i].shard_sizes[1],
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[i],
                        global_metadata=global_metadata,
                        dtensor_metadata=dtensor_metadata,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                        fused_params=info.fused_params,
                    )
                )

        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        return self._combined_embedding_dims

    def embedding_names(self) -> List[str]:
        return self._combined_embedding_names

    def embedding_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                embedding_shard_metadata.extend(config.embedding_shard_metadata())
        return embedding_shard_metadata

    def feature_names(self) -> List[str]:
        feature_names = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                feature_names.extend(config.feature_names())
        return feature_names

    def _get_feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes: List[int] = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                feature_hash_sizes.extend(config.feature_hash_sizes())
        return feature_hash_sizes

    def _dim_sum_per_node(self) -> List[int]:
        dim_sum_per_node = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_node:
            dim_sum = 0
            for grouped_config in grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            dim_sum_per_node.append(dim_sum)
        return dim_sum_per_node

    def _emb_dim_per_node_per_feature(self) -> List[List[int]]:
        emb_dim_per_node_per_feature = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_node:
            emb_dim_per_feature = []
            for grouped_config in grouped_embedding_configs:
                emb_dim_per_feature += grouped_config.embedding_dims()
            emb_dim_per_node_per_feature.append(emb_dim_per_feature)
        return emb_dim_per_node_per_feature

    def _features_per_rank(
        self, group: List[List[GroupedEmbeddingConfig]]
    ) -> List[int]:
        features_per_rank = []
        for grouped_embedding_configs in group:
            num_features = 0
            for grouped_config in grouped_embedding_configs:
                num_features += grouped_config.num_features()
            features_per_rank.append(num_features)
        return features_per_rank


class GridPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        rank: int,
        cross_pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        dim_sum_per_node: List[int],
        emb_dim_per_node_per_feature: List[List[int]],
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self._rank = rank
        self._intra_pg: dist.ProcessGroup = intra_pg
        self._cross_pg: dist.ProcessGroup = cross_pg
        self._dim_sum_per_node = dim_sum_per_node
        self._emb_dim_per_node_per_feature = emb_dim_per_node_per_feature
        self._device = device
        self._intra_codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None
        )
        self._cross_codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name, None)
            if qcomm_codecs_registry
            else None
        )
        self._intra_dist: Optional[
            Union[
                PooledEmbeddingsReduceScatter,
                VariableBatchPooledEmbeddingsReduceScatter,
            ]
        ] = None
        self._cross_dist: Optional[
            Union[
                PooledEmbeddingsAllToAll,
                VariableBatchPooledEmbeddingsAllToAll,
            ]
        ] = None
        self._callbacks = callbacks

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor followed by
        AlltoAll pooled operation.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """
        if self._intra_dist is None or self._cross_dist is None:
            self._create_output_dist_modules(sharding_ctx)
        local_rank = self._rank % self._intra_pg.size()
        if sharding_ctx is not None and len(set(sharding_ctx.batch_size_per_rank)) > 1:
            # preprocess batch_size_per_rank
            (
                batch_size_per_rank_by_cross_group,
                batch_size_sum_by_cross_group,
            ) = self._preprocess_batch_size_per_rank(
                self._intra_pg.size(),
                self._cross_pg.size(),
                sharding_ctx.batch_size_per_rank,
            )
            # Perform ReduceScatterV within one host
            rs_result = cast(PooledEmbeddingsReduceScatter, self._intra_dist)(
                local_embs, input_splits=batch_size_sum_by_cross_group
            ).wait()
            return cast(PooledEmbeddingsAllToAll, self._cross_dist)(
                rs_result,
                batch_size_per_rank=batch_size_per_rank_by_cross_group[local_rank],
            )
        else:
            return cast(PooledEmbeddingsAllToAll, self._cross_dist)(
                cast(PooledEmbeddingsReduceScatter, self._intra_dist)(local_embs).wait()
            )

    def _preprocess_batch_size_per_rank(
        self, local_size: int, nodes: int, batch_size_per_rank: List[int]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Reorders `batch_size_per_rank` so it's aligned with reordered features after
        AlltoAll.
        """
        batch_size_per_rank_by_cross_group: List[List[int]] = []
        batch_size_sum_by_cross_group: List[int] = []
        for local_rank in range(local_size):
            batch_size_per_rank_: List[int] = []
            batch_size_sum = 0
            for node in range(nodes):
                batch_size_per_rank_.append(
                    batch_size_per_rank[local_rank + node * local_size]
                )
                batch_size_sum += batch_size_per_rank[local_rank + node * local_size]
            batch_size_per_rank_by_cross_group.append(batch_size_per_rank_)
            batch_size_sum_by_cross_group.append(batch_size_sum)

        return batch_size_per_rank_by_cross_group, batch_size_sum_by_cross_group

    def _create_output_dist_modules(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        self._intra_dist = PooledEmbeddingsReduceScatter(
            pg=self._intra_pg,
            codecs=self._intra_codecs,
        )
        self._cross_dist = PooledEmbeddingsAllToAll(
            pg=self._cross_pg,
            dim_sum_per_rank=self._dim_sum_per_node,
            device=self._device,
            codecs=self._cross_codecs,
            callbacks=self._callbacks,
        )


class GridPooledEmbeddingSharding(
    BaseGridEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags into column wise shards and shards each CW shard table wise row wise within a node
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        feature_hash_sizes = self._get_feature_hash_sizes()
        assert self._pg is not None
        assert self._intra_pg is not None
        return TwRwSparseFeaturesDist(
            pg=self._pg,
            local_size=self._intra_pg.size(),
            features_per_rank=features_per_rank,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._rank],
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.TABLE_ROW_WISE,
            env=self._env,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]:
        embedding_permute_op: Optional[PermutePooledEmbeddingsSplit] = None
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
        if self._embedding_order != list(range(len(self._embedding_order))):
            assert len(self._embedding_order) == len(self._embedding_dims)
            embedding_permute_op = PermutePooledEmbeddingsSplit(
                self._embedding_dims, self._embedding_order, device=self._device
            )
            callbacks = [embedding_permute_op]
        return GridPooledEmbeddingDist(
            rank=self._rank,
            cross_pg=cast(dist.ProcessGroup, self._cross_pg),
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            dim_sum_per_node=self._dim_sum_per_node(),
            emb_dim_per_node_per_feature=self._emb_dim_per_node_per_feature(),
            device=device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            callbacks=callbacks,
        )
