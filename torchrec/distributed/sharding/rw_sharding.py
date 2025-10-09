#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math

from typing import Any, cast, Dict, List, Optional, overload, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor.placement_types import Replicate, Shard
from torchrec.distributed.dist_data import (
    EmbeddingsAllToOneReduce,
    KJEAllToAll,
    KJTAllToAll,
    KJTOneToAll,
    PooledEmbeddingsReduceScatter,
    VariableBatchPooledEmbeddingsReduceScatter,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    BaseSparseFeaturesWriteDist,
    bucketize_embeddings_before_all2all_write,
    bucketize_kjt_before_all2all,
    bucketize_kjt_inference,
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
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

logger: logging.Logger = logging.getLogger(__name__)

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


def get_embedding_shard_metadata(
    grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]],
) -> Tuple[List[List[int]], bool]:
    is_even_sharding: bool = True
    world_size = len(grouped_embedding_configs_per_rank)

    def get_even_shard_sizes(hash_size: int, world_size: int) -> List[int]:
        block_size: int = math.ceil(hash_size / world_size)
        last_rank: int = hash_size // block_size

        expected_even_shard_sizes = [block_size] * last_rank
        if hash_size % world_size != 0:
            expected_even_shard_sizes.append(hash_size - sum(expected_even_shard_sizes))
        return expected_even_shard_sizes

    embed_sharding = []
    for group in grouped_embedding_configs_per_rank[0]:
        for table in group.embedding_tables:
            embed_sharding_per_feature = []
            total_rows = 0
            sizes = []
            # pyre-ignore [16]: `Optional` has no attribute `shards_metadata`
            for metadata in table.global_metadata.shards_metadata:
                embed_sharding_per_feature.append(metadata.shard_offsets[0])
                total_rows += metadata.shard_sizes[0]
                sizes.append(metadata.shard_sizes[0])
            embed_sharding_per_feature.append(total_rows)
            embed_sharding.extend(
                [embed_sharding_per_feature] * len(table.embedding_names)
            )
            expected_even_sizes = get_even_shard_sizes(total_rows, world_size)
            if sizes != expected_even_sizes:
                is_even_sharding = False

    return (embed_sharding, is_even_sharding)


class BaseRwEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for row-wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._env = env
        self._is_2D_parallel: bool = isinstance(env, ShardingEnv2D)
        self._pg: Optional[dist.ProcessGroup] = (
            self._env.sharding_pg  # pyre-ignore[16]
            if self._is_2D_parallel
            else self._env.process_group
        )
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        if device is None:
            device = torch.device("cpu")
        self._device: torch.device = device
        self._device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = (
            device_type_from_sharding_infos
        )
        sharded_tables_per_rank = self._shard(sharding_infos)
        self._need_pos = need_pos
        self._grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]] = (
            []
        )
        self._grouped_embedding_configs_per_rank = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs: List[GroupedEmbeddingConfig] = (
            self._grouped_embedding_configs_per_rank[self._rank]
        )
        self.enable_embedding_update: bool = any(
            grouped_config.enable_embedding_update
            for grouped_config in self._grouped_embedding_configs
        )

        self._has_feature_processor: bool = False
        for group_config in self._grouped_embedding_configs:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for _ in range(self._world_size)
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
                placements = (
                    (Replicate(), Shard(0)) if self._is_2D_parallel else (Shard(0),)
                )
                dtensor_metadata = DTensorMetadata(
                    mesh=self._env.device_mesh,
                    placements=placements,
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

            for rank in range(self._world_size):
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
                        local_rows=shards[rank].shard_sizes[0],
                        local_cols=info.embedding_config.embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[rank],
                        global_metadata=global_metadata,
                        dtensor_metadata=dtensor_metadata,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                        fused_params=info.fused_params,
                        num_embeddings_post_pruning=info.embedding_config.num_embeddings_post_pruning,
                        total_num_buckets=info.embedding_config.total_num_buckets,
                        use_virtual_table=info.embedding_config.use_virtual_table,
                        virtual_table_eviction_policy=info.embedding_config.virtual_table_eviction_policy,
                        enable_embedding_update=info.embedding_config.enable_embedding_update,
                    )
                )
        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_config in self._grouped_embedding_configs:
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
        for grouped_config in self._grouped_embedding_configs:
            embedding_shard_metadata.extend(grouped_config.embedding_shard_metadata())
        return embedding_shard_metadata

    def feature_names(self) -> List[str]:
        feature_names = []
        for grouped_config in self._grouped_embedding_configs:
            feature_names.extend(grouped_config.feature_names())
        return feature_names

    def embedding_tables(self) -> List[ShardedEmbeddingTable]:
        embedding_tables = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_tables.extend(grouped_config.embedding_tables)
        return embedding_tables

    def _get_num_features(self) -> int:
        return sum(
            group_config.num_features()
            for group_config in self._grouped_embedding_configs
        )

    def _get_feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes: List[int] = []
        for group_config in self._grouped_embedding_configs:
            feature_hash_sizes.extend(group_config.feature_hash_sizes())
        return feature_hash_sizes

    def _get_num_writable_features(self) -> int:
        return sum(
            group_config.num_features()
            for group_config in self._grouped_embedding_configs
            if group_config.enable_embedding_update
        )

    def _get_writable_feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes: List[int] = []
        for group_config in self._grouped_embedding_configs:
            if group_config.enable_embedding_update:
                feature_hash_sizes.extend(group_config.feature_hash_sizes())
        return feature_hash_sizes


class RwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
            communication.
        num_features (int): total number of features.
        feature_hash_sizes (List[int]): hash sizes of features.
        feature_total_num_buckets (Optional[List[int]]): total number of buckets, if provided will be >= world size.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).

    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        feature_total_num_buckets: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        keep_original_indices: bool = False,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._num_features = num_features

        feature_block_sizes: List[int] = []

        for i, hash_size in enumerate(feature_hash_sizes):
            block_divisor = self._world_size
            if feature_total_num_buckets is not None:
                assert feature_total_num_buckets[i] % self._world_size == 0
                block_divisor = feature_total_num_buckets[i]
            feature_block_sizes.append((hash_size + block_divisor - 1) // block_divisor)

        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self._has_multiple_blocks_per_shard: bool = (
            feature_total_num_buckets is not None
        )
        if self._has_multiple_blocks_per_shard:
            self.register_buffer(
                "_feature_total_num_blocks_tensor",
                torch.tensor(
                    [feature_total_num_buckets],
                    device=device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

        self._dist = KJTAllToAll(
            pg=pg,
            splits=[self._num_features] * self._world_size,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None
        self._keep_original_indices = keep_original_indices

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Bucketizes sparse feature values into world size number of buckets and then
        performs AlltoAll operation.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            total_num_blocks=(
                self._feature_total_num_blocks_tensor
                if self._has_multiple_blocks_per_shard
                else None
            ),
            output_permute=self._is_sequence,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            keep_original_indices=self._keep_original_indices,
        )

        return self._dist(bucketized_features)


class RwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion by performing a reduce-scatter
    operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for reduce-scatter communication.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        embedding_dims: List[int],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._dist: Optional[
            Union[
                PooledEmbeddingsReduceScatter,
                VariableBatchPooledEmbeddingsReduceScatter,
            ]
        ] = None
        self._pg = pg
        self._qcomm_codecs_registry = qcomm_codecs_registry
        self._codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None
        )
        self._embedding_dims = embedding_dims

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.
            sharding_ctx (Optional[EmbeddingShardingContext]): shared context from
                KJTAllToAll operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """
        if self._dist is None:
            self._create_output_dist_module(sharding_ctx)

        if sharding_ctx is None:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(local_embs)
        elif sharding_ctx.variable_batch_per_feature:
            return cast(VariableBatchPooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                batch_size_per_rank_per_feature=sharding_ctx.batch_size_per_rank_per_feature,
                embedding_dims=self._embedding_dims,
            )
        else:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                input_splits=sharding_ctx.batch_size_per_rank,
            )

    def _create_output_dist_module(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._dist = VariableBatchPooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
        else:
            self._dist = PooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )


class InferRwPooledEmbeddingDist(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion with an AlltoOne operation.

    Args:
        device (torch.device): device on which the tensors will be communicated to.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOneReduce = EmbeddingsAllToOneReduce(
            device=device,
            world_size=world_size,
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> torch.Tensor:
        """
        Performs AlltoOne operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """

        return self._dist(local_embs)


class RwPooledEmbeddingSharding(
    BaseRwEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags row-wise, i.e.. a given embedding table is evenly distributed
    by rows and table slices are placed on all ranks.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            pg=self._pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=False,
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
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.ROW_WISE,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]:
        return RwPooledEmbeddingDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            embedding_dims=self.embedding_dims(),
        )


class RwSparseFeaturesWriteDist(BaseSparseFeaturesWriteDist[KeyedJaggedTensor]):
    """
    Accepts sparse feature embedding weights in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        num_features (int): total number of features.
        feature_hash_sizes (List[int]): hash sizes of features.
        feature_total_num_buckets (Optional[List[int]]): total number of buckets, if provided will be >= world size.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).

    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        feature_total_num_buckets: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        keep_original_indices: bool = False,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._num_features = num_features

        feature_block_sizes: List[int] = []

        for i, hash_size in enumerate(feature_hash_sizes):
            block_divisor = self._world_size
            if feature_total_num_buckets is not None:
                assert feature_total_num_buckets[i] % self._world_size == 0
                block_divisor = feature_total_num_buckets[i]
            feature_block_sizes.append((hash_size + block_divisor - 1) // block_divisor)

        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self._has_multiple_blocks_per_shard: bool = (
            feature_total_num_buckets is not None
        )
        if self._has_multiple_blocks_per_shard:
            self.register_buffer(
                "_feature_total_num_blocks_tensor",
                torch.tensor(
                    [feature_total_num_buckets],
                    device=device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

        self._dist = KJEAllToAll(
            pg=pg,
            splits=[self._num_features] * self._world_size,
        )
        self._is_sequence = is_sequence
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None
        self._keep_original_indices = keep_original_indices

    def forward(
        self,
        embeddings: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Bucketizes sparse feature values into world size number of buckets and then
        performs AlltoAll operation.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_embeddings_before_all2all_write(
            embeddings,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            total_num_blocks=(
                self._feature_total_num_blocks_tensor
                if self._has_multiple_blocks_per_shard
                else None
            ),
            output_permute=self._is_sequence,
            bucketize_pos=False,
            keep_original_indices=self._keep_original_indices,
        )

        return self._dist(bucketized_features)


@overload
def convert_tensor(t: torch.Tensor, feature: KeyedJaggedTensor) -> torch.Tensor: ...
@overload
def convert_tensor(t: None, feature: KeyedJaggedTensor) -> None: ...


def convert_tensor(
    t: Union[torch.Tensor, None],
    feature: KeyedJaggedTensor,
) -> Union[torch.Tensor, None]:
    # comparing to Optional[Tensor], this solution will keep output as Tensor when input is not None
    if t is None:
        return None
    else:
        return t.to(
            device=feature.device(),
            dtype=feature.values().dtype,
        )


class InferRwSparseFeaturesDist(BaseSparseFeaturesDist[InputDistOutputs]):
    def __init__(
        self,
        world_size: int,
        num_features: int,
        feature_hash_sizes: List[int],
        feature_total_num_buckets: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        embedding_shard_metadata: Optional[List[List[int]]] = None,
        keep_original_indices: bool = False,
    ) -> None:
        super().__init__()
        logger.info(
            f"InferRwSparseFeaturesDist: {world_size=}, {num_features=}, {feature_hash_sizes=}, {feature_total_num_buckets=}, {device=}, {is_sequence=}, {has_feature_processor=}, {need_pos=}, {embedding_shard_metadata=}"
            f", keep_original_indices={keep_original_indices}"
        )
        self._world_size: int = world_size
        self._num_features = num_features
        self._feature_total_num_buckets: Optional[List[int]] = feature_total_num_buckets
        feature_block_sizes: List[int] = []
        for i, hash_size in enumerate(feature_hash_sizes):
            block_divisor = self._world_size
            if (
                feature_total_num_buckets is not None
                and embedding_shard_metadata is None
            ):
                assert feature_total_num_buckets[i] % self._world_size == 0
                block_divisor = feature_total_num_buckets[i]
            feature_block_sizes.append((hash_size + block_divisor - 1) // block_divisor)
        self.register_buffer(
            "feature_block_sizes",
            torch.tensor(feature_block_sizes),
        )

        self._dist = KJTOneToAll(
            splits=self._world_size * [self._num_features],
            world_size=world_size,
            device=device,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos
        embedding_shard_metadata = embedding_shard_metadata or []
        for i, row_pos in enumerate(embedding_shard_metadata):
            self.register_buffer(f"row_pos_{i}", torch.tensor(row_pos))
        self.embedding_shard_metadata_len: int = len(embedding_shard_metadata)
        self._keep_original_indices = keep_original_indices
        # pyre-ignore[8]
        self.register_buffer(
            "feature_total_num_buckets",
            (
                torch.tensor(feature_total_num_buckets)
                if feature_total_num_buckets
                else None
            ),
        )
        self.forwarded: bool = False

    def get_block_bucketize_row_pos(self) -> Optional[List[torch.Tensor]]:
        return [
            getattr(self, f"row_pos_{i}")
            for i in range(self.embedding_shard_metadata_len)
        ] or None

    def move_buffer(self, sparse_features: KeyedJaggedTensor) -> None:
        # buffer should only be moved once, even if this method being executed multiple times. as later 'to' should return same tensor after first convert
        self.feature_block_sizes = convert_tensor(
            t=self.feature_block_sizes, feature=sparse_features
        )
        self.feature_total_num_buckets = convert_tensor(
            t=self.feature_total_num_buckets, feature=sparse_features
        )
        for i in range(self.embedding_shard_metadata_len):
            setattr(
                self,
                f"row_pos_{i}",
                convert_tensor(
                    t=getattr(self, f"row_pos_{i}"),
                    feature=sparse_features,
                ),
            )

    def forward(self, sparse_features: KeyedJaggedTensor) -> InputDistOutputs:
        if not self.forwarded:
            # after fx tracing, 'if' will be removed, and below line will actually be called multiple times. but it's ok as 'to' will return same tensor after first convert.
            self.move_buffer(sparse_features)
            self.forwarded = True
        (
            bucketized_features,
            unbucketize_permute_tensor,
            bucket_mapping_tensor_opt,
        ) = bucketize_kjt_inference(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self.feature_block_sizes,
            total_num_buckets=self.feature_total_num_buckets,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            block_bucketize_row_pos=self.get_block_bucketize_row_pos(),
            is_sequence=self._is_sequence,
            keep_original_indices=self._keep_original_indices,
        )
        # KJTOneToAll
        dist_kjt = self._dist.forward(bucketized_features)
        return InputDistOutputs(
            features=dist_kjt,
            unbucketize_permute_tensor=(
                unbucketize_permute_tensor if self._is_sequence else None
            ),
            bucket_mapping_tensor=(
                bucket_mapping_tensor_opt if self._is_sequence else None
            ),
            bucketized_length=(
                bucketized_features.lengths().view(
                    self._world_size * self._num_features, -1
                )
                if self._is_sequence
                else None
            ),
        )


class InferRwPooledEmbeddingSharding(
    BaseRwEmbeddingSharding[
        NullShardingContext, InputDistOutputs, List[torch.Tensor], torch.Tensor
    ]
):
    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[InputDistOutputs]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()

        (embed_sharding, is_even_sharding) = get_embedding_shard_metadata(
            self._grouped_embedding_configs_per_rank
        )

        return InferRwSparseFeaturesDist(
            world_size=self._world_size,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            embedding_shard_metadata=embed_sharding if not is_even_sharding else None,
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
            device_type_from_sharding_infos=self._device_type_from_sharding_infos,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]:
        assert device is not None
        return InferRwPooledEmbeddingDist(
            device=device,
            world_size=self._world_size,
        )
