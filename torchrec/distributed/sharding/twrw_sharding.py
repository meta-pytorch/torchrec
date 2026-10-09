#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.distributed_c10d import get_process_group_ranks
from torchrec.distributed import utils as distributed_utils
from torchrec.distributed.comm import (
    get_2d_pod_size,
    get_local_size,
    get_resolved_pod_size,
    intra_and_cross_node_pg,
    intra_and_cross_node_pg_2D,
    is_2d_pod_size_enabled,
)
from torchrec.distributed.dist_data import (
    KJTAllToAll,
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
    bucketize_kjt_before_all2all,
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
from torchrec.distributed.logging_handlers import EventLoggingHandler, TorchrecComponent
from torchrec.distributed.logging_utils import EventType
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
from torchrec.distributed.utils import none_throws
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_single_group_ranks(
    table_name: str,
    plan_ranks: List[int],
    num_shards: int,
    table_group: int,
    local_size: int,
    is_2D_parallel: bool,
) -> List[int]:
    if not is_2D_parallel and len(plan_ranks) > local_size:
        raise ValueError(
            f"'{table_name}': the plan places it on {len(plan_ranks)} ranks, "
            f"more than the {local_size} in a TWRW group, but does not set "
            "num_twrw_groups. The planner and runtime disagree on the group width."
        )
    if num_shards < local_size:
        raise ValueError(
            f"'{table_name}': a single-group table needs at least one shard per "
            f"rank, but the plan has {num_shards} shards for a TWRW group of "
            f"{local_size} ranks."
        )
    if not is_2D_parallel and num_shards > local_size:
        raise ValueError(
            f"'{table_name}': a single-group table needs exactly {local_size} "
            f"shards outside 2D parallelism, but the plan has {num_shards}."
        )
    return list(range(table_group * local_size, (table_group + 1) * local_size))


def _resolve_multi_group_ranks(
    table_name: str,
    num_twrw_groups: int,
    plan_ranks: List[int],
    num_shards: int,
    local_size: int,
    is_2D_parallel: bool,
) -> List[int]:
    if not distributed_utils.is_twrw_multi_group_enabled():
        raise NotImplementedError(
            f"'{table_name}': TABLE_ROW_WISE num_twrw_groups={num_twrw_groups} "
            "is disabled by killswitch."
        )
    if is_2D_parallel:
        raise ValueError(
            f"'{table_name}': TABLE_ROW_WISE "
            f"num_twrw_groups={num_twrw_groups} is not supported under "
            "2D parallelism."
        )
    if len(plan_ranks) != num_shards:
        raise ValueError(
            f"'{table_name}': a multi-group table has {len(plan_ranks)} placement "
            f"ranks for {num_shards} shards."
        )
    expected_ranks = num_twrw_groups * local_size
    if len(plan_ranks) != expected_ranks:
        raise ValueError(
            f"'{table_name}': num_twrw_groups={num_twrw_groups} needs "
            f"{expected_ranks} ranks at a TWRW group width of {local_size}, but "
            f"the plan places it on {len(plan_ranks)}."
        )
    if len(set(plan_ranks)) != len(plan_ranks):
        raise ValueError(
            f"'{table_name}': placement ranks {plan_ranks} repeat a rank; each "
            "shard needs its own."
        )
    groups = {rank // local_size for rank in plan_ranks}
    if len(groups) != num_twrw_groups:
        raise ValueError(
            f"'{table_name}': num_twrw_groups={num_twrw_groups} but its "
            f"{len(plan_ranks)} ranks span {len(groups)} groups."
        )
    # At this point there are exactly num_twrw_groups complete groups of
    # local_size ranks.
    return plan_ranks


@dataclass(frozen=True)
class TwRwFeatureDistLayout:
    """Feature-ID distribution layout for variable TWRW group counts.

    `hash_sizes` and `shard_counts` are in logical feature order. Each feature
    is bucketized using its shard count. `rank_order_indices` then places the
    resulting `(feature, shard)` entries in destination-rank order across all
    TWRW-sharded tables. `features_per_rank` gives the AlltoAll splits.

    Example: A/C have 2 shards and B has 4. Their shard-count groups bucketize
    to `(A0, C0, A1, C1)` and `(B0, B1, B2, B3)`. Concatenating those groups,
    then reordering to `(A0, B0, C0, A1, B1, C1, B2, B3)` by destination rank,
    gives `rank_order_indices=(0, 4, 1, 2, 5, 3, 6, 7)`.
    """

    hash_sizes: tuple[int, ...]
    shard_counts: tuple[int, ...]
    rank_order_indices: tuple[int, ...]
    features_per_rank: tuple[int, ...]


@dataclass(frozen=True)
class TwRwEmbeddingDistLayout:
    """Logical pooled-output dimensions and each group's physical partials.

    `embedding_dims` is in logical feature order. Each inner tuple maps one
    TWRW group's physical partials back to those logical features.

    Example: logical features A/B/C have dims `(4, 8, 6)`. Group indices
    `((0, 1, 2), (1,))` mean group 0 emits `(A, B, C)` and group 1 emits `(B)`.
    The physical order is `(A, B, C, B)`; both B partials map to logical B
    and are summed.
    """

    embedding_dims: tuple[int, ...]
    feature_indices_by_twrw_group: tuple[tuple[int, ...], ...]


@dataclass
class _LogicalFeatureMetadata:
    """Names and shard-zero metadata for one logical feature."""

    feature_name: str
    embedding_name: str
    shard_metadata: Optional[ShardMetadata]


def _shard_count_groups(
    shard_counts: tuple[int, ...],
) -> list[tuple[int, list[int]]]:
    """Group logical features that can share one bucketization call.

    Groups follow the first logical feature with each shard count. Features
    within each group remain in logical feature order.
    """
    features_by_shard_count: Dict[int, List[int]] = {}
    for feature, shard_count in enumerate(shard_counts):
        features_by_shard_count.setdefault(shard_count, []).append(feature)
    return list(features_by_shard_count.items())


def _rank_order_indices(
    shard_counts: tuple[int, ...], feature_shards: list[tuple[int, int]]
) -> tuple[int, ...]:
    """Map bucketization output into destination-rank order.

    Bucketization is shard-major within each shard-count group; `feature_shards`
    lists those same `(feature, shard)` entries in rank order.
    """
    index_by_feature_shard: Dict[Tuple[int, int], int] = {}
    group_offset = 0
    for shard_count, features in _shard_count_groups(shard_counts):
        for shard in range(shard_count):
            for position, feature in enumerate(features):
                index_by_feature_shard[(feature, shard)] = (
                    group_offset + shard * len(features) + position
                )
        group_offset += shard_count * len(features)
    return tuple(index_by_feature_shard[item] for item in feature_shards)


def _uniform_feature_dist_layout(
    hash_sizes: List[int], features_per_rank: List[int], local_size: int
) -> TwRwFeatureDistLayout:
    """Build the legacy uniform layout used by GRID's input distribution."""
    features_per_group = features_per_rank[::local_size]
    group_offsets = [0] + list(itertools.accumulate(features_per_group))
    return TwRwFeatureDistLayout(
        hash_sizes=tuple(hash_sizes),
        shard_counts=(local_size,) * len(hash_sizes),
        rank_order_indices=tuple(
            shard * len(hash_sizes) + feature
            for group in range(len(features_per_group))
            for shard in range(local_size)
            for feature in range(group_offsets[group], group_offsets[group + 1])
        ),
        features_per_rank=tuple(features_per_rank),
    )


def _build_twrw_dist_layouts(
    grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
    table_placement_ranks: Dict[str, List[int]],
    local_size: int,
) -> tuple[
    TwRwFeatureDistLayout,
    TwRwEmbeddingDistLayout,
    list[_LogicalFeatureMetadata],
]:
    """Build both distribution layouts in one traversal by destination rank.

    Returns feature-ID routing, pooled-partial reconstruction, and the logical
    names/shard-zero metadata exposed by the sharding API.
    """
    feature_metadata: list[_LogicalFeatureMetadata] = []
    hash_sizes: list[int] = []
    shard_counts: list[int] = []
    embedding_dims: list[int] = []
    feature_index_by_key: dict[tuple[str, int], int] = {}
    feature_shards: list[tuple[int, int]] = []
    features_per_rank: list[int] = []
    feature_indices_by_group: list[list[int]] = [
        [] for _ in range(len(grouped_configs_per_rank) // local_size)
    ]
    rank_to_shard_by_table = {
        table: {rank: shard for shard, rank in enumerate(ranks)}
        for table, ranks in table_placement_ranks.items()
    }
    for rank, grouped_configs in enumerate(grouped_configs_per_rank):
        rank_feature_count = 0
        tables = itertools.chain.from_iterable(
            config.embedding_tables for config in grouped_configs
        )
        for table in tables:
            placement_ranks = table_placement_ranks[table.name]
            shard = rank_to_shard_by_table[table.name][rank]
            for position, feature_name in enumerate(table.feature_names):
                key = (table.name, position)
                feature_index = feature_index_by_key.get(key)
                if feature_index is None:
                    feature_index = len(hash_sizes)
                    feature_index_by_key[key] = feature_index
                    feature_metadata.append(
                        _LogicalFeatureMetadata(
                            feature_name=feature_name,
                            embedding_name=table.embedding_names[position],
                            shard_metadata=None,
                        )
                    )
                    hash_sizes.append(table.num_embeddings)
                    shard_counts.append(len(placement_ranks))
                    embedding_dims.append(table.local_cols)
                if shard == 0:
                    feature_metadata[feature_index].shard_metadata = (
                        table.local_metadata
                    )
                if rank % local_size == 0:
                    feature_indices_by_group[rank // local_size].append(feature_index)
                feature_shards.append((feature_index, shard))
                rank_feature_count += 1
        features_per_rank.append(rank_feature_count)

    shard_counts_tuple = tuple(shard_counts)
    return (
        TwRwFeatureDistLayout(
            hash_sizes=tuple(hash_sizes),
            shard_counts=shard_counts_tuple,
            rank_order_indices=_rank_order_indices(shard_counts_tuple, feature_shards),
            features_per_rank=tuple(features_per_rank),
        ),
        TwRwEmbeddingDistLayout(
            embedding_dims=tuple(embedding_dims),
            feature_indices_by_twrw_group=tuple(
                tuple(indices) for indices in feature_indices_by_group
            ),
        ),
        feature_metadata,
    )


class BaseTwRwEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for table wise row wise sharding.
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
        self._env = env
        self._is_2D_parallel: bool = isinstance(env, ShardingEnv2D)
        self._pg: Optional[dist.ProcessGroup] = (
            # pyrefly: ignore[missing-attribute]
            self._env.sharding_pg
            if self._is_2D_parallel
            else self._env.process_group
        )
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        self._device = device
        self._need_pos = need_pos
        if self._is_2D_parallel:
            intra_pg, cross_pg = intra_and_cross_node_pg_2D(
                # pyrefly: ignore[bad-argument-type]
                self._env,
                device=device,
            )
        else:
            intra_pg, cross_pg = intra_and_cross_node_pg(
                device, backend=dist.get_backend(self._pg)
            )
        self._intra_pg: Optional[dist.ProcessGroup] = intra_pg
        self._cross_pg: Optional[dist.ProcessGroup] = cross_pg
        self._node_group_size: Optional[int] = (
            # pyrefly: ignore[missing-attribute]
            self._env.node_group_size
            if self._is_2D_parallel
            else None
        )
        # Must match the world size the real builder feeds get_local_size(), because
        # get_local_size() falls back to its argument when LOCAL_WORLD_SIZE is unset or
        # does not divide it -- any divergence here manufactures the very node-width
        # mismatch the guard in _shard() exists to catch. intra_and_cross_node_pg_2D
        # uses dist.get_world_size() (the default WORLD group), so use exactly that;
        # ShardingEnv2D always implies an initialised dist. The 1D fallback stays on
        # env.world_size instead, since a ShardingEnv.from_local() env has no process
        # group at all and calling dist here would raise.
        self._global_world_size: int = (
            dist.get_world_size() if self._is_2D_parallel else self._world_size
        )
        if intra_pg is not None:
            local_size = intra_pg.size()
        else:
            # Meta-device construction gets no process groups back, so mirror the
            # width the real intra group would have had -- pod_size included. Without
            # it this node width disagrees with a plan cut against
            # Topology.intra_group_size whenever pod_size > 1 (the NVL72 SKUs).
            # 2D goes through the knobbed accessor so the killswitch reaches this
            # path too; 1D keeps its ungated pod_size from D105663332.
            pod_size = (
                get_2d_pod_size()
                if self._is_2D_parallel
                else (get_resolved_pod_size() or 1)
            )
            local_size = pod_size * (
                self._node_group_size
                if self._node_group_size
                else get_local_size(self._global_world_size)
            )
        self._local_size: int = local_size

        sharded_tables_per_rank, table_placement_ranks = self._shard(sharding_infos)
        self._grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]] = (
            []
        )
        self._grouped_embedding_configs_per_rank = group_tables(sharded_tables_per_rank)
        self._has_feature_processor: bool = False
        for group_config in self._grouped_embedding_configs_per_rank[
            self._rank // self._local_size
        ]:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

        (
            self._feature_dist_layout,
            self._embedding_dist_layout,
            self._feature_metadata,
        ) = _build_twrw_dist_layouts(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            table_placement_ranks=table_placement_ranks,
            local_size=self._local_size,
        )

    def _resolve_placement_ranks(
        self,
        table_name: str,
        num_twrw_groups: int,
        plan_ranks: List[int],
        num_shards: int,
        table_group: int,
    ) -> List[int]:
        """Resolve one rank per shard.

        Single-group plans preserve the legacy `ranks[0]` behavior. Multi-group
        plans preserve `ranks[i]` to `shards[i]`; `num_twrw_groups` explicitly
        selects the path.
        """
        local_size = self._local_size
        if num_twrw_groups < 1:
            raise ValueError(
                f"'{table_name}': num_twrw_groups={num_twrw_groups} must be >= 1."
            )

        if num_twrw_groups == 1:
            return _resolve_single_group_ranks(
                table_name,
                plan_ranks,
                num_shards,
                table_group,
                local_size,
                self._is_2D_parallel,
            )

        return _resolve_multi_group_ranks(
            table_name,
            num_twrw_groups,
            plan_ranks,
            num_shards,
            local_size,
            self._is_2D_parallel,
        )

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> Tuple[List[List[ShardedEmbeddingTable]], Dict[str, List[int]]]:
        """Assign each shard to its resolved placement rank.

        Returns the sharded tables assigned to each rank and each table's
        placement ranks in shard order: `placement_ranks[i]` owns `shards[i]`.
        """
        world_size = self._world_size
        local_size = self._local_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for _ in range(world_size)
        ]
        table_placement_ranks: Dict[str, List[int]] = {}
        peer_group = get_process_group_ranks(self._pg) if self._is_2D_parallel else None
        for info in sharding_infos:
            # Under 2D parallelism we transform rank to the logical ordering in a regular parallelism scheme
            planner_rank = none_throws(info.param_sharding.ranks)[0]
            if peer_group is not None:
                pg_members: List[int] = peer_group
                try:
                    rank = pg_members.index(planner_rank)
                except ValueError:
                    logger.warning(
                        "[2d-sharding-diag] peer_group_index_failed table=%s planner_rank=%d (see Scuba torchrec_event_logging)",
                        info.embedding_config.name,
                        planner_rank,
                    )
                    EventLoggingHandler.log_event(
                        component=TorchrecComponent.SHARDER.value,
                        event_name="TwRwBaseSharding.2d_diag.peer_group_index_failed",
                        event_type=EventType.INFO,
                        metadata={
                            "table_name": info.embedding_config.name,
                            "planner_rank": str(planner_rank),
                            "peer_group_size": str(len(pg_members)),
                            "peer_group_head": str(pg_members[:8]),
                            "sharding_pg_size": str(self._world_size),
                            "global_world_size": str(dist.get_world_size()),
                        },
                    )
                    raise
            else:
                rank = planner_rank
            table_group = rank // local_size
            # pyrefly: ignore[missing-attribute]
            shards = info.param_sharding.sharding_spec.shards

            table_name = info.embedding_config.name
            num_twrw_groups: int = info.param_sharding.num_twrw_groups or 1
            placement_ranks = self._resolve_placement_ranks(
                table_name=table_name,
                num_twrw_groups=num_twrw_groups,
                plan_ranks=list(none_throws(info.param_sharding.ranks)),
                num_shards=len(shards),
                table_group=table_group,
            )
            expected_shards = num_twrw_groups * local_size

            # Keep the pod-size killswitch reversible: when disabled, the runtime
            # group is deliberately narrower than the plan.
            if len(shards) != expected_shards and is_2d_pod_size_enabled():
                consequence = (
                    f"placing only the first {expected_shards} and dropping "
                    f"{len(shards) - expected_shards} while still reporting the full "
                    f"tensor"
                    if len(shards) > expected_shards
                    else f"leaving {expected_shards - len(shards)} placement ranks "
                    "with no shard"
                )
                raise ValueError(
                    f"TWRW group span mismatch for table "
                    f"'{info.embedding_config.name}': the plan has {len(shards)} "
                    f"shards but num_twrw_groups={num_twrw_groups} with a "
                    f"runtime TWRW group of {local_size} ranks requires "
                    f"{expected_shards}. is_2D={self._is_2D_parallel}, "
                    f"pod_size={get_resolved_pod_size()}, "
                    f"node_group_size={self._node_group_size}, "
                    f"local_world_size={get_local_size(self._global_world_size)}. "
                    f"A TWRW plan must carry one shard per rank in every "
                    f"requested TWRW group; "
                    f"continuing would mean {consequence}. Check that the plan was "
                    f"built against Topology.intra_group_size rather than "
                    f"local_world_size."
                )

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
                dtensor_metadata = DTensorMetadata(
                    mesh=self._env.device_mesh,
                    placements=(
                        (Replicate(), Shard(1)) if self._is_2D_parallel else (Shard(1),)
                    ),
                    size=(
                        info.embedding_config.num_embeddings,
                        info.embedding_config.embedding_dim,
                    ),
                    stride=info.param.stride(),
                )

            table_placement_ranks[table_name] = placement_ranks
            for rank_idx, rank in enumerate(placement_ranks):
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
                        local_rows=shards[rank_idx].shard_sizes[0],
                        local_cols=info.embedding_config.embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[rank_idx],
                        global_metadata=global_metadata,
                        dtensor_metadata=dtensor_metadata,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                        fused_params=info.fused_params,
                        use_virtual_table=info.embedding_config.use_virtual_table,
                        stash_weights=info.embedding_config.stash_weights,
                    )
                )

        return tables_per_rank, table_placement_ranks

    def embedding_dims(self) -> List[int]:
        return list(self._embedding_dist_layout.embedding_dims)

    def embedding_names(self) -> List[str]:
        return [metadata.embedding_name for metadata in self._feature_metadata]

    def embedding_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        return [metadata.shard_metadata for metadata in self._feature_metadata]

    def feature_names(self) -> List[str]:
        return [metadata.feature_name for metadata in self._feature_metadata]

    def _get_feature_hash_sizes(self) -> List[int]:
        return list(self._feature_dist_layout.hash_sizes)

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


class TwRwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes feature IDs and redistributes them with AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        local_size (int): number of ranks in each TWRW group.
        features_per_rank (Optional[List[int]]): legacy GRID feature splits,
            required only when `layout` is not provided.
        feature_hash_sizes (Optional[List[int]]): legacy GRID hash sizes,
            required only when `layout` is not provided.
        device (Optional[torch.device]): device on which buffers will be allocated.
        has_feature_processor (bool): existence of a feature processor (ie. position
            weighted features).
        need_pos (bool): whether to bucketize positions, used in place of
            `has_feature_processor` once the features carry weights.
        layout (Optional[TwRwFeatureDistLayout]): hash sizes, shard counts, and
            AlltoAll ordering. None builds GRID's legacy uniform layout.

    Example::

        2 TWRW groups of 2 ranks. Each `(feature, shard)` pair becomes one
        key in the bucketized KJT, listed under the rank owning that shard.

        Single-group tables: every table sits in one group, so each feature
        routes to `local_size` = 2 shards in its group. Here f0 and f1 are
        in group 0, f2 in group 1::

            rank 0: (f0, 0) (f1, 0)    rank 2: (f2, 0)
            rank 1: (f0, 1) (f1, 1)    rank 3: (f2, 1)

        Multi-group table: fb spans both groups, so it is cut into
        `num_twrw_groups * local_size` = 4 shards, one per rank, while the
        single-group fa keeps 2::

            rank 0: (fa, 0) (fb, 0)    rank 2: (fb, 2)
            rank 1: (fa, 1) (fb, 1)    rank 3: (fb, 3)

        `TwRwFeatureDistLayout` orders those pairs by destination rank.
        Different shard counts use separate bucketize calls because
        `num_buckets` affects out-of-range ids.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        local_size: int,
        features_per_rank: Optional[List[int]] = None,
        feature_hash_sizes: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        layout: Optional[TwRwFeatureDistLayout] = None,
    ) -> None:
        super().__init__()
        assert pg.size() % local_size == 0, "currently group granularity must be node"

        self._world_size: int = pg.size()
        self._local_size: int = local_size
        self._num_cross_nodes: int = self._world_size // self._local_size

        if layout is None:
            if features_per_rank is None or feature_hash_sizes is None:
                raise ValueError(
                    "features_per_rank and feature_hash_sizes are required when "
                    "layout is not provided"
                )
            layout = _uniform_feature_dist_layout(
                feature_hash_sizes, features_per_rank, local_size
            )
        self._shard_count_groups = _shard_count_groups(layout.shard_counts)
        if not self._shard_count_groups:
            # Keep the single-call path: KJT.concat cannot build an empty result.
            self._shard_count_groups = [(local_size, [])]

        feature_block_sizes = [
            math.ceil(layout.hash_sizes[feature] / shard_count)
            for shard_count, features in self._shard_count_groups
            for feature in features
        ]
        self._rank_order_indices: List[int] = list(layout.rank_order_indices)
        shard_count_group_feature_indices = [
            feature for _, features in self._shard_count_groups for feature in features
        ]

        # Not persistent: all three follow from the sharding plan, so a
        # checkpoint taken under a different one must not restore them.
        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rank_order_indices_tensor",
            torch.tensor(
                self._rank_order_indices,
                device=device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_shard_count_group_feature_indices_tensor",
            torch.tensor(
                shard_count_group_feature_indices,
                device=device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self._dist = KJTAllToAll(
            pg=pg,
            splits=list(layout.features_per_rank),
            stagger=self._num_cross_nodes,
        )
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos

    @EventLoggingHandler.event_logger(
        TorchrecComponent.INPUT_DIST, n=1000, add_wait_counter=True
    )
    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Partitions feature IDs by destination shard, orders those partitions by
        destination rank, and sends them with AlltoAll.

        Args:
            sparse_features (KeyedJaggedTensor): feature IDs to bucketize and
                redistribute, represented as a KJT.

        Returns:
            Awaitable[KeyedJaggedTensor]: awaitable of KeyedJaggedTensor.
        """

        bucketized_features = self._bucketize(
            sparse_features,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
        )

        return self._dist(
            bucketized_features.permute(
                self._rank_order_indices,
                # pyrefly: ignore[bad-argument-type]
                self._rank_order_indices_tensor,
            )
        )

    def _bucketize(
        self,
        sparse_features: KeyedJaggedTensor,
        bucketize_pos: bool,
    ) -> KeyedJaggedTensor:
        """Run one FBGEMM bucketize call per shard count."""
        feature_block_sizes = cast(torch.Tensor, self._feature_block_sizes_tensor)
        if len(self._shard_count_groups) == 1:
            return bucketize_kjt_before_all2all(
                sparse_features,
                num_buckets=self._shard_count_groups[0][0],
                block_sizes=feature_block_sizes,
                output_permute=False,
                bucketize_pos=bucketize_pos,
            )[0]

        shard_count_group_feature_indices = cast(
            torch.Tensor, self._shard_count_group_feature_indices_tensor
        )
        bucketized_features: List[KeyedJaggedTensor] = []
        start = 0
        for shard_count, features in self._shard_count_groups:
            end = start + len(features)
            bucketized_features.append(
                bucketize_kjt_before_all2all(
                    sparse_features.permute(
                        features,
                        shard_count_group_feature_indices[start:end],
                    ),
                    num_buckets=shard_count,
                    block_sizes=feature_block_sizes[start:end],
                    output_permute=False,
                    bucketize_pos=bucketize_pos,
                )[0]
            )
            start = end
        return KeyedJaggedTensor.concat(bucketized_features)


@dataclass
class _SumSlice:
    """A contiguous tensor slice added from `src` to `dst`."""

    src: int
    dst: int
    length: int


class TwRwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embeddings with reduce-scatter inside each TWRW group,
    followed by AlltoAll across groups.

    Args:
        cross_pg (dist.ProcessGroup): global level ProcessGroup for AlltoAll
            communication.
        intra_pg (dist.ProcessGroup): TWRW-group ProcessGroup for reduce-scatter.
        dim_sum_per_node (Optional[List[int]]): legacy per-group dimension sums,
            required only when `layout` is not provided.
        emb_dim_per_node_per_feature (Optional[List[List[int]]]): legacy
            dimensions per group, required only when `layout` is not provided.
        device (Optional[torch.device]): device on which buffers will be allocated.
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]):
        layout (Optional[TwRwEmbeddingDistLayout]): logical dimensions and the
            physical partial order produced by each TWRW group.
    """

    def __init__(
        self,
        rank: int,
        cross_pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        dim_sum_per_node: Optional[List[int]] = None,
        emb_dim_per_node_per_feature: Optional[List[List[int]]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        layout: Optional[TwRwEmbeddingDistLayout] = None,
    ) -> None:
        super().__init__()
        if layout is None:
            # Legacy metadata has one physical partial per logical feature;
            # synthesize its identity layout while preserving caller sizes.
            if dim_sum_per_node is None or emb_dim_per_node_per_feature is None:
                raise ValueError(
                    "dim_sum_per_node and emb_dim_per_node_per_feature are required "
                    "when layout is not provided"
                )
            group_offsets = list(
                itertools.accumulate(
                    (len(dimensions) for dimensions in emb_dim_per_node_per_feature),
                    initial=0,
                )
            )
            layout = TwRwEmbeddingDistLayout(
                embedding_dims=tuple(
                    itertools.chain.from_iterable(emb_dim_per_node_per_feature)
                ),
                feature_indices_by_twrw_group=tuple(
                    tuple(range(group_offsets[group], group_offsets[group + 1]))
                    for group in range(len(emb_dim_per_node_per_feature))
                ),
            )
            embedding_dims_by_group = emb_dim_per_node_per_feature
            dim_sums_by_group = dim_sum_per_node
        else:
            embedding_dims_by_group = [
                [layout.embedding_dims[index] for index in group]
                for group in layout.feature_indices_by_twrw_group
            ]
            dim_sums_by_group = [
                sum(dimensions) for dimensions in embedding_dims_by_group
            ]
        self._rank = rank
        self._layout = layout
        self._logical_feature_indices_by_partial: List[int] = list(
            itertools.chain.from_iterable(layout.feature_indices_by_twrw_group)
        )
        self._intra_pg: dist.ProcessGroup = intra_pg
        self._cross_pg: dist.ProcessGroup = cross_pg
        self._dim_sums_by_twrw_group = dim_sums_by_group
        self._embedding_dims_by_twrw_group = embedding_dims_by_group
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
        self._intra_dist: Optional[PooledEmbeddingsReduceScatter] = None
        self._cross_dist: Optional[PooledEmbeddingsAllToAll] = None
        self._variable_intra_dist: Optional[
            VariableBatchPooledEmbeddingsReduceScatter
        ] = None
        self._variable_cross_dist: Optional[VariableBatchPooledEmbeddingsAllToAll] = (
            None
        )

    def _combine_callback(
        self, feature_sizes: List[int]
    ) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Build the physical-partial-to-logical-output sum.

        Returns no callback when every logical feature has exactly one partial.
        """
        if len(feature_sizes) == len(self._logical_feature_indices_by_partial):
            return None
        slices, destination_size = self._combine_slices(
            feature_sizes, self._logical_feature_indices_by_partial
        )

        def _combine(tensor: torch.Tensor) -> torch.Tensor:
            out = tensor.new_zeros((*tensor.shape[:-1], destination_size))
            for s in slices:
                out[..., s.dst : s.dst + s.length] += tensor[
                    ..., s.src : s.src + s.length
                ]
            return out

        return _combine

    def _combine_slices(
        self, feature_sizes: List[int], feature_indices: List[int]
    ) -> Tuple[List[_SumSlice], int]:
        """Coalesce adjacent partials with contiguous destinations."""
        feature_offsets = list(itertools.accumulate(feature_sizes, initial=0))
        source_sizes = [feature_sizes[index] for index in feature_indices]
        source_offsets = list(itertools.accumulate(source_sizes, initial=0))
        slices: List[_SumSlice] = []
        for position, feature_index in enumerate(feature_indices):
            src = source_offsets[position]
            dst = feature_offsets[feature_index]
            length = source_sizes[position]
            if slices and dst == slices[-1].dst + slices[-1].length:
                slices[-1].length += length
            else:
                slices.append(_SumSlice(src, dst, length))
        return slices, feature_offsets[-1]

    def _variable_batch_combine(
        self, batch_size_per_feature: List[int]
    ) -> Tuple[List[int], Optional[Callable[[torch.Tensor], torch.Tensor]]]:
        """Prepare variable-batch metadata and output reconstruction.

        Repeat each logical feature's batch size for every TWRW group that
        emits that feature. The callback then sums the per-group pooled results
        for the same logical feature. Each result occupies
        `batch_size * embedding_dim` values in the flattened tensor.
        """
        if len(batch_size_per_feature) != len(self._layout.embedding_dims):
            raise ValueError(
                f"expected {len(self._layout.embedding_dims)} feature batch sizes, "
                f"got {len(batch_size_per_feature)}"
            )
        if len(batch_size_per_feature) == len(self._logical_feature_indices_by_partial):
            return batch_size_per_feature, None
        feature_sizes = [
            batch_size * embedding_dim
            for batch_size, embedding_dim in zip(
                batch_size_per_feature, self._layout.embedding_dims
            )
        ]
        return (
            [
                batch_size_per_feature[index]
                for index in self._logical_feature_indices_by_partial
            ],
            self._combine_callback(feature_sizes),
        )

    @EventLoggingHandler.event_logger(
        TorchrecComponent.OUTPUT_DIST, n=1000, add_wait_counter=True
    )
    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Reduce-scatters physical pooled partials within each TWRW group, then
        exchanges the group results with AlltoAll.

        Args:
            local_embs (torch.Tensor): physical pooled partials to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """
        if self._intra_dist is None or self._cross_dist is None:
            self._create_output_dist_modules(sharding_ctx)
        local_rank = self._rank % self._intra_pg.size()
        current_group = self._rank // self._intra_pg.size()
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            (
                batch_size_per_rank_per_feature_by_cross_group,
                batch_size_per_feature_sum_by_cross_group,
            ) = self._preprocess_batch_size_per_rank_per_feature(
                self._intra_pg.size(),
                self._cross_pg.size(),
                sharding_ctx.batch_size_per_rank_per_feature,
            )
            rs_result = cast(
                VariableBatchPooledEmbeddingsReduceScatter, self._variable_intra_dist
            )(
                local_embs,
                batch_size_per_rank_per_feature=batch_size_per_feature_sum_by_cross_group,
                embedding_dims=self._embedding_dims_by_twrw_group[current_group],
            ).wait()
            batch_size_per_partial, combine_callback = self._variable_batch_combine(
                sharding_ctx.batch_size_per_feature_pre_a2a
            )
            awaitable = cast(
                VariableBatchPooledEmbeddingsAllToAll, self._variable_cross_dist
            )(
                rs_result,
                batch_size_per_rank_per_feature=batch_size_per_rank_per_feature_by_cross_group[
                    local_rank
                ],
                batch_size_per_feature_pre_a2a=batch_size_per_partial,
            )
            if combine_callback is not None:
                awaitable.callbacks.append(combine_callback)
            return awaitable
        elif (
            sharding_ctx is not None and len(set(sharding_ctx.batch_size_per_rank)) > 1
        ):
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

    def _preprocess_batch_size_per_rank_per_feature(
        self,
        local_size: int,
        nodes: int,
        batch_size_per_rank_per_feature_stagger: List[List[int]],
    ) -> Tuple[List[List[List[int]]], List[List[int]]]:
        """
        Reorders `batch_size_per_rank_per_feature_stagger` so it's aligned with
        reordered features after AlltoAll.
        """
        if not batch_size_per_rank_per_feature_stagger:
            return [[]] * local_size, []
        batch_size_per_rank_per_feature_by_cross_group: List[List[List[int]]] = []
        batch_size_per_feature_sum_by_cross_group: List[List[int]] = []
        for local_rank in range(local_size):
            batch_size_by_node_per_rank_per_feature: List[List[int]] = []
            batch_size_per_feature_sum = [0] * len(
                batch_size_per_rank_per_feature_stagger[0]
            )
            for node in range(nodes):
                batch_size = batch_size_per_rank_per_feature_stagger[
                    local_rank * nodes + node
                ]
                batch_size_by_node_per_rank_per_feature.append(batch_size)
                batch_size_per_feature_sum = [
                    sum(x) for x in zip(batch_size_per_feature_sum, batch_size)
                ]
            batch_size_per_rank_per_feature_by_cross_group.append(
                batch_size_by_node_per_rank_per_feature
            )
            batch_size_per_feature_sum_by_cross_group.append(batch_size_per_feature_sum)

        return (
            batch_size_per_rank_per_feature_by_cross_group,
            batch_size_per_feature_sum_by_cross_group,
        )

    def _create_output_dist_modules(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._variable_intra_dist = VariableBatchPooledEmbeddingsReduceScatter(
                pg=self._intra_pg,
                codecs=self._intra_codecs,
            )
            self._variable_cross_dist = VariableBatchPooledEmbeddingsAllToAll(
                pg=self._cross_pg,
                emb_dim_per_rank_per_feature=self._embedding_dims_by_twrw_group,
                device=self._device,
                # Bound per step in `forward`, since offsets are batch dependent.
                callbacks=None,
                codecs=self._cross_codecs,
            )
        self._intra_dist = PooledEmbeddingsReduceScatter(
            pg=self._intra_pg,
            codecs=self._intra_codecs,
        )
        # Fixed batch sizes make slice widths layout-only, so this callback can
        # be reused across steps.
        combine_callback = self._combine_callback(list(self._layout.embedding_dims))
        self._cross_dist = PooledEmbeddingsAllToAll(
            pg=self._cross_pg,
            dim_sum_per_rank=self._dim_sums_by_twrw_group,
            device=self._device,
            codecs=self._cross_codecs,
            callbacks=[combine_callback] if combine_callback is not None else None,
        )


class TwRwPooledEmbeddingSharding(
    BaseTwRwEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise then row-wise.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        assert self._pg is not None
        assert self._intra_pg is not None
        return TwRwSparseFeaturesDist(
            pg=self._pg,
            local_size=self._intra_pg.size(),
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
            layout=self._feature_dist_layout,
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
        return TwRwPooledEmbeddingDist(
            rank=self._rank,
            cross_pg=cast(dist.ProcessGroup, self._cross_pg),
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            device=device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            layout=self._embedding_dist_layout,
        )
