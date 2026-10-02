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


@dataclass
class _ShardCountGroup:
    """Features routed across the same number of shards.

    They can share one FBGEMM bucketize call because its `num_buckets` equals
    the number of destination shards.
    """

    num_shards: int
    features: List[int]


@dataclass(frozen=True)
class _FeatureShard:
    """One feature and one of its destination shards."""

    feature: int
    shard: int


class _InputPlan:
    """Input destinations grouped by shard count, plus rank-order permutation.

    In TWRW, each shard is a row partition of its table. Each feature has one
    input partition per shard. Features with different shard counts need
    separate FBGEMM bucketize calls because `num_buckets` also controls
    out-of-range routing.
    `rank_order_indices` puts the grouped output into destination-rank order.
    """

    shard_count_groups: List[_ShardCountGroup]
    rank_order_indices: List[int]

    def __init__(
        self,
        num_shards_per_feature: List[int],
        feature_shards: List[_FeatureShard],
        default_num_shards: int,
    ) -> None:
        features_by_shard_count: Dict[int, List[int]] = {}
        for feature, num_shards in enumerate(num_shards_per_feature):
            features_by_shard_count.setdefault(num_shards, []).append(feature)
        # An empty input still needs one group so `_bucketize` can use its
        # single-call path.
        self.shard_count_groups = [
            _ShardCountGroup(num_shards, features)
            for num_shards, features in (
                features_by_shard_count or {default_num_shards: []}
            ).items()
        ]

        index_by_feature_shard: Dict[_FeatureShard, int] = {}
        group_offset = 0
        for group in self.shard_count_groups:
            for shard in range(group.num_shards):
                for position, feature in enumerate(group.features):
                    index_by_feature_shard[_FeatureShard(feature, shard)] = (
                        group_offset + shard * len(group.features) + position
                    )
            group_offset += group.num_shards * len(group.features)

        self.rank_order_indices = [
            index_by_feature_shard[feature_shard] for feature_shard in feature_shards
        ]

    @classmethod
    def uniform(
        cls,
        num_features: int,
        features_per_rank: List[int],
        local_size: int,
    ) -> "_InputPlan":
        """Build the legacy plan: one shard per rank in a table's group."""
        groups = len(features_per_rank) // local_size
        features_per_group = [
            features_per_rank[group * local_size] for group in range(groups)
        ]
        group_offsets = [0] + list(itertools.accumulate(features_per_group))
        return cls(
            [local_size] * num_features,
            [
                _FeatureShard(feature, shard)
                for group in range(groups)
                for shard in range(local_size)
                for feature in range(group_offsets[group], group_offsets[group + 1])
            ],
            default_num_shards=local_size,
        )


@dataclass
class _FeatureInfo:
    """One table feature shared by input routing and final output metadata."""

    feature_name: str
    hash_size: int
    num_shards: int
    embedding_name: str
    embedding_dim: int
    shard_metadata: Optional[ShardMetadata]


@dataclass
class _SumSlice:
    """A contiguous tensor slice added from `src` to `dst`."""

    src: int
    dst: int
    length: int


class _FeatureLayout:
    """Feature metadata plus input and partial-output orderings.

    Multi-group features have one partial per group in `_feature_indices`;
    single-group features appear once.
    """

    features: List[_FeatureInfo]
    input_plan: _InputPlan
    _feature_indices: List[int]

    def __init__(
        self,
        grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        table_placement_ranks: Dict[str, List[int]],
        local_size: int,
    ) -> None:
        """Derive feature metadata and orderings from rank-local tables.

        Walking ranks in order produces the AlltoAll destination order. Each
        rank-local table contributes one `(feature, shard)` destination, while
        `features` keeps one metadata entry per logical feature.
        """
        features: List[_FeatureInfo] = []
        feature_index_by_key: Dict[Tuple[str, int], int] = {}
        feature_indices: List[int] = []
        feature_shards: List[_FeatureShard] = []
        rank_to_shard_by_table = {
            table: {rank: shard for shard, rank in enumerate(ranks)}
            for table, ranks in table_placement_ranks.items()
        }
        for rank, grouped_configs in enumerate(grouped_configs_per_rank):
            for grouped_config in grouped_configs:
                for table in grouped_config.embedding_tables:
                    placement_ranks = table_placement_ranks[table.name]
                    shard = rank_to_shard_by_table[table.name][rank]
                    for position, feature_name in enumerate(table.feature_names):
                        key = (table.name, position)
                        feature_index = feature_index_by_key.get(key)
                        if feature_index is None:
                            feature_index = len(features)
                            feature_index_by_key[key] = feature_index
                            features.append(
                                _FeatureInfo(
                                    feature_name=feature_name,
                                    hash_size=table.num_embeddings,
                                    num_shards=len(placement_ranks),
                                    embedding_name=table.embedding_names[position],
                                    embedding_dim=table.local_cols,
                                    shard_metadata=None,
                                )
                            )
                        # Keep one deterministic representative per logical
                        # feature, matching legacy TWRW even if ranks are reordered.
                        if shard == 0:
                            features[feature_index].shard_metadata = (
                                table.local_metadata
                            )
                        if rank % local_size == 0:
                            feature_indices.append(feature_index)
                        feature_shards.append(_FeatureShard(feature_index, shard))
        self.features = features
        self._feature_indices = feature_indices
        self.input_plan = _InputPlan(
            [feature.num_shards for feature in features],
            feature_shards,
            default_num_shards=local_size,
        )

    def combine_callback(self) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """The fixed-batch combine, or nothing when no feature repeats."""
        if not self._needs_combine():
            return None
        return self._combine_callback(
            [feature.embedding_dim for feature in self.features]
        )

    def variable_batch_combine(
        self, batch_size_per_feature: List[int]
    ) -> Tuple[List[int], Optional[Callable[[torch.Tensor], torch.Tensor]]]:
        """Return partial batch sizes and the flattened-output combine."""
        if len(batch_size_per_feature) != len(self.features):
            raise ValueError(
                f"expected {len(self.features)} feature batch sizes, "
                f"got {len(batch_size_per_feature)}"
            )
        if not self._needs_combine():
            return batch_size_per_feature, None
        batch_size_per_partial = [
            batch_size_per_feature[index] for index in self._feature_indices
        ]
        feature_sizes = [
            batch_size * feature.embedding_dim
            for batch_size, feature in zip(batch_size_per_feature, self.features)
        ]
        return batch_size_per_partial, self._combine_callback(feature_sizes)

    def _needs_combine(self) -> bool:
        return len(self.features) != len(self._feature_indices)

    def _combine_callback(
        self, feature_sizes: List[int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        slices, destination_size = self._combine_slices(feature_sizes)

        def _combine(tensor: torch.Tensor) -> torch.Tensor:
            out = tensor.new_zeros((*tensor.shape[:-1], destination_size))
            for s in slices:
                out[..., s.dst : s.dst + s.length] += tensor[
                    ..., s.src : s.src + s.length
                ]
            return out

        return _combine

    def _combine_slices(self, feature_sizes: List[int]) -> Tuple[List[_SumSlice], int]:
        """Coalesce adjacent partials with contiguous destinations."""
        feature_offsets = list(itertools.accumulate(feature_sizes, initial=0))
        source_sizes = [feature_sizes[index] for index in self._feature_indices]
        source_offsets = list(itertools.accumulate(source_sizes, initial=0))
        slices: List[_SumSlice] = []
        for position, feature_index in enumerate(self._feature_indices):
            src = source_offsets[position]
            dst = feature_offsets[feature_index]
            length = source_sizes[position]
            if slices and dst == slices[-1].dst + slices[-1].length:
                slices[-1].length += length
            else:
                slices.append(_SumSlice(src, dst, length))
        return slices, feature_offsets[-1]


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

        self._feature_layout = _FeatureLayout(
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
        return [feature.embedding_dim for feature in self._feature_layout.features]

    def embedding_names(self) -> List[str]:
        return [feature.embedding_name for feature in self._feature_layout.features]

    def embedding_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        return [feature.shard_metadata for feature in self._feature_layout.features]

    def feature_names(self) -> List[str]:
        return [feature.feature_name for feature in self._feature_layout.features]

    def _get_feature_hash_sizes(self) -> List[int]:
        return [feature.hash_size for feature in self._feature_layout.features]

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


class TwRwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes sparse features and redistributes them with AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        local_size (int): number of ranks in each TWRW group.
        features_per_rank (List[int]): number of routed feature/shard entries
            sent to each rank.
        feature_hash_sizes (List[int]): hash size of each input feature.
        device (Optional[torch.device]): device on which buffers will be allocated.
        has_feature_processor (bool): existence of a feature processor (ie. position
            weighted features).
        need_pos (bool): whether to bucketize positions, used in place of
            `has_feature_processor` once the features carry weights.
        input_plan (Optional[_InputPlan]): shard counts and AlltoAll rank
            order. None uses the legacy uniform plan.

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

        `_InputPlan` orders those pairs by destination rank. Different shard
        counts use separate bucketize calls because `num_buckets` affects
        out-of-range ids.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        local_size: int,
        features_per_rank: List[int],
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        input_plan: Optional[_InputPlan] = None,
    ) -> None:
        super().__init__()
        assert pg.size() % local_size == 0, "currently group granularity must be node"

        self._world_size: int = pg.size()
        self._local_size: int = local_size
        self._num_cross_nodes: int = self._world_size // self._local_size

        if input_plan is None:
            input_plan = _InputPlan.uniform(
                len(feature_hash_sizes), features_per_rank, local_size
            )
        self._shard_count_groups: List[_ShardCountGroup] = input_plan.shard_count_groups

        feature_block_sizes = [
            math.ceil(feature_hash_sizes[feature] / group.num_shards)
            for group in input_plan.shard_count_groups
            for feature in group.features
        ]
        self._rank_order_indices: List[int] = input_plan.rank_order_indices
        shard_count_group_feature_indices = [
            feature
            for group in input_plan.shard_count_groups
            for feature in group.features
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
            splits=features_per_rank,
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
        Partitions sparse feature values by destination shard, reorders
        them into rank order, and then performs an AlltoAll operation.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

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
                num_buckets=self._shard_count_groups[0].num_shards,
                block_sizes=feature_block_sizes,
                output_permute=False,
                bucketize_pos=bucketize_pos,
            )[0]

        shard_count_group_feature_indices = cast(
            torch.Tensor, self._shard_count_group_feature_indices_tensor
        )
        bucketized_features: List[KeyedJaggedTensor] = []
        start = 0
        for group in self._shard_count_groups:
            end = start + len(group.features)
            bucketized_features.append(
                bucketize_kjt_before_all2all(
                    sparse_features.permute(
                        group.features,
                        shard_count_group_feature_indices[start:end],
                    ),
                    num_buckets=group.num_shards,
                    block_sizes=feature_block_sizes[start:end],
                    output_permute=False,
                    bucketize_pos=bucketize_pos,
                )[0]
            )
            start = end
        return KeyedJaggedTensor.concat(bucketized_features)


class TwRwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in TWRW fashion by performing a reduce-scatter
    operation row wise on the host level and then an AlltoAll operation table wise on
    the global level.

    Args:
        cross_pg (dist.ProcessGroup): global level ProcessGroup for AlltoAll
            communication.
        intra_pg (dist.ProcessGroup): host level ProcessGroup for reduce-scatter
            communication.
        dim_sum_per_node (List[int]): number of features (sum of dimensions) of the
            embedding for each host.
        emb_dim_per_node_per_feature (List[List[int]]):
        feature_layout (_FeatureLayout): feature order and per-group partials.
        device (Optional[torch.device]): device on which buffers will be allocated.
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]):
    """

    def __init__(
        self,
        rank: int,
        cross_pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        dim_sum_per_node: List[int],
        emb_dim_per_node_per_feature: List[List[int]],
        feature_layout: _FeatureLayout,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._rank = rank
        self._feature_layout = feature_layout
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
        self._intra_dist: Optional[PooledEmbeddingsReduceScatter] = None
        self._cross_dist: Optional[PooledEmbeddingsAllToAll] = None
        self._variable_intra_dist: Optional[
            VariableBatchPooledEmbeddingsReduceScatter
        ] = None
        self._variable_cross_dist: Optional[VariableBatchPooledEmbeddingsAllToAll] = (
            None
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
        current_node = self._rank // self._intra_pg.size()
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
                embedding_dims=self._emb_dim_per_node_per_feature[current_node],
            ).wait()
            batch_size_per_partial, combine_callback = (
                self._feature_layout.variable_batch_combine(
                    sharding_ctx.batch_size_per_feature_pre_a2a
                )
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
                emb_dim_per_rank_per_feature=self._emb_dim_per_node_per_feature,
                device=self._device,
                # Bound per step in `forward`, since offsets are batch dependent.
                callbacks=None,
                codecs=self._cross_codecs,
            )
        self._intra_dist = PooledEmbeddingsReduceScatter(
            pg=self._intra_pg,
            codecs=self._intra_codecs,
        )
        # The two-dimensional pooled output has a fixed feature layout, so one
        # persistent callback is safe here.
        combine_callback = self._feature_layout.combine_callback()
        self._cross_dist = PooledEmbeddingsAllToAll(
            pg=self._cross_pg,
            dim_sum_per_rank=self._dim_sum_per_node,
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
            input_plan=self._feature_layout.input_plan,
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
            dim_sum_per_node=self._dim_sum_per_node(),
            emb_dim_per_node_per_feature=self._emb_dim_per_node_per_feature(),
            feature_layout=self._feature_layout,
            device=device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
