#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Platform-specific construction seam for the sharding planner.

The orchestrator (``ShardingPlannerAPI``) owns the planning *flow* and the
executor owns planner *invocation*; neither knows how to build the
platform-specific inputs a planner needs. Those live behind ``PlannerProvider``:
given the request/context it builds the ``Topology`` and constructs the planner
for the request's ``PlannerVariant``.

``DefaultPlannerProvider`` is the OSS implementation — it faithfully reconstructs
the planner object graph (enumerator + estimators, partitioner, perf model,
proposers) from ``PlannerConfig`` so the plan is deterministic and matches what
the frameworks build today. The Meta-internal provider (``torchrec/fb``)
subclasses it, overriding only the fb-specific pieces (HUM topology, HW-based
perf estimator, fb proposers/partitioner, LinearProgramming/Manifold planners) —
so the OSS layer never imports fb. The provider is *injected* (not registered by
import side effect), so the wiring is explicit, typed, and testable.
"""

from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable, Union

from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
    SortBy,
)
from torchrec.distributed.planner.perf_models import NoopStorageModel
from torchrec.distributed.planner.planners import (
    EmbeddingPlannerBase,
    EmbeddingShardingPlanner,
)
from torchrec.distributed.planner.proposers import (
    EmbeddingOffloadScaleupProposer,
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
    InferenceStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    HardwareConfig,
    KernelConfig,
    ParameterConstraints,
    Partitioner,
    PerfModel,
    PlannerConfig,
    PlannerSessionContext,
    PlannerVariant,
    Proposer,
    ShardEstimator,
    ShardingPlanRequest,
    StorageReservation,
    StorageReservationPolicy,
    Topology,
    TopologyFactory,
    TrainerConfig,
)
from torchrec.distributed.types import PipelineType

_BYTES_PER_GB: int = 1024**3
# Matches EmbeddingShardingPlanner's own heuristical default, so an unset
# percentage reserves the same fraction the OSS planner would by default.
_DEFAULT_RESERVATION_PERCENTAGE: float = 0.15

# OSS proposer_type selectors -> factory (legacy simple selection path; the
# richer proposer_config path is handled in _build_proposer).
_OSS_PROPOSERS: Dict[str, Callable[[], Proposer]] = {
    "greedy": GreedyProposer,
    "grid_search": GridSearchProposer,
    "uniform": UniformProposer,
}


@runtime_checkable
class PlannerProvider(Protocol):
    """Builds the platform-specific pieces the planner flow needs.

    Injected into the orchestrator so the OSS layer never imports fb: the OSS
    default is ``DefaultPlannerProvider``; Meta overrides it (HUM topology,
    LinearProgramming/Manifold planners) via ``FbPlannerProvider`` in
    ``torchrec/fb``.
    """

    def build_topology(self, sku: str, request: ShardingPlanRequest) -> Topology:
        """Build the ``Topology`` for ``sku`` (the target GPU-SKU)."""
        ...

    def build_storage_reservation(
        self, sku: str, request: ShardingPlanRequest
    ) -> StorageReservation:
        """Resolve the StorageReservation for ``sku`` from the request."""
        ...

    def build_planner(
        self,
        sku: str,
        *,
        topology: Topology,
        storage_reservation: StorageReservation,
        ctx: PlannerSessionContext,
    ) -> EmbeddingPlannerBase:
        """Construct the planner for ``ctx.request``'s ``planner_variant``.

        ``sku`` is the target being planned (the fb provider uses it to resolve
        the hardware capability for the HW-based perf estimator).
        """
        ...


class DefaultPlannerProvider(PlannerProvider):
    """OSS provider: topology from the request's caps + a faithfully-reconstructed
    OSS ``EmbeddingShardingPlanner``.

    Inherits the ``PlannerProvider`` protocol so conformance is explicit. Meta's
    ``FbPlannerProvider`` (``torchrec/fb``) subclasses this, overriding the
    fb-specific hooks (HUM topology, HW perf estimator, fb proposers/partitioner,
    fb perf model) and ``build_planner`` (LP/Manifold), delegating OSS variants
    back here via ``super()``.
    """

    def build_topology(self, sku: str, request: ShardingPlanRequest) -> Topology:
        # OSS builds from the request's explicit caps; the SKU -> hardware mapping
        # is a Meta (HUM) concern handled by FbPlannerProvider. ``sku`` is unused
        # here but is part of the seam the fb override keys on.
        trainer_config = TrainerConfig(
            world_size=request.world_size,
            local_world_size=request.local_world_size,
            pod_size=request.pod_size,
            hbm_cap_bytes=(
                int(request.hbm_gb * _BYTES_PER_GB)
                if request.hbm_gb is not None
                else None
            ),
            ddr_cap_bytes=(
                int(request.ddr_gb * _BYTES_PER_GB)
                if request.ddr_gb is not None
                else None
            ),
        )
        # Honor the caller's runtime device so the plan's shard placements match
        # the model's device (cpu/cuda/mtia); unset -> KernelConfig's own default.
        kernel_config = (
            KernelConfig(compute_device=request.compute_device)
            if request.compute_device
            else KernelConfig()
        )
        return TopologyFactory.create_topology(
            trainer_config=trainer_config,
            hardware_config=HardwareConfig(),
            kernel_config=kernel_config,
        )

    def build_storage_reservation(
        self, sku: str, request: ShardingPlanRequest
    ) -> StorageReservation:
        # A caller-supplied reservation wins; otherwise map the config policy to
        # the matching OSS StorageReservation. ``sku`` is unused here but is part
        # of the seam (fb may reserve per-SKU).
        if request.storage_reservation is not None:
            return request.storage_reservation
        cfg = request.planner_config
        percentage = (
            cfg.storage_reservation_percentage
            if cfg.storage_reservation_percentage is not None
            else _DEFAULT_RESERVATION_PERCENTAGE
        )
        policy = cfg.storage_reservation_policy
        if policy in (
            StorageReservationPolicy.UNSET,
            StorageReservationPolicy.HEURISTICAL,
        ):
            return HeuristicalStorageReservation(percentage=percentage)
        if policy == StorageReservationPolicy.FIXED_PERCENTAGE:
            return FixedPercentageStorageReservation(percentage=percentage)
        if policy == StorageReservationPolicy.INFERENCE:
            return InferenceStorageReservation(percentage=percentage)
        raise NotImplementedError(
            f"storage_reservation_policy {policy.value!r} ({policy.name}) is not "
            "buildable by DefaultPlannerProvider (SKU_AWARE is planned)."
        )

    # ---- Overridable construction hooks (fb overrides the fb-specific ones) ----

    def _build_perf_estimator(
        self,
        sku: str,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]],
        ctx: PlannerSessionContext,
    ) -> ShardEstimator:
        # OSS uses the default perf estimator; fb overrides with the HW-capability
        # estimator when use_hardware_based_compute is set.
        return EmbeddingPerfEstimator(
            topology=topology,
            constraints=constraints,  # pyre-ignore[6]
        )

    def _build_storage_estimator(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]],
        cfg: PlannerConfig,
    ) -> ShardEstimator:
        pipeline_type = (
            PipelineType(cfg.pipeline_type)
            if cfg.pipeline_type is not None
            else PipelineType.NONE
        )
        return EmbeddingStorageEstimator(
            topology=topology,
            constraints=constraints,  # pyre-ignore[6]
            pipeline_type=pipeline_type,
        )

    def _build_enumerator(
        self, sku: str, topology: Topology, ctx: PlannerSessionContext
    ) -> Enumerator:
        cfg = ctx.request.planner_config
        constraints = ctx.request.constraints
        return EmbeddingEnumerator(
            topology=topology,
            batch_size=ctx.request.batch_size,
            constraints=constraints,
            estimator=[
                self._build_perf_estimator(sku, topology, constraints, ctx),
                self._build_storage_estimator(topology, constraints, cfg),
            ],
        )

    def _build_partitioner(self, cfg: PlannerConfig) -> Optional[Partitioner]:
        if cfg.partitioner_type == "greedy_perf":
            sort_by = (
                SortBy(cfg.partitioner_sort_by)
                if cfg.partitioner_sort_by is not None
                else SortBy.STORAGE
            )
            return GreedyPerfPartitioner(
                sort_by=sort_by, balance_modules=cfg.balance_modules
            )
        if cfg.partitioner_type == "memory_balanced":
            kwargs: Dict[str, object] = {"balance_modules": cfg.balance_modules}
            if cfg.memory_balanced_max_search_count is not None:
                kwargs["max_search_count"] = cfg.memory_balanced_max_search_count
            if cfg.memory_balanced_tolerance is not None:
                kwargs["tolerance"] = cfg.memory_balanced_tolerance
            return MemoryBalancedPartitioner(**kwargs)  # pyre-ignore[6]
        return None  # None / fb-only selectors -> planner default / fb override

    def _build_performance_model(
        self, cfg: PlannerConfig, topology: Topology
    ) -> Optional[PerfModel]:
        if cfg.performance_model == "storage":
            return NoopStorageModel(topology)
        # "table_size" (NoopTableSizeModel) is fb -> handled by the fb override.
        return None

    def _build_proposer(
        self, cfg: PlannerConfig, local_world_size: int
    ) -> Optional[Union[Proposer, List[Proposer]]]:
        # local_world_size is unused by the OSS proposers here; it is part of the
        # hook signature so the fb override (DynamicColDim / grouped-DCD, which take
        # local_world_size) can consume it without a signature divergence.
        # (proposer_type / proposer_config mutual exclusion is enforced upstream by
        # PlannerConfig.__post_init__, so it is not re-checked here.)
        pc = cfg.proposer_config
        if pc is None:
            # Legacy simple selection via proposer_type. Unrecognized / fb-only
            # values return None (planner default / fb override) rather than raising,
            # since the OSS base cannot distinguish an fb selector from a typo.
            if cfg.proposer_type is not None:
                factory = _OSS_PROPOSERS.get(cfg.proposer_type)
                return factory() if factory is not None else None
            return None
        if pc.kind == "default":
            return None  # keep the planner's default proposer set
        if pc.kind == "greedy":
            return GreedyProposer()
        if pc.kind == "uniform":
            return UniformProposer()
        if pc.kind == "grid_search":
            return (
                GridSearchProposer(max_proposals=pc.max_proposals)
                if pc.max_proposals is not None
                else GridSearchProposer()
            )
        if pc.kind == "embedding_offload_scaleup":
            # Only forward use_depth when set, so an unset value keeps the
            # proposer's own default rather than hardcoding it here.
            eos_kwargs: Dict[str, object] = {}
            if pc.use_depth is not None:
                eos_kwargs["use_depth"] = pc.use_depth
            return EmbeddingOffloadScaleupProposer(**eos_kwargs)  # pyre-ignore[6]
        # dynamic_col_dim / embedding_offload_cache_scaling / grouped_dynamic_col_dim
        # are fb -> fb override.
        return None

    def build_planner(
        self,
        sku: str,
        *,
        topology: Topology,
        storage_reservation: StorageReservation,
        ctx: PlannerSessionContext,
    ) -> EmbeddingPlannerBase:
        variant = ctx.request.planner_config.planner_variant
        if variant not in (PlannerVariant.UNSET, PlannerVariant.OSS):
            raise NotImplementedError(
                f"DefaultPlannerProvider (OSS) does not build planner_variant "
                f"{variant.value!r} ({variant.name}); LinearProgramming/Manifold "
                "live in torchrec/fb and are built by FbPlannerProvider. Inject "
                "the fb provider (e.g. via the fb create_sharding_plan entrypoint)."
            )
        cfg = ctx.request.planner_config
        return EmbeddingShardingPlanner(
            topology=topology,
            batch_size=ctx.request.batch_size,
            enumerator=self._build_enumerator(sku, topology, ctx),
            storage_reservation=storage_reservation,
            proposer=self._build_proposer(cfg, ctx.request.local_world_size),
            partitioner=self._build_partitioner(cfg),
            performance_model=self._build_performance_model(cfg, topology),
            constraints=ctx.request.constraints,
            stats=ctx.stats,  # built by the orchestrator's reporter; None -> default
            debug=cfg.debug,
            timeout_seconds=cfg.timeout_seconds,
        )
