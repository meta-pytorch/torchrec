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

``DefaultPlannerProvider`` is the OSS implementation — it builds a topology from
the request's explicit caps and constructs the OSS ``EmbeddingShardingPlanner``.
The Meta-internal provider (``torchrec/fb``) subclasses it to add HUM-based
topology and the LinearProgramming/Manifold planners, so the OSS layer never
imports fb. The provider is *injected* (not registered by import side effect),
so the wiring is explicit, typed, and testable.
"""

from typing import Callable, Dict, Optional, Protocol, runtime_checkable

from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.planners import (
    EmbeddingPlannerBase,
    EmbeddingShardingPlanner,
)
from torchrec.distributed.planner.proposers import (
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
    InferenceStorageReservation,
)
from torchrec.distributed.planner.types import (
    HardwareConfig,
    KernelConfig,
    Partitioner,
    PlannerConfig,
    PlannerSessionContext,
    PlannerVariant,
    Proposer,
    ShardingPlanRequest,
    StorageReservation,
    StorageReservationPolicy,
    Topology,
    TopologyFactory,
    TrainerConfig,
)

_BYTES_PER_GB: int = 1024**3
# Matches EmbeddingShardingPlanner's own heuristical default, so an unset
# percentage reserves the same fraction the OSS planner would by default.
_DEFAULT_RESERVATION_PERCENTAGE: float = 0.15


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
        *,
        topology: Topology,
        storage_reservation: StorageReservation,
        ctx: PlannerSessionContext,
    ) -> EmbeddingPlannerBase:
        """Construct the planner for ``ctx.request``'s ``planner_variant``."""
        ...


# Component factories: map PlannerConfig scalar selectors -> concrete OSS
# components. Selectors naming fb-only components (e.g. "dynamic_col_dim",
# "memory_balanced") return None so the OSS planner keeps its default; the fb
# provider maps those in torchrec/fb.
_OSS_PROPOSERS: Dict[str, Callable[[], Proposer]] = {
    "greedy": GreedyProposer,
    "grid_search": GridSearchProposer,
    "uniform": UniformProposer,
}


def _build_proposer(cfg: PlannerConfig) -> Optional[Proposer]:
    if cfg.proposer_type is None:
        return None  # keep the planner's default proposer set
    factory = _OSS_PROPOSERS.get(cfg.proposer_type)
    return factory() if factory is not None else None


def _build_partitioner(cfg: PlannerConfig) -> Optional[Partitioner]:
    if cfg.partitioner_type == "greedy_perf":
        return GreedyPerfPartitioner()
    return None  # None / fb-only selectors -> planner default (GreedyPerfPartitioner)


class DefaultPlannerProvider(PlannerProvider):
    """OSS provider: a topology from the request's caps + the OSS planner.

    Inherits the ``PlannerProvider`` protocol so conformance is explicit (not
    just structural). Meta's ``FbPlannerProvider`` (``torchrec/fb``) subclasses
    this, overriding ``build_topology`` (HUM) and ``build_planner`` (LP/Manifold)
    while delegating the OSS variants back here via ``super()``.
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
        return TopologyFactory.create_topology(
            trainer_config=trainer_config,
            hardware_config=HardwareConfig(),
            kernel_config=KernelConfig(),
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

    def build_planner(
        self,
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
            storage_reservation=storage_reservation,
            constraints=ctx.request.constraints,
            proposer=_build_proposer(cfg),
            partitioner=_build_partitioner(cfg),
            debug=cfg.debug,
            timeout_seconds=cfg.timeout_seconds,
        )
