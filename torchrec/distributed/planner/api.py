#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Callable, Dict, List, Mapping, Optional

import torch.distributed as dist
from torchrec.distributed.planner.protocols import (
    PlannerExecutor,
    StorageReservationResolver,
)
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
    Topology,
)


class ShardingPlannerAPI(abc.ABC):
    """Abstract base for sharding planner orchestrators.

    Owns the shared per-target planning loop as a template method (``plan``):
    for each target it builds a topology, resolves the storage reservation, and
    delegates the planner invocation to the injected PlannerExecutor, aggregating
    per-target ShardingPlanResults onto ``ctx.results`` and into the returned map.

    Subclasses supply only what differs — the set of targets — via ``_targets``:
    ``ProductionPlannerOrchestrator`` resolves launcher_hardware -> concrete SKU;
    ``DryRunOrchestrator`` expands the request's SKU list. Framework/dry-run
    behavior is injected (executor, storage-reservation resolver, topology
    builder), and the broadcast process group is owned here.

    (There is intentionally no separate concrete "orchestrator" class beyond the
    prod/dry-run subclasses — the shared loop lives here, so the subclasses stay
    thin and there is a single contract.)
    """

    def __init__(
        self,
        executor: PlannerExecutor,
        storage_reservation_resolver: StorageReservationResolver,
        topology_builder: Callable[[str, ShardingPlanRequest], Topology],
    ) -> None:
        self._executor = executor
        self._storage_reservation_resolver = storage_reservation_resolver
        self._topology_builder = topology_builder

    def plan(
        self,
        request: ShardingPlanRequest,
        ctx: PlannerSessionContext,
    ) -> Mapping[str, ShardingPlanResult]:
        """Run the shared per-target planning loop and aggregate the results.

        Returns a Mapping (read-only, covariant in the value type) keyed by SKU,
        so subclasses may narrow the result type — e.g. DryRunOrchestrator
        returns Mapping[str, DryRunResult].

        Invariant: ``request`` must be the same object carried by ``ctx``
        (``ctx.request``). This method reads target/topology inputs from
        ``request`` while the executor reads model/sharders/constraints from
        ``ctx.request``; callers build ``ctx = PlannerSessionContext(request=
        request, ...)`` so the two never diverge.

        Distributed transfer: only the ShardingPlan is broadcast to all ranks —
        that is all a rank needs to build DMP. The rich ShardingPlanResult
        metadata (sharding_options, peak estimates, solve time) is produced only
        on the planning rank (rank 0) and is intentionally NOT broadcast; it is
        for logging/observability, which is a rank-0 concern. So on non-zero
        ranks the result carries the plan with empty metadata.
        """
        # request is ctx.request by construction (see the invariant above); the
        # executor reads model/sharders/constraints from ctx.request.
        pg = self._get_broadcast_pg()
        results: Dict[str, ShardingPlanResult] = {}
        for sku in self._targets(request):
            topology = self._topology_builder(sku, request)
            storage_reservation = self._storage_reservation_resolver.resolve(sku, ctx)
            # The executor reads model/sharders/constraints from ctx.request; only
            # the per-target, API-resolved inputs are passed explicitly.
            result = self._executor.run(
                sku=sku,
                topology=topology,
                storage_reservation=storage_reservation,
                ctx=ctx,
                pg=pg,
            )
            ctx.results[sku] = result
            results[sku] = result
        return results

    @abc.abstractmethod
    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        """Concrete GPU-SKUs to plan for.

        Each value becomes a ShardingPlanResult.sku and a key in the result map,
        so it must be a GPU-SKU (e.g. "H100"/"GB200") — not a launcher_hardware
        value ("ZIONEX"/"TC_ANY"). ProductionPlannerOrchestrator resolves
        launcher_hardware -> SKU; DryRunOrchestrator returns the request's
        sku_list.
        """
        ...

    def _get_broadcast_pg(self) -> Optional[dist.ProcessGroup]:
        """Process group for the plan broadcast, owned by the API.

        Returns a group in a distributed context so the executor runs
        ``collective_plan`` (rank-0 plan + broadcast); returns None offline /
        single-rank so the executor runs a local ``plan``.

        Process-group *initialization* is intentionally NOT done here — it is the
        caller's (trainer/framework's) responsibility, via
        ``torch.distributed.init_process_group`` during training setup, before
        planning. This method never initializes ``dist`` (doing so would clash
        with the trainer's own PG/backend/timeout); it only detects whether a
        distributed context already exists via ``dist.is_initialized()``. When it
        is not initialized (offline / dry-run / single process) it returns None
        and the executor plans locally.

        FOLLOW-UP: the plan is broadcast as a pickled Python object, and NCCL
        object-broadcast is known to corrupt on some SKUs (e.g. GB300). So this
        should return a dedicated Gloo/CPU group — reuse an existing Gloo control
        group if one exists (as APF's cpu_ctl_dist_pg does), else lazily create
        one spanning the training ranks (cached; created collectively on all
        ranks). Owning it here makes that mitigation the unified default for all
        frameworks. Returning the default (NCCL) WORLD group is a placeholder
        until the Gloo group is wired.
        """
        if dist.is_available() and dist.is_initialized():
            return dist.group.WORLD
        return None


class ProductionPlannerOrchestrator(ShardingPlannerAPI):
    """Production orchestrator: plans for the single concrete SKU the job runs on.

    Resolves the request's launcher_hardware (which may be an abstract type such
    as "TC_ANY") to the concrete GPU-SKU the job is scheduled on. That hardware
    resolution goes through HUM (get_training_hardware); wiring it is a follow-up,
    so ``_targets`` currently raises rather than emit a launcher-type value as a
    SKU.
    """

    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        raise NotImplementedError(
            "ProductionPlannerOrchestrator must resolve launcher_hardware to a "
            "concrete GPU-SKU via HUM hardware detection; that wiring is pending."
        )
