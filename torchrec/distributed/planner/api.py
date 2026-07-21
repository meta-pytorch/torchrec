#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from typing import Dict, List, Mapping, Optional

import torch.distributed as dist
from torchrec.distributed.planner.protocols import PlannerExecutor
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
)

logger: logging.Logger = logging.getLogger(__name__)


class ShardingPlannerAPI(abc.ABC):
    """Abstract base for sharding planner orchestrators.

    Owns the shared per-target planning loop as a template method (``plan``):
    for each target it delegates to the injected PlannerExecutor — which builds
    the topology, resolves the storage reservation, and constructs and runs the
    planner for that SKU — aggregating per-target ShardingPlanResults onto
    ``ctx.results`` and into the returned map.

    Subclasses supply only what differs — the set of targets — via ``_targets``:
    ``ProductionPlannerOrchestrator`` resolves launcher_hardware -> concrete SKU;
    ``DryRunOrchestrator`` expands the request's SKU list. The executor is
    injected (its PlannerProvider selects OSS vs Meta topology/storage/planner),
    and the broadcast process group is owned here.

    (There is intentionally no separate concrete "orchestrator" class beyond the
    prod/dry-run subclasses — the shared loop lives here, so the subclasses stay
    thin and there is a single contract.)
    """

    def __init__(
        self,
        executor: PlannerExecutor,
        broadcast_pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._executor = executor
        # Optional caller-injected process group for the rank-0 plan + broadcast.
        # APF passes its Gloo control group (cpu_ctl_dist_pg) here to avoid NCCL
        # object-broadcast corruption on some SKUs (e.g. GB300). When None,
        # _get_broadcast_pg falls back to the default WORLD group (offline -> None).
        self._broadcast_pg = broadcast_pg

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
        (``ctx.request``). This method reads only the target set from ``request``
        (via ``_targets``); the executor reads model/sharders/constraints/config
        from ``ctx.request``. Callers build ``ctx = PlannerSessionContext(request=
        request, ...)`` so the two never diverge.

        Distributed transfer: only the ShardingPlan is broadcast to all ranks —
        that is all a rank needs to build DMP. The rich ShardingPlanResult
        metadata (sharding_options, peak estimates, solve time) is produced only
        on the planning rank (rank 0) and is intentionally NOT broadcast; it is
        for logging/observability, which is a rank-0 concern. So on non-zero
        ranks the result carries the plan with empty metadata.
        """
        # request is ctx.request by construction (see the invariant above); the
        # executor reads model/sharders/constraints/config from ctx.request and
        # builds the topology, storage reservation, and planner for each SKU.
        # Enforce it so a caller passing a divergent request fails fast rather
        # than silently planning targets from one request against another. A raise
        # (not assert) so the check survives ``python -O``.
        if request is not ctx.request:
            raise ValueError("request must be the same object as ctx.request")
        pg = self._get_broadcast_pg()
        results: Dict[str, ShardingPlanResult] = {}
        # dict.fromkeys dedups while preserving order: a subclass returning a
        # duplicate SKU would otherwise trigger a redundant executor.run and
        # silently overwrite the earlier result.
        for sku in dict.fromkeys(self._targets(request)):
            try:
                result = self._executor.run(sku=sku, ctx=ctx, pg=pg)
                result = self._finalize_result(sku, result, request)
            except Exception as e:
                # Fail-fast by default (production plans a single SKU: a broken
                # plan must surface, not be swallowed). ``DryRunOrchestrator``
                # opts into collect-and-continue so one bad SKU (model factory,
                # enum coercion, HUM lookup, ...) does not lose the rest of the
                # what-if sweep -- it is recorded as an unsuccessful result and
                # the loop keeps going. (``PlannerError`` is already turned into a
                # success=False result inside the executor; this governs only the
                # unexpected errors that would otherwise propagate.)
                if not self._isolate_target_errors():
                    raise
                logger.warning(
                    "[planner] target sku=%s failed; recording as unsuccessful "
                    "and continuing the sweep",
                    sku,
                    exc_info=True,
                )
                result = self._finalize_result(
                    sku, self._failed_result(sku, ctx, e), request
                )
            ctx.results[sku] = result
            results[sku] = result
        return results

    def _isolate_target_errors(self) -> bool:
        """Whether a single target's unexpected failure is isolated to that target.

        Base returns False: production fails fast (it plans one SKU and any
        unexpected error must surface). ``DryRunOrchestrator`` overrides to True
        so a multi-SKU sweep returns partial results instead of aborting on the
        first bad SKU. Governs only non-``PlannerError`` exceptions; the executor
        already converts ``PlannerError`` into a success=False result on both paths.
        """
        return False

    def _failed_result(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        error: Exception,
    ) -> ShardingPlanResult:
        """Build a success=False result for a target that raised an unexpected error.

        Used only on the collect-and-continue path so the failing SKU is surfaced
        (with its error as ``planner_failure_reason``) rather than lost.
        """
        return ShardingPlanResult(
            sku=sku,
            success=False,
            sharding_plan=None,
            planner_failure_reason=f"{type(error).__name__}: {error}",
            estimated_max_hbm_bytes=0,
            estimated_max_ddr_bytes=0,
            request_hash=ctx.request.request_hash,
            request_id=ctx.request.request_id,
        )

    def _finalize_result(
        self,
        sku: str,
        result: ShardingPlanResult,
        request: ShardingPlanRequest,
    ) -> ShardingPlanResult:
        """Post-process the per-SKU result before it is aggregated.

        Base implementation returns the result unchanged. Subclasses may enrich
        it — e.g. DryRunOrchestrator returns a DryRunResult that adds the per-SKU
        request fingerprint. The return type is covariant, so a subclass returning
        a ShardingPlanResult subclass narrows the whole result map accordingly.
        """
        return result

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

        The plan is broadcast as a pickled Python object, and NCCL
        object-broadcast is known to corrupt on some SKUs (e.g. GB300). To avoid
        that, a caller in a distributed context should inject a dedicated Gloo/CPU
        group via the constructor (``broadcast_pg``) — APF passes its existing Gloo
        control group (``cpu_ctl_dist_pg``), validated to span the same ranks as
        WORLD. When an injected group is present it is used; otherwise this falls
        back to the default (NCCL) WORLD group when distributed is initialized, and
        None offline (single-rank / dry-run -> local plan).
        """
        if self._broadcast_pg is not None:
            return self._broadcast_pg
        if dist.is_available() and dist.is_initialized():
            return dist.group.WORLD
        return None


class ProductionPlannerOrchestrator(ShardingPlannerAPI):
    """Production orchestrator: plans for the single concrete SKU the job runs on.

    The concrete GPU-SKU is resolved by the caller and passed at construction;
    ``_targets`` simply returns it. Resolving launcher_hardware -> SKU goes through
    HUM hardware detection, which is fb-only, so it happens outside this OSS class:
    the Meta ``create_sharding_plan`` entrypoint resolves the SKU via
    ``get_training_hardware`` before constructing this orchestrator, keeping the
    OSS layer free of any fb/HUM import.
    """

    def __init__(
        self,
        executor: PlannerExecutor,
        sku: str,
        broadcast_pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(executor, broadcast_pg=broadcast_pg)
        self._sku = sku

    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        return [self._sku]
