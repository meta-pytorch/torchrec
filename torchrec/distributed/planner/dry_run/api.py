#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, List, Mapping

from torchrec.distributed.planner.api import ShardingPlannerAPI
from torchrec.distributed.planner.dry_run.types import DryRunRequest, DryRunResult
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
)


class DryRunOrchestrator(ShardingPlannerAPI):
    """Dry-run orchestrator: plans every SKU in the request's ``sku_list``.

    Reuses the shared template ``plan`` from ShardingPlannerAPI and supplies the
    dry-run target set via ``_targets`` — the DryRunRequest's concrete sku_list —
    mirroring ProductionPlannerOrchestrator (which resolves launcher_hardware to a
    single SKU). The executor is injected at construction; its PlannerProvider
    selects OSS vs Meta topology/storage/planner.

    Lives in ``dry_run/api.py`` (not ``protocols.py``) because it is a concrete
    orchestrator, not a Protocol -- mirroring the base ``ShardingPlannerAPI`` in
    ``planner/api.py`` -- so ``protocols.py`` stays protocol-only and consumers
    that only need ``PlanCache`` don't transitively pull in ``planner:api``.
    """

    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        if not isinstance(request, DryRunRequest):
            raise TypeError(
                "DryRunOrchestrator requires a DryRunRequest; got "
                f"{type(request).__name__}"
            )
        # Defensive copy: callers/the shared template may sort or filter the
        # returned list, which must not mutate the request's sku_list in place.
        return list(request.sku_list)

    def _isolate_target_errors(self) -> bool:
        # A dry-run is a what-if sweep across SKUs: one bad SKU (model factory,
        # enum coercion, HUM lookup, ...) should be recorded as an unsuccessful
        # result and the remaining SKUs still planned -- collect-and-continue --
        # rather than aborting the whole sweep. (Production keeps the base
        # fail-fast, since it plans a single SKU and a broken plan must surface.)
        return True

    def plan(
        self,
        request: ShardingPlanRequest,
        ctx: PlannerSessionContext,
    ) -> Mapping[str, DryRunResult]:
        # Runs the shared template; every value is a DryRunResult (see
        # _finalize_result), so narrow the return type for dry-run callers.
        return cast(Mapping[str, DryRunResult], super().plan(request, ctx))

    def _finalize_result(
        self,
        sku: str,
        result: ShardingPlanResult,
        request: ShardingPlanRequest,
    ) -> DryRunResult:
        """Enrich the per-SKU result with its (request, SKU) fingerprint."""
        # _targets already validated the request is a DryRunRequest. from_result
        # derives request_fingerprint from (request, sku), so the top-level sku
        # and the SKU encoded in the fingerprint cannot disagree.
        return DryRunResult.from_result(cast(DryRunRequest, request), sku, result)
