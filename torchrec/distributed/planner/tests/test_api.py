#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, List, Optional

import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.planner.api import (
    ProductionPlannerOrchestrator,
    ShardingPlannerAPI,
)
from torchrec.distributed.planner.protocols import PlannerExecutor
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
)


class _RecordingExecutor:
    """PlannerExecutor test double that records how the API invokes it.

    The executor builds topology/storage/planner internally now, so the API only
    hands it ``sku``/``ctx``/``pg``; the double records those.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        self.calls.append({"sku": sku, "pg": pg})
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=None,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=123,
            estimated_max_ddr_bytes=456,
        )


class _RaisingExecutor:
    """PlannerExecutor double that raises a non-PlannerError for one SKU.

    Models an unexpected failure (model factory, enum coercion, HUM lookup, ...)
    that is not a ``PlannerError`` and so is not already turned into a
    success=False result inside the executor.
    """

    def __init__(self, fail_sku: str) -> None:
        self.calls: List[str] = []
        self._fail_sku = fail_sku

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        self.calls.append(sku)
        if sku == self._fail_sku:
            raise RuntimeError(f"boom for {sku}")
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=None,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=1,
            estimated_max_ddr_bytes=1,
        )


class _FixedTargetsOrchestrator(ShardingPlannerAPI):
    """Concrete ShardingPlannerAPI whose _targets is a fixed SKU list.

    Exercises the shared template loop in ShardingPlannerAPI.plan without the
    (pending) production launcher_hardware -> SKU resolution.
    """

    def __init__(self, executor: PlannerExecutor, targets: List[str]) -> None:
        super().__init__(executor)
        self._fixed_targets = targets

    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        return self._fixed_targets


class _IsolatingOrchestrator(_FixedTargetsOrchestrator):
    """Orchestrator that opts into per-target error isolation (like dry-run)."""

    def _isolate_target_errors(self) -> bool:
        return True


class ShardingPlannerAPITest(unittest.TestCase):
    def _request(self, **kwargs: object) -> ShardingPlanRequest:
        defaults: Dict[str, object] = {
            "model": nn.Linear(10, 10),
            "sharders": [],
            "world_size": 8,
            "local_world_size": 8,
            "batch_size": 512,
        }
        defaults.update(kwargs)
        return ShardingPlanRequest(**defaults)  # pyre-ignore[6]

    def _orchestrator(
        self, executor: PlannerExecutor, targets: List[str]
    ) -> _FixedTargetsOrchestrator:
        return _FixedTargetsOrchestrator(executor, targets)

    def test_plan_delegates_and_aggregates_each_target(self) -> None:
        # The template loop delegates to the executor per target and aggregates
        # into both the returned map and ctx.results.
        executor = _RecordingExecutor()
        request = self._request()
        ctx = PlannerSessionContext(request=request, results={})

        results = self._orchestrator(executor, ["H100", "GB200"]).plan(request, ctx)

        self.assertEqual(set(results), {"H100", "GB200"})
        self.assertEqual(results["H100"].sku, "H100")
        self.assertEqual(results["GB200"].estimated_max_hbm_bytes, 123)
        self.assertIs(ctx.results["GB200"], results["GB200"])
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual({c["sku"] for c in executor.calls}, {"H100", "GB200"})

    def test_plan_runs_local_when_not_distributed(self) -> None:
        # Offline (no initialized process group) => API passes pg=None so the
        # executor runs a local plan rather than collective_plan.
        self.assertFalse(dist.is_available() and dist.is_initialized())
        executor = _RecordingExecutor()
        request = self._request()
        self._orchestrator(executor, ["H100"]).plan(
            request, PlannerSessionContext(request=request, results={})
        )
        self.assertIsNone(executor.calls[0]["pg"])

    def test_isolation_disabled_by_default(self) -> None:
        # Base default is fail-fast (production must not swallow an unexpected error).
        self.assertFalse(
            self._orchestrator(_RecordingExecutor(), [])._isolate_target_errors()
        )

    def test_fail_fast_propagates_unexpected_error(self) -> None:
        # Without isolation, an unexpected error aborts the whole sweep.
        executor = _RaisingExecutor(fail_sku="H100")
        request = self._request()
        with self.assertRaises(RuntimeError):
            self._orchestrator(executor, ["H100", "GB200"]).plan(
                request, PlannerSessionContext(request=request, results={})
            )

    def test_isolate_errors_collects_and_continues(self) -> None:
        # With isolation on, a failing SKU becomes a success=False result and the
        # remaining SKUs still plan (collect-and-continue sweep).
        executor = _RaisingExecutor(fail_sku="H100")
        request = self._request()
        ctx = PlannerSessionContext(request=request, results={})
        results = _IsolatingOrchestrator(executor, ["H100", "GB200"]).plan(request, ctx)
        self.assertEqual(set(results), {"H100", "GB200"})
        self.assertFalse(results["H100"].success)
        self.assertIn("RuntimeError", results["H100"].planner_failure_reason or "")
        self.assertTrue(results["GB200"].success)
        self.assertEqual(executor.calls, ["H100", "GB200"])

    def test_injected_broadcast_pg_is_used(self) -> None:
        # A caller-injected broadcast PG (e.g. APF's Gloo cpu_ctl_dist_pg) is
        # preferred over the default WORLD group and threaded to the executor,
        # so the rank-0 plan broadcast runs over the injected group.
        fake_pg = cast(dist.ProcessGroup, object())
        executor = _RecordingExecutor()
        request = self._request()
        ProductionPlannerOrchestrator(executor, "H100", broadcast_pg=fake_pg).plan(
            request, PlannerSessionContext(request=request, results={})
        )
        self.assertIs(executor.calls[0]["pg"], fake_pg)

    def test_production_orchestrator_plans_resolved_sku(self) -> None:
        # The SKU is resolved by the caller (the fb entrypoint via HUM) and passed
        # at construction; the prod orchestrator plans for exactly that SKU.
        executor = _RecordingExecutor()
        request = self._request()
        results = ProductionPlannerOrchestrator(executor, "GRANDTETON").plan(
            request, PlannerSessionContext(request=request, results={})
        )
        self.assertEqual(set(results), {"GRANDTETON"})
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0]["sku"], "GRANDTETON")
