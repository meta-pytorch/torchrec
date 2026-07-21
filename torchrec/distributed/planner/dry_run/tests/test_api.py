#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional

import torch.nn as nn
from torchrec.distributed.planner.api import ShardingPlannerAPI
from torchrec.distributed.planner.dry_run.api import DryRunOrchestrator
from torchrec.distributed.planner.dry_run.types import DryRunRequest, DryRunResult
from torchrec.distributed.planner.protocols import PlannerExecutor
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
)


class _RecordingExecutor:
    """PlannerExecutor test double returning a successful result per SKU.

    The executor builds topology/storage/planner internally (via its provider);
    the double simply returns a result, so the DryRunOrchestrator can be
    exercised without a real provider.
    """

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: object = None,
    ) -> ShardingPlanResult:
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=None,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=0,
            estimated_max_ddr_bytes=0,
        )


class _FailingExecutor:
    """Test double returning a FAILED result per SKU (planner could not fit)."""

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: object = None,
    ) -> ShardingPlanResult:
        return ShardingPlanResult(
            sku=sku,
            success=False,
            sharding_plan=None,
            planner_failure_reason="OOM_HBM",
            estimated_max_hbm_bytes=0,
            estimated_max_ddr_bytes=0,
        )


class _RaisingExecutor:
    """Test double that raises for a specific SKU (models an executor blowup)."""

    def __init__(self, raise_on: str) -> None:
        self._raise_on = raise_on

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: object = None,
    ) -> ShardingPlanResult:
        if sku == self._raise_on:
            raise RuntimeError(f"executor blew up on {sku}")
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=None,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=0,
            estimated_max_ddr_bytes=0,
        )


class DryRunOrchestratorTest(unittest.TestCase):
    def _request(self, **kwargs: object) -> DryRunRequest:
        defaults: dict[str, object] = {
            "model": nn.Linear(10, 10),
            "sharders": [],
            "sku_list": ["H100"],
            "training_framework": "apf",
            "world_size": 8,
            "local_world_size": 8,
            "batch_size": 512,
        }
        defaults.update(kwargs)
        return DryRunRequest(**defaults)  # pyre-ignore[6]

    def _orchestrator(
        self, executor: Optional[PlannerExecutor] = None
    ) -> DryRunOrchestrator:
        return DryRunOrchestrator(executor or _RecordingExecutor())

    def test_is_subclass_of_sharding_planner_api(self) -> None:
        self.assertTrue(issubclass(DryRunOrchestrator, ShardingPlannerAPI))

    def test_targets_returns_sku_list(self) -> None:
        request = self._request(sku_list=["H100", "GB200"])
        self.assertEqual(self._orchestrator()._targets(request), ["H100", "GB200"])

    def test_targets_rejects_non_dry_run_request(self) -> None:
        request = ShardingPlanRequest(
            model=nn.Linear(10, 10),
            sharders=[],
            world_size=8,
            local_world_size=8,
            batch_size=512,
        )
        with self.assertRaisesRegex(TypeError, "DryRunRequest"):
            self._orchestrator()._targets(request)

    def test_plan_runs_every_sku_via_template(self) -> None:
        # E2E through the shared ShardingPlannerAPI.plan template: one result per
        # SKU in the request's sku_list, aggregated into the returned map and
        # onto ctx.results.
        request = self._request(sku_list=["H100", "GB200"])
        ctx = PlannerSessionContext(request=request, results={})
        results = self._orchestrator().plan(request, ctx)
        self.assertEqual(set(results), {"H100", "GB200"})
        self.assertTrue(results["H100"].success)
        self.assertEqual(results["GB200"].sku, "GB200")
        self.assertIs(ctx.results["H100"], results["H100"])

    def test_plan_produces_dry_run_result_with_fingerprint(self) -> None:
        # The dry-run orchestrator enriches each result into a DryRunResult
        # carrying the per-(request, SKU) fingerprint.
        request = self._request(sku_list=["H100", "GB200"])
        ctx = PlannerSessionContext(request=request, results={})
        results = self._orchestrator().plan(request, ctx)
        for sku in ("H100", "GB200"):
            self.assertIsInstance(results[sku], DryRunResult)
            self.assertEqual(results[sku].request_fingerprint, request.fingerprint(sku))
        # Distinct SKUs get distinct fingerprints.
        self.assertNotEqual(
            results["H100"].request_fingerprint, results["GB200"].request_fingerprint
        )

    def test_plan_finalizes_failed_result(self) -> None:
        # A failed per-SKU result must still be finalized into a DryRunResult with
        # its fingerprint (DryRunResult.from_result accepts success=False), not
        # dropped -- so a caller sees which SKUs could not be planned.
        request = self._request(sku_list=["H100", "GB200"])
        ctx = PlannerSessionContext(request=request, results={})
        results = self._orchestrator(_FailingExecutor()).plan(request, ctx)
        for sku in ("H100", "GB200"):
            self.assertIsInstance(results[sku], DryRunResult)
            self.assertFalse(results[sku].success)
            self.assertEqual(results[sku].planner_failure_reason, "OOM_HBM")
            self.assertEqual(results[sku].request_fingerprint, request.fingerprint(sku))

    def test_plan_isolates_executor_exception_and_continues(self) -> None:
        # A dry-run is a what-if sweep: if the executor raises an unexpected error
        # (not a PlannerError it converts to a failed result) on one SKU, that SKU
        # is recorded as an unsuccessful DryRunResult (with the error as its
        # failure reason) and the remaining SKUs are still planned -- rather than
        # aborting the whole sweep. This is the DryRunOrchestrator override of the
        # base fail-fast contract (_isolate_target_errors).
        request = self._request(sku_list=["H100", "GB200"])
        ctx = PlannerSessionContext(request=request, results={})
        results = self._orchestrator(_RaisingExecutor(raise_on="H100")).plan(
            request, ctx
        )
        self.assertEqual(set(results), {"H100", "GB200"})
        # The failing SKU: an unsuccessful DryRunResult carrying the error, with
        # its fingerprint preserved so the caller sees which SKU could not plan.
        self.assertIsInstance(results["H100"], DryRunResult)
        self.assertFalse(results["H100"].success)
        self.assertIn("RuntimeError", results["H100"].planner_failure_reason or "")
        self.assertIn(
            "executor blew up on H100", results["H100"].planner_failure_reason or ""
        )
        self.assertEqual(
            results["H100"].request_fingerprint, request.fingerprint("H100")
        )
        # The other SKU still planned successfully.
        self.assertTrue(results["GB200"].success)

    def test_production_path_still_fails_fast(self) -> None:
        # The isolation is dry-run-only: the base contract stays fail-fast so a
        # broken production plan is never silently swallowed.
        from torchrec.distributed.planner.api import ProductionPlannerOrchestrator

        request = ShardingPlanRequest(
            model=nn.Linear(10, 10),
            sharders=[],
            world_size=8,
            local_world_size=8,
            batch_size=512,
        )
        ctx = PlannerSessionContext(request=request, results={})
        with self.assertRaisesRegex(RuntimeError, "executor blew up on H100"):
            ProductionPlannerOrchestrator(
                _RaisingExecutor(raise_on="H100"), "H100"
            ).plan(request, ctx)
