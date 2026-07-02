#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Callable, cast, Dict, List, Optional
from unittest.mock import MagicMock

import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.planner.api import (
    ProductionPlannerOrchestrator,
    ShardingPlannerAPI,
)
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
    ShardingPlanResult,
    StorageReservation,
    Topology,
)


class _RecordingExecutor:
    """PlannerExecutor test double that records how the API invokes it."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []

    def run(
        self,
        sku: str,
        topology: Topology,
        storage_reservation: StorageReservation,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        self.calls.append({"sku": sku, "pg": pg, "topology": topology})
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=None,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=123,
            estimated_max_ddr_bytes=456,
        )


class _FakeResolver:
    """StorageReservationResolver test double returning a placeholder reservation."""

    def resolve(self, sku: str, ctx: PlannerSessionContext) -> StorageReservation:
        return cast(StorageReservation, MagicMock())


def _topology(sku: str, request: ShardingPlanRequest) -> Topology:
    return Topology(
        world_size=request.world_size, compute_device="cuda", hbm_cap=1024**3
    )


class _FixedTargetsOrchestrator(ShardingPlannerAPI):
    """Concrete ShardingPlannerAPI whose _targets is a fixed SKU list.

    Exercises the shared template loop in ShardingPlannerAPI.plan without the
    (pending) production launcher_hardware -> SKU resolution.
    """

    def __init__(
        self,
        executor: _RecordingExecutor,
        storage_reservation_resolver: _FakeResolver,
        topology_builder: Callable[[str, ShardingPlanRequest], Topology],
        targets: List[str],
    ) -> None:
        super().__init__(executor, storage_reservation_resolver, topology_builder)
        self._fixed_targets = targets

    def _targets(self, request: ShardingPlanRequest) -> List[str]:
        return self._fixed_targets


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
        self, executor: _RecordingExecutor, targets: List[str]
    ) -> _FixedTargetsOrchestrator:
        return _FixedTargetsOrchestrator(
            executor=executor,
            storage_reservation_resolver=_FakeResolver(),
            topology_builder=_topology,
            targets=targets,
        )

    def test_plan_delegates_and_aggregates_each_target(self) -> None:
        # The template loop builds a topology, delegates to the executor per
        # target, and aggregates into both the returned map and ctx.results.
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
        self.assertEqual(cast(Topology, executor.calls[0]["topology"]).world_size, 8)

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

    def test_production_orchestrator_target_resolution_pending(self) -> None:
        # Production launcher_hardware -> SKU resolution is not yet wired, so the
        # prod orchestrator raises rather than emit a launcher-type value as a SKU.
        request = self._request(launcher_hardware="TC_ANY")
        planner = ProductionPlannerOrchestrator(
            executor=_RecordingExecutor(),
            storage_reservation_resolver=_FakeResolver(),
            topology_builder=_topology,
        )
        with self.assertRaisesRegex(NotImplementedError, "launcher_hardware"):
            planner.plan(request, PlannerSessionContext(request=request, results={}))
