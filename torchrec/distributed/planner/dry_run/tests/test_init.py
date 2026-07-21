#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torchrec.distributed.planner.dry_run as dry_run


class InitReExportTest(unittest.TestCase):
    """Verify that all public types and protocols are re-exported from __init__.py."""

    def test_types_reexported(self) -> None:
        expected_types = [
            "CacheConfig",
            "DryRunRequest",
            "DryRunResult",
            "SkuOverride",
        ]
        for name in expected_types:
            self.assertTrue(
                hasattr(dry_run, name),
                f"{name} not re-exported from dry_run.__init__",
            )

    def test_protocols_reexported(self) -> None:
        expected_protocols = [
            "CacheMetadata",
            "DryRunOrchestrator",
            "PlanCache",
            "PlannerExecutor",
        ]
        for name in expected_protocols:
            self.assertTrue(
                hasattr(dry_run, name),
                f"{name} not re-exported from dry_run.__init__",
            )

    def test_import_from_package_directly(self) -> None:
        from torchrec.distributed.planner.dry_run import (
            DryRunRequest,
            DryRunResult,
            PlanCache,
            PlannerExecutor,
        )

        self.assertIsNotNone(DryRunRequest)
        self.assertIsNotNone(DryRunResult)
        self.assertIsNotNone(PlanCache)
        self.assertIsNotNone(PlannerExecutor)

    def test_types_are_same_objects(self) -> None:
        from torchrec.distributed.planner.dry_run.types import DryRunRequest

        self.assertIs(dry_run.DryRunRequest, DryRunRequest)

    def test_protocols_are_same_objects(self) -> None:
        from torchrec.distributed.planner.dry_run.protocols import PlanCache

        self.assertIs(dry_run.PlanCache, PlanCache)
