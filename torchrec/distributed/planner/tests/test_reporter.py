#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch.nn as nn
from torchrec.distributed.planner.reporter import DefaultPlanReporter, PlanReporter
from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanRequest,
)


def _ctx() -> PlannerSessionContext:
    request = ShardingPlanRequest(
        model=nn.Linear(4, 4),
        sharders=[],
        world_size=2,
        local_world_size=2,
        batch_size=512,
    )
    return PlannerSessionContext(request=request, results={})


class DefaultPlanReporterTest(unittest.TestCase):
    def test_conforms_to_protocol(self) -> None:
        self.assertIsInstance(DefaultPlanReporter(), PlanReporter)

    def test_build_stats_console_only(self) -> None:
        stats = DefaultPlanReporter().build_stats(_ctx(), "H100")
        self.assertEqual(len(stats), 1)
        self.assertIsInstance(stats[0], EmbeddingStats)

    def test_build_stats_returns_fresh_instances_per_call(self) -> None:
        # The per-target loop calls build_stats once per SKU; each call must
        # return fresh (stateful) sinks so a multi-SKU sweep never reuses them.
        reporter = DefaultPlanReporter()
        ctx = _ctx()
        first = reporter.build_stats(ctx, "H100")
        second = reporter.build_stats(ctx, "GB200")
        self.assertIsNot(first[0], second[0])
