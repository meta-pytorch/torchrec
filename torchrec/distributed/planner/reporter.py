#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Observability seam for the sharding planner.

Plan reporting (the Stats sinks the planner logs to) is a cross-cutting concern
owned by the orchestrator — written once in the shared ``plan`` loop and reused
by dry-run and production, independent of the provider. ``PlanReporter`` builds
the sinks; the OSS default is console-only (``EmbeddingStats``). Meta's
``FbPlanReporter`` (``torchrec/fb``) adds Manifold + Scuba from
``PlanReportMetadata``. Injected into the orchestrator (not the provider), so
observability lives in one place.
"""

from typing import List, Protocol, runtime_checkable

from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.types import PlannerSessionContext, Stats


@runtime_checkable
class PlanReporter(Protocol):
    """Builds the Stats sinks the planner logs to for a planning session."""

    def build_stats(self, ctx: PlannerSessionContext, sku: str) -> List[Stats]:
        """Return the stats sinks for the SKU about to be planned.

        ``sku`` is the target being planned next; a reporter that persists a
        per-SKU artifact (e.g. the Manifold plan upload) uses it to give each SKU
        in a multi-SKU sweep its own destination instead of colliding on one path.
        Reads ``ctx.report_metadata`` if it needs framework provenance.
        """
        ...


class DefaultPlanReporter(PlanReporter):
    """OSS reporter: console stats only (``EmbeddingStats``).

    Meta's ``FbPlanReporter`` subclasses this to add the Manifold/Scuba sinks
    from ``ctx.report_metadata``; the console sink is common to both.
    """

    def build_stats(self, ctx: PlannerSessionContext, sku: str) -> List[Stats]:
        # Console stats are SKU-independent (no persisted artifact), so sku is
        # unused here; it is part of the seam for reporters that write per-SKU.
        return [EmbeddingStats()]
