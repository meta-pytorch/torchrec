#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Extension-point protocols (structural interfaces) for the sharding planner.

These live in their own module — separate from the data types (``types.py``) and
the concrete implementations (``api.py``, ``executor.py``) — on purpose:

- Dependency inversion / leaf module: a Protocol is the seam that consumers
  depend on and implementers conform to. This module imports only ``types`` (and
  torch), so both the consumer (``api.py`` takes these as constructor deps) and
  the implementers (``executor.py`` conforms to ``PlannerExecutor``;
  ``dry_run/protocols.py`` re-exports them) can import the contract without
  pulling in the heavier api/executor modules — which keeps the import graph
  acyclic (defining these in ``api.py`` would risk an api<->executor cycle).
- Separation of concerns / discoverability: the planner's pluggable seams live
  in one place, distinct from data (``types.py``) and behavior (``api``/``executor``).
- Fine-grained build target: the ``:protocols`` Buck target depends only on
  ``:types`` + torch, so consumers pull just the lightweight contract rather than
  the concrete implementations.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch.distributed as dist
from torchrec.distributed.planner.types import PlannerSessionContext, ShardingPlanResult


@runtime_checkable
class PlannerExecutor(Protocol):
    """Protocol for building and running the planner for a single target (SKU).

    Implementations construct the planner for ``ctx.request``'s variant (via an
    injected PlannerProvider that also builds the topology and resolves the
    storage reservation), run it, and return a ShardingPlanResult with the
    sharding plan or failure details.
    """

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        """Build and run the planner for one target (``sku``); return its result.

        Only the per-target inputs are passed explicitly: ``sku`` (the target
        being planned; it labels/keys the result) and ``pg``. Everything else is
        read from ``ctx.request`` — model, sharders, constraints, batch size, and
        planner config — rather than duplicated as parameters; the implementation
        materializes ``ctx.request.model`` if it is a factory. The topology,
        storage reservation, and planner for this SKU are all built inside ``run``
        (the implementation owns a PlannerProvider), so no pre-built topology or
        reservation is passed in.

        ``pg`` is supplied and owned by the ShardingPlannerAPI, not the caller:
        when provided (distributed prod) the implementation runs the planner
        collectively (``collective_plan``); when None (offline/dry-run, single
        rank) it runs locally (``plan``).
        """
        ...
