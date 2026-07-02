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
from torchrec.distributed.planner.types import (
    PlannerSessionContext,
    ShardingPlanResult,
    StorageReservation,
    Topology,
)


@runtime_checkable
class PlannerExecutor(Protocol):
    """Protocol for executing the embedding planner on a single topology.

    Implementations wrap the EmbeddingShardingPlanner and return a
    ShardingPlanResult with the sharding plan or failure details.
    """

    def run(
        self,
        sku: str,
        topology: Topology,
        storage_reservation: StorageReservation,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        """Run the planner for one target/topology and return its result.

        Only the per-target, API-resolved inputs are passed explicitly:
        ``sku`` (the target being planned; it labels/keys the result),
        ``topology`` (built per-SKU), ``storage_reservation`` (resolved per-SKU),
        and ``pg``. Everything else is read from ``ctx.request`` — model,
        sharders, constraints, batch size, and planner config — rather than
        duplicated as parameters; the implementation materializes
        ``ctx.request.model`` if it is a factory. (These were previously separate
        arguments, but the executor already reads ``ctx.request`` for batch size
        and planner config, so passing them again was redundant.)

        ``pg`` is supplied and owned by the ShardingPlannerAPI, not the caller:
        when provided (distributed prod) the implementation runs the planner
        collectively (``collective_plan``); when None (offline/dry-run, single
        rank) it runs locally (``plan``).
        """
        ...


@runtime_checkable
class StorageReservationResolver(Protocol):
    """Protocol for resolving a StorageReservation strategy.

    Implementations select and configure the appropriate StorageReservation for a
    target from the session context — reading what they need from ``ctx.request``
    (training_framework, planner_config's storage_reservation_policy/percentage,
    memory caps) and optionally recording the result into ``ctx``.
    """

    def resolve(
        self,
        sku: str,
        ctx: PlannerSessionContext,
    ) -> StorageReservation:
        """Return the StorageReservation for one target (``sku``).

        Like PlannerExecutor.run, this takes only the per-target ``sku`` plus the
        session ``ctx``; the framework and request are read from ``ctx.request``
        rather than duplicated as parameters.
        """
        ...
