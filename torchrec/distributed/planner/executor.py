#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.planner.protocols import PlannerExecutor
from torchrec.distributed.planner.provider import (
    DefaultPlannerProvider,
    PlannerProvider,
)
from torchrec.distributed.planner.types import (
    PlannerError,
    PlannerSessionContext,
    ShardingOption,
    ShardingOptionDetail,
    ShardingPlanResult,
)


def _peak_per_rank_storage(best_plan: List[ShardingOption]) -> Tuple[int, int]:
    """Peak per-rank (HBM, DDR) in bytes across the chosen shards.

    Aggregates each shard's storage onto its assigned rank and returns the
    maximum over ranks — the memory the tightest device must hold, which is what
    ShardingPlanResult.estimated_max_*_bytes reports. Shards without resolved
    storage or rank are skipped, so the reported peak is a lower bound if the
    planner left any chosen shard un-costed.
    """
    hbm_by_rank: Dict[int, int] = {}
    ddr_by_rank: Dict[int, int] = {}
    for option in best_plan:
        for shard in option.shards:
            if shard.storage is None or shard.rank is None:
                continue
            hbm_by_rank[shard.rank] = hbm_by_rank.get(shard.rank, 0) + shard.storage.hbm
            ddr_by_rank[shard.rank] = ddr_by_rank.get(shard.rank, 0) + shard.storage.ddr
    return (
        max(hbm_by_rank.values(), default=0),
        max(ddr_by_rank.values(), default=0),
    )


class DefaultPlannerExecutor(PlannerExecutor):
    """The single concrete PlannerExecutor behind ShardingPlannerAPI.

    There is intentionally ONE executor, not one per planner variant: ``run`` is
    the single integration point that builds the topology, storage reservation,
    and planner for the request's SKU (via the injected PlannerProvider), runs it
    (collectively when the API supplies a process group, else locally), and
    returns a ShardingPlanResult — including the per-table ``sharding_options``
    breakdown and per-rank memory estimates. This is the single place that
    replaces the topology/storage/planner-selection-and-invocation block each
    framework (Pyper/MVAI/APS) currently duplicates.

    The PlannerProvider is the OSS-vs-Meta seam: ``DefaultPlannerProvider`` (OSS)
    builds the OSS topology + EmbeddingShardingPlanner; ``FbPlannerProvider``
    (torchrec/fb) overrides it for HUM topology and the LinearProgramming/Manifold
    planners — so this OSS executor never imports fb, and variant support is
    injected (not registered by import side effect). Inherits the
    ``PlannerExecutor`` protocol so conformance is explicit, not just structural.
    """

    def __init__(self, provider: Optional[PlannerProvider] = None) -> None:
        # Default to the OSS provider; Meta binaries inject FbPlannerProvider
        # (e.g. via the fb create_sharding_plan entrypoint).
        self._provider: PlannerProvider = (
            provider if provider is not None else DefaultPlannerProvider()
        )

    def run(
        self,
        sku: str,
        ctx: PlannerSessionContext,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlanResult:
        # model/sharders come from the request (not passed separately);
        # materialize the model if it is a factory.
        model = ctx.request.model
        if not isinstance(model, nn.Module):
            # Guard callable first so a non-module, non-factory value gets the
            # intended message instead of a bare "X object is not callable".
            if not callable(model):
                raise TypeError(
                    "request.model must be an nn.Module or a factory callable "
                    f"returning one, got {type(model).__name__}"
                )
            model = model()
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "request.model factory must return an nn.Module, got "
                    f"{type(model).__name__}"
                )
        sharders = ctx.request.sharders

        # On the collective path only rank 0 plans and populates the per-shard
        # breakdown; other ranks get the broadcast plan but no breakdown/estimates.
        # Flag which result is authoritative so aggregators can filter. Local path
        # (pg=None) always plans on the caller's rank. Use pg.rank() (rank within
        # the broadcast group) rather than dist.get_rank(pg): the latter routes
        # through _get_default_group() and so requires an initialized default WORLD
        # group, a dependency the legacy path never had.
        is_planning_rank = pg is None or pg.rank() == 0

        # Build every per-SKU input and the planner itself via the injected
        # provider — the single dispatch point across OSS/LP/Manifold variants.
        topology = self._provider.build_topology(sku, ctx.request)
        storage_reservation = self._provider.build_storage_reservation(sku, ctx.request)
        planner = self._provider.build_planner(
            sku,
            topology=topology,
            storage_reservation=storage_reservation,
            ctx=ctx,
        )

        start = time.perf_counter()
        try:
            # pg is owned by the API: present -> collective (rank-0 plan +
            # broadcast); None -> local single-rank plan.
            if pg is not None:
                sharding_plan = planner.collective_plan(model, sharders, pg)
            else:
                sharding_plan = planner.plan(model, sharders)
        except PlannerError as e:
            return ShardingPlanResult(
                sku=sku,
                success=False,
                sharding_plan=None,
                planner_failure_reason=str(e),
                estimated_max_hbm_bytes=0,
                estimated_max_ddr_bytes=0,
                request_id=ctx.request.request_id,
                request_hash=ctx.request.request_hash,
                solve_time_ms=(time.perf_counter() - start) * 1000.0,
                is_planning_rank=is_planning_rank,
            )
        solve_time_ms = (time.perf_counter() - start) * 1000.0

        # get_selected_options() is the chosen List[ShardingOption] with per-shard
        # storage/perf populated; it is the only source of the full cost breakdown
        # that the deployment-facing ShardingPlan drops. It is a documented planner
        # contract (EmbeddingPlannerBase.get_selected_options), so a planner that
        # doesn't expose its plan fails loudly here rather than silently returning
        # empty options + zero-byte estimates (the old getattr(_best_plan) read).
        #
        # In the collective path only the planning rank (rank 0) computes the
        # plan, so on other ranks this is empty: those ranks return the
        # authoritative broadcast sharding_plan but an empty sharding_options
        # breakdown and zero estimates (the per-shard cost detail is not carried
        # in the broadcast ShardingPlan, so it cannot be recomputed off-rank).
        # Consumers that need the breakdown/estimates should read the planning
        # rank's result.
        best_plan: List[ShardingOption] = planner.get_selected_options()
        sharding_options = tuple(
            ShardingOptionDetail.from_sharding_option(option) for option in best_plan
        )
        max_hbm_bytes, max_ddr_bytes = _peak_per_rank_storage(best_plan)
        return ShardingPlanResult(
            sku=sku,
            success=True,
            sharding_plan=sharding_plan,
            planner_failure_reason=None,
            estimated_max_hbm_bytes=max_hbm_bytes,
            estimated_max_ddr_bytes=max_ddr_bytes,
            request_id=ctx.request.request_id,
            request_hash=ctx.request.request_hash,
            sharding_options=sharding_options,
            solve_time_ms=solve_time_ms,
            is_planning_rank=is_planning_rank,
        )
