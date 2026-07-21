#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Command-line tool for offline dry-run sharding-plan generation.

Parses dry-run arguments, builds a ``DryRunRequest``, runs it through the
dry-run orchestrator (one plan per SKU in ``--sku-list``), and prints the
per-SKU ``DryRunResult`` inline. Uses the OSS provider; a Meta counterpart can
reuse ``build_request`` / ``print_results`` and pass an orchestrator wired with
``FbPlannerProvider`` for HUM topology + LinearProgramming/Manifold.
"""

from __future__ import annotations

import argparse
from typing import cast, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.dry_run.api import DryRunOrchestrator
from torchrec.distributed.planner.dry_run.types import DryRunRequest, DryRunResult
from torchrec.distributed.planner.executor import DefaultPlannerExecutor
from torchrec.distributed.planner.types import (
    PlannerConfig,
    PlannerSessionContext,
    PlannerVariant,
    StorageReservationPolicy,
    TrainingFramework,
)
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline dry-run sharding planning across SKUs."
    )
    parser.add_argument(
        "--sku-list",
        required=True,
        help="Comma-separated SKUs to plan for (TrainingHardware names for the "
        "fb provider, e.g. 'GRANDTETON,GB200').",
    )
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--local-world-size",
        type=int,
        default=None,
        help="Devices per host; defaults to --world-size.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--training-framework",
        choices=["apf", "pyper", "mvai"],
        default="apf",
    )
    parser.add_argument("--hbm-gb", type=float, default=None)
    parser.add_argument("--ddr-gb", type=float, default=None)
    parser.add_argument(
        "--planner-variant",
        choices=["oss", "linear_programming", "manifold"],
        default="oss",
    )
    parser.add_argument(
        "--storage-reservation-policy",
        choices=["heuristical", "fixed_percentage", "inference"],
        default=None,
    )
    parser.add_argument("--storage-reservation-percentage", type=float, default=None)
    parser.add_argument("--manifold-path", default=None)
    # Synthetic-model knobs (used by the default build_model_and_sharders).
    parser.add_argument("--num-tables", type=int, default=4)
    parser.add_argument("--num-embeddings", type=int, default=100_000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    return parser


def build_model_and_sharders(
    args: argparse.Namespace,
) -> Tuple[nn.Module, List[ModuleSharder[nn.Module]]]:
    """Build a synthetic EmbeddingBagCollection so the CLI runs standalone."""
    tables = [
        EmbeddingBagConfig(
            name=f"table_{i}",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=[f"feature_{i}"],
        )
        for i in range(args.num_tables)
    ]
    model = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
    sharders = cast(List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()])
    return model, sharders


def build_planner_config(args: argparse.Namespace) -> PlannerConfig:
    policy = (
        StorageReservationPolicy(args.storage_reservation_policy)
        if args.storage_reservation_policy is not None
        else StorageReservationPolicy.UNSET
    )
    return PlannerConfig(
        planner_variant=PlannerVariant(args.planner_variant),
        storage_reservation_policy=policy,
        storage_reservation_percentage=args.storage_reservation_percentage,
        manifold_path=args.manifold_path,
    )


def build_request(
    args: argparse.Namespace,
    model: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
) -> DryRunRequest:
    sku_list = [s.strip() for s in args.sku_list.split(",") if s.strip()]
    return DryRunRequest(
        model=model,
        sharders=sharders,
        sku_list=sku_list,
        training_framework=TrainingFramework(args.training_framework),
        world_size=args.world_size,
        local_world_size=(
            args.local_world_size
            if args.local_world_size is not None
            else args.world_size
        ),
        batch_size=args.batch_size,
        hbm_gb=args.hbm_gb,
        ddr_gb=args.ddr_gb,
        planner_config=build_planner_config(args),
    )


def run(
    args: argparse.Namespace,
    orchestrator: Optional[DryRunOrchestrator] = None,
) -> Mapping[str, DryRunResult]:
    """Plan every SKU and return the per-SKU results.

    ``orchestrator`` defaults to the OSS ``DefaultPlannerExecutor``; a Meta
    caller passes one wired with ``FbPlannerProvider``.
    """
    model, sharders = build_model_and_sharders(args)
    request = build_request(args, model, sharders)
    ctx = PlannerSessionContext(request=request, results={})
    orchestrator = orchestrator or DryRunOrchestrator(DefaultPlannerExecutor())
    return orchestrator.plan(request, ctx)


def print_results(results: Mapping[str, DryRunResult]) -> None:
    print(f"Dry-run sharding plan: {len(results)} SKU(s)")
    for sku in sorted(results):
        result = results[sku]
        status = "OK" if result.success else "FAIL"
        print(
            f"  [{status}] {sku} "
            f"peak_hbm_bytes={result.estimated_max_hbm_bytes} "
            f"peak_ddr_bytes={result.estimated_max_ddr_bytes} "
            f"tables={len(result.sharding_options)} "
            f"fingerprint={result.request_fingerprint}"
        )
        if not result.success:
            print(f"    reason: {result.planner_failure_reason}")


def main(
    argv: Optional[List[str]] = None,
    orchestrator: Optional[DryRunOrchestrator] = None,
) -> int:
    args = build_arg_parser().parse_args(argv)
    results = run(args, orchestrator=orchestrator)
    print_results(results)
    # Non-zero exit if any SKU failed to plan, so scripts can gate on it.
    return 0 if all(r.success for r in results.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
