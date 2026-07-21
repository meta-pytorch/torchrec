#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List

import torch
import torch.nn as nn
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.dry_run.api import DryRunOrchestrator
from torchrec.distributed.planner.dry_run.types import DryRunRequest
from torchrec.distributed.planner.executor import DefaultPlannerExecutor
from torchrec.distributed.planner.types import PlannerSessionContext, TrainingFramework
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class DryRunEndToEndTest(unittest.TestCase):
    """End-to-end dry run through the real stack.

    DryRunOrchestrator + the real DefaultPlannerExecutor + the OSS
    DefaultPlannerProvider produce one real ShardingPlanResult per SKU in
    sku_list. Runs offline (pg=None), so the executor plans locally for each
    candidate SKU. This exercises the whole template loop end to end (no mocks),
    complementing test_executor (single SKU, direct) and test_api (mocked loop).
    """

    def _model(self) -> TestSparseNN:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(4)
        ]
        return TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

    def _sharders(self) -> List[ModuleSharder[nn.Module]]:
        return cast(List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()])

    def _request(
        self, sku_list: List[str], model: object | None = None
    ) -> DryRunRequest:
        return DryRunRequest(
            model=model if model is not None else self._model(),  # pyre-ignore[6]
            sharders=self._sharders(),
            sku_list=sku_list,
            training_framework=TrainingFramework.APF,
            world_size=2,
            local_world_size=2,
            batch_size=512,
        )

    def test_plans_every_sku_end_to_end(self) -> None:
        request = self._request(["H100", "GB200"])
        ctx = PlannerSessionContext(request=request, results={})

        results = DryRunOrchestrator(DefaultPlannerExecutor()).plan(request, ctx)

        self.assertEqual(set(results), {"H100", "GB200"})
        for sku in ("H100", "GB200"):
            result = results[sku]
            self.assertTrue(result.success, result.planner_failure_reason)
            self.assertEqual(result.sku, sku)
            self.assertIsNotNone(result.sharding_plan)
            # Real plan -> per-table breakdown and a positive peak-HBM estimate.
            self.assertTrue(result.sharding_options)
            self.assertGreater(result.estimated_max_hbm_bytes, 0)
            # Aggregated onto the session context as well as the returned map.
            self.assertIs(ctx.results[sku], result)

    def test_plans_with_model_factory(self) -> None:
        # A factory model on the request is materialized by the executor.
        built = self._model()
        request = self._request(["H100"], model=lambda: built)
        ctx = PlannerSessionContext(request=request, results={})

        results = DryRunOrchestrator(DefaultPlannerExecutor()).plan(request, ctx)

        self.assertTrue(results["H100"].success, results["H100"].planner_failure_reason)
        self.assertTrue(results["H100"].sharding_options)
