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
from torchrec.distributed.planner.executor import DefaultPlannerExecutor
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
)
from torchrec.distributed.planner.proposers import GreedyProposer, UniformProposer
from torchrec.distributed.planner.protocols import PlannerExecutor
from torchrec.distributed.planner.provider import DefaultPlannerProvider
from torchrec.distributed.planner.types import (
    PlannerConfig,
    PlannerSessionContext,
    PlannerVariant,
    ShardingPlanRequest,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class DefaultPlannerExecutorTest(unittest.TestCase):
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

    def _ctx(self, model: nn.Module, **cfg: object) -> PlannerSessionContext:
        request = ShardingPlanRequest(
            model=model,
            sharders=self._sharders(),
            world_size=2,
            local_world_size=2,
            batch_size=512,
            planner_config=PlannerConfig(**cfg),  # pyre-ignore[6]
        )
        return PlannerSessionContext(request=request, results={})

    def test_conforms_to_protocol(self) -> None:
        self.assertIsInstance(DefaultPlannerExecutor(), PlannerExecutor)

    def test_embedding_plan_end_to_end(self) -> None:
        # The executor builds topology + storage reservation + planner from the
        # request via its (default OSS) provider; only sku/ctx/pg are passed.
        model = self._model()
        result = DefaultPlannerExecutor().run(sku="H100", ctx=self._ctx(model), pg=None)
        self.assertTrue(result.success, result.planner_failure_reason)
        self.assertEqual(result.sku, "H100")
        self.assertIsNotNone(result.sharding_plan)
        self.assertIsNone(result.planner_failure_reason)
        # The per-table sharding-option table is populated from the chosen plan.
        self.assertTrue(result.sharding_options)
        self.assertTrue(all(so.fqn for so in result.sharding_options))
        # Peak per-rank HBM is a positive estimate for a non-empty plan.
        self.assertGreater(result.estimated_max_hbm_bytes, 0)
        self.assertIsNotNone(result.solve_time_ms)

    def test_run_populates_session_observability(self) -> None:
        # The executor records per-phase timing + the modeled topology, storage
        # reservation, and hardware provenance onto the session context.
        model = self._model()
        ctx = self._ctx(model)
        DefaultPlannerExecutor().run(sku="H100", ctx=ctx, pg=None)
        for phase in (
            "topology_build:H100",
            "storage_reservation:H100",
            "planner_construction:H100",
            "plan_call:H100",
        ):
            self.assertIn(phase, ctx.timing)
            self.assertGreaterEqual(ctx.timing[phase], 0.0)
        # world_size=2, local_world_size=2 -> exactly one cached topology entry.
        self.assertEqual(len(ctx.topology_cache), 1)
        self.assertIn("H100", ctx.storage_reservations_used)
        # OSS builds from the request caps, so that is the recorded HW source.
        self.assertEqual(ctx.hw_source["H100"], "request_caps")
        self.assertIn("H100", ctx.hw_overrides_applied)

    def test_embedding_plan_materializes_model_factory(self) -> None:
        # A factory model on the request is materialized by the executor
        # (model is read from ctx.request, not passed to run()).
        built = self._model()
        request = ShardingPlanRequest(
            model=lambda: built,
            sharders=self._sharders(),
            world_size=2,
            local_world_size=2,
            batch_size=512,
        )
        ctx = PlannerSessionContext(request=request, results={})
        result = DefaultPlannerExecutor().run(sku="H100", ctx=ctx, pg=None)
        self.assertTrue(result.success, result.planner_failure_reason)
        self.assertTrue(result.sharding_options)

    def test_unsupported_variant_raises(self) -> None:
        # The OSS provider only builds OSS variants; LinearProgramming/Manifold
        # live in torchrec/fb (FbPlannerProvider), so the OSS default rejects them.
        model = self._model()
        with self.assertRaisesRegex(
            NotImplementedError, "does not build planner_variant"
        ):
            DefaultPlannerExecutor().run(
                sku="H100",
                ctx=self._ctx(model, planner_variant=PlannerVariant.LINEAR_PROGRAMMING),
                pg=None,
            )


class BuildComponentsFromConfigTest(unittest.TestCase):
    """The OSS provider turns PlannerConfig scalar selectors into concrete OSS
    components (overridable hooks; fb-only selectors fall through to fb)."""

    def setUp(self) -> None:
        self.provider = DefaultPlannerProvider()

    def test_build_proposer_maps_known_selectors(self) -> None:
        self.assertIsInstance(
            self.provider._build_proposer(
                PlannerConfig(proposer_type="greedy"), local_world_size=8
            ),
            GreedyProposer,
        )
        self.assertIsInstance(
            self.provider._build_proposer(
                PlannerConfig(proposer_type="uniform"), local_world_size=8
            ),
            UniformProposer,
        )

    def test_build_proposer_none_and_fb_selectors_use_default(self) -> None:
        # None -> planner default set; fb-only selectors aren't OSS-buildable.
        self.assertIsNone(
            self.provider._build_proposer(PlannerConfig(), local_world_size=8)
        )
        self.assertIsNone(
            self.provider._build_proposer(
                PlannerConfig(proposer_type="dynamic_col_dim"), local_world_size=8
            )
        )

    def test_build_topology_records_request_caps_source(self) -> None:
        # OSS resolves capabilities from the request's explicit caps (no registry /
        # live detection), so the recorded provenance is "request_caps".
        request = ShardingPlanRequest(
            model=nn.Linear(4, 4),
            sharders=[],
            world_size=2,
            local_world_size=2,
            batch_size=512,
        )
        ctx = PlannerSessionContext(request=request, results={})
        self.provider.build_topology("H100", request, ctx)
        self.assertEqual(ctx.hw_source["H100"], "request_caps")
        self.assertIn("H100", ctx.hw_overrides_applied)

    def test_build_partitioner(self) -> None:
        self.assertIsInstance(
            self.provider._build_partitioner(
                PlannerConfig(partitioner_type="greedy_perf")
            ),
            GreedyPerfPartitioner,
        )
        self.assertIsInstance(
            self.provider._build_partitioner(
                PlannerConfig(partitioner_type="memory_balanced")
            ),
            MemoryBalancedPartitioner,
        )
        self.assertIsNone(self.provider._build_partitioner(PlannerConfig()))
