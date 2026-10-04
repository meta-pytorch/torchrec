#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# pyre-strict

import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings, strategies as st
from torch.optim import Optimizer
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
)
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    TrainPipelineFusedSparseDist,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_FUSED_PARAMS: dict[str, bool] = {"stochastic_rounding": False}
_NUM_BATCHES = 12
_BATCH_SIZE = 32
_DATA_PARALLEL_FEATURE = "data_parallel_feature"


class _DPShardedEmbeddingTrainModel(torch.nn.Module):
    def __init__(
        self,
        sequence: ShardedEmbeddingCollection,
        pooled: ShardedEmbeddingBagCollection,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.sequence = sequence
        self.pooled = pooled
        self.dense_weight = torch.nn.Parameter(torch.ones(4, device=device))

    def forward(self, features: KeyedJaggedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_embeddings = self.sequence(features)[_DATA_PARALLEL_FEATURE].values()
        pooled_embeddings = self.pooled(features)[_DATA_PARALLEL_FEATURE]
        prediction = torch.sum(
            (sequence_embeddings + pooled_embeddings) * self.dense_weight
        )
        return prediction.square(), prediction


class TrainPipelineFusedSparseDistTest(TrainPipelineSparseDistTestBase):
    def _create_dp_sharded_ec(self) -> ShardedEmbeddingCollection:
        module = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="data_parallel_table",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=[_DATA_PARALLEL_FEATURE],
                )
            ],
            device=self.device,
        )
        sharder = EmbeddingCollectionSharder()
        plan = construct_module_sharding_plan(
            module,
            per_param_sharding={"data_parallel_table": data_parallel()},
            local_size=1,
            world_size=1,
            device_type=self.device.type,
            sharder=cast(ModuleSharder[torch.nn.Module], sharder),
        )
        return sharder.shard(
            module=module,
            params=plan,
            env=ShardingEnv.from_process_group(self.pg),
            device=self.device,
        )

    def _create_dp_sharded_ebc(self) -> ShardedEmbeddingBagCollection:
        module = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="data_parallel_table",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=[_DATA_PARALLEL_FEATURE],
                )
            ],
            device=self.device,
        )
        sharder = EmbeddingBagCollectionSharder()
        plan = construct_module_sharding_plan(
            module,
            per_param_sharding={"data_parallel_table": data_parallel()},
            local_size=1,
            world_size=1,
            device_type=self.device.type,
            sharder=cast(ModuleSharder[torch.nn.Module], sharder),
        )
        return sharder.shard(
            module=module,
            params=plan,
            env=ShardingEnv.from_process_group(self.pg),
            device=self.device,
        )

    def _run_deferred_dp_lookup_parity(
        self,
        reference_model: _DPShardedEmbeddingTrainModel,
        pipeline_model: _DPShardedEmbeddingTrainModel,
    ) -> TrainPipelineFusedSparseDist[KeyedJaggedTensor, torch.Tensor]:
        copy_state_dict(reference_model.state_dict(), pipeline_model.state_dict())

        # Citrine C2: use foreach updates for multi-tensor optimizers.
        reference_optimizer = torch.optim.SGD(
            reference_model.parameters(), lr=0.1, foreach=True
        )
        pipeline_optimizer = torch.optim.SGD(
            pipeline_model.parameters(), lr=0.1, foreach=True
        )
        batches = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=[_DATA_PARALLEL_FEATURE],
                values=torch.tensor([0, 1], device=self.device),
                lengths=torch.tensor([1, 1], device=self.device),
            )
            for _ in range(4)
        ]
        pipeline = TrainPipelineFusedSparseDist(
            model=pipeline_model,
            optimizer=pipeline_optimizer,
            device=self.device,
            execute_all_batches=True,
            emb_lookup_stream="current",
            embedding_lookup_before_optimizer=True,
            defer_dp_lookup=True,
        )
        dataloader = iter(batches)

        for batch in batches[:2]:
            reference_optimizer.zero_grad()
            loss, expected = reference_model(batch)
            loss.backward()
            reference_optimizer.step()

            actual = pipeline.progress(dataloader)

            torch.testing.assert_close(expected, actual)

        return pipeline

    def _assert_equal_to_non_pipelined(
        self,
        sharding_type: str,
        verify_lookup_before_optimizer: bool = False,
        **pipeline_kwargs: object,
    ) -> None:
        """Runs the pipeline against a non-pipelined reference and compares.

        Args:
            sharding_type: sharding type for both models.
            verify_lookup_before_optimizer: verify that the next batch's lookup
                has started when the optimizer step begins.
            pipeline_kwargs: extra kwargs forwarded to TrainPipelineFusedSparseDist.
        """
        data = self._generate_data(
            num_batches=_NUM_BATCHES,
            batch_size=_BATCH_SIZE,
        )
        dataloader = iter(data)

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, EmbeddingComputeKernel.FUSED.value, _FUSED_PARAMS
        )
        (
            sharded_model_pipelined,
            optim_pipelined,
        ) = self._generate_sharded_model_and_optimizer(
            model, sharding_type, EmbeddingComputeKernel.FUSED.value, _FUSED_PARAMS
        )
        copy_state_dict(
            sharded_model.state_dict(), sharded_model_pipelined.state_dict()
        )

        pipeline = TrainPipelineFusedSparseDist(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
            # pyre-ignore[6]: kwargs are forwarded verbatim.
            **pipeline_kwargs,
        )

        lookup_verifications = 0

        def verify_lookup_started(
            _optimizer: Optimizer,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            nonlocal lookup_verifications
            context = cast(EmbeddingTrainPipelineContext, pipeline.contexts[1])
            self.assertTrue(context.embedding_a2a_requests)
            lookup_verifications += 1

        if verify_lookup_before_optimizer:
            optim_pipelined.register_step_pre_hook(verify_lookup_started)

        for batch in data[:-2]:
            batch = batch.to(self.device)
            optim.zero_grad()
            loss, pred = sharded_model(batch)
            loss.backward()
            optim.step()

            pred_pipeline = pipeline.progress(dataloader)

            torch.testing.assert_close(pred, pred_pipeline)

        if verify_lookup_before_optimizer:
            self.assertEqual(lookup_verifications, len(data) - 2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch("torch._utils_internal.justknobs_check", return_value=True)
    def test_new_emb_lookup_stream_equal_to_non_pipelined(
        self, _mock_justknobs_check: MagicMock
    ) -> None:
        """
        Tests that running the embedding lookup on a dedicated stream, rather
        than reusing the data-dist stream, still matches non-pipelined results.
        The lookup then consumes input-dist output across a stream boundary.
        """
        self._assert_equal_to_non_pipelined(
            sharding_type=ShardingType.TABLE_WISE.value,
            emb_lookup_stream="new",
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch("torch._utils_internal.justknobs_check", return_value=True)
    def test_lookup_before_optimizer_equal_to_non_pipelined(
        self, _mock_justknobs_check: MagicMock
    ) -> None:
        """
        Tests hoisting batch i+1's embedding lookup above the dense optimizer.

        Parity holds only because every table here is on the fused compute
        kernel, so its weights are final once backward returns and the dense
        optimizer step the lookup now precedes updates nothing the lookup reads.
        Tables the dense optimizer owns instead -- DATA_PARALLEL shards, or any
        table on a non-fused kernel -- would be read one iteration stale, which
        this configuration deliberately does not cover.
        """
        for sharding_type in (
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
        ):
            with self.subTest(sharding_type=sharding_type):
                self._assert_equal_to_non_pipelined(
                    sharding_type=sharding_type,
                    emb_lookup_stream="current",
                    embedding_lookup_before_optimizer=True,
                    verify_lookup_before_optimizer=True,
                )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_embedding_lookup_placements_are_mutually_exclusive(self) -> None:
        model = self._setup_model()
        sharded_model, optimizer = self._generate_sharded_model_and_optimizer(
            model,
            ShardingType.TABLE_WISE.value,
            EmbeddingComputeKernel.FUSED.value,
            _FUSED_PARAMS,
        )

        with self.assertRaisesRegex(ValueError, "alternative placements"):
            TrainPipelineFusedSparseDist(
                model=sharded_model,
                optimizer=optimizer,
                device=self.device,
                embedding_lookup_after_data_dist=True,
                embedding_lookup_before_optimizer=True,
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_deferred_dp_lookup_matches_non_pipelined(self) -> None:
        """Deferred EC and EBC lookups observe the current optimizer update."""
        reference_model = _DPShardedEmbeddingTrainModel(
            self._create_dp_sharded_ec(),
            self._create_dp_sharded_ebc(),
            self.device,
        )
        pipeline_model = _DPShardedEmbeddingTrainModel(
            self._create_dp_sharded_ec(),
            self._create_dp_sharded_ebc(),
            self.device,
        )
        # Verifies numerical parity after each optimizer update.
        pipeline = self._run_deferred_dp_lookup_parity(reference_model, pipeline_model)

        # Verifies deferred DP state for both embedding context types.
        context = cast(EmbeddingTrainPipelineContext, pipeline.contexts[0])
        ec_contexts = {
            name: module_context
            for name, module_context in context.module_contexts.items()
            if isinstance(module_context, EmbeddingCollectionContext)
        }
        self.assertEqual(1, len(ec_contexts))
        self.assertTrue(all(ctx.defer_dp_lookup for ctx in ec_contexts.values()))
        ebc_contexts = {
            name: module_context
            for name, module_context in context.module_contexts.items()
            if isinstance(module_context, EmbeddingBagCollectionContext)
        }
        self.assertEqual(1, len(ebc_contexts))
        self.assertTrue(all(ctx.defer_dp_lookup for ctx in ebc_contexts.values()))
        self.assertCountEqual(
            [*ec_contexts, *ebc_contexts],
            pipeline._deferred_dp_module_context_names or [],
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # 8 combinations of the axes below; sample all of them so coverage does not
    # depend on which examples hypothesis happens to draw.
    @settings(max_examples=8, deadline=None)
    @given(
        enqueue_batch_after_forward=st.booleans(),
        embedding_lookup_after_data_dist=st.booleans(),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
    )
    def test_equal_to_non_pipelined(
        self,
        enqueue_batch_after_forward: bool,
        embedding_lookup_after_data_dist: bool,
        sharding_type: str,
    ) -> None:
        """
        Tests TrainPipelineFusedSparseDist with various parameter combinations
        produces same results as non-pipelined execution.
        """
        self._assert_equal_to_non_pipelined(
            sharding_type=sharding_type,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            embedding_lookup_after_data_dist=embedding_lookup_after_data_dist,
        )
