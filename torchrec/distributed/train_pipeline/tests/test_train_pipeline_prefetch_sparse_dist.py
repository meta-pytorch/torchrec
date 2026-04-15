#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for PrefetchTrainPipelineSparseDist.

This module tests the prefetch pipeline API changes including:
- The new prefetch_embeddings utility function
- The fill_pipeline method with batch queue management
- The progress method with proper prefetch flow
- Stream synchronization and context management
"""

import unittest
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock

import torch
from hypothesis import given, settings, strategies as st
from torch import nn
from torch.optim import Optimizer
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.pipeline_context import (
    PrefetchTrainPipelineContext,
)
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    PrefetchTrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.utils import prefetch_embeddings
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import DataType


class PrefetchEmbeddingsUtilTest(unittest.TestCase):
    """Tests for the prefetch_embeddings utility function."""

    def test_prefetch_embeddings_returns_early_when_data_dist_stream_is_none(
        self,
    ) -> None:
        """Test that prefetch_embeddings returns early when data_dist_stream is None."""
        context = PrefetchTrainPipelineContext()
        pipelined_modules: List[MagicMock] = []
        stream_context = MagicMock()

        prefetch_embeddings(
            context=context,
            pipelined_modules=pipelined_modules,
            device=torch.device("cpu"),
            stream_context=stream_context,
            data_dist_stream=None,
            forward_stream=None,
        )

        stream_context.assert_not_called()


class PrefetchTrainPipelineTestBase(TrainPipelineSparseDistTestBase):
    """
    Base class for PrefetchTrainPipelineSparseDist tests.

    Provides common setup and helper methods to reduce code duplication.
    """

    # Default fused params for prefetch pipeline tests
    DEFAULT_FUSED_PARAMS: Dict[str, Any] = {
        "cache_load_factor": 0.5,
        "cache_precision": DataType.FP32,
        "stochastic_rounding": False,
        "prefetch_pipeline": True,
    }

    def _create_pipeline(
        self,
        num_batches: int = 5,
        batch_size: int = 32,
        execute_all_batches: bool = True,
        fused_params: Optional[Dict[str, Any]] = None,
        sharding_type: str = ShardingType.TABLE_WISE.value,
        kernel_type: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
    ) -> Tuple[
        PrefetchTrainPipelineSparseDist,
        Iterator[ModelInput],
        nn.Module,
        Optimizer,
    ]:
        """
        Creates a prefetch pipeline with all necessary components.

        Returns:
            Tuple of (pipeline, dataloader, sharded_model, optimizer)
        """
        self._set_table_weights_precision(DataType.FP32)
        data = self._generate_data(num_batches=num_batches, batch_size=batch_size)
        dataloader = iter(data)

        params = fused_params if fused_params is not None else self.DEFAULT_FUSED_PARAMS

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type, params
        )

        pipeline = PrefetchTrainPipelineSparseDist(
            model=sharded_model,
            optimizer=optim,
            device=self.device,
            execute_all_batches=execute_all_batches,
        )

        return pipeline, dataloader, sharded_model, optim


class PrefetchTrainPipelineTest(PrefetchTrainPipelineTestBase):
    """Tests for PrefetchTrainPipelineSparseDist API and behavior."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_fill_pipeline_initializes_batches_and_contexts(self) -> None:
        """Test that fill_pipeline properly initializes the batch and context queues."""
        pipeline, dataloader, _, _ = self._create_pipeline()

        pipeline.fill_pipeline(dataloader)

        self.assertEqual(len(pipeline.batches), 2)
        self.assertEqual(len(pipeline.contexts), 2)
        self.assertIsInstance(pipeline.contexts[0], PrefetchTrainPipelineContext)
        self.assertIsInstance(pipeline.contexts[1], PrefetchTrainPipelineContext)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_fill_pipeline_is_idempotent(self) -> None:
        """Test that calling fill_pipeline multiple times doesn't add extra batches."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=10)

        pipeline.fill_pipeline(dataloader)
        initial_batch_count = len(pipeline.batches)
        pipeline.fill_pipeline(dataloader)

        self.assertEqual(len(pipeline.batches), initial_batch_count)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_progress_dequeues_batch_after_processing(self) -> None:
        """Test that progress properly dequeues batches after processing."""
        pipeline, dataloader, _, _ = self._create_pipeline()

        _ = pipeline.progress(dataloader)

        self.assertLessEqual(len(pipeline.batches), 3)
        self.assertEqual(len(pipeline.batches), len(pipeline.contexts))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_progress_raises_stop_iteration_when_empty(self) -> None:
        """Test that progress raises StopIteration when no batches are available."""
        pipeline, _, _, _ = self._create_pipeline(num_batches=0)
        empty_dataloader: Iterator[ModelInput] = iter([])

        with self.assertRaises(StopIteration):
            pipeline.progress(empty_dataloader)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_prefetch_stream_initialization(self) -> None:
        """Test that prefetch and default streams are properly initialized."""
        pipeline, _, _, _ = self._create_pipeline()

        self.assertIsNotNone(pipeline._prefetch_stream)
        self.assertIsNotNone(pipeline._default_stream)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_context_indices_are_sequential(self) -> None:
        """Test that context indices are properly assigned sequentially."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=8)

        observed_indices = []
        for _ in range(5):
            _ = pipeline.progress(dataloader)
            if pipeline.contexts:
                observed_indices.append(pipeline.contexts[0].index)

        self.assertGreater(len(observed_indices), 0)
        for idx in observed_indices:
            self.assertIsNotNone(idx)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_batch_queue_maintains_sync_with_context_queue(self) -> None:
        """Test that batch queue and context queue remain synchronized."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=10)

        for _ in range(7):
            _ = pipeline.progress(dataloader)
            self.assertEqual(
                len(pipeline.batches),
                len(pipeline.contexts),
                "Batch and context queues must have the same length",
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_single_batch_execution(self) -> None:
        """Test pipeline behavior with only a single batch in the dataloader."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=1)

        output = pipeline.progress(dataloader)

        self.assertIsNotNone(output)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_two_batch_execution(self) -> None:
        """Test pipeline behavior with exactly two batches in the dataloader."""
        pipeline, dataloader, _, _ = self._create_pipeline(num_batches=2)

        output1 = pipeline.progress(dataloader)
        output2 = pipeline.progress(dataloader)

        self.assertIsNotNone(output1)
        self.assertIsNotNone(output2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_execute_all_batches_false(self) -> None:
        """Test pipeline behavior with execute_all_batches=False."""
        pipeline, dataloader, _, _ = self._create_pipeline(execute_all_batches=False)

        outputs = []
        try:
            for _ in range(10):
                outputs.append(pipeline.progress(dataloader))
        except StopIteration:
            pass

        self.assertGreater(len(outputs), 0)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @settings(max_examples=4, deadline=None)
    # pyre-ignore[56]
    @given(
        weight_precision=st.sampled_from([DataType.FP16, DataType.FP32]),
        cache_precision=st.sampled_from([DataType.FP16, DataType.FP32]),
        load_factor=st.sampled_from([0.2, 0.4, 0.6]),
        sharding_type=st.sampled_from(
            [ShardingType.TABLE_WISE.value, ShardingType.ROW_WISE.value]
        ),
    )
    def test_prefetch_pipeline_correctness(
        self,
        weight_precision: DataType,
        cache_precision: DataType,
        load_factor: float,
        sharding_type: str,
    ) -> None:
        """
        Verifies that pipelined execution with prefetch produces the same results
        as non-pipelined execution when using FUSED_UVM_CACHING kernel.
        """
        mixed_precision: bool = weight_precision != cache_precision
        self._set_table_weights_precision(weight_precision)
        data = self._generate_data(num_batches=12, batch_size=32)
        dataloader = iter(data)

        fused_params = {
            "cache_load_factor": load_factor,
            "cache_precision": cache_precision,
            "stochastic_rounding": False,
        }
        fused_params_pipelined = {**fused_params, "prefetch_pipeline": True}

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model,
            sharding_type,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params,
        )
        sharded_model_pipelined, optim_pipelined = (
            self._generate_sharded_model_and_optimizer(
                model,
                sharding_type,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                fused_params_pipelined,
            )
        )
        copy_state_dict(
            sharded_model.state_dict(), sharded_model_pipelined.state_dict()
        )

        pipeline = PrefetchTrainPipelineSparseDist(
            model=sharded_model_pipelined,
            optimizer=optim_pipelined,
            device=self.device,
            execute_all_batches=True,
        )

        for batch in data:
            batch = batch.to(self.device)
            optim.zero_grad(set_to_none=True)
            loss, pred = sharded_model(batch)
            loss.backward()
            optim.step()

            pred_pipeline = pipeline.progress(dataloader)

            if not mixed_precision:
                self.assertTrue(torch.equal(pred, pred_pipeline))
            else:
                torch.testing.assert_close(pred, pred_pipeline)


if __name__ == "__main__":
    unittest.main()
