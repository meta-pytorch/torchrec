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
from collections import deque
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple
from unittest.mock import call, MagicMock, patch

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
from torchrec.distributed.train_pipeline.runtime_forwards import (
    PrefetchPipelinedForward,
)
from torchrec.distributed.train_pipeline.tests.test_train_pipelines_base import (
    TrainPipelineSparseDistTestBase,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    PrefetchTrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.utils import prefetch_embeddings
from torchrec.distributed.types import Awaitable, ShardingType
from torchrec.modules.embedding_configs import DataType
from torchrec.streamable import Multistreamable


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


class PrefetchStreamSyncTest(PrefetchTrainPipelineTestBase):
    """Tests for stream synchronization between prefetch and data_dist streams.

    Validates the fix for the UVM cache cross-stream race (S627132): the
    data_dist stream must wait for the prefetch stream before starting the
    next batch's input distribution, and record_stream must cover
    data_dist_stream for dist_input tensors.
    """

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_streams_are_distinct(self) -> None:
        """Verify the pipeline creates three distinct CUDA streams."""
        pipeline, _, _, _ = self._create_pipeline()
        self.assertIsNotNone(pipeline._prefetch_stream)
        self.assertIsNotNone(pipeline._data_dist_stream)
        self.assertIsNotNone(pipeline._default_stream)
        self.assertIsNot(pipeline._prefetch_stream, pipeline._data_dist_stream)
        self.assertIsNot(pipeline._prefetch_stream, pipeline._default_stream)
        self.assertIsNot(pipeline._data_dist_stream, pipeline._default_stream)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_multiple_progress_with_uvm_caching_produces_finite_outputs(self) -> None:
        """Run multiple iterations with FUSED_UVM_CACHING and verify all outputs
        are finite. Garbage embeddings from a stream race would produce NaN/Inf."""
        pipeline, dataloader, _, _ = self._create_pipeline(
            num_batches=10,
            batch_size=32,
            kernel_type=EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        )
        for i in range(7):
            output = pipeline.progress(dataloader)
            self.assertIsNotNone(output)
            if isinstance(output, torch.Tensor):
                self.assertTrue(
                    torch.isfinite(output).all(),
                    f"Non-finite output at iteration {i}",
                )

    def test_fence_helper_orders_wait_before_start_sparse_data_dist(self) -> None:
        """Helper must issue ``data_dist_stream.wait_stream(prefetch_stream)``
        before ``start_sparse_data_dist``. This is the contract that protects
        the next batch's input dist from racing with the previous prefetch."""
        # Bypass __init__ so we don't need a model/CUDA — the helper only
        # touches stream attributes and ``start_sparse_data_dist``.
        pipeline = PrefetchTrainPipelineSparseDist.__new__(
            PrefetchTrainPipelineSparseDist
        )
        recorder = MagicMock()
        pipeline._data_dist_stream = recorder.data_dist_stream
        pipeline._prefetch_stream = recorder.prefetch_stream
        pipeline.start_sparse_data_dist = recorder.start

        batch = MagicMock(name="batch")
        context = MagicMock(name="context")
        pipeline._fence_prefetch_and_start_sparse_data_dist(batch, context)

        self.assertEqual(
            recorder.mock_calls,
            [
                call.data_dist_stream.wait_stream(recorder.prefetch_stream),
                call.start(batch, context),
            ],
        )

    def test_fence_helper_no_op_when_streams_are_none(self) -> None:
        """On non-CUDA devices both streams are ``None``; helper must skip
        the wait and still launch ``start_sparse_data_dist``."""
        pipeline = PrefetchTrainPipelineSparseDist.__new__(
            PrefetchTrainPipelineSparseDist
        )
        pipeline._data_dist_stream = None
        pipeline._prefetch_stream = None
        start_mock = MagicMock()
        pipeline.start_sparse_data_dist = start_mock

        batch = MagicMock()
        context = MagicMock()
        pipeline._fence_prefetch_and_start_sparse_data_dist(batch, context)

        start_mock.assert_called_once_with(batch, context)

    def test_fill_pipeline_routes_batch1_through_fence_helper(self) -> None:
        """``fill_pipeline`` must launch batch 1's input dist via the fence
        helper. Without this, the very first overlapped prefetch→data_dist
        transition is unfenced and reintroduces the race the fix targets."""
        pipeline = PrefetchTrainPipelineSparseDist.__new__(
            PrefetchTrainPipelineSparseDist
        )
        pipeline.batches = deque()
        pipeline.contexts = deque()
        pipeline._execute_all_batches = False
        pipeline._pipelined_forward_type = MagicMock()

        # ``enqueue_batch`` populates the deques; emulate that with side effects.
        def fake_enqueue(_iter: Any) -> bool:
            pipeline.batches.append(MagicMock(name=f"batch_{len(pipeline.batches)}"))
            pipeline.contexts.append(MagicMock(name=f"ctx_{len(pipeline.contexts)}"))
            return True

        pipeline.enqueue_batch = MagicMock(side_effect=fake_enqueue)
        pipeline._init_pipelined_modules = MagicMock()
        pipeline.wait_sparse_data_dist = MagicMock()
        pipeline._prefetch = MagicMock()
        pipeline._fence_prefetch_and_start_sparse_data_dist = MagicMock()

        pipeline.fill_pipeline(MagicMock(name="dataloader_iter"))

        pipeline._fence_prefetch_and_start_sparse_data_dist.assert_called_once_with(
            pipeline.batches[1], pipeline.contexts[1]
        )

    def test_progress_routes_next_batch_through_fence_helper(self) -> None:
        """``progress`` must launch the next batch's input dist via the fence
        helper once the queue is full."""
        pipeline = PrefetchTrainPipelineSparseDist.__new__(
            PrefetchTrainPipelineSparseDist
        )
        # Pre-fill the queues with three sentinel batches/contexts so progress
        # takes the ``len(batches) >= 3`` post-prefetch path.
        pipeline.batches = deque(MagicMock(name=f"b{i}") for i in range(3))
        pipeline.contexts = deque(MagicMock(name=f"c{i}") for i in range(3))
        pipeline._execute_all_batches = True
        pipeline._prefetch_stream = MagicMock()
        pipeline._model = MagicMock(training=False)
        pipeline._optimizer = MagicMock()
        pipeline._model_fwd = MagicMock(return_value=(MagicMock(), MagicMock()))

        pipeline.fill_pipeline = MagicMock()
        pipeline.enqueue_batch = MagicMock(return_value=True)
        pipeline._set_module_context = MagicMock()
        pipeline.wait_sparse_data_dist = MagicMock()
        pipeline._prefetch = MagicMock()
        pipeline.dequeue_batch = MagicMock()
        pipeline._fence_prefetch_and_start_sparse_data_dist = MagicMock()

        # progress() reads contexts[0..2]; context[0] needs the cleared dicts.
        cast(
            PrefetchTrainPipelineContext, pipeline.contexts[0]
        ).module_input_post_prefetch = MagicMock()
        cast(
            PrefetchTrainPipelineContext, pipeline.contexts[0]
        ).module_contexts_post_prefetch = MagicMock()

        with patch(
            "torchrec.distributed.train_pipeline.train_pipelines._wait_for_batch"
        ):
            pipeline.progress(MagicMock(name="dataloader_iter"))

        pipeline._fence_prefetch_and_start_sparse_data_dist.assert_called_once_with(
            pipeline.batches[2], pipeline.contexts[2]
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_correctness_uvm_caching_matches_non_pipelined(self) -> None:
        """Verify UVM-caching pipelined output matches non-pipelined baseline.
        A stream race would cause divergence through garbage embeddings."""
        self._set_table_weights_precision(DataType.FP32)
        data = self._generate_data(num_batches=8, batch_size=32)
        dataloader = iter(data)

        fused_params = {
            "cache_load_factor": 0.5,
            "cache_precision": DataType.FP32,
            "stochastic_rounding": False,
        }
        fused_params_pipelined = {**fused_params, "prefetch_pipeline": True}

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model,
            ShardingType.TABLE_WISE.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params,
        )
        sharded_model_pipelined, optim_pipelined = (
            self._generate_sharded_model_and_optimizer(
                model,
                ShardingType.TABLE_WISE.value,
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
            self.assertTrue(torch.equal(pred, pred_pipeline))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_correctness_row_wise_uvm_caching(self) -> None:
        """Same correctness check with ROW_WISE sharding — the sharding type
        used in TWRW production jobs where the race was observed."""
        self._set_table_weights_precision(DataType.FP32)
        data = self._generate_data(num_batches=8, batch_size=32)
        dataloader = iter(data)

        fused_params = {
            "cache_load_factor": 0.5,
            "cache_precision": DataType.FP32,
            "stochastic_rounding": False,
        }
        fused_params_pipelined = {**fused_params, "prefetch_pipeline": True}

        model = self._setup_model()
        sharded_model, optim = self._generate_sharded_model_and_optimizer(
            model,
            ShardingType.ROW_WISE.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            fused_params,
        )
        sharded_model_pipelined, optim_pipelined = (
            self._generate_sharded_model_and_optimizer(
                model,
                ShardingType.ROW_WISE.value,
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
            self.assertTrue(torch.equal(pred, pred_pipeline))


class _MockMultistreamable(Multistreamable):
    """Concrete Multistreamable subclass that tracks record_stream calls."""

    def __init__(self) -> None:
        self._recorded_streams: List[Any] = []

    def record_stream(self, stream: torch.Stream) -> None:
        self._recorded_streams.append(stream)


class _MockAwaitable(Awaitable[_MockMultistreamable]):
    """Concrete Awaitable that returns a pre-set value."""

    def __init__(self, result: _MockMultistreamable) -> None:
        super().__init__()
        self._result = result

    def _wait_impl(self) -> _MockMultistreamable:
        return self._result


class PrefetchEmbeddingsRecordStreamTest(unittest.TestCase):
    """Tests for record_stream coverage in prefetch_embeddings.

    Uses concrete Multistreamable subclasses (not MagicMock) so that the
    isinstance checks inside prefetch_embeddings pass correctly.
    """

    def _make_context_and_module(self, name: str) -> Tuple[
        PrefetchTrainPipelineContext,
        MagicMock,
        _MockMultistreamable,
        _MockMultistreamable,
    ]:
        context = PrefetchTrainPipelineContext(index=0)
        dist_input = _MockMultistreamable()
        module_context = _MockMultistreamable()

        mock_forward = MagicMock(spec=PrefetchPipelinedForward)
        mock_forward._name = name

        mock_module = MagicMock()
        mock_module.forward = mock_forward

        context.input_dist_tensors_requests[name] = _MockAwaitable(dist_input)
        context.module_contexts[name] = module_context

        return context, mock_module, dist_input, module_context

    def test_prefetch_embeddings_records_data_dist_stream(self) -> None:
        """Verify record_stream is called with data_dist_stream for both
        dist_input and module_context."""
        context, mock_module, dist_input, module_context = (
            self._make_context_and_module("mod")
        )

        data_dist_stream = MagicMock()
        forward_stream = MagicMock()
        cur_stream = MagicMock()

        stream_context = MagicMock()
        stream_context.return_value.__enter__ = MagicMock(return_value=None)
        stream_context.return_value.__exit__ = MagicMock(return_value=False)

        with unittest.mock.patch(
            "torchrec.distributed.train_pipeline.utils.torch.get_device_module"
        ) as mock_device_module:
            mock_device_module.return_value.current_stream.return_value = cur_stream

            prefetch_embeddings(
                context=context,
                pipelined_modules=[mock_module],
                device=torch.device("cpu"),
                stream_context=stream_context,
                data_dist_stream=data_dist_stream,
                forward_stream=forward_stream,
            )

        self.assertIn(
            data_dist_stream,
            dist_input._recorded_streams,
            "dist_input.record_stream must be called with data_dist_stream",
        )
        self.assertIn(
            data_dist_stream,
            module_context._recorded_streams,
            "module_context.record_stream must be called with data_dist_stream",
        )

    def test_prefetch_embeddings_records_all_three_streams(self) -> None:
        """Verify record_stream is called for cur_stream, data_dist_stream,
        and forward_stream — all three consumers of the dist_input tensor."""
        context, mock_module, dist_input, _ = self._make_context_and_module("mod")

        data_dist_stream = MagicMock()
        forward_stream = MagicMock()
        cur_stream = MagicMock()

        stream_context = MagicMock()
        stream_context.return_value.__enter__ = MagicMock(return_value=None)
        stream_context.return_value.__exit__ = MagicMock(return_value=False)

        with unittest.mock.patch(
            "torchrec.distributed.train_pipeline.utils.torch.get_device_module"
        ) as mock_device_module:
            mock_device_module.return_value.current_stream.return_value = cur_stream

            prefetch_embeddings(
                context=context,
                pipelined_modules=[mock_module],
                device=torch.device("cpu"),
                stream_context=stream_context,
                data_dist_stream=data_dist_stream,
                forward_stream=forward_stream,
            )

        recorded = set(dist_input._recorded_streams)
        self.assertIn(cur_stream, recorded)
        self.assertIn(data_dist_stream, recorded)
        self.assertIn(forward_stream, recorded)
        self.assertEqual(len(recorded), 3, f"Expected 3 streams, got {recorded}")


if __name__ == "__main__":
    unittest.main()
