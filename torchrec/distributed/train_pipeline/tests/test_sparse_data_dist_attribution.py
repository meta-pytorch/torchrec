#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from contextlib import ExitStack, nullcontext
from typing import cast, Iterator, Literal
from unittest.mock import patch

import torch
from torch import nn
from torchrec.distributed.embedding_sharding import (
    KJTListAwaitable,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionContext
from torchrec.distributed.train_pipeline.batch_attribution import (
    AttributedBatch,
    get_exception_attribution,
)
from torchrec.distributed.train_pipeline.experimental_pipelines import (
    EvalPipelineCPUSparse,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    CPUEmbeddingTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.pipeline_stage import (
    PipelineStage,
    SparseDataDistUtil,
)
from torchrec.distributed.train_pipeline.runtime_forwards import PipelinedForward
from torchrec.distributed.train_pipeline.train_pipelines import (
    StagedTrainPipeline,
    TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.types import ArgInfo, BaseArgInfoStep, CallArgs
from torchrec.distributed.types import Awaitable, LazyAwaitable, ShardedModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


class _Batch(Pipelineable):
    def __init__(self) -> None:
        self.features = object()

    def to(self, device: torch.device, non_blocking: bool) -> _Batch:
        raise AssertionError("CPU distribution must not copy the batch")

    def record_stream(self, stream: torch.Stream) -> None:
        raise AssertionError("CPU distribution must not record streams")


class _FakeStream:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.waited_on: list[_FakeStream] = []

    def wait_stream(self, stream: _FakeStream) -> None:
        self.waited_on.append(stream)


class _FeatureArgStep(BaseArgInfoStep):
    def process(self, arg: object) -> object:
        assert isinstance(arg, _Batch)
        return arg.features


class _FailingSplitsAwaitable(Awaitable[Awaitable[KeyedJaggedTensor]]):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def _wait_impl(self) -> Awaitable[KeyedJaggedTensor]:
        raise self.error


class _ShardedModule(
    ShardedModule[KJTList, KJTList, KJTList, EmbeddingBagCollectionContext]
):
    def __init__(self) -> None:
        super().__init__()
        self.start_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.received_feature: object | None = None
        self.request: KJTListSplitsAwaitable[EmbeddingBagCollectionContext] | None = (
            None
        )

    def create_context(self) -> EmbeddingBagCollectionContext:
        return EmbeddingBagCollectionContext()

    def input_dist(
        self, ctx: EmbeddingBagCollectionContext, *input: object, **kwargs: object
    ) -> KJTListSplitsAwaitable[EmbeddingBagCollectionContext]:
        self.received_feature = input[0]
        if self.start_error is not None:
            raise self.start_error
        awaitables: list[Awaitable[Awaitable[KeyedJaggedTensor]]] = []
        if self.wait_error is not None:
            awaitables.append(_FailingSplitsAwaitable(self.wait_error))
        self.request = KJTListSplitsAwaitable(awaitables, ctx)
        return self.request

    def compute(
        self, ctx: EmbeddingBagCollectionContext, dist_input: KJTList
    ) -> KJTList:
        raise AssertionError("Only input distribution is under test")

    def output_dist(
        self, ctx: EmbeddingBagCollectionContext, output: KJTList
    ) -> LazyAwaitable[KJTList]:
        raise AssertionError("Only input distribution is under test")

    @property
    def unsharded_module_type(self) -> type[nn.Module]:
        return nn.Module


class SparseDataDistAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)

    def _make_pipeline(
        self, context: TrainPipelineContext
    ) -> tuple[TrainPipelineSparseDist[Pipelineable, torch.Tensor], _ShardedModule]:
        device = torch.device("cpu")
        model = nn.Linear(1, 1, device=device)
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineSparseDist[Pipelineable, torch.Tensor](
            model,
            optimizer,
            device,
        )
        module = _ShardedModule()
        self._patches.enter_context(
            patch.object(
                module,
                "forward",
                PipelinedForward(
                    "sparse",
                    CallArgs([ArgInfo([_FeatureArgStep()])], {}),
                    module,
                    context,
                ),
            )
        )
        pipeline._pipelined_modules = [module]
        return pipeline, module

    def _make_staged_sdd(
        self,
        context: TrainPipelineContext,
        *,
        with_prefetch: bool = False,
    ) -> tuple[SparseDataDistUtil[_Batch], _ShardedModule]:
        device = torch.device("cpu")
        model = nn.Linear(1, 1, device=device)
        module = _ShardedModule()
        self._patches.enter_context(
            patch.object(
                module,
                "forward",
                PipelinedForward(
                    "sparse",
                    CallArgs([ArgInfo([_FeatureArgStep()])], {}),
                    module,
                    context,
                ),
            )
        )
        data_dist_stream = _FakeStream(device)
        sdd = SparseDataDistUtil[_Batch](
            model=model,
            data_dist_stream=cast(torch.Stream, data_dist_stream),
            prefetch_stream=(
                cast(torch.Stream, data_dist_stream) if with_prefetch else None
            ),
        )
        self._patches.enter_context(
            patch.object(
                sdd,
                "_stream_context",
                side_effect=lambda _stream: nullcontext(),
            )
        )
        self._patches.enter_context(
            patch(
                "torchrec.distributed.train_pipeline.pipeline_stage._rewrite_model",
                return_value=([module], model, [], [], None),
            )
        )
        return sdd, module

    def _make_cpu_sparse_pipeline(
        self,
        *,
        multi_thread: bool,
    ) -> tuple[EvalPipelineCPUSparse[_Batch, torch.Tensor], _ShardedModule]:
        device = torch.device("cpu")
        model = nn.Linear(1, 1, device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = EvalPipelineCPUSparse[_Batch, torch.Tensor](
            model,
            optimizer,
            device,
            multi_thread=multi_thread,
        )
        if pipeline._sparse_executor is not None:
            self.addCleanup(pipeline._sparse_executor.shutdown)

        module = _ShardedModule()
        self._patches.enter_context(
            patch.object(
                module,
                "forward",
                PipelinedForward(
                    "sparse",
                    CallArgs([ArgInfo([_FeatureArgStep()])], {}),
                    module,
                    CPUEmbeddingTrainPipelineContext(index=0, version=1),
                ),
            )
        )
        pipeline._pipelined_modules = [module]
        return pipeline, module

    def _assert_attribution(
        self,
        error: Exception,
        operation: str,
        metadata: object,
        batch_index: int,
    ) -> None:
        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual(operation, attribution.operation)
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(batch_index, attribution.batch_index)

    def test_start_and_wait_preserve_raw_features_and_requests(self) -> None:
        context = TrainPipelineContext(
            index=73,
            version=1,
            batch_metadata=object(),
        )
        pipeline, module = self._make_pipeline(context)
        batch = _Batch()

        self.assertIsNone(pipeline.start_sparse_data_dist(batch=batch, context=context))

        self.assertIs(batch.features, module.received_feature)
        self.assertIs(module.request, context.input_dist_splits_requests["sparse"])
        self.assertIsInstance(module.request, KJTListSplitsAwaitable)

        self.assertIsNone(pipeline.wait_sparse_data_dist(context=context))

        self.assertIsInstance(
            context.input_dist_tensors_requests["sparse"], KJTListAwaitable
        )
        self.assertEqual({}, context.input_dist_splits_requests)
        self.assertEqual([], context.fused_splits_awaitables)

    def test_start_failure_uses_context_metadata(self) -> None:
        for metadata in (None, object()):
            with self.subTest(attributed=metadata is not None):
                context = TrainPipelineContext(
                    index=73,
                    version=1,
                    batch_metadata=metadata,
                )
                pipeline, module = self._make_pipeline(context)
                error = ValueError("input dist launch failed")
                module.start_error = error
                batch = _Batch()

                with self.assertRaises(ValueError) as raised:
                    pipeline.start_sparse_data_dist(batch=batch, context=context)

                self.assertIs(error, raised.exception)
                self.assertIs(batch.features, module.received_feature)
                if metadata is None:
                    self.assertIsNone(get_exception_attribution(error))
                else:
                    self._assert_attribution(error, "input_dist_start", metadata, 73)

    def test_wait_failure_uses_supplied_context_metadata(self) -> None:
        for metadata in (None, object()):
            with self.subTest(attributed=metadata is not None):
                context = TrainPipelineContext(
                    index=73,
                    version=1,
                    batch_metadata=metadata,
                )
                pipeline, module = self._make_pipeline(context)
                error = RuntimeError("input dist wait failed")
                module.wait_error = error
                pipeline.contexts.append(
                    TrainPipelineContext(
                        index=72,
                        version=1,
                        batch_metadata=object(),
                    )
                )
                self.assertIsNone(
                    pipeline.start_sparse_data_dist(
                        batch=_Batch(),
                        context=context,
                    )
                )

                with self.assertRaises(RuntimeError) as raised:
                    pipeline.wait_sparse_data_dist(context=context)

                self.assertIs(error, raised.exception)
                if metadata is None:
                    self.assertIsNone(get_exception_attribution(error))
                else:
                    self._assert_attribution(error, "input_dist_wait", metadata, 73)

    def test_staged_start_failure_uses_outer_context(self) -> None:
        metadata = object()
        context = TrainPipelineContext(
            index=73,
            version=1,
            batch_metadata=metadata,
        )
        sdd, module = self._make_staged_sdd(TrainPipelineContext(index=0, version=1))
        error = ValueError("input dist launch failed")
        module.start_error = error

        with self.assertRaises(ValueError) as raised:
            sdd.start_sparse_data_dist(_Batch(), context=context)

        self.assertIs(error, raised.exception)
        self._assert_attribution(error, "input_dist_start", metadata, 73)

    def test_staged_pipeline_passes_batch_context_to_sparse_data_dist(self) -> None:
        metadata = object()
        sdd, module = self._make_staged_sdd(TrainPipelineContext(index=0, version=1))
        error = ValueError("input dist launch failed")
        module.start_error = error
        stream = _FakeStream(torch.device("cpu"))
        pipeline = StagedTrainPipeline[_Batch, _Batch](
            pipeline_stages=[
                PipelineStage(
                    name="start_sparse_data_dist",
                    runnable=sdd.start_sparse_data_dist,
                    stream=cast(torch.Stream, stream),
                    pass_batch_context=True,
                )
            ],
            compute_stream=cast(torch.Stream, stream),
        )
        pipeline._stream_context = lambda _stream: nullcontext()

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(iter([cast(_Batch, AttributedBatch(_Batch(), metadata))]))

        self.assertIs(error, raised.exception)
        self._assert_attribution(error, "input_dist_start", metadata, 0)

    def test_staged_start_failure_uses_latest_outer_context(self) -> None:
        first_context = TrainPipelineContext(
            index=73,
            version=1,
            batch_metadata=object(),
        )
        sdd, module = self._make_staged_sdd(first_context)
        sdd.start_sparse_data_dist(_Batch(), context=first_context)
        sdd.wait_sparse_data_dist()

        metadata = object()
        context = TrainPipelineContext(
            index=74,
            version=1,
            batch_metadata=metadata,
        )
        error = ValueError("later input dist launch failed")
        module.start_error = error

        with self.assertRaises(ValueError) as raised:
            sdd.start_sparse_data_dist(_Batch(), context=context)

        self.assertIs(error, raised.exception)
        self._assert_attribution(error, "input_dist_start", metadata, 74)

    def test_staged_wait_failure_uses_outer_context(self) -> None:
        for with_prefetch in (False, True):
            with self.subTest(with_prefetch=with_prefetch):
                metadata = object()
                context = TrainPipelineContext(
                    index=73,
                    version=1,
                    batch_metadata=metadata,
                )
                sdd, module = self._make_staged_sdd(
                    context,
                    with_prefetch=with_prefetch,
                )
                error = RuntimeError("input dist wait failed")
                module.wait_error = error
                sdd.start_sparse_data_dist(_Batch(), context=context)

                with self.assertRaises(RuntimeError) as raised:
                    sdd.wait_sparse_data_dist()

                self.assertIs(error, raised.exception)
                self._assert_attribution(error, "input_dist_wait", metadata, 73)

    def _prepare_cpu_sparse_failure(
        self,
        *,
        multi_thread: bool,
        fail_on: Literal["start", "wait"],
    ) -> tuple[
        EvalPipelineCPUSparse[_Batch, torch.Tensor],
        Iterator[_Batch],
        RuntimeError,
        object,
    ]:
        pipeline, module = self._make_cpu_sparse_pipeline(multi_thread=multi_thread)
        self.assertTrue(pipeline.enqueue_batch(iter([_Batch()])))

        metadata = object()
        error = RuntimeError(f"CPU sparse input dist {fail_on} failed")
        if fail_on == "start":
            module.start_error = error
        else:
            module.wait_error = error
        dataloader = iter([cast(_Batch, AttributedBatch(_Batch(), metadata))])
        return pipeline, dataloader, error, metadata

    def test_cpu_sparse_progress_failure_uses_context(self) -> None:
        for fail_on, operation in (
            ("start", "input_dist_start"),
            ("wait", "input_dist_wait"),
        ):
            with self.subTest(fail_on=fail_on):
                pipeline, dataloader, error, metadata = (
                    self._prepare_cpu_sparse_failure(
                        multi_thread=False,
                        fail_on=fail_on,
                    )
                )

                with (
                    patch.object(pipeline, "fill_pipeline"),
                    patch.object(
                        pipeline,
                        "_dense_forward",
                        return_value=torch.zeros(1),
                    ),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    pipeline.progress(dataloader)

                self.assertEqual(0, pipeline.get_current_batch_index())
                self.assertIs(error, raised.exception)
                self._assert_attribution(error, operation, metadata, 1)

    def test_cpu_sparse_future_failure_uses_context(self) -> None:
        for fail_on, operation in (
            ("start", "input_dist_start"),
            ("wait", "input_dist_wait"),
        ):
            with self.subTest(fail_on=fail_on):
                pipeline, dataloader, error, metadata = (
                    self._prepare_cpu_sparse_failure(
                        multi_thread=True,
                        fail_on=fail_on,
                    )
                )

                with (
                    patch.object(pipeline, "fill_pipeline"),
                    patch.object(
                        pipeline,
                        "_dense_forward",
                        return_value=torch.zeros(1),
                    ),
                ):
                    pipeline.progress(dataloader)
                    with self.assertRaises(RuntimeError) as raised:
                        pipeline.progress(iter(()))

                self.assertIs(error, raised.exception)
                self._assert_attribution(error, operation, metadata, 1)
                self.assertEqual(1, pipeline.get_current_batch_index())

    def test_staged_without_context_clears_attribution(self) -> None:
        for fail_on in ("start", "wait"):
            with self.subTest(fail_on=fail_on):
                context = TrainPipelineContext(
                    index=73,
                    version=1,
                    batch_metadata=object(),
                )
                sdd, module = self._make_staged_sdd(context)
                sdd.start_sparse_data_dist(_Batch(), context=context)
                sdd.wait_sparse_data_dist()
                error = RuntimeError(f"input dist {fail_on} failed")

                if fail_on == "start":
                    module.start_error = error
                    with self.assertRaises(RuntimeError) as raised:
                        sdd.start_sparse_data_dist(_Batch(), context=None)
                else:
                    module.wait_error = error
                    sdd.start_sparse_data_dist(_Batch(), context=None)
                    with self.assertRaises(RuntimeError) as raised:
                        sdd.wait_sparse_data_dist()

                self.assertIs(error, raised.exception)
                self.assertIsNone(get_exception_attribution(error))
