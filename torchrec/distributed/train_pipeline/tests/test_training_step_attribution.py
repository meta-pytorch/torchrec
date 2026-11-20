#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from itertools import product
from typing import cast, Protocol
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from torchrec.distributed.train_pipeline.batch_attribution import (
    AttributedBatch,
    get_exception_attribution,
)
from torchrec.distributed.train_pipeline.experimental_pipelines import (
    PrefetchTrainPipelineFusedSparseDist,
    TrainEvalHybridPipelineBase,
    TrainPipelinePrefetchEMS,
    TrainPipelineSparseDistBwdOpt,
    TrainPipelineSparseDistOptStash,
)
from torchrec.distributed.train_pipeline.pec_train_pipeline import TrainPipelinePEC
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.train_pipelines import (
    EvalPipelineSparseDist,
    PrefetchTrainPipelineSparseDist,
    TorchCompileConfig,
    TrainPipeline,
    TrainPipelineBase,
    TrainPipelineFusedSparseDist,
    TrainPipelinePT2,
    TrainPipelineSparseDist,
    TrainPipelineSparseDistCompAutograd,
    TrainPipelineSparseDistLite,
)
from torchrec.streamable import Pipelineable


class _Batch(Pipelineable):
    def to(self, device: torch.device, non_blocking: bool) -> _Batch:
        raise AssertionError("Warm CPU training must not copy the batch")

    def record_stream(self, stream: torch.Stream) -> None:
        raise AssertionError("CPU training must not record streams")


class _CopyableBatch(_Batch):
    def to(self, device: torch.device, non_blocking: bool) -> _CopyableBatch:
        return self


class _PECBatch(_CopyableBatch):
    def record_stream(self, stream: torch.Stream) -> None:
        pass


class _FakeStream:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.waited_on: list[_FakeStream] = []

    def wait_stream(self, stream: _FakeStream) -> None:
        self.waited_on.append(stream)


class _FakeDeviceModule:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream
        self.entered_streams: list[_FakeStream] = []

    def Stream(self, priority: int = 0) -> _FakeStream:
        return self._stream

    def current_stream(self) -> _FakeStream:
        return self._stream

    def stream(self, stream: _FakeStream) -> AbstractContextManager[None]:
        self.entered_streams.append(stream)
        return nullcontext()


class _ProgressPipeline(Protocol):
    def progress(self, dataloader_iter: Iterator[Pipelineable]) -> object: ...


class _TrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, device="cpu"))
        self.output = torch.tensor([7.0], device="cpu")
        self.custom_output = torch.tensor([9.0], device="cpu")
        self.received_batches: list[_Batch] = []
        self.custom_batches: list[Pipelineable | None] = []
        self.error: Exception | None = None

    def forward(self, batch: Pipelineable | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(batch, _Batch)
        self.received_batches.append(batch)
        if self.error is not None:
            raise self.error
        return self.weight * 2, self.output

    def custom_forward(
        self, batch: Pipelineable | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.custom_batches.append(batch)
        loss, _ = self(batch)
        return loss * 2, self.custom_output


class _ObservedContext:
    def __init__(self) -> None:
        self.active = False
        self.observations: list[bool] = []

    @contextmanager
    def enter(self) -> Iterator[None]:
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def record(self) -> None:
        self.observations.append(self.active)


class TrainingStepAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)

    def _register_forward_hooks(
        self, model: _TrainingModel
    ) -> tuple[list[tuple[object, ...]], list[object]]:
        hook_inputs: list[tuple[object, ...]] = []
        hook_outputs: list[object] = []
        self.addCleanup(
            model.register_forward_pre_hook(
                lambda module, args: hook_inputs.append(args)
            ).remove
        )
        self.addCleanup(
            model.register_forward_hook(
                lambda module, args, output: hook_outputs.append(output)
            ).remove
        )
        return hook_inputs, hook_outputs

    @staticmethod
    def _wrap_batch(batch: _CopyableBatch, metadata: object | None) -> Pipelineable:
        if metadata is None:
            return batch
        return cast(Pipelineable, AttributedBatch(batch, metadata))

    def _make_sparse_pipeline(
        self,
        model: _TrainingModel,
        custom_model_fwd: (
            Callable[[Pipelineable | None], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        optimizer: torch.optim.Optimizer | None = None,
        pipeline_type: type[
            TrainPipelineSparseDist[Pipelineable, torch.Tensor]
        ] = TrainPipelineSparseDist,
    ) -> TrainPipelineSparseDist[Pipelineable, torch.Tensor]:
        if optimizer is None:
            # Citrine C2: foreach=True for multi-tensor optimizer execution.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = pipeline_type(
            model,
            optimizer,
            torch.device("cpu"),
            custom_model_fwd=custom_model_fwd,
        )
        self._patches.enter_context(
            patch.object(
                pipeline,
                "_init_pipelined_modules",
            )
        )
        return pipeline

    def _make_pipeline(
        self,
        pipeline_type: (
            type[TrainPipelineBase[Pipelineable, torch.Tensor]]
            | type[TrainPipelinePT2[Pipelineable, torch.Tensor]]
        ),
        model: _TrainingModel,
    ) -> (
        TrainPipelineBase[Pipelineable, torch.Tensor]
        | TrainPipelinePT2[Pipelineable, torch.Tensor]
    ):
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        if pipeline_type is TrainPipelinePT2:
            return TrainPipelinePT2(
                model,
                optimizer,
                torch.device("cpu"),
                compile_configs=TorchCompileConfig(compile_on_iter=100),
            )
        return TrainPipelineBase(model, optimizer, torch.device("cpu"))

    def _make_comp_autograd_pipeline(
        self,
        model: _TrainingModel,
    ) -> TrainPipelineSparseDistCompAutograd[Pipelineable, torch.Tensor]:
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineSparseDistCompAutograd[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
        )
        self._patches.enter_context(patch.object(pipeline, "_init_pipelined_modules"))
        return pipeline

    def _make_prefetch_pipeline(
        self,
        model: _TrainingModel,
    ) -> PrefetchTrainPipelineSparseDist[Pipelineable, torch.Tensor]:
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = PrefetchTrainPipelineSparseDist[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
        )
        self._patches.enter_context(patch.object(pipeline, "_init_pipelined_modules"))
        self._patches.enter_context(patch.object(pipeline, "_prefetch"))
        return pipeline

    def _make_bwd_opt_pipeline(
        self,
        model: _TrainingModel,
        optimizer: torch.optim.Optimizer,
        stack: ExitStack,
    ) -> TrainPipelineSparseDistBwdOpt[Pipelineable, torch.Tensor]:
        pipeline = TrainPipelineSparseDistBwdOpt[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
            site_fqn="sparse",
        )
        stack.enter_context(patch.object(pipeline, "_init_pipelined_modules"))

        registered_hook: Callable[[torch.Tensor], None] | None = None

        def capture_backward_hook(
            site: object,
            hooked_model: nn.Module,
            hook: Callable[[torch.Tensor], None],
        ) -> object:
            nonlocal registered_hook
            self.assertIs(pipeline._output_dist_site, site)
            self.assertIs(model, hooked_model)
            registered_hook = hook
            return object()

        stack.enter_context(patch.object(TrainPipelineSparseDist, "_pipeline_model"))
        low_level_hook = stack.enter_context(
            patch(
                "torchrec.distributed.train_pipeline.train_pipelines."
                "register_backward_hook",
                side_effect=capture_backward_hook,
            )
        )
        pipeline._pipeline_model(
            None,
            TrainPipelineContext(index=0, version=1),
        )
        low_level_hook.assert_called_once()
        assert registered_hook is not None
        hook = registered_hook

        def optimizer_step(gradient: torch.Tensor) -> torch.Tensor:
            hook(gradient)
            return gradient

        weight_hook = model.weight.register_hook(optimizer_step)
        stack.callback(weight_hook.remove)
        return pipeline

    def _make_memory_stashing_pipeline(
        self,
        model: _TrainingModel,
        optimizer: torch.optim.Optimizer,
        stack: ExitStack,
        pipeline_type: (
            type[TrainPipelineSparseDistOptStash[Pipelineable, torch.Tensor]]
            | type[TrainPipelinePrefetchEMS[Pipelineable, torch.Tensor]]
        ),
    ) -> (
        TrainPipelineSparseDistOptStash[Pipelineable, torch.Tensor]
        | TrainPipelinePrefetchEMS[Pipelineable, torch.Tensor]
    ):
        with (
            patch(
                "torchrec.distributed.train_pipeline.experimental_pipelines."
                "MemoryStashingManager.set_streams"
            ),
            patch(
                "torchrec.distributed.train_pipeline.experimental_pipelines."
                "torch.cuda.Stream",
                return_value=MagicMock(),
            ),
            patch(
                "torchrec.distributed.train_pipeline.experimental_pipelines."
                "log_memory_stashing_config"
            ),
        ):
            pipeline = pipeline_type(
                model,
                optimizer,
                torch.device("cpu"),
                site_fqn="sparse",
            )
        stack.enter_context(patch.object(pipeline, "_init_pipelined_modules"))
        return pipeline

    def _make_prefetch_fused_sparse_pipeline(
        self,
        model: _TrainingModel,
        optimizer: torch.optim.Optimizer,
        stack: ExitStack,
    ) -> PrefetchTrainPipelineFusedSparseDist[Pipelineable, torch.Tensor]:
        stream = _FakeStream(torch.device("cuda"))
        device_module = _FakeDeviceModule(stream)
        stack.enter_context(
            patch("torch.get_device_module", return_value=device_module)
        )

        pipeline = PrefetchTrainPipelineFusedSparseDist[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cuda"),
        )
        stack.enter_context(patch.object(pipeline, "_init_pipelined_modules"))
        return pipeline

    def _assert_attribution(
        self,
        error: Exception,
        metadata: object | None,
        batch_index: int = 0,
        *,
        operation: str = "forward",
    ) -> None:
        attribution = get_exception_attribution(error)
        if metadata is None:
            self.assertIsNone(attribution)
            return
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual(operation, attribution.operation)
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(batch_index, attribution.batch_index)

    def _assert_forward_preserved(
        self,
        model: _TrainingModel,
        raw_batch: _CopyableBatch,
        hook_inputs: list[tuple[object, ...]],
        hook_outputs: list[object],
        output: torch.Tensor,
        expected_output: torch.Tensor,
        expected_weight: float,
    ) -> None:
        self.assertIs(expected_output, output)
        self.assertEqual([raw_batch], model.received_batches)
        self.assertEqual([(raw_batch,)], hook_inputs)
        self.assertEqual(1, len(hook_outputs))
        hook_output = hook_outputs[0]
        assert isinstance(hook_output, tuple)
        torch.testing.assert_close(hook_output[0], torch.tensor([2.0]))
        self.assertIs(model.output, hook_output[1])
        torch.testing.assert_close(
            model.weight.detach(),
            torch.tensor([expected_weight]),
        )

    def _assert_backward_failure(
        self,
        pipeline: TrainPipeline[Pipelineable, torch.Tensor],
        model: _TrainingModel,
        dataloader_iter: Iterator[Pipelineable],
        metadata: object | None,
        *,
        batch_index: int = 0,
        on_backward: Callable[[], None] | None = None,
    ) -> RemovableHandle:
        error = ValueError("backward failed")

        def fail(_gradient: torch.Tensor) -> torch.Tensor:
            if on_backward is not None:
                on_backward()
            raise error

        hook = model.weight.register_hook(fail)
        self.addCleanup(hook.remove)
        weight_before = model.weight.detach().clone()

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(dataloader_iter)

        self.assertIs(error, raised.exception)
        torch.testing.assert_close(model.weight.detach(), weight_before)
        self._assert_attribution(
            error,
            metadata,
            operation="backward",
            batch_index=batch_index,
        )
        return hook

    def _assert_optimizer_failure(
        self,
        pipeline: _ProgressPipeline,
        optimizer: torch.optim.Optimizer,
        dataloader_iter: Iterator[Pipelineable],
        metadata: object | None,
        *,
        batch_index: int = 0,
    ) -> None:
        error = ValueError("optimizer failed")

        def fail(*_args: object) -> None:
            raise error

        self.addCleanup(optimizer.register_step_pre_hook(fail).remove)

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(dataloader_iter)

        self.assertIs(error, raised.exception)
        self._assert_attribution(
            error,
            metadata,
            operation="optimizer",
            batch_index=batch_index,
        )

    def test_sparse_progress_preserves_hooks_custom_forward_and_runtime_types(
        self,
    ) -> None:
        for custom, metadata in product((False, True), (None, object())):
            with self.subTest(
                custom=custom,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                raw_batch = _CopyableBatch()
                hook_inputs, hook_outputs = self._register_forward_hooks(model)
                pipeline = self._make_sparse_pipeline(
                    model,
                    model.custom_forward if custom else None,
                )
                batch = self._wrap_batch(raw_batch, metadata)

                output = pipeline.progress(iter((batch, _CopyableBatch())))

                self.assertEqual([raw_batch] if custom else [], model.custom_batches)
                self._assert_forward_preserved(
                    model,
                    raw_batch,
                    hook_inputs,
                    hook_outputs,
                    output,
                    model.custom_output if custom else model.output,
                    0.6 if custom else 0.8,
                )

    def test_sparse_forward_failure_uses_current_context(self) -> None:
        for custom, metadata in product((False, True), (None, object())):
            with self.subTest(
                custom=custom,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                first_batch = _CopyableBatch()
                raw_batch = _CopyableBatch()
                error = ValueError("forward failed")
                pipeline = self._make_sparse_pipeline(
                    model,
                    model.custom_forward if custom else None,
                )
                data_iter = iter(
                    (
                        self._wrap_batch(first_batch, object()),
                        self._wrap_batch(raw_batch, metadata),
                        self._wrap_batch(_CopyableBatch(), object()),
                        self._wrap_batch(_CopyableBatch(), object()),
                    )
                )

                pipeline.progress(data_iter)
                model.error = error
                with self.assertRaises(ValueError) as raised:
                    pipeline.progress(data_iter)

                self.assertIs(error, raised.exception)
                self.assertEqual([first_batch, raw_batch], model.received_batches)
                self.assertEqual(
                    [first_batch, raw_batch] if custom else [], model.custom_batches
                )
                self._assert_attribution(error, metadata, 1)

    def test_eval_sparse_forward_failure_uses_current_context(self) -> None:
        for metadata in (None, object()):
            with self.subTest(attributed=metadata is not None):
                model = _TrainingModel()
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=0.1,
                    foreach=True,
                )
                pipeline = EvalPipelineSparseDist[Pipelineable, torch.Tensor](
                    model,
                    optimizer,
                    torch.device("cpu"),
                )
                self.addCleanup(pipeline.reset)
                self._patches.enter_context(
                    patch.object(pipeline, "_init_pipelined_modules")
                )
                raw_batch = _CopyableBatch()
                error = ValueError("forward failed")
                model.error = error

                with self.assertRaises(ValueError) as raised:
                    pipeline.progress(iter((self._wrap_batch(raw_batch, metadata),)))

                self.assertIs(error, raised.exception)
                self.assertEqual([raw_batch], model.received_batches)
                self._assert_attribution(error, metadata, 0)

    def test_base_and_pt2_success_preserves_hooks_and_runtime_types(self) -> None:
        for pipeline_type, metadata in product(
            (TrainPipelineBase, TrainPipelinePT2),
            (None, object()),
        ):
            with self.subTest(
                pipeline_type=pipeline_type.__name__,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                pipeline = self._make_pipeline(pipeline_type, model)
                raw_batch = _CopyableBatch()
                batch = self._wrap_batch(raw_batch, metadata)
                hook_inputs, hook_outputs = self._register_forward_hooks(model)

                output = pipeline.progress(iter((batch,)))

                self._assert_forward_preserved(
                    model,
                    raw_batch,
                    hook_inputs,
                    hook_outputs,
                    output,
                    model.output,
                    0.8,
                )

    def test_sparse_pipeline_backward_failure_uses_current_context(self) -> None:
        pipeline_factories = (
            ("sparse", self._make_sparse_pipeline),
            ("compiled_autograd", self._make_comp_autograd_pipeline),
            ("prefetch", self._make_prefetch_pipeline),
        )
        for (pipeline_name, make_pipeline), metadata in product(
            pipeline_factories,
            (None, object()),
        ):
            with self.subTest(
                pipeline=pipeline_name,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                raw_batch = _CopyableBatch()
                pipeline = make_pipeline(model)
                compiled_autograd_context = _ObservedContext()
                on_backward: Callable[[], None] | None = None

                if pipeline_name == "compiled_autograd":
                    self._patches.enter_context(
                        patch.object(
                            pipeline,
                            "get_compiled_autograd_ctx",
                            side_effect=compiled_autograd_context.enter,
                        )
                    )
                    on_backward = compiled_autograd_context.record
                batch = self._wrap_batch(raw_batch, metadata)
                lookahead_batch = self._wrap_batch(_CopyableBatch(), object())

                self._assert_backward_failure(
                    pipeline,
                    model,
                    iter((batch, lookahead_batch)),
                    metadata,
                    on_backward=on_backward,
                )

                self.assertEqual([raw_batch], model.received_batches)
                if pipeline_name == "compiled_autograd":
                    self.assertEqual([True], compiled_autograd_context.observations)

    def test_base_and_pt2_backward_failure_uses_current_context(self) -> None:
        for pipeline_type, metadata in product(
            (TrainPipelineBase, TrainPipelinePT2),
            (None, object()),
        ):
            with self.subTest(
                pipeline_type=pipeline_type.__name__,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                pipeline = self._make_pipeline(pipeline_type, model)
                raw_batch = _CopyableBatch()
                batch = self._wrap_batch(raw_batch, metadata)
                next_batch = self._wrap_batch(_CopyableBatch(), object())

                self._assert_backward_failure(
                    pipeline,
                    model,
                    iter((batch, next_batch)),
                    metadata,
                )

                self.assertEqual([raw_batch], model.received_batches)

    def test_base_second_backward_failure_uses_second_batch_context(self) -> None:
        for first_step_fails in (True, False):
            with self.subTest(first_step_fails=first_step_fails):
                model = _TrainingModel()
                pipeline = self._make_pipeline(TrainPipelineBase, model)
                first_batch = _CopyableBatch()
                second_batch = _CopyableBatch()
                first_metadata = object()
                second_metadata = object()
                dataloader_iter = iter(
                    (
                        self._wrap_batch(first_batch, first_metadata),
                        self._wrap_batch(second_batch, second_metadata),
                        self._wrap_batch(_CopyableBatch(), object()),
                    )
                )

                if first_step_fails:
                    backward_hook = self._assert_backward_failure(
                        pipeline,
                        model,
                        dataloader_iter,
                        first_metadata,
                    )
                    backward_hook.remove()
                else:
                    self.assertIs(model.output, pipeline.progress(dataloader_iter))

                self._assert_backward_failure(
                    pipeline,
                    model,
                    dataloader_iter,
                    second_metadata,
                    batch_index=1,
                )
                self.assertEqual(
                    [first_batch, second_batch],
                    model.received_batches,
                )

    def test_base_pt2_and_sparse_optimizer_failure_uses_new_context_after_success(
        self,
    ) -> None:
        for pipeline_type, second_metadata in product(
            (TrainPipelineBase, TrainPipelinePT2, TrainPipelineSparseDist),
            (None, object()),
        ):
            with self.subTest(
                pipeline_type=pipeline_type.__name__,
                attributed=second_metadata is not None,
            ):
                model = _TrainingModel()
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
                pipeline = (
                    self._make_sparse_pipeline(model, optimizer=optimizer)
                    if pipeline_type is TrainPipelineSparseDist
                    else pipeline_type(
                        model,
                        optimizer,
                        torch.device("cpu"),
                    )
                )
                first_batch = _CopyableBatch()
                second_batch = _CopyableBatch()
                second_input = self._wrap_batch(second_batch, second_metadata)
                batches: list[Pipelineable] = [
                    cast(
                        Pipelineable,
                        AttributedBatch(first_batch, object()),
                    ),
                    second_input,
                ]
                if pipeline_type is TrainPipelineSparseDist:
                    batches.append(
                        cast(
                            Pipelineable,
                            AttributedBatch(_CopyableBatch(), object()),
                        )
                    )
                dataloader_iter = iter(batches)

                self.assertIs(model.output, pipeline.progress(dataloader_iter))
                self._assert_optimizer_failure(
                    pipeline,
                    optimizer,
                    dataloader_iter,
                    second_metadata,
                    batch_index=1,
                )

                self.assertEqual(
                    [first_batch, second_batch],
                    model.received_batches,
                )

    def test_optimizer_failure_uses_current_context(self) -> None:
        def make_compiled_autograd(
            model: _TrainingModel,
            optimizer: torch.optim.Optimizer,
            stack: ExitStack,
        ) -> tuple[_ProgressPipeline, type[_CopyableBatch]]:
            pipeline = self._make_sparse_pipeline(
                model,
                optimizer=optimizer,
                pipeline_type=TrainPipelineSparseDistCompAutograd,
            )
            stack.enter_context(
                patch.object(
                    pipeline,
                    "get_compiled_autograd_ctx",
                    side_effect=nullcontext,
                )
            )
            return pipeline, _CopyableBatch

        def make_prefetch_ems(
            model: _TrainingModel,
            optimizer: torch.optim.Optimizer,
            stack: ExitStack,
        ) -> tuple[_ProgressPipeline, type[_CopyableBatch]]:
            pipeline = self._make_memory_stashing_pipeline(
                model,
                optimizer,
                stack,
                TrainPipelinePrefetchEMS,
            )
            stack.enter_context(patch.object(pipeline, "_restore_and_prefetch"))
            return pipeline, _CopyableBatch

        pipeline_factories: dict[
            str,
            Callable[
                [_TrainingModel, torch.optim.Optimizer, ExitStack],
                tuple[_ProgressPipeline, type[_CopyableBatch]],
            ],
        ] = {
            "base": lambda model, optimizer, _stack: (
                TrainPipelineBase(model, optimizer, torch.device("cpu")),
                _CopyableBatch,
            ),
            "pt2": lambda model, optimizer, _stack: (
                TrainPipelinePT2(model, optimizer, torch.device("cpu")),
                _CopyableBatch,
            ),
            "sparse": lambda model, optimizer, _stack: (
                self._make_sparse_pipeline(model, optimizer=optimizer),
                _CopyableBatch,
            ),
            "sparse_lite": lambda model, optimizer, _stack: (
                self._make_sparse_pipeline(
                    model,
                    optimizer=optimizer,
                    pipeline_type=TrainPipelineSparseDistLite,
                ),
                _CopyableBatch,
            ),
            "fused_sparse": lambda model, optimizer, _stack: (
                self._make_sparse_pipeline(
                    model,
                    optimizer=optimizer,
                    pipeline_type=TrainPipelineFusedSparseDist,
                ),
                _CopyableBatch,
            ),
            "prefetch_sparse": lambda model, optimizer, _stack: (
                self._make_sparse_pipeline(
                    model,
                    optimizer=optimizer,
                    pipeline_type=PrefetchTrainPipelineSparseDist,
                ),
                _CopyableBatch,
            ),
            "bwd_opt": lambda model, optimizer, stack: (
                self._make_bwd_opt_pipeline(model, optimizer, stack),
                _CopyableBatch,
            ),
            "compiled_autograd": make_compiled_autograd,
            "train_eval_hybrid": lambda model, optimizer, _stack: (
                self._make_sparse_pipeline(
                    model,
                    optimizer=optimizer,
                    pipeline_type=TrainEvalHybridPipelineBase,
                ),
                _CopyableBatch,
            ),
            "prefetch_ems": make_prefetch_ems,
            "prefetch_fused_sparse": lambda model, optimizer, stack: (
                self._make_prefetch_fused_sparse_pipeline(model, optimizer, stack),
                _PECBatch,
            ),
            "optimizer_state_stashing": lambda model, optimizer, stack: (
                self._make_memory_stashing_pipeline(
                    model,
                    optimizer,
                    stack,
                    TrainPipelineSparseDistOptStash,
                ),
                _CopyableBatch,
            ),
        }
        for (pipeline_name, make_pipeline), metadata in product(
            pipeline_factories.items(), (None, object())
        ):
            with (
                self.subTest(
                    pipeline=pipeline_name,
                    attributed=metadata is not None,
                ),
                ExitStack() as stack,
            ):
                model = _TrainingModel()
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
                pipeline, batch_type = make_pipeline(model, optimizer, stack)
                raw_batch = batch_type()
                batch = self._wrap_batch(raw_batch, metadata)
                next_batch = cast(
                    Pipelineable,
                    AttributedBatch(batch_type(), object()),
                )
                lookahead_batch = cast(
                    Pipelineable,
                    AttributedBatch(batch_type(), object()),
                )

                self._assert_optimizer_failure(
                    pipeline,
                    optimizer,
                    iter((batch, next_batch, lookahead_batch)),
                    metadata,
                )

                self.assertEqual([raw_batch], model.received_batches)

    def test_pec_optimizer_failure_uses_current_context(self) -> None:
        for metadata in (None, object()):
            with self.subTest(attributed=metadata is not None):
                model = _TrainingModel()
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
                with patch(
                    "torchrec.distributed.train_pipeline.pec_train_pipeline."
                    "torch.cuda.Event",
                    return_value=MagicMock(),
                ):
                    pipeline = TrainPipelinePEC(
                        model,
                        optimizer,
                        torch.device("cpu"),
                    )
                raw_batch = _PECBatch()
                batch = self._wrap_batch(raw_batch, metadata)

                with patch.object(pipeline, "_init_pipelined_modules"):
                    self._assert_optimizer_failure(
                        pipeline,
                        optimizer,
                        iter((batch, _PECBatch())),
                        metadata,
                    )

                self.assertEqual([raw_batch], model.received_batches)

    def test_base_and_pt2_forward_failure_uses_current_context(self) -> None:
        for pipeline_type, metadata in product(
            (TrainPipelineBase, TrainPipelinePT2),
            (None, object()),
        ):
            with self.subTest(
                pipeline_type=pipeline_type.__name__,
                attributed=metadata is not None,
            ):
                model = _TrainingModel()
                error = ValueError("forward failed")
                pipeline = self._make_pipeline(pipeline_type, model)
                first_batch = _CopyableBatch()
                raw_batch = _CopyableBatch()
                data_iter = iter(
                    (
                        self._wrap_batch(first_batch, object()),
                        self._wrap_batch(raw_batch, metadata),
                    )
                )

                pipeline.progress(data_iter)
                model.error = error
                with self.assertRaises(ValueError) as raised:
                    pipeline.progress(data_iter)

                self.assertIs(error, raised.exception)
                self.assertEqual([first_batch, raw_batch], model.received_batches)
                self._assert_attribution(error, metadata, 1)
