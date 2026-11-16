#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import product
from typing import cast
from unittest.mock import patch

import torch
from torch import nn
from torchrec.distributed.train_pipeline.batch_attribution import (
    AttributedBatch,
    get_exception_attribution,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    EvalPipelineSparseDist,
    TorchCompileConfig,
    TrainPipeline,
    TrainPipelineBase,
    TrainPipelinePT2,
    TrainPipelineSparseDist,
    TrainPipelineSparseDistCompAutograd,
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
    ) -> TrainPipelineSparseDist[Pipelineable, torch.Tensor]:
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineSparseDist[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
            custom_model_fwd=custom_model_fwd,
        )
        self.enterContext(
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
        self.enterContext(patch.object(pipeline, "_init_pipelined_modules"))
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
    ) -> None:
        error = ValueError("backward failed")

        def fail(_gradient: torch.Tensor) -> torch.Tensor:
            if on_backward is not None:
                on_backward()
            raise error

        self.addCleanup(model.weight.register_hook(fail).remove)

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(dataloader_iter)

        self.assertIs(error, raised.exception)
        self._assert_attribution(
            error,
            metadata,
            operation="backward",
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
                self.enterContext(patch.object(pipeline, "_init_pipelined_modules"))
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
                    self.enterContext(
                        patch.object(
                            pipeline,
                            "get_compiled_autograd_ctx",
                            side_effect=compiled_autograd_context.enter,
                        )
                    )
                    on_backward = compiled_autograd_context.record
                batch = (
                    cast(
                        Pipelineable,
                        AttributedBatch(raw_batch, metadata),
                    )
                    if metadata is not None
                    else raw_batch
                )

                self._assert_backward_failure(
                    pipeline,
                    model,
                    iter((batch, _CopyableBatch())),
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
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
                pipeline = pipeline_type(
                    model,
                    optimizer,
                    torch.device("cpu"),
                )
                raw_batch = _CopyableBatch()
                batch = (
                    cast(
                        _CopyableBatch,
                        AttributedBatch(raw_batch, metadata),
                    )
                    if metadata is not None
                    else raw_batch
                )
                next_batch = cast(
                    _CopyableBatch,
                    AttributedBatch(_CopyableBatch(), object()),
                )

                self._assert_backward_failure(
                    pipeline,
                    model,
                    iter((batch, next_batch)),
                    metadata,
                )

                self.assertEqual([raw_batch], model.received_batches)

    def test_base_backward_failure_does_not_reuse_context(self) -> None:
        model = _TrainingModel()
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineBase[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
        )
        first_batch = _CopyableBatch()
        second_batch = _CopyableBatch()
        first_metadata = object()
        second_metadata = object()
        data_iter = iter(
            (
                self._wrap_batch(first_batch, first_metadata),
                self._wrap_batch(second_batch, second_metadata),
                self._wrap_batch(_CopyableBatch(), object()),
            )
        )
        backward_error = ValueError("backward failed")

        def fail_backward(_gradient: torch.Tensor) -> torch.Tensor:
            raise backward_error

        backward_hook = model.weight.register_hook(fail_backward)
        self.addCleanup(backward_hook.remove)

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(data_iter)

        self.assertIs(backward_error, raised.exception)
        self._assert_attribution(
            backward_error,
            first_metadata,
            operation="backward",
        )

        backward_hook.remove()
        forward_error = ValueError("forward failed")
        model.error = forward_error

        with self.assertRaises(ValueError) as raised:
            pipeline.progress(data_iter)

        self.assertIs(forward_error, raised.exception)
        self._assert_attribution(
            forward_error,
            second_metadata,
            operation="forward",
            batch_index=1,
        )
        self.assertEqual([first_batch, second_batch], model.received_batches)

    def test_base_backward_failure_uses_new_context_after_success(self) -> None:
        model = _TrainingModel()
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineBase[Pipelineable, torch.Tensor](
            model,
            optimizer,
            torch.device("cpu"),
        )
        first_batch = _CopyableBatch()
        second_batch = _CopyableBatch()
        first_metadata = object()
        second_metadata = object()
        dataloader_iter = iter(
            (
                cast(
                    _CopyableBatch,
                    AttributedBatch(first_batch, first_metadata),
                ),
                cast(
                    _CopyableBatch,
                    AttributedBatch(second_batch, second_metadata),
                ),
            )
        )

        self.assertIs(model.output, pipeline.progress(dataloader_iter))
        self._assert_backward_failure(
            pipeline,
            model,
            dataloader_iter,
            second_metadata,
            batch_index=1,
        )

        self.assertEqual([first_batch, second_batch], model.received_batches)

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
