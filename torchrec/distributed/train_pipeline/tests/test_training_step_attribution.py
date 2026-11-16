#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from itertools import product
from typing import Callable, cast
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
    TrainPipelineBase,
    TrainPipelinePT2,
    TrainPipelineSparseDist,
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

    def _assert_attribution(
        self,
        error: Exception,
        metadata: object | None,
        expected_batch_index: int,
    ) -> None:
        attribution = get_exception_attribution(error)
        if metadata is None:
            self.assertIsNone(attribution)
            return
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual("forward", attribution.operation)
        self.assertIs(metadata, attribution.metadata)
        self.assertEqual(expected_batch_index, attribution.batch_index)

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
