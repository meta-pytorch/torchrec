#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from functools import partial
from typing import cast, Sequence
from unittest.mock import patch

import torch
from torch import nn
from torchrec.distributed.train_pipeline.batch_attribution import (
    AttributedBatch,
    get_exception_attribution,
)
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
    PartialWindowPolicy,
)
from torchrec.distributed.train_pipeline.pipeline_context import TrainPipelineContext
from torchrec.distributed.train_pipeline.train_pipelines import (
    TrainPipelineBase,
    TrainPipelinePT2,
    TrainPipelineSparseDist,
)
from torchrec.streamable import Pipelineable


class _Batch(Pipelineable):
    def to(self, device: torch.device, non_blocking: bool) -> _Batch:
        raise AssertionError("CPU training must not copy the batch")

    def record_stream(self, stream: torch.Stream) -> None:
        raise AssertionError("CPU training must not record streams")


class _CopyableBatch(_Batch):
    def to(self, device: torch.device, non_blocking: bool) -> _CopyableBatch:
        return self


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, device="cpu"))
        self.output = torch.tensor([7.0], device="cpu")

    def forward(self, batch: Pipelineable | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(batch, _Batch)
        return self.weight * 2, self.output


def _raise_error(error: Exception, *_args: object) -> None:
    raise error


def _preload(
    pipeline: TrainPipelineSparseDist[Pipelineable, torch.Tensor],
    metadata: Sequence[object | None],
    start_index: int = 73,
) -> None:
    pipeline.batches.clear()
    pipeline.contexts.clear()
    pipeline.batches.extend(_Batch() for _ in metadata)
    for index, item in enumerate(metadata, start=start_index):
        context = TrainPipelineContext(index=index, version=1)
        context.batch_metadata = item
        pipeline.contexts.append(context)


class GradientAccumulationAttributionTest(unittest.TestCase):
    def _make_wrapper(
        self,
        metadata: Sequence[object | None],
        num_steps: int = 3,
        policy: PartialWindowPolicy = PartialWindowPolicy.STEP,
    ) -> tuple[
        GradientAccumulationWrapper[Pipelineable, torch.Tensor],
        TrainPipelineSparseDist[Pipelineable, torch.Tensor],
        torch.optim.SGD,
        _Model,
    ]:
        model = _Model()
        # Citrine C2: foreach=True for multi-tensor optimizer execution.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
        pipeline = TrainPipelineSparseDist(
            model,
            optimizer,
            torch.device("cpu"),
            dmp_collection_sync_interval_batches=None,
        )
        _preload(pipeline, metadata)
        wrapper = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(num_steps=num_steps),
            partial_window_policy=policy,
        )
        return wrapper, pipeline, optimizer, model

    def _fail_optimizer(
        self, optimizer: torch.optim.Optimizer, error: Exception
    ) -> None:
        self.addCleanup(
            optimizer.register_step_pre_hook(partial(_raise_error, error)).remove
        )

    def _observe_optimizer_gradient(
        self,
        optimizer: torch.optim.Optimizer,
        parameter: torch.Tensor,
    ) -> list[torch.Tensor]:
        observed: list[torch.Tensor] = []

        def record_gradient(
            _optimizer: torch.optim.Optimizer,
            _args: tuple[object, ...],
            _kwargs: dict[str, object],
        ) -> None:
            assert parameter.grad is not None
            observed.append(parameter.grad.clone())

        self.addCleanup(optimizer.register_step_pre_hook(record_gradient).remove)
        return observed

    def _assert_optimizer_attribution(
        self,
        error: Exception,
        metadata: list[object | None],
        batch_index: int,
    ) -> None:
        attribution = get_exception_attribution(error)
        self.assertIsNotNone(attribution)
        assert attribution is not None
        self.assertEqual("optimizer", attribution.operation)
        self.assertEqual(metadata, attribution.metadata)
        self.assertEqual(batch_index, attribution.batch_index)

    def test_pipeline_families_attribute_optimizer_failure(self) -> None:
        for pipeline_type in (TrainPipelineBase, TrainPipelinePT2):
            with self.subTest(pipeline_type=pipeline_type.__name__):
                metadata = [object(), object()]
                model = _Model()
                # Citrine C2: foreach=True for multi-tensor optimizer execution.
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)
                pipeline = pipeline_type(
                    model,
                    optimizer,
                    torch.device("cpu"),
                )
                wrapper = GradientAccumulationWrapper(
                    pipeline,
                    optimizer,
                    model,
                    GradientAccumulationConfig(num_steps=2),
                )
                dataloader_iter = iter(
                    cast(
                        Pipelineable,
                        AttributedBatch(_CopyableBatch(), item),
                    )
                    for item in metadata
                )

                error = ValueError("optimizer failed")
                self._fail_optimizer(optimizer, error)
                self.assertIs(model.output, wrapper.progress(dataloader_iter))
                with self.assertRaises(ValueError) as raised:
                    wrapper.progress(dataloader_iter)

                self.assertIs(error, raised.exception)
                self._assert_optimizer_attribution(
                    error,
                    metadata,
                    1,
                )

    def test_optimizer_failure_attributes_the_complete_window(self) -> None:
        metadata = [object(), object(), object()]
        wrapper, _, optimizer, model = self._make_wrapper(metadata)
        error = ValueError("optimizer failed")
        self._fail_optimizer(optimizer, error)

        for _ in range(2):
            self.assertIs(model.output, wrapper.progress(iter(())))
        with self.assertRaises(ValueError) as raised:
            wrapper.progress(iter(()))

        self.assertIs(error, raised.exception)
        self._assert_optimizer_attribution(raised.exception, metadata, 75)

    def test_forced_and_exhaustion_steps_use_the_partial_window(self) -> None:
        for route in ("forced", "exhaustion"):
            with self.subTest(route=route):
                metadata = [object(), object()]
                wrapper, _, optimizer, model = self._make_wrapper(metadata)
                error = ValueError("optimizer failed")
                self._fail_optimizer(optimizer, error)

                self.assertIs(model.output, wrapper.progress(iter(())))
                if route == "forced":
                    with self.assertRaises(ValueError) as raised:
                        wrapper.progress(iter(()), is_last_batch=True)
                else:
                    self.assertIs(model.output, wrapper.progress(iter(())))
                    self.assertEqual(
                        metadata,
                        wrapper.optimizer_wrapper.batch_metadata,
                    )
                    self.assertEqual(74, wrapper.optimizer_wrapper.index)
                    with self.assertRaises(ValueError) as raised:
                        wrapper.progress(iter(()))

                self.assertIs(error, raised.exception)
                self._assert_optimizer_attribution(raised.exception, metadata, 74)

    def test_unknown_microbatch_keeps_its_window_position(self) -> None:
        for unknown_position in ("first", "last"):
            with self.subTest(unknown_position=unknown_position):
                attributed_metadata = object()
                metadata = (
                    [None, attributed_metadata]
                    if unknown_position == "first"
                    else [attributed_metadata, None]
                )
                wrapper, _, optimizer, model = self._make_wrapper(metadata, num_steps=2)
                error = ValueError("optimizer failed")
                self._fail_optimizer(optimizer, error)

                self.assertIs(model.output, wrapper.progress(iter(())))
                with self.assertRaises(ValueError) as raised:
                    wrapper.progress(iter(()))

                self._assert_optimizer_attribution(raised.exception, metadata, 74)

    def test_unattributed_window_preserves_raw_optimizer_failure(self) -> None:
        wrapper, _, optimizer, model = self._make_wrapper([None, None], num_steps=2)
        error = ValueError("optimizer failed")
        self._fail_optimizer(optimizer, error)

        self.assertIs(model.output, wrapper.progress(iter(())))
        with self.assertRaises(ValueError) as raised:
            wrapper.progress(iter(()))

        self.assertIs(error, raised.exception)
        self.assertIsNone(get_exception_attribution(raised.exception))

    def test_context_lookup_failure_does_not_change_optimizer_execution(self) -> None:
        wrapper, pipeline, _, model = self._make_wrapper(
            [object(), object()], num_steps=2
        )
        with (
            patch.object(
                pipeline,
                "get_optimizer_context",
                side_effect=RuntimeError("context lookup failed"),
            ),
            patch(
                "torchrec.distributed.train_pipeline.gradient_accumulation."
                "logger.warning"
            ) as warning,
        ):
            self.assertIs(model.output, wrapper.progress(iter(())))
            self.assertIs(model.output, wrapper.progress(iter(())))

        self.assertEqual(2, warning.call_count)
        self.assertIn(
            "optimizer attribution context",
            warning.call_args.args[0],
        )
        self.assertIs(True, warning.call_args.kwargs.get("exc_info"))
        torch.testing.assert_close(
            model.weight.detach(), torch.tensor([0.6], device="cpu")
        )

    def test_successful_update_clears_the_previous_window(self) -> None:
        first_window = [object(), object()]
        wrapper, pipeline, optimizer, model = self._make_wrapper(
            first_window, num_steps=2
        )
        for _ in first_window:
            self.assertIs(model.output, wrapper.progress(iter(())))
        self.assertIsNone(wrapper.optimizer_wrapper.batch_metadata)
        self.assertIsNone(wrapper.optimizer_wrapper.index)

        next_metadata = object()
        _preload(pipeline, [next_metadata], start_index=80)
        error = ValueError("next window failed")
        self._fail_optimizer(optimizer, error)
        with self.assertRaises(ValueError) as raised:
            wrapper.progress(iter(()), is_last_batch=True)

        self._assert_optimizer_attribution(raised.exception, [next_metadata], 80)

    def test_failed_backward_discards_the_previous_window(self) -> None:
        metadata = [object(), object()]
        wrapper, pipeline, optimizer, model = self._make_wrapper(metadata)
        self.assertIs(model.output, wrapper.progress(iter(())))

        backward_error = ValueError("backward failed")
        hook = model.weight.register_hook(partial(_raise_error, backward_error))
        try:
            with self.assertRaises(ValueError) as raised:
                wrapper.progress(iter(()))
        finally:
            hook.remove()
        self.assertIs(backward_error, raised.exception)
        self.assertIsNone(wrapper.optimizer_wrapper.batch_metadata)
        self.assertIsNone(wrapper.optimizer_wrapper.index)

        next_metadata = object()
        _preload(pipeline, [next_metadata], start_index=80)
        optimizer_error = ValueError("next window failed")
        self._fail_optimizer(optimizer, optimizer_error)
        with self.assertRaises(ValueError) as raised:
            wrapper.progress(iter(()), is_last_batch=True)

        self._assert_optimizer_attribution(raised.exception, [next_metadata], 80)

    def test_window_discard_paths_drop_previous_contributor(self) -> None:
        for route in ("reset", "set_step", "discard"):
            with self.subTest(route=route):
                policy = (
                    PartialWindowPolicy.DISCARD
                    if route == "discard"
                    else PartialWindowPolicy.STEP
                )
                wrapper, pipeline, optimizer, model = self._make_wrapper(
                    [object()],
                    num_steps=2,
                    policy=policy,
                )
                self.assertIs(model.output, wrapper.progress(iter(())))

                if route == "reset":
                    wrapper.reset(drop_partial=True)
                elif route == "set_step":
                    wrapper.set_step(0, drop_partial=True)
                else:
                    with self.assertRaises(StopIteration):
                        wrapper.progress(iter(()))

                next_metadata = object()
                _preload(pipeline, [next_metadata], start_index=80)
                error = ValueError("next window failed")
                observed_gradients = self._observe_optimizer_gradient(
                    optimizer,
                    model.weight,
                )
                self._fail_optimizer(optimizer, error)
                with self.assertRaises(ValueError) as raised:
                    wrapper.progress(iter(()), is_last_batch=True)

                self.assertIs(error, raised.exception)
                torch.testing.assert_close(
                    observed_gradients,
                    [torch.tensor([2.0], device="cpu")],
                )
                self._assert_optimizer_attribution(
                    raised.exception,
                    [next_metadata],
                    80,
                )

    def test_realign_window_preserves_previous_contributor(self) -> None:
        first_metadata = object()
        wrapper, pipeline, optimizer, model = self._make_wrapper(
            [first_metadata],
            num_steps=2,
        )
        self.assertIs(model.output, wrapper.progress(iter(())))

        wrapper.optimizer_wrapper.realign_window()
        next_metadata = object()
        _preload(pipeline, [next_metadata], start_index=80)
        error = ValueError("realigned window failed")
        observed_gradients = self._observe_optimizer_gradient(
            optimizer,
            model.weight,
        )
        self._fail_optimizer(optimizer, error)

        with self.assertRaises(ValueError) as raised:
            wrapper.progress(iter(()), is_last_batch=True)

        self.assertIs(error, raised.exception)
        torch.testing.assert_close(
            observed_gradients,
            [torch.tensor([4.0], device="cpu")],
        )
        self._assert_optimizer_attribution(
            raised.exception,
            [first_metadata, next_metadata],
            80,
        )
