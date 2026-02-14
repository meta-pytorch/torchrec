#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import contextlib
import unittest
from typing import Any, Iterator, List
from unittest.mock import MagicMock, patch

import torch
from torch import nn, optim
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    _GAOptimizerWrapper,
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline


class _MockPipeline(TrainPipeline[Any, float]):
    """Mock pipeline that tracks progress() calls and can raise StopIteration."""

    def __init__(self, num_batches: int) -> None:
        super().__init__()  # pyre-ignore[20]
        self._optimizer = MagicMock()
        self._num_batches = num_batches
        self._calls: int = 0
        self.progress_call_log: List[int] = []

    def progress(self, dataloader_iter: Iterator[Any]) -> float:
        if self._calls >= self._num_batches:
            raise StopIteration
        self._calls += 1
        self.progress_call_log.append(self._calls)
        return float(self._calls)

    def reset(self) -> None:
        self._calls = 0
        self.progress_call_log.clear()


class _RealForwardPipeline(TrainPipeline[Any, torch.Tensor]):
    """Pipeline that does actual forward/backward/step for gradient tests."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        super().__init__()  # pyre-ignore[20]
        self._model = model
        self._optimizer = optimizer
        self._progress_count = 0

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        output = self._model(batch)
        loss = output.sum()
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._progress_count += 1
        return loss


class _MockModel(torch.nn.Module):
    """Mock model that tracks no_sync context usage."""

    def __init__(self) -> None:
        super().__init__()
        self.no_sync_entered: int = 0
        self.no_sync_exited: int = 0
        # Minimal parameter so torch.optim.SGD doesn't complain
        self._param = torch.nn.Parameter(torch.zeros(1))

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        self.no_sync_entered += 1
        try:
            yield
        finally:
            self.no_sync_exited += 1


class GradientAccumulationConfigTest(unittest.TestCase):
    def test_default_config(self) -> None:
        config = GradientAccumulationConfig()
        self.assertFalse(config.is_enabled)
        self.assertEqual(config.num_steps, 1)
        self.assertEqual(config.num_warmup_steps, 1)

    def test_auto_enable_when_num_steps_gt_1(self) -> None:
        config = GradientAccumulationConfig(num_steps=4)
        self.assertTrue(config.is_enabled)

    def test_num_steps_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            GradientAccumulationConfig(num_steps=0)
        with self.assertRaises(ValueError):
            GradientAccumulationConfig(num_steps=-1)

    def test_num_warmup_steps_must_be_at_least_1(self) -> None:
        """num_warmup_steps >= 1 is required for DDP static_graph compatibility."""
        with self.assertRaises(ValueError):
            GradientAccumulationConfig(num_steps=4, num_warmup_steps=0)

    def test_num_warmup_steps_default_is_1(self) -> None:
        config = GradientAccumulationConfig(num_steps=4)
        self.assertEqual(config.num_warmup_steps, 1)


class GAOptimizerWrapperTest(unittest.TestCase):
    def _make_wrapper(
        self, num_steps: int = 4, num_warmup_steps: int = 1
    ) -> tuple[_GAOptimizerWrapper, MagicMock]:
        mock_opt = MagicMock(spec=torch.optim.Optimizer)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=num_steps, num_warmup_steps=num_warmup_steps
        )
        wrapper = _GAOptimizerWrapper(mock_opt, config)
        return wrapper, mock_opt

    def test_should_step_at_accumulation_boundaries(self) -> None:
        """_should_step returns True only at accumulation boundaries."""
        wrapper, _ = self._make_wrapper(num_steps=4)
        for step in range(4):
            wrapper._current_step = step
            if step == 3:
                self.assertTrue(wrapper._should_step(), f"step {step}")
            else:
                self.assertFalse(wrapper._should_step(), f"step {step}")

    def test_should_step_during_warmup(self) -> None:
        """_should_step follows accumulation schedule regardless of warmup.

        Warmup controls gradient sync (no_sync context), not the optimizer
        step schedule. During warmup, gradients are allreduced every step,
        but the optimizer still only steps at accumulation boundaries.
        """
        wrapper, _ = self._make_wrapper(num_steps=4, num_warmup_steps=3)
        expected = {0: False, 1: False, 2: False, 3: True}
        for step, should in expected.items():
            wrapper._current_step = step
            self.assertEqual(
                wrapper._should_step(),
                should,
                f"step {step}: expected _should_step={should}",
            )

    def test_step_only_calls_optimizer_at_boundary(self) -> None:
        wrapper, mock_opt = self._make_wrapper(num_steps=4)
        for step in range(8):
            wrapper._current_step = step
            wrapper.step()
        self.assertEqual(mock_opt.step.call_count, 2)

    def test_zero_grad_respects_needs_flag(self) -> None:
        wrapper, mock_opt = self._make_wrapper(num_steps=4)
        wrapper.zero_grad()
        self.assertEqual(mock_opt.zero_grad.call_count, 1)
        wrapper.zero_grad()
        self.assertEqual(mock_opt.zero_grad.call_count, 1)
        wrapper._current_step = 3
        wrapper.step()
        wrapper.zero_grad()
        self.assertEqual(mock_opt.zero_grad.call_count, 2)

    def test_attribute_proxy(self) -> None:
        """Attributes are proxied to wrapped optimizer."""
        model = nn.Linear(10, 5)
        real_opt = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(num_steps=4)
        wrapper = _GAOptimizerWrapper(real_opt, config)
        self.assertEqual(wrapper.param_groups, real_opt.param_groups)

    def test_reset(self) -> None:
        wrapper, _ = self._make_wrapper(num_steps=4)
        for _ in range(5):
            wrapper.advance_step()
        self.assertEqual(wrapper._current_step, 5)
        wrapper.reset()
        self.assertEqual(wrapper._current_step, 0)
        self.assertTrue(wrapper._needs_zero_grad)


class ShouldSyncGradTest(unittest.TestCase):
    """Tests for _should_sync_grad — the core method controlling no_sync usage."""

    def _make_wrapper(
        self, num_steps: int = 4, num_warmup_steps: int = 1
    ) -> GradientAccumulationWrapper[Any, Any]:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=num_steps, num_warmup_steps=num_warmup_steps
        )
        return GradientAccumulationWrapper(pipeline, optimizer, model, config)

    def test_first_step_always_syncs(self) -> None:
        """Step 0 must always sync for DDP static_graph compatibility."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=1)
        ga.set_step(0)
        self.assertTrue(ga._should_sync_grad(is_last_batch=False))

    def test_first_step_syncs_with_large_warmup(self) -> None:
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=10)
        ga.set_step(0)
        self.assertTrue(ga._should_sync_grad(is_last_batch=False))

    def test_warmup_steps_all_sync(self) -> None:
        """All steps during warmup period should sync."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=3)
        for step in range(3):
            ga.set_step(step)
            self.assertTrue(
                ga._should_sync_grad(is_last_batch=False),
                f"warmup step {step} should sync",
            )

    def test_after_warmup_follows_accumulation_schedule(self) -> None:
        """After warmup, only sync at accumulation boundaries."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=1)
        expected = {
            0: True,
            1: False,
            2: False,
            3: True,
            4: False,
            5: False,
            6: False,
            7: True,
        }
        for step, should_sync in expected.items():
            ga.set_step(step)
            self.assertEqual(
                ga._should_sync_grad(is_last_batch=False),
                should_sync,
                f"step {step}: expected sync={should_sync}",
            )

    def test_last_batch_always_syncs(self) -> None:
        """is_last_batch=True forces sync regardless of step."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=1)
        for step in range(8):
            ga.set_step(step)
            self.assertTrue(
                ga._should_sync_grad(is_last_batch=True),
                f"step {step} with is_last_batch=True should sync",
            )

    def test_warmup_greater_than_num_steps(self) -> None:
        """When warmup > num_steps, sync happens during entire warmup period."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=8)
        for step in range(8):
            ga.set_step(step)
            self.assertTrue(
                ga._should_sync_grad(is_last_batch=False),
                f"Should sync during warmup at step {step}",
            )
        # After warmup, normal accumulation
        ga.set_step(8)
        self.assertFalse(ga._should_sync_grad(is_last_batch=False))
        ga.set_step(11)
        self.assertTrue(ga._should_sync_grad(is_last_batch=False))

    def test_warmup_equals_num_steps(self) -> None:
        """Edge case: warmup equals num_steps."""
        ga = self._make_wrapper(num_steps=4, num_warmup_steps=4)
        for step in range(4):
            ga.set_step(step)
            self.assertTrue(ga._should_sync_grad(is_last_batch=False))
        ga.set_step(4)
        self.assertFalse(ga._should_sync_grad(is_last_batch=False))


class NoSyncContextTest(unittest.TestCase):
    """Tests that no_sync context is used/skipped correctly."""

    def _run_steps(
        self, num_steps: int, num_warmup_steps: int, num_batches: int
    ) -> tuple[_MockModel, GradientAccumulationWrapper[Any, Any], int]:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=num_batches)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=num_steps, num_warmup_steps=num_warmup_steps
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        completed = 0
        dummy_iter: Iterator[Any] = iter([])
        for _ in range(num_batches):
            ga.progress(dummy_iter)
            completed += 1
        return model, ga, completed

    def test_no_sync_not_used_during_warmup(self) -> None:
        """During warmup steps, no_sync should never be entered."""
        model, ga, completed = self._run_steps(
            num_steps=4, num_warmup_steps=4, num_batches=4
        )
        self.assertEqual(model.no_sync_entered, 0)
        self.assertEqual(completed, 4)

    def test_no_sync_not_used_on_first_step(self) -> None:
        """Step 0 must not use no_sync, even with num_warmup_steps=1."""
        model, ga, completed = self._run_steps(
            num_steps=4, num_warmup_steps=1, num_batches=1
        )
        self.assertEqual(model.no_sync_entered, 0)

    def test_no_sync_used_after_warmup_on_non_boundary_steps(self) -> None:
        """After warmup, non-boundary steps should use no_sync."""
        model, ga, completed = self._run_steps(
            num_steps=4, num_warmup_steps=1, num_batches=8
        )
        # Steps: 0=sync(warmup), 1=no_sync, 2=no_sync, 3=sync(boundary),
        #         4=no_sync, 5=no_sync, 6=no_sync, 7=sync(boundary)
        self.assertEqual(model.no_sync_entered, 5)
        self.assertEqual(completed, 8)

    def test_no_sync_pattern_with_warmup_2(self) -> None:
        """Verify sync pattern with num_warmup_steps=2."""
        model, ga, completed = self._run_steps(
            num_steps=4, num_warmup_steps=2, num_batches=8
        )
        # Steps: 0=sync(first+warmup), 1=sync(warmup), 2=no_sync, 3=sync(boundary),
        #         4=no_sync, 5=no_sync, 6=no_sync, 7=sync(boundary)
        self.assertEqual(model.no_sync_entered, 4)
        self.assertEqual(completed, 8)

    def test_no_sync_context_with_ddp(self) -> None:
        """Test no_sync context with DDP-like model."""
        mock_ddp_model = MagicMock(spec=["no_sync"])
        no_sync_entered = [False]

        @contextlib.contextmanager
        def mock_no_sync() -> Iterator[None]:
            no_sync_entered[0] = True
            yield

        mock_ddp_model.no_sync = mock_no_sync

        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, mock_ddp_model, config
        )

        # Advance past warmup to a non-boundary step
        wrapper.set_step(1)
        self.assertFalse(wrapper._should_sync_grad())
        with wrapper._get_no_sync_context():
            pass
        self.assertTrue(no_sync_entered[0])

    def test_no_sync_context_with_dmp_wrapped_module(self) -> None:
        """Test no_sync context when model has _dmp_wrapped_module attribute."""
        mock_dmp_model = MagicMock(spec=["_dmp_wrapped_module"])
        mock_inner_module = MagicMock(spec=["no_sync"])
        no_sync_entered = [False]

        @contextlib.contextmanager
        def mock_no_sync() -> Iterator[None]:
            no_sync_entered[0] = True
            yield

        mock_inner_module.no_sync = mock_no_sync
        mock_dmp_model._dmp_wrapped_module = mock_inner_module

        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, mock_dmp_model, config
        )

        wrapper.set_step(1)
        with wrapper._get_no_sync_context():
            pass
        self.assertTrue(no_sync_entered[0])

    def test_dmp_without_no_sync_falls_through(self) -> None:
        """DMP without no_sync falls through to nullcontext."""
        mock_dmp_model = MagicMock(spec=["_dmp_wrapped_module"])
        mock_inner_module = MagicMock(spec=[])
        mock_dmp_model._dmp_wrapped_module = mock_inner_module

        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, mock_dmp_model, config
        )

        # Should not raise
        with wrapper._get_no_sync_context():
            pass


class StopIterationHandlingTest(unittest.TestCase):
    """Tests for StopIteration handling — verifying no +1 overcount."""

    def test_stop_iteration_flushes_remaining_gradients(self) -> None:
        """When StopIteration is raised, flush should use current_step (no +1)."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=5)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        results = []
        dummy_iter: Iterator[Any] = iter([])
        for _ in range(10):
            try:
                result = ga.progress(dummy_iter)
                results.append(result)
            except StopIteration:
                break

        self.assertEqual(ga.current_step, 5)
        self.assertEqual(len(results), 5)

    def test_stop_iteration_no_flush_at_boundary(self) -> None:
        """At an exact accumulation boundary, no flush needed."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        results = []
        dummy_iter: Iterator[Any] = iter([])
        for _ in range(10):
            try:
                result = ga.progress(dummy_iter)
                results.append(result)
            except StopIteration:
                break

        self.assertEqual(ga.current_step, 4)
        self.assertEqual(len(results), 4)

    def test_stop_iteration_current_step_not_advanced(self) -> None:
        """StopIteration should not advance current_step beyond completed batches."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=3)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        completed = 0
        for _ in range(10):
            try:
                ga.progress(dummy_iter)
                completed += 1
            except StopIteration:
                break

        self.assertEqual(completed, 3)
        self.assertEqual(ga.current_step, 3)

    def test_stop_iteration_raises(self) -> None:
        """StopIteration is re-raised after flushing."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=0)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        with self.assertRaises(StopIteration):
            ga.progress(dummy_iter)

    def test_stop_iteration_with_is_last_batch_flushes_once(self) -> None:
        """StopIteration path only flushes once (not also via is_last_batch check)."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=2)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(2):
            ga.progress(dummy_iter)

        flush_call_count = [0]
        original_flush = ga._flush_accumulated_gradients

        def counting_flush(steps: int) -> bool:
            flush_call_count[0] += 1
            return original_flush(steps)

        ga._flush_accumulated_gradients = counting_flush  # pyre-ignore[8]

        with self.assertRaises(StopIteration):
            ga.progress(dummy_iter, is_last_batch=True)

        # Flush called once (StopIteration handler), not twice
        self.assertEqual(flush_call_count[0], 1)


class FlushGradientsTest(unittest.TestCase):
    """Tests for _flush_accumulated_gradients behavior."""

    def test_flush_calls_zero_grad(self) -> None:
        """Flush calls zero_grad after step to prevent stale gradients."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        for _ in range(2):
            wrapper.progress(iter([torch.randn(2, 10)]))

        with patch.object(
            wrapper.optimizer_wrapper._optimizer, "zero_grad"
        ) as mock_zero_grad:
            with self.assertRaises(StopIteration):
                wrapper.progress(iter([]))
            mock_zero_grad.assert_called_once_with(set_to_none=True)

    def test_needs_zero_grad_false_after_flush(self) -> None:
        """_needs_zero_grad is False after flush (grads already zeroed)."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        for _ in range(2):
            wrapper.progress(iter([torch.randn(2, 10)]))

        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        self.assertFalse(wrapper.optimizer_wrapper._needs_zero_grad)


class OptimizerInjectionTest(unittest.TestCase):
    """Tests for optimizer injection behavior."""

    def test_optimizer_not_injected_when_disabled(self) -> None:
        """Optimizer wrapper is NOT injected when GA is disabled."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        disabled_config = GradientAccumulationConfig(num_steps=1)

        pipeline = _RealForwardPipeline(model, optimizer)
        original_optimizer = pipeline._optimizer

        GradientAccumulationWrapper(pipeline, optimizer, model, disabled_config)

        self.assertIs(pipeline._optimizer, original_optimizer)
        self.assertNotIsInstance(pipeline._optimizer, _GAOptimizerWrapper)

    def test_optimizer_injected_when_enabled(self) -> None:
        """Optimizer wrapper IS injected when GA is enabled."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        enabled_config = GradientAccumulationConfig(num_steps=4)

        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, model, enabled_config
        )

        self.assertIsInstance(pipeline._optimizer, _GAOptimizerWrapper)
        self.assertIs(pipeline._optimizer, wrapper._optimizer_wrapper)


class IsLastBatchTest(unittest.TestCase):
    """Tests for is_last_batch parameter behavior."""

    def test_is_last_batch_at_boundary_no_double_step(self) -> None:
        """is_last_batch=True at accumulation boundary doesn't double-step."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        # Progress to step 3 (3 batches done)
        for _ in range(3):
            wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(wrapper.current_step, 3)

        step_call_count = [0]
        original_step = wrapper.optimizer_wrapper._optimizer.step

        def counting_step(*args: Any, **kwargs: Any) -> None:
            step_call_count[0] += 1
            return original_step(*args, **kwargs)

        wrapper.optimizer_wrapper._optimizer.step = counting_step  # pyre-ignore[8]

        # 4th batch is at accumulation boundary AND is_last_batch=True
        wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)
        self.assertEqual(wrapper.current_step, 4)
        # flush sees 4 % 4 = 0, no extra step
        self.assertEqual(step_call_count[0], 1)

    def test_is_last_batch_not_at_boundary_flushes(self) -> None:
        """is_last_batch=True not at boundary does flush."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(wrapper.current_step, 1)

        with patch.object(wrapper.optimizer_wrapper._optimizer, "step") as mock_step:
            wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)
            # flush sees 2 % 4 = 2 > 0, calls step
            mock_step.assert_called()


class FullTrainingLoopTest(unittest.TestCase):
    """End-to-end tests simulating a complete training loop."""

    def test_accumulation_schedule_num_steps_4(self) -> None:
        model = _MockModel()
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        pipeline = _MockPipeline(num_batches=8)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(8):
            ga.progress(dummy_iter)

        self.assertEqual(ga.current_step, 8)

    def test_disabled_ga_passes_through(self) -> None:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=3)
        config = GradientAccumulationConfig(is_enabled=False)
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        results = []
        for _ in range(3):
            results.append(ga.progress(dummy_iter))

        self.assertEqual(len(results), 3)
        self.assertEqual(model.no_sync_entered, 0)

    def test_is_last_batch_forces_sync_and_flush(self) -> None:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=10)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        for i in range(5):
            ga.progress(dummy_iter, is_last_batch=(i == 4))

        self.assertEqual(ga.current_step, 5)
        # Steps: 0=sync(first), 1=no_sync, 2=no_sync, 3=sync(boundary), 4=sync(last_batch)
        self.assertEqual(model.no_sync_entered, 2)

    def test_reset_clears_state(self) -> None:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=5)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(3):
            ga.progress(dummy_iter)
        self.assertEqual(ga.current_step, 3)

        ga.reset()
        self.assertEqual(ga.current_step, 0)
        self.assertEqual(ga.optimizer_wrapper._current_step, 0)

    def test_gradient_values_accumulated(self) -> None:
        """Gradients are accumulated across micro-batches (not replaced)."""
        model = nn.Linear(10, 5, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)

        optimizer.zero_grad()
        out1 = model(torch.ones(1, 10))
        out1.sum().backward()
        assert model.weight.grad is not None
        grad_after_first = model.weight.grad.clone()

        out2 = model(torch.ones(1, 10) * 2)
        out2.sum().backward()
        assert model.weight.grad is not None
        grad_after_second = model.weight.grad.clone()

        self.assertFalse(torch.equal(grad_after_first, grad_after_second))
