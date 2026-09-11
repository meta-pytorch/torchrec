#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Citrine missing_for_each_optimizer is suppressed file-wide below. These are CPU unit
# tests whose whole purpose is to pin the optimizer's INTERACTION with the GA wrapper
# (zero_grad interception, the step/no-step boundary, and bucket-view accumulation).
# `foreach=True` swaps SGD's internal implementation to the multi-tensor kernels, which is
# exactly the behaviour under test, and the perf benefit the rule is chasing is nil on a
# CPU unit test.
# @lint-ignore-every CITRINE

import contextlib
import unittest
from typing import Any, cast, Iterator, List
from unittest.mock import MagicMock, patch

import torch
from torch import nn, optim
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    _GAOptimizerWrapper,
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
    PartialWindowPolicy,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline


class _MockPipeline(TrainPipeline[Any, float]):
    """Mock pipeline that tracks progress() calls and can raise StopIteration."""

    def __init__(self, num_batches: int) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
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
        super().__init__()  # pyrefly: ignore[missing-argument]
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


class _EvalAwareForwardPipeline(TrainPipeline[Any, torch.Tensor]):
    """Forward/backward/step pipeline that respects model.training (like the real APS
    pipeline): training micros zero_grad -> forward -> backward -> step (so grads
    ACCUMULATE across a K-micro window via the GA-wrapped optimizer's gating), while eval
    micros are forward-only under no_grad (no backward, no optimizer step). Used to exercise
    the eval-reuse GA-state fix end-to-end."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._model = model
        self._optimizer = optimizer

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        if not self._model.training:
            with torch.no_grad():
                return self._model(batch).sum()
        # zero_grad FIRST (GA-wrapped: only zeros at micro 0 of a window), so backward
        # accumulates across the K micros; step is GA-wrapped (only at the boundary).
        self._optimizer.zero_grad()
        output = self._model(batch)
        loss = output.sum()
        loss.backward()
        self._optimizer.step()
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

        ga._flush_accumulated_gradients = (
            counting_flush  # pyrefly: ignore[bad-assignment]
        )

        with self.assertRaises(StopIteration):
            ga.progress(dummy_iter, is_last_batch=True)

        # Flush called once (StopIteration handler), not twice
        self.assertEqual(flush_call_count[0], 1)


class EvalInterludeGATest(unittest.TestCase):
    """Eval-reuse GA-state fix (Layer 1: _advance_state gated on model.training; plus the
    StopIteration flush guard). In-trainer / checkpoint eval reuses the SAME
    train_step -> progress() dispatch (keyed on _ga_config, not train/eval). Before the fix
    _advance_state() ran outside the model.training guard, so an eval interlude advanced the
    GA micro-step counter and desynced the K-micro window boundaries. These tests go RED
    without the fix (counter advances in eval / post-eval weights diverge / eval flushes) and
    GREEN with it."""

    def _make_wrapper(
        self, model: nn.Module, num_steps: int = 2
    ) -> GradientAccumulationWrapper[Any, torch.Tensor]:
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=num_steps, num_warmup_steps=1
        )
        pipeline = _EvalAwareForwardPipeline(model, optimizer)
        return GradientAccumulationWrapper(pipeline, optimizer, model, config)

    def test_eval_interlude_does_not_advance_counter(self) -> None:
        """Core red/green (Layer 1): an eval interlude must NOT advance the GA counter."""
        model = nn.Linear(10, 5)
        wrapper = self._make_wrapper(model, num_steps=2)
        batch = torch.randn(4, 10)

        model.train()
        wrapper.progress(iter([batch]))  # micro 0 -> current_step 1
        wrapper.progress(iter([batch]))  # micro 1 (boundary) -> current_step 2
        self.assertEqual(wrapper.current_step, 2)

        # Eval interlude: 3 micros (odd, NOT a multiple of K=2). Without the fix the counter
        # would jump to 5 (misaligned); with the fix it stays frozen at 2.
        model.eval()
        for _ in range(3):
            wrapper.progress(iter([batch]))
        self.assertEqual(
            wrapper.current_step,
            2,
            "eval advanced the GA micro-step counter -> K-micro window boundaries desynced",
        )

        # Resume training: the next window starts on a clean K-boundary.
        model.train()
        wrapper.progress(iter([batch]))  # current_step 3
        wrapper.progress(iter([batch]))  # current_step 4
        self.assertEqual(wrapper.current_step, 4)

    def test_eval_interlude_weight_parity_vs_no_eval(self) -> None:
        """End-to-end red/green: post-eval training weights are BIT-IDENTICAL to a no-eval
        control (the eval interlude must not perturb the training trajectory)."""
        torch.manual_seed(0)
        control = nn.Linear(10, 5)
        test = nn.Linear(10, 5)
        test.load_state_dict(control.state_dict())

        g = torch.Generator().manual_seed(123)
        train_batches = [torch.randn(4, 10, generator=g) for _ in range(8)]
        eval_batches = [torch.randn(4, 10, generator=g) for _ in range(3)]

        # Control: 8 training micros, no eval.
        wc = self._make_wrapper(control, num_steps=2)
        control.train()
        for b in train_batches:
            wc.progress(iter([b]))

        # Test: 4 train micros, eval interlude (3 = odd, non-multiple of K), 4 train micros.
        wt = self._make_wrapper(test, num_steps=2)
        test.train()
        for b in train_batches[:4]:
            wt.progress(iter([b]))
        test.eval()
        for b in eval_batches:
            wt.progress(iter([b]))
        test.train()
        for b in train_batches[4:]:
            wt.progress(iter([b]))

        # Weight parity FIRST so a regression fails on the actual weight divergence (not just
        # the counter): without the fix, the eval interlude desyncs the windows and the
        # post-eval training trajectory diverges from the no-eval control.
        for (name, pc), (_, pt) in zip(
            control.named_parameters(), test.named_parameters()
        ):
            self.assertTrue(
                torch.equal(pc, pt),
                f"post-eval weight divergence on {name}: the eval interlude perturbed training",
            )
        self.assertEqual(wt.current_step, 8)

    def test_eval_interlude_does_not_change_weights(self) -> None:
        """An eval interlude (forward-only) must not modify any training weight."""
        model = nn.Linear(10, 5)
        wrapper = self._make_wrapper(model, num_steps=2)
        batch = torch.randn(4, 10)
        model.train()
        for _ in range(2):
            wrapper.progress(iter([batch]))
        snapshot = [p.detach().clone() for p in model.parameters()]
        model.eval()
        for _ in range(3):
            wrapper.progress(iter([batch]))
        for before, p in zip(snapshot, model.parameters()):
            self.assertTrue(
                torch.equal(before, p), "eval interlude modified a training weight"
            )

    def test_eval_stop_iteration_does_not_flush(self) -> None:
        """D3 guard: an eval StopIteration must NEVER raw-flush / step the optimizer, even
        with pending mid-window gradients."""
        model = nn.Linear(10, 5)
        wrapper = self._make_wrapper(model, num_steps=2)
        batch = torch.randn(4, 10)

        # One training micro -> mid-window, _pending_uncommitted True.
        model.train()
        wrapper.progress(iter([batch]))
        self.assertTrue(wrapper._pending_uncommitted)

        flush_calls = [0]
        original_flush = wrapper._flush_accumulated_gradients

        def counting_flush(steps: int) -> bool:
            flush_calls[0] += 1
            return original_flush(steps)

        # pyre-ignore[8]: monkeypatch for the test
        wrapper._flush_accumulated_gradients = counting_flush
        model.eval()
        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))  # exhausted iter -> StopIteration under eval
        self.assertEqual(
            flush_calls[0], 0, "eval StopIteration triggered a raw flush/step"
        )


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

    def test_flush_partial_window_multi_rank_fails_closed(self) -> None:
        """ws>1 + a partial (uncommitted) window must FAIL-CLOSED rather than step
        un-reduced rank-local grads (replica divergence). The divisible-N path never
        reaches here; this guards the checkpoint-resume runtime-partial hole. Simulated
        via a mocked world_size=2 (single process)."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "partial final window"):
                wrapper._flush_accumulated_gradients(2)  # remaining = 2 % 4 = 2 > 0


class GAPartialWindowAbortTest(unittest.TestCase):
    """A3: a partial-window guard that raises rank-locally at ``world_size > 1`` strands its
    peers in the next collective until the ~30-minute NCCL watchdog fires, burying the real
    cause. These guards must abort every process group first.

    The matrix is deliberate about WHICH cells abort:

    * raw flush at ws>1 -- aborts under **both** policies. That branch fires BEFORE policy
      evaluation and is unconditional, so gating the abort on RAISE would leave the hang in
      place for every STEP caller.
    * ws<=1 -- never aborts, under either policy. No peers to strand, so the raise keeps its
      exact previous behavior.
    """

    _ABORT = (
        "torchrec.distributed.train_pipeline.gradient_accumulation"
        ".torch.distributed.distributed_c10d._abort_process_group"
    )

    def _wrapper(self, policy: PartialWindowPolicy) -> GradientAccumulationWrapper:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        return GradientAccumulationWrapper(
            pipeline, optimizer, model, config, partial_window_policy=policy
        )

    @contextlib.contextmanager
    def _force_world_size(self, world_size: int) -> Iterator[None]:
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=world_size):
            yield

    def test_raw_flush_ws2_aborts_under_raise(self) -> None:
        w = self._wrapper(PartialWindowPolicy.RAISE)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            with self.assertRaisesRegex(RuntimeError, "partial final window"):
                w._flush_accumulated_gradients(2)
            mock_abort.assert_called_once_with(None)

    def test_raw_flush_ws2_aborts_under_step(self) -> None:
        # The corrected cell. An earlier design gated this abort on RAISE, which would have
        # preserved the hang for every STEP caller -- and STEP is the default.
        w = self._wrapper(PartialWindowPolicy.STEP)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            with self.assertRaisesRegex(RuntimeError, "partial final window"):
                w._flush_accumulated_gradients(2)
            mock_abort.assert_called_once_with(None)

    def test_raw_flush_ws1_raises_without_abort_under_raise(self) -> None:
        w = self._wrapper(PartialWindowPolicy.RAISE)
        with patch(self._ABORT) as mock_abort, self._force_world_size(1):
            with self.assertRaisesRegex(RuntimeError, "world_size <= 1"):
                w._flush_accumulated_gradients(2)
            mock_abort.assert_not_called()

    def test_raw_flush_ws1_steps_without_abort_under_step(self) -> None:
        # Historical single-process behavior: sanctioned local step, no raise, no abort.
        w = self._wrapper(PartialWindowPolicy.STEP)
        with patch(self._ABORT) as mock_abort, self._force_world_size(1):
            self.assertTrue(w._flush_accumulated_gradients(2))
            mock_abort.assert_not_called()

    def test_full_window_never_aborts(self) -> None:
        # remaining == 0 -> not a partial window -> no raise and no abort at any world_size.
        w = self._wrapper(PartialWindowPolicy.RAISE)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            self.assertFalse(w._flush_accumulated_gradients(4))
            mock_abort.assert_not_called()


class GAPartialWindowDiscardTest(unittest.TestCase):
    """``PartialWindowPolicy.DISCARD``: a partial window at exhaustion is thrown away
    rank-locally -- zeroed, no optimizer step, no collective, no process-group abort -- so a
    consume-all phase simply ends instead of raising.

    The load-bearing detail is the ZERO. ``_GAOptimizerWrapper.zero_grad`` early-returns when
    ``_needs_zero_grad`` is False, and False is exactly the mid-window state, so a bare
    ``zero_grad()`` here is a guaranteed no-op and the discarded gradients would silently
    survive into the next window. The implementation re-arms the flag first; these tests pin
    that, not just "the branch was taken".
    """

    _ABORT = (
        "torchrec.distributed.train_pipeline.gradient_accumulation"
        ".torch.distributed.distributed_c10d._abort_process_group"
    )

    def _make(
        self, policy: PartialWindowPolicy, accumulate_into_buckets: bool = False
    ) -> tuple[GradientAccumulationWrapper, nn.Module, optim.Optimizer]:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=4,
            num_warmup_steps=1,
            accumulate_into_buckets=accumulate_into_buckets,
        )
        pipeline = _RealForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, model, config, partial_window_policy=policy
        )
        return wrapper, model, optimizer

    def _accumulate_partial_window(
        self, wrapper: GradientAccumulationWrapper, model: nn.Module, micros: int = 2
    ) -> None:
        """Run ``micros`` of a K=4 window and assert the trap preconditions actually hold:
        gradients are non-zero AND ``_needs_zero_grad`` is False."""
        for _ in range(micros):
            wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertFalse(
            wrapper.optimizer_wrapper._needs_zero_grad,
            "precondition broken: mid-window _needs_zero_grad should be False -- without "
            "it the zero_grad no-op trap this test exists for is not being exercised",
        )
        self.assertTrue(
            any(
                p.grad is not None and torch.count_nonzero(p.grad) > 0
                for p in model.parameters()
            ),
            "precondition broken: expected accumulated non-zero grads before the discard",
        )

    @contextlib.contextmanager
    def _force_world_size(self, world_size: int) -> Iterator[None]:
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=world_size):
            yield

    def test_discard_actually_zeroes_the_gradients(self) -> None:
        """THE regression test for the ``_needs_zero_grad`` no-op: non-zero grads in, zero
        (or None) grads out."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)

        self.assertFalse(wrapper._flush_accumulated_gradients(2))

        for name, p in model.named_parameters():
            self.assertTrue(
                p.grad is None or torch.count_nonzero(p.grad) == 0,
                f"discarded gradients survived on {name}",
            )

    def test_discard_takes_no_optimizer_step(self) -> None:
        """Weights must be bit-identical across the discard, and the real optimizer's
        ``step`` must not be called."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        before = [p.detach().clone() for p in model.parameters()]

        with patch.object(
            wrapper.optimizer_wrapper._optimizer, "step"
        ) as mock_step, patch.object(
            wrapper.optimizer_wrapper, "step"
        ) as mock_wrapper_step:
            wrapper._flush_accumulated_gradients(2)
            mock_step.assert_not_called()
            mock_wrapper_step.assert_not_called()

        for b, p in zip(before, model.parameters()):
            self.assertTrue(
                torch.equal(b, p), "DISCARD moved the weights (an optimizer step ran)"
            )

    def test_discard_never_aborts_process_groups_at_ws2(self) -> None:
        """The three-way matrix at ``world_size > 1``. STEP and RAISE both abort-then-raise
        (unchanged); DISCARD does neither -- an exhaustion-time partial window is the normal
        ending of a consume-all phase, not a fault."""
        for policy in (PartialWindowPolicy.STEP, PartialWindowPolicy.RAISE):
            with self.subTest(policy=policy):
                wrapper, model, _optimizer = self._make(policy)
                self._accumulate_partial_window(wrapper, model)
                with patch(self._ABORT) as mock_abort, self._force_world_size(2):
                    with self.assertRaisesRegex(RuntimeError, "partial final window"):
                        wrapper._flush_accumulated_gradients(2)
                    mock_abort.assert_called_once_with(None)

        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            self.assertFalse(wrapper._flush_accumulated_gradients(2))
            mock_abort.assert_not_called()

    def test_discard_does_not_rewind_the_counter_or_replay_warmup(self) -> None:
        """DISCARD must NOT call ``reset()``. Zeroing ``_current_step`` / ``_window_base``
        would re-enter GA/DDP warmup on every new iterator."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        step_before = wrapper.optimizer_wrapper._current_step
        base_before = wrapper.optimizer_wrapper._window_base

        wrapper._flush_accumulated_gradients(2)

        self.assertEqual(wrapper.optimizer_wrapper._current_step, step_before)
        self.assertEqual(wrapper.optimizer_wrapper._window_base, base_before)

    def test_discard_does_not_realign_the_window_itself(self) -> None:
        """``realign_window()`` is the CALLER's job (progress() does it right after the
        flush). Doing it here too would double-anchor."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)

        with patch.object(
            wrapper.optimizer_wrapper, "realign_window"
        ) as mock_realign, patch.object(
            wrapper.optimizer_wrapper, "reset"
        ) as mock_reset:
            wrapper._flush_accumulated_gradients(2)
            mock_realign.assert_not_called()
            mock_reset.assert_not_called()

    def test_discard_routes_through_the_wrapper_to_preserve_bucket_views(self) -> None:
        """Under ``accumulate_into_buckets`` the zero must go through
        ``_GAOptimizerWrapper.zero_grad`` -> ``_window_start_zero_grad`` (alias-preserving),
        NOT straight to the real optimizer, or the DDP ``gradient_as_bucket_view`` aliases
        are dropped."""
        wrapper, model, _optimizer = self._make(
            PartialWindowPolicy.DISCARD, accumulate_into_buckets=True
        )
        self._accumulate_partial_window(wrapper, model)

        with patch.object(wrapper, "_window_start_zero_grad") as mock_window_zero:
            wrapper._flush_accumulated_gradients(2)
            mock_window_zero.assert_called_once_with()

    def test_discard_is_inert_on_a_complete_window(self) -> None:
        """remaining == 0 is not a partial window: no zero, no step, no branch."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        grads_before = [
            None if p.grad is None else p.grad.detach().clone()
            for p in model.parameters()
        ]

        self.assertFalse(wrapper._flush_accumulated_gradients(4))

        for g, p in zip(grads_before, model.parameters()):
            if g is None:
                self.assertIsNone(p.grad)
            else:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.equal(g, p.grad))

    def test_explicit_is_last_batch_commits_in_band_rather_than_discarding(
        self,
    ) -> None:
        """Contract pin (raised by an independent critic). DISCARD scopes to the EXHAUSTION
        path only. An explicit ``is_last_batch=True`` partial window still commits in-band,
        exactly as under STEP -- ``is_last_batch`` forces ``should_sync``, so the grads are
        cross-rank reduced and the step is replica-safe AND keeps the data, which is
        strictly better than discarding it. Asymmetry with RAISE (which DOES fence the
        explicit path) is intentional, so lock it down."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        before = [p.detach().clone() for p in model.parameters()]

        # Micro 3 of a K=4 window, flagged as the last batch -> forced in-band step.
        wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)

        moved = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        self.assertTrue(
            moved,
            "explicit is_last_batch under DISCARD must still take the synchronized in-band "
            "step, not silently discard the window",
        )
        self.assertFalse(
            wrapper._pending_uncommitted,
            "the in-band commit must clear _pending_uncommitted so a later exhaustion does "
            "not double-handle the window",
        )

    def test_explicit_is_last_batch_still_raises_under_raise(self) -> None:
        """Negative control for the test above: RAISE keeps fencing the explicit path, so
        the DISCARD carve-out is genuinely policy-scoped and not a blanket removal."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.RAISE)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT):
            with self.assertRaisesRegex(RuntimeError, "is_last_batch"):
                wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)

    def test_discard_end_to_end_through_stop_iteration(self) -> None:
        """The real path: a partial window then an exhausted iterator. StopIteration still
        propagates, the grads are gone, and the caller's realign leaves the counter
        monotonic."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        step_before = wrapper.optimizer_wrapper._current_step

        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        for name, p in model.named_parameters():
            self.assertTrue(
                p.grad is None or torch.count_nonzero(p.grad) == 0,
                f"discarded gradients survived on {name} via the StopIteration path",
            )
        self.assertFalse(wrapper._pending_uncommitted)
        self.assertEqual(wrapper.optimizer_wrapper._current_step, step_before)
        self.assertEqual(wrapper.optimizer_wrapper._window_base, step_before)


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

        wrapper.optimizer_wrapper._optimizer.step = (
            counting_step  # pyrefly: ignore[bad-assignment]
        )

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

    def test_disabled_ga_does_not_signal_the_observer(self) -> None:
        """A disabled wrapper is a pure pass-through: it must leave the caller's own
        boundary state alone rather than forcing it True every batch."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=3)
        observer = MagicMock()
        ga = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=False),
            window_observer=observer,
        )

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(3):
            ga.progress(dummy_iter)

        observer.assert_not_called()
        # And the wrapper must not have injected the retired attribute names either.
        self.assertFalse(hasattr(pipeline, "_ga_should_step"))
        self.assertFalse(hasattr(pipeline, "_ga_at_window_start"))

    def test_enabled_ga_signals_observer_before_inner_progress(self) -> None:
        """The boundary must arrive BEFORE the inner progress(), or a sub-step that
        bypasses the wrapped optimizer would gate on the previous micro's answer."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)
        seen_at_progress: List[tuple[bool, bool]] = []
        state: dict[str, bool] = {"should_step": False, "at_window_start": False}

        def observer(*, should_step: bool, at_window_start: bool) -> None:
            state["should_step"] = should_step
            state["at_window_start"] = at_window_start

        inner_progress = pipeline.progress

        def recording_progress(dataloader_iter: Iterator[Any]) -> float:
            seen_at_progress.append((state["should_step"], state["at_window_start"]))
            return inner_progress(dataloader_iter)

        # pyre-ignore[8]: test double rebinds the bound method on the instance.
        pipeline.progress = recording_progress
        ga = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=True, num_steps=2),
            window_observer=observer,
        )
        model.train()

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(4):
            ga.progress(dummy_iter)

        # K=2: window starts on micros 0 and 2, boundary steps land on micros 1 and 3.
        self.assertEqual(
            seen_at_progress,
            [(False, True), (True, False), (False, True), (True, False)],
        )

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

        # Mid-window (3 of 4 micros) => an open partial window; reset must be an
        # explicit drop_partial opt-in (S2 clean-boundary guard).
        ga.reset(drop_partial=True)
        self.assertEqual(ga.current_step, 0)
        self.assertEqual(ga.optimizer_wrapper._current_step, 0)

    def test_reset_raises_on_open_partial_window(self) -> None:
        """S2: reset() with an open partial window fail-closes unless drop_partial."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=8)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        # 3 of 4 micros -> mid-window, un-stepped grads pending.
        for _ in range(3):
            ga.progress(dummy_iter)
        self.assertTrue(ga._pending_uncommitted)
        with self.assertRaises(RuntimeError):
            ga.reset()

        # Complete the window (4th micro commits/steps) -> clean boundary -> reset ok.
        ga.progress(dummy_iter)
        self.assertFalse(ga._pending_uncommitted)
        ga.reset()
        self.assertEqual(ga.current_step, 0)

    def test_gradient_values_accumulated(self) -> None:
        """Gradients are accumulated across micro-batches (not replaced)."""
        model = nn.Linear(10, 5, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)

        optimizer.zero_grad()
        out1 = model(torch.ones(1, 10))
        out1.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        grad_after_first = model.weight.grad.clone()

        out2 = model(torch.ones(1, 10) * 2)
        out2.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        grad_after_second = model.weight.grad.clone()

        self.assertFalse(torch.equal(grad_after_first, grad_after_second))


class _MockDDPModule(torch.nn.Module):
    """Mock module that simulates DistributedDataParallel with no_sync support.

    Tracks how many times no_sync is entered/exited so tests can verify that
    GradientAccumulationWrapper discovers and suppresses gradient sync on
    nested DDP instances (not just the outermost one).
    """

    def __init__(self) -> None:
        super().__init__()
        self.no_sync_entered: int = 0
        self.no_sync_exited: int = 0
        self._param = torch.nn.Parameter(torch.zeros(1))

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        self.no_sync_entered += 1
        try:
            yield
        finally:
            self.no_sync_exited += 1


class _ModelWithNestedDDP(torch.nn.Module):
    """Model with a REGISTERED inner DDP-like submodule (in ``_modules``).

    This exercises the discovery mechanism for inner DDPs that ARE registered submodules
    (found by the ``root.modules()`` walk and entered via no_sync).

    NOTE: production ``ShardedVariableLengthEmbeddingArch`` stores its lookup DDPs in a
    PLAIN Python list (``self._lookups``), NOT a registered submodule/ModuleList, so those
    DDPs are NOT discovered and reduce every micro (numerically correct, no reclaim). This
    mock is therefore the structural OPPOSITE of real VLE; the plain-list-not-discovered
    contract is covered by test_ga_bucket_view_alias.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dense_layer = torch.nn.Linear(10, 5)
        # A REGISTERED inner DDP (contrast: real VLE uses an unregistered plain list)
        self.inner_ddp = _MockDDPModule()


class NestedDDPNoSyncTest(unittest.TestCase):
    """Tests that _get_no_sync_context propagates to REGISTERED nested DDP modules.

    Verifies the discovery mechanism for inner DDPs that are registered submodules (in
    ``_modules``): _get_no_sync_context enters no_sync on them, so intermediate GA
    micro-batches accumulate locally instead of all-reducing.

    NOTE: this does NOT reflect production VLE. ShardedVariableLengthEmbeddingArch stores
    its lookup DDPs in an UNREGISTERED plain list, so they are NOT discovered and reduce
    every micro (numerically correct; no standalone-grad duplication to reclaim). See
    test_ga_bucket_view_alias for the plain-list contract + the real bucket-view alias
    lifecycle.
    """

    def setUp(self) -> None:
        self._patcher = patch(
            "torchrec.distributed.train_pipeline.gradient_accumulation.DistributedDataParallel",
            _MockDDPModule,
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    def _make_wrapper_with_nested_ddp(
        self,
        num_steps: int = 4,
        num_warmup_steps: int = 1,
        num_batches: int = 100,
    ) -> tuple[
        _ModelWithNestedDDP,
        _MockDDPModule,
        GradientAccumulationWrapper[Any, Any],
    ]:
        """Create a GradientAccumulationWrapper around a model with a REGISTERED nested DDP.

        The model itself does NOT have no_sync (it's not wrapped in an outer DDP), but it
        contains a REGISTERED inner DDP child module (in ``_modules``), which the discovery
        walk finds. NOTE: this is the structural OPPOSITE of production VLE, whose lookup
        DDPs live in an UNREGISTERED plain list and are NOT discovered (see
        test_ga_bucket_view_alias for the real plain-list contract).
        """
        model = _ModelWithNestedDDP()
        inner_ddp = model.inner_ddp
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=num_batches)
        config = GradientAccumulationConfig(
            is_enabled=True,
            num_steps=num_steps,
            num_warmup_steps=num_warmup_steps,
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)
        return model, inner_ddp, wrapper

    def test_inner_ddp_discovered_by_no_sync_context(self) -> None:
        """_get_no_sync_context enters no_sync on the inner DDP module."""
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp()
        wrapper.set_step(1)  # non-boundary, non-warmup → should use no_sync
        self.assertFalse(wrapper._should_sync_grad())

        with wrapper._get_no_sync_context():
            self.assertEqual(inner_ddp.no_sync_entered, 1)
        self.assertEqual(inner_ddp.no_sync_exited, 1)

    def test_dmp_wrapped_non_module_with_no_sync(self) -> None:
        """When _dmp_wrapped_module is NOT an nn.Module but has no_sync,
        its no_sync context is entered."""

        class _NonModuleWrapper:
            def __init__(self) -> None:
                self.no_sync_entered: int = 0
                self.no_sync_exited: int = 0

            @contextlib.contextmanager
            def no_sync(self) -> Iterator[None]:
                self.no_sync_entered += 1
                try:
                    yield
                finally:
                    self.no_sync_exited += 1

        non_module_wrapper = _NonModuleWrapper()
        model = torch.nn.Linear(10, 5)
        model._dmp_wrapped_module = non_module_wrapper  # type: ignore[assignment]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        wrapper.set_step(1)
        with wrapper._get_no_sync_context():
            self.assertEqual(non_module_wrapper.no_sync_entered, 1)
        self.assertEqual(non_module_wrapper.no_sync_exited, 1)

    def test_multiple_sibling_ddp_modules(self) -> None:
        """Multiple sibling DDP modules at the same level all get no_sync."""
        model = torch.nn.Module()
        model._param = torch.nn.Parameter(torch.zeros(1))
        inner_ddp_1 = _MockDDPModule()
        inner_ddp_2 = _MockDDPModule()
        model.add_module("inner_ddp_1", inner_ddp_1)
        model.add_module("inner_ddp_2", inner_ddp_2)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        wrapper.set_step(1)
        with wrapper._get_no_sync_context():
            self.assertEqual(inner_ddp_1.no_sync_entered, 1)
            self.assertEqual(inner_ddp_2.no_sync_entered, 1)
        self.assertEqual(inner_ddp_1.no_sync_exited, 1)
        self.assertEqual(inner_ddp_2.no_sync_exited, 1)

    def test_deeply_nested_ddp_modules(self) -> None:
        """DDP modules nested multiple levels deep are discovered."""
        model = torch.nn.Module()
        model._param = torch.nn.Parameter(torch.zeros(1))
        middle_layer = torch.nn.Module()
        deep_ddp = _MockDDPModule()
        middle_layer.add_module("deep_ddp", deep_ddp)
        model.add_module("middle_layer", middle_layer)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        wrapper.set_step(1)
        with wrapper._get_no_sync_context():
            self.assertEqual(deep_ddp.no_sync_entered, 1)
        self.assertEqual(deep_ddp.no_sync_exited, 1)

    def test_inner_ddp_no_sync_used_on_non_boundary_steps(self) -> None:
        """Inner DDP gets no_sync on non-boundary, post-warmup steps."""
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp(
            num_steps=4, num_warmup_steps=1, num_batches=8
        )

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(8):
            wrapper.progress(dummy_iter)

        # Steps: 0=sync(first), 1=no_sync, 2=no_sync, 3=sync(boundary),
        #         4=no_sync, 5=no_sync, 6=no_sync, 7=sync(boundary)
        self.assertEqual(inner_ddp.no_sync_entered, 5)
        self.assertEqual(inner_ddp.no_sync_exited, 5)

    def test_inner_ddp_no_sync_not_used_during_warmup(self) -> None:
        """Inner DDP no_sync should NOT be entered during warmup."""
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp(
            num_steps=4, num_warmup_steps=4, num_batches=4
        )

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(4):
            wrapper.progress(dummy_iter)

        self.assertEqual(inner_ddp.no_sync_entered, 0)

    def test_both_outer_and_inner_ddp_get_no_sync(self) -> None:
        """When model has BOTH outer DDP (via _dmp_wrapped_module) and inner
        DDP, both should get no_sync."""
        model = _ModelWithNestedDDP()
        inner_ddp = model.inner_ddp

        # Wrap the model in a mock DMP that has an outer DDP
        outer_ddp = _MockDDPModule()
        outer_ddp.add_module("inner_model", model)

        dmp_model = torch.nn.Module()
        dmp_model._dmp_wrapped_module = outer_ddp  # type: ignore[assignment]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, dmp_model, config)

        wrapper.set_step(1)  # non-boundary, non-warmup
        with wrapper._get_no_sync_context():
            self.assertEqual(outer_ddp.no_sync_entered, 1)
            self.assertEqual(inner_ddp.no_sync_entered, 1)

        self.assertEqual(outer_ddp.no_sync_exited, 1)
        self.assertEqual(inner_ddp.no_sync_exited, 1)

    def test_no_ddp_modules_yields_without_error(self) -> None:
        """Model with no DDP modules at all should yield without error."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        pipeline = _MockPipeline(num_batches=100)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        wrapper.set_step(1)
        entered = False
        with wrapper._get_no_sync_context():
            entered = True
        self.assertTrue(entered, "no_sync context should yield successfully")

    def test_inner_ddp_no_sync_exited_on_exception(self) -> None:
        """Inner DDP no_sync is properly exited even if body raises."""
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp()
        wrapper.set_step(1)

        with self.assertRaises(RuntimeError):
            with wrapper._get_no_sync_context():
                self.assertEqual(inner_ddp.no_sync_entered, 1)
                raise RuntimeError("test error")

        self.assertEqual(inner_ddp.no_sync_exited, 1)


class _SplitOrDefaultModePipeline(TrainPipeline[Any, torch.Tensor]):
    """Faithful CPU model of the APS optimizer-step paths for GA cross-window testing.

    APS ``train_pipeline.py`` zeros grads at the top of ``progress()`` through the
    (GA-wrapper-replaced) ``self._optimizer.zero_grad()`` (train_pipeline.py:1401,
    gated by ``_needs_zero_grad``), then dispatches the optimizer STEP one of two ways
    (train_pipeline.py:1461-1470):

    * ``bypass_wrapper_step=False`` (``_step_optimizer_default``): step via the
      wrapper ``self._optimizer.step()`` -> the wrapper resets ``_needs_zero_grad``
      at the accumulation boundary (gradient_accumulation.py:101).
    * ``bypass_wrapper_step=True`` (``_step_optimizer_embedding_lookup_fwd`` /
      ``_step_optimizer_fp_allreduce``): step a CHILD optimizer directly (mirrors
      ``self._dense_optimizer``/``self._sparse_optimizer``), gated on the GA consume
      boundary the wrapper delivers via ``window_observer`` (== ``_ga_at_consume_boundary()``).
      This path NEVER calls the wrapper ``.step()``, so nothing resets ``_needs_zero_grad``.

    Records the post-backward grad of ``model.weight`` at every micro-batch so a test
    can detect grads leaking ACROSS logical windows (a stale-grad-reset bug).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        bypass_wrapper_step: bool,
    ) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._model = model
        # self._optimizer is replaced by the GA wrapper on enable; _child_optimizer
        # keeps the raw reference the split path would step directly.
        self._optimizer = optimizer
        self._child_optimizer = optimizer
        self._bypass_wrapper_step = bypass_wrapper_step
        self.post_backward_grads: List[torch.Tensor] = []
        self.child_step_count: int = 0
        # Owned by this pipeline, defaulted to the non-GA answer; the wrapper updates it
        # through window_observer when GA is enabled.
        self._ga_should_step: bool = True
        self._ga_at_window_start: bool = True

    def set_ga_window_state(self, *, should_step: bool, at_window_start: bool) -> None:
        self._ga_should_step = should_step
        self._ga_at_window_start = at_window_start

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        # Top-of-progress zero_grad through the (wrapper-replaced) optimizer.
        self._optimizer.zero_grad()
        out = self._model(batch)
        loss = out.sum()
        loss.backward()
        weight = cast(torch.Tensor, self._model.weight)
        assert weight.grad is not None
        self.post_backward_grads.append(weight.grad.detach().clone())
        if self._bypass_wrapper_step:
            # Split-optimizer path: step the child directly, gated on the GA boundary.
            if self._ga_should_step:
                self.child_step_count += 1
                self._child_optimizer.step()
        else:
            # Default path: step through the wrapper (self-gated at the boundary).
            self._optimizer.step()
        return loss


class SplitModeCrossWindowGradTest(unittest.TestCase):
    """P1.1 probe / R3 red-green regression for the split-mode cross-window
    gradient-zeroing contract.

    Uses x==ones so every micro-batch contributes an identical grad (== ones) to a
    bias-free Linear (grad == input, weight-independent), and lr==0 so weights stay
    fixed and grads stay input-determined. With K=2 and 4 micro-batches (2 full
    windows), each window START (micro 0 and micro 2) must see a FRESH single-micro
    grad if zero_grad fires once per window:

      CORRECT: post_backward_grads == [1g, 2g, 1g, 2g]
      LEAK:    post_backward_grads == [1g, 2g, 3g, 4g]  (grads never re-zeroed)
    """

    def _run(
        self,
        bypass_wrapper_step: bool,
        k: int = 2,
        num_micros: int = 4,
        mark_last: bool = False,
    ) -> _SplitOrDefaultModePipeline:
        torch.manual_seed(0)
        model = nn.Linear(4, 1, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        pipeline = _SplitOrDefaultModePipeline(model, optimizer, bypass_wrapper_step)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=k, num_warmup_steps=1
        )
        wrapper = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            config,
            window_observer=pipeline.set_ga_window_state,
        )
        data: Iterator[torch.Tensor] = iter(
            [torch.ones(1, 4) for _ in range(num_micros)]
        )
        for i in range(num_micros):
            wrapper.progress(data, is_last_batch=(mark_last and i == num_micros - 1))
        return pipeline

    def test_default_mode_zeros_grad_each_window(self) -> None:
        """CONTROL: default step path (wrapper.step) resets _needs_zero_grad, so each
        window start sees a fresh single-micro grad. Must PASS on current code."""
        pipeline = self._run(bypass_wrapper_step=False)
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 4)
        # Window starts (micro 0, micro 2) see a fresh single-micro grad.
        self.assertTrue(
            torch.allclose(g[2], g[0]),
            f"default-mode window-2-start grad {g[2].flatten().tolist()} != "
            f"window-1-start grad {g[0].flatten().tolist()}",
        )
        # Within-window accumulation still holds (boundary == 2x window-start).
        self.assertTrue(torch.allclose(g[1], 2.0 * g[0]))
        self.assertTrue(torch.allclose(g[3], 2.0 * g[2]))

    def test_split_mode_zeros_grad_each_window(self) -> None:
        """R3 RED->GREEN: split step path (child.step bypasses the wrapper) must ALSO
        zero grads once per window. On current code _needs_zero_grad is never reset in
        split mode, so grads leak across windows and this FAILS (red)."""
        pipeline = self._run(bypass_wrapper_step=True)
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 4)
        self.assertTrue(
            torch.allclose(g[2], g[0]),
            "CROSS-WINDOW GRAD LEAK in split mode: window-2-start grad "
            f"{g[2].flatten().tolist()} != window-1-start grad "
            f"{g[0].flatten().tolist()} (expected equal; unequal => _needs_zero_grad "
            "never reset because split child-step bypasses _GAOptimizerWrapper.step()).",
        )
        self.assertTrue(torch.allclose(g[3], 2.0 * g[2]))

    def test_split_mode_partial_window_commits_in_band(self) -> None:
        """R3 partial window (N%K!=0): K=2, 3 micros = 1 full window [0,1] + 1 partial
        window [2] marked is_last_batch. The partial window must commit in-band (the
        split child steps once) with freshly-zeroed grads and no double-step."""
        pipeline = self._run(
            bypass_wrapper_step=True, k=2, num_micros=3, mark_last=True
        )
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 3)
        # Window starts (micro 0, micro 2) see a fresh single-micro grad; the full
        # window boundary (micro 1) sees 2x.
        self.assertTrue(torch.allclose(g[0], torch.ones_like(g[0])))
        self.assertTrue(torch.allclose(g[1], 2.0 * torch.ones_like(g[1])))
        self.assertTrue(torch.allclose(g[2], torch.ones_like(g[2])))
        # ceil(3/2) == 2 optimizer steps: micro-1 boundary + micro-2 forced last batch.
        self.assertEqual(pipeline.child_step_count, 2)

    def test_default_mode_partial_window_commits_in_band(self) -> None:
        """Same partial window for default mode: the one-shot forced wrapper step handles
        the final r<K window and grads are zeroed per window (no cross-window leak)."""
        pipeline = self._run(
            bypass_wrapper_step=False, k=2, num_micros=3, mark_last=True
        )
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 3)
        self.assertTrue(torch.allclose(g[0], torch.ones_like(g[0])))
        self.assertTrue(torch.allclose(g[1], 2.0 * torch.ones_like(g[1])))
        self.assertTrue(torch.allclose(g[2], torch.ones_like(g[2])))

    def test_partial_window_trailing_stop_iteration_no_double_step(self) -> None:
        """After an explicit partial window commits in-band (is_last_batch on the final
        r<K micro), a trailing progress() that raises StopIteration must NOT re-flush /
        re-step: the window already committed so _pending_uncommitted is False. Covers
        both split and default modes."""
        for bypass in (True, False):
            torch.manual_seed(0)
            model = nn.Linear(4, 1, bias=False)
            optimizer = optim.SGD(model.parameters(), lr=0.0)
            pipeline = _SplitOrDefaultModePipeline(model, optimizer, bypass)
            config = GradientAccumulationConfig(
                is_enabled=True, num_steps=2, num_warmup_steps=1
            )
            wrapper = GradientAccumulationWrapper(
                pipeline,
                optimizer,
                model,
                config,
                window_observer=pipeline.set_ga_window_state,
            )
            data: Iterator[torch.Tensor] = iter([torch.ones(1, 4) for _ in range(3)])
            wrapper.progress(data)  # micro 0 (window-1 start)
            wrapper.progress(data)  # micro 1 (window-1 boundary)
            wrapper.progress(data, is_last_batch=True)  # micro 2 (partial, in-band)
            self.assertFalse(
                wrapper._pending_uncommitted,
                f"partial window not committed in-band (bypass={bypass})",
            )
            flush_calls = [0]
            orig_flush = wrapper._flush_accumulated_gradients

            def _counting_flush(steps: int, _o=orig_flush, _c=flush_calls) -> bool:
                _c[0] += 1
                return _o(steps)

            wrapper._flush_accumulated_gradients = (
                _counting_flush  # pyrefly: ignore[bad-assignment]
            )
            with self.assertRaises(StopIteration):
                wrapper.progress(iter([]))  # trailing: exhausted -> StopIteration
            self.assertEqual(
                flush_calls[0],
                0,
                f"trailing StopIteration re-flushed after in-band commit (bypass={bypass})",
            )


class _IteratorDrivenStepPipeline(TrainPipeline[Any, int]):
    """Pipeline that consumes from the PASSED iterator and steps the (GA-wrapped) optimizer.

    Consuming the passed iterator -- rather than an internal counter -- is what lets a test
    exhaust iterator A and then drive iterator B through the SAME wrapper, which is the
    scenario the window-anchor exists for. ``self._optimizer`` is replaced by the
    ``_GAOptimizerWrapper`` at wrapper construction, so ``step()`` here is gated and the
    underlying mock optimizer records only the calls that actually reached it.
    """

    def __init__(self) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._optimizer = MagicMock()
        # Owned by this pipeline; the wrapper updates them through window_observer.
        self._ga_at_window_start: bool = False
        self._ga_should_step: bool = False

    def set_ga_window_state(self, *, should_step: bool, at_window_start: bool) -> None:
        self._ga_should_step = should_step
        self._ga_at_window_start = at_window_start

    def progress(self, dataloader_iter: Iterator[Any]) -> int:
        value = next(dataloader_iter)  # StopIteration when exhausted
        self._optimizer.step()
        return value


class WindowRealignmentTest(unittest.TestCase):
    """Re-anchoring the K-window when a data iterator is exhausted.

    The optimizer-step / grad-sync / window-start boundaries are computed off a
    free-running micro counter. A phase that consumes a NON-multiple of K micro-batches
    before exhausting leaves that counter off-modulo, so every SUBSEQUENT window's boundary
    is shifted off the trainer's logical step. ``realign_window()`` moves the window anchor
    to the exhaustion point; it deliberately does NOT reset ``current_step`` (which would
    break the public accessor's monotonic contract and replay ``num_warmup_steps`` on every
    new iterator).
    """

    K: int = 4

    def _make(self, num_warmup_steps: int = 1) -> tuple[
        GradientAccumulationWrapper[Any, int],
        _IteratorDrivenStepPipeline,
        _MockModel,
        MagicMock,
    ]:
        model = _MockModel()
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        pipeline = _IteratorDrivenStepPipeline()
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=self.K, num_warmup_steps=num_warmup_steps
        )
        wrapper: GradientAccumulationWrapper[Any, int] = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            config,
            window_observer=pipeline.set_ga_window_state,
        )
        model.train()
        return wrapper, pipeline, model, optimizer

    @staticmethod
    def _exhaust(wrapper: GradientAccumulationWrapper[Any, int], n: int) -> None:
        """Consume n micros from a fresh iterator, then drive it to StopIteration."""
        it: Iterator[int] = iter(range(n))
        for _ in range(n):
            wrapper.progress(it)
        try:
            wrapper.progress(it)
        except StopIteration:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("iterator did not raise StopIteration")

    def test_combined_boundary_moves_together_after_exhaustion(self) -> None:
        """THE load-bearing test: after a non-K-multiple exhaustion, the optimizer step, the
        DDP sync decision and the ``at_window_start`` signal must ALL move to the
        SAME new boundary.

        Asserting optimizer steps alone can pass while grad sync stays on the old boundary,
        which is precisely the silent replica-divergence bug this guards.
        """
        wrapper, pipeline, model, optimizer = self._make()

        # Iterator A: 6 micros, 6 % 4 == 2 -> exhausts OFF the window boundary.
        self._exhaust(wrapper, 6)
        # The partial window was committed in-band by the flush (world_size <= 1, policy
        # STEP), then the window was re-anchored at global micro 6.
        self.assertEqual(wrapper.steps_in_window, 0)
        self.assertEqual(wrapper.current_step, 6)

        optimizer.step.reset_mock()
        sync_before = model.no_sync_entered

        # Iterator B: record all three signals per micro.
        stepped: list[bool] = []
        synced: list[bool] = []
        at_start: list[bool] = []
        it: Iterator[int] = iter(range(self.K))
        for _ in range(self.K):
            n_steps_before = optimizer.step.call_count
            no_sync_before = model.no_sync_entered
            wrapper.progress(it)
            stepped.append(optimizer.step.call_count > n_steps_before)
            # no_sync NOT entered => this micro ran under nullcontext => DDP synced.
            synced.append(model.no_sync_entered == no_sync_before)
            at_start.append(pipeline._ga_at_window_start)

        # Without realignment the boundary would land on iterator B's SECOND micro (global
        # step 7, since (7 + 1) % 4 == 0) instead of its fourth -- so `stepped` would read
        # [False, True, False, False] and every one of these assertions would fail.
        self.assertEqual(
            stepped, [False, False, False, True], "optimizer-step boundary"
        )
        self.assertEqual(synced, [False, False, False, True], "DDP grad-sync boundary")
        self.assertEqual(at_start, [True, False, False, False], "window-start signal")
        self.assertGreater(model.no_sync_entered, sync_before)

    def test_partial_flush_after_realignment_uses_window_relative_count(self) -> None:
        """Mutation guard for the flush CALL-SITE input.

        Every other exhaustion test in this class runs while the anchor is still 0, where
        ``steps_in_window == current_step`` -- so passing the global counter to
        ``_flush_accumulated_gradients`` is indistinguishable from passing the
        window-relative one, and reverting that call site would keep them all green.

        Here the anchor is 6 and 2 micros are accumulated. The correct input is 2
        (``remaining = 2 % 4 = 2`` -> partial window -> sanctioned in-band step), whereas the
        global 8 gives ``remaining = 8 % 4 == 0`` -> the flush returns False and the partial
        window is silently dropped, un-stepped.
        """
        wrapper, _pipeline, _model, optimizer = self._make()

        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper.steps_in_window, 0)
        self.assertEqual(wrapper.current_step, 6)

        optimizer.step.reset_mock()

        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it)
        # Global counter is ON a K boundary; the window-relative one is NOT.
        self.assertEqual(wrapper.current_step, 8)
        self.assertEqual(wrapper.steps_in_window, 2)
        self.assertEqual(
            optimizer.step.call_count, 0, "no boundary reached inside a partial window"
        )

        with self.assertRaises(StopIteration):
            wrapper.progress(it)

        self.assertEqual(
            optimizer.step.call_count,
            1,
            "the partial window (remaining=2) must be committed in-band; feeding the "
            "flush the GLOBAL counter computes remaining=0 and skips the commit",
        )

    def test_current_step_stays_monotonic_across_exhaustion(self) -> None:
        """Anti-regression for the REJECTED counter-reset design: the public accessor's
        monotonic contract must survive re-anchoring."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper.current_step, 6)

        seen = [wrapper.current_step]
        it: Iterator[int] = iter(range(4))
        for _ in range(4):
            wrapper.progress(it)
            seen.append(wrapper.current_step)
        self.assertEqual(seen, [6, 7, 8, 9, 10])

    def test_warmup_does_not_replay_on_a_new_iterator(self) -> None:
        """``num_warmup_steps`` counts against the GLOBAL micro counter, so a re-anchored
        window must NOT re-enter warmup (a counter reset would have)."""
        wrapper, _pipeline, model, _optimizer = self._make(num_warmup_steps=3)
        self._exhaust(wrapper, 6)

        # First micro of the new window: warmup is long over (current_step == 6 >= 3), and
        # it is not a boundary micro, so it must run under no_sync.
        before = model.no_sync_entered
        it: Iterator[int] = iter(range(1))
        wrapper.progress(it)
        self.assertEqual(
            model.no_sync_entered,
            before + 1,
            "warmup replayed on the new iterator (forced a sync on a non-boundary micro)",
        )

    def test_eval_exhaustion_does_not_realign(self) -> None:
        """An eval interlude whose iterator exhausts must leave GA state untouched, so a
        mid-window TRAINING window survives it."""
        wrapper, _pipeline, model, _optimizer = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it)
        self.assertEqual(wrapper.steps_in_window, 2)

        model.eval()
        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))
        self.assertEqual(
            wrapper.steps_in_window,
            2,
            "an eval StopIteration re-anchored the window and dropped the open training window",
        )
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)

    def test_bucket_views_ready_survives_realignment(self) -> None:
        """Re-anchoring is a window-boundary concern; the DDP bucket-view aliases are a
        property of the (unchanged) model + DDP instances and must not be invalidated.
        """
        wrapper, _pipeline, _model, _optimizer = self._make()
        wrapper._bucket_views_ready = True
        self._exhaust(wrapper, 6)
        self.assertTrue(wrapper._bucket_views_ready)

    def test_set_step_after_realignment_uses_raw_step_semantics(self) -> None:
        """``set_step`` callers place the wrapper at a point in the accumulation cycle and
        assume ``(step + 1) % K`` on the RAW value, so it must also drop the anchor -- else a
        prior realignment leaves a stale base and steps_in_window goes negative."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 6)

        wrapper.set_step(3)  # K=4 => (3 + 1) % 4 == 0 => on-boundary FULL window
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)
        self.assertEqual(wrapper.steps_in_window, 3)
        self.assertTrue(wrapper.optimizer_wrapper._should_step())

    def test_reset_clears_the_window_anchor(self) -> None:
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 6)
        wrapper.reset()
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)
        self.assertEqual(wrapper.current_step, 0)
        self.assertEqual(wrapper.steps_in_window, 0)

    def test_clean_boundary_exhaustion_also_realigns(self) -> None:
        """Exhausting ON a boundary leaves nothing to flush, but the anchor must still track
        the counter so the two never drift apart."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, self.K)
        self.assertEqual(wrapper.current_step, self.K)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, self.K)
        self.assertEqual(wrapper.steps_in_window, 0)

    def test_is_last_batch_realigns_after_advancing(self) -> None:
        """The generic-caller path: an explicit last batch closes the phase, so the next
        iterator starts a fresh window even though this one ended off-modulo."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it, is_last_batch=True)
        self.assertEqual(wrapper.current_step, 2)
        self.assertEqual(wrapper.steps_in_window, 0)

    def test_steps_in_window_never_negative_across_orderings(self) -> None:
        """Guards the stale-anchor family of bugs across the reachable public orderings."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertGreaterEqual(wrapper.steps_in_window, 0)
        wrapper.set_step(2)
        self.assertGreaterEqual(wrapper.steps_in_window, 0)
        self._exhaust(wrapper, 3)
        self.assertGreaterEqual(wrapper.steps_in_window, 0)
        wrapper.reset()
        self.assertGreaterEqual(wrapper.steps_in_window, 0)


class UnsupportedPipelineTest(unittest.TestCase):
    """C3: an ENABLED GA config plus a pipeline that exposes no ``_optimizer`` at all must
    fail closed. Previously the injection was silently skipped and that pipeline then
    stepped on every micro-batch while the caller believed gradients accumulated over K.

    Scope is deliberately narrow: ``hasattr('_optimizer')`` is necessary, not sufficient --
    a pipeline can expose ``_optimizer``, accept the replacement, and still step a
    separately captured optimizer. That needs a GA capability/rebind contract and is NOT
    what this guard claims.
    """

    class _NoOptimizerPipeline(TrainPipeline[Any, int]):
        def __init__(self) -> None:
            super().__init__()  # pyrefly: ignore[missing-argument]

        def progress(self, dataloader_iter: Iterator[Any]) -> int:
            return 0

    def test_enabled_ga_with_no_optimizer_attr_raises(self) -> None:
        pipeline = self._NoOptimizerPipeline()
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(is_enabled=True, num_steps=4)
        with self.assertRaises(RuntimeError) as ctx:
            GradientAccumulationWrapper(pipeline, optimizer, model, config)
        msg = str(ctx.exception)
        self.assertIn("_NoOptimizerPipeline", msg)
        self.assertIn("_optimizer", msg)

    def test_disabled_ga_with_no_optimizer_attr_does_not_raise(self) -> None:
        """The raise lives inside ``if config.is_enabled``: a disabled config keeps the old
        no-injection behavior for any pipeline."""
        pipeline = self._NoOptimizerPipeline()
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(num_steps=1)  # disabled
        GradientAccumulationWrapper(
            pipeline, optimizer, model, config
        )  # must not raise
        self.assertFalse(hasattr(pipeline, "_optimizer"))

    def test_double_wrapping_the_same_pipeline_raises(self) -> None:
        """Wrapping a pipeline twice would nest the optimizer gates, stepping once per
        K**2 micro-batches instead of per K -- silent under-stepping, not a crash."""
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)
        config = GradientAccumulationConfig(is_enabled=True, num_steps=4)

        GradientAccumulationWrapper(pipeline, optimizer, model, config)
        with self.assertRaises(RuntimeError) as ctx:
            GradientAccumulationWrapper(pipeline, optimizer, model, config)
        self.assertIn("already has a GA-wrapped", str(ctx.exception))

    def test_disabled_second_wrap_does_not_raise(self) -> None:
        """The guard sits inside ``if config.is_enabled``: a disabled second wrapper
        injects nothing, so it cannot nest anything."""
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)

        GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=True, num_steps=4),
        )
        first_wrapper = pipeline._optimizer
        GradientAccumulationWrapper(
            pipeline, optimizer, model, GradientAccumulationConfig(num_steps=1)
        )  # must not raise
        self.assertIs(pipeline._optimizer, first_wrapper)
