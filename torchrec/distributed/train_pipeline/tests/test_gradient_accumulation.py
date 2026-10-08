#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# These CPU tests use the default SGD implementation because optimizer behavior is under
# test; enabling foreach would change that behavior without improving test performance.
# @lint-ignore-every CITRINE
# @lint-ignore-every CITRINEAGENT missing_for_each_optimizer

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
    """Pipeline with training updates and forward-only evaluation batches."""

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
    """Tests evaluation through the same wrapper used for training."""

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
        model = nn.Linear(10, 5)
        wrapper = self._make_wrapper(model, num_steps=2)
        batch = torch.randn(4, 10)

        model.train()
        wrapper.progress(iter([batch]))
        wrapper.progress(iter([batch]))
        self.assertEqual(wrapper.current_step, 2)

        # Use three eval batches so accidental counter updates would change the boundary.
        model.eval()
        for _ in range(3):
            wrapper.progress(iter([batch]))
        self.assertEqual(
            wrapper.current_step,
            2,
            "evaluation advanced the training accumulation counter",
        )

        model.train()
        wrapper.progress(iter([batch]))
        wrapper.progress(iter([batch]))
        self.assertEqual(wrapper.current_step, 4)

    def test_eval_interlude_weight_parity_vs_no_eval(self) -> None:
        """Keep training weights unchanged relative to a run without evaluation."""
        torch.manual_seed(0)
        control = nn.Linear(10, 5)
        test = nn.Linear(10, 5)
        test.load_state_dict(control.state_dict())

        g = torch.Generator().manual_seed(123)
        train_batches = [torch.randn(4, 10, generator=g) for _ in range(8)]
        eval_batches = [torch.randn(4, 10, generator=g) for _ in range(3)]

        wc = self._make_wrapper(control, num_steps=2)
        control.train()
        for b in train_batches:
            wc.progress(iter([b]))

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

        for (name, pc), (_, pt) in zip(
            control.named_parameters(), test.named_parameters()
        ):
            self.assertTrue(
                torch.equal(pc, pt),
                f"evaluation changed the final training weight for {name}",
            )
        self.assertEqual(wt.current_step, 8)

    def test_eval_interlude_does_not_change_weights(self) -> None:
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
        """Do not apply pending training gradients when evaluation input ends."""
        model = nn.Linear(10, 5)
        wrapper = self._make_wrapper(model, num_steps=2)
        batch = torch.randn(4, 10)

        model.train()
        wrapper.progress(iter([batch]))
        self.assertTrue(wrapper._pending_uncommitted)

        flush_calls = [0]
        original_flush = wrapper._flush_accumulated_gradients

        def counting_flush(steps: int) -> bool:
            flush_calls[0] += 1
            return original_flush(steps)

        # pyrefly: ignore[bad-assignment]: monkeypatch for the test
        wrapper._flush_accumulated_gradients = counting_flush
        model.eval()
        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))
        self.assertEqual(
            flush_calls[0], 0, "evaluation applied pending training gradients"
        )

    def test_the_observer_is_training_gated_and_fires_before_the_pipeline(self) -> None:
        """Call the observer before training progress and skip it during evaluation."""
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=2, num_warmup_steps=1
        )
        events: list[str] = []

        def observer(*, should_step: bool, at_window_start: bool) -> None:
            events.append("observer")

        pipeline = _EvalAwareForwardPipeline(model, optimizer)
        inner_progress = pipeline.progress

        def recording_progress(dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
            events.append("pipeline")
            return inner_progress(dataloader_iter)

        # pyrefly: ignore[bad-assignment]: monkeypatch for the test
        pipeline.progress = recording_progress
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, model, config, window_observer=observer
        )
        batch = torch.randn(4, 10)

        model.train()
        wrapper.progress(iter([batch]))
        self.assertEqual(events, ["observer", "pipeline"])

        del events[:]
        model.eval()
        for _ in range(3):
            wrapper.progress(iter([batch]))
        self.assertEqual(
            events, ["pipeline"] * 3, "the observer fired during an eval interlude"
        )

        del events[:]
        model.train()
        wrapper.progress(iter([batch]))
        self.assertEqual(
            events,
            ["observer", "pipeline"],
            "the window boundary was published after the inner pipeline ran",
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

    def test_flush_partial_window_multi_rank_steps_under_default_policy(self) -> None:
        """STEP applies unsynchronized partial gradients and logs a warning."""
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
            with patch.object(
                wrapper.optimizer_wrapper._optimizer, "step"
            ) as mock_step, self.assertLogs(
                "torchrec.distributed.train_pipeline.gradient_accumulation", "WARNING"
            ) as logs:
                self.assertTrue(wrapper._flush_accumulated_gradients(2))
            mock_step.assert_called_once_with()
            self.assertIn("model replicas will diverge", "\n".join(logs.output))


class GAPartialWindowAbortTest(unittest.TestCase):
    """Tests process-group aborts before a multi-rank local error.

    ``RAISE`` aborts only with multiple ranks. ``STEP`` applies gradients without
    aborting, and a single process never needs an abort.
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
            with self.assertRaisesRegex(
                RuntimeError, "partial gradient-accumulation window"
            ):
                w._flush_accumulated_gradients(2)
            mock_abort.assert_called_once_with(None)

    def test_raw_flush_ws2_steps_and_warns_under_step(self) -> None:
        w = self._wrapper(PartialWindowPolicy.STEP)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            with patch.object(
                w.optimizer_wrapper._optimizer, "step"
            ) as mock_step, self.assertLogs(
                "torchrec.distributed.train_pipeline.gradient_accumulation", "WARNING"
            ) as logs:
                self.assertTrue(w._flush_accumulated_gradients(2))
            mock_step.assert_called_once_with()
            self.assertIn("model replicas will diverge", "\n".join(logs.output))
            mock_abort.assert_not_called()

    def test_raw_flush_ws1_raises_without_abort_under_raise(self) -> None:
        w = self._wrapper(PartialWindowPolicy.RAISE)
        with patch(self._ABORT) as mock_abort, self._force_world_size(1):
            with self.assertRaisesRegex(RuntimeError, "PartialWindowPolicy.RAISE"):
                w._flush_accumulated_gradients(2)
            mock_abort.assert_not_called()

    def test_raw_flush_ws1_steps_without_abort_under_step(self) -> None:
        w = self._wrapper(PartialWindowPolicy.STEP)
        with patch(self._ABORT) as mock_abort, self._force_world_size(1):
            self.assertTrue(w._flush_accumulated_gradients(2))
            mock_abort.assert_not_called()

    def test_full_window_never_aborts(self) -> None:
        w = self._wrapper(PartialWindowPolicy.RAISE)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            self.assertFalse(w._flush_accumulated_gradients(4))
            mock_abort.assert_not_called()


class GAPartialWindowDiscardTest(unittest.TestCase):
    """Tests that ``DISCARD`` clears a partial window without stepping."""

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
        """Accumulate part of a four-batch window and verify pending gradients."""
        for _ in range(micros):
            wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertFalse(
            wrapper.optimizer_wrapper._needs_zero_grad,
            "mid-window zero_grad must remain disabled",
        )
        self.assertTrue(
            any(
                p.grad is not None and torch.count_nonzero(p.grad) > 0
                for p in model.parameters()
            ),
            "expected nonzero gradients before discarding the window",
        )

    @contextlib.contextmanager
    def _force_world_size(self, world_size: int) -> Iterator[None]:
        with patch("torch.distributed.is_available", return_value=True), patch(
            "torch.distributed.is_initialized", return_value=True
        ), patch("torch.distributed.get_world_size", return_value=world_size):
            yield

    def test_discard_actually_zeroes_the_gradients(self) -> None:
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)

        self.assertFalse(wrapper._flush_accumulated_gradients(2))

        for name, p in model.named_parameters():
            self.assertTrue(
                p.grad is None or torch.count_nonzero(p.grad) == 0,
                f"discarded gradients survived on {name}",
            )

    def test_discard_takes_no_optimizer_step(self) -> None:
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
                torch.equal(b, p), "discarding the window changed model weights"
            )

    def test_discard_never_aborts_process_groups_at_ws2(self) -> None:
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.RAISE)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            with self.assertRaisesRegex(
                RuntimeError, "partial gradient-accumulation window"
            ):
                wrapper._flush_accumulated_gradients(2)
            mock_abort.assert_called_once_with(None)

        wrapper, model, _optimizer = self._make(PartialWindowPolicy.STEP)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            self.assertTrue(wrapper._flush_accumulated_gradients(2))
            mock_abort.assert_not_called()

        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT) as mock_abort, self._force_world_size(2):
            self.assertFalse(wrapper._flush_accumulated_gradients(2))
            mock_abort.assert_not_called()

    def test_discard_does_not_rewind_the_counter_or_replay_warmup(self) -> None:
        """Keep the global counter and warmup state after discarding."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        step_before = wrapper.optimizer_wrapper._current_step
        base_before = wrapper.optimizer_wrapper._window_base

        wrapper._flush_accumulated_gradients(2)

        self.assertEqual(wrapper.optimizer_wrapper._current_step, step_before)
        self.assertEqual(wrapper.optimizer_wrapper._window_base, base_before)

    def test_discard_does_not_realign_the_window_itself(self) -> None:
        """Leave window realignment to ``progress()`` after the flush."""
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
        """Preserve bucket-view aliases when discarding gradients."""
        wrapper, model, _optimizer = self._make(
            PartialWindowPolicy.DISCARD, accumulate_into_buckets=True
        )
        self._accumulate_partial_window(wrapper, model)

        with patch.object(wrapper, "_window_start_zero_grad") as mock_window_zero:
            wrapper._flush_accumulated_gradients(2)
            mock_window_zero.assert_called_once_with()

    def test_step_routes_through_the_wrapper_to_preserve_bucket_views(self) -> None:
        """Preserve bucket-view aliases after applying a partial window."""
        wrapper, model, _optimizer = self._make(
            PartialWindowPolicy.STEP, accumulate_into_buckets=True
        )
        self._accumulate_partial_window(wrapper, model)

        with patch.object(
            wrapper, "_window_start_zero_grad"
        ) as mock_window_zero, patch.object(
            wrapper.optimizer_wrapper._optimizer, "zero_grad"
        ) as mock_raw_zero:
            self.assertTrue(wrapper._flush_accumulated_gradients(2))
            mock_window_zero.assert_called_once_with()
            mock_raw_zero.assert_not_called()

    def test_discard_is_inert_on_a_complete_window(self) -> None:
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
        """Apply an explicit final batch even when exhaustion uses ``DISCARD``."""
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.DISCARD)
        self._accumulate_partial_window(wrapper, model)
        before = [p.detach().clone() for p in model.parameters()]

        wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)

        moved = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
        self.assertTrue(
            moved,
            "the explicit final batch did not apply the partial window",
        )
        self.assertFalse(
            wrapper._pending_uncommitted,
            "the explicit final batch left pending gradients",
        )

    def test_explicit_is_last_batch_still_raises_under_raise(self) -> None:
        wrapper, model, _optimizer = self._make(PartialWindowPolicy.RAISE)
        self._accumulate_partial_window(wrapper, model)
        with patch(self._ABORT):
            with self.assertRaisesRegex(RuntimeError, "PartialWindowPolicy.RAISE"):
                wrapper.progress(iter([torch.randn(2, 10)]), is_last_batch=True)

    def test_discard_end_to_end_through_stop_iteration(self) -> None:
        """Discard pending gradients when the iterator is exhausted."""
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
    def test_optimizer_not_injected_when_disabled(self) -> None:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        disabled_config = GradientAccumulationConfig(num_steps=1)

        pipeline = _RealForwardPipeline(model, optimizer)
        original_optimizer = pipeline._optimizer

        GradientAccumulationWrapper(pipeline, optimizer, model, disabled_config)

        self.assertIs(pipeline._optimizer, original_optimizer)
        self.assertNotIsInstance(pipeline._optimizer, _GAOptimizerWrapper)

    def test_optimizer_injected_when_enabled(self) -> None:
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
        """Do not call the observer when accumulation is disabled."""
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
        self.assertFalse(hasattr(pipeline, "_ga_should_step"))
        self.assertFalse(hasattr(pipeline, "_ga_at_window_start"))

    def test_enabled_ga_signals_observer_before_inner_progress(self) -> None:
        """Publish the boundary before the inner pipeline runs."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)
        seen_at_progress: list[tuple[bool, bool]] = []
        state: dict[str, bool] = {"should_step": False, "at_window_start": False}

        def observer(*, should_step: bool, at_window_start: bool) -> None:
            state["should_step"] = should_step
            state["at_window_start"] = at_window_start

        inner_progress = pipeline.progress

        def recording_progress(dataloader_iter: Iterator[Any]) -> float:
            seen_at_progress.append((state["should_step"], state["at_window_start"]))
            return inner_progress(dataloader_iter)

        # pyrefly: ignore[bad-assignment]: test double rebinds the bound method.
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

        # With two batches per window, starts are 0 and 2 and boundaries are 1 and 3.
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

        # Resetting an open window requires explicit permission to drop its gradients.
        ga.reset(drop_partial=True)
        self.assertEqual(ga.current_step, 0)
        self.assertEqual(ga.optimizer_wrapper._current_step, 0)

    def test_dropped_partial_window_grads_do_not_reach_the_next_window(self) -> None:
        """A dropped partial window's gradients are gone before the next one accumulates."""
        model = nn.Linear(10, 5, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(
            _EvalAwareForwardPipeline(model, optimizer), optimizer, model, config
        )
        model.train()

        batch = torch.ones(1, 10)
        for _ in range(3):
            ga.progress(iter([batch]))
        self.assertIsNotNone(model.weight.grad)
        dropped = model.weight.grad.clone()
        self.assertTrue(torch.any(dropped != 0))

        ga.reset(drop_partial=True)
        ga.progress(iter([batch]))

        # The next window start clears the gradient while preserving its bucket view.
        self.assertIsNotNone(model.weight.grad)
        torch.testing.assert_close(model.weight.grad, dropped / 3)

    def test_reset_raises_on_open_partial_window(self) -> None:
        """Require ``drop_partial`` when resetting an open window."""
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=8)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(pipeline, optimizer, model, config)

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(3):
            ga.progress(dummy_iter)
        self.assertTrue(ga._pending_uncommitted)
        with self.assertRaises(RuntimeError):
            ga.reset()

        # Reset is allowed after the fourth batch completes the window.
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
    """Tracks ``no_sync`` calls for a mock distributed data-parallel module."""

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
    """Model with a registered distributed data-parallel submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.dense_layer = torch.nn.Linear(10, 5)
        self.inner_ddp = _MockDDPModule()


class NestedDDPNoSyncTest(unittest.TestCase):
    """Tests ``no_sync`` propagation to registered nested modules."""

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
        """Wrap a model whose nested module provides ``no_sync``."""
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
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp()
        wrapper.set_step(1)  # non-boundary, non-warmup → should use no_sync
        self.assertFalse(wrapper._should_sync_grad())

        with wrapper._get_no_sync_context():
            self.assertEqual(inner_ddp.no_sync_entered, 1)
        self.assertEqual(inner_ddp.no_sync_exited, 1)

    def test_dmp_wrapped_non_module_with_no_sync(self) -> None:
        """Use ``no_sync`` from a wrapped object that is not an ``nn.Module``."""

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
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp(
            num_steps=4, num_warmup_steps=4, num_batches=4
        )

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(4):
            wrapper.progress(dummy_iter)

        self.assertEqual(inner_ddp.no_sync_entered, 0)

    def test_both_outer_and_inner_ddp_get_no_sync(self) -> None:
        model = _ModelWithNestedDDP()
        inner_ddp = model.inner_ddp

        # Add an outer wrapper around the model with the nested module.
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
        _, inner_ddp, wrapper = self._make_wrapper_with_nested_ddp()
        wrapper.set_step(1)

        with self.assertRaises(RuntimeError):
            with wrapper._get_no_sync_context():
                self.assertEqual(inner_ddp.no_sync_entered, 1)
                raise RuntimeError("test error")

        self.assertEqual(inner_ddp.no_sync_exited, 1)


class _SplitOrDefaultModePipeline(TrainPipeline[Any, torch.Tensor]):
    """Models optimizer steps through either the wrapper or a child optimizer.

    The child path uses the boundary observer because it bypasses the wrapper's
    ``step()`` method. Recorded gradients expose failures to clear between windows.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        bypass_wrapper_step: bool,
    ) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._model = model
        # The wrapper replaces ``_optimizer``; the split path retains the child optimizer.
        self._optimizer = optimizer
        self._child_optimizer = optimizer
        self._bypass_wrapper_step = bypass_wrapper_step
        self.post_backward_grads: list[torch.Tensor] = []
        self.child_step_count: int = 0
        # The observer updates these values before each inner pipeline call.
        self._ga_should_step: bool = True
        self._ga_at_window_start: bool = True

    def set_ga_window_state(self, *, should_step: bool, at_window_start: bool) -> None:
        self._ga_should_step = should_step
        self._ga_at_window_start = at_window_start

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        self._optimizer.zero_grad()
        out = self._model(batch)
        loss = out.sum()
        loss.backward()
        weight = cast(torch.Tensor, self._model.weight)
        assert weight.grad is not None
        self.post_backward_grads.append(weight.grad.detach().clone())
        if self._bypass_wrapper_step:
            if self._ga_should_step:
                self.child_step_count += 1
                self._child_optimizer.step()
        else:
            self._optimizer.step()
        return loss


class SplitModeCrossWindowGradTest(unittest.TestCase):
    """Tests gradient clearing when a split pipeline steps a child optimizer.

    Constant inputs and a zero learning rate make accumulated gradients follow
    ``[1g, 2g, 1g, 2g]`` across two two-batch windows.
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
        pipeline = self._run(bypass_wrapper_step=False)
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 4)
        self.assertTrue(
            torch.allclose(g[2], g[0]),
            f"default-mode window-2-start grad {g[2].flatten().tolist()} != "
            f"window-1-start grad {g[0].flatten().tolist()}",
        )
        self.assertTrue(torch.allclose(g[1], 2.0 * g[0]))
        self.assertTrue(torch.allclose(g[3], 2.0 * g[2]))

    def test_split_mode_zeros_grad_each_window(self) -> None:
        """Clear gradients when the child optimizer bypasses the wrapper's step."""
        pipeline = self._run(bypass_wrapper_step=True)
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 4)
        self.assertTrue(
            torch.allclose(g[2], g[0]),
            "the split path retained gradients across windows: second-window gradient "
            f"{g[2].flatten().tolist()} != window-1-start grad "
            f"{g[0].flatten().tolist()}",
        )
        self.assertTrue(torch.allclose(g[3], 2.0 * g[2]))

    def test_split_mode_partial_window_commits_in_band(self) -> None:
        """Apply an explicit partial window through the child optimizer."""
        pipeline = self._run(
            bypass_wrapper_step=True, k=2, num_micros=3, mark_last=True
        )
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 3)
        self.assertTrue(torch.allclose(g[0], torch.ones_like(g[0])))
        self.assertTrue(torch.allclose(g[1], 2.0 * torch.ones_like(g[1])))
        self.assertTrue(torch.allclose(g[2], torch.ones_like(g[2])))
        self.assertEqual(pipeline.child_step_count, 2)

    def test_default_mode_partial_window_commits_in_band(self) -> None:
        """Apply an explicit partial window through the wrapped optimizer."""
        pipeline = self._run(
            bypass_wrapper_step=False, k=2, num_micros=3, mark_last=True
        )
        g = pipeline.post_backward_grads
        self.assertEqual(len(g), 3)
        self.assertTrue(torch.allclose(g[0], torch.ones_like(g[0])))
        self.assertTrue(torch.allclose(g[1], 2.0 * torch.ones_like(g[1])))
        self.assertTrue(torch.allclose(g[2], torch.ones_like(g[2])))

    def test_partial_window_trailing_stop_iteration_no_double_step(self) -> None:
        """Do not apply an explicit partial window again on later exhaustion."""
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
            wrapper.progress(data)
            wrapper.progress(data)
            wrapper.progress(data, is_last_batch=True)
            self.assertFalse(
                wrapper._pending_uncommitted,
                f"explicit partial window remained pending (bypass={bypass})",
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
                wrapper.progress(iter([]))
            self.assertEqual(
                flush_calls[0],
                0,
                f"exhaustion applied the explicit partial window twice (bypass={bypass})",
            )


class _IteratorDrivenStepPipeline(TrainPipeline[Any, int]):
    """Consumes the provided iterator and steps the wrapped optimizer."""

    def __init__(self) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._optimizer = MagicMock()
        self._ga_at_window_start: bool = False
        self._ga_should_step: bool = False

    def set_ga_window_state(self, *, should_step: bool, at_window_start: bool) -> None:
        self._ga_should_step = should_step
        self._ga_at_window_start = at_window_start

    def progress(self, dataloader_iter: Iterator[Any]) -> int:
        value = next(dataloader_iter)
        self._optimizer.step()
        return value


class WindowRealignmentTest(unittest.TestCase):
    """Tests window realignment after an iterator is exhausted.

    A new iterator starts a fresh window while ``current_step`` remains a global,
    monotonic micro-batch count.
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
        """Consume ``n`` micro-batches and exhaust the iterator."""
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
        """Move step, synchronization, and window-start signals together."""
        wrapper, pipeline, model, optimizer = self._make()

        # Six batches leave half of a four-batch window open.
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper.micro_batches_into_window, 0)
        self.assertEqual(wrapper.current_step, 6)

        optimizer.step.reset_mock()
        sync_before = model.no_sync_entered

        stepped: list[bool] = []
        synced: list[bool] = []
        at_start: list[bool] = []
        it: Iterator[int] = iter(range(self.K))
        for _ in range(self.K):
            n_steps_before = optimizer.step.call_count
            no_sync_before = model.no_sync_entered
            wrapper.progress(it)
            stepped.append(optimizer.step.call_count > n_steps_before)
            # A batch synchronizes when ``no_sync`` is not entered.
            synced.append(model.no_sync_entered == no_sync_before)
            at_start.append(pipeline._ga_at_window_start)

        self.assertEqual(
            stepped, [False, False, False, True], "optimizer-step boundary"
        )
        self.assertEqual(synced, [False, False, False, True], "DDP grad-sync boundary")
        self.assertEqual(at_start, [True, False, False, False], "window-start signal")
        self.assertGreater(model.no_sync_entered, sync_before)

    def test_partial_flush_after_realignment_uses_window_relative_count(self) -> None:
        """Flush based on the current window, not the global batch count."""
        wrapper, _pipeline, _model, optimizer = self._make()

        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper.micro_batches_into_window, 0)
        self.assertEqual(wrapper.current_step, 6)

        optimizer.step.reset_mock()

        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it)
        self.assertEqual(wrapper.current_step, 8)
        self.assertEqual(wrapper.micro_batches_into_window, 2)
        self.assertEqual(
            optimizer.step.call_count, 0, "no boundary reached inside a partial window"
        )

        with self.assertRaises(StopIteration):
            wrapper.progress(it)

        self.assertEqual(
            optimizer.step.call_count,
            1,
            "exhaustion did not apply the two pending batches",
        )

    def test_current_step_stays_monotonic_across_exhaustion(self) -> None:
        """Keep ``current_step`` monotonic across iterator exhaustion."""
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
        """Do not repeat warmup after starting a new iterator."""
        wrapper, _pipeline, model, _optimizer = self._make(num_warmup_steps=3)
        self._exhaust(wrapper, 6)

        before = model.no_sync_entered
        it: Iterator[int] = iter(range(1))
        wrapper.progress(it)
        self.assertEqual(
            model.no_sync_entered,
            before + 1,
            "the new iterator repeated gradient-synchronization warmup",
        )

    def test_eval_exhaustion_does_not_realign(self) -> None:
        """Keep an open training window unchanged when evaluation input ends."""
        wrapper, _pipeline, model, _optimizer = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it)
        self.assertEqual(wrapper.micro_batches_into_window, 2)

        model.eval()
        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))
        self.assertEqual(
            wrapper.micro_batches_into_window,
            2,
            "an eval StopIteration re-anchored the window and dropped the open training window",
        )
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)

    def test_bucket_views_ready_survives_realignment(self) -> None:
        """Keep bucket-view readiness when the window is realigned."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        wrapper._bucket_views_ready = True
        self._exhaust(wrapper, 6)
        self.assertTrue(wrapper._bucket_views_ready)

    def test_set_step_after_realignment_uses_raw_step_semantics(self) -> None:
        """Make ``set_step`` relative to the start of the accumulation cycle."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 6)

        wrapper.set_step(3)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)
        self.assertEqual(wrapper.micro_batches_into_window, 3)
        self.assertTrue(wrapper.optimizer_wrapper._should_step())

    def test_reset_clears_the_window_anchor(self) -> None:
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 6)
        wrapper.reset()
        self.assertEqual(wrapper._optimizer_wrapper._window_base, 0)
        self.assertEqual(wrapper.current_step, 0)
        self.assertEqual(wrapper.micro_batches_into_window, 0)

    def test_clean_boundary_exhaustion_also_realigns(self) -> None:
        """Realign after exhaustion at a complete window boundary."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, self.K)
        self.assertEqual(wrapper.current_step, self.K)
        self.assertEqual(wrapper._optimizer_wrapper._window_base, self.K)
        self.assertEqual(wrapper.micro_batches_into_window, 0)

    def test_is_last_batch_realigns_after_advancing(self) -> None:
        """Start a fresh window after an explicit final batch."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.progress(it, is_last_batch=True)
        self.assertEqual(wrapper.current_step, 2)
        self.assertEqual(wrapper.micro_batches_into_window, 0)

    def test_micro_batches_into_window_stays_in_range_across_orderings(self) -> None:
        """Keep the within-window counter in ``[0, K)`` across state changes."""
        wrapper, _pipeline, _model, _optimizer = self._make()
        self._exhaust(wrapper, 6)
        self._assert_in_window_range(wrapper)
        wrapper.set_step(2)
        self._assert_in_window_range(wrapper)
        self._exhaust(wrapper, 3)
        self._assert_in_window_range(wrapper)
        wrapper.reset()
        self._assert_in_window_range(wrapper)

    def _assert_in_window_range(self, wrapper: Any) -> None:
        value = wrapper.micro_batches_into_window
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, self.K)


class UnsupportedPipelineTest(unittest.TestCase):
    """Reject enabled wrappers when the pipeline optimizer cannot be replaced."""

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
        pipeline = self._NoOptimizerPipeline()
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(num_steps=1)  # disabled
        GradientAccumulationWrapper(
            pipeline, optimizer, model, config
        )  # must not raise
        self.assertFalse(hasattr(pipeline, "_optimizer"))

    def test_double_wrapping_the_same_pipeline_raises(self) -> None:
        model = _MockModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=4)
        config = GradientAccumulationConfig(is_enabled=True, num_steps=4)

        GradientAccumulationWrapper(pipeline, optimizer, model, config)
        with self.assertRaises(RuntimeError) as ctx:
            GradientAccumulationWrapper(pipeline, optimizer, model, config)
        self.assertIn(
            "already uses a gradient-accumulation optimizer", str(ctx.exception)
        )

    def test_disabled_second_wrap_does_not_raise(self) -> None:
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


class LiveKSingleSourceTest(unittest.TestCase):
    """Use the wrapper's stored window size for every boundary calculation."""

    K: int = 4

    def test_mutating_the_callers_config_does_not_move_the_boundary(self) -> None:
        model = _MockModel()
        pipeline = _IteratorDrivenStepPipeline()
        config = GradientAccumulationConfig(is_enabled=True, num_steps=self.K)
        wrapper: GradientAccumulationWrapper[Any, int] = GradientAccumulationWrapper(
            pipeline, MagicMock(spec=torch.optim.Optimizer), model, config
        )
        model.train()

        config.num_steps = 2
        self.assertEqual(self.K, wrapper.num_micro_batches_per_step)
