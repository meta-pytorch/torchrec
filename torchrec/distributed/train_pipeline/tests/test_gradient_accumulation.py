#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# These CPU tests use the default SGD implementation because optimizer behavior is under
# test; enabling foreach would change that behavior without improving test performance.
# @lint-ignore-every CITRINE
# @lint-ignore-every CITRINEAGENT missing_for_each_optimizer
# @lint-ignore-every CITRINECI missing_for_each_optimizer

import contextlib
import tempfile
import unittest
from typing import Any, Callable, cast, Iterator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    _fold_targets,
    _GAOptimizerWrapper,
    GAStepGradScale,
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
    PartialWindowPolicy,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline
from torchrec.optim.clipping import GradientClipping, GradientClippingOptimizer
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
)


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
        # Clear before backward so each window keeps its first micro-batch's gradient.
        self._optimizer.zero_grad()
        output = self._model(batch)
        loss = output.sum()
        loss.backward()
        self._optimizer.step()
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


class FsdpSyncSuppressionSeamTest(unittest.TestCase):
    """Test FSDP2 flag ordering and cleanup in GradientAccumulationWrapper."""

    def _make(
        self, policy: PartialWindowPolicy = PartialWindowPolicy.STEP
    ) -> GradientAccumulationWrapper:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        return GradientAccumulationWrapper(
            _RealForwardPipeline(model, optimizer),
            optimizer,
            model,
            config,
            partial_window_policy=policy,
        )

    @contextlib.contextmanager
    def _record(
        self, wrapper: GradientAccumulationWrapper, calls: List[str]
    ) -> Iterator[None]:
        """Record which suppression method the wrapper calls, in order."""
        suppression = wrapper._fsdp_sync_suppression
        with contextlib.ExitStack() as stack:
            for name in ("suppress", "restore", "clear_partial", "ensure_sharded"):
                stack.enter_context(
                    patch.object(
                        suppression,
                        name,
                        side_effect=lambda _targets, name=name: calls.append(name),
                    )
                )
            yield

    def test_restore_precedes_the_syncing_micro_batch(self) -> None:
        # Check when restore() runs; ensure_sharded() can mask late resharding.
        wrapper = self._make()
        calls: List[str] = []
        with self._record(wrapper, calls):
            with patch.object(
                wrapper._pipeline,
                "progress",
                side_effect=lambda _it: calls.append("progress") or torch.zeros(1),
            ):
                for _ in range(4):
                    wrapper.progress(iter([torch.randn(2, 10)]))
        # Native FSDP2 synchronizes only the step microbatch, in the first window too.
        self.assertEqual(
            calls,
            [
                "suppress",
                "progress",
                "suppress",
                "progress",
                "suppress",
                "progress",
                "restore",
                "progress",
            ],
        )

    def test_ensure_sharded_runs_before_the_optimizer_step(self) -> None:
        wrapper = self._make()
        calls: List[str] = []
        with self._record(wrapper, calls):
            with patch.object(
                wrapper.optimizer_wrapper._optimizer,
                "step",
                side_effect=lambda *a, **k: calls.append("optimizer.step"),
            ):
                for _ in range(4):
                    wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertIn("ensure_sharded", calls)
        self.assertLess(calls.index("ensure_sharded"), calls.index("optimizer.step"))

    def test_suppression_follows_the_optimizer_step_boundaries(self) -> None:
        wrapper = self._make()
        calls: List[str] = []
        with self._record(wrapper, calls):
            for _ in range(8):
                wrapper.progress(iter([torch.randn(2, 10)]))
        # Compare synchronization flags independently of optimizer validation.
        flags = [c for c in calls if c in ("suppress", "restore")]
        # Every window, the first included, synchronizes only its final microbatch.
        self.assertEqual(
            flags,
            [
                "suppress",
                "suppress",
                "suppress",
                "restore",
                "suppress",
                "suppress",
                "suppress",
                "restore",
            ],
        )

    def test_a_restarted_schedule_synchronizes_only_its_step_micro_batch(self) -> None:
        # A compile warmup's reset() must not make FSDP2 all-reduce microbatch 0 again.
        for api in ("reset", "set_step"):
            with self.subTest(api=api):
                wrapper = self._make()
                for _ in range(4):
                    wrapper.progress(iter([torch.randn(2, 10)]))
                if api == "reset":
                    wrapper.reset()
                else:
                    wrapper.set_step(0)
                calls: List[str] = []
                with self._record(wrapper, calls):
                    for _ in range(4):
                        wrapper.progress(iter([torch.randn(2, 10)]))
                self.assertEqual(
                    [c for c in calls if c in ("suppress", "restore")],
                    ["suppress", "suppress", "suppress", "restore"],
                )

    def test_no_sync_warmup_does_not_synchronize_fsdp2(self) -> None:
        cases = {
            4: ["suppress", "suppress", "suppress", "restore"],
            # Warmup spans two windows.
            2: ["suppress", "restore", "suppress", "restore"],
        }
        for num_steps, expected in cases.items():
            with self.subTest(num_steps=num_steps):
                model = _MockModel()
                wrapper = GradientAccumulationWrapper(
                    _MockPipeline(num_batches=4),
                    optim.SGD(model.parameters(), lr=0.01),
                    model,
                    GradientAccumulationConfig(
                        is_enabled=True, num_steps=num_steps, num_warmup_steps=3
                    ),
                )
                calls: List[str] = []
                with self._record(wrapper, calls):
                    for _ in range(4):
                        wrapper.progress(iter([]))
                self.assertEqual(model.no_sync_entered, 0)
                self.assertEqual(
                    [c for c in calls if c in ("suppress", "restore")], expected
                )

    def test_k1_synchronizes_every_micro_batch(self) -> None:
        # A compile warmup may run at K=1 on a wrapper built for a larger K.
        wrapper = self._make()
        wrapper.set_num_micro_batches_per_step(1)
        calls: List[str] = []
        with self._record(wrapper, calls):
            for _ in range(2):
                wrapper.progress(iter([torch.randn(2, 10)]))
            wrapper.reset()
            for _ in range(2):
                wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(
            [c for c in calls if c in ("suppress", "restore")], ["restore"] * 4
        )

    def test_disabled_accumulation_leaves_fsdp_state_alone(self) -> None:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        wrapper = GradientAccumulationWrapper(
            _RealForwardPipeline(model, optimizer),
            optimizer,
            model,
            GradientAccumulationConfig(),
        )
        calls: List[str] = []
        with self._record(wrapper, calls):
            for _ in range(2):
                wrapper.progress(iter([torch.randn(2, 10)]))
            wrapper.reset()
            wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(calls, [])

    def test_exhaustion_after_the_first_micro_batch_drops_its_fsdp_partial(
        self,
    ) -> None:
        wrapper = self._make(PartialWindowPolicy.STEP)
        calls: List[str] = []
        with self._record(wrapper, calls), patch.object(
            wrapper._fsdp_sync_suppression, "has_partial", return_value=True
        ), self.assertLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", "WARNING"
        ):
            wrapper.progress(iter([torch.randn(2, 10)]))
            with self.assertRaises(StopIteration):
                wrapper.progress(iter([]))
        self.assertEqual(
            calls,
            [
                "suppress",
                "suppress",
                "restore",
                "ensure_sharded",
                "clear_partial",
                "restore",
                "ensure_sharded",
            ],
        )

    def test_the_targets_are_collected_once(self) -> None:
        wrapper = self._make()
        with patch.object(
            wrapper._fsdp_sync_suppression, "collect", return_value=[]
        ) as collect:
            for _ in range(4):
                wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(collect.call_count, 1)

    def test_attach_drops_the_cached_targets(self) -> None:
        wrapper = self._make()
        wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertIsNotNone(wrapper._cached_fsdp_targets)

        with patch.object(wrapper._pipeline, "attach", create=True, return_value=None):
            wrapper.attach(wrapper._model)

        self.assertIsNone(wrapper._cached_fsdp_targets)

    def test_discard_clears_the_pending_reduction(self) -> None:
        wrapper = self._make(PartialWindowPolicy.DISCARD)
        for _ in range(2):
            wrapper.progress(iter([torch.randn(2, 10)]))
        calls: List[str] = []
        with self._record(wrapper, calls):
            self.assertFalse(wrapper._flush_accumulated_gradients(2))
        self.assertEqual(calls, ["clear_partial"])

    def test_step_partial_window_restores_and_reshards_before_stepping(self) -> None:
        wrapper = self._make(PartialWindowPolicy.STEP)
        for _ in range(2):
            wrapper.progress(iter([torch.randn(2, 10)]))
        calls: List[str] = []
        with self._record(wrapper, calls):
            with patch.object(
                wrapper.optimizer_wrapper._optimizer,
                "step",
                side_effect=lambda *a, **k: calls.append("optimizer.step"),
            ):
                self.assertTrue(wrapper._flush_accumulated_gradients(2))
        self.assertEqual(calls[:2], ["restore", "ensure_sharded"])
        self.assertLess(calls.index("ensure_sharded"), calls.index("optimizer.step"))

    def test_step_drops_a_pending_fsdp_partial_with_a_warning(self) -> None:
        wrapper = self._make(PartialWindowPolicy.STEP)
        for _ in range(2):
            wrapper.progress(iter([torch.randn(2, 10)]))
        calls: List[str] = []
        with self._record(wrapper, calls), patch.object(
            wrapper._fsdp_sync_suppression, "has_partial", return_value=True
        ), self.assertLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", "WARNING"
        ) as logs:
            self.assertTrue(wrapper._flush_accumulated_gradients(2))
        self.assertEqual(calls[:3], ["clear_partial", "restore", "ensure_sharded"])
        self.assertIn("Dropping the native-FSDP2 gradients", "\n".join(logs.output))

    def test_every_discard_api_drops_the_pending_reduction(self) -> None:
        # zero_grad() leaves FSDP2 reduction buffers intact.
        # Every discard API must reset them before the next window.
        for api in ("reset", "set_step", "set_k"):
            with self.subTest(api=api):
                wrapper = self._make()
                wrapper.progress(iter([torch.randn(2, 10)]))
                calls: List[str] = []
                with self._record(wrapper, calls):
                    if api == "reset":
                        wrapper.reset(drop_partial=True)
                    elif api == "set_step":
                        wrapper.set_step(0, drop_partial=True)
                    else:
                        wrapper.set_num_micro_batches_per_step(
                            2, allow_open_window=True
                        )
                self.assertIn("clear_partial", calls)

    def test_reset_touches_fsdp_state_only_with_an_open_window(self) -> None:
        completed = self._make()
        for _ in range(4):
            completed.progress(iter([torch.randn(2, 10)]))
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        disabled = GradientAccumulationWrapper(
            _RealForwardPipeline(model, optimizer),
            optimizer,
            model,
            GradientAccumulationConfig(),
        )
        for name, wrapper in (("completed window", completed), ("disabled", disabled)):
            with self.subTest(name):
                calls: List[str] = []
                with self._record(wrapper, calls):
                    wrapper.reset()
                self.assertNotIn("clear_partial", calls)

    def test_a_failed_micro_batch_resets_fsdp_state(self) -> None:
        wrapper = self._make()
        calls: List[str] = []
        with self._record(wrapper, calls):
            with patch.object(
                wrapper._pipeline, "progress", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertIn("clear_partial", calls)
        self.assertIn("restore", calls)
        # Restarting a failed window must discard dense and FSDP2 gradients together.
        self.assertFalse(wrapper._pending_uncommitted)

    def test_a_failed_window_does_not_leak_dense_gradients(self) -> None:
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        wrapper = GradientAccumulationWrapper(
            _RealForwardPipeline(model, optimizer),
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=True, num_steps=4),
        )
        batch = torch.ones(2, 10)
        wrapper.progress(iter([batch]))
        self.assertIsNotNone(model.weight.grad)
        one_micro_batch = cast(torch.Tensor, model.weight.grad).clone()
        wrapper.progress(iter([batch]))
        torch.testing.assert_close(model.weight.grad, 2 * one_micro_batch)
        with patch.object(
            wrapper._pipeline, "progress", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                wrapper.progress(iter([batch]))
        wrapper.progress(iter([batch]))
        # The retry must contain only gradients from its new microbatch.
        torch.testing.assert_close(model.weight.grad, one_micro_batch)

    def test_eval_batch_restores_fsdp_flags(self) -> None:
        wrapper = self._make()
        wrapper.set_step(4)
        wrapper._model.eval()
        calls: List[str] = []
        with self._record(wrapper, calls), patch.object(
            wrapper._pipeline, "progress", return_value=torch.zeros(1)
        ):
            wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertEqual(
            [c for c in calls if c in ("suppress", "restore")], ["restore"]
        )

    def test_eval_exhaustion_restores_fsdp_flags(self) -> None:
        wrapper = self._make()
        wrapper.set_step(4)
        wrapper._model.eval()
        calls: List[str] = []
        with self._record(wrapper, calls):
            with self.assertRaises(StopIteration):
                wrapper.progress(iter([]))
        self.assertEqual(
            [c for c in calls if c in ("suppress", "restore")], ["restore"]
        )

    def test_exhaustion_unsuppresses_the_window(self) -> None:
        # Restore flags at exhaustion before evaluation or checkpointing reads parameters.
        wrapper = self._make(PartialWindowPolicy.DISCARD)
        wrapper.progress(iter([torch.randn(2, 10)]))
        calls: List[str] = []
        with self._record(wrapper, calls):
            with self.assertRaises(StopIteration):
                wrapper.progress(iter([]))
        self.assertIn("restore", calls)
        self.assertLess(calls.index("restore"), calls.index("clear_partial"))

    def test_cleanup_failure_does_not_mask_the_pipeline_error(self) -> None:
        wrapper = self._make()
        with patch.object(
            wrapper, "_discard_fsdp_window", side_effect=ValueError("cleanup")
        ):
            with patch.object(
                wrapper._pipeline, "progress", side_effect=RuntimeError("boom")
            ):
                with self.assertRaisesRegex(RuntimeError, "boom") as caught:
                    wrapper.progress(iter([torch.randn(2, 10)]))
        self.assertIsInstance(caught.exception.__cause__, ValueError)


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

    def test_set_step_refuses_an_open_partial_window(self) -> None:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        ga = GradientAccumulationWrapper(
            _MockPipeline(num_batches=8), optimizer, model, config
        )

        dummy_iter: Iterator[Any] = iter([])
        for _ in range(3):
            ga.progress(dummy_iter)
        self.assertTrue(ga._pending_uncommitted)

        with self.assertRaises(RuntimeError):
            ga.set_step(0)

        # drop_partial clears this window's gradients before the next window.
        ga.set_step(0, drop_partial=True)
        self.assertFalse(ga._pending_uncommitted)
        self.assertTrue(ga._optimizer_wrapper._needs_zero_grad)
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


class OptimizerStepReportingTest(unittest.TestCase):
    """Tests optimizer-step status for each reader batch."""

    K: int = 4

    def _make(
        self, k: Optional[int] = None, is_enabled: bool = True
    ) -> tuple[GradientAccumulationWrapper[Any, int], _MockModel]:
        model = _MockModel()
        pipeline = _IteratorDrivenStepPipeline()
        config = GradientAccumulationConfig(
            is_enabled=is_enabled, num_steps=k if k is not None else self.K
        )
        wrapper: GradientAccumulationWrapper[Any, int] = GradientAccumulationWrapper(
            pipeline, MagicMock(spec=torch.optim.Optimizer), model, config
        )
        model.train()
        return wrapper, model

    def test_completion_is_reported_only_on_the_boundary_micro(self) -> None:
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(2 * self.K))
        completed = []
        for _ in range(2 * self.K):
            wrapper.progress(it)
            completed.append(wrapper.optimizer_step_completed)
        self.assertEqual(
            [False, False, False, True, False, False, False, True], completed
        )

    def test_prediction_matches_the_outcome_it_predicts(self) -> None:
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(2 * self.K))
        for i in range(2 * self.K):
            predicted = wrapper.will_complete_optimizer_step
            wrapper.progress(it)
            self.assertEqual(
                predicted,
                wrapper.optimizer_step_completed,
                f"prediction disagreed with the outcome at micro-batch {i}",
            )

    def test_uncommitted_gradients_track_the_open_window(self) -> None:
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(self.K))
        open_after = []
        for _ in range(self.K):
            wrapper.progress(it)
            open_after.append(wrapper.has_uncommitted_gradients)
        self.assertEqual([True, True, True, False], open_after)

    def test_disabled_ga_reports_every_batch_as_a_completed_step(self) -> None:
        # num_steps > 1 enables gradient accumulation in __post_init__.
        wrapper, _model = self._make(k=1, is_enabled=False)
        it: Iterator[int] = iter(range(3))
        for _ in range(3):
            wrapper.progress(it)
            self.assertTrue(wrapper.optimizer_step_completed)

    def test_eval_never_reports_a_completed_step(self) -> None:
        wrapper, model = self._make()
        it: Iterator[int] = iter(range(self.K))
        for _ in range(self.K - 1):
            wrapper.progress(it)
        model.eval()
        wrapper.progress(it)
        self.assertFalse(wrapper.optimizer_step_completed)


class CountedWindowValidationTest(unittest.TestCase):
    """Tests the micro-batch count in completed accumulation windows.

    ``set_step()`` may start between boundaries, so its first incomplete window warns
    instead of raising.
    """

    K: int = 4

    def _make(self) -> tuple[GradientAccumulationWrapper[Any, int], _MockModel]:
        model = _MockModel()
        pipeline = _IteratorDrivenStepPipeline()
        config = GradientAccumulationConfig(is_enabled=True, num_steps=self.K)
        wrapper: GradientAccumulationWrapper[Any, int] = GradientAccumulationWrapper(
            pipeline, MagicMock(spec=torch.optim.Optimizer), model, config
        )
        model.train()
        return wrapper, model

    def test_a_full_window_is_silent(self) -> None:
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(2 * self.K))
        with self.assertNoLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ):
            for _ in range(2 * self.K):
                wrapper.progress(it)

    def test_set_step_between_boundaries_warns_and_names_both_counts(self) -> None:
        wrapper, _model = self._make()
        wrapper.set_step(2)
        it: Iterator[int] = iter(range(2))
        with self.assertLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ) as logs:
            wrapper.progress(it)
            wrapper.progress(it)
        message = "\n".join(logs.output)
        self.assertIn("2 micro-batch(es)", message)
        self.assertIn(f"configured window size is {self.K}", message)

    def test_realignment_starts_a_fresh_full_window(self) -> None:
        wrapper, _model = self._make()
        first: Iterator[int] = iter(range(2))
        wrapper.progress(first)
        wrapper.progress(first)
        with self.assertRaises(StopIteration):
            wrapper.progress(first)
        second: Iterator[int] = iter(range(self.K))
        with self.assertNoLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ):
            for _ in range(self.K):
                wrapper.progress(second)

    def test_forced_last_batch_partial_window_is_not_held_to_k(self) -> None:
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        with self.assertNoLogs(
            "torchrec.distributed.train_pipeline.gradient_accumulation", level="WARNING"
        ):
            wrapper.progress(it, is_last_batch=True)

    def test_a_missed_anchor_reset_raises(self) -> None:
        """Simulate a re-anchor that does not reset the micro-batch count."""
        wrapper, _model = self._make()
        it: Iterator[int] = iter(range(2))
        wrapper.progress(it)
        wrapper.optimizer_wrapper._window_base -= 2
        with self.assertRaises(RuntimeError) as caught:
            wrapper.progress(it)
        self.assertIn("2 micro-batch(es)", str(caught.exception))
        self.assertIn(f"configured window size is {self.K}", str(caught.exception))


class LiveKSingleSourceTest(unittest.TestCase):
    """Use copied wrapper state for every boundary calculation.

    Mutating the caller's original config cannot change the active window size.
    """

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

        it: Iterator[int] = iter(range(self.K))
        completed = []
        for _ in range(self.K):
            wrapper.progress(it)
            completed.append(wrapper.optimizer_step_completed)
        self.assertEqual([False, False, False, True], completed)


class GAFinalPartialWindowVisibilityTest(unittest.TestCase):
    """Tests a partial-window step performed during iterator exhaustion.

    With ``STEP``, the optimizer runs before ``StopIteration`` prevents ``progress()``
    from reporting the step.
    """

    def _make(
        self, policy: PartialWindowPolicy
    ) -> tuple[GradientAccumulationWrapper[Any, torch.Tensor], nn.Linear]:
        torch.manual_seed(0)
        model = nn.Linear(10, 5)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        config = GradientAccumulationConfig(
            is_enabled=True, num_steps=4, num_warmup_steps=1
        )
        pipeline = _EvalAwareForwardPipeline(model, optimizer)
        wrapper = GradientAccumulationWrapper(
            pipeline, optimizer, model, config, partial_window_policy=policy
        )
        return wrapper, model

    @staticmethod
    def _two_training_micros(
        wrapper: GradientAccumulationWrapper[Any, torch.Tensor], model: nn.Linear
    ) -> torch.Tensor:
        batch = torch.randn(4, 10)
        model.train()
        wrapper.progress(iter([batch]))
        wrapper.progress(iter([batch]))
        return model.weight.detach().clone()

    def test_exhausted_partial_window_under_step_moves_the_weights(self) -> None:
        wrapper, model = self._make(PartialWindowPolicy.STEP)
        before = self._two_training_micros(wrapper, model)

        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        self.assertFalse(
            torch.equal(before, model.weight.detach()),
            "the partial final window was not committed",
        )

    def test_the_committed_partial_window_is_not_reported(self) -> None:
        wrapper, model = self._make(PartialWindowPolicy.STEP)
        self._two_training_micros(wrapper, model)

        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        self.assertFalse(wrapper.optimizer_step_completed)

    def test_exhausted_partial_window_under_discard_leaves_the_weights_alone(
        self,
    ) -> None:
        wrapper, model = self._make(PartialWindowPolicy.DISCARD)
        before = self._two_training_micros(wrapper, model)

        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        self.assertTrue(
            torch.equal(before, model.weight.detach()),
            "DISCARD took an optimizer step on the partial window",
        )

    def test_exhaustion_during_eval_does_not_commit_the_open_window(self) -> None:
        wrapper, model = self._make(PartialWindowPolicy.STEP)
        before = self._two_training_micros(wrapper, model)

        model.eval()
        with self.assertRaises(StopIteration):
            wrapper.progress(iter([]))

        self.assertTrue(
            torch.equal(before, model.weight.detach()),
            "an eval exhaustion flushed the open training window",
        )


class SetNumMicroBatchesPerStepTest(unittest.TestCase):
    """Tests the runtime accumulation-size change used during PT2 warmup.

    Changing the size starts a new window without resetting ``current_step``.
    """

    def _make_wrapper(
        self, num_steps: int = 4, is_enabled: bool = True
    ) -> GradientAccumulationWrapper[Any, Any]:
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        pipeline = _MockPipeline(num_batches=1000)
        config = GradientAccumulationConfig(
            is_enabled=is_enabled, num_steps=num_steps, num_warmup_steps=1
        )
        return GradientAccumulationWrapper(pipeline, optimizer, model, config)

    def test_setter_moves_the_boundary_and_reanchors(self) -> None:
        ga = self._make_wrapper(num_steps=4)
        ga.set_num_micro_batches_per_step(1)

        self.assertEqual(ga.num_micro_batches_per_step, 1)
        self.assertEqual(ga.micro_batches_into_window, 0)

    def test_rejects_k_below_one(self) -> None:
        ga = self._make_wrapper(num_steps=4)
        with self.assertRaises(ValueError):
            ga.set_num_micro_batches_per_step(0)
        with self.assertRaises(ValueError):
            ga.set_num_micro_batches_per_step(-1)
        self.assertEqual(ga.num_micro_batches_per_step, 4)

    def test_rejects_when_gradient_accumulation_is_disabled(self) -> None:
        ga = self._make_wrapper(num_steps=1, is_enabled=False)
        with self.assertRaises(RuntimeError):
            ga.set_num_micro_batches_per_step(1)

    def test_rejects_with_uncommitted_gradients(self) -> None:
        ga = self._make_wrapper(num_steps=4)
        ga.progress(iter([1]))  # one micro of a four-micro window
        self.assertTrue(ga.has_uncommitted_gradients)

        with self.assertRaises(RuntimeError):
            ga.set_num_micro_batches_per_step(1)
        self.assertEqual(ga.num_micro_batches_per_step, 4)

    def test_allow_open_window_waives_the_refusal(self) -> None:
        """Discard the partial window before changing the accumulation size."""
        ga = self._make_wrapper(num_steps=4)
        ga.progress(iter([1]))
        self.assertTrue(ga.has_uncommitted_gradients)
        param = next(iter(ga._model.parameters()))
        param.grad = torch.ones_like(param)

        ga.set_num_micro_batches_per_step(2, allow_open_window=True)

        self.assertEqual(ga.num_micro_batches_per_step, 2)
        self.assertFalse(ga.has_uncommitted_gradients)
        self.assertTrue(ga.optimizer_wrapper._needs_zero_grad)

        ga.optimizer_wrapper.zero_grad()

        self.assertTrue(
            param.grad is None
            or bool(torch.equal(param.grad, torch.zeros_like(param))),
            "the abandoned window's gradients survived into the next window",
        )

    def test_every_progress_completes_a_step_at_k_one(self) -> None:
        ga = self._make_wrapper(num_steps=4)
        ga.set_num_micro_batches_per_step(1)

        for _ in range(3):
            ga.progress(iter([1]))
            self.assertTrue(ga.optimizer_step_completed)
            self.assertFalse(ga.has_uncommitted_gradients)

    def test_round_trip_closes_the_next_window_holding_exactly_k(self) -> None:
        """Restore the original size and complete a full accumulation window."""
        ga = self._make_wrapper(num_steps=4)
        ga.set_num_micro_batches_per_step(1)
        for _ in range(3):  # a non-multiple of 4, deliberately
            ga.progress(iter([1]))
        ga.set_num_micro_batches_per_step(4)

        for i in range(4):
            ga.progress(iter([1]))
            expected_step = i == 3
            self.assertEqual(ga.optimizer_step_completed, expected_step)
        self.assertFalse(ga.has_uncommitted_gradients)

    def test_current_step_stays_monotonic_across_the_round_trip(self) -> None:
        ga = self._make_wrapper(num_steps=4)
        observed = [ga.current_step]

        for _ in range(4):  # one complete K=4 window, so nothing is left pending
            ga.progress(iter([1]))
        observed.append(ga.current_step)
        self.assertFalse(ga.has_uncommitted_gradients)

        ga.set_num_micro_batches_per_step(1)
        observed.append(ga.current_step)
        ga.progress(iter([1]))  # at K=1 this window closes immediately
        observed.append(ga.current_step)
        ga.set_num_micro_batches_per_step(4)
        observed.append(ga.current_step)
        ga.progress(iter([1]))
        observed.append(ga.current_step)

        self.assertEqual(
            observed,
            sorted(observed),
            f"current_step went backwards across a K round trip: {observed}",
        )
        self.assertEqual(
            observed[-1], 6, f"micros were lost or double-counted: {observed}"
        )


class GradScaleTest(unittest.TestCase):
    """Tests the per-step gradient factor on every path that takes an optimizer step.

    Each micro-batch contributes a gradient of exactly 1, and the factor ``10 + q`` names
    the micro-batch count ``q`` it was computed for.
    """

    def _run(
        self,
        num_steps: int,
        num_micros: int,
        bypass_wrapper_step: bool = False,
        mark_last: bool = False,
        grad_scale_fn: Optional[Callable[[int], float]] = lambda q: 10.0 + q,
    ) -> tuple[_SplitOrDefaultModePipeline, list[float]]:
        model = nn.Linear(1, 1, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        pipeline = _SplitOrDefaultModePipeline(model, optimizer, bypass_wrapper_step)
        stepped_grads: list[float] = []

        def record_step(*args: Any, **kwargs: Any) -> None:
            if bypass_wrapper_step:
                # The installed optimizer is the wrapper's; the child is stepped directly.
                installed = pipeline._optimizer
                assert isinstance(installed, GAStepGradScale)
                installed.apply_step_grad_scale()
            stepped_grads.append(cast(torch.Tensor, model.weight.grad).item())

        optimizer.step = record_step  # pyrefly: ignore[bad-assignment]
        wrapper = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=True, num_steps=num_steps),
            window_observer=pipeline.set_ga_window_state,
        )
        wrapper.set_grad_scale_fn(grad_scale_fn)
        data = iter([torch.ones(1, 1) for _ in range(num_micros)])
        for i in range(num_micros):
            wrapper.progress(data, is_last_batch=mark_last and i == num_micros - 1)
        return pipeline, stepped_grads

    def test_scheduled_steps_scale_by_the_full_window(self) -> None:
        _, stepped = self._run(num_steps=2, num_micros=4)
        self.assertEqual(stepped, [2.0 * 12.0, 2.0 * 12.0])

    def test_forced_final_step_scales_by_its_partial_count(self) -> None:
        _, stepped = self._run(num_steps=4, num_micros=3, mark_last=True)
        self.assertEqual(stepped, [3.0 * 13.0])

    def test_partial_window_flush_scales_by_its_partial_count(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        pipeline = _SplitOrDefaultModePipeline(model, optimizer, False)
        stepped: list[float] = []
        optimizer.step = (
            lambda *a, **k: stepped.append(  # pyrefly: ignore[bad-assignment]
                cast(torch.Tensor, model.weight.grad).item()
            )
        )
        wrapper = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            model,
            GradientAccumulationConfig(is_enabled=True, num_steps=4),
        )
        wrapper.set_grad_scale_fn(lambda q: 10.0 + q)
        data = iter([torch.ones(1, 1) for _ in range(3)])
        for _ in range(3):
            wrapper.progress(data)
        with self.assertRaises(StopIteration):
            wrapper.progress(data)
        self.assertEqual(stepped, [3.0 * 13.0])

    def test_pipeline_stepping_directly_reads_the_same_factor(self) -> None:
        pipeline, stepped = self._run(
            num_steps=2, num_micros=4, bypass_wrapper_step=True
        )
        self.assertEqual(pipeline.child_step_count, 2)
        self.assertEqual(stepped, [2.0 * 12.0, 2.0 * 12.0])

    def test_no_scale_fn_leaves_gradients_unchanged(self) -> None:
        _, stepped = self._run(num_steps=2, num_micros=4, grad_scale_fn=None)
        self.assertEqual(stepped, [2.0, 2.0])

    def test_repeated_boundary_steps_scale_once(self) -> None:
        param = nn.Parameter(torch.zeros(1))
        optimizer = optim.SGD([param], lr=0.0)
        wrapper = _GAOptimizerWrapper(
            optimizer,
            GradientAccumulationConfig(is_enabled=True, num_steps=2),
            grad_scale_fn=lambda q: 1.0 / q,
        )
        param.grad = torch.ones(1)
        wrapper._current_step = 1
        wrapper._micro_batches_counted = 1
        wrapper.step()
        wrapper.step()
        self.assertEqual(cast(torch.Tensor, param.grad).item(), 0.5)
        # The next window is scaled again even when gradients are cleared elsewhere.
        param.grad = torch.ones(1)
        wrapper._current_step = 3
        wrapper.step()
        self.assertEqual(cast(torch.Tensor, param.grad).item(), 0.5)

    def test_rewinding_the_step_counter_rescales_the_next_window(self) -> None:
        param = nn.Parameter(torch.zeros(1))
        wrapper = _GAOptimizerWrapper(
            optim.SGD([param], lr=0.0),
            GradientAccumulationConfig(is_enabled=True, num_steps=2),
            grad_scale_fn=lambda q: 1.0 / q,
        )
        for restart in (wrapper.set_step, lambda _: wrapper.realign_window()):
            param.grad = torch.ones(1)
            wrapper.scale_grads(0.5)
            restart(wrapper._current_step)
            param.grad = torch.ones(1)
            wrapper.scale_grads(0.5)
            self.assertEqual(cast(torch.Tensor, param.grad).item(), 0.5)


def _sgd(params: dict[str, nn.Parameter]) -> KeyedOptimizerWrapper:
    return KeyedOptimizerWrapper(params, lambda ps: optim.SGD(ps, lr=0.0))


def _norm_clipper(inner: KeyedOptimizer) -> GradientClippingOptimizer:
    return GradientClippingOptimizer(
        optimizer=inner, clipping=GradientClipping.NORM, max_gradient=1e6
    )


_GA_MODULE = "torchrec.distributed.train_pipeline.gradient_accumulation"


class StepGradScaleFoldTest(unittest.TestCase):
    """Tests where the per-step gradient scale is applied."""

    def setUp(self) -> None:
        self.w1 = nn.Parameter(torch.zeros(2))
        self.w2 = nn.Parameter(torch.zeros(3))

    def _wrapper(self, optimizer: torch.optim.Optimizer) -> _GAOptimizerWrapper:
        return _GAOptimizerWrapper(
            optimizer, GradientAccumulationConfig(is_enabled=True, num_steps=2)
        )

    def _set_grads(self) -> None:
        self.w1.grad = torch.tensor([2.0, 4.0])
        self.w2.grad = torch.tensor([8.0, 16.0, 32.0])

    def _assert_grads(self, factor: float) -> None:
        torch.testing.assert_close(
            self.w1.grad, factor * torch.tensor([2.0, 4.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.w2.grad, factor * torch.tensor([8.0, 16.0, 32.0]), rtol=0, atol=0
        )

    def test_norm_clippers_covering_every_gradient_take_the_scale(self) -> None:
        """NORM clippers that cover every gradient once apply the scale in their step."""
        combined = CombinedOptimizer(
            [
                ("a", _norm_clipper(_sgd({"w1": self.w1}))),
                ("b", _norm_clipper(_sgd({"w2": self.w2}))),
            ]
        )
        wrapper = self._wrapper(combined)
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(1.0)
        combined.step()
        self._assert_grads(0.5)

    def test_gradients_outside_every_clipper_are_multiplied_and_the_rest_folded(
        self,
    ) -> None:
        """A gradient outside every clipper is multiplied; the clipper's take the scale once."""
        clipper = _norm_clipper(_sgd({"w1": self.w1}))
        combined = CombinedOptimizer([("a", clipper), ("b", _sgd({"w2": self.w2}))])
        wrapper = self._wrapper(combined)
        self._set_grads()
        wrapper.scale_grads(0.5)
        torch.testing.assert_close(
            self.w1.grad, torch.tensor([2.0, 4.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            self.w2.grad, torch.tensor([4.0, 8.0, 16.0]), rtol=0, atol=0
        )
        combined.step()
        self._assert_grads(0.5)

    def test_fold_plan_is_computed_once_across_steps(self) -> None:
        """Later steps reuse the plan instead of walking the optimizers again."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        with patch(f"{_GA_MODULE}._fold_targets", wraps=_fold_targets) as walk:
            for _ in range(3):
                self._set_grads()
                wrapper.scale_grads(0.5)
                clipper.step()
                self._assert_grads(0.5)
                wrapper.advance_step()
        self.assertEqual(walk.call_count, 1)

    def test_registering_stepped_optimizers_recomputes_the_plan(self) -> None:
        """A plan made for the wrapped optimizer is not reused for registered ones."""
        inner_a, inner_b = _sgd({"w1": self.w1}), _sgd({"w2": self.w2})
        across = _norm_clipper(CombinedOptimizer([("a", inner_a), ("b", inner_b)]))
        wrapper = self._wrapper(across)
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(1.0)
        wrapper.advance_step()
        wrapper.set_stepped_optimizers([inner_a, inner_b])
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(0.5)

    def test_adding_a_param_group_recomputes_the_plan(self) -> None:
        """A parameter group added after the first step is scaled too."""
        optimizer = optim.SGD([self.w1], lr=0.0)
        wrapper = self._wrapper(optimizer)
        self.w1.grad = torch.tensor([2.0, 4.0])
        wrapper.scale_grads(0.5)
        wrapper.advance_step()
        optimizer.add_param_group({"params": [self.w2]})
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(0.5)

    def test_parameter_without_a_gradient_at_plan_time_is_still_scaled(self) -> None:
        """The plan covers every parameter, not only those with a gradient when made."""
        w3 = nn.Parameter(torch.zeros(1))
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w3": w3}))
        combined = CombinedOptimizer([("a", clipper), ("b", _sgd({"w2": self.w2}))])
        wrapper = self._wrapper(combined)
        self.w1.grad = torch.tensor([2.0, 4.0])
        wrapper.scale_grads(0.5)
        combined.step()
        wrapper.advance_step()
        self._set_grads()
        w3.grad = torch.tensor([8.0])
        wrapper.scale_grads(0.5)
        combined.step()
        self._assert_grads(0.5)
        torch.testing.assert_close(w3.grad, torch.tensor([4.0]), rtol=0, atol=0)

    def test_failed_plan_is_kept_and_every_step_multiplies(self) -> None:
        """A plan that cannot fold is kept, and each step multiplies every gradient once."""
        optimizer = _norm_clipper(_norm_clipper(_sgd({"w1": self.w1, "w2": self.w2})))
        wrapper = self._wrapper(optimizer)
        with patch(f"{_GA_MODULE}._fold_targets", wraps=_fold_targets) as walk:
            for _ in range(2):
                self._set_grads()
                wrapper.scale_grads(0.5)
                self._assert_grads(0.5)
                optimizer.step()
                self._assert_grads(0.5)
                wrapper.advance_step()
        self.assertEqual(walk.call_count, 1)

    def test_nested_or_non_norm_clippers_are_multiplied_once(self) -> None:
        """Nested clippers and non-NORM clippers make the wrapper multiply."""
        params = {"w1": self.w1, "w2": self.w2}
        cases = {
            "nested": _norm_clipper(_norm_clipper(_sgd(params))),
            "value": GradientClippingOptimizer(
                optimizer=_sgd(params),
                clipping=GradientClipping.VALUE,
                max_gradient=1e6,
            ),
        }
        for name, optimizer in cases.items():
            with self.subTest(case=name):
                wrapper = self._wrapper(optimizer)
                self._set_grads()
                wrapper.scale_grads(0.5)
                self._assert_grads(0.5)
                optimizer.step()
                self._assert_grads(0.5)

    def test_parameter_listed_by_two_optimizers_is_scaled_once(self) -> None:
        """A parameter that two optimizers list, as semi-sync DiLoCo does, is scaled once."""
        combined = CombinedOptimizer(
            [("local", _sgd({"w1": self.w1})), ("global", _sgd({"w1": self.w1}))]
        )
        wrapper = self._wrapper(combined)
        self.w1.grad = torch.tensor([2.0, 4.0])
        wrapper.scale_grads(0.5)
        torch.testing.assert_close(
            self.w1.grad, torch.tensor([1.0, 2.0]), rtol=0, atol=0
        )

    def test_clipper_above_the_stepped_optimizers_does_not_take_the_scale(self) -> None:
        """A clipper above the optimizers a pipeline steps directly never runs its clip."""
        inner_a, inner_b = _sgd({"w1": self.w1}), _sgd({"w2": self.w2})
        across = _norm_clipper(CombinedOptimizer([("a", inner_a), ("b", inner_b)]))
        wrapper = self._wrapper(across)
        wrapper.set_stepped_optimizers([inner_a, inner_b])
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(0.5)
        inner_a.step()
        inner_b.step()
        self._assert_grads(0.5)

    def test_no_registered_stepped_optimizers_multiply(self) -> None:
        """A pipeline that registers no stepped optimizers gets the multiply."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        wrapper.set_stepped_optimizers([])
        self._set_grads()
        wrapper.scale_grads(0.5)
        self._assert_grads(0.5)

    def test_scale_is_applied_once_per_step_through_the_clippers(self) -> None:
        """A second scale call in the same step does not scale the gradients again."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        self._set_grads()
        wrapper.scale_grads(0.5)
        wrapper.scale_grads(0.5)
        clipper.step()
        self._assert_grads(0.5)

    def test_scale_of_a_step_whose_clipper_never_ran_is_dropped(self) -> None:
        """A step that an outer optimizer skips does not pass its scale to the next step."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        self._set_grads()
        wrapper.scale_grads(0.5)
        # The clipper's step does not run, as when NanGradientSkipOptimizer skips it.
        wrapper.advance_step()
        wrapper.scale_grads(1.0)
        clipper.step()
        self._assert_grads(1.0)

    def test_dropped_scale_is_not_applied_on_top_of_a_multiply(self) -> None:
        """A dropped scale is not applied again when the next step multiplies instead."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        self._set_grads()
        wrapper.scale_grads(0.5)
        wrapper.advance_step()
        wrapper.set_stepped_optimizers([])
        self._set_grads()
        wrapper.scale_grads(0.5)
        clipper.step()
        self._assert_grads(0.5)

    def test_reset_keeps_dropping_the_scale_of_a_clipper_that_never_ran(self) -> None:
        """After reset(), a scale whose clipper never ran is still dropped."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        self._set_grads()
        wrapper.scale_grads(0.5)
        wrapper.advance_step()
        wrapper.reset()
        self._set_grads()
        wrapper.scale_grads(1.0)
        clipper.step()
        self._assert_grads(1.0)

    def test_reset_then_multiply_does_not_apply_a_dropped_scale(self) -> None:
        """After reset(), multiplying does not also apply the dropped scale."""
        clipper = _norm_clipper(_sgd({"w1": self.w1, "w2": self.w2}))
        wrapper = self._wrapper(clipper)
        self._set_grads()
        wrapper.scale_grads(0.5)
        wrapper.advance_step()
        wrapper.reset()
        wrapper.set_stepped_optimizers([])
        self._set_grads()
        wrapper.scale_grads(0.5)
        clipper.step()
        self._assert_grads(0.5)


class ShardedTensorStepGradScaleTest(unittest.TestCase):
    def test_sharded_tensors_are_skipped_without_a_dispatch(self) -> None:
        """ShardedTensors are skipped unread; dense gradients are still scaled once."""
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        dist.init_process_group(
            "gloo", init_method=f"file://{store_dir.name}/store", rank=0, world_size=1
        )
        self.addCleanup(dist.destroy_process_group)
        table = sharded_tensor.empty(
            ChunkShardingSpec(dim=0, placements=["rank:0/cpu"]), 4, 2
        )
        w = nn.Parameter(torch.zeros(2))
        cases: dict[str, KeyedOptimizer] = {
            "multiply": _sgd({"w": w}),
            "fold": _norm_clipper(_sgd({"w": w})),
        }
        for name, dense in cases.items():
            with self.subTest(case=name):
                sparse = KeyedOptimizer({"table": table}, {}, [{"params": [table]}])
                wrapper = _GAOptimizerWrapper(
                    CombinedOptimizer([("dense", dense), ("sparse", sparse)]),
                    GradientAccumulationConfig(is_enabled=True, num_steps=2),
                )
                w.grad = torch.tensor([2.0, 4.0])
                with patch.object(
                    sharded_tensor.ShardedTensor,
                    "__torch_function__",
                    side_effect=AssertionError("ShardedTensor was dispatched"),
                ):
                    wrapper.scale_grads(0.5)
                dense.step()
                torch.testing.assert_close(
                    w.grad, torch.tensor([1.0, 2.0]), rtol=0, atol=0
                )


class DTensorStepGradScaleTest(unittest.TestCase):
    def test_plain_and_dtensor_gradients_on_two_meshes_are_each_scaled_once(
        self,
    ) -> None:
        """One step scales plain gradients and DTensor gradients on two meshes once."""
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        dist.init_process_group(
            "gloo", init_method=f"file://{store_dir.name}/store", rank=0, world_size=1
        )
        self.addCleanup(dist.destroy_process_group)
        mesh_a = init_device_mesh("cpu", (1,), mesh_dim_names=("a",))
        mesh_b = init_device_mesh("cpu", (1,), mesh_dim_names=("b",))
        plain = nn.Parameter(torch.ones(2))
        on_a = nn.Parameter(distribute_tensor(torch.ones(3), mesh_a, [Replicate()]))
        on_b = nn.Parameter(distribute_tensor(torch.ones(4), mesh_b, [Shard(0)]))
        plain.grad = torch.full((2,), 2.0)
        on_a.grad = distribute_tensor(torch.full((3,), 4.0), mesh_a, [Replicate()])
        on_b.grad = distribute_tensor(torch.full((4,), 8.0), mesh_b, [Shard(0)])
        wrapper = _GAOptimizerWrapper(
            optim.SGD([plain, on_a, on_b], lr=0.0),
            GradientAccumulationConfig(is_enabled=True, num_steps=2),
        )
        wrapper.scale_grads(0.5)
        wrapper.scale_grads(0.5)
        torch.testing.assert_close(plain.grad, torch.full((2,), 1.0), rtol=0, atol=0)
        for param, value in ((on_a, 2.0), (on_b, 4.0)):
            grad = cast(DTensor, param.grad)
            self.assertIsInstance(grad, DTensor)
            torch.testing.assert_close(
                grad.full_tensor(), torch.full(grad.shape, value), rtol=0, atol=0
            )


_GradSnapshot = List[Optional[Tuple[int, torch.Tensor]]]


class _GradSnapshotPipeline(TrainPipeline[Any, torch.Tensor]):
    """Pipeline that records each gradient's storage and value after zero_grad and backward."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer) -> None:
        super().__init__()  # pyrefly: ignore[missing-argument]
        self._model = model
        self._optimizer = optimizer
        self.after_zero_grad: List[_GradSnapshot] = []
        self.after_backward: List[_GradSnapshot] = []

    def _snapshot(self) -> _GradSnapshot:
        return [
            (
                None
                if p.grad is None
                else (p.grad.untyped_storage().data_ptr(), p.grad.detach().clone())
            )
            for p in self._model.parameters()
        ]

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        self._optimizer.zero_grad()
        self.after_zero_grad.append(self._snapshot())
        loss = self._model(batch).sum()
        loss.backward()
        self.after_backward.append(self._snapshot())
        self._optimizer.step()
        return loss


class OneMicroBatchWindowZeroingTest(unittest.TestCase):
    """Tests window-start zeroing of one-micro-batch windows under accumulate_into_buckets."""

    def test_one_micro_batch_window_clears_like_the_wrapped_optimizer(self) -> None:
        """A K = 1 window sets gradients to None and skips the bucket-view path."""
        model = nn.Linear(4, 2, bias=False)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        wrapper = GradientAccumulationWrapper(
            _RealForwardPipeline(model, optimizer),
            optimizer,
            model,
            GradientAccumulationConfig(
                is_enabled=True, num_steps=2, accumulate_into_buckets=True
            ),
        )
        wrapper.set_num_micro_batches_per_step(1)
        model(torch.ones(1, 4)).sum().backward()
        wrapper._bucket_views_ready = True
        wrapper.optimizer_wrapper._needs_zero_grad = True

        with patch.object(wrapper, "_window_start_zero_grad") as mock_window_zero:
            wrapper.optimizer_wrapper.zero_grad()

        mock_window_zero.assert_not_called()
        self.assertIsNone(model.weight.grad)

    def test_bucket_aliases_follow_the_window_length_through_a_ramp(self) -> None:
        """K = 1 windows clear to None; a K = 2 window accumulates in the reducer's bucket."""
        store_dir = tempfile.TemporaryDirectory()
        self.addCleanup(store_dir.cleanup)
        dist.init_process_group(
            "gloo", init_method=f"file://{store_dir.name}/store", rank=0, world_size=1
        )
        self.addCleanup(dist.destroy_process_group)
        torch.manual_seed(0)
        ddp_model = DistributedDataParallel(
            nn.Linear(4, 2, bias=False), gradient_as_bucket_view=True
        )
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.0)
        pipeline = _GradSnapshotPipeline(ddp_model, optimizer)
        wrapper = GradientAccumulationWrapper(
            pipeline,
            optimizer,
            ddp_model,
            GradientAccumulationConfig(
                is_enabled=True, num_steps=2, accumulate_into_buckets=True
            ),
        )
        x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        x2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

        wrapper.set_num_micro_batches_per_step(1)
        for x in (torch.ones(1, 4), torch.full((1, 4), 2.0)):
            wrapper.progress(iter([x]))
        for snapshot in pipeline.after_zero_grad:
            self.assertEqual(snapshot, [None])
        # A synchronized backward leaves each gradient as a view of its reducer bucket.
        bucket_ptr = cast(Tuple[int, torch.Tensor], pipeline.after_backward[-1][0])[0]

        wrapper.set_num_micro_batches_per_step(2)
        wrapper.progress(iter([x1]))
        # A no_sync backward keeps the bucket alias only if zero_grad kept it.
        for snapshot in (pipeline.after_zero_grad[-1], pipeline.after_backward[-1]):
            entry = snapshot[0]
            self.assertIsNotNone(entry)
            self.assertEqual(cast(Tuple[int, torch.Tensor], entry)[0], bucket_ptr)
        torch.testing.assert_close(
            cast(Tuple[int, torch.Tensor], pipeline.after_zero_grad[-1][0])[1],
            torch.zeros(2, 4),
            rtol=0,
            atol=0,
        )
        wrapper.progress(iter([x2]))
        expected = (x1 + x2).expand(2, 4)
        torch.testing.assert_close(
            ddp_model.module.weight.grad, expected, rtol=0, atol=0
        )

        wrapper.set_num_micro_batches_per_step(1)
        wrapper.progress(iter([x1]))
        self.assertEqual(pipeline.after_zero_grad[-1], [None])


def _one_micro_batch_window_grads(
    rank: int, world_size: int, accumulate: bool
) -> list[torch.Tensor]:
    """Return each step's synchronized gradient on this rank, with or without the wrapper."""
    torch.manual_seed(0)
    model = DistributedDataParallel(
        nn.Linear(16, 8, bias=False), gradient_as_bucket_view=True
    )
    optimizer = optim.SGD(model.parameters(), lr=0.0)
    pipeline = _GradSnapshotPipeline(model, optimizer)
    wrapper = GradientAccumulationWrapper(
        pipeline,
        optimizer,
        model,
        GradientAccumulationConfig(
            is_enabled=True, num_steps=2, accumulate_into_buckets=True
        ),
    )
    wrapper.set_num_micro_batches_per_step(1)
    inputs = torch.randn(
        3, 4, 16, generator=torch.Generator().manual_seed(1 + rank)
    ).unbind()
    for x in inputs:
        if accumulate:
            wrapper.progress(iter([x]))
        else:
            pipeline.progress(iter([x]))
    return [cast(Tuple[int, torch.Tensor], s[0])[1] for s in pipeline.after_backward]


def _run_one_micro_batch_window_matches_no_accumulation(
    rank: int, world_size: int
) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        expected = _one_micro_batch_window_grads(rank, world_size, accumulate=False)
        actual = _one_micro_batch_window_grads(rank, world_size, accumulate=True)
        for step, (got, want) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(
                got, want, rtol=0, atol=0, msg=lambda m, step=step: f"step {step}: {m}"
            )
    finally:
        dist.destroy_process_group()


class OneMicroBatchWindowMultiRankTest(MultiProcessTestBase):
    def test_one_micro_batch_windows_match_no_accumulation_at_three_ranks(
        self,
    ) -> None:
        """At W = 3, where DDP's two ways of averaging round differently, K = 1 is bitwise a plain step."""
        self._run_multi_process_test(
            callable=_run_one_micro_batch_window_matches_no_accumulation,
            world_size=3,
        )
