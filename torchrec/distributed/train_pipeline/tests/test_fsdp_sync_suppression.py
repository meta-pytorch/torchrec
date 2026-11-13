#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test FSDP2 module discovery and flag changes with CPU stubs.

The stubs record method calls without running native FSDP2 collectives.
"""

import unittest

import torch
from torch.distributed.fsdp import FSDPModule
from torchrec.distributed.train_pipeline.fsdp_sync_suppression import (
    _SIMPLE_FSDP_MARKER,
    FsdpSyncSuppression,
)


class _StubGroup:
    """Represent the FSDP parameter-group state used by these tests."""

    def __init__(self, reshard_after_forward: bool, shardable: bool = True) -> None:
        self._reshard_after_forward = reshard_after_forward
        self.is_sharded = True
        self.shardable = shardable
        self.reshard_calls = 0
        self._partial_reduce_output: torch.Tensor | None = None

    def reshard(self) -> None:
        self.reshard_calls += 1
        if self.shardable:
            self.is_sharded = True


class _StubState:
    def __init__(
        self,
        groups: list[_StubGroup],
        is_root: bool | None = True,
        auto_reshard_after_forward: bool | None = False,
    ) -> None:
        self._fsdp_param_groups = groups
        self._is_root = is_root
        # False identifies an explicit reshard_after_forward setting.
        self._auto_reshard_after_forward = auto_reshard_after_forward
        self.reset_calls = 0

    def _reset_iter_state(self) -> None:
        self.reset_calls += 1


class _StubFsdp(FSDPModule, torch.nn.Module):
    """Record FSDPModule flag calls without initializing native FSDP state."""

    def __new__(cls, *args: object, **kwargs: object) -> "_StubFsdp":
        # Bypass FSDPModule.__new__, which would construct a plain nn.Module.
        return object.__new__(cls)

    def __init__(
        self,
        reshard_after_forward: bool = False,
        is_root: bool | None = True,
        shardable: bool = True,
        auto_reshard_after_forward: bool | None = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.all_reduce_calls: list[bool] = []
        self.reshard_flag_calls: list[bool] = []
        self.forbidden_calls: list[str] = []
        self.stub_state = _StubState(
            [_StubGroup(reshard_after_forward, shardable)],
            is_root,
            auto_reshard_after_forward,
        )

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        self.all_reduce_calls.append(requires_all_reduce)

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        self.reshard_flag_calls.append(reshard_after_backward)

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        self.forbidden_calls.append("set_requires_gradient_sync")

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        self.forbidden_calls.append("set_is_last_backward")

    # pyrefly: ignore[bad-override]
    def _get_fsdp_state(self) -> _StubState:
        return self.stub_state


class FsdpSyncSuppressionCollectTest(unittest.TestCase):
    def test_collects_native_fsdp_modules_including_nested(self) -> None:
        root = _StubFsdp()
        inner = _StubFsdp()
        root.add_module("inner", inner)
        model = torch.nn.Module()
        model.add_module("root", root)
        model.add_module("plain", torch.nn.Linear(2, 2))
        targets = FsdpSyncSuppression().collect(model)
        self.assertEqual([id(t) for t in targets], [id(root), id(inner)])

    def test_a_model_with_no_fsdp_collects_nothing(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.assertEqual(FsdpSyncSuppression().collect(model), [])

    def test_simple_fsdp_is_excluded_and_its_subtree_is_still_searched(self) -> None:
        # SimpleFSDP manages its own gradients; native FSDP2 controls do not apply.
        simple = _StubFsdp()
        setattr(simple, _SIMPLE_FSDP_MARKER, {})
        native = _StubFsdp()
        simple.add_module("native", native)
        model = torch.nn.Module()
        model.add_module("simple", simple)
        targets = FsdpSyncSuppression().collect(model)
        self.assertEqual([id(t) for t in targets], [id(native)])


class FsdpSyncSuppressionFlagTest(unittest.TestCase):
    def test_suppress_then_restore_toggles_both_flags(self) -> None:
        module = _StubFsdp(reshard_after_forward=False)
        suppression = FsdpSyncSuppression()
        suppression.suppress([module])
        suppression.restore([module])
        self.assertEqual(module.all_reduce_calls, [False, True])
        self.assertEqual(module.reshard_flag_calls, [False, True])

    def test_the_two_forbidden_setters_are_never_written(self) -> None:
        # Keep reduce-scatter enabled and wait for its completion on every backward.
        module = _StubFsdp()
        suppression = FsdpSyncSuppression()
        suppression.suppress([module])
        suppression.restore([module])
        suppression.ensure_sharded([module])
        suppression.clear_partial([module])
        self.assertEqual(module.forbidden_calls, [])

    def test_reshard_flag_is_not_touched_on_a_full_shard_module(self) -> None:
        # Preserve FULL_SHARD's release of unsharded parameters after backward.
        # Deferral would retain them between microbatches.
        full_shard = _StubFsdp(reshard_after_forward=True)
        FsdpSyncSuppression().suppress([full_shard])
        self.assertEqual(full_shard.reshard_flag_calls, [])
        # The all-reduce flag changes even when resharding is not deferred.
        self.assertEqual(full_shard.all_reduce_calls, [False])

    def test_reshard_flag_is_not_touched_on_a_default_built_root(self) -> None:
        # Lazy initialization makes default roots report reshard_after_forward=False.
        # The automatic setting still excludes them from reshard deferral.
        auto_root = _StubFsdp(
            reshard_after_forward=False, auto_reshard_after_forward=True
        )
        FsdpSyncSuppression().suppress([auto_root])
        self.assertEqual(auto_root.reshard_flag_calls, [])
        self.assertEqual(auto_root.all_reduce_calls, [False])

    def test_reshard_flag_is_touched_on_a_shard_grad_op_module(self) -> None:
        # SHARD_GRAD_OP retains unsharded parameters through forward and backward.
        # Deferring resharding reuses them across microbatches.
        shard_grad_op = _StubFsdp(reshard_after_forward=False)
        FsdpSyncSuppression().suppress([shard_grad_op])
        self.assertEqual(shard_grad_op.reshard_flag_calls, [False])


class FsdpSyncSuppressionEnsureShardedTest(unittest.TestCase):
    def test_ensure_sharded_reshards_every_target(self) -> None:
        modules = [_StubFsdp(), _StubFsdp()]
        FsdpSyncSuppression().ensure_sharded(modules)
        self.assertEqual(
            [m.stub_state._fsdp_param_groups[0].reshard_calls for m in modules], [1, 1]
        )

    def test_ensure_sharded_raises_when_a_group_stays_unsharded(self) -> None:
        module = _StubFsdp(shardable=False)
        module.stub_state._fsdp_param_groups[0].is_sharded = False
        with self.assertRaisesRegex(RuntimeError, "still unsharded"):
            FsdpSyncSuppression().ensure_sharded([module])


class FsdpSyncSuppressionClearPartialTest(unittest.TestCase):
    def test_clear_partial_resets_root_iteration_state(self) -> None:
        module = _StubFsdp(is_root=True)
        FsdpSyncSuppression().clear_partial([module])
        self.assertEqual(module.stub_state.reset_calls, 1)

    def test_clear_partial_resets_a_pre_lazy_init_module(self) -> None:
        # _is_root=None is valid before lazy initialization.
        # Only initialized non-root modules reject reset_iter_state().
        module = _StubFsdp(is_root=None)
        FsdpSyncSuppression().clear_partial([module])
        self.assertEqual(module.stub_state.reset_calls, 1)

    def test_clear_partial_skips_a_non_root(self) -> None:
        module = _StubFsdp(is_root=False)
        FsdpSyncSuppression().clear_partial([module])
        self.assertEqual(module.stub_state.reset_calls, 0)

    def test_has_partial_reports_only_a_pending_reduction(self) -> None:
        idle, pending = _StubFsdp(), _StubFsdp()
        group = pending.stub_state._fsdp_param_groups[0]
        group._partial_reduce_output = torch.zeros(1)
        suppression = FsdpSyncSuppression()
        self.assertFalse(suppression.has_partial([idle]))
        self.assertTrue(suppression.has_partial([idle, pending]))
