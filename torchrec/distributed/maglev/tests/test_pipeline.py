#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Maglev pass: the dense orders, one fused sparse update per pass, the
dense-weight checks, the loss scale, and what a schedule refuses."""

from __future__ import annotations

import copy
import unittest
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.distributed.pipelining._backward import (
    _get_grad_fn_or_grad_acc,
    stage_backward_input,
)
from torchrec.distributed.maglev.checks import DenseWeightChecks, weight_only_roots
from torchrec.distributed.maglev.customized_pipeline import (
    MaglevRailCustom,
    ScheduleValidator,
)
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.maglev.pipeline import MaglevRail
from torchrec.distributed.maglev.stage import MaglevPassState, StageWrapper
from torchrec.distributed.maglev.tests.utils import (
    as_chain_layer,
    assert_reduced_layer,
    ChainModel,
    input_for,
    mock_groups,
    split_rail,
    stage_mock,
    zero_bubble_rail,
)

WIDTH = 4
ROWS = 8
MICROBATCHES = 4
SPARSE_LR = 0.5

# MaglevRail's 1F1B order for p=4 m=8, as its docstring draws it.
ONE_F_ONE_B_P4_M8 = [
    "F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7",
    "F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7",
    "F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7",
    "F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7",
]


def _recording_stage(
    num_stages: int, stage_index: int, m: int
) -> tuple[MagicMock, list[str]]:
    """A mock stage that records each receive and each dense op it is asked for.

    Ops are named as a custom order names them: ``B`` and ``I`` by the forward
    they consume, ``W`` by the input half it completes.
    """
    calls: list[str] = []
    backwards = iter(range(m))
    deferred: deque[int] = deque()
    stage = MagicMock(
        num_stages=num_stages,
        stage_index=stage_index,
        is_first=stage_index == 0,
        is_last=stage_index == num_stages - 1,
        module=nn.Linear(1, 1),
        pending_weight_work=0,
    )

    def backward_act() -> None:
        deferred.append(next(backwards))
        stage.pending_weight_work = len(deferred)
        calls.append(f"I{deferred[-1]}")

    def backward_weight(**_kwargs: Any) -> None:
        calls.append(f"W{deferred.popleft()}")
        stage.pending_weight_work = len(deferred)

    stage.in_activation_specs.return_value = (
        () if stage_index == 0 else (ActivationSpec(torch.Size([-1, 1])),)
    )
    stage.start_recv_act.side_effect = lambda _input: calls.append("ra")
    stage.dense_forward_micro.side_effect = lambda _input, index, _seams: calls.append(
        f"F{index}"
    )
    stage.start_recv_grad.side_effect = lambda: calls.append("rg")
    stage.dense_backward_micro.side_effect = lambda: calls.append(f"B{next(backwards)}")
    stage.dense_backward_act_micro.side_effect = backward_act
    stage.dense_backward_weight_micro.side_effect = backward_weight
    return stage, calls


def _assert_receives_precede_their_ops(
    test: unittest.TestCase, calls: list[str]
) -> None:
    """Each receive is posted immediately before the op consuming it."""
    for index, call in enumerate(calls):
        if call[0] in "FBI":
            test.assertEqual(calls[index - 1], "ra" if call[0] == "F" else "rg")


SCHEDULES: dict[str, Callable[..., MaglevRail]] = {
    "1f1b": MaglevRail,
    "zero_bubble": zero_bubble_rail,
    "zero_bubble_split": split_rail,
}


def _pass_state(m: int) -> MaglevPassState:
    return MaglevPassState(
        pooled=[None],
        seams=[[None] for _ in range(m)],
        dense_inputs=[[k] for k in range(m)],
    )


class DenseOrderTest(unittest.TestCase):
    def test_each_stage_runs_1f1b_with_whole_backwards(self) -> None:
        num_stages, m = 4, 8
        for stage_index in range(num_stages):
            with self.subTest(stage_index=stage_index):
                stage, calls = _recording_stage(num_stages, stage_index, m)
                MaglevRail(stage, MagicMock(), m)._dense(_pass_state(m))
                ops = [call for call in calls if call[0] in "FB"]
                self.assertEqual(" ".join(ops), ONE_F_ONE_B_P4_M8[stage_index])
                _assert_receives_precede_their_ops(self, calls)
                stage.dense_backward_act_micro.assert_not_called()
                stage.dense_backward_weight_micro.assert_not_called()

    def test_rail_runs_the_order_the_custom_generator_writes(self) -> None:
        # Where each W lands is invisible to the numerics, so pin it here: the
        # derived order, and a custom one handed the generator's rows, both run
        # exactly those rows.
        num_stages, m = 4, 8
        rows = ScheduleValidator.generate(num_stages, m)
        for stage_index in range(num_stages):
            for name in ("derived", "custom"):
                with self.subTest(stage_index=stage_index, schedule=name):
                    stage, calls = _recording_stage(num_stages, stage_index, m)
                    if name == "derived":
                        pipeline = zero_bubble_rail(stage, MagicMock(), m)
                    else:
                        pipeline = MaglevRailCustom(
                            stage, MagicMock(), m, rows[stage_index].split()
                        )
                    pipeline._dense(_pass_state(m))
                    ops = [call for call in calls if call[0] in "FBIW"]
                    self.assertEqual(" ".join(ops), rows[stage_index])
                    _assert_receives_precede_their_ops(self, calls)

    def test_a_custom_row_runs_as_validated_however_it_is_split(self) -> None:
        # The row is validated as the tokens its actions split into, so those
        # tokens are what must run, whatever strings carried them.
        num_stages, m = 4, 8
        row = ScheduleValidator.generate(num_stages, m)[1]
        stage, calls = _recording_stage(num_stages, 1, m)
        MaglevRailCustom(stage, MagicMock(), m, [row])._dense(_pass_state(m))
        ops = [call for call in calls if call[0] in "FBIW"]
        self.assertEqual(" ".join(ops), row)


class _FusedLookup(torch.autograd.Function):
    """``value * weight``, whose backward steps ``weight`` itself, as a TBE's
    fused optimizer does, and records each update's gradient."""

    @staticmethod
    def forward(
        ctx: Any, value: torch.Tensor, weight: torch.Tensor, owner: _FusedSparse
    ) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.owner = owner
        return value * weight

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, None, None]:
        (grad,) = grad_outputs
        (value,) = ctx.saved_tensors
        update = (grad * value).sum(dim=0)
        ctx.owner.updates.append(update.clone())
        with torch.no_grad():
            ctx.owner.weight.sub_(SPARSE_LR * update)
        return None, None, None


class _FusedSparse(nn.Module):
    """A sparse half with a fused optimizer: its weight never holds a gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, WIDTH))
        self.updates: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _FusedLookup.apply(value, self.weight, self)


class _Dense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(
        self, pooled: torch.Tensor, value: torch.Tensor, incoming: Activations
    ) -> Activations:
        return (torch.tanh(self.linear(pooled)),)


class _Layer(MaglevLayer):
    def __init__(self) -> None:
        super().__init__(sparse=_FusedSparse(), dense=_Dense())

    def in_activation_specs(self) -> tuple[ActivationSpec, ...]:
        return ()

    def out_activation_specs(self) -> tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, WIDTH])),)

    def split_dense_input(
        self, layer_input: torch.Tensor, batch_size: int, num_microbatches: int
    ) -> list[torch.Tensor]:
        return list(layer_input.chunk(num_microbatches))


class _Model(MaglevModuleList):
    def get_batch_size(self, layer_inputs: Sequence[Any]) -> int:
        return layer_inputs[0].shape[0]

    def postproc(
        self, activations: Activations, layer_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A row sum, so the microbatches' losses add up to the batch's.
        return (activations[0] - layer_input).square().sum(), activations[0]


def _layer(model: _Model) -> MaglevLayer:
    return cast(MaglevLayer, model[0])


def _sparse(model: _Model) -> _FusedSparse:
    sparse = _layer(model).sparse
    assert isinstance(sparse, _FusedSparse)
    return sparse


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class FusedSparseUpdateTest(unittest.TestCase):
    def _train(
        self,
        schedule: Callable[..., MaglevRail],
        batches: list[torch.Tensor],
    ) -> tuple[_Model, list[int]]:
        """Train a fresh model one pass per batch; the fused updates per pass."""
        torch.manual_seed(0)
        model = _Model([_Layer()])
        stage = StageWrapper(model, [1], 1, process_groups=mock_groups())
        optimizer = torch.optim.SGD(
            _layer(model).require_dense().parameters(), lr=0.1, foreach=True
        )
        pipeline = schedule(stage, optimizer, MICROBATCHES)
        updates = []
        for batch in batches:
            before = len(_sparse(model).updates)
            with patch.object(stage, "take_global_inputs", return_value=[batch]):
                pipeline.progress(iter([]))
            updates.append(len(_sparse(model).updates) - before)
        return model, updates

    def _reference(self, batches: list[torch.Tensor]) -> _Model:
        """One forward and backward on each whole batch."""
        torch.manual_seed(0)
        model = _Model([_Layer()])
        optimizer = torch.optim.SGD(
            _layer(model).require_dense().parameters(), lr=0.1, foreach=True
        )
        for batch in batches:
            optimizer.zero_grad()
            model([batch])[0].backward()
            optimizer.step()
        return model

    def test_fused_sparse_update_runs_once_per_pass_on_the_global_batch(
        self, *_mocks: Any
    ) -> None:
        torch.manual_seed(1)
        batches = [torch.randn(ROWS, WIDTH) for _ in range(2)]
        reference = self._reference(batches)
        for name, schedule in SCHEDULES.items():
            with self.subTest(schedule=name), patch(
                "torchrec.distributed.maglev.stage.stage_backward_input",
                wraps=stage_backward_input,
            ) as split:
                model, updates = self._train(schedule, batches)
                self.assertEqual(updates, [1, 1])
                # 1F1B runs whole backwards; both zero-bubble arms split every
                # backward, stage 0's included.
                splits = 0 if name == "1f1b" else MICROBATCHES
                self.assertEqual(split.call_count, len(batches) * splits)
                for (name, parameter), expected in zip(
                    model.named_parameters(), reference.parameters()
                ):
                    torch.testing.assert_close(parameter, expected, msg=name)


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class WholeBackwardReachTest(unittest.TestCase):
    def test_unreached_dense_weight_is_refused_only_when_disallowed(
        self, *_mocks: Any
    ) -> None:
        model = _Model([_Layer()])
        stage = StageWrapper(model, [1], 1, process_groups=mock_groups())
        checks = stage._checks
        x = torch.randn(ROWS, WIDTH, requires_grad=True)
        reaches_all = cast(_Dense, _layer(model).require_dense()).linear(x).sum()
        reaches_none = x.square().sum()

        checks.before_whole_backward([reaches_none], stage.dense_parameters)
        checks.set_unreached_allowed(False)
        checks.before_whole_backward([reaches_all], stage.dense_parameters)
        with self.assertRaisesRegex(RuntimeError, "set_reduce_scatter_unused_params"):
            checks.before_whole_backward([reaches_none], stage.dense_parameters)
        # Structural, so walked on the first pass only.
        checks.end_first_pass()
        checks.before_whole_backward([reaches_none], stage.dense_parameters)


class _Schedule:
    """Stands in for a schedule: the stage only tells one from another."""


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class ConfigurePassTest(unittest.TestCase):
    def test_a_new_schedule_repeats_the_first_pass_checks(self, *_mocks: Any) -> None:
        model = _Model([_Layer()])
        stage = StageWrapper(model, [1], 1, process_groups=mock_groups())
        checks = stage._checks
        x = torch.randn(ROWS, WIDTH, requires_grad=True)
        reaches_none = x.square().sum()
        first, second = _Schedule(), _Schedule()

        stage.configure_pass(first, loss_scale=1.0, allow_unreached=False)
        with self.assertRaisesRegex(RuntimeError, "set_reduce_scatter_unused_params"):
            checks.before_whole_backward([reaches_none], stage.dense_parameters)
        checks.end_first_pass()
        stage.configure_pass(first, loss_scale=1.0, allow_unreached=False)
        checks.before_whole_backward([reaches_none], stage.dense_parameters)
        # Another order builds another graph, so its first pass checks again.
        stage.configure_pass(second, loss_scale=1.0, allow_unreached=False)
        with self.assertRaisesRegex(RuntimeError, "set_reduce_scatter_unused_params"):
            checks.before_whole_backward([reaches_none], stage.dense_parameters)


class DenseWeightChecksTest(unittest.TestCase):
    def test_counts_roots_that_reach_weights_but_no_input(self) -> None:
        weight = nn.Parameter(torch.ones(WIDTH))
        seam = torch.randn(ROWS, WIDTH, requires_grad=True)
        raw = torch.randn(ROWS, WIDTH)
        from_seam = seam * weight
        from_weights_only = raw * weight
        roots = [_get_grad_fn_or_grad_acc(t) for t in (from_seam, from_weights_only)]
        self.assertEqual(weight_only_roots(roots, [seam], [weight]), 1)
        self.assertEqual(weight_only_roots(roots[:1], [seam], [weight]), 0)

    def test_coverage_refuses_a_reached_weight_without_a_group(self) -> None:
        value = torch.ones(2, requires_grad=True)
        used = nn.Parameter(torch.tensor(2.0))
        independent = nn.Parameter(torch.tensor(3.0))
        unreached = nn.Parameter(torch.tensor(4.0))
        weights = [used, independent, unreached]
        # Both outputs are used, but the second has no differentiable activation
        # input, so upstream's I/W grouping records no group for its parameter.
        roots = [(value * used).sum(), independent.square()]
        # Held before the split backward detaches the roots, as
        # dense_backward_act_micro does.
        root_nodes = [_get_grad_fn_or_grad_acc(root) for root in roots]
        _input_grads, groups = stage_backward_input(roots, None, [value], iter(weights))
        checks = DenseWeightChecks(0, 1, MagicMock(), lambda: torch.device("cpu"))
        # Past the first pass, so only the coverage check runs.
        checks.end_first_pass()
        with self.assertRaisesRegex(RuntimeError, "so 1 would receive no gradient"):
            checks.after_input_half(groups, weights, root_nodes, [value])
        # An unreached weight alone is allowed, and its set is not walked again.
        checks.after_input_half(groups, [used, unreached], root_nodes, [value])
        self.assertEqual(checks._unreached_verified, {frozenset({id(unreached)})})
        checks.set_unreached_allowed(False)
        with self.assertRaisesRegex(RuntimeError, "Under FSDP2"):
            checks.after_input_half(groups, [used, unreached], root_nodes, [value])
        # A reached weight without a group is that error under either policy.
        with self.assertRaisesRegex(RuntimeError, "so 1 would receive no gradient"):
            checks.after_input_half(groups, [used, independent], root_nodes, [value])

    def test_first_pass_refuses_a_weight_only_output(self) -> None:
        checks = DenseWeightChecks(0, 1, MagicMock(), lambda: torch.device("cpu"))
        weight = nn.Parameter(torch.ones(WIDTH))
        roots = [_get_grad_fn_or_grad_acc(torch.randn(ROWS, WIDTH) * weight)]
        # The weight has a group, so only the weight-only output is wrong.
        groups = [{"params": [_get_grad_fn_or_grad_acc(weight)]}]
        with self.assertRaisesRegex(RuntimeError, "run its stage's backwards whole"):
            checks.after_input_half(groups, [weight], roots, [])
        checks.end_first_pass()
        checks.after_input_half(groups, [weight], roots, [])


class _MeanModel(_Model):
    def postproc(
        self, activations: Activations, layer_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A mean over the rows, as a data-parallel loss would be reduced.
        return (activations[0] - layer_input).square().mean(), activations[0]


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class LossScaleTest(unittest.TestCase):
    def test_scaled_mean_losses_train_like_the_whole_batch_mean(
        self, *_mocks: Any
    ) -> None:
        torch.manual_seed(1)
        batch = torch.randn(ROWS, WIDTH)
        torch.manual_seed(0)
        reference = _MeanModel([_Layer()])
        reference_optimizer = torch.optim.SGD(
            _layer(reference).require_dense().parameters(), lr=0.1, foreach=True
        )
        expected_losses = torch.stack(
            [reference([part])[0].detach() for part in batch.chunk(MICROBATCHES)]
        )
        reference([batch])[0].backward()
        reference_optimizer.step()
        for name, schedule in SCHEDULES.items():
            with self.subTest(schedule=name):
                torch.manual_seed(0)
                model = _MeanModel([_Layer()])
                stage = StageWrapper(model, [1], 1, process_groups=mock_groups())
                optimizer = torch.optim.SGD(
                    _layer(model).require_dense().parameters(), lr=0.1, foreach=True
                )
                pipeline = schedule(stage, optimizer, MICROBATCHES, scale_grads=True)
                # A schedule built later on the same stage does not change the
                # first one's scale: each pass sets its own.
                schedule(stage, optimizer, MICROBATCHES)
                with patch.object(stage, "take_global_inputs", return_value=[batch]):
                    losses = pipeline.progress(iter([]))
                assert losses is not None
                torch.testing.assert_close(losses, expected_losses)
                self.assertFalse(losses.requires_grad)
                for (name, parameter), expected in zip(
                    model.named_parameters(), reference.parameters()
                ):
                    torch.testing.assert_close(parameter, expected, msg=name)


class ScheduleConstructionTest(unittest.TestCase):
    """What a schedule refuses when it is built over a stage."""

    def test_refuses_activations_a_boundary_stage_cannot_have(self) -> None:
        stage = stage_mock()
        stage.in_activation_specs.return_value = (ActivationSpec(torch.Size([-1, 1])),)
        with self.assertRaisesRegex(ValueError, "stage 0 declares an incoming"):
            MaglevRail(stage, MagicMock(spec=torch.optim.Optimizer), 2)
        deeper = stage_mock()
        deeper.num_stages, deeper.stage_index, deeper.is_first = 2, 1, False
        with self.assertRaisesRegex(ValueError, "stage 1 declares no incoming"):
            zero_bubble_rail(deeper, MagicMock(spec=torch.optim.Optimizer), 2)

    def test_refuses_an_unknown_dense_schedule(self) -> None:
        for dense_schedule in ("zb", "ZERO_BUBBLE", ""):
            with self.subTest(dense_schedule=dense_schedule):
                with self.assertRaisesRegex(ValueError, "dense_schedule must be one"):
                    MaglevRail(
                        stage_mock(),
                        MagicMock(spec=torch.optim.Optimizer),
                        2,
                        dense_schedule=dense_schedule,
                    )

    def test_refuses_fewer_microbatches_than_stages(self) -> None:
        stage = stage_mock()
        stage.num_stages = 3
        with self.assertRaisesRegex(ValueError, "at least one microbatch per stage"):
            MaglevRail(stage, MagicMock(spec=torch.optim.Optimizer), 2)

    def test_refuses_a_nested_custom_reducer(self) -> None:
        class CustomReducer(nn.Linear):
            def no_sync(self) -> Any:
                return MagicMock()

        stage = stage_mock()
        stage.module = nn.Sequential(CustomReducer(1, 1))
        with self.assertRaisesRegex(ValueError, "CustomReducer"):
            zero_bubble_rail(stage, MagicMock(spec=torch.optim.Optimizer), 2)

    def test_refuses_model_trackers(self) -> None:
        # DistributedModelParallel steps its trackers on every forward call,
        # which a pass makes once per microbatch.
        stage = stage_mock()
        stage.module.model_trackers = {"RAW_ID_TRACKER": MagicMock()}
        with self.assertRaisesRegex(ValueError, "model_tracker_configs"):
            MaglevRail(stage, MagicMock(spec=torch.optim.Optimizer), 2)
        stage.module.model_trackers = {}
        pipeline = MaglevRail(stage, MagicMock(spec=torch.optim.Optimizer), 2)
        # One added after build (init_torchrec_delta_tracker) is refused at the
        # next pass.
        stage.module.model_trackers = {"DELTA_TRACKER": MagicMock()}
        with self.assertRaisesRegex(ValueError, "model_tracker_configs"):
            pipeline._configure_stage()


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class PassFailureTest(unittest.TestCase):
    def test_only_a_failed_input_round_poisons_the_stage(self, *_mocks: Any) -> None:
        stage = StageWrapper(_Model([_Layer()]), [1], 1, process_groups=mock_groups())
        pipeline = MaglevRail(stage, MagicMock(spec=torch.optim.Optimizer), 1)
        batch = torch.randn(ROWS, WIDTH)
        # The end of the data: nothing ran, so the stage stays usable.
        with patch.object(stage, "take_global_inputs", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                pipeline.progress(iter([]))
        with patch.object(stage, "take_global_inputs", return_value=[batch]):
            pipeline.progress(iter([]))
        # A failed round leaves the peers' inputs out of step.
        with patch.object(
            stage, "take_global_inputs", side_effect=RuntimeError("round failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "round failed"):
                pipeline.progress(iter([]))
        with patch.object(stage, "take_global_inputs", return_value=[batch]):
            with self.assertRaisesRegex(RuntimeError, "restart the job"):
                pipeline.progress(iter([]))


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class FrozenParameterTest(unittest.TestCase):
    def test_a_frozen_dense_bias_stays_frozen_over_three_passes(
        self, _get_rank: Any, driver: Any
    ) -> None:
        for schedule in (MaglevRail, zero_bubble_rail, split_rail):
            with self.subTest(schedule=schedule.__name__):
                torch.manual_seed(122)
                model = ChainModel()
                as_chain_layer(model[0]).dense_half().projection.bias.requires_grad_(
                    False
                )
                reference = copy.deepcopy(model)
                stage = StageWrapper(model, [2], 1, process_groups=mock_groups())
                optimizer = torch.optim.SGD(
                    stage.module.parameters(), lr=0.01, foreach=True
                )
                reference_optimizer = torch.optim.SGD(
                    reference.parameters(), lr=0.01, foreach=True
                )
                pipeline = schedule(stage, optimizer, num_microbatches=4)
                for pass_index in range(3):
                    value = input_for(0, pass_index, torch.device("cpu"))
                    reference_optimizer.zero_grad()
                    reference(value)[0].backward()
                    reference_optimizer.step()
                    driver.return_value.take.return_value = [[value, value]]
                    pipeline.progress(iter([]))
                    for index, layer in enumerate(stage._layers):
                        assert_reduced_layer(layer, reference[index])
