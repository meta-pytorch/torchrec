#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Any, cast, Dict, List, Sequence, Tuple
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torchrec.distributed.maglev.customized_pipeline import (
    MaglevRailCustom,
    ScheduleError,
    ScheduleValidator,
)
from torchrec.distributed.maglev.maglev_pp import (
    MaglevPP,
    MaglevPPConfig,
    MaglevSchedule,
)
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.maglev.pipeline import Maglev1F1B, MaglevRail
from torchrec.distributed.maglev.stage import MaglevProcessGroups
from torchrec.distributed.maglev.tests.utils import (
    assert_reduced_layer,
    ChainModel,
    input_for,
    microbatches,
    mock_groups,
    partial_devices,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)

WIDTH = 3
ROWS = 4
MICROBATCHES = 2


class _Sparse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, WIDTH))

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        return layer_input * self.scale


class _Dense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(
        self, sparse_output: Any, layer_input: torch.Tensor, in_activations: Activations
    ) -> Activations:
        x = layer_input if sparse_output is None else sparse_output
        if in_activations:
            x = x + in_activations[0]
        return (torch.tanh(self.linear(x)),)


class _Layer(MaglevLayer):
    def __init__(self, is_first: bool, sparse: bool = False) -> None:
        super().__init__(sparse=_Sparse() if sparse else None, dense=_Dense())
        self.is_first = is_first

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return () if self.is_first else (ActivationSpec(torch.Size([-1, WIDTH])),)

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        return (ActivationSpec(torch.Size([-1, WIDTH])),)

    def split_dense_input(
        self, layer_input: torch.Tensor, batch_size: int, num_microbatches: int
    ) -> List[torch.Tensor]:
        return list(layer_input.chunk(num_microbatches))


class _Model(MaglevModuleList):
    def __init__(self, layers: List[MaglevLayer], mean_loss: bool) -> None:
        super().__init__(layers)
        self.mean_loss = mean_loss

    def get_batch_size(self, layer_inputs: Sequence[Any]) -> int:
        return layer_inputs[0].shape[0]

    def postproc(
        self, activations: Activations, layer_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        squares = (activations[0] - layer_input).square()
        return squares.mean() if self.mean_loss else squares.sum(), activations[0]


def _model(sparse: bool = False, mean_loss: bool = True) -> _Model:
    torch.manual_seed(0)
    layers: List[MaglevLayer] = [
        _Layer(is_first=True, sparse=sparse),
        _Layer(is_first=False),
    ]
    return _Model(layers, mean_loss)


def _config(schedule: MaglevSchedule, **overrides: Any) -> MaglevPPConfig:
    fields = {
        "schedule": schedule,
        "num_microbatches": MICROBATCHES,
        "layers_per_stage": (2,),
        "stage_size": 1,
    }
    fields.update(overrides)
    return MaglevPPConfig(**fields)


# (schedule, MaglevPPConfig overrides) on one stage of two microbatches.
_SCHEDULE_CASES: Tuple[Tuple[MaglevSchedule, Dict[str, Any]], ...] = (
    (MaglevSchedule.ONE_F_ONE_B, {}),
    (MaglevSchedule.RAIL, {}),
    (MaglevSchedule.RAIL, {"dense_schedule": "zero_bubble"}),
    (MaglevSchedule.RAIL, {"schedule_actions": ("F0 I0 F1 W0 I1 W1",)}),
)


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class MaglevPPTest(unittest.TestCase):
    def _pp(self, schedule: MaglevSchedule, **overrides: Any) -> MaglevPP:
        return MaglevPP(_model(), _config(schedule, **overrides), mock_groups())

    def test_config_routes_to_the_schedule(self, *_mocks: Any) -> None:
        pp = self._pp(MaglevSchedule.ONE_F_ONE_B)
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        self.assertIs(type(pp.pipeline), Maglev1F1B)
        for dense_schedule, expected in (
            (None, "1f1b"),
            ("1f1b", "1f1b"),
            ("zero_bubble", "zero_bubble"),
        ):
            with self.subTest(dense_schedule=dense_schedule):
                pp = self._pp(MaglevSchedule.RAIL, dense_schedule=dense_schedule)
                pp.build(MagicMock(spec=torch.optim.Optimizer))
                self.assertIs(type(pp.pipeline), MaglevRail)
                self.assertEqual(cast(MaglevRail, pp.pipeline).dense_schedule, expected)

    def test_rail_actions_build_the_custom_schedule(self, *_mocks: Any) -> None:
        for dense_schedule in (None, "zero_bubble"):
            with self.subTest(dense_schedule=dense_schedule):
                pp = self._pp(
                    MaglevSchedule.RAIL,
                    schedule_actions=("F0 I0 W0 F1 I1 W1",),
                    dense_schedule=dense_schedule,
                )
                pp.build(MagicMock(spec=torch.optim.Optimizer))
                self.assertIsInstance(pp.pipeline, MaglevRailCustom)
                self.assertEqual(
                    cast(MaglevRail, pp.pipeline).dense_schedule, "zero_bubble"
                )

    def test_config_converts_names_and_lists(self, *_mocks: Any) -> None:
        config = MaglevPPConfig(
            schedule="rail",  # pyrefly: ignore[bad-argument-type]
            num_microbatches=MICROBATCHES,
            layers_per_stage=[2],  # pyrefly: ignore[bad-argument-type]
            stage_size=1,
        )
        self.assertIs(config.schedule, MaglevSchedule.RAIL)
        self.assertEqual(config.layers_per_stage, (2,))
        self.assertTrue(config.scale_grads)
        with self.assertRaises(ValueError):
            _config("zero-bubble")  # pyrefly: ignore[bad-argument-type]

    def test_config_refusals(self, *_mocks: Any) -> None:
        with self.assertRaisesRegex(ValueError, "would ignore them"):
            _config(MaglevSchedule.ONE_F_ONE_B, schedule_actions=("F0 B0",))
        with self.assertRaisesRegex(ValueError, "1 stages"):
            _config(MaglevSchedule.RAIL, schedule_actions=("F0", "F0"))
        with self.assertRaisesRegex(ValueError, "at least the stage count"):
            _config(MaglevSchedule.RAIL, num_microbatches=1, layers_per_stage=(1, 1))
        with self.assertRaisesRegex(ValueError, "one entry per stage"):
            _config(MaglevSchedule.RAIL, layers_per_stage=())
        with self.assertRaises(ScheduleError):
            self._pp(MaglevSchedule.RAIL, schedule_actions=("F0 I0 W0",))
        with self.assertRaisesRegex(ValueError, "'rail' only"):
            _config(MaglevSchedule.ONE_F_ONE_B, dense_schedule="1f1b")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            _config(MaglevSchedule.RAIL, dense_schedule="interleaved")
        with self.assertRaisesRegex(ValueError, "cannot run dense_schedule '1f1b'"):
            _config(
                MaglevSchedule.RAIL, schedule_actions=("F0 B0",), dense_schedule="1f1b"
            )

    def test_input_batch_size_follows_the_schedule(self, *_mocks: Any) -> None:
        rail = self._pp(MaglevSchedule.RAIL)
        self.assertEqual(rail.input_batch_size(ROWS), ROWS * MICROBATCHES)
        one_f_one_b = self._pp(MaglevSchedule.ONE_F_ONE_B)
        self.assertEqual(one_f_one_b.input_batch_size(ROWS), ROWS)

    def test_each_stage_builds_its_own_row(self, get_rank: Any, _driver: Any) -> None:
        rows = ScheduleValidator.generate(2, MICROBATCHES)
        get_rank.return_value = 1
        pp = MaglevPP(
            _model(),
            _config(
                MaglevSchedule.RAIL, layers_per_stage=(1, 1), schedule_actions=rows
            ),
            mock_groups(2),
        )
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        self.assertEqual(pp.stage.stage_index, 1)
        self.assertEqual(
            cast(MaglevRailCustom, pp.pipeline).actions, tuple(rows[1].split())
        )

    def test_a_deadlocking_grid_is_refused_before_the_stage_exists(
        self, *_mocks: Any
    ) -> None:
        # Each row is legal on its own; together they deadlock.
        rows = ("F0 I0 W0 F1 I1 W1", "F0 F1 I0 W0 I1 W1")
        with patch("torchrec.distributed.maglev.maglev_pp.StageWrapper") as stage:
            with self.assertRaisesRegex(ScheduleError, "deadlocks"):
                MaglevPP(
                    _model(),
                    _config(
                        MaglevSchedule.RAIL,
                        layers_per_stage=(1, 1),
                        schedule_actions=rows,
                    ),
                    mock_groups(2),
                )
        stage.assert_not_called()

    def test_close_releases_the_schedule(self, *_mocks: Any) -> None:
        pp = self._pp(MaglevSchedule.RAIL)
        pp.close()
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        with patch.object(pp.pipeline, "close") as close:
            pp.close()
        close.assert_called_once_with()

    def test_module_is_wrapped_before_build(self, *_mocks: Any) -> None:
        pp = self._pp(MaglevSchedule.ONE_F_ONE_B)
        wrapper = nn.Sequential(pp.module)
        pp.module = wrapper
        self.assertIs(pp.stage.module, wrapper)
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        with self.assertRaisesRegex(RuntimeError, "before build"):
            pp.module = nn.Sequential()
        with self.assertRaisesRegex(RuntimeError, "runs once"):
            pp.build(MagicMock(spec=torch.optim.Optimizer))

    def test_to_places_the_stage_and_chains(self, *_mocks: Any) -> None:
        pp = self._pp(MaglevSchedule.RAIL)
        with patch.object(pp.stage, "to") as to:
            self.assertIs(pp.to(torch.device("cpu")), pp)
        to.assert_called_once_with(torch.device("cpu"))

    def test_progress_needs_build(self, *_mocks: Any) -> None:
        with self.assertRaisesRegex(RuntimeError, "build"):
            self._pp(MaglevSchedule.RAIL).progress(iter([]))

    def test_one_model_trains_like_the_whole_batch_under_every_schedule(
        self, *_mocks: Any
    ) -> None:
        # MaglevRail with scale_grads trains mean-loss microbatches on the whole
        # batch's mean; Maglev1F1B sums its microbatch losses, so a summed loss
        # trains on the whole batch's sum.
        torch.manual_seed(1)
        parts = [torch.randn(ROWS, WIDTH) for _ in range(MICROBATCHES)]
        whole = torch.cat(parts)
        for schedule, overrides in _SCHEDULE_CASES:
            with self.subTest(schedule=schedule, **overrides):
                mean_loss = schedule is MaglevSchedule.RAIL
                reference = _model(sparse=True, mean_loss=mean_loss)
                reference([whole, whole])[0].backward()
                model = _model(sparse=True, mean_loss=mean_loss)
                pp = MaglevPP(model, _config(schedule, **overrides), mock_groups())
                pp.build(MagicMock(spec=torch.optim.Optimizer))
                if mean_loss:
                    with patch.object(
                        pp.stage, "take_global_inputs", return_value=[whole, whole]
                    ):
                        losses = pp.progress(iter([]))
                    assert losses is not None
                    self.assertEqual(losses.shape, (MICROBATCHES,))
                else:
                    with patch.object(
                        pp.stage,
                        "take_inputs",
                        return_value=[[part, part] for part in parts],
                    ):
                        self.assertIsNone(pp.progress(iter([])))
                for (name, parameter), expected in zip(
                    model.named_parameters(), reference.parameters()
                ):
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, expected.grad)

    def test_1f1b_gradients_are_num_microbatches_times_rail(self, *_mocks: Any) -> None:
        # One mean-loss model and batch: 1f1b sums the microbatch losses, rail
        # with scale_grads averages them.
        torch.manual_seed(1)
        parts = [torch.randn(ROWS, WIDTH) for _ in range(MICROBATCHES)]
        whole = torch.cat(parts)
        grads: Dict[MaglevSchedule, List[torch.Tensor]] = {}
        for schedule in (MaglevSchedule.ONE_F_ONE_B, MaglevSchedule.RAIL):
            model = _model(sparse=True)
            pp = MaglevPP(model, _config(schedule), mock_groups())
            pp.build(MagicMock(spec=torch.optim.Optimizer))
            if schedule is MaglevSchedule.RAIL:
                feed = patch.object(
                    pp.stage, "take_global_inputs", return_value=[whole, whole]
                )
            else:
                feed = patch.object(
                    pp.stage,
                    "take_inputs",
                    return_value=[[part, part] for part in parts],
                )
            with feed:
                pp.progress(iter([]))
            grads[schedule] = []
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"{schedule.value}: {name} has no gradient")
                grads[schedule].append(parameter.grad)
        for one_f_one_b, rail in zip(
            grads[MaglevSchedule.ONE_F_ONE_B], grads[MaglevSchedule.RAIL]
        ):
            torch.testing.assert_close(one_f_one_b, MICROBATCHES * rail)


# Each W right after its I on every stage, unlike the derived order's lag.
_SPLIT_1F1B_P3_M4 = (
    "F0 F1 F2 I0 W0 F3 I1 W1 I2 W2 I3 W3",
    "F0 F1 I0 W0 F2 I1 W1 F3 I2 W2 I3 W3",
    "F0 I0 W0 F1 I1 W1 F2 I2 W2 F3 I3 W3",
)


def _run_three_stage_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    schedule: MaglevSchedule,
    overrides: Dict[str, Any],
) -> None:
    """Three one-rank stages through MaglevPP against the whole batch: under
    MaglevRail a mean loss with ``scale_grads`` seeded once on the last stage,
    under Maglev1F1B a summed loss over its four raw microbatches."""
    if world_size != 3:
        raise ValueError("This case requires three stages of one rank")
    rail = schedule is MaglevSchedule.RAIL
    torch.manual_seed(0)
    model = ChainModel(num_layers=3, mean_loss=rail)
    reference = copy.deepcopy(model)
    batch = input_for(0, 0, torch.device("cpu"))
    reference(batch)[0].backward()
    config = MaglevPPConfig(
        schedule=schedule,
        num_microbatches=4,
        layers_per_stage=(1, 1, 1),
        stage_size=1,
        **overrides,
    )
    groups = MaglevProcessGroups.from_scratch(stage_size=1, num_stages=3)
    with patch("torchrec.distributed.maglev.stage.InputDistDriver") as driver:
        pp = MaglevPP(model, config, groups)
    pp.to(torch.device("cpu"))
    # A mock optimizer leaves the weights, and so the gradients, comparable.
    pp.build(MagicMock(spec=torch.optim.Optimizer))
    driver.return_value.take.return_value = (
        [[batch]] if rail else microbatches(batch, 4)
    )
    losses = pp.progress(iter([]))
    if rail and pp.stage.is_last and (losses is None or losses.shape != (4,)):
        raise AssertionError(f"expected four microbatch losses, got {losses}")
    if not rail and losses is not None:
        raise AssertionError(f"Maglev1F1B returned {losses}")
    assert_reduced_layer(pp.stage._layers[0], reference[pp.stage.stage_index])


@unittest.skipIf(partial_devices(3), "each rank binds the CUDA device of its index")
class ThreeStageGlooTest(unittest.TestCase):
    def test_every_schedule_trains_like_the_whole_batch(self) -> None:
        for schedule, overrides in (
            (MaglevSchedule.ONE_F_ONE_B, {}),
            (MaglevSchedule.RAIL, {}),
            (MaglevSchedule.RAIL, {"dense_schedule": "zero_bubble"}),
            (MaglevSchedule.RAIL, {"schedule_actions": _SPLIT_1F1B_P3_M4}),
        ):
            with self.subTest(schedule=schedule, **overrides):
                run_local_multi_process_func(
                    _run_three_stage_case,
                    world_size=3,
                    backend="gloo",
                    schedule=schedule,
                    overrides=overrides,
                )
