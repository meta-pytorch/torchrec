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
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    MaglevRail1F1B,
    MaglevRailZB,
)
from torchrec.distributed.maglev.process_groups import MaglevProcessGroups
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
    (MaglevSchedule.RAIL_1F1B, {}),
    (MaglevSchedule.RAIL_ZB, {}),
    (MaglevSchedule.RAIL_CUSTOM, {"schedule_actions": ("F0 I0 F1 W0 I1 W1",)}),
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
        for schedule, expected in (
            (MaglevSchedule.RAIL_1F1B, MaglevRail1F1B),
            (MaglevSchedule.RAIL_ZB, MaglevRailZB),
        ):
            with self.subTest(schedule=schedule):
                pp = self._pp(schedule)
                pp.build(MagicMock(spec=torch.optim.Optimizer))
                self.assertIs(type(pp.pipeline), expected)

    def test_rail_custom_builds_the_custom_schedule(self, *_mocks: Any) -> None:
        pp = self._pp(
            MaglevSchedule.RAIL_CUSTOM, schedule_actions=("F0 I0 W0 F1 I1 W1",)
        )
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        self.assertIs(type(pp.pipeline), MaglevRailCustom)

    def test_config_converts_names_and_lists(self, *_mocks: Any) -> None:
        config = MaglevPPConfig(
            schedule="rail_zb",  # pyrefly: ignore[bad-argument-type]
            num_microbatches=MICROBATCHES,
            layers_per_stage=[2],  # pyrefly: ignore[bad-argument-type]
            stage_size=1,
        )
        self.assertIs(config.schedule, MaglevSchedule.RAIL_ZB)
        self.assertIs(MaglevSchedule("rail_custom"), MaglevSchedule.RAIL_CUSTOM)
        self.assertEqual(config.layers_per_stage, (2,))
        self.assertTrue(config.scale_grads)
        with self.assertRaisesRegex(ValueError, "expected one of"):
            _config("zero-bubble")  # pyrefly: ignore[bad-argument-type]
        with self.assertRaisesRegex(ValueError, "'rail' was split into 'rail_1f1b'"):
            _config("rail")  # pyrefly: ignore[bad-argument-type]

    def test_config_refusals(self, *_mocks: Any) -> None:
        for schedule in (
            MaglevSchedule.ONE_F_ONE_B,
            MaglevSchedule.RAIL_1F1B,
            MaglevSchedule.RAIL_ZB,
        ):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "under schedule 'rail_custom'"):
                    _config(schedule, schedule_actions=("F0 I0 F1 W0 I1 W1",))
        with self.assertRaisesRegex(ValueError, "'rail_custom' runs the dense order"):
            _config(MaglevSchedule.RAIL_CUSTOM)
        with self.assertRaisesRegex(ValueError, "1 stages"):
            _config(MaglevSchedule.RAIL_CUSTOM, schedule_actions=("F0", "F0"))
        with self.assertRaisesRegex(ValueError, "at least the stage count"):
            _config(
                MaglevSchedule.RAIL_1F1B, num_microbatches=1, layers_per_stage=(1, 1)
            )
        with self.assertRaisesRegex(ValueError, "one entry per stage"):
            _config(MaglevSchedule.RAIL_1F1B, layers_per_stage=())
        with self.assertRaises(ScheduleError):
            self._pp(MaglevSchedule.RAIL_CUSTOM, schedule_actions=("F0 I0 W0",))

    def test_input_batch_size_follows_the_schedule(self, *_mocks: Any) -> None:
        for schedule, overrides in _SCHEDULE_CASES[1:]:
            rail = self._pp(schedule, **overrides)
            self.assertEqual(rail.input_batch_size(ROWS), ROWS * MICROBATCHES)
        one_f_one_b = self._pp(MaglevSchedule.ONE_F_ONE_B)
        self.assertEqual(one_f_one_b.input_batch_size(ROWS), ROWS)

    def test_each_stage_builds_its_own_row(self, get_rank: Any, _driver: Any) -> None:
        rows = ScheduleValidator.generate(2, MICROBATCHES)
        get_rank.return_value = 1
        pp = MaglevPP(
            _model(),
            _config(
                MaglevSchedule.RAIL_CUSTOM,
                layers_per_stage=(1, 1),
                schedule_actions=rows,
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
                        MaglevSchedule.RAIL_CUSTOM,
                        layers_per_stage=(1, 1),
                        schedule_actions=rows,
                    ),
                    mock_groups(2),
                )
        stage.assert_not_called()

    def test_build_takes_a_subclass_of_the_selected_schedule(
        self, *_mocks: Any
    ) -> None:
        class _Hooked(Maglev1F1B):
            def __init__(self, *args: Any, hook: object, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.hook = hook

        hook = object()
        pp = self._pp(MaglevSchedule.ONE_F_ONE_B, scale_grads=False)
        pp.build(MagicMock(spec=torch.optim.Optimizer), schedule_cls=_Hooked, hook=hook)
        pipeline = cast(_Hooked, pp.pipeline)
        self.assertIs(type(pipeline), _Hooked)
        self.assertIs(pipeline.hook, hook)
        self.assertEqual(pipeline.num_microbatches, MICROBATCHES)
        self.assertEqual(pipeline._loss_scale, 1.0)

    def test_build_refuses_a_class_the_schedule_does_not_select(
        self, *_mocks: Any
    ) -> None:
        pp = self._pp(MaglevSchedule.ONE_F_ONE_B)
        with self.assertRaisesRegex(TypeError, "must subclass Maglev1F1B"):
            pp.build(MagicMock(spec=torch.optim.Optimizer), schedule_cls=MaglevRail1F1B)
        with self.assertRaisesRegex(TypeError, "need a schedule_cls"):
            pp.build(MagicMock(spec=torch.optim.Optimizer), hook=object())
        self.assertIsNone(pp._pipeline)

    def test_close_releases_the_schedule(self, *_mocks: Any) -> None:
        pp = self._pp(MaglevSchedule.RAIL_1F1B)
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
        pp = self._pp(MaglevSchedule.RAIL_1F1B)
        with patch.object(pp.stage, "to") as to:
            self.assertIs(pp.to(torch.device("cpu")), pp)
        to.assert_called_once_with(torch.device("cpu"))

    def test_progress_needs_build(self, *_mocks: Any) -> None:
        with self.assertRaisesRegex(RuntimeError, "build"):
            self._pp(MaglevSchedule.RAIL_1F1B).progress(iter([]))

    def _pass_grads(
        self,
        model: _Model,
        schedule: MaglevSchedule,
        parts: List[torch.Tensor],
        **overrides: Any,
    ) -> List[torch.Tensor]:
        """Each parameter's gradient after one pass of ``model`` over ``parts``."""
        whole = torch.cat(parts)
        pp = MaglevPP(model, _config(schedule, **overrides), mock_groups())
        pp.build(MagicMock(spec=torch.optim.Optimizer))
        if schedule is MaglevSchedule.ONE_F_ONE_B:
            with patch.object(
                pp.stage, "take_inputs", return_value=[[part, part] for part in parts]
            ):
                self.assertIsNone(pp.progress(iter([])))
        else:
            with patch.object(pp.stage, "take_inputs", return_value=[[whole, whole]]):
                losses = pp.progress(iter([]))
            assert losses is not None
            self.assertEqual(losses.shape, (MICROBATCHES,))
        grads = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"{schedule.value}: {name} has no gradient")
            grads.append(parameter.grad)
        return grads

    def test_one_model_trains_like_the_whole_batch_under_every_schedule(
        self, *_mocks: Any
    ) -> None:
        # With scale_grads, mean-loss microbatches train on the whole batch's
        # mean; without it, summed-loss microbatches train on its sum.
        torch.manual_seed(1)
        parts = [torch.randn(ROWS, WIDTH) for _ in range(MICROBATCHES)]
        whole = torch.cat(parts)
        for scale_grads in (True, False):
            reference = _model(sparse=True, mean_loss=scale_grads)
            reference([whole, whole])[0].backward()
            for schedule, overrides in _SCHEDULE_CASES:
                with self.subTest(
                    schedule=schedule, scale_grads=scale_grads, **overrides
                ):
                    grads = self._pass_grads(
                        _model(sparse=True, mean_loss=scale_grads),
                        schedule,
                        parts,
                        scale_grads=scale_grads,
                        **overrides,
                    )
                    for grad, expected in zip(grads, reference.parameters()):
                        torch.testing.assert_close(grad, expected.grad)

    def test_switching_schedule_does_not_rescale_the_gradients(
        self, *_mocks: Any
    ) -> None:
        # One mean-loss model and batch: the schedules agree under either
        # setting, and scale_grads divides every gradient by num_microbatches.
        torch.manual_seed(1)
        parts = [torch.randn(ROWS, WIDTH) for _ in range(MICROBATCHES)]
        grads = {
            (schedule, scale_grads): self._pass_grads(
                _model(sparse=True), schedule, parts, scale_grads=scale_grads
            )
            for schedule in (MaglevSchedule.ONE_F_ONE_B, MaglevSchedule.RAIL_1F1B)
            for scale_grads in (True, False)
        }
        for scale_grads in (True, False):
            for one_f_one_b, rail in zip(
                grads[(MaglevSchedule.ONE_F_ONE_B, scale_grads)],
                grads[(MaglevSchedule.RAIL_1F1B, scale_grads)],
            ):
                torch.testing.assert_close(one_f_one_b, rail)
        for summed, scaled in zip(
            grads[(MaglevSchedule.ONE_F_ONE_B, False)],
            grads[(MaglevSchedule.ONE_F_ONE_B, True)],
        ):
            torch.testing.assert_close(summed, MICROBATCHES * scaled)


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
    """Three one-rank stages through MaglevPP against the whole batch: a mean
    loss over four microbatches, each seeded with a quarter on the last stage;
    the Rail schedules read the batch once, Maglev1F1B as four raw microbatches."""
    if world_size != 3:
        raise ValueError("This case requires three stages of one rank")
    rail = schedule is not MaglevSchedule.ONE_F_ONE_B
    torch.manual_seed(0)
    model = ChainModel(num_layers=3, mean_loss=True)
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
            (MaglevSchedule.RAIL_1F1B, {}),
            (MaglevSchedule.RAIL_ZB, {}),
            (MaglevSchedule.RAIL_CUSTOM, {"schedule_actions": _SPLIT_1F1B_P3_M4}),
        ):
            with self.subTest(schedule=schedule, **overrides):
                run_local_multi_process_func(
                    _run_three_stage_case,
                    world_size=3,
                    backend="gloo",
                    schedule=schedule,
                    overrides=overrides,
                )
