#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maglev1F1B and MaglevRail (both dense schedules) with ``fully_shard`` units in
the sparse and dense halves, without (zero2) or with (zero3) resharding after
each forward, against an averaged reference; and the FSDP2 pass reduction."""

from __future__ import annotations

import copy
import re
import unittest
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, cast
from unittest.mock import call, MagicMock, patch

import torch
import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo, FSDPMeshInfo
from torch.distributed.pipelining._backward import _get_grad_fn_or_grad_acc
from torchrec.distributed.maglev.checks import graph_nodes
from torchrec.distributed.maglev.data_parallel import FSDP2PassReduction
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    MaglevPipelineBase,
    MaglevRail,
)
from torchrec.distributed.maglev.stage import (
    MaglevPassState,
    MaglevProcessGroups,
    StageWrapper,
)
from torchrec.distributed.maglev.tests.utils import (
    as_chain_layer,
    assert_reduced_layer,
    averaged_passes,
    ChainModel,
    input_for,
    microbatches,
    partial_devices,
    rail_averaged_passes,
    set_use_extra,
    split_rail,
    step_averaged_reference,
    two_by_two_stage,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)

# MaglevRail cells, by the dense_schedule they pass.
_RAIL_SCHEDULES = {"rail_1f1b": "1f1b", "rail_zero_bubble": "zero_bubble"}


def _counting(name: str) -> AbstractContextManager[MagicMock]:
    return patch.object(fsdp_param_group, name, wraps=getattr(fsdp_param_group, name))


def _check_1f1b_pass(pipeline: Maglev1F1B, dp_mode: str, num_units: int) -> None:
    """``zero2`` reduces each unit once per pass and all-gathers it once;
    ``zero3`` reduce-scatters each unit per microbatch and all-gathers it again."""
    with (
        _counting("foreach_reduce") as reduce,
        _counting("foreach_all_gather") as all_gather,
    ):
        pipeline.progress(iter([]))
    per_unit = pipeline.num_microbatches if dp_mode == "zero3" else 1
    if reduce.call_count != num_units * per_unit:
        raise AssertionError(
            f"Expected {per_unit} sparse-frontend and {per_unit} dense "
            f"reductions per pass, got {reduce.call_count}"
        )
    if dp_mode == "zero2" and all_gather.call_count != num_units:
        raise AssertionError(
            f"Expected one all-gather per unit per pass, got {all_gather.call_count}"
        )
    if dp_mode == "zero3" and all_gather.call_count <= num_units:
        raise AssertionError(
            "Expected a resharded unit to all-gather again within the pass, "
            f"got {all_gather.call_count} all-gathers"
        )


def _check_rail_pass(
    pipeline: MaglevRail, raw: torch.Tensor, reductions: int
) -> list[torch.Tensor]:
    """Run one pass, which must run ``reductions`` reduce-scatters; return the
    seam gradients, read just before the sparse backward releases them."""
    seam_gradients: list[list[torch.Tensor]] = []
    sparse_backward = pipeline.stage.sparse_backward_global

    def capture_seam_gradients(state: MaglevPassState) -> None:
        gradients: list[torch.Tensor] = []
        for seams in state.seams:
            seam = seams[0]
            gradient = seam.grad if isinstance(seam, torch.Tensor) else None
            if gradient is None:
                raise AssertionError("Missing dense-input seam gradient")
            gradients.append(gradient.clone())
        seam_gradients.append(gradients)
        sparse_backward(state)

    with (
        _counting("foreach_reduce") as reduce,
        patch.object(pipeline.stage, "sparse_backward_global", capture_seam_gradients),
    ):
        pipeline.progress(iter([raw]))
    if reduce.call_count != reductions:
        raise AssertionError(
            f"Expected {reductions} reductions in the pass, got {reduce.call_count}"
        )
    if len(seam_gradients) != 1:
        raise AssertionError("Expected one whole-pass sparse backward")
    return seam_gradients[0]


def _run_reduction_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    dp_mode: str,
    schedule: str = "1f1b",
    cpu: bool = False,
) -> None:
    """2 stages x 2 replicas against an averaged single-process reference.

    ``cpu`` runs on Gloo. ``dp_mode`` wraps the sparse frontend and the dense
    half with ``fully_shard`` without (``zero2``) or with (``zero3``) resharding
    after each forward. MaglevRail under ``zero3`` reduce-scatters the sparse
    frontend once, after the pass's one sparse backward, and the dense half per
    microbatch; ``rail_zero_bubble`` must refuse ``zero3``.
    """
    if world_size != 4:
        raise ValueError("This reduction reference requires two stages/two replicas")
    device = torch.device("cpu") if cpu else ctx.device
    torch.manual_seed(122)
    model = ChainModel().to(device)
    reference = copy.deepcopy(model)
    stage, driver = two_by_two_stage(
        model, device, stub_driver=cpu or schedule == "1f1b"
    )
    mesh = DeviceMesh.from_group(stage.stage_pg, device.type)
    layer = as_chain_layer(stage._layers[0])
    units = (layer.sparse_half().frontend, layer.dense_half())
    for unit in units:
        fully_shard(unit, mesh=mesh, reshard_after_forward=dp_mode == "zero3")
    # Citrine C2: foreach combines the real parameter updates into tensor batches.
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    num_microbatches = 4

    if schedule == "1f1b":
        pipeline = Maglev1F1B(stage, optimizer, num_microbatches)
        for _raw in averaged_passes(
            stage,
            reference,
            reference_optimizer,
            driver,
            device,
            divergent=False,
            num_microbatches=num_microbatches,
        ):
            _check_1f1b_pass(pipeline, dp_mode, len(units))
        return

    dense_schedule = _RAIL_SCHEDULES[schedule]
    if dense_schedule == "zero_bubble" and dp_mode == "zero3":
        try:
            MaglevRail(
                stage, optimizer, num_microbatches, dense_schedule=dense_schedule
            )
        except ValueError as error:
            if "reshard_after_forward=False" not in str(error):
                raise
            return
        raise AssertionError("zero-bubble MaglevRail accepted a ZeRO-3 module")
    rail = MaglevRail(stage, optimizer, num_microbatches, dense_schedule=dense_schedule)
    reductions = 1 + num_microbatches if dp_mode == "zero3" else 2
    reference_seams: list[torch.Tensor] = []

    def retain_reference_seam(
        _module: torch.nn.Module, _args: Any, output: torch.Tensor
    ) -> None:
        output.retain_grad()
        reference_seams.append(output)

    as_chain_layer(reference[stage.stage_index]).sparse_half().register_forward_hook(
        retain_reference_seam
    )
    for raw in rail_averaged_passes(
        stage, reference, reference_optimizer, driver, device, divergent=False
    ):
        input_gradients = _check_rail_pass(rail, raw, reductions)
        expected = reference_seams[stage.position].grad
        reference_seams.clear()
        if expected is None:
            raise AssertionError("Missing independent reference input gradient")
        # Weight reductions average DP replicas. Input gradients before those
        # reductions remain local, while the ordinary reference loss averages two.
        torch.testing.assert_close(
            torch.cat(input_gradients), expected * 2, rtol=1e-4, atol=1e-5
        )


class _PassFSDPCells:
    def _run(self, dp_mode: str, schedule: str = "1f1b") -> None:
        raise NotImplementedError

    def test_1f1b_zero2_sharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero2")

    def test_1f1b_zero3_resharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero3")

    def test_rail_1f1b_zero2_sharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero2", "rail_1f1b")

    def test_rail_1f1b_zero3_resharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero3", "rail_1f1b")

    def test_rail_zero_bubble_zero2_sharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero2", "rail_zero_bubble")

    def test_rail_zero_bubble_zero3_is_refused(self) -> None:
        self._run("zero3", "rail_zero_bubble")


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires four allocated CUDA devices")
class PassFSDPTest(_PassFSDPCells, unittest.TestCase):
    def _run(self, dp_mode: str, schedule: str = "1f1b") -> None:
        run_local_multi_process_func(
            _run_reduction_case, world_size=4, dp_mode=dp_mode, schedule=schedule
        )


@unittest.skipIf(partial_devices(4), "each rank binds the CUDA device of its index")
class PassFSDPGlooTest(_PassFSDPCells, unittest.TestCase):
    """The 4-GPU cases on CPU: 2 stages x 2 replicas over Gloo."""

    def _run(self, dp_mode: str, schedule: str = "1f1b") -> None:
        run_local_multi_process_func(
            _run_reduction_case,
            world_size=4,
            backend="gloo",
            dp_mode=dp_mode,
            schedule=schedule,
            cpu=True,
        )


_ONE_STAGE_SCHEDULES: dict[str, Callable[..., MaglevPipelineBase]] = {
    "1f1b": Maglev1F1B,
    "rail_1f1b": MaglevRail,
    "rail_split": split_rail,
}


def _one_stage(
    reshard_after_forward: bool = False,
    reduce_unused: bool = True,
    with_unused: bool = False,
    divergent: bool = True,
) -> tuple[StageWrapper, MagicMock, ChainModel]:
    """One two-layer stage x two replicas under ``fully_shard`` on Gloo; with
    ``divergent``, replica 1 skips ``extra``."""
    torch.manual_seed(122)
    model = ChainModel(unused=with_unused)
    reference = copy.deepcopy(model)
    groups = MaglevProcessGroups.from_scratch(stage_size=2, num_stages=1)
    with patch("torchrec.distributed.maglev.stage.InputDistDriver") as driver:
        stage = StageWrapper(model, [2], stage_size=2, process_groups=groups)
    del model[:]
    stage.to(torch.device("cpu"))
    mesh = DeviceMesh.from_group(groups.stage_pg, "cpu")
    for module in stage._layers:
        layer = as_chain_layer(module)
        for unit in (layer.sparse_half().frontend, layer.dense_half()):
            fully_shard(unit, mesh=mesh, reshard_after_forward=reshard_after_forward)
            if reduce_unused:
                cast(Any, unit).set_reduce_scatter_unused_params(True)
    set_use_extra(stage.module, stage.position == 0 or not divergent)
    return stage, driver, reference


def _run_one_stage_passes(
    stage: StageWrapper,
    driver: MagicMock,
    reference: ChainModel,
    pipeline: MaglevPipelineBase,
    divergent: bool,
) -> None:
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    device = torch.device("cpu")
    for pass_index in range(3):
        step_averaged_reference(
            reference, reference_optimizer, pass_index, device, divergent=divergent
        )
        value = input_for(stage.position, pass_index, device)
        driver.return_value.take.return_value = (
            microbatches(value, 4, 2)
            if isinstance(pipeline, Maglev1F1B)
            else [[value, value]]
        )
        pipeline.progress(iter([]))
        for index, layer in enumerate(stage._layers):
            assert_reduced_layer(layer, reference[index])


def _run_unused_parameter_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    schedule: str,
    with_unused: bool,
    reduce_unused: bool,
) -> None:
    """Without ``reduce_scatter_unused_params`` the replicas would reduce-scatter
    different parameter sets, so the first pass must refuse on both of them.
    With it, the result matches the averaged reference."""
    if world_size != 2:
        raise ValueError("This case requires one stage with two replicas")
    stage, driver, reference = _one_stage(
        reduce_unused=reduce_unused, with_unused=with_unused
    )
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    pipeline = _ONE_STAGE_SCHEDULES[schedule](stage, optimizer, num_microbatches=4)
    if reduce_unused:
        _run_one_stage_passes(stage, driver, reference, pipeline, divergent=True)
        return
    value = input_for(stage.position, 0, torch.device("cpu"))
    driver.return_value.take.return_value = [[value, value]]
    try:
        pipeline.progress(iter([]))
    except RuntimeError as error:
        expected = "set_reduce_scatter_unused_params|failed a structural check"
        if not re.search(expected, str(error)):
            raise
        return
    raise AssertionError("FSDP2 accepted a dense parameter without gradient")


@unittest.skipIf(partial_devices(2), "each rank binds the CUDA device of its index")
class PassFSDPUnusedParameterTest(unittest.TestCase):
    def _run(
        self, schedules: Sequence[str], with_unused: bool, reduce_unused: bool
    ) -> None:
        for schedule in schedules:
            with self.subTest(schedule=schedule):
                run_local_multi_process_func(
                    _run_unused_parameter_case,
                    world_size=2,
                    backend="gloo",
                    schedule=schedule,
                    with_unused=with_unused,
                    reduce_unused=reduce_unused,
                )

    def test_rank_divergent_parameter_is_refused_on_every_replica(self) -> None:
        self._run(("rail_1f1b", "rail_split"), with_unused=False, reduce_unused=False)

    def test_parameter_no_replica_uses_is_refused(self) -> None:
        self._run(("rail_1f1b", "rail_split"), with_unused=True, reduce_unused=False)

    def test_reduce_scatter_unused_params_matches_the_reference(self) -> None:
        self._run(tuple(_ONE_STAGE_SCHEDULES), with_unused=False, reduce_unused=True)


def _unsharded_graph_parameters(stage: StageWrapper) -> list[torch.Tensor]:
    return [
        fsdp_param._unsharded_param
        for module in stage._layers
        for group in cast(FSDPModule, as_chain_layer(module).dense_half())
        ._get_fsdp_state()
        ._fsdp_param_groups
        for fsdp_param in group.fsdp_params
    ]


def _run_zero3_graph_parameter_case(
    ctx: SingleProcessContext, rank: int, world_size: int
) -> None:
    """MaglevRail 1f1b over ZeRO-3: each forward reshards, re-registering the
    sharded parameters, so the weights the whole-backward check walks to must be
    the unsharded tensors the graph holds. One stage: no p2p to hang on."""
    stage, driver, reference = _one_stage(reshard_after_forward=True, divergent=False)
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    pipeline = MaglevRail(stage, optimizer, num_microbatches=4)
    checks = stage._checks
    check = checks.before_whole_backward
    checked: list[int] = []

    def check_graph_parameters(
        roots: Sequence[torch.Tensor],
        weights: Callable[[], Sequence[torch.nn.Parameter]],
    ) -> None:
        parameters = weights()
        expected = _unsharded_graph_parameters(stage)
        if sorted(map(id, parameters)) != sorted(map(id, expected)):
            raise AssertionError(
                "dense_parameters() is not the unsharded parameters the graph holds"
            )
        visited = graph_nodes([_get_grad_fn_or_grad_acc(root) for root in roots])
        unreached = [
            p for p in parameters if _get_grad_fn_or_grad_acc(p) not in visited
        ]
        if unreached:
            raise AssertionError(f"{len(unreached)} dense parameters are unreached")
        checked.append(len(parameters))
        check(roots, weights)

    with patch.object(checks, "before_whole_backward", check_graph_parameters):
        _run_one_stage_passes(stage, driver, reference, pipeline, divergent=False)
    if not checked:
        raise AssertionError("The whole-backward check never ran")


@unittest.skipIf(partial_devices(2), "each rank binds the CUDA device of its index")
class RailZero3GraphParameterTest(unittest.TestCase):
    def test_rail_1f1b_zero3_checks_the_unsharded_graph_parameters(self) -> None:
        run_local_multi_process_func(
            _run_zero3_graph_parameter_case, world_size=2, backend="gloo"
        )


def _local(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if to_local is not None else tensor


def _run_dense_parameters_after_pass_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    schedule: str,
    reshard_after_forward: bool,
) -> None:
    """Between passes, ``dense_parameters()`` is the registered parameters, all
    with live storage, although the unsharded tensors the last backward's graph
    held have been freed."""
    stage, driver, reference = _one_stage(
        reshard_after_forward=reshard_after_forward, divergent=False
    )
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    pipeline = _ONE_STAGE_SCHEDULES[schedule](stage, optimizer, num_microbatches=4)
    _run_one_stage_passes(stage, driver, reference, pipeline, divergent=False)
    if not any(
        _local(p).numel() and not _local(p).untyped_storage().size()
        for p in stage._graph_dense_parameters()
    ):
        raise AssertionError("FSDP2 freed no unsharded parameter after the pass")
    registered = [
        p
        for module in stage._layers
        for p in as_chain_layer(module).dense_half().parameters()
        if p.requires_grad
    ]
    parameters = stage.dense_parameters()
    if [id(p) for p in parameters] != [id(p) for p in registered]:
        raise AssertionError("dense_parameters() is not the registered parameters")
    for p in parameters:
        local = _local(p).detach()
        if local.numel() and not local.untyped_storage().size():
            raise AssertionError("dense_parameters() returned freed storage")
        torch.testing.assert_close(local.clone(), local, rtol=0, atol=0)


@unittest.skipIf(partial_devices(2), "each rank binds the CUDA device of its index")
class DenseParametersAfterPassTest(unittest.TestCase):
    def test_registered_parameters_with_live_storage_under_zero2_and_zero3(
        self,
    ) -> None:
        for schedule in ("1f1b", "rail_1f1b"):
            for reshard_after_forward in (False, True):
                with self.subTest(
                    schedule=schedule, reshard_after_forward=reshard_after_forward
                ):
                    run_local_multi_process_func(
                        _run_dense_parameters_after_pass_case,
                        world_size=2,
                        backend="gloo",
                        schedule=schedule,
                        reshard_after_forward=reshard_after_forward,
                    )


def _fsdp_module(*groups: MagicMock) -> MagicMock:
    module = MagicMock(spec=FSDPModule)
    module._get_fsdp_state.return_value._fsdp_param_groups = list(groups)
    return module


def _group(
    mesh_info: type[Any] = FSDPMeshInfo, reshard_after_forward: bool = False
) -> MagicMock:
    return MagicMock(
        reduce_scatter_unused_params=False,
        mesh_info=MagicMock(spec=mesh_info),
        post_forward_mesh_info=(
            MagicMock(spec=FSDPMeshInfo) if reshard_after_forward else None
        ),
    )


class FSDP2PassReductionTest(unittest.TestCase):
    def test_holds_gradient_sync_for_the_pass(self) -> None:
        group = _group()
        fsdp_module = _fsdp_module(group)
        state = fsdp_module._get_fsdp_state.return_value
        reduction = FSDP2PassReduction([fsdp_module])

        with self.assertRaisesRegex(RuntimeError, "pass failed"):
            with reduction.accumulate():
                fsdp_module.set_requires_gradient_sync.assert_called_once_with(
                    False, recurse=False
                )
                raise RuntimeError("pass failed")
        # Re-armed even though the pass raised.
        self.assertEqual(
            fsdp_module.set_is_last_backward.call_args_list, [call(False), call(True)]
        )

        reduction.reduce()
        group.post_backward.assert_called_once_with()
        state._root_post_backward_final_callback.assert_called_once_with()

    def test_leaves_reshard_after_forward_to_the_caller(self) -> None:
        sharded = _fsdp_module(_group(FSDPMeshInfo))
        replicated = _fsdp_module(_group(DDPMeshInfo))
        reduction = FSDP2PassReduction([sharded, replicated])
        with reduction.accumulate():
            pass
        reduction.reduce()
        reduction.close()
        sharded.set_reshard_after_forward.assert_not_called()
        replicated.set_reshard_after_forward.assert_not_called()

    def test_resharding_modules_reduce_per_microbatch(self) -> None:
        held_group = _group()
        resharded_group = _group(reshard_after_forward=True)
        held = _fsdp_module(held_group)
        resharded = _fsdp_module(resharded_group)
        reduction = FSDP2PassReduction([held, resharded])

        with reduction.accumulate():
            held.set_requires_gradient_sync.assert_called_once_with(
                False, recurse=False
            )
        reduction.reduce()

        resharded.set_requires_gradient_sync.assert_not_called()
        resharded.set_is_last_backward.assert_not_called()
        resharded.set_reshard_after_backward.assert_not_called()
        resharded_group.post_backward.assert_not_called()
        resharded_state = resharded._get_fsdp_state.return_value
        resharded_state._root_post_backward_final_callback.assert_not_called()
        held_group.post_backward.assert_called_once_with()

    def test_refuses_held_and_resharding_modules_in_one_state_context(self) -> None:
        held = _fsdp_module(_group())
        resharded = _fsdp_module(_group(reshard_after_forward=True))
        resharded._get_fsdp_state.return_value._state_ctx = (
            held._get_fsdp_state.return_value._state_ctx
        )
        with self.assertRaisesRegex(ValueError, "share a state context"):
            FSDP2PassReduction([held, resharded])

    def test_reduces_each_state_and_state_context_once(self) -> None:
        # fully_shard([a, b]) maps two modules onto one state; nested FSDP2
        # roots share one state context, whose callback walks every state.
        shared_group, inner_group = _group(), _group()
        first = _fsdp_module(shared_group)
        second = MagicMock(spec=FSDPModule)
        second._get_fsdp_state.return_value = first._get_fsdp_state.return_value
        inner = _fsdp_module(inner_group)
        outer_state = first._get_fsdp_state.return_value
        inner_state = inner._get_fsdp_state.return_value
        inner_state._state_ctx = outer_state._state_ctx

        FSDP2PassReduction([first, second, inner]).reduce()

        shared_group.post_backward.assert_called_once_with()
        inner_group.post_backward.assert_called_once_with()
        outer_state._root_post_backward_final_callback.assert_called_once_with()
        inner_state._root_post_backward_final_callback.assert_not_called()
