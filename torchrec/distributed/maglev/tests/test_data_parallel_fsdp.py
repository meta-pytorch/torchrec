#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maglev1F1B reduction with ``fully_shard`` units in both the sparse and the
dense halves, without (zero2) or with (zero3) resharding after each forward,
against an averaged reference, and the FSDP2 pass reduction."""

from __future__ import annotations

import copy
import unittest
from typing import Any, cast
from unittest.mock import call, MagicMock, patch

import torch
import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import DDPMeshInfo, FSDPMeshInfo
from torchrec.distributed.maglev.data_parallel import FSDP2PassReduction
from torchrec.distributed.maglev.pipeline import Maglev1F1B
from torchrec.distributed.maglev.stage import MaglevProcessGroups, StageWrapper
from torchrec.distributed.maglev.tests.utils import (
    as_chain_layer,
    assert_reduced_layer,
    averaged_passes,
    ChainModel,
    input_for,
    microbatches,
    partial_devices,
    set_use_extra,
    step_averaged_reference,
    two_by_two_stage,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)


def _run_reduction_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    dp_mode: str,
    cpu: bool = False,
) -> None:
    """2 stages x 2 replicas against an averaged single-process reference.

    ``cpu`` runs on Gloo; either way the input driver is stubbed. ``dp_mode``
    wraps with ``fully_shard`` without (``zero2``) or with (``zero3``)
    resharding after each forward. ``zero2`` must run one reduction per unit
    per pass, after its last microbatch backward, and all-gather each unit once
    per pass; ``zero3`` must reduce-scatter each unit per microbatch and
    all-gather it again within the pass.
    """
    if world_size != 4:
        raise ValueError("This reduction reference requires two stages/two replicas")
    device = torch.device("cpu") if cpu else ctx.device
    torch.manual_seed(122)
    model = ChainModel().to(device)
    reference = copy.deepcopy(model)
    stage, driver = two_by_two_stage(model, device)
    mesh = DeviceMesh.from_group(stage.stage_pg, device.type)
    layer = as_chain_layer(stage._layers[0])
    units = (layer.sparse_half().frontend, layer.dense_half())
    for unit in units:
        fully_shard(unit, mesh=mesh, reshard_after_forward=dp_mode == "zero3")
    # Citrine C2: foreach combines the real parameter updates into tensor batches.
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    pipeline = Maglev1F1B(stage, optimizer, num_microbatches=4)

    for _raw in averaged_passes(
        stage,
        reference,
        reference_optimizer,
        driver,
        device,
        divergent=False,
        num_microbatches=pipeline.num_microbatches,
    ):
        with (
            patch.object(
                fsdp_param_group,
                "foreach_reduce",
                wraps=fsdp_param_group.foreach_reduce,
            ) as reduce,
            patch.object(
                fsdp_param_group,
                "foreach_all_gather",
                wraps=fsdp_param_group.foreach_all_gather,
            ) as all_gather,
        ):
            pipeline.progress(iter([]))
        per_unit = pipeline.num_microbatches if dp_mode == "zero3" else 1
        if reduce.call_count != len(units) * per_unit:
            raise AssertionError(
                f"Expected {per_unit} sparse-frontend and {per_unit} dense "
                f"reductions per pass, got {reduce.call_count}"
            )
        if dp_mode == "zero2" and all_gather.call_count != len(units):
            raise AssertionError(
                f"Expected one all-gather per unit per pass, got {all_gather.call_count}"
            )
        if dp_mode == "zero3" and all_gather.call_count <= len(units):
            raise AssertionError(
                "Expected a resharded unit to all-gather again within the pass, "
                f"got {all_gather.call_count} all-gathers"
            )


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires four allocated CUDA devices")
class PassFSDPTest(unittest.TestCase):
    def _run(self, dp_mode: str) -> None:
        run_local_multi_process_func(_run_reduction_case, world_size=4, dp_mode=dp_mode)

    def test_1f1b_zero2_sharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero2")

    def test_1f1b_zero3_resharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero3")


@unittest.skipIf(partial_devices(4), "each rank binds the CUDA device of its index")
class PassFSDPGlooTest(unittest.TestCase):
    """The 4-GPU cases on CPU: 2 stages x 2 replicas over Gloo."""

    def _run(self, dp_mode: str) -> None:
        run_local_multi_process_func(
            _run_reduction_case,
            world_size=4,
            backend="gloo",
            dp_mode=dp_mode,
            cpu=True,
        )

    def test_1f1b_zero2_sharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero2")

    def test_1f1b_zero3_resharded_sparse_frontend_and_dense(self) -> None:
        self._run("zero3")


def _run_unused_parameter_case(
    ctx: SingleProcessContext, rank: int, world_size: int
) -> None:
    """One stage x two replicas under fully_shard() on Gloo; replica 1 skips ``extra``.

    With ``reduce_scatter_unused_params`` the replicas reduce-scatter the same
    parameter set, and the result matches the averaged reference.
    """
    if world_size != 2:
        raise ValueError("This case requires one stage with two replicas")
    device = torch.device("cpu")
    torch.manual_seed(122)
    model = ChainModel()
    reference = copy.deepcopy(model)
    groups = MaglevProcessGroups.from_scratch(stage_size=2, num_stages=1)
    with patch("torchrec.distributed.maglev.stage.InputDistDriver") as driver:
        stage = StageWrapper(model, [2], stage_size=2, process_groups=groups)
    del model[:]
    stage.to(device)
    mesh = DeviceMesh.from_group(groups.stage_pg, device.type)
    for module in stage._layers:
        layer = as_chain_layer(module)
        for unit in (layer.sparse_half().frontend, layer.dense_half()):
            fully_shard(unit, mesh=mesh, reshard_after_forward=False)
            cast(Any, unit).set_reduce_scatter_unused_params(True)
    set_use_extra(stage.module, stage.position == 0)
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    pipeline = Maglev1F1B(stage, optimizer, num_microbatches=4)
    for pass_index in range(3):
        step_averaged_reference(
            reference, reference_optimizer, pass_index, device, divergent=True
        )
        value = input_for(stage.position, pass_index, device)
        driver.return_value.take.return_value = microbatches(value, 4, 2)
        pipeline.progress(iter([]))
        for index, layer in enumerate(stage._layers):
            assert_reduced_layer(layer, reference[index])


@unittest.skipIf(partial_devices(2), "each rank binds the CUDA device of its index")
class PassFSDPUnusedParameterTest(unittest.TestCase):
    def test_reduce_scatter_unused_params_matches_the_reference(self) -> None:
        run_local_multi_process_func(
            _run_unused_parameter_case, world_size=2, backend="gloo"
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
