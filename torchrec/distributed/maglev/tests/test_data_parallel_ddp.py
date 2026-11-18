#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Maglev1F1B with a DistributedDataParallel stage: against an averaged
reference, the pass reduction's bucket priming, and the DDP configurations
the schedule refuses."""

import copy
import tempfile
import unittest
from collections import Counter
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.data_parallel import DDPPassReduction
from torchrec.distributed.maglev.pipeline import Maglev1F1B, Maglev1F1BRecvAhead
from torchrec.distributed.maglev.stage import StageWrapper
from torchrec.distributed.maglev.tests.utils import (
    as_chain_layer,
    assert_reduced_layer,
    averaged_passes,
    ChainModel,
    input_for,
    microbatches,
    mock_groups,
    partial_devices,
    set_use_extra,
    stage_mock,
    two_by_two_stage,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)

_SCHEDULES: tuple[type[Maglev1F1B], ...] = (Maglev1F1B, Maglev1F1BRecvAhead)


def _ddp(
    module: torch.nn.Module,
    process_group: dist.ProcessGroup | None,
    device: torch.device,
    gradient_as_bucket_view: bool,
    bucket_cap_mb: float | None = None,
) -> DistributedDataParallel:
    # The APS/DMP configuration, apart from the bucket settings under test.
    return DistributedDataParallel(
        module,
        device_ids=None if device.type == "cpu" else [device],
        process_group=process_group,
        find_unused_parameters=True,
        static_graph=False,
        gradient_as_bucket_view=gradient_as_bucket_view,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb,
    )


def _reduction(pipeline: Maglev1F1B) -> DDPPassReduction:
    reduction = pipeline._reduction
    if not isinstance(reduction, DDPPassReduction):
        raise AssertionError(f"expected a DDP reduction, got {type(reduction)}")
    return reduction


def _count_bucket_reductions(ddp: DistributedDataParallel) -> list[int]:
    """Record the buffer of each bucket the module all-reduces.

    By buffer, not ``bucket.index()``: a flush outside the backward reports
    every bucket under one index. DDP checks the hook's annotations against the
    real classes, so this module leaves them unpostponed.
    """
    reductions: list[int] = []

    def counting_allreduce(
        process_group: dist.ProcessGroup, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        reductions.append(bucket.buffer().untyped_storage().data_ptr())
        return allreduce_hook(process_group, bucket)

    ddp.register_comm_hook(ddp.process_group, counting_allreduce)
    return reductions


def _check_bucket_reductions(reductions: list[int]) -> int:
    """Each bucket was all-reduced once this pass; return the bucket count."""
    counts = Counter(reductions)
    if not counts or set(counts.values()) != {1}:
        raise AssertionError(
            f"bucket all-reduces in one pass: {sorted(counts.values())}, "
            "expected 1 per bucket"
        )
    reductions.clear()
    return len(counts)


def _run_ddp_case(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    gradient_as_bucket_view: bool,
    schedule: type[Maglev1F1B] = Maglev1F1B,
    cpu: bool = False,
    passes: int = 3,
) -> None:
    """2 stages x 2 replicas against an averaged single-process reference.

    ``cpu`` runs on Gloo. Either way the input driver is stubbed, and each
    stage's one layer consumes its microbatch of the raw batch as-is.
    """
    if world_size != 4:
        raise ValueError("This reduction reference requires two stages/two replicas")
    device = torch.device("cpu") if cpu else ctx.device
    torch.manual_seed(122)
    model = ChainModel(unused=True).to(device)
    reference = copy.deepcopy(model)
    stage, driver = two_by_two_stage(model, device)
    ddp = _ddp(stage.module, stage.stage_pg, device, gradient_as_bucket_view)
    reductions = _count_bucket_reductions(ddp)
    stage.module = ddp
    # Replica 0 alone runs `extra`; its gradient exists on one side only.
    set_use_extra(stage._layers[0], stage.position == 0)
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    pipeline = schedule(stage, optimizer, num_microbatches=4)

    for _raw in averaged_passes(
        stage,
        reference,
        reference_optimizer,
        driver,
        device,
        divergent=True,
        num_microbatches=pipeline.num_microbatches,
        passes=passes,
    ):
        with patch.object(
            DistributedDataParallel,
            "forward",
            autospec=True,
            side_effect=DistributedDataParallel.forward,
        ) as ddp_forward:
            pipeline.progress(iter([]))
        if ddp_forward.call_count != pipeline.num_microbatches:
            raise AssertionError(
                f"Expected {pipeline.num_microbatches} DDP forwards, got "
                f"{ddp_forward.call_count}"
            )
        _check_bucket_reductions(reductions)


@unittest.skipIf(torch.cuda.device_count() < 4, "Requires four allocated CUDA devices")
class PassDDPTest(unittest.TestCase):
    def test_1f1b_reduces_once_per_pass_with_bucket_views(self) -> None:
        run_local_multi_process_func(
            _run_ddp_case, world_size=4, gradient_as_bucket_view=True
        )

    def test_1f1b_reduces_once_per_pass_without_bucket_views(self) -> None:
        run_local_multi_process_func(
            _run_ddp_case, world_size=4, gradient_as_bucket_view=False
        )

    def test_recv_ahead_reduces_once_per_pass(self) -> None:
        run_local_multi_process_func(
            _run_ddp_case,
            world_size=4,
            gradient_as_bucket_view=True,
            schedule=Maglev1F1BRecvAhead,
        )


@unittest.skipIf(partial_devices(4), "each rank binds the CUDA device of its index")
class PassDDPGlooTest(unittest.TestCase):
    """The 4-GPU parity cases on CPU: 2 stages x 2 replicas over Gloo."""

    def _run(
        self, gradient_as_bucket_view: bool, schedule: type[Maglev1F1B] = Maglev1F1B
    ) -> None:
        run_local_multi_process_func(
            _run_ddp_case,
            world_size=4,
            backend="gloo",
            gradient_as_bucket_view=gradient_as_bucket_view,
            schedule=schedule,
            cpu=True,
        )

    def test_1f1b_reduces_once_per_pass_with_bucket_views(self) -> None:
        self._run(gradient_as_bucket_view=True)

    def test_1f1b_reduces_once_per_pass_without_bucket_views(self) -> None:
        self._run(gradient_as_bucket_view=False)

    def test_recv_ahead_reduces_once_per_pass(self) -> None:
        self._run(gradient_as_bucket_view=True, schedule=Maglev1F1BRecvAhead)

    def test_1f1b_flushes_multi_bucket_views(self) -> None:
        run_local_multi_process_func(
            _run_bucket_view_case, world_size=4, backend="gloo"
        )


class OneStageDDPCPUTest(unittest.TestCase):
    def _run(self, driver: Any) -> None:
        torch.manual_seed(122)
        model = ChainModel(unused=True)
        reference = copy.deepcopy(model)
        stage = StageWrapper(model, [2], 1, process_groups=mock_groups())
        stage.module = _ddp(
            stage.module,
            dist.group.WORLD,
            torch.device("cpu"),
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=0.01, foreach=True
        )
        pipeline = Maglev1F1B(stage, optimizer, num_microbatches=4)
        for pass_index in range(3):
            value = input_for(0, pass_index, torch.device("cpu"))
            reference_optimizer.zero_grad()
            reference(value)[0].backward()
            reference_optimizer.step()
            driver.return_value.take.return_value = microbatches(value, 4, 2)
            pipeline.progress(iter([]))
            for index, layer in enumerate(stage._layers):
                assert_reduced_layer(layer, reference[index])

    @patch("torchrec.distributed.maglev.stage.InputDistDriver")
    @patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
    def test_two_layers_on_one_ddp_stage_match_over_three_passes(
        self, _get_rank: Any, driver: Any
    ) -> None:
        # Two layers, each microbatch's sparse then dense halves, under bucket views.
        with tempfile.TemporaryDirectory() as temp_dir:
            dist.init_process_group(
                "gloo", init_method=f"file://{temp_dir}/store", rank=0, world_size=1
            )
            try:
                self._run(driver)
            finally:
                dist.destroy_process_group()


def _run_bucket_view_case(
    ctx: SingleProcessContext, rank: int, world_size: int
) -> None:
    """Maglev1F1B under bucket-view DDP, 2 stages x 2 replicas on Gloo.

    The bucket cap splits each stage's parameters over several buckets. After
    each pass ``unused`` must still be ``None``, every bucket must have been
    all-reduced exactly once, and the step must match the averaged reference.
    """
    if world_size != 4:
        raise ValueError("This reduction reference requires two stages/two replicas")
    device = torch.device("cpu")
    torch.manual_seed(122)
    model = ChainModel(unused=True)
    reference = copy.deepcopy(model)
    stage, driver = two_by_two_stage(model, device)
    ddp = _ddp(
        stage.module,
        stage.stage_pg,
        device,
        gradient_as_bucket_view=True,
        bucket_cap_mb=0.0001,
    )
    reductions = _count_bucket_reductions(ddp)
    stage.module = ddp
    layer = as_chain_layer(stage._layers[0])
    dense = layer.dense_half()
    # Replica 0 alone runs `extra`; replica 1's gradient for it comes from the flush.
    set_use_extra(layer, stage.position == 0)
    optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=True)
    pipeline = Maglev1F1B(stage, optimizer, num_microbatches=2)
    for _raw in averaged_passes(
        stage,
        reference,
        reference_optimizer,
        driver,
        device,
        divergent=True,
        num_microbatches=pipeline.num_microbatches,
    ):
        pipeline.progress(iter([]))
        if any(p.grad is not None for p in dense.require_unused().parameters()):
            raise AssertionError("an unused parameter has a gradient after the flush")
        buckets = _check_bucket_reductions(reductions)
        if buckets < 2:
            raise AssertionError(f"the bucket cap left {buckets} bucket")


@patch("torchrec.distributed.maglev.stage.InputDistDriver")
@patch("torchrec.distributed.maglev.stage.dist.get_rank", return_value=0)
class DDPFlushTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self._temp_dir.name}/store",
            rank=0,
            world_size=1,
        )

    def tearDown(self) -> None:
        dist.destroy_process_group()
        self._temp_dir.cleanup()

    @staticmethod
    def _stage() -> StageWrapper:
        torch.manual_seed(122)
        stage = StageWrapper(
            ChainModel(unused=True), [2], 1, process_groups=mock_groups()
        )
        stage.module = _ddp(
            stage.module,
            dist.group.WORLD,
            torch.device("cpu"),
            gradient_as_bucket_view=True,
        )
        return stage

    def _pipeline(self, driver: Any, stage: StageWrapper) -> Maglev1F1B:
        value = input_for(0, 0, torch.device("cpu"))
        driver.return_value.take.return_value = microbatches(value, 4, 2)
        optimizer = torch.optim.SGD(stage.module.parameters(), lr=0.01, foreach=True)
        return Maglev1F1B(stage, optimizer, num_microbatches=4)

    def test_the_flush_rearms_the_buffer_sync(
        self, _get_rank: Any, driver: Any
    ) -> None:
        # Every forward of the pass ran under no_sync, which clears it; the next
        # pass's first forward must sync the buffers again.
        stage = self._stage()
        self._pipeline(driver, stage).progress(iter([]))
        self.assertTrue(
            cast(DistributedDataParallel, stage.module).require_forward_param_sync
        )


def _ddp_mock(**overrides: Any) -> MagicMock:
    ddp = MagicMock(spec=DistributedDataParallel)
    ddp.static_graph = False
    ddp.find_unused_parameters = True
    ddp.mixed_precision = None
    ddp.require_manual_backward_finalization = False
    # Set in DDP.__init__, so the class spec does not carry them.
    ddp.reducer = MagicMock()
    ddp.gradient_as_bucket_view = False
    for name, value in overrides.items():
        setattr(ddp, name, value)
    ddp.modules.side_effect = lambda: iter((ddp,))
    return ddp


def _unflushable_ddp_cases() -> list[tuple[str, dict[str, Any], str]]:
    """DDP settings with which a hook-free flush after the pass trains wrongly."""
    in_backward = torch.nn.Parameter(torch.ones(1))
    vars(in_backward)["_in_backward_optimizers"] = [MagicMock()]
    return [
        ("static graph", {"static_graph": True}, "static_graph=False"),
        (
            "used parameters not found",
            {"find_unused_parameters": False},
            "find_unused_parameters=True",
        ),
        ("mixed_precision", {"mixed_precision": MagicMock()}, "mixed_precision"),
        (
            "optimizer in backward",
            {"_module_parameters": [in_backward]},
            "optimizers applied in backward",
        ),
    ]


class DDPValidationTest(unittest.TestCase):
    """What a schedule accepts from a DDP stage, checked when it is built."""

    def test_schedules_build_a_ddp_reduction(self) -> None:
        for schedule in _SCHEDULES:
            with self.subTest(schedule.__name__):
                stage = stage_mock()
                stage.module = _ddp_mock()
                pipeline = schedule(stage, MagicMock(spec=torch.optim.Optimizer), 2)
                self.assertIsInstance(pipeline._reduction, DDPPassReduction)
                self.assertEqual(pipeline._ddp_modules, [stage.module])

    def test_schedules_reject_ddp_the_flush_cannot_drive(self) -> None:
        for schedule in _SCHEDULES:
            for label, overrides, message in _unflushable_ddp_cases():
                with self.subTest(schedule=schedule.__name__, case=label):
                    stage = stage_mock()
                    stage.module = _ddp_mock(**overrides)
                    with self.assertRaisesRegex(ValueError, message):
                        schedule(stage, MagicMock(spec=torch.optim.Optimizer), 2)

    def test_schedule_rejects_ddp_mixed_with_fsdp2(self) -> None:
        layer = torch.nn.Linear(1, 1)
        layer.__class__ = type("FSDPLinear", (FSDPModule, torch.nn.Linear), {})
        ddp = _ddp_mock()
        ddp.modules.side_effect = lambda: iter((ddp, layer))
        stage = stage_mock()
        stage.module = ddp
        with self.assertRaisesRegex(ValueError, "not both"):
            Maglev1F1B(stage, MagicMock(spec=torch.optim.Optimizer), 2)

    def test_schedules_flush_ddp_once_after_the_last_backward(self) -> None:
        for schedule in _SCHEDULES:
            with self.subTest(schedule.__name__):
                stage = stage_mock()
                ddp = _ddp_mock()
                stage.module = ddp
                optimizer = MagicMock(spec=torch.optim.Optimizer)
                pipeline = schedule(stage, optimizer, 2)
                order = MagicMock()
                order.attach_mock(stage.take_inputs, "take_inputs")
                order.attach_mock(optimizer.zero_grad, "zero_grad")
                order.attach_mock(stage.drain_sends, "drain_sends")
                order.attach_mock(
                    ddp.reducer.prepare_for_backward, "prepare_for_backward"
                )
                order.attach_mock(ddp.reducer._delay_all_reduce, "delay_all_reduce")
                order.attach_mock(ddp.finalize_backward, "finalize_backward")
                order.attach_mock(optimizer.step, "step")
                with patch.object(pipeline, "_run_microbatches") as run:
                    order.attach_mock(run, "microbatches")
                    pipeline.progress(iter([]))
                self.assertEqual(
                    [entry[0] for entry in order.mock_calls],
                    [
                        "take_inputs",
                        "zero_grad",
                        "microbatches",
                        "drain_sends",
                        "prepare_for_backward",
                        "delay_all_reduce",
                        "step",
                    ],
                )
                ddp.reducer.prepare_for_backward.assert_called_once_with([])

                ddp.require_manual_backward_finalization = True
                ddp.reducer.reset_mock()
                with patch.object(pipeline, "_run_microbatches"):
                    pipeline.progress(iter([]))
                ddp.finalize_backward.assert_called_once_with()
