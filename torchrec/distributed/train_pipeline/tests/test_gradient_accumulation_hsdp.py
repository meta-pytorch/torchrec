#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The recording optimizer is SGD with lr=0; foreach would not change what is tested.
# @lint-ignore-every CITRINEAGENT missing_for_each_optimizer

import unittest
from collections.abc import Callable, Iterator
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.train_pipeline.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
    PartialWindowPolicy,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipeline

_WORLD_SIZE = 4
_K = 2
_GA_LOGGER = "torchrec.distributed.train_pipeline.gradient_accumulation"


def _fsdp_modules(model: nn.Module) -> list[FSDPModule]:
    return [m for m in model.modules() if isinstance(m, FSDPModule)]


def _param_groups(model: nn.Module) -> list[Any]:
    return [
        group
        for module in _fsdp_modules(model)
        for group in module._get_fsdp_state()._fsdp_param_groups
    ]


def _residency(model: nn.Module) -> list[bool]:
    return [group.is_sharded for group in _param_groups(model)]


def _integer_mlp(mesh: DeviceMesh, device: torch.device) -> nn.Sequential:
    """Build an MLP with the same integer weights on every rank, sharded as HSDP."""
    model = nn.Sequential(
        nn.Linear(8, 8, bias=False, device=device),
        nn.ReLU(),
        nn.Linear(8, 2, bias=False, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randint(-2, 3, param.shape, generator=generator, device=device)
            )
    # An explicit False lets suppression defer post-backward resharding.
    fully_shard(model[0], mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    return model


def _micro_batches(
    count: int, rank: int, device: torch.device, seed: int
) -> list[torch.Tensor]:
    """Integer inputs; ranks differ so the reductions are not trivial."""
    generator = torch.Generator(device=device).manual_seed(100 * seed + rank)
    return [
        torch.randint(
            -2, 3, (2, 8), generator=generator, device=device, dtype=torch.float32
        )
        for _ in range(count)
    ]


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, model: nn.Module) -> None:
        # Keep weights fixed so the same reference model stays valid across windows.
        super().__init__(model.parameters(), lr=0.0)
        self._model = model
        self.grads: list[list[torch.Tensor | None]] = []
        self.sharded: list[list[bool]] = []

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.grads.append(
            [
                None if p.grad is None else p.grad.to_local().clone()
                for p in self.param_groups[0]["params"]
            ]
        )
        self.sharded.append(_residency(self._model))
        return super().step(closure)


class _Pipeline(TrainPipeline[Any, torch.Tensor]):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer

    def progress(self, dataloader_iter: Iterator[torch.Tensor]) -> torch.Tensor:
        batch = next(dataloader_iter)
        self._optimizer.zero_grad()
        loss = self._model(batch).sum()
        loss.backward()
        self._optimizer.step()
        return loss


class _Setup:
    def __init__(
        self,
        ctx: MultiProcessContext,
        policy: PartialWindowPolicy = PartialWindowPolicy.STEP,
    ) -> None:
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
        model = _integer_mlp(mesh, ctx.device)
        reference = _integer_mlp(mesh, ctx.device)
        optimizer = _RecordingSGD(model)
        self.tc = unittest.TestCase()
        self.rank: int = ctx.rank
        self.device: torch.device = ctx.device
        self.mesh: DeviceMesh = mesh
        self.model: nn.Sequential = model
        self.reference: nn.Sequential = reference
        self._reference_params: list[torch.Tensor] = list(reference.parameters())
        self.optimizer: _RecordingSGD = optimizer
        self.wrapper: GradientAccumulationWrapper[Any, torch.Tensor] = (
            GradientAccumulationWrapper(
                _Pipeline(model, optimizer),
                optimizer,
                model,
                GradientAccumulationConfig(is_enabled=True, num_steps=_K),
                partial_window_policy=policy,
            )
        )

    def assert_fp32_reductions(self) -> None:
        """Call after the first forward, when FSDP2 has resolved reduction dtypes."""
        for group in _param_groups(self.model):
            self.tc.assertTrue(group._is_hsdp)
            self.tc.assertIsNone(group._reduce_dtype)
            for fsdp_param in group.fsdp_params:
                self.tc.assertEqual(fsdp_param.orig_dtype, torch.float32)
                self.tc.assertIsNone(fsdp_param.param_dtype)
                self.tc.assertIsNone(fsdp_param.reduce_dtype)

    def has_partial(self) -> bool:
        return self.wrapper._fsdp_sync_suppression.has_partial(
            self.wrapper._get_fsdp_targets()
        )

    def batches(self, count: int, seed: int) -> list[torch.Tensor]:
        return _micro_batches(count, self.rank, self.device, seed)

    def run(self, batches: list[torch.Tensor]) -> None:
        for batch in batches:
            self.wrapper.progress(iter([batch]))

    def run_window(self, seed: int) -> None:
        """Compare K=2 accumulated gradients with one backward over both microbatches."""
        window = self.batches(_K, seed)
        steps = len(self.optimizer.grads)
        self.run(window[:1])
        self.assert_fp32_reductions()
        # The first microbatch reduce-scatters but defers its all-reduce.
        self.tc.assertTrue(self.has_partial())
        self.tc.assertEqual(len(self.optimizer.grads), steps)
        self.run(window[1:])
        self.tc.assertFalse(self.has_partial())
        self.tc.assertEqual(len(self.optimizer.grads), steps + 1)
        self.assert_last_step_matches(window)

    def assert_last_step_matches(self, window: list[torch.Tensor]) -> None:
        self.reference(torch.cat(window)).sum().backward()
        expected = [
            cast(DTensor, p.grad).to_local().clone() for p in self._reference_params
        ]
        for param in self._reference_params:
            param.grad = None
        recorded = self.optimizer.grads[-1]
        self.tc.assertEqual(len(recorded), len(expected))
        for got, want in zip(recorded, expected):
            assert got is not None
            self.tc.assertGreater(int(torch.count_nonzero(want)), 0)
            self.tc.assertTrue(torch.equal(got, want), f"{got} != {want}")
            self._assert_replicas_agree(got)
        self.tc.assertEqual(self.optimizer.sharded[-1], [True, True])

    def _assert_replicas_agree(self, grad: torch.Tensor) -> None:
        group = self.mesh.get_group("replicate")
        gathered = [torch.empty_like(grad) for _ in range(group.size())]
        dist.all_gather(gathered, grad.contiguous(), group=group)
        for other in gathered[1:]:
            self.tc.assertTrue(torch.equal(gathered[0], other))


def _fresh_window(rank: int, world_size: int, backend: str = "nccl") -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        setup = _Setup(ctx)
        setup.run_window(seed=1)
        setup.run_window(seed=2)


def _k1_warmup_then_reset(rank: int, world_size: int, backend: str = "nccl") -> None:
    # A compile warmup runs K=1, then restores K and calls reset().
    with MultiProcessContext(rank, world_size, backend) as ctx:
        setup = _Setup(ctx)
        setup.wrapper.set_num_micro_batches_per_step(1)
        for seed in range(3):
            batch = setup.batches(1, seed=10 + seed)
            setup.run(batch)
            setup.tc.assertFalse(setup.has_partial())
            setup.assert_last_step_matches(batch)
        setup.wrapper.set_num_micro_batches_per_step(_K)
        setup.wrapper.reset()
        setup.run_window(seed=20)


def _exhaustion_drops_the_first_micro_batch(
    rank: int, world_size: int, backend: str = "nccl"
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        setup = _Setup(ctx, PartialWindowPolicy.STEP)
        setup.run(setup.batches(1, seed=30))
        setup.tc.assertTrue(setup.has_partial())
        with setup.tc.assertLogs(_GA_LOGGER, "WARNING"):
            with setup.tc.assertRaises(StopIteration):
                setup.wrapper.progress(iter([]))
        setup.tc.assertEqual(len(setup.optimizer.grads), 1)
        setup.tc.assertEqual(setup.optimizer.grads[0], [None, None])
        setup.tc.assertFalse(setup.has_partial())


def _reduction_hooks_run_once_per_window(
    rank: int, world_size: int, backend: str = "nccl"
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        setup = _Setup(ctx)
        groups = _param_groups(setup.model)
        all_reduces = [0] * len(groups)
        accumulations = [0] * len(groups)

        def count(counter: list[int], index: int) -> Callable[[Any], None]:
            def hook(_: Any) -> None:
                counter[index] += 1

            return hook

        for index, module in enumerate(_fsdp_modules(setup.model)):
            module.set_all_reduce_hook(count(all_reduces, index))
        for index, group in enumerate(groups):
            group.fsdp_params[0].sharded_param.register_post_accumulate_grad_hook(
                count(accumulations, index)
            )
        for window in (1, 2):
            setup.run(setup.batches(_K, seed=40 + window))
            setup.tc.assertEqual(all_reduces, [window] * len(groups))
            setup.tc.assertEqual(accumulations, [window] * len(groups))


def _discard_apis_reshard_and_recover(
    rank: int, world_size: int, backend: str = "nccl"
) -> None:
    with MultiProcessContext(rank, world_size, backend) as ctx:
        setup = _Setup(ctx)
        for seed, api in enumerate(("reset", "set_step", "set_k", "exception")):
            setup.run(setup.batches(1, seed=50 + seed))
            # Deferred resharding keeps the suppressed microbatch's parameters resident.
            setup.tc.assertEqual(_residency(setup.model), [False, False], api)
            setup.tc.assertTrue(setup.has_partial(), api)
            if api == "reset":
                setup.wrapper.reset(drop_partial=True)
            elif api == "set_step":
                setup.wrapper.set_step(0, drop_partial=True)
            elif api == "set_k":
                setup.wrapper.set_num_micro_batches_per_step(_K, allow_open_window=True)
            else:
                with patch.object(
                    setup.wrapper._pipeline,
                    "progress",
                    side_effect=RuntimeError("boom"),
                ), setup.tc.assertRaisesRegex(RuntimeError, "boom"):
                    setup.wrapper.progress(iter([]))
            setup.tc.assertEqual(_residency(setup.model), [True, True], api)
            setup.tc.assertFalse(setup.has_partial(), api)
            setup.run_window(seed=60 + seed)


@unittest.skipIf(
    torch.cuda.device_count() < _WORLD_SIZE,
    f"Need at least {_WORLD_SIZE} GPUs for a 2x2 HSDP mesh",
)
class GradientAccumulationHsdpTest(MultiProcessTestBase):
    """Native FSDP2 HSDP gradients under accumulation equal one backward per window."""

    def test_a_fresh_wrapper_all_reduces_only_on_the_step_micro_batch(self) -> None:
        self._run_multi_process_test(callable=_fresh_window, world_size=_WORLD_SIZE)

    def test_k1_warmup_then_reset_all_reduces_only_on_the_step_micro_batch(
        self,
    ) -> None:
        self._run_multi_process_test(
            callable=_k1_warmup_then_reset, world_size=_WORLD_SIZE
        )

    def test_exhaustion_after_the_first_micro_batch_drops_its_gradients(self) -> None:
        self._run_multi_process_test(
            callable=_exhaustion_drops_the_first_micro_batch, world_size=_WORLD_SIZE
        )

    def test_hooks_fire_once_per_window(self) -> None:
        self._run_multi_process_test(
            callable=_reduction_hooks_run_once_per_window, world_size=_WORLD_SIZE
        )

    def test_discard_apis_reshard_and_the_next_window_matches(self) -> None:
        self._run_multi_process_test(
            callable=_discard_apis_reshard_and_recover, world_size=_WORLD_SIZE
        )
