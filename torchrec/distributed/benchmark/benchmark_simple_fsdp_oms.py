#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise compiled SimpleFSDP optimizer-state prefetch on two local GPUs.

Example usage (internal):
    buck2 run @fbcode//mode/opt @fbcode//mode/inplace \
        fbcode//torchrec/distributed/benchmark:benchmark_simple_fsdp_oms -- \
        --swap --expect-hook --verify-hook-gradients --output=/tmp/simple-fsdp-oms

Run with --swap to exercise APS's DTensor-to-local-parameter decorators.
The shared process runner owns rendezvous, device binding, and process groups.
Every rank writes a GPU trace, hook counts, and final local parameters.
This synthetic workload checks compatibility, not production throughput.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fb.simple_fsdp.fx_passes import set_passes_for_simple_fsdp
from torch.distributed.fb.simple_fsdp.simple_fsdp import (
    simple_fsdp_context_manager_decorator,
    SimpleFSDP_data_parallel,
)
from torch.distributed.fb.simple_fsdp.utils import (
    simple_fsdp_pipeline_backward_decorator,
    simple_fsdp_pipeline_forward_decorator,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.profiler import profile, ProfilerActivity, record_function
from torchrec.distributed.memory_stashing import MemoryStashingManager
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)

logger: logging.Logger = logging.getLogger(__name__)


class PrefetchProbe:
    def __init__(self, parameter: torch.Tensor) -> None:
        self.parameter = parameter
        self.step = -1
        self.calls: list[dict[str, int | str]] = []
        self.active = False

    def __call__(self, parameter: torch.Tensor) -> None:
        assert parameter is self.parameter, "Hook received the replacement parameter"
        if not self.active:
            return
        assert parameter.grad is not None
        with record_function("OMS_PARAM_GRAD_PREFETCH"):
            self.calls.append(
                {
                    "step": self.step,
                    "time_ns": time.perf_counter_ns(),
                    "parameter_type": type(parameter).__name__,
                }
            )
            MemoryStashingManager.restore_optimizer_state_next()


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def optimizer_tensors(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [
        local_tensor(value)
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_cuda
    ]


def build_training(
    args: argparse.Namespace,
    device: torch.device,
    world_size: int,
) -> tuple[torch.nn.Module, Callable[..., torch.Tensor], Callable[..., None]]:
    layers = []
    for _ in range(args.layers):
        layers.extend([torch.nn.Linear(args.width, args.width), torch.nn.GELU()])
    model = torch.nn.Sequential(*layers).to(device)
    SimpleFSDP_data_parallel(
        model,
        init_device_mesh(device.type, (world_size,)),
        MixedPrecisionPolicy(),
        mode="fully_shard",
    )
    set_passes_for_simple_fsdp(
        {
            "dedup_wait_nodes": None,
            "bucket_fsdp_all_gather_concat": {"all_gather_bucket_cap_mb": 500},
            "bucket_fsdp_reduce_scatter_concat": {"reduce_scatter_bucket_cap_mb": 500},
            "force_recompute_fsdp_all_gather": {"force_recompute": True},
        }
    )
    forward = torch.compile(model, fullgraph=True)
    backward = torch.Tensor.backward
    if args.swap:
        mappings = {}
        forward = simple_fsdp_pipeline_forward_decorator(forward, model, mappings)
        backward = simple_fsdp_pipeline_backward_decorator(backward, model, mappings)
    return (
        model,
        simple_fsdp_context_manager_decorator(forward),
        simple_fsdp_context_manager_decorator(backward),
    )


def train_step(
    step: int,
    forward: Callable[..., torch.Tensor],
    backward: Callable[..., None],
    optimizer: torch.optim.Optimizer,
    probe: PrefetchProbe,
    inputs: torch.Tensor,
) -> dict[str, int | float]:
    optimizer.zero_grad(set_to_none=True)
    expected = [tensor.clone() for tensor in optimizer_tensors(optimizer)]
    state_bytes = sum(t.numel() * t.element_size() for t in expected)
    with record_function("OMS_STASH"):
        await_restore, _ = MemoryStashingManager.stash_optimizer_state(optimizer)
    probe.step = step
    probe.active = True
    with record_function("SIMPLE_FSDP_FORWARD"):
        loss = forward(inputs).square().mean()
    backward_start = time.perf_counter_ns()
    with record_function("SIMPLE_FSDP_BACKWARD"):
        backward(loss)
    backward_end = time.perf_counter_ns()
    with record_function("OMS_OPTIMIZER_FALLBACK_RESTORE"):
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)
    torch.cuda.synchronize()
    for actual, saved in zip(optimizer_tensors(optimizer), expected, strict=True):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "step": step,
        "loss": loss.item(),
        "state_bytes": state_bytes,
        "backward_start_ns": backward_start,
        "backward_end_ns": backward_end,
    }


def verify_hook_gradients(
    parameter: torch.Tensor,
    forward: Callable[..., torch.Tensor],
    backward: Callable[..., None],
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    backward(forward(inputs).square().mean())
    assert parameter.grad is not None
    reference = parameter.grad.detach().clone()
    calls = []

    def scale_gradient(original: torch.Tensor) -> None:
        assert original is parameter
        assert isinstance(original.grad, DTensor)
        calls.append(original)
        original.grad.mul_(0.5)

    handle = parameter.register_post_accumulate_grad_hook(scale_gradient)
    optimizer.zero_grad(set_to_none=True)
    backward(forward(inputs).square().mean())
    assert len(calls) == 1
    torch.testing.assert_close(parameter.grad, reference * 0.5, rtol=0, atol=0)
    handle.remove()
    optimizer.zero_grad(set_to_none=True)
    backward(forward(inputs).square().mean())
    assert len(calls) == 1
    torch.testing.assert_close(parameter.grad, reference, rtol=0, atol=0)


def save_parameters(
    model: torch.nn.Module, rank: int, args: argparse.Namespace
) -> None:
    parameters = {
        name: local_tensor(p).detach().cpu() for name, p in model.named_parameters()
    }
    if args.compare_to:
        reference = torch.load(
            args.compare_to / f"rank-{rank}.parameters.pt", weights_only=True
        )
        torch.testing.assert_close(parameters, reference, rtol=0, atol=0)
    torch.save(parameters, args.output / f"rank-{rank}.parameters.pt")


def _run_training(
    ctx: SingleProcessContext, rank: int, world_size: int, args: argparse.Namespace
) -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(1234)
    model, forward, backward = build_training(args, ctx.device, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True)
    MemoryStashingManager.set_streams(torch.cuda.Stream(), torch.cuda.Stream())
    MemoryStashingManager.set_optimizer_trunk_size(256 * 1024**2, 256 * 1024**2)
    parameter = next(model.parameters())
    probe = PrefetchProbe(parameter)
    handle = parameter.register_post_accumulate_grad_hook(probe)
    torch.manual_seed(4321 + rank)
    inputs = torch.randn(args.batch_size, args.width, device=ctx.device)
    for _ in range(args.warmups):
        optimizer.zero_grad(set_to_none=True)
        backward(forward(inputs).square().mean())
        optimizer.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        results = [
            train_step(step, forward, backward, optimizer, probe, inputs)
            for step in range(args.steps)
        ]
    prof.export_chrome_trace(str(args.output / f"rank-{rank}.trace.json"))
    counts = [
        sum(call["step"] == step for call in probe.calls) for step in range(args.steps)
    ]
    assert counts == [int(args.expect_hook)] * args.steps, counts
    handle.remove()
    before = len(probe.calls)
    optimizer.zero_grad(set_to_none=True)
    backward(forward(inputs).square().mean())
    assert len(probe.calls) == before, "Removed hook still fired"
    if args.verify_hook_gradients:
        verify_hook_gradients(parameter, forward, backward, optimizer, inputs)
    save_parameters(model, rank, args)
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "swap": args.swap,
                "results": results,
                "hook_calls": probe.calls,
                "hook_gradients_verified": args.verify_hook_gradients,
                "parameters_compared_to": (
                    str(args.compare_to) if args.compare_to else None
                ),
            },
            indent=2,
        )
    )
    logger.info("Rank %s passed: hook counts %s; state round trips exact", rank, counts)


def benchmark_runner(
    ctx: SingleProcessContext, rank: int, world_size: int, args: argparse.Namespace
) -> None:
    try:
        _run_training(ctx, rank, world_size, args)
    finally:
        MemoryStashingManager.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--expect-hook", action="store_true")
    parser.add_argument("--verify-hook-gradients", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.steps < 1 or args.warmups < 1:
        parser.error("--steps and --warmups must be positive")
    if not 1 <= args.world_size <= torch.cuda.device_count():
        parser.error("--world-size must fit the available CUDA devices")
    args.output.mkdir(parents=True, exist_ok=True)
    run_local_multi_process_func(
        benchmark_runner,
        world_size=args.world_size,
        disable_cuda_tf_32=False,
        args=args,
    )


if __name__ == "__main__":
    main()
