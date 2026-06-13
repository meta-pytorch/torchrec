#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_data_transfer -- \
        lazyawaitable --name=$(hg whereami | cut -c 1-10)

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_data_transfer \
        lazyawaitable --name=$(git rev-parse --short HEAD || echo $USER)

see README.md for more details
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    cmd_conf,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.types import DeviceToHostTensorAwaitable

logger: logging.Logger = logging.getLogger(__name__)

# pyrefly: ignore[missing-argument]
_cc = cmd_conf()


#################################### util functions ####################################
def _compute(
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    a dummy compute function to simulate the GPU load for computing, all
    operations are on the GPU side, no need to block CPU operations
    """
    if x is None:
        x = torch.rand(dim, dim, device=ctx.device) - 0.5
    for _ in range(num_mul):
        x = F.normalize(x @ x) * 10
    x = torch.sigmoid(x).reshape(1, dim, dim) + ctx.rank
    return torch.concat([x] * num_concat)


def _validate(x: torch.Tensor, ctx: MultiProcessContext) -> torch.Tensor:
    """
    validate the correctness of the comms result, the validation is done on GPU
    returns a GPU tensor with a single boolean value, non-blocking on CPU
    """
    mixed_ranks = x.to(torch.int).reshape(ctx.world_size, -1)
    checks = torch.empty(ctx.world_size, dtype=torch.bool, device=ctx.device)
    for i in range(ctx.world_size):
        checks[i] = torch.all(mixed_ranks[i, :] == i)
    return torch.all(checks)


################################# framework components #################################


def _unset_benchmark(*_args: Any, **_kwargs: Any) -> None:
    # placeholder default for DataCopyConfig.func; each benchmark config sets its own
    raise NotImplementedError("DataCopyConfig.func must be set by a subclass")


@dataclass
class DataCopyConfig(BenchFuncConfig):
    name: str = ""
    world_size: int = 2
    dim: int = 2048
    profile_dir: str = "."
    num_benchmarks: int = 2
    num_profiles: int = 2
    num_mul: int = 5
    num_concat: int = 100
    debug_mode: bool = False
    backend: str = "nccl"
    func: Callable[..., None] = _unset_benchmark


# single-rank runner
def single_rank_runner(rank: int, world_size: int, arg: DataCopyConfig) -> None:
    if arg.backend == "nccl":
        # Ensure GPUs are available and we have enough of them
        assert (
            torch.cuda.is_available() and torch.cuda.device_count() >= world_size
        ), "CUDA not available or insufficient GPUs for the requested world_size"

    arg.set_log_level()

    # debug mode only works with vscode for now.
    if arg.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    new_keys = {f.name for f in fields(type(arg))} - {
        f.name for f in fields(DataCopyConfig)
    }
    new_kwargs = {k: getattr(arg, k) for k in new_keys}
    func_name = getattr(arg.func, "__name__", arg.name)
    name: str = f"{func_name}_{arg.name}" if arg.name else func_name

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend=arg.backend,
        use_deterministic_algorithms=False,
    ) as ctx:
        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            benchmark_func_kwargs={
                "ctx": ctx,
                "dim": arg.dim,
                "num_mul": arg.num_mul,
                "num_concat": arg.num_concat,
            }
            | new_kwargs,
            func_to_benchmark=arg.func,
            rank=rank,
            # Input is empty, actual traffic is determined by the benchmark function
            sample_count=0,
            **arg.benchmark_func_kwargs(name=name),
        )

        if rank == 0:
            print(result)


###################################### benchmarks ######################################
def benchmark_lazyawaitable(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## pre-comms compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## all_to_all_single ##"):
        # use zeros instead of empty to make sure no previous data used
        post_comms = torch.zeros_like(pre_comms)
        req = dist.all_to_all_single(
            output=post_comms,
            input=pre_comms,
            group=ctx.pg,
            async_op=True,
        )

    with record_function("## irrelevant compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## comms check ##"):
        # assertion fails without wait(), this wait() makes the main cuda stream wait
        # for the comms to finish, so the post-comms compute will be blocked until
        # the comms is done
        assert req is not None
        req.wait()
        check_awaitable = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert check_awaitable.item()


@dataclass
class LazyAwaitableConfig(DataCopyConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer lazyawaitable

    use case:
        demonstrate the use of the device-to-host lazy awaitable. The
        DeviceToHostTensorAwaitable wraps a non-blocking device-to-host transfer
        together with a CUDA event and defers the cudaEventSync until the value is
        actually read on the host. This removes a CPU-blocking sync point, so the
        post-comms compute can be scheduled ahead of the host-side assertion --
        useful for sync-point removal in training optimization.
        see https://github.com/meta-pytorch/torchrec/pull/3477 for more details
    """

    name: str = "lazyawaitable"
    func: Callable[..., None] = benchmark_lazyawaitable


# pyrefly: ignore[missing-attribute]
@_cc.register
def lazyawaitable(arg: LazyAwaitableConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


def benchmark_h2d_data_copy(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    preallocated: bool = False,
    use_data_copy_stream: bool = True,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = (
            torch.cuda.Stream() if use_data_copy_stream else nullcontext()
        )
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

        # the host to device data transfer will block cuda execution without the `pin_memory()`
        host_data = (torch.rand(dim, dim) - 0.5).pin_memory()
        if preallocated:
            # pre-allocate memory on the device for the incoming data transfer from the host
            device_data = torch.empty_like(host_data, device=ctx.device)
        else:
            device_data = torch.empty(0, device=ctx.device)

    with record_function("## irrelevant compute before h2d ##"):
        # result intentionally discarded: this only simulates GPU compute load
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=irrelevant_data)

    with record_function("## copy data to device ##"):
        with data_copy_stream:
            if preallocated:
                # copy data to device, this will not block the main stream
                device_data.copy_(host_data, non_blocking=True)
            else:
                device_data = host_data.to(ctx.device, non_blocking=True)

    with record_function("## irrelevant compute after h2d ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        # result intentionally discarded: this only simulates GPU compute load
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=irrelevant_data)

    with record_function("## pre-comms compute ##"):
        # make sure the data copy is done before the pre-comms compute
        if use_data_copy_stream:
            # pyrefly: ignore[bad-argument-type]
            main_stream.wait_stream(data_copy_stream)
        # result intentionally discarded: this only simulates GPU compute load
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=device_data)


@dataclass
class H2DCopyConfig(DataCopyConfig):
    """
    run commands:
    1. default: non-blocking host-to-device data copy, w/o pre-allocated memory
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy \
        --name=non_blocking_h2d_copy

    2. pre-allocated: non-blocking host-to-device data copy, w/ pre-allocated memory
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy \
        --name=pre_allocated_h2d_copy \
        --preallocated=True

    3. blocking: blocking host-to-device data copy
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy \
        --name=blocking_h2d_copy \
        --use_data_copy_stream=False

    use case:
        study the CUDA cached-memory footprint of host-to-device copies. A
        non-blocking copy on a side stream inflates that stream's reserved
        memory: the copied tensor is allocated on the side stream and is not
        shared back with the main stream by the caching allocator.
        Pre-allocating on the main stream and doing an in-place copy on the side
        stream keeps the footprint as low as a blocking copy.
        see https://github.com/meta-pytorch/torchrec/pull/3485 and
        https://github.com/meta-pytorch/torchrec/pull/3510 for more details
    """

    memory_snapshot: bool = True
    preallocated: bool = False
    use_data_copy_stream: bool = True
    func: Callable[..., None] = benchmark_h2d_data_copy


# pyrefly: ignore[missing-attribute]
@_cc.register
def h2d_data_copy(arg: H2DCopyConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


####################################### backups ########################################


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
