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
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_comms -- \
        a2a_single --name=a2a_sync_base-$(hg whereami | cut -c 1-10)

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_comms \
        a2a_single --name=a2a_sync_base-$(git rev-parse --short HEAD || echo $USER)

see README.md for more details
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

# pyrefly: ignore[missing-argument]
_cc = cmd_conf()


@dataclass
class AllToAllSingleRunConfig(BenchFuncConfig):
    name: str = "all_to_all_single"
    world_size: int = 2
    dim: int = 2048
    profile_dir: str = "."
    num_benchmarks: int = 2
    num_profiles: int = 2
    num_mul: int = 5
    num_concat: int = 100


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


# all_to_all_single with sync and single stream
def a2a_sync_base(
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
        post_comms = torch.empty_like(pre_comms)
        req = dist.all_to_all_single(output=post_comms, input=pre_comms, group=ctx.pg)

    with record_function("## comms validation ##"):
        # this non-blocking copy to CPU will trigger a device-to-host data transfer
        # however, since it's from the device side, CPU doesn't know if it's finished
        # so we'll need a cuda event to mark if it's done from the device side
        # the trace looks very interesting without cuda.event in this case
        # all cpu-side operations are non-blocking, and finished before the comms
        # and hence failed the validation assertion
        checks = _validate(post_comms, ctx).to(torch.device("cpu"), non_blocking=True)
        ev_d2h = torch.cuda.Event()
        ev_d2h.record()

    with record_function("## irrelevant compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        # explained above, this event.synchroize() is needed to make sure the
        # device-to-host data transfer is done before the assertion
        ev_d2h.synchronize()
        assert checks


# all_to_all_single with sync and single stream
def a2a_async_base(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
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

    with record_function("## comms pre-check ##"):
        # pre-check is performed before comms' done
        pre_checks = _validate(post_comms, ctx).to("cpu", non_blocking=True)
        # need this cuda.event to record the device-to-host data transfer
        ev_d2h = torch.cuda.Event()
        ev_d2h.record()

    with record_function("## irrelevant compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    ev_d2h.synchronize()  # make sure the pre_checks is available from cpu side
    with record_function(f"## comms check and pre-check: {pre_checks} ##"):
        # assertion fails without wait(), this wait() makes the main cuda stream wait
        # for the comms to finish, so the post-comms compute will be blocked until
        # the comms is done
        req.wait()
        checks = _validate(post_comms, ctx).to("cpu", non_blocking=True)
        ev_d2h.record()  # record the device-to-host data transfer

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        # again, make sure the device-to-host data transfer is done before the assertion
        ev_d2h.synchronize()
        assert checks


# all_to_all_single with sync and single stream
def a2a_async_twice(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## pre-comms compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## pre-allocation ##"):
        # use zeros instead of empty to make sure no previous data used
        post_comms1 = torch.zeros_like(pre_comms)
        post_comms2 = torch.zeros_like(pre_comms)

    with record_function("## comms1 ##"):
        req1 = dist.all_to_all_single(
            output=post_comms1,
            input=pre_comms,
            group=ctx.pg,
            async_op=True,
        )

    with record_function("## comms1 pre-validation ##"):
        # pre-check is performed before comms' done
        pre_checks1 = _validate(post_comms1, ctx).to("cpu", non_blocking=True)
        # need this cuda.event to record the device-to-host data transfer
        ev_d2h = torch.cuda.Event()
        ev_d2h.record()

    with record_function("## comms2 ##"):
        side_stream = torch.cuda.Stream()
        post_comms2.record_stream(side_stream)
        with torch.cuda.stream(side_stream):
            req1.wait()  # let the side stream wait for comms1 to finish
            pre_comms = torch.sigmoid(post_comms1) + ctx.rank
            req2 = dist.all_to_all_single(
                output=post_comms2,
                input=pre_comms,
                group=ctx.pg,
                async_op=True,
            )

    with record_function("## irrelevant compute1 ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## comms2 pre-validation ##"):
        # pre-check is performed before comms' done, actually even before comms2 starts
        pre_checks2 = _validate(post_comms2, ctx).to("cpu", non_blocking=True)
        ev_d2h.record()  # record the device-to-host data transfer

    with record_function("## irrelevant compute2 ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    ev_d2h.synchronize()  # make sure the pre_checks is available from cpu side
    with record_function(f"## comms1 checks and pre-checks1 {pre_checks1} ##"):
        req1.wait()  # let the main stream wait for comms1 to finish
        checks1 = _validate(post_comms1, ctx).to("cpu", non_blocking=True)
    with record_function(f"## comms2 checks and pre-checks2 {pre_checks2} ##"):
        req2.wait()  # let the main stream wait for comms2 to finish
        checks2 = _validate(post_comms2, ctx).to("cpu", non_blocking=True)
        ev_d2h.record()  # record the device-to-host data transfer

    with record_function("## post-comms comput ##"):
        post_comms2 = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms2[0]
        )

    with record_function("## assert ##"):
        # again, make sure the device-to-host data transfer is done before the assertion
        ev_d2h.synchronize()
        assert checks1 and checks2


# LazyAwaitable
def lazyawaitable(
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
        req.wait()
        check_awaitable = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert check_awaitable.item()


# muti-stream memory footprint
def multi_stream_memory(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    multi_stream: bool = True,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = torch.cuda.Stream() if multi_stream else nullcontext()
        data_dist_stream = torch.cuda.Stream() if multi_stream else nullcontext()
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

        # the host to device data transfer will block cuda execution without the `pin_memory()`
        host_data = (torch.rand(dim, dim) - 0.5).pin_memory()

    with record_function("## irrelevant compute before h2d ##"):
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## copy data to device ##"):
        # use a separate stream to copy data to device, this will not block the main stream
        with data_copy_stream:
            device_data = host_data.to(ctx.device, non_blocking=True)
            # record the data to main stream, so it won't be freed accidently in the data_copy_stream
            device_data.record_stream(main_stream)

    with record_function("## irrelevant compute after h2d ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## pre-comms compute ##"):
        if isinstance(data_copy_stream, torch.cuda.Stream):
            # make sure the data copy is done before the pre-comms compute
            main_stream.wait_stream(data_copy_stream)
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=device_data
        )

    # use a separate stream to do the comms, this will not block the main stream
    with data_dist_stream:
        with record_function("## all_to_all_single ##"):
            if isinstance(data_dist_stream, torch.cuda.Stream):
                # make sure the pre-comms compute is done before the comms
                data_dist_stream.wait_stream(main_stream)
            post_comms = torch.zeros_like(pre_comms)
            req = dist.all_to_all_single(
                output=post_comms,
                input=pre_comms,
                group=ctx.pg,
                async_op=True,
            )
            # record the data to main stream, so it won't be freed accidently in the data_dist_stream
            post_comms.record_stream(main_stream)
        with record_function("## a2a comm validation ##"):
            # the comm validation is also done in this separate stream since
            # there's no data dependency afterwards
            req.wait()
            checks = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## irrelevant compute after a2a ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## post-comms compute ##"):
        req.wait()
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert checks.item()


def single_stream_memory(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    return multi_stream_memory(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        multi_stream=False,
    )


# an optimized version of muti-stream memory footprint
def multi_stream_optimized(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = torch.cuda.Stream()
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

        # the host to device data transfer will block cuda execution without the `pin_memory()`
        host_data = (torch.rand(dim, dim) - 0.5).pin_memory()
        # pre-allocate memory on the device for the incoming data transfer from the host
        device_data = torch.empty_like(host_data, device=ctx.device)

    with record_function("## irrelevant compute before h2d ##"):
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## copy data to device ##"):
        with data_copy_stream:
            # copy data to device, this will not block the main stream
            device_data.copy_(host_data, non_blocking=True)

    with record_function("## irrelevant compute after h2d ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## pre-comms compute ##"):
        # make sure the data copy is done before the pre-comms compute
        main_stream.wait_stream(data_copy_stream)
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=device_data
        )

    with record_function("## pre-allocate memory for a2a on main stream ##"):
        post_comms = torch.zeros_like(pre_comms)

    with record_function("## all_to_all_single ##"):
        # the all_to_all_single from torch.dist has async feature
        # it automaically uses a separate stream to do the comms
        # without introducing extra memory footprint
        req = dist.all_to_all_single(
            output=post_comms,
            input=pre_comms,
            group=ctx.pg,
            async_op=True,
        )

    with record_function("## irrelevant compute after a2a ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## a2a comm validation ##"):
        # this req.wait() can be wrapped into a LazyAwaitable
        req.wait()
        # still want the compute on the main stream if possible
        checks = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert checks.item()


# an optimized version of muti-stream memory footprint
def non_blocking_copy(
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
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=irrelevant_data
        )

    with record_function("## copy data to device ##"):
        with data_copy_stream:
            if preallocated:
                # copy data to device, this will not block the main stream
                device_data.copy_(host_data, non_blocking=True)
            else:
                device_data = host_data.to(ctx.device, non_blocking=True)

    with record_function("## irrelevant compute after h2d ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=irrelevant_data
        )

    with record_function("## pre-comms compute ##"):
        # make sure the data copy is done before the pre-comms compute
        if use_data_copy_stream:
            main_stream.wait_stream(data_copy_stream)
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx, x=device_data
        )


def preallocated_non_blocking_copy(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    return non_blocking_copy(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        preallocated=True,
    )


def blocking_copy(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    return non_blocking_copy(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        use_data_copy_stream=False,
    )


def threading_copy(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    multithreading: bool = True,
    **_kwargs: Dict[str, Any],
) -> None:
    num_tensors = 512
    dummy_dim = 256

    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = torch.cuda.Stream()

        # create a list of small tensors on cpu, pinned for async H2D copy
        host_tensors = [
            torch.rand(dummy_dim, dummy_dim).pin_memory() for _ in range(num_tensors)
        ]
        # pre-allocate gpu memory so the copy thread only does .copy_()
        device_tensors = [torch.empty_like(t, device=ctx.device) for t in host_tensors]

        # large tensor on gpu for main-stream compute
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

    def _copy_worker() -> None:
        torch.cuda.set_device(ctx.device)
        with torch.cuda.stream(data_copy_stream):
            for i in range(num_tensors):
                device_tensors[i].copy_(host_tensors[i], non_blocking=True)

    # launch the copy via ThreadPoolExecutor
    with record_function("## submit copy to executor ##"):
        if multithreading:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_copy_worker)
        else:
            _copy_worker()

    # run slow gpu operations on the main stream — these should overlap with the copies
    with record_function("## main stream compute (should overlap with copy) ##"):
        for _ in range(num_mul):
            irrelevant_data = _compute(
                dim=dim, num_mul=1, num_concat=1, ctx=ctx, x=irrelevant_data
            )

    with record_function("## wait for executor future ##"):
        if multithreading:
            future.result()
            executor.shutdown(wait=False)

    with record_function("## wait for copy stream ##"):
        main_stream.wait_stream(data_copy_stream)

    # use the copied data to prove it arrived correctly
    with record_function("## use copied data ##"):
        _ = _compute(
            dim=dummy_dim, num_mul=1, num_concat=1, ctx=ctx, x=device_tensors[0]
        )


def single_thread_copy(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    multithreading: bool = True,
    **_kwargs: Dict[str, Any],
):
    return threading_copy(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        multithreading=False,
    )


# single-rank runner
def a2a_single_runner(rank: int, world_size: int, arg: AllToAllSingleRunConfig) -> None:
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:
        match arg.name.lower():
            case "a2a_sync_base":
                func = a2a_sync_base
            case "a2a_async_base":
                func = a2a_async_base
            case "a2a_async_twice":
                func = a2a_async_twice
            case "lazyawaitable":
                func = lazyawaitable
            case "multi_stream_memory":
                func = multi_stream_memory
            case "single_stream_memory":
                func = single_stream_memory
            case "multi_stream_optimized":
                func = multi_stream_optimized
            case "non_blocking_copy":
                func = non_blocking_copy
            case "preallocated_non_blocking_copy":
                func = preallocated_non_blocking_copy
            case "blocking_copy":
                func = blocking_copy
            case "threading_copy":
                func = threading_copy
            case "single_thread_copy":
                func = single_thread_copy
            case _:
                raise ValueError(f"Unknown benchmark name: {arg.name}")

        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            benchmark_func_kwargs={
                "ctx": ctx,
                "dim": arg.dim,
                "num_mul": arg.num_mul,
                "num_concat": arg.num_concat,
            },
            func_to_benchmark=func,
            rank=rank,
            **arg.benchmark_func_kwargs(),
        )

        if rank == 0:
            print(result)


# pyrefly: ignore[missing-attribute]
@_cc.register
def a2a_single(arg: AllToAllSingleRunConfig) -> None:
    run_multi_process_func(func=a2a_single_runner, world_size=arg.world_size, arg=arg)


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
