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

from dataclasses import dataclass
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

_cc = cmd_conf()


@dataclass
class AllToAllSingleRunConfig(BenchFuncConfig):
    name: str = "all_to_all_single"
    world_size: int = 2
    dim: int = 2048
    profile_dir: str = "."
    num_benchmarks: int = 1
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

        if arg.name.startswith("a2a_sync_base"):
            func = a2a_sync_base
        elif arg.name.startswith("a2a_async_base"):
            func = a2a_async_base
        elif arg.name.startswith("a2a_async_twice"):
            func = a2a_async_twice
        else:
            func = a2a_sync_base

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


@_cc.register
def a2a_single(arg: AllToAllSingleRunConfig) -> None:
    run_multi_process_func(func=a2a_single_runner, world_size=arg.world_size, arg=arg)


if __name__ == "__main__":
    _cc.main()
