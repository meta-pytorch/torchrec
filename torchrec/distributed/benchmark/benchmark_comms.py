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

import logging
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
from torchrec.distributed.collective_utils import create_on_rank_and_share_result
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.types import DeviceToHostTensorAwaitable

logger: logging.Logger = logging.getLogger(__name__)

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
    debug_mode: bool = False
    backend: str = "nccl"


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
        assert req is not None
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
            assert req1 is not None
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
        assert req1 is not None
        req1.wait()  # let the main stream wait for comms1 to finish
        checks1 = _validate(post_comms1, ctx).to("cpu", non_blocking=True)
    with record_function(f"## comms2 checks and pre-checks2 {pre_checks2} ##"):
        assert req2 is not None
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


def shared_memory_across_process(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    CPU shared memory benchmark: rank 0 creates a large CPU tensor in POSIX
    shared memory (/dev/shm) and shares the storage metadata with other ranks
    via a broadcast collective. Other ranks open the same shared memory region
    and access the tensor without any data copy.

    Both processes hold the mapping open simultaneously so you can verify
    from the host that only one copy of the data exists:
        ls -lh /dev/shm/torch_*
    """

    # Use (num_concat * dim, dim) to make the shared region large enough
    # to be clearly visible in /dev/shm.
    # Default: num_concat=100, dim=2048 => 100*2048*2048*4 bytes ≈ 1.6 GB
    shm_shape = (num_concat * dim, dim)

    assert ctx.pg is not None
    with record_function("## create and share tensor ##"):
        shared_tensor = create_on_rank_and_share_result(
            ctx.pg,
            0,
            creator=lambda: torch.full(shm_shape, fill_value=42.0, dtype=torch.float32),
            extractor=lambda t: [t],
            constructor=lambda ts: ts[0],  # pyrefly: ignore[bad-argument-type]
        )
        # Keep the tensor alive across benchmark iterations so the shared
        # memory file in /dev/shm is not cleaned up prematurely.
        # pyrefly: ignore[missing-attribute]
        shared_memory_across_process._shm_tensor = shared_tensor

    if ctx.rank != 0:
        with record_function(f"## rank-{ctx.rank}: validate shared data ##"):
            assert torch.all(
                shared_tensor == 42.0
            ).item(), "Shared memory validation failed"
            logger.info(
                f"[rank-{ctx.rank}] validated shared tensor {list(shared_tensor.shape)}, "
                f"data_ptr matches rank 0's /dev/shm region"
            )


def multi_async_comms(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Rank 0 and rank 1 issue two different collectives (a2a_single + all_reduce)
    in different CPU-side order. The a2a uses ctx.pg and the all_reduce uses
    a separate process group, so they have independent NCCL communicators.

    Rank 0: a2a → compute → all_reduce
    Rank 1: all_reduce → compute → a2a

    The two collectives operate on different-sized tensors
    (input_a: num_concat, input_b: num_concat*2).
    """
    with record_function("## setup ##"):
        rank = ctx.rank
        ar_pg = multi_async_comms._ar_pg  # pyrefly: ignore[missing-attribute]
        # Warm up the NCCL communicator so subsequent ops are truly async;
        # the first collective on a new pg blocks for ncclCommInitRank
        dist.barrier(group=ar_pg)

    with record_function("## pre-comms compute ##"):
        input_a = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)
        input_b = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat * 2, ctx=ctx)

    def do_a2a(
        input_tensor: torch.Tensor,
        label: str,
    ) -> tuple[torch.Tensor, dist.Work]:
        out = torch.zeros_like(input_tensor)
        with record_function(f"## rank {rank}: {label} ##"):
            req = dist.all_to_all_single(
                output=out,
                input=input_tensor,
                group=ctx.pg,
                async_op=True,
            )
        assert req is not None
        return out, req

    def do_all_reduce(
        input_tensor: torch.Tensor,
        label: str,
    ) -> tuple[torch.Tensor, dist.Work]:
        out = input_tensor.clone()
        with record_function(f"## rank {rank}: {label} ##"):
            req = dist.all_reduce(
                out,
                op=dist.ReduceOp.SUM,
                group=ar_pg,
                async_op=True,
            )
        return out, req

    if rank == 0:
        out_a, req_a = do_a2a(input_a, "a2a")
        with record_function("## irrelevant compute ##"):
            _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)
        out_b, req_b = do_all_reduce(input_b, "all_reduce")
    else:
        out_b, req_b = do_all_reduce(input_b, "all_reduce")
        with record_function("## irrelevant compute ##"):
            _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)
        out_a, req_a = do_a2a(input_a, "a2a")

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## wait and validate ##"):
        req_a.wait()
        req_b.wait()
        checks_a = _validate(out_a, ctx)
        # all_reduce(SUM) of values from rank 0 in (0,1) and rank 1 in (1,2)
        # produces sums in (1,3), so int values >= 1
        checks_b = torch.all(out_b.to(torch.int) >= 1)
        checks = DeviceToHostTensorAwaitable(checks_a & checks_b)

    with record_function("## post-comms compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=out_a[0])

    with record_function("## assert ##"):
        assert checks.item()


def competing_comms(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    wait_for_comm1: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Two collectives (a2a on pg1, all_reduce on pg2) on separate streams
    with independent NCCL communicators. When wait_for_comm1=False (default),
    both comms compete freely. When wait_for_comm1=True, stream_b waits
    for comm1 to finish before issuing comm2.
    """
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        # pyrefly: ignore[missing-attribute]
        ar_pg = competing_comms._ar_pg
        dist.barrier(group=ar_pg)

    with record_function("## pre-comms compute ##"):
        input_a = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)
        input_b = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat * 2, ctx=ctx)

    with record_function("## comm1: a2a on stream_a ##"):
        out_a = torch.zeros_like(input_a)
        out_a.record_stream(stream_a)
        with torch.cuda.stream(stream_a):
            stream_a.wait_stream(main_stream)
            req_a = dist.all_to_all_single(
                output=out_a,
                input=input_a,
                group=ctx.pg,
                async_op=True,
            )
            assert req_a is not None

    with record_function("## comm2: all_reduce on stream_b ##"):
        out_b = input_b.clone()
        out_b.record_stream(stream_b)
        with torch.cuda.stream(stream_b):
            if wait_for_comm1:
                assert req_a is not None
                req_a.wait()
            stream_b.wait_stream(main_stream)
            req_b = dist.all_reduce(
                out_b,
                op=dist.ReduceOp.SUM,
                group=ar_pg,
                async_op=True,
            )
            assert req_b is not None

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## wait and validate ##"):
        assert req_a is not None
        assert req_b is not None
        req_a.wait()
        req_b.wait()
        main_stream.wait_stream(stream_a)
        main_stream.wait_stream(stream_b)
        checks_a = _validate(out_a, ctx)
        checks_b = torch.all(out_b.to(torch.int) >= 1)
        checks = DeviceToHostTensorAwaitable(checks_a & checks_b)

    with record_function("## post-comms compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=out_a[0])

    with record_function("## assert ##"):
        assert checks.item()


def competing_comms_with_req_wait(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    return competing_comms(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        wait_for_comm1=True,
    )


def tolist_overlap_comms(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    num_comms_rounds: int = 5,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Test whether .tolist() on AMD waits for in-flight async comms to complete.

    Two .tolist() calls for comparison:
    1. On comms stream: comms launched (async_op) -> compute -> .tolist()
       while comms are still in flight. On AMD this may block until comms
       finish; on NVIDIA it should only wait for the compute D2H.
    2. On main stream: after comms are waited -> compute -> .tolist()
       This should be fast on both platforms since no comms are in flight.
    """
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        comms_stream = torch.cuda.Stream()

    with record_function("## pre-comms compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## comms stream: async comms + tolist during comms ##"):
        post_comms = torch.zeros_like(pre_comms)
        post_comms.record_stream(comms_stream)
        with torch.cuda.stream(comms_stream):
            comms_stream.wait_stream(main_stream)
            requests = [
                dist.all_to_all_single(
                    output=post_comms,
                    input=pre_comms,
                    group=ctx.pg,
                    async_op=True,
                )
                for _ in range(num_comms_rounds)
            ]
            compute_result1 = _compute(
                dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx
            )
            _ = compute_result1[0][0].tolist()

            for req in requests:
                assert req is not None
                req.wait()

    with record_function("## tolist after comms (no in-flight comms) ##"):
        compute_result2 = _compute(dim=dim, num_mul=1, num_concat=1, ctx=ctx)
        _ = compute_result2[0][0].tolist()

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=1, num_concat=1, ctx=ctx)

    with record_function("## wait and validate ##"):
        for req in requests:
            assert req is not None
            req.wait()
        main_stream.wait_stream(comms_stream)
        checks = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## post-comms compute ##"):
        _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert checks.item()


def h2d_execution_order(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Test the execution order of H2D copies across three streams.

    Each stream issues a different amount of compute before its H2D copy,
    so the copies are submitted to the copy engine at different times.
    The CPU issues the long-compute stream first, but its copy reaches
    the copy engine last:
      stream_c: sync a2a       → H2D copy C (issued first, submitted last)
      stream_b: medium compute → H2D copy B (issued second)
      stream_a: short compute  → H2D copy A (issued last, submitted first)

    Check the trace to observe the actual execution order of the H2D
    copies on the copy engine. If the copy engine is FIFO by GPU-side
    submission order (not CPU issue order), copy A should start first
    despite being issued last from the CPU.
    """
    copy_size = num_concat * dim // 5

    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        stream_c = torch.cuda.Stream()

    small_size = 1024

    with record_function("## allocate tensors ##"):
        # bulk copies
        src_a = torch.full((copy_size, dim), fill_value=1.0).pin_memory()
        src_b = torch.full((copy_size, dim), fill_value=2.0).pin_memory()
        src_c = torch.full((copy_size, dim), fill_value=3.0).pin_memory()

        # small copies before/after each bulk copy (int32 ~ 4 KB each)
        small_before = [
            torch.full(
                (small_size,), fill_value=float(100 + i), dtype=torch.int32
            ).pin_memory()
            for i in range(3)
        ]
        small_after = [
            torch.full(
                (small_size,), fill_value=float(200 + i), dtype=torch.int32
            ).pin_memory()
            for i in range(3)
        ]

    with record_function("## pre-compute ##"):
        _compute(dim=dim, num_mul=num_mul * 3, num_concat=num_concat, ctx=ctx)

    with record_function("## stream_c: sync a2a + h2d (issued first) ##"):
        a2a_input = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)
        a2a_output = torch.zeros_like(a2a_input)
        with torch.cuda.stream(stream_c):
            stream_c.wait_stream(main_stream)
            dist.all_to_all_single(
                output=a2a_output,
                input=a2a_input,
                group=ctx.pg,
                async_op=False,
            )
            small_dev_c_before = small_before[2].to(ctx.device, non_blocking=True)
            small_dev_c_before += 1
            dev_c = src_c.to(ctx.device, non_blocking=True)
            small_dev_c_before -= 1
            small_dev_c_after = small_after[2].to(ctx.device, non_blocking=True)
            dev_c.record_stream(main_stream)
            small_dev_c_before.record_stream(main_stream)
            small_dev_c_after.record_stream(main_stream)

    with record_function("## stream_b: medium compute + h2d (issued second) ##"):
        with torch.cuda.stream(stream_b):
            stream_b.wait_stream(main_stream)
            _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx)
            small_dev_b_before = small_before[1].to(ctx.device, non_blocking=True)
            small_dev_b_before += 1
            dev_b = src_b.to(ctx.device, non_blocking=True)
            small_dev_b_before -= 1
            small_dev_b_after = small_after[1].to(ctx.device, non_blocking=True)
            dev_b.record_stream(main_stream)
            small_dev_b_before.record_stream(main_stream)
            small_dev_b_after.record_stream(main_stream)

    with record_function("## stream_a: short compute + h2d (issued last) ##"):
        with torch.cuda.stream(stream_a):
            stream_a.wait_stream(main_stream)
            _compute(dim=dim, num_mul=1, num_concat=1, ctx=ctx)
            small_dev_a_before = small_before[0].to(ctx.device, non_blocking=True)
            small_dev_a_before += 1
            dev_a = src_a.to(ctx.device, non_blocking=True)
            small_dev_a_before -= 1
            small_dev_a_after = small_after[0].to(ctx.device, non_blocking=True)
            dev_a.record_stream(main_stream)
            small_dev_a_before.record_stream(main_stream)
            small_dev_a_after.record_stream(main_stream)

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## wait and validate ##"):
        torch.cuda.synchronize()

        a_correct = bool(torch.all(dev_a == 1.0).item())
        b_correct = bool(torch.all(dev_b == 2.0).item())
        c_correct = bool(torch.all(dev_c == 3.0).item())

        small_correct = (
            bool(torch.all(small_dev_a_before == 100).item())
            and bool(torch.all(small_dev_a_after == 200).item())
            and bool(torch.all(small_dev_b_before == 101).item())
            and bool(torch.all(small_dev_b_after == 201).item())
            and bool(torch.all(small_dev_c_before == 102).item())
            and bool(torch.all(small_dev_c_after == 202).item())
        )

    with record_function("## post-copy compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## assert ##"):
        assert a_correct, f"Copy A validation failed on rank {ctx.rank}"
        assert b_correct, f"Copy B validation failed on rank {ctx.rank}"
        assert c_correct, f"Copy C validation failed on rank {ctx.rank}"
        assert small_correct, f"Small copy validation failed on rank {ctx.rank}"


def a2a_d2h_contention(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    concurrent: bool = True,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Compare resource contention between all_to_all_single (NVLink/network)
    and device-to-host copy (PCIe). When concurrent=True, both run on
    separate streams simultaneously — contention on shared resources (PCIe,
    memory bandwidth) will show up as increased latency in the trace.
    When concurrent=False, D2H waits for all2all to complete first,
    providing a contention-free baseline for comparison.
    """
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        comms_stream = torch.cuda.Stream()
        d2h_stream = torch.cuda.Stream()

    with record_function("## pre-comms compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## allocate d2h tensors ##"):
        d2h_source = torch.full(
            (num_concat * dim, dim),
            fill_value=float(ctx.rank + 1),
            device=ctx.device,
        )
        cpu_buffer = torch.empty_like(d2h_source, device="cpu").pin_memory()

    with record_function("## barrier — align ranks ##"):
        dist.barrier(group=ctx.pg)

    with record_function("## d2h copy on d2h_stream ##"):
        with torch.cuda.stream(d2h_stream):
            d2h_stream.wait_stream(main_stream)
            cpu_buffer.copy_(d2h_source, non_blocking=True)

    with record_function("## all2all on comms_stream ##"):
        post_comms = torch.zeros_like(pre_comms)
        post_comms.record_stream(comms_stream)
        with torch.cuda.stream(comms_stream):
            comms_stream.wait_stream(main_stream)
            if not concurrent:
                comms_stream.wait_stream(d2h_stream)
            req = dist.all_to_all_single(
                output=post_comms,
                input=pre_comms,
                group=ctx.pg,
                async_op=True,
            )

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## wait and validate ##"):
        assert req is not None
        req.wait()
        main_stream.wait_stream(comms_stream)
        d2h_stream.synchronize()
        checks_a2a = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))
        expected = float(ctx.rank + 1)
        d2h_correct = bool(torch.all(cpu_buffer == expected).item())

    with record_function("## post-comms compute ##"):
        _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert checks_a2a.item()
        assert d2h_correct, f"D2H copy validation failed on rank {ctx.rank}"


def a2a_d2h_sequential(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    **_kwargs: Dict[str, Any],
) -> None:
    return a2a_d2h_contention(
        _batch_inputs=_batch_inputs,
        dim=dim,
        num_mul=num_mul,
        num_concat=num_concat,
        ctx=ctx,
        concurrent=False,
    )


# single-rank runner
def a2a_single_runner(rank: int, world_size: int, arg: AllToAllSingleRunConfig) -> None:
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

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend=arg.backend,
        use_deterministic_algorithms=False,
    ) as ctx:
        match arg.name.lower():
            case "a2a_sync_base":
                func = a2a_sync_base
            case "a2a_async_base":
                func = a2a_async_base
            case "a2a_async_twice":
                func = a2a_async_twice
            case "shared_memory_across_process":
                func = shared_memory_across_process
            case "multi_async_comms":
                func = multi_async_comms
                # pyrefly: ignore[missing-attribute]
                multi_async_comms._ar_pg = dist.new_group(
                    ranks=list(range(ctx.world_size))
                )
            case "competing_comms":
                func = competing_comms
                # pyrefly: ignore[missing-attribute]
                competing_comms._ar_pg = dist.new_group(
                    ranks=list(range(ctx.world_size))
                )
            case "competing_comms_with_req_wait":
                func = competing_comms_with_req_wait
                # pyrefly: ignore[missing-attribute]
                competing_comms._ar_pg = dist.new_group(
                    ranks=list(range(ctx.world_size))
                )
            case s if s.startswith("tolist_overlap_comms"):
                func = tolist_overlap_comms
            case "h2d_execution_order":
                func = h2d_execution_order
            case "a2a_d2h_contention":
                func = a2a_d2h_contention
            case "a2a_d2h_sequential":
                func = a2a_d2h_sequential
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
            # Input is empty, actual traffic is determined by the benchmark function
            sample_count=0,
            **arg.benchmark_func_kwargs(name=f"{arg.name}_{arg.backend}"),
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
