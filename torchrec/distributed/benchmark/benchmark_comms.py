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
        all_to_all_base --name=$(hg whereami | cut -c 1-10)

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_comms \
        all_to_all_base --name=$(git rev-parse --short HEAD || echo $USER)

see README.md for more details
"""

import logging
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List

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


#################################### util functions ####################################
def _compute(
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    x: torch.Tensor | None = None,
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
    # placeholder default for CommsConfig.func; each benchmark config sets its own
    raise NotImplementedError("CommsConfig.func must be set by a subclass")


@dataclass
class CommsConfig(BenchFuncConfig):
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
def single_rank_runner(rank: int, world_size: int, arg: CommsConfig) -> None:
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
        f.name for f in fields(CommsConfig)
    }
    new_kwargs = {k: getattr(arg, k) for k in new_keys}
    func_name = getattr(arg.func, "__name__", arg.name)
    if func_name.startswith("benchmark_"):
        func_name = func_name[len("benchmark_") :]
    name: str = f"{func_name}_{arg.name}" if arg.name else func_name

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend=arg.backend,
        use_deterministic_algorithms=False,
    ) as ctx:
        # benchmarks that issue a second collective need a dedicated all-reduce
        # process group, created here inside the initialized MultiProcessContext
        if new_kwargs.pop("needs_ar_pg", False):
            new_kwargs["ar_pg"] = dist.new_group(ranks=list(range(ctx.world_size)))

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
# all_to_all_single on a single stream, sync or async
def benchmark_all_to_all_base(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    async_op: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## pre-comms compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## all_to_all_single ##"):
        # use zeros instead of empty to make sure no previous data is used
        post_comms = torch.zeros_like(pre_comms)
        req = dist.all_to_all_single(
            output=post_comms,
            input=pre_comms,
            group=ctx.pg,
            async_op=async_op,
        )

    with record_function("## comms pre-check ##"):
        # in the async case this runs before the comms is done (a "pre-check");
        # in the sync case the (blocking) comms has completed so it is the real check.
        # this non-blocking copy to CPU triggers a device-to-host data transfer; since
        # it's issued from the device side, the CPU doesn't know when it finishes, so we
        # need a cuda event to mark completion before reading it on the host.
        pre_checks = _validate(post_comms, ctx).to("cpu", non_blocking=True)
        ev_d2h = torch.cuda.Event()
        ev_d2h.record()

    with record_function("## irrelevant compute ##"):
        pre_comms = _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    ev_d2h.synchronize()  # make sure the pre_checks is available from cpu side
    with record_function(f"## comms check and pre-check: {pre_checks} ##"):
        if async_op:
            # assertion fails without wait(): this wait() makes the main cuda stream
            # wait for the comms to finish before the post-comms compute
            assert req is not None
            req.wait()
        checks = _validate(post_comms, ctx).to("cpu", non_blocking=True)
        ev_d2h.record()  # record the device-to-host data transfer

    with record_function("## post-comms compute ##"):
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        # make sure the device-to-host data transfer is done before the assertion
        ev_d2h.synchronize()
        assert checks


@dataclass
class AllToAllBaseConfig(CommsConfig):
    """
    run commands:
    1. sync (default): blocking all_to_all_single on a single stream
    > python -m torchrec.distributed.benchmark.benchmark_comms all_to_all_base \
        --name=sync

    2. async: async_op=True, the main stream waits via req.wait() before post-comms
    > python -m torchrec.distributed.benchmark.benchmark_comms all_to_all_base \
        --name=async \
        --async_op=True

    use case:
        benchmark all_to_all_single with and without async_op. With async_op the comms
        runs on a separate (comms) stream, non-blocking for the following main-stream
        ops, but the pre-allocated output is not valid until the comms is done (the
        pre-check fails); when a later op depends on the output, the caller must
        req.wait() so the main stream waits on the comms stream. Either way, the
        device-side validation result is read on the host via a non-blocking D2H copy
        gated by a CUDA event, so the host assertion only blocks until the value is
        actually available -- not on the whole comms.
        see https://github.com/meta-pytorch/torchrec/pull/3436 for more details
    """

    async_op: bool = False
    func: Callable[..., None] = benchmark_all_to_all_base


# pyrefly: ignore[missing-attribute]
@_cc.register
def all_to_all_base(arg: AllToAllBaseConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


# all_to_all_single with sync and single stream
def benchmark_a2a_async_twice(
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


@dataclass
class A2AAsyncTwiceConfig(CommsConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_comms a2a_async_twice

    use case:
        model the data-dependent comms chain in TWRW output dist (intra-node
        reduce_scatter -> decode/convert -> cross-node all_to_all). The second a2a and
        the decode run on a side stream so the main stream's following compute
        (rw-lookup / dense forward) overlaps instead of blocking on req.wait(). The
        side stream's allocations inflate GPU reserved memory (the caching allocator
        reserves per-stream pools) -- the compute-overlap vs memory-footprint tradeoff.
        see https://github.com/meta-pytorch/torchrec/pull/3440 for more details
    """

    memory_snapshot: bool = True
    func: Callable[..., None] = benchmark_a2a_async_twice


# pyrefly: ignore[missing-attribute]
@_cc.register
def a2a_async_twice(arg: A2AAsyncTwiceConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


def benchmark_shared_memory_across_process(
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
        benchmark_shared_memory_across_process._shm_tensor = shared_tensor

    if ctx.rank != 0:
        with record_function(f"## rank-{ctx.rank}: validate shared data ##"):
            assert torch.all(
                shared_tensor == 42.0
            ).item(), "Shared memory validation failed"
            logger.info(
                f"[rank-{ctx.rank}] validated shared tensor {list(shared_tensor.shape)}, "
                f"data_ptr matches rank 0's /dev/shm region"
            )


@dataclass
class SharedMemoryAcrossProcessConfig(CommsConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_comms shared_memory_across_process

    use case:
        for read-only eval/inference, place the full embedding table in CPU shared
        memory (/dev/shm) once and let all ranks on the host map the same physical
        pages -- no sharding, no input/output dist, zero data copies (except one copy
        on the creator rank). Host-level embedding footprint stays the size of the full
        table regardless of rank count. Delegates to create_on_rank_and_share_result;
        CPU-only, same-host (pass an intra-node process group).
        see https://github.com/meta-pytorch/torchrec/pull/3810 for more details
    """

    memory_snapshot: bool = True
    func: Callable[..., None] = benchmark_shared_memory_across_process


# pyrefly: ignore[missing-attribute]
@_cc.register
def shared_memory_across_process(arg: SharedMemoryAcrossProcessConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


def benchmark_multi_async_comms(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    ar_pg: dist.ProcessGroup | None = None,
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
        assert ar_pg is not None
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


@dataclass
class MultiAsyncCommsConfig(CommsConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_comms multi_async_comms

    use case:
        demonstrate two collectives (a2a + all_reduce) issued in different CPU-side
        order across two ranks, over two independent process groups (ctx.pg + ar_pg):
        rank 0 runs a2a then all_reduce, rank 1 runs them in the opposite order. Both
        ranks still complete correctly since each collective matches on its own group.
        see https://github.com/meta-pytorch/torchrec/pull/4149 for more details
    """

    # request a dedicated all-reduce process group (passed to func as ar_pg)
    needs_ar_pg: bool = True
    all_rank_traces: bool = True
    func: Callable[..., None] = benchmark_multi_async_comms


# pyrefly: ignore[missing-attribute]
@_cc.register
def multi_async_comms(arg: MultiAsyncCommsConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


def benchmark_competing_comms(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    serialized: bool = False,
    ar_pg: dist.ProcessGroup | None = None,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Two collectives (a2a on pg1, all_reduce on pg2) on separate streams
    with independent NCCL communicators. When serialized=False (default),
    both comms compete freely. When serialized=True, stream_b waits
    for comm1 to finish before issuing comm2.
    """
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        assert ar_pg is not None
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
            if serialized:
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


@dataclass
class CompetingCommsConfig(CommsConfig):
    """
    run commands:
    1. default: a2a and all_reduce compete freely on separate streams
    > python -m torchrec.distributed.benchmark.benchmark_comms competing_comms

    2. wait for comm1: stream_b waits for the a2a before issuing the all_reduce
    > python -m torchrec.distributed.benchmark.benchmark_comms competing_comms \
        --name=serialized \
        --serialized=True

    use case:
        collectives on different process groups have independent NCCL communicators, so
        two of them issued concurrently genuinely compete for NVLink bandwidth (e.g.
        sparse-dist a2a overlapping with dense-grad all_reduce). This runs a2a on
        stream_a and all_reduce on stream_b (separate PGs). With serialized=True,
        stream_b calls req_a.wait() before issuing the all_reduce, serializing the two
        so each gets full bandwidth -- without blocking the CPU or main stream (only the
        side stream waits). Compare against the default (free overlap) to see the
        contention vs serialization tradeoff.
        see https://github.com/meta-pytorch/torchrec/pull/4251 for more details
    """

    serialized: bool = False
    all_rank_traces: bool = True
    # request a dedicated all-reduce process group (passed to func as ar_pg)
    needs_ar_pg: bool = True
    func: Callable[..., None] = benchmark_competing_comms


# pyrefly: ignore[missing-attribute]
@_cc.register
def competing_comms(arg: CompetingCommsConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


def benchmark_tolist_overlap_comms(
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


@dataclass
class TolistOverlapCommsConfig(CommsConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_comms tolist_overlap_comms

    use case:
        probe how .tolist() (a CPU-blocking D2H sync) interacts with in-flight async
        comms across streams, used to investigate an AMD NCCL hang. .tolist() only
        blocks the CPU thread; it should not stall comms already enqueued on other
        streams -- which holds on NVIDIA, and on AMD too once the streams stay separate.
        The real divergence is comm-stream allocation: on AMD a side stream can collapse
        into the main stream, serializing comm+compute and making .tolist() appear to
        wait on the comms (a device-wide sync that was never issued). So .tolist() is
        not the root cause; the AMD side-stream collapse is.
        see https://github.com/meta-pytorch/torchrec/pull/4299 for more details
    """

    num_comms_rounds: int = 5
    func: Callable[..., None] = benchmark_tolist_overlap_comms


# pyrefly: ignore[missing-attribute]
@_cc.register
def tolist_overlap_comms(arg: TolistOverlapCommsConfig) -> None:
    run_multi_process_func(func=single_rank_runner, world_size=arg.world_size, arg=arg)


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
