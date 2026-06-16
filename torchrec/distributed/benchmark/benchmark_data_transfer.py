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
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
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
from torchrec.distributed.memory_stashing import chunked_copy_
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


# single-rank runner
def single_rank_runner(
    rank: int,
    world_size: int,
    arg: DataCopyConfig,
    bench_func: Callable[..., None],
) -> None:
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
    func_name = getattr(bench_func, "__name__", arg.name)
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
            func_to_benchmark=bench_func,
            rank=rank,
            # Input is empty, actual traffic is determined by the benchmark function
            sample_count=0,
            **arg.benchmark_func_kwargs(name=name),
        )

        if rank == 0:
            print(result)


def register_benchmark(
    config: type[DataCopyConfig],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Decorator factory: register a benchmark function with the CLI, bound to the
    given config class. The decorated function is the per-iteration benchmark and
    its name is the CLI subcommand. Define the config class first, then:

    @register_benchmark(LazyAwaitableConfig)
    def lazyawaitable(_batch_inputs, ..., ctx, **_kwargs): ...
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        def dispatch(arg: DataCopyConfig) -> None:
            run_multi_process_func(
                func=single_rank_runner,
                world_size=arg.world_size,
                arg=arg,
                bench_func=func,
            )

        # CLI subcommand key = benchmark function name; the annotation must be the
        # concrete config subclass so cmd_conf builds its argparse from its fields
        dispatch.__name__ = func.__name__
        dispatch.__annotations__ = {"arg": config, "return": None}
        # pyrefly: ignore[missing-attribute]
        _cc.register(dispatch)
        return func

    return decorator


###################################### benchmarks ######################################
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


@register_benchmark(LazyAwaitableConfig)
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
class H2DCopyConfig(DataCopyConfig):
    """
    run commands:
    1. non-blocking (default): host-to-device data copy, w/o pre-allocated memory
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy

    2. pre-allocated: non-blocking host-to-device data copy, w/ pre-allocated memory
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy \
        --name=preallocated \
        --preallocated=True

    3. blocking: blocking host-to-device data copy
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_data_copy \
        --name=blocking \
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


@register_benchmark(H2DCopyConfig)
def h2d_data_copy(
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
class StreamMemoryConfig(DataCopyConfig):
    """
    run commands:
    1. single stream: copy + a2a on the main stream
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer stream_memory \
        --name=single \
        --multi_stream=False

    2. multi stream (default): copy on a side stream, a2a on a dedicated dist stream
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer stream_memory

    3. optimized: pre-allocated in-place copy on a side stream, a2a on the main stream
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer stream_memory \
        --name=optimized \
        --preallocated=True

    use case:
        study the CUDA memory footprint when overlapping an H2D copy and an
        all-to-all with compute across streams. A side-stream copy/comm overlaps
        well but inflates that stream's reserved memory; pre-allocating on the main
        stream + in-place copy, and letting all_to_all_single use NCCL's own async
        stream, keeps the overlap while avoiding the extra footprint.
        see https://github.com/meta-pytorch/torchrec/pull/3480 for more details
    """

    memory_snapshot: bool = True
    multi_stream: bool = True
    preallocated: bool = False


@register_benchmark(StreamMemoryConfig)
def stream_memory(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    multi_stream: bool = True,
    preallocated: bool = False,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Consolidates the three multi-stream memory-footprint benchmarks. It overlaps an
    H2D copy and an all-to-all with compute, and the flags select the strategy:
      - single_stream     (multi_stream=False): copy and a2a on the main stream
      - multi_stream       (multi_stream=True): copy on a side stream, a2a on a
                            dedicated dist stream
      - optimized          (multi_stream=True, preallocated=True): pre-allocated
                            in-place copy on a side stream, and a2a on the main
                            stream relying on NCCL's own async stream -- keeps the
                            overlap without the extra reserved-memory footprint
    """
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = torch.cuda.Stream() if multi_stream else nullcontext()
        # the optimized variant lets dist.all_to_all_single use its own async stream,
        # so a dedicated dist stream is only needed for the plain multi-stream case
        data_dist_stream = (
            torch.cuda.Stream() if multi_stream and not preallocated else nullcontext()
        )
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

        # the host to device data transfer will block cuda execution without the `pin_memory()`
        host_data = (torch.rand(dim, dim) - 0.5).pin_memory()
        if preallocated:
            # pre-allocate on the main stream so the freed memory can be reused by it
            device_data = torch.empty_like(host_data, device=ctx.device)
        else:
            # the .to() copy below allocates the device tensor on the copy stream
            device_data = torch.empty(0, device=ctx.device)

    with record_function("## irrelevant compute before h2d ##"):
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    with record_function("## copy data to device ##"):
        # use a separate stream to copy data to device, this will not block the main stream
        with data_copy_stream:
            if preallocated:
                # in-place copy into the main-stream allocation
                device_data.copy_(host_data, non_blocking=True)
            else:
                device_data = host_data.to(ctx.device, non_blocking=True)
                if multi_stream:
                    # record to the main stream so it isn't freed early on the copy stream
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

    # the optimized variant runs a2a on the main stream (NCCL async); the others run
    # it on data_dist_stream (a side stream when multi_stream)
    checks: DeviceToHostTensorAwaitable | None = None
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
            if isinstance(data_dist_stream, torch.cuda.Stream):
                # record to the main stream so it isn't freed early on the dist stream
                post_comms.record_stream(main_stream)
        if not preallocated:
            with record_function("## a2a comm validation ##"):
                # validate in the dist stream since there's no data dependency after
                assert req is not None
                req.wait()
                checks = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## irrelevant compute after a2a ##"):
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5
        pre_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=irrelevant_data
        )

    if preallocated:
        with record_function("## a2a comm validation ##"):
            # this req.wait() can be wrapped into a LazyAwaitable
            assert req is not None
            req.wait()
            # still want the compute on the main stream if possible
            checks = DeviceToHostTensorAwaitable(_validate(post_comms, ctx))

    with record_function("## post-comms compute ##"):
        assert req is not None
        req.wait()
        post_comms = _compute(
            dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx, x=post_comms[0]
        )

    with record_function("## assert ##"):
        assert checks is not None
        assert checks.item()


@dataclass
class ThreadingCopyConfig(DataCopyConfig):
    """
    run commands:
    1. multi-threaded (default): issue the H2D copies from a worker thread
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer threading_copy

    2. single-threaded: run the copies inline on the main thread
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer threading_copy \
        --name=single_thread \
        --multithreading=False

    use case:
        in production models the input batch has 500+ tensors copied host-to-device
        one by one; even though each copy is non-blocking, the per-call CPU overhead
        (Python loop + CUDA driver dispatch) accumulates and blocks the main thread
        from dispatching forward-pass kernels. This benchmark simulates that with many
        small pinned tensors and checks whether issuing the copies from a separate
        Python thread (via concurrent.futures.ThreadPoolExecutor, on a side CUDA
        stream) frees the main thread and lets the copies overlap main-stream compute.
        CUDA calls release the GIL so the background thread does not block the main
        thread, though the Python copy loop itself still contends on the GIL.
        see https://github.com/meta-pytorch/torchrec/pull/3774 for more details
    """

    memory_snapshot: bool = True
    multithreading: bool = True


@register_benchmark(ThreadingCopyConfig)
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
        device_tensors: List[torch.Tensor] = [
            torch.empty_like(t, device=ctx.device) for t in host_tensors
        ]

        # large tensor on gpu for main-stream compute
        irrelevant_data = torch.rand(dim, dim, device=ctx.device) - 0.5

    def _copy_worker() -> None:
        torch.cuda.set_device(ctx.device)
        with torch.cuda.stream(data_copy_stream):
            for i in range(num_tensors):
                device_tensors[i].copy_(host_tensors[i], non_blocking=True)

    # launch the copy via ThreadPoolExecutor
    executor: ThreadPoolExecutor | None = None
    future: Future | None = None
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
            assert future is not None
            future.result()
            assert executor is not None
            executor.shutdown(wait=False)

    with record_function("## wait for copy stream ##"):
        main_stream.wait_stream(data_copy_stream)

    # use the copied data to prove it arrived correctly
    with record_function("## use copied data ##"):
        # result intentionally discarded: this only proves the copy arrived
        _compute(dim=dummy_dim, num_mul=1, num_concat=1, ctx=ctx, x=device_tensors[0])


@dataclass
class DataCopyRecordStreamConfig(DataCopyConfig):
    """
    run commands:
    1. with record_stream (correct, default)
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer data_copy_record_stream

    2. without record_stream (reproduces the bug)
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer data_copy_record_stream \
        --name=no_record_stream \
        --use_record_stream=False

    use case:
        reproduce the NaN corruption bug fixed by D102202725 in EMS memory stashing.
        record_stream() tells the CUDA caching allocator to defer reuse of a tensor's
        memory until the recording stream catches up. Without it, freeing the source
        via untyped_storage().resize_(0) on the main stream lets the allocator hand the
        block to a new allocation while an async D2H copy on a side stream is still
        reading it, corrupting the copied data (NaN / wrong values). With
        use_record_stream the copy validates correctly; without it validation fails.
        see https://github.com/meta-pytorch/torchrec/pull/4231 for more details
    """

    memory_snapshot: bool = True
    use_record_stream: bool = True


@register_benchmark(DataCopyRecordStreamConfig)
def data_copy_record_stream(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    use_record_stream: bool = True,
    **_kwargs: Dict[str, Any],
) -> None:
    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        data_copy_stream = torch.cuda.Stream()

        gpu_tensor = torch.full(
            (num_concat * dim, dim),
            fill_value=float(ctx.rank + 1),
            device=ctx.device,
        )
        cpu_buffer = torch.empty_like(gpu_tensor, device="cpu").pin_memory()

    with record_function("## async D2H copy on side stream ##"):
        with torch.cuda.stream(data_copy_stream):
            data_copy_stream.wait_stream(main_stream)
            cpu_buffer.copy_(gpu_tensor, non_blocking=True)

        if use_record_stream:
            gpu_tensor.record_stream(data_copy_stream)

    with record_function("## free source tensor ##"):
        # untyped_storage().resize_(0) releases the underlying GPU memory
        # back to the caching allocator. Without record_stream() above,
        # the allocator may immediately reuse this memory for new
        # allocations while the D2H copy is still reading from it.
        gpu_tensor.untyped_storage().resize_(0)

    with record_function("## new allocations that reuse freed HBM ##"):
        _: list[torch.Tensor] = [
            torch.full(
                (num_concat * dim, dim),
                fill_value=-1.0,
                device=ctx.device,
            )
            for _ in range(num_mul)
        ]

    with record_function("## wait and validate ##"):
        data_copy_stream.synchronize()
        expected = float(ctx.rank + 1)
        all_correct = bool(torch.all(cpu_buffer == expected).item())
        has_nan = bool(torch.any(torch.isnan(cpu_buffer)).item())
        logger.info(
            f"[rank-{ctx.rank}] record_stream={use_record_stream}: "
            f"correct={all_correct}, has_nan={has_nan}"
        )

    with record_function("## validation result ##"):
        if has_nan:
            logger.error(f"[rank-{ctx.rank}] NaN detected — missing record_stream()")
        if not all_correct:
            logger.error(f"[rank-{ctx.rank}] Data corruption — missing record_stream()")


@dataclass
class CopyEngineContentionConfig(DataCopyConfig):
    """
    run commands:
    1. default: back-to-back trunked H2D vs small copies on the main stream
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer copy_engine_contention

    2. dummy compute between trunks
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer copy_engine_contention \
        --name=dummy_compute \
        --dummy_compute=True

    3. dummy compute + 10x trunks (smaller trunks, compute fires more often)
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer copy_engine_contention \
        --name=dummy_compute_10x_trunk \
        --dummy_compute=True \
        --trunk_count=30

    use case:
        reproduce and mitigate CUDA copy-engine contention. A bulk H2D copy (e.g. EMS
        embedding restore) saturates the GPU's single same-direction DMA copy engine,
        blocking small H2D copies -- and the dependent compute -- on other streams. The
        copy engine is priority-blind and current-stream-first, so consecutive
        same-stream trunks run back-to-back and chunking alone never yields. Here the
        h2d_stream runs one bulk H2D split into trunk_count trunks via chunked_copy_,
        while the main stream interleaves a small D2H and two small H2D copies with
        compute. With dummy_compute a tiny op is inserted between trunks, forcing the
        engine to yield so the main stream's small H2D copies slip into the gaps (the
        D2H is unaffected -- it uses a separate copy engine). Smaller trunks (larger
        trunk_count) tighten the contention latency at a small bandwidth cost.
        see https://github.com/meta-pytorch/torchrec/pull/4303 and
        https://github.com/meta-pytorch/torchrec/pull/4339 for more details
    """

    memory_snapshot: bool = True
    dummy_compute: bool = False
    trunk_count: int = 3


@register_benchmark(CopyEngineContentionConfig)
def copy_engine_contention(
    _batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
    dummy_compute: bool = False,
    trunk_count: int = 3,
    **_kwargs: Dict[str, Any],
) -> None:
    """
    Test copy engine contention between two streams issuing PCIe transfers.

    h2d_stream: one large H2D transfer split into ``trunk_count`` trunks via
                ``chunked_copy_`` (saturates copy engine)
    main stream: 1 small D2H copy + 2 small H2D copies (interleaved with
                 small compute ops)

    When dummy_compute=True, ``chunked_copy_`` inserts a tiny compute op
    (tensor.add_(1)) between trunks on the h2d_stream, testing whether the copy
    engine treats back-to-back copies differently from compute-separated ones.
    """
    small_size = 1024

    with record_function("## setup ##"):
        main_stream = torch.cuda.current_stream()
        h2d_stream = torch.cuda.Stream()

    with record_function("## allocate tensors ##"):
        # One large H2D source (pinned CPU -> GPU), copied in trunks via
        # chunked_copy_. The chunk size is one trunk, so the transfer is split
        # into ``trunk_count`` back-to-back copies on the copy engine.
        trunk_rows = num_concat * dim // 5
        large_h2d_source = torch.full(
            (trunk_count * trunk_rows, dim),
            fill_value=float(ctx.rank + 1),
        ).pin_memory()
        large_h2d_device = torch.empty_like(large_h2d_source, device=ctx.device)
        trunk_size_bytes = trunk_rows * dim * large_h2d_source.element_size()

        # 1 small D2H source on GPU
        d2h_source = torch.full(
            (small_size,), fill_value=float(ctx.rank + 10), device=ctx.device
        )
        d2h_cpu_buffer = torch.empty_like(d2h_source, device="cpu").pin_memory()

        # 2 small H2D sources (pinned CPU)
        small_h2d_sources = [
            torch.full(
                (small_size,), fill_value=float(ctx.rank + 20 + i), dtype=torch.int32
            ).pin_memory()
            for i in range(2)
        ]

    with record_function("## pre-copy compute ##"):
        _compute(dim=dim, num_mul=num_mul * 5, num_concat=num_concat, ctx=ctx)

    # --- h2d_stream: one large H2D copy, trunked via chunked_copy_ ---
    with record_function("## large trunked h2d on h2d_stream ##"):
        with torch.cuda.stream(h2d_stream):
            h2d_stream.wait_stream(main_stream)
            chunked_copy_(
                large_h2d_device,
                large_h2d_source,
                chunk_size_bytes=trunk_size_bytes,
                dummy_compute=dummy_compute,
            )
            large_h2d_device.record_stream(main_stream)

    # --- main stream: compute, small d2h, compute, small h2d, compute, small h2d ---
    with record_function("## small compute 0 ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx)

    with record_function("## small d2h on main stream ##"):
        d2h_cpu_buffer.copy_(d2h_source, non_blocking=True)

    with record_function("## small compute 1 ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx)

    with record_function("## small h2d 0 on main stream ##"):
        small_h2d_device_0 = small_h2d_sources[0].to(ctx.device, non_blocking=True)

    with record_function("## small compute 2 ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=1, ctx=ctx)

    with record_function("## small h2d 1 on main stream ##"):
        small_h2d_device_1 = small_h2d_sources[1].to(ctx.device, non_blocking=True)

    with record_function("## irrelevant compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## wait and validate ##"):
        h2d_stream.synchronize()
        torch.cuda.synchronize()

        # validate large H2D copy
        large_correct = bool(torch.all(large_h2d_device == float(ctx.rank + 1)).item())

        # validate small D2H copy
        d2h_correct = bool(torch.all(d2h_cpu_buffer == float(ctx.rank + 10)).item())

        # validate small H2D copies
        small_h2d_correct_0 = bool(
            torch.all(small_h2d_device_0 == ctx.rank + 20).item()
        )
        small_h2d_correct_1 = bool(
            torch.all(small_h2d_device_1 == ctx.rank + 21).item()
        )

        logger.info(
            f"[rank-{ctx.rank}] dummy_compute={dummy_compute}: "
            f"large_h2d={large_correct}, d2h={d2h_correct}, "
            f"small_h2d_0={small_h2d_correct_0}, small_h2d_1={small_h2d_correct_1}"
        )

    with record_function("## post-copy compute ##"):
        _compute(dim=dim, num_mul=num_mul, num_concat=num_concat, ctx=ctx)

    with record_function("## assert ##"):
        assert large_correct, f"Large H2D validation failed on rank {ctx.rank}"
        assert d2h_correct, f"D2H validation failed on rank {ctx.rank}"
        assert small_h2d_correct_0, f"Small H2D 0 validation failed on rank {ctx.rank}"
        assert small_h2d_correct_1, f"Small H2D 1 validation failed on rank {ctx.rank}"


@dataclass
class H2DExecutionOrderConfig(DataCopyConfig):
    """
    run commands:
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer h2d_execution_order

    use case:
        observe the copy engine's execution order across three streams that reach their
        H2D copies at different times (controlled by different pre-copy compute).
        Reverse-engineered model: the engine uses readiness-based dispatch (it runs
        whichever copy becomes ready next, regardless of CPU issue order) with
        current-stream-first scheduling (consecutive same-stream copies run back-to-back
        unless a dummy op breaks their readiness). The CPU issues the long-compute
        stream first but its copy reaches the engine last -- check the trace to confirm.
        see https://github.com/meta-pytorch/torchrec/pull/4315 for more details
    """

    memory_snapshot: bool = True


@register_benchmark(H2DExecutionOrderConfig)
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


@dataclass
class A2AD2HContentionConfig(DataCopyConfig):
    """
    run commands:
    1. concurrent (default): a2a and D2H run on separate streams simultaneously
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer a2a_d2h_contention

    2. sequential: D2H waits for the a2a to finish (contention-free baseline)
    > python -m torchrec.distributed.benchmark.benchmark_data_transfer a2a_d2h_contention \
        --name=sequential \
        --concurrent=False

    use case:
        measure whether a device-to-host copy (PCIe copy engine) contends with
        all_to_all_single (NVLink/NVSwitch) when overlapped. Running both concurrently
        vs sequentially (the contention-free baseline) shows comparable a2a latency:
        D2H and NCCL collectives use largely independent data paths, so they are safe to
        overlap. The concurrent variant surfaces any shared-resource contention (copy
        engine, memory controller, L2 bandwidth) as increased latency in the trace.
        see https://github.com/meta-pytorch/torchrec/pull/4292 for more details
    """

    memory_snapshot: bool = True
    concurrent: bool = True


@register_benchmark(A2AD2HContentionConfig)
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


if __name__ == "__main__":
    # pyrefly: ignore[missing-attribute]
    _cc.main()
