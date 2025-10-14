#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16]

#!/usr/bin/env python3

import argparse
import functools
import inspect
import json
import logging
import os
import resource
import sys
import time
import timeit
from dataclasses import dataclass, fields, is_dataclass, MISSING
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import yaml
from torch import multiprocessing as mp
from torch.autograd.profiler import record_function
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger()

# Reference: https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    4833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]

EMBEDDING_DIM: int = 128
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100_000


class CompileMode(Enum):
    EAGER = "eager"
    FX_SCRIPT = "fx_script"


@dataclass
class GPUMemoryStats:
    rank: int
    malloc_retries: int
    max_mem_allocated_mbs: int
    max_mem_reserved_mbs: int

    @classmethod
    def for_device(cls, rank: int) -> "GPUMemoryStats":
        stats = torch.cuda.memory_stats(rank)
        alloc_retries = stats.get("num_alloc_retries", 0)
        max_allocated = stats.get("allocated_bytes.all.peak", 0)
        max_reserved = stats.get("reserved_bytes.all.peak", 0)
        return cls(
            rank,
            alloc_retries,
            max_allocated // 1024 // 1024,
            max_reserved // 1024 // 1024,
        )

    def __str__(self) -> str:
        return f"Rank {self.rank}: retries={self.malloc_retries}, allocated={self.max_mem_allocated_mbs:7}mb, reserved={self.max_mem_reserved_mbs:7}mb"


@dataclass
class CPUMemoryStats:
    rank: int
    peak_rss_mbs: int

    @classmethod
    def for_process(cls, rank: int) -> "CPUMemoryStats":
        # Peak RSS from resource.getrusage (in KB on CentOS/Linux)
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss_kb // 1024

        return cls(rank, peak_rss_mb)

    def __str__(self) -> str:
        return f"Rank {self.rank}: CPU Memory Peak RSS: {self.peak_rss_mbs/1000:.2f} GB"


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    gpu_elapsed_time: torch.Tensor  # milliseconds
    cpu_elapsed_time: torch.Tensor  # milliseconds
    gpu_mem_stats: List[GPUMemoryStats]  # GPU memory stats per rank
    cpu_mem_stats: List[CPUMemoryStats]  # CPU memory stats per rank
    rank: int = -1

    def __str__(self) -> str:
        gpu_runtime = (
            f"GPU Runtime (P90): {self.runtime_percentile(90, device='gpu'):.2f} ms"
        )
        cpu_runtime = (
            f"CPU Runtime (P90): {self.runtime_percentile(90, device='cpu'):.2f} ms"
        )
        cpu_mem = f"CPU Peak RSS (P90): {self.cpu_mem_percentile(90)/1000:.2f} GB"

        if len(self.gpu_mem_stats) == 0:
            return (
                f"{self.short_name: <{35}} |  {gpu_runtime} | {cpu_runtime} | {cpu_mem}"
            )
        mem_alloc = f"GPU Peak Memory alloc (P90): {self.max_mem_alloc_percentile(90)/1000:.2f} GB"
        mem_reserved = f"GPU Peak Memory reserved (P90): {self.max_mem_reserved_percentile(90)/1000:.2f} GB"
        malloc_retries = f"Malloc retries (P50/P90/P100): {self.mem_retries(50)} / {self.mem_retries(90)} / {self.mem_retries(100)}"
        return f"{self.short_name: <{35}} | {malloc_retries} | {gpu_runtime} | {cpu_runtime} | {mem_alloc} | {mem_reserved} | {cpu_mem}"

    def runtime_percentile(
        self,
        percentile: int = 50,
        interpolation: str = "nearest",
        device: str = "gpu",
    ) -> torch.Tensor:
        """Return the runtime percentile for the requested timer.

        Args:
            percentile: Percentile to compute.
            interpolation: See ``torch.quantile``.
            device: 'gpu' for CUDA event timings, 'cpu' for active CPU timings.
        """
        timings = self.gpu_elapsed_time if device == "gpu" else self.cpu_elapsed_time
        return torch.quantile(
            timings,
            percentile / 100.0,
            interpolation=interpolation,
        )

    def max_mem_alloc_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_allocated_mbs, percentile, interpolation
        )

    def max_mem_reserved_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_reserved_mbs, percentile, interpolation
        )

    def mem_retries(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.malloc_retries, percentile, interpolation
        )

    def _mem_percentile(
        self,
        mem_selector: Callable[[GPUMemoryStats], int],
        percentile: int = 50,
        interpolation: str = "nearest",
    ) -> torch.Tensor:
        mem_data = torch.tensor(
            [mem_selector(mem_stat) for mem_stat in self.gpu_mem_stats],
            dtype=torch.float,
        )
        return torch.quantile(mem_data, percentile / 100.0, interpolation=interpolation)

    def cpu_mem_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        """Return the CPU memory percentile for peak RSS."""
        cpu_mem_data = torch.tensor(
            [cpu_stat.peak_rss_mbs for cpu_stat in self.cpu_mem_stats],
            dtype=torch.float,
        )
        return torch.quantile(
            cpu_mem_data, percentile / 100.0, interpolation=interpolation
        )


T = TypeVar("T", bound=torch.nn.Module)


def write_report(
    benchmark_results: List[BenchmarkResult],
    report_file: str,
    report_str: str,
    num_requests: int,
) -> None:
    for benchmark_res in benchmark_results:
        # GPU statistics
        avg_dur_s_gpu = benchmark_res.gpu_elapsed_time.mean().item() * 1e-3  # sec
        std_dur_s_gpu = benchmark_res.gpu_elapsed_time.std().item() * 1e-3  # sec

        # CPU statistics
        avg_dur_s_cpu = benchmark_res.cpu_elapsed_time.mean().item() * 1e-3  # sec
        std_dur_s_cpu = benchmark_res.cpu_elapsed_time.std().item() * 1e-3  # sec

        qps_gpu = int(num_requests / avg_dur_s_gpu)

        mem_str = ""
        for gpu_memory_stats in benchmark_res.gpu_mem_stats:
            mem_str += f"{gpu_memory_stats}\n"

        for cpu_memory_stats in benchmark_res.cpu_mem_stats:
            mem_str += f"{cpu_memory_stats}\n"

        report_str += (
            f"{benchmark_res.short_name:40} "
            f"Avg QPS(GPU):{qps_gpu:10} "
            f"GPU Avg: {int(1000*avg_dur_s_gpu):5}ms ±{(1000*std_dur_s_gpu):.2f}ms "
            f"CPU Avg: {int(1000*avg_dur_s_cpu):5}ms ±{(1000*std_dur_s_cpu):.2f}ms\n"
        )
        report_str += f"\tMemory Allocated Per Rank:\n\t{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def multi_process_benchmark(
    callable: Callable[
        ...,
        None,
    ],
    # pyre-ignore
    **kwargs,
) -> BenchmarkResult:

    def setUp() -> None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = str("localhost")
            os.environ["MASTER_PORT"] = str(get_free_port())

    assert "world_size" in kwargs
    world_size = kwargs["world_size"]

    setUp()
    benchmark_res_per_rank = []
    # kineto has a known problem with fork-server: it'll hang
    # when dumping the trace. Workaround with spawn
    ctx = mp.get_context("spawn")
    qq = ctx.SimpleQueue()
    processes = []

    for rank in range(world_size):
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        kwargs["queue"] = qq
        p = ctx.Process(
            target=callable,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    for _ in range(world_size):
        res = qq.get()

        benchmark_res_per_rank.append(res)
        assert len(res.gpu_mem_stats) == 1
        assert len(res.cpu_mem_stats) == 1

    for p in processes:
        p.join()
        assert 0 == p.exitcode

    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        gpu_mem_stats=[GPUMemoryStats(rank, 0, 0, 0) for rank in range(world_size)],
        cpu_mem_stats=[CPUMemoryStats(rank, 0) for rank in range(world_size)],
        rank=0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 GPU and 1 CPU memory measurement
        total_benchmark_res.gpu_mem_stats[res.rank] = res.gpu_mem_stats[0]
        total_benchmark_res.cpu_mem_stats[res.rank] = res.cpu_mem_stats[0]

    return total_benchmark_res


def set_embedding_config(
    embedding_config_json: str,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    the config file should follow this pattern: {feature: {num_embeddings: int, embedding_dim: int}}
    """
    embedding_configs = []
    pooling_configs = []
    has_pooling_config = False
    try:
        if os.path.exists(embedding_config_json):
            with open(embedding_config_json, "r") as f:
                embedding_config_json = json.load(f)

            for _, config in embedding_config_json.items():
                embedding_configs.append(
                    (config["num_embeddings"], config["embedding_dim"])
                )
                if "pooling_factor" in config:
                    pooling_configs.append(config["pooling_factor"])
                    has_pooling_config = True
                else:
                    if has_pooling_config:
                        raise RuntimeError(
                            "We cannot handle some features have pooling factor and others don't."
                        )
        else:
            raise RuntimeError(
                f"Could not find embedding config json at path {embedding_config_json}"
            )
    except BaseException as e:
        logger.warning(
            f"Failed to load embedding config because {e}, fallback to DLRM config"
        )
        embedding_configs = [
            (num_embeddings, EMBEDDING_DIM)
            for num_embeddings in DLRM_NUM_EMBEDDINGS_PER_FEATURE
        ]

    return embedding_configs, pooling_configs


class cmd_conf:
    """
    Decorator for run functions in command line.
    parse input arguments into the function's arguments and config (dataclass)

    Example 1: direct decorating (see the overloaded __new__ method below)
    ```
    @cmd_conf  # you might need "pyre-ignore [56]"
    def main(
        run_option: RunOptions,
        table_config: EmbeddingTablesConfig,
        model_selection: ModelSelectionConfig,
        pipeline_config: PipelineConfig,
        model_config: Optional[BaseModelConfig] = None,
        integer: int
    ) -> None:
        pass

    if __name__ == "__main__":
        main()
    ```

    Example 2: register multiple function
    invoke with: -- (run1|run2) --arg1=...
    ```
    _cc = cmd_conf()
    @_cc.register
    def func1(input_config: CONF1):
        pass

    @_cc.register
    def func2(input_config: CONF2):
        pass

    if __name__ == "__main__":
        _cc.main()
    ```
    """

    def __init__(self) -> None:
        # pyre-ignore [24]
        self.programs: Dict[str, Callable] = {}

    @classmethod
    # pyre-ignore [24]
    def __new__(cls, _, func: Optional[Callable] = None) -> Union["cmd_conf", Callable]:
        if not func:
            return super().__new__(cls)
        else:
            return cmd_conf.call(func)

    @staticmethod
    def call(func: Callable) -> Callable:  # pyre-ignore [24]

        def _load_config_file(
            config_path: str, is_json: bool = False
        ) -> Dict[str, Any]:
            if not config_path:
                return {}

            try:
                with open(config_path, "r") as f:
                    if is_json:
                        return json.load(f) or {}
                    else:
                        return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(
                    f"Failed to load config because {e}. Proceeding without it."
                )
                return {}

        @functools.wraps(func)
        def wrapper() -> Any:  # pyre-ignore [3]
            sig = inspect.signature(func)
            parser = argparse.ArgumentParser(func.__doc__)

            parser.add_argument(
                "--yaml_config",
                type=str,
                default=None,
                help="YAML config file for benchmarking",
            )

            parser.add_argument(
                "--json_config",
                type=str,
                default=None,
                help="JSON config file for benchmarking",
            )

            # Add loglevel argument with current logger level as default
            parser.add_argument(
                "--loglevel",
                type=str,
                default=logging._levelToName[logger.level],
                help="Set the logging level (e.g. info, debug, warning, error)",
            )

            pre_args, _ = parser.parse_known_args()

            yaml_defaults: Dict[str, Any] = (
                _load_config_file(pre_args.yaml_config, is_json=False)
                if pre_args.yaml_config
                else {}
            )
            json_defaults: Dict[str, Any] = (
                _load_config_file(pre_args.json_config, is_json=True)
                if pre_args.json_config
                else {}
            )
            # Merge the two dictionaries, JSON overrides YAML
            merged_defaults = {**yaml_defaults, **json_defaults}

            seen_args = set()  # track all --<name> we've added

            for _name, param in sig.parameters.items():
                cls = param.annotation
                if not is_dataclass(cls):
                    continue

                for f in fields(cls):
                    arg_name = f.name
                    if arg_name in seen_args:
                        logger.warning(f"WARNING: duplicate argument {arg_name}")
                        continue
                    seen_args.add(arg_name)

                    ftype = f.type
                    origin = get_origin(ftype)

                    # Unwrapping Optional[X] to X
                    if origin is Union and type(None) in get_args(ftype):
                        non_none = [t for t in get_args(ftype) if t is not type(None)]
                        if len(non_none) == 1:
                            ftype = non_none[0]
                            origin = get_origin(ftype)

                    # Handle default_factory value and allow config to override
                    default_value = merged_defaults.get(
                        arg_name,  # flat lookup
                        merged_defaults.get(cls.__name__, {}).get(  # hierarchy lookup
                            arg_name,
                            (
                                f.default_factory()  # pyre-ignore [29]
                                if f.default_factory is not MISSING
                                else f.default
                            ),
                        ),
                    )

                    arg_kwargs = {
                        "default": default_value,
                        "help": f"({cls.__name__}) {arg_name}",
                    }

                    if origin in (list, List):
                        elem_type = get_args(ftype)[0]
                        arg_kwargs.update(nargs="*", type=elem_type)
                    elif ftype is bool:
                        # Special handling for boolean arguments
                        arg_kwargs.update(
                            type=lambda x: x.lower() in ["true", "1", "yes"]
                        )
                    else:
                        arg_kwargs.update(type=ftype)

                    parser.add_argument(f"--{arg_name}", **arg_kwargs)

            args = parser.parse_args()
            logger.setLevel(logging.INFO)

            # Build the dataclasses
            kwargs = {}
            for name, param in sig.parameters.items():
                cls = param.annotation
                if is_dataclass(cls):
                    data = {f.name: getattr(args, f.name) for f in fields(cls)}
                    config_instance = cls(**data)  # pyre-ignore [29]
                    kwargs[name] = config_instance
                    logger.info(config_instance)

            loglevel = logging._nameToLevel[args.loglevel.upper()]
            logger.setLevel(loglevel)

            return func(**kwargs)

        return wrapper

    # pyre-ignore [24]
    def register(self, func: Callable) -> Callable:
        wrapper = cmd_conf.call(func)
        self.programs[func.__name__] = wrapper
        return wrapper

    def main(self) -> None:
        program = sys.argv[1]
        if program in self.programs:
            sys.argv[:] = [sys.argv[0]] + (sys.argv[2:] if len(sys.argv) > 2 else [])
            self.programs[program]()
        else:
            print(
                f"Invalid command. Please use select program from {', '.join(self.programs.keys())}."
            )


def create_trace_file_name(profile_name: str, rank: int) -> str:
    """Create a unique trace file name for the given rank and profile name."""
    return f"trace-{profile_name}-rank{rank}.json.gz"


def init_argparse_and_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=500)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--max_num_embeddings", type=int, default=1000000)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=5)
    parser.add_argument("--embedding_config_json", type=str, default="")
    parser.add_argument("--device_type", type=str, default="cuda")

    args = parser.parse_args()

    return args


def _run_benchmark_core(
    name: str,
    run_iter_fn: Callable[[], None],
    profile_iter_fn: Optional[Callable[[Any], None]],  # pyre-ignore [2]
    world_size: int,
    rank: int,
    num_benchmarks: int,
    device_type: str,
    output_dir: str,
    pre_gpu_load: int = 0,
    export_stacks: bool = False,
    reset_accumulated_memory_stats: bool = False,
    all_rank_traces: bool = False,
    memory_snapshot: bool = False,
) -> BenchmarkResult:
    """Internal helper that contains the core benchmarking logic shared by
    ``benchmark`` and ``benchmark_func``.  All heavy–lifting (timing, memory
    accounting, optional profiling) happens here so the public helpers can stay
    small and focused on preparing the callables to execute.

    Args:
        name: Human-readable benchmark name.

        run_iter_fn: Zero-arg callable that executes one measured iteration.
        profile_iter_fn: Optional callable that receives a ``torch.profiler``
            instance and runs the iterations that should be captured.

        world_size, rank: Distributed context to correctly reset / collect GPU
            stats. ``rank == -1`` means single-process mode.
        num_benchmarks: Number of measured iterations.

        device_type: "cuda" or "cpu".
        output_dir: Where to write chrome traces / stack files.
        pre_gpu_load: Number of dummy matmul operations to run before the first
            measured iteration (helps simulating a loaded allocator).
        export_stacks: Whether to export flamegraph-compatible stack files.
        reset_accumulated_memory_stats: Whether to reset accumulated memory
            stats in addition to peak memory stats.
    """

    # Preparation & memory reset
    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                torch.cuda.reset_peak_memory_stats(di)
                if reset_accumulated_memory_stats:
                    torch.cuda.reset_accumulated_memory_stats(di)
        else:
            torch.cuda.reset_peak_memory_stats(rank)
            if reset_accumulated_memory_stats:
                torch.cuda.reset_accumulated_memory_stats(rank)

        # Optional allocator warm-up to create fragmentation similar to production
        if pre_gpu_load:
            _tmp = torch.rand(16384, 16384, device="cuda")
            for _ in range(pre_gpu_load):
                _tmp = _tmp * torch.rand(16384, 16384, device="cuda")

    # Timings
    start_events, end_events, times = [], [], []

    if device_type == "cuda":
        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)
        ]
        end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)
        ]
        # Capture per-iteration active CPU cycles (excludes time the thread is truly idle/asleep) using `process_time_ns`.
        cpu_times_active_ns: List[int] = []

        for i in range(num_benchmarks):
            # Ensure that outstanding GPU work from the previous iteration has
            # finished so that we do not attribute its wait time to the next
            # CPU measurement.
            if i > 0:
                torch.cuda.synchronize(rank if rank >= 0 else 0)

            start_events[i].record()
            cpu_start_active_ns = time.process_time_ns()

            run_iter_fn()

            cpu_end_active_ns = time.process_time_ns()
            end_events[i].record()
            cpu_times_active_ns.append(cpu_end_active_ns - cpu_start_active_ns)

        # Convert to milliseconds and drop the first iteration
        cpu_elapsed_time = (
            torch.tensor([t / 1e6 for t in cpu_times_active_ns[1:]], dtype=torch.float)
            if num_benchmarks >= 2
            else torch.zeros(1, dtype=torch.float)
        )

        # Make sure all kernels are finished before reading timers / stats
        if rank == -1:
            for di in range(world_size):
                torch.cuda.synchronize(di)
        else:
            torch.cuda.synchronize(rank)

        gpu_elapsed_time = (
            torch.tensor(
                [s.elapsed_time(e) for s, e in zip(start_events[1:], end_events[1:])]
            )
            if num_benchmarks >= 2
            else torch.zeros(1, dtype=torch.float)
        )
    else:
        # For CPU-only benchmarks we fall back to wall-clock timing via ``timeit``.
        times = timeit.repeat(run_iter_fn, number=1, repeat=num_benchmarks)
        cpu_elapsed_time = torch.tensor(times) * 1e3  # convert to ms

        # mirror CPU timings for overall consistency
        gpu_elapsed_time = cpu_elapsed_time.clone()

    # Memory statistics collection
    gpu_mem_stats: List[GPUMemoryStats] = []
    cpu_mem_stats = [CPUMemoryStats.for_process(rank)]

    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                gpu_mem_stats.append(GPUMemoryStats.for_device(di))
        else:
            gpu_mem_stats.append(GPUMemoryStats.for_device(rank))
    # CPU memory stats are collected for both GPU and CPU-only runs

    # Optional detailed profiling
    if output_dir and profile_iter_fn and device_type == "cuda":

        def _trace_handler(prof: torch.profiler.profile) -> None:
            total_avg = prof.profiler.total_average()
            logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_avg}")
            if not all_rank_traces and rank > 0:
                # only save trace for rank 0 when all_rank_traces is disabled
                return
            trace_file = f"{output_dir}/{create_trace_file_name(name, rank)}"
            logger.info(f" PROFILE[{name}].chrome_trace:{trace_file}")
            prof.export_chrome_trace(trace_file)

            if export_stacks:
                prof.export_stacks(
                    f"{output_dir}/stacks-cpu-{name}.stacks", "self_cpu_time_total"
                )
                prof.export_stacks(
                    f"{output_dir}/stacks-cuda-{name}.stacks", "self_cuda_time_total"
                )

        if memory_snapshot:
            torch.cuda.memory._record_memory_history(
                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
            )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=export_stacks,
            on_trace_ready=_trace_handler,
        ) as prof:
            profile_iter_fn(prof)

        # Synchronize again after profiling to guarantee deterministic ordering
        if rank == -1:
            for di in range(torch.cuda.device_count()):
                torch.cuda.synchronize(torch.device(f"cuda:{di}"))
        else:
            torch.cuda.synchronize(rank)

        if memory_snapshot:
            try:
                torch.cuda.memory._dump_snapshot(
                    f"{output_dir}/memory-{name}-rank{rank}.pickle"
                )
            except Exception as e:
                logger.error(f"Failed to capture memory snapshot {e}")

            # Stop recording memory snapshot history.
            torch.cuda.memory._record_memory_history(enabled=None)

    return BenchmarkResult(
        short_name=name,
        gpu_elapsed_time=gpu_elapsed_time,
        cpu_elapsed_time=cpu_elapsed_time,
        gpu_mem_stats=gpu_mem_stats,
        cpu_mem_stats=cpu_mem_stats,
        rank=rank,
    )


def benchmark_model_with_warmup(
    name: str,
    model: torch.nn.Module,
    warmup_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    bench_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    prof_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    world_size: int,
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
    enable_logging: bool = True,
    device_type: str = "cuda",
    benchmark_unsharded_module: bool = False,
    export_stacks: bool = False,
) -> BenchmarkResult:
    if enable_logging:
        logger.info(f" BENCHMARK_MODEL[{name}]:\n{model}")

    # Warm-up forwards to stabilize kernels / JIT compilation
    for _input in warmup_inputs:
        model(_input)

    if benchmark_func_kwargs is None:
        benchmark_func_kwargs = {}

    run_iter_fn: Callable[[], None] = lambda: func_to_benchmark(
        model, bench_inputs, **benchmark_func_kwargs
    )

    def _profile_iter_fn(prof: torch.profiler.profile) -> None:
        for _input in prof_inputs:
            with record_function("## forward ##"):
                model(_input)
                prof.step()

    return _run_benchmark_core(
        name=name,
        run_iter_fn=run_iter_fn,
        profile_iter_fn=_profile_iter_fn if output_dir else None,
        world_size=world_size,
        rank=rank,
        num_benchmarks=num_benchmarks,
        device_type=device_type,
        output_dir=output_dir,
        pre_gpu_load=0,
        export_stacks=export_stacks,
        reset_accumulated_memory_stats=False,
    )


@dataclass
class BenchFuncConfig:
    name: str
    world_size: int
    num_profiles: int
    num_benchmarks: int
    profile_dir: str
    device_type: str = "cuda"
    pre_gpu_load: int = 0
    export_stacks: bool = False
    all_rank_traces: bool = False
    memory_snapshot: bool = False

    # pyre-ignore [2]
    def benchmark_func_kwargs(self, **kwargs_to_override) -> Dict[str, Any]:
        return {
            "name": self.name,
            "world_size": self.world_size,
            "num_profiles": self.num_profiles,
            "num_benchmarks": self.num_benchmarks,
            "profile_dir": self.profile_dir,
            "device_type": self.device_type,
            "pre_gpu_load": self.pre_gpu_load,
            "export_stacks": self.export_stacks,
            "all_rank_traces": self.all_rank_traces,
            "memory_snapshot": self.memory_snapshot,
        } | kwargs_to_override


def benchmark_func(
    name: str,
    rank: int,
    world_size: int,
    func_to_benchmark: Any,  # pyre-ignore[2]
    bench_inputs: List[Dict[str, Any]],
    prof_inputs: List[Dict[str, Any]],
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    num_profiles: int,
    num_benchmarks: int,
    profile_dir: str,
    device_type: str = "cuda",
    pre_gpu_load: int = 0,
    export_stacks: bool = False,
    all_rank_traces: bool = False,
    memory_snapshot: bool = False,
) -> BenchmarkResult:
    """
    Args:
        name: Human-readable benchmark name.
        world_size, rank: Distributed context to correctly reset / collect GPU
            stats. ``rank == -1`` means single-process mode.

        func_to_benchmark: Callable that executes one measured iteration.
            func_to_benchmark(batch_inputs, **kwargs) -> None
        bench_inputs, prof_inputs: List[Dict[str, Any]] this argument will be fed
            to the function at once, and bench_inputs will be used for benchmarking
            while prof_inputs will be used for profiling
        benchmark_func_kwargs: kwargs to be passed to func_to_benchmark

        num_profiles, num_benchmarks: Number of measured iterations, i.e., how many
            times the function will be called
        profile_dir: Where to write chrome traces / stack files.

        device_type: "cuda" or "cpu".
        pre_gpu_load: Number of dummy matmul operations to run before the first
            measured iteration (helps simulating a loaded allocator).
        export_stacks: Whether to export flamegraph-compatible stack files.
        all_rank_traces: Whether to export traces from all ranks.
        memory_snapshot: Whether to capture memory snapshot during the profiling
    """
    if benchmark_func_kwargs is None:
        benchmark_func_kwargs = {}

    run_iter_fn: Callable[[], None] = lambda: func_to_benchmark(
        bench_inputs, **benchmark_func_kwargs
    )

    def _profile_iter_fn(prof: torch.profiler.profile) -> None:
        for i in range(num_profiles):
            with record_function(f"## profile {i} ##"):
                func_to_benchmark(prof_inputs, **benchmark_func_kwargs)
                prof.step()

    return _run_benchmark_core(
        name=name,
        run_iter_fn=run_iter_fn,
        profile_iter_fn=_profile_iter_fn if profile_dir else None,
        world_size=world_size,
        rank=rank,
        num_benchmarks=num_benchmarks,
        device_type=device_type,
        output_dir=profile_dir,
        pre_gpu_load=pre_gpu_load,
        export_stacks=export_stacks,
        reset_accumulated_memory_stats=True,
        all_rank_traces=all_rank_traces,
        memory_snapshot=memory_snapshot,
    )
