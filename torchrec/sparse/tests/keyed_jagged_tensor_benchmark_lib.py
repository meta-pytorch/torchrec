#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import random
import sys
import time
import timeit
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Need this for PT2 compile
# Otherwise will get error
# NotImplementedError: fbgemm::permute_1D_sparse_data: We could not find the abstract impl for this operator.
from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torchrec.distributed.benchmark.base import (
    BenchmarkResult,
    CPUMemoryStats,
    GPUMemoryStats,
)
from torchrec.distributed.dist_data import _get_recat

from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", stream=sys.stdout)
logger.setLevel(logging.DEBUG)

DEFTAULT_BENCHMARK_FUNCS = [
    "permute",
    "to_dict",
    "split",
    "concat",
    "__getitem__",
    "dist_splits",
    "dist_init",
]


class TransformType(Enum):
    DEFAULT = "DEFAULT"
    JIT_SCRIPT = "JIT_SCRIPT"
    VBE = "VBE"
    PT2_AOT_EAGER = "AOT_EAGER"


class BenchmarkFormatter:
    TABLE_HEADERS: List[List[str]] = [
        [
            "Method Name",
            "Transform",
            "Variable Batch",
            "Batch Size",
            "# Features",
            "Avg Pooling Factor",
            "Runtime (P50)",
            "Runtime (P90)",
        ],
        [
            "-------------",
            "-------------",
            "----------------",
            "------------",
            "------------",
            "--------------------",
            "------------------------------",
            "------------------------------",
        ],
    ]

    def __init__(
        self,
        batch_size: int,
        num_features: int,
        mean_pooling_factor: int,
    ) -> None:
        self._batch_size = batch_size
        self._num_features = num_features
        self._mean_pooling_factor = mean_pooling_factor
        self._delimiter = "|"
        # map method name -> (p50 runtime, p90 runtime) in ms
        self._runtime_baseline: Dict[str, Tuple[float, float]] = {}
        self._divider_widths: List[int] = [
            len(divider) for divider in BenchmarkFormatter.TABLE_HEADERS[1]
        ]

    def set_baseline(
        self, method_name: str, p50_runtime: float, p90_runtime: float
    ) -> None:
        self._runtime_baseline[method_name] = (p50_runtime, p90_runtime)

    def print_headers(self) -> None:
        row_format = "|".join(
            [" {:<" + str(w - 2) + "} " for w in self._divider_widths]
        )
        # headers
        logger.info(row_format.format(*self.TABLE_HEADERS[0]))
        # dividers
        logger.info("+".join(self.TABLE_HEADERS[1]))

    def format_width(self, s: str, col_idx: int) -> str:
        return f"{s:<{self._divider_widths[col_idx]-2}}"

    def get_runtime_delta(self, duration: float, baseline: float) -> str:
        if duration <= baseline:
            delta_pct = (baseline - duration) / duration
            direction = "faster"
        else:
            delta_pct = (duration - baseline) / baseline
            direction = "slower"
        return f"{delta_pct * 100.0:.1f}% {direction}"

    def format_runtime(
        self, duration: float, baseline_duration: float, is_baseline: bool
    ) -> str:
        return f"{duration * 1000:<8.3g} ms ({'baseline' if is_baseline else self.get_runtime_delta(duration, baseline_duration)})"

    def print_formatted(
        self,
        method_name: str,
        transform_type: TransformType,
        is_vb: bool,
        p50_runtime: float,
        p90_runtime: float,
        is_baseline: bool,
    ) -> None:
        cols = [
            method_name,
            transform_type.value,
            "Yes" if is_vb else "No",
            self._batch_size,
            self._num_features,
            self._mean_pooling_factor,
            self.format_runtime(
                p50_runtime, self._runtime_baseline[method_name][0], is_baseline
            ),
            self.format_runtime(
                p90_runtime, self._runtime_baseline[method_name][1], is_baseline
            ),
        ]

        row_format = "|".join(
            [" {:<" + str(w - 2) + "} " for w in self._divider_widths]
        )
        logger.info(row_format.format(*cols))


def generate_kjt(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> KeyedJaggedTensor:
    global_input = ModelInput.generate(
        batch_size=batch_size,
        world_size=1,  # 1 for cpu
        num_float_features=0,
        tables=tables,
        weighted_tables=[],
        # mean pooling factor per feature
        tables_pooling=[mean_pooling_factor] * len(tables),
        # returns KJTs with values all set to 0
        # we don't care about KJT values for benchmark, and this saves time
        randomize_indices=True,
        device=device,
    )[0]
    assert isinstance(global_input.idlist_features, KeyedJaggedTensor)
    return global_input.idlist_features


def build_kjts(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> KeyedJaggedTensor:
    start = time.perf_counter()
    logger.info("Starting to build KJTs")

    kjt = generate_kjt(
        tables,
        batch_size,
        mean_pooling_factor,
        device,
    )

    end = time.perf_counter()
    time_taken_s = end - start
    logger.info(f"Took {time_taken_s * 1000:.1f}ms to build KJT\n")
    return kjt


def benchmark_kjt(
    test_name: str,
    # pyre-ignore[2]
    test_module: Union[torch.nn.Module, Callable[..., Any]],
    kjt: KeyedJaggedTensor,
    num_repeat: int,
    num_warmup: int,
    bench_formatter: BenchmarkFormatter,
    fn_kwargs: Dict[str, Any],
    transform_type: TransformType,
    is_vb: bool = False,
    is_baseline: bool = False,
    print_formatted: bool = True,
) -> BenchmarkResult:
    for _ in range(num_warmup):
        # Reset cached states
        kjt.unsync()
        kjt._jt_dict = None
        test_module(**fn_kwargs)

    times = []
    for _ in range(num_repeat):
        # Reset cached states
        kjt.unsync()
        kjt._jt_dict = None

        time_elapsed = timeit.timeit(lambda: test_module(**fn_kwargs), number=1)
        # remove length_per_key and offset_per_key cache for fairer comparison
        times.append(time_elapsed)

    result = BenchmarkResult(
        short_name=f"{test_name}-{transform_type.name}",
        gpu_elapsed_time=torch.tensor(times),
        cpu_elapsed_time=torch.tensor(times),
        gpu_mem_stats=[GPUMemoryStats(0, 0, 0, 0)],
        cpu_mem_stats=[CPUMemoryStats.for_process(0)],
    )

    p50_runtime = result.runtime_percentile(50, interpolation="linear").item()
    p90_runtime = result.runtime_percentile(90, interpolation="linear").item()

    if is_baseline:
        bench_formatter.set_baseline(test_name, p50_runtime, p90_runtime)

    if print_formatted:
        bench_formatter.print_formatted(
            method_name=test_name,
            transform_type=transform_type,
            is_vb=is_vb,
            p50_runtime=p50_runtime,
            p90_runtime=p90_runtime,
            is_baseline=is_baseline,
        )

    return result


def get_k_splits(n: int, k: int) -> List[int]:
    split_size, _ = divmod(n, k)
    splits = [split_size] * (k - 1) + [n - split_size * (k - 1)]
    return splits


def gen_dist_split_input(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    num_workers: int,
    num_features: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
    batch_size_per_rank = get_k_splits(n=batch_size, k=num_workers)
    kjts = [
        generate_kjt(tables, batch_size_rank, mean_pooling_factor, device)
        for batch_size_rank in batch_size_per_rank
    ]
    kjt_lengths = torch.cat([kjt.lengths() for kjt in kjts])
    kjt_values = torch.cat([kjt.values() for kjt in kjts])
    recat = _get_recat(
        local_split=num_features,
        num_splits=num_workers,
        device=device,
        batch_size_per_rank=batch_size_per_rank,
    )

    return (kjt_lengths, kjt_values, batch_size_per_rank, recat)


class KJTPermute(torch.nn.Module):
    def forward(self, kjt: KeyedJaggedTensor, indices: List[int]) -> KeyedJaggedTensor:
        return kjt.permute(indices)


class KJTToDict(torch.nn.Module):
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        return kjt.to_dict()


class KJTSplit(torch.nn.Module):
    def forward(
        self, kjt: KeyedJaggedTensor, segments: List[int]
    ) -> List[KeyedJaggedTensor]:
        return kjt.split(segments)


class KJTConcat(torch.nn.Module):
    def forward(
        self,
        inputs: List[KeyedJaggedTensor],
    ) -> KeyedJaggedTensor:
        return KeyedJaggedTensor.concat(inputs)


class KJTGetItem(torch.nn.Module):
    def forward(self, kjt: KeyedJaggedTensor, key: str) -> JaggedTensor:
        return kjt[key]


class KJTDistSplits(torch.nn.Module):
    def forward(self, kjt: KeyedJaggedTensor, key_splits: List[int]) -> List[List[int]]:
        return kjt.dist_splits(key_splits)


class KJTDistInit(torch.nn.Module):
    def forward(
        self,
        keys: List[str],
        tensors: List[torch.Tensor],
        variable_stride_per_key: bool,
        num_workers: int,
        recat: Optional[torch.Tensor],
        stride_per_rank: Optional[List[int]],
    ) -> KeyedJaggedTensor:
        return KeyedJaggedTensor.dist_init(
            keys, tensors, variable_stride_per_key, num_workers, recat, stride_per_rank
        )


# pyre-ignore
def dynamo_compile(
    method_name: str,
    kjt_module: torch.nn.Module,
    backend: str,
    fullgraph: bool,
    fn_kwargs: Dict[str, Any],
) -> Callable[..., Any]:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    compiled_mod = torch.compile(kjt_module, backend=backend, fullgraph=fullgraph)
    return compiled_mod


def bench(
    num_repeat: int,
    num_warmup: int,
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    num_workers: int,
    test_pt2: bool,
    kjt_funcs: str,
    run_modes: str,
) -> List[BenchmarkResult]:
    # TODO: support CUDA benchmark
    device: torch.device = torch.device("cpu")

    tables: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            num_embeddings=20,  # determines indices range
            embedding_dim=10,  # doesn't matter for benchmark
            name=f"table_{i}",
            feature_names=[f"feature_{i}"],
        )
        for i in range(num_features)
    ]

    kjt_funcs_to_benchmark: List[str] = kjt_funcs.split(",")
    run_modes_to_benchmark: List[str] = run_modes.split(",")

    kjt = build_kjts(
        tables,
        batch_size,
        mean_pooling_factor,
        device,
    )

    splits = get_k_splits(n=num_features, k=8)

    permute_indices = random.sample(range(num_features), k=num_features)

    key = f"feature_{random.randint(0, num_features - 1)}"

    kjt_lengths, kjt_values, strides_per_rank, recat = gen_dist_split_input(
        tables, batch_size, num_workers, num_features, mean_pooling_factor, device
    )

    benchmarked_methods: List[Tuple[str, Dict[str, Any], torch.nn.Module]] = [
        ("permute", {"kjt": kjt, "indices": permute_indices}, KJTPermute()),
        ("to_dict", {"kjt": kjt}, KJTToDict()),
        ("split", {"kjt": kjt, "segments": splits}, KJTSplit()),
        ("concat", {"inputs": [kjt, kjt]}, KJTConcat()),
        ("__getitem__", {"kjt": kjt, "key": key}, KJTGetItem()),
        ("dist_splits", {"kjt": kjt, "key_splits": splits}, KJTDistSplits()),
        (
            "dist_init",
            {
                "keys": kjt.keys(),
                "tensors": [
                    # lengths from each rank, should add up to num_features x batch_size in total
                    kjt_lengths,
                    # values from each rank
                    kjt_values,
                ],
                "variable_stride_per_key": False,
                "num_workers": num_workers,
                "recat": recat,
                "stride_per_rank": strides_per_rank,
            },
            KJTDistInit(),
        ),
    ]

    bench_formatter = BenchmarkFormatter(
        batch_size,
        num_features,
        mean_pooling_factor,
    )
    bench_formatter.print_headers()

    all_results: List[BenchmarkResult] = []

    filtered_benchmarked_methods = [
        row for row in benchmarked_methods if row[0] in kjt_funcs_to_benchmark
    ]

    for method_name, fn_kwargs, kjt_module in filtered_benchmarked_methods:
        if "DEFAULT" in run_modes_to_benchmark:
            # Test Eager
            result = benchmark_kjt(
                test_name=method_name,
                kjt=kjt,
                test_module=kjt_module,
                num_repeat=num_repeat,
                num_warmup=num_warmup,
                fn_kwargs=fn_kwargs,
                transform_type=TransformType.DEFAULT,
                bench_formatter=bench_formatter,
                is_baseline=True,
            )
            all_results.append(result)

        if "JIT_SCRIPT" in run_modes_to_benchmark:
            # Test JIT script
            result = benchmark_kjt(
                test_name=method_name,
                kjt=kjt,
                test_module=torch.jit.script(kjt_module),
                num_repeat=num_repeat,
                num_warmup=num_warmup,
                fn_kwargs=fn_kwargs,
                transform_type=TransformType.JIT_SCRIPT,
                bench_formatter=bench_formatter,
            )

            all_results.append(result)

        if "VBE" in run_modes_to_benchmark:
            # Test Eager VBE
            vbe_kjt = KeyedJaggedTensor(
                keys=kjt.keys(),
                values=kjt._values,
                lengths=kjt._lengths,
                stride_per_key_per_rank=kjt._stride_per_key_per_rank,
            )
            vbe_fn_kwargs = fn_kwargs.copy()
            if "kjt" in fn_kwargs:
                vbe_fn_kwargs["kjt"] = vbe_kjt

            result = benchmark_kjt(
                test_name=method_name,
                kjt=vbe_kjt,
                test_module=kjt_module,
                num_repeat=num_repeat,
                num_warmup=num_warmup,
                fn_kwargs=vbe_fn_kwargs,
                transform_type=TransformType.VBE,
                is_vb=True,
                bench_formatter=bench_formatter,
            )
            all_results.append(result)

        # PT2 (Eager Inductor)
        if test_pt2 or "AOT_EAGER" in run_modes_to_benchmark:
            vbe_kjt = KeyedJaggedTensor(
                keys=kjt.keys(),
                values=kjt._values,
                lengths=kjt._lengths,
                stride_per_key_per_rank=kjt._stride_per_key_per_rank,
            )
            vbe_fn_kwargs = fn_kwargs.copy()
            if "kjt" in fn_kwargs:
                vbe_fn_kwargs["kjt"] = vbe_kjt
            dynamo_compiled_mod = dynamo_compile(
                method_name,
                kjt_module,
                backend="aot_eager",
                fullgraph=True,
                fn_kwargs=vbe_fn_kwargs,
            )

            result = benchmark_kjt(
                test_name=method_name,
                kjt=vbe_kjt,
                test_module=dynamo_compiled_mod,
                num_repeat=num_repeat,
                num_warmup=num_warmup,
                # simulate VBE, otherwise torch.compile currently fails
                fn_kwargs=vbe_fn_kwargs,
                transform_type=TransformType.PT2_AOT_EAGER,
                is_vb=True,
                bench_formatter=bench_formatter,
            )

            all_results.append(result)
        # Leave a gap between methods
        print("")
    return all_results
