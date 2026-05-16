#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for permute_2D_sparse_data with and without pre-allocated output tensors.

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/sparse/tests:permute_2d_benchmark
    buck2 run @fbcode//mode/opt fbcode//torchrec/sparse/tests:permute_2d_benchmark -- --num_features=170 --batch_size=256 --device_type=cuda
"""

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")

from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
)

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", stream=sys.stdout)
logger.setLevel(logging.DEBUG)


def _generate_sparse_data(
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    device: torch.device,
    has_weight: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Generate sparse data for permute_2D_sparse_data benchmarking."""
    lengths = torch.randint(
        low=0,
        high=2 * mean_pooling_factor,
        size=(num_features, batch_size),
        dtype=torch.int32,
        device=device,
    )
    total = int(lengths.sum().item())
    indices = torch.randint(
        low=0,
        high=int(1e5),
        size=(total,),
        dtype=torch.int32,
        device=device,
    )
    permute = torch.tensor(
        random.sample(range(num_features), k=num_features),
        dtype=torch.int32,
        device=device,
    )
    weights = (
        torch.rand(total, dtype=torch.float32, device=device) if has_weight else None
    )
    return permute, lengths, indices, weights


@dataclass
class RunOptions(BenchFuncConfig):
    """Configuration for permute_2D_sparse_data benchmark."""

    ALL_NAMES: List[str] = field(
        default_factory=lambda: [
            "permute_2d_default",
            "permute_2d_preallocated",
        ],
        repr=False,
    )

    name: str = "all"
    num_features: int = 170
    batch_size: int = 128
    mean_pooling_factor: int = 50
    has_weight: bool = False

    # Override defaults from BenchFuncConfig
    world_size: int = 1
    num_benchmarks: int = 100
    num_profiles: int = 10
    device_type: str = "cuda"
    debug_mode: bool = False
    profile_dir: str = "."
    memory_snapshot: bool = True

    _iter_index: int = field(default=0, repr=False)

    def __iter__(self) -> "RunOptions":
        self._iter_index = 0
        return self

    def __next__(self) -> "RunOptions":
        if self.name != "all":
            if self._iter_index == 0:
                self._iter_index += 1
                return self
            raise StopIteration

        if self._iter_index >= len(self.ALL_NAMES):
            raise StopIteration

        import copy

        name = self.ALL_NAMES[self._iter_index]
        self._iter_index += 1
        new_option = copy.copy(self)
        new_option.name = name
        return new_option


def runner(
    run_option: RunOptions,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    device = torch.device(run_option.device_type)

    permute, lengths, indices, weights = _generate_sparse_data(
        run_option.num_features,
        run_option.batch_size,
        run_option.mean_pooling_factor,
        device,
        run_option.has_weight,
    )

    total = int(lengths.sum().item())
    T = permute.numel()
    B = lengths.size(1)

    use_preallocated = run_option.name == "permute_2d_preallocated"

    # Only pre-allocate output tensors for the preallocated path
    permuted_lengths_out: Optional[torch.Tensor] = None
    permuted_indices_out: Optional[torch.Tensor] = None
    permuted_weights_out: Optional[torch.Tensor] = None
    if use_preallocated:
        permuted_lengths_out = torch.empty(T, B, dtype=lengths.dtype, device=device)
        permuted_indices_out = torch.empty(total, dtype=indices.dtype, device=device)
        permuted_weights_out = (
            torch.empty(total, dtype=torch.float32, device=device)
            if run_option.has_weight
            else None
        )

    # Warmup
    if use_preallocated:
        torch.ops.fbgemm.permute_2D_sparse_data(
            permute,
            lengths,
            indices,
            weights,
            total,
            permuted_lengths_out,
            permuted_indices_out,
            permuted_weights_out,
        )
    else:
        torch.ops.fbgemm.permute_2D_sparse_data(
            permute,
            lengths,
            indices,
            weights,
            total,
        )

    def _func_to_benchmark(
        _bench_inputs: List[Any],
    ) -> None:
        if use_preallocated:
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute,
                lengths,
                indices,
                weights,
                total,
                permuted_lengths_out,
                permuted_indices_out,
                permuted_weights_out,
            )
        else:
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute,
                lengths,
                indices,
                weights,
                total,
            )

    result = benchmark_func(
        rank=0,
        func_to_benchmark=_func_to_benchmark,
        bench_inputs=[{}],
        prof_inputs=[{}] * run_option.num_profiles,
        benchmark_func_kwargs={},
        **run_option.benchmark_func_kwargs(),
    )

    logger.info(result.prettify())
    logger.info("\nMarkdown format:\n%s", result)

    return result


@cmd_conf
def main(
    run_option: RunOptions,
) -> None:
    """Main entry point for the permute_2D_sparse_data benchmark."""
    run_option.set_log_level()

    results: List[BenchmarkResult] = []
    for option in run_option:
        results.append(runner(option))

    print(BenchmarkResult.print_table(results))


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
