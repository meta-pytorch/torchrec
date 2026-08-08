#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
)
from torchrec.metrics.cpu_offloaded_metric_module import _foreach_clone_dict

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RunOptions(BenchFuncConfig):
    name: str = "zorm_clone_helper"
    world_size: int = 1
    num_profiles: int = 1
    num_benchmarks: int = 7
    profile_dir: str = ""
    device_type: str = "cuda"
    memory_snapshot: bool = False

    warmups: int = 2
    dependency_delay_cycles: int = 245_000_000
    prefill_kernel_count: int = 64
    tensor_count_multiplier: int = 1


def _add_tensors(
    model_out: dict[str, torch.Tensor],
    prefix: str,
    count: int,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    for index in range(count):
        model_out[f"{prefix}_{index}"] = torch.ones(
            shape,
            dtype=torch.float32,
            device=device,
        )


def make_production_like_model_out(
    device: torch.device, tensor_count_multiplier: int
) -> dict[str, torch.Tensor]:
    model_out: dict[str, torch.Tensor] = {}
    for replica in range(tensor_count_multiplier):
        prefix = f"replica_{replica}"
        _add_tensors(model_out, f"{prefix}_column", 864, (4096, 1), device)
        _add_tensors(model_out, f"{prefix}_vector", 62, (4096,), device)
        _add_tensors(model_out, f"{prefix}_scalar", 32, (), device)
        _add_tensors(model_out, f"{prefix}_two_column", 1, (4096, 2), device)
        _add_tensors(model_out, f"{prefix}_five_column", 1, (4096, 5), device)
        _add_tensors(model_out, f"{prefix}_wide", 1, (4096, 512), device)
        model_out[f"{prefix}_non_dense_stride_4"] = torch.ones(
            (4096, 4), dtype=torch.float32, device=device
        )[:, :1]
        model_out[f"{prefix}_non_dense_stride_7"] = torch.ones(
            (4096, 7), dtype=torch.float32, device=device
        )[:, :1]
        model_out[f"{prefix}_double_scalar"] = torch.ones(
            (), dtype=torch.float64, device=device
        )
        model_out[f"{prefix}_int_labels"] = torch.ones(
            (4096,), dtype=torch.int64, device=device
        )
    return model_out


def _prepare_queue_pressure(
    dependency_delay_cycles: int,
    prefill_kernel_count: int,
    dependency_stream: torch.cuda.Stream,
    prefill_tensor: torch.Tensor,
) -> None:
    current_stream = torch.cuda.current_stream()
    dependency_stream.wait_stream(current_stream)
    with torch.cuda.stream(dependency_stream):
        if dependency_delay_cycles > 0:
            # pyrefly: ignore[missing-module-attribute]
            torch.cuda._sleep(dependency_delay_cycles)
    current_stream.wait_stream(dependency_stream)
    for _ in range(prefill_kernel_count):
        prefill_tensor.add_(1)


def _host_wall_time_ms(result: BenchmarkResult) -> torch.Tensor:
    return torch.where(
        result.cpu_utilization > 0,
        result.cpu_elapsed_time / result.cpu_utilization,
        torch.zeros_like(result.cpu_elapsed_time),
    )


def runner(run_option: RunOptions) -> BenchmarkResult:
    if run_option.device_type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if run_option.world_size != 1:
        raise ValueError("This benchmark supports world_size=1")
    if run_option.tensor_count_multiplier < 1:
        raise ValueError("tensor_count_multiplier must be positive")

    device = torch.device(run_option.device_type, 0)
    torch.cuda.set_device(0)
    model_out = make_production_like_model_out(
        device, run_option.tensor_count_multiplier
    )
    dependency_stream = torch.cuda.Stream(device=device)
    prefill_tensor = torch.ones((4096,), dtype=torch.float32, device=device)
    result_ref: Any = None

    def clone_model_out(_bench_inputs: list[dict[str, Any]]) -> None:
        nonlocal result_ref
        _prepare_queue_pressure(
            run_option.dependency_delay_cycles,
            run_option.prefill_kernel_count,
            dependency_stream,
            prefill_tensor,
        )
        result_ref = _foreach_clone_dict(model_out)

    for _ in range(run_option.warmups):
        torch.cuda.synchronize()
        clone_model_out([])
    torch.cuda.synchronize()

    logger.info(
        "Benchmarking %d tensors / %d bytes on %s",
        len(model_out),
        sum(t.numel() * t.element_size() for t in model_out.values()),
        torch.cuda.get_device_name(device),
    )
    result = benchmark_func(
        rank=0,
        func_to_benchmark=clone_model_out,
        bench_inputs=[{}],
        prof_inputs=[{}] * run_option.num_profiles,
        benchmark_func_kwargs={},
        sample_count=0,
        **run_option.benchmark_func_kwargs(),
    )

    host_wall_time_ms = _host_wall_time_ms(result)
    logger.info(
        "Host submission (P50/P90): %.2f / %.2f ms",
        torch.quantile(host_wall_time_ms, 0.5, interpolation="nearest").item(),
        torch.quantile(host_wall_time_ms, 0.9, interpolation="nearest").item(),
    )
    logger.info(result.prettify())
    return result


@cmd_conf
def main(run_option: RunOptions) -> None:
    run_option.set_log_level()
    print(BenchmarkResult.print_table([runner(run_option)]))


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
