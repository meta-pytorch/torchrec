#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and forward-performance sweep for the batched TPU EBC kernel."""

import json
import logging
import math
import os
import statistics
import time
from typing import Any, Tuple

import jax
import torch

# pyre-ignore[21]: Provided by the TPU pod runtime.
import torch_tpu  # noqa: F401

# pyre-ignore[21]: Provided by the TPU pod runtime.
from torch_tpu._internal import sync
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.experimental.torch_tpu.modules.embedding_modules import (
    _bucket_index_count,
    PallasTableBatchedEmbeddingBags,
)
from torchrec.experimental.torch_tpu.pallas import impl as pallas_impl  # noqa: F401
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


if os.environ.get("EXPLAIN_CACHE_MISSES") == "1":
    try:
        jax.config.update("jax_explain_cache_misses", True)
    except Exception as exc:  # pragma: no cover - depends on the installed JAX
        print(f"jax_explain_cache_misses unavailable: {exc}", flush=True)
else:
    logging.getLogger("jax._src.interpreters.partial_eval").setLevel(logging.ERROR)


MIN_WARMUP = 3
MAX_WARMUP = 10
ITERATIONS = 10
BATCH_SIZE = 2048
NUM_ROWS = 1_000_000
EMBEDDING_DIMS: list[int] = [512]
TABLE_COUNTS = [12]
FEATURES_PER_TABLE = 4
POOLING_FACTORS = [1]
BASE_SEED = 0


def _create_inputs(
    num_tables: int, pooling_factor: int, embedding_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    tables = [
        EmbeddingBagConfig(
            name=f"table_{table}",
            embedding_dim=embedding_dim,
            num_embeddings=NUM_ROWS,
            feature_names=[
                f"feature_{table}_{feature}" for feature in range(FEATURES_PER_TABLE)
            ],
        )
        for table in range(num_tables)
    ]
    model_input, _ = ModelInput.generate(
        batch_size=BATCH_SIZE,
        world_size=1,
        num_float_features=0,
        tables=tables,
        weighted_tables=[],
        tables_pooling=[pooling_factor] * num_tables,
        use_offsets=True,
        indices_dtype=torch.int32,
        offsets_dtype=torch.int32,
        lengths_dtype=torch.int32,
        random_seed=seed,
    )
    features = model_input.idlist_features

    assert isinstance(features, KeyedJaggedTensor)
    return features.values(), features.offsets()


def _cpu_reference(
    module: PallasTableBatchedEmbeddingBags,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    num_tables: int,
) -> torch.Tensor:
    outputs = []
    weights = [weight.detach().cpu() for weight in module.split_embedding_weights()]
    for feature, table in enumerate(module.feature_table_map):
        weight = weights[table]
        feature_outputs = []
        for batch in range(BATCH_SIZE):
            bag = feature * BATCH_SIZE + batch
            start = int(offsets[bag])
            end = int(offsets[bag + 1])
            feature_outputs.append(weight[indices[start:end].long()].sum(dim=0))
        outputs.append(torch.stack(feature_outputs))
    assert len(module.split_embedding_weights()) == num_tables
    assert len(outputs) == len(module.feature_table_map)
    return torch.cat(outputs, dim=1)


def _benchmark(
    module: PallasTableBatchedEmbeddingBags,
    dim: int,
    num_tables: int,
    pooling_factor: int,
    seed: int,
) -> dict[str, Any]:
    iteration_timings: list[dict[str, int | float]] = []

    # Pad to the bucket on the HOST, before the transfer.
    requests: list[Tuple[torch.Tensor, torch.Tensor, int]] = []
    for request_index in range(MAX_WARMUP + ITERATIONS):
        indices_cpu, offsets_cpu = _create_inputs(
            num_tables, pooling_factor, dim, seed + request_index
        )
        num_indices = indices_cpu.numel()
        bucket = _bucket_index_count(num_indices)
        if bucket != num_indices:
            indices_cpu = torch.nn.functional.pad(
                indices_cpu, (0, bucket - num_indices)
            )
        indices = indices_cpu.to("tpu")
        offsets = offsets_cpu.to("tpu")
        sync.synchronize([indices, offsets], wait=True)
        requests.append((indices, offsets, num_indices))

    # pyre-ignore[16]
    misses_start: int = torch.tpu._get_cache_misses()
    warmup_misses = [misses_start]
    timed_compiles = 0

    with torch.no_grad():
        warmup_iterations = MAX_WARMUP
        for warmup_iteration in range(MAX_WARMUP):
            indices, offsets, _ = requests[warmup_iteration]

            output = module(indices=indices, offsets=offsets)
            sync.synchronize(output, wait=True)

            # pyre-ignore[16]
            warmup_misses.append(torch.tpu._get_cache_misses())
            if (
                warmup_iteration + 1 >= MIN_WARMUP
                and warmup_misses[-1] == warmup_misses[-2]
            ):
                warmup_iterations = warmup_iteration + 1
                break

        # pyre-ignore[16]
        misses_after_warmup: int = torch.tpu._get_cache_misses()

        for iteration in range(ITERATIONS):
            indices, offsets, num_indices = requests[MAX_WARMUP + iteration]

            # Bracket only the timed region
            # pyre-ignore[16]
            misses_before: int = torch.tpu._get_cache_misses()
            start_ns = time.perf_counter_ns()
            output = module(indices=indices, offsets=offsets)
            sync.synchronize(output, wait=True)
            end_ns = time.perf_counter_ns()
            # pyre-ignore[16]
            iteration_compiles: int = torch.tpu._get_cache_misses() - misses_before
            timed_compiles += iteration_compiles

            iteration_timings.append(
                {
                    "iteration": iteration,
                    "seed": seed + MAX_WARMUP + iteration,
                    "num_indices": num_indices,
                    "padded_num_indices": indices.numel(),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ms": (end_ns - start_ns) / 1.0e6,
                    "compiles": iteration_compiles,
                }
            )

    # pyre-ignore[16]
    misses_end: int = torch.tpu._get_cache_misses()

    times_ms = [float(timing["duration_ms"]) for timing in iteration_timings]
    mean_ms = statistics.fmean(times_ms)
    ordered_times_ms = sorted(times_ms)
    p95_index = min(len(ordered_times_ms) - 1, math.ceil(0.95 * len(times_ms)) - 1)

    # Average the id count over the timed iterations
    mean_num_indices = statistics.fmean(
        float(timing["num_indices"]) for timing in iteration_timings
    )
    weight_bytes = mean_num_indices * dim * torch.float32.itemsize
    input_bytes = mean_num_indices * indices.element_size()
    input_bytes += offsets.numel() * offsets.element_size()
    output_bytes = (
        BATCH_SIZE * len(module.feature_table_map) * dim * torch.float32.itemsize
    )

    effective_gbps = (
        (weight_bytes + input_bytes + output_bytes)
        / (statistics.median(times_ms) / 1.0e3)
        / 1.0e9
    )

    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "p95_ms": ordered_times_ms[p95_index],
        "min_ms": ordered_times_ms[0],
        "max_ms": ordered_times_ms[-1],
        "stddev_ms": statistics.pstdev(times_ms),
        "effective_gbps": effective_gbps,
        "warmup_iterations": warmup_iterations,
        "warmup_converged": warmup_iterations < MAX_WARMUP,
        "warmup_compiles": misses_after_warmup - misses_start,
        "timed_compiles": timed_compiles,
        "timed_loop_compiles": misses_end - misses_after_warmup,
        "mean_num_indices": mean_num_indices,
        "warmup_cache_misses": warmup_misses,
        "iterations": iteration_timings,
    }


def _run_case(
    dim: int,
    num_tables: int,
    pooling_factor: int,
    seed: int,
) -> dict[str, Any]:
    feature_table_map = [
        table for table in range(num_tables) for _ in range(FEATURES_PER_TABLE)
    ]
    module = PallasTableBatchedEmbeddingBags(
        embedding_specs=[(NUM_ROWS, dim)] * num_tables,
        feature_table_map=feature_table_map,
    )
    with torch.no_grad():
        for table, weight in enumerate(module.split_embedding_weights()):
            row_values = torch.arange(weight.shape[0], device=weight.device)
            weight.copy_(
                (row_values.remainder(97) + table * 100).unsqueeze(1).expand_as(weight)
            )

    indices_cpu, offsets_cpu = _create_inputs(num_tables, pooling_factor, dim, seed)
    expected = _cpu_reference(module, indices_cpu, offsets_cpu, num_tables)
    indices = indices_cpu.to("tpu")
    offsets = offsets_cpu.to("tpu")

    with torch.no_grad():
        actual = module(indices=indices, offsets=offsets)
        sync.synchronize(actual, wait=True)
        actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu, expected, atol=1e-4, rtol=1e-4)

    return _benchmark(
        module,
        dim,
        num_tables,
        pooling_factor,
        seed + 1,
    )


def _write_results(
    results_path: str,
    rank: int,
    results: list[dict[str, Any]],
) -> None:
    temporary_path = f"{results_path}.tmp"
    with open(temporary_path, "w") as results_file:
        json.dump(
            {
                "base_seed": BASE_SEED,
                "rank": rank,
                "min_warmup": MIN_WARMUP,
                "max_warmup": MAX_WARMUP,
                "iterations": ITERATIONS,
                "results": results,
            },
            results_file,
            indent=2,
        )
    os.replace(temporary_path, results_path)


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    rank_seed = BASE_SEED + rank * 1_000_000
    torch.manual_seed(rank_seed)

    results_dir = os.environ.get(
        "BENCHMARK_RESULTS_DIR",
        "/workspace/traces/batched_tpu_ebc",
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"benchmark_results_rank_{rank}.json")

    results: list[dict[str, Any]] = []
    has_printed = False
    case_index = 0
    seeds_per_case = MAX_WARMUP + ITERATIONS + 1

    for dim in EMBEDDING_DIMS:
        for num_tables in TABLE_COUNTS:
            for pooling_factor in POOLING_FACTORS:
                case_seed = rank_seed + case_index * seeds_per_case
                stats = _run_case(dim, num_tables, pooling_factor, case_seed)
                result = {
                    "dim": dim,
                    "num_tables": num_tables,
                    "features_per_table": FEATURES_PER_TABLE,
                    "pooling_factor": pooling_factor,
                    "case_seed": case_seed,
                    **stats,
                }
                results.append(result)
                _write_results(results_path, rank, results)
                case_index += 1

                if not has_printed and rank == 0:
                    print(
                        f"{'dim':>5} {'tables':>7} {'feat/table':>10} {'pool':>6} "
                        f"{'mean_ms':>10} {'median_ms':>10} {'p95_ms':>10} "
                        f"{'min_ms':>10} {'max_ms':>10} {'std_ms':>10} "
                        f"{'warm_it':>8} {'warm_cmp':>9} {'timed_cmp':>10} "
                        f"{'effective_GB/s':>16}",
                        flush=True,
                    )
                    has_printed = True
                if rank == 0:
                    print(
                        f"{dim:>5} {num_tables:>7} {FEATURES_PER_TABLE:>10} "
                        f"{pooling_factor:>6} {stats['mean_ms']:>10.3f} "
                        f"{stats['median_ms']:>10.3f} {stats['p95_ms']:>10.3f} "
                        f"{stats['min_ms']:>10.3f} {stats['max_ms']:>10.3f} "
                        f"{stats['stddev_ms']:>10.3f} "
                        f"{stats['warmup_iterations']:>8} "
                        f"{stats['warmup_compiles']:>9} "
                        f"{stats['timed_compiles']:>10} "
                        f"{stats['effective_gbps']:>16.2f}",
                        flush=True,
                    )

    if rank == 0:
        print(f"benchmark results: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
