#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Compare the padded and offset SparseCore pooled lookup kernels.

Run with: ./run_pod.sh run experimental/torch_tpu/benchmarks/compare_pooled_lookup.py
"""

from __future__ import annotations

import functools
import statistics
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_tpu  # pyre-ignore[21]: Available in the TPU pod environment.
from torch_tpu._internal import sync  # pyre-ignore[21]
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.experimental.torch_tpu.pallas import impl as pallas_impl
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

NUM_ROWS = 1_000_000
EMB_DIM = 64
ITERS = 30
WARMUP = 5
TRIALS = 7
SEED = 100
POOL_ALIGNMENT = 8
MIN_POOL = 16

# (global batch size, average pool factor)
CONFIGS = [
    (512, 16),
    (2048, 1),
    (2048, 8),
    (2048, 32),
    (4096, 8),
    (4096, 32),
    (4096, 64),
]

Lookup = Callable[[], torch.Tensor]


def measure_first_call(lookup: Lookup) -> float:
    """Return synchronized first-call latency in milliseconds."""
    start = time.perf_counter()
    output = lookup()
    sync.synchronize(output, wait=True)
    return (time.perf_counter() - start) * 1e3


def benchmark_trial(lookup: Lookup) -> float:
    start = time.perf_counter()
    for _ in range(ITERS):
        output = lookup()
        sync.synchronize(output, wait=True)
    return (time.perf_counter() - start) / ITERS * 1e3


def benchmark_steady_state(padded: Lookup, offset: Lookup) -> tuple[float, float]:
    """Return median trial-average latency with padded measured before offset."""
    for _ in range(WARMUP):
        for lookup in (padded, offset):
            output = lookup()
            sync.synchronize(output, wait=True)

    padded_trials: list[float] = []
    offset_trials: list[float] = []
    for _ in range(TRIALS):
        padded_trials.append(benchmark_trial(padded))
        offset_trials.append(benchmark_trial(offset))

    return statistics.median(padded_trials), statistics.median(offset_trials)


def prepare_padded_indices(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Convert jagged indices to the padded kernel's fixed-pool format."""
    num_bags = offsets.numel() - 1
    max_length = int(lengths.max().item())
    pool = max(
        ((max_length + POOL_ALIGNMENT - 1) // POOL_ALIGNMENT) * POOL_ALIGNMENT,
        MIN_POOL,
    )

    padded_indices = F.pad(indices, (0, num_bags * pool - indices.numel()))
    columns = torch.arange(pool, dtype=offsets.dtype)
    positions = offsets[:-1, None] + columns[None, :]
    valid = columns[None, :] < lengths[:, None]
    fixed_pool_indices = torch.where(
        valid,
        padded_indices[positions],
        torch.full_like(positions, NUM_ROWS),
    )
    return fixed_pool_indices.flatten().to("tpu"), pool


def padded_lookup(
    indices: torch.Tensor, table: torch.Tensor, pool: int
) -> torch.Tensor:
    return torch.ops.torchrec.embedding_pooled_lookup(indices, table, pool)


def offset_lookup(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    table: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    return torch.ops.torchrec.embedding_pooled_lookup_offset(
        indices, offsets, table, emb_dim
    )


def main() -> None:
    # These imports register the TPU device and both TorchRec TPU operators.
    _ = torch_tpu, pallas_impl

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()

    torch.manual_seed(0)
    weight_cpu = torch.randn(NUM_ROWS, EMB_DIM, dtype=torch.float32)
    weight_tpu = weight_cpu.to("tpu")
    padded_weight_tpu = torch.cat(
        (weight_tpu, weight_tpu.new_zeros((1, EMB_DIM))), dim=0
    )

    table = EmbeddingConfig(
        name="t0",
        embedding_dim=EMB_DIM,
        num_embeddings=NUM_ROWS,
        feature_names=["f0"],
    )

    if rank == 0:
        print(
            f"\nPadded vs offset SparseCore pooled lookup "
            f"[table {NUM_ROWS}x{EMB_DIM}, {TRIALS} trials x {ITERS} iterations]\n"
            f"{'config':<18}{'ids':>10}{'maxlen':>9}{'padded K':>10}"
            f"{'padded rows':>13}{'offset rows':>13}{'padded compile*':>17}"
            f"{'offset compile*':>17}{'padded ms':>11}{'offset ms':>11}{'speedup':>10}",
            flush=True,
        )

    for batch, average_pool in CONFIGS:
        model_input, _ = ModelInput.generate(
            batch_size=batch,
            world_size=1,
            num_float_features=0,
            tables=[table],
            weighted_tables=[],
            pooling_avg=average_pool,
            indices_dtype=torch.int32,
            offsets_dtype=torch.int32,
            lengths_dtype=torch.int32,
            random_seed=SEED,
        )
        kjt = model_input.idlist_features
        assert isinstance(kjt, KeyedJaggedTensor)
        indices = kjt.values()
        offsets = kjt.offsets()
        lengths = kjt.lengths()
        total_ids = indices.numel()
        num_bags = offsets.numel() - 1
        max_length = int(lengths.max().item())

        padded_indices, pool = prepare_padded_indices(indices, offsets, lengths)
        offset_indices = indices.to("tpu")
        offset_offsets = offsets.to("tpu")

        run_padded: Lookup = functools.partial(
            padded_lookup, padded_indices, padded_weight_tpu, pool
        )
        run_offset: Lookup = functools.partial(
            offset_lookup,
            indices=offset_indices,
            offsets=offset_offsets,
            table=weight_tpu,
            emb_dim=EMB_DIM,
        )

        for tensor in (
            padded_indices,
            padded_weight_tpu,
            offset_indices,
            offset_offsets,
            weight_tpu,
        ):
            sync.synchronize(tensor, wait=True)

        padded_first_call = measure_first_call(run_padded)
        offset_first_call = measure_first_call(run_offset)

        padded_time, offset_time = benchmark_steady_state(run_padded, run_offset)
        padded_compile = max(padded_first_call - padded_time, 0.0)
        offset_compile = max(offset_first_call - offset_time, 0.0)

        reference = F.embedding_bag(
            input=indices.to(torch.int64),
            weight=weight_cpu,
            offsets=offsets[:-1].to(torch.int64),
            mode="sum",
        )
        padded_output = run_padded()
        sync.synchronize(padded_output, wait=True)
        padded_output = padded_output.to("cpu")
        offset_output = run_offset()
        sync.synchronize(offset_output, wait=True)
        offset_output = offset_output.to("cpu")

        padded_error = (reference - padded_output).abs().max().item()
        offset_error = (reference - offset_output).abs().max().item()
        assert torch.allclose(
            reference, padded_output, atol=1e-3
        ), f"padded B={batch}: error={padded_error}"
        assert torch.allclose(
            reference, offset_output, atol=1e-3
        ), f"offset B={batch}: error={offset_error}"
        padded_rows = num_bags * pool

        if rank == 0:
            config = f"B={num_bags} avg={average_pool}"
            print(
                f"{config:<18}{total_ids:>10}{max_length:>9}{pool:>10}"
                f"{padded_rows:>13}{total_ids:>13}{padded_compile:>17.1f}"
                f"{offset_compile:>17.1f}{padded_time:>11.3f}{offset_time:>11.3f}"
                f"{padded_time / offset_time:>9.2f}x",
                flush=True,
            )

    if rank == 0:
        print(
            "\n* Compile is estimated as first synchronized call minus the "
            "steady-state median; TPU cache lookup/loading is included."
        )
        print("All comparisons passed against torch embedding_bag.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
