#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Performance benchmark for sharded relay allreduce on MI350X.

Measures and compares four approaches:

  A) COALESCED — allreduce_coalesced on 2-rank sub-group PGs (legacy baseline)
  B) FUSED     — 1 call with flat-concat buffers + per-group passthrough helpers
                 (phase-sync kernel: all groups in lockstep, passthrough-at-helper)
  C) KERNEL    — 1 direct call, one large tensor per group (kernel BW validation)
  NCCL         — 4 parallel 2-rank dist.all_reduce calls (pre-sharded-relay baseline)

Memory model (phase-sync kernel, passthrough-at-helper, batched forward):
  Each rank holds:
    - 1 active flat buffer (= per-group total for its active group)
    - 3 helper buffers (= nActiveRanks × chunkSize each, passthrough minimum)
    - 1 relay scratch (= numHelpers × chunkSize, batched recv from all helpers)
    - 1 direct-exchange scratch (= 1 × directChunkSize)
  Each helper group has its own buffer (no aliasing) because phase-sync
  processes all groups simultaneously.

  | Approach                          | Active | Helper          | Scratch         | Total  |
  |:----------------------------------|-------:|----------------:|----------------:|-------:|
  | Pre-OOM-fix (phase-sync, reduce)  | 24 GiB | 3×24 = 72 GiB  | ~3.2 GiB        | ~99 GiB|
  | **Passthrough (batched forward)** | 24 GiB | 3×6.8= 20.6 GiB| ~24 GiB         | ~69 GiB|

BM-FM production numbers (from aps-bm_fm_amd_srinathb_20260420_200640-ea51247ebd):
  - 64 trainers (8 nodes × 8 GPUs per MI350X node)
  - 2d_weight_sync (fp16) per_group_total_counts:
      [12_002_982_488, 12_245_126_152, 12_014_370_640, 12_057_805_952]
      ≈ 22.4 GiB per group (fp16)
  - 2d_optimizer_sync (fp32) per_group_total_counts:
      [479_250_475, 553_440_942, 634_386_550, 560_128_334]
      ≈ 1.8–2.4 GiB per group (fp32)

The production per-group totals above are capped to a 1 GiB-per-group default so
the benchmark fits alongside other work on a shared GPU host and in CI.  Run with
no extra flags:
    buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //torchrec/distributed/tests:bench_sharded_relay_perf

Production scale (uncapped, needs an otherwise-idle host):
    BENCH_TABLE_SIZE=12002982488 BENCH_KERNEL_SIZE_GB=22.4 \\
        buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //torchrec/distributed/tests:bench_sharded_relay_perf

Optimizer-sync scale (fp32):
    BENCH_DTYPE=fp32 buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //torchrec/distributed/tests:bench_sharded_relay_perf

Small-scale smoke run (101 tables × 1M elements ≈ 100M per group):
    BENCH_NUM_TABLES=101 BENCH_TABLE_SIZE=1048576 \\
        buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //torchrec/distributed/tests:bench_sharded_relay_perf

Environment variables (all optional):
    BENCH_DTYPE          bf16|fp16|fp32   (default: fp16)
    BENCH_NUM_TABLES     int              (default: 1)
    BENCH_TABLE_SIZE     int              (default: capped production total, <=1 GiB/group)
    BENCH_KERNEL_SIZE_GB float            (default: capped production total, <=1 GiB/group)
    BENCH_WARMUP_ITERS   int              (default: 10)
    BENCH_BENCH_ITERS    int              (default: 20)
    BENCH_LOG_SIZES      1                (print sizes and exit; for calibration)

The benchmark automatically sweeps BOTH 2-active and 4-active sharded relay
groups and prints a full report for each. The 4-active sweep covers all four
collectives: allreduce, reduce-scatter, all-to-all, and all-gather.

Message-size sweep (all collectives, 2- & 4-rank, bf16):
    A separate test method, test_collectives_msg_size_sweep, sweeps a fixed list
    of message sizes (4 KB .. 1 GB) for every collective (allreduce,
    reduce-scatter, all-to-all, all-gather) at both 2 and 4 active ranks, and
    reports the sharded-relay speedup over NCCL in the FUSED scenario:
    (NUM_GPUS // A) concurrent A-rank groups in one multi-group relay kernel vs
    the matching NCCL baseline run in parallel on each rank's A-rank sub-group.
    It prints one table per (collective, active-rank-count). The swept size is the
    per-active-rank input tensor byte size, so sizes stay comparable across
    collectives. The single-group scenario is benchmarked separately (see
    test_parallel_collectives_msg_size_sweep). Run it selectively so the production
    test above does not also run:
        buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
            //torchrec/distributed/tests:bench_sharded_relay_perf -- \\
            torchrec.distributed.tests.bench_sharded_relay_perf.BenchShardedRelayPerfTest.test_collectives_msg_size_sweep

Parallel separate-job sweep (all collectives, 2- & 4-rank, bf16):
    A separate test method, test_parallel_collectives_msg_size_sweep, sweeps the
    same fixed message sizes for every collective (allreduce, reduce-scatter,
    all-to-all, all-gather) at both 2 and 4 active ranks. It models N = NUM_GPUS
    // A independent workloads the way production runs them: as N SEPARATE
    co-resident jobs on one 8-GPU node, each owning its OWN full 8-rank
    communicator and issuing exactly ONE single-group A-rank sharded relay call
    (its own process, CUDA context, and GPU_MAX_HW_QUEUES budget). For each A it
    spawns N * NUM_GPUS processes (process p → job = p // NUM_GPUS,
    rank_in_job = p % NUM_GPUS on cuda:rank_in_job), so the N jobs are
    co-resident (N processes per GPU); a gloo PG across all processes overlaps
    the jobs and the reported time is the max across jobs. This is the
    production-faithful counterpart to the FUSED scenario: because each workload
    owns a separate communicator the multi-group fused kernel cannot be used, so
    it measures single-group A-rank relay performance under real inter-process
    XGMI contention. The NCCL baseline is N disjoint A-rank NCCL collectives run
    as separate processes (one per GPU), overlapped, max across jobs. It prints
    one table per (collective, active-rank-count). Run it selectively:
        buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
            //torchrec/distributed/tests:bench_sharded_relay_perf -- \\
            torchrec.distributed.tests.bench_sharded_relay_perf.BenchShardedRelayPerfTest.test_parallel_collectives_msg_size_sweep
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import unittest
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from caffe2.torch.distributed.fb.sharded_relay_process_group import (  # type: ignore[import]
        FusedShardedRelayMultiGroup,
    )

    FUSED_AVAILABLE: bool = True
except ImportError:
    FusedShardedRelayMultiGroup = None  # type: ignore[misc, assignment]
    FUSED_AVAILABLE = False

try:
    from torchcomms import new_comm as _torchcomms_new_comm  # type: ignore[import]

    RCCLX_AVAILABLE: bool = True
except ImportError:
    _torchcomms_new_comm = None  # type: ignore[misc, assignment]
    RCCLX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _find_free_port() -> int:
    """Return a currently-free localhost TCP port.

    Chosen at runtime (in the parent process, before mp.spawn) and passed to the
    workers so rank 0's TCPStore never collides with a hardcoded port left in
    TIME_WAIT by a prior run or by another phase of the same test.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _want(name: str) -> bool:
    """BENCH_ONLY optionally restricts the run to a single collective
    (allreduce|reduce_scatter|all_to_all|all_gather) so tuning one 4-active
    collective doesn't pay for the other three. Default "all" runs everything.
    """
    bo = os.environ.get("BENCH_ONLY", "all")
    return bo == "all" or bo == name


def _get_dtype() -> torch.dtype:
    name = _env_str("BENCH_DTYPE", "fp16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Production numbers (defaults)
# ---------------------------------------------------------------------------

# fp16 weight sync — one total per sparse group (local ranks 0-1, 2-3, 4-5, 6-7)
_PROD_TOTALS_FP16: list[int] = [
    12_002_982_488,  # group 0
    12_245_126_152,  # group 1
    12_014_370_640,  # group 2
    12_057_805_952,  # group 3
]

# fp32 optimizer sync — one total per sparse group
_PROD_TOTALS_FP32: list[int] = [
    479_250_475,  # group 0
    553_440_942,  # group 1
    634_386_550,  # group 2
    560_128_334,  # group 3
]

# The production totals above are ~22 GiB per group, which needs ~69 GiB per rank
# once helper and scratch buffers are included and OOMs whenever the host is
# shared (or in CI). Cap the per-group default and use BENCH_TABLE_SIZE /
# BENCH_KERNEL_SIZE_GB to measure at production scale on an idle host.
_DEFAULT_MAX_BYTES_PER_GROUP: int = 1024**3  # 1 GiB


def _default_totals(dtype: torch.dtype) -> list[int]:
    totals = _PROD_TOTALS_FP16 if dtype != torch.float32 else _PROD_TOTALS_FP32
    cap = _DEFAULT_MAX_BYTES_PER_GROUP // dtype.itemsize
    return [min(total, cap) for total in totals]


# ---------------------------------------------------------------------------
# Message-size sweep (test_allreduce_msg_size_sweep)
# ---------------------------------------------------------------------------

_KIB: int = 1024
_MIB: int = 1024 * 1024

# (human label, bytes) for each swept allreduce message size. Each value is the
# byte size of the input tensor a single active rank contributes to the
# allreduce; element count = bytes // dtype.itemsize.
_MSG_SWEEP_SIZES: list[tuple[str, int]] = [
    ("4 KB", 4 * _KIB),
    ("9 KB", 9 * _KIB),
    ("18 KB", 18 * _KIB),
    ("36 KB", 36 * _KIB),
    ("72 KB", 72 * _KIB),
    ("144 KB", 144 * _KIB),
    ("288 KB", 288 * _KIB),
    ("576 KB", 576 * _KIB),
    ("4.5 MB", int(4.5 * _MIB)),
    ("9 MB", 9 * _MIB),
    ("13.5 MB", int(13.5 * _MIB)),
    ("27 MB", 27 * _MIB),
    ("31.5 MB", int(31.5 * _MIB)),
    ("36 MB", 36 * _MIB),
    ("63 MB", 63 * _MIB),
    ("67.5 MB", int(67.5 * _MIB)),
    ("72 MB", 72 * _MIB),
    ("135 MB", 135 * _MIB),
    ("144 MB", 144 * _MIB),
    ("256 MB", 256 * _MIB),
    ("512 MB", 512 * _MIB),
    ("1 GB", 1024 * _MIB),
]

# Active-rank counts and collectives swept by test_collectives_msg_size_sweep.
# The sweep prints one table per (collective, active-rank-count) = 8 tables.
_MSG_SWEEP_ACTIVE_RANKS: tuple[int, ...] = (2, 4)
_MSG_SWEEP_COLLECTIVES: tuple[str, ...] = (
    "allreduce",
    "reduce_scatter",
    "all_to_all",
    "all_gather",
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _setup_rcclx_comm(
    local_rank: int, local_size: int, node_idx: int, store: Any
) -> Any | None:
    """Create a single 8-rank intra-node RCCLX communicator using the provided store."""
    if not RCCLX_AVAILABLE or _torchcomms_new_comm is None:
        return None
    device = torch.device(f"cuda:{local_rank}")

    orig_rank = os.environ.get("TORCHCOMM_RANK")
    orig_size = os.environ.get("TORCHCOMM_SIZE")
    try:
        os.environ["TORCHCOMM_RANK"] = str(local_rank)
        os.environ["TORCHCOMM_SIZE"] = str(local_size)
        group_store = dist.PrefixStore(f"bench_rcclx_node{node_idx}", store)
        comm = _torchcomms_new_comm(
            backend="rcclx",
            device=device,
            name=f"bench_node{node_idx}",
            store=group_store,
        )
        return comm
    finally:
        if orig_rank is None:
            os.environ.pop("TORCHCOMM_RANK", None)
        else:
            os.environ["TORCHCOMM_RANK"] = orig_rank
        if orig_size is None:
            os.environ.pop("TORCHCOMM_SIZE", None)
        else:
            os.environ["TORCHCOMM_SIZE"] = orig_size


def _make_fused(
    rcclx_comm: Any,
    local_rank: int,
    local_size: int,
    sharding_group_size: int = 2,
) -> Any | None:
    if FusedShardedRelayMultiGroup is None or rcclx_comm is None:
        return None
    num_sparse_groups = local_size // sharding_group_size
    all_active_ranks = [
        list(range(g * sharding_group_size, (g + 1) * sharding_group_size))
        for g in range(num_sparse_groups)
    ]
    return FusedShardedRelayMultiGroup(
        rcclx_comm=rcclx_comm,
        world_size=local_size,
        rank=local_rank,
        all_active_ranks=all_active_ranks,
    )


from torchrec.distributed.sharded_relay_utils import _passthrough_helper_size


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Benchmark B: fused flat approach (phase-sync kernel, passthrough helper)
# ---------------------------------------------------------------------------


def bench_fused_flat(
    fused: Any,
    my_tensors: list[torch.Tensor],
    num_sparse_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
    local_size: int,
    intra_pg: dist.ProcessGroup | None,
    sparse_group_size: int,
    flat_bufs: list[torch.Tensor],
    meta_cache: dict[str, list[int]],
) -> None:
    """ONE fused call with all tensors concatenated per group.

    Uses per-group passthrough-sized helper buffers (nActiveRanks × chunkSize).
    Each helper group has its own buffer — no aliasing — because the
    phase-sync kernel processes all groups simultaneously.
    """
    device = my_tensors[0].device
    dtype = my_tensors[0].dtype
    my_total = sum(t.numel() for t in my_tensors)

    # Pack: single fused CUDA kernel via torch.cat(out=) into pre-allocated buffer.
    active_flat = flat_bufs[my_sparse_group]
    if active_flat.numel() < my_total:
        flat_bufs[my_sparse_group] = torch.empty(my_total, dtype=dtype, device=device)
        active_flat = flat_bufs[my_sparse_group]
    active_flat = active_flat.narrow(0, 0, my_total)
    torch.cat([t.flatten() for t in my_tensors], out=active_flat)

    meta_key = "bench" + str(dtype)
    if meta_key not in meta_cache:
        if intra_pg is not None:
            # Use all_gather to learn per-group totals (heterogeneous groups).
            count_tensor = torch.tensor([my_total], dtype=torch.int64, device=device)
            all_counts = [
                torch.zeros(1, dtype=torch.int64, device=device)
                for _ in range(local_size)
            ]
            dist.all_gather(all_counts, count_tensor, group=intra_pg)
            meta_cache[meta_key] = [
                int(all_counts[g * sparse_group_size].item())
                for g in range(num_sparse_groups)
            ]
        else:
            # Bench controls table count: all groups have the same total.
            meta_cache[meta_key] = [my_total] * num_sparse_groups
    per_group_totals = meta_cache[meta_key]

    # Compute per-group passthrough helper sizes.
    num_chunks = (local_size - sparse_group_size) + 1

    group_tensors: list[torch.Tensor] = []
    group_sizes: list[int] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            group_tensors.append(active_flat)
            group_sizes.append(my_total)
        else:
            total_g = per_group_totals[g]
            if sparse_group_size > 2:
                # Flat A>2 allreduce: helper holds A source chunks + 1 reduced
                # chunk per offload slice = (A+1)*oChunk <= 1.25*total_g for any
                # offload fraction f<=1; 2*total_g covers it with margin.
                helper_size_g = 2 * total_g
            else:
                helper_size_g = _passthrough_helper_size(
                    total_g, sparse_group_size, num_chunks
                )
            # Ensure the per-group helper buffer in flat_bufs is large enough.
            if flat_bufs[g].numel() < helper_size_g:
                flat_bufs[g] = torch.empty(helper_size_g, dtype=dtype, device=device)
            helper_buf = flat_bufs[g]
            group_tensors.append(
                helper_buf
                if helper_buf.numel() == helper_size_g
                else helper_buf.narrow(0, 0, helper_size_g)
            )
            group_sizes.append(total_g)  # full count goes to the kernel

    fused.allreduce_multi_group(
        tensors=group_tensors,
        num_groups=num_sparse_groups,
        per_group_sizes=group_sizes,
        all_active_ranks=all_active_ranks,
        op=dist.ReduceOp.AVG,
        skip_validation=True,
    )

    # Unpack: single batched operation via _foreach_copy_.
    slices = active_flat.split([t.numel() for t in my_tensors])
    torch._foreach_copy_(
        my_tensors,
        [s.view(t.shape) for s, t in zip(slices, my_tensors)],
    )


# ---------------------------------------------------------------------------
# Benchmark C: kernel-level direct call
# ---------------------------------------------------------------------------


def bench_kernel_direct(
    fused: Any,
    tensor: torch.Tensor,
    num_sparse_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
    scratch_tensors: list[torch.Tensor],
    per_group_declared_sizes: list[int] | None = None,
) -> None:
    """Direct kernel call with one large tensor per group — kernel BW validation.

    per_group_declared_sizes: the declared element count for each group passed
    to allreduce_multi_group().  All ranks MUST agree on these values so that
    the RCCLX kernel computes the same chunkSize on every rank.  When None,
    falls back to tensor.numel() for all groups (safe only when all groups have
    the same active tensor size, e.g. the tiny warmup).
    """
    group_tensors: list[torch.Tensor] = []
    group_sizes: list[int] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            group_tensors.append(tensor)
            group_sizes.append(tensor.numel())
        else:
            group_tensors.append(scratch_tensors[g])
            # Use the caller-supplied declared size for this helper group so
            # that every rank computes the same chunkSize.  If not provided,
            # fall back to this rank's tensor size (only correct when all
            # groups share the same element count).
            declared = (
                per_group_declared_sizes[g]
                if per_group_declared_sizes is not None
                else tensor.numel()
            )
            group_sizes.append(declared)

    fused.allreduce_multi_group(
        tensors=group_tensors,
        num_groups=num_sparse_groups,
        per_group_sizes=group_sizes,
        all_active_ranks=all_active_ranks,
        op=dist.ReduceOp.AVG,
        skip_validation=True,
    )


# ---------------------------------------------------------------------------
# Benchmark D: fused reduce-scatter (sharded relay)
# ---------------------------------------------------------------------------


def bench_reduce_scatter_flat(
    fused: Any,
    input_flat: torch.Tensor,
    output_flat: torch.Tensor,
    num_sparse_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
    helper_bufs: list[torch.Tensor],
    per_group_recv_counts: list[int],
) -> None:
    """ONE fused reduce-scatter call across all sparse groups.

    Each group contributes ONE contiguous tensor: the active group passes the
    single contiguous input_flat/output_flat; each helper group passes its
    single passthrough scratch buffer.
    """
    input_group_tensors: list[torch.Tensor] = []
    output_group_tensors: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            input_group_tensors.append(input_flat)
            output_group_tensors.append(output_flat)
        else:
            input_group_tensors.append(helper_bufs[g])
            output_group_tensors.append(helper_bufs[g])

    fused.reduce_scatter_multi_group(
        input_tensors=input_group_tensors,
        output_tensors=output_group_tensors,
        num_groups=num_sparse_groups,
        per_group_recv_counts=per_group_recv_counts,
        all_active_ranks=all_active_ranks,
        op=dist.ReduceOp.AVG,
        skip_validation=True,
    )


# ---------------------------------------------------------------------------
# Benchmark E: fused all-to-all (sharded relay)
# ---------------------------------------------------------------------------


def bench_all_to_all_flat(
    fused: Any,
    input_flat: torch.Tensor,
    output_flat: torch.Tensor,
    num_sparse_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
    helper_bufs: list[torch.Tensor],
    per_group_segment_counts: list[int],
) -> None:
    """ONE fused all-to-all call across all sparse groups.

    Each group contributes ONE contiguous tensor: the active group passes the
    single contiguous input_flat/output_flat; each helper group passes its
    single passthrough scratch buffer.
    """
    input_group_tensors: list[torch.Tensor] = []
    output_group_tensors: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            input_group_tensors.append(input_flat)
            output_group_tensors.append(output_flat)
        else:
            input_group_tensors.append(helper_bufs[g])
            output_group_tensors.append(helper_bufs[g])

    fused.all_to_all_multi_group(
        input_tensors=input_group_tensors,
        output_tensors=output_group_tensors,
        num_groups=num_sparse_groups,
        per_group_segment_counts=per_group_segment_counts,
        all_active_ranks=all_active_ranks,
        skip_validation=True,
    )


# ---------------------------------------------------------------------------
# Benchmark F: fused all-gather (sharded relay)
# ---------------------------------------------------------------------------


def bench_all_gather_flat(
    fused: Any,
    input_flat: torch.Tensor,
    output_flat: torch.Tensor,
    num_sparse_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
    helper_bufs: list[torch.Tensor],
    per_group_send_counts: list[int],
) -> None:
    """ONE fused all-gather call across all sparse groups.

    Each group contributes ONE contiguous tensor: the active group passes the
    single contiguous input_flat/output_flat; each helper group passes its
    single passthrough scratch buffer.
    """
    input_group_tensors: list[torch.Tensor] = []
    output_group_tensors: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            input_group_tensors.append(input_flat)
            output_group_tensors.append(output_flat)
        else:
            input_group_tensors.append(helper_bufs[g])
            output_group_tensors.append(helper_bufs[g])

    fused.all_gather_multi_group(
        input_tensors=input_group_tensors,
        output_tensors=output_group_tensors,
        num_groups=num_sparse_groups,
        per_group_send_counts=per_group_send_counts,
        all_active_ranks=all_active_ranks,
        skip_validation=True,
    )


# ---------------------------------------------------------------------------
# NCCL baseline: 4 parallel 2-rank dist.all_reduce calls
# ---------------------------------------------------------------------------


def bench_nccl_baseline(
    tensor: torch.Tensor,
    my_pg: dist.ProcessGroup,
) -> None:
    """Standard NCCL 2-rank allreduce — pre-sharded-relay baseline."""
    dist.all_reduce(tensor, group=my_pg, op=dist.ReduceOp.SUM)


# ---------------------------------------------------------------------------
# Timer helper
# ---------------------------------------------------------------------------


def _measure_ms(
    fn: Any,
    warmup: int,
    iters: int,
    device_barrier_group: Any = None,
) -> tuple[float, float]:
    """Time fn() over 'iters' runs after 'warmup' warmups.

    Returns (best_ms, std_ms). The BEST (min) is the headline metric: collective
    microbenchmarks are noise-dominated, so the mean is skewed by stray slow
    iterations (OS jitter, DVFS, cross-rank skew). The min is the steady-state,
    reproducible time — the same convention nccl-tests / rccl-tests use. std is
    returned only to show the spread.

    Methodology for a stable, fair measurement:
    - A full-world barrier precedes each timed iteration so all ranks start the
      collective aligned (rank skew otherwise inflates the measured time, and
      differently every run). The barrier is OUTSIDE the timed region.
    - device_barrier_group (optional): an additional barrier on this group is
      issued after the default barrier. The parallel sweep passes a per-job NCCL
      group so each job's participants are aligned ON-DEVICE (removing P2P
      partner skew that inflates latency-bound relay times), while the default
      barrier (a global CPU/gloo group there) deterministically overlaps all
      co-resident jobs each iteration. Both baselines pass the same pair, so the
      comparison is apples-to-apples.
    - Timing uses on-device CUDA/hipEvents, so host-side scheduling jitter is
      excluded.
    Note: GPU clocks should also be locked (e.g. rocm-smi) to remove DVFS
    variance — that is operational, not done here.
    """
    have_dist = dist.is_available() and dist.is_initialized()

    def _align() -> None:
        if not have_dist:
            return
        dist.barrier()
        if device_barrier_group is not None:
            dist.barrier(group=device_barrier_group)

    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    _align()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        # Align all ranks' starts; excluded from the timed region below.
        _align()
        torch.cuda.synchronize()
        start_ev.record()
        fn()
        end_ev.record()
        end_ev.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    best = min(times)
    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return best, std


# ---------------------------------------------------------------------------
# Per-rank benchmark worker — mimics _benchmark_worker from the old
# test_sharded_relay_2d_integration.py pattern:
#   - receives an explicit TCPStore instead of using _get_default_store()
#   - sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT itself
# ---------------------------------------------------------------------------

NUM_GPUS: int = 8
_NCCL_PORT: int = 29500
_TCPSTORE_PORT: int = 29502


def _bench_a_worker(
    rank: int,
    world_size: int,
    results_dict: Any,
    sharding_group_size: int = 2,
    port_offset: int = 0,
) -> None:
    """Worker for bench A: serial all_reduce on isolated sub-group PGs.

    Runs in its own mp.spawn with its own dist.init_process_group/destroy cycle
    (on different ports) so that NCCL sub-group communicators are fully isolated
    from bench B/C's RCCLX communicators. sharding_group_size selects the
    active-ranks-per-group (2 or 4); port_offset isolates ports across the
    2-rank and 4-rank sweep iterations.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT + 10 + port_offset)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT + 10 + port_offset,
        world_size=world_size,
        is_master=is_master,
        wait_for_workers=True,
    )

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        store=store,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype = _get_dtype()
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 20))

    sparse_group_size = sharding_group_size
    num_sparse_groups = world_size // sparse_group_size
    my_sparse_group = local_rank // sparse_group_size
    all_active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]

    prod_totals = _default_totals(dtype)

    num_tables = _env_int("BENCH_NUM_TABLES", 1)
    table_sizes_all = [
        _env_int("BENCH_TABLE_SIZE", prod_totals[g]) for g in range(num_sparse_groups)
    ]
    all_tensors_sizes = [
        [max(1024, ts // (1 + (i % 5))) for i in range(num_tables)]
        for ts in table_sizes_all
    ]
    sizes = all_tensors_sizes[my_sparse_group]
    my_tensors = [torch.randn(sz, dtype=dtype, device=device) for sz in sizes]

    # Create one sub-group PG per pair (0/1, 2/3, 4/5, 6/7).
    sub_pgs = [
        dist.new_group(ranks=all_active_ranks[g]) for g in range(num_sparse_groups)
    ]
    my_pg = sub_pgs[my_sparse_group]

    # CPU-only barrier via TCPStore (no GPU/NCCL).
    _barrier_cnt: list[int] = [0]

    def _store_barrier() -> None:
        _barrier_cnt[0] += 1
        tag = f"_bench_a_barrier_{_barrier_cnt[0]}"
        store.set(f"{tag}_{rank}", "1")
        for r in range(world_size):
            store.wait([f"{tag}_{r}"])

    # Warmup the sub-group communicator.
    _warmup_t = torch.ones(1, dtype=dtype, device=device)
    dist.all_reduce(_warmup_t, op=dist.ReduceOp.SUM, group=my_pg)
    torch.cuda.synchronize()
    del _warmup_t
    _store_barrier()

    # Match the legacy weight-sync path: one allreduce_coalesced call with
    # all tensors batched. Use SUM + manual divide instead of AVG because
    # RCCL on MI350X lacks the compiled kernel for AVG + fp16 (ncclDevFuncId
    # not found → GPU memory access fault).
    _ar_opts = dist.AllreduceCoalescedOptions()
    _ar_opts.reduceOp = dist.ReduceOp.SUM

    def run_coalesced() -> None:
        for t in my_tensors:
            t.fill_(1.0)
        my_pg.allreduce_coalesced(my_tensors, opts=_ar_opts).wait()
        for t in my_tensors:
            t.div_(sparse_group_size)

    mean_serial, std_serial = (
        _measure_ms(run_coalesced, warmup, bench_iters)
        if _want("allreduce")
        else (0.0, 0.0)
    )
    _store_barrier()

    if rank == 0:
        results_dict["A"] = (mean_serial, std_serial)

    # NCCL reduce-scatter baseline runs for all active-rank counts (paired with
    # the now-4-active sharded relay reduce-scatter bench). recv_count =
    # prod_total // sparse_group_size so the input (A x recv_count) matches the
    # sharded-relay reduce-scatter benchmark's active footprint. Free the
    # allreduce tensors first to reclaim HBM.
    my_tensors = []
    torch.cuda.empty_cache()
    rs_recv = prod_totals[my_sparse_group] // sparse_group_size
    rs_in = torch.ones(sparse_group_size * rs_recv, dtype=dtype, device=device)
    rs_out = torch.empty(rs_recv, dtype=dtype, device=device)

    # Use SUM + manual divide (RCCL on MI350X lacks the AVG+fp16 kernel).
    def run_rs_baseline() -> None:
        rs_in.fill_(1.0)
        dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM, group=my_pg)
        rs_out.div_(sparse_group_size)

    mean_rs_base, std_rs_base = (
        _measure_ms(run_rs_baseline, warmup, bench_iters)
        if _want("reduce_scatter")
        else (0.0, 0.0)
    )
    _store_barrier()

    if rank == 0:
        results_dict["A_rs"] = (mean_rs_base, std_rs_base)

    # NCCL all-to-all baseline runs for all active-rank counts (paired with the
    # now-4-active sharded relay all-to-all bench). Free the reduce-scatter
    # baseline tensors first. segment_count = prod_total // (2*sparse_group_size)
    # so the out-of-place in/out buffers (each A x segment_count = prod_total/2)
    # match the sharded-relay all-to-all footprint regardless of A.
    rs_in = torch.empty(0, dtype=dtype, device=device)
    rs_out = torch.empty(0, dtype=dtype, device=device)
    torch.cuda.empty_cache()
    a2a_seg = prod_totals[my_sparse_group] // (2 * sparse_group_size)
    a2a_in = torch.ones(sparse_group_size * a2a_seg, dtype=dtype, device=device)
    a2a_out = torch.empty(sparse_group_size * a2a_seg, dtype=dtype, device=device)

    def run_a2a_baseline() -> None:
        a2a_in.fill_(1.0)
        dist.all_to_all_single(a2a_out, a2a_in, group=my_pg)

    mean_a2a_base, std_a2a_base = (
        _measure_ms(run_a2a_baseline, warmup, bench_iters)
        if _want("all_to_all")
        else (0.0, 0.0)
    )
    _store_barrier()

    if rank == 0:
        results_dict["A_a2a"] = (mean_a2a_base, std_a2a_base)

    # NCCL all-gather baseline runs for all active-rank counts (paired with the
    # now-4-active sharded relay all-gather bench). Free the all-to-all baseline
    # tensors first. send_count = prod_total // (2 * sparse_group_size) so the
    # output (A x send_count = prod_total/2) matches the sharded-relay all-gather
    # footprint regardless of A.
    a2a_in = torch.empty(0, dtype=dtype, device=device)
    a2a_out = torch.empty(0, dtype=dtype, device=device)
    torch.cuda.empty_cache()
    ag_send = prod_totals[my_sparse_group] // (2 * sparse_group_size)
    ag_in = torch.ones(ag_send, dtype=dtype, device=device)
    ag_out = torch.empty(sparse_group_size * ag_send, dtype=dtype, device=device)

    def run_ag_baseline() -> None:
        ag_in.fill_(1.0)
        dist.all_gather_into_tensor(ag_out, ag_in, group=my_pg)

    mean_ag_base, std_ag_base = (
        _measure_ms(run_ag_baseline, warmup, bench_iters)
        if _want("all_gather")
        else (0.0, 0.0)
    )
    _store_barrier()

    if rank == 0:
        results_dict["A_ag"] = (mean_ag_base, std_ag_base)

    dist.destroy_process_group()


def _benchmark_worker(
    rank: int,
    world_size: int,
    results_dict: Any,
    sharding_group_size: int = 2,
    port_offset: int = 0,
) -> None:
    """
    Worker for bench B (fused flat) and C (kernel direct).

    Uses an explicit TCPStore (same pattern as the deleted
    test_sharded_relay_2d_integration.py) so that RCCLX comm creation does
    not depend on dist._get_default_store(), which can hang when called from
    spawned child processes in the Meta environment.

    sharding_group_size selects the active-ranks-per-group (2 or 4). All four
    collectives (allreduce, reduce-scatter, all-to-all, all-gather) run at both
    2 and 4. port_offset isolates ports across the
    2-rank and 4-rank sweep iterations.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT + port_offset)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    # Explicit TCPStore — same as the integration test pattern.
    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT + port_offset,
        world_size=world_size,
        is_master=is_master,
        wait_for_workers=True,
    )

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        store=store,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Config
    dtype = _get_dtype()
    # Enforce minimum 1 warmup: on AMD GPUs the first RCCLX kernel call triggers
    # JIT compilation which takes several minutes.  Without warmup the timed run
    # shows compilation time, not runtime.
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 20))
    log_sizes = _env_str("BENCH_LOG_SIZES", "0") == "1"

    sparse_group_size = sharding_group_size
    num_sparse_groups = world_size // sparse_group_size
    my_sparse_group = local_rank // sparse_group_size
    all_active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]

    # Production per-group totals: groups have heterogeneous sizes matching BM-FM.
    prod_totals = _default_totals(dtype)

    # Tensors for Benchmark B.
    # Default: one flat tensor per group at the production total for this rank's
    # active group.  Groups are heterogeneous as in real BM-FM training.
    # Override with BENCH_NUM_TABLES + BENCH_TABLE_SIZE for smaller-scale runs.
    num_tables = _env_int("BENCH_NUM_TABLES", 1)
    table_sizes_all = [
        _env_int("BENCH_TABLE_SIZE", prod_totals[g]) for g in range(num_sparse_groups)
    ]
    all_tensors_sizes = [
        [max(1024, ts // (1 + (i % 5))) for i in range(num_tables)]
        for ts in table_sizes_all
    ]

    table_size = table_sizes_all[my_sparse_group]
    sizes = all_tensors_sizes[my_sparse_group]
    my_tensors = [torch.randn(sz, dtype=dtype, device=device) for sz in sizes]

    if log_sizes:
        if rank == 0:
            total = sum(sizes)
            print(f"[BENCH_LOG_SIZES] num_tables={num_tables}, dtype={dtype}")
            print(f"  sizes (first 10): {sizes[:10]}")
            print(f"  total_elements_per_group: {total}")
            print(
                f"  total_bytes_per_group: {total * dtype.itemsize / 1024 / 1024:.1f} MB"
            )
        dist.destroy_process_group()
        return

    # -------------------------------------------------------------------------
    # Benchmark B: 1 fused call with flat-concat buffers (proposed fix).
    # -------------------------------------------------------------------------

    # RCCLX comm — pass the explicit store directly (no _get_default_store())
    rcclx_comm = _setup_rcclx_comm(local_rank, world_size, 0, store)
    fused = _make_fused(rcclx_comm, local_rank, world_size, sparse_group_size)
    if fused is None:
        if rank == 0:
            print("[bench] FusedShardedRelayMultiGroup not available. Exiting.")
        dist.destroy_process_group()
        return

    # Helper sizes for Bench B and C
    total_my = sum(sizes)
    # With the passthrough kernel, each helper group gets its own buffer
    # sized to nActiveRanks × chunkSize (passthrough minimum).
    num_chunks = (world_size - sparse_group_size) + 1

    # Benchmark B flat buffers: active = total_my, each helper = passthrough size.
    flat_bufs: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            flat_bufs.append(torch.zeros(total_my, dtype=dtype, device=device))
        else:
            helper_total_g = sum(all_tensors_sizes[g])
            if sparse_group_size > 2:
                helper_size_g = 2 * helper_total_g
            else:
                helper_size_g = _passthrough_helper_size(
                    helper_total_g, sparse_group_size, num_chunks
                )
            flat_bufs.append(torch.zeros(helper_size_g, dtype=dtype, device=device))
    meta_cache_b: dict[str, list[int]] = {
        "bench"
        + str(dtype): [sum(all_tensors_sizes[g]) for g in range(num_sparse_groups)]
    }

    # Tensors for Benchmark C: one large tensor per group at production scale.
    # Override with BENCH_KERNEL_SIZE_GB for a different size.
    # All ranks must pass the same per_group_sizes vector for every group index.
    kernel_gb = _env_float("BENCH_KERNEL_SIZE_GB", 0.0)
    kernel_elements = (
        int(kernel_gb * 1024**3 / dtype.itemsize)
        if kernel_gb > 0
        else prod_totals[my_sparse_group]
    )
    kernel_declared_sizes: list[int] = (
        [kernel_elements] * num_sparse_groups
        if kernel_gb > 0
        else [prod_totals[g] for g in range(num_sparse_groups)]
    )
    kernel_tensor = torch.ones(kernel_elements, dtype=dtype, device=device)
    # For kernel bench, each helper group gets its own passthrough-sized buffer.
    kernel_scratch: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            kernel_scratch.append(kernel_tensor)
        else:
            declared_g = kernel_declared_sizes[g]
            if sparse_group_size > 2:
                helper_size_g = 2 * declared_g
            else:
                helper_size_g = _passthrough_helper_size(
                    declared_g, sparse_group_size, num_chunks
                )
            kernel_scratch.append(
                torch.empty(helper_size_g, dtype=dtype, device=device)
            )

    # One barrier + kernel warmup to trigger HIP JIT compilation before timing.
    torch.cuda.synchronize()
    dist.barrier()

    # Warmup with 1024 elements: must be >= numChunks * CACHE_LINE_SIZE = 7 * 64 = 448
    # so chunkSize > 0 after alignment, avoiding the degenerate chunkSize==count fallback.
    _tiny = torch.ones(1024, dtype=dtype, device=device)
    _tiny_scratch = [
        torch.ones(1024, dtype=dtype, device=device) for _ in range(num_sparse_groups)
    ]
    for _ in range(3):
        bench_kernel_direct(
            fused=fused,
            tensor=_tiny,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            scratch_tensors=_tiny_scratch,
        )
    torch.cuda.synchronize()
    del _tiny, _tiny_scratch

    def run_fused() -> None:
        for t in my_tensors:
            t.fill_(1.0)
        bench_fused_flat(
            fused=fused,
            my_tensors=my_tensors,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            local_size=world_size,
            intra_pg=None,
            sparse_group_size=sparse_group_size,
            flat_bufs=flat_bufs,
            meta_cache=meta_cache_b,
        )

    mean_fused, std_fused = (
        _measure_ms(run_fused, warmup, bench_iters)
        if _want("allreduce")
        else (0.0, 0.0)
    )

    # -------------------------------------------------------------------------
    # Benchmark C: direct kernel call with one large tensor per group.
    # -------------------------------------------------------------------------
    def run_kernel() -> None:
        kernel_tensor.fill_(1.0)
        fused.allreduce_multi_group(
            tensors=kernel_scratch,
            num_groups=num_sparse_groups,
            per_group_sizes=kernel_declared_sizes,
            all_active_ranks=all_active_ranks,
            op=dist.ReduceOp.AVG,
            skip_validation=True,
        )

    mean_kernel, std_kernel = (
        _measure_ms(run_kernel, warmup, bench_iters)
        if _want("allreduce")
        else (0.0, 0.0)
    )

    # Measure peak HBM usage — the whole point of this work.
    if _want("allreduce"):
        torch.cuda.reset_peak_memory_stats()
        run_fused()
        torch.cuda.synchronize()
        peak_hbm_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_hbm_bytes = 0

    # All sharded-relay collectives (reduce-scatter, all-to-all, all-gather) run
    # for both 2 and 4 active ranks.

    # -------------------------------------------------------------------------
    # Benchmark D: fused reduce-scatter (runs for all active-rank counts).
    # Release the Bench C kernel tensors first to reclaim HBM — reduce-scatter's
    # input buffer is A x its output (A blocks), so the active footprint is
    # larger than allreduce's. Reassign (instead of `del`) so the run_kernel
    # closure stays valid.
    # -------------------------------------------------------------------------
    kernel_tensor = torch.empty(0, dtype=dtype, device=device)
    kernel_scratch = []
    torch.cuda.empty_cache()

    # recv_count = prod_total // sparse_group_size so the input (A x recv_count)
    # matches the allreduce active footprint (one prod_total per group). Index by
    # range(num_sparse_groups) — NOT `for t in prod_totals` — because prod_totals
    # always has 4 entries while num_sparse_groups = local_size // sparse_group_size
    # (2 at 4-active), and per_group_recv_counts must match input_tensors length.
    rs_recv_counts = [
        prod_totals[g] // sparse_group_size for g in range(num_sparse_groups)
    ]
    my_rs_recv = rs_recv_counts[my_sparse_group]
    rs_input = torch.ones(sparse_group_size * my_rs_recv, dtype=dtype, device=device)
    rs_output = torch.empty(my_rs_recv, dtype=dtype, device=device)
    # Production per-table model tensors: packed into / unpacked from the flat
    # buffers so the measured cost includes the cat/copy the production util
    # (reduce_scatter_tensors_with_sharded_relay) performs.
    rs_in_model = [
        torch.ones(sparse_group_size * my_rs_recv, dtype=dtype, device=device)
    ]
    rs_out_model = [torch.empty(my_rs_recv, dtype=dtype, device=device)]
    rs_helper_bufs: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            rs_helper_bufs.append(rs_output)  # unused for the active group
        else:
            rs_helper_size_g = _passthrough_helper_size(
                rs_recv_counts[g], sparse_group_size, num_chunks
            )
            rs_helper_bufs.append(
                torch.empty(rs_helper_size_g, dtype=dtype, device=device)
            )

    def run_reduce_scatter() -> None:
        # Production path: pack the per-table model input into the contiguous
        # flat send buffer, run the fused call, then unpack into the model output.
        for t in rs_in_model:
            t.fill_(1.0)
        rs_input.copy_(rs_in_model[0].reshape(-1))
        bench_reduce_scatter_flat(
            fused=fused,
            input_flat=rs_input,
            output_flat=rs_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=rs_helper_bufs,
            per_group_recv_counts=rs_recv_counts,
        )
        rs_out_model[0].reshape(-1).copy_(rs_output)

    mean_rs, std_rs = (
        _measure_ms(run_reduce_scatter, warmup, bench_iters)
        if _want("reduce_scatter")
        else (0.0, 0.0)
    )

    def run_reduce_scatter_kernel() -> None:
        # Kernel-direct: raw kernel on the contiguous flat buffers, no pack/copy.
        rs_input.fill_(1.0)
        bench_reduce_scatter_flat(
            fused=fused,
            input_flat=rs_input,
            output_flat=rs_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=rs_helper_bufs,
            per_group_recv_counts=rs_recv_counts,
        )

    mean_rs_kernel, std_rs_kernel = (
        _measure_ms(run_reduce_scatter_kernel, warmup, bench_iters)
        if _want("reduce_scatter")
        else (0.0, 0.0)
    )

    # -------------------------------------------------------------------------
    # Benchmark E: fused all-to-all (runs for all active-rank counts). Release
    # the reduce-scatter buffers first. segment_count =
    # prod_total // (2 * sparse_group_size) so the out-of-place in/out buffers
    # (each A x segment_count = prod_total/2) stay within HBM. Indexed by
    # range(num_sparse_groups) (NOT `for t in prod_totals`) so the segment-count
    # list length matches num_sparse_groups (2 at 4-active).
    # -------------------------------------------------------------------------
    rs_input = torch.empty(0, dtype=dtype, device=device)
    rs_output = torch.empty(0, dtype=dtype, device=device)
    rs_helper_bufs = []
    rs_in_model = [torch.empty(0, dtype=dtype, device=device)]
    rs_out_model = [torch.empty(0, dtype=dtype, device=device)]
    torch.cuda.empty_cache()

    a2a_seg_counts = [
        prod_totals[g] // (2 * sparse_group_size) for g in range(num_sparse_groups)
    ]
    my_a2a_seg = a2a_seg_counts[my_sparse_group]
    a2a_input = torch.ones(sparse_group_size * my_a2a_seg, dtype=dtype, device=device)
    a2a_output = torch.empty(sparse_group_size * my_a2a_seg, dtype=dtype, device=device)
    a2a_in_model = [
        torch.ones(sparse_group_size * my_a2a_seg, dtype=dtype, device=device)
    ]
    a2a_out_model = [
        torch.empty(sparse_group_size * my_a2a_seg, dtype=dtype, device=device)
    ]
    a2a_helper_bufs: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            a2a_helper_bufs.append(a2a_output)  # unused for the active group
        else:
            if sparse_group_size > 2:
                # Flat A>2 all-to-all is pure-direct (no helper relay), so the
                # helper does no work for this group -- a tiny placeholder is
                # all the per-group tensor-list slot needs.
                a2a_helper_size_g = 1
            else:
                a2a_helper_size_g = _passthrough_helper_size(
                    a2a_seg_counts[g], sparse_group_size, num_chunks
                )
            a2a_helper_bufs.append(
                torch.empty(a2a_helper_size_g, dtype=dtype, device=device)
            )

    def run_all_to_all() -> None:
        # Production path: pack the per-table model input into the contiguous
        # flat send buffer, run the fused call, then unpack into the model output.
        for t in a2a_in_model:
            t.fill_(1.0)
        a2a_input.copy_(a2a_in_model[0].reshape(-1))
        bench_all_to_all_flat(
            fused=fused,
            input_flat=a2a_input,
            output_flat=a2a_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=a2a_helper_bufs,
            per_group_segment_counts=a2a_seg_counts,
        )
        a2a_out_model[0].reshape(-1).copy_(a2a_output)

    mean_a2a, std_a2a = (
        _measure_ms(run_all_to_all, warmup, bench_iters)
        if _want("all_to_all")
        else (0.0, 0.0)
    )

    def run_all_to_all_kernel() -> None:
        # Kernel-direct: raw kernel on the contiguous flat buffers, no pack/copy.
        a2a_input.fill_(1.0)
        bench_all_to_all_flat(
            fused=fused,
            input_flat=a2a_input,
            output_flat=a2a_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=a2a_helper_bufs,
            per_group_segment_counts=a2a_seg_counts,
        )

    mean_a2a_kernel, std_a2a_kernel = (
        _measure_ms(run_all_to_all_kernel, warmup, bench_iters)
        if _want("all_to_all")
        else (0.0, 0.0)
    )

    # -------------------------------------------------------------------------
    # Benchmark F: fused all-gather (runs for all active-rank counts). Release
    # the all-to-all buffers first. send_count = prod_total // (2*sparse_group_size)
    # so the output (A x send_count = prod_total/2) stays within HBM. Indexed by
    # range(num_sparse_groups) (NOT `for t in prod_totals`) so the send-count list
    # length matches num_sparse_groups (2 at 4-active).
    # -------------------------------------------------------------------------
    a2a_input = torch.empty(0, dtype=dtype, device=device)
    a2a_output = torch.empty(0, dtype=dtype, device=device)
    a2a_helper_bufs = []
    a2a_in_model = [torch.empty(0, dtype=dtype, device=device)]
    a2a_out_model = [torch.empty(0, dtype=dtype, device=device)]
    torch.cuda.empty_cache()

    ag_send_counts = [
        prod_totals[g] // (2 * sparse_group_size) for g in range(num_sparse_groups)
    ]
    my_ag_send = ag_send_counts[my_sparse_group]
    ag_input = torch.ones(my_ag_send, dtype=dtype, device=device)
    ag_output = torch.empty(sparse_group_size * my_ag_send, dtype=dtype, device=device)
    ag_in_model = [torch.ones(my_ag_send, dtype=dtype, device=device)]
    ag_out_model = [
        torch.empty(sparse_group_size * my_ag_send, dtype=dtype, device=device)
    ]
    ag_helper_bufs: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            ag_helper_bufs.append(ag_output)  # unused for the active group
        else:
            if sparse_group_size > 2:
                # Flat A>2 path: each helper stores one offload chunk per active
                # source (A*cs <= send_count). A*send_count safely covers that
                # (and matches the C++ tests).
                ag_helper_size_g = sparse_group_size * ag_send_counts[g]
            else:
                ag_helper_size_g = _passthrough_helper_size(
                    ag_send_counts[g], sparse_group_size, num_chunks
                )
            ag_helper_bufs.append(
                torch.empty(ag_helper_size_g, dtype=dtype, device=device)
            )

    def run_all_gather() -> None:
        # Production path: pack the per-table model input into the contiguous
        # flat send buffer, run the fused call, then unpack into the model output.
        for t in ag_in_model:
            t.fill_(1.0)
        ag_input.copy_(ag_in_model[0].reshape(-1))
        bench_all_gather_flat(
            fused=fused,
            input_flat=ag_input,
            output_flat=ag_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=ag_helper_bufs,
            per_group_send_counts=ag_send_counts,
        )
        ag_out_model[0].reshape(-1).copy_(ag_output)

    mean_ag, std_ag = (
        _measure_ms(run_all_gather, warmup, bench_iters)
        if _want("all_gather")
        else (0.0, 0.0)
    )

    def run_all_gather_kernel() -> None:
        # Kernel-direct: raw kernel on the contiguous flat buffers, no pack/copy.
        ag_input.fill_(1.0)
        bench_all_gather_flat(
            fused=fused,
            input_flat=ag_input,
            output_flat=ag_output,
            num_sparse_groups=num_sparse_groups,
            my_sparse_group=my_sparse_group,
            all_active_ranks=all_active_ranks,
            helper_bufs=ag_helper_bufs,
            per_group_send_counts=ag_send_counts,
        )

    mean_ag_kernel, std_ag_kernel = (
        _measure_ms(run_all_gather_kernel, warmup, bench_iters)
        if _want("all_gather")
        else (0.0, 0.0)
    )

    if rank == 0:
        total_bytes = sum(sz * dtype.itemsize for sz in sizes)
        kernel_bytes = kernel_elements * dtype.itemsize

        def bw(nbytes: int, ms: float) -> float:
            return 2 * nbytes / (ms / 1000) / 1e9 if ms > 0 else 0.0

        bench_a_mean = float(results_dict["A"][0]) if "A" in results_dict else 0.0
        bench_a_std = float(results_dict["A"][1]) if "A" in results_dict else 0.0
        bench_a_rs_mean = (
            float(results_dict["A_rs"][0]) if "A_rs" in results_dict else 0.0
        )
        bench_a_rs_std = (
            float(results_dict["A_rs"][1]) if "A_rs" in results_dict else 0.0
        )
        bench_a_a2a_mean = (
            float(results_dict["A_a2a"][0]) if "A_a2a" in results_dict else 0.0
        )
        bench_a_a2a_std = (
            float(results_dict["A_a2a"][1]) if "A_a2a" in results_dict else 0.0
        )
        bench_a_ag_mean = (
            float(results_dict["A_ag"][0]) if "A_ag" in results_dict else 0.0
        )
        bench_a_ag_std = (
            float(results_dict["A_ag"][1]) if "A_ag" in results_dict else 0.0
        )

        rs_output_bytes = my_rs_recv * dtype.itemsize

        def rs_bw(nbytes: int, ms: float) -> float:
            # Reduce-scatter busBW factor for n active ranks is (n-1)/n.
            return (
                (sparse_group_size - 1) / sparse_group_size * nbytes / (ms / 1000) / 1e9
                if ms > 0
                else 0.0
            )

        print("\n" + "=" * 72)
        print(
            f"Sharded Relay Benchmark — MI350X ({world_size} GPUs), "
            f"{sparse_group_size} active ranks/group"
        )
        print("=" * 72)
        print(f"  dtype:          {dtype}")
        print(f"  num_tables:     {num_tables}")
        print(f"  active ranks/group: {sparse_group_size}")
        print("  times:          best-of-N (min), barrier-aligned + hipEvent-timed")
        print(
            f"  avg table size: {table_size:,} elements "
            f"({table_size * dtype.itemsize / 1024 / 1024:.1f} MB)"
        )
        print(f"  total data/grp: {total_bytes / 1024 / 1024:.1f} MB")

        # ----- ALLREDUCE -----
        print()
        print("-" * 72)
        print("ALLREDUCE")
        print("-" * 72)
        if bench_a_mean > 0:
            print(f"  [A] NCCL COALESCED (baseline, {num_tables} tensors):")
            print(
                f"       {bench_a_mean:.2f} ms  ±  {bench_a_std:.2f} ms  |  "
                f"{bw(total_bytes, bench_a_mean):.1f} GB/s"
            )
        else:
            print("  [A] NCCL COALESCED (baseline): N/A")
        print("  [B] SHARDED RELAY (1 fused call, passthrough helpers):")
        print(
            f"       {mean_fused:.2f} ms  ±  {std_fused:.2f} ms  |  "
            f"{bw(total_bytes, mean_fused):.1f} GB/s"
        )
        print(f"       Peak HBM: {peak_hbm_bytes / 1024 / 1024 / 1024:.2f} GiB")
        print(
            f"  [C] KERNEL DIRECT (1 large tensor, {kernel_bytes / 1024 / 1024:.0f} MB):"
        )
        print(
            f"       {mean_kernel:.2f} ms  ±  {std_kernel:.2f} ms  |  "
            f"{bw(kernel_bytes, mean_kernel):.1f} GB/s"
        )
        if bench_a_mean > 0 and mean_fused > 0:
            speedup_ar = bench_a_mean / mean_fused
            print(
                f"  >> Allreduce speedup (NCCL coalesced → sharded relay): "
                f"{speedup_ar:.2f}x"
            )
        if bench_a_mean > 0 and mean_kernel > 0:
            print(
                f"  >> Allreduce KERNEL speedup (NCCL → kernel-direct): "
                f"{bench_a_mean / mean_kernel:.2f}x"
            )

        # ----- REDUCE-SCATTER (runs for 2 or 4 active ranks) -----
        print()
        print("-" * 72)
        print("REDUCE-SCATTER")
        print("-" * 72)
        if bench_a_rs_mean > 0:
            print(
                f"  [A] NCCL reduce_scatter_tensor (baseline, "
                f"{rs_output_bytes / 1024 / 1024:.0f} MB output/group):"
            )
            print(
                f"       {bench_a_rs_mean:.2f} ms  ±  {bench_a_rs_std:.2f} ms  |  "
                f"{rs_bw(rs_output_bytes, bench_a_rs_mean):.1f} GB/s"
            )
        else:
            print("  [A] NCCL reduce_scatter_tensor (baseline): N/A")
        print(
            f"  [B] SHARDED RELAY (1 fused call, passthrough helpers, "
            f"{rs_output_bytes / 1024 / 1024:.0f} MB output/group):"
        )
        print(
            f"       {mean_rs:.2f} ms  ±  {std_rs:.2f} ms  |  "
            f"{rs_bw(rs_output_bytes, mean_rs):.1f} GB/s"
        )
        print(
            f"  [C] KERNEL DIRECT (no pack/copy, "
            f"{rs_output_bytes / 1024 / 1024:.0f} MB output/group):"
        )
        print(
            f"       {mean_rs_kernel:.2f} ms  ±  {std_rs_kernel:.2f} ms  |  "
            f"{rs_bw(rs_output_bytes, mean_rs_kernel):.1f} GB/s"
        )
        if bench_a_rs_mean > 0 and mean_rs > 0:
            speedup_rs = bench_a_rs_mean / mean_rs
            print(
                f"  >> Reduce-scatter speedup (NCCL → sharded relay): "
                f"{speedup_rs:.2f}x"
            )
        if bench_a_rs_mean > 0 and mean_rs_kernel > 0:
            print(
                f"  >> Reduce-scatter KERNEL speedup (NCCL → kernel-direct): "
                f"{bench_a_rs_mean / mean_rs_kernel:.2f}x"
            )

        # ----- ALL-TO-ALL (runs for 2 or 4 active ranks) -----
        a2a_seg_bytes = my_a2a_seg * dtype.itemsize
        print()
        print("-" * 72)
        print("ALL-TO-ALL")
        print("-" * 72)
        if bench_a_a2a_mean > 0:
            print(
                f"  [A] NCCL all_to_all_single (baseline, "
                f"{a2a_seg_bytes / 1024 / 1024:.0f} MB segment/group):"
            )
            print(
                f"       {bench_a_a2a_mean:.2f} ms  ±  {bench_a_a2a_std:.2f} ms  |  "
                f"{rs_bw(a2a_seg_bytes, bench_a_a2a_mean):.1f} GB/s"
            )
        else:
            print("  [A] NCCL all_to_all_single (baseline): N/A")
        print(
            f"  [B] SHARDED RELAY (1 fused call, passthrough helpers, "
            f"{a2a_seg_bytes / 1024 / 1024:.0f} MB segment/group):"
        )
        print(
            f"       {mean_a2a:.2f} ms  ±  {std_a2a:.2f} ms  |  "
            f"{rs_bw(a2a_seg_bytes, mean_a2a):.1f} GB/s"
        )
        print(
            f"  [C] KERNEL DIRECT (no pack/copy, "
            f"{a2a_seg_bytes / 1024 / 1024:.0f} MB segment/group):"
        )
        print(
            f"       {mean_a2a_kernel:.2f} ms  ±  {std_a2a_kernel:.2f} ms  |  "
            f"{rs_bw(a2a_seg_bytes, mean_a2a_kernel):.1f} GB/s"
        )
        if bench_a_a2a_mean > 0 and mean_a2a > 0:
            speedup_a2a = bench_a_a2a_mean / mean_a2a
            print(
                f"  >> All-to-all speedup (NCCL → sharded relay): "
                f"{speedup_a2a:.2f}x"
            )
        if bench_a_a2a_mean > 0 and mean_a2a_kernel > 0:
            print(
                f"  >> All-to-all KERNEL speedup (NCCL → kernel-direct): "
                f"{bench_a_a2a_mean / mean_a2a_kernel:.2f}x"
            )

        # ----- ALL-GATHER (runs for 2 or 4 active ranks) -----
        ag_send_bytes = my_ag_send * dtype.itemsize
        print()
        print("-" * 72)
        print("ALL-GATHER")
        print("-" * 72)
        if bench_a_ag_mean > 0:
            print(
                f"  [A] NCCL all_gather_into_tensor (baseline, "
                f"{ag_send_bytes / 1024 / 1024:.0f} MB send/group):"
            )
            print(
                f"       {bench_a_ag_mean:.2f} ms  ±  {bench_a_ag_std:.2f} ms  |  "
                f"{rs_bw(ag_send_bytes, bench_a_ag_mean):.1f} GB/s"
            )
        else:
            print("  [A] NCCL all_gather_into_tensor (baseline): N/A")
        print(
            f"  [B] SHARDED RELAY (1 fused call, passthrough helpers, "
            f"{ag_send_bytes / 1024 / 1024:.0f} MB send/group):"
        )
        print(
            f"       {mean_ag:.2f} ms  ±  {std_ag:.2f} ms  |  "
            f"{rs_bw(ag_send_bytes, mean_ag):.1f} GB/s"
        )
        print(
            f"  [C] KERNEL DIRECT (no pack/copy, "
            f"{ag_send_bytes / 1024 / 1024:.0f} MB send/group):"
        )
        print(
            f"       {mean_ag_kernel:.2f} ms  ±  {std_ag_kernel:.2f} ms  |  "
            f"{rs_bw(ag_send_bytes, mean_ag_kernel):.1f} GB/s"
        )
        if bench_a_ag_mean > 0 and mean_ag > 0:
            speedup_ag = bench_a_ag_mean / mean_ag
            print(
                f"  >> All-gather speedup (NCCL → sharded relay): " f"{speedup_ag:.2f}x"
            )
        if bench_a_ag_mean > 0 and mean_ag_kernel > 0:
            print(
                f"  >> All-gather KERNEL speedup (NCCL → kernel-direct): "
                f"{bench_a_ag_mean / mean_ag_kernel:.2f}x"
            )
        print("=" * 72)

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Message-size sweep workers (test_collectives_msg_size_sweep)
#
# Sweeps every collective (allreduce, reduce-scatter, all-to-all, all-gather) at
# both 2 and 4 active ranks over a fixed list of message sizes and reports, per
# size, the sharded-relay speedup over NCCL for the FUSED scenario:
#   FUSED — (NUM_GPUS // A) concurrent A-rank groups in one multi-group kernel
#           call, vs the NCCL baseline run in parallel on each rank's A-rank
#           sub-group.
# The single-group scenario is benchmarked separately (see the parallel
# independent-comm sweep, test_parallel_collectives_msg_size_sweep).
# Fixed at bf16. NCCL allreduce/reduce-scatter baselines use SUM + manual divide
# (RCCL on MI350X lacks the AVG kernel); the relay path uses AVG in the kernel.
# all-to-all / all-gather do no reduction. The swept nbytes is the per-active-rank
# input tensor byte size, so sizes stay comparable across collectives.
# ---------------------------------------------------------------------------

# Data type for the message-size sweep comparison.
_MSG_SWEEP_DTYPE: torch.dtype = torch.bfloat16


# ----- Per-collective buffer/shape helpers (shared by NCCL + relay) ---------


def _active_io(
    collective: str,
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(input, output) tensors an ACTIVE rank contributes for one collective.

    `elements` is the per-active-rank input size (the swept size axis).
    """
    if collective == "allreduce":
        t = torch.ones(elements, dtype=dtype, device=device)
        return t, t  # in place
    if collective == "reduce_scatter":
        recv = elements // active_ranks
        return (
            torch.ones(elements, dtype=dtype, device=device),  # A * recv
            torch.empty(recv, dtype=dtype, device=device),
        )
    if collective == "all_to_all":
        return (
            torch.ones(elements, dtype=dtype, device=device),  # A * seg
            torch.empty(elements, dtype=dtype, device=device),
        )
    if collective == "all_gather":
        return (
            torch.ones(elements, dtype=dtype, device=device),  # send
            torch.empty(active_ranks * elements, dtype=dtype, device=device),
        )
    raise ValueError(f"unknown collective {collective!r}")


def _relay_helper_size(
    collective: str, active_ranks: int, elements: int, num_chunks: int
) -> int:
    """Helper-buffer element count for a single helper group's relay tensor.

    Mirrors the per-collective helper sizing _benchmark_worker uses for A=2 vs A>2.
    """
    if collective == "allreduce":
        if active_ranks > 2:
            return 2 * elements
        return _passthrough_helper_size(elements, active_ranks, num_chunks)
    if collective == "reduce_scatter":
        recv = elements // active_ranks
        return _passthrough_helper_size(recv, active_ranks, num_chunks)
    if collective == "all_to_all":
        if active_ranks > 2:
            return 1  # pure-direct A>2 all-to-all: helper does no relay work
        seg = elements // active_ranks
        return _passthrough_helper_size(seg, active_ranks, num_chunks)
    if collective == "all_gather":
        if active_ranks > 2:
            return active_ranks * elements
        return _passthrough_helper_size(elements, active_ranks, num_chunks)
    raise ValueError(f"unknown collective {collective!r}")


def _relay_counts(
    collective: str, active_ranks: int, elements: int, num_groups: int
) -> list[int]:
    """Per-group declared count vector passed to the fused relay call."""
    if collective in ("allreduce", "all_gather"):
        return [elements] * num_groups
    if collective in ("reduce_scatter", "all_to_all"):
        return [elements // active_ranks] * num_groups
    raise ValueError(f"unknown collective {collective!r}")


def _build_relay_bufs(
    collective: str,
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    num_groups: int,
    my_sparse_group: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Build (active_in, active_out, per_group_helper_bufs) for one FUSED relay
    call. Every rank is active in exactly one of the num_groups groups; the
    helper slot for the active group is a placeholder (unused by the kernel)."""
    active_in, active_out = _active_io(
        collective, active_ranks, elements, dtype, device
    )
    helper_size = _relay_helper_size(collective, active_ranks, elements, num_chunks)
    bufs: list[torch.Tensor] = []
    for g in range(num_groups):
        if g == my_sparse_group:
            bufs.append(active_out)  # active group slot — unused by the kernel
        else:
            bufs.append(torch.empty(helper_size, dtype=dtype, device=device))
    return active_in, active_out, bufs


def _build_relay_group_lists(
    active_in: torch.Tensor,
    active_out: torch.Tensor,
    bufs: list[torch.Tensor],
    num_groups: int,
    my_sparse_group: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-group (input, output) tensor lists for a FUSED relay call, built ONCE
    outside the timed loop: the active group slot holds this rank's real
    tensors; every other (helper) slot holds that group's passthrough scratch.
    For allreduce (in-place) out_list mirrors in_list and is otherwise unused."""
    in_list: list[torch.Tensor] = []
    out_list: list[torch.Tensor] = []
    for g in range(num_groups):
        if g == my_sparse_group:
            in_list.append(active_in)
            out_list.append(active_out)
        else:
            in_list.append(bufs[g])
            out_list.append(bufs[g])
    return in_list, out_list


def _run_relay_once(
    collective: str,
    relay: Any,
    in_list: list[torch.Tensor],
    out_list: list[torch.Tensor],
    counts: list[int],
    num_groups: int,
    my_sparse_group: int,
    all_active_ranks: list[list[int]],
) -> None:
    """One sharded-relay collective call with PRE-BUILT per-group tensor lists.

    The per-group input/output lists (and inputs) are constructed once outside
    the timed loop, so the timed region is just the wrapper dispatch — on par
    with the NCCL baseline's single call. Inputs stay ones (AVG of ones = ones;
    the non-allreduce collectives are out-of-place), so no per-call refill."""
    if collective == "allreduce":
        relay.allreduce_multi_group(
            tensors=in_list,
            num_groups=num_groups,
            per_group_sizes=counts,
            all_active_ranks=all_active_ranks,
            op=dist.ReduceOp.AVG,
            skip_validation=True,
        )
    elif collective == "reduce_scatter":
        relay.reduce_scatter_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=num_groups,
            per_group_recv_counts=counts,
            all_active_ranks=all_active_ranks,
            op=dist.ReduceOp.AVG,
            skip_validation=True,
        )
    elif collective == "all_to_all":
        relay.all_to_all_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=num_groups,
            per_group_segment_counts=counts,
            all_active_ranks=all_active_ranks,
            skip_validation=True,
        )
    elif collective == "all_gather":
        relay.all_gather_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=num_groups,
            per_group_send_counts=counts,
            all_active_ranks=all_active_ranks,
            skip_validation=True,
        )
    else:
        raise ValueError(f"unknown collective {collective!r}")


# ----- NCCL baseline op builders (one per collective) ----------------------


def _nccl_ops_allreduce(
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    fused_pg: dist.ProcessGroup,
    ar_opts: Any,
) -> Any:
    ft = torch.ones(elements, dtype=dtype, device=device)

    def run_fused() -> None:
        fused_pg.allreduce_coalesced([ft], opts=ar_opts).wait()
        ft.div_(active_ranks)

    return run_fused


def _nccl_ops_reduce_scatter(
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    fused_pg: dist.ProcessGroup,
    _ar_opts: Any,
) -> Any:
    recv = elements // active_ranks
    fin = torch.ones(elements, dtype=dtype, device=device)  # A * recv
    fout = torch.empty(recv, dtype=dtype, device=device)

    def run_fused() -> None:
        dist.reduce_scatter_tensor(fout, fin, op=dist.ReduceOp.SUM, group=fused_pg)
        fout.div_(active_ranks)

    return run_fused


def _nccl_ops_all_to_all(
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    fused_pg: dist.ProcessGroup,
    _ar_opts: Any,
) -> Any:
    fin = torch.ones(elements, dtype=dtype, device=device)  # A * seg
    fout = torch.empty(elements, dtype=dtype, device=device)

    def run_fused() -> None:
        dist.all_to_all_single(fout, fin, group=fused_pg, async_op=True).wait()

    return run_fused


def _nccl_ops_all_gather(
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    fused_pg: dist.ProcessGroup,
    _ar_opts: Any,
) -> Any:
    fin = torch.ones(elements, dtype=dtype, device=device)  # send
    fout = torch.empty(active_ranks * elements, dtype=dtype, device=device)

    def run_fused() -> None:
        dist.all_gather_into_tensor(fout, fin, group=fused_pg, async_op=True).wait()

    return run_fused


def _nccl_baseline_op(
    collective: str,
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
    fused_pg: dist.ProcessGroup,
    ar_opts: Any,
) -> Any:
    """Return the FUSED NCCL baseline closure for one size (runs on this rank's
    A-rank sub-group)."""
    builders = {
        "allreduce": _nccl_ops_allreduce,
        "reduce_scatter": _nccl_ops_reduce_scatter,
        "all_to_all": _nccl_ops_all_to_all,
        "all_gather": _nccl_ops_all_gather,
    }
    return builders[collective](
        active_ranks, elements, dtype, device, fused_pg, ar_opts
    )


def _msg_sweep_nccl_worker(
    rank: int,
    world_size: int,
    results_dict: Any,
    port_offset: int = 0,
) -> None:
    """NCCL FUSED baselines for the message-size sweep.

    Writes, for each (collective, active-rank-count A, size index i) (rank 0 only):
      results_dict["nccl_{collective}_a{A}_fused_{i}"] = (best_ms, std_ms)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT + 200 + port_offset)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT + 200 + port_offset,
        world_size=world_size,
        is_master=is_master,
        wait_for_workers=True,
    )
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, store=store
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype = _MSG_SWEEP_DTYPE
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 100))

    # Build the A-rank sub-group PGs once per A (all ranks call new_group for
    # every group collectively). pgs_by_a[A][rank // A] is this rank's FUSED
    # sub-group.
    pgs_by_a: dict[int, list[dist.ProcessGroup]] = {}
    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        num_groups = world_size // active_ranks
        groups = [
            list(range(g * active_ranks, (g + 1) * active_ranks))
            for g in range(num_groups)
        ]
        pgs_by_a[active_ranks] = [dist.new_group(ranks=grp) for grp in groups]

    # SUM + manual divide (RCCL on MI350X lacks the AVG kernel).
    _ar_opts = dist.AllreduceCoalescedOptions()
    _ar_opts.reduceOp = dist.ReduceOp.SUM

    # Warm up each FUSED sub-group communicator.
    _wt = torch.ones(1, dtype=dtype, device=device)
    for active_ranks, pgs in pgs_by_a.items():
        dist.all_reduce(_wt, op=dist.ReduceOp.SUM, group=pgs[rank // active_ranks])
    torch.cuda.synchronize()
    del _wt

    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        pgs = pgs_by_a[active_ranks]
        fused_pg = pgs[rank // active_ranks]
        for collective in _MSG_SWEEP_COLLECTIVES:
            for i, (_label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
                elements = nbytes // dtype.itemsize
                if elements % active_ranks != 0:
                    continue
                run_fused_base = _nccl_baseline_op(
                    collective,
                    active_ranks,
                    elements,
                    dtype,
                    device,
                    fused_pg,
                    _ar_opts,
                )
                best_f, std_f = _measure_ms(run_fused_base, warmup, bench_iters)
                # Drop the per-size tensors held by the closure before the next
                # (larger) size.
                del run_fused_base
                torch.cuda.empty_cache()

                if rank == 0:
                    key = f"{collective}_a{active_ranks}"
                    results_dict[f"nccl_{key}_fused_{i}"] = (best_f, std_f)

    # Match the relay worker: let every rank finish its collectives before any
    # rank tears the process group down.
    dist.barrier()
    dist.destroy_process_group()


def _measure_relay_for(
    collective: str,
    active_ranks: int,
    relay_fused: Any,
    nbytes: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    warmup: int,
    bench_iters: int,
) -> tuple[float, float]:
    """Time the FUSED (num_groups = world // A) relay collective for one message
    size. Returns (best_ms, std_ms)."""
    elements = nbytes // dtype.itemsize
    if active_ranks == 0 or elements % active_ranks != 0:
        return 0.0, 0.0
    num_chunks = (world_size - active_ranks) + 1

    num_groups = world_size // active_ranks
    my_sparse_group = rank // active_ranks
    fused_active_ranks = [
        list(range(g * active_ranks, (g + 1) * active_ranks)) for g in range(num_groups)
    ]
    active_in, active_out, bufs = _build_relay_bufs(
        collective,
        active_ranks,
        elements,
        dtype,
        device,
        num_groups,
        my_sparse_group,
        num_chunks,
    )
    counts = _relay_counts(collective, active_ranks, elements, num_groups)
    in_list, out_list = _build_relay_group_lists(
        active_in, active_out, bufs, num_groups, my_sparse_group
    )
    best_f, std_f = _measure_ms(
        partial(
            _run_relay_once,
            collective,
            relay_fused,
            in_list,
            out_list,
            counts,
            num_groups,
            my_sparse_group,
            fused_active_ranks,
        ),
        warmup,
        bench_iters,
    )
    del active_in, active_out, bufs, in_list, out_list
    torch.cuda.empty_cache()
    return best_f, std_f


def _msg_sweep_relay_warmup(
    fused_by_a: dict[int, Any],
    rank: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Tiny FUSED relay calls for every (A, collective) so each distinct HIP
    kernel config JIT-compiles before timing."""
    torch.cuda.synchronize()
    dist.barrier()
    # The A=2 reduce-scatter/all-to-all 2-active relay path requires the chunked
    # quantity Q = elements // A >= 128 * numChunks (= 128 * (world - A + 1)); for
    # 8 ranks, A=2 that is Q >= 896, i.e. elements >= A * 896 = 1792. Use 2048 so
    # every (collective, A) warmup config clears the floor (the real sweep starts
    # at 4 KB = 2048 elements, so it is always valid too).
    warm_elems = 2048
    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        num_chunks = (world_size - active_ranks) + 1
        num_groups = world_size // active_ranks
        my_sparse_group = rank // active_ranks
        fused_active_ranks = [
            list(range(g * active_ranks, (g + 1) * active_ranks))
            for g in range(num_groups)
        ]
        for collective in _MSG_SWEEP_COLLECTIVES:
            a_in, a_out, bufs = _build_relay_bufs(
                collective,
                active_ranks,
                warm_elems,
                dtype,
                device,
                num_groups,
                my_sparse_group,
                num_chunks,
            )
            counts = _relay_counts(collective, active_ranks, warm_elems, num_groups)
            in_list, out_list = _build_relay_group_lists(
                a_in, a_out, bufs, num_groups, my_sparse_group
            )
            for _ in range(3):
                _run_relay_once(
                    collective,
                    fused_by_a[active_ranks],
                    in_list,
                    out_list,
                    counts,
                    num_groups,
                    my_sparse_group,
                    fused_active_ranks,
                )
            del a_in, a_out, bufs, in_list, out_list
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _msg_sweep_relay_worker(
    rank: int,
    world_size: int,
    results_dict: Any,
    port_offset: int = 0,
) -> None:
    """Sharded-relay FUSED collectives for the message-size sweep.

    Writes, for each (collective, active-rank-count A, size index i) (rank 0 only):
      results_dict["relay_{collective}_a{A}_fused_{i}"] = (best_ms, std_ms)

    Rank 0 prints the final summary tables using both the relay results and the
    NCCL results written earlier by _msg_sweep_nccl_worker.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT + 300 + port_offset)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT + 300 + port_offset,
        world_size=world_size,
        is_master=is_master,
        wait_for_workers=True,
    )
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, store=store
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dtype = _MSG_SWEEP_DTYPE
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 100))

    # One shared intra-node RCCLX comm, then a FUSED (num_groups = world // A)
    # relay object per active-rank count. Every rank constructs all objects
    # collectively.
    rcclx_comm = _setup_rcclx_comm(local_rank, world_size, 0, store)
    fused_by_a: dict[int, Any] = {}
    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        fused_by_a[active_ranks] = _make_fused(
            rcclx_comm, local_rank, world_size, active_ranks
        )
    if rcclx_comm is None or any(v is None for v in fused_by_a.values()):
        if rank == 0:
            print("[bench] FusedShardedRelayMultiGroup not available. Exiting.")
        dist.destroy_process_group()
        return

    # Pre-compile every distinct HIP kernel config before timing.
    _msg_sweep_relay_warmup(fused_by_a, rank, world_size, dtype, device)

    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        relay_fused = fused_by_a[active_ranks]
        for collective in _MSG_SWEEP_COLLECTIVES:
            for i, (_label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
                best_f, std_f = _measure_relay_for(
                    collective=collective,
                    active_ranks=active_ranks,
                    relay_fused=relay_fused,
                    nbytes=nbytes,
                    dtype=dtype,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    warmup=warmup,
                    bench_iters=bench_iters,
                )
                if rank == 0:
                    key = f"{collective}_a{active_ranks}"
                    results_dict[f"relay_{key}_fused_{i}"] = (best_f, std_f)

    dist.barrier()
    dist.destroy_process_group()


def _sweep_get_ms(results_dict: Any, key: str) -> float:
    v = results_dict.get(key)
    return float(v[0]) if v is not None else 0.0


def _sweep_fmt_ms(v: float) -> str:
    return f"{v:.3f}" if v > 0 else "N/A"


def _sweep_fmt_speedup(nccl: float, relay: float) -> str:
    return f"{nccl / relay:.2f}x" if nccl > 0 and relay > 0 else "N/A"


def _emit_report(lines: list[str], default_basename: str) -> None:
    """Write the assembled results tables to a dedicated file AND to stdout as a
    single atomic write.

    The sweep spawns up to N * NUM_GPUS worker processes, each emitting glog /
    thrift / RCCLX C++ init logging to the shared stdout/stderr. Interleaving
    that firehose with per-line print() mangles the tables (a stray token can
    land mid-row). Assembling the whole report as one string and writing it once
    — and also to an isolated file no other process touches — keeps the results
    clean and diffable. BENCH_RESULTS_FILE overrides the file path.
    """
    report = "\n".join(lines) + "\n"
    path = os.environ.get(
        "BENCH_RESULTS_FILE",
        os.path.join(tempfile.gettempdir(), default_basename),
    )
    try:
        with open(path, "w") as f:
            f.write(report)
    except OSError:
        path = ""
    banner = "#" * 55
    header = "# BENCH RESULTS" + (f" (also written to {path})" if path else "")
    sys.stdout.write(f"\n{banner}\n{header}\n{banner}\n{report}{banner}\n")
    sys.stdout.flush()


def _format_sweep_table(
    results_dict: Any, dtype: torch.dtype, collective: str, active_ranks: int
) -> list[str]:
    """Build the lines for one (collective, A) FUSED message-size sweep table."""
    num_groups = NUM_GPUS // active_ranks
    key = f"{collective}_a{active_ranks}"
    title = collective.replace("_", "-").upper()
    width = 55
    line = "=" * width
    out: list[str] = [
        "",
        line,
        f"Sharded Relay {title} — Message-Size Sweep "
        f"(MI350X, {NUM_GPUS} GPUs, {dtype})",
        f"  {active_ranks} active ranks/group; times = best-of-N (min), "
        "barrier-aligned + hipEvent-timed",
        f"  FUSED = {num_groups}-group sharded relay  vs  "
        f"NCCL baseline ({num_groups} concurrent {active_ranks}-rank groups)",
        line,
        f"{'':>10} | {f'FUSED ({num_groups} groups)':^33}",
        f"{'Msg Size':>10} | {'NCCL(ms)':>10} {'Relay(ms)':>10} {'Speedup':>9}",
        "-" * width,
    ]
    for i, (label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
        elements = nbytes // dtype.itemsize
        if elements % active_ranks != 0:
            continue
        nf = _sweep_get_ms(results_dict, f"nccl_{key}_fused_{i}")
        rf = _sweep_get_ms(results_dict, f"relay_{key}_fused_{i}")
        out.append(
            f"{label:>10} | {_sweep_fmt_ms(nf):>10} {_sweep_fmt_ms(rf):>10} "
            f"{_sweep_fmt_speedup(nf, rf):>9}"
        )
    out.append(line)
    return out


def _print_msg_sweep_report(results_dict: Any, dtype: torch.dtype) -> None:
    """Emit one FUSED message-size sweep table per (collective, active-rank-count)
    as a single clean, diffable block (file + atomic stdout write)."""
    lines: list[str] = []
    for collective in _MSG_SWEEP_COLLECTIVES:
        for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
            lines.extend(
                _format_sweep_table(results_dict, dtype, collective, active_ranks)
            )
    _emit_report(lines, "bench_sharded_relay_fused_sweep_results.txt")


# ---------------------------------------------------------------------------
# Parallel independent-comm sweep workers (test_parallel_collectives_msg_size_sweep)
#
# Models N = NUM_GPUS // A independent workloads, each with its OWN 8-rank
# communicator and a disjoint active rank group, all launching a SINGLE-GROUP
# A-rank sharded relay collective IN PARALLEL (async, one per comm). Unlike the
# fused multi-group kernel (which coordinates all groups in lockstep phases to
# remove XGMI link contention), these calls cannot be fused — each workload owns
# a separate communicator — so this measures single-group A-rank relay collective
# performance under the link contention of several overlapping independent
# relays. Covers all four collectives (allreduce, reduce-scatter, all-to-all,
# all-gather) at both 2 and 4 active ranks → one table per (collective, A).
#
# Topology (8 GPUs, A active ranks/group); each comm spans ALL 8 ranks, e.g. A=2:
#   comm 0: active [0,1], helpers [2,3,4,5,6,7]
#   comm 1: active [2,3], helpers [0,1,4,5,6,7]  ... (N=4 comms)
# Each rank is active in exactly one comm and a helper in the others.
#
# Fixed at bf16. NCCL baseline: N disjoint A-rank NCCL collectives running
# concurrently (identical to the FUSED sweep's baseline; reuses _nccl_baseline_op).
# NCCL allreduce/reduce-scatter use SUM + manual divide (RCCL on MI350X lacks the
# AVG kernel); the relay path uses AVG in the RCCLX kernel.
# ---------------------------------------------------------------------------


def _build_per_job_nccl_group(total_procs: int, my_job: int) -> Any:
    """Collectively build each job's NUM_GPUS-rank NCCL group (over global ranks
    [j*NUM_GPUS .. j*NUM_GPUS+NUM_GPUS-1]); return this proc's own job group.

    Called with a global (gloo) default PG so dist.barrier(group=...) on the
    returned group is an on-device NCCL barrier that aligns the job's NUM_GPUS
    participants — matching how the NCCL baseline aligns its participants."""
    my_group = None
    num_jobs = total_procs // NUM_GPUS
    for j in range(num_jobs):
        ranks = list(range(j * NUM_GPUS, (j + 1) * NUM_GPUS))
        g = dist.new_group(ranks=ranks, backend="nccl")
        if j == my_job:
            my_group = g
    return my_group


def _build_per_job_active_subgroup(
    total_procs: int, my_job: int, active_ranks: int
) -> Any:
    """Collectively build each job's A-rank NCCL sub-group used for the NCCL
    baseline collective. Job j's active ranks are rank_in_job [j*A .. j*A+A-1]
    (device j*A..), i.e. global ranks [j*NUM_GPUS + r]. Returns this proc's own
    job sub-group (a non-member handle if this proc is a helper)."""
    my_sub = None
    num_jobs = total_procs // NUM_GPUS
    for j in range(num_jobs):
        active_in_job = range(j * active_ranks, (j + 1) * active_ranks)
        ranks = [j * NUM_GPUS + r for r in active_in_job]
        g = dist.new_group(ranks=ranks, backend="nccl")
        if j == my_job:
            my_sub = g
    return my_sub


def _noop_fn() -> None:
    """Timed no-op for idle co-resident ranks in the parallel NCCL baseline."""
    return None


def _parallel_msg_sweep_nccl_worker(
    p: int,
    total_procs: int,
    active_ranks: int,
    results_dict: Any,
    store_port: int,
) -> None:
    """NCCL baseline under the SAME topology/co-residency as the relay sweep.

    To compare the collective ALGORITHMS apples-to-apples (not the deployment
    topology), this mirrors the relay worker exactly: N = NUM_GPUS // A separate
    8-rank jobs are spawned as N * NUM_GPUS processes (N per GPU); process p is
    (job = p // NUM_GPUS, rank_in_job = p % NUM_GPUS) on cuda:rank_in_job. Each
    job builds its OWN 8-rank NCCL world (per-job PrefixStore, 1 rank/GPU) and,
    within it, an A-rank sub-group [job*A .. job*A+A-1]. The A active ranks run
    the A-rank NCCL collective on that sub-group; the other ranks are idle
    co-resident processes (exactly like the relay's helper ranks). Alignment uses
    the same per-job 8-rank NCCL device barrier (_measure_ms's dist.barrier), so
    both baselines pay identical co-residency and get identical device-side
    alignment. Records per job's rank-0 (rank_in_job == job*A):
      results_dict["nccl_parallel_{collective}_a{A}_{i}_job{job}"] = (best, std)
    """
    job = p // NUM_GPUS
    rank_in_job = p % NUM_GPUS
    device_idx = rank_in_job

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ["RANK"] = str(p)
    os.environ["WORLD_SIZE"] = str(total_procs)

    store = dist.TCPStore(
        host_name="localhost",
        port=store_port,
        world_size=total_procs,
        is_master=(p == 0),
        wait_for_workers=True,
    )
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")

    # Global (gloo) default PG across ALL N * NUM_GPUS procs: its dist.barrier
    # deterministically overlaps every co-resident job each iteration (identical
    # worst-case contention for both baselines). A per-job NUM_GPUS-rank NCCL
    # group provides the on-device within-job alignment (passed to _measure_ms).
    dist.init_process_group(
        backend="gloo",
        rank=p,
        world_size=total_procs,
        store=dist.PrefixStore("parallel_global", store),
    )
    job_device_group = _build_per_job_nccl_group(total_procs, job)

    dtype = _MSG_SWEEP_DTYPE
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 100))

    # A-rank sub-group (global ranks) for this job's NCCL collective.
    active_group = list(range(job * active_ranks, (job + 1) * active_ranks))
    subgroup = _build_per_job_active_subgroup(total_procs, job, active_ranks)
    is_active = rank_in_job in active_group
    is_job_rank0 = rank_in_job == active_group[0]

    # SUM + manual divide (RCCL on MI350X lacks the AVG kernel).
    _ar_opts = dist.AllreduceCoalescedOptions()
    _ar_opts.reduceOp = dist.ReduceOp.SUM

    # Warm up the sub-group communicator (active ranks only).
    if is_active:
        _wt = torch.ones(1, dtype=dtype, device=device)
        dist.all_reduce(_wt, op=dist.ReduceOp.SUM, group=subgroup)
        del _wt
    torch.cuda.synchronize()

    key = f"a{active_ranks}"
    for collective in _MSG_SWEEP_COLLECTIVES:
        for i, (_label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
            elements = nbytes // dtype.itemsize
            if elements % active_ranks != 0:
                continue
            if is_active:
                fn = _nccl_baseline_op(
                    collective,
                    active_ranks,
                    elements,
                    dtype,
                    device,
                    subgroup,
                    _ar_opts,
                )
            else:
                # Idle co-resident process (mirrors a relay helper rank); still
                # participates in the alignment barriers.
                fn = _noop_fn
            best, std = _measure_ms(
                fn, warmup, bench_iters, device_barrier_group=job_device_group
            )
            del fn
            torch.cuda.empty_cache()
            if is_job_rank0:
                results_dict[f"nccl_parallel_{collective}_{key}_{i}_job{job}"] = (
                    best,
                    std,
                )

    # Match the relay worker: let every rank finish its collectives before any
    # rank tears the process group down.
    dist.barrier()
    dist.destroy_process_group()


def _issue_relay_async(
    collective: str,
    comm: Any,
    in_list: list[torch.Tensor],
    out_list: list[torch.Tensor],
    count_list: list[int],
    all_active_ranks: list[list[int]],
) -> Any:
    """Issue ONE single-group (num_groups=1) sharded relay collective on a comm
    with async_op=True; returns its work handle."""
    if collective == "allreduce":
        return comm.allreduce_multi_group(
            tensors=in_list,
            num_groups=1,
            per_group_sizes=count_list,
            all_active_ranks=all_active_ranks,
            op=dist.ReduceOp.AVG,
            skip_validation=True,
            async_op=True,
        )
    if collective == "reduce_scatter":
        return comm.reduce_scatter_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=1,
            per_group_recv_counts=count_list,
            all_active_ranks=all_active_ranks,
            op=dist.ReduceOp.AVG,
            skip_validation=True,
            async_op=True,
        )
    if collective == "all_to_all":
        return comm.all_to_all_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=1,
            per_group_segment_counts=count_list,
            all_active_ranks=all_active_ranks,
            skip_validation=True,
            async_op=True,
        )
    if collective == "all_gather":
        return comm.all_gather_multi_group(
            input_tensors=in_list,
            output_tensors=out_list,
            num_groups=1,
            per_group_send_counts=count_list,
            all_active_ranks=all_active_ranks,
            skip_validation=True,
            async_op=True,
        )
    raise ValueError(f"unknown collective {collective!r}")


def _run_parallel_relay_once(
    collective: str,
    comms: list[Any],
    in_tensors: list[list[torch.Tensor]],
    out_tensors: list[list[torch.Tensor]],
    count_list: list[int],
    active_ranks_per_comm: list[list[list[int]]],
    num_parallel: int,
) -> None:
    """Issue N single-group relay collectives (one per comm) in parallel.

    Each call is async_op=True so it runs on its communicator's dedicated stream;
    the N calls therefore overlap. The wait() calls only enqueue device-side
    stream-waits on the current stream (non host-blocking), so the following
    end-event captures the LAST (max) completion across the N comms — i.e. the
    overlapped wall-time. Inputs are pre-filled with ones at buffer construction
    and stay ones, so no per-call refill is done inside the timed region.
    """
    works: list[Any] = []
    for k in range(num_parallel):
        works.append(
            _issue_relay_async(
                collective,
                comms[k],
                in_tensors[k],
                out_tensors[k],
                count_list,
                active_ranks_per_comm[k],
            )
        )
    for work in works:
        if work is not None:
            work.wait()


def _parallel_relay_single_call(
    collective: str,
    comm: Any,
    active_group: list[int],
    rank_in_job: int,
    active_ranks: int,
    elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """Zero-arg closure that issues this process's ONE single-group relay call.

    Active ranks (rank_in_job in active_group) pass real message buffers; helper
    ranks pass a 1-element placeholder (the kernel stages helpers into its own
    internal scratch and never reads the caller's buffer). Reuses
    _run_parallel_relay_once with num_parallel=1 so exactly one async relay
    collective is issued and waited on per call.
    """
    count_list = _relay_counts(collective, active_ranks, elements, 1)
    if rank_in_job in active_group:
        a_in, a_out = _active_io(collective, active_ranks, elements, dtype, device)
        in_list = [a_in]
        out_list = [a_out]
    else:
        # Helper slots are ignored by the kernel (it stages into internal
        # scratch), so a 1-element placeholder suffices for every collective.
        ph = torch.empty(1, dtype=dtype, device=device)
        in_list = [ph]
        out_list = [ph]
    return partial(
        _run_parallel_relay_once,
        collective,
        [comm],
        [in_list],
        [out_list],
        count_list,
        [[active_group]],
        1,
    )


def _parallel_relay_warmup_single(
    comm: Any,
    active_group: list[int],
    rank_in_job: int,
    active_ranks: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Tiny relay call per collective so each distinct single-group HIP kernel
    config JIT-compiles before timing. warm_elems=2048 clears the A=2
    reduce-scatter/all-to-all 2-active chunk floor (elements>=1792)."""
    torch.cuda.synchronize()
    dist.barrier()
    warm_elems = 2048
    for collective in _MSG_SWEEP_COLLECTIVES:
        fn = _parallel_relay_single_call(
            collective,
            comm,
            active_group,
            rank_in_job,
            active_ranks,
            warm_elems,
            dtype,
            device,
        )
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _parallel_msg_sweep_relay_worker(
    p: int,
    total_procs: int,
    active_ranks: int,
    results_dict: Any,
    store_port: int,
) -> None:
    """One production job's rank in the reconciled parallel relay sweep.

    Models N = NUM_GPUS // A independent relay jobs co-resident on one 8-GPU
    node (the production topology: N separate processes/jobs, each owning its own
    full 8-rank communicator, each issuing exactly ONE relay call). This spawn
    launches N * NUM_GPUS processes; process p maps to
    (job = p // NUM_GPUS, rank_in_job = p % NUM_GPUS) and runs on
    cuda:rank_in_job, so the N jobs are co-resident (N processes per GPU) with
    per-process CUDA contexts and their own GPU_MAX_HW_QUEUES=2 budget.

    Each process creates exactly ONE 8-rank RCCLX comm for its job (a per-job
    PrefixStore namespace via node_idx=job) and issues exactly ONE single-group
    A-rank relay call per timed iteration: active if rank_in_job is in the job's
    A-rank group [job*A .. job*A+A-1], helper otherwise. A gloo process group
    spanning all N * NUM_GPUS processes provides the cross-job barrier in
    _measure_ms so every job's timed region overlaps (true inter-process XGMI
    contention is the measurement target). Writes, per (collective, size index
    i), for each job's rank-0 (rank_in_job == job*A):
      results_dict["relay_parallel_{collective}_a{A}_{i}_job{job}"] = (best, std)
    The parent reduces max across jobs (overlapped wall-time).
    """
    job = p // NUM_GPUS
    rank_in_job = p % NUM_GPUS
    device_idx = rank_in_job

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ["RANK"] = str(p)
    os.environ["WORLD_SIZE"] = str(total_procs)

    store = dist.TCPStore(
        host_name="localhost",
        port=store_port,
        world_size=total_procs,
        is_master=(p == 0),
        wait_for_workers=True,
    )
    torch.cuda.set_device(device_idx)
    device = torch.device(f"cuda:{device_idx}")

    # Global (gloo) default PG across ALL N * NUM_GPUS procs: its dist.barrier
    # deterministically overlaps every co-resident job each iteration (same
    # worst-case contention every iter). A per-job NUM_GPUS-rank NCCL group gives
    # on-device within-job alignment (passed to _measure_ms as
    # device_barrier_group) so the P2P partner skew that would otherwise inflate
    # latency-bound small-size times is removed — identical treatment to the NCCL
    # baseline. This exactly matches the NCCL baseline's alignment, so relay vs
    # NCCL isolates the collective algorithm, not the harness.
    dist.init_process_group(
        backend="gloo",
        rank=p,
        world_size=total_procs,
        store=dist.PrefixStore("parallel_global", store),
    )
    job_device_group = _build_per_job_nccl_group(total_procs, job)

    dtype = _MSG_SWEEP_DTYPE
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 10))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 100))

    # This job owns ONE full 8-rank RCCLX comm in its own store namespace.
    raw_comm = _setup_rcclx_comm(
        rank_in_job,
        NUM_GPUS,
        job,
        dist.PrefixStore("parallel_relay_rcclx", store),
    )
    if FusedShardedRelayMultiGroup is None or raw_comm is None:
        if p == 0:
            print("[bench] FusedShardedRelayMultiGroup not available. Exiting.")
        dist.destroy_process_group()
        return

    # Job j's single active group is [j*A .. j*A+A-1], spreading the N jobs'
    # active ranks evenly across the 8 GPUs (each GPU is active for one job and a
    # helper for the other N-1) — matching production's balanced co-residency.
    active_group = list(range(job * active_ranks, (job + 1) * active_ranks))
    comm = FusedShardedRelayMultiGroup(
        rcclx_comm=raw_comm,
        world_size=NUM_GPUS,
        rank=rank_in_job,
        all_active_ranks=[active_group],
    )

    _parallel_relay_warmup_single(
        comm, active_group, rank_in_job, active_ranks, dtype, device
    )

    is_job_rank0 = rank_in_job == active_group[0]
    key = f"a{active_ranks}"
    for collective in _MSG_SWEEP_COLLECTIVES:
        for i, (_label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
            elements = nbytes // dtype.itemsize
            if elements % active_ranks != 0:
                continue
            fn = _parallel_relay_single_call(
                collective,
                comm,
                active_group,
                rank_in_job,
                active_ranks,
                elements,
                dtype,
                device,
            )
            best, std = _measure_ms(
                fn, warmup, bench_iters, device_barrier_group=job_device_group
            )
            del fn
            torch.cuda.empty_cache()
            if is_job_rank0:
                results_dict[f"relay_parallel_{collective}_{key}_{i}_job{job}"] = (
                    best,
                    std,
                )

    dist.barrier()
    dist.destroy_process_group()


def _reduce_parallel_jobs_max(results_dict: Any) -> None:
    """Collapse per-job parallel-sweep entries to the max across jobs.

    Each job writes its own best-of-N under ..._job{j}; the overlapped wall-time
    is the slowest (max) job, so reduce both the relay and NCCL per-job entries
    into the base keys the print/report helpers read.
    """
    for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
        num_jobs = NUM_GPUS // active_ranks
        for collective in _MSG_SWEEP_COLLECTIVES:
            key = f"{collective}_a{active_ranks}"
            for i, (_label, _nbytes) in enumerate(_MSG_SWEEP_SIZES):
                for prefix in ("relay_parallel", "nccl_parallel"):
                    vals = [
                        results_dict.get(f"{prefix}_{key}_{i}_job{j}")
                        for j in range(num_jobs)
                    ]
                    present = [v for v in vals if v is not None]
                    if present:
                        results_dict[f"{prefix}_{key}_{i}"] = max(
                            present, key=lambda v: v[0]
                        )


def _format_parallel_sweep_table(
    results_dict: Any, dtype: torch.dtype, collective: str, active_ranks: int
) -> list[str]:
    """Build the lines for one (collective, A) parallel separate-job table."""
    num_parallel = NUM_GPUS // active_ranks
    key = f"{collective}_a{active_ranks}"
    title = collective.replace("_", "-").upper()
    width = 55
    line = "=" * width
    out: list[str] = [
        "",
        line,
        f"Sharded Relay {title} — {num_parallel}x Parallel Separate-Job "
        f"Sweep (MI350X, {NUM_GPUS} GPUs, {dtype})",
        f"  {active_ranks} active ranks/group; times = best-of-N (min), "
        "barrier-aligned + hipEvent-timed; max across jobs",
        f"  PARALLEL = {num_parallel} co-resident jobs, each a separate 8-rank "
        f"comm issuing one {active_ranks}-rank relay (max across jobs)",
        f"  NCCL baseline = {num_parallel} concurrent {active_ranks}-rank NCCL "
        "collectives as separate processes (max across jobs)",
        line,
        f"{'':>10} | {f'PARALLEL ({num_parallel} jobs)':^33}",
        f"{'Msg Size':>10} | {'NCCL(ms)':>10} {'Relay(ms)':>10} {'Speedup':>9}",
        "-" * width,
    ]
    for i, (label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
        elements = nbytes // dtype.itemsize
        if elements % active_ranks != 0:
            continue
        n = _sweep_get_ms(results_dict, f"nccl_parallel_{key}_{i}")
        r = _sweep_get_ms(results_dict, f"relay_parallel_{key}_{i}")
        out.append(
            f"{label:>10} | {_sweep_fmt_ms(n):>10} {_sweep_fmt_ms(r):>10} "
            f"{_sweep_fmt_speedup(n, r):>9}"
        )
    out.append(line)
    return out


def _print_parallel_msg_sweep_report(results_dict: Any, dtype: torch.dtype) -> None:
    """Emit one parallel separate-job table per (collective, active-rank-count) as
    a single clean, diffable block (file + atomic stdout write)."""
    lines: list[str] = []
    for collective in _MSG_SWEEP_COLLECTIVES:
        for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
            lines.extend(
                _format_parallel_sweep_table(
                    results_dict, dtype, collective, active_ranks
                )
            )
    _emit_report(lines, "bench_sharded_relay_parallel_sweep_results.txt")


def _format_single_group_sweep_table(
    results_dict: Any, dtype: torch.dtype, collective: str, active_ranks: int
) -> list[str]:
    """Build the lines for one (collective, A) SINGLE-GROUP best-case table.

    Best case = exactly ONE A-rank relay group on the 8-GPU host with no other
    co-resident job competing for XGMI bandwidth or HW queues (num_jobs == 1).
    The relay still spans the full 8-rank comm (A active + 8-A helpers); the
    NCCL baseline runs the A-rank collective with the remaining ranks idle. The
    alignment (global barrier + per-job device barrier) matches the parallel
    sweep, so this is that sweep's num_jobs=1 point — the contention-free upper
    bound. Reads the same reduced base keys the parallel report reads.
    """
    key = f"{collective}_a{active_ranks}"
    title = collective.replace("_", "-").upper()
    width = 55
    line = "=" * width
    out: list[str] = [
        "",
        line,
        f"Sharded Relay {title} — Single-Group Best-Case Sweep "
        f"(MI350X, {NUM_GPUS} GPUs, {dtype})",
        f"  {active_ranks} active ranks/group; times = best-of-N (min), "
        "barrier-aligned + hipEvent-timed",
        f"  SINGLE GROUP = one {active_ranks}-rank relay on a full "
        f"{NUM_GPUS}-rank comm, no co-resident jobs (best case)",
        f"  NCCL baseline = one {active_ranks}-rank NCCL collective, "
        f"other {NUM_GPUS - active_ranks} ranks idle",
        line,
        f"{'':>10} | {'SINGLE GROUP (1 job)':^33}",
        f"{'Msg Size':>10} | {'NCCL(ms)':>10} {'Relay(ms)':>10} {'Speedup':>9}",
        "-" * width,
    ]
    for i, (label, nbytes) in enumerate(_MSG_SWEEP_SIZES):
        elements = nbytes // dtype.itemsize
        if elements % active_ranks != 0:
            continue
        n = _sweep_get_ms(results_dict, f"nccl_parallel_{key}_{i}")
        r = _sweep_get_ms(results_dict, f"relay_parallel_{key}_{i}")
        out.append(
            f"{label:>10} | {_sweep_fmt_ms(n):>10} {_sweep_fmt_ms(r):>10} "
            f"{_sweep_fmt_speedup(n, r):>9}"
        )
    out.append(line)
    return out


def _print_single_group_msg_sweep_report(results_dict: Any, dtype: torch.dtype) -> None:
    """Emit one single-group best-case table per (collective, active-rank-count)
    as a single clean, diffable block (file + atomic stdout write)."""
    lines: list[str] = []
    for collective in _MSG_SWEEP_COLLECTIVES:
        for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
            lines.extend(
                _format_single_group_sweep_table(
                    results_dict, dtype, collective, active_ranks
                )
            )
    _emit_report(lines, "bench_sharded_relay_single_group_sweep_results.txt")


# ---------------------------------------------------------------------------
# TestCase — works with both "buck2 test" and "buck2 run" (same as the old
# test_sharded_relay_2d_integration.py pattern).
# ---------------------------------------------------------------------------


class BenchShardedRelayPerfTest(unittest.TestCase):
    """
    Runs the sharded relay benchmark via mp.spawn inside a single TestCase.

    Bench A (NCCL allreduce_coalesced) runs in its own mp.spawn with its own
    dist.init/destroy cycle so that its NCCL sub-group communicators are fully
    isolated from bench B/C's RCCLX communicators.  Results are passed to the
    bench B/C worker via mp.Manager dict for a unified printout.

    Both of these work:
        buck2 test @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
            //torchrec/distributed/tests:bench_sharded_relay_perf

        buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
            //torchrec/distributed/tests:bench_sharded_relay_perf
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        if torch.cuda.device_count() < NUM_GPUS:
            self.skipTest(
                f"Benchmark requires {NUM_GPUS} GPUs, "
                f"found {torch.cuda.device_count()}"
            )
        if not (FUSED_AVAILABLE and RCCLX_AVAILABLE):
            self.skipTest("FusedShardedRelayMultiGroup or RCCLX not available")

    def test_benchmark(self) -> None:
        manager = mp.Manager()
        results: Any = manager.dict()

        # Sweep both 2-active and 4-active sharded relay groups, printing a full
        # report for each. The 4-active sweep covers all four collectives
        # (allreduce, reduce-scatter, all-to-all, all-gather). Each iteration
        # uses a distinct port_offset so the NCCL/TCPStore endpoints don't
        # collide across iterations.
        for i, sharding_group_size in enumerate((2, 4)):
            port_offset = i * 100

            # Phase 1: bench A — NCCL allreduce_coalesced on sub-group PGs.
            mp.spawn(
                _bench_a_worker,
                args=(NUM_GPUS, results, sharding_group_size, port_offset),
                nprocs=NUM_GPUS,
                join=True,
            )

            # Phase 2: bench B (fused flat) and C (kernel direct) via RCCLX.
            mp.spawn(
                _benchmark_worker,
                args=(NUM_GPUS, results, sharding_group_size, port_offset),
                nprocs=NUM_GPUS,
                join=True,
            )

        manager.shutdown()

    def test_collectives_msg_size_sweep(self) -> None:
        """All-collective, 2- & 4-rank sharded relay message-size sweep (bf16).

        Sweeps the fixed message sizes in _MSG_SWEEP_SIZES for every collective
        (allreduce, reduce-scatter, all-to-all, all-gather) at both 2 and 4
        active ranks and, per size, reports the sharded-relay FUSED speedup over
        NCCL (one table per (collective, active-rank-count)):
          FUSED — (NUM_GPUS // A) concurrent A-rank groups in one multi-group
                  kernel call vs the matching NCCL baseline on each rank's A-rank
                  sub-group.
        The single-group scenario is benchmarked separately (see
        test_parallel_collectives_msg_size_sweep).

        Run selectively (so the production test_benchmark does not also run).
        The buck2 test runner imports the module, so the selector must be the
        fully-qualified module.Class.method path:
            buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
                //torchrec/distributed/tests:bench_sharded_relay_perf -- \\
                torchrec.distributed.tests.bench_sharded_relay_perf.BenchShardedRelayPerfTest.test_collectives_msg_size_sweep
        """
        manager = mp.Manager()
        results: Any = manager.dict()

        # Pin the GPU HW-queue count to the production regime: BM-FM runs with
        # GPU_MAX_HW_QUEUES=2. Set before mp.spawn so the spawned workers inherit
        # it before the HIP runtime initializes. Fused is queue-insensitive (one
        # comm, one multi-group launch), so this only aligns it with the parallel
        # sweep, which needs =2 to avoid uncoordinated multi-comm XGMI contention.
        os.environ["GPU_MAX_HW_QUEUES"] = "2"

        # Phase 1: NCCL FUSED baselines per collective and A.
        mp.spawn(
            _msg_sweep_nccl_worker,
            args=(NUM_GPUS, results, 0),
            nprocs=NUM_GPUS,
            join=True,
        )

        # Phase 2: sharded relay FUSED per collective and A via RCCLX; the
        # parent emits all tables after the workers join (clean, un-interleaved).
        mp.spawn(
            _msg_sweep_relay_worker,
            args=(NUM_GPUS, results, 0),
            nprocs=NUM_GPUS,
            join=True,
        )

        _print_msg_sweep_report(results, _MSG_SWEEP_DTYPE)

        manager.shutdown()

    def test_parallel_collectives_msg_size_sweep(self) -> None:
        """N independent single-group A-rank relay JOBS in parallel (bf16).

        Production-faithful reconciliation: models N = NUM_GPUS // A independent
        relay workloads as SEPARATE co-resident jobs on one 8-GPU node (not one
        shared process issuing N calls). For each A, one mp.spawn launches
        N * NUM_GPUS processes — process p is (job = p // NUM_GPUS,
        rank_in_job = p % NUM_GPUS) on cuda:rank_in_job — so the N jobs are
        co-resident (N processes per GPU). Each process owns exactly ONE 8-rank
        RCCLX communicator for its job and issues exactly ONE single-group
        A-rank relay call per iteration, with its own CUDA context and
        GPU_MAX_HW_QUEUES=2 budget. A gloo PG spanning all N * NUM_GPUS
        processes overlaps every job's timed region; the reported time is the max
        across jobs (overlapped wall-time).

        The NCCL baseline is N disjoint A-rank NCCL collectives run as separate
        processes (A ranks/job → NUM_GPUS processes total, one per GPU),
        overlapped under the same barrier, max across jobs. The relay's 8-rank
        comm vs NCCL's A-rank comm asymmetry is exactly the production contrast.
        Prints one table per (collective, active-rank-count).

        Run selectively (so the production test_benchmark does not also run).
        The buck2 test runner imports the module, so the selector must be the
        fully-qualified module.Class.method path:
            buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
                //torchrec/distributed/tests:bench_sharded_relay_perf -- \\
                torchrec.distributed.tests.bench_sharded_relay_perf.BenchShardedRelayPerfTest.test_parallel_collectives_msg_size_sweep
        """
        manager = mp.Manager()
        results: Any = manager.dict()

        # Production regime: BM-FM runs with GPU_MAX_HW_QUEUES=2. Each spawned
        # process now inherits its OWN budget (N separate co-resident jobs), so
        # this reproduces the per-process queue budget of production rather than
        # one shared budget split across N calls. Set before mp.spawn so the
        # workers inherit it before HIP init.
        os.environ["GPU_MAX_HW_QUEUES"] = "2"

        # Each process builds several co-resident NCCL groups (a per-job device
        # group + the baseline's A-rank sub-group). The NCCL flight recorder /
        # heartbeat monitor names its dump pipe /tmp/nccl_trace_<sec>_rank_<r>.pipe
        # by (second, rank), so groups created in the same second with the same
        # rank number collide ("File exists" -> SIGABRT). Disable the monitor /
        # trace buffer for the benchmark (a debugging aid, not needed here).
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "0"

        # Both baselines use the SAME separate-job topology (N * NUM_GPUS procs
        # per A, N co-resident jobs each a full 8-rank world) and the SAME
        # alignment (global gloo cross-job barrier for deterministic overlap +
        # per-job NCCL device barrier for on-device within-job alignment), so
        # relay vs NCCL isolates the collective algorithm rather than the
        # deployment topology or the measurement harness. Per A, run the NCCL
        # baseline then the relay.
        for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
            num_jobs = NUM_GPUS // active_ranks
            total_procs = num_jobs * NUM_GPUS
            mp.spawn(
                _parallel_msg_sweep_nccl_worker,
                args=(total_procs, active_ranks, results, _find_free_port()),
                nprocs=total_procs,
                join=True,
            )
            mp.spawn(
                _parallel_msg_sweep_relay_worker,
                args=(total_procs, active_ranks, results, _find_free_port()),
                nprocs=total_procs,
                join=True,
            )

        # Reduce per-job entries to max-across-jobs, then print the tables.
        _reduce_parallel_jobs_max(results)
        _print_parallel_msg_sweep_report(results, _MSG_SWEEP_DTYPE)

        manager.shutdown()

    def test_single_group_collectives_msg_size_sweep(self) -> None:
        """Best-case SINGLE-GROUP A-rank relay sweep (bf16): exactly ONE group.

        Uses the same workers and harness as
        test_parallel_collectives_msg_size_sweep, but forces num_jobs == 1 for
        every A (total_procs = NUM_GPUS), so only ONE A-rank relay group runs on
        the 8-GPU host with no co-resident job competing for XGMI bandwidth or HW
        queues. This is the parallel sweep's num_jobs=1 point — the
        contention-free upper bound on both relay and NCCL. The relay still uses
        the full 8-rank comm (A active ranks [0..A-1] + 8-A helpers); the NCCL
        baseline runs the A-rank collective with the remaining ranks idle. Same
        alignment (global gloo barrier + per-job NCCL device barrier) as the
        parallel sweep. Prints one table per (collective, active-rank-count).

        Run selectively (so the production test_benchmark does not also run).
        The buck2 test runner imports the module, so the selector must be the
        fully-qualified module.Class.method path:
            buck2 run @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
                //torchrec/distributed/tests:bench_sharded_relay_perf -- \\
                torchrec.distributed.tests.bench_sharded_relay_perf.BenchShardedRelayPerfTest.test_single_group_collectives_msg_size_sweep
        """
        manager = mp.Manager()
        results: Any = manager.dict()

        # Same production regime + NCCL flight-recorder disable as the parallel
        # sweep (see test_parallel_collectives_msg_size_sweep for the rationale).
        os.environ["GPU_MAX_HW_QUEUES"] = "2"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "0"

        # One job only: total_procs = NUM_GPUS -> num_jobs = 1 in both workers,
        # so the single A-rank group [0..A-1] is the only workload on the host
        # (the other 8-A ranks are helpers/idle). Per A, run NCCL then relay.
        for active_ranks in _MSG_SWEEP_ACTIVE_RANKS:
            total_procs = NUM_GPUS
            mp.spawn(
                _parallel_msg_sweep_nccl_worker,
                args=(total_procs, active_ranks, results, _find_free_port()),
                nprocs=total_procs,
                join=True,
            )
            mp.spawn(
                _parallel_msg_sweep_relay_worker,
                args=(total_procs, active_ranks, results, _find_free_port()),
                nprocs=total_procs,
                join=True,
            )

        # Collapse the single job's per-job entries to the base keys, then print.
        _reduce_parallel_jobs_max(results)
        _print_single_group_msg_sweep_report(results, _MSG_SWEEP_DTYPE)

        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
