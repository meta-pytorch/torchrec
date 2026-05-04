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

The production per-group totals above are the defaults.  Run with no extra flags:
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
    BENCH_TABLE_SIZE     int              (default: production total for active group)
    BENCH_KERNEL_SIZE_GB float            (default: production total for active group)
    BENCH_WARMUP_ITERS   int              (default: 5)
    BENCH_BENCH_ITERS    int              (default: 20)
    BENCH_LOG_SIZES      1                (print sizes and exit; for calibration)
"""

from __future__ import annotations

import os
import time
import unittest
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


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


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


def _make_fused(rcclx_comm: Any, local_rank: int, local_size: int) -> Any | None:
    if FusedShardedRelayMultiGroup is None or rcclx_comm is None:
        return None
    sparse_group_size = 2
    num_sparse_groups = local_size // sparse_group_size
    all_active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
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


def _measure_ms(fn: Any, warmup: int, iters: int) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over 'iters' calls after 'warmup' warmup runs."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# Per-rank benchmark worker — mimics _benchmark_worker from the old
# test_sharded_relay_2d_integration.py pattern:
#   - receives an explicit TCPStore instead of using _get_default_store()
#   - sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT itself
# ---------------------------------------------------------------------------

NUM_GPUS: int = 8
_NCCL_PORT: int = 29500
_TCPSTORE_PORT: int = 29502


def _bench_a_worker(rank: int, world_size: int, results_dict: Any) -> None:
    """Worker for bench A: serial all_reduce on isolated 2-rank sub-group PGs.

    Runs in its own mp.spawn with its own dist.init_process_group/destroy cycle
    (on different ports) so that NCCL sub-group communicators are fully isolated
    from bench B/C's RCCLX communicators.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT + 10)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT + 10,
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
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 5))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 20))

    sparse_group_size = 2
    num_sparse_groups = world_size // sparse_group_size
    my_sparse_group = local_rank // sparse_group_size
    all_active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]

    prod_totals = _PROD_TOTALS_FP16 if dtype != torch.float32 else _PROD_TOTALS_FP32

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

    mean_serial, std_serial = _measure_ms(run_coalesced, warmup, bench_iters)
    _store_barrier()

    if rank == 0:
        results_dict["A"] = (mean_serial, std_serial)

    dist.destroy_process_group()


def _benchmark_worker(rank: int, world_size: int, results_dict: Any) -> None:
    """
    Worker for bench B (fused flat) and C (kernel direct).

    Uses an explicit TCPStore (same pattern as the deleted
    test_sharded_relay_2d_integration.py) so that RCCLX comm creation does
    not depend on dist._get_default_store(), which can hang when called from
    spawned child processes in the Meta environment.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_NCCL_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    local_rank = rank
    is_master = rank == 0

    # Explicit TCPStore — same as the integration test pattern.
    store = dist.TCPStore(
        host_name="localhost",
        port=_TCPSTORE_PORT,
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
    warmup = max(1, _env_int("BENCH_WARMUP_ITERS", 5))
    bench_iters = max(1, _env_int("BENCH_BENCH_ITERS", 20))
    log_sizes = _env_str("BENCH_LOG_SIZES", "0") == "1"

    sparse_group_size = 2
    num_sparse_groups = world_size // sparse_group_size
    my_sparse_group = local_rank // sparse_group_size
    all_active_ranks = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]

    # Production per-group totals: groups have heterogeneous sizes matching BM-FM.
    prod_totals = _PROD_TOTALS_FP16 if dtype != torch.float32 else _PROD_TOTALS_FP32

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
    fused = _make_fused(rcclx_comm, local_rank, world_size)
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
        [kernel_elements] * num_sparse_groups if kernel_gb > 0 else list(prod_totals)
    )
    kernel_tensor = torch.ones(kernel_elements, dtype=dtype, device=device)
    # For kernel bench, each helper group gets its own passthrough-sized buffer.
    kernel_scratch: list[torch.Tensor] = []
    for g in range(num_sparse_groups):
        if g == my_sparse_group:
            kernel_scratch.append(kernel_tensor)
        else:
            declared_g = kernel_declared_sizes[g]
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

    mean_fused, std_fused = _measure_ms(run_fused, warmup, bench_iters)

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

    mean_kernel, std_kernel = _measure_ms(run_kernel, warmup, bench_iters)

    # Measure peak HBM usage — the whole point of this work.
    torch.cuda.reset_peak_memory_stats()
    run_fused()
    torch.cuda.synchronize()
    peak_hbm_bytes = torch.cuda.max_memory_allocated()

    if rank == 0:
        total_bytes = sum(sz * dtype.itemsize for sz in sizes)
        kernel_bytes = kernel_elements * dtype.itemsize

        def bw(nbytes: int, ms: float) -> float:
            return 2 * nbytes / (ms / 1000) / 1e9

        bench_a_mean = float(results_dict["A"][0]) if "A" in results_dict else 0.0
        bench_a_std = float(results_dict["A"][1]) if "A" in results_dict else 0.0

        print("\n" + "=" * 72)
        print(f"Sharded Relay Allreduce Benchmark — MI350X ({world_size} GPUs)")
        print("=" * 72)
        print(f"  dtype:          {dtype}")
        print(f"  num_tables:     {num_tables}")
        print(
            f"  avg table size: {table_size:,} elements "
            f"({table_size * dtype.itemsize / 1024 / 1024:.1f} MB)"
        )
        print(f"  total data/grp: {total_bytes / 1024 / 1024:.1f} MB")
        print()
        if bench_a_mean > 0:
            print(f"  [A] COALESCED ({num_tables} tensors, allreduce_coalesced):")
            print(
                f"       {bench_a_mean:.2f} ms  ±  {bench_a_std:.2f} ms  |  "
                f"{bw(total_bytes, bench_a_mean):.1f} GB/s"
            )
        else:
            print("  [A] COALESCED: N/A")
        print()
        print("  [B] FUSED FLAT (1 call, passthrough helpers, phase-sync kernel):")
        print(
            f"       {mean_fused:.2f} ms  ±  {std_fused:.2f} ms  |  "
            f"{bw(total_bytes, mean_fused):.1f} GB/s"
        )
        print(f"       Peak HBM: {peak_hbm_bytes / 1024 / 1024 / 1024:.2f} GiB")
        print()
        print(
            f"  [C] KERNEL DIRECT (1 large tensor, {kernel_bytes / 1024 / 1024:.0f} MB):"
        )
        print(
            f"       {mean_kernel:.2f} ms  ±  {std_kernel:.2f} ms  |  "
            f"{bw(kernel_bytes, mean_kernel):.1f} GB/s"
        )
        print()
        if bench_a_mean > 0:
            speedup_ab = bench_a_mean / mean_fused if mean_fused > 0 else float("inf")
            print(f"  A→B speedup (coalesced → fused flat):  {speedup_ab:.2f}x")
        print("=" * 72)

    dist.barrier()
    dist.destroy_process_group()


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

        # Phase 1: bench A — allreduce_coalesced on 2-rank NCCL sub-group PGs.
        mp.spawn(
            _bench_a_worker,
            args=(NUM_GPUS, results),
            nprocs=NUM_GPUS,
            join=True,
        )

        # Phase 2: bench B (fused flat) and C (kernel direct) via RCCLX.
        mp.spawn(
            _benchmark_worker,
            args=(NUM_GPUS, results),
            nprocs=NUM_GPUS,
            join=True,
        )

        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
