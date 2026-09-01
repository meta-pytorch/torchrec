#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import triton  # @manual
import triton.language as tl  # @manual
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

OPTIM_TYPE_TO_INT: dict[OptimType, int] = {
    OptimType.EXACT_SGD: 0,
    OptimType.EXACT_ROWWISE_ADAGRAD: 1,
}


@triton.jit
def _stochastic_rounding_store(
    ptr,
    val,
    mask,
    seed,
    offset,
):
    """
    Store FP32 values to FP16 using stochastic rounding.

    Probabilistically rounds each element to one of the two nearest FP16
    values with probability proportional to proximity:
        P(round to ceil) = (val - floor_fp16) / ULP

    This gives unbiased rounding, critical for convergence with FP16 weights.

    Unlike the CUDA TBE approach (noise + __float2half_rz truncation),
    we use an explicit probability-based method because Triton lacks a
    round-toward-zero FP16 conversion intrinsic.
    """
    val_fp16 = val.to(tl.float16)
    val_rne = val_fp16.to(tl.float32)

    error = val - val_rne

    val_bits = val_fp16.to(tl.int16, bitcast=True)
    is_neg = val_bits < 0

    adjacent_bits = tl.where(
        error > 0,
        tl.where(is_neg, val_bits - 1, val_bits + 1),
        tl.where(is_neg, val_bits + 1, val_bits - 1),
    ).to(tl.int16)
    adjacent = adjacent_bits.to(tl.float16, bitcast=True)
    adjacent_f32 = adjacent.to(tl.float32)

    ulp = tl.abs(adjacent_f32 - val_rne)
    abs_error = tl.abs(error)

    rand = tl.rand(seed, offset)
    result = tl.where(rand * ulp < abs_error, adjacent, val_fp16)

    tl.store(ptr, result, mask=mask)


_LONG_RUN_THRESHOLD: int = 256


@triton.jit
def _classify_runs_kernel(
    cum_lengths_ptr,
    num_runs_ptr,
    short_run_ids_ptr,
    num_short_ptr,
    long_run_ids_ptr,
    num_long_ptr,
    infos_sorted_ptr,
    feature_bucket_id_ptr,
    bucket_base_ptr,
    info_B_num_bits,
    threshold: tl.constexpr,
    CLASSIFY_BLOCK: tl.constexpr,
    NUM_BUCKETS: tl.constexpr = 1,
) -> None:
    """
    Classify runs as short or long and compact into output arrays.
    Uses atomic counters for sync-free stream compaction.

    With NUM_BUCKETS > 1 the short runs are additionally split by the embedding
    dimension bucket of their feature, so each bucket can be launched with a
    BLOCK_SIZE that matches its rows instead of the global max. Bucket b's ids
    live at short_run_ids[bucket_base[b] : bucket_base[b] + num_short[b]].
    """
    pid = tl.program_id(0)
    num_runs = tl.load(num_runs_ptr)

    offsets = pid * CLASSIFY_BLOCK + tl.arange(0, CLASSIFY_BLOCK)
    mask = offsets < num_runs

    cum_start = tl.load(cum_lengths_ptr + offsets, mask=mask, other=0)
    cum_end = tl.load(cum_lengths_ptr + offsets + 1, mask=mask, other=0)
    run_len = cum_end - cum_start

    is_long = (run_len >= threshold) & mask
    is_short = (~is_long) & mask

    num_long_block = tl.sum(is_long.to(tl.int32))
    long_base = tl.atomic_add(num_long_ptr, num_long_block)
    long_local = tl.cumsum(is_long.to(tl.int32), axis=0) - 1
    long_pos = (long_base + long_local).to(tl.int64)
    tl.store(long_run_ids_ptr + long_pos, offsets.to(tl.int32), mask=is_long)

    if NUM_BUCKETS == 1:
        num_short_block = tl.sum(is_short.to(tl.int32))
        short_base = tl.atomic_add(num_short_ptr, num_short_block)
        short_local = tl.cumsum(is_short.to(tl.int32), axis=0) - 1
        short_pos = (short_base + short_local).to(tl.int64)
        tl.store(short_run_ids_ptr + short_pos, offsets.to(tl.int32), mask=is_short)
    else:
        # A run maps to exactly one feature, so one info load per run resolves
        # its dimension bucket.
        info = tl.load(infos_sorted_ptr + cum_start, mask=mask, other=0).to(tl.uint32)
        t = (info >> info_B_num_bits).to(tl.int32)
        bucket_id = tl.load(feature_bucket_id_ptr + t, mask=mask, other=0)
        for b in tl.static_range(NUM_BUCKETS):
            sel = is_short & (bucket_id == b)
            cnt = tl.sum(sel.to(tl.int32))
            base = tl.atomic_add(num_short_ptr + b, cnt)
            local = tl.cumsum(sel.to(tl.int32), axis=0) - 1
            pos = (tl.load(bucket_base_ptr + b) + base + local).to(tl.int64)
            tl.store(short_run_ids_ptr + pos, offsets.to(tl.int32), mask=sel)


@triton.jit
def _expand_long_runs_kernel(
    long_run_ids_ptr,
    cum_lengths_ptr,
    seg_starts_out_ptr,
    seg_ends_out_ptr,
    grad_buffer_ids_out_ptr,
    programs_per_long_run_out_ptr,
    num_programs_out_ptr,
    num_long_ptr,
    threshold: tl.constexpr,
) -> None:
    """
    Expand each long run into sub-programs with segment boundaries.
    One program per long run; uses atomic counter for output positions.
    """
    pid = tl.program_id(0)
    num_long = tl.load(num_long_ptr)
    if pid >= num_long:
        return

    run_id = tl.load(long_run_ids_ptr + pid)
    seg_start_orig = tl.load(cum_lengths_ptr + run_id)
    seg_end_orig = tl.load(cum_lengths_ptr + run_id + 1)
    run_len = seg_end_orig - seg_start_orig
    num_sub = (run_len + threshold - 1) // threshold

    base = tl.atomic_add(num_programs_out_ptr, num_sub.to(tl.int32))
    tl.store(programs_per_long_run_out_ptr + pid, num_sub.to(tl.int32))

    for j in range(num_sub):
        prog_idx = base + j
        start_val = seg_start_orig + j * threshold
        end_val = tl.minimum(
            seg_start_orig + (j + 1) * threshold,
            seg_end_orig,
        )
        tl.store(seg_starts_out_ptr + prog_idx, start_val.to(tl.int32))
        tl.store(seg_ends_out_ptr + prog_idx, end_val.to(tl.int32))
        tl.store(grad_buffer_ids_out_ptr + prog_idx, pid.to(tl.int32))


def _expand_long_runs(
    sorted_linear_indices_cumulative_run_lengths: torch.Tensor,
    sorted_linear_indices_num_runs: torch.Tensor,
    max_num_runs: int,
    max_sl_per_program: int = _LONG_RUN_THRESHOLD,
    infos_sorted: Optional[torch.Tensor] = None,
    feature_bucket_id: Optional[torch.Tensor] = None,
    bucket_base: Optional[torch.Tensor] = None,
    short_run_capacity: Optional[int] = None,
    num_buckets: int = 1,
    info_B_num_bits: int = 0,
) -> Tuple[
    torch.Tensor,  # short_run_ids
    torch.Tensor,  # num_short_runs (num_buckets-element GPU int64 tensor)
    torch.Tensor,  # (unused)
    torch.Tensor,  # long_run_program_seg_starts
    torch.Tensor,  # long_run_program_seg_ends
    torch.Tensor,  # num_long_run_programs (1-element GPU int32 tensor)
    torch.Tensor,  # num_long_runs (1-element GPU int32 tensor)
    torch.Tensor,  # long_run_grad_buffer_ids
    torch.Tensor,  # long_run_original_ids
    torch.Tensor,  # programs_per_long_run
]:
    """
    Split runs into short runs and long runs for 2-tier backward dispatch.
    Fully GPU-resident — no .item() calls or CPU-GPU sync points.
    Uses Triton kernels with atomic counters for stream compaction
    and direct expansion (no sort, no searchsorted).
    """
    device = sorted_linear_indices_cumulative_run_lengths.device
    max_long_runs = max_num_runs // max_sl_per_program + 1
    max_long_run_programs = 2 * max_num_runs // max_sl_per_program + 1

    # Allocate output tensors
    short_run_ids = torch.empty(
        short_run_capacity if short_run_capacity is not None else max_num_runs,
        dtype=torch.int32,
        device=device,
    )
    num_short_runs_t = torch.zeros(num_buckets, dtype=torch.int64, device=device)

    long_run_original_ids = torch.empty(max_long_runs, dtype=torch.int32, device=device)
    num_long_runs_t = torch.zeros(1, dtype=torch.int64, device=device)

    # Kernel 1: classify and compact runs
    CLASSIFY_BLOCK = 1024
    classify_grid = (max_num_runs + CLASSIFY_BLOCK - 1) // CLASSIFY_BLOCK
    _classify_runs_kernel[(classify_grid,)](
        sorted_linear_indices_cumulative_run_lengths,
        sorted_linear_indices_num_runs,
        short_run_ids,
        num_short_runs_t,
        long_run_original_ids,
        num_long_runs_t,
        infos_sorted,
        feature_bucket_id,
        bucket_base,
        info_B_num_bits,
        threshold=max_sl_per_program,
        CLASSIFY_BLOCK=CLASSIFY_BLOCK,
        NUM_BUCKETS=num_buckets,
    )

    # Allocate expansion output tensors
    long_run_program_seg_starts = torch.empty(
        max_long_run_programs, dtype=torch.int32, device=device
    )
    long_run_program_seg_ends = torch.empty(
        max_long_run_programs, dtype=torch.int32, device=device
    )
    long_run_grad_buffer_ids = torch.empty(
        max_long_run_programs, dtype=torch.int32, device=device
    )
    programs_per_long_run = torch.zeros(max_long_runs, dtype=torch.int32, device=device)
    num_long_run_programs_t = torch.zeros(1, dtype=torch.int64, device=device)

    # Kernel 2: expand long runs into sub-programs
    _expand_long_runs_kernel[(max_long_runs,)](
        long_run_original_ids,
        sorted_linear_indices_cumulative_run_lengths,
        long_run_program_seg_starts,
        long_run_program_seg_ends,
        long_run_grad_buffer_ids,
        programs_per_long_run,
        num_long_run_programs_t,
        num_long_runs_t,
        threshold=max_sl_per_program,
    )

    return (
        short_run_ids,
        num_short_runs_t,
        torch.empty(0, dtype=torch.int32, device=device),
        long_run_program_seg_starts,
        long_run_program_seg_ends,
        num_long_run_programs_t,
        num_long_runs_t,
        long_run_grad_buffer_ids,
        long_run_original_ids,
        programs_per_long_run.to(torch.int32),
    )
