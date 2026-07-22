#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton  # @manual
import triton.language as tl  # @manual
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

OPTIM_TYPE_TO_INT: dict[OptimType, int] = {
    OptimType.EXACT_SGD: 0,
    OptimType.EXACT_ROWWISE_ADAGRAD: 1,
}
from ads_mkl.ops.triton.amd.triton_table_batched_embeddings import (
    _FIXED_GRID as _AMD_FIXED_GRID,
)

_FIXED_GRID = 24576
_CLC_FIXED_GRID = 32 * 24576


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
    threshold: tl.constexpr,
    CLASSIFY_BLOCK: tl.constexpr,
) -> None:
    """
    Classify runs as short or long and compact into output arrays.
    Uses atomic counters for sync-free stream compaction.
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

    num_short_block = tl.sum(is_short.to(tl.int32))
    num_long_block = tl.sum(is_long.to(tl.int32))

    short_base = tl.atomic_add(num_short_ptr, num_short_block)
    long_base = tl.atomic_add(num_long_ptr, num_long_block)

    short_local = tl.cumsum(is_short.to(tl.int32), axis=0) - 1
    long_local = tl.cumsum(is_long.to(tl.int32), axis=0) - 1

    short_pos = (short_base + short_local).to(tl.int64)
    tl.store(short_run_ids_ptr + short_pos, offsets.to(tl.int32), mask=is_short)

    long_pos = (long_base + long_local).to(tl.int64)
    tl.store(long_run_ids_ptr + long_pos, offsets.to(tl.int32), mask=is_long)


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
) -> Tuple[
    torch.Tensor,  # short_run_ids
    torch.Tensor,  # num_short_runs (1-element GPU int32 tensor)
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
    short_run_ids = torch.empty(max_num_runs, dtype=torch.int32, device=device)
    num_short_runs_t = torch.zeros(1, dtype=torch.int64, device=device)

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
        threshold=max_sl_per_program,
        CLASSIFY_BLOCK=CLASSIFY_BLOCK,
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


def get_grid_size(
    is_amd: bool,
    max_num_runs: int,
    max_long_runs: int,
    max_long_run_programs: int,
    use_clc: bool,
) -> Tuple[int, int, int]:
    """
    Get grid size for Triton TBE kernels.
    """
    if is_amd:
        return (
            min(_AMD_FIXED_GRID, max_num_runs),
            min(_AMD_FIXED_GRID, max_long_run_programs),
            min(_AMD_FIXED_GRID, max_long_run_programs),
        )

    if use_clc:
        return (
            min(_CLC_FIXED_GRID, max_num_runs),
            min(_CLC_FIXED_GRID, max_long_run_programs),
            max_long_runs,  # Will not be used.
        )

    return (
        min(_FIXED_GRID, max_num_runs),
        min(_FIXED_GRID, max_long_run_programs),
        max_long_runs,
    )
