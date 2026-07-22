#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
triton table batched embedding bag with sum reduction mode
"""

import logging
import math
import os
from itertools import accumulate
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton  # @manual
import triton.language as tl  # @manual
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (  # noqa: F401
    generate_vbe_metadata,
)
from triton.language.core import constexpr  # @manual

has_tlx = True
try:
    import triton.language.extra.tlx as tlx

except ImportError:
    has_tlx = False

from torchrec.distributed.triton_tbe.triton_tbe_backward_long_run_fused import (
    triton_tbe_backward_long_run_fused_unweighted,
    triton_tbe_backward_long_run_fused_weighted,
)
from torchrec.distributed.triton_tbe.triton_tbe_backward_utils import (
    _expand_long_runs,
    _LONG_RUN_THRESHOLD,
    _stochastic_rounding_store,
    get_grid_size,
    OPTIM_TYPE_TO_INT,
)


def is_amd() -> bool:
    return torch.version.hip is not None


# AMD-compatible kernel variants (no CLC, FP32 accum, portable tl.range)
from ads_mkl.ops.triton.amd.triton_table_batched_embeddings import (  # noqa: F811
    _expand_long_runs as _amd_expand_long_runs,
    _FIXED_GRID as _AMD_FIXED_GRID,
    _nbit_TBE_forward_kernel_16bits as _amd_nbit_fwd_16bits,
    _nbit_TBE_forward_kernel_32bits as _amd_nbit_fwd_32bits,
    triton_tbe_backward_long_run_apply_unweighted as _amd_bwd_long_apply,
    triton_tbe_backward_long_run_grad_accum_unweighted as _amd_bwd_long_accum_unweighted,
    triton_tbe_backward_long_run_grad_accum_weighted as _amd_bwd_long_accum_weighted,
    triton_tbe_backward_short_run_unweighted as _amd_bwd_short_unweighted,
    triton_tbe_backward_short_run_weighted as _amd_bwd_short_weighted,
    triton_tbe_forward_unweighted_kernel as _amd_fwd_unweighted_kernel,
    triton_tbe_forward_weighted_kernel as _amd_fwd_weighted_kernel,
)


def lengths_to_offsets(lengths: List[int], keep_last: bool = False) -> List[int]:
    assert len(lengths) > 0
    offsets = [0] + list(accumulate(lengths))
    if not keep_last:
        offsets.pop()
    return offsets


@triton.jit
def table_batched_embedding_bag_forward_weighted_kernel(
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    per_sample_weights_ptr,
    # VBE-specific pointers (only used when vbe=T)
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    b_t_map_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    BLOCK_SIZE: tl.constexpr,
    vbe: tl.constexpr = False,
    info_B_num_bits=0,
    info_B_mask=0,
) -> None:

    b_t = tl.program_id(0).to(tl.int64)

    if vbe:
        info = tl.load(b_t_map_ptr + b_t).to(tl.uint32)
        t = (info >> info_B_num_bits).to(tl.int32)
        b = (info & info_B_mask).to(tl.int32)
    else:
        t = b_t // B  # feature id
        b = b_t % B  # batch id

    # Map feature index to table index for weight lookup
    table_idx = tl.load(feature_table_map_ptr + t)
    table_offset = tl.load(table_offsets_ptr + table_idx)
    # embedding_dim and embedding_offset are indexed by feature
    embedding_dim = tl.load(embedding_dims_ptr + t)
    embedding_offset = tl.load(embedding_offsets_ptr + t)

    start = tl.load(offsets_ptr + b_t)
    end = tl.load(offsets_ptr + b_t + 1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < embedding_dim
    bag_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

    # without type hint the unrolling performance will downgrade
    step: tl.constexpr = 4
    ns = (end - start) // step
    endn = start + step * ns

    for idx in range(start, endn, step):
        row_idx_0 = tl.load(indices_ptr + idx + 0)
        row_idx_1 = tl.load(indices_ptr + idx + 1)
        row_idx_2 = tl.load(indices_ptr + idx + 2)
        row_idx_3 = tl.load(indices_ptr + idx + 3)

        row_start_ptr_0 = weight_ptr + table_offset + row_idx_0 * embedding_dim
        row_start_ptr_1 = weight_ptr + table_offset + row_idx_1 * embedding_dim
        row_start_ptr_2 = weight_ptr + table_offset + row_idx_2 * embedding_dim
        row_start_ptr_3 = weight_ptr + table_offset + row_idx_3 * embedding_dim

        row_ptrs_0 = row_start_ptr_0 + col_offsets
        row_ptrs_1 = row_start_ptr_1 + col_offsets
        row_ptrs_2 = row_start_ptr_2 + col_offsets
        row_ptrs_3 = row_start_ptr_3 + col_offsets

        row_0 = tl.load(row_ptrs_0, mask=mask, other=0)
        row_1 = tl.load(row_ptrs_1, mask=mask, other=0)
        row_2 = tl.load(row_ptrs_2, mask=mask, other=0)
        row_3 = tl.load(row_ptrs_3, mask=mask, other=0)

        idx_weight_0 = tl.load(per_sample_weights_ptr + idx + 0)
        idx_weight_1 = tl.load(per_sample_weights_ptr + idx + 1)
        idx_weight_2 = tl.load(per_sample_weights_ptr + idx + 2)
        idx_weight_3 = tl.load(per_sample_weights_ptr + idx + 3)

        # Explicitly convert to float32 before accumulating to ensure
        # consistent precision with SplitTBE CUDA kernel
        bag_output += (
            row_0.to(tl.float32) * idx_weight_0
            + row_1.to(tl.float32) * idx_weight_1
            + row_2.to(tl.float32) * idx_weight_2
            + row_3.to(tl.float32) * idx_weight_3
        )

    for idx in range(endn, end):
        row_idx = tl.load(indices_ptr + idx)
        row_start_ptr = weight_ptr + table_offset + row_idx * embedding_dim
        row_ptrs = row_start_ptr + col_offsets
        row = tl.load(row_ptrs, mask=mask, other=0)

        idx_weight = tl.load(per_sample_weights_ptr + idx)
        # Explicitly convert to float32 before accumulating
        bag_output += row.to(tl.float32) * idx_weight

    if vbe:
        row_output_offset = tl.load(row_output_offsets_ptr + b_t)
        output_row_start_ptr = output_ptr + row_output_offset
    else:
        output_row_start_ptr = output_ptr + b * total_embedding_dim + embedding_offset
    output_row_ptrs = output_row_start_ptr + col_offsets

    bag_output_original = bag_output.to(tl.float32)
    tl.store(output_row_ptrs, bag_output_original, mask=mask)


@triton.jit
def table_batched_embedding_bag_forward_unweighted_kernel(
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    # VBE-specific pointers (only used when vbe=True)
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    vbe: tl.constexpr = False,
) -> None:

    b = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    if vbe:
        output_row_base = output_ptr  # unused, VBE uses row_output_offsets
    else:
        output_row_base = output_ptr + b * total_embedding_dim

    for t in range(T):
        if vbe:
            # VBE: check if this batch index is within feature t's batch size
            B_start = tl.load(B_offsets_ptr + t).to(tl.int64)
            B_end = tl.load(B_offsets_ptr + t + 1).to(tl.int64)
            B_t = B_end - B_start
            b_t = B_start + b
            in_bounds = b < B_t
        else:
            b_t = t * B + b
            in_bounds = True

        if in_bounds:
            # Map feature index to table index for weight lookup
            table_idx = tl.load(feature_table_map_ptr + t)
            table_offset = tl.load(table_offsets_ptr + table_idx)
            # embedding_dim and embedding_offset are indexed by feature
            embedding_dim = tl.load(embedding_dims_ptr + t)

            start = tl.load(offsets_ptr + b_t)
            end = tl.load(offsets_ptr + b_t + 1)

            mask = col_offsets < embedding_dim
            bag_output = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)

            step: tl.constexpr = 4
            ns = (end - start) // step
            endn = start + step * ns

            for idx in range(start, endn, step):
                row_idx_0 = tl.load(indices_ptr + idx + 0)
                row_idx_1 = tl.load(indices_ptr + idx + 1)
                row_idx_2 = tl.load(indices_ptr + idx + 2)
                row_idx_3 = tl.load(indices_ptr + idx + 3)

                row_start_ptr_0 = weight_ptr + table_offset + row_idx_0 * embedding_dim
                row_start_ptr_1 = weight_ptr + table_offset + row_idx_1 * embedding_dim
                row_start_ptr_2 = weight_ptr + table_offset + row_idx_2 * embedding_dim
                row_start_ptr_3 = weight_ptr + table_offset + row_idx_3 * embedding_dim

                row_ptrs_0 = row_start_ptr_0 + col_offsets
                row_ptrs_1 = row_start_ptr_1 + col_offsets
                row_ptrs_2 = row_start_ptr_2 + col_offsets
                row_ptrs_3 = row_start_ptr_3 + col_offsets

                row_0 = tl.load(row_ptrs_0, mask=mask, other=0)
                row_1 = tl.load(row_ptrs_1, mask=mask, other=0)
                row_2 = tl.load(row_ptrs_2, mask=mask, other=0)
                row_3 = tl.load(row_ptrs_3, mask=mask, other=0)

                bag_output += (
                    row_0.to(tl.float32)
                    + row_1.to(tl.float32)
                    + row_2.to(tl.float32)
                    + row_3.to(tl.float32)
                )

            for idx in range(endn, end):
                row_idx = tl.load(indices_ptr + idx)
                row_start_ptr = weight_ptr + table_offset + row_idx * embedding_dim
                row_ptrs = row_start_ptr + col_offsets
                row = tl.load(row_ptrs, mask=mask, other=0)
                bag_output += row.to(tl.float32)

            if vbe:
                row_output_offset = tl.load(row_output_offsets_ptr + b_t)
                output_row_ptrs = output_ptr + row_output_offset + col_offsets
            else:
                embedding_offset = tl.load(embedding_offsets_ptr + t)
                output_row_ptrs = output_row_base + embedding_offset + col_offsets
            bag_output_original = bag_output.to(tl.float32)
            tl.store(output_row_ptrs, bag_output_original, mask=mask)


@triton.jit
def triton_tbe_backward_short_run_unweighted(
    dout_ptr,
    weight_ptr,
    infos_sorted_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    short_run_ids_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    sorted_linear_indices_num_runs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    USE_CLC: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
    vbe: tl.constexpr = False,
) -> None:
    """Backward kernel for short runs only. Each program handles one short run."""
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    if USE_CLC:
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        tile_id = tl.program_id(0)
        num_tiles = tl.num_programs(0)
        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)
        c_runs = num_short_runs // num_tiles
        r_runs = num_short_runs - c_runs * num_tiles
    else:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)

        c = num_short_runs // num_programs
        r = num_short_runs - c * num_programs

        local_id = c * pid + tl.maximum(pid - num_programs + r, 0)
        local_id_end = local_id + c + (pid >= num_programs - r)

        # Early exit for excess programs when grid > num_short_runs
        if local_id >= local_id_end:
            return

    has_more_tile = True
    while has_more_tile:
        if USE_CLC:
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1
            local_id = c_runs * tile_id + tl.maximum(tile_id - num_tiles + r_runs, 0)
            local_id_end = local_id + c_runs + (tile_id >= num_tiles - r_runs)

        while local_id < local_id_end:
            run_id = tl.load(short_run_ids_ptr + local_id)

            linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
            segment_start = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id
            )
            segment_end = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id + 1
            )

            info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
            t = (info_start >> info_B_num_bits).to(tl.int32)

            table_idx = tl.load(feature_table_map_ptr + t)
            table_offset = tl.load(table_offsets_ptr + table_idx)
            embedding_dim = tl.load(embedding_dims_ptr + t)
            mask = col_offsets < embedding_dim

            grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            sz = (segment_end - segment_start) // buffer_size
            endn = segment_start + sz * buffer_size

            for idx in tl.range(
                segment_start,
                endn,
                buffer_size,
                flatten=True,
                loop_unroll_factor=2,
            ):
                info = tl.load(infos_sorted_ptr + idx + buffer_offsets).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr[:, None] + col_offsets[None, :]
                grad_buffer = tl.load(dout_row_ptrs, mask=mask[None, :], other=0).to(
                    tl.float32
                )
                grad += tl.sum(grad_buffer, axis=0)

            for idx in range(endn, segment_end):
                info = tl.load(infos_sorted_ptr + idx).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr + col_offsets
                dout_row = tl.load(dout_row_ptrs, mask=mask, other=0).to(tl.float32)
                grad += dout_row

            grad_original = grad.to(tl.float32)

            index_offset = tl.load(hash_size_cumsum_ptr + t)
            row_idx = linear_index - index_offset
            row_start_ptr = weight_ptr + table_offset + row_idx * embedding_dim
            row_ptrs = row_start_ptr + col_offsets
            row = tl.load(row_ptrs, mask=col_offsets < embedding_dim, other=0).to(
                tl.float32
            )

            row_update = row - learning_rate * grad_original

            if optimizer == 1:
                row_offset = tl.load(rows_cumsum_ptr + table_idx)
                momentum_idx = row_offset + row_idx

                grad_square = grad_original * grad_original
                grad_square_average = tl.sum(grad_square) / embedding_dim

                momentum = tl.load(momentum_ptr + momentum_idx)
                momentum_new = momentum + grad_square_average
                tl.store(momentum_ptr + momentum_idx, momentum_new)

                adaptive_learning_rate = learning_rate / (tl.sqrt(momentum_new) + eps)

                row_update = row - adaptive_learning_rate * grad_original

            if STOCHASTIC_ROUNDING:
                sr_offset = local_id * BLOCK_SIZE + col_offsets
                _stochastic_rounding_store(
                    row_ptrs, row_update, mask, stochastic_rounding_seed, sr_offset
                )
            else:
                tl.store(row_ptrs, row_update, mask=mask)

            local_id += 1

        if USE_CLC:
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
            clc_phase_consumer ^= 1
            has_more_tile = tile_id != -1
        else:
            has_more_tile = False


@triton.jit
def triton_tbe_backward_short_run_weighted(
    dout_ptr,
    weight_ptr,
    infos_sorted_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    short_run_ids_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    per_sample_weights_ptr,
    sorted_linear_indices_num_runs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    USE_CLC: tl.constexpr,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
    vbe: tl.constexpr = False,
) -> None:
    """Backward kernel for short runs only (weighted). Each program handles one short run."""
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    if USE_CLC:
        clc_phase_producer = 1
        clc_phase_consumer = 0
        clc_context = tlx.clc_create_context(1)

        tile_id = tl.program_id(0)
        num_tiles = tl.num_programs(0)
        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)
        c_runs = num_short_runs // num_tiles
        r_runs = num_short_runs - c_runs * num_tiles
    else:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)

        num_short_runs = tl.load(sorted_linear_indices_num_runs_ptr)

        c = num_short_runs // num_programs
        r = num_short_runs - c * num_programs

        local_id = c * pid + tl.maximum(pid - num_programs + r, 0)
        local_id_end = local_id + c + (pid >= num_programs - r)

        # Early exit for programs with no assigned work
        if local_id >= local_id_end:
            return

    has_more_tile = True
    while has_more_tile:
        if USE_CLC:
            tlx.clc_producer(clc_context, clc_phase_producer)
            clc_phase_producer ^= 1
            local_id = c_runs * tile_id + tl.maximum(tile_id - num_tiles + r_runs, 0)
            local_id_end = local_id + c_runs + (tile_id >= num_tiles - r_runs)

        while local_id < local_id_end:
            run_id = tl.load(short_run_ids_ptr + local_id)

            linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
            segment_start = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id
            )
            segment_end = tl.load(
                sorted_linear_indices_cumulative_run_lengths_ptr + run_id + 1
            )

            info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
            t = (info_start >> info_B_num_bits).to(tl.int32)

            table_idx = tl.load(feature_table_map_ptr + t)
            table_offset = tl.load(table_offsets_ptr + table_idx)
            embedding_dim = tl.load(embedding_dims_ptr + t)
            mask = col_offsets < embedding_dim

            grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

            sz = (segment_end - segment_start) // buffer_size
            endn = segment_start + sz * buffer_size

            for idx in tl.range(
                segment_start,
                endn,
                buffer_size,
                flatten=True,
                loop_unroll_factor=2,
            ):
                info = tl.load(infos_sorted_ptr + idx + buffer_offsets).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr[:, None] + col_offsets[None, :]
                grad_buffer = tl.load(dout_row_ptrs, mask=mask[None, :], other=0).to(
                    tl.float32
                )
                idx_weights = tl.load(per_sample_weights_ptr + idx + buffer_offsets)
                grad_buffer = grad_buffer * idx_weights[:, None]
                grad += tl.sum(grad_buffer, axis=0)

            for idx in range(endn, segment_end):
                info = tl.load(infos_sorted_ptr + idx).to(tl.uint32)
                b = (info & info_B_mask).to(tl.int64)
                t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
                if vbe:
                    b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                    dout_row_start_ptr = dout_ptr + tl.load(
                        row_output_offsets_ptr + b_t
                    )
                else:
                    embedding_offset_per_entry = tl.load(
                        embedding_offsets_ptr + t_per_entry
                    )
                    dout_row_start_ptr = (
                        dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                    )
                dout_row_ptrs = dout_row_start_ptr + col_offsets
                dout_row = tl.load(dout_row_ptrs, mask=mask, other=0).to(tl.float32)
                idx_weight = tl.load(per_sample_weights_ptr + idx)
                dout_row = dout_row * idx_weight
                grad += dout_row

            grad_original = grad.to(tl.float32)

            index_offset = tl.load(hash_size_cumsum_ptr + t)
            row_idx = linear_index - index_offset
            row_start_ptr = weight_ptr + table_offset + row_idx * embedding_dim
            row_ptrs = row_start_ptr + col_offsets
            row = tl.load(row_ptrs, mask=col_offsets < embedding_dim, other=0).to(
                tl.float32
            )

            row_update = row - learning_rate * grad_original

            if optimizer == 1:
                row_offset = tl.load(rows_cumsum_ptr + table_idx)
                momentum_idx = row_offset + row_idx

                grad_square = grad_original * grad_original
                grad_square_average = tl.sum(grad_square) / embedding_dim

                momentum = tl.load(momentum_ptr + momentum_idx)
                momentum_new = momentum + grad_square_average
                tl.store(momentum_ptr + momentum_idx, momentum_new)

                adaptive_learning_rate = learning_rate / (tl.sqrt(momentum_new) + eps)

                row_update = row - adaptive_learning_rate * grad_original

            if STOCHASTIC_ROUNDING:
                sr_offset = local_id * BLOCK_SIZE + col_offsets
                _stochastic_rounding_store(
                    row_ptrs, row_update, mask, stochastic_rounding_seed, sr_offset
                )
            else:
                tl.store(row_ptrs, row_update, mask=mask)

            local_id += 1

        if USE_CLC:
            tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
            clc_phase_consumer ^= 1
            has_more_tile = tile_id != -1
        else:
            has_more_tile = False


@triton.jit
def triton_tbe_backward_long_run_grad_accum_weighted(
    dout_ptr,
    infos_sorted_ptr,
    long_run_program_seg_starts_ptr,
    long_run_program_seg_ends_ptr,
    long_run_grad_buffer_ids_ptr,
    temp_grad_buffer_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    per_sample_weights_ptr,
    num_long_run_programs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    vbe: tl.constexpr = False,
) -> None:
    """
    Weighted version: each program accumulates weighted gradients for a
    sub-range of a long run and atomically adds into a temp gradient buffer.
    """
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_long_run_programs = tl.load(num_long_run_programs_ptr)

    c = num_long_run_programs // num_programs
    r = num_long_run_programs - c * num_programs

    local_id = c * pid + tl.maximum(pid - num_programs + r, 0)
    local_id_end = local_id + c + (pid >= num_programs - r)

    # Early exit for programs with no assigned work
    if local_id >= local_id_end:
        return

    has_more_tile = True
    while has_more_tile:

        segment_start = tl.load(long_run_program_seg_starts_ptr + local_id)
        segment_end = tl.load(long_run_program_seg_ends_ptr + local_id)
        grad_buffer_id = tl.load(long_run_grad_buffer_ids_ptr + local_id)

        info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
        t = (info_start >> info_B_num_bits).to(tl.int32)
        embedding_dim = tl.load(embedding_dims_ptr + t)
        mask = col_offsets < embedding_dim

        grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        sz = (segment_end - segment_start) // buffer_size
        endn = segment_start + sz * buffer_size

        for idx in tl.range(
            segment_start,
            endn,
            buffer_size,
            flatten=True,
            loop_unroll_factor=2,
        ):
            info = tl.load(infos_sorted_ptr + idx + buffer_offsets).to(tl.uint32)
            b = (info & info_B_mask).to(tl.int64)
            t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
            if vbe:
                b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                dout_row_start_ptr = dout_ptr + tl.load(row_output_offsets_ptr + b_t)
            else:
                embedding_offset_per_entry = tl.load(
                    embedding_offsets_ptr + t_per_entry
                )
                dout_row_start_ptr = (
                    dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                )
            dout_row_ptrs = dout_row_start_ptr[:, None] + col_offsets[None, :]
            grad_buffer = tl.load(dout_row_ptrs, mask=mask[None, :], other=0).to(
                tl.float32
            )
            idx_weights = tl.load(per_sample_weights_ptr + idx + buffer_offsets)
            grad_buffer = grad_buffer * idx_weights[:, None]
            grad += tl.sum(grad_buffer, axis=0)

        for idx in range(endn, segment_end):
            info = tl.load(infos_sorted_ptr + idx).to(tl.uint32)
            b = (info & info_B_mask).to(tl.int64)
            t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
            if vbe:
                b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                dout_row_start_ptr = dout_ptr + tl.load(row_output_offsets_ptr + b_t)
            else:
                embedding_offset_per_entry = tl.load(
                    embedding_offsets_ptr + t_per_entry
                )
                dout_row_start_ptr = (
                    dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                )
            dout_row_ptrs = dout_row_start_ptr + col_offsets
            dout_row = tl.load(dout_row_ptrs, mask=mask, other=0).to(tl.float32)
            idx_weight = tl.load(per_sample_weights_ptr + idx)
            dout_row = dout_row * idx_weight
            grad += dout_row

        # Atomically accumulate partial gradient into the temp buffer
        temp_grad_offset = grad_buffer_id.to(tl.int64) * BLOCK_SIZE
        tl.atomic_add(
            temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
            grad,
            mask=mask,
        )

        local_id += 1
        has_more_tile = local_id < local_id_end


@triton.jit
def triton_tbe_backward_long_run_grad_accum_unweighted(
    dout_ptr,
    infos_sorted_ptr,
    long_run_program_seg_starts_ptr,
    long_run_program_seg_ends_ptr,
    long_run_grad_buffer_ids_ptr,
    temp_grad_buffer_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    num_long_run_programs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    vbe: tl.constexpr = False,
) -> None:
    """
    Each program accumulates gradients for a sub-range of a long run
    and atomically adds the partial result into a temp gradient buffer.
    """
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    num_long_run_programs = tl.load(num_long_run_programs_ptr)

    c = num_long_run_programs // num_programs
    r = num_long_run_programs - c * num_programs

    local_id = c * pid + tl.maximum(pid - num_programs + r, 0)
    local_id_end = local_id + c + (pid >= num_programs - r)

    # Early exit for excess programs when grid > num_long_run_programs
    if local_id >= local_id_end:
        return

    has_more_tile = True
    while has_more_tile:

        segment_start = tl.load(long_run_program_seg_starts_ptr + local_id)
        segment_end = tl.load(long_run_program_seg_ends_ptr + local_id)
        grad_buffer_id = tl.load(long_run_grad_buffer_ids_ptr + local_id)

        info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
        t = (info_start >> info_B_num_bits).to(tl.int32)
        embedding_dim = tl.load(embedding_dims_ptr + t)
        mask = col_offsets < embedding_dim

        grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        sz = (segment_end - segment_start) // buffer_size
        endn = segment_start + sz * buffer_size

        for idx in tl.range(
            segment_start,
            endn,
            buffer_size,
            flatten=True,
            loop_unroll_factor=2,
        ):
            info = tl.load(infos_sorted_ptr + idx + buffer_offsets).to(tl.uint32)
            b = (info & info_B_mask).to(tl.int64)
            t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
            if vbe:
                b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                dout_row_start_ptr = dout_ptr + tl.load(row_output_offsets_ptr + b_t)
            else:
                embedding_offset_per_entry = tl.load(
                    embedding_offsets_ptr + t_per_entry
                )
                dout_row_start_ptr = (
                    dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                )
            dout_row_ptrs = dout_row_start_ptr[:, None] + col_offsets[None, :]
            grad_buffer = tl.load(dout_row_ptrs, mask=mask[None, :], other=0).to(
                tl.float32
            )
            grad += tl.sum(grad_buffer, axis=0)

        for idx in range(endn, segment_end):
            info = tl.load(infos_sorted_ptr + idx).to(tl.uint32)
            b = (info & info_B_mask).to(tl.int64)
            t_per_entry = (info.to(tl.uint32) >> info_B_num_bits).to(tl.int32)
            if vbe:
                b_t = tl.load(B_offsets_ptr + t_per_entry).to(tl.int64) + b
                dout_row_start_ptr = dout_ptr + tl.load(row_output_offsets_ptr + b_t)
            else:
                embedding_offset_per_entry = tl.load(
                    embedding_offsets_ptr + t_per_entry
                )
                dout_row_start_ptr = (
                    dout_ptr + b * total_embedding_dim + embedding_offset_per_entry
                )
            dout_row_ptrs = dout_row_start_ptr + col_offsets
            dout_row = tl.load(dout_row_ptrs, mask=mask, other=0).to(tl.float32)
            grad += dout_row

        # Atomically accumulate partial gradient into the temp buffer
        temp_grad_offset = grad_buffer_id.to(tl.int64) * BLOCK_SIZE
        tl.atomic_add(
            temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
            grad,
            mask=mask,
        )

        local_id += 1
        has_more_tile = local_id < local_id_end


@triton.jit
def triton_tbe_backward_long_run_apply_unweighted(
    weight_ptr,
    temp_grad_buffer_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    infos_sorted_ptr,
    long_run_original_ids_ptr,
    num_long_runs_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
) -> None:
    """
    One program per long run. Reads the accumulated gradient from the temp
    buffer and applies the optimizer update to the embedding row.
    """
    pid = tl.program_id(0)

    # Early exit for excess programs when grid > num_long_runs
    if pid >= tl.load(num_long_runs_ptr):
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)

    run_id = tl.load(long_run_original_ids_ptr + pid)
    linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
    segment_start = tl.load(sorted_linear_indices_cumulative_run_lengths_ptr + run_id)

    info_start = tl.load(infos_sorted_ptr + segment_start).to(tl.uint32)
    t = (info_start >> info_B_num_bits).to(tl.int32)

    table_idx = tl.load(feature_table_map_ptr + t)
    table_offset = tl.load(table_offsets_ptr + table_idx)
    embedding_dim = tl.load(embedding_dims_ptr + t)
    mask = col_offsets < embedding_dim

    # Read the accumulated gradient from temp buffer
    temp_grad_offset = pid.to(tl.int64) * BLOCK_SIZE
    grad_original = tl.load(
        temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
        mask=mask,
        other=0,
    )

    # Load embedding row
    index_offset = tl.load(hash_size_cumsum_ptr + t)
    row_idx = linear_index - index_offset
    row_start_ptr = weight_ptr + table_offset + row_idx * embedding_dim
    row_ptrs = row_start_ptr + col_offsets
    row = tl.load(row_ptrs, mask=mask, other=0)

    row_update = row - learning_rate * grad_original

    if optimizer == 1:
        row_offset = tl.load(rows_cumsum_ptr + table_idx)
        momentum_idx = row_offset + row_idx

        grad_square = grad_original * grad_original
        grad_square_average = tl.sum(grad_square) / embedding_dim

        momentum = tl.load(momentum_ptr + momentum_idx)
        momentum_new = momentum + grad_square_average
        tl.store(momentum_ptr + momentum_idx, momentum_new)

        adaptive_learning_rate = learning_rate / (tl.sqrt(momentum_new) + eps)

        row_update = row - adaptive_learning_rate * grad_original

    if STOCHASTIC_ROUNDING:
        sr_offset = tl.program_id(0) * BLOCK_SIZE + col_offsets
        _stochastic_rounding_store(
            row_ptrs, row_update, mask, stochastic_rounding_seed, sr_offset
        )
    else:
        tl.store(row_ptrs, row_update, mask=mask)


class TritonTBE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices,
        offsets,
        weight,
        table_offsets,
        embedding_dims,
        embedding_offsets,
        feature_table_map,
        total_embedding_dim: constexpr,
        T,
        hash_size_cumsum,
        total_hash_size_bits,
        learning_rate,
        block_size: constexpr,
        eps,
        optimizer,
        momentum,
        rows_cumsum,
        per_sample_weights,
        # pyre-fixme[2]: Parameter must be annotated.
        forward_event_callback,
        output_dtype,
        stochastic_rounding,
        # VBE parameters
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        # Cached VBE constants (avoid recomputing on every call)
        cached_feature_dims_cpu: Optional[torch.Tensor] = None,
        cached_D_offsets: Optional[torch.Tensor] = None,
        cached_max_D: int = 0,
        # Pre-computed VBE metadata from module.forward() (avoids double computation)
        precomputed_vbe_metadata: Optional[Any] = None,
        precomputed_row_output_offsets: Optional[torch.Tensor] = None,
        precomputed_b_t_map: Optional[torch.Tensor] = None,
        precomputed_info_B_num_bits: int = 0,
        precomputed_info_B_mask: int = 0,
        precomputed_total_B: int = 0,
        precomputed_max_B: int = 0,
    ) -> torch.Tensor:
        # VBE support: use pre-computed metadata if available, otherwise compute
        vbe = batch_size_per_feature_per_rank is not None
        if vbe:
            if precomputed_vbe_metadata is not None:
                vbe_metadata = precomputed_vbe_metadata
                row_output_offsets = precomputed_row_output_offsets
                b_t_map = precomputed_b_t_map
                info_B_num_bits = precomputed_info_B_num_bits
                info_B_mask = precomputed_info_B_mask
                total_B = precomputed_total_B
                B = precomputed_max_B
            else:
                assert batch_size_per_feature_per_rank is not None
                feature_dims_cpu = (
                    cached_feature_dims_cpu
                    if cached_feature_dims_cpu is not None
                    else embedding_dims.cpu().to(torch.int64)
                )
                vbe_metadata = generate_vbe_metadata(
                    offsets,
                    batch_size_per_feature_per_rank,
                    PoolingMode.SUM,
                    feature_dims_cpu,
                    weight.device,
                )
                total_B = sum(sum(bs) for bs in batch_size_per_feature_per_rank)
                B = vbe_metadata.max_B

                info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
                    vbe_metadata.B_offsets,
                    B,
                    T,
                )

                assert vbe_metadata.B_offsets is not None
                assert vbe_metadata.B_offsets_rank_per_feature is not None
                assert vbe_metadata.output_offsets_feature_rank is not None

                D_offsets = (
                    cached_D_offsets
                    if cached_D_offsets is not None
                    else torch.zeros(T + 1, device=weight.device, dtype=torch.int32)
                )
                if cached_D_offsets is None:
                    D_offsets[1:] = torch.cumsum(embedding_dims, dim=0)

                max_D = (
                    cached_max_D
                    if cached_max_D > 0
                    else int(embedding_dims.max().item())
                )

                row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
                    vbe_metadata.B_offsets,
                    vbe_metadata.B_offsets_rank_per_feature,
                    vbe_metadata.output_offsets_feature_rank,
                    D_offsets,
                    max_D,
                    False,
                    vbe_metadata.max_B_feature_rank,
                    info_B_num_bits,
                    total_B,
                )

            output = torch.empty(
                (vbe_metadata.output_size,), device=weight.device, dtype=output_dtype
            )
        else:
            B = (offsets.size(0) - 1) // T
            total_B = B * T
            row_output_offsets = None
            b_t_map = None
            info_B_num_bits = 0
            info_B_mask = 0
            output = torch.empty(
                (B, total_embedding_dim), device=weight.device, dtype=output_dtype
            )

        # For VBE backward, save row_output_offsets, B_offsets, and b_t_map
        if vbe:
            assert vbe_metadata.B_offsets is not None
            assert b_t_map is not None
            vbe_row_output_offsets = row_output_offsets
            vbe_B_offsets = vbe_metadata.B_offsets
            vbe_b_t_map = b_t_map
        else:
            vbe_row_output_offsets = torch.empty(
                0, device=weight.device, dtype=torch.int64
            )
            vbe_B_offsets = torch.empty(0, device=weight.device, dtype=torch.int32)
            vbe_b_t_map = torch.empty(0, device=weight.device, dtype=torch.int32)

        ctx.save_for_backward(
            indices,
            offsets,
            weight,
            table_offsets,
            embedding_dims,
            embedding_offsets,
            hash_size_cumsum,
            per_sample_weights if per_sample_weights is not None else torch.empty(0),
            vbe_row_output_offsets,
            vbe_B_offsets,
            vbe_b_t_map,
        )

        ctx.total_embedding_dim = total_embedding_dim
        ctx.total_hash_size_bits = total_hash_size_bits
        ctx.B = B
        ctx.T = T
        ctx.learning_rate = learning_rate
        ctx.block_size = block_size
        ctx.eps = eps
        ctx.optimizer = optimizer
        ctx.momentum = momentum
        ctx.rows_cumsum = rows_cumsum
        ctx.feature_table_map = feature_table_map
        ctx.stochastic_rounding = stochastic_rounding
        ctx.vbe = vbe

        num_warps = 1

        weighted = per_sample_weights is not None and per_sample_weights.numel() > 0

        # Prepare VBE pointers (use dummy tensor if not VBE)
        row_output_offsets_ptr = (
            row_output_offsets
            if vbe
            else torch.empty(0, device=weight.device, dtype=torch.int64)
        )
        b_t_map_ptr = (
            b_t_map if vbe else torch.empty(0, device=weight.device, dtype=torch.int32)
        )
        B_offsets_ptr = vbe_B_offsets

        if weighted:
            fwd_kernel = (
                _amd_fwd_weighted_kernel
                if is_amd()
                else table_batched_embedding_bag_forward_weighted_kernel
            )
            fwd_kernel[(total_B,)](
                output,
                indices,
                offsets,
                weight,
                table_offsets,
                embedding_dims,
                embedding_offsets,
                feature_table_map,
                per_sample_weights,
                row_output_offsets_ptr,
                b_t_map_ptr,
                total_embedding_dim,
                B,
                BLOCK_SIZE=block_size,
                vbe=vbe,
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
                num_warps=num_warps,
            )
        else:
            fwd_kernel = (
                _amd_fwd_unweighted_kernel
                if is_amd()
                else table_batched_embedding_bag_forward_unweighted_kernel
            )
            fwd_kernel[(B,)](
                output,
                indices,
                offsets,
                weight,
                table_offsets,
                embedding_dims,
                embedding_offsets,
                feature_table_map,
                row_output_offsets_ptr,
                B_offsets_ptr,
                total_embedding_dim,
                B,
                T,
                BLOCK_SIZE=block_size,
                vbe=vbe,
                num_warps=num_warps,
            )

        # Record a CUDA event to mark forward kernel completion.
        # This is needed for synchronization before NCCL collectives.
        if forward_event_callback is not None:
            forward_event_callback()

        return output

    @staticmethod
    #  inconsistently.
    def backward(ctx, dout) -> Tuple[None, ...]:
        # Ensure dout is contiguous for correct memory access in Triton kernels
        dout = dout.contiguous()

        (
            indices,
            offsets,
            weight,
            table_offsets,
            embedding_dims,
            embedding_offsets,
            hash_size_cumsum,
            per_sample_weights,
            vbe_row_output_offsets,
            vbe_B_offsets,
            vbe_b_t_map,
        ) = ctx.saved_tensors

        total_hash_size_bits = ctx.total_hash_size_bits
        total_embedding_dim = ctx.total_embedding_dim
        B = ctx.B
        T = ctx.T
        learning_rate = ctx.learning_rate
        eps = ctx.eps
        optimizer = ctx.optimizer
        momentum = ctx.momentum
        rows_cumsum = ctx.rows_cumsum
        block_size = ctx.block_size

        stochastic_rounding = ctx.stochastic_rounding
        vbe = ctx.vbe
        stochastic_rounding_seed = (
            torch.randint(0, 2**31, (1,), dtype=torch.int32).item()
            if stochastic_rounding
            else 0
        )

        weighted = per_sample_weights.numel() > 0

        # The first arg can be any device tensor
        info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
            indices, B, T
        )

        (
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            _,  # sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths,
        ) = torch.ops.fbgemm.transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            nobag=False,
            vbe_b_t_map=vbe_b_t_map if vbe else None,
            info_B_num_bits=info_B_num_bits,
            info_B_mask=info_B_mask,
        )

        # Debug: validate shapes and values
        if os.environ.get("TRITON_TBE_DEBUG"):
            num_runs = sorted_linear_indices_num_runs[0].item()
            print(
                f"[TritonTBE backward] dout.shape={dout.shape}, B={B}, T={T}, "
                f"total_embedding_dim={total_embedding_dim}, device={dout.device}, "
                f"num_runs={num_runs}, indices.shape={indices.shape}, "
                f"weight.shape={weight.shape}, info_B_num_bits={info_B_num_bits}, "
                f"info_B_mask={info_B_mask}"
            )

        if weighted:
            # linear_indices and per_sample_weights need to be sorted together so they matchs
            perm = torch.argsort(linear_indices)
            sorted_per_sample_weights = per_sample_weights[perm]
        else:
            sorted_per_sample_weights = torch.empty(0, device=weight.device)

        num_warps = 1

        use_clc = False
        if has_tlx:
            if torch.cuda.get_device_capability()[0] >= 10:
                use_clc = True

        # Common setup for 2-tier dispatch (both weighted and unweighted)
        max_num_runs = indices.numel()
        max_long_runs = max_num_runs // _LONG_RUN_THRESHOLD + 1
        max_long_run_programs = 2 * max_num_runs // _LONG_RUN_THRESHOLD + 1
        # Fixed grid sizes: work distribution while-loops handle
        # any num_short_runs / num_long_run_programs value.
        # Using 24576 matching the theoretical prediction (192 SMs * 32 warps/SM * 4 waves).
        # Below 24576, performance drops because there aren't enough programs to keep all
        # SMs busy. Above it, diminishing returns from scheduling overhead of excess programs.

        # NVIDIA: 192 SMs * 32 warps/SM * 4 waves
        # AMD: 256 CUs * 64 warps/CU * 4 waves
        short_run_grid_size, long_accum_or_fused_grid_size, long_apply_grid_size = (
            get_grid_size(
                is_amd=is_amd(),
                max_num_runs=max_num_runs,
                max_long_runs=max_long_runs,
                max_long_run_programs=max_long_run_programs,
                use_clc=use_clc,
            )
        )

        if is_amd():
            (
                short_run_ids,
                num_short_runs_t,
                _long_run_program_run_ids,
                long_run_program_seg_starts,
                long_run_program_seg_ends,
                num_long_run_programs_t,
                num_long_runs_t,
                long_run_grad_buffer_ids,
                long_run_original_ids,
            ) = _amd_expand_long_runs(
                sorted_linear_indices_cumulative_run_lengths,
                sorted_linear_indices_num_runs,
                max_num_runs,
            )
        else:
            (
                short_run_ids,
                num_short_runs_t,
                _long_run_program_run_ids,
                long_run_program_seg_starts,
                long_run_program_seg_ends,
                num_long_run_programs_t,
                num_long_runs_t,
                long_run_grad_buffer_ids,
                long_run_original_ids,
                programs_per_long_run,
            ) = _expand_long_runs(
                sorted_linear_indices_cumulative_run_lengths,
                sorted_linear_indices_num_runs,
                max_num_runs,
            )

        temp_grad_buffer = torch.zeros(
            (max_long_runs, block_size),
            dtype=torch.float32,
            device=weight.device,
        )

        # Select kernel variants based on hardware
        _use_amd = is_amd()

        if weighted:
            # Weighted: 2-tier dispatch with sync-free _expand_long_runs
            # Kernel 1: short-run kernel (weighted)
            bwd_short_w = (
                _amd_bwd_short_weighted
                if _use_amd
                else triton_tbe_backward_short_run_weighted
            )
            bwd_short_w[(short_run_grid_size,)](
                dout,
                weight,
                infos_sorted,
                sorted_linear_indices_run,
                sorted_linear_indices_cumulative_run_lengths,
                short_run_ids,
                table_offsets,
                embedding_dims,
                embedding_offsets,
                ctx.feature_table_map,
                hash_size_cumsum,
                momentum,
                rows_cumsum,
                sorted_per_sample_weights,
                num_short_runs_t,
                vbe_row_output_offsets,
                vbe_B_offsets,
                total_embedding_dim,
                B,
                learning_rate,
                eps,
                OPTIM_TYPE_TO_INT[optimizer],
                BLOCK_SIZE=block_size,
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
                num_warps=num_warps,
                USE_CLC=use_clc,
                STOCHASTIC_ROUNDING=stochastic_rounding,
                stochastic_rounding_seed=stochastic_rounding_seed,
                vbe=vbe,
            )
            if use_clc:
                # CLC path: fused long-run grad accumulation + optimizer apply
                # CLC Path is exclusive to CUDA B200+.
                grad_accum_counter = programs_per_long_run.clone()
                triton_tbe_backward_long_run_fused_weighted[
                    (long_accum_or_fused_grid_size,)
                ](
                    dout,
                    infos_sorted,
                    long_run_program_seg_starts,
                    long_run_program_seg_ends,
                    long_run_grad_buffer_ids,
                    temp_grad_buffer,
                    grad_accum_counter,
                    embedding_dims,
                    embedding_offsets,
                    sorted_per_sample_weights,
                    weight,
                    sorted_linear_indices_run,
                    sorted_linear_indices_cumulative_run_lengths,
                    long_run_original_ids,
                    table_offsets,
                    ctx.feature_table_map,
                    hash_size_cumsum,
                    momentum,
                    rows_cumsum,
                    num_long_run_programs_t,
                    vbe_row_output_offsets,
                    vbe_B_offsets,
                    total_embedding_dim,
                    learning_rate,
                    eps,
                    OPTIM_TYPE_TO_INT[optimizer],
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    STOCHASTIC_ROUNDING=stochastic_rounding,
                    stochastic_rounding_seed=stochastic_rounding_seed,
                    vbe=vbe,
                )
            else:
                # Non-CLC path: separate grad accumulation + apply kernels
                # Kernel 2: long-run grad accumulation (weighted)
                bwd_long_accum_w = (
                    _amd_bwd_long_accum_weighted
                    if _use_amd
                    else triton_tbe_backward_long_run_grad_accum_weighted
                )
                bwd_long_accum_w[(long_accum_or_fused_grid_size,)](
                    dout,
                    infos_sorted,
                    long_run_program_seg_starts,
                    long_run_program_seg_ends,
                    long_run_grad_buffer_ids,
                    temp_grad_buffer,
                    embedding_dims,
                    embedding_offsets,
                    sorted_per_sample_weights,
                    num_long_run_programs_t,
                    vbe_row_output_offsets,
                    vbe_B_offsets,
                    total_embedding_dim,
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    vbe=vbe,
                )
                # Kernel 3: apply optimizer (reuse unweighted — weight-independent)
                bwd_long_apply = (
                    _amd_bwd_long_apply
                    if _use_amd
                    else triton_tbe_backward_long_run_apply_unweighted
                )
                bwd_long_apply[(long_apply_grid_size,)](
                    weight,
                    temp_grad_buffer,
                    sorted_linear_indices_run,
                    sorted_linear_indices_cumulative_run_lengths,
                    infos_sorted,
                    long_run_original_ids,
                    num_long_runs_t,
                    table_offsets,
                    embedding_dims,
                    ctx.feature_table_map,
                    hash_size_cumsum,
                    momentum,
                    rows_cumsum,
                    learning_rate,
                    eps,
                    OPTIM_TYPE_TO_INT[optimizer],
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    STOCHASTIC_ROUNDING=stochastic_rounding,
                    stochastic_rounding_seed=stochastic_rounding_seed,
                )
        else:
            # Unweighted: 2-tier dispatch
            # Kernel 1: short-run kernel
            bwd_short_uw = (
                _amd_bwd_short_unweighted
                if _use_amd
                else triton_tbe_backward_short_run_unweighted
            )
            bwd_short_uw[(short_run_grid_size,)](
                dout,
                weight,
                infos_sorted,
                sorted_linear_indices_run,
                sorted_linear_indices_cumulative_run_lengths,
                short_run_ids,
                table_offsets,
                embedding_dims,
                embedding_offsets,
                ctx.feature_table_map,
                hash_size_cumsum,
                momentum,
                rows_cumsum,
                num_short_runs_t,
                vbe_row_output_offsets,
                vbe_B_offsets,
                total_embedding_dim,
                B,
                learning_rate,
                eps,
                OPTIM_TYPE_TO_INT[optimizer],
                BLOCK_SIZE=block_size,
                info_B_num_bits=info_B_num_bits,
                info_B_mask=info_B_mask,
                num_warps=num_warps,
                USE_CLC=use_clc,
                STOCHASTIC_ROUNDING=stochastic_rounding,
                stochastic_rounding_seed=stochastic_rounding_seed,
                vbe=vbe,
            )
            if use_clc:
                # CLC path: fused long-run grad accumulation + optimizer apply
                # CLC Path is exclusive to CUDA B200+.
                grad_accum_counter = programs_per_long_run.clone()
                triton_tbe_backward_long_run_fused_unweighted[
                    (long_accum_or_fused_grid_size,)
                ](
                    dout,
                    infos_sorted,
                    long_run_program_seg_starts,
                    long_run_program_seg_ends,
                    long_run_grad_buffer_ids,
                    temp_grad_buffer,
                    grad_accum_counter,
                    embedding_dims,
                    embedding_offsets,
                    weight,
                    sorted_linear_indices_run,
                    sorted_linear_indices_cumulative_run_lengths,
                    long_run_original_ids,
                    table_offsets,
                    ctx.feature_table_map,
                    hash_size_cumsum,
                    momentum,
                    rows_cumsum,
                    num_long_run_programs_t,
                    vbe_row_output_offsets,
                    vbe_B_offsets,
                    total_embedding_dim,
                    learning_rate,
                    eps,
                    OPTIM_TYPE_TO_INT[optimizer],
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    STOCHASTIC_ROUNDING=stochastic_rounding,
                    stochastic_rounding_seed=stochastic_rounding_seed,
                    vbe=vbe,
                )
            else:
                # Non-CLC path: separate grad accumulation + apply kernels
                # Kernel 2: long-run grad accumulation
                bwd_long_accum_uw = (
                    _amd_bwd_long_accum_unweighted
                    if _use_amd
                    else triton_tbe_backward_long_run_grad_accum_unweighted
                )
                bwd_long_accum_uw[(long_accum_or_fused_grid_size,)](
                    dout,
                    infos_sorted,
                    long_run_program_seg_starts,
                    long_run_program_seg_ends,
                    long_run_grad_buffer_ids,
                    temp_grad_buffer,
                    embedding_dims,
                    embedding_offsets,
                    num_long_run_programs_t,
                    vbe_row_output_offsets,
                    vbe_B_offsets,
                    total_embedding_dim,
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    vbe=vbe,
                )

                # Kernel 3: apply optimizer (direct mapping, no while-loop)
                bwd_long_apply = (
                    _amd_bwd_long_apply
                    if _use_amd
                    else triton_tbe_backward_long_run_apply_unweighted
                )
                bwd_long_apply[(long_apply_grid_size,)](
                    weight,
                    temp_grad_buffer,
                    sorted_linear_indices_run,
                    sorted_linear_indices_cumulative_run_lengths,
                    infos_sorted,
                    long_run_original_ids,
                    num_long_runs_t,
                    table_offsets,
                    embedding_dims,
                    ctx.feature_table_map,
                    hash_size_cumsum,
                    momentum,
                    rows_cumsum,
                    learning_rate,
                    eps,
                    OPTIM_TYPE_TO_INT[optimizer],
                    BLOCK_SIZE=block_size,
                    info_B_num_bits=info_B_num_bits,
                    info_B_mask=info_B_mask,
                    num_warps=num_warps,
                    STOCHASTIC_ROUNDING=stochastic_rounding,
                    stochastic_rounding_seed=stochastic_rounding_seed,
                )
        # Debug logging after backward kernel
        if os.environ.get("TRITON_TBE_DEBUG"):
            print("[TritonTBE backward] Backward kernel completed")
            print(
                f"[TritonTBE backward] weight after: min={weight.min().item():.4f}, max={weight.max().item():.4f}, "
                f"has_nan={torch.isnan(weight).any().item()}, has_inf={torch.isinf(weight).any().item()}"
            )
            print("[TritonTBE backward] ===== BACKWARD PASS END =====")

        return (
            None,  # indices
            None,  # offsets
            None,  # weight
            None,  # table_offsets
            None,  # embedding_dims
            None,  # embedding_offsets
            None,  # feature_table_map
            None,  # total_embedding_dim
            None,  # T
            None,  # hash_size_cumsum
            None,  # total_hash_size_bits
            None,  # learning_rate
            None,  # block_size
            None,  # eps
            None,  # optimizer
            None,  # momentum
            None,  # rows_cumsum
            None,  # per_sample_weights
            None,  # forward_event_callback
            None,  # output_dtype
            None,  # stochastic_rounding
            None,  # batch_size_per_feature_per_rank
            None,  # cached_feature_dims_cpu
            None,  # cached_D_offsets
            None,  # cached_max_D
            None,  # precomputed_vbe_metadata
            None,  # precomputed_row_output_offsets
            None,  # precomputed_b_t_map
            None,  # precomputed_info_B_num_bits
            None,  # precomputed_info_B_mask
            None,  # precomputed_total_B
            None,  # precomputed_max_B
        )


class TritonTableBatchedEmbeddingBags(torch.nn.Module):
    embedding_specs: List[Tuple[int, int]]
    # CUDA event to track forward kernel completion for stream synchronization.
    # This is used to ensure proper ordering with NCCL collectives.
    _forward_event: Optional[torch.cuda.Event]

    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],  # tuple of (rows, dims)
        feature_table_map: Optional[List[int]] = None,  # [T] maps features to tables
        weights_precision: torch.dtype = torch.float32,
        output_dtype: Optional[torch.dtype] = torch.float32,
        stochastic_rounding: bool = True,
        learning_rate: float = 0.01,
        eps: float = 0.1,
        optimizer: OptimType = OptimType.EXACT_SGD,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        logging.info("TritonTableBatchedEmbeddingBags init args: %s", locals())
        self.embedding_specs = embedding_specs
        # Initialize event as None; it will be set after forward kernel runs
        self._forward_event = None
        T_ = len(embedding_specs)  # num of physical tables

        # If feature_table_map is not provided, assume 1:1 mapping (one feature per table)
        if feature_table_map is None:
            feature_table_map = list(range(T_))

        self.feature_table_map: List[int] = feature_table_map
        self.T = len(
            feature_table_map
        )  # num of features (used for batch size calculation)

        if device is None:
            device = torch.device(torch.cuda.current_device())

        # Physical table properties (unchanged for weight storage)
        table_embedding_dims = [spec[1] for spec in embedding_specs]
        self.max_embedding_dim = max(table_embedding_dims)
        table_sizes = [spec[0] * spec[1] for spec in embedding_specs]

        # Feature-level properties (indexed by feature_table_map)
        # These are used for forward pass and output shape calculation
        feature_dims = [table_embedding_dims[t] for t in feature_table_map]
        self.total_embedding_dim = sum(feature_dims)
        self.table_offsets = torch.tensor(
            lengths_to_offsets(table_sizes), dtype=torch.int64, device=device
        )
        self.embedding_offsets = torch.tensor(
            lengths_to_offsets(feature_dims), dtype=torch.int64, device=device
        )
        self.embedding_dims = torch.tensor(
            feature_dims, dtype=torch.int64, device=device
        )
        # Store the feature_table_map as a tensor for kernel use
        self.feature_table_map_tensor = torch.tensor(
            feature_table_map, dtype=torch.int64, device=device
        )

        hash_sizes = [spec[0] for spec in embedding_specs]
        total_hash_size = sum(hash_sizes)
        self.total_hash_size_bits: int = int(math.log2(total_hash_size) + 1)
        # Compute table-level hash_size_cumsum first
        table_hash_size_cumsum = lengths_to_offsets(hash_sizes, keep_last=True)
        # Hash size cumsum indexed by feature (matching SplitTBE behavior):
        # hash_size_cumsum[f] = table_hash_size_cumsum[feature_table_map[f]]
        feature_hash_size_cumsum = [
            table_hash_size_cumsum[t] for t in feature_table_map
        ] + [total_hash_size]
        self.hash_size_cumsum = torch.tensor(
            feature_hash_size_cumsum,
            dtype=torch.int64,
            device=device,
        )

        # Pre-compute VBE-related constants (avoid recomputation on every forward)
        self._feature_dims_cpu = torch.tensor(feature_dims, dtype=torch.int64)
        self._D_offsets = torch.zeros(self.T + 1, device=device, dtype=torch.int32)
        self._D_offsets[1:] = torch.cumsum(self.embedding_dims.int(), dim=0)
        self._max_D = self.max_embedding_dim

        self.block_size = triton.next_power_of_2(self.max_embedding_dim)
        total_weight_size = sum(table_sizes)
        self.weight = torch.empty(
            [total_weight_size],
            dtype=weights_precision,
            device=device,
            requires_grad=True,
        )

        self.output_dtype = (
            output_dtype if output_dtype is not None else weights_precision
        )
        self.stochastic_rounding = stochastic_rounding
        self.learning_rate = learning_rate
        self.eps = eps
        self.optimizer = optimizer

        # Initialize optimizer state
        rows = [spec[0] for spec in embedding_specs]
        self.rows_cumsum = torch.tensor(
            lengths_to_offsets(rows, keep_last=True),
            dtype=torch.int64,
            device=device,
        )

        if optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            total_rows = sum(rows)
            self.momentum = torch.zeros(
                [total_rows],
                dtype=torch.float32,
                device=device,
            )
        else:
            self.momentum = torch.zeros([1], dtype=torch.float32, device=device)

        # Bounds checking configuration
        # rows_per_table needs to be indexed by feature (not table) for bounds_check_indices
        rows_per_feature = [rows[t] for t in feature_table_map]
        self.rows_per_table = torch.tensor(
            rows_per_feature, dtype=torch.int64, device=device
        )
        self.bounds_check_warning = torch.tensor([0], device=device, dtype=torch.int64)
        # Use WARNING mode by default. We don't support environment variable override
        # because TritonTBE only uses bounds check v1 kernel, while the env var
        # (FBGEMM_TBE_BOUNDS_CHECK_MODE) can also set v2 modes (V2_IGNORE, V2_WARNING, V2_FATAL).
        self.bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING

    def prepare_inputs(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare TBE inputs by running bounds check on indices.

        This method validates that all indices are within the valid row ranges
        for their respective tables, similar to SplitTableBatchedEmbeddingBagsCodegen.

        Args:
            indices (Tensor): Input indices
            offsets (Tensor): Input offsets
            per_sample_weights (Optional[Tensor]): Input per sample weights

        Returns:
            A tuple of (indices, offsets, per_sample_weights) after validation
        """
        # Input type casting: ensure offsets has the same dtype as indices
        # since the kernels assume same dtype. This follows the same pattern
        # as SplitTableBatchedEmbeddingBagsCodegen.
        if indices.dtype != offsets.dtype:
            offsets = offsets.to(dtype=indices.dtype)

        # Force casting per_sample_weights to float for numerical stability
        if (
            per_sample_weights is not None
            and not per_sample_weights.is_floating_point()
        ):
            per_sample_weights = per_sample_weights.float()

        if self.bounds_check_mode != BoundsCheckMode.NONE:
            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode,
                self.bounds_check_warning,
                per_sample_weights,
            )

        return indices, offsets, per_sample_weights

    def _generate_vbe_metadata_for_bounds_check(
        self,
        offsets: torch.Tensor,
        batch_size_per_feature_per_rank: List[List[int]],
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, int, int, int, int]:
        """Generate VBE metadata for bounds_check_indices and kernels."""
        vbe_metadata = generate_vbe_metadata(
            offsets,
            batch_size_per_feature_per_rank,
            PoolingMode.SUM,
            self._feature_dims_cpu,
            self.weight.device,
        )
        total_B = sum(sum(bs) for bs in batch_size_per_feature_per_rank)
        max_B: int = vbe_metadata.max_B  # pyre-ignore[8]
        info_B_num_bits, info_B_mask = torch.ops.fbgemm.get_infos_metadata(
            vbe_metadata.B_offsets,
            max_B,
            self.T,
        )
        assert vbe_metadata.B_offsets is not None
        assert vbe_metadata.B_offsets_rank_per_feature is not None
        assert vbe_metadata.output_offsets_feature_rank is not None
        row_output_offsets, b_t_map = torch.ops.fbgemm.generate_vbe_metadata(
            vbe_metadata.B_offsets,
            vbe_metadata.B_offsets_rank_per_feature,
            vbe_metadata.output_offsets_feature_rank,
            self._D_offsets,
            self._max_D,
            False,
            vbe_metadata.max_B_feature_rank,
            info_B_num_bits,
            total_B,
        )
        return (
            vbe_metadata,
            row_output_offsets,
            b_t_map,
            info_B_num_bits,
            info_B_mask,
            total_B,
            max_B,
        )

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
    ) -> torch.Tensor:
        # Input type casting
        if indices.dtype != offsets.dtype:
            offsets = offsets.to(dtype=indices.dtype)
        if (
            per_sample_weights is not None
            and not per_sample_weights.is_floating_point()
        ):
            per_sample_weights = per_sample_weights.float()

        # Pre-compute VBE metadata (used for both bounds check and kernel)
        vbe_metadata = None
        row_output_offsets = None
        b_t_map = None
        info_B_num_bits = 0
        info_B_mask = 0
        total_B = 0
        max_B = 0
        if batch_size_per_feature_per_rank is not None:
            (
                vbe_metadata,
                row_output_offsets,
                b_t_map,
                info_B_num_bits,
                info_B_mask,
                total_B,
                max_B,
            ) = self._generate_vbe_metadata_for_bounds_check(
                offsets, batch_size_per_feature_per_rank
            )

        # Bounds check (VBE-aware)
        if self.bounds_check_mode != BoundsCheckMode.NONE:
            torch.ops.fbgemm.bounds_check_indices(
                self.rows_per_table,
                indices,
                offsets,
                self.bounds_check_mode,
                self.bounds_check_warning,
                per_sample_weights,
                B_offsets=vbe_metadata.B_offsets if vbe_metadata is not None else None,
                max_B=max_B if max_B > 0 else -1,
                b_t_map=b_t_map,
                info_B_num_bits=info_B_num_bits if info_B_num_bits > 0 else -1,
                info_B_mask=info_B_mask if info_B_mask > 0 else -1,
            )

        return TritonTBE.apply(
            indices,
            offsets,
            self.weight,
            self.table_offsets,
            self.embedding_dims,
            self.embedding_offsets,
            self.feature_table_map_tensor,
            self.total_embedding_dim,
            self.T,
            self.hash_size_cumsum,
            self.total_hash_size_bits,
            self.learning_rate,
            self.block_size,
            self.eps,
            self.optimizer,
            self.momentum,
            self.rows_cumsum,
            per_sample_weights,
            self.record_forward_event,
            self.output_dtype,
            self.stochastic_rounding,
            batch_size_per_feature_per_rank,
            self._feature_dims_cpu,
            self._D_offsets,
            self._max_D,
            vbe_metadata,
            row_output_offsets,
            b_t_map,
            info_B_num_bits,
            info_B_mask,
            total_B,
            max_B,
        )

    def split_embedding_weights(self) -> List[torch.Tensor]:
        """
        Returns a list of embedding weights (view), split by table.

        Returns:
            A list of weights. Length = the number of tables
        """
        splits = []
        for t, (rows, dim) in enumerate(self.embedding_specs):
            offset = self.table_offsets[t].item()
            splits.append(
                self.weight.detach()[offset : offset + rows * dim].view(rows, dim)
            )
        return splits

    def flush(self) -> None:
        """No-op for Triton TBE (no cache to flush)."""
        pass

    def reset_cache_states(self) -> None:
        """No-op for Triton TBE (no cache to reset)."""
        pass

    def split_optimizer_states(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        Returns a list of optimizer states (view), split by table.

        For EXACT_ROWWISE_ADAGRAD, returns the momentum (accumulated squared gradients)
        for each table as a 1D tensor of shape [rows].

        Returns:
            A list of tuples of optimizer state tensors. Length = the number of tables.
            For EXACT_ROWWISE_ADAGRAD: [(momentum1,), (momentum1,), ...]
            For EXACT_SGD: [(), (), ...] (empty tuples, no optimizer state)
        """
        if self.optimizer == OptimType.EXACT_SGD:
            return [() for _ in self.embedding_specs]

        # For EXACT_ROWWISE_ADAGRAD, split the momentum tensor by table
        splits = []
        for t, (rows, _dim) in enumerate(self.embedding_specs):
            start_offset = self.rows_cumsum[t].item()
            end_offset = start_offset + rows
            splits.append((self.momentum[start_offset:end_offset],))
        return splits

    def get_optimizer_state(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get the optimizer state dict that matches the OSS Pytorch optims.

        Returns:
            A list of dicts, one per table, with optimizer state tensors.
            For EXACT_ROWWISE_ADAGRAD: [{"sum": tensor}, ...]
            For EXACT_SGD: [{}, {}, ...] (empty dicts)
        """
        split_optimizer_states = self.split_optimizer_states()

        if self.optimizer == OptimType.EXACT_SGD:
            return [{} for _ in split_optimizer_states]
        elif self.optimizer == OptimType.EXACT_ROWWISE_ADAGRAD:
            return [{"sum": states[0]} for states in split_optimizer_states]
        else:
            raise NotImplementedError(
                f"Getting optimizer state for {self.optimizer} is not implemented in TritonTBE"
            )

    def record_forward_event(self) -> None:
        """
        Record a CUDA event on the current stream to track forward kernel completion.

        This should be called after the forward pass to record when the forward
        kernel finishes. The event can then be used by downstream code (e.g., 2D
        sharding) to ensure proper stream synchronization before collective operations.

        This is necessary because Triton kernels run asynchronously, and NCCL
        collectives may run on a different stream. Without explicit synchronization,
        the collective might start before the forward kernel completes, causing
        NCCL timeouts when ranks reach the collective at different times.
        """
        if self._forward_event is None:
            self._forward_event = torch.cuda.Event(enable_timing=False)
        assert self._forward_event is not None
        self._forward_event.record(torch.cuda.current_stream())

    def wait_for_forward(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Wait for the forward kernel to complete on the specified stream.

        This should be called before collective operations (e.g., ALLTOALL for
        output distribution) to ensure the forward kernel has completed.
        If no stream is specified, uses the current stream.

        Args:
            stream: The stream to wait on. If None, uses torch.cuda.current_stream().
        """
        if self._forward_event is not None:
            target_stream = (
                stream if stream is not None else torch.cuda.current_stream()
            )
            target_stream.wait_event(self._forward_event)


@triton.jit
def _nbit_TBE_forward_kernel_16bits(
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    D_offsets_ptr,
    weight_Ds_ptr,
    weight_tys_ptr,
    total_D: tl.constexpr,
    B: tl.constexpr,
    block_size_fp16: tl.constexpr,
    block_size_int2: tl.constexpr,
) -> None:

    bag_idx = tl.program_id(0).to(tl.int64)

    t = bag_idx // B  # table id
    b = bag_idx % B  # batch id

    table_offset = tl.load(table_offsets_ptr + t)
    D_offset = tl.load(D_offsets_ptr + t)
    D = tl.load(D_offsets_ptr + t + 1) - D_offset
    weight_D = tl.load(weight_Ds_ptr + t)
    weight_ty = tl.load(weight_tys_ptr + t)

    start = tl.load(offsets_ptr + bag_idx)
    end = tl.load(offsets_ptr + bag_idx + 1)

    output_row_start_ptr = output_ptr + b * total_D + D_offset

    if weight_ty == 1:  # fp16
        col_offsets = tl.arange(0, block_size_fp16)
        output_row_ptrs = tl.multiple_of(output_row_start_ptr + col_offsets, 4)
        output = tl.zeros((block_size_fp16,), dtype=tl.float32)
        mask_weight = col_offsets < weight_D

        step_fp16: tl.constexpr = 4
        ns = (end - start) // step_fp16
        endn = start + step_fp16 * ns

        for idx in range(start, endn, step_fp16):
            for j in range(step_fp16):
                row_idx = tl.load(indices_ptr + idx + j)
                row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
                row_ptrs = row_start_ptr + col_offsets
                row = tl.load(row_ptrs, mask=mask_weight, other=0)
                # bitwise cast row from int16 to float16
                output += row.to(tl.float16, bitcast=True)

        for idx in range(endn, end):
            row_idx = tl.load(indices_ptr + idx)
            row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
            row_ptrs = row_start_ptr + col_offsets
            row = tl.load(row_ptrs, mask=mask_weight, other=0)
            # bitwise cast row from int16 to float16
            output += row.to(tl.float16, bitcast=True)

        output_fp32 = output.to(tl.float32)
        tl.store(output_row_ptrs, output_fp32, mask=col_offsets < D)

    elif weight_ty == 4:  # int2
        col_offsets = tl.arange(0, block_size_int2)
        output_row_ptrs = tl.multiple_of(output_row_start_ptr + col_offsets, 8)

        e_BS: tl.constexpr = block_size_int2 >> 3
        e_col_offsets = tl.arange(0, e_BS)
        mask_weight = e_col_offsets < (weight_D - 2)

        output_e0 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e1 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e2 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e3 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e4 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e5 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e6 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e7 = tl.zeros((e_BS,), dtype=tl.float32)

        for idx in range(start, end):
            row_idx = tl.load(indices_ptr + idx)
            row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
            row_ptrs = tl.multiple_of(row_start_ptr + 2 + e_col_offsets, 1)
            row = tl.load(row_ptrs, mask=mask_weight, other=0).to(
                tl.uint16, bitcast=True
            )
            scale = tl.load(row_start_ptr).to(tl.float16, bitcast=True)
            shift = tl.load(row_start_ptr + 1).to(tl.float16, bitcast=True)

            scale_512 = (scale * 512.0).to(tl.float16)
            scale_128 = (scale * 128.0).to(tl.float16)

            row_e0 = (
                (row & 3).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e1 = (
                (row & 12).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 4
            row_e2 = (
                (row & 3).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e3 = (
                (row & 12).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 4
            row_e4 = (
                (row & 3).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e5 = (
                (row & 12).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 4
            row_e6 = (
                (row & 3).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e7 = (
                (row & 12).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)

            output_e0 += row_e0 * scale_512 + shift
            output_e1 += row_e1 * scale_128 + shift
            output_e2 += row_e2 * scale_512 + shift
            output_e3 += row_e3 * scale_128 + shift
            output_e4 += row_e4 * scale_512 + shift
            output_e5 += row_e5 * scale_128 + shift
            output_e6 += row_e6 * scale_512 + shift
            output_e7 += row_e7 * scale_128 + shift

        mask_output = e_col_offsets < (D // 8)
        output_row_ptrs_e0 = output_row_start_ptr + 8 * e_col_offsets
        tl.store(output_row_ptrs_e0, output_e0, mask=mask_output)

        output_row_ptrs_e1 = output_row_start_ptr + 8 * e_col_offsets + 1
        tl.store(output_row_ptrs_e1, output_e1, mask=mask_output)

        output_row_ptrs_e2 = output_row_start_ptr + 8 * e_col_offsets + 2
        tl.store(output_row_ptrs_e2, output_e2, mask=mask_output)

        output_row_ptrs_e3 = output_row_start_ptr + 8 * e_col_offsets + 3
        tl.store(output_row_ptrs_e3, output_e3, mask=mask_output)

        output_row_ptrs_e4 = output_row_start_ptr + 8 * e_col_offsets + 4
        tl.store(output_row_ptrs_e4, output_e4, mask=mask_output)

        output_row_ptrs_e5 = output_row_start_ptr + 8 * e_col_offsets + 5
        tl.store(output_row_ptrs_e5, output_e5, mask=mask_output)

        output_row_ptrs_e6 = output_row_start_ptr + 8 * e_col_offsets + 6
        tl.store(output_row_ptrs_e6, output_e6, mask=mask_output)

        output_row_ptrs_e7 = output_row_start_ptr + 8 * e_col_offsets + 7
        tl.store(output_row_ptrs_e7, output_e7, mask=mask_output)


@triton.jit
def _nbit_TBE_forward_kernel_32bits(  # noqa
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    D_offsets_ptr,
    weight_Ds_ptr,
    weight_tys_ptr,
    total_D: tl.constexpr,
    B: tl.constexpr,
    block_size_fp32: tl.constexpr,
    block_size_int8: tl.constexpr,
    block_size_int4: tl.constexpr,
) -> None:

    bag_idx = tl.program_id(0).to(tl.int64)

    t = bag_idx // B  # table id
    b = bag_idx % B  # batch id

    table_offset = tl.load(table_offsets_ptr + t)
    D_offset = tl.load(D_offsets_ptr + t)
    D = tl.load(D_offsets_ptr + t + 1) - D_offset
    weight_D = tl.load(weight_Ds_ptr + t)
    weight_ty = tl.load(weight_tys_ptr + t)

    start = tl.load(offsets_ptr + bag_idx)
    end = tl.load(offsets_ptr + bag_idx + 1)

    output_row_start_ptr = output_ptr + b * total_D + D_offset

    if weight_ty == 0:  # fp32
        col_offsets = tl.arange(0, block_size_fp32)
        output_row_ptrs = tl.multiple_of(output_row_start_ptr + col_offsets, 4)
        output = tl.zeros((block_size_fp32,), dtype=tl.float64)
        mask_weight = col_offsets < weight_D

        step_fp32: tl.constexpr = 4
        ns = (end - start) // step_fp32
        endn = start + step_fp32 * ns

        for idx in range(start, endn, step_fp32):
            for j in range(step_fp32):
                row_idx = tl.load(indices_ptr + idx + j)
                row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
                row_ptrs = row_start_ptr + col_offsets
                row = tl.load(row_ptrs, mask=mask_weight, other=0)
                # bitwise cast row from int32 to float32
                output += row.to(tl.float32, bitcast=True)

        for idx in range(endn, end):
            row_idx = tl.load(indices_ptr + idx)
            row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
            row_ptrs = row_start_ptr + col_offsets
            row = tl.load(row_ptrs, mask=mask_weight, other=0)
            # bitwise cast row from int32 to float32
            output += row.to(tl.float32, bitcast=True)

        output_fp32 = output.to(tl.float32)
        tl.store(output_row_ptrs, output_fp32, mask=col_offsets < D)

    elif weight_ty == 2:  # int8
        col_offsets = tl.arange(0, block_size_int8)
        # this tl.multiple_of hint is crucial to make the numerics right
        output_row_ptrs = tl.multiple_of(output_row_start_ptr + col_offsets, 4)

        q_BS: tl.constexpr = block_size_int8 >> 2
        q_col_offsets = tl.arange(0, q_BS)
        mask_weight = q_col_offsets < (weight_D - 1)

        output_q0 = tl.zeros((q_BS,), dtype=tl.float32)
        output_q1 = tl.zeros((q_BS,), dtype=tl.float32)
        output_q2 = tl.zeros((q_BS,), dtype=tl.float32)
        output_q3 = tl.zeros((q_BS,), dtype=tl.float32)

        for idx in range(start, end):
            row_idx = tl.load(indices_ptr + idx)
            row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
            row_ptrs = row_start_ptr + 1 + q_col_offsets
            row = tl.load(row_ptrs, mask=mask_weight, other=0).to(
                tl.uint32, bitcast=True
            )
            scale_shift = tl.load(row_start_ptr)
            scale = (scale_shift & 65535).to(tl.int16).to(tl.float16, bitcast=True)
            shift = (scale_shift >> 16).to(tl.int16).to(tl.float16, bitcast=True)

            scale_512 = (scale * 512.0).to(tl.float16)

            row_q0 = (
                (row & 255).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_q1 = (
                (row & 255).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_q2 = (
                (row & 255).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_q3 = (
                (row & 255).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)

            output_q0 += row_q0 * scale_512 + shift
            output_q1 += row_q1 * scale_512 + shift
            output_q2 += row_q2 * scale_512 + shift
            output_q3 += row_q3 * scale_512 + shift

        mask_output = q_col_offsets < (D // 4)
        output_row_ptrs_q0 = output_row_start_ptr + 4 * q_col_offsets
        tl.store(output_row_ptrs_q0, output_q0, mask=mask_output)

        output_row_ptrs_q1 = output_row_start_ptr + 4 * q_col_offsets + 1
        tl.store(output_row_ptrs_q1, output_q1, mask=mask_output)

        output_row_ptrs_q2 = output_row_start_ptr + 4 * q_col_offsets + 2
        tl.store(output_row_ptrs_q2, output_q2, mask=mask_output)

        output_row_ptrs_q3 = output_row_start_ptr + 4 * q_col_offsets + 3
        tl.store(output_row_ptrs_q3, output_q3, mask=mask_output)

    elif weight_ty == 3:  # int4
        col_offsets = tl.arange(0, block_size_int4)
        # this tl.multiple_of hint is crucial to make the numerics right
        output_row_ptrs = tl.multiple_of(output_row_start_ptr + col_offsets, 8)

        e_BS: tl.constexpr = block_size_int4 >> 3
        e_col_offsets = tl.arange(0, e_BS)
        mask_weight = e_col_offsets < (weight_D - 1)

        output_e0 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e1 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e2 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e3 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e4 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e5 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e6 = tl.zeros((e_BS,), dtype=tl.float32)
        output_e7 = tl.zeros((e_BS,), dtype=tl.float32)

        for idx in range(start, end):
            row_idx = tl.load(indices_ptr + idx)
            row_start_ptr = weight_ptr + table_offset + row_idx * weight_D
            row_ptrs = row_start_ptr + 1 + e_col_offsets
            row = tl.load(row_ptrs, mask=mask_weight, other=0).to(
                tl.uint32, bitcast=True
            )
            scale_shift = tl.load(row_start_ptr)
            scale = (scale_shift & 65535).to(tl.int16).to(tl.float16, bitcast=True)
            shift = (scale_shift >> 16).to(tl.int16).to(tl.float16, bitcast=True)

            scale_512 = (scale * 512.0).to(tl.float16)
            scale_32 = (scale * 32.0).to(tl.float16)

            row_e0 = (
                (row & 15).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e1 = (
                (row & 240).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_e2 = (
                (row & 15).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e3 = (
                (row & 240).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_e4 = (
                (row & 15).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e5 = (
                (row & 240).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row = row >> 8
            row_e6 = (
                (row & 15).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)
            row_e7 = (
                (row & 240).to(tl.uint16).to(tl.float16, bitcast=True) * 32768.0
            ).to(tl.float16)

            output_e0 += row_e0 * scale_512 + shift
            output_e1 += row_e1 * scale_32 + shift
            output_e2 += row_e2 * scale_512 + shift
            output_e3 += row_e3 * scale_32 + shift
            output_e4 += row_e4 * scale_512 + shift
            output_e5 += row_e5 * scale_32 + shift
            output_e6 += row_e6 * scale_512 + shift
            output_e7 += row_e7 * scale_32 + shift

        mask_output = e_col_offsets < (D // 8)
        output_row_ptrs_e0 = output_row_start_ptr + 8 * e_col_offsets
        tl.store(output_row_ptrs_e0, output_e0, mask=mask_output)

        output_row_ptrs_e1 = output_row_start_ptr + 8 * e_col_offsets + 1
        tl.store(output_row_ptrs_e1, output_e1, mask=mask_output)

        output_row_ptrs_e2 = output_row_start_ptr + 8 * e_col_offsets + 2
        tl.store(output_row_ptrs_e2, output_e2, mask=mask_output)

        output_row_ptrs_e3 = output_row_start_ptr + 8 * e_col_offsets + 3
        tl.store(output_row_ptrs_e3, output_e3, mask=mask_output)

        output_row_ptrs_e4 = output_row_start_ptr + 8 * e_col_offsets + 4
        tl.store(output_row_ptrs_e4, output_e4, mask=mask_output)

        output_row_ptrs_e5 = output_row_start_ptr + 8 * e_col_offsets + 5
        tl.store(output_row_ptrs_e5, output_e5, mask=mask_output)

        output_row_ptrs_e6 = output_row_start_ptr + 8 * e_col_offsets + 6
        tl.store(output_row_ptrs_e6, output_e6, mask=mask_output)

        output_row_ptrs_e7 = output_row_start_ptr + 8 * e_col_offsets + 7
        tl.store(output_row_ptrs_e7, output_e7, mask=mask_output)


# util function taken from split_table_batched_embeddings_ops.py
def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def rounded_row_size_in_bytes(
    dim: int, weight_ty: SparseType, row_alignment: int
) -> int:
    r = unpadded_row_size_in_bytes(dim, weight_ty)
    # align each row to 16-byte boundaries.
    return round_up(r, row_alignment)


def unpadded_row_size_in_bytes(dim: int, weight_ty: SparseType) -> int:
    r = {
        SparseType.FP32.value: dim * 4,
        SparseType.FP16.value: dim * 2,
        SparseType.FP8.value: dim,
        SparseType.INT8.value: dim + 4,
        SparseType.INT4.value: dim // 2 + 4,
        SparseType.INT2.value: dim // 4 + 4,
    }[weight_ty.value]
    return r


def align_to_cacheline(a: int) -> int:
    # align each table to 128b cache line boundary.
    return round_up(a, 128)


class TritonNBitTBE(torch.nn.Module):

    embedding_specs: List[Tuple[int, int, SparseType]]

    def __init__(
        self,
        embedding_specs: List[
            Tuple[int, int, SparseType]
        ],  # tuple of (rows, dims, dtype)
        output_dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        row_alignment: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding_specs = embedding_specs
        self.T = len(embedding_specs)  # num of tables

        if device is None:
            device = torch.device(torch.cuda.current_device())
        self.device = device

        self.row_alignment = row_alignment if row_alignment else 16

        rows = [spec[0] for spec in embedding_specs]
        Ds = [spec[1] for spec in embedding_specs]
        weight_tys = [spec[2] for spec in embedding_specs]

        self.rows_per_table = torch.tensor(rows, dtype=torch.int64, device=device)
        self.bounds_check_warning = torch.tensor([0], device=device, dtype=torch.int64)

        self.output_dtype = output_dtype

        Ds_bytes = [
            rounded_row_size_in_bytes(spec[1], spec[2], self.row_alignment)
            for spec in embedding_specs
        ]

        table_sizes_bytes = [
            align_to_cacheline(row * dim_bytes)
            for (row, dim_bytes) in zip(rows, Ds_bytes)
        ]

        self.table_offsets = torch.tensor(
            lengths_to_offsets(table_sizes_bytes), dtype=torch.int64, device=device
        )

        self.total_D = sum(Ds)
        self.weight_Ds = torch.tensor(Ds_bytes, dtype=torch.int64, device=device)
        self.D_offsets = torch.tensor(
            lengths_to_offsets(Ds, keep_last=True), dtype=torch.int64, device=device
        )

        total_weight_size = sum(table_sizes_bytes)

        self.weight = torch.empty([total_weight_size], dtype=torch.uint8, device=device)
        self.weight_in_4bytes = self.weight.view(torch.int32)
        self.weight_in_2bytes = self.weight.view(torch.int16)

        self.table_offsets_in_4bytes = self.table_offsets // 4
        self.weight_Ds_in_4bytes = self.weight_Ds // 4

        self.table_offsets_in_2bytes = self.table_offsets // 2
        self.weight_Ds_in_2bytes = self.weight_Ds // 2

        self.weight_tys = torch.tensor(
            [ty.as_int() for ty in weight_tys], dtype=torch.uint8, device=device
        )

        self.calc_block_sizes()

        self.num_warps = 1

    def calc_block_sizes(self) -> None:
        min_block_size = {
            SparseType.FP32: 64,
            SparseType.FP16: 64,
            SparseType.INT8: 128,
            SparseType.INT4: 256,
            SparseType.INT2: 256,
        }

        self.launch_32bits_kernel = False
        self.launch_16bits_kernel = False

        tys_with_32bits_kernel = [SparseType.FP32, SparseType.INT8, SparseType.INT4]
        tys_with_16bits_kernel = [SparseType.FP16, SparseType.INT2]

        def block_size_ty(ty: SparseType) -> int:
            max_ty_D = max(
                [spec[1] for spec in self.embedding_specs if spec[2] == ty],
                default=1,
            )
            if ty in tys_with_32bits_kernel and max_ty_D > 1:
                self.launch_32bits_kernel = True
            if ty in tys_with_16bits_kernel and max_ty_D > 1:
                self.launch_16bits_kernel = True
            return max(triton.next_power_of_2(max_ty_D), min_block_size[ty])

        self.block_size_fp32: int = block_size_ty(SparseType.FP32)
        self.block_size_fp16: int = block_size_ty(SparseType.FP16)
        self.block_size_int8: int = block_size_ty(SparseType.INT8)
        self.block_size_int4: int = block_size_ty(SparseType.INT4)
        self.block_size_int2: int = block_size_ty(SparseType.INT2)

    def copy_weight(self, weight) -> None:
        assert weight.dtype == self.weight.dtype
        assert weight.shape == self.weight.shape
        self.weight.copy_(weight)
        self.weight_in_4bytes = self.weight.view(torch.int32)
        self.weight_in_2bytes = self.weight.view(torch.int16)

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = (
            offsets.size(0) - 1
        ) // self.T  # batch_size, offsets is of size (1 + T * B)

        output = torch.empty(
            (B, self.total_D), device=self.device, dtype=self.output_dtype
        )

        num_programs = B * self.T

        if self.launch_32bits_kernel:
            nbit_32_kernel = (
                _amd_nbit_fwd_32bits if is_amd() else _nbit_TBE_forward_kernel_32bits
            )
            nbit_32_kernel[(num_programs,)](
                output,
                indices,
                offsets,
                self.weight_in_4bytes,
                self.table_offsets_in_4bytes,
                self.D_offsets,
                self.weight_Ds_in_4bytes,
                self.weight_tys,
                self.total_D,
                B,
                self.block_size_fp32,
                self.block_size_int8,
                self.block_size_int4,
                num_warps=self.num_warps,
            )

        if self.launch_16bits_kernel:
            nbit_16_kernel = (
                _amd_nbit_fwd_16bits if is_amd() else _nbit_TBE_forward_kernel_16bits
            )
            nbit_16_kernel[(num_programs,)](
                output,
                indices,
                offsets,
                self.weight_in_2bytes,
                self.table_offsets_in_2bytes,
                self.D_offsets,
                self.weight_Ds_in_2bytes,
                self.weight_tys,
                self.total_D,
                B,
                self.block_size_fp16,
                self.block_size_int2,
                num_warps=self.num_warps,
            )

        return output
