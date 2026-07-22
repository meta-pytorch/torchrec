#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import triton  # @manual
import triton.language as tl  # @manual
from torchrec.distributed.triton_tbe.triton_tbe_backward_utils import (
    _stochastic_rounding_store,
)

has_tlx = True
try:
    import triton.language.extra.tlx as tlx

except ImportError:
    has_tlx = False


@triton.jit
def triton_tbe_backward_long_run_fused_weighted(
    dout_ptr,
    infos_sorted_ptr,
    long_run_program_seg_starts_ptr,
    long_run_program_seg_ends_ptr,
    long_run_grad_buffer_ids_ptr,
    temp_grad_buffer_ptr,
    grad_accum_counter_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    per_sample_weights_ptr,
    weight_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    long_run_original_ids_ptr,
    table_offsets_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    num_long_run_programs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
    vbe: tl.constexpr = False,
) -> None:
    """
    Fused weighted long-run kernel: accumulates weighted partial gradients
    via atomic add, uses tlx.fence("gpu") + atomic counter so the last sub-program
    applies the optimizer update — eliminating a separate apply kernel launch.
    """
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    clc_phase_producer = 1
    clc_phase_consumer = 0
    clc_context = tlx.clc_create_context(1)

    tile_id = tl.program_id(0)
    num_tiles = tl.num_programs(0)
    num_long_run_programs = tl.load(num_long_run_programs_ptr)
    c_runs = num_long_run_programs // num_tiles
    r_runs = num_long_run_programs - c_runs * num_tiles

    has_more_tile = True
    while has_more_tile:
        tlx.clc_producer(clc_context, clc_phase_producer)
        clc_phase_producer ^= 1
        local_id = c_runs * tile_id + tl.maximum(tile_id - num_tiles + r_runs, 0)
        local_id_end = local_id + c_runs + (tile_id >= num_tiles - r_runs)

        while local_id < local_id_end:
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

            # Atomically accumulate partial gradient into the temp buffer
            temp_grad_offset = grad_buffer_id.to(tl.int64) * BLOCK_SIZE
            tl.atomic_add(
                temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
                grad,
                mask=mask,
            )

            tlx.fence("gpu")

            remaining = tl.atomic_add(grad_accum_counter_ptr + grad_buffer_id, -1)
            if remaining == 1:
                grad_original = tl.load(
                    temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
                    mask=mask,
                    other=0,
                )

                run_id = tl.load(long_run_original_ids_ptr + grad_buffer_id)
                linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
                seg_start = tl.load(
                    sorted_linear_indices_cumulative_run_lengths_ptr + run_id
                )
                info_0 = tl.load(infos_sorted_ptr + seg_start).to(tl.uint32)
                t_0 = (info_0 >> info_B_num_bits).to(tl.int32)

                table_idx = tl.load(feature_table_map_ptr + t_0)
                table_offset = tl.load(table_offsets_ptr + table_idx)
                emb_dim = tl.load(embedding_dims_ptr + t_0)
                apply_mask = col_offsets < emb_dim

                index_offset = tl.load(hash_size_cumsum_ptr + t_0)
                row_idx = linear_index - index_offset
                row_start_ptr = weight_ptr + table_offset + row_idx * emb_dim
                row_ptrs = row_start_ptr + col_offsets
                row = tl.load(row_ptrs, mask=apply_mask, other=0)

                row_update = row - learning_rate * grad_original

                if optimizer == 1:
                    row_offset = tl.load(rows_cumsum_ptr + table_idx)
                    momentum_idx = row_offset + row_idx

                    grad_square = grad_original * grad_original
                    grad_square_average = tl.sum(grad_square) / emb_dim

                    momentum = tl.load(momentum_ptr + momentum_idx)
                    momentum_new = momentum + grad_square_average
                    tl.store(momentum_ptr + momentum_idx, momentum_new)

                    adaptive_learning_rate = learning_rate / (
                        tl.sqrt(momentum_new) + eps
                    )

                    row_update = row - adaptive_learning_rate * grad_original

                if STOCHASTIC_ROUNDING:
                    sr_offset = grad_buffer_id * BLOCK_SIZE + col_offsets
                    _stochastic_rounding_store(
                        row_ptrs,
                        row_update,
                        apply_mask,
                        stochastic_rounding_seed,
                        sr_offset,
                    )
                else:
                    tl.store(row_ptrs, row_update, mask=apply_mask)

            local_id += 1

        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
        clc_phase_consumer ^= 1
        has_more_tile = tile_id != -1


@triton.jit
def triton_tbe_backward_long_run_fused_unweighted(
    dout_ptr,
    infos_sorted_ptr,
    long_run_program_seg_starts_ptr,
    long_run_program_seg_ends_ptr,
    long_run_grad_buffer_ids_ptr,
    temp_grad_buffer_ptr,
    grad_accum_counter_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    weight_ptr,
    sorted_linear_indices_run_ptr,
    sorted_linear_indices_cumulative_run_lengths_ptr,
    long_run_original_ids_ptr,
    table_offsets_ptr,
    feature_table_map_ptr,
    hash_size_cumsum_ptr,
    momentum_ptr,
    rows_cumsum_ptr,
    num_long_run_programs_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    learning_rate,
    eps,
    optimizer: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    info_B_num_bits,
    info_B_mask,
    STOCHASTIC_ROUNDING: tl.constexpr,
    stochastic_rounding_seed,
    vbe: tl.constexpr = False,
) -> None:
    """
    Fused long-run kernel: accumulates partial gradients via atomic add,
    uses tlx.fence("gpu") + atomic counter so the last sub-program applies
    the optimizer update — eliminating a separate apply kernel launch.
    """
    col_offsets = tl.arange(0, BLOCK_SIZE)
    buffer_size: tl.constexpr = 16
    buffer_offsets = tl.arange(0, buffer_size)

    clc_phase_producer = 1
    clc_phase_consumer = 0
    clc_context = tlx.clc_create_context(1)

    tile_id = tl.program_id(0)
    num_tiles = tl.num_programs(0)
    num_long_run_programs = tl.load(num_long_run_programs_ptr)
    c_runs = num_long_run_programs // num_tiles
    r_runs = num_long_run_programs - c_runs * num_tiles

    has_more_tile = True
    while has_more_tile:
        tlx.clc_producer(clc_context, clc_phase_producer)
        clc_phase_producer ^= 1
        local_id = c_runs * tile_id + tl.maximum(tile_id - num_tiles + r_runs, 0)
        local_id_end = local_id + c_runs + (tile_id >= num_tiles - r_runs)

        while local_id < local_id_end:
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

            temp_grad_offset = grad_buffer_id.to(tl.int64) * BLOCK_SIZE
            tl.atomic_add(
                temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
                grad,
                mask=mask,
            )

            tlx.fence("gpu")

            remaining = tl.atomic_add(grad_accum_counter_ptr + grad_buffer_id, -1)
            if remaining == 1:
                grad_original = tl.load(
                    temp_grad_buffer_ptr + temp_grad_offset + col_offsets,
                    mask=mask,
                    other=0,
                )

                run_id = tl.load(long_run_original_ids_ptr + grad_buffer_id)
                linear_index = tl.load(sorted_linear_indices_run_ptr + run_id)
                seg_start = tl.load(
                    sorted_linear_indices_cumulative_run_lengths_ptr + run_id
                )
                info_0 = tl.load(infos_sorted_ptr + seg_start).to(tl.uint32)
                t_0 = (info_0 >> info_B_num_bits).to(tl.int32)

                table_idx = tl.load(feature_table_map_ptr + t_0)
                table_offset = tl.load(table_offsets_ptr + table_idx)
                emb_dim = tl.load(embedding_dims_ptr + t_0)
                apply_mask = col_offsets < emb_dim

                index_offset = tl.load(hash_size_cumsum_ptr + t_0)
                row_idx = linear_index - index_offset
                row_start_ptr = weight_ptr + table_offset + row_idx * emb_dim
                row_ptrs = row_start_ptr + col_offsets
                row = tl.load(row_ptrs, mask=apply_mask, other=0)

                row_update = row - learning_rate * grad_original

                if optimizer == 1:
                    row_offset = tl.load(rows_cumsum_ptr + table_idx)
                    momentum_idx = row_offset + row_idx

                    grad_square = grad_original * grad_original
                    grad_square_average = tl.sum(grad_square) / emb_dim

                    momentum = tl.load(momentum_ptr + momentum_idx)
                    momentum_new = momentum + grad_square_average
                    tl.store(momentum_ptr + momentum_idx, momentum_new)

                    adaptive_learning_rate = learning_rate / (
                        tl.sqrt(momentum_new) + eps
                    )

                    row_update = row - adaptive_learning_rate * grad_original

                if STOCHASTIC_ROUNDING:
                    sr_offset = grad_buffer_id * BLOCK_SIZE + col_offsets
                    _stochastic_rounding_store(
                        row_ptrs,
                        row_update,
                        apply_mask,
                        stochastic_rounding_seed,
                        sr_offset,
                    )
                else:
                    tl.store(row_ptrs, row_update, mask=apply_mask)

            local_id += 1

        tile_id = tlx.clc_consumer(clc_context, clc_phase_consumer)
        clc_phase_consumer ^= 1
        has_more_tile = tile_id != -1
