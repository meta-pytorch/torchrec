#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any

import torch
import triton  # @manual
import triton.language as tl  # @manual
from fbgemm_gpu.split_table_batched_embeddings_ops_common import CacheAlgorithm
from torchrec.distributed.triton_tbe.triton_table_batched_embeddings import (
    _load_checked_index,
    lengths_to_offsets,
)
from torchrec.distributed.triton_tbe.triton_uvm_table_batched_embeddings import (
    TritonUVMTableBatchedEmbeddingBags,
)


@triton.jit
def _load_cached_contiguous_weight_row(
    weight_ptr,
    cache_weights_ptr,
    cache_locations_ptr,
    index_position,
    cache_row_width: tl.constexpr,
    logical_row_start,
    col_offsets,
    mask,
):
    cache_location = tl.load(cache_locations_ptr + index_position)
    cache_hit = cache_location >= 0
    cached_row = tl.load(
        cache_weights_ptr + cache_location.to(tl.int64) * cache_row_width + col_offsets,
        mask=mask & cache_hit,
        other=0,
    )
    uvm_row = tl.load(
        weight_ptr + logical_row_start + col_offsets,
        mask=mask & ~cache_hit,
        other=0,
    )
    return tl.where(cache_hit, cached_row, uvm_row)


@triton.jit
def table_batched_embedding_bag_forward_weighted_cached_kernel(  # noqa: TR001
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    per_sample_weights_ptr,
    cache_weights_ptr,
    cache_locations_ptr,
    # VBE-specific pointers (only used when vbe=T)
    # pyre-fixme[2]: Parameter must be annotated.
    row_output_offsets_ptr,
    # pyre-fixme[2]: Parameter must be annotated.
    b_t_map_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    BLOCK_SIZE: tl.constexpr,
    CACHE_ROW_WIDTH: tl.constexpr,
    vbe: tl.constexpr = False,
    info_B_num_bits=0,
    info_B_mask=0,
    ENABLE_TRITON_TBE_OPTIMIZATIONS: tl.constexpr = False,
):

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
    accumulator_dtype: tl.constexpr = (
        tl.float32
        if ENABLE_TRITON_TBE_OPTIMIZATIONS or weight_ptr.dtype.element_ty != tl.float32
        else tl.float64
    )
    bag_output = tl.zeros((BLOCK_SIZE,), dtype=accumulator_dtype)

    # without type hint the unrolling performance will downgrade
    step: tl.constexpr = 4
    ns = (end - start) // step
    endn = start + step * ns

    for idx in range(start, endn, step):
        row_idx_0 = tl.load(indices_ptr + idx + 0)
        row_idx_1 = tl.load(indices_ptr + idx + 1)
        row_idx_2 = tl.load(indices_ptr + idx + 2)
        row_idx_3 = tl.load(indices_ptr + idx + 3)

        row_0 = _load_cached_contiguous_weight_row(
            weight_ptr,
            cache_weights_ptr,
            cache_locations_ptr,
            idx + 0,
            CACHE_ROW_WIDTH,
            table_offset + row_idx_0 * embedding_dim,
            col_offsets,
            mask,
        )
        row_1 = _load_cached_contiguous_weight_row(
            weight_ptr,
            cache_weights_ptr,
            cache_locations_ptr,
            idx + 1,
            CACHE_ROW_WIDTH,
            table_offset + row_idx_1 * embedding_dim,
            col_offsets,
            mask,
        )
        row_2 = _load_cached_contiguous_weight_row(
            weight_ptr,
            cache_weights_ptr,
            cache_locations_ptr,
            idx + 2,
            CACHE_ROW_WIDTH,
            table_offset + row_idx_2 * embedding_dim,
            col_offsets,
            mask,
        )
        row_3 = _load_cached_contiguous_weight_row(
            weight_ptr,
            cache_weights_ptr,
            cache_locations_ptr,
            idx + 3,
            CACHE_ROW_WIDTH,
            table_offset + row_idx_3 * embedding_dim,
            col_offsets,
            mask,
        )

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
        row = _load_cached_contiguous_weight_row(
            weight_ptr,
            cache_weights_ptr,
            cache_locations_ptr,
            idx,
            CACHE_ROW_WIDTH,
            table_offset + row_idx * embedding_dim,
            col_offsets,
            mask,
        )

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
# Triton TR001: BLOCK_SIZE is fixed by the embedding width.
def table_batched_embedding_bag_forward_unweighted_cached_kernel(  # noqa: C901, TR001
    output_ptr,
    indices_ptr,
    offsets_ptr,
    weight_ptr,
    table_offsets_ptr,
    embedding_dims_ptr,
    embedding_offsets_ptr,
    feature_table_map_ptr,
    rows_cumsum_ptr,
    bounds_check_warning_ptr,
    cache_weights_ptr,
    cache_locations_ptr,
    row_output_offsets_ptr,
    B_offsets_ptr,
    total_embedding_dim: tl.constexpr,
    B,
    T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_ROW_WIDTH: tl.constexpr,
    vbe: tl.constexpr = False,
    FEATURE_START: tl.constexpr = 0,
    FEATURE_END: tl.constexpr = -1,
    BAGS_PER_PROGRAM: tl.constexpr = 1,
    UNROLL8: tl.constexpr = False,
    FUSED_BOUNDS_CHECK: tl.constexpr = False,
    ENABLE_TRITON_TBE_OPTIMIZATIONS: tl.constexpr = False,
) -> None:
    base_b = tl.program_id(0).to(tl.int64) * BAGS_PER_PROGRAM
    col_offsets = tl.arange(0, BLOCK_SIZE)
    warning_count = 0

    feature_end: tl.constexpr = T if FEATURE_END < 0 else FEATURE_END
    for t in range(FEATURE_START, feature_end):
        table_idx = tl.load(feature_table_map_ptr + t)
        table_offset = tl.load(table_offsets_ptr + table_idx)
        embedding_dim = tl.load(embedding_dims_ptr + t)
        embedding_offset = tl.load(embedding_offsets_ptr + t)
        if FUSED_BOUNDS_CHECK:
            num_rows = tl.load(rows_cumsum_ptr + table_idx + 1) - tl.load(
                rows_cumsum_ptr + table_idx
            )

        if vbe:
            B_start = tl.load(B_offsets_ptr + t).to(tl.int64)
            B_end = tl.load(B_offsets_ptr + t + 1).to(tl.int64)
            B_t = B_end - B_start

        for bag_slot in tl.static_range(0, BAGS_PER_PROGRAM):
            b = base_b + bag_slot
            if vbe:
                b_t = B_start + b
                in_bounds = b < B_t
            else:
                b_t = t * B + b
                in_bounds = b < B

            if in_bounds:
                start = tl.load(offsets_ptr + b_t)
                end = tl.load(offsets_ptr + b_t + 1)
                mask = col_offsets < embedding_dim
                accumulator_dtype: tl.constexpr = (
                    tl.float32
                    if ENABLE_TRITON_TBE_OPTIMIZATIONS
                    or weight_ptr.dtype.element_ty != tl.float32
                    else tl.float64
                )
                bag_output = tl.zeros((BLOCK_SIZE,), dtype=accumulator_dtype)

                step: tl.constexpr = 8 if UNROLL8 else 4
                ns = (end - start) // step
                endn = start + step * ns

                for idx in range(start, endn, step):
                    row_idx_0, invalid_0 = _load_checked_index(
                        indices_ptr,
                        idx + 0,
                        num_rows if FUSED_BOUNDS_CHECK else 0,
                        True,
                        FUSED_BOUNDS_CHECK,
                    )
                    row_idx_1, invalid_1 = _load_checked_index(
                        indices_ptr,
                        idx + 1,
                        num_rows if FUSED_BOUNDS_CHECK else 0,
                        True,
                        FUSED_BOUNDS_CHECK,
                    )
                    row_idx_2, invalid_2 = _load_checked_index(
                        indices_ptr,
                        idx + 2,
                        num_rows if FUSED_BOUNDS_CHECK else 0,
                        True,
                        FUSED_BOUNDS_CHECK,
                    )
                    row_idx_3, invalid_3 = _load_checked_index(
                        indices_ptr,
                        idx + 3,
                        num_rows if FUSED_BOUNDS_CHECK else 0,
                        True,
                        FUSED_BOUNDS_CHECK,
                    )
                    if FUSED_BOUNDS_CHECK:
                        warning_count += (
                            invalid_0.to(tl.int32)
                            + invalid_1.to(tl.int32)
                            + invalid_2.to(tl.int32)
                            + invalid_3.to(tl.int32)
                        )
                    if UNROLL8:
                        row_idx_4, invalid_4 = _load_checked_index(
                            indices_ptr,
                            idx + 4,
                            num_rows if FUSED_BOUNDS_CHECK else 0,
                            True,
                            FUSED_BOUNDS_CHECK,
                        )
                        row_idx_5, invalid_5 = _load_checked_index(
                            indices_ptr,
                            idx + 5,
                            num_rows if FUSED_BOUNDS_CHECK else 0,
                            True,
                            FUSED_BOUNDS_CHECK,
                        )
                        row_idx_6, invalid_6 = _load_checked_index(
                            indices_ptr,
                            idx + 6,
                            num_rows if FUSED_BOUNDS_CHECK else 0,
                            True,
                            FUSED_BOUNDS_CHECK,
                        )
                        row_idx_7, invalid_7 = _load_checked_index(
                            indices_ptr,
                            idx + 7,
                            num_rows if FUSED_BOUNDS_CHECK else 0,
                            True,
                            FUSED_BOUNDS_CHECK,
                        )
                        if FUSED_BOUNDS_CHECK:
                            warning_count += (
                                invalid_4.to(tl.int32)
                                + invalid_5.to(tl.int32)
                                + invalid_6.to(tl.int32)
                                + invalid_7.to(tl.int32)
                            )
                    row_0 = _load_cached_contiguous_weight_row(
                        weight_ptr,
                        cache_weights_ptr,
                        cache_locations_ptr,
                        idx + 0,
                        CACHE_ROW_WIDTH,
                        table_offset + row_idx_0 * embedding_dim,
                        col_offsets,
                        mask,
                    )
                    row_1 = _load_cached_contiguous_weight_row(
                        weight_ptr,
                        cache_weights_ptr,
                        cache_locations_ptr,
                        idx + 1,
                        CACHE_ROW_WIDTH,
                        table_offset + row_idx_1 * embedding_dim,
                        col_offsets,
                        mask,
                    )
                    row_2 = _load_cached_contiguous_weight_row(
                        weight_ptr,
                        cache_weights_ptr,
                        cache_locations_ptr,
                        idx + 2,
                        CACHE_ROW_WIDTH,
                        table_offset + row_idx_2 * embedding_dim,
                        col_offsets,
                        mask,
                    )
                    row_3 = _load_cached_contiguous_weight_row(
                        weight_ptr,
                        cache_weights_ptr,
                        cache_locations_ptr,
                        idx + 3,
                        CACHE_ROW_WIDTH,
                        table_offset + row_idx_3 * embedding_dim,
                        col_offsets,
                        mask,
                    )
                    if UNROLL8:
                        row_4 = _load_cached_contiguous_weight_row(
                            weight_ptr,
                            cache_weights_ptr,
                            cache_locations_ptr,
                            idx + 4,
                            CACHE_ROW_WIDTH,
                            table_offset + row_idx_4 * embedding_dim,
                            col_offsets,
                            mask,
                        )
                        row_5 = _load_cached_contiguous_weight_row(
                            weight_ptr,
                            cache_weights_ptr,
                            cache_locations_ptr,
                            idx + 5,
                            CACHE_ROW_WIDTH,
                            table_offset + row_idx_5 * embedding_dim,
                            col_offsets,
                            mask,
                        )
                        row_6 = _load_cached_contiguous_weight_row(
                            weight_ptr,
                            cache_weights_ptr,
                            cache_locations_ptr,
                            idx + 6,
                            CACHE_ROW_WIDTH,
                            table_offset + row_idx_6 * embedding_dim,
                            col_offsets,
                            mask,
                        )
                        row_7 = _load_cached_contiguous_weight_row(
                            weight_ptr,
                            cache_weights_ptr,
                            cache_locations_ptr,
                            idx + 7,
                            CACHE_ROW_WIDTH,
                            table_offset + row_idx_7 * embedding_dim,
                            col_offsets,
                            mask,
                        )
                    bag_output += (
                        row_0.to(tl.float32)
                        + row_1.to(tl.float32)
                        + row_2.to(tl.float32)
                        + row_3.to(tl.float32)
                    )
                    if UNROLL8:
                        bag_output += (
                            row_4.to(tl.float32)
                            + row_5.to(tl.float32)
                            + row_6.to(tl.float32)
                            + row_7.to(tl.float32)
                        )

                for idx in range(endn, end):
                    row_idx, invalid = _load_checked_index(
                        indices_ptr,
                        idx,
                        num_rows if FUSED_BOUNDS_CHECK else 0,
                        True,
                        FUSED_BOUNDS_CHECK,
                    )
                    if FUSED_BOUNDS_CHECK:
                        warning_count += invalid.to(tl.int32)
                    row = _load_cached_contiguous_weight_row(
                        weight_ptr,
                        cache_weights_ptr,
                        cache_locations_ptr,
                        idx,
                        CACHE_ROW_WIDTH,
                        table_offset + row_idx * embedding_dim,
                        col_offsets,
                        mask,
                    )
                    bag_output += row.to(tl.float32)

                if vbe:
                    row_output_offset = tl.load(row_output_offsets_ptr + b_t)
                    output_row_ptrs = output_ptr + row_output_offset + col_offsets
                else:
                    output_row_ptrs = (
                        output_ptr
                        + b * total_embedding_dim
                        + embedding_offset
                        + col_offsets
                    )
                tl.store(output_row_ptrs, bag_output.to(tl.float32), mask=mask)

    if FUSED_BOUNDS_CHECK and warning_count > 0:
        tl.atomic_add(bounds_check_warning_ptr, warning_count.to(tl.int64))


def _run_uvm_caching_forward(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    table_offsets: torch.Tensor,
    embedding_dims: torch.Tensor,
    embedding_offsets: torch.Tensor,
    feature_table_map: torch.Tensor,
    rows_cumsum: torch.Tensor,
    total_embedding_dim: int,
    num_features: int,
    block_size: int,
    per_sample_weights: torch.Tensor | None,
    forward_event_callback: Any,
    output_dtype: torch.dtype,
    batch_size_per_feature_per_rank: list[list[int]] | None,
    vbe_metadata: Any | None,
    row_output_offsets: torch.Tensor | None,
    b_t_map: torch.Tensor | None,
    info_B_num_bits: int,
    info_B_mask: int,
    total_B: int,
    max_B: int,
    bounds_check_warning: torch.Tensor,
    fused_bounds_check: bool,
    enable_triton_tbe_optimizations: bool,
    cache_weights: torch.Tensor,
    cache_locations: torch.Tensor,
) -> torch.Tensor:
    vbe = batch_size_per_feature_per_rank is not None
    if vbe:
        assert vbe_metadata is not None
        assert vbe_metadata.B_offsets is not None
        assert row_output_offsets is not None
        assert b_t_map is not None
        batch_size = max_B
        output = torch.empty(
            (vbe_metadata.output_size,), device=weight.device, dtype=output_dtype
        )
        row_output_offsets_ptr = row_output_offsets
        b_t_map_ptr = b_t_map
        B_offsets_ptr = vbe_metadata.B_offsets
    else:
        batch_size = (offsets.size(0) - 1) // num_features
        total_B = batch_size * num_features
        output = torch.empty(
            (batch_size, total_embedding_dim),
            device=weight.device,
            dtype=output_dtype,
        )
        row_output_offsets_ptr = torch.empty(0, device=weight.device, dtype=torch.int64)
        b_t_map_ptr = torch.empty(0, device=weight.device, dtype=torch.int32)
        B_offsets_ptr = torch.empty(0, device=weight.device, dtype=torch.int32)

    if cache_locations.numel() != indices.numel():
        raise ValueError("cache_locations must contain one entry per index")
    weighted = per_sample_weights is not None and per_sample_weights.numel() > 0
    if fused_bounds_check and (weighted or vbe):
        raise ValueError("Invalid fused bounds-check configuration")

    if weighted:
        table_batched_embedding_bag_forward_weighted_cached_kernel[(total_B,)](
            output,
            indices,
            offsets,
            weight,
            table_offsets,
            embedding_dims,
            embedding_offsets,
            feature_table_map,
            per_sample_weights,
            cache_weights,
            cache_locations,
            row_output_offsets_ptr,
            b_t_map_ptr,
            total_embedding_dim,
            batch_size,
            BLOCK_SIZE=block_size,
            CACHE_ROW_WIDTH=cache_weights.size(1),
            vbe=vbe,
            info_B_num_bits=info_B_num_bits,
            info_B_mask=info_B_mask,
            ENABLE_TRITON_TBE_OPTIMIZATIONS=enable_triton_tbe_optimizations,
            num_warps=1,
        )
    else:
        bags_per_program = (
            2
            if enable_triton_tbe_optimizations
            and not vbe
            and batch_size >= 65536
            and weight.dtype != torch.float32
            else 1
        )
        table_batched_embedding_bag_forward_unweighted_cached_kernel[
            (triton.cdiv(batch_size, bags_per_program),)
        ](
            output,
            indices,
            offsets,
            weight,
            table_offsets,
            embedding_dims,
            embedding_offsets,
            feature_table_map,
            rows_cumsum,
            bounds_check_warning,
            cache_weights,
            cache_locations,
            row_output_offsets_ptr,
            B_offsets_ptr,
            total_embedding_dim,
            batch_size,
            num_features,
            BLOCK_SIZE=block_size,
            CACHE_ROW_WIDTH=cache_weights.size(1),
            vbe=vbe,
            FEATURE_START=0,
            FEATURE_END=num_features,
            BAGS_PER_PROGRAM=bags_per_program,
            UNROLL8=bags_per_program == 2,
            FUSED_BOUNDS_CHECK=fused_bounds_check,
            ENABLE_TRITON_TBE_OPTIMIZATIONS=enable_triton_tbe_optimizations,
            num_warps=1,
        )

    if forward_event_callback is not None:
        forward_event_callback()
    return output


@triton.jit
# Triton TR001: CACHE_ASSOC is a fixed cache-layout dimension.
def _triton_uvm_cache_reserve_lru_kernel(  # noqa: TR001
    linear_cache_indices_ptr,
    lxu_cache_state_ptr,
    lxu_state_ptr,
    cache_admission_counter_ptr,
    cache_miss_counter_ptr,
    num_indices,
    total_cache_hash_size,
    cache_sets,
    timestep,
    CACHE_ASSOC: tl.constexpr,
    CACHE_LOCATION_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    if tl.load(cache_miss_counter_ptr) == 0:
        return
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    position_offsets = tl.arange(0, BLOCK_N)
    ways = tl.arange(0, CACHE_ASSOC)
    while block_start < num_indices:
        positions = block_start + position_offsets
        valid = positions < num_indices
        linear_indices = tl.load(
            linear_cache_indices_ptr + positions,
            mask=valid,
            other=total_cache_hash_size,
        )
        valid &= (linear_indices >= 0) & (linear_indices < total_cache_hash_size)
        cache_set = tl.where(valid, linear_indices % cache_sets, 0)
        # Triton TR003: CACHE_ASSOC exactly spans an in-bounds cache set.
        states = tl.load(  # noqa: TR003
            lxu_cache_state_ptr + cache_set[:, None] * CACHE_ASSOC + ways[None, :],
            mask=valid[:, None],
            other=-1,
        )
        is_resident = (
            tl.sum((states == linear_indices[:, None]).to(tl.int32), axis=1) > 0
        )
        tickets = tl.atomic_add(
            cache_admission_counter_ptr + cache_set,
            1,
            mask=valid & ~is_resident,
            sem="relaxed",
        )
        ticket_valid = valid & ~is_resident & (tickets < CACHE_ASSOC)
        # Triton TR003: CACHE_ASSOC exactly spans an in-bounds cache set.
        last_access_by_way = tl.load(  # noqa: TR003
            lxu_state_ptr + cache_set[:, None] * CACHE_ASSOC + ways[None, :],
            mask=valid[:, None],
            other=timestep,
        )
        available = (states == -1) | ((states >= 0) & (last_access_by_way < timestep))
        victim_keys = tl.where(
            states == -1,
            ways[None, :],
            (last_access_by_way + 1) * CACHE_ASSOC + ways[None, :],
        )
        victim_keys = tl.where(available, victim_keys, 1 << 62)
        sorted_victim_keys = tl.sort(victim_keys, dim=1)
        victim_key = tl.sum(
            tl.where(ways[None, :] == tickets[:, None], sorted_victim_keys, 0),
            axis=1,
        )
        victim_locations = cache_set * CACHE_ASSOC + victim_key % CACHE_ASSOC
        ticket_valid &= victim_key < (1 << 62)
        expected_states = tl.load(
            lxu_cache_state_ptr + victim_locations,
            mask=ticket_valid,
            other=-2,
        )
        should_insert = ticket_valid & (expected_states >= -1)
        tl.atomic_xchg(
            lxu_cache_state_ptr + victim_locations,
            -(linear_indices + 2),
            mask=should_insert,
            sem="acq_rel",
        )
        work_items = (linear_indices << CACHE_LOCATION_BITS) | victim_locations.to(
            tl.int64
        )
        tl.store(
            linear_cache_indices_ptr + positions,
            tl.where(should_insert, work_items, -1),
            mask=valid,
        )
        block_start += block_stride


@triton.jit
# Triton TR001: CACHE_ASSOC is a fixed cache-layout dimension.
def _triton_uvm_cache_reserve_kernel(  # noqa: TR001
    linear_cache_indices_ptr,
    linear_cache_indices_length_ptr,
    linear_cache_indices_count_ptr,
    cache_commit_locations_ptr,
    lfu_frequency_ptr,
    lxu_cache_state_ptr,
    lxu_state_ptr,
    lfu_cache_timestamps_ptr,
    cache_admission_counter_ptr,
    cache_miss_counter_ptr,
    total_cache_hash_size,
    cache_sets,
    timestep,
    CACHE_ASSOC: tl.constexpr,
    CACHE_LOCATION_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    num_indices = tl.load(linear_cache_indices_length_ptr)
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    position_offsets = tl.arange(0, BLOCK_N)
    ways = tl.arange(0, CACHE_ASSOC)
    while block_start < num_indices:
        positions = block_start + position_offsets
        valid_positions = positions < num_indices
        linear_indices = tl.load(
            linear_cache_indices_ptr + positions,
            mask=valid_positions,
            other=total_cache_hash_size,
        )
        valid = (
            valid_positions
            & (linear_indices >= 0)
            & (linear_indices < total_cache_hash_size)
        )
        request_counts = tl.load(
            linear_cache_indices_count_ptr + positions,
            mask=valid,
            other=0,
        )
        # get_unique_indices guarantees one writer per LFU counter.
        request_frequencies = tl.load(
            lfu_frequency_ptr + linear_indices,
            mask=valid,
            other=0,
        )
        request_frequencies += request_counts
        tl.store(
            lfu_frequency_ptr + linear_indices,
            request_frequencies,
            mask=valid,
        )
        cache_set = tl.where(valid, linear_indices % cache_sets, 0)
        state_offsets = cache_set[:, None] * CACHE_ASSOC + ways[None, :]
        # Triton TR003: CACHE_ASSOC exactly spans an in-bounds cache set.
        states = tl.load(  # noqa: TR003
            lxu_cache_state_ptr + state_offsets,
            mask=valid[:, None],
            other=-1,
        )
        matched_way = (
            tl.max(
                tl.where(states == linear_indices[:, None], ways[None, :] + 1, 0),
                axis=1,
            )
            - 1
        )
        is_resident = matched_way >= 0
        resident_locations = cache_set * CACHE_ASSOC + matched_way
        tl.store(
            lxu_state_ptr + resident_locations,
            request_frequencies,
            mask=valid & is_resident,
        )
        tl.atomic_max(
            lfu_cache_timestamps_ptr + resident_locations,
            timestep,
            mask=valid & is_resident,
            sem="relaxed",
        )
        tickets = tl.atomic_add(
            cache_admission_counter_ptr + cache_set,
            1,
            mask=valid & ~is_resident,
            sem="relaxed",
        )
        segment_size: tl.constexpr = 8
        num_segments: tl.constexpr = CACHE_ASSOC // segment_size
        ticket_valid = valid & ~is_resident & (tickets < CACHE_ASSOC)
        # Interleaving tickets gives each admission a distinct segment/rank victim.
        segment = tickets % num_segments
        segment_rank = tickets // num_segments
        segment_ways = tl.arange(0, segment_size)
        candidate_ways = segment[:, None] * segment_size + segment_ways[None, :]
        candidate_locations = cache_set[:, None] * CACHE_ASSOC + candidate_ways
        # Triton TR003: segment_size exactly spans an in-bounds cache segment.
        frequencies = tl.load(  # noqa: TR003
            lxu_state_ptr + candidate_locations,
            mask=ticket_valid[:, None],
            other=0,
        ).to(tl.int64)
        # Triton TR003: segment_size exactly spans an in-bounds cache segment.
        last_access = tl.load(  # noqa: TR003
            lfu_cache_timestamps_ptr + candidate_locations,
            mask=ticket_valid[:, None],
            other=timestep,
        )
        available = last_access < timestep
        victim_keys = (frequencies + 1) * segment_size + segment_ways[None, :]
        victim_keys = tl.where(available, victim_keys, 1 << 62)
        sorted_victim_keys = tl.sort(victim_keys, dim=1)
        victim_key = tl.sum(
            tl.where(
                segment_ways[None, :] == segment_rank[:, None],
                sorted_victim_keys,
                0,
            ),
            axis=1,
        )
        victim_way = segment * segment_size + victim_key % segment_size
        victim_frequency = victim_key // segment_size - 1
        victim_locations = cache_set * CACHE_ASSOC + victim_way
        ticket_valid &= victim_key < (1 << 62)
        should_insert = ticket_valid & (
            (victim_frequency < 0) | (request_frequencies >= victim_frequency)
        )
        tl.atomic_xchg(
            lxu_cache_state_ptr + victim_locations,
            -(linear_indices + 2),
            mask=should_insert,
            sem="acq_rel",
        )
        tl.store(
            lxu_state_ptr + victim_locations,
            request_frequencies,
            mask=should_insert,
        )
        work_items = (linear_indices << CACHE_LOCATION_BITS) | victim_locations.to(
            tl.int64
        )
        tl.store(
            linear_cache_indices_ptr + positions,
            tl.where(should_insert, work_items, -1),
            mask=valid,
        )
        insert_flags = should_insert.to(tl.int32)
        num_inserts = tl.sum(insert_flags, axis=0)
        insert_base = tl.atomic_add(cache_miss_counter_ptr, num_inserts)
        insert_offsets = tl.cumsum(insert_flags, axis=0) - 1
        tl.store(
            cache_commit_locations_ptr + insert_base + insert_offsets,
            victim_locations,
            mask=should_insert,
        )
        block_start += block_stride


@triton.jit
# Triton TR001: BLOCK_D is a fixed cache-row dimension.
def _triton_uvm_cache_copy_kernel(  # noqa: TR001
    weight_ptr,
    cache_index_table_map_ptr,
    cache_hash_size_cumsum_ptr,
    cache_weights_offsets_ptr,
    D_offsets_ptr,
    cache_work_items_ptr,
    lxu_cache_state_ptr,
    lxu_cache_weights_ptr,
    cache_miss_counter_ptr,
    num_cache_locations_ptr,
    cache_rows,
    total_cache_hash_size,
    CACHE_ROW_WIDTH: tl.constexpr,
    CACHE_LOCATION_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    num_misses = tl.load(cache_miss_counter_ptr)
    if num_misses == 0:
        return
    num_cache_locations = tl.load(num_cache_locations_ptr)
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    row_offsets = tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_D)
    cache_location_mask = (1 << CACHE_LOCATION_BITS) - 1
    while block_start < num_cache_locations:
        positions = block_start + row_offsets
        valid_positions = positions < num_cache_locations
        work_items = tl.load(
            cache_work_items_ptr + positions,
            mask=valid_positions,
            other=-1,
        )
        cache_locations = work_items & cache_location_mask
        linear_indices = work_items >> CACHE_LOCATION_BITS
        valid_locations = (
            valid_positions
            & (work_items >= 0)
            & (cache_locations < cache_rows)
            & (linear_indices >= 0)
            & (linear_indices < total_cache_hash_size)
        )
        lock_states = tl.load(
            lxu_cache_state_ptr + cache_locations,
            mask=valid_locations,
            other=-1,
        )
        expected_lock_states = -(linear_indices + 2)
        acquired = valid_locations & (lock_states == expected_lock_states)
        feature = tl.load(
            cache_index_table_map_ptr + linear_indices,
            mask=acquired,
            other=0,
        ).to(tl.int64)
        feature_row_start = tl.load(
            cache_hash_size_cumsum_ptr + feature,
            mask=acquired,
            other=0,
        )
        weight_offset = tl.load(
            cache_weights_offsets_ptr + feature,
            mask=acquired,
            other=0,
        )
        embedding_dim = tl.load(
            D_offsets_ptr + feature + 1,
            mask=acquired,
            other=0,
        ) - tl.load(D_offsets_ptr + feature, mask=acquired, other=0)
        row = linear_indices - feature_row_start
        values = tl.load(
            weight_ptr
            + weight_offset[:, None]
            + row[:, None] * embedding_dim[:, None]
            + columns[None, :],
            mask=acquired[:, None] & (columns[None, :] < embedding_dim[:, None]),
            other=0,
        )
        tl.store(
            lxu_cache_weights_ptr
            + cache_locations[:, None] * CACHE_ROW_WIDTH
            + columns[None, :],
            values,
            mask=acquired[:, None] & (columns[None, :] < embedding_dim[:, None]),
        )
        block_start += block_stride


@triton.jit
# Triton TR001: BLOCK_N is a fixed compact commit tile.
def _triton_uvm_cache_commit_kernel(  # noqa: TR001
    cache_commit_locations_ptr,
    lxu_cache_state_ptr,
    cache_miss_counter_ptr,
    BLOCK_N: tl.constexpr,
) -> None:
    num_misses = tl.load(cache_miss_counter_ptr)
    if num_misses == 0:
        return
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    row_offsets = tl.arange(0, BLOCK_N)
    while block_start < num_misses:
        positions = block_start + row_offsets
        valid_positions = positions < num_misses
        cache_locations = tl.load(
            cache_commit_locations_ptr + positions,
            mask=valid_positions,
            other=0,
        )
        lock_states = tl.load(
            lxu_cache_state_ptr + cache_locations,
            mask=valid_positions,
            other=-1,
        )
        acquired = valid_positions & (lock_states < -1)
        linear_indices = -lock_states - 2
        # Copy validates the final lock owner, and this stream commits immediately.
        tl.atomic_xchg(
            lxu_cache_state_ptr + cache_locations,
            linear_indices,
            mask=acquired,
            sem="release",
        )
        block_start += block_stride


@triton.jit
# Triton TR001: BLOCK_D is a fixed cache-row dimension.
def _triton_uvm_cache_copy_lru_kernel(  # noqa: TR001
    weight_ptr,
    cache_index_table_map_ptr,
    cache_hash_size_cumsum_ptr,
    cache_weights_offsets_ptr,
    D_offsets_ptr,
    cache_work_items_ptr,
    lxu_cache_state_ptr,
    lxu_cache_weights_ptr,
    lxu_state_ptr,
    cache_miss_counter_ptr,
    num_cache_locations,
    cache_rows,
    total_cache_hash_size,
    timestep,
    CACHE_ROW_WIDTH: tl.constexpr,
    CACHE_LOCATION_BITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    if tl.load(cache_miss_counter_ptr) == 0:
        return
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    row_offsets = tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_D)
    invalid_expected_state = -(total_cache_hash_size + 2)
    cache_location_mask = (1 << CACHE_LOCATION_BITS) - 1
    while block_start < num_cache_locations:
        positions = block_start + row_offsets
        valid_positions = positions < num_cache_locations
        work_items = tl.load(
            cache_work_items_ptr + positions,
            mask=valid_positions,
            other=-1,
        )
        cache_locations = work_items & cache_location_mask
        linear_indices = work_items >> CACHE_LOCATION_BITS
        valid_locations = (
            valid_positions
            & (work_items >= 0)
            & (cache_locations < cache_rows)
            & (linear_indices >= 0)
            & (linear_indices < total_cache_hash_size)
        )
        lock_states = tl.load(
            lxu_cache_state_ptr + cache_locations,
            mask=valid_locations,
            other=-1,
        )
        expected_lock_states = -(linear_indices + 2)
        acquired = valid_locations & (lock_states == expected_lock_states)
        feature = tl.load(
            cache_index_table_map_ptr + linear_indices,
            mask=acquired,
            other=0,
        ).to(tl.int64)
        feature_row_start = tl.load(
            cache_hash_size_cumsum_ptr + feature,
            mask=acquired,
            other=0,
        )
        weight_offset = tl.load(
            cache_weights_offsets_ptr + feature,
            mask=acquired,
            other=0,
        )
        embedding_dim = tl.load(
            D_offsets_ptr + feature + 1,
            mask=acquired,
            other=0,
        ) - tl.load(D_offsets_ptr + feature, mask=acquired, other=0)
        row = linear_indices - feature_row_start
        values = tl.load(
            weight_ptr
            + weight_offset[:, None]
            + row[:, None] * embedding_dim[:, None]
            + columns[None, :],
            mask=acquired[:, None] & (columns[None, :] < embedding_dim[:, None]),
            other=0,
        )
        tl.store(
            lxu_cache_weights_ptr
            + cache_locations[:, None] * CACHE_ROW_WIDTH
            + columns[None, :],
            values,
            mask=acquired[:, None] & (columns[None, :] < embedding_dim[:, None]),
        )
        tl.store(
            lxu_state_ptr + cache_locations,
            timestep.to(tl.int64),
            mask=acquired,
        )
        tl.atomic_cas(
            lxu_cache_state_ptr + tl.where(acquired, cache_locations, 0),
            tl.where(acquired, expected_lock_states, invalid_expected_state),
            linear_indices,
            sem="release",
        )
        block_start += block_stride


@triton.jit
# Triton TR001: CACHE_ASSOC is fixed and BLOCK_N is an occupancy tradeoff.
def _triton_uvm_cache_lookup_kernel(  # noqa: TR001
    linear_cache_indices_ptr,
    lxu_cache_state_ptr,
    lxu_state_ptr,
    lfu_cache_timestamps_ptr,
    lxu_cache_locations_ptr,
    num_indices,
    total_cache_hash_size,
    cache_sets,
    timestep,
    cache_miss_counter_ptr,
    CACHE_ASSOC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_LFU: tl.constexpr,
    UPDATE_STATE: tl.constexpr,
    RELOOKUP_MISSES_ONLY: tl.constexpr,
    SKIP_IF_NO_MISSES: tl.constexpr,
) -> None:
    if SKIP_IF_NO_MISSES and tl.load(cache_miss_counter_ptr) == 0:
        return
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_N
    block_stride = tl.num_programs(0).to(tl.int64) * BLOCK_N
    lanes = tl.arange(0, BLOCK_N)
    while block_start < num_indices:
        positions = block_start + lanes
        valid = positions < num_indices
        linear_indices = tl.load(
            linear_cache_indices_ptr + positions,
            mask=valid,
            other=total_cache_hash_size,
        )
        valid &= (linear_indices >= 0) & (linear_indices < total_cache_hash_size)
        if RELOOKUP_MISSES_ONLY:
            valid &= (
                tl.load(
                    lxu_cache_locations_ptr + positions,
                    mask=valid,
                    other=-1,
                )
                < 0
            )
        cache_sets_for_indices = tl.where(valid, linear_indices % cache_sets, 0)
        ways = tl.arange(0, CACHE_ASSOC)
        # Triton TR003: CACHE_ASSOC exactly spans an in-bounds cache set.
        states = tl.load(  # noqa: TR003
            lxu_cache_state_ptr
            + cache_sets_for_indices[:, None] * CACHE_ASSOC
            + ways[None, :],
            mask=valid[:, None],
            other=-1,
        )
        matched_way = (
            tl.max(
                tl.where(states == linear_indices[:, None], ways[None, :] + 1, 0),
                axis=1,
            )
            - 1
        )
        cache_locations = tl.where(
            valid & (matched_way >= 0),
            cache_sets_for_indices * CACHE_ASSOC + matched_way,
            -1,
        )
        tl.store(
            lxu_cache_locations_ptr + positions,
            cache_locations,
            mask=(positions < num_indices) & ((not RELOOKUP_MISSES_ONLY) | valid),
        )
        if UPDATE_STATE:
            misses = valid & (cache_locations < 0)
            num_misses = tl.sum(misses.to(tl.int32), axis=0)
            if num_misses > 0:
                tl.atomic_add(cache_miss_counter_ptr, num_misses)
            if IS_LFU:
                pass
            else:
                tl.atomic_max(
                    lxu_state_ptr + cache_locations,
                    timestep.to(tl.int64),
                    mask=cache_locations >= 0,
                    sem="relaxed",
                )
        block_start += block_stride


class TritonUVMCachingTableBatchedEmbeddingBags(TritonUVMTableBatchedEmbeddingBags):
    """Forward-only Triton UVM TBE with a bounded HBM row cache."""

    lxu_cache_locations: torch.Tensor

    def __init__(  # noqa: C901
        self,
        *args: Any,
        cache_load_factor: float = 0.2,
        cache_sets: int = 0,
        cache_reserved_memory: float = 0.0,
        cache_algorithm: str | CacheAlgorithm = CacheAlgorithm.LRU,
        uvm_cache_lookup_block_limit: int = 0,
        uvm_cache_populate_block_limit: int = 0,
        uvm_cache_materialize_block_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        if not 0 < cache_load_factor <= 1:
            raise ValueError("cache_load_factor must be in (0, 1]")
        if cache_sets < 0:
            raise ValueError("cache_sets must be non-negative")
        if cache_reserved_memory < 0:
            raise ValueError("cache_reserved_memory must be non-negative")
        if uvm_cache_lookup_block_limit < 0:
            raise ValueError("uvm_cache_lookup_block_limit must be non-negative")
        if uvm_cache_populate_block_limit < 0:
            raise ValueError("uvm_cache_populate_block_limit must be non-negative")
        if uvm_cache_materialize_block_limit < 0:
            raise ValueError("uvm_cache_materialize_block_limit must be non-negative")
        super().__init__(*args, **kwargs)

        if isinstance(cache_algorithm, str):
            try:
                cache_algorithm = CacheAlgorithm[cache_algorithm.upper()]
            except KeyError as error:
                raise ValueError("cache_algorithm must be LRU or LFU") from error
        if cache_algorithm not in (CacheAlgorithm.LRU, CacheAlgorithm.LFU):
            raise ValueError("cache_algorithm must be LRU or LFU")

        cache_assoc = torch.cuda.get_device_properties(self.weight.device).warp_size
        total_rows = sum(rows for rows, _ in self.embedding_specs)
        bytes_per_cache_set = (
            cache_assoc * self.max_embedding_dim * self.weight.element_size()
        )
        total_cache_sets = (total_rows + cache_assoc - 1) // cache_assoc
        if cache_sets <= 0:
            cache_sets = max(
                1,
                (int(total_rows * cache_load_factor) + cache_assoc - 1) // cache_assoc,
            )
            total_memory = torch.cuda.get_device_properties(
                self.weight.device
            ).total_memory
            free_memory = (
                total_memory
                - torch.cuda.memory_reserved(self.weight.device)
                - int(cache_reserved_memory)
            )
            if free_memory <= 0:
                raise ValueError("no HBM is available for the UVM cache")
            if cache_sets * bytes_per_cache_set > free_memory:
                cache_sets = (
                    free_memory // self.max_embedding_dim // self.weight.element_size()
                    + cache_assoc
                    - 1
                ) // cache_assoc
        cache_sets = min(cache_sets, total_cache_sets)
        if cache_sets <= 0:
            raise ValueError("cache configuration must hold at least one cache set")
        if cache_algorithm == CacheAlgorithm.LFU and cache_sets >= 2**24 - 1:
            raise ValueError("LFU cache requires fewer than 2**24 - 1 cache sets")

        table_rows = [rows for rows, _ in self.embedding_specs]
        table_sizes = [rows * dim for rows, dim in self.embedding_specs]
        table_weight_offsets = lengths_to_offsets(table_sizes)
        table_row_offsets = lengths_to_offsets(table_rows, keep_last=True)
        feature_weight_offsets = [
            table_weight_offsets[table] for table in self.feature_table_map
        ]
        feature_row_offsets = [
            table_row_offsets[table] for table in self.feature_table_map
        ] + [total_rows]
        table_to_feature = [-1] * len(self.embedding_specs)
        for feature, table in enumerate(self.feature_table_map):
            table_to_feature[table] = feature
        if any(feature < 0 for feature in table_to_feature):
            raise ValueError("every cached physical table must map to a feature")

        device = self.weight.device
        self.cache_algorithm = cache_algorithm
        self.cache_assoc = cache_assoc
        self.cache_sets = cache_sets
        self.uvm_cache_lookup_block_limit = uvm_cache_lookup_block_limit
        self.uvm_cache_populate_block_limit = uvm_cache_populate_block_limit
        self.uvm_cache_materialize_block_limit = (
            uvm_cache_materialize_block_limit
            if uvm_cache_materialize_block_limit > 0
            else uvm_cache_populate_block_limit
        )
        self.uvm_cache_bytes = (
            cache_sets
            * cache_assoc
            * self.max_embedding_dim
            * self.weight.element_size()
        )
        self.total_cache_hash_size = total_rows
        self.cache_timestep = 1
        self.register_buffer(
            "cache_hash_size_cumsum",
            torch.tensor(feature_row_offsets, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "cache_weights_offsets",
            torch.tensor(feature_weight_offsets, dtype=torch.int64, device=device),
            persistent=False,
        )
        # Citrine C3: construct cache metadata directly on the target GPU.
        cache_table_ids = torch.empty(
            total_rows,
            dtype=torch.int32,
            device=device,
        )
        for table, feature in enumerate(table_to_feature):
            cache_table_ids[
                table_row_offsets[table] : table_row_offsets[table + 1]
            ].fill_(feature)
        self.register_buffer("cache_index_table_map", cache_table_ids, persistent=False)
        self.register_buffer(
            "lxu_cache_state",
            torch.full(
                (cache_sets, cache_assoc),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "lxu_cache_weights",
            torch.empty(
                cache_sets * cache_assoc,
                self.max_embedding_dim,
                dtype=self.weight.dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "lxu_state",
            torch.full(
                (cache_sets, cache_assoc),
                -1 if cache_algorithm == CacheAlgorithm.LFU else 0,
                dtype=(
                    torch.int32
                    if cache_algorithm == CacheAlgorithm.LFU
                    else torch.int64
                ),
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "lfu_cache_timestamps",
            (
                torch.zeros(
                    (cache_sets, cache_assoc),
                    dtype=torch.int32,
                    device=device,
                )
                if cache_algorithm == CacheAlgorithm.LFU
                else torch.empty(0, dtype=torch.int32, device=device)
            ),
            persistent=False,
        )
        self.register_buffer(
            "lfu_frequency",
            (
                torch.zeros(
                    total_rows,
                    dtype=torch.int32,
                    device=device,
                )
                if cache_algorithm == CacheAlgorithm.LFU
                else torch.empty(0, dtype=torch.int32, device=device)
            ),
            persistent=False,
        )
        self.register_buffer(
            "triton_cache_miss_counter",
            torch.zeros(1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "cache_admission_counter",
            torch.zeros(cache_sets, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.lxu_cache_locations = torch.empty(0, dtype=torch.int32, device=device)
        logging.info(
            "Initialized Triton UVM cache: algorithm=%s sets=%d rows=%d bytes=%d "
            "lookup_block_limit=%d populate_block_limit=%d "
            "materialize_block_limit=%d",
            cache_algorithm.name,
            cache_sets,
            cache_sets * cache_assoc,
            self.uvm_cache_bytes,
            uvm_cache_lookup_block_limit,
            uvm_cache_populate_block_limit,
            self.uvm_cache_materialize_block_limit,
        )

    @staticmethod
    def _cache_grid(work_items: int, block_limit: int) -> tuple[int]:
        return (min(work_items, block_limit) if block_limit > 0 else work_items,)

    def _cache_lookup_grid(self, num_indices: int) -> tuple[int]:
        return self._cache_grid(
            triton.cdiv(num_indices, 128), self.uvm_cache_lookup_block_limit
        )

    def _cache_populate_grid(
        self, num_rows: int, rows_per_program: int = 8
    ) -> tuple[int]:
        programs = triton.cdiv(triton.cdiv(num_rows, rows_per_program), 64)
        return self._cache_grid(programs, self.uvm_cache_materialize_block_limit)

    def _cache_reserve_grid(self, num_indices: int) -> tuple[int]:
        programs = triton.cdiv(triton.cdiv(num_indices, 4), 8192)
        return self._cache_grid(programs, self.uvm_cache_populate_block_limit)

    def _prepare_forward_cache(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        vbe_B_offsets: torch.Tensor | None,
        max_B: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_sets == 0:
            return (
                self.weight[:0],
                torch.empty(0, dtype=torch.int32, device=indices.device),
            )
        linear_cache_indices = torch.ops.fbgemm.linearize_cache_indices(
            self.cache_hash_size_cumsum,
            indices,
            offsets,
            vbe_B_offsets,
            max_B if vbe_B_offsets is not None else -1,
            0,
        )
        self.cache_timestep += 1
        if indices.numel() == 0:
            self.lxu_cache_locations = torch.empty(
                0, dtype=torch.int32, device=indices.device
            )
            return self.lxu_cache_weights, self.lxu_cache_locations

        self.lxu_cache_locations = torch.empty(
            indices.numel(), dtype=torch.int32, device=indices.device
        )
        (
            linear_unique_indices,
            linear_unique_indices_length,
            linear_unique_indices_count,
        ) = torch.ops.fbgemm.get_unique_indices(
            linear_cache_indices,
            self.total_cache_hash_size,
            self.cache_algorithm == CacheAlgorithm.LFU,
        )
        cache_rows = self.cache_sets * self.cache_assoc
        self.cache_admission_counter.zero_()
        self.triton_cache_miss_counter.zero_()
        if self.cache_algorithm == CacheAlgorithm.LRU:
            _triton_uvm_cache_lookup_kernel[self._cache_lookup_grid(indices.numel())](
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_state,
                self.lfu_cache_timestamps,
                self.lxu_cache_locations,
                indices.numel(),
                self.total_cache_hash_size,
                self.cache_sets,
                self.cache_timestep,
                self.triton_cache_miss_counter,
                CACHE_ASSOC=self.cache_assoc,
                BLOCK_N=128,
                IS_LFU=False,
                UPDATE_STATE=True,
                RELOOKUP_MISSES_ONLY=False,
                SKIP_IF_NO_MISSES=False,
                num_warps=4,
            )
            _triton_uvm_cache_reserve_lru_kernel[
                self._cache_reserve_grid(linear_unique_indices.numel())
            ](
                linear_unique_indices,
                self.lxu_cache_state,
                self.lxu_state,
                self.cache_admission_counter,
                self.triton_cache_miss_counter,
                linear_unique_indices.numel(),
                self.total_cache_hash_size,
                self.cache_sets,
                self.cache_timestep,
                CACHE_ASSOC=self.cache_assoc,
                CACHE_LOCATION_BITS=max(cache_rows - 1, 1).bit_length(),
                BLOCK_N=16,
                num_warps=4,
            )
        else:
            if linear_unique_indices_count is None:
                raise RuntimeError("LFU cache requires unique-index counts")
            _triton_uvm_cache_reserve_kernel[
                self._cache_reserve_grid(linear_unique_indices.numel())
            ](
                linear_unique_indices,
                linear_unique_indices_length,
                linear_unique_indices_count,
                self.lxu_cache_locations,
                self.lfu_frequency,
                self.lxu_cache_state,
                self.lxu_state,
                self.lfu_cache_timestamps,
                self.cache_admission_counter,
                self.triton_cache_miss_counter,
                self.total_cache_hash_size,
                self.cache_sets,
                self.cache_timestep,
                CACHE_ASSOC=self.cache_assoc,
                CACHE_LOCATION_BITS=max(cache_rows - 1, 1).bit_length(),
                BLOCK_N=16,
                num_warps=4,
            )
        materialize_rows = linear_unique_indices.numel()
        if self.cache_algorithm == CacheAlgorithm.LRU:
            _triton_uvm_cache_copy_lru_kernel[
                self._cache_populate_grid(materialize_rows)
            ](
                self.weight,
                self.cache_index_table_map,
                self.cache_hash_size_cumsum,
                self.cache_weights_offsets,
                self._D_offsets,
                linear_unique_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.lxu_state,
                self.triton_cache_miss_counter,
                materialize_rows,
                cache_rows,
                self.total_cache_hash_size,
                self.cache_timestep,
                CACHE_ROW_WIDTH=self.max_embedding_dim,
                CACHE_LOCATION_BITS=max(cache_rows - 1, 1).bit_length(),
                BLOCK_N=8,
                BLOCK_D=self.block_size,
                num_warps=4,
            )
        else:
            _triton_uvm_cache_copy_kernel[
                self._cache_populate_grid(materialize_rows, rows_per_program=2)
            ](
                self.weight,
                self.cache_index_table_map,
                self.cache_hash_size_cumsum,
                self.cache_weights_offsets,
                self._D_offsets,
                linear_unique_indices,
                self.lxu_cache_state,
                self.lxu_cache_weights,
                self.triton_cache_miss_counter,
                linear_unique_indices_length,
                cache_rows,
                self.total_cache_hash_size,
                CACHE_ROW_WIDTH=self.max_embedding_dim,
                CACHE_LOCATION_BITS=max(cache_rows - 1, 1).bit_length(),
                BLOCK_N=2,
                BLOCK_D=self.block_size,
                num_warps=2,
            )
            _triton_uvm_cache_commit_kernel[
                self._cache_populate_grid(
                    min(materialize_rows, cache_rows), rows_per_program=128
                )
            ](
                self.lxu_cache_locations,
                self.lxu_cache_state,
                self.triton_cache_miss_counter,
                BLOCK_N=128,
                num_warps=4,
            )
        _triton_uvm_cache_lookup_kernel[self._cache_lookup_grid(indices.numel())](
            linear_cache_indices,
            self.lxu_cache_state,
            self.lxu_state,
            self.lfu_cache_timestamps,
            self.lxu_cache_locations,
            indices.numel(),
            self.total_cache_hash_size,
            self.cache_sets,
            self.cache_timestep,
            self.triton_cache_miss_counter,
            CACHE_ASSOC=self.cache_assoc,
            BLOCK_N=128,
            IS_LFU=self.cache_algorithm == CacheAlgorithm.LFU,
            UPDATE_STATE=False,
            RELOOKUP_MISSES_ONLY=self.cache_algorithm == CacheAlgorithm.LRU,
            SKIP_IF_NO_MISSES=self.cache_algorithm == CacheAlgorithm.LRU,
            num_warps=4,
        )
        return self.lxu_cache_weights, self.lxu_cache_locations

    def _run_forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor | None,
        batch_size_per_feature_per_rank: list[list[int]] | None,
        vbe_metadata: Any | None,
        row_output_offsets: torch.Tensor | None,
        b_t_map: torch.Tensor | None,
        info_B_num_bits: int,
        info_B_mask: int,
        total_B: int,
        max_B: int,
        use_fused_bounds_check: bool,
        *,
        weight_ptrs: tuple[torch.Tensor, ...],
        weight_chunk_starts: tuple[int, ...],
        split_weight_row_starts: tuple[int, ...],
        feature_weight_chunk_ids: tuple[int, ...],
        feature_chunk_relative_table_offsets: tuple[int, ...],
    ) -> torch.Tensor:
        cache_weights, cache_locations = self._prepare_forward_cache(
            indices,
            offsets,
            vbe_metadata.B_offsets if vbe_metadata is not None else None,
            max_B,
        )
        if cache_weights.numel() == 0:
            return super()._run_forward(
                indices,
                offsets,
                per_sample_weights,
                batch_size_per_feature_per_rank,
                vbe_metadata,
                row_output_offsets,
                b_t_map,
                info_B_num_bits,
                info_B_mask,
                total_B,
                max_B,
                use_fused_bounds_check,
                weight_ptrs=weight_ptrs,
                weight_chunk_starts=weight_chunk_starts,
                split_weight_row_starts=split_weight_row_starts,
                feature_weight_chunk_ids=feature_weight_chunk_ids,
                feature_chunk_relative_table_offsets=(
                    feature_chunk_relative_table_offsets
                ),
            )
        if len(weight_ptrs) != 1:
            raise RuntimeError("Triton UVM caching requires contiguous weights")
        return _run_uvm_caching_forward(
            indices,
            offsets,
            weight_ptrs[0],
            self.table_offsets,
            self.embedding_dims,
            self.embedding_offsets,
            self.feature_table_map_tensor,
            self.rows_cumsum,
            self.total_embedding_dim,
            self.T,
            self.block_size,
            per_sample_weights,
            self.record_forward_event,
            self.output_dtype,
            batch_size_per_feature_per_rank,
            vbe_metadata,
            row_output_offsets,
            b_t_map,
            info_B_num_bits,
            info_B_mask,
            total_B,
            max_B,
            self.bounds_check_warning,
            use_fused_bounds_check,
            self.enable_triton_tbe_optimizations,
            cache_weights,
            cache_locations,
        )

    def lookup_cache_locations(
        self, linear_cache_indices: torch.Tensor
    ) -> torch.Tensor:
        locations = torch.empty(
            linear_cache_indices.numel(),
            dtype=torch.int32,
            device=linear_cache_indices.device,
        )
        if linear_cache_indices.numel() > 0:
            _triton_uvm_cache_lookup_kernel[
                self._cache_lookup_grid(linear_cache_indices.numel())
            ](
                linear_cache_indices,
                self.lxu_cache_state,
                self.lxu_state,
                self.lfu_cache_timestamps,
                locations,
                linear_cache_indices.numel(),
                self.total_cache_hash_size,
                self.cache_sets,
                self.cache_timestep,
                self.triton_cache_miss_counter,
                CACHE_ASSOC=self.cache_assoc,
                BLOCK_N=128,
                IS_LFU=self.cache_algorithm == CacheAlgorithm.LFU,
                UPDATE_STATE=False,
                RELOOKUP_MISSES_ONLY=False,
                SKIP_IF_NO_MISSES=False,
                num_warps=4,
            )
        return locations

    def reset_cache_states(self) -> None:
        self.lxu_cache_state.fill_(-1)
        if self.cache_algorithm == CacheAlgorithm.LFU:
            self.lxu_state.fill_(-1)
            self.lfu_frequency.zero_()
        else:
            self.lxu_state.zero_()
        self.lfu_cache_timestamps.zero_()
        self.cache_admission_counter.zero_()
        self.cache_timestep = 1
        self.lxu_cache_locations = self.lxu_cache_locations[:0]
