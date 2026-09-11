#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import triton
import triton.language as tl


_BUCKETING_MIN_OUTPUT_ELEMENTS = 4 * 1024 * 1024
_COUNTED_SCATTER_MIN_ROW_FRACTION = 0.5
_SORTED_REDUCTION_MIN_OUTPUT_ELEMENTS = 512 * 1024 * 1024


@triton.jit
# Triton TR001: the linearization pass uses one fixed streaming tile.
def _linearize_indices_kernel(  # noqa: TR001
    indices,
    row_offsets,
    linear_indices,
    num_indices,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices
    features = offsets // batch_size
    rows = tl.load(indices + offsets, mask=mask, other=0)
    feature_row_start = tl.load(row_offsets + features, mask=mask, other=0)
    tl.store(linear_indices + offsets, feature_row_start + rows, mask=mask)


@triton.jit
# Triton TR001: the run-detection pass uses one fixed streaming tile.
def _mark_sorted_runs_kernel(  # noqa: TR001
    sorted_rows,
    run_flags,
    num_indices,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices
    rows = tl.load(sorted_rows + offsets, mask=mask, other=0)
    previous_rows = tl.load(
        sorted_rows + offsets - 1,
        mask=mask & (offsets > 0),
        other=-1,
    )
    tl.store(run_flags + offsets, (offsets == 0) | (rows != previous_rows), mask=mask)


@triton.jit
# Triton TR001: the run-compaction pass uses one fixed streaming tile.
def _compact_sorted_runs_kernel(  # noqa: TR001
    sorted_rows,
    run_flags,
    run_ids,
    unique_rows,
    run_offsets,
    num_indices,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices
    flags = tl.load(run_flags + offsets, mask=mask, other=0)
    ids = tl.load(run_ids + offsets, mask=mask, other=0)
    rows = tl.load(sorted_rows + offsets, mask=mask, other=0)
    starts = mask & (flags != 0)
    tl.store(unique_rows + ids - 1, rows, mask=starts)
    tl.store(run_offsets + ids - 1, offsets, mask=starts)


@triton.jit
def _finish_sorted_runs_kernel(
    run_ids,
    run_offsets,
    num_runs,
    num_indices,
) -> None:
    runs = tl.load(run_ids + num_indices - 1)
    tl.store(num_runs, runs)
    tl.store(run_offsets + runs, num_indices)


@triton.jit
# Triton TR001: the heavy uniform-row reduction uses one fixed row/vector tile.
def _sorted_index_backward_kernel(  # noqa: TR001
    grad_output,
    sorted_positions,
    unique_rows,
    run_offsets,
    num_runs,
    grad_inputs,
    batch_size,
    total_columns,
    COLUMNS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
) -> None:
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    columns = tl.arange(0, COLUMNS)[None, :]
    runs = tl.load(num_runs)
    row_mask = rows < runs
    input_rows = tl.load(unique_rows + rows, mask=row_mask, other=0)
    starts = tl.load(run_offsets + rows, mask=row_mask, other=0)
    ends = tl.load(run_offsets + rows + 1, mask=row_mask, other=0)
    positions_in_run = tl.zeros((BLOCK_ROWS, 1), dtype=tl.int64)
    accumulator = tl.zeros((BLOCK_ROWS, COLUMNS), dtype=tl.float32)
    active = row_mask & (starts + positions_in_run < ends)
    while tl.max(active).to(tl.int1):
        positions = tl.load(
            sorted_positions + starts + positions_in_run,
            mask=active,
            other=0,
        )
        features = positions // batch_size
        batches = positions % batch_size
        output_offsets = batches * total_columns + features * COLUMNS + columns
        accumulator += tl.load(
            grad_output + output_offsets,
            mask=active & (columns < COLUMNS),
            other=0.0,
            eviction_policy="evict_first",
        )
        positions_in_run += 1
        active = row_mask & (starts + positions_in_run < ends)
    tl.store(
        grad_inputs + input_rows * COLUMNS + columns,
        accumulator,
        mask=row_mask & (columns < COLUMNS),
    )


@triton.jit
# Triton TR001: keep the auxiliary count pass on one fixed, bandwidth-sized tile.
def _count_index_occurrences_kernel(  # noqa: TR001
    indices,
    row_offsets,
    counts,
    num_indices,
    batch_size,
    RELAXED_ATOMICS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # Triton int64 pointer-cast guard: widen the scalar before large products.
    block_start = tl.program_id(0).to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_indices
    features = offsets // batch_size
    selected_rows = tl.load(indices + offsets, mask=mask, other=0).to(tl.int64)
    row_starts = tl.load(row_offsets + features, mask=mask, other=0)
    if RELAXED_ATOMICS:
        tl.atomic_add(
            counts + row_starts + selected_rows,
            1,
            mask=mask,
            sem="relaxed",
        )
    else:
        tl.atomic_add(counts + row_starts + selected_rows, 1, mask=mask)


@triton.jit
# Triton TR001: cached shape buckets select launch parameters without autotuning.
def _batch_index_select_dim0_kernel(  # noqa: TR001
    inputs,
    indices,
    input_offsets,
    column_offsets,
    feature_ids,
    output,
    batch_size,
    total_columns,
    NUM_FEATURES: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
) -> None:
    batch_feature = tl.program_id(0)
    feature = tl.load(
        feature_ids + batch_feature % NUM_FEATURES, eviction_policy="evict_last"
    )
    batch_tile = batch_feature // NUM_FEATURES
    batch_offsets = tl.arange(0, BLOCK_BATCH)[:, None]
    batches = batch_tile * BLOCK_BATCH + batch_offsets
    # Triton int64 pointer-cast guard: widen the scalar before the output stride.
    output_batches = batch_tile.to(tl.int64) * BLOCK_BATCH + batch_offsets
    columns = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)[None, :]

    column_start = tl.load(column_offsets + feature, eviction_policy="evict_last")
    columns_count = (
        tl.load(column_offsets + feature + 1, eviction_policy="evict_last")
        - column_start
    )
    input_start = tl.load(input_offsets + feature, eviction_policy="evict_last")

    batch_mask = batches < batch_size
    column_mask = columns < columns_count
    mask = batch_mask & column_mask
    selected_rows = tl.load(
        indices + feature * batch_size + batches,
        mask=batch_mask,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.int64)

    input_indices = input_start + selected_rows * columns_count + columns
    output_indices = output_batches * total_columns + column_start + columns
    values = tl.load(
        inputs + input_indices,
        mask=mask,
        eviction_policy="evict_first",
    )
    tl.store(output + output_indices, values, mask=mask)


@triton.jit
# Triton TR001: backward reuses the forward bucket's measured launch parameters.
def _batch_index_select_dim0_backward_kernel(  # noqa: TR001
    grad_output,
    indices,
    input_offsets,
    row_offsets,
    column_offsets,
    feature_ids,
    counts,
    grad_inputs,
    batch_size,
    total_columns,
    NUM_FEATURES: tl.constexpr,
    USE_COUNTS: tl.constexpr,
    RELAXED_ATOMICS: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
) -> None:
    batch_feature = tl.program_id(0)
    feature = tl.load(
        feature_ids + batch_feature % NUM_FEATURES, eviction_policy="evict_last"
    )
    batch_tile = batch_feature // NUM_FEATURES
    batch_offsets = tl.arange(0, BLOCK_BATCH)[:, None]
    batches = batch_tile * BLOCK_BATCH + batch_offsets
    # Triton int64 pointer-cast guard: widen the scalar before the output stride.
    output_batches = batch_tile.to(tl.int64) * BLOCK_BATCH + batch_offsets
    columns = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)[None, :]

    column_start = tl.load(column_offsets + feature, eviction_policy="evict_last")
    columns_count = (
        tl.load(column_offsets + feature + 1, eviction_policy="evict_last")
        - column_start
    )
    input_start = tl.load(input_offsets + feature, eviction_policy="evict_last")

    batch_mask = batches < batch_size
    column_mask = columns < columns_count
    mask = batch_mask & column_mask
    selected_rows = tl.load(
        indices + feature * batch_size + batches,
        mask=batch_mask,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.int64)

    output_indices = output_batches * total_columns + column_start + columns
    gradients = tl.load(
        grad_output + output_indices,
        mask=mask,
        eviction_policy="evict_first",
    )
    input_indices = input_start + selected_rows * columns_count + columns
    atomic_mask = mask
    if USE_COUNTS:
        row_start = tl.load(row_offsets + feature, eviction_policy="evict_last")
        occurrences = tl.load(
            counts + row_start + selected_rows,
            mask=batch_mask,
            other=0,
            eviction_policy="evict_last",
        )
        unique_mask = mask & (occurrences == 1)
        tl.store(grad_inputs + input_indices, gradients, mask=unique_mask)
        atomic_mask = mask & (occurrences > 1)
    if RELAXED_ATOMICS:
        tl.atomic_add(
            grad_inputs + input_indices,
            gradients,
            mask=atomic_mask,
            sem="relaxed",
        )
    else:
        tl.atomic_add(grad_inputs + input_indices, gradients, mask=atomic_mask)


@dataclass(frozen=True)
class _LaunchBucket:
    feature_ids: torch.Tensor
    max_columns: int


@dataclass(frozen=True)
class _Metadata:
    input_offsets: torch.Tensor
    row_offsets: torch.Tensor
    column_offsets: torch.Tensor
    buckets: tuple[_LaunchBucket, ...]
    combined_bucket: _LaunchBucket
    total_columns: int
    total_rows: int
    supports_relaxed_atomics: bool
    num_sms: int


def _prefix_sum(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(itertools.accumulate(values, initial=0))


def _launch_parameters(max_columns: int) -> tuple[int, int, int]:
    if max_columns <= 64:
        block_columns = 1 << (max_columns - 1).bit_length()
        return 256 // block_columns, block_columns, 2
    if max_columns <= 128:
        return 4, 128, 4
    block_columns = min(512, 1 << (max_columns - 1).bit_length())
    return 1, block_columns, 2 if block_columns == 256 else 4


@functools.lru_cache(maxsize=32)
def _cached_metadata(
    input_rows: tuple[int, ...],
    input_columns: tuple[int, ...],
    device: torch.device,
) -> _Metadata:
    input_offsets = _prefix_sum(
        tuple(rows * columns for rows, columns in zip(input_rows, input_columns))
    )
    row_offsets = _prefix_sum(input_rows)
    column_offsets = _prefix_sum(input_columns)

    feature_buckets: dict[int, list[int]] = {}
    for feature, columns in enumerate(input_columns):
        limit = 1 << (columns - 1).bit_length()
        feature_buckets.setdefault(limit, []).append(feature)

    buckets = []
    for feature_list in feature_buckets.values():
        buckets.append(
            _LaunchBucket(
                feature_ids=torch.tensor(
                    feature_list, dtype=torch.int32, device=device
                ),
                max_columns=max(input_columns[i] for i in feature_list),
            )
        )

    tensor_kwargs = {"dtype": torch.int64, "device": device}
    device_properties = torch.cuda.get_device_properties(device)
    return _Metadata(
        input_offsets=torch.tensor(input_offsets, **tensor_kwargs),
        row_offsets=torch.tensor(row_offsets, **tensor_kwargs),
        column_offsets=torch.tensor(column_offsets, **tensor_kwargs),
        buckets=tuple(buckets),
        combined_bucket=_LaunchBucket(
            feature_ids=torch.arange(
                len(input_columns), dtype=torch.int32, device=device
            ),
            max_columns=max(input_columns),
        ),
        total_columns=column_offsets[-1],
        total_rows=row_offsets[-1],
        supports_relaxed_atomics=device_properties.major >= 8,
        num_sms=device_properties.multi_processor_count,
    )


def _launch_buckets(metadata: _Metadata, batch_size: int) -> tuple[_LaunchBucket, ...]:
    if (
        len(metadata.buckets) > 1
        and batch_size * metadata.total_columns >= _BUCKETING_MIN_OUTPUT_ELEMENTS
    ):
        return metadata.buckets
    return (metadata.combined_bucket,)


def _use_counted_scatter(metadata: _Metadata, batch_size: int) -> bool:
    num_features = metadata.combined_bucket.feature_ids.numel()
    return (
        batch_size * metadata.total_columns >= _BUCKETING_MIN_OUTPUT_ELEMENTS
        and metadata.total_rows
        >= _COUNTED_SCATTER_MIN_ROW_FRACTION * batch_size * num_features
    )


def _use_fp32_accumulator(
    metadata: _Metadata, batch_size: int, dtype: torch.dtype
) -> bool:
    num_features = metadata.combined_bucket.feature_ids.numel()
    return (
        dtype != torch.float32
        and batch_size * metadata.total_columns >= _BUCKETING_MIN_OUTPUT_ELEMENTS
        and 4 * metadata.total_rows <= batch_size * num_features
    )


def _use_sorted_reduction(
    metadata: _Metadata,
    batch_size: int,
    input_columns: tuple[int, ...],
) -> bool:
    return (
        len(set(input_columns)) == 1
        and input_columns[0] > 256
        and batch_size * metadata.total_columns >= _SORTED_REDUCTION_MIN_OUTPUT_ELEMENTS
    )


def _sorted_backward_impl(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    input_numel: int,
    batch_size: int,
    input_columns: tuple[int, ...],
    metadata: _Metadata,
) -> torch.Tensor:
    num_indices = indices.numel()
    linear_indices = torch.empty_like(indices)
    linear_grid = (triton.cdiv(num_indices, 256),)
    _linearize_indices_kernel[linear_grid](
        indices,
        metadata.row_offsets,
        linear_indices,
        num_indices,
        batch_size,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=256,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    sorted_rows, sorted_positions = torch.sort(linear_indices)
    run_flags = torch.empty_like(indices, dtype=torch.int32)
    run_grid = (triton.cdiv(num_indices, 256),)
    _mark_sorted_runs_kernel[run_grid](
        sorted_rows,
        run_flags,
        num_indices,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=256,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    run_ids = torch.cumsum(run_flags, dim=0, dtype=torch.int32)
    unique_rows = torch.empty_like(indices)
    run_offsets = torch.empty(num_indices + 1, dtype=torch.int64, device=indices.device)
    num_runs = torch.empty(1, dtype=torch.int32, device=indices.device)
    _compact_sorted_runs_kernel[run_grid](
        sorted_rows,
        run_flags,
        run_ids,
        unique_rows,
        run_offsets,
        num_indices,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=256,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    _finish_sorted_runs_kernel[(1,)](
        run_ids,
        run_offsets,
        num_runs,
        num_indices,
    )
    grad_inputs = torch.zeros(
        input_numel,
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    grid = (triton.cdiv(num_indices, 4),)
    _sorted_index_backward_kernel[grid](
        grad_output,
        sorted_positions,
        unique_rows,
        run_offsets,
        num_runs,
        grad_inputs,
        batch_size,
        metadata.total_columns,
        # pyrefly: ignore[bad-argument-type]
        COLUMNS=input_columns[0],
        # pyrefly: ignore[bad-argument-type]
        BLOCK_ROWS=4,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=2,
    )
    return grad_inputs


def _validate_inputs(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: tuple[int, ...],
    input_columns: tuple[int, ...],
) -> None:
    if len(input_rows) != len(input_columns):
        raise ValueError("input_rows and input_columns must have equal length")
    if inputs.device.type != "cuda" or indices.device != inputs.device:
        raise ValueError("inputs and indices must be CUDA tensors on the same device")
    if not inputs.is_contiguous() or not indices.is_contiguous():
        raise ValueError("inputs and indices must be contiguous")
    if inputs.numel() != sum(
        rows * columns for rows, columns in zip(input_rows, input_columns)
    ):
        raise ValueError("inputs size does not match input_rows and input_columns")
    if indices.numel() != batch_size * len(input_columns):
        raise ValueError("indices size does not match batch_size and input_columns")
    if any(rows <= 0 for rows in input_rows):
        raise ValueError("all input row counts must be positive")
    if any(columns <= 0 for columns in input_columns):
        raise ValueError("all input column counts must be positive")


def _forward_impl(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: tuple[int, ...],
    input_columns: tuple[int, ...],
) -> torch.Tensor:
    _validate_inputs(inputs, indices, batch_size, input_rows, input_columns)
    if not input_columns:
        return inputs.new_empty(0)
    metadata = _cached_metadata(input_rows, input_columns, inputs.device)
    output = inputs.new_empty(batch_size * metadata.total_columns)
    launch_buckets = _launch_buckets(metadata, batch_size)
    for bucket in launch_buckets:
        block_batch, block_columns, num_warps = _launch_parameters(bucket.max_columns)
        if len(launch_buckets) == 1 and bucket.max_columns == 128:
            # Triton TR001: uniform D128 benefits from a wider batch tile.
            block_batch, block_columns, num_warps = 8, 64, 2
        if (
            len(launch_buckets) == 1
            and bucket.max_columns == 1024
            and batch_size * metadata.total_columns
            >= _SORTED_REDUCTION_MIN_OUTPUT_ELEMENTS
        ):
            block_batch, block_columns, num_warps = 1, 1024, 4
        num_features = bucket.feature_ids.numel()
        num_batch_tiles = triton.cdiv(batch_size, block_batch)
        grid = (
            num_features * num_batch_tiles,
            triton.cdiv(bucket.max_columns, block_columns),
        )
        _batch_index_select_dim0_kernel[grid](
            inputs,
            indices,
            metadata.input_offsets,
            metadata.column_offsets,
            bucket.feature_ids,
            output,
            batch_size,
            metadata.total_columns,
            # pyrefly: ignore[bad-argument-type]
            NUM_FEATURES=num_features,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_BATCH=block_batch,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_COLUMNS=block_columns,
            # pyrefly: ignore[unexpected-keyword]
            num_warps=num_warps,
        )
    return output


def _backward_impl(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    input_numel: int,
    batch_size: int,
    input_rows: tuple[int, ...],
    input_columns: tuple[int, ...],
) -> torch.Tensor:
    metadata = _cached_metadata(input_rows, input_columns, grad_output.device)
    if _use_sorted_reduction(metadata, batch_size, input_columns):
        return _sorted_backward_impl(
            grad_output,
            indices,
            input_numel,
            batch_size,
            input_columns,
            metadata,
        )
    use_fp32_accumulator = _use_fp32_accumulator(
        metadata, batch_size, grad_output.dtype
    )
    grad_inputs = torch.zeros(
        input_numel,
        dtype=torch.float32 if use_fp32_accumulator else grad_output.dtype,
        device=grad_output.device,
    )
    use_counts = _use_counted_scatter(metadata, batch_size)
    counts = metadata.row_offsets
    if use_counts:
        counts = torch.zeros(
            metadata.total_rows,
            dtype=torch.int32,
            device=grad_output.device,
        )
        num_indices = indices.numel()
        count_num_programs = triton.cdiv(num_indices, 256)
        count_grid = (count_num_programs,)
        # Triton atomic-add membar amortization uses relaxed ordering at 16+ waves.
        relaxed_atomics: Any = (
            metadata.supports_relaxed_atomics
            and count_num_programs >= 16 * metadata.num_sms
        )
        _count_index_occurrences_kernel[count_grid](
            indices,
            metadata.row_offsets,
            counts,
            num_indices,
            batch_size,
            RELAXED_ATOMICS=relaxed_atomics,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_SIZE=256,
            # pyrefly: ignore[unexpected-keyword]
            num_warps=4,
        )
    for bucket in _launch_buckets(metadata, batch_size):
        block_batch, block_columns, num_warps = _launch_parameters(bucket.max_columns)
        if use_fp32_accumulator and bucket.max_columns <= 128:
            # Triton TR001: high-contention atomics benefit from a wider batch tile.
            block_batch, block_columns, num_warps = 8, 64, 2
        num_features = bucket.feature_ids.numel()
        num_batch_tiles = triton.cdiv(batch_size, block_batch)
        num_column_tiles = triton.cdiv(bucket.max_columns, block_columns)
        num_programs = num_features * num_batch_tiles * num_column_tiles
        grid = (
            num_features * num_batch_tiles,
            num_column_tiles,
        )
        # Triton atomic-add membar amortization: terminal sums with at least
        # 16 waves do not need release ordering within the kernel.
        relaxed_atomics: Any = (
            metadata.supports_relaxed_atomics and num_programs >= 16 * metadata.num_sms
        )
        counted_scatter: Any = use_counts
        _batch_index_select_dim0_backward_kernel[grid](
            grad_output,
            indices,
            metadata.input_offsets,
            metadata.row_offsets,
            metadata.column_offsets,
            bucket.feature_ids,
            counts,
            grad_inputs,
            batch_size,
            metadata.total_columns,
            # pyrefly: ignore[bad-argument-type]
            NUM_FEATURES=num_features,
            USE_COUNTS=counted_scatter,
            RELAXED_ATOMICS=relaxed_atomics,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_BATCH=block_batch,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_COLUMNS=block_columns,
            # pyrefly: ignore[unexpected-keyword]
            num_warps=num_warps,
        )
    return (
        grad_inputs.to(dtype=grad_output.dtype) if use_fp32_accumulator else grad_inputs
    )


@torch.library.custom_op(
    "torchrec::triton_batch_index_select_dim0",
    mutates_args=(),
    schema="(Tensor inputs, Tensor indices, SymInt batch_size, SymInt[] input_rows, SymInt[] input_columns) -> Tensor",
)
def triton_batch_index_select_dim0(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    return _forward_impl(
        inputs,
        indices,
        batch_size,
        tuple(input_rows),
        tuple(input_columns),
    )


@triton_batch_index_select_dim0.register_fake
def _fake_triton_batch_index_select_dim0(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    return inputs.new_empty(batch_size * sum(input_columns))


@torch.library.custom_op(
    "torchrec::triton_batch_index_select_dim0_backward",
    mutates_args=(),
    schema="(Tensor grad_output, Tensor indices, SymInt input_numel, SymInt batch_size, SymInt[] input_rows, SymInt[] input_columns) -> Tensor",
)
def _triton_batch_index_select_dim0_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    input_numel: int,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    return _backward_impl(
        grad_output.contiguous(),
        indices,
        input_numel,
        batch_size,
        tuple(input_rows),
        tuple(input_columns),
    )


@_triton_batch_index_select_dim0_backward.register_fake
def _fake_triton_batch_index_select_dim0_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    input_numel: int,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    return grad_output.new_empty(input_numel)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    input_values, indices, batch_size, input_rows, input_columns = inputs
    ctx.save_for_backward(indices)
    ctx.input_numel = input_values.numel()
    ctx.batch_size = batch_size
    ctx.input_rows = input_rows
    ctx.input_columns = input_columns


def _backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None, None]:
    (indices,) = ctx.saved_tensors
    return (
        _triton_batch_index_select_dim0_backward(
            grad_output,
            indices,
            ctx.input_numel,
            ctx.batch_size,
            ctx.input_rows,
            ctx.input_columns,
        ),
        None,
        None,
        None,
        None,
    )


triton_batch_index_select_dim0.register_autograd(
    _backward,
    setup_context=_setup_context,
)
