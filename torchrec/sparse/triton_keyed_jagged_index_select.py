#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, Optional

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 4096
_MAX_FUSED_METADATA_SEGMENTS = 4096


@triton.jit
# Triton TR001: one program keeps the short prefix scans within a single kernel.
def _selected_metadata_kernel(  # noqa: TR001
    lengths,
    indices,
    selected_lengths,
    selected_offsets,
    selected_block_offsets,
    input_batch_size,
    output_batch_size,
    num_output_segments,
    BLOCK_SIZE: tl.constexpr,
    VALUE_BLOCK_SIZE: tl.constexpr,
) -> None:
    segments = tl.arange(0, BLOCK_SIZE)
    mask = segments < num_output_segments
    keys = segments // output_batch_size
    selected_batches = segments - keys * output_batch_size
    source_batches = tl.load(indices + selected_batches, mask=mask, other=0)
    source_segments = keys * input_batch_size + source_batches
    selected = tl.load(lengths + source_segments, mask=mask, other=0).to(tl.int64)
    length_ends = tl.cumsum(selected, axis=0)
    blocks = (selected + VALUE_BLOCK_SIZE - 1) // VALUE_BLOCK_SIZE
    block_ends = tl.cumsum(blocks, axis=0)
    tl.store(selected_lengths + segments, selected, mask=mask)
    tl.store(selected_offsets + segments, length_ends - selected, mask=mask)
    tl.store(selected_block_offsets + segments, block_ends - blocks, mask=mask)
    tl.store(selected_offsets + num_output_segments, tl.sum(selected))
    tl.store(selected_block_offsets + num_output_segments, tl.sum(blocks))


@triton.jit
# Triton TR001: output tiles are balanced independently of jagged segment length.
def _keyed_jagged_index_select_kernel(  # noqa: TR001
    values,
    weights,
    offsets,
    indices,
    selected_offsets,
    selected_lengths,
    selected_block_offsets,
    output_values,
    output_weights,
    input_batch_size,
    output_batch_size,
    num_output_segments,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    valid_block = block < tl.load(selected_block_offsets + num_output_segments)
    segment = 0
    upper = num_output_segments - 1
    while upper - segment > 0:
        middle = (segment + upper + 1) // 2
        is_after = tl.load(selected_block_offsets + middle) <= block
        segment = tl.where(is_after, middle, segment)
        upper = tl.where(is_after, upper, middle - 1)

    block_in_segment = block - tl.load(selected_block_offsets + segment)
    offsets_in_segment = block_in_segment * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = valid_block & (
        offsets_in_segment < tl.load(selected_lengths + segment)
    )
    keys = segment // output_batch_size
    selected_batches = segment - keys * output_batch_size
    source_batches = tl.load(indices + selected_batches)
    source_segments = keys * input_batch_size + source_batches
    source_starts = tl.load(offsets + source_segments)
    output_starts = tl.load(selected_offsets + segment)
    source_offsets = source_starts + offsets_in_segment
    output_offsets = output_starts + offsets_in_segment
    tl.store(
        output_values + output_offsets,
        tl.load(values + source_offsets, mask=output_mask),
        mask=output_mask,
    )
    if HAS_WEIGHTS:
        tl.store(
            output_weights + output_offsets,
            tl.load(weights + source_offsets, mask=output_mask),
            mask=output_mask,
        )


@triton.jit
# Triton TR001: source tiles avoid atomics when selection indices repeat.
def _keyed_jagged_index_select_backward_kernel(  # noqa: TR001
    grad_output_values,
    grad_output_weights,
    offsets,
    indices,
    selected_offsets,
    grad_values,
    grad_weights,
    input_batch_size,
    output_batch_size,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    source_segment = tl.program_id(0)
    block_in_segment = tl.program_id(1)
    source_start = tl.load(offsets + source_segment)
    source_end = tl.load(offsets + source_segment + 1)
    offsets_in_segment = block_in_segment * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    source_offsets = source_start + offsets_in_segment
    input_mask = source_offsets < source_end
    key = source_segment // input_batch_size
    source_batch = source_segment - key * input_batch_size
    grad_value = tl.zeros([BLOCK_SIZE], tl.float32)
    grad_weight = tl.zeros([BLOCK_SIZE], tl.float32)
    selected_batch = 0
    while selected_batch < output_batch_size:
        is_selected = tl.load(indices + selected_batch) == source_batch
        output_segment = key * output_batch_size + selected_batch
        output_start = tl.load(selected_offsets + output_segment)
        output_offsets = output_start + offsets_in_segment
        grad_value += tl.load(
            grad_output_values + output_offsets,
            mask=input_mask & is_selected,
            other=0.0,
        )
        if HAS_WEIGHTS:
            grad_weight += tl.load(
                grad_output_weights + output_offsets,
                mask=input_mask & is_selected,
                other=0.0,
            )
        selected_batch += 1

    tl.store(grad_values + source_offsets, grad_value, mask=input_mask)
    if HAS_WEIGHTS:
        tl.store(grad_weights + source_offsets, grad_weight, mask=input_mask)


def _validate_inputs(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    input_batch_size: int,
    weights: Optional[torch.Tensor],
) -> None:
    tensors = (values, lengths, offsets, indices)
    if values.device.type != "cuda" or any(
        tensor.device != values.device for tensor in tensors[1:]
    ):
        raise ValueError("all inputs must be CUDA tensors on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("values, lengths, offsets, and indices must be 1D")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must have int32 or int64 dtype")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets must have int32 or int64 dtype")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must have int32 or int64 dtype")
    if input_batch_size <= 0 or lengths.numel() % input_batch_size != 0:
        raise ValueError("input_batch_size must be positive and divide lengths")
    if offsets.numel() != lengths.numel() + 1:
        raise ValueError("offsets must contain one more element than lengths")
    if weights is not None:
        if weights.device != values.device or not weights.is_contiguous():
            raise ValueError("weights must be contiguous and on the values device")
        if weights.ndim != 1 or weights.numel() != values.numel():
            raise ValueError("weights must be 1D and match values")


def _selected_metadata(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    input_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_output_segments = lengths.numel() // input_batch_size * indices.numel()
    if num_output_segments == 0:
        selected_lengths = lengths.new_empty(0)
        empty_offsets = torch.zeros(1, dtype=torch.int64, device=lengths.device)
        return selected_lengths, empty_offsets, empty_offsets.clone()
    if num_output_segments <= _MAX_FUSED_METADATA_SEGMENTS:
        selected_lengths = lengths.new_empty(num_output_segments)
        selected_offsets = torch.empty(
            num_output_segments + 1,
            dtype=torch.int64,
            device=lengths.device,
        )
        selected_block_offsets = torch.empty_like(selected_offsets)
        _selected_metadata_kernel[(1,)](
            lengths,
            indices,
            selected_lengths,
            selected_offsets,
            selected_block_offsets,
            input_batch_size,
            indices.numel(),
            num_output_segments,
            # pyrefly: ignore[bad-argument-type]
            BLOCK_SIZE=triton.next_power_of_2(num_output_segments),
            # pyrefly: ignore[bad-argument-type]
            VALUE_BLOCK_SIZE=_BLOCK_SIZE,
        )
        return selected_lengths, selected_offsets, selected_block_offsets

    selected_lengths = (
        lengths.view(-1, input_batch_size)[:, indices.to(torch.int64)]
        .contiguous()
        .flatten()
    )
    selected_offsets = torch.zeros(
        selected_lengths.numel() + 1,
        dtype=torch.int64,
        device=lengths.device,
    )
    torch.cumsum(selected_lengths, dim=0, out=selected_offsets[1:])
    selected_blocks = torch.div(
        selected_lengths + _BLOCK_SIZE - 1,
        _BLOCK_SIZE,
        rounding_mode="floor",
    )
    block_offsets = torch.zeros(
        selected_lengths.numel() + 1,
        dtype=torch.int64,
        device=selected_lengths.device,
    )
    torch.cumsum(selected_blocks, dim=0, out=block_offsets[1:])
    return selected_lengths, selected_offsets, block_offsets


def _forward_impl(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    input_batch_size: int,
    weights: Optional[torch.Tensor],
    selected_lengths_sum: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    _validate_inputs(values, lengths, offsets, indices, input_batch_size, weights)
    selected_lengths, selected_offsets, selected_block_offsets = _selected_metadata(
        lengths,
        indices,
        input_batch_size,
    )
    output_numel = (
        selected_lengths_sum
        if selected_lengths_sum is not None
        else int(selected_offsets[-1].item())
    )
    output_values = values.new_empty(output_numel)
    output_weights = weights.new_empty(output_numel) if weights is not None else None
    if output_numel == 0:
        return output_values, selected_lengths, output_weights

    output_batch_size = indices.numel()
    max_num_blocks = triton.cdiv(output_numel, _BLOCK_SIZE) + selected_lengths.numel()
    _keyed_jagged_index_select_kernel[(max_num_blocks,)](
        values,
        weights if weights is not None else values,
        offsets,
        indices,
        selected_offsets,
        selected_lengths,
        selected_block_offsets,
        output_values,
        output_weights if output_weights is not None else output_values,
        input_batch_size,
        output_batch_size,
        selected_lengths.numel(),
        # pyrefly: ignore[bad-argument-type]
        HAS_WEIGHTS=weights is not None,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=8,
    )
    return output_values, selected_lengths, output_weights


@torch.library.custom_op(
    "torchrec::triton_keyed_jagged_index_select_dim1",
    mutates_args=(),
    schema=(
        "(Tensor values, Tensor lengths, Tensor offsets, Tensor indices, "
        "SymInt input_batch_size, Tensor? weights=None, "
        "SymInt? selected_lengths_sum=None) -> (Tensor, Tensor, Tensor?)"
    ),
)
def triton_keyed_jagged_index_select_dim1(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    input_batch_size: int,
    weights: Optional[torch.Tensor] = None,
    selected_lengths_sum: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    return _forward_impl(
        values,
        lengths,
        offsets,
        indices,
        input_batch_size,
        weights,
        selected_lengths_sum,
    )


@triton_keyed_jagged_index_select_dim1.register_fake
def _fake_triton_keyed_jagged_index_select_dim1(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
    input_batch_size: int,
    weights: Optional[torch.Tensor] = None,
    selected_lengths_sum: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    del offsets
    output_numel = selected_lengths_sum
    if output_numel is None:
        output_numel = torch.library.get_ctx().new_dynamic_size()
    num_keys = lengths.numel() // input_batch_size
    output_values = values.new_empty(output_numel)
    output_lengths = lengths.new_empty(num_keys * indices.numel())
    output_weights = weights.new_empty(output_numel) if weights is not None else None
    return output_values, output_lengths, output_weights


def _backward_impl(
    grad_output_values: torch.Tensor,
    grad_output_weights: Optional[torch.Tensor],
    offsets: torch.Tensor,
    indices: torch.Tensor,
    selected_lengths: torch.Tensor,
    input_numel: int,
    input_batch_size: int,
    has_weights: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    grad_output_values = grad_output_values.contiguous()
    if grad_output_weights is not None:
        grad_output_weights = grad_output_weights.contiguous()
    grad_values = grad_output_values.new_zeros(input_numel)
    grad_weights = (
        grad_output_weights.new_zeros(input_numel)
        if grad_output_weights is not None
        else None
    )
    if grad_output_values.numel() == 0:
        return grad_values, grad_weights

    selected_offsets = torch.zeros(
        selected_lengths.numel() + 1,
        dtype=torch.int64,
        device=selected_lengths.device,
    )
    torch.cumsum(selected_lengths, dim=0, out=selected_offsets[1:])
    max_input_length = int(torch.diff(offsets).max().item())
    num_input_segments = offsets.numel() - 1
    _keyed_jagged_index_select_backward_kernel[
        (num_input_segments, triton.cdiv(max_input_length, _BLOCK_SIZE))
    ](
        grad_output_values,
        grad_output_weights if grad_output_weights is not None else grad_output_values,
        offsets,
        indices,
        selected_offsets,
        grad_values,
        grad_weights if grad_weights is not None else grad_values,
        input_batch_size,
        indices.numel(),
        # pyrefly: ignore[bad-argument-type]
        HAS_WEIGHTS=has_weights and grad_output_weights is not None,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=8,
    )
    return grad_values, grad_weights


@torch.library.custom_op(
    "torchrec::triton_keyed_jagged_index_select_dim1_backward",
    mutates_args=(),
    schema=(
        "(Tensor grad_output_values, Tensor? grad_output_weights, Tensor offsets, "
        "Tensor indices, Tensor selected_lengths, SymInt input_numel, "
        "SymInt input_batch_size, bool has_weights) -> (Tensor, Tensor?)"
    ),
)
def _triton_keyed_jagged_index_select_dim1_backward(
    grad_output_values: torch.Tensor,
    grad_output_weights: Optional[torch.Tensor],
    offsets: torch.Tensor,
    indices: torch.Tensor,
    selected_lengths: torch.Tensor,
    input_numel: int,
    input_batch_size: int,
    has_weights: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return _backward_impl(
        grad_output_values,
        grad_output_weights,
        offsets,
        indices,
        selected_lengths,
        input_numel,
        input_batch_size,
        has_weights,
    )


@_triton_keyed_jagged_index_select_dim1_backward.register_fake
def _fake_triton_keyed_jagged_index_select_dim1_backward(
    grad_output_values: torch.Tensor,
    grad_output_weights: Optional[torch.Tensor],
    offsets: torch.Tensor,
    indices: torch.Tensor,
    selected_lengths: torch.Tensor,
    input_numel: int,
    input_batch_size: int,
    has_weights: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    del offsets, indices, selected_lengths, input_batch_size
    grad_values = grad_output_values.new_empty(input_numel)
    grad_weights = (
        grad_output_weights.new_empty(input_numel)
        if has_weights and grad_output_weights is not None
        else None
    )
    return grad_values, grad_weights


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Any, ...]) -> None:
    values, _lengths, offsets, indices, input_batch_size, weights, _sum = inputs
    _output_values, selected_lengths, _output_weights = output
    ctx.save_for_backward(offsets, indices, selected_lengths)
    ctx.input_numel = values.numel()
    ctx.input_batch_size = input_batch_size
    ctx.has_weights = weights is not None


def _backward(
    ctx: Any,
    grad_output_values: torch.Tensor,
    _grad_output_lengths: Optional[torch.Tensor],
    grad_output_weights: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    None,
    None,
    None,
    None,
    Optional[torch.Tensor],
    None,
]:
    offsets, indices, selected_lengths = ctx.saved_tensors
    grad_values, grad_weights = _triton_keyed_jagged_index_select_dim1_backward(
        grad_output_values,
        grad_output_weights,
        offsets,
        indices,
        selected_lengths,
        ctx.input_numel,
        ctx.input_batch_size,
        ctx.has_weights,
    )
    return grad_values, None, None, None, None, grad_weights, None


triton_keyed_jagged_index_select_dim1.register_autograd(
    _backward,
    setup_context=_setup_context,
)
