#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 1024


@triton.jit
# Triton TR001: one fixed streaming tile covers the bandwidth-bound copy.
def _pack_segments_kernel(  # noqa: TR001
    input,
    lengths,
    cumulative_lengths,
    output,
    max_length,
    cell_size,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < output_numel
    output_rows = offsets // cell_size
    columns = offsets - output_rows * cell_size
    batches = output_rows // max_length
    positions = output_rows - batches * max_length
    segment_lengths = tl.load(lengths + batches, mask=output_mask, other=0)
    segment_ends = tl.load(cumulative_lengths + batches, mask=output_mask, other=0)
    input_rows = segment_ends - segment_lengths + positions
    copy_mask = output_mask & (positions < segment_lengths)
    values = tl.load(
        input + input_rows * cell_size + columns,
        mask=copy_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    tl.store(output + offsets, values, mask=output_mask)


@triton.jit
# Triton TR001: backward reuses the forward streaming tile.
def _unpack_segments_kernel(  # noqa: TR001
    grad_output,
    lengths,
    cumulative_lengths,
    grad_input,
    max_length,
    cell_size,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < output_numel
    output_rows = offsets // cell_size
    columns = offsets - output_rows * cell_size
    batches = output_rows // max_length
    positions = output_rows - batches * max_length
    segment_lengths = tl.load(lengths + batches, mask=output_mask, other=0)
    segment_ends = tl.load(cumulative_lengths + batches, mask=output_mask, other=0)
    input_rows = segment_ends - segment_lengths + positions
    copy_mask = output_mask & (positions < segment_lengths)
    values = tl.load(
        grad_output + offsets,
        mask=copy_mask,
        eviction_policy="evict_first",
    )
    tl.store(
        grad_input + input_rows * cell_size + columns,
        values,
        mask=copy_mask,
    )


def _validate_inputs(
    input: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> None:
    if input.device.type != "cuda" or lengths.device != input.device:
        raise ValueError("input and lengths must be CUDA tensors on the same device")
    if input.ndim < 1 or lengths.ndim != 1:
        raise ValueError(
            "input must have at least one dimension and lengths must be 1D"
        )
    if not input.is_contiguous() or not lengths.is_contiguous():
        raise ValueError("input and lengths must be contiguous")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must have int32 or int64 dtype")
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")


def _forward_impl(
    input: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    _validate_inputs(input, lengths, max_length)
    output_shape = (lengths.numel(), max_length, *input.shape[1:])
    output = input.new_empty(output_shape)
    if output.numel() == 0:
        return output

    cell_size = input.stride(0)
    cumulative_lengths = torch.cumsum(lengths, dim=0)
    _pack_segments_kernel[(triton.cdiv(output.numel(), _BLOCK_SIZE),)](
        input,
        lengths,
        cumulative_lengths,
        output,
        max_length,
        cell_size,
        output.numel(),
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    return output


def _backward_impl(
    grad_output: torch.Tensor,
    lengths: torch.Tensor,
    input_shape: tuple[int, ...],
    max_length: int,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    grad_input = grad_output.new_zeros(input_shape)
    if grad_output.numel() == 0 or grad_input.numel() == 0:
        return grad_input

    cell_size = grad_input.numel() // max(input_shape[0], 1)
    cumulative_lengths = torch.cumsum(lengths, dim=0)
    _unpack_segments_kernel[(triton.cdiv(grad_output.numel(), _BLOCK_SIZE),)](
        grad_output,
        lengths,
        cumulative_lengths,
        grad_input,
        max_length,
        cell_size,
        grad_output.numel(),
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    return grad_input


@torch.library.custom_op(
    "torchrec::triton_pack_segments",
    mutates_args=(),
    schema="(Tensor input, Tensor lengths, SymInt max_length) -> Tensor",
)
def triton_pack_segments(
    input: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    return _forward_impl(input, lengths, max_length)


@triton_pack_segments.register_fake
def _fake_triton_pack_segments(
    input: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    return input.new_empty((lengths.numel(), max_length, *input.shape[1:]))


@torch.library.custom_op(
    "torchrec::triton_pack_segments_backward",
    mutates_args=(),
    schema="(Tensor grad_output, Tensor lengths, SymInt[] input_shape, SymInt max_length) -> Tensor",
)
def _triton_pack_segments_backward(
    grad_output: torch.Tensor,
    lengths: torch.Tensor,
    input_shape: list[int],
    max_length: int,
) -> torch.Tensor:
    return _backward_impl(grad_output, lengths, tuple(input_shape), max_length)


@_triton_pack_segments_backward.register_fake
def _fake_triton_pack_segments_backward(
    grad_output: torch.Tensor,
    lengths: torch.Tensor,
    input_shape: list[int],
    max_length: int,
) -> torch.Tensor:
    return grad_output.new_empty(input_shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    input, lengths, max_length = inputs
    ctx.save_for_backward(lengths)
    ctx.input_shape = input.shape
    ctx.max_length = max_length


def _backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
    (lengths,) = ctx.saved_tensors
    return (
        _triton_pack_segments_backward(
            grad_output,
            lengths,
            ctx.input_shape,
            ctx.max_length,
        ),
        None,
        None,
    )


triton_pack_segments.register_autograd(_backward, setup_context=_setup_context)
