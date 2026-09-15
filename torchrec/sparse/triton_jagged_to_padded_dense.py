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
# Triton TR001: rows are independent and each program streams one row tile.
def _jagged_to_padded_dense_kernel(  # noqa: TR001
    values,
    offsets,
    output,
    max_length,
    cell_size,
    output_row_size,
    padding_value,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    batch = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = columns < output_row_size
    positions = columns // cell_size
    dimensions = columns - positions * cell_size
    start = tl.load(offsets + batch)
    end = tl.load(offsets + batch + 1)
    copy_mask = output_mask & (positions < end - start)
    value_offsets = (start + positions) * cell_size + dimensions
    copied = tl.load(
        values + value_offsets,
        mask=copy_mask,
        other=padding_value,
        eviction_policy="evict_first",
    )
    tl.store(
        output + batch * output_row_size + columns,
        copied,
        mask=output_mask,
    )


@triton.jit
# Triton TR001: backward mirrors the coalesced forward copy.
def _padded_dense_to_jagged_kernel(  # noqa: TR001
    grad_output,
    offsets,
    grad_values,
    max_length,
    cell_size,
    output_row_size,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    batch = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = columns < output_row_size
    positions = columns // cell_size
    dimensions = columns - positions * cell_size
    start = tl.load(offsets + batch)
    end = tl.load(offsets + batch + 1)
    copy_mask = output_mask & (positions < end - start)
    value_offsets = (start + positions) * cell_size + dimensions
    gradients = tl.load(
        grad_output + batch * output_row_size + columns,
        mask=copy_mask,
        eviction_policy="evict_first",
    )
    tl.store(grad_values + value_offsets, gradients, mask=copy_mask)


def _validate_inputs(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
) -> None:
    if values.device.type != "cuda" or offsets.device != values.device:
        raise ValueError("values and offsets must be CUDA tensors on the same device")
    if values.ndim < 1 or offsets.ndim != 1:
        raise ValueError(
            "values must have at least one dimension and offsets must be 1D"
        )
    if not values.is_contiguous() or not offsets.is_contiguous():
        raise ValueError("values and offsets must be contiguous")
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("values must have float16, bfloat16, or float32 dtype")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets must have int32 or int64 dtype")
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one element")
    if max_length < 0:
        raise ValueError("max_length must be nonnegative")


def _forward_impl(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float,
) -> torch.Tensor:
    _validate_inputs(values, offsets, max_length)
    batch_size = offsets.numel() - 1
    output = values.new_empty((batch_size, max_length, *values.shape[1:]))
    if output.numel() == 0:
        return output

    cell_size = values.stride(0)
    output_row_size = max_length * cell_size
    _jagged_to_padded_dense_kernel[
        (batch_size, triton.cdiv(output_row_size, _BLOCK_SIZE))
    ](
        values,
        offsets,
        output,
        max_length,
        cell_size,
        output_row_size,
        padding_value,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    return output


def _backward_impl(
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    input_shape: tuple[int, ...],
    max_length: int,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    grad_values = grad_output.new_zeros(input_shape)
    if grad_output.numel() == 0 or grad_values.numel() == 0:
        return grad_values

    batch_size = offsets.numel() - 1
    cell_size = grad_values.stride(0)
    output_row_size = max_length * cell_size
    _padded_dense_to_jagged_kernel[
        (batch_size, triton.cdiv(output_row_size, _BLOCK_SIZE))
    ](
        grad_output,
        offsets,
        grad_values,
        max_length,
        cell_size,
        output_row_size,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=_BLOCK_SIZE,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    return grad_values


@torch.library.custom_op(
    "torchrec::triton_jagged_to_padded_dense",
    mutates_args=(),
    schema=(
        "(Tensor values, Tensor offsets, SymInt max_length, "
        "float padding_value=0.0) -> Tensor"
    ),
)
def triton_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    return _forward_impl(values, offsets, max_length, padding_value)


@triton_jagged_to_padded_dense.register_fake
def _fake_triton_jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: torch.Tensor,
    max_length: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    del padding_value
    return values.new_empty((offsets.numel() - 1, max_length, *values.shape[1:]))


@torch.library.custom_op(
    "torchrec::triton_jagged_to_padded_dense_backward",
    mutates_args=(),
    schema=(
        "(Tensor grad_output, Tensor offsets, SymInt[] input_shape, "
        "SymInt max_length) -> Tensor"
    ),
)
def _triton_jagged_to_padded_dense_backward(
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    input_shape: list[int],
    max_length: int,
) -> torch.Tensor:
    return _backward_impl(grad_output, offsets, tuple(input_shape), max_length)


@_triton_jagged_to_padded_dense_backward.register_fake
def _fake_triton_jagged_to_padded_dense_backward(
    grad_output: torch.Tensor,
    offsets: torch.Tensor,
    input_shape: list[int],
    max_length: int,
) -> torch.Tensor:
    del offsets, max_length
    return grad_output.new_empty(input_shape)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    values, offsets, max_length, _padding_value = inputs
    ctx.save_for_backward(offsets)
    ctx.input_shape = values.shape
    ctx.max_length = max_length


def _backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None]:
    (offsets,) = ctx.saved_tensors
    return (
        _triton_jagged_to_padded_dense_backward(
            grad_output,
            offsets,
            ctx.input_shape,
            ctx.max_length,
        ),
        None,
        None,
        None,
    )


triton_jagged_to_padded_dense.register_autograd(
    _backward,
    setup_context=_setup_context,
)
