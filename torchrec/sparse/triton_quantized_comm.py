#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import torch
import triton
import triton.language as tl


_MAX_COLUMNS = 4096


@triton.jit
# Triton TR001: row width determines the reduction tile and launch shape.
def _float_to_fused8bitrowwise_kernel(  # noqa: TR001
    input,
    output,
    output_qparams,
    num_rows,
    num_columns,
    output_columns,
    qparam_columns,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
) -> None:
    row_ids = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rows = row_ids[:, None]
    columns = tl.arange(0, BLOCK_COLUMNS)[None, :]
    row_mask = rows < num_rows
    value_mask = row_mask & (columns < num_columns)
    values = tl.load(
        input + rows * num_columns + columns,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)
    minimum = tl.min(tl.where(value_mask, values, float("inf")), axis=1)
    maximum = tl.max(tl.where(value_mask, values, -float("inf")), axis=1)
    value_range = maximum - minimum
    inverse_scale = 255.0 / (value_range + 1.0e-20)
    quantized = tl.floor((values - minimum[:, None]) * inverse_scale[:, None] + 0.5)
    payload_mask = row_mask & (columns < output_columns - 8)
    tl.store(
        output + rows * output_columns + columns,
        tl.where(value_mask, quantized, 0).to(tl.uint8),
        mask=payload_mask,
    )
    qparam_offsets = row_ids * qparam_columns + (output_columns - 8) // 4
    tl.store(
        output_qparams + qparam_offsets,
        value_range / 255.0,
        mask=row_ids < num_rows,
    )
    tl.store(
        output_qparams + qparam_offsets + 1,
        minimum,
        mask=row_ids < num_rows,
    )


@triton.jit
# Triton TR001: dequantization is a single fixed streaming tile.
def _fused8bitrowwise_to_float_kernel(  # noqa: TR001
    input,
    input_qparams,
    output,
    numel,
    input_columns,
    output_columns,
    qparam_columns,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    rows = offsets // output_columns
    columns = offsets % output_columns
    qparam_offsets = rows * qparam_columns + output_columns // 4
    scale = tl.load(input_qparams + qparam_offsets, mask=mask)
    bias = tl.load(input_qparams + qparam_offsets + 1, mask=mask)
    quantized = tl.load(
        input + rows * input_columns + columns,
        mask=mask,
        other=0,
    ).to(tl.float32)
    tl.store(output + offsets, quantized * scale + bias, mask=mask)


def _validate_quantize_input(input: torch.Tensor) -> None:
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if input.dtype != torch.float32:
        raise ValueError("input must have dtype torch.float32")
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if input.shape[-1] > _MAX_COLUMNS:
        raise ValueError(f"input row width cannot exceed {_MAX_COLUMNS}")


def _validate_dequantize_input(input: torch.Tensor) -> None:
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")
    if input.dtype != torch.uint8:
        raise ValueError("input must have dtype torch.uint8")
    if input.ndim == 0:
        raise ValueError("input must have at least one dimension")
    if input.shape[-1] < 8 or input.shape[-1] % 4 != 0:
        raise ValueError(
            "input row width must be a multiple of four and at least eight"
        )


@torch.library.custom_op(
    "torchrec::triton_float_to_fused8bitrowwise_quantized",
    mutates_args=(),
)
def triton_float_to_fused8bitrowwise_quantized(
    input: torch.Tensor,
) -> torch.Tensor:
    _validate_quantize_input(input)
    num_columns = input.shape[-1]
    payload_columns = (num_columns + 3) // 4 * 4
    output_columns = payload_columns + 8
    output = torch.empty(
        (*input.shape[:-1], output_columns),
        dtype=torch.uint8,
        device=input.device,
    )
    num_rows = output.numel() // output_columns
    if num_rows == 0:
        return output
    if num_columns == 0:
        return output.zero_()

    block_columns = max(4, triton.next_power_of_2(num_columns))
    block_rows = max(1, min(8, 256 // block_columns))
    _float_to_fused8bitrowwise_kernel[(triton.cdiv(num_rows, block_rows),)](
        input,
        output,
        output.view(torch.float32),
        num_rows,
        num_columns,
        output_columns,
        output_columns // 4,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_ROWS=block_rows,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_COLUMNS=block_columns,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=min(8, max(1, block_columns // 32)),
    )
    return output


@triton_float_to_fused8bitrowwise_quantized.register_fake
def _fake_float_to_fused8bitrowwise_quantized(
    input: torch.Tensor,
) -> torch.Tensor:
    output_columns = (input.shape[-1] + 3) // 4 * 4 + 8
    return input.new_empty((*input.shape[:-1], output_columns), dtype=torch.uint8)


@torch.library.custom_op(
    "torchrec::triton_fused8bitrowwise_quantized_to_float",
    mutates_args=(),
)
def triton_fused8bitrowwise_quantized_to_float(
    input: torch.Tensor,
) -> torch.Tensor:
    _validate_dequantize_input(input)
    input_columns = input.shape[-1]
    output_columns = input_columns - 8
    output = torch.empty(
        (*input.shape[:-1], output_columns),
        dtype=torch.float32,
        device=input.device,
    )
    if output.numel() == 0:
        return output

    _fused8bitrowwise_to_float_kernel[(triton.cdiv(output.numel(), 512),)](
        input,
        input.view(torch.float32),
        output,
        output.numel(),
        input_columns,
        output_columns,
        input_columns // 4,
        # pyrefly: ignore[bad-argument-type]
        BLOCK_SIZE=512,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=8,
    )
    return output


@triton_fused8bitrowwise_quantized_to_float.register_fake
def _fake_fused8bitrowwise_quantized_to_float(
    input: torch.Tensor,
) -> torch.Tensor:
    return input.new_empty(
        (*input.shape[:-1], input.shape[-1] - 8), dtype=torch.float32
    )
