#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Fused fp32 -> bf16 cast-and-scale for the DDP bf16 compress hook
(ONEFLOW_DDP_COMPRESS_FUSED in model_parallel.py).
"""

import contextlib
import os
from typing import Any, cast

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_NUM_WARPS = 8
# Offsets are int32.
_MAX_NUMEL: int = 2**31 - _BLOCK


@triton.jit
def _bf16_bits(y):
    # Round to nearest even on the bits, NaN -> 0x7FC0, as c10's
    # round_to_nearest_even does.
    bits = y.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(y != y, 0x7FC0, rounded).to(tl.uint16)


@triton.jit
def _scale_to_bf16_kernel(
    x_ptr,
    y_ptr,
    n,
    scale,
    BLOCK: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    if block_start + BLOCK <= n:
        # Full blocks are unmasked so that the loads and stores vectorize.
        y = tl.load(x_ptr + offsets) * scale
        tl.store(y_ptr + offsets, _bf16_bits(y))
    else:
        mask = offsets < n
        y = tl.load(x_ptr + offsets, mask=mask, other=0.0) * scale
        tl.store(y_ptr + offsets, _bf16_bits(y), mask=mask)


def supported(x: torch.Tensor) -> bool:
    """Whether ``scale_to_bf16`` takes ``x``: a contiguous fp32 GPU tensor (or
    a CPU tensor under the Triton interpreter) with int32-addressable numel."""
    on_device = x.is_cuda or (
        x.device.type == "cpu" and os.environ.get("TRITON_INTERPRET", "0") == "1"
    )
    return (
        on_device
        and x.dtype == torch.float32
        and x.is_contiguous()
        and x.numel() <= _MAX_NUMEL
    )


def scale_to_bf16(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Returns ``(x * scale).to(torch.bfloat16)`` from one kernel: the product
    is rounded to fp32, then to bf16 with round to nearest even."""
    y = torch.empty_like(x, dtype=torch.bfloat16)
    n = x.numel()
    if n > 0:
        with torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext():
            cast(Any, _scale_to_bf16_kernel)[(triton.cdiv(n, _BLOCK),)](
                x,
                y.view(torch.uint16),
                n,
                scale,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
    return y
