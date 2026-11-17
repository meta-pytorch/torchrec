#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""MX4 dequantize that reads its input rows through a row map.

All2All_Seq_Req's backward receives its gradient as an fbgemm MX4 payload, then
dequantizes it and permutes the rows by the backward recat. With D a multiple
of the group size every row is a whole number of packed groups, so writing
dequantize(payload row row_idx[j]) to output row j yields the permuted gradient
in one pass, without materializing the unpermuted one.
"""

import torch
import triton
import triton.language as tl
from fbgemm_gpu.triton.common import get_mx4_lookup_table


@triton.jit
def _kernel_dequantize_mx4_rowmap(
    A,
    row_idx,
    mx4_lookup_table,
    out,
    n_rows,
    ROW_BYTES: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    """out[j, :] = dequantize(A[row_idx[j] * ROW_BYTES :][:ROW_BYTES]) for the
    BLOCK_R rows j of this program.

    The per-group math is _kernel_dequantize_mx4's in fbgemm_gpu/triton/quantize.py
    op for op: LUT lookup of both nibbles, the scale built from the exponent byte,
    scale * value in fp32, low nibble at even positions, one cast on store.
    """
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1
    pid = tl.program_id(0)
    j = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = j < n_rows
    src = tl.load(row_idx + j, mask=row_mask, other=0).to(tl.int64)
    out_row = j.to(tl.int64) * D
    packed_offset = tl.arange(0, GROUP_SIZE // 2)
    output_offset = tl.arange(0, GROUP_SIZE)
    for g in tl.static_range(D // GROUP_SIZE):
        group_start = src * ROW_BYTES + g * PACKED_GROUP_SIZE
        a = tl.load(
            A + group_start[:, None] + packed_offset[None, :],
            mask=row_mask[:, None],
            other=0,
        )
        low_mx4 = a & 0xF
        high_mx4 = (a >> 4) & 0xF
        low_fp32 = tl.load(mx4_lookup_table + low_mx4)
        high_fp32 = tl.load(mx4_lookup_table + high_mx4)
        exp = tl.load(A + group_start + GROUP_SIZE // 2, mask=row_mask, other=0)
        exp = exp.to(tl.int32) - 127
        scale = tl.where(
            exp >= -126,
            ((exp + 127) << 23).to(tl.float32, bitcast=True),
            2.0**-127,
        )
        scaled_fp32 = tl.interleave(
            scale[:, None] * low_fp32, scale[:, None] * high_fp32
        )
        tl.store(
            out + out_row[:, None] + g * GROUP_SIZE + output_offset[None, :],
            scaled_fp32.to(out.dtype.element_ty),
            mask=row_mask[:, None],
        )


def dequantize_mx4_rowmap(
    payload: torch.Tensor,
    row_idx: torch.Tensor,
    D: int,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantizes rows `row_idx` of `payload`, an fbgemm MX4 (ebits=2, mbits=1)
    uint8 payload of D-value rows, into a flat [len(row_idx) * D] tensor of
    `output_dtype`. Bitwise equal to gathering those rows of
    `triton_dequantize_mx4(payload, group_size, output_dtype=output_dtype)`.
    """
    n_rows = row_idx.numel()
    out = torch.empty(n_rows * D, device=payload.device, dtype=output_dtype)
    if n_rows == 0:
        return out
    BLOCK_R = 16
    _kernel_dequantize_mx4_rowmap[(triton.cdiv(n_rows, BLOCK_R),)](
        payload,
        row_idx,
        get_mx4_lookup_table(2, 1, payload.device),
        out,
        n_rows,
        ROW_BYTES=D // 2 + D // group_size,
        D=D,
        GROUP_SIZE=group_size,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )
    return out
