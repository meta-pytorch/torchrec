#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Row-tiled MX4 kernels for the sequence all-to-all codec
(TORCHREC_SEQ_A2A_MX4_V2 in comm_ops).

A packed fbgemm MX4 (ebits=2, mbits=1) row of D values is D // group_size
groups of group_size // 2 nibble bytes followed by one exponent byte, and no
group spans two rows when D % group_size == 0. Each program here handles
BLOCK_R whole rows. The encode loads every row with 16-byte loads, takes each
group's max in registers and packs the nibbles without index math per value.
Its per-group and per-value arithmetic is fbgemm's (_kernel_quantize_mx4 in
fbgemm_gpu/triton/quantize.py) op for op, with two exact rewrites:

- it multiplies by 2^-e instead of dividing by 2^e. For the shared exponent e
  in [-127, 125], 2^-e is a normal fp32, so both round the same exact value
  (mul_rcp=False keeps the division);
- the mantissa shift masks its amount to 5 bits. fbgemm's amount reaches 45
  for values far below the group max, and the gfx950 shift instruction it
  compiles to uses the amount's low 5 bits, which the mask states explicitly.

So the payload bytes equal fbgemm's for rounding modes other than stochastic,
an optional row map and any input values: zeros, infinities, quiet NaNs,
subnormals. Signaling NaN bit patterns, which arithmetic never produces, are
the exception: the compiler takes a bf16 one for a quiet NaN when it orders a
group's max, so where it lands in that order decides the max.
"""

import torch
import triton
import triton.language as tl
from fbgemm_gpu.triton.quantize import _compute_exp

# Group sizes and row lengths the kernels take: the tile shapes are powers of
# two.
MX4V2_GROUP_SIZES = (8, 16, 32, 64, 128, 256)
MX4V2_MAX_D = 2048


def mx4v2_supported(D: int, group_size: int) -> bool:
    """Whether the kernels take rows of D values in groups of group_size."""
    return (
        group_size in MX4V2_GROUP_SIZES
        and group_size <= D <= MX4V2_MAX_D
        and D & (D - 1) == 0
    )


@triton.jit(do_not_specialize=["n_rows"])
def _mx4v2_encode_kernel(
    A,
    row_idx,
    out,
    n_rows,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_ROW_IDX: tl.constexpr,
    MUL_RCP: tl.constexpr,
) -> None:
    """Packed row j of `out` is the MX4 payload of row row_idx[j] (row j
    without a row map) of the [*, D] input A, for the BLOCK_R rows j of this
    program."""
    # pyrefly: ignore [bad-assignment]
    FP32_EXP_MASK: tl.constexpr = 0x7F800000
    # pyrefly: ignore [bad-assignment]
    FP32_EXP_OFFSET: tl.constexpr = 23
    # pyrefly: ignore [bad-assignment]
    FP32_EXP_BIAS: tl.constexpr = 127
    # pyrefly: ignore [bad-assignment]
    FP32_SIGN_OFFSET: tl.constexpr = 31
    # pyrefly: ignore [bad-assignment]
    SIGN_MASK: tl.constexpr = 0x1
    # pyrefly: ignore [bad-assignment]
    FP32_MANTISSA_MASK: tl.constexpr = 0x007FFFFF
    MBITS_IMPLICIT: tl.constexpr = MBITS + 1
    # pyrefly: ignore [bad-assignment]
    MAX_FP32_MANTISSA_BITS: tl.constexpr = 24
    # pyrefly: ignore [bad-assignment]
    IMPLIED_1_BIT: tl.constexpr = 1 << 23
    # pyrefly: ignore [bad-assignment]
    FP32_MIN_NORMAL: tl.constexpr = 2 ** (-126)
    # pyrefly: ignore [bad-assignment, unsupported-operation]
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1
    # pyrefly: ignore [bad-assignment, unsupported-operation]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1
    IMPLICIT_1_MASK: tl.constexpr = (1 << (MBITS_IMPLICIT - 1)) - 1
    # pyrefly: ignore [bad-assignment]
    SHIFT_MASK: tl.constexpr = 31
    G: tl.constexpr = D // GROUP_SIZE
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1
    ROW_BYTES: tl.constexpr = D // 2 + G

    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = r < n_rows
    if HAS_ROW_IDX:
        src = tl.load(row_idx + r, mask=row_mask, other=0).to(tl.int64)
    else:
        src = r.to(tl.int64)
    g = tl.arange(0, G)
    col = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    a = tl.load(
        A + src[:, None, None] * D + col[None, :, :],
        mask=row_mask[:, None, None],
        other=0,
    )

    # Scaling step, fbgemm's.
    group_max = tl.max(tl.abs(a), axis=2)
    group_max = tl.where(group_max == 0, FP32_MIN_NORMAL, group_max)
    group_exp = _compute_exp(group_max, ROUNDING_MODE, None, MBITS)
    group_exp = group_exp - EBITS
    group_exp = tl.clamp(group_exp, -127, 125)
    group_exp_int = group_exp.to(tl.int32)
    if MUL_RCP:
        # 1 / scale: below -126 the scale is 2^-127 (also for a NaN exponent,
        # whose int conversion is target-defined).
        rcp_exp = FP32_EXP_BIAS - tl.maximum(group_exp_int, -127)
        rcp = (rcp_exp << FP32_EXP_OFFSET).to(tl.float32, bitcast=True)
        scaled_a = a * rcp[:, :, None]
    else:
        scale = tl.where(
            group_exp_int >= -126,
            ((group_exp_int + 127) << 23).to(tl.float32, bitcast=True),
            2.0**-127,
        )
        scaled_a = a / scale[:, :, None]
    group_start = r.to(tl.int64)[:, None] * ROW_BYTES + g[None, :] * PACKED_GROUP_SIZE
    tl.store(
        out + group_start + GROUP_SIZE // 2,
        (group_exp + FP32_EXP_BIAS).to(tl.uint8),
        mask=row_mask[:, None],
    )

    # Quantization step, fbgemm's.
    scaled_a = scaled_a.to(tl.int32, bitcast=True)
    sign_bit = (scaled_a >> FP32_SIGN_OFFSET) & SIGN_MASK
    biased_exp = (scaled_a & FP32_EXP_MASK) >> FP32_EXP_OFFSET
    trailing_mantissa = scaled_a & FP32_MANTISSA_MASK
    new_biased_exp = biased_exp - FP32_EXP_BIAS + FP4_EXP_BIAS
    exp_diff = tl.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)
    exp_diff = tl.minimum(exp_diff, MAX_FP32_MANTISSA_BITS)
    is_subnorm = biased_exp == 0
    mantissa = tl.where(
        is_subnorm, trailing_mantissa, trailing_mantissa + IMPLIED_1_BIT
    )
    fp32_sig_bits = tl.where(is_subnorm, 23, 24).to(tl.int32)
    # Shift amounts reach 45 for tiny values; clamping to 31 gives 0 on every
    # backend, as fbgemm's quantize_mx4 does.
    mantissa = mantissa >> tl.minimum(fp32_sig_bits + exp_diff - MBITS_IMPLICIT - 1, 31)
    mantissa = (mantissa + 1) >> 1
    overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
    mantissa = tl.where(overflow & (~is_subnorm), mantissa >> 1, mantissa)
    new_biased_exp = tl.where(
        (new_biased_exp <= 0) & (mantissa == 2), 1, new_biased_exp
    )
    mantissa = mantissa & IMPLICIT_1_MASK
    new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
    mantissa = tl.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)
    new_biased_exp = tl.maximum(
        tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD), 0
    )
    mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
    mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value
    low_mx4, high_mx4 = tl.split(
        tl.reshape(mx4_value, [BLOCK_R, G, GROUP_SIZE // 2, 2])
    )
    packed_mx4 = ((high_mx4 << 4) | low_mx4).to(tl.int8)
    tl.store(
        out + group_start[:, :, None] + tl.arange(0, GROUP_SIZE // 2)[None, None, :],
        packed_mx4,
        mask=row_mask[:, None, None],
    )


def mx4_encode_rows(
    a: torch.Tensor,
    row_idx: "torch.Tensor | None",
    D: int,
    group_size: int,
    rounding_mode: int,
    block_r: int = 4,
    num_warps: int = 4,
    mul_rcp: bool = True,
) -> torch.Tensor:
    """Quantizes rows `row_idx` of `a`, a contiguous float tensor of D-value
    rows (all of its rows without a row map), into a flat uint8 fbgemm MX4
    payload. Bitwise equal to the payload of fp32_to_mx4 (on GPU
    triton_quantize_mx4) of those rows for mx4v2_supported(D, group_size) and
    a rounding mode other than stochastic."""
    assert mx4v2_supported(D, group_size) and int(rounding_mode) != 3
    n_rows = a.numel() // D if row_idx is None else row_idx.numel()
    out = torch.empty(
        n_rows * (D // 2 + D // group_size), device=a.device, dtype=torch.uint8
    )
    if n_rows == 0:
        return out
    _mx4v2_encode_kernel[(triton.cdiv(n_rows, block_r),)](
        a,
        a if row_idx is None else row_idx,
        out,
        n_rows,
        D=D,
        GROUP_SIZE=group_size,
        EBITS=2,
        MBITS=1,
        ROUNDING_MODE=int(rounding_mode),
        FP4_EXP_BIAS=1,
        BLOCK_R=block_r,
        HAS_ROW_IDX=row_idx is not None,
        MUL_RCP=mul_rcp,
        num_warps=num_warps,
    )
    return out


# The decode loads each packed row once, rebuilds the E2M1 values with integer
# ops instead of table loads and stores whole rows with 16-byte stores. Its
# arithmetic is fbgemm's (_kernel_dequantize_mx4): the same table values, the
# scale built from the exponent byte, scale * value in fp32, one cast on store.
# So the decoded bits equal fbgemm's for any payload bytes and row map.


@triton.jit
def _e2m1_to_fp32(code):
    """The fp32 E2M1 value of the low 4 bits s|e1|e0|m of the int32 `code`,
    fbgemm's E2M1_LUT entry: (code & 7) << 22 is the fp32 2^(e-127)(1+m/2),
    or m * 2^-127 for e == 0, so times 2^126 it is the magnitude exactly, and
    adding +0.0 turns code 8 into the table's +0.0."""
    bits = ((code & 7) << 22) | ((code & 8) << 28)
    return tl.fma(bits.to(tl.float32, bitcast=True), 2.0**126, 0.0)


@triton.jit(do_not_specialize=["n_rows"])
def _mx4v2_decode_kernel(
    A,
    row_idx,
    out,
    n_rows,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_ROW_IDX: tl.constexpr,
) -> None:
    """Row j of `out` is the dequantized packed row row_idx[j] (row j without
    a row map) of the MX4 payload A, for the BLOCK_R rows j of this program:
    scale * value in fp32, the low nibble first, one cast on store."""
    G: tl.constexpr = D // GROUP_SIZE
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1
    ROW_BYTES: tl.constexpr = D // 2 + G

    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = r < n_rows
    if HAS_ROW_IDX:
        src = tl.load(row_idx + r, mask=row_mask, other=0).to(tl.int64)
    else:
        src = r.to(tl.int64)
    g = tl.arange(0, G)
    group_start = src[:, None] * ROW_BYTES + g[None, :] * PACKED_GROUP_SIZE
    a = tl.load(
        A + group_start[:, :, None] + tl.arange(0, GROUP_SIZE // 2)[None, None, :],
        mask=row_mask[:, None, None],
        other=0,
    ).to(tl.int32)
    exp = tl.load(A + group_start + GROUP_SIZE // 2, mask=row_mask[:, None], other=0)
    exp = exp.to(tl.int32) - 127
    scale = tl.where(
        exp >= -126,
        ((exp + 127) << 23).to(tl.float32, bitcast=True),
        2.0**-127,
    )
    scaled_fp32 = tl.interleave(
        scale[:, :, None] * _e2m1_to_fp32(a), scale[:, :, None] * _e2m1_to_fp32(a >> 4)
    )
    col = g[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    tl.store(
        out + r.to(tl.int64)[:, None, None] * D + col[None, :, :],
        scaled_fp32.to(out.dtype.element_ty),
        mask=row_mask[:, None, None],
    )


def mx4_decode_rows(
    payload: torch.Tensor,
    row_idx: "torch.Tensor | None",
    n_rows: int,
    D: int,
    group_size: int,
    output_dtype: torch.dtype,
    block_r: int = 4,
    num_warps: int = 4,
) -> torch.Tensor:
    """Dequantizes packed rows `row_idx` (rows 0 .. n_rows - 1 without a row
    map) of `payload`, a flat uint8 fbgemm MX4 payload of D-value rows, into
    a flat [n_rows * D] tensor of output_dtype (float32 or bfloat16). Bitwise
    equal to those rows of triton_dequantize_mx4(payload, group_size,
    output_dtype=output_dtype) for mx4v2_supported(D, group_size)."""
    assert mx4v2_supported(D, group_size)
    assert output_dtype in (torch.float32, torch.bfloat16)
    out = torch.empty(n_rows * D, device=payload.device, dtype=output_dtype)
    if n_rows == 0:
        return out
    _mx4v2_decode_kernel[(triton.cdiv(n_rows, block_r),)](
        payload,
        payload if row_idx is None else row_idx,
        out,
        n_rows,
        D=D,
        GROUP_SIZE=group_size,
        BLOCK_R=block_r,
        HAS_ROW_IDX=row_idx is not None,
        num_warps=num_warps,
    )
    return out
