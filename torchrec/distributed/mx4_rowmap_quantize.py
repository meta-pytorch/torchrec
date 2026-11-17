#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""MX4 quantize that reads its input rows through a row map.

All2All_Seq_Req's forward permutes its rows by the forward recat and then
quantizes them to fbgemm MX4. With D a multiple of the group size every row is
a whole number of groups, and each group is quantized on its own, so writing
quantize(input row row_idx[j]) to packed row j yields the payload of the
permuted rows in one pass, without materializing the permuted rows or
permuting the payload.
"""

import math

import torch
import triton
import triton.language as tl
from fbgemm_gpu.triton.common import get_mx4_exp_bias, RoundingMode
from fbgemm_gpu.triton.quantize import _compute_exp, _kernel_quantize_mx4

# Groups per step, triton_quantize_mx4's GROUP_LOAD.
MX4_ROWMAP_GROUP_LOAD = 64


@triton.jit
def _kernel_quantize_mx4_rowmap(
    A,
    row_idx,
    out,
    n_rows,
    GROUPS_PER_THREAD,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantizes rows row_idx[0], row_idx[1], ... of the [*, D] input A into
    the packed MX4 payload `out` of n_rows rows.

    This is _kernel_quantize_mx4 of fbgemm_gpu/triton/quantize.py for an
    [n_rows * D] input with no row padding, rounding modes other than
    stochastic and no stochastic casting, except for where values are loaded
    from: value i of the output order is read from input row row_idx[i // D],
    column i % D. From the loaded [GROUP_LOAD * GROUP_SIZE] block on, the code
    is that kernel's, and it runs over the same work split. Needs
    GROUP_LOAD * GROUP_SIZE <= D.
    """
    # Define Constant Expressions.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    FP32_SIGN_OFFSET: tl.constexpr = 31  # type: ignore[Incompatible variable type]
    SIGN_MASK: tl.constexpr = 0x1  # type: ignore[Incompatible variable type]
    FP32_MANTISSA_MASK: tl.constexpr = 0x007FFFFF  # type: ignore[Incompatible variable type]
    # FP4 has 2 mantissa bits, one explicit one implicit.
    MBITS_IMPLICIT: tl.constexpr = MBITS + 1  # type: ignore[Incompatible variable type]
    MAX_FP32_MANTISSA_BITS: tl.constexpr = 24  # type: ignore[Incompatible variable type]
    IMPLIED_1_BIT: tl.constexpr = 1 << 23  # type: ignore[Incompatible variable type]
    FP32_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1  # type: ignore[Incompatible variable type]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1  # type: ignore[Incompatible variable type]
    IMPLICIT_1_MASK = (1 << (MBITS_IMPLICIT - 1)) - 1

    # Get the current thread number.
    pid = tl.program_id(0)
    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        n_rows = tl.cast(n_rows, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    PACKED_GROUP_SIZE: tl.constexpr = GROUP_SIZE // 2 + 1  # type: ignore[Incompatible variable type]
    NUM_GROUPS = n_rows * (D // GROUP_SIZE)
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 2 + GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 2 + NUM_GROUPS

    # Find starting offsets for this thread.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = output_start + GROUP_SIZE // 2
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 2))
    # We need to shift output offsets to make space for shared exponent storage.
    output_offset += output_offset // (GROUP_SIZE // 2) + output_start
    # Now create offsets for writing the shared exponent.
    exp_offset = tl.arange(0, GROUP_LOAD) * PACKED_GROUP_SIZE + exp_start

    # A block of GROUP_LOAD * GROUP_SIZE <= D values spans at most two output
    # rows, whose source rows are loaded one block ahead, so that each step
    # waits on a single round of loads, like _kernel_quantize_mx4.
    input_mask = (input_offset < (n_rows * D)) & (
        input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE)
    )
    first_row = input_start // D
    src_row0 = tl.load(row_idx + first_row, mask=first_row < n_rows, other=0)
    src_row1 = tl.load(row_idx + first_row + 1, mask=first_row + 1 < n_rows, other=0)

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        # Load a block of values.
        src_row = tl.where(input_offset // D == first_row, src_row0, src_row1)
        a = tl.load(
            A + (src_row.to(tl.int64) * D + input_offset % D),
            mask=input_mask,
            other=0,
        )
        input_offset += GROUP_LOAD * GROUP_SIZE
        input_mask = (input_offset < (n_rows * D)) & (
            input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE)
        )
        first_row = (input_start + (_k + 1) * (GROUP_LOAD * GROUP_SIZE)) // D
        src_row0 = tl.load(row_idx + first_row, mask=first_row < n_rows, other=0)
        src_row1 = tl.load(
            row_idx + first_row + 1, mask=first_row + 1 < n_rows, other=0
        )

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, FP32_MIN_NORMAL, group_max)
        # Compute shared exponent using specified rounding mode.
        group_exp = _compute_exp(group_max, ROUNDING_MODE, None, MBITS)
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - EBITS
        # Make sure exponent is in valid range.
        group_exp = tl.clamp(group_exp, -127, 125)

        # Next we scale A in preparation for quantization.
        # Compute scale = 2^group_exp exactly via FP32 bit construction instead of a slow
        # per-group FP64 exp2. group_exp is an integer in [-127, 125]; for e >= -126, 2^e is
        # a normal fp32 (zero mantissa) so we set the exponent field directly. e == -127 ->
        # 2^-127 is subnormal and cannot be built that way, so special-case it.
        group_exp_int = group_exp.to(tl.int32)
        scale = tl.where(
            group_exp_int >= -126,
            ((group_exp_int + 127) << 23).to(tl.float32, bitcast=True),
            2.0**-127,
        )
        # Apply scale to input. We do this by broadcasting scale.
        scaled_a = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]) / tl.reshape(
            scale, [GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp32_exp_bias for compatibility with cuda dequant.
        tl.store(
            out + exp_offset,
            (group_exp + FP32_EXP_BIAS).to(tl.uint8),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < OUTPUT_SIZE)
            & (exp_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Quantization step
        ###################

        # During quantization, we're going to be doing a lot of bitwise operations.
        # This is easier to work with in int32.
        scaled_a = scaled_a.to(tl.int32, bitcast=True)

        # Extract sign bit of value.
        sign_bit = (scaled_a >> FP32_SIGN_OFFSET) & SIGN_MASK

        # Extract exponent.
        biased_exp = (scaled_a & FP32_EXP_MASK) >> FP32_EXP_OFFSET

        # Extract mantissa.
        trailing_mantissa = scaled_a & FP32_MANTISSA_MASK

        # Adjust exponent bias for FP4.
        new_biased_exp = biased_exp - FP32_EXP_BIAS + FP4_EXP_BIAS

        # Compute difference between ideal exponent and what fp4 can represent.
        exp_diff = tl.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)

        # Clip this difference to maximum number of fp32 mantissa bits.
        exp_diff = tl.minimum(exp_diff, MAX_FP32_MANTISSA_BITS)

        # Now we round our fp32 mantissa down to fp4.
        is_subnorm = biased_exp == 0
        # Add implied 1 bit to normal values.
        mantissa = tl.where(
            is_subnorm, trailing_mantissa, trailing_mantissa + IMPLIED_1_BIT
        )
        # Compute base number of bits corresponding to the mantissa, smaller for subnorms
        # since implied one is included in exp_diff.
        fp32_sig_bits = tl.where(is_subnorm, 23, 24).to(tl.int32)
        # Now we're ready to shift down to target bitwidth (with an extra bit for rounding).
        # Clamped to 31 like fbgemm's quantize_mx4: tiny values give 0 on every backend.
        mantissa = mantissa >> tl.minimum(
            fp32_sig_bits + exp_diff - MBITS_IMPLICIT - 1, 31
        )
        # Perform rounding by adding 1 and shifting down.
        mantissa = (mantissa + 1) >> 1

        # Check for overflow and adjust exponent accordingly.
        overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
        # Allow subnorms to overflow into normals, otherwise shift away overflow.
        mantissa = tl.where(overflow and (not is_subnorm), mantissa >> 1, mantissa)
        # Special case where a value is subnormal and has a large mantissa, overflow it.
        new_biased_exp = tl.where(
            (new_biased_exp <= 0) and (mantissa == 2), 1, new_biased_exp
        )
        # Remove implicit 1.
        mantissa = mantissa & IMPLICIT_1_MASK
        # Add overflow to exponent.
        new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
        # If exp overflows, set mantissa to maximum value (equivalent to clamping).
        mantissa = tl.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)

        # Construct FP4 value from components.
        new_biased_exp = tl.maximum(
            tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD), 0
        )
        mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
        mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            packed_mx4,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        exp_offset += GROUP_LOAD * PACKED_GROUP_SIZE
        output_offset += GROUP_LOAD * PACKED_GROUP_SIZE


def quantize_mx4_rowmap(
    a: torch.Tensor,
    row_idx: torch.Tensor,
    D: int,
    group_size: int,
    rounding_mode: int,
) -> torch.Tensor:
    """Quantizes rows `row_idx` of `a`, a contiguous float tensor of D-value
    rows, into a flat uint8 fbgemm MX4 (ebits=2, mbits=1) payload of
    len(row_idx) rows. Bitwise equal to
    `triton_quantize_mx4(a.view(-1, D)[row_idx].view(-1), group_size,
    rounding_mode=rounding_mode)` for a power-of-two group_size with
    MX4_ROWMAP_GROUP_LOAD * group_size <= D and a rounding mode other than
    stochastic.
    """
    GROUP_LOAD = MX4_ROWMAP_GROUP_LOAD
    assert group_size & (group_size - 1) == 0 and GROUP_LOAD * group_size <= D
    n_rows = row_idx.numel()
    num_groups = n_rows * (D // group_size)
    out = torch.empty(
        (num_groups * group_size) // 2 + num_groups, device=a.device, dtype=torch.uint8
    )
    if n_rows == 0:
        return out
    # The work split of triton_quantize_mx4 for n_rows * D values.
    num_threads = math.ceil(math.sqrt(n_rows * D))
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1
    _kernel_quantize_mx4_rowmap[(triton.cdiv(num_groups, groups_per_thread),)](
        a,
        row_idx,
        out,
        n_rows,
        groups_per_thread,
        D=D,
        GROUP_SIZE=group_size,
        EBITS=2,
        MBITS=1,
        ROUNDING_MODE=rounding_mode,
        FP4_EXP_BIAS=get_mx4_exp_bias(2),
        GROUP_LOAD=GROUP_LOAD,
        USE_INT64=use_int64,
    )
    return out


# Row-chunked MX4 quantize of a [rows, D] tensor (quantize_mx4_rows_chunked):
# with D a multiple of the group size, packed rows r0 .. r1 - 1 of the payload
# are the payload of input rows r0 .. r1 - 1 alone, so fbgemm's quantize kernel
# can run once per row chunk with any number of groups per program. Each chunk
# then spans less than 2 GiB, which the AMD Triton backend needs to load it with
# buffer ops, and each program runs whole 64-group steps.

# Largest byte extent of a pointer argument that the AMD Triton backend
# specializes for buffer ops (HIPBackend.is_within_2gb).
MX4_BUFFER_OPS_MAX_BYTES = 2**31 - 1

# Row chunks start at multiples of this many rows, which keeps the chunks'
# input and payload pointers as 16-byte aligned as the whole tensors'.
MX4_CHUNK_ROW_ALIGN = 16


class PtrRangeView:
    """A contiguous tensor as a Triton pointer argument whose byte range is
    the tensor's own extent. The AMD backend gives a pointer argument with a
    ptr_range() of at most MX4_BUFFER_OPS_MAX_BYTES the buffer-ops
    specialization, and otherwise goes by the size of the tensor's whole
    storage, which for a slice of a large tensor is the large tensor's.
    """

    def __init__(self, t: torch.Tensor) -> None:
        assert t.is_contiguous()
        self.tensor = t
        self.dtype: torch.dtype = t.dtype
        self.device: torch.device = t.device

    def data_ptr(self) -> int:
        return self.tensor.data_ptr()

    def ptr_range(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


def _ptr_arg(t: torch.Tensor) -> "torch.Tensor | PtrRangeView":
    import os

    # The Triton interpreter takes tensors only.
    if t.is_cuda and os.environ.get("TRITON_INTERPRET", "0") != "1":
        return PtrRangeView(t)
    return t


def mx4_rows_chunk_plan(
    rows: int,
    D: int,
    element_size: int,
    group_size: int,
    max_chunk_bytes: int = MX4_BUFFER_OPS_MAX_BYTES,
) -> tuple[int, int, int] | None:
    """Returns (rows per chunk, number of chunks, groups per program) for
    quantize_mx4_rows_chunked of a [rows, D] input of element_size-byte
    values, rows > 0, or None when MX4_CHUNK_ROW_ALIGN rows exceed
    max_chunk_bytes.

    The chunks are balanced, start at multiples of MX4_CHUNK_ROW_ALIGN rows and
    span at most max_chunk_bytes of input and of payload. Groups per program is
    triton_quantize_mx4's for the whole input rounded to the nearest whole
    number of MX4_ROWMAP_GROUP_LOAD-group steps: the kernel computes every step
    on all of its lanes, so a program whose last step is mostly masked costs as
    much as one whose last step is full.
    """
    GROUP_LOAD = MX4_ROWMAP_GROUP_LOAD
    groups_per_row = D // group_size
    row_bytes = max(D * element_size, D // 2 + groups_per_row)
    max_rows = max_chunk_bytes // row_bytes // MX4_CHUNK_ROW_ALIGN * MX4_CHUNK_ROW_ALIGN
    if max_rows <= 0:
        return None
    num_chunks = triton.cdiv(rows, max_rows)
    rows_per_chunk = (
        triton.cdiv(triton.cdiv(rows, num_chunks), MX4_CHUNK_ROW_ALIGN)
        * MX4_CHUNK_ROW_ALIGN
    )
    # triton_quantize_mx4's work split for the whole input.
    num_groups = rows * groups_per_row
    num_threads = math.ceil(math.sqrt(rows * D))
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    groups_per_program = max(
        GROUP_LOAD, (groups_per_thread + GROUP_LOAD // 2) // GROUP_LOAD * GROUP_LOAD
    )
    return rows_per_chunk, num_chunks, groups_per_program


def quantize_mx4_rows_chunked(
    a: torch.Tensor,
    group_size: int,
    rounding_mode: int | None,
    rows_per_chunk: int,
    groups_per_program: int,
) -> torch.Tensor:
    """Quantizes a contiguous [rows, D] float tensor into a flat uint8 fbgemm
    MX4 (ebits=2, mbits=1) payload with fbgemm's _kernel_quantize_mx4, launched
    once per chunk of rows_per_chunk rows with groups_per_program groups per
    program (see mx4_rows_chunk_plan). A rounding_mode of None is
    RoundingMode.even, as in fp32_to_mx4.

    Bitwise equal to
    `fp32_to_mx4(a, group_size, rounding_mode=rounding_mode).view(-1)` (on GPU
    `triton_quantize_mx4`) for D % group_size == 0, a power-of-two group_size
    and a rounding mode other than stochastic: whatever the work split, the
    kernel quantizes each group of group_size values on its own into the
    group_size // 2 + 1 bytes at byte group_index * (group_size // 2 + 1), so
    packed row r starts at byte r * (D // 2 + D // group_size).
    """
    rows, D = a.shape
    assert a.is_contiguous() and D % group_size == 0
    if rounding_mode is None:
        rounding_mode = RoundingMode.even
    groups_per_row = D // group_size
    row_bytes = D // 2 + groups_per_row
    out = torch.empty(rows * row_bytes, device=a.device, dtype=torch.uint8)
    for r0 in range(0, rows, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows)
        grid = triton.cdiv((r1 - r0) * groups_per_row, groups_per_program)
        _kernel_quantize_mx4[(grid,)](
            _ptr_arg(a[r0:r1]),
            _ptr_arg(out[r0 * row_bytes : r1 * row_bytes]),
            rand_bits=None,
            M=r1 - r0,
            K=D,
            GROUPS_PER_ROW=groups_per_row,
            GROUPS_PER_THREAD=groups_per_program,
            ROW_PADDING=0,
            GROUP_SIZE=group_size,
            EBITS=2,
            MBITS=1,
            ROUNDING_MODE=rounding_mode,
            STOCHASTIC_CASTING=False,
            FP4_EXP_BIAS=get_mx4_exp_bias(2),
            GROUP_LOAD=MX4_ROWMAP_GROUP_LOAD,
            USE_INT64=grid * groups_per_program * group_size > 2**31 - 1,
        )
    return out
