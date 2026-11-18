#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Segment permute with an atomic-free, deterministic backward.

The forward is fbgemm's ``keyed_jagged_index_select_dim1``. Its backward
zero-fills the input gradient and atomically adds every output segment back.
Here each input segment instead sums its source segments in a fixed order, in
FP32, and writes its slice once. When ``recat`` is a permutation every input
segment has exactly one source, so the backward is a copy and bit-identical to
fbgemm's.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

_CHUNK = 1024


@triton.jit
def _segment_gather_sum_kernel(
    grad_output,
    grad_input,
    input_offsets,
    chunk_offsets,
    chunk_segment,
    source_ptr,
    source_starts,
    num_segments,
    CHUNK: tl.constexpr,
) -> None:
    # One program per chunk: a grid-stride chunk loop measured ~3x slower.
    pid = tl.program_id(0)
    segment = tl.load(chunk_segment + pid)
    if segment >= num_segments:
        return
    base = tl.load(input_offsets + segment).to(tl.int64)
    length = tl.load(input_offsets + segment + 1).to(tl.int64) - base
    start = (pid - tl.load(chunk_offsets + segment)).to(tl.int64) * CHUNK
    offsets = start + tl.arange(0, CHUNK)
    mask = offsets < length
    first = tl.load(source_ptr + segment)
    last = tl.load(source_ptr + segment + 1)
    if last - first == 1:
        src = tl.load(source_starts + first).to(tl.int64)
        values = tl.load(grad_output + src + offsets, mask=mask)
        tl.store(grad_input + base + offsets, values, mask=mask)
    else:
        acc = tl.zeros([CHUNK], dtype=tl.float32)
        for k in tl.range(first, last):
            src = tl.load(source_starts + k).to(tl.int64)
            acc += tl.load(grad_output + src + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
        tl.store(
            grad_input + base + offsets,
            acc.to(grad_input.dtype.element_ty),
            mask=mask,
        )


def _segment_gather_sum(
    grad_output: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    input_numel: int,
) -> torch.Tensor:
    grad_output = grad_output.contiguous()
    num_segments = segment_sizes.numel()
    grad_input = grad_output.new_empty(input_numel)
    if input_numel == 0 or num_segments == 0:
        return grad_input.zero_()
    sizes = segment_sizes.to(torch.int64)
    input_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(sizes)
    recat = recat.to(torch.int64)
    output_starts = torch.ops.fbgemm.asynchronous_exclusive_cumsum(sizes[recat])
    sorted_recat, order = torch.sort(recat, stable=True)
    source_ptr = torch.searchsorted(
        sorted_recat,
        torch.arange(num_segments + 1, device=recat.device, dtype=torch.int64),
    )
    source_starts = output_starts[order]
    chunk_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        (sizes + _CHUNK - 1) // _CHUNK
    )
    # Sum of per-segment chunk counts is at most this; spare programs exit early.
    grid = triton.cdiv(input_numel, _CHUNK) + num_segments
    chunk_segment = (
        torch.searchsorted(
            chunk_offsets,
            torch.arange(grid, device=recat.device, dtype=torch.int64),
            right=True,
        )
        - 1
    )
    _segment_gather_sum_kernel[(grid,)](
        grad_output,
        grad_input,
        input_offsets,
        chunk_offsets,
        chunk_segment,
        source_ptr,
        source_starts,
        num_segments,
        # pyrefly: ignore[bad-argument-type]
        CHUNK=_CHUNK,
        # pyrefly: ignore[unexpected-keyword]
        num_warps=4,
    )
    return grad_input


@torch.library.custom_op(
    "torchrec::permute_segments_gather_backward",
    mutates_args=(),
    schema="(Tensor values, Tensor segment_sizes, Tensor recat, SymInt output_size) -> Tensor",
)
def permute_segments_gather_backward(
    values: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """``_permute_tensor_by_segments`` for CUDA values without weights, whose
    backward is ``_segment_gather_sum`` instead of fbgemm's atomic add."""
    return torch.ops.fbgemm.keyed_jagged_index_select_dim1(
        values=values,
        lengths=segment_sizes,
        offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(segment_sizes),
        indices=recat,
        batch_size=segment_sizes.numel(),
        weights=None,
        selected_lengths_sum=output_size,
    )[0]


@permute_segments_gather_backward.register_fake
def _fake_permute_segments_gather_backward(
    values: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    return values.new_empty(output_size)


@torch.library.custom_op(
    "torchrec::segment_gather_sum",
    mutates_args=(),
    schema="(Tensor grad_output, Tensor segment_sizes, Tensor recat, SymInt input_numel) -> Tensor",
)
def segment_gather_sum(
    grad_output: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    input_numel: int,
) -> torch.Tensor:
    return _segment_gather_sum(grad_output, segment_sizes, recat, input_numel)


@segment_gather_sum.register_fake
def _fake_segment_gather_sum(
    grad_output: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    input_numel: int,
) -> torch.Tensor:
    return grad_output.new_empty(input_numel)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    values, segment_sizes, recat, _output_size = inputs
    ctx.save_for_backward(segment_sizes, recat)
    ctx.input_numel = values.numel()


def _backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None]:
    segment_sizes, recat = ctx.saved_tensors
    return (
        segment_gather_sum(grad_output, segment_sizes, recat, ctx.input_numel),
        None,
        None,
        None,
    )


permute_segments_gather_backward.register_autograd(
    _backward, setup_context=_setup_context
)
