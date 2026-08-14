#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Triton replacement for `fbgemm.permute_2D_sparse_data`, for the two regimes
where fbgemm's per-segment launch shape costs more than the copy itself.

fbgemm hands each block 32 consecutive segments and picks uint4 vs scalar loads
per segment from a 16-byte alignment test. That is efficient when there are few,
long, aligned segments. It degrades on two axes this model hits hard:

- **many segments**: per-segment overhead dominates. At 5.2M segments averaging
  0.6 elements it spends 610 us moving 3.1M values.
- **length skew across keys**: segments are laid out key-major, so a block's cost
  is set by which key it landed on. Measured on the dominant shape, fbgemm goes
  1470 -> 1889 -> 2173 us as key-level lognormal sigma goes 4 -> 6 -> 8, while
  the per-segment Triton kernel stays flat at 1172 -> 1257 -> 1300.

Below ~700k segments fbgemm wins on every traced shape and this module defers to
it -- see `should_use_triton`. Correctness is `torch.equal`, never a tolerance:
this is a pure gather-copy with no arithmetic, and a tolerance would hide the
failure mode that matters, a segment-boundary off-by-one whose values still look
plausible.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger(__name__)

# Segment counts below this went to fbgemm on every shape in the trace: it needs
# 32-103 us there, against 94-320 us for either Triton kernel. The wins all sit at
# >=786k segments, the losses all at <=654k, so the split is unambiguous.
MIN_SEGMENTS = 700_000

# Above this mean segment length one program per segment beats the load-balanced
# kernel (1172 vs 1522 us at mean 298); below it the ordering reverses and grows
# to 2.9x by mean 0.6.
PERSEG_MIN_MEAN = 128


@triton.jit
def _permute_2d_perseg_kernel(
    values_ptr,
    out_ptr,
    w_ptr,
    w_out_ptr,
    in_off_ptr,
    out_off_ptr,
    permute_ptr,
    B,
    n_seg,
    HAS_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per segment, chunked over BLOCK.

    BLOCK=1024 is not a tuning oversight: sweeping 64..1024 on the real shapes
    under key-stratified skew never beat 1024, because the long-key segments that
    dominate the runtime are far longer than a block anyway.
    """
    pid = tl.program_id(0)
    if pid >= n_seg:
        return

    b = pid % B
    t = pid // B
    src_t = tl.load(permute_ptr + t)

    out_start = tl.load(out_off_ptr + pid)
    seg_len = tl.load(out_off_ptr + pid + 1) - out_start
    in_start = tl.load(in_off_ptr + src_t * B + b)

    for base in range(0, seg_len, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < seg_len
        v = tl.load(values_ptr + in_start + offs, mask=mask)
        tl.store(out_ptr + out_start + offs, v, mask=mask)
        if HAS_W:
            w = tl.load(w_ptr + in_start + offs, mask=mask)
            tl.store(w_out_ptr + out_start + offs, w, mask=mask)


@triton.jit
def _permute_2d_blocked_kernel(
    values_ptr,
    out_ptr,
    w_ptr,
    w_out_ptr,
    in_off_ptr,
    out_off_ptr,
    permute_ptr,
    seg_lo_ptr,
    seg_hi_ptr,
    B,
    n_out,
    HAS_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per BLOCK of *output* elements, not per segment.

    Work per program is exactly BLOCK however the lengths fall, which is the whole
    point for short segments: giving one program a whole segment wastes a program
    on ~1 element, and giving one *lane* per segment serialises the tile on its
    longest member. Each lane binary-searches its own segment inside
    [seg_lo, seg_hi] -- the only segments this block can touch, precomputed by a
    sync-free searchsorted -- so the search is a few steps over an offset array
    small enough to stay cached.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_out

    seg = tl.zeros([BLOCK], tl.int64) + tl.load(seg_lo_ptr + pid)
    r = tl.zeros([BLOCK], tl.int64) + tl.load(seg_hi_ptr + pid)
    while tl.max(tl.where(m, r - seg, 0)) > 0:
        mid = (seg + r + 1) // 2
        c = tl.load(out_off_ptr + mid) <= offs
        seg = tl.where(c, mid, seg)
        r = tl.where(c, r, mid - 1)

    out_start = tl.load(out_off_ptr + seg)
    src_t = tl.load(permute_ptr + seg // B)
    src = tl.load(in_off_ptr + src_t * B + seg % B) + (offs - out_start)
    tl.store(out_ptr + offs, tl.load(values_ptr + src, mask=m), mask=m)
    if HAS_W:
        tl.store(w_out_ptr + offs, tl.load(w_ptr + src, mask=m), mask=m)


def should_use_triton(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    permuted_lengths_sum: Optional[int],
) -> bool:
    """Decided from shapes and the caller's `permuted_lengths_sum` only.

    Nothing here reads device memory, so the dispatch costs no synchronisation.
    `permuted_lengths_sum` is fbgemm's fifth argument and every internal caller
    passes it; when it is absent the output size is not knowable without a
    device-to-host sync, and paying one per call is worse than anything this
    module can win back.
    """
    if permuted_lengths_sum is None:
        return False
    n_seg = permute.shape[0] * lengths.shape[1]
    return n_seg >= MIN_SEGMENTS


def triton_permute_2d_sparse_data(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    permuted_lengths_sum: Optional[int] = None,
    block: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Returns (permuted_lengths, permuted_values, permuted_weights).

    `permute` is NOT required to be a permutation of range(T): callers select a
    subset of keys and may repeat one, which is why fbgemm's signature carries a
    separate `permuted_lengths_sum`. The traced call sites really do this --
    P=24 against T=51, P=295 against T=267.
    """
    B = lengths.shape[1]
    has_w = weights is not None

    # Indexing by `permute` also covers the subset and repeated-key cases; the
    # output segment count follows `permute`, not the input's T.
    permuted_lengths = lengths[permute.long()]
    n_seg = permuted_lengths.numel()

    # Sized from the caller's sum rather than `permuted_lengths.sum()`: the latter
    # is an .item() device sync on every call. Sizing it as empty_like(values)
    # instead is silently wrong under a subset permute -- the trailing rows are
    # garbage and it surfaces much later as a split_with_sizes size mismatch
    # inside construct_jagged_tensors.
    n_out = permuted_lengths_sum
    if n_out is None:
        n_out = int(permuted_lengths.sum())

    if n_seg == 0 or values.numel() == 0 or n_out == 0:
        # fbgemm early-outs on T==0/B==0 (sparse_permute_2d.cu:240-249).
        return (
            permuted_lengths,
            values.new_empty(0) if n_out == 0 else values.clone(),
            (
                (weights.new_empty(0) if n_out == 0 else weights.clone())
                if has_w
                else None
            ),
        )

    # in_off is indexed by *source* segment and out_off by *output* segment; those
    # counts differ under a subset or repeating permute, so in_off must be sized
    # from `lengths`, not from n_seg.
    in_off = torch.zeros(lengths.numel() + 1, dtype=torch.int64, device=values.device)
    out_off = torch.zeros(n_seg + 1, dtype=torch.int64, device=values.device)
    torch.cumsum(lengths.reshape(-1), 0, out=in_off[1:])
    torch.cumsum(permuted_lengths.reshape(-1), 0, out=out_off[1:])

    out = values.new_empty(n_out)
    w_out = weights.new_empty(n_out) if has_w else values

    if n_out >= n_seg * PERSEG_MIN_MEAN:
        _permute_2d_perseg_kernel[(n_seg,)](
            values,
            out,
            weights if has_w else values,
            w_out,
            in_off,
            out_off,
            permute,
            B,
            n_seg,
            # pyrefly: ignore[bad-argument-type]
            HAS_W=has_w,
            # pyrefly: ignore[bad-argument-type]
            BLOCK=block,
        )
    else:
        nb = triton.cdiv(n_out, block)
        starts = torch.arange(nb, device=values.device, dtype=torch.int64) * block
        ends = torch.clamp(starts + block - 1, max=n_out - 1)
        _permute_2d_blocked_kernel[(nb,)](
            values,
            out,
            weights if has_w else values,
            w_out,
            in_off,
            out_off,
            permute,
            torch.searchsorted(out_off, starts, right=True) - 1,
            torch.searchsorted(out_off, ends, right=True) - 1,
            B,
            n_out,
            # pyrefly: ignore[bad-argument-type]
            HAS_W=has_w,
            # pyrefly: ignore[bad-argument-type]
            BLOCK=block,
        )

    return permuted_lengths, out, (w_out if has_w else None)
