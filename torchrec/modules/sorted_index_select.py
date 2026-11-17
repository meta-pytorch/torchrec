#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
sorted_index_select: drop-in for ``embeddings.index_select(0, reverse_indices)``
whose BACKWARD uses a sorted segment-reduce instead of the native index_add
(``indexFuncLargeIndex`` bf16 atomic scatter). Forward is identical (a gather);
only the backward differs, so this is a training-only optimization.

Design (chunk-split, sync-free):
- Sort reverse_indices on GPU (radix sort; output shape == input shape, no D2H).
- Per-row offsets via searchsorted on the sorted indices (off[r] = #(idx < r)),
  which is atomic-free (NOT counts.scatter_add_, whose hot/sentinel row serializes
  ~N/2 atomicAdds onto one counter) and NOT bincount/unique_consecutive (which read
  a data-dependent size back to the host and stall). nch/nchoff are GPU cumsums.
- Each work item = (row, chunk) covering a CHUNK-sized slice of one row's sorted
  segment. A short row (1 chunk) does a single store (no atomic). A HOT row is
  split across ceil(count/CHUNK) programs, each reducing its slice in fp32
  registers then one atomic_add into the row's fp32 accumulator -> bounded
  atomics per hot row AND parallel (avoids the per-row serial long tail that
  regressed under skewed/zipf data).
- Sync-free launch: grid is a host-known static upper bound on the work-item
  count; the real total is read on-device (nchoff[num_rows]) and excess programs
  early-exit. Each program binary-searches nchoff to find its row (or reads it
  from an item->row map, see ONEFLOW_SEG_REDUCE_ITEM_ROW). No .item().
- sorted_index_select_split also splits the gathered rows (as
  construct_jagged_tensors does); with ONEFLOW_SEG_REDUCE_MULTISRC its backward
  reads the per-split gradients in place instead of their torch.cat.
"""

import logging
import os
from typing import (
    Any,
    cast,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import triton  # @manual
import triton.language as tl  # @manual
from torch.fx._symbolic_trace import is_fx_symbolic_tracing

logger: logging.Logger = logging.getLogger(__name__)

_CHUNK = 512  # sorted entries reduced per work item (bounds hot-row tail)
_BUF = 16  # rows buffered per vectorized load
_NUM_WARPS = 1
_NUM_STAGES = 2
# ATen's index_add_ switches to 64-bit index math once any operand reaches 2**31
# elements, which is ~2x slower per row on MI350; the sorted path has no such cliff.
_INDEX_ADD_32BIT_NUMEL = 1 << 31

# Env ONEFLOW_SEG_REDUCE_FUSED_PERM=1: _seg_chunk reads grad_out rows through the
# sort permutation instead of a materialized grad_out.index_select(0, perm).
# ONEFLOW_SEG_REDUCE_FUSED_PERM_CHECK=1 also compares the first few fused calls
# against the gathered path and falls back to it for good on a mismatch.
_FUSED_PERM_CHECK_CALLS = 3
_fused_perm_checks = 0
_fused_perm_disabled = False

# Env ONEFLOW_SEG_REDUCE_NARROW_SORT=1: sort the indices as int16/int32 keys when
# num_rows fits. Narrowing preserves order and the large-sort radix path is
# stable, so the permutation (and the fp32 summation order it fixes) is unchanged.
# ONEFLOW_SEG_REDUCE_NARROW_SORT_CHECK=1 also compares the first few narrowed
# permutations with the int64 sort and falls back to it for good on a mismatch.
# Duplicated in fbr/flash/triton/embedding_bwd_triton_override.py.
_NARROW_SORT_CHECK_CALLS = 3
_narrow_checks = 0
_narrow_sort_disabled = False

# Env ONEFLOW_SEG_REDUCE_ROW_DIRECT=1: for a non-fp32 grad_out, _seg_chunk writes
# every output row once in the grad_out dtype (an empty row gets one item that
# stores zeros) instead of filling an fp32 zeros buffer that a second pass casts.
# Hot rows still add their fp32 chunk sums atomically, into a small fp32 buffer
# that _seg_hot_rows then casts into place.
# ONEFLOW_SEG_REDUCE_ROW_DIRECT_CHECK=1 also compares the first few row-direct
# calls against the fp32-buffer path and falls back to it for good on a mismatch.
# Duplicated in fbr/flash/triton/embedding_bwd_triton_override.py.
_ROW_DIRECT_CHECK_CALLS = 3
_ROW_DIRECT_CHECK_ROWS = 1 << 18
_row_direct_checks = 0
_row_direct_disabled = False

# Env ONEFLOW_SEG_REDUCE_ITEM_ROW=1: each _seg_chunk program reads its row from an
# int32 item->row map that one searchsorted over nchoff builds per launch, instead
# of loading the total and binary-searching nchoff itself. The map holds the row
# the search finds (the r with nchoff[r] <= pid < nchoff[r + 1]) and num_rows for
# every pid >= total, so the same programs exit and the reduction is unchanged.
# ONEFLOW_SEG_REDUCE_ITEM_ROW_CHECK=1 also validates the first few maps against
# nchoff before their launch and falls back to the binary search for good on a
# mismatch.
_ITEM_ROW_CHECK_CALLS = 3
_item_row_checks = 0
_item_row_disabled = False

# Env ONEFLOW_SEG_REDUCE_MULTISRC=1: the backward of sorted_index_select_split
# passes the gradients of its (at most _MAX_SRC) splits to _seg_chunk_multisrc,
# which reads each sorted row from its split instead of from a torch.cat of the
# splits. The rows, their order and the reduction are those of _seg_chunk on the
# concatenation. Needs ONEFLOW_SEG_REDUCE_FUSED_PERM=1; otherwise, and for
# gradients whose layout differs from a fresh torch.cat output (see
# _split_sources), the backward concatenates. If the multi-source backward raises
# (e.g. the kernel does not compile), it logs the error, concatenates, and the
# path stays off for the rest of the process.
# ONEFLOW_SEG_REDUCE_MULTISRC_CHECK=1 also runs the first few multi-source
# backwards on the concatenated gradients and falls back to that for good on a
# mismatch.
_MAX_SRC = 8
# Gradient dtypes _seg_chunk_multisrc reads, as its G_TY.
_TL_DTYPES = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
_MULTISRC_CHECK_CALLS = 6
_MULTISRC_CHECK_ROWS = 1 << 16
_multisrc_checks = 0
_multisrc_disabled = False


@triton.jit
def _seg_chunk(
    g_ptr,  # grad_out [N, D]; gathered into sorted order unless HAS_PERM
    perm_ptr,  # [N] int64 sort permutation; sorted row i is g[perm[i]] if HAS_PERM
    off_ptr,  # [num_rows + 1] segment offsets in the sorted array
    nchoff_ptr,  # [num_rows + 1] cumsum of chunks-per-row; nchoff[num_rows] = total
    item_row_ptr,  # [grid] int32 row of each program (num_rows: exit) if ITEM_ROW
    out_ptr,  # [num_rows, D] fp32, or the grad_out dtype if ROW_DIRECT
    hcum_ptr,  # [num_rows + 1] exclusive cumsum of hot-row flags if ROW_DIRECT
    hot_ptr,  # [>= #hot rows, D] fp32 zeros for hot-row sums if ROW_DIRECT
    num_rows,
    D,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BUF: tl.constexpr,
    HAS_PERM: tl.constexpr,
    ROW_DIRECT: tl.constexpr,
    ITEM_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    if ITEM_ROW:
        row = tl.load(item_row_ptr + pid)
        if row >= num_rows:  # exactly the programs with pid >= total
            return
    else:
        total = tl.load(nchoff_ptr + num_rows)  # device read, NOT host
        if pid >= total:
            return
        # binary search: largest row r with nchoff[r] <= pid
        lo = 0
        hi = num_rows
        while hi - lo > 1:
            mid = (lo + hi) // 2
            v = tl.load(nchoff_ptr + mid)
            if v <= pid:
                lo = mid
            else:
                hi = mid
        row = lo
    base = tl.load(nchoff_ptr + row)
    nch_row = tl.load(nchoff_ptr + row + 1) - base
    chunk = pid - base
    rs = tl.load(off_ptr + row)
    re = tl.load(off_ptr + row + 1)
    seg_s = rs + chunk * CHUNK
    seg_e = tl.minimum(seg_s + CHUNK, re)
    col = tl.arange(0, BLOCK_D)
    mask = col < D
    bo = tl.arange(0, BUF)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    nfull = seg_s + ((seg_e - seg_s) // BUF) * BUF
    i = seg_s
    if HAS_PERM:
        # perm is read one block ahead so its load overlaps the row load it feeds.
        pnext = tl.load(
            perm_ptr + (seg_s + bo).to(tl.int64), mask=(seg_s + bo) < nfull, other=0
        )
    while i < nfull:
        if HAS_PERM:
            ridx = pnext.to(tl.int64)  # pyrefly: ignore[unbound-name]
            pnext = tl.load(
                perm_ptr + (i + BUF + bo).to(tl.int64),
                mask=(i + BUF + bo) < nfull,
                other=0,
            )
        else:
            ridx = (i + bo).to(tl.int64)
        blk = tl.load(
            g_ptr + ridx[:, None] * D + col[None, :], mask=mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(blk, axis=0)
        i += BUF
    if HAS_PERM:
        # seg_e - nfull < BUF: the whole tail is one masked block.
        r = i + bo
        rm = r < seg_e
        p = tl.load(perm_ptr + r.to(tl.int64), mask=rm, other=0).to(tl.int64)
        tail = tl.load(
            g_ptr + p[:, None] * D + col[None, :],
            mask=rm[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(tail, axis=0)
    else:
        while i < seg_e:
            acc += tl.load(g_ptr + i.to(tl.int64) * D + col, mask=mask, other=0.0).to(
                tl.float32
            )
            i += 1
    oo = out_ptr + tl.cast(row, tl.int64) * D + col
    if nch_row == 1:
        # short row: single writer, no atomic
        tl.store(oo, acc.to(out_ptr.dtype.element_ty), mask=mask)
    elif ROW_DIRECT:
        slot = tl.load(hcum_ptr + row)
        tl.atomic_add(hot_ptr + slot * D + col, acc, mask=mask)
    else:
        tl.atomic_add(oo, acc, mask=mask)  # hot row: few bounded atomics, parallel


@triton.jit
def _src_rows(
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    ridx,
    D,
    G_TY: tl.constexpr,
    NSRC: tl.constexpr,
):
    # Pointers to rows ridx (int64, in [0, N)) of the row-wise concatenation of
    # NSRC [rows, D] splits: split k holds rows o_k .. o_{k+1} - 1 (o_0 = 0 <=
    # o_1 <= ... <= o_NSRC = N), its row r at element r * D from byte address b_k.
    # Row ridx is in the last split k with o_k <= ridx, which is never empty as
    # o_{k+1} > ridx. An empty split k has o_k = o_{k+1} (= N if it is the last),
    # so every lane that selects b_k is overridden by a later split. The bases
    # are selected as integers (TritonAMDGPUCanonicalizePointers in Triton 3.5
    # rejects a select between pointers on a tensor condition). Each b_k is a
    # multiple of 16, so the row pointers have the alignment of a 16-byte
    # aligned tensor's rows, as in _seg_chunk.
    b = tl.where(ridx >= o1, b1, b0)
    if NSRC > 2:
        b = tl.where(ridx >= o2, b2, b)
    if NSRC > 3:
        b = tl.where(ridx >= o3, b3, b)
    if NSRC > 4:
        b = tl.where(ridx >= o4, b4, b)
    if NSRC > 5:
        b = tl.where(ridx >= o5, b5, b)
    if NSRC > 6:
        b = tl.where(ridx >= o6, b6, b)
    if NSRC > 7:
        b = tl.where(ridx >= o7, b7, b)
    rp = tl.cast(
        b.to(tl.int64), tl.pointer_type(G_TY)  # pyrefly: ignore[bad-argument-type]
    )  # pyrefly: ignore[bad-argument-type]
    return tl.multiple_of(rp, 16) + ridx * D


@triton.jit(
    do_not_specialize=[
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "o1",
        "o2",
        "o3",
        "o4",
        "o5",
        "o6",
        "o7",
    ]
)
def _seg_chunk_multisrc(
    perm_ptr,  # sorted row i is grad_out[perm[i]]
    off_ptr,
    nchoff_ptr,
    item_row_ptr,
    out_ptr,
    hcum_ptr,
    hot_ptr,
    num_rows,
    D,
    # b0..b7 (int64): split k of grad_out [N, D] has grad_out row r at element
    # r * D from byte address b_k
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    o1,  # grad_out row where split k starts, for k = 1..NSRC-1
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BUF: tl.constexpr,
    ROW_DIRECT: tl.constexpr,
    ITEM_ROW: tl.constexpr,
    NSRC: tl.constexpr,  # 2 .. 8
    G_TY: tl.constexpr,  # grad_out element type
):
    # _seg_chunk with HAS_PERM on grad_out held as NSRC row splits; the other
    # arguments are those of _seg_chunk. Programs, rows, chunks and the reduction
    # are the same; only the row addresses differ, and the perm prefetch runs to
    # seg_e so the tail rows come from the last prefetch.
    pid = tl.program_id(0)
    if ITEM_ROW:
        row = tl.load(item_row_ptr + pid)
        if row >= num_rows:
            return
    else:
        total = tl.load(nchoff_ptr + num_rows)
        if pid >= total:
            return
        lo = 0
        hi = num_rows
        while hi - lo > 1:
            mid = (lo + hi) // 2
            v = tl.load(nchoff_ptr + mid)
            if v <= pid:
                lo = mid
            else:
                hi = mid
        row = lo
    base = tl.load(nchoff_ptr + row)
    nch_row = tl.load(nchoff_ptr + row + 1) - base
    chunk = pid - base
    rs = tl.load(off_ptr + row)
    re = tl.load(off_ptr + row + 1)
    seg_s = rs + chunk * CHUNK
    seg_e = tl.minimum(seg_s + CHUNK, re)
    col = tl.arange(0, BLOCK_D)
    mask = col < D
    bo = tl.arange(0, BUF)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    nfull = seg_s + ((seg_e - seg_s) // BUF) * BUF
    i = seg_s
    pnext = tl.load(
        perm_ptr + (seg_s + bo).to(tl.int64), mask=(seg_s + bo) < seg_e, other=0
    )
    while i < nfull:
        ridx = pnext.to(tl.int64)
        pnext = tl.load(
            perm_ptr + (i + BUF + bo).to(tl.int64),
            mask=(i + BUF + bo) < seg_e,
            other=0,
        )
        rp = _src_rows(
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            o1,
            o2,
            o3,
            o4,
            o5,
            o6,
            o7,
            ridx,
            D,
            G_TY,
            NSRC,
        )
        blk = tl.load(rp[:, None] + col[None, :], mask=mask[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(blk, axis=0)
        i += BUF
    # seg_e - nfull < BUF: pnext holds perm[i + bo] where i + bo < seg_e, else 0.
    rm = (i + bo) < seg_e
    rp = _src_rows(
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        pnext.to(tl.int64),
        D,
        G_TY,
        NSRC,
    )
    tail = tl.load(
        rp[:, None] + col[None, :], mask=rm[:, None] & mask[None, :], other=0.0
    ).to(tl.float32)
    acc += tl.sum(tail, axis=0)
    oo = out_ptr + tl.cast(row, tl.int64) * D + col
    if nch_row == 1:
        tl.store(oo, acc.to(out_ptr.dtype.element_ty), mask=mask)
    elif ROW_DIRECT:
        slot = tl.load(hcum_ptr + row)
        tl.atomic_add(hot_ptr + slot * D + col, acc, mask=mask)
    else:
        tl.atomic_add(oo, acc, mask=mask)


@triton.jit
def _seg_hot_rows(
    hot_ptr,  # [>= #hot rows, D] fp32 hot-row sums from _seg_chunk ROW_DIRECT
    hcum_ptr,  # [num_rows + 1] exclusive cumsum of hot-row flags
    out_ptr,  # [num_rows, D] grad_out dtype
    num_rows,
    D,
    BLOCK_D: tl.constexpr,
):
    slot = tl.program_id(0)
    if slot >= tl.load(hcum_ptr + num_rows):
        return
    # binary search: the hot row of this slot is the largest r with hcum[r] <= slot
    lo = 0
    hi = num_rows
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = tl.load(hcum_ptr + mid)
        if v <= slot:
            lo = mid
        else:
            hi = mid
    col = tl.arange(0, BLOCK_D)
    mask = col < D
    acc = tl.load(hot_ptr + slot.to(tl.int64) * D + col, mask=mask, other=0.0)
    tl.store(
        out_ptr + tl.cast(lo, tl.int64) * D + col,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _rm_sel(
    ridx,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    v0,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    NSRC: tl.constexpr,
):
    # v_k of the split k holding each row ridx, selected as in _src_rows.
    v = tl.where(ridx >= o1, v1, v0)
    if NSRC > 2:
        v = tl.where(ridx >= o2, v2, v)
    if NSRC > 3:
        v = tl.where(ridx >= o3, v3, v)
    if NSRC > 4:
        v = tl.where(ridx >= o4, v4, v)
    if NSRC > 5:
        v = tl.where(ridx >= o5, v5, v)
    if NSRC > 6:
        v = tl.where(ridx >= o6, v6, v)
    if NSRC > 7:
        v = tl.where(ridx >= o7, v7, v)
    return v


@triton.jit(
    do_not_specialize=[
        "n",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "o1",
        "o2",
        "o3",
        "o4",
        "o5",
        "o6",
        "o7",
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "m7",
    ]
)
def _rm_compose(
    perm_ptr,  # [n] int64 sort permutation: sorted row i is grad_out[perm[i]]
    addr_ptr,  # [n] int64 output: byte address of sorted row i
    n,
    D,
    MS,  # row stride of the row-map splits' sources, in elements
    # b0..b7 (int64): byte address of split k's grad_out row 0 (plain), or of
    # its source (row map)
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    o1,  # grad_out row where split k starts, for k = 1..NSRC-1
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    # m0..m7 (int64): byte address of row-map split k's int64 map, minus
    # 8 * o_k; unused for a plain split
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    BLOCK: tl.constexpr,
    ESZ: tl.constexpr,  # grad_out element size in bytes
    NSRC: tl.constexpr,  # 2 .. 8
    MAPPED: tl.constexpr,  # bit k set: split k is read through its row map
):
    # The byte address of each sorted row i (ONEFLOW_SEG_MULTISRC_ROWMAP), for
    # _seg_chunk_rowaddr: split k holds grad_out rows o_k .. o_{k+1} - 1
    # (o_0 = 0). A plain split k has row r at element r * D from b_k, as in
    # _src_rows; a row-map split k has it at element map[r] * MS from b_k,
    # where map[r] is the int64 at byte address m_k + 8 * r. Every address is
    # a multiple of 16 bytes: so are b_k, D * ESZ and MS * ESZ.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < n
    ridx = tl.load(perm_ptr + i, mask=valid, other=0).to(tl.int64)
    is_map = (
        _rm_sel(
            ridx,
            o1,
            o2,
            o3,
            o4,
            o5,
            o6,
            o7,
            MAPPED & 1,
            (MAPPED >> 1) & 1,
            (MAPPED >> 2) & 1,
            (MAPPED >> 3) & 1,
            (MAPPED >> 4) & 1,
            (MAPPED >> 5) & 1,
            (MAPPED >> 6) & 1,
            (MAPPED >> 7) & 1,
            NSRC,
        )
        != 0
    )
    m = _rm_sel(ridx, o1, o2, o3, o4, o5, o6, o7, m0, m1, m2, m3, m4, m5, m6, m7, NSRC)
    mp = tl.cast(m.to(tl.int64), tl.pointer_type(tl.int64))
    mrow = tl.load(mp + ridx, mask=valid & is_map, other=0)
    b = _rm_sel(
        ridx, o1, o2, o3, o4, o5, o6, o7, b0, b1, b2, b3, b4, b5, b6, b7, NSRC
    ).to(tl.int64)
    addr = b + tl.where(is_map, mrow * MS, ridx * D) * ESZ
    tl.store(addr_ptr + i, addr, mask=valid)


@triton.jit(do_not_specialize=["num_rows"])
def _seg_chunk_rowaddr(
    addr_ptr,  # [N] int64 byte address of each sorted row, from _rm_compose
    off_ptr,
    nchoff_ptr,
    item_row_ptr,
    out_ptr,
    hcum_ptr,
    hot_ptr,
    num_rows,
    D,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BUF: tl.constexpr,
    ROW_DIRECT: tl.constexpr,
    ITEM_ROW: tl.constexpr,
    G_TY: tl.constexpr,  # grad_out element type
):
    # _seg_chunk_multisrc with the address of each sorted row precomputed
    # (ONEFLOW_SEG_MULTISRC_ROWMAP): the address array takes the place of perm,
    # with the same prefetch, so the loop's dependency depth is unchanged.
    # Programs, rows, chunks and the reduction are the same.
    pid = tl.program_id(0)
    if ITEM_ROW:
        row = tl.load(item_row_ptr + pid)
        if row >= num_rows:
            return
    else:
        total = tl.load(nchoff_ptr + num_rows)
        if pid >= total:
            return
        lo = 0
        hi = num_rows
        while hi - lo > 1:
            mid = (lo + hi) // 2
            v = tl.load(nchoff_ptr + mid)
            if v <= pid:
                lo = mid
            else:
                hi = mid
        row = lo
    base = tl.load(nchoff_ptr + row)
    nch_row = tl.load(nchoff_ptr + row + 1) - base
    chunk = pid - base
    rs = tl.load(off_ptr + row)
    re = tl.load(off_ptr + row + 1)
    seg_s = rs + chunk * CHUNK
    seg_e = tl.minimum(seg_s + CHUNK, re)
    col = tl.arange(0, BLOCK_D)
    mask = col < D
    bo = tl.arange(0, BUF)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    nfull = seg_s + ((seg_e - seg_s) // BUF) * BUF
    i = seg_s
    anext = tl.load(
        addr_ptr + (seg_s + bo).to(tl.int64), mask=(seg_s + bo) < seg_e, other=0
    )
    while i < nfull:
        a = anext.to(tl.int64)
        anext = tl.load(
            addr_ptr + (i + BUF + bo).to(tl.int64),
            mask=(i + BUF + bo) < seg_e,
            other=0,
        )
        rp = tl.multiple_of(
            tl.cast(a, tl.pointer_type(G_TY)), 16  # pyrefly: ignore[bad-argument-type]
        )  # pyrefly: ignore[bad-argument-type]
        blk = tl.load(rp[:, None] + col[None, :], mask=mask[None, :], other=0.0).to(
            tl.float32
        )
        acc += tl.sum(blk, axis=0)
        i += BUF
    # seg_e - nfull < BUF: anext holds addr[i + bo] where i + bo < seg_e, else 0.
    rm = (i + bo) < seg_e
    rp = tl.multiple_of(
        tl.cast(
            anext.to(tl.int64),
            tl.pointer_type(G_TY),  # pyrefly: ignore[bad-argument-type]
        ),
        16,
    )
    tail = tl.load(
        rp[:, None] + col[None, :], mask=rm[:, None] & mask[None, :], other=0.0
    ).to(tl.float32)
    acc += tl.sum(tail, axis=0)
    oo = out_ptr + tl.cast(row, tl.int64) * D + col
    if nch_row == 1:
        tl.store(oo, acc.to(out_ptr.dtype.element_ty), mask=mask)
    elif ROW_DIRECT:
        slot = tl.load(hcum_ptr + row)
        tl.atomic_add(hot_ptr + slot * D + col, acc, mask=mask)
    else:
        tl.atomic_add(oo, acc, mask=mask)


def _fused_perm_enabled() -> bool:
    return (
        not _fused_perm_disabled
        and os.environ.get("ONEFLOW_SEG_REDUCE_FUSED_PERM", "1") == "1"
    )


def _fused_perm_check_due(out: torch.Tensor) -> bool:
    return (
        _fused_perm_checks < _FUSED_PERM_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_REDUCE_FUSED_PERM_CHECK", "1") == "1"
        and out.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def _fused_perm_self_check(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Compare the fused-perm fp32 result with the legacy gathered-path result.

    Syncs once (tolist); only called on the first _FUSED_PERM_CHECK_CALLS calls.
    On mismatch, disables the fused path for the process and returns ``ref``.
    """
    global _fused_perm_checks, _fused_perm_disabled
    _fused_perm_checks += 1
    inf = float("inf")
    diff, ref_max = torch.stack(
        [torch.linalg.vector_norm(out - ref, inf), torch.linalg.vector_norm(ref, inf)]
    ).tolist()
    logger.warning(
        "seg_reduce fused-perm self-check %d/%d: out=%s max|fused - legacy|=%.3e "
        "max|legacy|=%.3e",
        _fused_perm_checks,
        _FUSED_PERM_CHECK_CALLS,
        tuple(out.shape),
        diff,
        ref_max,
    )
    if not diff <= 1e-4 * max(ref_max, 1e-6):
        logger.error(
            "seg_reduce fused-perm self-check FAILED; disabling "
            "ONEFLOW_SEG_REDUCE_FUSED_PERM for the rest of the process"
        )
        _fused_perm_disabled = True
        return ref
    return out


def _item_row_enabled() -> bool:
    return (
        not _item_row_disabled
        and os.environ.get("ONEFLOW_SEG_REDUCE_ITEM_ROW", "1") == "1"
    )


def _item_row_check_due(out: torch.Tensor) -> bool:
    return (
        _item_row_checks < _ITEM_ROW_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_REDUCE_ITEM_ROW_CHECK", "1") == "1"
        and out.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def _item_row_self_check(
    item_row: torch.Tensor, nchoff: torch.Tensor, pids: torch.Tensor
) -> Optional[torch.Tensor]:
    """Check that item_row[pid] is the row r with nchoff[r] <= pid < nchoff[r + 1]
    for every pid < nchoff[-1], and at least num_rows (exit) for every later pid.

    Syncs once (tolist); only called on the first _ITEM_ROW_CHECK_CALLS launches.
    On mismatch, disables the item->row map for the process and returns None.
    """
    global _item_row_checks, _item_row_disabled
    _item_row_checks += 1
    num_rows = nchoff.numel() - 1
    r = item_row.long()
    rc = r.clamp(0, num_rows - 1)
    in_row = (r == rc) & (nchoff[rc] <= pids) & (pids < nchoff[rc + 1])
    ok = torch.where(pids < nchoff[-1], in_row, r >= num_rows)
    n_bad, total = torch.stack([(~ok).sum(), nchoff[-1]]).tolist()
    logger.warning(
        "seg_reduce item-row self-check %d/%d: rows=%d items=%d grid=%d bad=%d",
        _item_row_checks,
        _ITEM_ROW_CHECK_CALLS,
        num_rows,
        total,
        pids.numel(),
        n_bad,
    )
    if n_bad != 0:
        logger.error(
            "seg_reduce item-row self-check FAILED; disabling "
            "ONEFLOW_SEG_REDUCE_ITEM_ROW for the rest of the process"
        )
        _item_row_disabled = True
        return None
    return item_row


def _item_row_map(
    nchoff: torch.Tensor, out: torch.Tensor, g_max: int
) -> Optional[torch.Tensor]:
    """[g_max] int32 row of each _seg_chunk program, or None to binary-search.

    nchoff[0] == 0 <= pid, so the count of nchoff[1:] entries <= pid is the
    largest r with nchoff[r] <= pid, the row the in-kernel search returns for
    every pid < nchoff[-1]; for every later pid it is num_rows, the exit value.
    """
    if not _item_row_enabled():
        return None
    pids = torch.arange(g_max, device=nchoff.device, dtype=nchoff.dtype)
    item_row = torch.searchsorted(nchoff[1:], pids, right=True, out_int32=True)
    if _item_row_check_due(out):
        return _item_row_self_check(item_row, nchoff, pids)
    return item_row


class _SplitGrads:
    """The [N, D] grad_out of an index_select whose output was split along dim 0:
    rows offs[k] .. offs[k] + parts[k].shape[0] - 1 of grad_out are parts[k].
    srcs[k] is parts[k] as a contiguous tensor, or for an empty part a non-empty
    one, which _src_rows never reads. bases[k] is the byte address grad_out row 0
    would have in srcs[k], so srcs[k] holds grad_out row r at
    bases[k] + r * row_bytes. The kernel reads the srcs through these integer
    addresses, so they are kept here until the backward returns."""

    def __init__(
        self, parts: List[torch.Tensor], srcs: List[torch.Tensor], offs: List[int]
    ) -> None:
        self.parts = parts
        self.srcs = srcs
        self.offs = offs
        self.shape: Tuple[int, int] = (
            offs[-1] + parts[-1].shape[0],
            srcs[0].shape[1],
        )
        self.dtype: torch.dtype = srcs[0].dtype
        self.device: torch.device = srcs[0].device
        self.row_bytes: int = self.shape[1] * srcs[0].element_size()
        self.bases: List[int] = [
            s.data_ptr() - o * self.row_bytes for s, o in zip(srcs, offs)
        ]

    def contiguous(self) -> "_SplitGrads":
        return self

    def cat(self) -> torch.Tensor:
        return torch.cat(self.parts)


class _RowMapSplitGrads(_SplitGrads):
    """A _SplitGrads (see ONEFLOW_SEG_MULTISRC_ROWMAP) where the parts k with bit
    k of ``mapped`` set are row-map gradients: grad_out row offs[k] + j is row
    rows[j] of ``sources[k]``, whose rows are ``map_stride`` elements apart. As
    _rm_compose reads them, bases[k] is that source's address
    and maps[k] the address of rows minus 8 * offs[k]; the other parts are as
    in _SplitGrads (maps[k] = 0). The tensors behind the addresses are kept
    until the backward returns."""

    def __init__(
        self,
        parts: List[torch.Tensor],
        srcs: List[torch.Tensor],
        offs: List[int],
        sources: List[Optional[torch.Tensor]],
        rows: List[Optional[torch.Tensor]],
        map_stride: int,
    ) -> None:
        super().__init__(parts, srcs, offs)
        self.sources = sources
        self.rows = rows
        self.map_stride = map_stride
        self.mapped = 0
        self.maps = [0] * len(parts)
        for k, (source, r) in enumerate(zip(sources, rows)):
            if source is not None and r is not None:
                self.mapped |= 1 << k
                self.bases[k] = source.data_ptr()
                self.maps[k] = r.data_ptr() - 8 * offs[k]


def _dense(g):
    """grad_out as one [N, D] tensor (a _SplitGrads is concatenated)."""
    return g.cat() if isinstance(g, _SplitGrads) else g


def _launch_seg_chunk(
    g: Union[torch.Tensor, _SplitGrads],
    perm: torch.Tensor,
    off: torch.Tensor,
    nchoff: torch.Tensor,
    out: torch.Tensor,
    g_max: int,
    has_perm: bool,
    hcum: Optional[torch.Tensor] = None,
    hot: Optional[torch.Tensor] = None,
    item_row: Optional[torch.Tensor] = None,
) -> None:
    num_rows, D = out.shape
    row_direct = hot is not None
    if item_row is None:
        item_row = _item_row_map(nchoff, out, g_max)
    if isinstance(g, _RowMapSplitGrads):
        nsrc = len(g.bases)
        assert has_perm and 1 < nsrc <= _MAX_SRC
        pad = _MAX_SRC - nsrc
        n = perm.numel()
        addr = torch.empty_like(perm)
        if n > 0:
            cast(Any, _rm_compose)[(triton.cdiv(n, _COMPOSE_BLOCK),)](
                perm,
                addr,
                n,
                D,
                g.map_stride,
                *g.bases,
                *([g.bases[0]] * pad),
                *g.offs[1:],
                *([0] * pad),
                *g.maps,
                *([0] * pad),
                BLOCK=_COMPOSE_BLOCK,
                ESZ=g.row_bytes // D,
                NSRC=nsrc,
                MAPPED=g.mapped,
                num_warps=4,
            )
        cast(Any, _seg_chunk_rowaddr)[(g_max,)](
            addr,
            off,
            nchoff,
            nchoff if item_row is None else item_row,
            out,
            hcum if row_direct else nchoff,
            hot if row_direct else out,
            num_rows,
            D,
            CHUNK=_CHUNK,
            BLOCK_D=triton.next_power_of_2(D),
            BUF=_BUF,
            ROW_DIRECT=row_direct,
            ITEM_ROW=item_row is not None,
            G_TY=_TL_DTYPES[g.dtype],
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        return
    if isinstance(g, _SplitGrads):
        nsrc = len(g.bases)
        assert has_perm and 1 < nsrc <= _MAX_SRC
        cast(Any, _seg_chunk_multisrc)[(g_max,)](
            perm,
            off,
            nchoff,
            nchoff if item_row is None else item_row,
            out,
            hcum if row_direct else nchoff,
            hot if row_direct else out,
            num_rows,
            D,
            *g.bases,
            *([g.bases[0]] * (_MAX_SRC - nsrc)),
            *g.offs[1:],
            *([0] * (_MAX_SRC - nsrc)),
            CHUNK=_CHUNK,
            BLOCK_D=triton.next_power_of_2(D),
            BUF=_BUF,
            ROW_DIRECT=row_direct,
            ITEM_ROW=item_row is not None,
            NSRC=nsrc,
            G_TY=_TL_DTYPES[g.dtype],
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        return
    cast(Any, _seg_chunk)[(g_max,)](
        g,
        perm,
        off,
        nchoff,
        nchoff if item_row is None else item_row,
        out,
        hcum if row_direct else nchoff,
        hot if row_direct else out,
        num_rows,
        D,
        CHUNK=_CHUNK,
        BLOCK_D=triton.next_power_of_2(D),
        BUF=_BUF,
        HAS_PERM=has_perm,
        ROW_DIRECT=row_direct,
        ITEM_ROW=item_row is not None,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )


def _row_direct_enabled() -> bool:
    return (
        not _row_direct_disabled
        and os.environ.get("ONEFLOW_SEG_REDUCE_ROW_DIRECT", "1") == "1"
    )


def _row_direct_check_due(out: torch.Tensor) -> bool:
    return (
        _row_direct_checks < _ROW_DIRECT_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_REDUCE_ROW_DIRECT_CHECK", "0") == "1"
        and out.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def _row_direct_self_check(
    out: torch.Tensor, ref: torch.Tensor, hot_rows: torch.Tensor
) -> torch.Tensor:
    """Compare the row-direct result with the fp32-buffer result ``ref``.

    A single-chunk row rounds the same fp32 sum, so it must equal ``ref`` cast to
    out.dtype exactly (NaN matching NaN). A hot row's atomic fp32 sum may run in
    another order, so it may also be off by one ulp of out.dtype plus 1e-4 of its
    row's max |ref|. Compared in row blocks to bound the extra memory; syncs once
    (tolist); only called on the first _ROW_DIRECT_CHECK_CALLS calls. On
    mismatch, disables the row-direct path for the process and returns ``ref``
    cast to out.dtype.
    """
    global _row_direct_checks, _row_direct_disabled
    _row_direct_checks += 1
    eps = torch.finfo(out.dtype).eps
    ref_max = torch.linalg.vector_norm(ref, float("inf"))
    n_bad = ref.new_zeros(())
    max_err = ref.new_zeros(())
    for s in range(0, ref.shape[0], _ROW_DIRECT_CHECK_ROWS):
        o = out[s : s + _ROW_DIRECT_CHECK_ROWS]
        r = ref[s : s + _ROW_DIRECT_CHECK_ROWS]
        ok = (o == r.to(o.dtype)) | (o.isnan() & r.isnan())
        err = (o.float() - r).abs_()
        r_abs = r.abs()
        row_atol = r_abs.amax(1, keepdim=True).mul_(1e-4)
        hot = hot_rows[s : s + _ROW_DIRECT_CHECK_ROWS, None]
        ok |= hot & (err <= r_abs.mul_(eps).add_(row_atol))
        n_bad += (~ok).sum()
        max_err = torch.maximum(max_err, err.max())
    n_bad_v, max_err_v, ref_max_v, n_hot_v = torch.stack(
        [n_bad, max_err, ref_max, hot_rows.sum().to(ref.dtype)]
    ).tolist()
    logger.warning(
        "seg_reduce row-direct self-check %d/%d: out=%s %s hot_rows=%d bad=%d "
        "max|row_direct - legacy|=%.3e max|legacy|=%.3e",
        _row_direct_checks,
        _ROW_DIRECT_CHECK_CALLS,
        tuple(out.shape),
        out.dtype,
        n_hot_v,
        n_bad_v,
        max_err_v,
        ref_max_v,
    )
    if n_bad_v != 0:
        logger.error(
            "seg_reduce row-direct self-check FAILED; disabling "
            "ONEFLOW_SEG_REDUCE_ROW_DIRECT for the rest of the process"
        )
        _row_direct_disabled = True
        return ref.to(out.dtype)
    return out


def _seg_reduce_row_direct(
    g: Union[torch.Tensor, _SplitGrads],
    perm: torch.Tensor,
    off: torch.Tensor,
    nch: torch.Tensor,
    out_dtype: torch.dtype,
    has_perm: bool,
) -> torch.Tensor:
    """Row-direct _seg_chunk: each output row is written once, in out_dtype.

    ``nch`` is the per-row chunk count of the fp32-buffer path. A hot row has
    more than _CHUNK entries, so there are at most N // (_CHUNK + 1) of them,
    a host-known bound that sizes the fp32 hot-row buffer without a sync.
    """
    num_rows = nch.numel()
    N = perm.numel()
    D = g.shape[1]
    dev = g.device
    nch_rd = nch.clamp(min=1)
    hot_rows = nch_rd > 1
    hcum = torch.cat([nch.new_zeros(1), hot_rows.cumsum(0)])
    nchoff = torch.cat([nch.new_zeros(1), nch_rd.cumsum(0)])
    max_hot = min(num_rows, N // (_CHUNK + 1))
    hot = torch.zeros(max(max_hot, 1), D, dtype=torch.float32, device=dev)
    out = torch.empty(num_rows, D, dtype=out_dtype, device=dev)
    g_max = num_rows + N // _CHUNK + 1  # bounds sum(max(nch, 1)) as well
    _launch_seg_chunk(g, perm, off, nchoff, out, g_max, has_perm, hcum, hot)
    if max_hot > 0:
        cast(Any, _seg_hot_rows)[(max_hot,)](
            hot,
            hcum,
            out,
            num_rows,
            D,
            BLOCK_D=triton.next_power_of_2(D),
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    if _row_direct_check_due(out):
        ref = torch.zeros(num_rows, D, dtype=torch.float32, device=dev)
        nchoff_ref = torch.cat([nch.new_zeros(1), nch.cumsum(0)])
        _launch_seg_chunk(g, perm, off, nchoff_ref, ref, g_max, has_perm)
        out = _row_direct_self_check(out, ref, hot_rows)
    return out


def _narrow_keys(idx: torch.Tensor, num_rows: int) -> torch.Tensor:
    if (
        _narrow_sort_disabled
        or os.environ.get("ONEFLOW_SEG_REDUCE_NARROW_SORT", "1") != "1"
    ):
        return idx
    if num_rows < 32767:
        return idx.to(torch.int16)
    if num_rows < 2**31 - 1:
        return idx.to(torch.int32)
    return idx


def _sort_indices(
    idx: torch.Tensor, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.sort(idx) on keys narrowed by _narrow_keys; perm is always int64.

    The self-check syncs once (torch.equal) on each of the first
    _NARROW_SORT_CHECK_CALLS narrowed calls only.
    """
    global _narrow_checks, _narrow_sort_disabled
    keys = _narrow_keys(idx, num_rows)
    sorted_idx, perm = torch.sort(keys)
    if (
        keys.dtype != idx.dtype
        and _narrow_checks < _NARROW_SORT_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_REDUCE_NARROW_SORT_CHECK", "1") == "1"
        and not torch.cuda.is_current_stream_capturing()
    ):
        _narrow_checks += 1
        ref_sorted, ref_perm = torch.sort(idx)
        ok = torch.equal(perm, ref_perm)
        logger.warning(
            "seg_reduce narrow-sort self-check %d/%d: n=%d num_rows=%d keys=%s ok=%s",
            _narrow_checks,
            _NARROW_SORT_CHECK_CALLS,
            idx.numel(),
            num_rows,
            keys.dtype,
            ok,
        )
        if not ok:
            logger.error(
                "seg_reduce narrow-sort self-check FAILED; disabling "
                "ONEFLOW_SEG_REDUCE_NARROW_SORT for the rest of the process"
            )
            _narrow_sort_disabled = True
            return ref_sorted, ref_perm
    return sorted_idx, perm


def _native_index_add_preferred(n: int, num_rows: int, dim: int) -> bool:
    """Low duplication (dest is at least half the index count) and both operands
    within 32-bit index math."""
    return num_rows * 2 >= n and max(n, num_rows) * dim < _INDEX_ADD_32BIT_NUMEL


def _sorted_segment_reduce_backward(grad_out, reverse_indices, num_rows, plan=None):
    """grad wrt embeddings for out = embeddings.index_select(0, reverse_indices).
    Fully on-device: no host scalar reads, no unique_consecutive/bincount/.item().
    grad_out is a tensor or a _SplitGrads; only the fused-perm chunk path reads a
    _SplitGrads in place, the other paths concatenate it. ``plan``, from
    build_sorted_plan, holds the index-only part (see ONEFLOW_SEG_PRE_PLAN); it is
    ignored if it does not fit."""
    grad_out = grad_out.contiguous()
    _, D = grad_out.shape
    dev = grad_out.device
    if plan is not None:
        out = _segment_reduce_with_plan(grad_out, reverse_indices, num_rows, plan)
        if out is not None:
            return out

    # Duplication-aware route. The chunk path only pays off when many gradient rows
    # collide into few destination rows (hot-row atomic contention). For the concat
    # backward the two operands are very different: the LEFT (small values_left) is
    # ~1.17M indices into ~1.9K rows (dup ~600x, sentinel hot) = 18ms native -> chunk
    # wins; the RIGHT (large values_right) is ~1.17M into ~1.17M rows (dup ~1) = 3ms
    # native, where the sort overhead makes the chunk path a LOSS. So when dup is low
    # (dest is at least half the index count) fall straight back to native index_add.
    N = reverse_indices.numel()
    if _native_index_add_preferred(N, num_rows, D):
        if isinstance(grad_out, _RowMapSplitGrads):
            _rowmap_log(
                grad_out.parts, "low duplication: index_add of the materialized splits"
            )
        out = torch.zeros(num_rows, D, dtype=grad_out.dtype, device=dev)
        out.index_add_(0, reverse_indices, _dense(grad_out))
        return out

    sorted_idx, perm = _sort_indices(reverse_indices, num_rows)
    fused_perm = _fused_perm_enabled()
    g = grad_out if fused_perm else _dense(grad_out).index_select(0, perm).contiguous()
    # Offsets straight from the SORTED indices: off[r] = #(idx < r). This is
    # atomic-free. The earlier counts.scatter_add_(0, sorted_idx, 1) version was a
    # HOT-ROW ATOMIC -- the sentinel/padding row appears in ~half the indices, so
    # ~N/2 atomicAdds land on counts[sentinel]. Atomic contention is latency-bound,
    # so the 1-wide long payload does not help: it was just as serialized as the
    # 128-wide index_add this kernel replaces (showed up slow in the MI350 trace).
    # searchsorted exploits the sortedness to avoid all atomics, and is still
    # sync-free (output size num_rows+1 is host-known, no data-dependent read-back).
    off = torch.searchsorted(
        sorted_idx, torch.arange(num_rows + 1, device=dev, dtype=sorted_idx.dtype)
    ).to(torch.int64)
    counts = off[1:] - off[:-1]
    nch = (counts + _CHUNK - 1) // _CHUNK  # chunks per row
    if grad_out.dtype != torch.float32 and _row_direct_enabled():
        return _seg_reduce_row_direct(g, perm, off, nch, grad_out.dtype, fused_perm)
    nchoff = torch.cat([nch.new_zeros(1), nch.cumsum(0)])  # nchoff[-1] = total items

    out = torch.zeros(num_rows, D, dtype=torch.float32, device=dev)
    # Static host-known upper bound on total work items: sum(ceil(c_i/CHUNK))
    # <= num_rows + N/CHUNK. Excess programs early-exit in the kernel.
    g_max = num_rows + reverse_indices.numel() // _CHUNK + 1
    _launch_seg_chunk(g, perm, off, nchoff, out, g_max, fused_perm)
    if fused_perm and _fused_perm_check_due(out):
        ref = torch.zeros_like(out)
        g_ref = _dense(grad_out).index_select(0, perm).contiguous()
        _launch_seg_chunk(g_ref, perm, off, nchoff, ref, g_max, False)
        del g_ref
        out = _fused_perm_self_check(out, ref)
    return out.to(grad_out.dtype)


class _SortedIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, reverse_indices):
        ctx.save_for_backward(reverse_indices)
        ctx.num_rows = embeddings.shape[0]
        return embeddings.index_select(0, reverse_indices)

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, grad_out):
        (reverse_indices,) = ctx.saved_tensors
        grad_emb = _sorted_segment_reduce_backward(
            grad_out, reverse_indices, ctx.num_rows, plan=_take_seg_plan(ctx)
        )
        return grad_emb, None


def sorted_index_select(embeddings, reverse_indices, seg_plan=None):
    """Drop-in for ``embeddings.index_select(0, reverse_indices)`` with a fast,
    fp32, sync-free sorted (chunk-split) backward. Training only. ``seg_plan``:
    see build_sorted_plan."""
    out = _SortedIndexSelect.apply(embeddings, reverse_indices)
    if seg_plan is not None:
        _attach_seg_plan(out, _SortedIndexSelect, seg_plan)
    return out


def _multisrc_enabled() -> bool:
    return (
        not _multisrc_disabled
        and os.environ.get("ONEFLOW_SEG_REDUCE_MULTISRC", "1") == "1"
    )


def _multisrc_check_due(out: torch.Tensor) -> bool:
    return (
        _multisrc_checks < _MULTISRC_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_REDUCE_MULTISRC_CHECK", "1") == "1"
        and out.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def _multisrc_self_check(
    out: torch.Tensor, ref: torch.Tensor, reverse_indices: torch.Tensor
) -> torch.Tensor:
    """Compare the multi-source result with ``ref``, the same backward run on the
    concatenated gradients.

    A row with at most _CHUNK entries reduces the same values in the same order,
    so it must match bit for bit (NaN matching NaN). A hot row adds its chunk
    sums with fp32 atomics whose order varies from run to run, so it may also be
    off by one ulp of out.dtype plus 1e-4 of its row's max |ref|. Compared in row
    blocks to bound the extra memory; syncs once (tolist); only called on the
    first _MULTISRC_CHECK_CALLS multi-source backwards. On mismatch, disables the
    multi-source path for the process and returns ``ref``.
    """
    global _multisrc_checks, _multisrc_disabled
    _multisrc_checks += 1
    num_rows = out.shape[0]
    counts = torch.zeros(num_rows, dtype=torch.int64, device=out.device)
    counts.index_add_(
        0, reverse_indices, torch.ones_like(reverse_indices, dtype=torch.int64)
    )
    hot_rows = counts > _CHUNK
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[out.element_size()]
    eps = torch.finfo(out.dtype).eps
    n_bad = counts.new_zeros(())
    for s in range(0, num_rows, _MULTISRC_CHECK_ROWS):
        o = out[s : s + _MULTISRC_CHECK_ROWS]
        r = ref[s : s + _MULTISRC_CHECK_ROWS]
        ok = (o.view(bits) == r.view(bits)) | (o.isnan() & r.isnan())
        r_abs = r.float().abs()
        err = (o.float() - r.float()).abs_()
        row_atol = r_abs.amax(1, keepdim=True).mul_(1e-4)
        hot = hot_rows[s : s + _MULTISRC_CHECK_ROWS, None]
        ok |= hot & (err <= r_abs.mul_(eps).add_(row_atol))
        n_bad += (~ok).sum()
    n_bad_v, n_hot_v = torch.stack([n_bad, hot_rows.sum()]).tolist()
    logger.warning(
        "seg_reduce multi-source self-check %d/%d: out=%s %s hot_rows=%d bad=%d",
        _multisrc_checks,
        _MULTISRC_CHECK_CALLS,
        tuple(out.shape),
        out.dtype,
        n_hot_v,
        n_bad_v,
    )
    if n_bad_v != 0:
        logger.error(
            "seg_reduce multi-source self-check FAILED; disabling "
            "ONEFLOW_SEG_REDUCE_MULTISRC for the rest of the process"
        )
        _multisrc_disabled = True
        return ref
    return out


def _split_sources(
    parts: List[torch.Tensor], D: int
) -> Optional[Union[torch.Tensor, _SplitGrads]]:
    """grad_out for the multi-source chunk path given its per-split gradients
    ``parts`` (each [n_k, D]): the only part, or a _SplitGrads with one source per
    part, so the kernel variant depends only on the number of splits and the
    dtype (and on whether a base fits in int32, which a device address does
    not). None (the caller concatenates) when the path is off, for more than
    _MAX_SRC parts or a dtype not in _TL_DTYPES, when a part is not a plain
    strided tensor (a ZeroTensor or a tensor subclass, which torch.cat
    materializes or unwraps) or its layout differs from a fresh torch.cat output
    (a neg/conj view, a non-empty part not 16-byte aligned, rows not a multiple
    of 16 bytes). The kernel then reads 16-byte aligned rows with the load
    layout of the concatenation. Host-only checks, no sync."""
    if (
        len(parts) > _MAX_SRC
        or not _multisrc_enabled()
        or not _fused_perm_enabled()
        or parts[0].dtype not in _TL_DTYPES
        or (D * parts[0].element_size()) % 16 != 0
        or any(
            type(p) is not torch.Tensor
            or p._is_zerotensor()
            or p.is_neg()
            or p.is_conj()
            or p.device != parts[0].device
            for p in parts
        )
    ):
        return None
    srcs = [p.contiguous() if p.shape[0] > 0 else p for p in parts]
    nonempty = [s for s in srcs if s.shape[0] > 0]
    if not nonempty or any(s.data_ptr() % 16 != 0 for s in nonempty):
        return None
    if len(srcs) == 1:
        return srcs[0]
    offs = [0]
    for p in parts[:-1]:
        offs.append(offs[-1] + p.shape[0])
    srcs = [s if s.shape[0] > 0 else nonempty[0] for s in srcs]
    return _SplitGrads(parts, srcs, offs)


# ONEFLOW_SEG_MULTISRC_ROWMAP (read per call by hammer.ops.jagged): the backward
# of hammer's _Concat2DJaggedIntoFunction may hand a split's gradient to the
# sorted_index_select_split backward as a row-map gradient, a torch.Tensor
# subclass with a true ``_oneflow_row_map_grad`` class attribute: row j is row
# rows[j] of ``source``, with (source, rows) = row_map_source(), and every op on
# it yields the values of its materialize(). With the multi-source path on, the
# backward composes the byte address of each sorted row (_rm_compose, one pass
# over the sorted entries) and reduces through those addresses
# (_seg_chunk_rowaddr); the rows, their order and the reduction are those of
# the multi-source backward on the materialized gradients. Otherwise, for a
# layout this path does not read, and after a kernel error (which turns the
# path off), the row-map gradients are materialized first. Each producer flag
# window (row_map_window) logs once (WARNING) whether row maps were read, and
# why not.
# ONEFLOW_SEG_MULTISRC_ROWMAP_CHECK=1 also runs the first few row-map backwards
# on the materialized gradients and compares as the multi-source self-check
# does; on a mismatch it returns that result and materializes for good. It also
# adds the device memory in use to the window logs.
_ROWMAP_CHECK_CALLS = 3
_COMPOSE_BLOCK = 1024
_rowmap_checks = 0
_rowmap_disabled = False
_ROWMAP_LOG_KEEP = 64
_rowmap_logged: List[Tuple[int, str]] = []


def row_map_grads_accepted() -> bool:
    """Whether the sorted_index_select_split backward reads row-map gradients
    through their maps now (ONEFLOW_SEG_MULTISRC_ROWMAP): the multi-source and
    fused-perm paths are on and the row-map path was not turned off. Producers
    check it so that they do not hand over gradients that would be
    materialized."""
    return not _rowmap_disabled and _multisrc_enabled() and _fused_perm_enabled()


def _rowmap_log(parts: List[torch.Tensor], message: str) -> None:
    """Logs ``message`` at WARNING once per producer flag window of the
    row-map ``parts``."""
    window = max(
        (getattr(p, "row_map_window", 0) for p in parts if _is_row_map_grad(p)),
        default=0,
    )
    if (window, message) in _rowmap_logged:
        return
    _rowmap_logged.append((window, message))
    del _rowmap_logged[:-_ROWMAP_LOG_KEEP]
    memory = ""
    dev = parts[0].device
    if (
        os.environ.get("ONEFLOW_SEG_MULTISRC_ROWMAP_CHECK", "1") == "1"
        and dev.type == "cuda"
    ):
        memory = "; device memory allocated %.2f GiB, peak %.2f GiB" % (
            torch.cuda.memory_allocated(dev) / 2**30,
            torch.cuda.max_memory_allocated(dev) / 2**30,
        )
    logger.warning(
        "ONEFLOW_SEG_MULTISRC_ROWMAP (window %d): %s%s", window, message, memory
    )


def _is_row_map_grad(p) -> bool:
    return getattr(type(p), "_oneflow_row_map_grad", False) is True


def _plain_part(p: torch.Tensor, dev: torch.device) -> bool:
    return (
        type(p) is torch.Tensor
        and not p._is_zerotensor()
        and not p.is_neg()
        and not p.is_conj()
        and p.device == dev
    )


def _row_map_sources(
    parts: List[torch.Tensor], D: int
) -> Union[_RowMapSplitGrads, str]:
    """grad_out for _rm_compose given per-split gradients of
    which some are row-map gradients, or why not (the caller then materializes
    them):
    when the path is off, for one or more than _MAX_SRC parts, a dtype not in
    _TL_DTYPES, a width D or a source row stride that is not a multiple of 16
    elements (so that the rows' alignment, which picks the kernel's load and
    reduction layout, is that of _seg_chunk_multisrc's rows), a plain part
    _split_sources would refuse, an empty row-map part, a source that is not a
    plain 2-D tensor of the part's dtype with unit column stride and a 16-byte
    aligned address, or sources with different row strides. Host-only checks,
    no sync."""
    if _rowmap_disabled:
        return "turned off for the rest of the process (self-check or kernel failure)"
    if not _multisrc_enabled() or not _fused_perm_enabled():
        return "the multi-source or fused-perm path is off"
    if not 1 < len(parts) <= _MAX_SRC:
        return f"{len(parts)} splits"
    if parts[0].dtype not in _TL_DTYPES or D % 16 != 0:
        return f"{parts[0].dtype} rows of width {D}"
    dev = parts[0].device
    srcs: List[torch.Tensor] = []
    sources: List[Optional[torch.Tensor]] = []
    rows_kept: List[Optional[torch.Tensor]] = []
    strides: List[int] = []
    for p in parts:
        if _is_row_map_grad(p):
            source, rows = cast(Any, p).row_map_source()
            if (
                p.shape[0] == 0
                or not _plain_part(source, dev)
                or source.dim() != 2
                or source.dtype != p.dtype
                or source.shape[1] != D
                or source.stride(1) != 1
                or source.stride(0) % 16 != 0
                or source.data_ptr() % 16 != 0
                or type(rows) is not torch.Tensor
                or rows.dtype != torch.int64
                or rows.shape != (p.shape[0],)
                or not rows.is_contiguous()
                or rows.device != dev
            ):
                return (
                    f"row-map source {tuple(source.shape)}/{source.stride()}/"
                    f"{source.dtype} or rows {rows.dtype}{tuple(rows.shape)} "
                    "do not fit"
                )
            # _SplitGrads takes a source's width, dtype and (replaced) address.
            srcs.append(source)
            sources.append(source)
            rows_kept.append(rows)
            strides.append(source.stride(0))
            continue
        if not _plain_part(p, dev):
            return f"a split is a {type(p).__name__} or a view the kernels do not read"
        s = p.contiguous() if p.shape[0] > 0 else p
        if s.shape[0] > 0 and s.data_ptr() % 16 != 0:
            return "a split is not 16-byte aligned"
        srcs.append(s)
        sources.append(None)
        rows_kept.append(None)
    if len(set(strides)) != 1:
        return f"row-map sources with row strides {sorted(set(strides))}"
    nonempty = [s for s, p in zip(srcs, parts) if p.shape[0] > 0]
    srcs = [s if p.shape[0] > 0 else nonempty[0] for s, p in zip(srcs, parts)]
    offs = [0]
    for p in parts[:-1]:
        offs.append(offs[-1] + p.shape[0])
    return _RowMapSplitGrads(parts, srcs, offs, sources, rows_kept, strides[0])


def _rowmap_check_due(out: torch.Tensor) -> bool:
    return (
        _rowmap_checks < _ROWMAP_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_MULTISRC_ROWMAP_CHECK", "1") == "1"
        and out.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def _rowmap_self_check(
    out: torch.Tensor,
    ref: torch.Tensor,
    reverse_indices: torch.Tensor,
    parts: List[torch.Tensor],
) -> torch.Tensor:
    """_multisrc_self_check for the row-map backward: rows with at most _CHUNK
    entries must match ``ref`` (the backward on the materialized gradients) bit
    for bit, NaN matching NaN; a hot row, whose fp32 atomic chunk sums run in a
    varying order, may also be off by one ulp of out.dtype plus 1e-4 of its
    row's max |ref|. Syncs once; on mismatch, turns the row-map path off here
    and in the producer of each row-map part and returns ``ref``."""
    global _rowmap_checks, _rowmap_disabled
    _rowmap_checks += 1
    num_rows = out.shape[0]
    counts = torch.zeros(num_rows, dtype=torch.int64, device=out.device)
    counts.index_add_(
        0, reverse_indices, torch.ones_like(reverse_indices, dtype=torch.int64)
    )
    hot_rows = counts > _CHUNK
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[out.element_size()]
    eps = torch.finfo(out.dtype).eps
    n_bad = counts.new_zeros(())
    for s in range(0, num_rows, _MULTISRC_CHECK_ROWS):
        o = out[s : s + _MULTISRC_CHECK_ROWS]
        r = ref[s : s + _MULTISRC_CHECK_ROWS]
        ok = (o.view(bits) == r.view(bits)) | (o.isnan() & r.isnan())
        r_abs = r.float().abs()
        err = (o.float() - r.float()).abs_()
        row_atol = r_abs.amax(1, keepdim=True).mul_(1e-4)
        hot = hot_rows[s : s + _MULTISRC_CHECK_ROWS, None]
        ok |= hot & (err <= r_abs.mul_(eps).add_(row_atol))
        n_bad += (~ok).sum()
    n_bad_v, n_hot_v = torch.stack([n_bad, hot_rows.sum()]).tolist()
    logger.warning(
        "seg_reduce row-map self-check %d/%d: out=%s %s hot_rows=%d bad=%d",
        _rowmap_checks,
        _ROWMAP_CHECK_CALLS,
        tuple(out.shape),
        out.dtype,
        n_hot_v,
        n_bad_v,
    )
    if n_bad_v != 0:
        logger.error(
            "seg_reduce row-map self-check FAILED; disabling "
            "ONEFLOW_SEG_MULTISRC_ROWMAP for the rest of the process"
        )
        _rowmap_disabled = True
        for p in parts:
            if _is_row_map_grad(p):
                cast(Any, p).row_map_failed()
        return ref
    return out


def _sorted_segment_reduce_backward_row_map(
    parts, split_sizes, reverse_indices, num_rows, D, dtype, plan
):
    """_sorted_segment_reduce_backward_split for parts that include row-map
    gradients (see ONEFLOW_SEG_MULTISRC_ROWMAP)."""
    global _rowmap_disabled
    g = _row_map_sources(parts, D)
    out = None
    if isinstance(g, str):
        _rowmap_log(parts, f"materializing the row-map gradients: {g}")
    else:
        try:
            out = _sorted_segment_reduce_backward(g, reverse_indices, num_rows, plan)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            logger.exception(
                "seg_reduce row-map backward failed; disabling "
                "ONEFLOW_SEG_MULTISRC_ROWMAP for the rest of the process"
            )
            _rowmap_disabled = True
            for p in parts:
                if _is_row_map_grad(p):
                    cast(Any, p).row_map_failed()
    if out is None:
        dense = [p.materialize() if _is_row_map_grad(p) else p for p in parts]
        return _sorted_segment_reduce_backward_split(
            dense, split_sizes, reverse_indices, num_rows, D, dtype, plan
        )
    assert not isinstance(g, str)
    if num_rows * 2 < reverse_indices.numel():  # else logged as low duplication
        _rowmap_log(
            parts,
            f"{bin(g.mapped).count('1')} of {len(parts)} splits read through row maps",
        )
    if _rowmap_check_due(out):
        dense = [p.materialize() if _is_row_map_grad(p) else p for p in parts]
        ref = _sorted_segment_reduce_backward_split(
            dense, split_sizes, reverse_indices, num_rows, D, dtype
        )
        out = _rowmap_self_check(out, ref, reverse_indices, parts)
    return out


def _sorted_segment_reduce_backward_split(
    grads, split_sizes, reverse_indices, num_rows, D, dtype, plan=None
):
    """grad wrt embeddings for outs = embeddings.index_select(0, reverse_indices)
    .split(split_sizes); grads[k] is None for an output without a gradient.

    A missing gradient is zeros, as in the split backward. With the multi-source
    path on, the chunk path reads the splits in place; otherwise, or if that
    raises anything but an out-of-memory error, the splits are concatenated as
    the split backward does.
    """
    global _multisrc_disabled
    parts = []
    for gk, n in zip(grads, split_sizes):
        if gk is None:
            gk = torch.zeros(n, D, dtype=dtype, device=reverse_indices.device)
        assert gk.shape == (n, D) and gk.dtype == dtype, (gk.shape, gk.dtype, n, D)
        parts.append(gk)
    if any(_is_row_map_grad(p) for p in parts):
        return _sorted_segment_reduce_backward_row_map(
            parts, split_sizes, reverse_indices, num_rows, D, dtype, plan
        )
    g = _split_sources(parts, D)
    if isinstance(g, torch.Tensor):  # the only part
        return _sorted_segment_reduce_backward(g, reverse_indices, num_rows, plan)
    out = None
    if g is not None:
        try:
            out = _sorted_segment_reduce_backward(g, reverse_indices, num_rows, plan)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            logger.exception(
                "seg_reduce multi-source backward failed; disabling "
                "ONEFLOW_SEG_REDUCE_MULTISRC for the rest of the process"
            )
            _multisrc_disabled = True
    if out is None:
        return _sorted_segment_reduce_backward(
            torch.cat(parts), reverse_indices, num_rows, plan
        )
    if _multisrc_check_due(out):
        ref = _sorted_segment_reduce_backward(
            torch.cat(parts), reverse_indices, num_rows
        )
        out = _multisrc_self_check(out, ref, reverse_indices)
    return out


# Env ONEFLOW_G2C=1 (read per call by hammer.v2.modules.hstu_cint, which arms
# it with g2c_arm around the first access of the EC output, for the content
# features that ONEFLOW_CONTENT_CONCAT_INTO merges into the content buffer):
# the forward of sorted_index_select_split, on the multi-source path (see
# maybe_sorted_index_select_split), does not gather the rows of an armed
# feature's split. It returns a _G2CRows for that split instead, and the
# _Concat2DJaggedIntoFunction forward of hammer.ops.jagged that takes it as
# values_a gathers its rows straight into their concat output rows
# (_g2c_concat_kernel, which also copies values_b) instead of copying them from
# a gathered split. Every other use of a _G2CRows runs on its materialize(), the
# rows the split would have held, and logs the op once (WARNING). The values
# are copies either way, so the concat output is the same bits; the backward is
# unchanged. Once any op has read a _G2CRows (materialize()), the concat copies
# those materialized rows instead, so an in-place op on them, or on a view of
# them, reaches the concat output as without the flag.
# ONEFLOW_G2C_CHECK=1 also runs the concat of the first few gather-concats on
# the materialized rows into a scratch tensor and compares the bits (a
# rank-local verdict, no collective; syncs). On a mismatch, or when the kernel
# raises, it logs an ERROR with the rank, writes the old concat's rows and turns
# the path off for the rest of the process.
_G2C_CHECK_CALLS = 3
_g2c_checks = 0
_g2c_disabled = False
_g2c_armed: Optional[FrozenSet[str]] = None
_G2C_LOG_KEEP = 64
_g2c_logged: List[str] = []


def g2c_enabled() -> bool:
    """ONEFLOW_G2C=1 (read per call) and the path was not turned off."""
    return not _g2c_disabled and os.environ.get("ONEFLOW_G2C", "0") == "1"


def g2c_arm(keys: Optional[Iterable[str]]) -> None:
    """Arms ONEFLOW_G2C for the splits of the features ``keys`` (None or
    empty: disarms). The caller disarms once the EC output was waited."""
    global _g2c_armed
    armed = frozenset(keys) if keys is not None else frozenset()
    _g2c_armed = armed if armed else None


def _g2c_rank() -> str:
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        return f"rank {dist.get_rank()}"
    return "rank -"


def _g2c_log(message: str) -> None:
    """Logs ``message`` at WARNING once (the last _G2C_LOG_KEEP kept)."""
    if message in _g2c_logged:
        return
    _g2c_logged.append(message)
    del _g2c_logged[:-_G2C_LOG_KEEP]
    logger.warning("ONEFLOW_G2C (%s): %s", _g2c_rank(), message)


def _is_g2c_rows(t: object) -> bool:
    return getattr(type(t), "_oneflow_g2c_rows", False) is True


def _g2c_deferred(
    names: Optional[Sequence[str]],
    split_sizes: Sequence[int],
    embeddings: torch.Tensor,
    reverse_indices: torch.Tensor,
) -> Optional[Tuple[int, ...]]:
    """The splits (of ``names``, one name per split) whose gather ONEFLOW_G2C
    defers: armed, enabled, a name that occurs once, a non-empty split, plain
    embeddings rows with a unit column stride and contiguous 1-D int32/int64
    indices.
    None when there are none. Host-only checks."""
    armed = _g2c_armed
    if (
        armed is None
        or not g2c_enabled()
        or names is None
        or len(names) != len(split_sizes)
        or type(embeddings) is not torch.Tensor
        or embeddings.stride(1) != 1
        or type(reverse_indices) is not torch.Tensor
        or reverse_indices.dim() != 1
        or reverse_indices.stride(0) != 1
        or reverse_indices.dtype not in (torch.int32, torch.int64)
    ):
        return None
    deferred = tuple(
        k
        for k, (name, n) in enumerate(zip(names, split_sizes))
        if name in armed and names.count(name) == 1 and n > 0
    )
    return deferred if deferred else None


@triton.jit
def _g2c_concat_kernel(
    OffsetsA,
    Emb,  # [*, D] embedding rows: values_a row i is Emb[Idx[i]]
    Idx,  # [len(values_a)] int32/int64
    OffsetsB,
    ValuesB,
    Out,
    D,
    stride_e,
    stride_bd,
    stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The jagged-jagged concat of hammer's _Concat2DJaggedIntoFunction (the
    # multirow kernel without prefix, dense or replace parts): output row
    # OffsetsA[z] + OffsetsB[z] + n of sequence z is values_a row OffsetsA[z] + n
    # for n < its length, read from Emb through Idx, and then values_b.
    off_z = tl.program_id(1)
    start_n = tl.program_id(0) * BLOCK_N
    seq_start_a = tl.load(OffsetsA + off_z)
    seq_len_a = tl.load(OffsetsA + off_z + 1) - seq_start_a
    seq_start_b = tl.load(OffsetsB + off_z)
    seq_len_b = tl.load(OffsetsB + off_z + 1) - seq_start_b
    seq_len = seq_len_a + seq_len_b
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d[None, :] < D
    valid = offs_n < seq_len
    to_a = valid & (offs_n < seq_len_a)
    to_b = valid & (offs_n >= seq_len_a)
    out_ptrs = (
        Out
        + (seq_start_a + seq_start_b + offs_n[:, None]).to(tl.int64) * stride_od
        + offs_d[None, :]
    )
    rows = tl.load(Idx + (seq_start_a + offs_n).to(tl.int64), mask=to_a, other=0)
    a_mask = to_a[:, None] & d_mask
    v_a = tl.load(
        Emb + rows[:, None].to(tl.int64) * stride_e + offs_d[None, :],
        mask=a_mask,
        other=0.0,
    )
    tl.store(out_ptrs, v_a, mask=a_mask)
    b_mask = to_b[:, None] & d_mask
    v_b = tl.load(
        ValuesB
        + (seq_start_b + offs_n[:, None] - seq_len_a).to(tl.int64) * stride_bd
        + offs_d[None, :],
        mask=b_mask,
        other=0.0,
    )
    tl.store(out_ptrs, v_b, mask=b_mask)


def _g2c_config(D: int, element_size: int) -> Tuple[int, int, int]:
    """(BLOCK_D, BLOCK_N, num_warps) of _g2c_concat_kernel for rows of D
    elements: up to 8 rows of at most 2048 elements per program, and one
    16-byte vector per thread where the tile allows (at most 4 warps)."""
    block_d = triton.next_power_of_2(D)
    block_n = max(1, min(8, 2048 // block_d))
    num_warps = max(1, min(4, block_n * block_d * element_size // (16 * 64)))
    return block_d, block_n, num_warps


class _G2CRows(torch.Tensor):
    """A split of sorted_index_select_split whose gather ONEFLOW_G2C deferred:
    row i is row ``indices[i]`` of ``embeddings``. It has the split's size,
    contiguous strides, dtype and device, but no storage of its own: every op
    on it (and data_ptr()) runs on materialize(), the gathered rows, so its
    values are the split's. hammer.ops.jagged recognizes it by
    ``_oneflow_g2c_rows`` and calls concat_into instead."""

    _oneflow_g2c_rows = True
    _emb: torch.Tensor
    _idx: torch.Tensor
    _dense: Optional[torch.Tensor]

    @staticmethod
    def __new__(cls, embeddings: torch.Tensor, indices: torch.Tensor) -> "_G2CRows":
        rows = torch.Tensor._make_wrapper_subclass(
            cls,
            (indices.numel(), embeddings.shape[1]),
            dtype=embeddings.dtype,
            device=embeddings.device,
            requires_grad=False,
        )
        rows._emb = embeddings.detach()
        rows._idx = indices
        rows._dense = None
        return rows

    __torch_function__: Any = torch._C._disabled_torch_function_impl

    @classmethod
    # pyre-ignore[3]
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # pyre-ignore[2]
        def dense(x):  # pyre-ignore[2, 3]
            return x.materialize() if isinstance(x, _G2CRows) else x

        from torch.utils._pytree import tree_map

        _g2c_log(f"a deferred split was materialized by {func}")
        return func(*tree_map(dense, args), **tree_map(dense, kwargs or {}))

    def data_ptr(self) -> int:  # pyre-ignore[14]
        _g2c_log("a deferred split was materialized by data_ptr()")
        return self.materialize().data_ptr()

    def dense(self) -> torch.Tensor:
        """The rows the split would have held, as a new tensor."""
        return self._emb.index_select(0, self._idx)

    def materialize(self) -> torch.Tensor:
        """dense(), computed once."""
        if self._dense is None:
            self._dense = self.dense()
        return self._dense

    def g2c_materialize(self) -> torch.Tensor:
        """dense() as a tensor whose gradient flows to these rows, for a
        consumer other than the gather-concat (logged once)."""
        _g2c_log("a deferred split was materialized for a consumer besides the concat")
        return _G2CMaterialize.apply(self)

    def concat_into(
        self,
        values_b: torch.Tensor,
        offsets_a: torch.Tensor,
        offsets_b: torch.Tensor,
        out: torch.Tensor,
        max_seq_len: int,
    ) -> None:
        """Writes the rows of the jagged concat of these rows (values_a) and
        values_b (unit column stride) into ``out``, as
        _Concat2DJaggedIntoFunction does, gathering these rows in place."""
        B = offsets_a.numel() - 1
        D = self.shape[1]
        # no rows to write: the old concat grid (cdiv(max_seq_len, BLOCK_N), B) is empty
        if B <= 0 or max_seq_len <= 0:
            return
        block_d, block_n, num_warps = _g2c_config(D, self.element_size())
        cast(Any, _g2c_concat_kernel)[(triton.cdiv(max_seq_len, block_n), B)](
            offsets_a,
            self._emb,
            self._idx,
            offsets_b,
            values_b,
            out,
            D,
            self._emb.stride(0),
            values_b.stride(0),
            out.stride(0),
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

    def g2c_use_dense(self) -> bool:
        """Whether the concat copies materialize() instead of gathering: a
        failure turned ONEFLOW_G2C off since these rows were deferred, or an
        op already read them (and may have changed them in place)."""
        return _g2c_disabled or self._dense is not None

    def g2c_check_due(self) -> bool:
        """True (and counted) for the first _G2C_CHECK_CALLS gather-concats
        with ONEFLOW_G2C_CHECK=1."""
        global _g2c_checks
        if (
            _g2c_checks >= _G2C_CHECK_CALLS
            or os.environ.get("ONEFLOW_G2C_CHECK", "0") != "1"
            or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        ):
            return False
        _g2c_checks += 1
        return True

    def g2c_check_result(self, equal: bool, shape: Tuple[int, ...]) -> None:
        """Logs a self-check verdict; a mismatch turns the path off."""
        global _g2c_disabled
        logger.warning(
            "ONEFLOW_G2C self-check %d/%d (%s): concat of %s rows into %s equal=%s",
            _g2c_checks,
            _G2C_CHECK_CALLS,
            _g2c_rank(),
            self.shape[0],
            shape,
            equal,
        )
        if not equal:
            logger.error(
                "ONEFLOW_G2C self-check FAILED (%s); using the old concat and "
                "turning ONEFLOW_G2C off for the rest of the process",
                _g2c_rank(),
            )
            _g2c_disabled = True

    def g2c_failed(self) -> None:
        """In an exception handler: logs the error and turns the path off."""
        global _g2c_disabled
        logger.exception(
            "ONEFLOW_G2C gather-concat failed (%s); using the old concat and "
            "turning ONEFLOW_G2C off for the rest of the process",
            _g2c_rank(),
        )
        _g2c_disabled = True


class _G2CMaterialize(torch.autograd.Function):
    """The rows of a _G2CRows as a new tensor, with an identity backward."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, rows: _G2CRows) -> torch.Tensor:  # pyre-ignore[2]
        return rows.dense()

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # pyre-ignore[2]
        return grad


def g2c_materialize(t: torch.Tensor) -> torch.Tensor:
    """``t``, or for a _G2CRows its rows as a new tensor whose gradient flows
    to it (ONEFLOW_G2C: for a consumer other than the gather-concat)."""
    return cast(Any, t).g2c_materialize() if _is_g2c_rows(t) else t


def _g2c_split_forward(
    embeddings: torch.Tensor,
    reverse_indices: torch.Tensor,
    split_sizes: Tuple[int, ...],
    deferred: Tuple[int, ...],
) -> Tuple[torch.Tensor, ...]:
    """embeddings.index_select(0, reverse_indices).split(split_sizes) with a
    _G2CRows for each split in ``deferred``: each run of other splits is one
    gather of its indices, split."""
    starts = [0]
    for n in split_sizes:
        starts.append(starts[-1] + n)
    outs: List[torch.Tensor] = []
    k = 0
    while k < len(split_sizes):
        if k in deferred:
            outs.append(
                _G2CRows(
                    embeddings, reverse_indices.narrow(0, starts[k], split_sizes[k])
                )
            )
            k += 1
            continue
        j = k
        while j < len(split_sizes) and j not in deferred:
            j += 1
        rows = embeddings.index_select(
            0, reverse_indices.narrow(0, starts[k], starts[j] - starts[k])
        )
        outs.extend(rows.split(split_sizes[k:j]))
        k = j
    return tuple(outs)


class _SortedIndexSelectSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, reverse_indices, split_sizes, g2c_deferred=None):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(reverse_indices)
        ctx.num_rows = embeddings.shape[0]
        ctx.dim = embeddings.shape[1]
        ctx.dtype = embeddings.dtype
        ctx.split_sizes = tuple(split_sizes)
        ctx.g2c = g2c_deferred is not None
        if g2c_deferred is not None:
            return _g2c_split_forward(
                embeddings, reverse_indices, ctx.split_sizes, g2c_deferred
            )
        return embeddings.index_select(0, reverse_indices).split(ctx.split_sizes)

    @staticmethod
    # pyrefly: ignore[bad-override]
    def backward(ctx, *grads):
        (reverse_indices,) = ctx.saved_tensors
        grad_emb = _sorted_segment_reduce_backward_split(
            grads,
            ctx.split_sizes,
            reverse_indices,
            ctx.num_rows,
            ctx.dim,
            ctx.dtype,
            plan=_take_seg_plan(ctx),
        )
        if ctx.g2c:
            return grad_emb, None, None, None
        return grad_emb, None, None


def sorted_index_select_split(
    embeddings,
    reverse_indices,
    split_sizes: List[int],
    seg_plan=None,
    g2c_deferred: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, ...]:
    """``embeddings.index_select(0, reverse_indices).split(split_sizes)`` (2-D
    embeddings) with the sorted_index_select backward, which with
    ONEFLOW_SEG_REDUCE_MULTISRC=1 reads the per-split gradients without
    concatenating them. Training only. ``seg_plan``: see build_sorted_plan.
    ``g2c_deferred`` (ONEFLOW_G2C, from _g2c_deferred): the splits returned as
    _G2CRows."""
    if g2c_deferred is None:
        outs = _SortedIndexSelectSplit.apply(embeddings, reverse_indices, split_sizes)
    else:
        outs = _SortedIndexSelectSplit.apply(
            embeddings, reverse_indices, split_sizes, g2c_deferred
        )
    if seg_plan is not None and len(outs) > 0:
        _attach_seg_plan(outs[0], _SortedIndexSelectSplit, seg_plan)
    return outs


def maybe_sorted_index_select_split(
    embeddings,
    reverse_indices,
    features,
    seg_plan=None,
    names: Optional[Sequence[str]] = None,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """sorted_index_select_split(embeddings, reverse_indices,
    features.length_per_key()) when ONEFLOW_SEG_REDUCE_MULTISRC=1 and
    ONEFLOW_SEG_REDUCE_FUSED_PERM=1 and its backward can take the multi-source
    chunk path, else None: the caller then runs sorted_index_select and
    torch.split itself.

    Eager training only: None under TorchScript, FX symbolic tracing,
    torch.compile or export, with grad mode off, or when embeddings needs no
    gradient. None also for non-2-D embeddings, a dtype not in _TL_DTYPES, rows
    that are not a multiple of 16 bytes, low duplication (the backward
    index_adds the concatenation) and more than _MAX_SRC keys.
    features.length_per_key() is only called once the other conditions hold.
    torch.compiler.is_compiling() comes first so that dynamo stops at a
    constant, and the flags right after it, so with them off nothing else runs.
    ``seg_plan`` is passed to sorted_index_select_split. ``names`` (one per
    split) are the features ONEFLOW_G2C matches against its armed ones."""
    if (
        torch.jit.is_scripting()
        or torch.compiler.is_compiling()
        or not _multisrc_enabled()
        or not _fused_perm_enabled()
        or is_fx_symbolic_tracing()
        or not torch.is_grad_enabled()
        or not embeddings.requires_grad
        or embeddings.dim() != 2
        or embeddings.dtype not in _TL_DTYPES
        or (embeddings.shape[1] * embeddings.element_size()) % 16 != 0
        or embeddings.shape[0] * 2 >= reverse_indices.numel()
    ):
        return None
    split_sizes = features.length_per_key()
    if len(split_sizes) > _MAX_SRC:
        return None
    g2c_deferred = (
        None
        if _g2c_armed is None
        else _g2c_deferred(names, split_sizes, embeddings, reverse_indices)
    )
    return sorted_index_select_split(
        embeddings,
        reverse_indices,
        split_sizes,
        seg_plan=seg_plan,
        g2c_deferred=g2c_deferred,
    )


def maybe_sorted_index_select(embeddings, indices, use_sorted: bool):
    """sorted_index_select when ``use_sorted`` is True, else the plain
    index_select(0) fallback. The caller decides (e.g. from its own config or
    env); only the backward reads the opt-in ONEFLOW_SEG_REDUCE_* flags.

    Bypassed under torch.jit.script (publish/inference lowering): the custom
    autograd.Function and its triton kernel are not TorchScript-compilable, and
    this is a training-only (backward) optimization -- the forward gather is
    identical -- so scripted inference just uses the native index_select."""
    if torch.jit.is_scripting():
        return embeddings.index_select(0, indices)
    if use_sorted:
        return sorted_index_select(embeddings, indices)
    return embeddings.index_select(0, indices)


# Env ONEFLOW_SEG_PRE_PLAN=1 (read per call by the code that builds the plans,
# torchrec.distributed.embedding for the EC dedup expand): build_sorted_plan runs
# the index-only part of the sorted_index_select(_split) backward (the sort, the
# segment and chunk offsets, the hot-row map and buffer, the item->row map) while
# the forward waits for its embeddings, and that backward, given the plan,
# launches only the kernels that read the gradient. The plan tensors are the ones
# the backward would compute from the same indices in the same
# ONEFLOW_SEG_REDUCE_* modes, so the gradient is unchanged. A plan whose indices
# or modes no longer match, or that a stream not ordered after its build would
# read, is ignored; the first few ignored plans are logged.
# ONEFLOW_SEG_PRE_PLAN_CHECK=1 also rebuilds the first few plans in the backward
# and compares them. On a mismatch it uses the rebuilt plan and stops using plans
# for the rest of the process.
# Host cost: the builds are eager ops in the forward (in the hstu job about 150
# per step, 7 of them sorts, issued while the EC all-to-all is in flight).
# PlanBuildTimer logs their host time once; a profiler inflates it several-fold,
# so judge the flag by unprofiled QPS.
# Duplicated in fbr/flash/triton/embedding_bwd_triton_override.py.
_SEG_PRE_PLAN_CHECK_CALLS = 3
_seg_pre_plan_checks = 0
_seg_pre_plan_disabled = False
_SEG_PLAN_NOT_USED_LOGS = 3
_seg_plans_not_used = 0
_PLAN_BUILD_SKIP = 50
_PLAN_BUILD_TIMED = 100


def _narrow_sort_on() -> bool:
    """Whether _narrow_keys narrows (for a num_rows that fits)."""
    return (
        not _narrow_sort_disabled
        and os.environ.get("ONEFLOW_SEG_REDUCE_NARROW_SORT", "1") == "1"
    )


def _seg_modes(dtype: torch.dtype) -> Tuple[bool, bool, bool]:
    """The ONEFLOW_SEG_REDUCE_* modes the plan tensors depend on, for a gradient
    of ``dtype``: narrow sort, row-direct, item->row map."""
    return (
        _narrow_sort_on(),
        dtype != torch.float32 and _row_direct_enabled(),
        _item_row_enabled(),
    )


class SortedSegPlan:
    """Index-only tensors of the chunk path of _sorted_segment_reduce_backward
    for the 1-D ``idx`` (reverse_indices), ``num_rows`` embedding rows and a
    gradient of width ``dim``, built in the modes ``modes`` (see _seg_modes).

    ``hot`` is the zeroed fp32 hot-row buffer of a row-direct plan; the first
    backward takes it, and any later one allocates its own.

    ``stream`` is a CUDA stream that may read the tensors (None on CPU): the
    one that built them, until for_current_stream orders another one after
    the build. ``ready`` is the event record_sorted_plans_ready recorded on
    the build stream after the build, or None.
    """

    def __init__(
        self,
        idx: torch.Tensor,
        num_rows: int,
        dim: int,
        modes: Tuple[bool, bool, bool],
        perm: torch.Tensor,
        off: torch.Tensor,
        nchoff: torch.Tensor,
        item_row: Optional[torch.Tensor],
        g_max: int,
        hcum: Optional[torch.Tensor],
        hot: Optional[torch.Tensor],
        max_hot: int,
    ) -> None:
        self.idx = idx
        self.idx_version: int = idx._version
        self.num_rows = num_rows
        self.dim = dim
        self.modes = modes
        self.row_direct: bool = modes[1]
        self.perm = perm
        self.off = off
        self.nchoff = nchoff
        self.item_row = item_row
        self.g_max = g_max
        self.hcum = hcum
        self.hot = hot
        self.max_hot = max_hot
        self.stream: Optional[torch.cuda.Stream] = _plan_stream(idx.device)
        self.ready: Optional[torch.cuda.Event] = None

    def take_hot(self) -> Optional[torch.Tensor]:
        hot, self.hot = self.hot, None
        return hot

    def tensors(self) -> List[torch.Tensor]:
        ts = [self.perm, self.off, self.nchoff, self.item_row, self.hcum, self.hot]
        return [t for t in ts if t is not None]

    def record_stream(self, stream: torch.Stream) -> None:
        for t in self.tensors():
            t.record_stream(stream)

    def for_current_stream(self) -> Optional["SortedSegPlan"]:
        """This plan if the current stream may read it, else None (logged).

        It may if it is ``stream``, or once it waits for ``ready`` (a stream
        wait, no host sync); it then also holds the tensors for the caching
        allocator and becomes ``stream``.
        """
        cur = _plan_stream(self.idx.device)
        if cur is None or cur == self.stream:
            return self
        if self.ready is None:
            _sorted_plan_not_used("stream", self, self.num_rows, self.dim)
            return None
        cur.wait_event(self.ready)
        self.record_stream(cur)
        self.stream = cur
        return self


def _plan_stream(device: torch.device) -> Optional[torch.cuda.Stream]:
    """The current stream of a CUDA ``device``; None for other devices."""
    return torch.cuda.current_stream(device) if device.type == "cuda" else None


def record_sorted_plans_ready(plans: List[Optional[SortedSegPlan]]) -> None:
    """Records one event on the current stream, which built ``plans``, for
    for_current_stream on another stream to wait for. Not under graph
    capture: such plans are only used on their build stream."""
    built = [p for p in plans if p is not None and p.stream is not None]
    if not built or torch.cuda.is_current_stream_capturing():
        return
    stream = _plan_stream(built[0].idx.device)
    assert stream is not None
    ready = cast(torch.cuda.Event, stream.record_event())
    for p in built:
        p.ready = ready


def build_sorted_plan(
    reverse_indices: torch.Tensor, num_rows: int, dim: int, dtype: torch.dtype
) -> Optional[SortedSegPlan]:
    """The SortedSegPlan of the sorted_index_select(_split) backward for 1-D
    ``reverse_indices`` into ``num_rows`` embedding rows of width ``dim``, with
    a gradient of ``dtype``, in the current modes.

    The statements are the index-only ones of _sorted_segment_reduce_backward,
    _seg_reduce_row_direct and _launch_seg_chunk, on the same inputs. None
    where that backward does not sort (low duplication: index_add) and after a
    failed plan self-check.
    """
    N = reverse_indices.numel()
    if _seg_pre_plan_disabled or _native_index_add_preferred(N, num_rows, dim):
        return None
    dev = reverse_indices.device
    sorted_idx, perm = _sort_indices(reverse_indices, num_rows)
    off = torch.searchsorted(
        sorted_idx, torch.arange(num_rows + 1, device=dev, dtype=sorted_idx.dtype)
    ).to(torch.int64)
    del sorted_idx
    counts = off[1:] - off[:-1]
    nch = (counts + _CHUNK - 1) // _CHUNK
    row_direct = dtype != torch.float32 and _row_direct_enabled()
    hcum = None
    hot = None
    max_hot = 0
    if row_direct:
        nch_rd = nch.clamp(min=1)
        hcum = torch.cat([nch.new_zeros(1), (nch_rd > 1).cumsum(0)])
        nchoff = torch.cat([nch.new_zeros(1), nch_rd.cumsum(0)])
        max_hot = min(num_rows, N // (_CHUNK + 1))
        hot = torch.zeros(max(max_hot, 1), dim, dtype=torch.float32, device=dev)
    else:
        nchoff = torch.cat([nch.new_zeros(1), nch.cumsum(0)])
    g_max = num_rows + N // _CHUNK + 1
    # _item_row_map reads its output argument only for numel() (its self-check
    # skips an empty output), which a meta tensor gives without allocating.
    item_row = _item_row_map(nchoff, torch.empty(num_rows, dim, device="meta"), g_max)
    return SortedSegPlan(
        idx=reverse_indices,
        num_rows=num_rows,
        dim=dim,
        modes=(_narrow_sort_on(), row_direct, item_row is not None),
        perm=perm,
        off=off,
        nchoff=nchoff,
        item_row=item_row,
        g_max=g_max,
        hcum=hcum,
        hot=hot,
        max_hot=max_hot,
    )


def _attach_seg_plan(out: torch.Tensor, fn, plan: SortedSegPlan) -> None:
    """Gives ``plan`` to the backward of the ``fn`` call that returned ``out``:
    the ctx of a custom autograd Function is its output's grad_fn. Dropped when
    ``out`` was not recorded for backward."""
    if torch.compiler.is_compiling():
        return
    node: Any = out.grad_fn
    if getattr(node, "_forward_cls", None) is fn:
        node.seg_plan = plan


def _take_seg_plan(ctx) -> Optional[SortedSegPlan]:
    """The plan _attach_seg_plan gave this backward, once."""
    plan = getattr(ctx, "seg_plan", None)
    if plan is not None:
        ctx.seg_plan = None
    return plan


def _sorted_plan_misfit(
    plan: SortedSegPlan,
    idx: torch.Tensor,
    grad_rows: int,
    num_rows: int,
    dim: int,
    dtype: torch.dtype,
) -> Optional[str]:
    """None if ``plan`` was built from ``idx`` (unmodified since) for these
    sizes, in the modes this backward runs in; else the first term that
    differs."""
    p = plan.idx
    terms = (
        ("plans disabled", not _seg_pre_plan_disabled),
        ("num_rows", plan.num_rows == num_rows),
        ("dim", plan.dim == dim),
        ("grad rows", grad_rows == idx.numel()),
        ("index version", idx._version == plan.idx_version),
        (
            "index tensor",
            (idx.dtype, idx.device, idx.shape, idx.stride(), idx.data_ptr())
            == (p.dtype, p.device, p.shape, p.stride(), p.data_ptr()),
        ),
        ("modes", plan.modes == _seg_modes(dtype)),
    )
    for term, ok in terms:
        if not ok:
            return term
    return None


def _sorted_plan_not_used(
    term: str, plan: SortedSegPlan, num_rows: int, dim: int
) -> None:
    """Logs the first _SEG_PLAN_NOT_USED_LOGS plans a backward ignores."""
    global _seg_plans_not_used
    _seg_plans_not_used += 1
    if _seg_plans_not_used <= _SEG_PLAN_NOT_USED_LOGS:
        logger.warning(
            "seg_reduce pre-plan not used (%s differs; num_rows %d/%d, dim %d/%d, "
            "plan/call): plan-free backward",
            term,
            plan.num_rows,
            num_rows,
            plan.dim,
            dim,
        )


def _seg_plan_check_due() -> bool:
    return (
        _seg_pre_plan_checks < _SEG_PRE_PLAN_CHECK_CALLS
        and os.environ.get("ONEFLOW_SEG_PRE_PLAN_CHECK", "1") == "1"
        and not torch.cuda.is_current_stream_capturing()
    )


def _sorted_plans_equal(a: SortedSegPlan, b: SortedSegPlan) -> bool:
    """Same scalars and bit-identical tensors (the hot buffers only if both
    still have one). Syncs once."""
    if (a.modes, a.g_max, a.max_hot) != (b.modes, b.g_max, b.max_hot):
        return False
    pairs = [
        (a.perm, b.perm),
        (a.off, b.off),
        (a.nchoff, b.nchoff),
        (a.item_row, b.item_row),
        (a.hcum, b.hcum),
    ]
    if a.hot is not None and b.hot is not None:
        pairs.append((a.hot, b.hot))
    same = []
    for x, y in pairs:
        if x is None or y is None:
            if x is not y:
                return False
            continue
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if x.is_floating_point():
            x, y = x.view(torch.int32), y.view(torch.int32)
        same.append((x == y).all())
    return bool(torch.stack(same).all())


def _sorted_plan_self_check(
    plan: SortedSegPlan,
    idx: torch.Tensor,
    num_rows: int,
    dim: int,
    dtype: torch.dtype,
) -> Optional[SortedSegPlan]:
    """Compare ``plan`` with one rebuilt from the indices now.

    Syncs once; only called on the first _SEG_PRE_PLAN_CHECK_CALLS planned
    backwards. On mismatch, disables plans for the process and returns the
    rebuilt plan.
    """
    global _seg_pre_plan_checks, _seg_pre_plan_disabled
    _seg_pre_plan_checks += 1
    ref = build_sorted_plan(idx, num_rows, dim, dtype)
    ok = ref is not None and _sorted_plans_equal(plan, ref)
    logger.warning(
        "seg_reduce pre-plan self-check %d/%d: n=%d rows=%d modes=%s ok=%s",
        _seg_pre_plan_checks,
        _SEG_PRE_PLAN_CHECK_CALLS,
        idx.numel(),
        num_rows,
        plan.modes,
        ok,
    )
    if not ok:
        logger.error(
            "seg_reduce pre-plan self-check FAILED; disabling "
            "ONEFLOW_SEG_PRE_PLAN for the rest of the process"
        )
        _seg_pre_plan_disabled = True
        return ref
    return plan


def _segment_reduce_with_plan(grad_out, reverse_indices, num_rows, plan):
    """_sorted_segment_reduce_backward(grad_out, reverse_indices, num_rows) for
    a contiguous grad_out (or _SplitGrads) on the tensors of ``plan``: the
    launches that read grad_out, with their self-checks. None if the plan does
    not fit or this stream may not read it; the caller then runs the plan-free
    path."""
    N, D = grad_out.shape
    misfit = _sorted_plan_misfit(plan, reverse_indices, N, num_rows, D, grad_out.dtype)
    if misfit is not None:
        _sorted_plan_not_used(misfit, plan, num_rows, D)
        return None
    plan = plan.for_current_stream()
    if plan is None:
        return None
    if _seg_plan_check_due():
        plan = _sorted_plan_self_check(
            plan, reverse_indices, num_rows, D, grad_out.dtype
        )
        if plan is None:
            return None
    dev = grad_out.device
    perm, off, nchoff, g_max = plan.perm, plan.off, plan.nchoff, plan.g_max
    fused_perm = _fused_perm_enabled()
    g = grad_out if fused_perm else _dense(grad_out).index_select(0, perm).contiguous()
    if plan.row_direct:
        hot = plan.take_hot()
        if hot is None:
            hot = torch.zeros(max(plan.max_hot, 1), D, dtype=torch.float32, device=dev)
        out = torch.empty(num_rows, D, dtype=grad_out.dtype, device=dev)
        _launch_seg_chunk(
            g, perm, off, nchoff, out, g_max, fused_perm, plan.hcum, hot, plan.item_row
        )
        if plan.max_hot > 0:
            _seg_hot_rows[(plan.max_hot,)](
                hot,
                plan.hcum,
                out,
                num_rows,
                D,
                BLOCK_D=triton.next_power_of_2(D),
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
        if _row_direct_check_due(out):
            nch = (off[1:] - off[:-1] + _CHUNK - 1) // _CHUNK
            ref = torch.zeros(num_rows, D, dtype=torch.float32, device=dev)
            nchoff_ref = torch.cat([nch.new_zeros(1), nch.cumsum(0)])
            _launch_seg_chunk(g, perm, off, nchoff_ref, ref, g_max, fused_perm)
            out = _row_direct_self_check(out, ref, nch.clamp(min=1) > 1)
        return out
    out = torch.zeros(num_rows, D, dtype=torch.float32, device=dev)
    _launch_seg_chunk(
        g, perm, off, nchoff, out, g_max, fused_perm, item_row=plan.item_row
    )
    if fused_perm and _fused_perm_check_due(out):
        ref = torch.zeros_like(out)
        g_ref = _dense(grad_out).index_select(0, perm).contiguous()
        _launch_seg_chunk(g_ref, perm, off, nchoff, ref, g_max, False)
        del g_ref
        out = _fused_perm_self_check(out, ref)
    return out.to(grad_out.dtype)


class PlanBuildTimer:
    """Host time of one call site's plan builds, measured by the caller with
    time.perf_counter (no device sync), logged once on rank 0 over builds
    _PLAN_BUILD_SKIP + 1 .. _PLAN_BUILD_SKIP + _PLAN_BUILD_TIMED: past the
    first builds, which run the ONEFLOW_SEG_REDUCE_*_CHECK syncs."""

    def __init__(self, site: str) -> None:
        self.site = site
        self.builds = 0
        self.times: List[float] = []

    def add(self, seconds: float) -> None:
        """Counts one build that took ``seconds`` of host time."""
        last = _PLAN_BUILD_SKIP + _PLAN_BUILD_TIMED
        if self.builds >= last:
            return
        self.builds += 1
        if self.builds > _PLAN_BUILD_SKIP:
            self.times.append(seconds)
        if self.builds < last:
            return
        ts, self.times = sorted(self.times), []
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.warning(
            "seg_reduce pre-plan builds (%s), host time over builds %d-%d: "
            "median %.3f ms, p90 %.3f ms, max %.3f ms",
            self.site,
            _PLAN_BUILD_SKIP + 1,
            last,
            1e3 * ts[len(ts) // 2],
            1e3 * ts[len(ts) * 9 // 10],
            1e3 * ts[-1],
        )


EC_PLAN_BUILD_TIMER: PlanBuildTimer = PlanBuildTimer("EmbeddingCollection dedup expand")
