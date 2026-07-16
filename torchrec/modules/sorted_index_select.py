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
  early-exit. Each program binary-searches nchoff to find its row. No .item().
"""

import torch
import triton  # @manual
import triton.language as tl  # @manual


_CHUNK = 512  # sorted entries reduced per work item (bounds hot-row tail)
_BUF = 16  # rows buffered per vectorized load
_NUM_WARPS = 1
_NUM_STAGES = 2


@triton.jit
def _seg_chunk(
    g_ptr,  # grad_out gathered into sorted order, [N, D]
    off_ptr,  # [num_rows + 1] segment offsets in the sorted array
    nchoff_ptr,  # [num_rows + 1] cumsum of chunks-per-row; nchoff[num_rows] = total
    out_ptr,  # [num_rows, D] fp32
    num_rows,
    D,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BUF: tl.constexpr,
):
    pid = tl.program_id(0)
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
    while i < nfull:
        ridx = (i + bo).to(tl.int64)
        blk = tl.load(
            g_ptr + ridx[:, None] * D + col[None, :], mask=mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(blk, axis=0)
        i += BUF
    while i < seg_e:
        acc += tl.load(g_ptr + i.to(tl.int64) * D + col, mask=mask, other=0.0).to(
            tl.float32
        )
        i += 1
    oo = out_ptr + tl.cast(row, tl.int64) * D + col
    if nch_row == 1:
        tl.store(oo, acc, mask=mask)  # short row: single writer, no atomic
    else:
        tl.atomic_add(oo, acc, mask=mask)  # hot row: few bounded atomics, parallel


def _sorted_segment_reduce_backward(grad_out, reverse_indices, num_rows):
    """grad wrt embeddings for out = embeddings.index_select(0, reverse_indices).
    Fully on-device: no host scalar reads, no unique_consecutive/bincount/.item()."""
    grad_out = grad_out.contiguous()
    _, D = grad_out.shape
    dev = grad_out.device

    # Duplication-aware route. The chunk path only pays off when many gradient rows
    # collide into few destination rows (hot-row atomic contention). For the concat
    # backward the two operands are very different: the LEFT (small values_left) is
    # ~1.17M indices into ~1.9K rows (dup ~600x, sentinel hot) = 18ms native -> chunk
    # wins; the RIGHT (large values_right) is ~1.17M into ~1.17M rows (dup ~1) = 3ms
    # native, where the sort overhead makes the chunk path a LOSS. So when dup is low
    # (dest is at least half the index count) fall straight back to native index_add.
    N = reverse_indices.numel()
    if num_rows * 2 >= N:
        out = torch.zeros(num_rows, D, dtype=grad_out.dtype, device=dev)
        out.index_add_(0, reverse_indices, grad_out)
        return out

    sorted_idx, perm = torch.sort(reverse_indices)
    g_sorted = grad_out.index_select(0, perm).contiguous()
    # Offsets straight from the SORTED indices: off[r] = #(idx < r). This is
    # atomic-free. The earlier counts.scatter_add_(0, sorted_idx, 1) version was a
    # HOT-ROW ATOMIC -- the sentinel/padding row appears in ~half the indices, so
    # ~N/2 atomicAdds land on counts[sentinel]. Atomic contention is latency-bound,
    # so the 1-wide long payload does not help: it was just as serialized as the
    # 128-wide index_add this kernel replaces (showed up slow in the MI350 trace).
    # searchsorted exploits the sortedness to avoid all atomics, and is still
    # sync-free (output size num_rows+1 is host-known, no data-dependent read-back).
    off = torch.searchsorted(sorted_idx, torch.arange(num_rows + 1, device=dev)).to(
        torch.int64
    )
    counts = off[1:] - off[:-1]
    nch = (counts + _CHUNK - 1) // _CHUNK  # chunks per row
    nchoff = torch.cat([nch.new_zeros(1), nch.cumsum(0)])  # nchoff[-1] = total items

    out = torch.zeros(num_rows, D, dtype=torch.float32, device=dev)
    # Static host-known upper bound on total work items: sum(ceil(c_i/CHUNK))
    # <= num_rows + N/CHUNK. Excess programs early-exit in the kernel.
    g_max = num_rows + reverse_indices.numel() // _CHUNK + 1
    _seg_chunk[(g_max,)](
        g_sorted,
        off,
        nchoff,
        out,
        num_rows,
        D,
        CHUNK=_CHUNK,
        BLOCK_D=triton.next_power_of_2(D),
        BUF=_BUF,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
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
            grad_out, reverse_indices, ctx.num_rows
        )
        return grad_emb, None


def sorted_index_select(embeddings, reverse_indices):
    """Drop-in for ``embeddings.index_select(0, reverse_indices)`` with a fast,
    fp32, sync-free sorted (chunk-split) backward. Training only."""
    return _SortedIndexSelect.apply(embeddings, reverse_indices)


def maybe_sorted_index_select(embeddings, indices, use_sorted: bool):
    """sorted_index_select when ``use_sorted`` is True, else the plain
    index_select(0) fallback. The caller decides (e.g. from its own config or
    env); this module reads no global/env state itself.

    Bypassed under torch.jit.script (publish/inference lowering): the custom
    autograd.Function and its triton kernel are not TorchScript-compilable, and
    this is a training-only (backward) optimization -- the forward gather is
    identical -- so scripted inference just uses the native index_select."""
    if torch.jit.is_scripting():
        return embeddings.index_select(0, indices)
    if use_sorted:
        return sorted_index_select(embeddings, indices)
    return embeddings.index_select(0, indices)
