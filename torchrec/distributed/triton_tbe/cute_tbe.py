#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL kernels for Table Batched Embedding (TBE) forward and backward.

Drop-in replacements for the Triton TBE kernels in this directory, enabled with
``TritonTableBatchedEmbeddingBags(cute_kernels=True)`` (see cute_tbe_launchers):

* ``TbeForward``: pooled (SUM) forward, unweighted and weighted, with VBE,
  replacing ``table_batched_embedding_bag_forward_{unweighted,weighted}_kernel``.
  1.4-2x faster than Triton on GB300 production shards A-D (A 17.7 -> 11.8 ms,
  B 5.7 -> 3.6, C 4.67 -> 2.99, D 1.41 -> 0.69; VBE A 19.8 -> 11.5).
* ``TbeBackwardPacked`` (default): short-run backward with exact row-wise
  Adagrad, unweighted and weighted, with VBE, replacing
  ``triton_tbe_backward_short_run_{unweighted,weighted}``. Short runs are mostly
  1-3 lookups, so it packs several runs per warp instruction (16-lane groups
  at D <= 128) with runs per batch scaled to the bucket. GB300 short-run kernel
  vs Triton: shard A ~38 -> 22.9 ms, shard C 12.3 -> 6.65, shard C weighted 5.07
  -> 3.22, shard B 4.7 -> 2.3, shard D 1.91 -> 1.79.
* ``TbeLongRunAccum`` + ``TbeLongRunApply``: unweighted long-run tier (rows with
  >= 256 lookups, VBE too), replacing
  ``triton_tbe_backward_long_run_{grad_accum,fused,apply}_unweighted``. Hot rows
  come from small tables with big bags, so repeated dout rows in a chunk are
  loaded once and scaled by their count, chunks run in (feature, sample) order
  so hot rows sharing a sample range hit L2, and hot narrow features read a
  compact copy of their dout columns. shard A accumulate 2.5 -> 2.15 ms (incl.
  the copy; ~86% of the L2 gather floor for its 16 B-aligned dout rows), apply
  0.59 -> 0.04 ms. Only D % 4 != 0 chunks are left to Triton.
* ``TbeBackwardTiled`` (CUTE_TBE_BWD_TILE_B): regroups lookups by (batch tile,
  row) so a tile's dout stays in L2 (shard D DRAM reads 8.2 -> 2.8 GB; rows
  spanning tiles are summed per tile and combined in tile order).
  ``TileRunsPrep`` builds the tiled runs with a counting sort. Only pays where
  dout re-reads dominate: shard D reaches parity end to end, shards C / B lose.

Why the forward wins (all measured with ncu on GB300):

1. Flat index stream. Each warp owns 32 consecutive bags of one feature; their
   offsets arrive in one coalesced load and, because their indices are
   contiguous, the warp walks them as one stream: 32 indices per load and
   ``unroll`` row loads in flight regardless of bag boundaries. Triton pays an
   offsets -> indices -> rows dependent chain plus setup per bag, which
   dominates at L ~ 1-17.
2. Fewer instructions per row. Triton's forward is issue-bound (~80 warp
   instructions per gathered row at ~76% issue utilisation, DRAM at 29%).
   Mixed-precision ``add.rn.f32.f16`` folds the fp16 -> fp32 widening into the
   accumulate, branch-guarded row slots replace mask multiplies, and int32
   indices are shuffled as 32-bit values.
3. Multi-warp CTAs. Triton's num_warps=1 programs hit Blackwell's 32 CTA/SM
   limit (50% occupancy); here occupancy is bounded by registers.

Constraints (host checks): forward needs even D and table offsets (16B lanes
when both are multiples of 8), 4-element aligned output rows (D % 4 == 0 under
VBE), fp16/bf16 weights, SUM pooling. The short-run backward needs D % 8 == 0,
D <= 256, no stochastic rounding (unweighted and weighted, with or without VBE);
the long-run tier D % 4 == 0 (D % 8 == 0 above 128), unweighted only; tiling
also needs unweighted, non-VBE and T * tiles per feature <= 256.
"""

import os
from functools import partial
from typing import Dict, Tuple

import cuda.bindings.driver as cuda  # pyre-fixme[21]: Missing module stub
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op, T

VEC = 8  # elements per lane chunk (16B of fp16 weights)
WARP = 32
FULL = 0xFFFFFFFF  # full-warp shuffle/vote mask

_TORCH_TO_CUTE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.uint8: cutlass.Uint8,
}

# Per-feature metadata columns (int64 [T, META_COLS]).
META_TABLE_OFFSET = 0  # element offset of the feature's table in the weights
META_D = 1  # embedding dim
META_OUT_OFFSET = 2  # column offset of the feature in the pooled output
META_HASH_CUMSUM = 3  # linear-index base of the feature (hash_size_cumsum[t])
META_ROWS_CUMSUM = 4  # momentum base of the feature's table (rows_cumsum[table])
META_NUM_ROWS = 5  # rows in the feature's table (bounds check)
META_RPW = 6  # rows per warp instruction: max(32 // (D / 8), 1)
META_SLOT_MAGIC = 7  # ceil(2^16 / (D / 8)): lane // (D / 8) == (lane * magic) >> 16
META_FEATURE = 8  # global feature index (forward launches take feature subsets)
META_COLS = 12
ROW_MAX = 256  # widest row (elements) a cross-tile partial sum holds


def _ptr(t: torch.Tensor, align: int = 16) -> cute.Pointer:
    return make_ptr(
        _TORCH_TO_CUTE[t.dtype],
        t.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=align,
    )


_pinned: Dict[int, torch.Tensor] = {}


def _tensor_key(t: torch.Tensor) -> int:
    """t's address, as a key for host metadata derived from t (Triton passes the
    same module tensors on every call, so the lookup is a dict hit). t is kept
    alive with it: a freed address reused by a different tensor would otherwise
    hit a stale entry."""
    _pinned[t.data_ptr()] = t
    return t.data_ptr()


@cute.jit
def _addr(base: cute.Pointer, off: Int64, dtype: cutlass.Constexpr) -> Int64:
    """Byte address of base[off]. Integer math on purpose: the equivalent
    ``(base + off).toint()`` made the grouped forward 15% slower (shard A 13.6 vs
    11.8 ms, shard B 3.77 vs 3.60 ms on GB300)."""
    # pyre-fixme[16]: Incomplete cutlass stubs (Pointer.toint, Constexpr.width)
    return base.toint() + off * (dtype.width // 8)


@cute.jit
def _loadn(
    addr: Int64, frag: cute.Tensor, dtype: cutlass.Constexpr, n: cutlass.Constexpr
):
    """frag[:n] = *addr with the widest vector loads n allows (<= 128 bits).

    The pointer is rebuilt from the integer address on purpose:
    ``ptr + cute.assume(off, divby=...)`` loses the alignment (it comes out
    element-aligned) and silently splits the copy into scalar loads.
    """
    # pyre-fixme[16]: Incomplete cutlass stubs (Constexpr.width)
    bits = min(128, n * dtype.width)
    epa = bits // dtype.width  # pyre-fixme[16]
    tiled = cute.make_layout((epa, n // epa))
    src = cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=bits // 8),
        tiled,
    )
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=bits
    )
    cute.copy(atom, src, cute.make_tensor(frag.iterator, tiled))


@cute.jit
def _storen(
    addr: Int64, frag: cute.Tensor, dtype: cutlass.Constexpr, n: cutlass.Constexpr
):
    # pyre-fixme[16]: Incomplete cutlass stubs (Constexpr.width)
    bits = min(128, n * dtype.width)
    epa = bits // dtype.width  # pyre-fixme[16]
    tiled = cute.make_layout((epa, n // epa))
    dst = cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=bits // 8),
        tiled,
    )
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=bits
    )
    cute.copy(atom, cute.make_tensor(frag.iterator, tiled), dst)


@dsl_user_op
def _acc_add_half8(
    acc: cute.Tensor, frag: cute.Tensor, ty: str, n: int = 8, *, loc=None, ip=None
):
    """acc[i] += float(frag[i]) for n (even) fp16/bf16 values, one instruction each.

    Blackwell's mixed-precision ``add.rn.f32.{f16,bf16}`` folds the widening
    into the add; the generic ``.to(Float32)`` + add costs two per element and
    the gather loop is issue-bound.
    """
    w = n // 2
    words = cute.recast_tensor(frag, Uint32)
    ins = [Uint32(words[i]).ir_value(loc=loc, ip=ip) for i in range(w)]
    ins += [Float32(acc[i]).ir_value(loc=loc, ip=ip) for i in range(n)]
    asm = f"{{\n.reg .b16 h<{n}>;\n"
    asm += "".join(f"mov.b32 {{h{2 * i}, h{2 * i + 1}}}, ${n + i};\n" for i in range(w))
    asm += "".join(f"add.rn.f32.{ty} ${i}, h{i}, ${n + w + i};\n" for i in range(n))
    asm += "}\n"
    res = llvm.inline_asm(
        # pyre-fixme[16]: Missing from llvm stubs
        llvm.StructType.get_literal([T.f32()] * n),
        ins,
        asm,
        ",".join(["=f"] * n + ["r"] * w + ["f"] * n),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    for i in range(n):
        acc[i] = Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip))


@cute.jit
def _copy4(
    addr: Int64,
    frag: cute.Tensor,
    offset: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    store: cutlass.Constexpr = False,  # pyre-fixme[9]: Incomplete Constexpr stub
):
    """frag[offset:offset+4] <-> *addr (4 contiguous elements, one vector access)."""
    bits = 4 * dtype.width  # pyre-fixme[16]: Incomplete cutlass stubs
    g = cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=bits // 8),
        cute.make_layout(4),
    )
    # pyre-fixme[58]: Incomplete Pointer/Constexpr operator stubs
    r = cute.make_tensor(frag.iterator + offset, cute.make_layout(4))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=bits
    )
    if const_expr(store):
        cute.copy(atom, r, g)
    else:
        cute.copy(atom, g, r)


@cute.jit
def _reduce_slots(acc: cute.Tensor, lane: Int32, nch: Int32, rpw: Int32):
    """Sum the per-slot partial rows into slot 0 (slot s lives at lanes s*nch..).
    Shuffles read an untouched copy: updating in place double-counts once rpw > 2."""
    part = cute.make_rmem_tensor((cute.size(acc),), Float32)
    part.store(acc.load())
    for s in cutlass.range(1, rpw):
        for i in cutlass.range_constexpr(cute.size(acc)):
            acc[i] = acc[i] + cute.arch.shuffle_sync(part[i], lane + s * nch)


class TbeForward:
    """Unweighted SUM-pooled TBE forward.

    Grid (ceil(B / (warps * 32)), T). Each warp owns 32 consecutive bags of
    feature t. Their offsets come in with one coalesced load, and since the
    bags' indices are contiguous the warp walks them as one flat stream: 32
    indices per coalesced load, ``unroll`` row loads in flight regardless of
    where bags start and end, flushing the accumulator to the output at each
    bag boundary. Short bags therefore cost neither a dependent
    offsets->indices->rows latency chain nor per-bag setup.

    Features whose rows are narrower than a warp (D < 256 for fp16) pack
    several rows per warp instruction and take a per-bag path instead.
    """

    def __init__(
        self,
        w_dtype,
        idx_dtype,
        bounds_check: bool,
        unroll: int,
        warps: int,
        weighted: bool = False,
        vec: int = VEC,
        narrow: str = "grouped",
        vbe: bool = False,
    ):
        # Variable batch size per feature: bag b of feature t is b_t =
        # B_offsets[t] + b and its output starts at row_output_offsets[b_t].
        self.vbe = vbe
        self.weighted = weighted  # per-sample weights (fp32, one per index)
        # Elements per lane per row (8 = 16B of fp16). vec < 8 only for D not a
        # multiple of 8; those rows take the flat stream with narrower lanes.
        self.vec = vec
        # Rows narrower than a warp (rpw > 1): "grouped" = per-lane-group flat
        # streams; "packed" = rpw rows per instruction, one bag at a time. Neither
        # wins on every shard, so tbe_forward picks per shape on first use.
        self.narrow = narrow if vec == VEC else "grouped"
        self.w_dtype = w_dtype
        self.half_ty = {cutlass.Float16: "f16", cutlass.BFloat16: "bf16"}.get(w_dtype)
        self.idx_dtype = idx_dtype
        self.bounds_check = bounds_check
        self.unroll = unroll
        self.warps = warps

    @cute.jit
    def __call__(
        self,
        out_ptr: cute.Pointer,
        idx_ptr: cute.Pointer,
        off_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        psw_ptr: cute.Pointer,
        warn_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        total_D: Int32,
        B: Int32,
        T: Int32,
        stream: cuda.CUstream,
    ):
        mOff = cute.make_tensor(off_ptr, cute.make_layout(1 << 30))
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        self.forward_kernel(
            out_ptr,
            idx_ptr,
            mOff,
            w_ptr,
            mMeta,
            psw_ptr,
            warn_ptr,
            rofs_ptr,
            bofs_ptr,
            total_D,
            B,
        ).launch(
            grid=[cute.ceil_div(B, self.warps * WARP), T, 1],
            block=[WARP * self.warps, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _accumulate(self, acc, frag, w):
        if const_expr(self.weighted):
            acc.store(acc.load() + frag.load().to(Float32) * w)
        elif const_expr(self.half_ty is None):  # fp32 rows
            acc.store(acc.load() + frag.load())
        else:
            _acc_add_half8(acc, frag, self.half_ty, self.vec)

    @cute.jit
    def _load_psw(self, mPsw, pos, end, lane):
        """Lane l gets per_sample_weights[pos + l] (1.0 when unweighted)."""
        my_w = Float32(1.0)
        if const_expr(self.weighted):
            if pos + lane < end:
                my_w = Float32(mPsw[pos + lane])
        return my_w

    @cute.jit
    def _load_indices(
        self, mIdx, pos, end, lane, num_rows, warn_ptr, first_pass, gmask
    ):
        """Group lane l gets index[pos + l] (pos + l < end); out-of-range ones repaired."""
        my_idx = self.idx_dtype(0)
        if pos + lane < end:
            my_idx = self.idx_dtype(mIdx[pos + lane])
        if const_expr(self.bounds_check):
            # Triton's fused-check contract: repair out-of-range indices to 0
            # in place and count them. -1 (pruned) is left alone, skipped later.
            bad = (
                (pos + lane < end)
                & (my_idx != -1)
                & ((my_idx < 0) | (Int64(my_idx) >= num_rows))
            )
            nbad = cute.arch.popc(cute.arch.vote_ballot_sync(bad, mask=gmask))
            if nbad > 0:
                if bad:
                    my_idx = self.idx_dtype(0)
                    if first_pass:
                        mIdx[pos + lane] = my_idx
                if (lane == 0) & first_pass:
                    cute.arch.atomic_add(warn_ptr, Int64(nbad))
        return my_idx

    @cute.kernel
    def forward_kernel(
        self,
        out_ptr: cute.Pointer,
        idx_ptr: cute.Pointer,
        mOff: cute.Tensor,
        w_ptr: cute.Pointer,
        mMeta: cute.Tensor,
        psw_ptr: cute.Pointer,
        warn_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        total_D: Int32,
        B: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, t, _ = cute.arch.block_idx()
        lane = tidx % WARP
        b0 = (bx * self.warps + tidx // WARP) * WARP
        gt = Int32(mMeta[t, META_FEATURE])
        B_t = B  # this feature's batch size (B = max over features under VBE)
        b_start = gt * B
        if const_expr(self.vbe):
            mBofs = cute.make_tensor(bofs_ptr, cute.make_layout(1 << 20))
            b_start = Int32(mBofs[gt])
            B_t = Int32(mBofs[gt + 1]) - b_start
        if b0 < B_t:
            nb = cutlass.min(B_t - b0, WARP)
            bt0 = b_start + b0
            tbl_off = Int64(mMeta[t, META_TABLE_OFFSET])
            D = Int32(mMeta[t, META_D])
            out_off = Int64(mMeta[t, META_OUT_OFFSET])
            num_rows = Int64(mMeta[t, META_NUM_ROWS])
            rpw = Int32(1)
            if const_expr(self.vec == VEC):
                rpw = Int32(mMeta[t, META_RPW])
            magic = Int32(mMeta[t, META_SLOT_MAGIC])
            nch = D // self.vec  # vec-element chunks per row
            slot = (lane * magic) >> 16  # lane // nch without a divide
            c_in = lane - slot * nch
            row_bytes = Int64(D) * (self.w_dtype.width // 8)
            # Bag k of this warp spans [start(k), start(k+1)); lane k holds start(k)
            # relative to the warp's first index (Int32 positions from here on).
            g_start = Int64(0)
            if lane < nb:
                g_start = Int64(mOff[bt0 + lane])
            g0 = cute.arch.shuffle_sync(g_start, 0)
            s_end = Int32(Int64(mOff[bt0 + nb]) - g0)
            my_start = Int32(g_start - g0)
            mIdx = cute.make_tensor(idx_ptr + g0, cute.make_layout(s_end + WARP))
            mPsw = cute.make_tensor(psw_ptr + g0, cute.make_layout(s_end + WARP))
            out_row0 = Int64(b0) * total_D + out_off
            # Lane k holds bag k's output element offset (VBE: from the metadata).
            my_out = Int64(0)
            if const_expr(self.vbe):
                if lane < nb:
                    my_out = Int64(
                        cute.make_tensor(rofs_ptr, cute.make_layout(1 << 30))[
                            bt0 + lane
                        ]
                    )

            if rpw == 1:
                self._warp_stream(
                    out_ptr,
                    w_ptr,
                    mIdx,
                    mPsw,
                    warn_ptr,
                    tbl_off,
                    out_row0,
                    total_D,
                    row_bytes,
                    num_rows,
                    nch,
                    lane,
                    nb,
                    s_end,
                    my_start,
                    my_out,
                )
            elif const_expr(self.narrow == "packed"):
                self._packed_bags(
                    out_ptr,
                    w_ptr,
                    mIdx,
                    mPsw,
                    warn_ptr,
                    tbl_off,
                    out_row0,
                    total_D,
                    row_bytes,
                    num_rows,
                    nch,
                    lane,
                    slot,
                    c_in,
                    rpw,
                    nb,
                    s_end,
                    my_start,
                    my_out,
                )
            else:
                # Lane groups: a group is the lanes covering one row (the whole warp
                # when D/vec >= 32, else rpw groups of nch lanes). Group g owns a
                # contiguous run of the warp's bags and walks their indices as one flat
                # stream. Groups diverge, so their shuffles/votes use the group mask.
                grpw = cutlass.select_(rpw == 1, Int32(WARP), nch)
                grp = cutlass.select_(rpw == 1, Int32(0), slot)
                glane = lane - grp * grpw
                gbase = grp * grpw
                gmask = cutlass.select_(
                    grpw == WARP,
                    cutlass.Uint32(0xFFFFFFFF),
                    ((cutlass.Uint32(1) << cutlass.Uint32(grpw)) - cutlass.Uint32(1))
                    << cutlass.Uint32(gbase),
                )
                bpg = (nb + rpw - 1) // rpw
                kb0 = cutlass.min(grp * bpg, nb)
                kb1 = cutlass.min(kb0 + bpg, nb)
                # Converged here: hand each group its bag starts (group lane i: bags
                # kb0 + i and kb0 + grpw + i) and its stream bounds.
                sa = cute.arch.shuffle_sync(
                    my_start, cutlass.min(kb0 + glane, WARP - 1)
                )
                sb = cute.arch.shuffle_sync(
                    my_start, cutlass.min(kb0 + grpw + glane, WARP - 1)
                )
                oa = Int64(0)
                ob = Int64(0)
                if const_expr(self.vbe):
                    oa = cute.arch.shuffle_sync(
                        my_out, cutlass.min(kb0 + glane, WARP - 1)
                    )
                    ob = cute.arch.shuffle_sync(
                        my_out, cutlass.min(kb0 + grpw + glane, WARP - 1)
                    )
                gs0 = cutlass.select_(
                    kb0 < nb,
                    cute.arch.shuffle_sync(my_start, cutlass.min(kb0, WARP - 1)),
                    s_end,
                )
                gs1 = cutlass.select_(
                    kb1 < nb,
                    cute.arch.shuffle_sync(my_start, cutlass.min(kb1, WARP - 1)),
                    s_end,
                )
                if (grp < rpw) & (kb0 < kb1):
                    self._group_stream(
                        w_ptr,
                        out_ptr,
                        mIdx,
                        mPsw,
                        warn_ptr,
                        tbl_off,
                        out_row0,
                        total_D,
                        row_bytes,
                        num_rows,
                        nch,
                        glane,
                        gbase,
                        grpw,
                        gmask,
                        kb0,
                        kb1,
                        gs0,
                        gs1,
                        sa,
                        sb,
                        oa,
                        ob,
                    )

    @cute.jit
    def _warp_stream(
        self,
        out_ptr,
        w_ptr,
        mIdx,
        mPsw,
        warn_ptr,
        tbl_off,
        out_row0,
        total_D,
        row_bytes,
        num_rows,
        nch,
        lane,
        nb,
        s_end,
        my_start,
        my_out,
    ):
        """Rows of >= 32 vec chunks (one row per warp instruction): the warp walks
        its bags' indices as one flat stream, storing each bag as it ends."""
        s0 = Int32(0)
        for cbase in cutlass.range(0, nch, WARP):
            chunk = cbase + lane
            lane_ok = chunk < nch
            col = Int64(cutlass.select_(lane_ok, chunk * self.vec, 0))
            lane_base = _addr(w_ptr, tbl_off + col, self.w_dtype)
            out_base = _addr(out_ptr, out_row0 + col, Float32)
            flush = (out_ptr, out_base, my_out, col, total_D)
            acc = cute.make_rmem_tensor((self.vec,), Float32)
            acc.fill(0.0)
            frags = [
                cute.make_rmem_tensor((self.vec,), self.w_dtype)
                for _ in range(self.unroll)
            ]
            # Flush bags that end before position p (empty bags emit zeros).
            cur = Int32(0)
            cur_end = cutlass.select_(
                nb > 1, cute.arch.shuffle_sync(my_start, 1), s_end
            )
            cur, cur_end = self._flush_bags(
                acc, s0, cur, cur_end, nb, s_end, my_start, lane_ok, flush
            )
            for p0 in cutlass.range(s0, s_end, WARP):
                n = cutlass.min(s_end - p0, WARP)
                my_idx = self._load_indices(
                    mIdx, p0, s_end, lane, num_rows, warn_ptr, cbase == 0, FULL
                )
                my_w = self._load_psw(mPsw, p0, s_end, lane)
                for j in cutlass.range(0, n, self.unroll):
                    oks = []
                    ws = []
                    for u in cutlass.range_constexpr(self.unroll):
                        row = cute.arch.shuffle_sync(my_idx, j + u)
                        ws.append(
                            cute.arch.shuffle_sync(my_w, j + u)
                            if const_expr(self.weighted)
                            else my_w
                        )
                        ok = (j + u < n) & (row != -1)
                        if ok:
                            _loadn(
                                lane_base + Int64(row) * row_bytes,
                                frags[u],
                                self.w_dtype,
                                self.vec,
                            )
                        oks.append(ok)
                    for u in cutlass.range_constexpr(self.unroll):
                        if oks[u]:
                            self._accumulate(acc, frags[u], ws[u])
                        if j + u < n:
                            cur, cur_end = self._flush_bags(
                                acc,
                                p0 + j + u + 1,
                                cur,
                                cur_end,
                                nb,
                                s_end,
                                my_start,
                                lane_ok,
                                flush,
                            )

    @cute.jit
    def _packed_bags(
        self,
        out_ptr,
        w_ptr,
        mIdx,
        mPsw,
        warn_ptr,
        tbl_off,
        out_row0,
        total_D,
        row_bytes,
        num_rows,
        nch,
        lane,
        slot,
        c_in,
        rpw,
        nb,
        s_end,
        my_start,
        my_out,
    ):
        """Narrow rows, packed: rpw rows per warp instruction, one bag at a time."""
        col = Int64(cutlass.select_(slot < rpw, c_in * VEC, 0))
        lane_base = _addr(w_ptr, tbl_off + col, self.w_dtype)
        G = rpw * self.unroll
        frags = [
            cute.make_rmem_tensor((VEC,), self.w_dtype) for _ in range(self.unroll)
        ]
        for k in cutlass.range(0, nb):
            start = cute.arch.shuffle_sync(my_start, k)
            nxt = cute.arch.shuffle_sync(my_start, k + 1)
            end = cutlass.select_(k + 1 < nb, nxt, s_end)
            acc = cute.make_rmem_tensor((VEC,), Float32)
            acc.fill(0.0)
            for p0 in cutlass.range(start, end, WARP):
                n = cutlass.min(end - p0, WARP)
                my_idx = self._load_indices(
                    mIdx, p0, end, lane, num_rows, warn_ptr, True, FULL
                )
                my_w = self._load_psw(mPsw, p0, end, lane)
                for j in cutlass.range(0, n, G):
                    oks = []
                    ws = []
                    for u in cutlass.range_constexpr(self.unroll):
                        r = j + u * rpw + slot
                        row = cute.arch.shuffle_sync(my_idx, r)
                        ws.append(
                            cute.arch.shuffle_sync(my_w, r)
                            if const_expr(self.weighted)
                            else my_w
                        )
                        ok = (r < n) & (row != -1)
                        if ok:
                            _loadn(
                                lane_base + Int64(row) * row_bytes,
                                frags[u],
                                self.w_dtype,
                                VEC,
                            )
                        oks.append(ok)
                    for u in cutlass.range_constexpr(self.unroll):
                        if oks[u]:
                            self._accumulate(acc, frags[u], ws[u])
            _reduce_slots(acc, lane, nch, rpw)
            o_addr = self._out_addr(
                out_ptr,
                _addr(out_ptr, out_row0 + col, Float32),
                my_out,
                k,
                col,
                total_D,
            )
            if slot == 0:
                _storen(o_addr, acc, Float32, VEC)

    @cute.jit
    def _flush_bags(self, acc, p, cur, cur_end, nb, s_end, my_start, lane_ok, flush):
        """Whole-warp stream: store bag `cur` and every later bag that ends at
        stream position p (empty bags store zeros), resetting acc; returns the
        next (cur, cur_end). flush = (out_ptr, out_base, my_out, col, total_D)."""
        out_ptr, out_base, my_out, col, total_D = flush
        while (cur < nb) & (cur_end == p):
            o_addr = self._out_addr(out_ptr, out_base, my_out, cur, col, total_D)
            if lane_ok:
                _storen(o_addr, acc, Float32, self.vec)
            acc.fill(0.0)
            cur = cur + 1
            nxt = cute.arch.shuffle_sync(my_start, cur + 1)
            cur_end = cutlass.select_(cur + 1 < nb, nxt, s_end)
        return cur, cur_end

    @cute.jit
    def _flush_group_bags(self, acc, p, cur, cur_end, lane_ok, flush):
        """_flush_bags for a lane group's bags kb0..kb1. flush = (out_ptr,
        out_base, col, total_D, kb0, kb1, gs1, sa, sb, oa, ob, gbase, grpw,
        gmask)."""
        out_ptr, out_base, col, total_D, kb0, kb1, gs1, sa, sb, oa, ob = flush[:11]
        gbase, grpw, gmask = flush[11:]
        while (cur < kb1) & (cur_end == p):
            o_addr = self._group_out_addr(
                out_ptr, out_base, cur, col, total_D, kb0, oa, ob, gbase, grpw, gmask
            )
            if lane_ok:
                _storen(o_addr, acc, Float32, self.vec)
            acc.fill(0.0)
            cur = cur + 1
            cur_end = self._bag_start(
                cur + 1, kb0, kb1, gs1, sa, sb, gbase, grpw, gmask
            )
        return cur, cur_end

    @cute.jit
    def _out_addr(self, out_ptr, out_base, my_out, k, col, total_D):
        """Byte address of this lane's part of bag k's output row (full warp)."""
        if const_expr(self.vbe):
            return _addr(out_ptr, cute.arch.shuffle_sync(my_out, k) + col, Float32)
        return out_base + Int64(k) * total_D * 4

    @cute.jit
    def _group_stream(
        self,
        w_ptr,
        out_ptr,
        mIdx,
        mPsw,
        warn_ptr,
        tbl_off,
        out_row0,
        total_D,
        row_bytes,
        num_rows,
        nch,
        glane,
        gbase,
        grpw,
        gmask,
        kb0,
        kb1,
        gs0,
        gs1,
        sa,
        sb,
        oa,
        ob,
    ):
        for cbase in cutlass.range(
            0, nch, grpw
        ):  # > 1 pass only for rows wider than 32 lanes
            chunk = cbase + glane
            lane_ok = chunk < nch
            col = Int64(cutlass.select_(lane_ok, chunk * self.vec, 0))
            lane_base = _addr(w_ptr, tbl_off + col, self.w_dtype)
            out_base = _addr(out_ptr, out_row0 + col, Float32)
            flush = (out_ptr, out_base, col, total_D, kb0, kb1, gs1, sa, sb, oa, ob)
            flush += (gbase, grpw, gmask)
            acc = cute.make_rmem_tensor((self.vec,), Float32)
            acc.fill(0.0)
            frags = [
                cute.make_rmem_tensor((self.vec,), self.w_dtype)
                for _ in range(self.unroll)
            ]
            # Flush bags that end at position gs0 (empty bags emit zeros).
            cur = kb0
            cur_end = self._bag_start(
                kb0 + 1, kb0, kb1, gs1, sa, sb, gbase, grpw, gmask
            )
            cur, cur_end = self._flush_group_bags(
                acc, gs0, cur, cur_end, lane_ok, flush
            )
            for p0 in cutlass.range(gs0, gs1, grpw):
                n = cutlass.min(gs1 - p0, grpw)
                my_idx = self._load_indices(
                    mIdx, p0, gs1, glane, num_rows, warn_ptr, cbase == 0, gmask
                )
                my_w = self._load_psw(mPsw, p0, gs1, glane)
                for j in cutlass.range(0, n, self.unroll):
                    oks = []
                    ws = []
                    for u in cutlass.range_constexpr(self.unroll):
                        src = gbase + cutlass.min(j + u, grpw - 1)
                        row = cute.arch.shuffle_sync(my_idx, src, mask=gmask)
                        ws.append(
                            cute.arch.shuffle_sync(my_w, src, mask=gmask)
                            if const_expr(self.weighted)
                            else my_w
                        )
                        ok = (j + u < n) & (row != -1)
                        if ok:
                            _loadn(
                                lane_base + Int64(row) * row_bytes,
                                frags[u],
                                self.w_dtype,
                                self.vec,
                            )
                        oks.append(ok)
                    for u in cutlass.range_constexpr(self.unroll):
                        if oks[u]:
                            self._accumulate(acc, frags[u], ws[u])
                        if j + u < n:
                            cur, cur_end = self._flush_group_bags(
                                acc, p0 + j + u + 1, cur, cur_end, lane_ok, flush
                            )

    @cute.jit
    def _group_val(self, k, kb0, va, vb, gbase, grpw, gmask):
        """Group bag k's value, spread over group lanes as va (bags kb0..) / vb."""
        j = k - kb0
        a = cute.arch.shuffle_sync(va, gbase + cutlass.min(j, grpw - 1), mask=gmask)
        b = cute.arch.shuffle_sync(vb, gbase + cutlass.max(j - grpw, 0), mask=gmask)
        return cutlass.select_(j < grpw, a, b)

    @cute.jit
    def _bag_start(self, k, kb0, kb1, gs1, sa, sb, gbase, grpw, gmask):
        """Stream position where group bag k starts (gs1 past the group's last bag)."""
        return cutlass.select_(
            k >= kb1, gs1, self._group_val(k, kb0, sa, sb, gbase, grpw, gmask)
        )

    @cute.jit
    def _group_out_addr(
        self, out_ptr, out_base, k, col, total_D, kb0, oa, ob, gbase, grpw, gmask
    ):
        if const_expr(self.vbe):
            o = self._group_val(k, kb0, oa, ob, gbase, grpw, gmask)
            return _addr(out_ptr, o + col, Float32)
        return out_base + Int64(k) * total_D * 4


class TbeBackwardShortRun:
    """Unweighted short-run backward + exact row-wise Adagrad. One warp per run.

    Warps pull batches of ``rpb`` runs from an atomic counter (work stealing:
    run lengths vary 1..255, so a static split leaves stragglers), with
    rpb = clamp(num_runs / num_warps, 1, 32) so small buckets still spread
    over every warp. Lane l prefetches run (batch + l)'s metadata and its first
    info in one load each; a one-lookup run then needs no further dependent
    load before its gradient row, weight row and momentum are all in flight.
    """

    def __init__(self, w_dtype, unroll: int, warps: int, small_T: bool):
        self.w_dtype = w_dtype
        self.unroll = unroll
        self.warps = warps
        self.small_T = small_T
        self.min_blocks = max(
            int(os.environ.get("CUTE_TBE_BWD_MIN_WARPS_PER_SM", 32)) // warps, 1
        )

    @cute.jit
    def __call__(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_runs_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        T: Int32,
        total_D: Int32,
        max_runs: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        lr: Float32,
        eps: Float32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        self.kernel(
            dout_ptr,
            w_ptr,
            mom_ptr,
            infos_ptr,
            lin_run_ptr,
            cum_ptr,
            run_ids_ptr,
            num_runs_ptr,
            counter_ptr,
            mMeta,
            T,
            total_D,
            max_runs,
            info_B_num_bits,
            info_B_mask,
            lr,
            eps,
        ).launch(
            grid=[grid, 1, 1],
            block=[WARP * self.warps, 1, 1],
            smem=0,
            # Register cap via launch bounds: the kernel is latency-bound.
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_runs_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        mMeta: cute.Tensor,
        T: Int32,
        total_D: Int32,
        max_runs: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        lr: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        nwarps = gdim * self.warps

        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mLin = cute.make_tensor(lin_run_ptr, cute.make_layout(max_runs + 1))
        mCum = cute.make_tensor(cum_ptr, cute.make_layout(max_runs + 1))
        mRunIds = cute.make_tensor(run_ids_ptr, cute.make_layout(max_runs + 1))
        num_runs = Int32(cute.make_tensor(num_runs_ptr, cute.make_layout(1))[0])
        rpb = cutlass.min(cutlass.max(cute.ceil_div(num_runs, nwarps), 1), WARP)

        # Feature t's output column offset lives in lane t when T <= 32.
        lane_out_off = Int32(0)
        if const_expr(self.small_T):
            if lane < T:
                lane_out_off = Int32(mMeta[lane, META_OUT_OFFSET])
        bits = cutlass.Uint32(info_B_num_bits)
        bmask = cutlass.Uint32(info_B_mask)
        dout_base = dout_ptr.toint()  # pyre-fixme[16]: Incomplete cutlass stubs

        batch = Int32(0)
        if lane == 0:
            batch = Int32(cute.arch.atomic_add(counter_ptr, rpb))
        batch = cute.arch.shuffle_sync(batch, 0)
        while batch < num_runs:
            nb = cutlass.min(num_runs - batch, rpb)
            p_seg_start = Int32(0)
            p_seg_len = Int32(0)
            p_info0 = Int32(0)
            p_row = Int64(0)
            if lane < nb:
                run_id = Int32(mRunIds[batch + lane])
                p_seg_start = Int32(mCum[run_id])
                p_seg_len = Int32(mCum[run_id + 1]) - p_seg_start
                p_info0 = Int32(mInfos[p_seg_start])
                p_t = Int32(cutlass.Uint32(p_info0) >> bits)
                p_row = Int64(mLin[run_id]) - Int64(mMeta[p_t, META_HASH_CUMSUM])

            # Grab the next batch now so the atomic's latency hides behind this one.
            nxt = Int32(0)
            if lane == 0:
                nxt = Int32(cute.arch.atomic_add(counter_ptr, rpb))

            for r in cutlass.range(0, nb):
                seg_start = cute.arch.shuffle_sync(p_seg_start, r)
                L = cute.arch.shuffle_sync(p_seg_len, r)
                info0 = cute.arch.shuffle_sync(p_info0, r)
                row = cute.arch.shuffle_sync(p_row, r)
                t = Int32(cutlass.Uint32(info0) >> bits)
                D = Int32(mMeta[t, META_D])
                tbl_off = Int64(mMeta[t, META_TABLE_OFFSET])
                mom_idx = Int64(mMeta[t, META_ROWS_CUMSUM]) + row
                rpw = Int32(mMeta[t, META_RPW])
                nch = D // VEC
                slot = (lane * Int32(mMeta[t, META_SLOT_MAGIC])) >> 16
                # Lane c owns elements [4c, 4c+4) and [D/2 + 4c, D/2 + 4c + 4), so
                # each gradient-row load instruction covers 512 contiguous bytes.
                col = Int64((lane - slot * nch) * (VEC // 2))
                half = Int64(D // 2)
                w_addr = _addr(w_ptr, tbl_off + row * D + col, self.w_dtype)
                w_addr_hi = w_addr + half * (self.w_dtype.width // 8)
                lane_dout = dout_base + col * 4
                dout_hi = half * 4

                # Optimizer-state loads go first; they overlap the gather.
                w_row = cute.make_rmem_tensor((VEC,), self.w_dtype)
                mMom = cute.make_tensor(mom_ptr + mom_idx, cute.make_layout(1))
                _copy4(w_addr, w_row, 0, self.w_dtype)
                _copy4(w_addr_hi, w_row, 4, self.w_dtype)
                mom = Float32(mMom[0])

                acc = cute.make_rmem_tensor((VEC,), Float32)
                acc.fill(0.0)
                frags = [
                    cute.make_rmem_tensor((VEC,), Float32) for _ in range(self.unroll)
                ]
                G = rpw * self.unroll
                for j0 in cutlass.range(0, L, WARP):
                    n = cutlass.min(L - j0, WARP)
                    my_info = info0  # one-lookup runs reuse the prefetched info
                    if L > 1:
                        if lane < n:
                            my_info = Int32(mInfos[seg_start + j0 + lane])
                    for j in cutlass.range(0, n, G):
                        oks = []
                        for u in cutlass.range_constexpr(self.unroll):
                            e = j + u * rpw + slot
                            ui = cutlass.Uint32(cute.arch.shuffle_sync(my_info, e))
                            eb = Int64(Int32(ui & bmask))
                            et = Int32(ui >> bits)
                            if const_expr(self.small_T):
                                oo = cute.arch.shuffle_sync(lane_out_off, et)
                            else:
                                oo = Int32(mMeta[et, META_OUT_OFFSET])
                            d_addr = lane_dout + (eb * total_D + Int64(oo)) * 4
                            ok = (e < n) & (slot < rpw)
                            if ok:
                                _copy4(d_addr, frags[u], 0, Float32)
                                _copy4(d_addr + dout_hi, frags[u], 4, Float32)
                            oks.append(ok)
                        for u in cutlass.range_constexpr(self.unroll):
                            if oks[u]:
                                acc.store(acc.load() + frags[u].load())
                _reduce_slots(acc, lane, nch, rpw)
                self._adagrad(
                    acc, w_row, mom, mMom, w_addr, w_addr_hi, slot, lane, D, lr, eps
                )
            batch = cute.arch.shuffle_sync(nxt, 0)

    @cute.jit
    def _adagrad(
        self, acc, w_row, mom, mMom, w_addr, w_addr_hi, slot, lane, D, lr, eps
    ):
        # Row-wise Adagrad: mean of grad^2 over the row (slot-0 lanes hold it).
        sq = Float32(0.0)
        for i in cutlass.range_constexpr(VEC):
            sq = sq + acc[i] * acc[i]
        sq = cutlass.select_(slot == 0, sq, Float32(0.0))
        for sh in cutlass.range_constexpr(5):
            sq = sq + cute.arch.shuffle_sync_bfly(sq, 1 << sh)
        mom_new = mom + sq / Float32(D)
        alr = lr / (cute.math.sqrt(mom_new) + eps)
        if lane == 0:
            mMom[0] = mom_new
        if slot == 0:
            upd = cute.make_rmem_tensor((VEC,), self.w_dtype)
            upd.store((w_row.load().to(Float32) - acc.load() * alr).to(self.w_dtype))
            _copy4(w_addr, upd, 0, self.w_dtype, True)
            _copy4(w_addr_hi, upd, 4, self.w_dtype, True)


class TbeBackwardPacked:
    """Row-order short-run backward + exact row-wise Adagrad, several runs per
    warp instruction.

    Short runs are mostly 1-3 lookups (shard C: 30M runs, mean 1.76 lookups, 58%
    single), so per-run work dominates: metadata, the weight row, momentum and
    the Adagrad reduction. A warp takes up to 32 runs off an atomic queue, and
    lane l prefetches run l's metadata and first dout offset in one round of loads.
    The warp then works through them in groups of G lanes, one run per group
    (G = the launch's block size / 8: 16 lanes for D <= 128, two runs per
    instruction, where a warp per run leaves 18 of 32 lanes idle at D=112).
    Lane c of a group owns elements [4c, 4c+4) and [D/2 + 4c, D/2 + 4c + 4).
    Weighted: each lookup's dout row is scaled by its per-sample weight (sorted
    like the lookups) before it is summed, as in Triton's weighted kernel.
    """

    def __init__(
        self, w_dtype, G: int, warps: int, unroll: int, weighted: bool, vbe: bool
    ):
        self.weighted = weighted
        self.vbe = vbe  # dout is the flat VBE output (see _dout_row)
        self.w_dtype = w_dtype
        self.G = G
        self.unroll = unroll
        self.log2_G = G.bit_length() - 1  # (a power of 2)
        self.warps = warps
        self.min_blocks = max(
            int(os.environ.get("CUTE_TBE_BWD_MIN_WARPS_PER_SM", 32)) // warps, 1
        )

    @cute.jit
    def __call__(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_runs_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        psw_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        T: Int32,
        total_D: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        lr: Float32,
        eps: Float32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        self.kernel(
            dout_ptr,
            w_ptr,
            mom_ptr,
            infos_ptr,
            lin_run_ptr,
            cum_ptr,
            run_ids_ptr,
            num_runs_ptr,
            counter_ptr,
            psw_ptr,
            rofs_ptr,
            bofs_ptr,
            mMeta,
            total_D,
            info_B_num_bits,
            info_B_mask,
            lr,
            eps,
        ).launch(
            grid=[grid, 1, 1],
            block=[WARP * self.warps, 1, 1],
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_runs_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        psw_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        mMeta: cute.Tensor,
        total_D: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        lr: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % WARP
        g = lane >> self.log2_G
        c = lane & (self.G - 1)
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mLin = cute.make_tensor(lin_run_ptr, cute.make_layout(1 << 30))
        mCum = cute.make_tensor(cum_ptr, cute.make_layout(1 << 30))
        mRun = cute.make_tensor(run_ids_ptr, cute.make_layout(1 << 30))
        mPsw = cute.make_tensor(psw_ptr, cute.make_layout(1 << 30))
        mRofs = cute.make_tensor(rofs_ptr, cute.make_layout(1 << 30))
        mBofs = cute.make_tensor(bofs_ptr, cute.make_layout(1 << 20))
        num_runs = Int32(cute.make_tensor(num_runs_ptr, cute.make_layout(1))[0])
        bits = cutlass.Uint32(info_B_num_bits)
        bmask = cutlass.Uint32(info_B_mask)
        wsz = self.w_dtype.width // 8
        # pyrefly: ignore [missing-attribute]
        dout_base = dout_ptr.toint()
        # pyrefly: ignore [missing-attribute]
        w_base = w_ptr.toint()
        col = Int64(c * (VEC // 2))
        acc = cute.make_rmem_tensor((VEC,), Float32)
        frags = [cute.make_rmem_tensor((VEC,), Float32) for _ in range(self.unroll)]
        w_row = cute.make_rmem_tensor((VEC,), self.w_dtype)
        acc.fill(0.0)

        # Runs per batch: up to 32, fewer when the bucket has few runs so they
        # still spread over every warp (shard D's D=128 bucket: 10K runs of ~100
        # lookups would otherwise occupy ~300 warps).
        gdim, _, _ = cute.arch.grid_dim()
        rpb = cutlass.min(
            cutlass.max(
                cute.ceil_div(num_runs, gdim * self.warps), Int32(WARP // self.G)
            ),
            WARP,
        )
        b0 = Int32(0)
        if lane == 0:
            b0 = Int32(cute.arch.atomic_add(counter_ptr, rpb))
        b0 = cute.arch.shuffle_sync(b0, 0)
        while b0 < num_runs:
            nxt = Int32(0)
            if lane == 0:
                nxt = Int32(cute.arch.atomic_add(counter_ptr, rpb))
            nb = cutlass.min(num_runs - b0, rpb)
            # Lane l: run b0 + l's lookup range, row width, weight and momentum
            # offsets, and its first lookup's dout element offset.
            p_s = Int32(0)
            p_L = Int32(0)
            p_D = Int32(0)
            p_w = Int64(0)
            p_m = Int64(0)
            p_d0 = Int64(0)
            p_pw = Float32(1.0)  # first lookup's per-sample weight
            if lane < nb:
                rid = Int32(mRun[b0 + lane])
                p_s = Int32(mCum[rid])
                p_L = Int32(mCum[rid + 1]) - p_s
                u0 = cutlass.Uint32(Int32(mInfos[p_s]))
                t = Int32(u0 >> bits)
                p_D = Int32(mMeta[t, META_D])
                row = Int64(mLin[rid]) - Int64(mMeta[t, META_HASH_CUMSUM])
                p_w = Int64(mMeta[t, META_TABLE_OFFSET]) + row * p_D
                p_m = Int64(mMeta[t, META_ROWS_CUMSUM]) + row
                p_d0 = _dout_row(
                    u0,
                    bits,
                    bmask,
                    total_D,
                    Int64(mMeta[t, META_OUT_OFFSET]),
                    mRofs,
                    mBofs,
                    self.vbe,
                )
                if const_expr(self.weighted):
                    p_pw = Float32(mPsw[p_s])
            for i in cutlass.range(0, nb, WARP // self.G):  # one run per group
                k = i + g  # this group's run within the batch
                src = cutlass.min(k, WARP - 1)
                s_k = cute.arch.shuffle_sync(p_s, src)
                L_k = cute.arch.shuffle_sync(p_L, src)
                D_k = cute.arch.shuffle_sync(p_D, src)
                w_k = cute.arch.shuffle_sync(p_w, src)
                m_k = cute.arch.shuffle_sync(p_m, src)
                d0_k = cute.arch.shuffle_sync(p_d0, src)
                pw_k = cute.arch.shuffle_sync(p_pw, src)
                ok = k < nb
                act = ok & (c < D_k // VEC)
                half = Int64(D_k // 2)
                w_addr = w_base + (w_k + col) * wsz
                mom = Float32(0.0)
                if act:
                    _copy4(w_addr, w_row, 0, self.w_dtype)
                    _copy4(w_addr + half * wsz, w_row, 4, self.w_dtype)
                    d_addr = dout_base + (d0_k + col) * 4
                    _copy4(d_addr, acc, 0, Float32)
                    _copy4(d_addr + half * 4, acc, 4, Float32)
                    if const_expr(self.weighted):
                        acc.store(acc.load() * pw_k)
                if ok:
                    mom = Float32(
                        cute.make_tensor(mom_ptr + m_k, cute.make_layout(1))[0]
                    )
                # Further lookups, G at a time: lane c of the group decodes lookup
                # j + c's dout offset (its own feature: a run of a shared table can
                # span features), then the group walks them by shuffle with up to
                # `unroll` dout rows in flight.
                gbase = lane - c
                j = Int32(1)
                while cute.arch.vote_ballot_sync(ok & (j < L_k)) != 0:
                    my_d = Int64(0)
                    my_pw = Float32(0.0)
                    if ok & (j + c < L_k):
                        if const_expr(self.weighted):
                            my_pw = Float32(mPsw[s_k + j + c])
                        u = cutlass.Uint32(Int32(mInfos[s_k + j + c]))
                        my_d = _dout_row(
                            u,
                            bits,
                            bmask,
                            total_D,
                            Int64(mMeta[Int32(u >> bits), META_OUT_OFFSET]),
                            mRofs,
                            mBofs,
                            self.vbe,
                        )
                    n_k = cutlass.min(L_k - j, self.G)
                    jj = Int32(0)
                    while cute.arch.vote_ballot_sync(ok & (jj < n_k)) != 0:
                        ws = []
                        for v in cutlass.range_constexpr(self.unroll):
                            srcv = gbase + cutlass.min(jj + v, self.G - 1)
                            d = cute.arch.shuffle_sync(my_d, srcv)
                            if const_expr(self.weighted):
                                ws.append(cute.arch.shuffle_sync(my_pw, srcv))
                            dd = dout_base + (d + col) * 4
                            if act & (jj + v < n_k):
                                _copy4(dd, frags[v], 0, Float32)
                                _copy4(dd + half * 4, frags[v], 4, Float32)
                        for v in cutlass.range_constexpr(self.unroll):
                            if act & (jj + v < n_k):
                                if const_expr(self.weighted):
                                    acc.store(acc.load() + frags[v].load() * ws[v])
                                else:
                                    acc.store(acc.load() + frags[v].load())
                        jj = jj + self.unroll
                    j = j + self.G
                # Row-wise Adagrad: mean of grad^2 over the row, reduced in-group.
                sq = Float32(0.0)
                for e in cutlass.range_constexpr(VEC):
                    sq = sq + acc[e] * acc[e]
                sq = cutlass.select_(act, sq, Float32(0.0))
                for sh in cutlass.range_constexpr(self.log2_G):
                    sq = sq + cute.arch.shuffle_sync_bfly(sq, 1 << sh)
                mom_new = mom + sq / Float32(cutlass.max(D_k, 1))
                alr = lr / (cute.math.sqrt(mom_new) + eps)
                if act:
                    upd = cute.make_rmem_tensor((VEC,), self.w_dtype)
                    upd.store(
                        (w_row.load().to(Float32) - acc.load() * alr).to(self.w_dtype)
                    )
                    _copy4(w_addr, upd, 0, self.w_dtype, True)
                    _copy4(w_addr + half * wsz, upd, 4, self.w_dtype, True)
                if ok & (c == 0):
                    cute.make_tensor(mom_ptr + m_k, cute.make_layout(1))[0] = mom_new
            b0 = cute.arch.shuffle_sync(nxt, 0)


class TbeBackwardTiled(TbeBackwardShortRun):
    """Short-run backward over (batch tile, row) runs, processed tile-major.

    ``_tile_runs`` re-sorts the short runs' lookups by (tile, row) with
    tile = feature * tiles_per_feature + b // tile_b, so each tile's slice of
    dout stays in L2 while every lookup that reads it runs (row order re-reads
    each dout row from DRAM once per lookup; 2.7x more DRAM traffic on shard D).
    A row whose lookups all fall in one tile (~90%) is a single run here and
    gets row-wise Adagrad directly. A row spanning tiles writes one fp32 partial
    per tile; ``finalize`` adds them in tile order and applies Adagrad, so the
    result is deterministic.

    Each warp takes ``batch`` consecutive runs and walks their lookups as one
    flat stream, like the forward: per-run metadata in one coalesced load per
    field, 32 dout offsets per load, up to ``unroll`` dout rows in flight
    regardless of run boundaries. The per-run epilogue sits outside the
    unrolled loop so it is emitted once.
    """

    def __init__(self, w_dtype, unroll: int, warps: int, batch: int):
        super().__init__(w_dtype, unroll, warps, small_T=False)
        self.batch = batch

    @cute.jit
    def tiled(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        doff_ptr: cute.Pointer,
        tcum_ptr: cute.Pointer,
        lin_ptr: cute.Pointer,
        rt_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        nruns_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        part_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        T: Int32,
        lr: Float32,
        eps: Float32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.tiled_kernel(
            dout_ptr,
            w_ptr,
            mom_ptr,
            doff_ptr,
            tcum_ptr,
            lin_ptr,
            rt_ptr,
            slot_ptr,
            nruns_ptr,
            counter_ptr,
            part_ptr,
            meta_ptr,
            T,
            lr,
            eps,
        ).launch(
            grid=[grid, 1, 1],
            block=[WARP * self.warps, 1, 1],
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def tiled_kernel(
        self,
        dout_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        doff_ptr: cute.Pointer,
        tcum_ptr: cute.Pointer,
        lin_ptr: cute.Pointer,
        rt_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        nruns_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        part_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        T: Int32,
        lr: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % WARP
        nruns = Int32(cute.make_tensor(nruns_ptr, cute.make_layout(1))[0])
        mCum = cute.make_tensor(tcum_ptr, cute.make_layout(1 << 30))
        mDoff = cute.make_tensor(doff_ptr, cute.make_layout(1 << 30))
        mLin = cute.make_tensor(lin_ptr, cute.make_layout(1 << 30))
        mRt = cute.make_tensor(rt_ptr, cute.make_layout(1 << 30))
        mSlot = cute.make_tensor(slot_ptr, cute.make_layout(1 << 30))
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        wsz = self.w_dtype.width // 8
        # pyrefly: ignore [missing-attribute]
        dout_base = dout_ptr.toint()
        # pyrefly: ignore [missing-attribute]
        part_base = part_ptr.toint()

        acc = cute.make_rmem_tensor((VEC,), Float32)
        w_row = cute.make_rmem_tensor((VEC,), self.w_dtype)
        frags = [cute.make_rmem_tensor((VEC,), Float32) for _ in range(self.unroll)]
        for u in cutlass.range_constexpr(self.unroll):
            frags[u].fill(0.0)  # idle lanes never load; keep their sums finite
        # Batches come off an atomic queue, so warps stay on nearby runs (the
        # same tile or two) however unevenly their batches take.
        RB = self.batch
        r0 = Int32(0)
        if lane == 0:
            r0 = Int32(cute.arch.atomic_add(counter_ptr, Int32(RB)))
        r0 = cute.arch.shuffle_sync(r0, 0)
        while r0 < nruns:
            nxt = Int32(0)
            if lane == 0:
                nxt = Int32(cute.arch.atomic_add(counter_ptr, Int32(RB)))
            nb = cutlass.min(nruns - r0, RB)
            # Lane k: run r0 + k's end, weight row, momentum index, partial slot
            # and row width (runs of one launch may differ in D).
            my_end = Int32(0)
            my_woff = Int64(0)
            my_midx = Int64(0)
            my_slot = Int32(-1)
            my_D = Int32(0)
            if lane < nb:
                my_end = Int32(mCum[r0 + lane + 1])
                my_slot = Int32(mSlot[r0 + lane])
                rt = Int32(mRt[r0 + lane])
                my_D = Int32(mMeta[rt, META_D])
                row = Int64(mLin[r0 + lane]) - Int64(mMeta[rt, META_HASH_CUMSUM])
                my_woff = Int64(mMeta[rt, META_TABLE_OFFSET]) + row * my_D
                my_midx = Int64(mMeta[rt, META_ROWS_CUMSUM]) + row
            s0 = Int32(mCum[r0])
            s1 = cute.arch.shuffle_sync(my_end, nb - 1)

            cur = Int32(0)
            cur_end = cute.arch.shuffle_sync(my_end, 0)
            w_addr, m_idx, pslot, mom, D, col = self._run_start(
                cur, lane, my_woff, my_midx, my_slot, my_D, w_ptr, mom_ptr, w_row
            )
            acc.fill(0.0)
            for c0 in cutlass.range(s0, s1, WARP):
                n = cutlass.min(s1 - c0, WARP)
                my_doff = Int32(0)
                if c0 + lane < s1:
                    my_doff = Int32(mDoff[c0 + lane])
                j = Int32(0)
                while j < n:
                    # Up to `unroll` lookups, never past the current run's end.
                    take = cutlass.min(
                        cutlass.min(cur_end - c0 - j, n - j), Int32(self.unroll)
                    )
                    for u in cutlass.range_constexpr(self.unroll):
                        d = cute.arch.shuffle_sync(
                            my_doff, cutlass.min(j + u, WARP - 1)
                        )
                        d_addr = dout_base + (Int64(d) + col) * 4
                        if (take > u) & (lane < D // VEC):
                            _copy4(d_addr, frags[u], 0, Float32)
                            _copy4(d_addr + Int64(D // 2) * 4, frags[u], 4, Float32)
                    for u in cutlass.range_constexpr(self.unroll):
                        if take > u:
                            acc.store(acc.load() + frags[u].load())
                    j = j + take
                    if c0 + j == cur_end:
                        mMom = cute.make_tensor(mom_ptr + m_idx, cute.make_layout(1))
                        if pslot < 0:
                            self._adagrad(
                                acc,
                                w_row,
                                mom,
                                mMom,
                                w_addr,
                                w_addr + Int64(D // 2) * wsz,
                                cutlass.select_(lane < D // VEC, Int32(0), Int32(1)),
                                lane,
                                D,
                                lr,
                                eps,
                            )
                        else:
                            if lane < D // VEC:
                                p_lo = part_base + (Int64(pslot) * ROW_MAX + col) * 4
                                _copy4(p_lo, acc, 0, Float32, True)
                                _copy4(p_lo + Int64(D // 2) * 4, acc, 4, Float32, True)
                        acc.fill(0.0)
                        cur = cur + 1
                        if cur < nb:
                            cur_end = cute.arch.shuffle_sync(my_end, cur)
                            w_addr, m_idx, pslot, mom, D, col = self._run_start(
                                cur,
                                lane,
                                my_woff,
                                my_midx,
                                my_slot,
                                my_D,
                                w_ptr,
                                mom_ptr,
                                w_row,
                            )
            r0 = cute.arch.shuffle_sync(nxt, 0)

    @cute.jit
    def _run_start(
        self, cur, lane, my_woff, my_midx, my_slot, my_D, w_ptr, mom_ptr, w_row
    ):
        """Run cur's lane layout and addresses. Lane c owns elements [4c, 4c+4)
        and [D/2 + 4c, D/2 + 4c + 4) of the row (lanes c >= D/8 idle), so each
        load instruction covers 512 contiguous bytes. A run applied here
        prefetches its weight row and momentum, which load behind its gather.
        Returns (weight address, momentum index, slot, momentum, D, column)."""
        wsz = self.w_dtype.width // 8
        D = cute.arch.shuffle_sync(my_D, cur)
        col = Int64(cutlass.select_(lane < D // VEC, lane, 0) * (VEC // 2))
        w_addr = w_ptr.toint() + (cute.arch.shuffle_sync(my_woff, cur) + col) * wsz
        m_idx = cute.arch.shuffle_sync(my_midx, cur)
        pslot = cute.arch.shuffle_sync(my_slot, cur)
        mom = Float32(0.0)
        if pslot < 0:
            _copy4(w_addr, w_row, 0, self.w_dtype)
            _copy4(w_addr + Int64(D // 2) * wsz, w_row, 4, self.w_dtype)
            mom = Float32(cute.make_tensor(mom_ptr + m_idx, cute.make_layout(1))[0])
        return w_addr, m_idx, pslot, mom, D, col

    @cute.jit
    def finalize(
        self,
        w_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        mr_ptr: cute.Pointer,
        n_mr_ptr: cute.Pointer,
        part_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        T: Int32,
        info_B_num_bits: Int32,
        lr: Float32,
        eps: Float32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        """Second pass for rows that span tiles: add their per-tile partial sums in
        tile order (deterministic) and apply row-wise Adagrad. One warp per row."""
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        self.fin_kernel(
            w_ptr,
            mom_ptr,
            infos_ptr,
            lin_run_ptr,
            cum_ptr,
            mr_ptr,
            n_mr_ptr,
            part_ptr,
            mMeta,
            info_B_num_bits,
            lr,
            eps,
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def fin_kernel(
        self,
        w_ptr,
        mom_ptr,
        infos_ptr,
        lin_run_ptr,
        cum_ptr,
        mr_ptr,
        n_mr_ptr,
        part_ptr,
        mMeta,
        info_B_num_bits: Int32,
        lr: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        bits = cutlass.Uint32(info_B_num_bits)
        n_mr = Int32(cute.make_tensor(n_mr_ptr, cute.make_layout(1))[0])
        mMR = cute.make_tensor(mr_ptr, cute.make_layout((1 << 26, 3), stride=(3, 1)))
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mCum = cute.make_tensor(cum_ptr, cute.make_layout(1 << 30))
        mLin = cute.make_tensor(lin_run_ptr, cute.make_layout(1 << 30))
        for i in cutlass.range(
            bidx * self.warps + tidx // WARP, n_mr, gdim * self.warps
        ):
            run_id = Int32(mMR[i, 0])
            base = Int32(mMR[i, 1])
            nseg = Int32(mMR[i, 2])
            t = Int32(cutlass.Uint32(Int32(mInfos[Int32(mCum[run_id])])) >> bits)
            row = Int64(mLin[run_id]) - Int64(mMeta[t, META_HASH_CUMSUM])
            D = Int32(mMeta[t, META_D])
            nch = D // VEC
            slot = cutlass.select_(lane < nch, Int32(0), Int32(1))
            col = Int64(cutlass.select_(lane < nch, lane, 0) * (VEC // 2))
            half = Int64(D // 2)
            w_addr = _addr(
                w_ptr, Int64(mMeta[t, META_TABLE_OFFSET]) + row * D + col, self.w_dtype
            )
            w_addr_hi = w_addr + half * (self.w_dtype.width // 8)
            w_row = cute.make_rmem_tensor((VEC,), self.w_dtype)
            _copy4(w_addr, w_row, 0, self.w_dtype)
            _copy4(w_addr_hi, w_row, 4, self.w_dtype)
            mMom = cute.make_tensor(
                mom_ptr + Int64(mMeta[t, META_ROWS_CUMSUM]) + row, cute.make_layout(1)
            )
            mom = Float32(mMom[0])
            acc = cute.make_rmem_tensor((VEC,), Float32)
            acc.fill(0.0)
            frag = cute.make_rmem_tensor((VEC,), Float32)
            for k in cutlass.range(0, nseg):
                p_lo = _addr(part_ptr, Int64(base + k) * ROW_MAX + col, Float32)
                _copy4(p_lo, frag, 0, Float32)
                _copy4(p_lo + half * 4, frag, 4, Float32)
                acc.store(acc.load() + frag.load())
            self._adagrad(
                acc, w_row, mom, mMom, w_addr, w_addr_hi, slot, lane, D, lr, eps
            )


class TileRunsPrep:
    """Builds TbeBackwardTiled's (batch tile, row) runs from the row-sorted short
    runs with a two-pass counting sort by tile (see _tile_runs).

    Both passes give each warp the same batch of runs: one lane per run, up to
    32 per warp (fewer when the bucket has few runs, so they spread over every
    warp). Runs of <= 32 lookups are walked by their own lane, longer ones by
    the whole warp. A run's lookups are in (feature, b) order, so its tiles are
    nondecreasing and each tile's lookups form one contiguous segment.

    * ``count``: per run, its tile count and whether to split it (split runs
      get nseg consecutive partial slots and a finalize record); per tile, the
      number of segments and lookups that go to it.
    * ``scatter``: each segment reserves its run index and lookup range within
      its tile with one packed 64-bit atomic (segments << 32 | lookups), then
      writes the run's start, linear index, feature and slot, and its lookups'
      dout element offsets. Order within a tile is atomic order; it only affects
      scheduling (each row still sums in b order, tiles in tile order).
    """

    # Each tile's counter is split over SUB sub-counters picked by block index
    # (its range becomes SUB consecutive sub-ranges), each on its own 128 B line
    # (PAD int64s): one counter per tile took ~17K atomics per address on shard D
    # and serialized, and same-line counters contend like one.
    SUB = 32
    PAD = 16

    def __init__(self, warps: int = 8):
        self.warps = warps

    @cute.jit
    def count(
        self,
        infos_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_short_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        rbase_ptr: cute.Pointer,
        ctr_ptr: cute.Pointer,
        mr_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        T: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        tpf: Int32,
        tile_shift: Int32,
        min_per_tile: Int32,
        cap: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        self.count_kernel(
            infos_ptr,
            cum_ptr,
            run_ids_ptr,
            num_short_ptr,
            mMeta,
            rbase_ptr,
            ctr_ptr,
            mr_ptr,
            cnt_ptr,
            info_B_num_bits,
            info_B_mask,
            tpf,
            tile_shift,
            min_per_tile,
            cap,
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def count_kernel(
        self,
        infos_ptr,
        cum_ptr,
        run_ids_ptr,
        num_short_ptr,
        mMeta,
        rbase_ptr,
        ctr_ptr,
        mr_ptr,
        cnt_ptr,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        tpf: Int32,
        tile_shift: Int32,
        min_per_tile: Int32,
        cap: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        num_short = Int32(cute.make_tensor(num_short_ptr, cute.make_layout(1))[0])
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mCum = cute.make_tensor(cum_ptr, cute.make_layout(1 << 30))
        mRun = cute.make_tensor(run_ids_ptr, cute.make_layout(1 << 30))
        mRbase = cute.make_tensor(rbase_ptr, cute.make_layout(1 << 30))
        mMR = cute.make_tensor(mr_ptr, cute.make_layout((1 << 26, 3), stride=(3, 1)))
        bits = cutlass.Uint32(info_B_num_bits)
        bmask = cutlass.Uint32(info_B_mask)
        sh = cutlass.Uint32(tile_shift)
        nwarps = gdim * self.warps
        wid = bidx * self.warps + tidx // WARP
        rpb = cutlass.min(cutlass.max(cute.ceil_div(num_short, nwarps), 1), WARP)

        # Static batches (scatter balances with a queue); the sub-counter is
        # picked by batch index, so both passes agree on it.
        for b0 in cutlass.range(wid * rpb, num_short, nwarps * rpb):
            sub = (b0 // rpb) % self.SUB
            cnt_b = cnt_ptr + sub * self.PAD  # tile t: + t * SUB * PAD
            i = b0 + lane
            ok = (lane < rpb) & (i < num_short)
            rid = Int32(0)
            s0 = Int32(0)
            L = Int32(0)
            tf = Int32(0)
            if ok:
                rid = Int32(mRun[i])
                s0 = Int32(mCum[rid])
                L = Int32(mCum[rid + 1]) - s0
                u0 = cutlass.Uint32(Int32(mInfos[s0]))
                tf = Int32(u0 >> bits) * tpf + Int32((u0 & bmask) >> sh)
            small = ok & (L <= WARP)
            nseg = Int32(1)
            if small:
                prev = tf
                # pyrefly: ignore [no-matching-overload]
                for j in cutlass.range(1, L, unroll=4):
                    tile = _tile_of(mInfos, s0 + j, bits, bmask, sh, tpf)
                    nseg = nseg + cutlass.select_(tile != prev, Int32(1), Int32(0))
                    prev = tile
            big = cutlass.Uint32(cute.arch.vote_ballot_sync(ok & (L > WARP)))
            m = big
            while m != cutlass.Uint32(0):
                p = _lowest(m)
                m = m & (m - cutlass.Uint32(1))
                n_p = self._segments(
                    mInfos,
                    cute.arch.shuffle_sync(s0, p),
                    cute.arch.shuffle_sync(L, p),
                    lane,
                    bits,
                    bmask,
                    sh,
                    tpf,
                    cnt_b,
                    None,
                    0,
                )
                if lane == p:
                    nseg = n_p

            # Split only if the run averages >= min_per_tile lookups per tile:
            # each extra tile costs a 1 KB partial written and read back, more
            # than the L2 reuse of one or two dout rows saves.
            # The partial pool holds `cap` slots; a run that doesn't fit stays whole
            # (correct, just without the L2 reuse). rb >= 0 marks split runs.
            split = ok & (nseg > 1) & (L >= min_per_tile * nseg)
            rb = Int32(-1)
            if split:
                rb = Int32(cute.arch.atomic_add(ctr_ptr, nseg))
                if rb + nseg > cap:
                    rb = Int32(-1)
                else:
                    mi = Int32(cute.arch.atomic_add(ctr_ptr + 1, Int32(1)))
                    mMR[mi, 0] = rid
                    mMR[mi, 1] = rb
                    mMR[mi, 2] = nseg
            if ok:
                mRbase[i] = rb

            # Per-tile (segments, lookups), packed as segments << 32 | lookups.
            if small:
                if rb >= 0:
                    seg0 = Int32(0)
                    cur = tf
                    # pyrefly: ignore [no-matching-overload]
                    for j in cutlass.range(1, L, unroll=4):
                        tile = _tile_of(mInfos, s0 + j, bits, bmask, sh, tpf)
                        if tile != cur:
                            cute.arch.atomic_add(
                                cnt_b + cur * self.SUB * self.PAD, _pack(j - seg0)
                            )
                            seg0 = j
                            cur = tile
                    cute.arch.atomic_add(
                        cnt_b + cur * self.SUB * self.PAD, _pack(L - seg0)
                    )
                else:
                    cute.arch.atomic_add(cnt_b + tf * self.SUB * self.PAD, _pack(L))
            m = big
            while m != cutlass.Uint32(0):
                p = _lowest(m)
                m = m & (m - cutlass.Uint32(1))
                if cute.arch.shuffle_sync(rb, p) >= 0:  # split
                    self._segments(
                        mInfos,
                        cute.arch.shuffle_sync(s0, p),
                        cute.arch.shuffle_sync(L, p),
                        lane,
                        bits,
                        bmask,
                        sh,
                        tpf,
                        cnt_b,
                        None,
                        1,
                    )
                else:
                    tf_p = cute.arch.shuffle_sync(tf, p)  # (shuffles stay converged)
                    L_p = cute.arch.shuffle_sync(L, p)
                    if lane == 0:
                        cute.arch.atomic_add(
                            cnt_b + tf_p * self.SUB * self.PAD, _pack(L_p)
                        )

    @cute.jit
    def scatter(
        self,
        infos_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        run_ids_ptr: cute.Pointer,
        num_short_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        rbase_ptr: cute.Pointer,
        queue_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        off_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        tcum_ptr: cute.Pointer,
        lin_ptr: cute.Pointer,
        rt_ptr: cute.Pointer,
        slot_ptr: cute.Pointer,
        doff_ptr: cute.Pointer,
        T: Int32,
        total_D: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        tpf: Int32,
        tile_shift: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        mMeta = cute.make_tensor(
            meta_ptr, cute.make_layout((T, META_COLS), stride=(META_COLS, 1))
        )
        out = (res_ptr, off_ptr, lin_run_ptr, tcum_ptr, lin_ptr, rt_ptr, slot_ptr)
        self.scatter_kernel(
            infos_ptr,
            cum_ptr,
            run_ids_ptr,
            num_short_ptr,
            mMeta,
            rbase_ptr,
            queue_ptr,
            *out,
            doff_ptr,
            total_D,
            info_B_num_bits,
            info_B_mask,
            tpf,
            tile_shift,
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def scatter_kernel(
        self,
        infos_ptr,
        cum_ptr,
        run_ids_ptr,
        num_short_ptr,
        mMeta,
        rbase_ptr,
        queue_ptr,
        res_ptr,
        off_ptr,
        lin_run_ptr,
        tcum_ptr,
        lin_ptr,
        rt_ptr,
        slot_ptr,
        doff_ptr,
        total_D: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        tpf: Int32,
        tile_shift: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        num_short = Int32(cute.make_tensor(num_short_ptr, cute.make_layout(1))[0])
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mCum = cute.make_tensor(cum_ptr, cute.make_layout(1 << 30))
        mRun = cute.make_tensor(run_ids_ptr, cute.make_layout(1 << 30))
        mRbase = cute.make_tensor(rbase_ptr, cute.make_layout(1 << 30))
        bits = cutlass.Uint32(info_B_num_bits)
        bmask = cutlass.Uint32(info_B_mask)
        sh = cutlass.Uint32(tile_shift)
        nwarps = gdim * self.warps
        rpb = cutlass.min(cutlass.max(cute.ceil_div(num_short, nwarps), 1), WARP)
        dst0 = (
            mMeta,
            res_ptr,
            off_ptr,
            lin_run_ptr,
            tcum_ptr,
            lin_ptr,
            rt_ptr,
            slot_ptr,
            doff_ptr,
            total_D,
            mInfos,
            bits,
            bmask,
            tpf,
            self.SUB,
        )
        # Batches come off an atomic queue (run lengths vary, so a static split
        # straggles); the sub-counter goes with the batch, so both passes agree.
        b0 = Int32(0)
        if lane == 0:
            b0 = Int32(cute.arch.atomic_add(queue_ptr, rpb))
        b0 = cute.arch.shuffle_sync(b0, 0)
        while b0 < num_short:
            nxt = Int32(0)
            if lane == 0:
                nxt = Int32(cute.arch.atomic_add(queue_ptr, rpb))
            sub = (b0 // rpb) % self.SUB
            # tile t's counter: + t * SUB * PAD
            dst = (mMeta, res_ptr + sub * self.PAD) + dst0[2:] + (sub,)
            i = b0 + lane
            ok = (lane < rpb) & (i < num_short)
            rid = Int32(0)
            s0 = Int32(0)
            L = Int32(0)
            tf = Int32(0)
            rb = Int32(-1)
            if ok:
                rid = Int32(mRun[i])
                s0 = Int32(mCum[rid])
                L = Int32(mCum[rid + 1]) - s0
                tf = _tile_of(mInfos, s0, bits, bmask, sh, tpf)
                rb = Int32(mRbase[i])
            if ok & (L <= WARP):
                if rb >= 0:
                    seg0 = Int32(0)
                    cur = tf
                    k = Int32(0)
                    # pyrefly: ignore [no-matching-overload]
                    for j in cutlass.range(1, L, unroll=4):
                        tile = _tile_of(mInfos, s0 + j, bits, bmask, sh, tpf)
                        if tile != cur:
                            _emit_lane(dst, rid, s0 + seg0, j - seg0, cur, rb + k)
                            seg0 = j
                            cur = tile
                            k = k + 1
                    _emit_lane(dst, rid, s0 + seg0, L - seg0, cur, rb + k)
                else:
                    _emit_lane(dst, rid, s0, L, tf, Int32(-1))
            m = cutlass.Uint32(cute.arch.vote_ballot_sync(ok & (L > WARP)))
            while m != cutlass.Uint32(0):
                p = _lowest(m)
                m = m & (m - cutlass.Uint32(1))
                s0_p = cute.arch.shuffle_sync(s0, p)
                L_p = cute.arch.shuffle_sync(L, p)
                rb_p = cute.arch.shuffle_sync(rb, p)
                rid_p = cute.arch.shuffle_sync(rid, p)
                tf_p = cute.arch.shuffle_sync(tf, p)
                if rb_p >= 0:
                    self._segments(
                        mInfos,
                        s0_p,
                        L_p,
                        lane,
                        bits,
                        bmask,
                        sh,
                        tpf,
                        (dst, rid_p, rb_p),
                        None,
                        2,
                    )
                else:
                    _emit_warp(dst, lane, rid_p, s0_p, L_p, tf_p, Int32(-1))
            b0 = cute.arch.shuffle_sync(nxt, 0)

    @cute.jit
    def offsets(
        self,
        cnt_ptr: cute.Pointer,
        off_ptr: cute.Pointer,
        nrun_ptr: cute.Pointer,
        tcum_ptr: cute.Pointer,
        n: Int32,
        stream: cuda.CUstream,
    ):
        self.offsets_kernel(cnt_ptr, off_ptr, nrun_ptr, tcum_ptr, n).launch(
            grid=[1, 1, 1], block=[WARP, 1, 1], stream=stream
        )

    @cute.kernel
    def offsets_kernel(self, cnt_ptr, off_ptr, nrun_ptr, tcum_ptr, n: Int32):
        """One warp: exclusive scan of the n packed (segments, lookups) counters
        (stride PAD) into per-counter run and lookup offsets; total runs, and the
        end sentinel tcum[runs] = lookups. Lane l scans a contiguous chunk."""
        lane, _, _ = cute.arch.thread_idx()
        mCnt = cute.make_tensor(cnt_ptr, cute.make_layout(1 << 24))
        mOff = cute.make_tensor(off_ptr, cute.make_layout((1 << 20, 2), stride=(2, 1)))
        per = (n + WARP - 1) // WARP
        lo = cutlass.min(lane * per, n)
        hi = cutlass.min(lo + per, n)
        tot = Int64(0)
        # pyrefly: ignore [no-matching-overload]
        for i in cutlass.range(lo, hi, unroll=8):
            tot = tot + Int64(mCnt[i * self.PAD])
        incl = tot
        for sh in cutlass.range_constexpr(5):
            v = cute.arch.shuffle_sync_up(incl, 1 << sh, mask_and_clamp=0)
            if lane >= (1 << sh):
                incl = incl + v
        run = incl - tot  # packed exclusive prefix: no carries (fields < 2^32)
        for i in cutlass.range(lo, hi):
            # pyrefly: ignore [unsupported-operation]
            mOff[i, 0] = Int32(run >> 32)
            # pyrefly: ignore [unsupported-operation]
            mOff[i, 1] = Int32(run & 0xFFFFFFFF)
            run = run + Int64(mCnt[i * self.PAD])
        if lane == WARP - 1:
            # pyrefly: ignore [unsupported-operation]
            cute.make_tensor(nrun_ptr, cute.make_layout(1))[0] = Int32(incl >> 32)
            # pyrefly: ignore [unsupported-operation]
            cute.make_tensor(tcum_ptr, cute.make_layout(1 << 30))[Int32(incl >> 32)] = (
                # pyrefly: ignore [unsupported-operation]
                Int32(incl & 0xFFFFFFFF)
            )

    @cute.jit
    def _segments(
        self,
        mInfos,
        s0,
        L,
        lane,
        bits,
        bmask,
        sh,
        tpf,
        out,
        unused,
        mode: cutlass.Constexpr,
    ):
        """Whole warp walks one run 32 lookups at a time, closing a segment at
        each tile change. mode 0: return the tile count. mode 1: add each
        segment to the per-tile counts (out = cnt_ptr). mode 2: emit each
        segment (out = (dst, rid, first slot))."""
        prev = Int32(-1)  # tile of the previous lookup (carried across chunks)
        cur = Int32(0)
        seg0 = Int32(0)
        k = Int32(0)
        for c0 in cutlass.range(0, L, WARP):
            in_run = c0 + lane < L
            tile = Int32(-2)
            if in_run:
                tile = _tile_of(mInfos, s0 + c0 + lane, bits, bmask, sh, tpf)
            up = cute.arch.shuffle_sync_up(tile, 1, mask_and_clamp=0)
            before = cutlass.select_(lane == 0, prev, up)
            starts = cutlass.Uint32(
                cute.arch.vote_ballot_sync(in_run & (tile != before) & (c0 + lane > 0))
            )
            if c0 == 0:
                cur = cute.arch.shuffle_sync(tile, 0)
            if const_expr(mode == 0):
                k = k + cute.arch.popc(Int32(starts))
            else:
                while starts != cutlass.Uint32(0):
                    p = _lowest(starts)
                    starts = starts & (starts - cutlass.Uint32(1))
                    self._close(out, lane, s0, seg0, c0 + p - seg0, cur, k, mode)
                    seg0 = c0 + p
                    cur = cute.arch.shuffle_sync(tile, p)
                    k = k + 1
            prev = cute.arch.shuffle_sync(tile, cutlass.min(L - c0, WARP) - 1)
        if const_expr(mode != 0):
            self._close(out, lane, s0, seg0, L - seg0, cur, k, mode)
        return k + 1

    @cute.jit
    def _close(self, out, lane, s0, seg0, n, tile, k, mode: cutlass.Constexpr):
        if const_expr(mode == 1):
            if lane == 0:
                cute.arch.atomic_add(out + tile * self.SUB * self.PAD, _pack(n))
        else:
            dst, rid, rb = out
            _emit_warp(dst, lane, rid, s0 + seg0, n, tile, rb + k)


@cute.jit
def _tile_of(mInfos, pos, bits, bmask, sh, tpf):
    ui = cutlass.Uint32(Int32(mInfos[pos]))
    return Int32(ui >> bits) * tpf + Int32((ui & bmask) >> sh)


@cute.jit
def _lowest(m):
    """Index of the lowest set bit of a nonzero mask."""
    return cute.arch.popc(Int32((m & (~m + cutlass.Uint32(1))) - cutlass.Uint32(1)))


@cute.jit
def _pack(n):
    """One segment of n lookups, as added to a tile's 64-bit counter."""
    return Int64(n) + Int64(1 << 32)


@cute.jit
def _reserve(dst, rid, n, tile, slot):
    """Reserve a segment's run index and lookup range in its tile; write the
    run's metadata. Returns (first lookup position, feature, its dout column
    offset)."""
    mMeta, res_ptr, off_ptr, lin_run_ptr, tcum_ptr, lin_ptr, rt_ptr, slot_ptr = dst[:8]
    S, sub = dst[14], dst[15]
    mOff = cute.make_tensor(off_ptr, cute.make_layout((1 << 20, 2), stride=(2, 1)))
    old = Int64(cute.arch.atomic_add(res_ptr + tile * S * TileRunsPrep.PAD, _pack(n)))
    r = Int32(mOff[tile * S + sub, 0]) + Int32(old >> 32)
    pos = Int32(mOff[tile * S + sub, 1]) + Int32(old & 0xFFFFFFFF)
    t = tile // dst[13]
    cute.make_tensor(tcum_ptr, cute.make_layout(1 << 30))[r] = pos
    cute.make_tensor(lin_ptr, cute.make_layout(1 << 30))[r] = Int64(
        cute.make_tensor(lin_run_ptr, cute.make_layout(1 << 30))[rid]
    )
    cute.make_tensor(rt_ptr, cute.make_layout(1 << 30))[r] = t
    cute.make_tensor(slot_ptr, cute.make_layout(1 << 30))[r] = slot
    return pos, t, Int32(mMeta[t, META_OUT_OFFSET])


@cute.jit
def _dout_off(mMeta, ui, total_D, bits, bmask, t_seg, oo_seg):
    """Element offset of a lookup's dout row, from its own feature: that is the
    segment's feature t_seg (column offset oo_seg) except in runs of a shared
    table that span features without being split."""
    t = Int32(ui >> bits)
    oo = oo_seg
    if t != t_seg:
        oo = Int32(mMeta[t, META_OUT_OFFSET])
    return Int32(ui & bmask) * total_D + oo


@cute.jit
def _emit_lane(dst, rid, start, n, tile, slot):
    """One lane emits a segment: lookups [start, start + n) of infos."""
    pos, t_seg, oo_seg = _reserve(dst, rid, n, tile, slot)
    doff_ptr, total_D, mInfos, bits, bmask = dst[8:13]
    mDoff = cute.make_tensor(doff_ptr, cute.make_layout(1 << 30))
    for j in cutlass.range(0, n, unroll=4):
        ui = cutlass.Uint32(Int32(mInfos[start + j]))
        mDoff[pos + j] = _dout_off(dst[0], ui, total_D, bits, bmask, t_seg, oo_seg)


@cute.jit
def _emit_warp(dst, lane, rid, start, n, tile, slot):
    """The whole warp emits a segment (lane 0 reserves it)."""
    pos = Int32(0)
    t_seg = Int32(0)
    oo_seg = Int32(0)
    if lane == 0:
        pos, t_seg, oo_seg = _reserve(dst, rid, n, tile, slot)
    pos = cute.arch.shuffle_sync(pos, 0)
    t_seg = cute.arch.shuffle_sync(t_seg, 0)
    oo_seg = cute.arch.shuffle_sync(oo_seg, 0)
    doff_ptr, total_D, mInfos, bits, bmask = dst[8:13]
    mDoff = cute.make_tensor(doff_ptr, cute.make_layout(1 << 30))
    for c0 in cutlass.range(0, n, WARP):
        if c0 + lane < n:
            ui = cutlass.Uint32(Int32(mInfos[start + c0 + lane]))
            mDoff[pos + c0 + lane] = _dout_off(
                dst[0], ui, total_D, bits, bmask, t_seg, oo_seg
            )


class TbeLongRunAccum:
    """Long-run gradient accumulation (rows with >= 256 lookups), replacing
    ``triton_tbe_backward_long_run_grad_accum_unweighted``. Each chunk (<= 256
    lookups of one row, from Triton's _expand_long_runs) is summed and
    atomically added to the row's fp32 buffer, as in Triton; Triton's apply
    kernel then runs row-wise Adagrad.

    Hot rows come from small tables with big bags, so within a chunk (lookups
    sorted by sample) the same dout row repeats: shard B's 46-row table averages 4.2
    lookups per distinct (row, sample). Each distinct (feature, sample) dout
    row is loaded once and scaled by its multiplicity. A chunk's row takes G =
    next_pow2(D / 8) lanes (chosen per chunk: one launch covers every feature),
    so narrow rows pack 32 / G lookups per instruction: lane group g takes the
    distinct lookups among window lanes [g*G, (g+1)*G), and the groups are
    reduced at the end of the chunk.

    Chunks come off an atomic queue in (feature, first sample) order (``keys``
    + an argsort): concurrent chunks of different hot rows then read the same
    dout slice from L2 and spread their atomics over rows (in row order every
    warp is on one row's accumulator: shard B 6.9 vs 1.4 ms). Unsupported chunks (D
    not a multiple of 4, or of 8 above 128) sort last and are left to Triton.
    """

    def __init__(self, warps: int, unroll: int, vbe: bool = False):
        self.vbe = vbe  # dout is the flat VBE output (see _dout_row)
        self.warps = warps
        self.unroll = unroll

    @cute.jit
    def keys(
        self,
        infos_ptr: cute.Pointer,
        seg_starts_ptr: cute.Pointer,
        seg_ends_ptr: cute.Pointer,
        dims_ptr: cute.Pointer,
        noff_ptr: cute.Pointer,
        num_ptr: cute.Pointer,
        key_ptr: cute.Pointer,
        nrest_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        cap: Int32,
        info_B_num_bits: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.keys_kernel(
            infos_ptr,
            seg_starts_ptr,
            seg_ends_ptr,
            dims_ptr,
            noff_ptr,
            num_ptr,
            key_ptr,
            nrest_ptr,
            cnt_ptr,
            cap,
            info_B_num_bits,
        ).launch(grid=[grid, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def keys_kernel(
        self,
        infos_ptr,
        seg_starts_ptr,
        seg_ends_ptr,
        dims_ptr,
        noff_ptr,
        num_ptr,
        key_ptr,
        nrest_ptr,
        cnt_ptr,
        cap: Int32,
        info_B_num_bits: Int32,
    ):
        """Sort key per chunk: its first lookup's 32-bit info (feature << bits |
        sample) >> 1 (sample parity is irrelevant to locality), with the sign bit
        set for supported chunks so they sort first; unused slots get INT32_MAX.
        Counts the unsupported chunks (D % 4 != 0, or D > 128 and D % 8 != 0),
        and the long-run lookups of each narrow feature (see _narrow_layout)."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mSS = cute.make_tensor(seg_starts_ptr, cute.make_layout(1 << 30))
        mSE = cute.make_tensor(seg_ends_ptr, cute.make_layout(1 << 30))
        mDims = cute.make_tensor(dims_ptr, cute.make_layout(1 << 20))
        mNoff = cute.make_tensor(noff_ptr, cute.make_layout(1 << 20))
        mKey = cute.make_tensor(key_ptr, cute.make_layout(1 << 30))
        num = Int32(cute.make_tensor(num_ptr, cute.make_layout(1))[0])
        bits = cutlass.Uint32(info_B_num_bits)
        lane = tidx % WARP
        # Warp-uniform loop (the narrow counts below use full-warp shuffles).
        for i0 in cutlass.range(bidx * 256 + tidx - lane, cap, gdim * 256):
            i = i0 + lane
            k = Int32(0x7FFFFFFF)
            tn = Int32(-1)  # narrow feature of the chunk
            n = Int32(0)
            if i < num:
                u = cutlass.Uint32(Int32(mInfos[Int32(mSS[i])]))
                half_u = Int32(u >> cutlass.Uint32(1))
                t = Int32(u >> bits)
                d = Int32(mDims[t])
                if (d % 4 != 0) | ((d > 4 * WARP) & (d % VEC != 0)):  # see kernel
                    k = half_u
                    cute.arch.atomic_add(nrest_ptr, Int32(1))
                else:
                    k = half_u | Int32(-0x80000000)
                    if Int32(mNoff[t]) >= 0:
                        tn = t
                        n = Int32(mSE[i]) - Int32(mSS[i])
            if i < cap:
                mKey[i] = k
            # One atomic per feature per warp (per-chunk atomics on shard A's 24
            # counters cost 0.04 ms).
            if cute.arch.vote_ballot_sync(tn >= 0) != 0:
                grp = _match_any(tn)
                tot = _redux_add(n, grp)
                if (tn >= 0) & (lane == _lowest(grp)):
                    _red_add_s32(cnt_ptr.toint() + Int64(tn) * 4, tot)

    @cute.jit
    def compact(
        self,
        dout_ptr: cute.Pointer,
        nar_ptr: cute.Pointer,
        src_ptr: cute.Pointer,
        pf_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        total_D: Int32,
        npieces: Int32,
        B: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.compact_kernel(
            dout_ptr, nar_ptr, src_ptr, pf_ptr, cnt_ptr, total_D, npieces, B
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def compact_kernel(
        self,
        dout_ptr: cute.Pointer,
        nar_ptr: cute.Pointer,
        src_ptr: cute.Pointer,
        pf_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        total_D: Int32,
        npieces: Int32,
        B: Int32,
    ):
        """Compact copy of the narrow features' dout columns (see
        _narrow_layout): warp per sample, 4 floats per lane from column src[p]
        of feature pf[p], for features with >= B long-run lookups."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        mSrc = cute.make_tensor(src_ptr, cute.make_layout(1 << 20))
        mPf = cute.make_tensor(pf_ptr, cute.make_layout(1 << 20))
        mCnt = cute.make_tensor(cnt_ptr, cute.make_layout(1 << 20))
        v = cute.make_rmem_tensor((4,), Float32)
        anyq = False
        for p in cutlass.range(lane, npieces, WARP):
            anyq = anyq | (Int32(mCnt[Int32(mPf[p])]) >= B)
        b0 = bidx * self.warps + tidx // WARP
        if cute.arch.vote_ballot_sync(anyq) == 0:
            b0 = B
        for b in cutlass.range(b0, B, gdim * self.warps):
            for p in cutlass.range(lane, npieces, WARP):
                if Int32(mCnt[Int32(mPf[p])]) >= B:
                    src = Int64(b) * total_D + Int32(mSrc[p])
                    # pyrefly: ignore [missing-attribute]
                    _copy4(dout_ptr.toint() + src * 4, v, 0, Float32)
                    dst = (Int64(b) * npieces + p) * 16
                    # pyrefly: ignore [missing-attribute]
                    _copy4(nar_ptr.toint() + dst, v, 0, Float32, True)

    @cute.jit
    def __call__(
        self,
        dout_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        perm_ptr: cute.Pointer,
        seg_starts_ptr: cute.Pointer,
        seg_ends_ptr: cute.Pointer,
        buf_ids_ptr: cute.Pointer,
        temp_ptr: cute.Pointer,
        dims_ptr: cute.Pointer,
        out_offs_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        num_ptr: cute.Pointer,
        nrest_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        nar_ptr: cute.Pointer,
        noff_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        total_D: Int32,
        ncols: Int32,
        nar_min: Int32,
        row_stride: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            dout_ptr,
            infos_ptr,
            perm_ptr,
            seg_starts_ptr,
            seg_ends_ptr,
            buf_ids_ptr,
            temp_ptr,
            dims_ptr,
            out_offs_ptr,
            rofs_ptr,
            bofs_ptr,
            num_ptr,
            nrest_ptr,
            counter_ptr,
            nar_ptr,
            noff_ptr,
            cnt_ptr,
            total_D,
            ncols,
            nar_min,
            row_stride,
            info_B_num_bits,
            info_B_mask,
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        dout_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        perm_ptr: cute.Pointer,
        seg_starts_ptr: cute.Pointer,
        seg_ends_ptr: cute.Pointer,
        buf_ids_ptr: cute.Pointer,
        temp_ptr: cute.Pointer,
        dims_ptr: cute.Pointer,
        out_offs_ptr: cute.Pointer,
        rofs_ptr: cute.Pointer,
        bofs_ptr: cute.Pointer,
        num_ptr: cute.Pointer,
        nrest_ptr: cute.Pointer,
        counter_ptr: cute.Pointer,
        nar_ptr: cute.Pointer,
        noff_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        total_D: Int32,
        ncols: Int32,
        nar_min: Int32,
        row_stride: Int32,
        info_B_num_bits: Int32,
        info_B_mask: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % WARP
        mNoff = cute.make_tensor(noff_ptr, cute.make_layout(1 << 20))
        mCnt = cute.make_tensor(cnt_ptr, cute.make_layout(1 << 20))
        mInfos = cute.make_tensor(infos_ptr, cute.make_layout(1 << 30))
        mSS = cute.make_tensor(seg_starts_ptr, cute.make_layout(1 << 30))
        mSE = cute.make_tensor(seg_ends_ptr, cute.make_layout(1 << 30))
        mBuf = cute.make_tensor(buf_ids_ptr, cute.make_layout(1 << 30))
        mDims = cute.make_tensor(dims_ptr, cute.make_layout(1 << 20))
        mOut = cute.make_tensor(out_offs_ptr, cute.make_layout(1 << 20))
        mPerm = cute.make_tensor(perm_ptr, cute.make_layout(1 << 30))
        mRofs = cute.make_tensor(rofs_ptr, cute.make_layout(1 << 30))
        mBofs = cute.make_tensor(bofs_ptr, cute.make_layout(1 << 20))
        num = Int32(cute.make_tensor(num_ptr, cute.make_layout(1))[0]) - Int32(
            cute.make_tensor(nrest_ptr, cute.make_layout(1))[0]
        )
        bits = cutlass.Uint32(info_B_num_bits)
        bmask = cutlass.Uint32(info_B_mask)
        acc = cute.make_rmem_tensor((VEC,), Float32)
        frags = [cute.make_rmem_tensor((VEC,), Float32) for _ in range(self.unroll)]

        i = Int32(0)
        if lane == 0:
            i = Int32(cute.arch.atomic_add(counter_ptr, Int32(1)))
        i = cute.arch.shuffle_sync(i, 0)
        while i < num:
            nxt = Int32(0)
            if lane == 0:
                nxt = Int32(cute.arch.atomic_add(counter_ptr, Int32(1)))
            j = Int32(mPerm[i])
            s = Int32(mSS[j])
            e = Int32(mSE[j])
            buf = Int64(mBuf[j])
            t0 = Int32(cutlass.Uint32(Int32(mInfos[s])) >> bits)
            D = Int32(mDims[t0])
            # Narrow features with >= B long-run lookups read a compact copy
            # of their dout columns (see _narrow_layout).
            nar = False
            if ncols > 0:
                nar = (Int32(mNoff[t0]) >= 0) & (Int32(mCnt[t0]) >= nar_min)
            # pyrefly: ignore [missing-attribute]
            base = cutlass.select_(nar, nar_ptr.toint(), dout_ptr.toint())
            stride = cutlass.select_(nar, ncols, total_D)
            oo0 = Int64(cutlass.select_(nar, Int32(mNoff[t0]), Int32(mOut[t0])))
            # Rows of D >= 64 (D % 8 == 0) take D/8 lanes, lane c owning elements
            # [4c, 4c+4) and [D/2+4c, D/2+4c+4): two lane groups per warp at D=128
            # keep twice the rows in flight of a whole-warp row (1.5 vs 2.2 ms).
            # Narrower rows take D/4 lanes owning [4c, 4c+4): a lane loading both
            # halves of a <= 128 B row costs two L1 wavefronts for one cache line
            # (D=8 was L1-bound). Lane groups of G = next_pow2(lanes) lanes.
            two = (D >= 64) & (D % VEC == 0)
            lanes = cutlass.select_(two, D // VEC, D // 4)
            lg = Int32(0)
            # pyrefly: ignore [unsupported-operation]
            while (1 << lg) < lanes:
                lg = lg + 1
            G = Int32(1) << lg
            c = lane & (G - 1)
            g = lane >> lg
            gbase = lane - c
            gmask = cutlass.select_(
                G == WARP,
                cutlass.Uint32(0xFFFFFFFF),
                (cutlass.Uint32(1) << cutlass.Uint32(G)) - cutlass.Uint32(1),
            )
            col = Int64(c * (VEC // 2))
            act = c < lanes
            half = Int64(cutlass.select_(two, D // 2, 0))
            acc.fill(0.0)
            # Windows of 32 lookups; the next window's infos load during this one.
            my_next = Int32(-1)
            if lane < e - s:
                my_next = Int32(mInfos[s + lane])
            for w0 in cutlass.range(s, e, WARP):
                n = cutlass.min(e - w0, WARP)
                my = my_next
                my_next = Int32(-1)
                if w0 + WARP + lane < e:
                    my_next = Int32(mInfos[w0 + WARP + lane])
                # Leaders: first lookup of each run of equal infos in the window;
                # cnt = lookups until the next leader.
                up = cute.arch.shuffle_sync_up(my, 1, mask_and_clamp=0)
                lead = (lane < n) & ((lane == 0) | (my != up))
                bal = cutlass.Uint32(cute.arch.vote_ballot_sync(lead))
                above = bal & ~(
                    (cutlass.Uint32(2) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
                )
                nxt_lead = cutlass.select_(
                    above != cutlass.Uint32(0), _lowest(above), n
                )
                cnt = Float32(nxt_lead - lane)
                # This group's leaders are window lanes [gbase, gbase + G). Compact
                # them: group lane c takes leader c (fns = n-th set bit), computes
                # its dout address, and the group then walks the list by shuffle
                # with no dependency between unrolled steps.
                mine = (bal >> cutlass.Uint32(gbase)) & gmask
                nl = cute.arch.popc(Int32(mine))
                has_c = c < nl
                src = gbase + cutlass.select_(has_c, Int32(_fns(mine, c + 1)), 0)
                ui = cutlass.Uint32(cute.arch.shuffle_sync(my, src))
                my_w = cutlass.select_(
                    has_c, cute.arch.shuffle_sync(cnt, src), Float32(0.0)
                )
                et = Int32(ui >> bits)
                oo = oo0
                if has_c & (et != t0):  # a run of a shared table can span features
                    oo = Int64(cutlass.select_(nar, Int32(mNoff[et]), Int32(mOut[et])))
                # dout element offset (fits in 32 bits, see tbe_long_run): one
                # shuffle per step instead of two for a 64-bit address. Lanes
                # without a leader skip it: under VBE it loads B_offsets[t] and
                # row_output_offsets with t from the window's -1 padding.
                my_dd = Int32(0)
                if has_c:
                    my_dd = Int32(
                        _dout_row(ui, bits, bmask, stride, oo, mRofs, mBofs, self.vbe)
                    )
                # A group has at most G leaders, so steps v >= G are skipped (D=8
                # groups are 2 lanes: their 3rd/4th steps would be pure overhead).
                k0 = Int32(0)
                while cute.arch.vote_ballot_sync(k0 < nl) != 0:
                    ws = []
                    for v in cutlass.range_constexpr(self.unroll):
                        w = Float32(0.0)
                        if (v < 2) | (G > v):
                            sv = gbase + cutlass.min(k0 + v, G - 1)
                            dd = (
                                base
                                + (Int64(cute.arch.shuffle_sync(my_dd, sv)) + col) * 4
                            )
                            wv = cute.arch.shuffle_sync(my_w, sv)
                            ok = (k0 + v < nl) & act
                            if ok:
                                _copy4(dd, frags[v], 0, Float32)
                                if two:
                                    _copy4(dd + half * 4, frags[v], 4, Float32)
                            w = cutlass.select_(ok, wv, Float32(0.0))
                        ws.append(w)
                    for v in cutlass.range_constexpr(self.unroll):
                        if ws[v] != Float32(0.0):
                            acc.store(acc.load() + frags[v].load() * ws[v])
                    k0 = k0 + cutlass.min(G, self.unroll)
            # Reduce the lane groups (same column, different lookups).
            for sh in cutlass.range(lg, 5):
                for k in cutlass.range_constexpr(VEC):
                    acc[k] = acc[k] + cute.arch.shuffle_sync_bfly(
                        acc[k], Int32(1) << sh
                    )
            if act & (g == 0):
                # pyrefly: ignore [missing-attribute]
                row = temp_ptr.toint() + (buf * row_stride + col) * 4
                for k in cutlass.range_constexpr(VEC // 2):
                    _atomic_add_f32(row + k * 4, acc[k])
                if two:
                    for k in cutlass.range_constexpr(VEC // 2):
                        _atomic_add_f32(row + (half + k) * 4, acc[VEC // 2 + k])
            i = cute.arch.shuffle_sync(nxt, 0)


class TbeLongRunApply:
    """Row-wise Adagrad for the long runs from their accumulated fp32 gradients,
    replacing ``triton_tbe_backward_long_run_apply_unweighted``, which launches
    one program per possible long run (1.17M on shard A for ~69K real rows). One
    warp per real run (grid-stride to the device-side count), same math. The
    row layout matches TbeLongRunAccum (one or two 4-element pieces per lane)."""

    def __init__(self, w_dtype, warps: int = 8):
        self.w_dtype = w_dtype
        self.warps = warps

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        temp_ptr: cute.Pointer,
        lin_run_ptr: cute.Pointer,
        cum_ptr: cute.Pointer,
        infos_ptr: cute.Pointer,
        orig_ptr: cute.Pointer,
        num_ptr: cute.Pointer,
        tbl_off_ptr: cute.Pointer,
        dims_ptr: cute.Pointer,
        ftm_ptr: cute.Pointer,
        hsc_ptr: cute.Pointer,
        mom_ptr: cute.Pointer,
        rcs_ptr: cute.Pointer,
        row_stride: Int32,
        info_B_num_bits: Int32,
        lr: Float32,
        eps: Float32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            w_ptr,
            temp_ptr,
            lin_run_ptr,
            cum_ptr,
            infos_ptr,
            orig_ptr,
            num_ptr,
            tbl_off_ptr,
            dims_ptr,
            ftm_ptr,
            hsc_ptr,
            mom_ptr,
            rcs_ptr,
            row_stride,
            info_B_num_bits,
            lr,
            eps,
        ).launch(grid=[grid, 1, 1], block=[WARP * self.warps, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        w_ptr,
        temp_ptr,
        lin_run_ptr,
        cum_ptr,
        infos_ptr,
        orig_ptr,
        num_ptr,
        tbl_off_ptr,
        dims_ptr,
        ftm_ptr,
        hsc_ptr,
        mom_ptr,
        rcs_ptr,
        row_stride: Int32,
        info_B_num_bits: Int32,
        lr: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tidx % WARP
        big = cute.make_layout(1 << 30)
        small = cute.make_layout(1 << 20)
        mLin = cute.make_tensor(lin_run_ptr, big)
        mCum = cute.make_tensor(cum_ptr, big)
        mInfos = cute.make_tensor(infos_ptr, big)
        mOrig = cute.make_tensor(orig_ptr, big)
        mTbl = cute.make_tensor(tbl_off_ptr, small)
        mDims = cute.make_tensor(dims_ptr, small)
        mFtm = cute.make_tensor(ftm_ptr, small)
        mHsc = cute.make_tensor(hsc_ptr, small)
        mRcs = cute.make_tensor(rcs_ptr, small)
        num = Int32(cute.make_tensor(num_ptr, cute.make_layout(1))[0])
        bits = cutlass.Uint32(info_B_num_bits)
        wsz = self.w_dtype.width // 8
        col = Int64(lane * (VEC // 2))
        grad = cute.make_rmem_tensor((VEC,), Float32)
        w_row = cute.make_rmem_tensor((VEC,), self.w_dtype)
        for k in cutlass.range(
            bidx * self.warps + tidx // WARP, num, gdim * self.warps
        ):
            run = Int32(mOrig[k])
            t = Int32(cutlass.Uint32(Int32(mInfos[Int32(mCum[run])])) >> bits)
            tbl = Int32(mFtm[t])
            D = Int32(mDims[t])
            row = Int64(mLin[run]) - Int64(mHsc[t])
            two = D % VEC == 0
            act = lane < cutlass.select_(two, D // VEC, D // 4)
            half = Int64(cutlass.select_(two, D // 2, 0))
            w_addr = w_ptr.toint() + (Int64(mTbl[tbl]) + row * D + col) * wsz
            g_addr = temp_ptr.toint() + (Int64(k) * row_stride + col) * 4
            mMom = cute.make_tensor(
                mom_ptr + Int64(mRcs[tbl]) + row, cute.make_layout(1)
            )
            grad.fill(0.0)
            if act:
                _copy4(w_addr, w_row, 0, self.w_dtype)
                _copy4(g_addr, grad, 0, Float32)
                if two:
                    _copy4(w_addr + half * wsz, w_row, 4, self.w_dtype)
                    _copy4(g_addr + half * 4, grad, 4, Float32)
            mom = Float32(mMom[0])
            sq = Float32(0.0)
            for e in cutlass.range_constexpr(VEC):
                sq = sq + grad[e] * grad[e]
            for sh in cutlass.range_constexpr(5):
                sq = sq + cute.arch.shuffle_sync_bfly(sq, 1 << sh)
            mom_new = mom + sq / Float32(D)
            alr = lr / (cute.math.sqrt(mom_new) + eps)
            if act:
                upd = cute.make_rmem_tensor((VEC,), self.w_dtype)
                upd.store(
                    (w_row.load().to(Float32) - grad.load() * alr).to(self.w_dtype)
                )
                _copy4(w_addr, upd, 0, self.w_dtype, True)
                if two:
                    _copy4(w_addr + half * wsz, upd, 4, self.w_dtype, True)
            if lane == 0:
                mMom[0] = mom_new


@cute.jit
def _dout_row(ui, bits, bmask, total_D, out_off, mRofs, mBofs, vbe: cutlass.Constexpr):
    """Element offset of a lookup's dout row from its info (feature t, sample b).
    Dense output: b * total_D + out_off (t's column offset). VBE (flat output,
    per-feature batch sizes): row_output_offsets[B_offsets[t] + b]."""
    if const_expr(vbe):
        return Int64(mRofs[Int32(mBofs[Int32(ui >> bits)]) + Int32(ui & bmask)])
    return Int64(Int32(ui & bmask)) * total_D + out_off


@dsl_user_op
def _fns(mask, n, *, loc=None, ip=None):
    """Bit position of the n-th (1-based) set bit of mask, 0xFFFFFFFF if none
    (PTX fns.b32)."""
    res = llvm.inline_asm(
        T.i32(),
        [
            cutlass.Uint32(mask).ir_value(loc=loc, ip=ip),
            Int32(n).ir_value(loc=loc, ip=ip),
        ],
        "fns.b32 $0, $1, 0, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(res)


@dsl_user_op
def _atomic_add_f32(addr: Int64, val: Float32, *, loc=None, ip=None):
    """red.global.add.f32 at a raw address (no return value)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(val).ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_add_s32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """red.global.add.s32 at a raw address (no return value)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "red.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _match_any(v, *, loc=None, ip=None):
    """Mask of the lanes holding the same value as this one (match.any.sync)."""
    res = llvm.inline_asm(
        T.i32(),
        [Int32(v).ir_value(loc=loc, ip=ip)],
        "match.any.sync.b32 $0, $1, 0xffffffff;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(res)


@dsl_user_op
def _redux_add(v, mask, *, loc=None, ip=None):
    """Sum of v over the lanes in mask (redux.sync.add.s32; every lane in mask
    passes the same mask)."""
    res = llvm.inline_asm(
        T.i32(),
        [
            Int32(v).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(mask).ir_value(loc=loc, ip=ip),
        ],
        "redux.sync.add.s32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(res)


# ----------------------------------------------------------------------------
# Host wrappers
# ----------------------------------------------------------------------------

_cache: Dict[Tuple, object] = {}


def build_meta(
    feature_table_map,
    table_offsets,
    dims,
    out_offsets,
    hash_size_cumsum,
    rows_cumsum,
    device,
) -> torch.Tensor:
    T = len(feature_table_map)
    meta = torch.zeros((T, META_COLS), dtype=torch.int64)
    for t, tbl in enumerate(feature_table_map):
        d = int(dims[t])
        assert d % 2 == 0, f"feature {t}: D={d} must be even"
        meta[t, META_TABLE_OFFSET] = int(table_offsets[tbl])
        meta[t, META_D] = d
        meta[t, META_OUT_OFFSET] = int(out_offsets[t])
        meta[t, META_HASH_CUMSUM] = (
            0 if hash_size_cumsum is None else int(hash_size_cumsum[t])
        )
        meta[t, META_ROWS_CUMSUM] = int(rows_cumsum[tbl])
        meta[t, META_NUM_ROWS] = int(rows_cumsum[tbl + 1]) - int(rows_cumsum[tbl])
        nch = max(d // VEC, 1)
        meta[t, META_RPW] = max(WARP // nch, 1)
        meta[t, META_SLOT_MAGIC] = -(-(1 << 16) // max(nch, 1))
        meta[t, META_FEATURE] = t
    return meta.to(device)


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _vec_for(d: int, tbl_off: int) -> int:
    """Lane vector width for a feature. 16B lanes whenever alignment allows: rows
    narrower than a warp then pack several rows per instruction, which measured
    faster than 8B/4B lanes on the flat stream (D=112: 1.43 vs 1.80 ms; the kernel
    is issue-bound). Narrower lanes only for D or offsets not a multiple of 8."""
    for v in (8, 4, 2):
        if d % v == 0 and tbl_off % v == 0:
            return v
    raise AssertionError(f"D={d}, table offset {tbl_off}: needs even D and offset")


_vec_groups: Dict[Tuple, list] = {}
_aligned_ok: Dict[Tuple, bool] = {}


def _check_output_alignment(meta, total_D: int, vbe: bool) -> None:
    """Output rows are stored with 16B (fp32 x4) vectors: every feature's output
    start must be a multiple of 4 elements. Checked once per meta tensor."""
    key = (_tensor_key(meta), meta.shape[0], total_D, vbe)
    if key not in _aligned_ok:
        m = meta.cpu()
        dims = m[:, META_D].tolist()
        outs = m[:, META_OUT_OFFSET].tolist()
        ok = all(o % 4 == 0 for o in outs) and (total_D % 4 == 0 or vbe)
        if vbe:  # row_output_offsets are sums of B_t * D_t
            ok = ok and all(d % 4 == 0 for d in dims)
        assert (
            ok
        ), f"CuTe TBE forward needs 4-element aligned outputs (D={dims}, total_D={total_D})"
        _aligned_ok[key] = True


_narrow_choice: Dict[Tuple, str] = {}


def _forward_groups(meta):
    """[(vec, meta rows of the features using it)], cached per meta tensor."""
    key = (_tensor_key(meta), meta.shape[0])
    if key not in _vec_groups:
        m = meta.cpu()
        by_vec: Dict[int, list] = {}
        for t in range(m.shape[0]):
            v = _vec_for(int(m[t, META_D]), int(m[t, META_TABLE_OFFSET]))
            by_vec.setdefault(v, []).append(t)
        _vec_groups[key] = [
            (v, meta[torch.tensor(ts, device=meta.device)].contiguous())
            for v, ts in sorted(by_vec.items())
        ]
    return _vec_groups[key]


def tbe_forward(
    weights,
    indices,
    offsets,
    meta,
    B,
    T,
    total_D,
    out=None,
    warn=None,
    per_sample_weights=None,
    unroll=4,
    warps=4,
    vbe_B_offsets=None,
    vbe_row_output_offsets=None,
):
    """Pooled forward into `out` ([B, total_D], or 1-D VBE output). VBE: pass the
    per-feature B_offsets and row_output_offsets; B is then the max batch size."""
    unroll = int(os.environ.get("CUTE_TBE_FWD_UNROLL", unroll))
    warps = int(os.environ.get("CUTE_TBE_FWD_WARPS", warps))
    if out is None:
        out = torch.empty((B, total_D), dtype=torch.float32, device=weights.device)
    assert out.dtype == torch.float32 and out.is_contiguous()
    bounds_check = warn is not None
    weighted = per_sample_weights is not None
    if weighted:
        assert per_sample_weights.dtype == torch.float32
    vbe = vbe_B_offsets is not None
    if vbe:
        assert vbe_row_output_offsets is not None and vbe_B_offsets.dtype == torch.int32
        assert vbe_row_output_offsets.dtype == torch.int64
    _check_output_alignment(meta, total_D, vbe)

    def launch(narrow, warn_t):
        for vec, gmeta in _forward_groups(meta):
            key = (
                "fwd",
                weights.dtype,
                indices.dtype,
                offsets.dtype,
                bounds_check,
                unroll,
                warps,
                weighted,
                vec,
                narrow,
                vbe,
            )
            args = (
                _ptr(out),
                _ptr(indices, indices.element_size()),
                _ptr(offsets, offsets.element_size()),
                _ptr(weights),
                _ptr(gmeta, 8),
                _ptr(per_sample_weights if weighted else out, 4),
                _ptr(warn_t if bounds_check else meta, 8),
                _ptr(vbe_row_output_offsets if vbe else gmeta, 8),
                _ptr(vbe_B_offsets if vbe else out, 4),
                Int32(total_D),
                Int32(B),
                Int32(gmeta.shape[0]),
            )
            _compiled(
                key,
                partial(
                    TbeForward,
                    _TORCH_TO_CUTE[weights.dtype],
                    _TORCH_TO_CUTE[indices.dtype],
                    bounds_check,
                    unroll,
                    warps,
                    weighted,
                    vec,
                    narrow,
                    vbe,
                ),
                args,
            )

    # Strategy is chosen once per model (meta) and kept: VBE batch sizes change
    # every step, and re-timing on each new shape would cost more than it saves.
    tkey = (_tensor_key(meta), weighted, weights.dtype, vbe)
    mode = os.environ.get("CUTE_TBE_FWD_NARROW") or _narrow_choice.get(tkey)
    if mode is None:
        # First call for this shape: the real launch (bounds check counted once),
        # then time both narrow-row strategies on the now-repaired indices with a
        # scratch warning counter, and remember the faster one.
        launch("grouped", warn)
        scratch = torch.zeros(1, dtype=torch.int64, device=out.device)
        best = None
        for cand in ("grouped", "packed"):
            launch(cand, scratch)
            s0, e0 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            s0.record()
            for _ in range(3):
                launch(cand, scratch)
            e0.record()
            e0.synchronize()
            if best is None or s0.elapsed_time(e0) < best[0]:
                best = (s0.elapsed_time(e0), cand)
        _narrow_choice[tkey] = best[1]
        launch(best[1], scratch)  # leave `out` from the chosen kernel
    else:
        launch(mode, warn)
    return out


def _compiled(key, make_op, args):
    """Compile once (TVM-FFI calling convention), then launch on the current
    stream. Pointers go in as raw addresses: the TVM-FFI call path costs ~50 us
    of CPU per launch vs ~200+ us for the default one, which showed up as GPU
    idle time on short backward passes (shard D)."""
    if key not in _cache:
        _cache[key] = cute.compile(
            make_op(), *args, _stream(), options="--enable-tvm-ffi"
        )
    _cache[key](
        *[
            a.__tvm_ffi_opaque_ptr__() if isinstance(a, cute.Pointer) else a
            for a in args
        ],
        torch.cuda.current_stream().cuda_stream,
    )


def _tile_runs(
    infos_sorted, lin_run, cum, run_ids, num_short, meta, total_D, bits, mask, B, tile_b
):
    """Split the short runs into (batch tile, row) runs, tile-major, for
    TbeBackwardTiled: a counting sort by tile (TileRunsPrep.count / .offsets /
    .scatter). Returns per-lookup dout element offsets, per-run [start, linear
    index, feature, partial slot (-1: apply in place)], and finalize's (run id,
    first slot, tiles) records, in worst-case-sized buffers reused across calls
    (no host sync; CPU launch cost matters as much as the kernels here).
    """
    dev = infos_sorted.device
    N = infos_sorted.numel()
    if N == 0:
        return None
    tpf = -(-B // tile_b)
    nt = meta.shape[0] * tpf
    assert nt <= 256, "at most 256 tiles"
    assert B * total_D < 1 << 31, "dout offsets are int32"
    ws = _workspace(dev, N)
    ws["zero"].zero_()  # counters: one memset
    i32 = lambda t: _ptr(t, t.element_size())  # noqa: E731
    shift = tile_b.bit_length() - 1
    static = ws["ptrs"]
    cargs = (
        i32(infos_sorted),
        i32(cum),
        i32(run_ids),
        i32(num_short),
        _ptr(meta, 8),
        *static["count"],
        Int32(meta.shape[0]),
        Int32(bits),
        Int32(mask),
        Int32(tpf),
        Int32(shift),
        Int32(int(os.environ.get("CUTE_TBE_BWD_MIN_PER_TILE", 2))),
        Int32(_partial_cap()),
        Int32(ws["grid"]),
    )
    ck = ("prep_count", infos_sorted.dtype, cum.dtype, run_ids.dtype)
    _compiled(ck, lambda: TileRunsPrep().count, cargs)
    oargs = (*static["offsets"], Int32(nt * TileRunsPrep.SUB))
    _compiled("prep_offsets", lambda: TileRunsPrep().offsets, oargs)
    sargs = (
        i32(infos_sorted),
        i32(cum),
        i32(run_ids),
        i32(num_short),
        _ptr(meta, 8),
        *static["scatter"],
        i32(lin_run),
        *static["scatter_out"],
        Int32(meta.shape[0]),
        Int32(total_D),
        Int32(bits),
        Int32(mask),
        Int32(tpf),
        Int32(shift),
        Int32(ws["grid"]),
    )
    sk = ("prep_scatter", infos_sorted.dtype, cum.dtype, run_ids.dtype, lin_run.dtype)
    _compiled(sk, lambda: TileRunsPrep().scatter, sargs)
    return ws


_ws: Dict[Tuple, dict] = {}


def _workspace(dev, N):
    """Prep/backward buffers for N lookups (short runs <= lookups), reused
    across calls (same stream, so each call's kernels finish before the next
    call's overwrite them), with their pointer arguments prebuilt."""
    key = (dev, N)
    if key not in _ws:
        _ws.clear()
        sp = TileRunsPrep.SUB * TileRunsPrep.PAD
        # Zeroed every call: count/scatter tile counters, then int32 counters
        # (slots, records, -, -, -, -, scatter queue, backward queue).
        zero = torch.zeros(2 * 256 * sp + 4, dtype=torch.int64, device=dev)
        cnt = zero[: 2 * 256 * sp].view(2, 256 * sp)
        ctr = zero[2 * 256 * sp :].view(torch.int32)
        w = dict(
            zero=zero,
            ctr=ctr,
            rbase=torch.empty(N, dtype=torch.int32, device=dev),
            mr=torch.empty((N // 4 + 1, 3), dtype=torch.int32, device=dev),
            off=torch.empty((256 * TileRunsPrep.SUB, 2), dtype=torch.int32, device=dev),
            nrun=torch.empty(1, dtype=torch.int32, device=dev),
            tcum=torch.empty(N + 1, dtype=torch.int32, device=dev),
            lin=torch.empty(N, dtype=torch.int64, device=dev),
            rt=torch.empty(N, dtype=torch.int32, device=dev),
            slot=torch.empty(N, dtype=torch.int32, device=dev),
            doff=torch.empty(N, dtype=torch.int32, device=dev),
            part=torch.empty(
                (_partial_cap(), ROW_MAX), dtype=torch.float32, device=dev
            ),
            grid=torch.cuda.get_device_properties(dev).multi_processor_count * 8,
        )
        w["ptrs"] = dict(
            count=(_ptr(w["rbase"], 4), _ptr(ctr, 4), _ptr(w["mr"], 4), _ptr(cnt, 8)),
            offsets=(
                _ptr(cnt, 8),
                _ptr(w["off"], 4),
                _ptr(w["nrun"], 4),
                _ptr(w["tcum"], 4),
            ),
            scatter=(
                _ptr(w["rbase"], 4),
                _ptr(ctr[6:], 4),
                _ptr(cnt[1], 8),
                _ptr(w["off"], 4),
            ),
            scatter_out=tuple(
                _ptr(w[k], w[k].element_size())
                for k in ("tcum", "lin", "rt", "slot", "doff")
            ),
            main=(
                _ptr(w["doff"], 4),
                _ptr(w["tcum"], 4),
                _ptr(w["lin"], 8),
                _ptr(w["rt"], 4),
                _ptr(w["slot"], 4),
                _ptr(w["nrun"], 4),
                _ptr(ctr[7:], 4),
                _ptr(w["part"]),
            ),
            fin=(_ptr(w["mr"], 4), _ptr(ctr[1:], 4), _ptr(w["part"])),
        )
        _ws[key] = w
    return _ws[key]


def _partial_cap():
    """Partial-sum slots (ROW_MAX fp32 each). shard D uses ~0.5M at 32K tiles."""
    return int(os.environ.get("CUTE_TBE_BWD_PARTIAL_CAP", 1 << 20))


def tbe_backward_short_runs(
    dout,
    weights,
    momentum,
    infos_sorted,
    sorted_linear_indices_run,
    cumulative_run_lengths,
    short_run_ids,
    num_short_runs,
    meta,
    total_D,
    info_B_num_bits,
    info_B_mask,
    lr,
    eps,
    B=0,
    grid=None,
    unroll=4,
    warps=8,
    block_size=0,
    per_sample_weights=None,
    vbe_B_offsets=None,
    vbe_row_output_offsets=None,
):
    """Short-run backward. Default: TbeBackwardPacked (row order, several runs
    per warp instruction; needs the launch's block_size). With
    CUTE_TBE_BWD_TILE_B=<power of 2> (and B) lookups are regrouped into batch
    tiles and processed tile-major (TbeBackwardTiled): only worth it where dout
    re-reads dominate the backward's traffic (shard D, not shard C or shard B).
    CUTE_TBE_BWD_RUN_PER_WARP=1 selects the older warp-per-run kernel.
    per_sample_weights (weighted TBE, sorted like the lookups) and VBE (flat
    dout; vbe_B_offsets int32, vbe_row_output_offsets int64): packed only."""
    assert dout.dtype == torch.float32 and dout.is_contiguous()
    warps = int(os.environ.get("CUTE_TBE_BWD_WARPS", warps))
    tile_b = int(os.environ.get("CUTE_TBE_BWD_TILE_B", 0)) if B else 0
    assert tile_b & (tile_b - 1) == 0, "tile size must be a power of 2"
    dev = dout.device
    warps_per_sm = int(os.environ.get("CUTE_TBE_BWD_WARPS_PER_SM", 64))
    if grid is None:
        # One full wave; the work loops cover the rest.
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        grid = sms * max(warps_per_sm // warps, 1)
    T = meta.shape[0]

    def i32(t):
        return _ptr(t, t.element_size())

    weighted = per_sample_weights is not None
    vbe = vbe_B_offsets is not None
    if weighted or vbe:
        # Weighted (sorted per-sample weights, fp32) / VBE: run-packed kernel only.
        assert block_size
        assert not weighted or per_sample_weights.dtype == torch.float32
        tile_b = 0
    if tile_b:
        return _backward_tiled(
            dout,
            weights,
            momentum,
            infos_sorted,
            sorted_linear_indices_run,
            cumulative_run_lengths,
            short_run_ids,
            num_short_runs,
            meta,
            total_D,
            info_B_num_bits,
            info_B_mask,
            lr,
            eps,
            B,
            tile_b,
            grid,
            warps,
        )

    if (
        weighted
        or vbe
        or (block_size and not os.environ.get("CUTE_TBE_BWD_RUN_PER_WARP"))
    ):
        # Several runs per warp instruction (G = block_size / 8 lanes per run).
        G = min(max(block_size // VEC, 1), WARP)
        # Dout rows in flight per group: 4 for full-warp rows (D > 128; long runs
        # want more in flight), 2 below (4 spills there: shard C 6.5 -> 9.3 ms).
        punroll = int(os.environ.get("CUTE_TBE_BWD_UNROLL", 4 if G == WARP else 2))
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        pgrid = sms * max(
            int(os.environ.get("CUTE_TBE_BWD_TILED_WARPS_PER_SM", 32)) // warps, 1
        )
        counter = torch.zeros(1, dtype=torch.int32, device=dev)
        args = (
            _ptr(dout),
            _ptr(weights),
            _ptr(momentum, 4),
            i32(infos_sorted),
            i32(sorted_linear_indices_run),
            i32(cumulative_run_lengths),
            i32(short_run_ids),
            i32(num_short_runs),
            _ptr(counter, 4),
            _ptr(per_sample_weights if weighted else counter, 4),
            _ptr(vbe_row_output_offsets if vbe else counter, 8 if vbe else 4),
            _ptr(vbe_B_offsets if vbe else counter, 4),
            _ptr(meta, 8),
            Int32(T),
            Int32(total_D),
            Int32(info_B_num_bits),
            Int32(info_B_mask),
            Float32(lr),
            Float32(eps),
            Int32(pgrid),
        )
        key = (
            "bwd_packed",
            punroll,
            weighted,
            vbe,
            weights.dtype,
            infos_sorted.dtype,
            cumulative_run_lengths.dtype,
            sorted_linear_indices_run.dtype,
            short_run_ids.dtype,
            G,
            warps,
            os.environ.get("CUTE_TBE_BWD_MIN_WARPS_PER_SM"),
        )
        _compiled(
            key,
            partial(
                TbeBackwardPacked,
                _TORCH_TO_CUTE[weights.dtype],
                G,
                warps,
                punroll,
                weighted,
                vbe,
            ),
            args,
        )
        return
    unroll = int(os.environ.get("CUTE_TBE_BWD_UNROLL", unroll))
    small_T = T <= WARP
    max_runs = short_run_ids.numel()
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    key = (
        "bwd",
        weights.dtype,
        infos_sorted.dtype,
        cumulative_run_lengths.dtype,
        sorted_linear_indices_run.dtype,
        short_run_ids.dtype,
        unroll,
        warps,
        small_T,
        os.environ.get("CUTE_TBE_BWD_MIN_WARPS_PER_SM"),
    )
    args = (
        _ptr(dout),
        _ptr(weights),
        _ptr(momentum, 4),
        i32(infos_sorted),
        i32(sorted_linear_indices_run),
        i32(cumulative_run_lengths),
        i32(short_run_ids),
        i32(num_short_runs),
        _ptr(counter, 4),
        _ptr(meta, 8),
        Int32(T),
        Int32(total_D),
        Int32(max_runs),
        Int32(info_B_num_bits),
        Int32(info_B_mask),
        Float32(lr),
        Float32(eps),
        Int32(grid),
    )
    _compiled(
        key,
        partial(
            TbeBackwardShortRun, _TORCH_TO_CUTE[weights.dtype], unroll, warps, small_T
        ),
        args,
    )


_dims_ok: set = set()


def _backward_tiled(
    dout,
    weights,
    momentum,
    infos_sorted,
    lin_run,
    cum,
    run_ids,
    num_short,
    meta,
    total_D,
    bits,
    mask,
    lr,
    eps,
    B,
    tile_b,
    grid,
    warps,
):
    unroll = int(os.environ.get("CUTE_TBE_BWD_UNROLL", 2))
    batch = int(os.environ.get("CUTE_TBE_BWD_BATCH", 8))
    sms = torch.cuda.get_device_properties(dout.device).multi_processor_count
    grid = sms * max(
        int(os.environ.get("CUTE_TBE_BWD_TILED_WARPS_PER_SM", 32)) // warps, 1
    )
    r = _tile_runs(
        infos_sorted,
        lin_run,
        cum,
        run_ids,
        num_short,
        meta,
        total_D,
        bits,
        mask,
        B,
        tile_b,
    )
    if r is None:
        return
    dims = meta[:, META_D].tolist() if _tensor_key(meta) not in _dims_ok else []
    assert all(d % VEC == 0 and d <= ROW_MAX for d in dims), dims
    _dims_ok.add(_tensor_key(meta))
    wdt = _TORCH_TO_CUTE[weights.dtype]
    op_fn = partial(TbeBackwardTiled, wdt, unroll, warps, batch)
    p = r["ptrs"]
    args = (
        _ptr(dout),
        _ptr(weights),
        _ptr(momentum, 4),
        *p["main"][:7],
        p["main"][7],
        _ptr(meta, 8),
        Int32(meta.shape[0]),
        Float32(lr),
        Float32(eps),
        Int32(grid),
    )
    key = (
        "bwd_tiled",
        weights.dtype,
        unroll,
        warps,
        batch,
        os.environ.get("CUTE_TBE_BWD_MIN_WARPS_PER_SM"),
    )
    _compiled(key, lambda: op_fn().tiled, args)
    fargs = (
        _ptr(weights),
        _ptr(momentum, 4),
        _ptr(infos_sorted, infos_sorted.element_size()),
        _ptr(lin_run, lin_run.element_size()),
        _ptr(cum, cum.element_size()),
        *p["fin"],
        _ptr(meta, 8),
        Int32(meta.shape[0]),
        Int32(bits),
        Float32(lr),
        Float32(eps),
        Int32(grid),
    )
    fkey = (
        "bwd_fin",
        weights.dtype,
        warps,
        infos_sorted.dtype,
        lin_run.dtype,
        cum.dtype,
    )
    _compiled(fkey, lambda: op_fn().finalize, fargs)


_narrow_cache = {}


def _narrow_layout(dims, out_offsets, total_D):
    """Compact layout for TbeLongRunAccum's narrow features (D = 4, 8, 16). In the
    full [B, total_D] output a narrow row is alone in its cache line (shard A: 32
    B every 22 KB), so long-run reads of a hot narrow table are sector-bound;
    from a compact [B, 4 * src.numel()] copy shard A's D=8 chunks take 0.22
    instead of 0.48 ms, for 0.08 ms of copying. Only features with >= B
    long-run lookups (counted on device by keys) are copied and read compact:
    break-even is ~1.25 B for D=8.

    Returns (noff, src, pf): noff[t] = t's column in the copy (-1 for other
    features), src[p] / pf[p] = dout column / feature of 4-float piece p (rows
    padded to 16 floats, widest first, so rows stay aligned). src and pf are
    None if there are no narrow features or total_D isn't a multiple of 4.
    Host lists, cached per tensor."""
    key = (_tensor_key(dims), _tensor_key(out_offsets), dims.numel(), total_D)
    if key not in _narrow_cache:
        ds, oo = dims.tolist(), out_offsets.tolist()
        nf = [t for t, d in enumerate(ds) if d in (4, 8, 16) and oo[t] % 4 == 0]
        nf.sort(key=lambda t: -ds[t])
        noff, src, pf = [-1] * len(ds), [], []
        for t in nf:
            noff[t] = 4 * len(src)
            src += range(oo[t], oo[t] + ds[t], 4)
            pf += [t] * (ds[t] // 4)
        src += src[:1] * (-len(src) % 4)
        pf += pf[:1] * (len(src) - len(pf))
        dev = dims.device
        ok = bool(nf) and total_D % 4 == 0
        _narrow_cache[key] = (
            torch.tensor(noff, dtype=torch.int32, device=dev),
            torch.tensor(src, dtype=torch.int32, device=dev) if ok else None,
            torch.tensor(pf, dtype=torch.int32, device=dev) if ok else None,
        )
    return _narrow_cache[key]


def tbe_long_run(
    dout,
    infos_sorted,
    seg_starts,
    seg_ends,
    buf_ids,
    temp_grad_buffer,
    dims,
    out_offsets,
    num,
    total_D,
    info_B_num_bits,
    info_B_mask,
    warps=8,
    vbe_B_offsets=None,
    vbe_row_output_offsets=None,
):
    """Long-run accumulation (TbeLongRunAccum) over Triton's chunk arrays,
    reading them through a (feature, first sample) order. Returns (perm,
    n_rest): chunks perm[num - n_rest : num] have D % 4 != 0 and are left to
    the caller (Triton's kernel). Launches keys, the narrow copy (when there
    are narrow features), and the accumulation around an argsort; no host sync
    after the first call per dims tensor (_narrow_layout)."""
    dev = dout.device
    cap = seg_starts.numel()
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    i32 = lambda t: _ptr(t, t.element_size())  # noqa: E731
    vbe = vbe_B_offsets is not None
    noff, src, pf = _narrow_layout(dims, out_offsets, total_D)
    if vbe:  # flat output: no columns to compact
        src = None
    key = torch.empty(cap, dtype=torch.int32, device=dev)
    # n_rest, queue, per-feature narrow long-run lookups
    ctr = torch.zeros(2 + dims.numel(), dtype=torch.int32, device=dev)
    # dout element offsets are shuffled as int32.
    assert dout.numel() < 1 << 31, "dout too large for int32 offsets"
    op = partial(
        TbeLongRunAccum, warps, int(os.environ.get("CUTE_TBE_LONG_UNROLL", 4)), vbe
    )
    kargs = (
        i32(infos_sorted),
        i32(seg_starts),
        i32(seg_ends),
        i32(dims),
        i32(noff),
        i32(num),
        _ptr(key, 4),
        _ptr(ctr, 4),
        _ptr(ctr[2:], 4),
        Int32(cap),
        Int32(info_B_num_bits),
        Int32(min(-(-cap // 256), sms * 16)),
    )
    kk = ("long_keys", infos_sorted.dtype, seg_starts.dtype, dims.dtype, num.dtype)
    _compiled(kk, lambda: op().keys, kargs)
    nar, B = dout, dout.numel() // total_D
    if src is not None:
        nar = torch.empty(B, 4 * src.numel(), dtype=dout.dtype, device=dev)
        cargs = (_ptr(dout), _ptr(nar), _ptr(src, 4), _ptr(pf, 4))
        cargs += (_ptr(ctr[2:], 4), Int32(total_D), Int32(src.numel()), Int32(B))
        _compiled(("long_compact",), lambda: op().compact, cargs + (Int32(sms * 8),))
    perm = torch.argsort(key, stable=True)
    args = (
        _ptr(dout),
        i32(infos_sorted),
        _ptr(perm, 8),
        i32(seg_starts),
        i32(seg_ends),
        i32(buf_ids),
        _ptr(temp_grad_buffer, 4),
        i32(dims),
        i32(out_offsets),
        _ptr(vbe_row_output_offsets if vbe else ctr, 8 if vbe else 4),
        _ptr(vbe_B_offsets if vbe else ctr, 4),
        i32(num),
        _ptr(ctr, 4),
        _ptr(ctr[1:], 4),
        _ptr(nar),
        i32(noff),
        _ptr(ctr[2:], 4),
        Int32(total_D),
        Int32(0 if src is None else 4 * src.numel()),
        Int32(B),
        Int32(temp_grad_buffer.stride(0)),
        Int32(info_B_num_bits),
        Int32(info_B_mask),
        Int32(sms * max(int(os.environ.get("CUTE_TBE_LONG_WPS", 32)) // warps, 1)),
    )
    key2 = (
        "long_accum",
        warps,
        infos_sorted.dtype,
        seg_starts.dtype,
        buf_ids.dtype,
        dims.dtype,
        out_offsets.dtype,
        num.dtype,
        os.environ.get("CUTE_TBE_LONG_UNROLL"),
        vbe,
    )
    _compiled(key2, op, args)
    return perm, ctr[:1]


def tbe_long_run_apply(
    weights,
    temp_grad_buffer,
    lin_run,
    cum,
    infos_sorted,
    orig_ids,
    num_long_runs,
    table_offsets,
    dims,
    feature_table_map,
    hash_size_cumsum,
    momentum,
    rows_cumsum,
    info_B_num_bits,
    lr,
    eps,
    warps=8,
):
    """TbeLongRunApply: Adagrad on the num_long_runs (device) long-run rows."""
    sms = torch.cuda.get_device_properties(weights.device).multi_processor_count
    i32 = lambda t: _ptr(t, t.element_size())  # noqa: E731
    tensors = (
        lin_run,
        cum,
        infos_sorted,
        orig_ids,
        num_long_runs,
        table_offsets,
        dims,
        feature_table_map,
        hash_size_cumsum,
    )
    args = (
        _ptr(weights),
        _ptr(temp_grad_buffer, 16),
        *[i32(t) for t in tensors],
        _ptr(momentum, 4),
        i32(rows_cumsum),
        Int32(temp_grad_buffer.stride(0)),
        Int32(info_B_num_bits),
        Float32(lr),
        Float32(eps),
        Int32(sms * max(32 // warps, 1)),
    )
    key = ("long_apply", weights.dtype, warps) + tuple(t.dtype for t in tensors)
    key += (rows_cumsum.dtype,)
    _compiled(key, partial(TbeLongRunApply, _TORCH_TO_CUTE[weights.dtype], warps), args)
