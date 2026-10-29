#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launchers that run the CuTe DSL TBE kernels (``cute_tbe``) in place of Triton
TBE kernel launches, for ``TritonTableBatchedEmbeddingBags(cute_kernels=True)``.

Each launcher takes its Triton kernel's launch arguments unchanged, so
``triton_table_batched_embeddings`` only swaps the kernel object at its launch
sites (``_cute_or``), and falls back to that Triton kernel for what the CuTe
kernels don't cover: split weight tensors, stochastic rounding, optimizers other
than exact row-wise Adagrad, non-fp32 outputs, and unsupported embedding dims.
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torchrec.distributed.triton_tbe.triton_table_batched_embeddings as ttbe
from torchrec.distributed.triton_tbe import cute_tbe

_ROWWISE_ADAGRAD = 1  # OPTIM_TYPE_TO_INT[EXACT_ROWWISE_ADAGRAD]

_meta_cache: Dict[Tuple, torch.Tensor] = {}
_ok_cache: Dict[Tuple, bool] = {}


def _key(t: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
    return None if t is None else (cute_tbe._tensor_key(t), t.numel())


def _meta(table_offsets, dims, out_offsets, ftm, hash_cumsum, rows_cumsum):
    """cute_tbe metadata for the module tensors (same objects every call)."""
    tensors = (table_offsets, dims, out_offsets, ftm, hash_cumsum, rows_cumsum)
    key = tuple(_key(t) for t in tensors)
    if key not in _meta_cache:
        _meta_cache[key] = cute_tbe.build_meta(
            ftm.tolist(),
            table_offsets.tolist(),
            dims.tolist(),
            out_offsets.tolist(),
            None if hash_cumsum is None else hash_cumsum.tolist(),
            rows_cumsum.tolist(),
            table_offsets.device,
        )
    return _meta_cache[key]


def _cached(key: Tuple, check) -> bool:
    if key not in _ok_cache:
        _ok_cache[key] = bool(check())
    return _ok_cache[key]


def _fwd_ok(weight, output, meta, total_D: int, vbe: bool) -> bool:
    """cute_tbe.tbe_forward's constraints: fp16/bf16 weights, fp32 output, even D
    and table offsets, 4-element aligned output rows (D % 4 == 0 under VBE)."""

    def check():
        m = meta.cpu()
        dims = m[:, cute_tbe.META_D].tolist()
        tbl = m[:, cute_tbe.META_TABLE_OFFSET].tolist()
        outs = m[:, cute_tbe.META_OUT_OFFSET].tolist()
        return (
            all(d % 2 == 0 and o % 2 == 0 for d, o in zip(dims, tbl))
            and all(o % 4 == 0 for o in outs)
            and (all(d % 4 == 0 for d in dims) if vbe else total_D % 4 == 0)
        )

    if weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    if output.dtype != torch.float32:
        return False
    return _cached(("fwd", _key(meta), total_D, vbe), check)


def _short_ok(embedding_dims, num_short, block_size: int) -> bool:
    """Whether every feature this short-run launch covers suits the CuTe
    backward (D % 8 == 0, D <= 256). With dimension buckets Triton passes
    num_short as a slice of a per-bucket counter, and the launch covers the
    features whose max(32, next_pow2(D)) is its BLOCK_SIZE; otherwise it covers
    every feature."""
    bucketed = num_short.untyped_storage().nbytes() > num_short.element_size()

    def check():
        dims = embedding_dims.tolist()
        if bucketed:
            dims = [d for d in dims if max(32, 1 << (d - 1).bit_length()) == block_size]
        return block_size <= 256 and all(d % 8 == 0 for d in dims)

    return _cached(("short", _key(embedding_dims), block_size, bucketed), check)


def _has_rest(dims) -> bool:
    """Any feature TbeLongRunAccum leaves to Triton (D % 4 != 0, or D > 128 and
    D % 8 != 0)."""
    return _cached(
        ("rest", _key(dims)),
        lambda: any(d % 4 or (d > 128 and d % 8) for d in dims.tolist()),
    )


class _Launcher:
    """Stands in for one Triton kernel (``self.triton``, the fallback).
    ``specialized``: whether Triton launches it through _specialize_weight_ptrs
    (the fallback must then be the 1-weight specialization)."""

    def __init__(self, triton_kernel: Any, specialized: bool = True) -> None:
        self.triton = triton_kernel
        self.specialized = specialized

    def specialize_weight_ptrs(self, num_weight_ptrs: int) -> Any:
        """For ttbe._specialize_weight_ptrs: the CuTe kernels take one weight
        tensor; split weights run the Triton kernel."""
        if num_weight_ptrs == 1:
            return self
        return ttbe._specialize_weight_ptrs(self.triton, num_weight_ptrs)

    def __getitem__(self, grid: Any) -> Any:
        return partial(self._launch, grid)

    def _fallback(self, grid: Any, *args: Any, **kw: Any) -> None:
        kernel = self.triton
        if self.specialized:
            kernel = ttbe._specialize_weight_ptrs(kernel, 1)
        kernel[grid](*args, **kw)

    def _launch(self, grid: Any, *args: Any, **kw: Any) -> None:
        raise NotImplementedError


class _Forward(_Launcher):
    """table_batched_embedding_bag_forward_unweighted_kernel."""

    def _launch(self, grid, *args, **kw):
        (output, indices, offsets, weight, table_offsets, embedding_dims) = args[:6]
        (embedding_offsets, feature_table_map, rows_cumsum, warn) = args[6:10]
        (row_output_offsets, B_offsets, total_D, B, T) = args[10:15]
        vbe = bool(kw.get("vbe"))
        meta = _meta(  # hash_size_cumsum is not needed in forward
            table_offsets,
            embedding_dims,
            embedding_offsets,
            feature_table_map,
            None,
            rows_cumsum,
        )
        if not _fwd_ok(weight, output, meta, total_D, vbe):
            return self._fallback(grid, *args, **kw)
        fs = kw.get("FEATURE_START", 0)
        fe = kw.get("FEATURE_END", -1)
        fe = T if fe < 0 else fe
        cute_tbe.tbe_forward(  # meta rows carry their global feature index
            weight,
            indices,
            offsets,
            meta[fs:fe],
            B,
            fe - fs,
            total_D,
            out=output,
            warn=warn if kw.get("FUSED_BOUNDS_CHECK") else None,
            vbe_B_offsets=B_offsets if vbe else None,
            vbe_row_output_offsets=row_output_offsets if vbe else None,
        )


class _WeightedForward(_Launcher):
    """table_batched_embedding_bag_forward_weighted_kernel."""

    def _launch(self, grid, *args, **kw):
        (output, indices, offsets, weight, _chunk_starts, _split_starts) = args[:6]
        (table_offsets, embedding_dims, embedding_offsets, ftm) = args[6:10]
        (per_sample_weights, rows_cumsum, warn, row_output_offsets) = args[10:14]
        (b_t_map, total_D, B) = args[14:17]
        vbe = bool(kw.get("vbe"))
        T = embedding_dims.numel()
        meta = _meta(
            table_offsets, embedding_dims, embedding_offsets, ftm, None, rows_cumsum
        )
        if per_sample_weights.dtype != torch.float32 or not _fwd_ok(
            weight, output, meta, total_D, vbe
        ):
            return self._fallback(grid, *args, **kw)
        B_offsets = None
        if vbe:
            # Triton's weighted kernel gets b_t_map, not B_offsets: b_t is
            # feature-major, so B_offsets[t] = first b_t of feature t.
            # (b_t_map is int32 holding a uint32: t's top bit can be the sign bit)
            feat = (b_t_map.to(torch.int64) & 0xFFFFFFFF) >> kw["info_B_num_bits"]
            B_offsets = torch.searchsorted(
                feat, torch.arange(T + 1, device=feat.device)
            ).to(torch.int32)
        cute_tbe.tbe_forward(
            weight,
            indices,
            offsets,
            meta,
            B,
            T,
            total_D,
            out=output,
            warn=warn if kw.get("FUSED_BOUNDS_CHECK") else None,
            per_sample_weights=per_sample_weights,
            vbe_B_offsets=B_offsets,
            vbe_row_output_offsets=row_output_offsets if vbe else None,
        )


class _ShortRun(_Launcher):
    """triton_tbe_backward_short_run_{unweighted,weighted}. Weighted launches
    take the sorted per-sample weights after rows_cumsum."""

    def __init__(self, triton_kernel: Any, weighted: bool) -> None:
        super().__init__(triton_kernel)
        self.weighted = weighted

    def _launch(self, grid, dout, *args, **kw):
        rest = list(args)
        psw = rest.pop(14) if self.weighted else None
        (weight, _cs, _ss, infos, lin_run, cum, run_ids, tbl_off, dims) = rest[:9]
        (embo, ftm, hsc, momentum, rows_cumsum, num_short) = rest[9:15]
        (vbe_rows, vbe_B, total_D, B, lr, eps, optimizer) = rest[15:22]
        block = kw.get("BLOCK_SIZE", 0)
        if (
            optimizer != _ROWWISE_ADAGRAD
            or kw.get("STOCHASTIC_ROUNDING")
            or dout.dtype != torch.float32
            or not _short_ok(dims, num_short, block)
        ):
            return self._fallback(grid, dout, *args, **kw)
        vbe = bool(kw.get("vbe"))
        cute_tbe.tbe_backward_short_runs(
            dout,
            weight,
            momentum,
            infos,
            lin_run,
            cum,
            run_ids,
            num_short,
            _meta(tbl_off, dims, embo, ftm, hsc, rows_cumsum),
            total_D,
            kw["info_B_num_bits"],
            kw["info_B_mask"],
            lr,
            eps,
            B=B,
            block_size=block,
            per_sample_weights=psw,
            vbe_B_offsets=vbe_B if vbe else None,
            vbe_row_output_offsets=vbe_rows if vbe else None,
        )


class _LongRun(_Launcher):
    """triton_tbe_backward_long_run_grad_accum_unweighted (fused=False) and the
    CLC triton_tbe_backward_long_run_fused_unweighted (fused=True, accumulate +
    apply): cute_tbe.tbe_long_run reorders and accumulates the chunks, Triton's
    accumulate takes the chunks it leaves (if any table has such a D), then the
    fused entry point applies with cute_tbe.tbe_long_run_apply."""

    def __init__(self, triton_kernel: Any, fused: bool, accum: Any) -> None:
        super().__init__(triton_kernel, specialized=fused)
        self.fused = fused
        self.accum = accum  # Triton's unfused accumulate, for the left chunks

    def _launch(self, grid, *args, **kw):
        vbe = bool(kw.get("vbe"))
        bits, mask = kw["info_B_num_bits"], kw["info_B_mask"]
        if self.fused:
            (dout, infos, ss, se, gid, temp, counter, dims, embo, w, _cs, _ws) = args[
                :12
            ]
            lin_run, cum, orig_ids, tbl_off, ftm, hsc, mom, rcs = args[12:20]
            max_long_runs, num, vr, vb, total_D, lr, eps, optim = args[20:28]
            if optim != _ROWWISE_ADAGRAD or kw.get("STOCHASTIC_ROUNDING"):
                return self._fallback(grid, *args, **kw)
        else:
            (dout, infos, ss, se, gid, temp, dims, embo, num, vr, vb, total_D) = args
        if dout.dtype != torch.float32:
            return self._fallback(grid, *args, **kw)
        perm, n_rest = cute_tbe.tbe_long_run(
            dout,
            infos,
            ss,
            se,
            gid,
            temp,
            dims,
            embo,
            num,
            total_D,
            bits,
            mask,
            vbe_B_offsets=vb if vbe else None,
            vbe_row_output_offsets=vr if vbe else None,
        )
        if _has_rest(dims):
            # Chunks CuTe leaves sort last: perm[num - n_rest : num].
            tail = perm[
                (torch.arange(perm.numel(), device=perm.device) + num - n_rest)
                % perm.numel()
            ]
            self.accum[(grid[0],)](
                dout,
                infos,
                ss[tail],
                se[tail],
                gid[tail],
                temp,
                dims,
                embo,
                n_rest,
                vr,
                vb,
                total_D,
                BLOCK_SIZE=kw["BLOCK_SIZE"],
                info_B_num_bits=bits,
                info_B_mask=mask,
                num_warps=kw.get("num_warps", 1),
                vbe=vbe,
                ENABLE_TRITON_TBE_OPTIMIZATIONS=kw.get(
                    "ENABLE_TRITON_TBE_OPTIMIZATIONS", True
                ),
            )
        if self.fused:
            num_long_runs = (counter[:max_long_runs] > 0).sum(dtype=torch.int32).view(1)
            cute_tbe.tbe_long_run_apply(
                w,
                temp,
                lin_run,
                cum,
                infos,
                orig_ids,
                num_long_runs,
                tbl_off,
                dims,
                ftm,
                hsc,
                mom,
                rcs,
                bits,
                lr,
                eps,
            )


class _LongRunApply(_Launcher):
    """triton_tbe_backward_long_run_apply_unweighted (non-fused path)."""

    def _launch(self, grid, *args, **kw):
        # Triton's (w, chunk_starts, split_starts, temp, lin_run, cum, infos,
        # orig_ids, num, tbl_off, dims, ftm, hsc, mom, rcs, lr, eps, optim).
        if args[17] != _ROWWISE_ADAGRAD or kw.get("STOCHASTIC_ROUNDING"):
            return self._fallback(grid, *args, **kw)
        bits = kw["info_B_num_bits"]
        cute_tbe.tbe_long_run_apply(args[0], *args[3:15], bits, *args[15:17])


_launchers: Dict[Any, _Launcher] = {}


def launcher(triton_kernel: Any) -> Any:
    """The CuTe launcher standing in for triton_kernel (the kernel itself if
    there is none)."""
    if not _launchers:
        accum = ttbe.triton_tbe_backward_long_run_grad_accum_unweighted
        for launch in (
            _Forward(ttbe.table_batched_embedding_bag_forward_unweighted_kernel, False),
            _WeightedForward(ttbe.table_batched_embedding_bag_forward_weighted_kernel),
            _ShortRun(ttbe.triton_tbe_backward_short_run_unweighted, weighted=False),
            _ShortRun(ttbe.triton_tbe_backward_short_run_weighted, weighted=True),
            _LongRun(accum, fused=False, accum=accum),
            _LongRun(ttbe.triton_tbe_backward_long_run_fused_unweighted, True, accum),
            _LongRunApply(ttbe.triton_tbe_backward_long_run_apply_unweighted),
        ):
            _launchers[launch.triton] = launch
    return _launchers.get(triton_kernel, triton_kernel)
