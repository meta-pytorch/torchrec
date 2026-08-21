#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SparseCore recat kernel for the KJT post-all2all reconstruction.

After the TW/CW input all2all, `KeyedJaggedTensor.dist_init` reorders the received
tensors from rank-major `[T_local*W, B]` to feature-major `[T_local, W*B]` via
`fbgemm.permute_2D_sparse_data` (a variable-length segment gather). That op has no
TPU kernel, so today it runs on CPU (values round-trip TPU->CPU->TPU every step —
~100MB/rank at MLPerf-16chip scale).

This replaces the values (and weights) segment-gather with a **SparseCore** flat
gather, keeping the id payload on-device. The gather index and permuted lengths are
compile-time constant for FIXED multi-hot sizes (the MLPerf regime), so they are
computed once (from the first batch's lengths), moved to the device, and cached; per
step only the SparseCore value gather runs.

Public entry point: `sc_recat_tensors(output_tensors, recat, stride_per_rank)` ->
the feature-major `[lengths, values, (weights)]` to hand to `dist_init(recat=None)`.
"""

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
from torch_tpu._internal import pallas

# Cache: signature -> (gather_idx on device, permuted_lengths on device). The recat
# permutation and per-segment sizes are static for fixed multi-hot, so a plan
# computed from the first batch is reused for every subsequent step.
_PLAN_CACHE: Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_params_sc(n: int, num_subcores: int) -> Tuple[int, int, int]:
    """Chunk/grid sizing for a width-1 SparseCore gather.

    A width-1 output row is padded to the SC vector lane width (LANE) in tile_spmem,
    so an `(chunk, 1)` VMEM tile actually costs `chunk*LANE` words. Budget the chunk
    against the ~131071-word SPMEM limit with double-buffering (idx + out) and margin,
    else the pipeline hits CompileTimeSparseCoreAllocationFailure.
    """
    TILE = 8
    LANE = 16
    SPMEM_WORDS = 131072
    # 4x = double-buffered (2x) idx+out tiles, each padded to LANE, plus headroom.
    max_chunk = SPMEM_WORDS // (4 * (LANE + 1))
    indices_chunk = min(max(n, 1), max_chunk)
    indices_chunk = max((indices_chunk // TILE) * TILE, TILE)
    grid_size = (n + indices_chunk - 1) // indices_chunk
    grid_size = ((grid_size + num_subcores - 1) // num_subcores) * num_subcores
    return indices_chunk, grid_size, grid_size * indices_chunk


@jax.jit
def _run_sc_gather(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather rows of ``values`` ([N, LANE]) by ``indices`` ([M]) on the SparseCore.

    Returns [M, LANE] of ``values.dtype``. ``out[k] = values[indices[k]]``. Rows must
    be the SC vector-lane width (a width-1 gather silently returns zeros).
    """
    # pyre-ignore[16]: the buck-pinned jax stubs predate the SparseCore Pallas
    # API that the pod's jax exposes.
    assert (sc_info := pltpu.get_tpu_info().sparse_core)
    # pyre-ignore[20]: `num_cores` is defaulted in the pod's jax, not in the stubs.
    vector_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore"
    )
    sc_num_subcores = sc_info.num_subcores

    n = indices.shape[0]
    width = values.shape[1]
    indices_chunk, grid_size, padded_n = _get_params_sc(n, sc_num_subcores)
    if padded_n > n:
        indices = jnp.pad(indices, (0, padded_n - n), constant_values=0)

    # pyre-ignore[16]: `pl.kernel` exists in the pod's jax, not in the stubs.
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((padded_n, width), values.dtype),
        mesh=vector_mesh,
    )
    def _kernel(values_hbm, indices_hbm, out_hbm):
        def body(idx_smem, out_vmem):
            pltpu.sync_copy(values_hbm.at[idx_smem], out_vmem)

        pltpu.emit_pipeline(
            body,
            grid=(grid_size,),
            in_specs=[
                pl.BlockSpec((indices_chunk,), lambda i: (i,), memory_space=pltpu.VMEM)
            ],
            out_specs=[pl.BlockSpec((indices_chunk, width), lambda i: (i, 0))],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(indices_hbm, out_hbm)

    return _kernel(values, indices)[:n, :]


# torch op over the SparseCore gather (no autograd: recat runs on ids under no_grad).
_recat_gather_fn = pallas.jax_op("pallas::recat_gather", _run_sc_gather)


def _sc_gather_1d(values: torch.Tensor, gather_idx: torch.Tensor) -> torch.Tensor:
    """``values[gather_idx]`` on the SparseCore, restoring the input dtype.

    The gather runs in int32 (SparseCore ids are int32, matching the unfused lookup
    kernel), so the op has a single, stable specialization regardless of whether the
    KJT ships int32 or int64 ids. Ids exceed int32 only above ~2.1B rows.
    """
    orig_dtype = values.dtype
    # A width-1 SparseCore gather silently returns zeros, so pad each id to the
    # 16-wide SC vector lane (value in column 0) and take column 0 back. Gather in
    # float32 (the proven run_sc_lookup dtype); exact for ids < 2**24 (shrunk).
    lane = 16
    n = values.numel()
    padded = torch.zeros(n, lane, dtype=torch.float32, device=values.device)
    padded[:, 0] = values.view(-1).to(torch.float32)
    out = _recat_gather_fn(
        values=padded,
        indices=gather_idx.to(dtype=torch.int32, device=values.device),
    )
    return out[:, 0].round().to(orig_dtype)


def _tc_gather_1d(values: torch.Tensor, gather_idx: torch.Tensor) -> torch.Tensor:
    """``values[gather_idx]`` on the TensorCore via a native width-1 gather.

    A plain `index_select` lowers to an optimized TensorCore gather — width-1
    native, so (unlike the SparseCore path) there is no 16-lane padding / 16x
    bandwidth waste. Keeps the id payload on-device (no CPU round-trip).
    """
    return values.view(-1).index_select(
        0, gather_idx.to(dtype=torch.int64, device=values.device)
    )


def _compute_recat_plan(
    lengths: torch.Tensor, recat: torch.Tensor, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the value gather-index and permuted (feature-major) lengths on CPU.

    Mirrors `permute_2D_sparse_data(recat, lengths.view(T*W, stride), values)`:
    rows are reordered by ``recat`` and each row carries its contiguous value block.
    Done once per signature (static for fixed multi-hot), so CPU cost is amortized.
    """
    lengths_cpu = lengths.detach().to("cpu")
    recat_cpu = recat.detach().to("cpu").to(torch.int64)
    tw = recat_cpu.numel()
    lengths_2d = lengths_cpu.view(tw, stride)

    # Feature-major lengths: rows reordered by recat, flattened back.
    permuted_lengths = lengths_2d[recat_cpu].reshape(-1).to(lengths.dtype)

    # Per-source-row value block sizes and start offsets (rank-major layout).
    row_size = lengths_2d.sum(dim=1).to(torch.int64)  # [T*W]
    src_off = torch.zeros(tw + 1, dtype=torch.int64)
    torch.cumsum(row_size, dim=0, out=src_off[1:])

    # For each output row k (feature-major), its block gathers source row recat[k].
    psize = row_size[recat_cpu]  # [T*W]
    starts = src_off[recat_cpu]  # [T*W]
    total = int(psize.sum().item())
    out_row = torch.repeat_interleave(torch.arange(tw), psize)  # [total]
    block_start = torch.cumsum(psize, dim=0) - psize  # exclusive prefix in output order
    within = torch.arange(total) - block_start[out_row]  # position within block
    gather_idx = (starts[out_row] + within).to(torch.int32)

    device = lengths.device
    return gather_idx.to(device), permuted_lengths.to(device)


def recat_tensors(
    output_tensors: List[torch.Tensor],
    recat: torch.Tensor,
    stride_per_rank: List[int],
    mode: str = "sc",
) -> List[torch.Tensor]:
    """Feature-major `[lengths, values, (weights)]` via an on-device value gather.

    Drop-in for the CPU `permute_2D_sparse_data` recat on the fixed-stride TW/CW TPU
    path: pass the result to `dist_init(..., recat=None)` (already permuted). Only
    supports single (equal) stride per rank — the fixed-batch case. Both backends
    keep the id payload on-device (no CPU round-trip); they differ only in engine.

    Args:
        output_tensors: post-all2all `[lengths, values]` or `[lengths, values, weights]`.
        recat: rank-major -> feature-major segment permutation (`_get_recat`).
        stride_per_rank: per-rank batch sizes; all equal on this path.
        mode: "sc" (SparseCore gather) or "tc" (TensorCore native gather).

    Returns:
        `[permuted_lengths, permuted_values, (permuted_weights)]`, all on-device.
    """
    lengths = output_tensors[0]
    values = output_tensors[1]
    weights = output_tensors[2] if len(output_tensors) == 3 else None
    stride = stride_per_rank[0]

    sig = (
        str(values.device),
        int(recat.numel()),
        int(stride),
        int(lengths.numel()),
        int(values.numel()),
    )
    plan = _PLAN_CACHE.get(sig)
    if plan is None:
        plan = _compute_recat_plan(lengths, recat, stride)
        _PLAN_CACHE[sig] = plan
    gather_idx, permuted_lengths = plan

    gather = _tc_gather_1d if mode == "tc" else _sc_gather_1d
    permuted = [permuted_lengths, gather(values, gather_idx)]
    if weights is not None:
        permuted.append(gather(weights, gather_idx))
    return permuted
