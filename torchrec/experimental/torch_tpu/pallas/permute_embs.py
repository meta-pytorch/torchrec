#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc


@jax.jit(static_argnames=["col_block_size"])  # pyre-ignore[6]
def permute_pooled_embs_auto_grad_split_kernel(
    pooled_embs: jax.Array,  # row-major order
    offset_dim_list: jax.Array,
    permute_list: jax.Array,
    inv_offset_dim_list: jax.Array,
    # inv_permute_list: jax.Array,
    # allow_duplicates: bool,
    col_block_size: int,
) -> jax.Array:
    """Naive implementation to permute the embeddings."""
    if permute_list.shape[0] != offset_dim_list.shape[0] - 1:
        raise NotImplementedError("Subset permutations are not supported")
    _ = col_block_size
    T = permute_list.shape[0]
    B = pooled_embs.shape[0]
    D = pooled_embs.shape[1]

    sc = pltpu.get_tpu_info().sparse_core  # pyre-ignore[16]
    num_sub = sc.num_cores * sc.num_subcores
    UNIT = 8
    NL = 16

    perm = permute_list.astype(jnp.int32)
    src_start = offset_dim_list[perm].astype(jnp.int32)  # input col start per out-table
    src_end = offset_dim_list[perm + 1].astype(jnp.int32)
    n_units = ((src_end - src_start) // UNIT).astype(jnp.int32)  # UNIT-copies per table
    dst_start = inv_offset_dim_list[:T].astype(jnp.int32)  # output col start per table

    slot = jnp.arange(T) * NL
    src_start = jnp.zeros((T * NL,), jnp.int32).at[slot].set(src_start)
    dst_start = jnp.zeros((T * NL,), jnp.int32).at[slot].set(dst_start)
    n_units = jnp.zeros((T * NL,), jnp.int32).at[slot].set(n_units)

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="t",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyre-ignore[28]
    )
    grid = ((T * B + num_sub - 1) // num_sub,)

    def permute_kernel(
        src_start_hbm,
        dst_start_hbm,
        n_units_hbm,
        in_pooled_embs_hbm,
        out_pooled_embs_hbm,
        ss_v,
        ds_v,
        nu_v,
        vmem_in,
    ):
        s = jax.lax.axis_index("t") * sc.num_subcores + jax.lax.axis_index("s")

        @functools.partial(
            pltpu.emit_pipeline,
            grid=grid,
        )
        def _inner():
            k = pl.program_id(0) * num_sub + s

            @pl.when(k < T * B)
            def _kernel():
                id_table = k // B
                id_batch = k % B

                idx = pl.multiple_of(id_table * NL, 8)
                pltpu.sync_copy(src_start_hbm.at[pl.ds(idx, NL)], ss_v)
                pltpu.sync_copy(dst_start_hbm.at[pl.ds(idx, NL)], ds_v)
                pltpu.sync_copy(n_units_hbm.at[pl.ds(idx, NL)], nu_v)
                in_start = ss_v[...][0]
                out_start = ds_v[...][0]
                num_units = nu_v[...][0]

                @pl.loop(0, num_units)
                def _copy(u):
                    in_col = pl.multiple_of(in_start + u * UNIT, UNIT)
                    out_col = pl.multiple_of(out_start + u * UNIT, UNIT)
                    pltpu.sync_copy(
                        in_pooled_embs_hbm.at[pl.ds(id_batch, 1), pl.ds(in_col, UNIT)],
                        vmem_in,
                    )
                    pltpu.sync_copy(
                        vmem_in,
                        out_pooled_embs_hbm.at[
                            pl.ds(id_batch, 1), pl.ds(out_col, UNIT)
                        ],
                    )

        _inner()

    ker = pl.kernel(  # pyre-ignore[16]
        permute_kernel,
        out_type=jax.ShapeDtypeStruct((B, D), pooled_embs.dtype),
        scratch_types=[
            pltpu.VMEM((NL,), jnp.int32),  # src_start window
            pltpu.VMEM((NL,), jnp.int32),  # dst_start window
            pltpu.VMEM((NL,), jnp.int32),  # n_units window
            pltpu.VMEM((1, UNIT), pooled_embs.dtype),  # data copy buffer
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyre-ignore[28]
            needs_layout_passes=False,  # pyre-ignore[28]
        ),
        mesh=mesh,
    )
    permuted_embeddings = ker(src_start, dst_start, n_units, pooled_embs)
    return permuted_embeddings


# -------------------------
# TensorCore implementation
# --------------------------


def _permute_col_index(
    offset_dim_list: jax.Array, permute_list: jax.Array, D: int
) -> jax.Array:
    """Per-output-column source index (length D)."""
    in_off = offset_dim_list.astype(jnp.int32)  # [T+1] input block offsets
    perm = permute_list.astype(jnp.int32)
    dims = in_off[1:] - in_off[:-1]  # [T] input block widths
    out_off = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(dims[perm]).astype(jnp.int32)]
    )  # [T+1] output block offsets
    d_out = jnp.arange(D, dtype=jnp.int32)
    blk = jnp.searchsorted(out_off[1:], d_out, side="right")  # output block per col
    within = d_out - out_off[blk]  # column position within the block
    return in_off[perm[blk]] + within


def permute_pooled_embs_tc(
    pooled_embs: jax.Array,
    offset_dim_list: jax.Array,
    permute_list: jax.Array,
) -> jax.Array:
    """Permute [B, D] pooled embeddings by reordering their column blocks."""
    if permute_list.shape[0] != offset_dim_list.shape[0] - 1:
        raise NotImplementedError("Subset permutations are not supported")
    D = pooled_embs.shape[1]
    col_perm = _permute_col_index(offset_dim_list, permute_list, D)
    return jnp.take(pooled_embs, col_perm, axis=1)


def permute_pooled_embs_tc_bwd(
    grad_out: jax.Array,
    offset_dim_list: jax.Array,
    permute_list: jax.Array,
) -> jax.Array:
    """Gradient w.r.t. the input: scatter-add grad_out back to its source columns."""
    if permute_list.shape[0] != offset_dim_list.shape[0] - 1:
        raise NotImplementedError("Subset permutations are not supported")
    B, D = grad_out.shape
    col_perm = _permute_col_index(offset_dim_list, permute_list, D)
    return jnp.zeros((B, D), grad_out.dtype).at[:, col_perm].add(grad_out)
