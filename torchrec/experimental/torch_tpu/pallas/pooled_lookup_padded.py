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


NL, NACC = 16, 16


def _rblk(K, D):
    """Rows-per-block: largest power-of-two (<=4) bag block whose double-buffered
    gather buffer stays under the ~500 KiB VMEM budget."""
    r = 1
    while 2 * (r * 2) * K * D * 4 <= 500 * 1024 and r * 2 <= 4:
        r *= 2
    return r


# Fused SparseCore gather-sum kernel (from Google/torus bench_tpu_gather_pool_pallas_sc)
def _kernel(idx_hbm, table_hbm, out_hbm, idx_v, buf, ov, sem, *, RBLK, K):
    sc = pltpu.get_tpu_info().sparse_core
    num_sub = sc.num_cores * sc.num_subcores
    B, D, TILE = out_hbm.shape[0], buf.shape[-1], RBLK * K
    nb = pl.cdiv(B, RBLK * num_sub)

    @functools.partial(
        pltpu.emit_pipeline,
        grid=(nb + 1, num_sub),
        core_axis_name=("c", "s"),
        dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
    )
    def _inner():
        blk, core = pl.program_id(0), pl.program_id(1)

        @pl.when(blk < nb)  # gather block blk
        def _():
            p = jax.lax.rem(blk, 2)
            o0 = (blk * num_sub + core) * RBLK
            pltpu.sync_copy(idx_hbm.at[pl.ds(o0 * K, TILE)], idx_v.at[p])
            pltpu.make_async_copy(
                table_hbm.at[plsc.Indices(idx_v.at[p])], buf.at[p], sem.at[p]
            ).start()

        @pl.when(blk > 0)  # reduce block blk-1 (overlaps gather)
        def _():
            q = jax.lax.rem(blk - 1, 2)
            o0 = ((blk - 1) * num_sub + core) * RBLK
            pltpu.make_async_copy(
                table_hbm.at[plsc.Indices(idx_v.at[q])], buf.at[q], sem.at[q]
            ).wait()
            for r in range(RBLK):
                base = r * K

                @plsc.parallel_loop(0, D, step=NL)
                def _cols(c0, base=base, r=r):  # bind loop vars (flake8 B023)
                    cs = pl.ds(c0, NL)
                    accs = [buf[q, base + a, cs] for a in range(NACC)]
                    for k in range(NACC, K):
                        accs[k % NACC] = accs[k % NACC] + buf[q, base + k, cs]
                    acc = accs[0]
                    for a in range(1, NACC):
                        acc = acc + accs[a]
                    ov[r, cs] = acc

            pltpu.sync_copy(ov, out_hbm.at[pl.ds(o0, RBLK)])

    _inner()


def build(B, K, D):
    """f(table[V, D] f32, idx[B*K] i32) -> out[B, D] f32 (fused gather + sum-pool).

    Returns the plain JAX function (not jitted) so ``jax_op`` can trace/export it.
    """
    RBLK = _rblk(K, D)
    sc = pltpu.get_tpu_info().sparse_core
    num_sub = sc.num_cores * sc.num_subcores
    Bpad = pl.cdiv(B, RBLK * num_sub) * RBLK * num_sub
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
    )
    ker = pl.kernel(
        functools.partial(_kernel, RBLK=RBLK, K=K),
        out_type=jax.ShapeDtypeStruct((Bpad, D), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,
            needs_layout_passes=False,
        ),
        scratch_types=[
            pltpu.VMEM((2, RBLK * K), jnp.int32),
            pltpu.VMEM((2, RBLK * K, D), jnp.float32),
            pltpu.VMEM((RBLK, D), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        mesh=mesh,
    )

    def run(table, idx):
        idxp = idx if Bpad == B else jnp.pad(idx, (0, (Bpad - B) * K))
        return ker(idxp, table)[:B]

    return run


def embedding_pooled_lookup_jax(
    idx: jax.Array, table: jax.Array, pool: int
) -> jax.Array:
    """Fused uniform gather + sum-pool: idx[B*pool] i32, table[V, D] f32 -> [B, D]."""
    B = idx.shape[0] // pool
    D = table.shape[1]
    return build(B, pool, D)(table, idx)


def embedding_pooled_gather_bwd_jax(
    grad_out: jax.Array, idx: jax.Array, pool: int, num_rows: int, emb_dim: int
) -> jax.Array:
    grad_expanded = jnp.repeat(
        grad_out, pool, axis=0
    )  # [B*pool, D]; row j -> grad_out[j//pool]
    return (
        jnp.zeros((num_rows, emb_dim), dtype=grad_out.dtype).at[idx].add(grad_expanded)
    )
