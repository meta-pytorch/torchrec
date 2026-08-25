#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""Fused jagged pooled (gather + sum) TPU embedding lookup on the SparseCore (v2).

This version is fully offset-driven.

The only host-side work is a `concat` padding `offsets` up to the grid size (+ window slack) and
a small tail pad on `indices` so the last aligned chunk read stays in bounds -- the out-of-bag
ids such reads gather are masked out in the reduction.

"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc


# NL: SparseCore vector lane (for v7)
#    emb_dim must be a multiple of NL, and int32 vector loads must be exactly (NL,)
# NACC: independent accumulators for pooling
# TILE: SparseCore id-tile that HBM dim-0 indices must align to.
# CHUNK: Number of ids gathered per step
# RB: batches per step in subcore
NL, NACC, TILE, CHUNK, RB = 16, 16, 8, 16, 8


@jax.jit(static_argnames=["emb_dim"])  # pyre-ignore[20]
def run_sc_pooled_lookup(  # noqa: C901
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    emb_dim: int,
) -> jax.Array:
    assert emb_dim % NL == 0, f"emb_dim={emb_dim} must be a multiple of {NL}"
    assert (sc := pltpu.get_tpu_info().sparse_core)  # pyre-ignore[16]
    num_sub = sc.num_cores * sc.num_subcores
    num_bags = offsets.shape[0] - 1
    n_slices = emb_dim // NL

    num_blocks = (num_bags + RB - 1) // RB
    nb = (num_blocks + num_sub - 1) // num_sub
    grid_blocks = nb * num_sub
    grid_rows = grid_blocks * RB

    # Host op #1: pad offsets so (a) the bag count reaches grid_rows and (b) each block's
    # NL-wide offset-window DMA stays in bounds.
    pad_off = grid_rows + NL - offsets.shape[0]
    offsets_p = jnp.concatenate(
        [offsets, jnp.full((pad_off,), offsets[-1], offsets.dtype)]
    )
    # Host op #2: tail-pad the flat ids so a bag's last TILE-aligned chunk
    indices_p = jnp.pad(indices, (0, CHUNK), constant_values=0)

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyre-ignore[28]
    )

    # pyre-ignore[16]
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((grid_rows, emb_dim), dev_weights.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyre-ignore[28]
            needs_layout_passes=False,  # pyre-ignore[28]
        ),
        scratch_types=[
            pltpu.VMEM(
                (NL,), offsets.dtype
            ),  # NL-wide offset window (start/end extracted)
            pltpu.VMEM((2, CHUNK), indices.dtype),  # ping-pong staged gather indices
            pltpu.VMEM(
                (2, CHUNK, emb_dim), dev_weights.dtype
            ),  # ping-pong gathered rows
            pltpu.VMEM((RB, emb_dim), dev_weights.dtype),  # RB pooled rows
            pltpu.SemaphoreType.DMA((2,)),  # per-slot gather completion
        ],
        mesh=mesh,
    )
    def _kernel(
        weights_hbm, indices_hbm, offsets_hbm, out_hbm, off_v, idx_v, buf, acc_v, sem
    ):
        @functools.partial(
            pltpu.emit_pipeline,
            grid=(nb, num_sub),
            core_axis_name=("c", "s"),
            dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
        )
        def _inner():
            block_id = pl.program_id(0) * num_sub + pl.program_id(1)
            base_row = pl.multiple_of(block_id * RB, TILE)

            # One aligned DMA for the block's RB+1 offsets; extract per-bag scalars in-register.
            pltpu.sync_copy(offsets_hbm.at[pl.ds(base_row, NL)], off_v)
            ow = off_v[...]

            for j in range(RB):
                for s in range(n_slices):
                    acc_v[j, pl.ds(s * NL, NL)] = jnp.zeros((NL,), acc_v.dtype)

            def _gather(base, slot):
                base = pl.multiple_of(base, TILE)
                pltpu.sync_copy(indices_hbm.at[pl.ds(base, CHUNK)], idx_v.at[slot])
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(idx_v.at[slot])],
                    buf.at[slot],
                    sem.at[slot],
                ).start()

            def _wait(slot):
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(idx_v.at[slot])],
                    buf.at[slot],
                    sem.at[slot],
                ).wait()

            for j in range(RB):
                start, end = ow[j], ow[j + 1]
                aligned = (start // TILE) * TILE
                num_chunks = jnp.where(
                    end == start, 0, (end - aligned + CHUNK - 1) // CHUNK
                )

                @pl.when(num_chunks > 0)
                def _(aligned=aligned):
                    _gather(aligned, 0)

                @pl.loop(0, num_chunks)
                def _chunk(
                    c, j=j, start=start, end=end, aligned=aligned, num_chunks=num_chunks
                ):
                    slot = jax.lax.rem(c, 2)

                    @pl.when(c + 1 < num_chunks)
                    def _(c=c, aligned=aligned):
                        _gather(aligned + (c + 1) * CHUNK, jax.lax.rem(c + 1, 2))

                    _wait(slot)

                    base = aligned + c * CHUNK
                    one = jnp.ones((NL,), acc_v.dtype)
                    zero = jnp.zeros((NL,), acc_v.dtype)
                    masks = [
                        jnp.where((base + r >= start) & (base + r < end), one, zero)
                        for r in range(CHUNK)
                    ]

                    @plsc.parallel_loop(0, emb_dim, step=NL)
                    def _cols(c0, j=j, slot=slot, masks=masks):
                        cs = pl.ds(c0, NL)
                        accs = [buf[slot, r, cs] * masks[r] for r in range(NACC)]
                        for r in range(NACC, CHUNK):
                            accs[r % NACC] = (
                                accs[r % NACC] + buf[slot, r, cs] * masks[r]
                            )
                        acc = acc_v[j, cs]
                        for a in range(NACC):
                            acc = acc + accs[a]
                        acc_v[j, cs] = acc

            pltpu.sync_copy(acc_v, out_hbm.at[pl.ds(base_row, RB)])

        _inner()

    return _kernel(dev_weights, indices_p, offsets_p)[:num_bags, :]


def embedding_pooled_lookup_jax(
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    emb_dim: int,
) -> jax.Array:
    return run_sc_pooled_lookup(indices, offsets, dev_weights, emb_dim)


def embedding_pooled_lookup_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    """Dense full-table gradient on the TensorCore (mirrors single_lookup's backward).

    sum-pool backward is `grad_weight[indices[i]] += grad_out[bag(i)]`. Each flat id's bag is
    found from `offsets` with a searchsorted (no host `repeat_interleave`), the per-bag grad is
    expanded to per-id, then scatter-added into a zero `[num_rows, emb_dim]` table gradient.
    """
    total_ids = indices.shape[0]
    # bag(i) = # of offset boundaries <= i; side="right" puts a bag's first id in that bag.
    seg = jnp.searchsorted(
        offsets[1:], jnp.arange(total_ids, dtype=offsets.dtype), side="right"
    )
    grad_per_id = grad_out[seg]  # [total_ids, emb_dim]
    return jnp.zeros((num_rows, emb_dim), grad_out.dtype).at[indices].add(grad_per_id)
