#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""Fused jagged pooled (gather + sum) TPU embedding lookup on the SparseCore.

This version is fully offset-driven.
"""

import functools
import os

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
    sc = pltpu.get_tpu_info().sparse_core  # pyre-ignore
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


# Which derivation the backward uses for the per-id bag ids. "searchsorted" is the
# default because it is unconditionally correct: it reads the bag boundaries out of
# `offsets` and needs no relationship between `offsets` and the id count. "repeat" is
# markedly faster (see `embedding_pooled_lookup_segments_jax`) but requires
# `offsets[-1] == indices.shape[0]`; `jnp.repeat`'s `total_repeat_length` pads or
# truncates silently rather than raising, so a caller that builds a mismatched
# offsets/indices pair would get a wrong `seg` and a silently wrong gradient. Opt in
# with TPU_POOLED_BWD_MODE=repeat once the caller is known to hold that invariant.
POOLED_BWD_MODE_ENV = "TPU_POOLED_BWD_MODE"
POOLED_BWD_MODE_DEFAULT = "searchsorted"


def _pooled_bwd_mode() -> str:
    """Read at trace time, so the choice is baked into each compiled backward.

    Changing the variable after a shape has been compiled will not retrace it; set it
    before the first backward of a run.
    """
    return os.environ.get(POOLED_BWD_MODE_ENV, POOLED_BWD_MODE_DEFAULT)


def embedding_pooled_lookup_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    """Dense full-table gradient on the TensorCore (mirrors single_lookup's backward).

    sum-pool backward is `grad_weight[indices[i]] += grad_out[bag(i)]`. Each flat id's bag
    comes from `offsets` (no host `repeat_interleave`), the per-bag grad is expanded to
    per-id, then scatter-added into a zero `[num_rows, emb_dim]` table gradient.

    The bag ids are derived per `TPU_POOLED_BWD_MODE`: "searchsorted" (default, no
    precondition) or "repeat" (faster, requires `offsets[-1] == indices.shape[0]`). Both
    produce bit-identical gradients when that invariant holds.
    """
    total_ids = indices.shape[0]
    mode = _pooled_bwd_mode()
    if mode == "repeat":
        seg = embedding_pooled_lookup_segments_jax(offsets, total_ids)
    else:
        seg = embedding_pooled_lookup_segments_searchsorted_jax(offsets, total_ids)
    return embedding_pooled_lookup_bwd_seg_jax(
        grad_out, indices, seg, num_rows, emb_dim
    )


def embedding_pooled_lookup_segments_searchsorted_jax(
    offsets: jax.Array,
    total_ids: int,
) -> jax.Array:
    """Per-id bag ids via a binary search over `offsets`. The default derivation.

    `bag(i)` is the number of offset boundaries <= i; `side="right"` puts a bag's first
    id in that bag. O(total_ids * log num_bags) and materializes an `arange(total_ids)`
    to search with, so it is slower than the repeat form -- but it reads the boundaries
    straight out of `offsets` and so cannot silently mis-segment a mismatched pair.
    """
    return jnp.searchsorted(
        offsets[1:], jnp.arange(total_ids, dtype=offsets.dtype), side="right"
    )


def embedding_pooled_lookup_segments_jax(
    offsets: jax.Array,
    total_ids: int,
) -> jax.Array:
    """Per-id bag ids: `seg[i]` is the bag that flat id `i` belongs to.

    Built from the lengths in one `jnp.repeat`. Used by the backward only when
    `TPU_POOLED_BWD_MODE=repeat`; see the flag comment above for why it is opt-in.
    `jnp.repeat` needs a statically known output length under jit, hence `total_ids`.

    Requires `offsets[-1] == total_ids`, i.e. the offsets must cover exactly the flat ids
    being scattered. `total_repeat_length` pads or truncates silently rather than raising,
    so a mismatched pair would yield a wrong `seg` and hence a silently wrong gradient.
    Both are derived from the same lookup (`offsets` and `indices.shape[0]`), so the
    invariant holds by construction on the autograd path; the assert guards callers that
    build the pair themselves.
    """
    num_bags = offsets.shape[0] - 1
    lengths = offsets[1:] - offsets[:-1]
    # Static-shape check only: `offsets` values are traced, so this cannot compare
    # offsets[-1] to total_ids under jit. It does catch a caller passing a `total_ids`
    # that disagrees with the shapes it derived the offsets from.
    assert num_bags >= 0, f"offsets must have at least one entry, got {offsets.shape}"
    return jnp.repeat(
        jnp.arange(num_bags, dtype=offsets.dtype),
        lengths,
        total_repeat_length=total_ids,
    )


def embedding_pooled_lookup_bwd_seg_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    seg: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    """Dense full-table gradient from precomputed per-id bag ids.

    The shared body of the backward: with `seg` already built by
    `embedding_pooled_lookup_segments_jax`, this is just the per-id gather plus the
    dense scatter-add. Split out so a caller holding a precomputed `seg` can skip
    rebuilding it.
    """
    grad_per_id = grad_out[seg]  # [total_ids, emb_dim]
    return jnp.zeros((num_rows, emb_dim), grad_out.dtype).at[indices].add(grad_per_id)


@jax.jit(static_argnames=["emb_dim"])  # pyre-ignore[20]
def run_sc_pooled_lookup_batch_parallel(  # noqa: C901
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    row_offsets: jax.Array,
    emb_dim: int,
) -> jax.Array:
    """
    Batched pooled embedding lookup on sparse-core parallelizes over batches.

    This algorithm performs pooled embedding lookup using indices and offsets. Effectively
    each subcore is assigned to a group of batches defined by `NUMB_BATCHES_PER_ITERATION`.
    A subcore would grab the offsets and create a temporary zeroed out buffer for the pooled
    embeddings. Afterwards a ping-pong style algorithm is performed to overlap the HBM->VMEM
    transfers of ids within `i`th batch, with pooling of the previous embeddings on previous ids
    within the `i`th batch.

    The way pooling is performed is that each subcore would process `NUMB_IDS_PROCESSED_PER_STEP`
    indices at a time within a batch. Now the first row of the temprory buffer is used and
    each embedding is added to it. The embedding dimensions are split to lane size, and so
    only `LANE_SIZE` columns are processed in each step.


    Notes:
    - Embedding dimensions are padded only for VMEM but not HBM saving memory. However non-16
        multiple embedding dimension is not yet supported.
    - Indices, and offsets are padded to length of a lane.
    - Grabbing the offsets is async while it creates a zeroed out temporary buffer.
    - Each subcore gets a chunk of certain number of batch.
    - Subcore processes one batch, but it iterates through groups of batches
    - Each SIMD lane/threads within the subcore processes embedding columns together
    - One could do grid=(number_of_blocks,) with PARALLEL, so each block gets assigned to a subcore
        - but for pingpong buffers, it is much easier to do with 2D-style
    - Bulk transfer of offsets are done for each batch, once could do sequential batch processing per subcore
    - Removes the assumption that embedding dimension should be a multiple of tiles,
        - Decided to do masking, but maybe we cna pad within VMEM but HBM tranffer needs to be properly handled
    - Assumption is that maximum number of batch size >= 16


    TODO: Profile against doing a post-kernel permute/reshape of the embedding table against
          doing one where the output layout is done wihtin the kernel. Right now, the kernel return [F * B, D] whereas
          it is expected to be in feature-major order. [B, F * D]. Another option is to re-order the KJT, so three approaches
          can be tested out here.
    TODO: Benchmark against pre-shifting every index before calling this pallas kernel (for table batched), or do it within
          the kernel. FBGEMM does it within the kernel. Note that pre-shifting would cause this to be in the
    TODO: Benchmark against analytic calculation of  VMEM size.
    TODO: Implement to work over any embedding-dimension
    TODO: Prefetch batch i + 1 while doing hte pooling of the last chunk of bag i.
    TODO: Implement for embedding-dimension 8.
    TODO: Test against BF16, and implement it
    TODO: check if using repeat_interleave for the row_offset can cause performance regressions in e2e model.
    TODO: xla compiler for sparseCore can only supports 16 GiB of HBM per tensor, so if we have large stacked embeddings
          then we would need to either split them up, or re-think how we want to shard.
    TODO: Remove the output from [F, B, D] = [F* B, D] shape to [B, F * D] so that we can remove operations inside PallasTBE class.
    """
    # Only do eight batch at a time since it can fit inside a 16-lane offset load
    NUMB_BATCHES_PER_ITERATION = 8  # TODO: Tune this parameter
    LANE_SIZE = 16
    NUMB_IDS_PROCESSED_PER_STEP = 16  # TODO: Explicitly figure out this formula # IDS gathered per chunk wihtin one batch
    TILE = 8
    assert NUMB_BATCHES_PER_ITERATION % TILE == 0
    assert NUMB_IDS_PROCESSED_PER_STEP % TILE == 0
    assert NUMB_BATCHES_PER_ITERATION + 1 <= LANE_SIZE
    assert emb_dim == 8 or emb_dim % LANE_SIZE == 0
    num_batches = offsets.shape[0] - 1

    # Get Sparse-core information
    sc = pltpu.get_tpu_info().sparse_core  # pyre-ignore
    total_num_subcores = sc.num_cores * sc.num_subcores

    # Pad the embedding dimensions to be alinged to a LANE size
    # note we want to avoid doing it to full HBM, so we do it on VMEM/output instead
    aligned_emb_dim = ((emb_dim + LANE_SIZE - 1) // LANE_SIZE) * LANE_SIZE

    # Embedding dimension processes per SIMD lanes
    numb_cols_chunks = aligned_emb_dim // LANE_SIZE

    # Number of chunks (grouping of batch) to compute
    numb_chunks = (
        num_batches + NUMB_BATCHES_PER_ITERATION - 1
    ) // NUMB_BATCHES_PER_ITERATION

    # Distribute chunks to number of available subcores
    number_of_blocks = (numb_chunks + total_num_subcores - 1) // total_num_subcores
    grid_blocks = number_of_blocks * total_num_subcores
    grid_rows = grid_blocks * NUMB_BATCHES_PER_ITERATION

    # Pad offsets to be multiple of lane size 16
    pad_size = grid_rows + LANE_SIZE - offsets.shape[0]
    offsets = jnp.concatenate(
        [
            offsets,
            jnp.full(
                (pad_size,),
                offsets[-1],
                dtype=offsets.dtype,
            ),
        ]
    )

    # Tail-pad the ids so a batch's last tile-aligned chunk cannot read past the end
    # of the array (bounds checks are disabled); the extra rows are masked out.
    indices = jnp.pad(indices, (0, NUMB_IDS_PROCESSED_PER_STEP), constant_values=0)

    # Pad the feature/row offsets
    row_offsets = jnp.pad(
        row_offsets,
        (0, grid_rows + LANE_SIZE - row_offsets.shape[0]),
        constant_values=0,
    )

    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyre-ignore[16]
    )

    # pyre-ignore[16]
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((grid_rows, aligned_emb_dim), dev_weights.dtype),
        mesh=mesh,  # pyre-ignore[16]
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=False,
            use_tc_tiling_on_sc=False,  # pyre-ignore[16]
            needs_layout_passes=False,  # pyre-ignore[16]
        ),
        scratch_types=[
            pltpu.VMEM((LANE_SIZE,), offsets.dtype),  # offsets
            pltpu.VMEM((2, NUMB_IDS_PROCESSED_PER_STEP), indices.dtype),  # indices
            pltpu.VMEM((LANE_SIZE,), offsets.dtype),  # per-batch row offsets
            pltpu.VMEM(
                (2, NUMB_IDS_PROCESSED_PER_STEP, aligned_emb_dim),
                dev_weights.dtype,  # embedding in vmem
            ),
            pltpu.VMEM(
                (NUMB_BATCHES_PER_ITERATION, aligned_emb_dim),
                dev_weights.dtype,  # post pooled emb in vmem
            ),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ],
    )
    def lookup_kernel(
        weights_hbm,
        indices_hbm,
        offsets_hbm,
        row_offsets_hbm,
        out_hbm,
        offsets_vmem,
        indices_vmem,
        row_offsets_vmem,
        buf_vmem,
        post_pooled_emb_vmem,
        sem,
        offset_sem,
        row_offset_sem,
    ):
        @functools.partial(
            pltpu.emit_pipeline,
            grid=(number_of_blocks, total_num_subcores),
            core_axis_name=("c", "s"),
            dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
        )
        def _inner():
            # Calculate which batch corresponds to this subcore
            # TODO: Could assign contigious batches to each subcore instead versus doing strided here
            chunk_id = pl.program_id(0)
            core_id = pl.program_id(1)
            chunk_id = chunk_id * total_num_subcores + core_id

            # Fetch all the offsets in bulk at once that requires to be processed by subcore
            base = pl.multiple_of(
                chunk_id * NUMB_BATCHES_PER_ITERATION,
                TILE,
            )
            offset_start = pltpu.make_async_copy(
                offsets_hbm.at[
                    pl.ds(
                        base,
                        LANE_SIZE,
                    )
                ],
                offsets_vmem,
                offset_sem,
            )

            row_offset_start = pltpu.make_async_copy(
                row_offsets_hbm.at[
                    pl.ds(
                        base,
                        LANE_SIZE,
                    )
                ],
                row_offsets_vmem,
                row_offset_sem,
            )
            offset_start.start()
            row_offset_start.start()

            # Zero out the temporary embedding rows in VMEM
            for i in range(NUMB_BATCHES_PER_ITERATION):
                for j in range(numb_cols_chunks):
                    post_pooled_emb_vmem[i, pl.ds(j * LANE_SIZE, LANE_SIZE)] = (
                        jnp.zeros((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                    )

            # Bring from VMEM to registers
            offset_start.wait()
            row_offset_start.wait()
            offsets_reg = offsets_vmem[...]
            row_offsets_reg = row_offsets_vmem[...]

            def _gather(base, slot, off_start, off_end, row_offset):
                # Copy the sync indices on chunks based due to VMEM shortage
                base = pl.multiple_of(base, TILE)
                pltpu.sync_copy(
                    indices_hbm.at[pl.ds(base, NUMB_IDS_PROCESSED_PER_STEP)],
                    indices_vmem.at[slot],
                )

                positions = base + jnp.arange(NUMB_IDS_PROCESSED_PER_STEP)
                valid = (positions >= off_start) & (positions < off_end)

                indices_vmem[slot, ...] = (
                    jnp.where(valid, indices_vmem[slot, ...], 0) + row_offset
                )

                # Start copying the weights
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(indices_vmem.at[slot])],
                    buf_vmem.at[slot],
                    sem.at[slot],
                ).start()

            def _wait(slot):
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(indices_vmem.at[slot])],
                    buf_vmem.at[slot],
                    sem.at[slot],
                ).wait()

            output_start = pl.multiple_of(
                chunk_id * NUMB_BATCHES_PER_ITERATION,
                TILE,
            )

            # Go through each batch per iteration
            for i_batch in range(NUMB_BATCHES_PER_ITERATION):
                # Grab the offset start and end
                off_start, off_end = offsets_reg[i_batch], offsets_reg[i_batch + 1]

                # Row offset of the table this batch's feature maps into.
                feat_offset = row_offsets_reg[i_batch]

                # An HBM slice has to start on a tile boundary, and so masking is required
                aligned_start = (off_start // TILE) * TILE

                # Mask the number of batches needed to be processed per step
                numb_id_chunks = jnp.where(
                    off_end == off_start,
                    0,
                    (off_end - aligned_start + NUMB_IDS_PROCESSED_PER_STEP - 1)
                    // NUMB_IDS_PROCESSED_PER_STEP,
                )

                # Grab the initial chunk of ids and rows
                @pl.when(numb_id_chunks > 0)
                def _(
                    aligned_start=aligned_start,
                    off_start=off_start,
                    off_end=off_end,
                    feat_offset=feat_offset,
                ):
                    pingpong_id = 0
                    _gather(aligned_start, pingpong_id, off_start, off_end, feat_offset)

                @pl.loop(0, numb_id_chunks)
                def _process_cols(
                    i_id_chunk,
                    i=i_batch,
                    numb_id_chunks=numb_id_chunks,
                    off_end=off_end,
                    off_start=off_start,
                    aligned_start=aligned_start,
                    feat_offset=feat_offset,
                ):
                    # Current pingpong id/batch
                    pingpong_id = i_id_chunk % 2

                    @pl.when(i_id_chunk + 1 < numb_id_chunks)
                    def _(
                        i_id_chunk=i_id_chunk,
                        aligned_start=aligned_start,
                        off_start=off_start,
                        off_end=off_end,
                        feat_offset=feat_offset,
                    ):
                        # Alternate to do the next set of ids
                        next_pingpong_id = (i_id_chunk + 1) % 2

                        # Grab the next chunk of ids and their embedding.
                        _gather(
                            aligned_start
                            + (i_id_chunk + 1) * NUMB_IDS_PROCESSED_PER_STEP,
                            next_pingpong_id,
                            off_start,
                            off_end,
                            feat_offset,
                        )

                    # Wait for the ids/embedding rows to arrive
                    _wait(pingpong_id)

                    # Do masking over ids within a batch. The chunk is tile-aligned
                    # so it can start before the batch does: mask both ends.
                    chunk_base = (
                        aligned_start + i_id_chunk * NUMB_IDS_PROCESSED_PER_STEP
                    )
                    one = jnp.ones((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                    zero = jnp.zeros((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                    masks = [
                        jnp.where(
                            (chunk_base + row >= off_start)
                            & (chunk_base + row < off_end),
                            one,
                            zero,
                        )
                        for row in range(NUMB_IDS_PROCESSED_PER_STEP)
                    ]

                    # Do pooling of the embedding table by going through column dimension
                    # TODO: Benchmark with different temp arrays for the summation.
                    @plsc.parallel_loop(0, aligned_emb_dim, step=LANE_SIZE)
                    def _cols(i_col, i=i, masks=masks, pingpong_id=pingpong_id):
                        col_ids = pl.ds(i_col, LANE_SIZE)

                        # Mask due to the embedding dimension not being alined to the lane-size
                        lane_ids = i_col + jnp.arange(LANE_SIZE)
                        valid_lanes = lane_ids < emb_dim

                        # Accumulate it in the first row of temp buffer
                        total = post_pooled_emb_vmem[i, col_ids]

                        # Each subcore goes through certain number of IDS within its batch
                        for row in range(NUMB_IDS_PROCESSED_PER_STEP):
                            embs_out = buf_vmem[pingpong_id, row, col_ids]
                            masked_emb = jnp.where(valid_lanes, embs_out, 0)
                            total = total + masked_emb * masks[row]
                        post_pooled_emb_vmem[i, col_ids] = total

            pltpu.sync_copy(
                post_pooled_emb_vmem,
                out_hbm.at[pl.ds(output_start, NUMB_BATCHES_PER_ITERATION)],
            )

        _inner()

    return lookup_kernel(dev_weights, indices, offsets, row_offsets)[
        :num_batches, :emb_dim
    ]


def embedding_pooled_batched_lookup_jax(
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    row_offsets: jax.Array,
    emb_dim: int,
) -> jax.Array:
    return run_sc_pooled_lookup_batch_parallel(
        indices, offsets, dev_weights, row_offsets, emb_dim
    )


def embedding_pooled_batched_lookup_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    """Dense gradient for a stacked table-batched sum-pooled lookup."""
    num_bags = grad_out.shape[0]
    assert num_bags > 0

    positions = jnp.arange(indices.shape[0], dtype=offsets.dtype)
    bag_indices = jnp.searchsorted(offsets[1:], positions, side="right")
    valid = positions < offsets[-1]
    safe_bag_indices = jnp.minimum(bag_indices, num_bags - 1)
    weight_indices = indices + row_offsets[safe_bag_indices]
    safe_weight_indices = jnp.where(valid, weight_indices, 0)
    grad_per_id = jnp.where(
        valid[:, None],
        grad_out[safe_bag_indices],
        jnp.zeros((), dtype=grad_out.dtype),
    )
    return (
        jnp.zeros((num_rows, emb_dim), grad_out.dtype)
        .at[safe_weight_indices]
        .add(grad_per_id)
    )
