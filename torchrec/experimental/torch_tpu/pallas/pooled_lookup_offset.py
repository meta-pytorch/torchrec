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
from typing import cast

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


@functools.partial(jax.jit, static_argnames=["emb_dim"])  # pyre-ignore[20]
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
        name="pooled_lookup_offset_sc",
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


# The single-table backward defaults to searchsorted; repeat requires offsets[-1]
# to match the id count there. The batched backward masks bucket-padded ids.
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


def _validate_vbe_metadata(
    B_offsets: jax.Array | None,
    B_offsets_rank_per_feature: jax.Array | None,
    output_offsets_feature_rank: jax.Array | None,
) -> None:
    if B_offsets is None:
        assert (
            B_offsets_rank_per_feature is None
        ), "VBE metadata must be all None or all provided"
        assert (
            output_offsets_feature_rank is None
        ), "VBE metadata must be all None or all provided"
    else:
        assert (
            B_offsets_rank_per_feature is not None
        ), "B_offsets_rank_per_feature is required for VBE"
        assert (
            output_offsets_feature_rank is not None
        ), "output_offsets_feature_rank is required for VBE"
        assert B_offsets.ndim == 1, "B_offsets must be 1D"
        assert (
            B_offsets_rank_per_feature.ndim == 2
        ), "B_offsets_rank_per_feature must be 2D"
        assert (
            output_offsets_feature_rank.ndim == 1
        ), "output_offsets_feature_rank must be 1D"
        num_features = B_offsets.shape[0] - 1
        num_ranks = B_offsets_rank_per_feature.shape[1] - 1
        assert num_features > 0, "VBE requires at least one feature"
        assert num_ranks > 0, "VBE requires at least one rank"
        assert (
            B_offsets_rank_per_feature.shape[0] == num_features
        ), "VBE feature counts must match"
        assert (
            output_offsets_feature_rank.shape[0] == num_ranks * num_features + 1
        ), "VBE output offsets must have num_ranks * num_features + 1 entries"


def _generate_row_output_offsets(
    B_offsets,
    B_offsets_rank_per_feature,
    output_offsets_feature_rank,
    num_batches,
    aligned_emb_dim,
    dtype,
):
    """Generate the output row offsets used to get rank-major order for VBE lookups"""
    # Follows fbgemm_gpu generate_vbe_metadata.cu
    row_output_offsets = jnp.zeros((num_batches,), dtype=dtype)
    num_tables = B_offsets.shape[0] - 1
    num_ranks = B_offsets_rank_per_feature.shape[1] - 1
    # Gets the global input bag of shape [F, W + 1]
    # e.g. F=2, W=2, [[0, 2, 5], [5, 6, 8]], so f0: rank0: [0, 2), f1:rank0:[5, 6]
    bag_boundaries_for_each_feature = (
        B_offsets[:-1, None] + B_offsets_rank_per_feature
    )  # Shape [F, W + 1], F=num_feature, W=world-size
    chunk_counts = jnp.diff(B_offsets_rank_per_feature, axis=1).reshape(-1)

    chunk_ids = jnp.repeat(
        jnp.arange(num_tables * num_ranks),
        chunk_counts,
        total_repeat_length=num_batches,
    )
    t = chunk_ids // num_ranks
    r = chunk_ids % num_ranks
    input_bag_ids = jnp.arange(num_batches, dtype=B_offsets.dtype)

    # Bag index within its feature/rank chunk.
    b = input_bag_ids - bag_boundaries_for_each_feature[t, r]

    # Output metadata is indexed in rank-major order.
    output_chunk_starts = output_offsets_feature_rank[r * num_tables + t]

    row_output_offsets = output_chunk_starts + b * aligned_emb_dim
    return row_output_offsets


@functools.partial(jax.jit, static_argnames=["emb_dim"])  # pyre-ignore[20]
def run_sc_pooled_lookup_batch_parallel(  # noqa: C901
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    row_offsets: jax.Array,
    emb_dim: int,
    row_output_offsets: jax.Array | None = None,
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
    TODO: For vbe metadata to calculat ehte row_output_offsets, test whether it should be inside it's own kernel (similar to generate_vbe_metadata.cu)
          or, put it inside this pallas kernel.
    TODO: Return the correct order in non-VBE as well.
    """
    # Only do eight batch at a time since it can fit inside a 16-lane offset load
    NUMB_BATCHES_PER_ITERATION = 8  # TODO: Tune this parameter
    LANE_SIZE = 16
    NUMB_IDS_PROCESSED_PER_STEP = 16  # Gather X number of IDS, indices per bag can vary so we need a fix size here
    IDS_PER_FETCH = 32  # How many embedding rows to fetch into each buffer
    TILE = 8
    assert NUMB_BATCHES_PER_ITERATION % TILE == 0
    assert NUMB_IDS_PROCESSED_PER_STEP % TILE == 0
    assert NUMB_BATCHES_PER_ITERATION + 1 <= LANE_SIZE
    assert IDS_PER_FETCH % TILE == 0
    assert IDS_PER_FETCH % LANE_SIZE == 0
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
    indices = jnp.pad(indices, (0, IDS_PER_FETCH), constant_values=0)

    # Pad the feature/row offsets
    row_offsets = jnp.pad(
        row_offsets,
        (0, grid_rows + LANE_SIZE - row_offsets.shape[0]),
        constant_values=0,
    )

    if row_output_offsets is None:
        row_output_offsets = jnp.arange(num_batches, dtype=jnp.int32) * emb_dim
    else:
        assert row_output_offsets.ndim == 1, "row_output_offsets must be 1D"
        assert (
            row_output_offsets.shape[0] == num_batches
        ), "row_output_offsets must have one entry per input bag"
    # Row indices for your uniform-width with shape (num_batches,)
    output_rows = (row_output_offsets // emb_dim).astype(jnp.int32)
    output_rows = jnp.pad(
        output_rows,
        (0, grid_rows + LANE_SIZE - output_rows.shape[0]),
        constant_values=-1,
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
            pltpu.VMEM((2, IDS_PER_FETCH), indices.dtype),  # indices
            pltpu.VMEM((LANE_SIZE,), offsets.dtype),  # per-batch row offsets
            pltpu.VMEM(
                (2, IDS_PER_FETCH, aligned_emb_dim),
                dev_weights.dtype,  # embedding in vmem
            ),
            pltpu.VMEM(
                (NUMB_BATCHES_PER_ITERATION, aligned_emb_dim),
                dev_weights.dtype,  # post pooled emb in vmem
            ),
            pltpu.VMEM((LANE_SIZE,), offsets.dtype),  # output row offsets
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SemaphoreType.DMA(()),
        ],
        name="batched_pooled_lookup_offset_sc",
    )
    def lookup_kernel(
        weights_hbm,
        indices_hbm,
        offsets_hbm,
        row_offsets_hbm,
        output_rows_hbm,
        out_hbm,
        offsets_vmem,
        indices_vmem,
        row_offsets_vmem,
        buf_vmem,
        post_pooled_emb_vmem,
        output_rows_vmem,
        sem,
        offset_sem,
        row_offset_sem,
        output_rows_sem,
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
            # in bulk implies it grabs all `NUMB_BATCHES_PER_ITERATION` offsets
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

            output_offsets_start = pltpu.make_async_copy(
                output_rows_hbm.at[
                    pl.ds(
                        base,
                        LANE_SIZE,
                    )
                ],
                output_rows_vmem,
                output_rows_sem,
            )
            offset_start.start()
            row_offset_start.start()
            output_offsets_start.start()

            # Zero out the temporary embedding rows in VMEM
            for i in range(NUMB_BATCHES_PER_ITERATION):
                for j in range(numb_cols_chunks):
                    post_pooled_emb_vmem[i, pl.ds(j * LANE_SIZE, LANE_SIZE)] = (
                        jnp.zeros((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                    )

            # Bring from VMEM to registers
            offset_start.wait()
            row_offset_start.wait()
            output_offsets_start.wait()
            offsets_reg = offsets_vmem[...]
            row_offsets_reg = row_offsets_vmem[...]

            def _gather_group_of_embeddings(
                base,
                slot,
                group_start,
                group_end,
                group_offsets,
                group_row_offsets,
                num_group_bags,
            ):
                """Gather a `IDS_PER_FETCH` number of embedding rows from `indices`."""
                # 1. Grab `IDS_PER_FETCH` indices
                base = pl.multiple_of(base, TILE)
                pltpu.sync_copy(
                    indices_hbm.at[pl.ds(base, IDS_PER_FETCH)],
                    indices_vmem.at[slot],
                )

                # 2. Convert indices to be global across tables
                # Go through each lane_size of IDS_PER_FETCH
                @pl.loop(0, IDS_PER_FETCH, step=LANE_SIZE, unroll=True)
                def _create_global_indices(i_lane, indices_vmem=indices_vmem):
                    # Make sure the indices grabbed are within the group of bag offsets.
                    positions = base + jnp.arange(LANE_SIZE) + i_lane
                    valid = (positions >= group_start) & (positions < group_end)

                    # `per_id_offsets` is where the global offsets are stored
                    indices_offsets = jnp.zeros(
                        (LANE_SIZE,), dtype=row_offsets_reg.dtype
                    )

                    # Bags can cross boundaries of indices, e.g. [i_0, i_1, i_2] can be 2/3 bags
                    # TODO: Check if better to make this unrolled loop
                    for i_bag in range(num_group_bags):
                        is_part_of_this_bag = (positions >= group_offsets[i_bag]) & (
                            positions < group_offsets[i_bag + 1]
                        )

                        # Update the indices of this bag
                        indices_offsets = jnp.where(
                            is_part_of_this_bag,
                            group_row_offsets[i_bag],
                            indices_offsets,
                        )

                    # Figure out which indices are part of which table/feature
                    indices_vmem[slot, pl.ds(i_lane, LANE_SIZE)] = jnp.where(
                        valid,
                        indices_vmem[slot, pl.ds(i_lane, LANE_SIZE)] + indices_offsets,
                        0,
                    )

                # 3. Grab the `IDS_PER_FETCH` embeddings rows from global indices
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(indices_vmem.at[slot])],
                    buf_vmem.at[slot],
                    sem.at[slot],
                ).start()

            def _wait_for_embeddings(slot):
                pltpu.make_async_copy(
                    weights_hbm.at[plsc.Indices(indices_vmem.at[slot])],
                    buf_vmem.at[slot],
                    sem.at[slot],
                ).wait()

            # First bag and last bag to calculate the number of IDS to do at a time
            off_start = offsets_reg[0]
            off_end = offsets_reg[NUMB_BATCHES_PER_ITERATION]
            aligned_start = (off_start // TILE) * TILE
            num_group_bags = NUMB_BATCHES_PER_ITERATION
            num_id_chunks = jnp.where(
                off_start == off_end,
                0,
                (off_end - aligned_start + IDS_PER_FETCH - 1) // IDS_PER_FETCH,
            )

            @pl.loop(0, num_id_chunks)
            def _process_shared_id_chunk(i_id_chunk):
                # The goal is to fetch `IDS_PER_FETCH` at a time
                new_base = aligned_start + i_id_chunk * IDS_PER_FETCH

                # Grab the initial chunk of ids and rows
                @pl.when(i_id_chunk == 0)
                def _get_initial_emb(
                    new_base=new_base,
                    off_start=off_start,
                    off_end=off_end,
                    offsets_reg=offsets_reg,
                    row_offsets_reg=row_offsets_reg,
                ):
                    pingpong_id = 0
                    _gather_group_of_embeddings(
                        new_base,
                        pingpong_id,
                        off_start,
                        off_end,
                        offsets_reg,
                        row_offsets_reg,
                        num_group_bags,
                    )

                @pl.when(i_id_chunk >= 0)
                def _start_emb_pool_loop():
                    # Current pingpong id/batch
                    pingpong_id = i_id_chunk % 2

                    @pl.when(i_id_chunk + 1 < num_id_chunks)
                    def _gather_next_set_of_embeddings(
                        i_id_chunk=i_id_chunk,
                        off_start=off_start,
                        off_end=off_end,
                        offsets_reg=offsets_reg,
                        row_offsets_reg=row_offsets_reg,
                    ):
                        # Alternate to do the next set of ids
                        next_pingpong_id = (i_id_chunk + 1) % 2

                        # Grab the next chunk of ids and their embedding.
                        _gather_group_of_embeddings(
                            aligned_start + (i_id_chunk + 1) * IDS_PER_FETCH,
                            next_pingpong_id,
                            off_start,
                            off_end,
                            offsets_reg,
                            row_offsets_reg,
                            num_group_bags,
                        )

                    # Wait for the ids/embedding rows to arrive
                    _wait_for_embeddings(pingpong_id)

                    for i_batch in range(NUMB_BATCHES_PER_ITERATION):
                        # Get the offsets for this bag
                        bag_start, bag_end = (
                            offsets_reg[i_batch],
                            offsets_reg[i_batch + 1],
                        )

                        # Do masking over ids within a batch. The chunk is tile-aligned
                        # so it can start before the batch does: mask both ends.
                        # Because our IDs chuck can span across different bags
                        chunk_base = aligned_start + i_id_chunk * IDS_PER_FETCH
                        one = jnp.ones((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                        zero = jnp.zeros((LANE_SIZE,), dtype=post_pooled_emb_vmem.dtype)
                        masks = [
                            jnp.where(
                                (chunk_base + row >= bag_start)
                                & (chunk_base + row < bag_end),
                                one,
                                zero,
                            )
                            for row in range(IDS_PER_FETCH)
                        ]

                        # Do pooling of the embedding table by going through column dimension
                        # TODO: Benchmark with different temp arrays for the summation.
                        @plsc.parallel_loop(0, aligned_emb_dim, step=LANE_SIZE)
                        def _cols(
                            i_col, i=i_batch, masks=masks, pingpong_id=pingpong_id
                        ):
                            col_ids = pl.ds(i_col, LANE_SIZE)

                            # Mask due to the embedding dimension not being alined to the lane-size
                            lane_ids = i_col + jnp.arange(LANE_SIZE)
                            valid_lanes = lane_ids < emb_dim

                            # Accumulate it in the first row of temp buffer
                            total = post_pooled_emb_vmem[i, col_ids]

                            # Each subcore goes through certain number of IDS within its batch
                            for row in range(IDS_PER_FETCH):
                                embs_out = buf_vmem[pingpong_id, row, col_ids]
                                masked_emb = jnp.where(valid_lanes, embs_out, 0)
                                total = total + jnp.where(
                                    masks[row] != 0, masked_emb, 0
                                )
                            post_pooled_emb_vmem[i, col_ids] = total

            # Post pooled_emb_vmem has size (NUMB_BATCHES_PER_ITERATION, aligned_emb_dim)
            destination_indices = output_rows_vmem.at[
                pl.ds(0, NUMB_BATCHES_PER_ITERATION)
            ]
            pltpu.sync_copy(
                post_pooled_emb_vmem,
                out_hbm.at[plsc.Indices(destination_indices, ignored_value=-1)],
            )

        _inner()

    return lookup_kernel(dev_weights, indices, offsets, row_offsets, output_rows)[
        :num_batches, :emb_dim
    ]


def embedding_pooled_batched_lookup_jax(
    indices: jax.Array,
    offsets: jax.Array,
    dev_weights: jax.Array,
    row_offsets: jax.Array,
    emb_dim: int,
    row_output_offsets: jax.Array | None = None,
) -> jax.Array:
    return run_sc_pooled_lookup_batch_parallel(
        indices,
        offsets,
        dev_weights.reshape((-1, emb_dim)),
        row_offsets,
        emb_dim,
        row_output_offsets=row_output_offsets,
    )


def _gradient_map(
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
    bag_row_offsets: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_bags = offsets.shape[0] - 1
    assert num_bags > 0
    n = indices.shape[0]
    bag_updates = (
        jnp.ones((num_bags,), dtype=offsets.dtype)
        if bag_row_offsets is None
        else jnp.diff(bag_row_offsets, prepend=0)
    )
    updates = jnp.stack((bag_updates, jnp.diff(row_offsets, prepend=0)))
    markers = jnp.zeros((2, n), dtype=updates.dtype)
    prefixes = cast(
        jax.Array, jnp.cumsum(markers.at[:, offsets[:-1]].add(updates), axis=1)
    )
    valid = jnp.arange(n, dtype=offsets.dtype) < offsets[-1]
    bags = (
        jnp.minimum(prefixes[0] - 1, num_bags - 1)
        if bag_row_offsets is None
        else jnp.where(valid, prefixes[0], 0)
    )
    rows = jnp.where(valid, indices + prefixes[1], 0)
    return rows, bags, valid


def embedding_pooled_batched_lookup_gradient_map_jax(
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map each flat ID to (stacked row, global bag, valid bool).

    This mapping is per occurrence, not deduplicated: repeated rows may contribute
    gradients from different bags, which the backward's scatter-add aggregates.
    Padded IDs are marked invalid.
    """
    # Make sure they have the same dtype, since this impl combines both together
    if offsets.dtype == row_offsets.dtype == jnp.int32:
        return _gradient_map(indices, offsets, row_offsets)

    num_bags = offsets.shape[0] - 1
    assert num_bags > 0

    positions = jnp.arange(indices.shape[0], dtype=offsets.dtype)
    bag_starts = offsets[:-1]
    bag_markers = jnp.zeros((indices.shape[0],), dtype=offsets.dtype)
    bag_indices = jnp.cumsum(bag_markers.at[bag_starts].add(1), axis=0) - 1
    valid = positions < offsets[-1]
    safe_bag_indices = jnp.minimum(bag_indices, num_bags - 1)
    # Base deltas telescope even when empty bags have the same start.
    base_deltas = jnp.diff(row_offsets, prepend=0)
    base_markers = jnp.zeros((indices.shape[0],), dtype=row_offsets.dtype)
    weight_indices = indices + cast(
        jax.Array,
        jnp.cumsum(base_markers.at[bag_starts].add(base_deltas), axis=0),
    )
    safe_weight_indices = jnp.where(valid, weight_indices, 0)
    return safe_weight_indices, safe_bag_indices, valid


def embedding_pooled_batched_lookup_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
    num_rows: int,
    emb_dim: int,
    flat_weights: bool = False,
    row_output_offsets: jax.Array | None = None,
) -> jax.Array:
    """Preprocessing for dense gradient for a stacked table-batched sum-pooled lookup."""
    safe_weight_indices, safe_bag_indices, valid = (
        embedding_pooled_batched_lookup_gradient_map_jax(indices, offsets, row_offsets)
    )

    safe_weight_indices = jnp.where(valid, safe_weight_indices, num_rows)
    ind = jnp.argsort(safe_weight_indices)
    safe_weight_indices = safe_weight_indices[ind]
    safe_bag_indices = safe_bag_indices[ind]
    valid = valid[ind]
    if row_output_offsets is not None:
        safe_bag_indices = (row_output_offsets[safe_bag_indices] // emb_dim).astype(
            jnp.int32
        )

    grad_per_id = jnp.where(
        valid[:, None],
        grad_out[safe_bag_indices],
        jnp.zeros((), dtype=grad_out.dtype),
    )
    if flat_weights:
        columns = jnp.arange(emb_dim, dtype=safe_weight_indices.dtype)
        return (
            jnp.zeros((num_rows * emb_dim,), grad_out.dtype)
            .at[safe_weight_indices[:, None] * emb_dim + columns]
            .add(grad_per_id)
        )
    return (
        jnp.zeros((num_rows, emb_dim), grad_out.dtype)
        .at[safe_weight_indices]
        .add(grad_per_id)
    )


def embedding_pooled_batched_lookup_row_runs_jax(
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
    num_rows: int,
    bag_row_offsets: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Preprocessing for fused gradient for a stacked table-batched pooled lookup"""
    rows, bags, valid = (
        embedding_pooled_batched_lookup_gradient_map_jax(indices, offsets, row_offsets)
        if bag_row_offsets is None
        else _gradient_map(indices, offsets, row_offsets, bag_row_offsets)
    )

    sorted_rows, sorted_bags = jax.lax.sort_key_val(
        jnp.where(valid, rows, num_rows),
        bags,
        dimension=0,
    )

    n = indices.shape[0]
    run_starts = (sorted_rows != num_rows) & jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            sorted_rows[1:] != sorted_rows[:-1],
        )
    )[:n]
    num_runs = jnp.sum(run_starts, dtype=offsets.dtype)
    run_offsets = jnp.nonzero(run_starts, size=min(n, num_rows), fill_value=n)[0]
    run_offsets = (
        jnp.concatenate((run_offsets, jnp.array([n], dtype=run_offsets.dtype)))
        .at[num_runs]
        .set(offsets[-1])
    )

    return sorted_rows, sorted_bags, run_offsets, num_runs
