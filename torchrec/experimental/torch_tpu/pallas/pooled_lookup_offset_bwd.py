#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc
from torchrec.experimental.torch_tpu.pallas import pooled_lookup_offset


def _rsqrt_sc(a: jax.Array) -> jax.Array:
    """Reciprocal square root on the SparseCore vector subcore.

    The SC vector subcore has no `sqrt`/`rsqrt`/`pow`/`log`
    so Adagrad's 1/sqrt is computed with the classic fast-inverse-sqrt
    bit-trick seed + 3 Newton iterations (~f32 precision).
    Assumes `a >= 0`; `a == 0` returns a large finite value, and callers
    form `sqrt(a) = a * rsqrt(a)` so the 0 case collapses to 0 (no NaN).
    """
    i = jax.lax.bitcast_convert_type(a, jnp.int32)
    i = jnp.int32(0x5F3759DF) - jax.lax.shift_right_logical(i, 1)
    y = jax.lax.bitcast_convert_type(i, jnp.float32)
    half_a = a * jnp.float32(0.5)
    for _ in range(3):
        y = y * (jnp.float32(1.5) - half_a * y * y)
    return y


@functools.partial(jax.jit, static_argnames=["emb_dim", "eps"])
def fused_rowwise_adagrad_update(  # noqa: C901
    indices: jax.Array,
    bag_ids: jax.Array,
    run_offsets: jax.Array,
    num_runs: jax.Array,
    grad_out: jax.Array,
    emb_tables: jax.Array,
    momentum: jax.Array,
    lr: jax.Array,
    eps: float,
    emb_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Fused rowwise Adagrad update for sparse embedding tables on SparseCore.

    Args:
        grad_out: [M, D] Gradient for each bag
        indices: [N] the sorted global indices of the table
        bag_index: [N] which bag the index is part of, used to get grad_out
        run_offsets: Offset of each uniqu eindex
        emb_tables: Full (stacked) embedding table
        emb_dim: Dimension D of embedding table

    Algorithm is as follows:
    1) Obtain a mapping from unique indices to which bag it is part of
    2) For each subcore, assign a unique index to it
    3) Gather the gradients for that unique index
    4) Pool the gradients
    5) Do the rowwise adagrad update

    TODO: Convert most of these sync copies with async copies to overlap with compute
    TODO: Add double buffering/ping-pong algo, and also zero out temp buffers, e.g. could do batches
          or embedding tables
    TODO: Optimize the hyper-parameters e.g. NUMB_BAGS_PER_ITERATION
    """
    # Kernel parameters
    NUMB_BAGS_PER_ITERATION = 8
    LANE_SIZE = 16
    lr = jnp.broadcast_to(jnp.asarray(lr, dtype=jnp.float32), (LANE_SIZE,))
    NUMB_IDS_PROCESSED_PER_BAG = 16
    TILE = 8
    assert emb_dim % LANE_SIZE == 0

    # Typical sparse-core subcore information
    sc = pltpu.get_tpu_info().sparse_core  # pyrefly: ignore [missing-attribute]
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyrefly: ignore [unexpected-keyword]
    )
    num_subcores = sc.num_cores * sc.num_subcores

    # Parallelization strategy is to assign each unique index to a sparse-core
    # Then the lanes work over embedding dimensions.
    max_runs = run_offsets.shape[0] - 1
    numb_chunks = (max_runs + NUMB_BAGS_PER_ITERATION - 1) // NUMB_BAGS_PER_ITERATION
    number_of_blocks = max(1, (numb_chunks + num_subcores - 1) // num_subcores)
    grid = (number_of_blocks, num_subcores)
    launched_runs = number_of_blocks * num_subcores * NUMB_BAGS_PER_ITERATION
    run_offsets = jnp.pad(
        run_offsets,
        (
            0,
            launched_runs + LANE_SIZE - NUMB_BAGS_PER_ITERATION - run_offsets.shape[0],
        ),
        mode="edge",
    )
    num_runs_tile = jnp.full((LANE_SIZE,), num_runs, dtype=run_offsets.dtype)
    indices = jnp.pad(indices, (0, LANE_SIZE))
    bag_ids = jnp.pad(bag_ids, (0, NUMB_IDS_PROCESSED_PER_BAG))

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=(),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((LANE_SIZE,), jnp.int32),  # indices_vmem
            pltpu.VMEM((NUMB_IDS_PROCESSED_PER_BAG,), jnp.int32),  # bag_ids_vmem
            pltpu.VMEM((LANE_SIZE,), jnp.int32),  # run_offsets_vmem
            pltpu.VMEM((LANE_SIZE,), jnp.int32),  # num_runs_vmem
            pltpu.VMEM((NUMB_IDS_PROCESSED_PER_BAG, emb_dim), grad_out.dtype),
            pltpu.VMEM((LANE_SIZE,), momentum.dtype),
            pltpu.VMEM((LANE_SIZE, emb_dim), emb_tables.dtype),
            pltpu.VMEM((emb_dim,), emb_tables.dtype),
            pltpu.VMEM((LANE_SIZE,), jnp.float32),
        ],
        name="rowwise_adagrad_sc_bwd",
        mesh=mesh,
    )
    def kernel(
        indices_hbm,
        bag_ids_hbm,
        run_offsets_hbm,
        num_runs_hbm,
        grad_out_hbm,
        momentum_hbm,
        emb_tables_hbm,
        lr_hbm,
        indices_vmem,
        bag_ids_vmem,
        run_offsets_vmem,
        num_runs_vmem,
        grad_out_vmem,
        momentum_vmem,
        emb_tables_vmem,
        grad_buffer_vmem,
        lr_vmem,
    ):
        pltpu.sync_copy(lr_hbm, lr_vmem)

        @functools.partial(
            pltpu.emit_pipeline,
            grid=grid,
            core_axis_name=("c", "s"),
            dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL),
        )
        def _inner():
            # Using run_offsets, get the offset associated with this run
            chunk_id = pl.program_id(0)
            core_id = pl.program_id(1)
            run_id = (chunk_id * num_subcores + core_id) * NUMB_BAGS_PER_ITERATION
            pltpu.sync_copy(num_runs_hbm, num_runs_vmem)
            pltpu.sync_copy(
                run_offsets_hbm.at[pl.ds(run_id, LANE_SIZE)],
                run_offsets_vmem,
            )

            valid_runs = jnp.minimum(
                NUMB_BAGS_PER_ITERATION,
                jnp.maximum(0, num_runs_vmem[...][0] - run_id),
            )

            # Loop over numb bags to process in this subcore
            @pl.loop(0, valid_runs)
            def rowwise_adagrad_per_unique_row(unique_row_id):
                start = plsc.load_gather(
                    run_offsets_vmem,
                    (jnp.full((LANE_SIZE,), unique_row_id, dtype=jnp.int32),),
                )[0]
                end = plsc.load_gather(
                    run_offsets_vmem,
                    (jnp.full((LANE_SIZE,), unique_row_id + 1, dtype=jnp.int32),),
                )[0]

                # Check validiity here for the bag_ids
                aligned_start = pl.multiple_of((start // TILE) * TILE, TILE)

                # Then using that grab the bag_ids and single index from that offset
                pltpu.sync_copy(
                    indices_hbm.at[pl.ds(aligned_start, LANE_SIZE)], indices_vmem
                )
                row = plsc.load_gather(
                    indices_vmem,
                    (jnp.full((LANE_SIZE,), start - aligned_start, dtype=jnp.int32),),
                )[0]
                indices_vmem[...] = jnp.where(jnp.arange(LANE_SIZE) == 0, row, -1)
                row_indices = plsc.Indices(indices_vmem, ignored_value=-1)
                pltpu.sync_copy(
                    bag_ids_hbm.at[pl.ds(aligned_start, NUMB_IDS_PROCESSED_PER_BAG)],
                    bag_ids_vmem,
                )
                positions = aligned_start + jnp.arange(NUMB_IDS_PROCESSED_PER_BAG)
                valid = (positions >= start) & (positions < end)
                bag_ids_vmem[...] = jnp.where(valid, bag_ids_vmem[...], -1)

                # Pull all gradients using bag_ids
                pltpu.sync_copy(
                    grad_out_hbm.at[plsc.Indices(bag_ids_vmem, ignored_value=-1)],
                    grad_out_vmem,
                )
                momentum_vmem[...] = jnp.zeros((LANE_SIZE,), dtype=momentum_vmem.dtype)
                pltpu.sync_copy(momentum_hbm.at[row_indices], momentum_vmem)

                # Grab the single embedding row
                pltpu.sync_copy(emb_tables_hbm.at[row_indices], emb_tables_vmem)

                # Zero out a total gradient
                @pl.loop(0, emb_dim, step=LANE_SIZE)
                def zero_grad(i):
                    col = pl.ds(i, LANE_SIZE)
                    grad_buffer_vmem[col] = jnp.zeros(
                        (LANE_SIZE,), dtype=grad_out_vmem.dtype
                    )

                # Grab the gradients from the bag_ids
                num_chunks = (
                    end - aligned_start + NUMB_IDS_PROCESSED_PER_BAG - 1
                ) // NUMB_IDS_PROCESSED_PER_BAG

                @pl.loop(0, num_chunks)
                def process_chunk(chunk):
                    chunk_start = aligned_start + chunk * NUMB_IDS_PROCESSED_PER_BAG

                    @pl.when(chunk > 0)
                    def load_chunk():
                        pltpu.sync_copy(
                            bag_ids_hbm.at[
                                pl.ds(chunk_start, NUMB_IDS_PROCESSED_PER_BAG)
                            ],
                            bag_ids_vmem,
                        )
                        positions = chunk_start + jnp.arange(NUMB_IDS_PROCESSED_PER_BAG)
                        valid = (positions >= start) & (positions < end)
                        bag_ids_vmem[...] = jnp.where(valid, bag_ids_vmem[...], -1)
                        pltpu.sync_copy(
                            grad_out_hbm.at[
                                plsc.Indices(bag_ids_vmem, ignored_value=-1)
                            ],
                            grad_out_vmem,
                        )

                    @pl.loop(0, NUMB_IDS_PROCESSED_PER_BAG)
                    def process_single_bag_id(run_id):
                        @pl.when(
                            (chunk_start + run_id >= start)
                            & (chunk_start + run_id < end)
                        )
                        def add_gradient():
                            @pl.loop(0, emb_dim, step=LANE_SIZE)
                            def process_gradient(col_id):
                                cols = pl.ds(col_id, LANE_SIZE)
                                grad_buffer_vmem[cols] += grad_out_vmem[run_id, cols]

                # Calculate mean sq gradient
                mean_sq_grad = jnp.float32(0)
                for col in range(0, emb_dim, LANE_SIZE):
                    grad = grad_buffer_vmem[pl.ds(col, LANE_SIZE)]
                    mean_sq_grad += jnp.sum(grad * grad)
                mean_sq_grad *= 1.0 / emb_dim

                # Get the momentum update
                new_m = momentum_vmem[...][0] + mean_sq_grad
                sqrt_m = new_m * _rsqrt_sc(new_m)

                # Update embedding table
                step = lr_vmem[...] / jnp.full(
                    (LANE_SIZE,), sqrt_m + eps, dtype=jnp.float32
                )

                @pl.loop(0, emb_dim, step=LANE_SIZE)
                def update_table(i):
                    col = pl.ds(i, LANE_SIZE)
                    emb_tables_vmem[0, col] -= step * grad_buffer_vmem[col]

                pltpu.sync_copy(emb_tables_vmem, emb_tables_hbm.at[row_indices])
                momentum_vmem[...] = jnp.full(
                    (LANE_SIZE,), new_m, dtype=momentum_vmem.dtype
                )
                pltpu.sync_copy(momentum_vmem, momentum_hbm.at[row_indices])

        _inner()

    weights_ref = jax.new_ref(emb_tables)
    momentum_ref = jax.new_ref(momentum)
    kernel(
        indices,
        bag_ids,
        run_offsets,
        num_runs_tile,
        grad_out,
        momentum_ref,
        weights_ref,
        lr,
    )
    return weights_ref[...], momentum_ref[...]


@functools.partial(jax.jit, static_argnames=["eps", "emb_dim"])
def embedding_pooled_batched_lookup_fused_adagrad_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    row_offsets: jax.Array,
    emb_tables: jax.Array,
    momentum: jax.Array,
    lr: jax.Array,
    eps: float,
    emb_dim: int,
) -> tuple[jax.Array, jax.Array]:
    sorted_rows, bag_ids, run_offsets, num_runs = (
        pooled_lookup_offset.embedding_pooled_batched_lookup_row_runs_jax(
            indices, offsets, row_offsets, emb_tables.size // emb_dim
        )
    )
    # Reshaping is required since XLA compiler converts the layouts of the
    # 2-D tables from row-major to col-major before passing it to the kernel
    # Thus decided to keep emb_tables one-dimensional and do a reshape to 2-D
    # right before the kernel.
    new_tables, new_momentum = fused_rowwise_adagrad_update(
        sorted_rows,
        bag_ids,
        run_offsets,
        num_runs,
        grad_out,
        emb_tables.reshape((-1, emb_dim)),
        momentum,
        lr,
        eps,
        emb_dim,
    )
    return new_tables.reshape(emb_tables.shape), new_momentum
