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


@functools.partial(jax.jit, static_argnames=["total_columns"])
def batch_index_select_sc(
    inputs: jax.Array,
    indices: jax.Array,
    input_rows: jax.Array,
    input_columns: jax.Array,
    total_columns: int,
) -> jax.Array:
    """Expand deduplicated VBE rows into batch-major output on SparseCore.

    inputs: flattened concatenated embedding tables
    indices: flattened reverse_indices, conceptually shaped [K, B]
    input_rows: deduped batch size for each uncombined feature
    input_columns: embedding dimension for each uncombined feature
    total_columns: combined embedding dimension of the output

    VBE feature-segment parallelism splits each feature reverse_indices into
    NL-sized segments. Each subcore gathers one segment of deduped embedding
    rows and writes it into the corresponding embedding output rows.
    """
    # Typical sparse-core subcore information
    sc = pltpu.get_tpu_info().sparse_core  # pyrefly: ignore [missing-attribute]
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyrefly: ignore [unexpected-keyword]
    )
    num_subcores = sc.num_cores * sc.num_subcores
    NL = 16
    COL_TILE = 16

    # The kernel views the output as [batch, combined embedding dim] for a
    # rectangular output DMA, then flattens it to match the FBGEMM contract.
    num_features = input_columns.shape[0]

    batch_size = indices.shape[0] // num_features
    if indices.shape[0] % num_features != 0:
        raise ValueError(
            "Total number of elements in `indices` should be a multiple of `num_features`"
        )
    if batch_size % NL != 0:
        raise NotImplementedError(f"batch size must be divisible by {NL}")
    if total_columns % COL_TILE != 0 or inputs.shape[0] % COL_TILE != 0:
        raise NotImplementedError(
            f"embedding dimensions must be divisible by {COL_TILE}"
        )

    # Parallelism scheme: each subcore handles one NL-sized segment of
    # one feature reverse_indices and reconstructs that embedding output segment.
    tiles_per_feature = batch_size // NL
    total_tiles = num_features * tiles_per_feature
    n_blocks = (total_tiles + num_subcores - 1) // num_subcores

    # Materialize three separate offset arrays on TPU.
    rows = jnp.asarray(input_rows, dtype=jnp.int32)
    columns = jnp.asarray(input_columns, dtype=jnp.int32)
    zero = jnp.zeros((1,), dtype=jnp.int32)
    input_offsets = jnp.concatenate((zero, jnp.cumsum(rows * columns)[:-1]))
    column_offsets = jnp.concatenate((zero, jnp.cumsum(columns)[:-1]))
    # Repeat each feature scalar across all lanes so the kernel can read lane 0
    # statically; SparseCore does not lower a dynamic VMEM lane selection.
    input_offsets = jnp.repeat(input_offsets, NL)
    column_offsets = jnp.repeat(column_offsets, NL)
    columns = jnp.repeat(columns, NL)
    # View the flat input as COL_TILE-wide rows so plsc.Indices can gather one
    # column chunk from all NL selected embedding rows in a single indexed DMA.
    input_tiles = inputs.reshape((-1, COL_TILE))
    indices = indices.astype(jnp.int32)

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct((batch_size, total_columns), inputs.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL, COL_TILE), inputs.dtype),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
        ],
        name="batch_index_select_sc",
        mesh=mesh,
    )
    def _kernel(
        input_tiles_hbm,
        indices_hbm,
        input_offsets_hbm,
        columns_hbm,
        column_offsets_hbm,
        out_hbm,
        indices_vmem,
        inputs_vmem,
        input_offsets_vmem,
        columns_vmem,
        column_offsets_vmem,
        gather_sem,
    ):
        @functools.partial(pltpu.emit_pipeline, grid=(n_blocks,))
        def _inner():
            # Grab the global subcore ID across all sparse cores.
            core_index = jax.lax.axis_index("c")
            subcore_in_core = jax.lax.axis_index("s")
            subcore_id = core_index * sc.num_subcores + subcore_in_core
            # Grab the tile and feature/batch_tile assigned to this subcore.
            tile = pl.program_id(0) * num_subcores + subcore_id

            # Only work on tiles that are less than the logical tile count.
            @pl.when(tile < total_tiles)
            def _valid_tile():
                feature = tile // tiles_per_feature
                feature_block_start = pl.multiple_of(feature * NL, NL)
                # Load the feature metadata into VMEM.
                pltpu.sync_copy(
                    columns_hbm.at[pl.ds(feature_block_start, NL)], columns_vmem
                )
                pltpu.sync_copy(
                    column_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                    column_offsets_vmem,
                )
                pltpu.sync_copy(
                    input_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                    input_offsets_vmem,
                )
                col_length = columns_vmem[...][0]
                col_start = column_offsets_vmem[...][0]
                input_start = input_offsets_vmem[...][0]

                # Stage this feature segment reverse_indices in VMEM.
                batch_tile = tile % tiles_per_feature
                batch_start = batch_tile * NL
                index_start = pl.multiple_of(feature * batch_size + batch_start, 8)
                pltpu.sync_copy(indices_hbm.at[pl.ds(index_start, NL)], indices_vmem)
                # Convert selected embedding rows to COL_TILE-row indices in the
                # reshaped input. Incrementing them advances one column chunk.
                chunks_per_row = col_length // COL_TILE
                indices_vmem[...] = (
                    input_start // COL_TILE + indices_vmem[...] * chunks_per_row
                )

                # Gather NL noncontiguous input rows, then write one rectangular
                # [NL, COL_TILE] output block for each feature-column chunk.
                @pl.loop(0, chunks_per_row)
                def _copy_chunk(chunk):
                    # Grab the embeddings using this segment reverse_indices.
                    # TODO remove this unnecessary async copy.
                    gather = pltpu.make_async_copy(
                        input_tiles_hbm.at[plsc.Indices(indices_vmem)],
                        inputs_vmem,
                        gather_sem,
                    )
                    gather.start()
                    gather.wait()
                    output_col = pl.multiple_of(col_start + chunk * COL_TILE, COL_TILE)
                    # Write the gathered embedding segment into emb_output.
                    pltpu.sync_copy(
                        inputs_vmem,
                        out_hbm.at[pl.ds(batch_start, NL), pl.ds(output_col, COL_TILE)],
                    )
                    indices_vmem[...] = indices_vmem[...] + 1

        _inner()

    # FBGEMM returns a flat tensor; this reshape is metadata-only.
    return _kernel(
        input_tiles, indices, input_offsets, columns, column_offsets
    ).reshape(-1)


@functools.partial(jax.jit, static_argnames=["num_features", "batch_size"])
def group_batches_by_unique(
    indices: jax.Array,
    num_features: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Gets a mapping from unique bags to which gradient it corresponds to.

    Reverse-indices has shape (F, B), and tells us how to go from deduped to expanded
    For SC, we need a mapping from the unique indices to which gradient it belongs to
    so that a sparseCore subcore can own one bag and sum its gradients effectively.

    Returns:
        Tuple[jax.Array, jax.Array]:
            - grad_rows: array size [F, B] corresponding to which index the bag corresponds to
            - bag_offsets: Offset boundaries inside `grad_rows` that corresponds to each bag.

    Examples:

        indices     = [[0, 0, 1, 0], [0, 1, 0, 0]]
        grad_rows        = [[0, 1, 3, 2], [0, 2, 3, 1]]
        bag_offsets = [[0, 3, 4, 4, 4], [0, 3, 4, 4, 4]]
        # f=0: bag 0 -> samples [0, 1, 3], bag 1 -> [2]
    """

    indices = indices.reshape(num_features, batch_size)
    # First do a 2-D argsort, e.g. feature-0 becomes [0, 1, 3, 2]
    grad_rows = jnp.argsort(indices, axis=1, stable=True)
    # Found the boundaries that corresponds to each bag b=i
    # e.g. feature_0 becomes [b=0, b=1] -> [[0, 1, 3], [2]]
    # This can be done by adding how many duplicates each bag has
    rows = jnp.arange(num_features)[:, None]
    counts = jnp.zeros((num_features, batch_size), jnp.int32).at[rows, indices].add(1)
    # Get the offsets to get the boundaries of the bags
    # e.g. feature-0 becomes [0, 3, 4, 4, 4, 4]
    bag_offsets = jnp.pad(
        jnp.cumsum(counts, axis=1),
        (
            (0, 0),
            (1, 0),
        ),  # (before, after) per axis: no padding on axis 0, one 0 before axis 1
    )
    return grad_rows, bag_offsets


def batch_index_select_bwd_sc(
    grad_output: jax.Array,
    indices: jax.Array,
    input_rows: jax.Array,
    input_columns: jax.Array,
    permute_output_dim_0_1: bool,
    output_size: int,
) -> jax.Array:
    """
    Backward pallas kernel for the batch index select

    This takes the expanded gradient, and produces the deduped summed gradient. This assumes the assumption made
    inside forward still hold, e.g. B % LANE_SIZE == 0

    Args:
        grad_output: The flattened (row-major order, [B, F, D]) gradient of the expanded embedding table lookup.
        indices: Per-feature reverse indices, flattened, size (F, B)
        input_rows: The number of rows per deduped embedding lookup result, size (F,)
        input_cols: The number of cols per deduped embedding lookup result, size (F,)
        permute_output_dim_0_1: Whether output is batch-major order
        output_size: Total number of elements in the deduped embedding lookup.

    Returns:
        jax.Array:
            Flatten array of the deduped gradients in feature order, i.e. (F, B, D) row-major order

    Algorithm:
    1. For every feature, grab the mapping from unique indices -> (which gradient need to be summed)
       Mapping feature bag |-> [index of gradients for that feature ], see Mapping below
    2. Assign each unique index to a sub-core
    3. Each sub-core grabs the unique-index, then grabs which gradients indices that it needs (does so by NL-chunks)
    4. Grabs the gradients but in NL-chunk columns at a time
    5. Pools the gradient for each NL-chunk columns into a single temporary gradient buffer
    6. Then writes that gradient back to HBM.

    Caveats:
    - load imbalance: if pooling sizes get very big
    - The gradients are worked into `COL_TILE` chunks, since VBE can cause different gradient dimensions to be different

    TODO: Move the computation done inside the forward pallas kernel to the backward to store inside ctx
    TODO: Try out similar to rowwise adagrad to do sort and group unique rows toegether and assign them to a subcore
    TODO: Consider doing a feature per sub-core implementation with sparse-core coordination, gets around load imbalance
    TODO: whether n_blocks should be sum(input_rows) (which can trigger recompilation) or F*B or a padded
    TODO: Check reshaping grad_output to have COl_TILE dimension is better to do than when gathering the gradients inside the kernel
    TODO: Optimize the memory tranfers/compute with utilizing double the memory
    TODO: Optimize the number of grid-blocks needed

    Notes:
        Inverse-indices Mapping:
            Feature-offsets of input-columns F_0 = [0, D_0, D_0 + D_1, ..., sum(D)]
            Mapping would be (f_i, g_j, d_k) |->  g_j * F_0[-1] + F_0[f_i] + d_k

        Deduped Gradient:
            Embedding dimensions of each feature D_i
            Total offsets O = [0, R_0 * D_0, R_0 * D_0 + R_1 * D_1, ...] (summation over row/cols)
            (f_i, r_i, d_j) |-> gradient output index ([R_f, D_f] blocks per feature) = O[f_i] + r_i * D[f_i] + d_j

    """
    NL = 16
    COL_TILE = 16
    ROW_TILE = 8
    if not permute_output_dim_0_1:
        raise NotImplementedError(
            f"`permute_output_dim_0_1` is not implemented yet. Reach out to oncall."
        )
    # Typical sparse-core subcore information
    sc = pltpu.get_tpu_info().sparse_core  # pyrefly: ignore [missing-attribute]
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyrefly: ignore [unexpected-keyword]
    )
    num_subcores = sc.num_cores * sc.num_subcores

    num_features = input_rows.shape[0]
    batch_size = indices.shape[0] // num_features

    # Size of output
    output_shape = (output_size,)

    # First get a mapping from (feature, bag index) -> [gradient index i that needs to be pooled]
    grad_rows, bag_offsets = group_batches_by_unique(indices, num_features, batch_size)

    # Parallelism scheme: each subcore handles one deduped row, grabs the gradients, and pools them
    # Choose F*B for the number of blocks for now, but will have to optimize this later on
    # TODO: Optimize the number of grid blocks
    n_blocks = (num_features * batch_size + num_subcores - 1) // num_subcores

    # Convert into jax arrays with int32
    rows = jnp.asarray(input_rows, dtype=jnp.int32)
    columns = jnp.asarray(input_columns, dtype=jnp.int32)
    zero = jnp.zeros((1,), dtype=jnp.int32)
    # output_offsets helps with the mapping to get the index in the deduped gradient
    # assuming the deduped gradient is shape (F, B, D) row-major order
    output_offsets = jnp.concatenate((zero, jnp.cumsum(rows * columns)[:-1]))
    # needed to get the mapping from reverse-indices to the expanded gradient
    # assumes the expanded gradient is (B, D, D) row-major order
    column_offsets = jnp.concatenate((zero, jnp.cumsum(columns)[:-1]))
    # Repeat so that a lane can read the same scalar value
    deduped_offsets = jnp.repeat(output_offsets, NL)
    column_offsets = jnp.repeat(column_offsets, NL)
    bag_offsets = jnp.repeat(bag_offsets, NL)
    rows = jnp.repeat(rows, NL)
    columns = jnp.repeat(columns, NL)

    # Reshape the gradient into COL_TILE-wide rows for indexed DMA.
    total_col_dim = grad_output.size // batch_size
    chunks_per_batch = total_col_dim // COL_TILE
    grad_output = grad_output.reshape((-1, COL_TILE))

    # Pad by NL so the last bag's NL-wide read stays in bounds; padded lanes are masked.
    grad_rows = jnp.pad(grad_rows.reshape(-1), (0, NL))

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct(output_shape, grad_output.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((NL, COL_TILE), grad_output.dtype),  # expanded_grad_vmem
            pltpu.VMEM((NL,), jnp.int32),  # grad_rows_vmem
            pltpu.VMEM((NL,), jnp.int32),  # bag_offsets_vmem
            pltpu.VMEM((NL,), jnp.int32),  # cols_offsets_vmem
            pltpu.VMEM((NL,), jnp.int32),  # rows_vmem
            pltpu.VMEM((NL,), jnp.int32),  # columns_vmem
            pltpu.VMEM((NL,), jnp.int32),  # deduped_offsets_vmem
            pltpu.VMEM((COL_TILE,), grad_output.dtype),  # grad_buffer_vmem
        ],
        name="batch_index_select_bwd_sc",
        mesh=mesh,
    )
    def _kernel(
        expanded_grad_hbm,
        grad_rows_hbm,
        bag_offsets_hbm,
        cols_offsets_hbm,
        rows_hbm,
        columns_hbm,
        deduped_offsets_hbm,
        out_hbm,
        expanded_grad_vmem,
        grad_rows_vmem,
        bag_offsets_vmem,
        cols_offsets_vmem,
        rows_vmem,
        columns_vmem,
        deduped_offsets_vmem,
        grad_buffer_vmem,
    ):
        @functools.partial(pltpu.emit_pipeline, grid=(n_blocks,))
        def _inner():
            # Grab the global subcore ID across all sparse cores.
            core_index = jax.lax.axis_index("c")
            subcore_in_core = jax.lax.axis_index("s")
            subcore_id = core_index * sc.num_subcores + subcore_in_core
            # Grab the feature/batch assigned to this subcore.
            idx = pl.program_id(0) * num_subcores + subcore_id

            # Only work if it's less than total number of expanded gradients
            @pl.when(idx < num_features * batch_size)
            def _valid_idx():
                # Grab the feature and batch this subcore responsible for
                feat_id = idx // batch_size
                batch_id = idx % batch_size
                feature_block_start = pl.multiple_of(feat_id * NL, NL)

                # Need to make sure the batch id for this feature doesn't overextend
                pltpu.sync_copy(
                    rows_hbm.at[pl.ds(feature_block_start, NL)],
                    rows_vmem,
                )

                @pl.when(batch_id < rows_vmem[...][0])
                def _get_gradients():
                    # Based on feat_id, batch_id, pull the bag_offsets
                    base = (feat_id * (batch_size + 1) + batch_id) * NL
                    pltpu.sync_copy(
                        bag_offsets_hbm.at[pl.ds(base, NL)],
                        bag_offsets_vmem,
                    )
                    start_bag = bag_offsets_vmem[...][0]

                    pltpu.sync_copy(
                        bag_offsets_hbm.at[pl.ds(base + NL, NL)],
                        bag_offsets_vmem,
                    )
                    end_bag = bag_offsets_vmem[...][0]
                    numb_grads_to_pool = end_bag - start_bag

                    # Pull the gradients from the grad indices
                    # (f_i, b_i, g_j, d_k) |->  g_j * F_0[-1] + F_0[i] + d_k
                    pltpu.sync_copy(
                        cols_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                        cols_offsets_vmem,
                    )
                    col_start = cols_offsets_vmem[...][0]

                    # Get the embedding dimension for this feature
                    pltpu.sync_copy(
                        columns_hbm.at[pl.ds(feature_block_start, NL)],
                        columns_vmem,
                    )
                    emb_dim = columns_vmem[...][0]
                    # Assumes that emb-dimension is a multiple of COL_TILE
                    numb_chunks = emb_dim // COL_TILE

                    # Grab the deduped offsets for the output
                    pltpu.sync_copy(
                        deduped_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                        deduped_offsets_vmem,
                    )
                    total_offset = deduped_offsets_vmem[...][0]

                    # Go through 16 gradients at a time
                    @pl.loop(0, numb_chunks)
                    def _loop_col(i):
                        # Make a temporary buffer of size `COL_TILE` to pool the gradients
                        grad_buffer_vmem[...] = jnp.zeros(COL_TILE, out_hbm.dtype)

                        @pl.loop(0, numb_grads_to_pool, step=ROW_TILE)
                        def _chunk_gradients(s):
                            # Grab 16 gradients indices
                            grad_start = feat_id * batch_size + start_bag + s
                            aligned_start = pl.multiple_of(
                                (grad_start // ROW_TILE) * ROW_TILE, ROW_TILE
                            )
                            pltpu.sync_copy(
                                grad_rows_hbm.at[pl.ds(aligned_start, NL)],
                                grad_rows_vmem,
                            )
                            # Minimum number of gradients to pool
                            min_grads = jnp.minimum(ROW_TILE, numb_grads_to_pool - s)
                            lanes = jnp.arange(NL)
                            valid = lanes < min_grads
                            source_lanes = jnp.where(
                                valid, grad_start - aligned_start + lanes, 0
                            )
                            gathered_rows = plsc.load_gather(
                                grad_rows_vmem, (source_lanes,)
                            )
                            grad_tile = col_start // COL_TILE + i
                            grad_rows_vmem[...] = jnp.where(
                                valid,
                                gathered_rows * chunks_per_batch + grad_tile,
                                -1,
                            )

                            # Grab the expanded gradients
                            pltpu.sync_copy(
                                expanded_grad_hbm.at[
                                    plsc.Indices(grad_rows_vmem, ignored_value=-1)
                                ],
                                expanded_grad_vmem,  # [NL, COl_TILE]
                            )

                            @pl.loop(0, min_grads)
                            def _pool_gradient(grad_idx):
                                grad_buffer_vmem[...] += expanded_grad_vmem[grad_idx, :]

                        # Write gradient into the correct HBM location
                        # Note that (f_i, b_i, d_j) |-> gradient output index (F, B, D) row-major = f_i * O[i] +  b_i * D[i] + d_j
                        out_start = pl.multiple_of(
                            total_offset + batch_id * emb_dim + i * COL_TILE,
                            COL_TILE,
                        )

                        # Copy the result to the output
                        pltpu.sync_copy(
                            grad_buffer_vmem,
                            out_hbm.at[pl.ds(out_start, COL_TILE)],
                        )

        _inner()

    # The output should be in (F, B) row-major order
    return _kernel(
        grad_output,
        grad_rows,
        bag_offsets,
        column_offsets,
        rows,
        columns,
        deduped_offsets,
    )
