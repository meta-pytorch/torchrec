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
from torchrec.experimental.torch_tpu.pallas.ops import _validate_vbe_metadata_shapes


@functools.partial(
    jax.jit, static_argnames=["num_bags", "emb_dim", "max_B_feature_rank"]
)
def generate_vbe_metadata_sc(
    B_offsets: jax.Array,
    B_offsets_rank_per_feature: jax.Array,
    output_offsets_feature_rank: jax.Array,
    num_bags: int,
    emb_dim: int,
    max_B_feature_rank: int,
) -> jax.Array:
    """Write input-bag element offsets from feature/rank chunks on SparseCore.

    Packed rank-major output addresses must fit int32, as in the pooled kernels.
    """
    _validate_vbe_metadata_shapes(
        B_offsets.shape,
        B_offsets_rank_per_feature.shape,
        output_offsets_feature_rank.shape,
    )
    if num_bags == 0:
        return jnp.empty((0,), dtype=output_offsets_feature_rank.dtype)
    if max_B_feature_rank <= 0:
        raise ValueError("max_B_feature_rank must be positive for nonempty VBE")

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
    BAGS_PER_TILE = 1024

    # Get the required information
    num_features = B_offsets.shape[0] - 1
    num_ranks = B_offsets_rank_per_feature.shape[1] - 1
    input_starts = B_offsets[:-1, None] + B_offsets_rank_per_feature[:, :-1]
    counts = jnp.diff(B_offsets_rank_per_feature, axis=1)
    output_starts = output_offsets_feature_rank[:-1].reshape(num_ranks, num_features).T
    chunk_metadata = jnp.stack(
        (input_starts.flatten(), counts.flatten(), output_starts.flatten()), axis=1
    )
    # Broadcast only the compact chunk metadata, so scalar loads use aligned DMA.
    chunk_metadata = jnp.broadcast_to(
        chunk_metadata[:, :, None], (num_features * num_ranks, 3, NL)
    ).reshape(-1)

    # Each program owns a feature/rank chunk and one tile of its bags.
    tiles_per_chunk = (max_B_feature_rank + NL - 1 + BAGS_PER_TILE - 1) // BAGS_PER_TILE
    num_programs = num_features * num_ranks * tiles_per_chunk
    number_of_blocks = (num_programs + num_subcores - 1) // num_subcores

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct((num_bags,), output_offsets_feature_rank.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((3 * NL,), chunk_metadata.dtype),
            pltpu.VMEM((BAGS_PER_TILE,), jnp.int32),
            pltpu.VMEM((BAGS_PER_TILE,), output_offsets_feature_rank.dtype),
        ],
        name="generate_vbe_metadata_sc",
        mesh=mesh,
    )
    def _kernel(
        metadata_hbm,
        row_output_offsets_hbm,
        metadata_vmem,
        destinations_vmem,
        values_vmem,
    ):
        @functools.partial(pltpu.emit_pipeline, grid=(number_of_blocks,))
        def _inner():
            subcore_id = jax.lax.axis_index("c") * sc.num_subcores + jax.lax.axis_index(
                "s"
            )
            program = pl.program_id(0) * num_subcores + subcore_id

            @pl.when(program < num_programs)
            def _valid_program():
                chunk = program // tiles_per_chunk
                tile_index = program % tiles_per_chunk
                metadata_start = pl.multiple_of(chunk * 3 * NL, NL)
                pltpu.sync_copy(
                    metadata_hbm.at[pl.ds(metadata_start, 3 * NL)], metadata_vmem
                )
                input_start = metadata_vmem[pl.ds(0, NL)][0]
                count = metadata_vmem[pl.ds(NL, NL)][0]
                output_start = metadata_vmem[pl.ds(2 * NL, NL)][0]
                tile_base = pl.multiple_of(
                    (input_start // NL) * NL + tile_index * BAGS_PER_TILE, NL
                )
                tile_start = tile_base - input_start

                @pl.when(tile_start < count)
                def _write_tile():
                    @pl.loop(0, BAGS_PER_TILE, step=NL)
                    def _generate_lanes(lane_start):
                        b = tile_start + lane_start + jnp.arange(NL, dtype=jnp.int32)
                        valid = (b >= 0) & (b < count)
                        destinations_vmem[pl.ds(lane_start, NL)] = jnp.where(
                            valid, input_start + b, -1
                        )
                        values_vmem[pl.ds(lane_start, NL)] = output_start + b * emb_dim

                    full_tile = (tile_start >= 0) & (
                        tile_start + BAGS_PER_TILE <= count
                    )

                    @pl.when(full_tile)
                    def _write_full_tile():
                        pltpu.sync_copy(
                            values_vmem,
                            row_output_offsets_hbm.at[pl.ds(tile_base, BAGS_PER_TILE)],
                        )

                    @pl.when(~full_tile)
                    def _write_partial_tile():
                        pltpu.sync_copy(
                            values_vmem,
                            row_output_offsets_hbm.at[
                                plsc.Indices(destinations_vmem, ignored_value=-1)
                            ],
                        )

        _inner()

    return _kernel(chunk_metadata)
