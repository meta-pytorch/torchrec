#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc


def lookup_mode() -> str:
    """Which kernel variant to run, from ``$LOOKUP_MODE``."""
    _VALID_LOOKUP_MODES = ("v1_tpu", "v1_sc")
    _DEFAULT_LOOKUP_MODE = "v1_sc"

    mode = os.environ.get("LOOKUP_MODE", _DEFAULT_LOOKUP_MODE)
    if mode not in _VALID_LOOKUP_MODES:
        raise ValueError(f"LOOKUP_MODE={mode!r} must be one of {_VALID_LOOKUP_MODES}")
    return mode


def _kernel_v1(indices_ref, dev_weights_ref, embeddings_ref):
    emb_dim = embeddings_ref.shape[1]

    @pl.loop(0, indices_ref.shape[0])
    def gather(i):
        idx = indices_ref[i]
        pltpu.sync_copy(
            dev_weights_ref.at[pl.ds(idx, 1), pl.ds(0, emb_dim)],
            embeddings_ref.at[pl.ds(i, 1), pl.ds(0, emb_dim)],
        )


def get_params(indices: jax.Array, emb_dim: int):
    # Calculate optimal number of indices to chunk
    VMEM_CAPCITY = 524288  # 67108864 // 4

    n = indices.shape[0]
    maximal_batch = int(VMEM_CAPCITY / (4 * (emb_dim + 1)))
    if n <= maximal_batch:
        indices_chunk = n
    else:
        indices_chunk = 2 ** int(math.log2(maximal_batch))
        indices_chunk = max(indices_chunk, 128)

    # indices_chunk = ((indices_chunk + 128 - 1) // 128) * 128
    grid_size = (n + indices_chunk - 1) // indices_chunk

    # Pad indices up to grid_size * indices_chunk (a multiple of TILE).
    padded_n = grid_size * indices_chunk
    if padded_n > n:
        indices = jnp.pad(indices, (0, padded_n - n), constant_values=0)

    return (
        indices,
        indices_chunk,
        grid_size,
    )


@jax.jit(static_argnames=["emb_dim"])  # pyre-ignore[6]
def run_tpu_lookup(
    indices: jax.Array,
    dev_weights: jax.Array,
    emb_dim: int,
) -> jax.Array:
    kernel = _kernel_v1
    n = indices.shape[0]
    indices, indices_chunk, grid_size = get_params(indices, emb_dim)

    return pl.pallas_call(
        kernel,
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((indices_chunk,), lambda i: i, memory_space=pltpu.SMEM),
            # keep embedding table in HBM
            pl.BlockSpec(dev_weights.shape, lambda i: (0, 0), memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((indices_chunk, emb_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((indices.shape[0], emb_dim), jnp.float32),
        debug=False,  # True,
    )(indices, dev_weights)[:n, :]


def get_params_sc(indices: jax.Array, emb_dim: int, num_subcores: int = 1):
    # Calculate optimal number of indices to chunk
    VMEM_CAPCITY = 524288 // 4  # 67108864 // 4

    TILE = 8
    n = indices.shape[0]
    maximal_batch = int(VMEM_CAPCITY / (4 * (emb_dim + 1)))
    if n <= maximal_batch:
        indices_chunk = n
    else:
        indices_chunk = 2 ** int(math.log2(maximal_batch))

    # Round the chunk up to a multiple of TILE so every window offset
    indices_chunk = ((indices_chunk + TILE - 1) // TILE) * TILE

    grid_size = (n + indices_chunk - 1) // indices_chunk

    # Distribute the grid size to number of num_subcores
    grid_size = (((grid_size + num_subcores - 1)) // num_subcores) * num_subcores

    # Pad indices up to grid_size * indices_chunk (a multiple of TILE).
    padded_n = grid_size * indices_chunk
    if padded_n > n:
        indices = jnp.pad(indices, (0, padded_n - n), constant_values=0)

    # print(f"Grid Size {grid_size}, and indices chunk {indices_chunk}")
    return (
        indices,
        indices_chunk,
        grid_size,
    )


# @jax.custom_vjp Needed for backward pass
@jax.jit(static_argnames=["emb_dim"])  # pyre-ignore[6]
def run_sc_lookup(
    indices: jax.Array,
    dev_weights: jax.Array,
    emb_dim: int,
) -> jax.Array:
    # Get Hardware information
    assert (sc_info := pltpu.get_tpu_info().sparse_core)  # pyre-ignore[16]

    vector_mesh = plsc.VectorSubcoreMesh(  # pyre-ignore[6]
        core_axis_name="core", subcore_axis_name="subcore"
    )

    _, sc_num_subcores = sc_info.num_cores, sc_info.num_subcores

    n = indices.shape[0]
    padded_indices, indices_chunk, grid_size = get_params_sc(
        indices, emb_dim, sc_num_subcores
    )
    padded_n = padded_indices.shape[0]

    @pl.kernel(  # pyre-ignore[16]
        out_type=jax.ShapeDtypeStruct((padded_n, emb_dim), dev_weights.dtype),
        mesh=vector_mesh,
    )
    def _kernel_v2(weights_hbm, indices_hbm, out_hbm):
        # idx_smem and out_vmem are mapped automatically by emit_pipeline
        def body(idx_smem, out_vmem):
            pltpu.sync_copy(weights_hbm.at[idx_smem], out_vmem)

        pltpu.emit_pipeline(
            body,
            grid=(grid_size,),
            in_specs=[
                pl.BlockSpec((indices_chunk,), lambda i: (i,), memory_space=pltpu.VMEM)
            ],
            out_specs=[pl.BlockSpec((indices_chunk, emb_dim), lambda i: (i, 0))],
            core_axis_name="subcore",
            dimension_semantics=(pltpu.PARALLEL,),
        )(indices_hbm, out_hbm)

    out = _kernel_v2(dev_weights, padded_indices)
    return out[:n, :]


def embedding_lookup_jax(
    indices: jax.Array,
    dev_weights: jax.Array,
    emb_dim: int,
) -> jax.Array:
    if lookup_mode() == "v1_tpu":
        return run_tpu_lookup(indices, dev_weights, emb_dim)
    return run_sc_lookup(indices, dev_weights, emb_dim)


def embedding_lookup_bwd_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    # Runs on TensorCores
    # note: it returns a matrix of the whole embedding table here
    #       atomic adds work on tensor cores
    #       fills the zero gradient array with the gradients at indices
    return (
        jnp.zeros((num_rows, emb_dim), dtype=grad_out.dtype).at[indices].add(grad_out)
    )
