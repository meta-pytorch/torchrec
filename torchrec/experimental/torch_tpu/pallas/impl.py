#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tpu._internal import pallas  # pyre-ignore[21]
from torchrec.experimental.torch_tpu.pallas import (
    lookup,
    ops,
    permute_embs,
    pooled_lookup_offset,
    pooled_lookup_padded,
)

_fwd = pallas.jax_op("torchrec_pallas::embedding_lookup", lookup.embedding_lookup_jax)
_bwd_tc = pallas.jax_op(
    "torchrec_pallas::embedding_lookup_bwd_tc", lookup.embedding_lookup_bwd_jax
)
_pooled_padded_fwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup_padded",
    pooled_lookup_padded.embedding_pooled_lookup_jax,
)
_pooled_padded_bwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup_padded_bwd",
    pooled_lookup_padded.embedding_pooled_gather_bwd_jax,
)
_pooled_offset_fwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup_offset",
    pooled_lookup_offset.embedding_pooled_lookup_jax,
)
_pooled_offset_bwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup_offset_bwd",
    pooled_lookup_offset.embedding_pooled_lookup_bwd_jax,
)
_pooled_batched_offset_fwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_batched_lookup_offset",
    pooled_lookup_offset.embedding_pooled_batched_lookup_jax,
)
_pooled_batched_offset_bwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_batched_lookup_offset_bwd",
    pooled_lookup_offset.embedding_pooled_batched_lookup_bwd_jax,
)
_permute_pooled_embs_sc_fwd = pallas.jax_op(
    "torchrec_pallas::permute_pooled_embs_auto_grad_split",
    permute_embs.permute_pooled_embs_auto_grad_split_kernel,
)
_permute_pooled_embs_tc_fwd = pallas.jax_op(
    "torchrec_pallas::permute_pooled_embs",
    permute_embs.permute_pooled_embs_tc,
)
_permute_pooled_embs_tc_bwd = pallas.jax_op(
    "torchrec_pallas::permute_pooled_embs_backward",
    permute_embs.permute_pooled_embs_tc_bwd,
)


def embedding_lookup_tpu(indices, weights, emb_dim):
    return _fwd(indices=indices, dev_weights=weights, emb_dim=emb_dim)


def embedding_lookup_backward_tpu(grad_out, indices, num_rows, emb_dim):
    return _bwd_tc(
        grad_out=grad_out, indices=indices, num_rows=num_rows, emb_dim=emb_dim
    )


def embedding_pooled_lookup_tpu(idx, table, pool):
    return _pooled_padded_fwd(idx=idx, table=table, pool=pool)


def embedding_pooled_lookup_backward_tpu(grad_out, idx, pool, num_rows, emb_dim):
    return _pooled_padded_bwd(
        grad_out=grad_out,
        idx=idx,
        pool=pool,
        num_rows=num_rows,
        emb_dim=emb_dim,
    )


def embedding_pooled_lookup_offset_tpu(indices, offsets, weights, emb_dim):
    return _pooled_offset_fwd(
        indices=indices,
        offsets=offsets,
        dev_weights=weights,
        emb_dim=emb_dim,
    )


def embedding_pooled_lookup_offset_backward_tpu(
    grad_out, indices, offsets, num_rows, emb_dim
):
    return _pooled_offset_bwd(
        grad_out=grad_out,
        indices=indices,
        offsets=offsets,
        num_rows=num_rows,
        emb_dim=emb_dim,
    )


def embedding_pooled_batched_lookup_offset_tpu(
    indices, offsets, weights, row_offsets, emb_dim
):
    return _pooled_batched_offset_fwd(
        indices=indices,
        offsets=offsets,
        dev_weights=weights,
        row_offsets=row_offsets,
        emb_dim=emb_dim,
    )


def embedding_pooled_batched_lookup_offset_backward_tpu(
    grad_out, indices, offsets, row_offsets, num_rows, emb_dim
):
    return _pooled_batched_offset_bwd(
        grad_out=grad_out,
        indices=indices,
        offsets=offsets,
        row_offsets=row_offsets,
        num_rows=num_rows,
        emb_dim=emb_dim,
    )


def permute_pooled_embs_tpu(pooled_embs, offset_dim_list, permute_list):
    return _permute_pooled_embs_tc_fwd(
        pooled_embs=pooled_embs,
        offset_dim_list=offset_dim_list,
        permute_list=permute_list,
    )


def permute_pooled_embs_backward_tpu(grad_out, offset_dim_list, permute_list):
    return _permute_pooled_embs_tc_bwd(
        grad_out=grad_out,
        offset_dim_list=offset_dim_list,
        permute_list=permute_list,
    )


def permute_pooled_embs_sparse_core_tpu(
    pooled_embs,
    offset_dim_list,
    permute_list,
    output_offset_dim_list,
    col_block_size,
):
    return _permute_pooled_embs_sc_fwd(
        pooled_embs=pooled_embs,
        offset_dim_list=offset_dim_list,
        permute_list=permute_list,
        inv_offset_dim_list=output_offset_dim_list,
        col_block_size=col_block_size,
    )


lib: torch.library.Library = torch.library.Library("torchrec", "IMPL")
lib.impl("embedding_lookup", embedding_lookup_tpu, "TPU")
lib.impl("embedding_lookup_backward", embedding_lookup_backward_tpu, "TPU")
lib.impl("embedding_pooled_lookup", embedding_pooled_lookup_tpu, "TPU")
lib.impl(
    "embedding_pooled_lookup_backward", embedding_pooled_lookup_backward_tpu, "TPU"
)
lib.impl("embedding_pooled_lookup_offset", embedding_pooled_lookup_offset_tpu, "TPU")
lib.impl(
    "embedding_pooled_lookup_offset_backward",
    embedding_pooled_lookup_offset_backward_tpu,
    "TPU",
)
lib.impl(
    "embedding_pooled_batched_lookup_offset",
    embedding_pooled_batched_lookup_offset_tpu,
    "TPU",
)
lib.impl(
    "embedding_pooled_batched_lookup_offset_backward",
    embedding_pooled_batched_lookup_offset_backward_tpu,
    "TPU",
)
lib.impl("permute_pooled_embs", permute_pooled_embs_tpu, "TPU")
lib.impl("permute_pooled_embs_backward", permute_pooled_embs_backward_tpu, "TPU")


_ = ops
