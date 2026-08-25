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

_ = ops
