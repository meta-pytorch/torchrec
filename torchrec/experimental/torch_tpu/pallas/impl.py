#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch_tpu._internal import pallas  # pyre-ignore[21]
from torchrec.experimental.torch_tpu.pallas import (  # noqa: F401
    lookup,
    ops,
    pooled_lookup,
)

_fwd = pallas.jax_op("torchrec_pallas::embedding_lookup", lookup.embedding_lookup_jax)
_bwd_tc = pallas.jax_op(
    "torchrec_pallas::embedding_lookup_bwd_tc", lookup.embedding_lookup_bwd_jax
)
_pooled_fwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup",
    pooled_lookup.embedding_pooled_lookup_jax,
)
_pooled_bwd = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_lookup_bwd",
    pooled_lookup.embedding_pooled_gather_bwd_jax,
)


def embedding_lookup_tpu(indices, weights, emb_dim):
    return _fwd(indices=indices, dev_weights=weights, emb_dim=emb_dim)


def embedding_lookup_backward_tpu(grad_out, indices, num_rows, emb_dim):
    # Both modes use TC backward until SC backward is implemented
    return _bwd_tc(
        grad_out=grad_out, indices=indices, num_rows=num_rows, emb_dim=emb_dim
    )


def embedding_pooled_lookup_tpu(idx, table, pool):
    return _pooled_fwd(idx=idx, table=table, pool=pool)


def embedding_pooled_lookup_backward_tpu(grad_out, idx, pool, num_rows, emb_dim):
    return _pooled_bwd(
        grad_out=grad_out,
        idx=idx,
        pool=pool,
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
