#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


lib: torch.library.Library = torch.library.Library("torchrec", "FRAGMENT")

lib.define("embedding_lookup(Tensor indices, Tensor weights, int emb_dim) -> Tensor")
lib.define(
    "embedding_lookup_backward("
    "Tensor grad_out, Tensor indices, int num_rows, int emb_dim) -> Tensor"
)


def embedding_lookup_cpu(
    indices: torch.Tensor, weights: torch.Tensor, emb_dim: int
) -> torch.Tensor:
    return weights[indices.long()]


def embedding_lookup_backward_cpu(
    grad_out: torch.Tensor, indices: torch.Tensor, num_rows: int, emb_dim: int
) -> torch.Tensor:
    return grad_out.new_zeros((num_rows, emb_dim)).index_add_(
        0, indices.long(), grad_out
    )


lib.impl("embedding_lookup", embedding_lookup_cpu, "CPU")
lib.impl("embedding_lookup_backward", embedding_lookup_backward_cpu, "CPU")


def _setup_context(ctx, inputs, output) -> None:  # pyre-ignore[2]
    indices, weights, emb_dim = inputs
    ctx.save_for_backward(indices)
    ctx.num_rows = weights.shape[0]
    ctx.emb_dim = emb_dim


def _backward(ctx, grad_out: torch.Tensor):  # pyre-ignore[2,3]
    (indices,) = ctx.saved_tensors
    grad_weights = torch.ops.torchrec.embedding_lookup_backward(
        grad_out.contiguous(), indices, ctx.num_rows, ctx.emb_dim
    )
    # One grad per forward input: (indices, weights, emb_dim).
    return None, grad_weights, None


torch.library.register_autograd(
    "torchrec::embedding_lookup", _backward, setup_context=_setup_context
)
