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
lib.define("embedding_pooled_lookup(Tensor idx, Tensor table, int pool) -> Tensor")
lib.define(
    "embedding_pooled_lookup_backward("
    "Tensor grad_out, Tensor idx, int pool, int num_rows, int emb_dim) -> Tensor"
)
lib.define(
    "embedding_pooled_lookup_offset("
    "Tensor indices, Tensor offsets, Tensor weights, int emb_dim) -> Tensor"
)
lib.define(
    "embedding_pooled_lookup_offset_backward("
    "Tensor grad_out, Tensor indices, Tensor offsets, "
    "int num_rows, int emb_dim) -> Tensor"
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


def embedding_pooled_lookup_cpu(
    idx: torch.Tensor, table: torch.Tensor, pool: int
) -> torch.Tensor:
    return table[idx.long()].reshape(-1, pool, table.shape[1]).sum(dim=1)


def embedding_pooled_lookup_backward_cpu(
    grad_out: torch.Tensor,
    idx: torch.Tensor,
    pool: int,
    num_rows: int,
    emb_dim: int,
) -> torch.Tensor:
    return grad_out.new_zeros((num_rows, emb_dim)).index_add_(
        0,
        idx.long(),
        grad_out.repeat_interleave(pool, dim=0),
    )


def _bag_indices(offsets: torch.Tensor) -> torch.Tensor:
    lengths = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(offsets.numel() - 1, device=offsets.device),
        lengths.long(),
    )


def embedding_pooled_lookup_offset_cpu(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    bag_indices = _bag_indices(offsets)
    return weights.new_zeros((offsets.numel() - 1, emb_dim)).index_add_(
        0,
        bag_indices,
        weights[indices.long()],
    )


def embedding_pooled_lookup_offset_backward_cpu(
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    num_rows: int,
    emb_dim: int,
) -> torch.Tensor:
    return grad_out.new_zeros((num_rows, emb_dim)).index_add_(
        0,
        indices.long(),
        grad_out[_bag_indices(offsets)],
    )


lib.impl("embedding_pooled_lookup", embedding_pooled_lookup_cpu, "CPU")
lib.impl(
    "embedding_pooled_lookup_backward", embedding_pooled_lookup_backward_cpu, "CPU"
)
lib.impl("embedding_pooled_lookup_offset", embedding_pooled_lookup_offset_cpu, "CPU")
lib.impl(
    "embedding_pooled_lookup_offset_backward",
    embedding_pooled_lookup_offset_backward_cpu,
    "CPU",
)


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
    return None, grad_weights, None


torch.library.register_autograd(
    "torchrec::embedding_lookup", _backward, setup_context=_setup_context
)


def _setup_pooled_context(ctx, inputs, output) -> None:  # pyre-ignore[2]
    idx, table, pool = inputs
    ctx.save_for_backward(idx)
    ctx.pool = pool
    ctx.num_rows = table.shape[0]
    ctx.emb_dim = table.shape[1]


def _pooled_backward(ctx, grad_out: torch.Tensor):  # pyre-ignore[2,3]
    (idx,) = ctx.saved_tensors
    grad_table = torch.ops.torchrec.embedding_pooled_lookup_backward(
        grad_out.contiguous(), idx, ctx.pool, ctx.num_rows, ctx.emb_dim
    )
    return None, grad_table, None


def _setup_pooled_offset_context(ctx, inputs, output) -> None:  # pyre-ignore[2]
    indices, offsets, weights, emb_dim = inputs
    ctx.save_for_backward(indices, offsets)
    ctx.num_rows = weights.shape[0]
    ctx.emb_dim = emb_dim


def _pooled_offset_backward(ctx, grad_out: torch.Tensor):  # pyre-ignore[2,3]
    indices, offsets = ctx.saved_tensors
    grad_weights = torch.ops.torchrec.embedding_pooled_lookup_offset_backward(
        grad_out.contiguous(), indices, offsets, ctx.num_rows, ctx.emb_dim
    )
    return None, None, grad_weights, None


torch.library.register_autograd(
    "torchrec::embedding_pooled_lookup",
    _pooled_backward,
    setup_context=_setup_pooled_context,
)
torch.library.register_autograd(
    "torchrec::embedding_pooled_lookup_offset",
    _pooled_offset_backward,
    setup_context=_setup_pooled_offset_context,
)
