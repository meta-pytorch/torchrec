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
lib.define(
    "embedding_pooled_batched_lookup_offset("
    "Tensor indices, Tensor offsets, Tensor weights, Tensor row_offsets, "
    "int emb_dim) -> Tensor"
)
lib.define(
    "embedding_pooled_batched_lookup_offset_backward("
    "Tensor grad_out, Tensor indices, Tensor offsets, Tensor row_offsets, "
    "int num_rows, int emb_dim) -> Tensor"
)
lib.define(
    "permute_pooled_embs("
    "Tensor pooled_embs, Tensor offset_dim_list, Tensor permute_list"
    ") -> Tensor"
)
lib.define(
    "permute_pooled_embs_backward("
    "Tensor grad_out, Tensor offset_dim_list, Tensor permute_list"
    ") -> Tensor"
)


def embedding_lookup_cpu(
    indices: torch.Tensor, weights: torch.Tensor, emb_dim: int
) -> torch.Tensor:
    _ = emb_dim
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


def embedding_pooled_batched_lookup_offset_cpu(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
    row_offsets: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    bag_indices = _bag_indices(offsets)
    active_indices = indices[: bag_indices.numel()]
    weight_indices = active_indices.long() + row_offsets[bag_indices].long()
    return weights.new_zeros((offsets.numel() - 1, emb_dim)).index_add_(
        0,
        bag_indices,
        weights[weight_indices],
    )


def embedding_pooled_batched_lookup_offset_backward_cpu(
    grad_out: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    row_offsets: torch.Tensor,
    num_rows: int,
    emb_dim: int,
) -> torch.Tensor:
    bag_indices = _bag_indices(offsets)
    active_indices = indices[: bag_indices.numel()]
    weight_indices = active_indices.long() + row_offsets[bag_indices].long()
    return grad_out.new_zeros((num_rows, emb_dim)).index_add_(
        0,
        weight_indices,
        grad_out[bag_indices],
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
lib.impl(
    "embedding_pooled_batched_lookup_offset",
    embedding_pooled_batched_lookup_offset_cpu,
    "CPU",
)
lib.impl(
    "embedding_pooled_batched_lookup_offset_backward",
    embedding_pooled_batched_lookup_offset_backward_cpu,
    "CPU",
)


def _permute_col_indices(
    offset_dim_list: torch.Tensor,
    permute_list: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    input_dims = offset_dim_list[1:] - offset_dim_list[:-1]
    output_offsets = torch.cat(
        [
            offset_dim_list.new_zeros(1),
            input_dims[permute_list.long()].cumsum(dim=0),
        ]
    )
    output_columns = torch.arange(
        output_size,
        dtype=offset_dim_list.dtype,
        device=offset_dim_list.device,
    )
    output_blocks = torch.searchsorted(output_offsets[1:], output_columns, right=True)
    return (
        offset_dim_list[permute_list.long()[output_blocks]]
        + output_columns
        - output_offsets[output_blocks]
    )


def permute_pooled_embs_cpu(
    pooled_embs: torch.Tensor,
    offset_dim_list: torch.Tensor,
    permute_list: torch.Tensor,
) -> torch.Tensor:
    col_indices = _permute_col_indices(
        offset_dim_list, permute_list, pooled_embs.shape[1]
    )
    return pooled_embs.index_select(1, col_indices.long())


def permute_pooled_embs_backward_cpu(
    grad_out: torch.Tensor,
    offset_dim_list: torch.Tensor,
    permute_list: torch.Tensor,
) -> torch.Tensor:
    col_indices = _permute_col_indices(offset_dim_list, permute_list, grad_out.shape[1])
    return grad_out.new_zeros(grad_out.shape).index_add_(
        1, col_indices.long(), grad_out
    )


lib.impl("permute_pooled_embs", permute_pooled_embs_cpu, "CPU")
lib.impl("permute_pooled_embs_backward", permute_pooled_embs_backward_cpu, "CPU")


def _setup_context(ctx, inputs, output) -> None:  # pyre-ignore[2]
    _ = output
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
    _ = output
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
    _ = output
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


def _setup_pooled_batched_offset_context(ctx, inputs, output) -> None:
    _ = output
    indices, offsets, weights, row_offsets, emb_dim = inputs
    ctx.save_for_backward(indices, offsets, row_offsets)
    ctx.num_rows = weights.shape[0]
    ctx.emb_dim = emb_dim


def _pooled_batched_offset_backward(ctx, grad_out: torch.Tensor):  # pyre-ignore[2,3]
    indices, offsets, row_offsets = ctx.saved_tensors
    grad_weights = torch.ops.torchrec.embedding_pooled_batched_lookup_offset_backward(
        grad_out.contiguous(),
        indices,
        offsets,
        row_offsets,
        ctx.num_rows,
        ctx.emb_dim,
    )
    return None, None, grad_weights, None, None


def _permute_setup_context(ctx, inputs, output) -> None:  # pyre-ignore[2]
    _ = output
    _, offset_dim_list, permute_list = inputs
    ctx.save_for_backward(offset_dim_list, permute_list)


def _permute_backward(ctx, grad_out: torch.Tensor):  # pyre-ignore[2,3]
    offset_dim_list, permute_list = ctx.saved_tensors
    grad_input = torch.ops.torchrec.permute_pooled_embs_backward(
        grad_out.contiguous(),
        offset_dim_list,
        permute_list,
    )
    return grad_input, None, None


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
torch.library.register_autograd(
    "torchrec::embedding_pooled_batched_lookup_offset",
    _pooled_batched_offset_backward,
    setup_context=_setup_pooled_batched_offset_context,
)
torch.library.register_autograd(
    "torchrec::permute_pooled_embs",
    _permute_backward,
    setup_context=_permute_setup_context,
)
