#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from torch_tpu._internal import pallas  # pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.pallas import (
    batch_index,
    lookup,
    ops,
    permute_embs,
    pooled_lookup_offset,
    pooled_lookup_offset_bwd,
    pooled_lookup_padded,
    segment_csr,
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
_pooled_batched_offset_adagrad_update = pallas.jax_op(
    "torchrec_pallas::embedding_pooled_batched_lookup_adagrad_update",
    pooled_lookup_offset_bwd.embedding_pooled_batched_lookup_fused_adagrad_bwd_jax,
    donate_argnums=(4, 5),
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
_batch_index_select_sc = pallas.jax_op(
    "torchrec_pallas::batch_index_select_dim0",
    batch_index.batch_index_select_sc,
)
_batch_index_select_sc_bwd = pallas.jax_op(
    "torchrec_pallas::batch_index_select_dim0_backward",
    batch_index.batch_index_select_bwd_sc,
)


def _batch_index_select_setup_context(ctx, inputs, output) -> None:
    _ = output
    input_tensor, indices, input_rows, input_columns, _total_columns = inputs
    ctx.save_for_backward(indices, input_rows, input_columns)
    ctx.input_numel = input_tensor.numel()


def _batch_index_select_backward(ctx, grad_output: torch.Tensor):
    indices, input_rows, input_columns = ctx.saved_tensors
    grad_inputs = _batch_index_select_sc_bwd(
        grad_output=grad_output.contiguous(),
        indices=indices,
        input_rows=input_rows,
        input_columns=input_columns,
        permute_output_dim_0_1=True,
        output_size=ctx.input_numel,
    )
    return grad_inputs, None, None, None, None


_batch_index_select_sc.register_autograd(
    _batch_index_select_backward, setup_context=_batch_index_select_setup_context
)

_segment_sum_csr_sc = pallas.jax_op(
    "torchrec_pallas::segment_sum_csr",
    segment_csr.segment_sum_csr_sc,
)


def segment_sum_csr_tpu(
    batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    output_dtype = values.dtype
    # TODO: Re-evaluate the int32
    csr_seg = csr_seg.to(torch.int32)
    if values.dtype == torch.int64:
        values = values.to(torch.int32)
    return _segment_sum_csr_sc(batch_size, csr_seg, values).to(output_dtype)


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


def embedding_pooled_batched_lookup_offset_fused_adagrad_tpu(
    indices, offsets, weights, row_offsets, momentum, lr, eps, emb_dim
):
    return embedding_pooled_batched_lookup_offset_tpu(
        indices, offsets, weights, row_offsets, emb_dim
    )


def embedding_pooled_batched_lookup_offset_adagrad_update_tpu(
    grad_out, indices, offsets, row_offsets, weights, momentum, lr, eps
):
    new_weights, new_momentum = _pooled_batched_offset_adagrad_update(
        grad_out=grad_out,
        indices=indices,
        offsets=offsets,
        row_offsets=row_offsets,
        emb_tables=weights,
        momentum=momentum,
        lr=lr,
        eps=eps,
        emb_dim=grad_out.shape[1],
    )
    weights.set_(new_weights)
    momentum.set_(new_momentum)


def embedding_pooled_batched_lookup_offset_backward_tpu(
    grad_out, indices, offsets, row_offsets, num_rows, emb_dim, flat_weights=False
):
    return _pooled_batched_offset_bwd(
        grad_out=grad_out,
        indices=indices,
        offsets=offsets,
        row_offsets=row_offsets,
        num_rows=num_rows,
        emb_dim=emb_dim,
        flat_weights=flat_weights,
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


def batch_index_select_dim0_tpu(inputs, indices, input_rows, input_columns):
    return _batch_index_select_sc(
        inputs=inputs,
        indices=indices.to(torch.int32),
        input_rows=torch.tensor(input_rows, dtype=torch.int32, device=inputs.device),
        input_columns=torch.tensor(
            input_columns, dtype=torch.int32, device=inputs.device
        ),
        total_columns=sum(input_columns),
    )


def batch_index_select_dim0_backward_tpu_impl(
    grad_output, indices, input_rows, input_columns, input_numel
):
    if any(columns % 16 != 0 for columns in input_columns):
        raise NotImplementedError(
            f"embedding dimensions must be divisible by 16, got {input_columns}"
        )
    # TODO: re-evaluate the int32 conversion here
    return _batch_index_select_sc_bwd(
        grad_output=grad_output,
        indices=indices.to(torch.int32),
        input_rows=torch.tensor(
            input_rows, dtype=torch.int32, device=grad_output.device
        ),
        input_columns=torch.tensor(
            input_columns, dtype=torch.int32, device=grad_output.device
        ),
        permute_output_dim_0_1=True,
        output_size=input_numel,
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
lib.impl(
    "embedding_pooled_batched_lookup_offset_fused_adagrad",
    embedding_pooled_batched_lookup_offset_fused_adagrad_tpu,
    "TPU",
)
lib.impl(
    "embedding_pooled_batched_lookup_offset_adagrad_update",
    embedding_pooled_batched_lookup_offset_adagrad_update_tpu,
    "TPU",
)
lib.impl("permute_pooled_embs", permute_pooled_embs_tpu, "TPU")
lib.impl("permute_pooled_embs_backward", permute_pooled_embs_backward_tpu, "TPU")
_ = ops
