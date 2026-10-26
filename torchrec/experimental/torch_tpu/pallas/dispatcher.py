#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file dispatches many of the FBGEMM kernels over to pallas kernels, or if they're not
implemented, the CPU version is used.
"""

from __future__ import annotations

import torch
from torch_tpu._internal import pallas  # pyrefly: ignore[missing-import]
from torchrec.experimental.torch_tpu.pallas import (
    associative_scan,
    impl,
    permute_1d,
    permute_2d,
)


_complete_cumsum = pallas.jax_op(
    "torchrec_pallas::associative_complete_cumsum",
    associative_scan.complete_cumsum_jax,
)
_permute_1d_sparse_data = pallas.jax_op(
    "torchrec_pallas::permute_1d_sparse_data",
    permute_1d.permute_1d_sparse_data_sc,
)
_permute_2d_sparse_data = pallas.jax_op(
    "torchrec_pallas::permute_2d_sparse_data",
    permute_2d.permute_2d_sparse_data_sc,
)
_permute_1d_sparse_data_bwd = pallas.jax_op(
    "torchrec_pallas::permute_1d_sparse_data_backward",
    permute_1d.permute_1d_sparse_data_backward_sc,
)


def _permute_1d_setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], output[0])
    ctx.mark_non_differentiable(output[0])


def _permute_1d_backward(ctx, _grad_lengths, grad_output):
    permute, permuted_lengths = ctx.saved_tensors
    grad_input = _permute_1d_sparse_data_bwd(
        grad_output=grad_output.contiguous(),
        permute=permute,
        permuted_lengths=permuted_lengths,
    )
    return None, None, grad_input, None, None


_permute_1d_sparse_data.register_autograd(
    _permute_1d_backward, setup_context=_permute_1d_setup_context
)
lib = torch.library.Library("fbgemm", "IMPL")


def permute_2D_sparse_data(
    permute, lengths, values, weights=None, permuted_lengths_sum=None
):
    if permuted_lengths_sum is not None and weights is None:
        values = values.to(torch.int32) if values.dtype == torch.int64 else values
        permuted_lengths, permuted_values = _permute_2d_sparse_data(
            permute=permute.to(torch.int32),
            lengths=lengths.to(torch.int32),
            values=values,
            weights=None,
            permuted_lengths_sum=permuted_lengths_sum,
        )
        return permuted_lengths, permuted_values, None

    (permuted_lengths_after_sparse_data_all2all, sharded_input_embeddings, w) = (
        torch.ops.fbgemm.permute_2D_sparse_data(
            permute.to("cpu"),
            lengths.to("cpu"),
            values.to("cpu"),
            weights.to("cpu") if weights is not None else None,
            permuted_lengths_sum,
        )
    )
    permuted_lengths_after_sparse_data_all2all = (
        permuted_lengths_after_sparse_data_all2all.to("tpu")
    )
    sharded_input_embeddings = sharded_input_embeddings.to("tpu")
    w = w.to("tpu") if w is not None else w
    return (permuted_lengths_after_sparse_data_all2all, sharded_input_embeddings, w)


def permute_1D_sparse_data(
    permute, lengths, indices, weights=None, permuted_lengths_sum=None
):
    # TODO: Re-evaluate the int32 path that is used.
    if permuted_lengths_sum is not None and weights is None:
        indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
        permuted_lengths, permuted_indices = _permute_1d_sparse_data(
            permute=permute.to(torch.int32),
            lengths=lengths.to(torch.int32),
            indices=indices,
            weights=None,
            permuted_lengths_sum=permuted_lengths_sum,
        )
        return permuted_lengths, permuted_indices, None

    (permuted_lengths_after_sparse_data_all2all, sharded_input_embeddings, w) = (
        torch.ops.fbgemm.permute_1D_sparse_data(
            permute.to("cpu"),
            lengths.to("cpu"),
            indices.to("cpu"),
            weights.to("cpu") if weights is not None else None,
            permuted_lengths_sum,
        )
    )
    permuted_lengths_after_sparse_data_all2all = (
        permuted_lengths_after_sparse_data_all2all.to("tpu")
    )
    sharded_input_embeddings = sharded_input_embeddings.to("tpu")
    w = w.to("tpu") if w is not None else w
    return (permuted_lengths_after_sparse_data_all2all, sharded_input_embeddings, w)


def block_bucketize_sparse_features(
    lengths: torch.Tensor,
    indices: torch.Tensor,
    bucketize_pos: bool,
    sequence: bool,
    block_sizes: torch.Tensor,
    my_size: int,
    weights: torch.Tensor | None = None,
    batch_size_per_feature: torch.Tensor | None = None,
    max_B: int = -1,
    block_bucketize_pos: torch.Tensor | None = None,
    keep_orig_idx: bool = False,
    total_num_blocks: torch.Tensor | None = None,
    keep_orig_idx_per_feature: torch.Tensor | None = None,
):
    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
    ) = torch.ops.fbgemm.block_bucketize_sparse_features(
        lengths.to("cpu"),
        indices.to("cpu"),
        bucketize_pos,
        sequence,
        block_sizes.to("cpu"),
        my_size,
        weights.to("cpu") if weights is not None else None,
        (
            batch_size_per_feature.to("cpu")
            if batch_size_per_feature is not None
            else None
        ),
        max_B,
        (
            [t.to("cpu") for t in block_bucketize_pos]
            if block_bucketize_pos is not None
            else None
        ),
        keep_orig_idx,
        total_num_blocks.to("cpu") if total_num_blocks is not None else None,
        (
            keep_orig_idx_per_feature.to("cpu")
            if keep_orig_idx_per_feature is not None
            else None
        ),
    )
    bucketized_lengths = bucketized_lengths.to("tpu")
    bucketized_indices = bucketized_indices.to("tpu")
    bucketized_weights = (
        bucketized_weights.to("tpu") if bucketized_weights is not None else None
    )
    pos = pos.to("tpu") if pos is not None else None
    unbucketize_permute = (
        unbucketize_permute.to("tpu") if unbucketize_permute is not None else None
    )
    return (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
    )


def invert_permute(permute):
    return torch.ops.fbgemm.invert_permute(permute.to("cpu")).to("tpu")


def permute_pooled_embs_auto_grad_split(
    pooled_embs,
    offset_dim_list,
    permute_list,
    inv_offset_dim_list,
    inv_permute_list,
):
    """CPU fallback for the CW output-dist feature-order permute.

    Restores feature order after the column-wise PooledEmbeddingsAllToAll. The fbgemm
    kernel has no TPU implementation, so round-trip through CPU like the sparse-side
    permutes above. Autograd flows through the `.to()` calls, so the op's backward runs
    on CPU as well.
    """
    return torch.ops.fbgemm.permute_pooled_embs_auto_grad_split(
        pooled_embs.to("cpu"),
        offset_dim_list.to("cpu"),
        permute_list.to("cpu"),
        inv_offset_dim_list.to("cpu"),
        inv_permute_list.to("cpu"),
    ).to("tpu")


def segment_sum_csr_tpu(
    batch_size: int, csr_seg: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    return impl.segment_sum_csr_tpu(batch_size, csr_seg, values)


def asynchronous_complete_cumsum_tpu(array):
    return _complete_cumsum(array=array.to(torch.int32))


def batch_index_select_dim0_tpu(
    inputs,
    indices,
    input_num_indices,
    input_rows,
    input_columns,
    permute_output_dim_0_1=False,
):
    batch_size = input_num_indices[0]
    if (
        permute_output_dim_0_1
        and batch_size % 16 == 0
        and all(size == batch_size for size in input_num_indices)
        and all(columns % 16 == 0 for columns in input_columns)
    ):
        # FBGEMM requires these conditions as well
        if input_num_indices[0] <= 0:
            raise ValueError(
                f"input_num_indices {input_num_indices[0]} elements should be non-zero positive."
            )
        if not all(size == input_num_indices[0] for size in input_num_indices):
            raise ValueError(
                "All input_num_indices must be the same if `permute_output_dim_0_1` is true."
                f"Instead it is {input_num_indices}."
            )

        return impl.batch_index_select_dim0_tpu(
            inputs, indices, input_rows, input_columns
        )

    return torch.ops.fbgemm.batch_index_select_dim0(
        inputs.to("cpu"),
        indices.to("cpu"),
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1,
    ).to(inputs.device)


# Register these functions to dispatcher with key 'TPU'
lib.impl("permute_2D_sparse_data", permute_2D_sparse_data, "TPU")
lib.impl("permute_1D_sparse_data", permute_1D_sparse_data, "TPU")
lib.impl("block_bucketize_sparse_features", block_bucketize_sparse_features, "TPU")
lib.impl("invert_permute", invert_permute, "TPU")
lib.impl(
    "permute_pooled_embs_auto_grad_split",
    permute_pooled_embs_auto_grad_split,
    "TPU",
)
lib.impl("segment_sum_csr", segment_sum_csr_tpu, "TPU")
lib.impl("asynchronous_complete_cumsum", asynchronous_complete_cumsum_tpu, "TPU")
lib.impl("batch_index_select_dim0", batch_index_select_dim0_tpu, "TPU")
lib.impl("batch_index_select_dim0", batch_index_select_dim0_tpu, "AutogradPrivateUse1")
_ = impl
