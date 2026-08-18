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

lib = torch.library.Library("fbgemm", "IMPL")


def permute_2D_sparse_data(
    permute, lengths, values, weights=None, permuted_lengths_sum=None
):
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


# Register these functions to dispatcher with key 'TPU'
lib.impl("permute_2D_sparse_data", permute_2D_sparse_data, "TPU")
lib.impl("permute_1D_sparse_data", permute_1D_sparse_data, "TPU")
lib.impl("block_bucketize_sparse_features", block_bucketize_sparse_features, "TPU")
lib.impl("invert_permute", invert_permute, "TPU")
