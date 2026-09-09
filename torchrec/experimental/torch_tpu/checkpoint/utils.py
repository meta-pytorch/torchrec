#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for PyTorch TPU Embedding."""

import torch


def mod_shard(
    unsharded_weight: torch.Tensor,
    num_shards: int,
) -> torch.Tensor:
    """MOD shards a sequential global tensor.

    Args:
      unsharded_weight: Tensor of shape [vocab_size, dim]
      num_shards: Number of shards (ranks * SCs per rank)

    Returns:
      MOD-sharded tensor of shape [vocab_size, dim] where the rows are permuted.
    """
    vocab_size = unsharded_weight.size(0)
    if vocab_size % num_shards != 0:
        raise ValueError(
            f"Vocab size {vocab_size} must be divisible by num_shards {num_shards}"
        )

    shard_size = vocab_size // num_shards
    extra_shape = list(unsharded_weight.shape[1:])

    # Reshape: [shard_size, num_shards, *extra_shape]
    t = unsharded_weight.view(shard_size, num_shards, *extra_shape)
    # Transpose: [num_shards, shard_size, *extra_shape]
    t = t.permute(1, 0, *range(2, t.ndim)).contiguous()
    # Flatten: [vocab_size, *extra_shape]
    return t.view(-1, *extra_shape)


def reverse_mod_shard(
    sharded_weight: torch.Tensor,
    vocab_size: int,
    embedding_dim: int,
    num_shards: int,
) -> torch.Tensor:
    """Un-MOD-shards a sharded tensor back to sequential layout.

    Args:
      sharded_weight: Consolidated sharded tensor of shape [padded_vocab_size,
        dim]
      vocab_size: Original (unpadded) vocabulary size.
      embedding_dim: Embedding dimension.
      num_shards: Number of shards (ranks * SCs per rank)

    Returns:
      Sequential unpadded tensor of shape [vocab_size, embedding_dim].
    """
    padded_vocab_size = sharded_weight.size(0)
    if padded_vocab_size % num_shards != 0:
        raise ValueError(
            f"Padded vocab size {padded_vocab_size} must be divisible by num_shards"
            f" {num_shards}"
        )

    shard_size = padded_vocab_size // num_shards
    extra_shape = list(sharded_weight.shape[1:])

    # Reshape: [num_shards, shard_size, *extra_shape]
    t = sharded_weight.view(num_shards, shard_size, *extra_shape)
    # Transpose: [shard_size, num_shards, *extra_shape]
    t = t.permute(1, 0, *range(2, t.ndim)).contiguous()
    # Flatten: [padded_vocab_size, *extra_shape]
    unsharded_weight = t.view(-1, *extra_shape)

    # Slice to remove padding if vocab_size is smaller than padded shape
    return unsharded_weight[:vocab_size, :embedding_dim]
