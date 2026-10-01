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

"""Utility functions for PyTorch TPU Embedding Checkpointing."""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    StackedSparseCoreEmbeddingConfig,
)


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


def serialize_stacked_configs(
    stacked_configs: Union[Sequence[StackedSparseCoreEmbeddingConfig], Dict[str, Any]],
) -> Dict[str, Any]:
    """Serializes stacked embedding configs into a DCP-compatible dictionary."""
    if hasattr(stacked_configs, "_stacked_configs"):
        stacked_configs = stacked_configs._stacked_configs

    if isinstance(stacked_configs, dict):
        return stacked_configs

    res = {}
    for sc in stacked_configs:
        if isinstance(sc, dict):
            stack_name = sc.get("stack_name") or sc.get("stack_table_name")
            res[stack_name] = sc
        elif hasattr(sc, "to_dict"):
            res[sc.stack_name] = sc.to_dict()
    return res


def unstack_and_unshard_global_tensor(
    sharded_weight: torch.Tensor,
    stack_config: Union[StackedSparseCoreEmbeddingConfig, Dict[str, Any]],
    num_shards: int,
) -> Dict[str, torch.Tensor]:
    """Unstacks and un-MOD-shards a full global stacked tensor into individual table weights.

    Args:
      sharded_weight: Consolidated sharded tensor of shape [stack_num_embeddings,
        stack_embedding_dim]
      stack_config: StackedSparseCoreEmbeddingConfig or serialized dict metadata.
      num_shards: Total number of shards (world_size * num_sc_per_device) used
        when saving.

    Returns:
      Dict mapping table name to sequential unpadded tensor [vocab_size,
      embedding_dim].
    """
    if isinstance(stack_config, dict):
        stack_config = StackedSparseCoreEmbeddingConfig.from_dict(stack_config)

    stack_num_embeddings = sharded_weight.size(0)
    stack_embedding_dim = sharded_weight.size(1)

    if stack_num_embeddings % num_shards != 0:
        raise ValueError(
            f"Stack vocab size {stack_num_embeddings} must be divisible by"
            f" num_shards {num_shards}"
        )

    global_3d = sharded_weight.view(num_shards, -1, stack_embedding_dim)
    table_weights = {}

    for table in stack_config.tables:
        table_name = table.name
        vocab_size = table.num_embeddings
        embedding_dim = table.embedding_dim
        padded_vocab_size = stack_config.padded_vocab_sizes[table_name]
        row_offset = stack_config.row_offsets_in_shard[table_name]
        rotation = stack_config.shard_rotations[table_name]

        chunk_size = padded_vocab_size // num_shards
        w_slice = global_3d[:, row_offset : row_offset + chunk_size, :]

        rot = rotation % num_shards
        if rot != 0:
            w_slice = torch.roll(w_slice, shifts=-rot, dims=0)

        unsharded_padded = reverse_mod_shard(
            w_slice.reshape(-1, stack_embedding_dim),
            vocab_size=padded_vocab_size,
            embedding_dim=stack_embedding_dim,
            num_shards=num_shards,
        )
        table_weights[table_name] = unsharded_padded[:vocab_size, :embedding_dim]

    return table_weights


def stack_and_shard_global_tensor(
    table_weights: Dict[str, torch.Tensor],
    stack_config: Union[StackedSparseCoreEmbeddingConfig, Dict[str, Any]],
    num_shards: int,
) -> torch.Tensor:
    """Stacks and MOD-shards individual table weights into a full global stacked tensor.

    Args:
      table_weights: Dict mapping table name to sequential tensor [vocab_size,
        embedding_dim]
      stack_config: StackedSparseCoreEmbeddingConfig or serialized dict metadata.
      num_shards: Total number of target shards (world_size * num_sc_per_device).

    Returns:
      Stacked and MOD-sharded tensor of shape [stack_num_embeddings,
      stack_embedding_dim].
    """
    if isinstance(stack_config, dict):
        stack_config = StackedSparseCoreEmbeddingConfig.from_dict(stack_config)

    stack_num_embeddings = stack_config.stack_num_embeddings
    stack_embedding_dim = stack_config.stack_embedding_dim

    if stack_num_embeddings % num_shards != 0:
        raise ValueError(
            f"Target stack vocab size {stack_num_embeddings} must be divisible by"
            f" target num_shards {num_shards}"
        )

    total_rows_per_shard = stack_num_embeddings // num_shards
    dtype = next(iter(table_weights.values())).dtype
    global_3d = torch.zeros(
        (num_shards, total_rows_per_shard, stack_embedding_dim),
        dtype=dtype,
    )

    for table in stack_config.tables:
        table_name = table.name
        w = table_weights[table_name]
        padded_vocab_size = stack_config.padded_vocab_sizes[table_name]
        row_offset = stack_config.row_offsets_in_shard[table_name]
        rotation = stack_config.shard_rotations[table_name]

        pad_rows = padded_vocab_size - w.size(0)
        pad_cols = stack_embedding_dim - w.size(1)
        if pad_rows > 0 or pad_cols > 0:
            w_padded = F.pad(w, (0, pad_cols, 0, pad_rows), value=0.0)
        else:
            w_padded = w

        sharded = mod_shard(w_padded, num_shards=num_shards)
        chunk_size = padded_vocab_size // num_shards
        t = sharded.view(num_shards, chunk_size, stack_embedding_dim)

        rot = rotation % num_shards
        if rot != 0:
            t = torch.roll(t, shifts=rot, dims=0)

        global_3d[:, row_offset : row_offset + chunk_size, :] = t

    return global_3d.view(-1, stack_embedding_dim)
