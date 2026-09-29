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

"""Table stacking methods for TPU SparseCore Embedding."""

import collections
import hashlib
import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torchrec.experimental.torch_tpu.modules.embedding_configs import (
    SparseCoreEmbeddingConfig,
    StackedSparseCoreEmbeddingConfig,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _get_stack_name(
    table_names: Sequence[str],
    use_short_names: bool = True,
    max_length: int = 50,
    hash_length: int = 12,
) -> str:
    """Returns a deterministic unique stack name for the given table names."""
    stack_name = "_".join(table_names)
    if not use_short_names:
        return stack_name
    shortened_name = stack_name[:max_length]
    if len(stack_name) > max_length:
        shortened_name += "_"
        shortened_name += hashlib.sha256(stack_name.encode("utf-8")).hexdigest()[
            :hash_length
        ]
    return shortened_name


def _split_groups_by_memory_limit(
    groups: List[List[SparseCoreEmbeddingConfig]],
    table_to_activation_mem_bytes: Dict[str, int],
    activation_mem_bytes_limit: int,
) -> List[List[SparseCoreEmbeddingConfig]]:
    """Splits table groups to respect the activation memory limit."""
    validated_groups = []
    for group in groups:
        split_groups: List[List[SparseCoreEmbeddingConfig]] = []
        for table in group:
            found = False
            for candidate_group in split_groups:
                accumulated_mem = sum(
                    table_to_activation_mem_bytes[t.name] for t in candidate_group
                )
                if (
                    accumulated_mem + table_to_activation_mem_bytes[table.name]
                    <= activation_mem_bytes_limit
                ):
                    candidate_group.append(table)
                    found = True
                    break
            if not found:
                split_groups.append([table])
        validated_groups.extend(split_groups)
    return validated_groups


def _group_tables_for_stacking(
    tables: List[SparseCoreEmbeddingConfig],
    global_device_count: int,
    num_sc_per_device: int,
    activation_mem_bytes_limit: int = 6 * 1024 * 1024,
    batch_size: int = 16,
) -> List[List[SparseCoreEmbeddingConfig]]:
    """Groups tables by compatibility (dim, pooling, config type) and memory limit."""
    table_groups = collections.defaultdict(list)

    for table in sorted(tables, key=lambda t: t.name):
        padded_dim = _round_up_to_multiple(table.embedding_dim, 8)
        if isinstance(table.config, EmbeddingBagConfig):
            key = (
                "EBC",
                padded_dim,
                table.pooling,
            )
        else:
            key = (
                "EC",
                padded_dim,
                table.max_seq_len,
            )
        table_groups[key].append(table)

    table_to_mem_bytes = {}
    for table in tables:
        padded_dim = _round_up_to_multiple(table.embedding_dim, 8)
        if isinstance(table.config, EmbeddingBagConfig):
            sample_count = (len(table.feature_names) * batch_size) // max(
                1, num_sc_per_device
            )
        else:
            if table.max_seq_len is None:
                raise ValueError(
                    f"Table '{table.name}' is an unpooled EmbeddingConfig but has"
                    " max_seq_len=None. `max_seq_len` must be specified for all"
                    " unpooled tables to compute activation memory requirements for"
                    " table stacking."
                )
            sample_count = (
                len(table.feature_names) * table.max_seq_len * batch_size
            ) // max(1, num_sc_per_device)
        table_to_mem_bytes[table.name] = padded_dim * sample_count * 4

    return _split_groups_by_memory_limit(
        list(table_groups.values()),
        table_to_mem_bytes,
        activation_mem_bytes_limit,
    )


def stack_tables(
    tables: List[SparseCoreEmbeddingConfig],
    table_names: Sequence[str],
    global_device_count: int = 1,
    num_sc_per_device: int = 2,
    batch_size: int = 1,
    stack_table_name: Optional[str] = None,
    rotation: Optional[int] = None,
    fail_on_excess_padding: bool = False,
) -> StackedSparseCoreEmbeddingConfig:
    """Creates a StackedSparseCoreEmbeddingConfig for the specified group of tables.

    Validates that tables have compatible config types (EBC vs EC) and pooling.
    Allows stacking tables with different embedding dimensions by padding smaller
    dimensions up to max_padded_dim in the group (unless
    fail_on_excess_padding=True).

    Assigns the resulting StackedSparseCoreEmbeddingConfig to
    table._stacked_config
    in-place for each table in table_names.
    """
    group = [t for t in tables if t.name in table_names]
    if len(group) != len(table_names):
        found = {t.name for t in group}
        missing = set(table_names) - found
        raise ValueError(f"Table names not found in tables list: {missing}")

    first_config = group[0].config
    first_is_ebc = isinstance(first_config, EmbeddingBagConfig)
    first_pooling = getattr(first_config, "pooling", None)

    for table in group:
        is_ebc = isinstance(table.config, EmbeddingBagConfig)
        if is_ebc != first_is_ebc:
            raise ValueError(
                "Cannot stack EmbeddingBagConfig and EmbeddingConfig tables together"
            )
        if is_ebc and getattr(table.config, "pooling", None) != first_pooling:
            raise ValueError(
                "Cannot stack EmbeddingBagConfig tables with different pooling"
                f" types: {first_pooling} vs {getattr(table.config, 'pooling', None)}"
            )

    if not stack_table_name:
        stack_table_name = _get_stack_name(
            [t.name for t in group], use_short_names=True
        )

    num_sparsecores = global_device_count * num_sc_per_device
    row_offsets_in_shard: Dict[str, int] = {}
    shard_rotations: Dict[str, int] = {}
    padded_vocab_sizes: Dict[str, int] = {}
    padded_embedding_dims: Dict[str, int] = {}
    feature_names: List[str] = []

    current_row_offset_in_shard = 0
    current_shard_rotation = 0
    total_padded_vocab_size = 0
    max_padded_dim = 0
    total_sample_count_multiplier = 0

    for table in group:
        padded_vocab = _round_up_to_multiple(table.num_embeddings, 8 * num_sparsecores)
        padded_dim = _round_up_to_multiple(table.embedding_dim, 8)
        padded_vocab_sizes[table.name] = padded_vocab
        padded_embedding_dims[table.name] = padded_dim
        if padded_dim > max_padded_dim:
            max_padded_dim = padded_dim

    if not all(dim == max_padded_dim for dim in padded_embedding_dims.values()):
        excess = sum(
            (max_padded_dim - padded_embedding_dims[t.name])
            * padded_vocab_sizes[t.name]
            for t in group
        )
        msg = (
            f"Excess padding detected for stacked table {stack_table_name}:"
            f" {excess} values."
        )
        if fail_on_excess_padding:
            raise ValueError(msg)
        else:
            logging.warning("WARNING during stack_tables: %s", msg)
        for name in padded_embedding_dims:
            padded_embedding_dims[name] = max_padded_dim

    rot_step = rotation if rotation is not None else num_sc_per_device
    for table in group:
        padded_vocab = padded_vocab_sizes[table.name]
        row_offsets_in_shard[table.name] = current_row_offset_in_shard
        shard_rotations[table.name] = current_shard_rotation

        for feat in table.feature_names:
            feature_names.append(feat)

        if isinstance(table.config, EmbeddingBagConfig):
            total_sample_count_multiplier += len(table.feature_names)
        else:
            if table.max_seq_len is None:
                raise ValueError(
                    f"Table '{table.name}' is an unpooled EmbeddingConfig but has"
                    " max_seq_len=None. `max_seq_len` must be specified for all"
                    " unpooled tables to stack tables."
                )
            total_sample_count_multiplier += (
                len(table.feature_names) * table.max_seq_len
            )

        num_rows_in_shard = padded_vocab // num_sparsecores
        current_row_offset_in_shard += num_rows_in_shard
        current_shard_rotation = (current_shard_rotation + rot_step) % num_sparsecores
        total_padded_vocab_size += padded_vocab

    stack_config = StackedSparseCoreEmbeddingConfig(
        stack_table_name=stack_table_name,
        tables=group,
        feature_names=feature_names,
        stack_num_embeddings=total_padded_vocab_size,
        stack_embedding_dim=max_padded_dim,
        total_sample_count_multiplier=total_sample_count_multiplier,
        row_offsets_in_shard=row_offsets_in_shard,
        shard_rotations=shard_rotations,
        padded_vocab_sizes=padded_vocab_sizes,
        padded_embedding_dims=padded_embedding_dims,
    )

    for table in group:
        table._stacked_config = stack_config

    return stack_config


def auto_stack_tables(
    tables: List[SparseCoreEmbeddingConfig],
    global_device_count: int = 1,
    num_sc_per_device: int = 2,
    batch_size: int = 1,
    activation_mem_bytes_limit: int = 6 * 1024 * 1024,
    use_short_names: bool = True,
) -> List[StackedSparseCoreEmbeddingConfig]:
    """Automatically groups and configures tables into stacked embedding configs."""
    if not tables:
        return []

    unstacked_tables = [t for t in tables if t.stacked_config is None]
    groups = _group_tables_for_stacking(
        unstacked_tables,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        activation_mem_bytes_limit=activation_mem_bytes_limit,
        batch_size=batch_size,
    )

    stacked_configs = []
    for group in groups:
        table_names = [t.name for t in group]
        stack_table_name = _get_stack_name(table_names, use_short_names=use_short_names)
        stack_config = stack_tables(
            tables=tables,
            table_names=table_names,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            batch_size=batch_size,
            stack_table_name=stack_table_name,
        )
        stacked_configs.append(stack_config)

    return stacked_configs


def prepare_tables_for_stacking(
    tables: List[SparseCoreEmbeddingConfig],
    global_device_count: int = 1,
    num_sc_per_device: int = 2,
) -> List[StackedSparseCoreEmbeddingConfig]:
    """Prepares tables for training by ensuring all tables have StackedSparseCoreEmbeddingConfig populated.

    For any tables not already grouped by auto_stack_tables, groups by custom
    stack_table_name if specified, or creates a 1-table stack and populates
    authoritative padded vocabulary and embedding dimensions.
    """
    groups: Dict[str, List[SparseCoreEmbeddingConfig]] = collections.defaultdict(list)
    for t in tables:
        if t.stacked_config is None:
            group_name = t.stack_table_name or t.name
            groups[group_name].append(t)

    for group_name, group_tables in groups.items():
        stack_tables(
            tables=tables,
            table_names=[t.name for t in group_tables],
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            stack_table_name=group_name,
        )

    seen_stacks = set()
    stacked_configs = []
    for t in tables:
        if t.stacked_config is None:
            raise RuntimeError(
                f"Table '{t.name}' was not associated with a"
                " StackedSparseCoreEmbeddingConfig."
            )
        if t.stacked_config.stack_table_name not in seen_stacks:
            stacked_configs.append(t.stacked_config)
            seen_stacks.add(t.stacked_config.stack_table_name)

    return stacked_configs


def stack_and_shard_tables(
    table_weights: Dict[str, torch.Tensor],
    stacked_configs: List[StackedSparseCoreEmbeddingConfig],
    global_device_count: int = 1,
    num_sc_per_device: int = 2,
) -> Dict[str, torch.Tensor]:
    """Stacks and shards individual table weights into physical stacked table parameters."""
    stacked_weights: Dict[str, torch.Tensor] = {}

    for stack_config in stacked_configs:
        sharded_slices = []
        for table in stack_config.tables:
            w = table_weights[table.name]
            padded_vocab_size = stack_config.padded_vocab_sizes[table.name]
            rotation = stack_config.shard_rotations[table.name]

            local_vocab = w.shape[0]
            local_padded_vocab = padded_vocab_size // global_device_count
            local_dim = w.shape[1]

            pad_rows = local_padded_vocab - local_vocab
            pad_cols = stack_config.stack_embedding_dim - local_dim
            if pad_rows > 0 or pad_cols > 0:
                w_padded = F.pad(w, (0, pad_cols, 0, pad_rows), value=0.0)
            else:
                w_padded = w

            chunk_size = local_padded_vocab // num_sc_per_device
            w_3d = w_padded.view(
                num_sc_per_device, chunk_size, stack_config.stack_embedding_dim
            )

            local_rot = rotation % num_sc_per_device
            if local_rot != 0:
                w_3d = torch.roll(w_3d, shifts=local_rot, dims=0)

            sharded_slices.append(w_3d)

        stacked_3d = torch.cat(sharded_slices, dim=1)
        stacked_weights[stack_config.stack_name] = stacked_3d.view(
            -1, stack_config.stack_embedding_dim
        )

    return stacked_weights


def unshard_and_unstack_tables(
    stacked_weights: Dict[str, torch.Tensor],
    stacked_configs: List[StackedSparseCoreEmbeddingConfig],
    global_device_count: int = 1,
    num_sc_per_device: int = 2,
) -> Dict[str, torch.Tensor]:
    """Unshards and unstacks physical stacked table parameters back into individual table weights."""
    table_weights: Dict[str, torch.Tensor] = {}

    for stack_config in stacked_configs:
        stacked_w = stacked_weights[stack_config.stack_name]
        stacked_3d = stacked_w.view(
            num_sc_per_device, -1, stack_config.stack_embedding_dim
        )

        for table in stack_config.tables:
            row_offset_in_shard = stack_config.row_offsets_in_shard[table.name]
            rotation = stack_config.shard_rotations[table.name]
            padded_vocab_size = stack_config.padded_vocab_sizes[table.name]
            chunk_size = (padded_vocab_size // global_device_count) // num_sc_per_device

            w_slice = stacked_3d[
                :, row_offset_in_shard : row_offset_in_shard + chunk_size, :
            ]

            local_rot = rotation % num_sc_per_device
            if local_rot != 0:
                w_slice = torch.roll(w_slice, shifts=-local_rot, dims=0)

            w_padded = w_slice.reshape(-1, stack_config.stack_embedding_dim)
            local_vocab = table.num_embeddings // global_device_count
            table_weights[table.name] = w_padded[:local_vocab, : table.embedding_dim]

    return table_weights
