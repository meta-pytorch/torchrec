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

"""Custom Embedding Configurations using SparseCore on TPU."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

from torchrec.modules import embedding_configs

PoolingType = embedding_configs.PoolingType
EmbeddingConfig = embedding_configs.EmbeddingConfig
EmbeddingBagConfig = embedding_configs.EmbeddingBagConfig
dataclass = dataclasses.dataclass


@dataclass
class SparseCoreEmbeddingConfig:
    """Wrapper config class that attaches TPU sequence bounds to a standard BaseEmbeddingConfig."""

    config: Union[EmbeddingConfig, EmbeddingBagConfig]
    max_seq_len: Optional[int] = None
    max_ids_per_partition: int = 256
    max_unique_ids_per_partition: int = 256
    suggested_coo_buffer_size_per_device: Optional[int] = None
    stack_table_name: Optional[str] = None
    _stacked_config: Optional["StackedSparseCoreEmbeddingConfig"] = (
        dataclasses.field(default=None, repr=False, compare=False)
    )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    @property
    def num_embeddings(self) -> int:
        return self.config.num_embeddings

    @property
    def feature_names(self) -> List[str]:
        return self.config.feature_names

    @property
    def init_fn(self) -> Any:
        return self.config.init_fn

    @property
    def pooling(self) -> PoolingType:
        if isinstance(self.config, EmbeddingBagConfig):
            return self.config.pooling
        raise AttributeError("Wrapped config is not an EmbeddingBagConfig")

    @property
    def stacked_config(self) -> Optional["StackedSparseCoreEmbeddingConfig"]:
        return self._stacked_config


@dataclass
class StackedSparseCoreEmbeddingConfig:
    """Combined configuration and physical placement map for a stacked TPU table."""

    stack_table_name: str
    tables: List[SparseCoreEmbeddingConfig]
    feature_names: List[str]

    # Stack-level physical dimensions
    stack_num_embeddings: int
    stack_embedding_dim: int
    total_sample_count_multiplier: int

    # Per-table placement metadata
    row_offsets_in_shard: Dict[str, int]
    shard_rotations: Dict[str, int]
    padded_vocab_sizes: Dict[str, int]
    padded_embedding_dims: Dict[str, int]

    _max_ids_per_partition: Optional[int] = None
    _max_unique_ids_per_partition: Optional[int] = None
    _suggested_coo_buffer_size_per_device: Optional[int] = None

    @property
    def max_ids_per_partition(self) -> int:
        if self._max_ids_per_partition is not None:
            return self._max_ids_per_partition
        return max(t.max_ids_per_partition for t in self.tables)

    @max_ids_per_partition.setter
    def max_ids_per_partition(self, value: Optional[int]) -> None:
        self._max_ids_per_partition = value

    @property
    def max_unique_ids_per_partition(self) -> int:
        if self._max_unique_ids_per_partition is not None:
            return self._max_unique_ids_per_partition
        return max(t.max_unique_ids_per_partition for t in self.tables)

    @max_unique_ids_per_partition.setter
    def max_unique_ids_per_partition(self, value: Optional[int]) -> None:
        self._max_unique_ids_per_partition = value

    @property
    def suggested_coo_buffer_size_per_device(self) -> Optional[int]:
        if self._suggested_coo_buffer_size_per_device is not None:
            return self._suggested_coo_buffer_size_per_device
        vals = [
            t.suggested_coo_buffer_size_per_device
            for t in self.tables
            if t.suggested_coo_buffer_size_per_device is not None
        ]
        return max(vals) if vals else None

    @suggested_coo_buffer_size_per_device.setter
    def suggested_coo_buffer_size_per_device(self, value: Optional[int]) -> None:
        self._suggested_coo_buffer_size_per_device = value

    @property
    def name(self) -> str:
        return self.stack_table_name

    @property
    def stack_name(self) -> str:
        return self.stack_table_name

    @property
    def embedding_dim(self) -> int:
        return self.stack_embedding_dim

    @property
    def num_embeddings(self) -> int:
        return self.stack_num_embeddings

    @property
    def pooling(self) -> PoolingType:
        if isinstance(self.tables[0].config, EmbeddingBagConfig):
            return self.tables[0].pooling
        raise AttributeError("Wrapped config is not an EmbeddingBagConfig")

    @property
    def is_unpooled(self) -> bool:
        return isinstance(self.tables[0].config, EmbeddingConfig)

    @property
    def max_seq_len(self) -> Optional[int]:
        return self.tables[0].max_seq_len

    @property
    def total_features(self) -> int:
        """Returns the total number of features across all tables in this stack."""
        return len(self.feature_names)

    def get_col_offset(self, table_name: str, num_sparsecores: int) -> int:
        """Returns the starting vocabulary column offset for a table in the stacked sparse matrix representation.

        In SparseCore COO minibatching, batch samples are indexed along rows while
        vocabulary IDs are indexed along columns. Across all SparseCores, this
        offset
        is row_offsets_in_shard[table_name] * num_sparsecores.
        """
        return self.row_offsets_in_shard[table_name] * num_sparsecores

    def get_vocab_offset(self, table_name: str, num_sparsecores: int) -> int:
        """Alias for get_col_offset returning the starting global vocabulary index offset."""
        return self.get_col_offset(table_name, num_sparsecores)

    def get_shard_rotation(self, table_name: str) -> int:
        """Returns the MOD-sharding rotation factor for a table."""
        return self.shard_rotations[table_name]

    def get_total_sample_count(self, batch_size: int) -> int:
        """Returns the total sample count for this stack given process-local batch_size."""
        return self.total_sample_count_multiplier * batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this stacked configuration into a DCP-compatible dictionary."""
        tables_info = []
        for t in self.tables:
            t_num_embeddings = (
                t.config.num_embeddings
                if hasattr(t, "config") and hasattr(t.config, "num_embeddings")
                else getattr(t, "num_embeddings", 0)
            )
            t_embedding_dim = (
                t.embedding_dim
                if hasattr(t, "embedding_dim")
                else getattr(t.config, "embedding_dim", 0)
            )
            is_bag = isinstance(getattr(t, "config", None), EmbeddingBagConfig)
            pooling = None
            max_seq_len = None
            if is_bag:
                pooling = (
                    t.config.pooling.name if hasattr(t.config, "pooling") else "SUM"
                )
            else:
                max_seq_len = getattr(t, "max_seq_len", None)

            tables_info.append({
                "name": t.name,
                "num_embeddings": t_num_embeddings,
                "embedding_dim": t_embedding_dim,
                "is_bag": is_bag,
                "pooling": pooling,
                "max_seq_len": max_seq_len,
                "feature_names": getattr(t, "feature_names", []),
                "padded_vocab_size": self.padded_vocab_sizes[t.name],
                "padded_embedding_dim": self.padded_embedding_dims[t.name],
                "row_offset_in_shard": self.row_offsets_in_shard[t.name],
                "shard_rotation": self.shard_rotations[t.name],
            })
        return {
            "stack_name": self.stack_name,
            "stack_table_name": self.stack_name,
            "feature_names": list(self.feature_names),
            "stack_num_embeddings": self.stack_num_embeddings,
            "stack_embedding_dim": self.stack_embedding_dim,
            "total_sample_count_multiplier": self.total_sample_count_multiplier,
            "row_offsets_in_shard": dict(self.row_offsets_in_shard),
            "shard_rotations": dict(self.shard_rotations),
            "padded_vocab_sizes": dict(self.padded_vocab_sizes),
            "padded_embedding_dims": dict(self.padded_embedding_dims),
            "tables": tables_info,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any]
    ) -> "StackedSparseCoreEmbeddingConfig":
        """Deserializes a dictionary into a StackedSparseCoreEmbeddingConfig."""
        tables = []
        for t_data in data["tables"]:
            is_bag = t_data.get("is_bag", False) or t_data.get("pooling") is not None
            pooling_str = t_data.get("pooling")
            feature_names = t_data.get("feature_names", [])
            if is_bag:
                pooling = (
                    PoolingType[pooling_str]
                    if pooling_str and pooling_str in PoolingType.__members__
                    else PoolingType.SUM
                )
                cfg = EmbeddingBagConfig(
                    name=t_data["name"],
                    embedding_dim=t_data["embedding_dim"],
                    num_embeddings=t_data["num_embeddings"],
                    pooling=pooling,
                    feature_names=feature_names,
                )
                tables.append(SparseCoreEmbeddingConfig(config=cfg))
            else:
                cfg = EmbeddingConfig(
                    name=t_data["name"],
                    embedding_dim=t_data["embedding_dim"],
                    num_embeddings=t_data["num_embeddings"],
                    feature_names=feature_names,
                )
                tables.append(
                    SparseCoreEmbeddingConfig(
                        config=cfg, max_seq_len=t_data.get("max_seq_len")
                    )
                )

        feature_names = data.get("feature_names")
        if not feature_names:
            feature_names = []
            for t in tables:
                feature_names.extend(getattr(t, "feature_names", []))

        res = cls(
            stack_table_name=data.get("stack_table_name") or data["stack_name"],
            tables=tables,
            feature_names=feature_names,
            stack_num_embeddings=data["stack_num_embeddings"],
            stack_embedding_dim=data["stack_embedding_dim"],
            total_sample_count_multiplier=data.get(
                "total_sample_count_multiplier", 0
            ),
            row_offsets_in_shard=data["row_offsets_in_shard"],
            shard_rotations=data["shard_rotations"],
            padded_vocab_sizes=data["padded_vocab_sizes"],
            padded_embedding_dims=data["padded_embedding_dims"],
        )
        for t in res.tables:
            t._stacked_config = res
        return res
