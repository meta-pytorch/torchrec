#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import DataType, PoolingType


def _make_config(
    num_tables: int = 1,
    features_per_table: int = 2,
    local_cols: int = 16,
) -> GroupedEmbeddingConfig:
    tables = []
    feat_idx = 0
    for t in range(num_tables):
        feature_names = [f"feature_{feat_idx + j}" for j in range(features_per_table)]
        feat_idx += features_per_table
        tables.append(
            ShardedEmbeddingTable(
                name=f"table_{t}",
                data_type=DataType.FP32,
                pooling=PoolingType.SUM,
                is_weighted=False,
                has_feature_processor=False,
                compute_kernel=EmbeddingComputeKernel.DENSE,
                embedding_dim=local_cols,
                local_cols=local_cols,
                num_embeddings=100,
                feature_names=feature_names,
            )
        )
    return GroupedEmbeddingConfig(
        data_type=DataType.FP32,
        pooling=PoolingType.SUM,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=EmbeddingComputeKernel.DENSE,
        embedding_tables=tables,
    )


def _make_features_mock(
    stride_per_key_per_rank: list[list[int]],
    keys: list[str],
) -> MagicMock:
    mock = MagicMock()
    mock.stride_per_key_per_rank.return_value = stride_per_key_per_rank
    mock.keys.return_value = keys
    return mock


def _make_lookup_mock(
    grouped_configs: list[GroupedEmbeddingConfig],
    world_size: int,
) -> MagicMock:
    mock = MagicMock(spec=GroupedPooledEmbeddingsLookup)
    mock.grouped_configs = grouped_configs
    mock._world_size = world_size
    return mock


class VbeSplitsTest(unittest.TestCase):
    def test_vbe_splits_normal_cases(self) -> None:
        with self.subTest("ranks_match_world_size"):
            config = _make_config(num_tables=1, features_per_table=2, local_cols=16)
            features = _make_features_mock(
                stride_per_key_per_rank=[[2, 3], [1, 4]],
                keys=["feature_0", "feature_1"],
            )
            lookup = _make_lookup_mock([config], world_size=2)
            result = GroupedPooledEmbeddingsLookup._vbe_splits(lookup, [features])
            self.assertEqual(result[0], [32, 16, 48, 64])

        with self.subTest("world_size_one"):
            config = _make_config(num_tables=1, features_per_table=2, local_cols=16)
            features = _make_features_mock(
                stride_per_key_per_rank=[[4], [6]],
                keys=["feature_0", "feature_1"],
            )
            lookup = _make_lookup_mock([config], world_size=1)
            result = GroupedPooledEmbeddingsLookup._vbe_splits(lookup, [features])
            self.assertEqual(result[0], [64, 96])

        with self.subTest("multiple_tables_single_group"):
            config = _make_config(num_tables=2, features_per_table=1, local_cols=16)
            features = _make_features_mock(
                stride_per_key_per_rank=[[3, 2], [7, 5]],
                keys=["feature_0", "feature_1"],
            )
            lookup = _make_lookup_mock([config], world_size=2)
            result = GroupedPooledEmbeddingsLookup._vbe_splits(lookup, [features])
            self.assertEqual(result[0], [48, 112, 32, 80])

        with self.subTest("multi_tbe_groups"):
            config_a = _make_config(num_tables=1, features_per_table=2, local_cols=16)
            config_b = _make_config(num_tables=1, features_per_table=1, local_cols=32)
            features_a = _make_features_mock(
                stride_per_key_per_rank=[[2, 3], [1, 4]],
                keys=["feature_0", "feature_1"],
            )
            features_b = _make_features_mock(
                stride_per_key_per_rank=[[5, 2]],
                keys=["feature_2"],
            )
            lookup = _make_lookup_mock([config_a, config_b], world_size=2)
            result = GroupedPooledEmbeddingsLookup._vbe_splits(
                lookup, [features_a, features_b]
            )
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], [32, 16, 48, 64])
            self.assertEqual(result[1], [160, 64])

    def test_vbe_splits_rank_mismatch_error(self) -> None:
        with self.subTest("n_ranks_greater_than_world_size"):
            config = _make_config(num_tables=1, features_per_table=2, local_cols=16)
            features = _make_features_mock(
                stride_per_key_per_rank=[[2, 3, 1], [1, 4, 2]],
                keys=["feature_0", "feature_1"],
            )
            lookup = _make_lookup_mock([config], world_size=2)
            with self.assertRaises(ValueError) as ctx:
                GroupedPooledEmbeddingsLookup._vbe_splits(lookup, [features])
            self.assertIn("3 ranks", str(ctx.exception))
            self.assertIn("world_size is 2", str(ctx.exception))

        with self.subTest("n_ranks_less_than_world_size"):
            config = _make_config(num_tables=1, features_per_table=2, local_cols=16)
            features = _make_features_mock(
                stride_per_key_per_rank=[[2], [1]],
                keys=["feature_0", "feature_1"],
            )
            lookup = _make_lookup_mock([config], world_size=4)
            with self.assertRaises(ValueError) as ctx:
                GroupedPooledEmbeddingsLookup._vbe_splits(lookup, [features])
            self.assertIn("1 ranks", str(ctx.exception))
            self.assertIn("world_size is 4", str(ctx.exception))
