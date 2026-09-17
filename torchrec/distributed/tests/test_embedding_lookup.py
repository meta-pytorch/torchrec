#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import OrderedDict
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
from torchrec.distributed.embedding_lookup import (
    _load_state_dict,
    BackendType,
    BatchedFusedEmbeddingBag,
    GroupedEmbeddingsLookup,
    GroupedPooledEmbeddingsLookup,
    TritonBatchedFusedEmbeddingBag,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.types import ShardingEnv2D, ShardingStrategy, ShardingType
from torchrec.modules.embedding_configs import DataType, PoolingType


class _FakeDTensor:
    def __init__(self, local_tensor: LocalShardsWrapper) -> None:
        self._local_tensor = local_tensor

    def to_local(self) -> LocalShardsWrapper:
        return self._local_tensor


def _make_config(
    num_tables: int = 1,
    features_per_table: int = 2,
    local_cols: int = 16,
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.DENSE,
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
                compute_kernel=compute_kernel,
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
        compute_kernel=compute_kernel,
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


class VbeTritonMergeTest(unittest.TestCase):
    def test_multi_group_triton_vbe_uses_concat_merge(self) -> None:
        lookup = MagicMock(spec=GroupedPooledEmbeddingsLookup)
        lookup._dummy_embs_tensor = MagicMock()
        lookup._feature_splits = [1, 1]
        lookup._world_size = 1
        lookup._emb_modules = [
            MagicMock(spec=TritonBatchedFusedEmbeddingBag),
            MagicMock(spec=BatchedFusedEmbeddingBag),
        ]
        lookup._forward.return_value = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        lookup._vbe_splits.return_value = [[2], [1]]
        lookup._merge_variable_batch_embeddings.side_effect = lambda embeddings, splits: GroupedPooledEmbeddingsLookup._merge_variable_batch_embeddings(
            lookup, embeddings, splits
        )
        sparse_features = MagicMock()
        sparse_features.variable_stride_per_key.return_value = True
        sparse_features.split.return_value = [MagicMock(), MagicMock()]

        result = GroupedPooledEmbeddingsLookup.forward(lookup, sparse_features)

        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))


def _make_virtual_table_config(
    compute_kernel: EmbeddingComputeKernel,
    enable_embedding_update: bool = True,
    fused_params: dict | None = None,
) -> GroupedEmbeddingConfig:
    table = ShardedEmbeddingTable(
        name="table_0",
        data_type=DataType.FP32,
        pooling=PoolingType.SUM,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=compute_kernel,
        embedding_dim=16,
        local_cols=16,
        num_embeddings=100,
        feature_names=["feature_0"],
        use_virtual_table=True,
    )
    return GroupedEmbeddingConfig(
        data_type=DataType.FP32,
        pooling=PoolingType.SUM,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=compute_kernel,
        embedding_tables=[table],
        fused_params=fused_params,
        enable_embedding_update=enable_embedding_update,
    )


class DramSsdVirtualTableKernelTest(unittest.TestCase):
    @patch("torchrec.distributed.embedding_lookup.ZeroCollisionEmbeddingCache")
    def test_dram_ssd_creates_embedding_cache(self, mock_cache: MagicMock) -> None:
        config = _make_virtual_table_config(
            EmbeddingComputeKernel.DRAM_SSD_VIRTUAL_TABLE,
            enable_embedding_update=True,
        )
        lookup = MagicMock(spec=GroupedEmbeddingsLookup)
        result = GroupedEmbeddingsLookup._create_embedding_kernel(
            lookup, config, None, None, None
        )
        mock_cache.assert_called_once()
        self.assertEqual(
            mock_cache.call_args.kwargs["backend_type"], BackendType.DRAM_SSD
        )
        self.assertIs(result, mock_cache.return_value)

    @patch(
        "torchrec.distributed.embedding_lookup.ZeroCollisionEmbeddingEnrichmentCache"
    )
    def test_dram_ssd_creates_enrichment_cache_with_policy(
        self, mock_cache: MagicMock
    ) -> None:
        kvzch_config = MagicMock()
        kvzch_config.enrichment_policy = MagicMock()
        config = _make_virtual_table_config(
            EmbeddingComputeKernel.DRAM_SSD_VIRTUAL_TABLE,
            enable_embedding_update=True,
            fused_params={"kvzch_tbe_config": kvzch_config},
        )
        lookup = MagicMock(spec=GroupedEmbeddingsLookup)
        result = GroupedEmbeddingsLookup._create_embedding_kernel(
            lookup, config, None, None, None
        )
        mock_cache.assert_called_once()
        self.assertEqual(
            mock_cache.call_args.kwargs["backend_type"], BackendType.DRAM_SSD
        )
        self.assertIs(result, mock_cache.return_value)

    def test_dram_ssd_without_embedding_update_raises(self) -> None:
        config = _make_virtual_table_config(
            EmbeddingComputeKernel.DRAM_SSD_VIRTUAL_TABLE,
            enable_embedding_update=False,
        )
        lookup = MagicMock(spec=GroupedEmbeddingsLookup)
        with self.assertRaises(ValueError) as ctx:
            GroupedEmbeddingsLookup._create_embedding_kernel(
                lookup, config, None, None, None
            )
        self.assertIn("enable_embedding_update", str(ctx.exception))

    def test_dram_ssd_pooled_embedding_bag_not_supported(self) -> None:
        config = _make_virtual_table_config(
            EmbeddingComputeKernel.DRAM_SSD_VIRTUAL_TABLE,
            enable_embedding_update=True,
        )
        lookup = MagicMock(spec=GroupedPooledEmbeddingsLookup)
        with self.assertRaises(ValueError) as ctx:
            GroupedPooledEmbeddingsLookup._create_embedding_kernel(
                lookup, config, None, None, None, None
            )
        self.assertIn("EmbeddingBagCollection", str(ctx.exception))


class FragmentedDTensorStateLoadTest(unittest.TestCase):
    def test_load_state_dict_copies_across_fragment_layouts(self) -> None:
        source_fragments = [
            torch.arange(6, dtype=torch.float32).view(2, 3),
            torch.arange(6, 15, dtype=torch.float32).view(3, 3),
        ]
        source = LocalShardsWrapper(
            source_fragments,
            [(10, 4), (12, 4)],
            logical_size=torch.Size([5, 3]),
        )
        destination_fragments = [
            torch.zeros((1, 3)),
            torch.zeros((3, 3)),
            torch.zeros((1, 3)),
        ]
        destination = LocalShardsWrapper(
            destination_fragments,
            [(10, 4), (11, 4), (14, 4)],
            logical_size=torch.Size([5, 3]),
        )
        embedding_module = MagicMock()
        embedding_module.state_dict.return_value = OrderedDict(
            {"table.weight": _FakeDTensor(destination)}
        )

        with patch("torchrec.distributed.embedding_lookup.DTensor", _FakeDTensor):
            missing, unexpected = _load_state_dict(
                cast(Any, [embedding_module]),
                cast(
                    Any,
                    OrderedDict({"table.weight": _FakeDTensor(source)}),
                ),
            )

        self.assertEqual([], missing)
        self.assertEqual([], unexpected)
        torch.testing.assert_close(source_fragments[0][:1], destination_fragments[0])
        torch.testing.assert_close(
            torch.cat([source_fragments[0][1:], source_fragments[1][:2]]),
            destination_fragments[1],
        )
        torch.testing.assert_close(source_fragments[1][2:], destination_fragments[2])


class ChunkedShardedTritonKernelRoutingTest(unittest.TestCase):
    def test_fully_sharded_2d_uses_chunked_sharded_triton(self) -> None:
        config = _make_config(compute_kernel=EmbeddingComputeKernel.FUSED_TRITON)
        lookup = MagicMock(spec=GroupedPooledEmbeddingsLookup)
        env = MagicMock(spec=ShardingEnv2D)
        env.sharding_strategy = ShardingStrategy.FULLY_SHARDED

        with patch(
            "torchrec.distributed.embedding_lookup.ChunkedShardedTritonBatchedFusedEmbeddingBag"
        ) as chunked_kernel, patch(
            "torchrec.distributed.embedding_lookup.TritonBatchedFusedEmbeddingBag"
        ) as unsharded_kernel:
            result = GroupedPooledEmbeddingsLookup._create_embedding_kernel(
                lookup,
                config,
                None,
                None,
                ShardingType.ROW_WISE,
                env,
            )

        chunked_kernel.assert_called_once_with(
            config=config,
            pg=None,
            device=None,
            sharding_type=ShardingType.ROW_WISE,
            env=env,
        )
        unsharded_kernel.assert_not_called()
        self.assertIs(result, chunked_kernel.return_value)

    def test_non_fully_sharded_2d_uses_regular_triton(self) -> None:
        config = _make_config(compute_kernel=EmbeddingComputeKernel.FUSED_TRITON)
        lookup = MagicMock(spec=GroupedPooledEmbeddingsLookup)
        env = MagicMock(spec=ShardingEnv2D)
        env.sharding_strategy = ShardingStrategy.DEFAULT

        with patch(
            "torchrec.distributed.embedding_lookup.ChunkedShardedTritonBatchedFusedEmbeddingBag"
        ) as chunked_kernel, patch(
            "torchrec.distributed.embedding_lookup.TritonBatchedFusedEmbeddingBag"
        ) as unsharded_kernel:
            result = GroupedPooledEmbeddingsLookup._create_embedding_kernel(
                lookup,
                config,
                None,
                None,
                ShardingType.ROW_WISE,
                env,
            )

        chunked_kernel.assert_not_called()
        unsharded_kernel.assert_called_once_with(
            config=config,
            pg=None,
            device=None,
        )
        self.assertIs(result, unsharded_kernel.return_value)
