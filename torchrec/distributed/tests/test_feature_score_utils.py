#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List
from unittest.mock import Mock

import torch
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.feature_score_utils import (
    create_sharding_type_to_feature_score_mapping,
    may_collect_feature_scores,
)
from torchrec.distributed.types import ParameterSharding
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    EmbeddingTableConfig,
    FeatureScoreBasedEvictionPolicy,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _convert_to_table_config(
    config: EmbeddingConfig | EmbeddingBagConfig,
) -> EmbeddingTableConfig:
    """Convert EmbeddingConfig or EmbeddingBagConfig to EmbeddingTableConfig for sharding info"""
    pooling = getattr(config, "pooling", None)
    if pooling is None:
        from torchrec.modules.embedding_configs import PoolingType

        pooling = PoolingType.SUM

    return EmbeddingTableConfig(
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        name=config.name,
        data_type=config.data_type,
        feature_names=config.feature_names,
        pooling=pooling,
        is_weighted=False,
        has_feature_processor=False,
        embedding_names=[config.name],
        weight_init_max=config.weight_init_max,
        weight_init_min=config.weight_init_min,
        use_virtual_table=config.use_virtual_table,
        virtual_table_eviction_policy=config.virtual_table_eviction_policy,
    )


class CreateShardingTypeToFeatureScoreMappingTest(unittest.TestCase):
    def test_no_virtual_tables(self) -> None:
        # Setup: create embedding configs without virtual tables
        embedding_configs = [
            EmbeddingConfig(
                name="table_0",
                embedding_dim=64,
                num_embeddings=100,
                feature_names=["feature_0", "feature_1"],
            ),
        ]
        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: both flags should be False and mapping should be empty
        self.assertFalse(enable_weight_acc)
        self.assertFalse(enable_auto_collection)
        self.assertEqual(mapping, {})

    def test_virtual_table_without_eviction_policy(self) -> None:
        # Setup: create virtual table without eviction policy
        embedding_configs = [
            EmbeddingConfig(
                name="table_0",
                embedding_dim=64,
                num_embeddings=100,
                feature_names=["feature_0"],
                use_virtual_table=True,
            ),
        ]
        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: both flags should be False and mapping should be empty
        self.assertFalse(enable_weight_acc)
        self.assertFalse(enable_auto_collection)
        self.assertEqual(mapping, {})

    def test_virtual_table_with_feature_score_policy_without_auto_collection(
        self,
    ) -> None:
        # Setup: create virtual table with feature score policy but without auto collection
        embedding_configs = [
            EmbeddingConfig(
                name="table_0",
                embedding_dim=64,
                num_embeddings=100,
                feature_names=["feature_0"],
                use_virtual_table=True,
                virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                    feature_score_mapping={"feature_0": 1.0},
                    enable_auto_feature_score_collection=False,
                ),
            ),
        ]
        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: weight accumulation is enabled but auto collection is not
        self.assertTrue(enable_weight_acc)
        self.assertFalse(enable_auto_collection)
        self.assertEqual(mapping, {})

    def test_virtual_table_with_auto_collection_enabled(self) -> None:
        # Setup: create virtual table with auto collection enabled
        mock_embedding_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.5, "feature_1": 2.0},
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: both flags are enabled and mapping contains feature scores
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertIn(ShardingType.TABLE_WISE.value, mapping)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 1.5, "feature_1": 2.0},
        )

    def test_virtual_table_with_default_value(self) -> None:
        # Setup: create virtual table with default value for missing features
        mock_embedding_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.5},
                feature_score_default_value=0.5,
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: mapping contains explicit score for feature_0 and default for feature_1
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 1.5, "feature_1": 0.5},
        )

    def test_data_parallel_sharding_type(self) -> None:
        # Setup: create virtual table with data parallel sharding
        mock_embedding_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.0},
                enable_auto_feature_score_collection=True,
            ),
        )

        embedding_configs = [mock_embedding_config]
        sharding_type_to_sharding_infos = {
            ShardingType.DATA_PARALLEL.value: [],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: data parallel sharding has empty mapping
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertIn(ShardingType.DATA_PARALLEL.value, mapping)
        self.assertEqual(mapping[ShardingType.DATA_PARALLEL.value], {})

    def test_eviction_ttl_mins_positive(self) -> None:
        # Setup: create virtual table with positive eviction_ttl_mins
        mock_embedding_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={},
                eviction_ttl_mins=60,
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: all features get 0.0 score when eviction_ttl_mins is positive
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 0.0, "feature_1": 0.0},
        )


class MayCollectFeatureScoresTest(unittest.TestCase):
    def test_auto_collection_disabled(self) -> None:
        # Setup: create input features without auto collection enabled
        input_features = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([3]),
        )
        input_feature_splits = [input_features]
        enabled_feature_score_auto_collection = False
        sharding_type_feature_score_mapping: Dict[str, Dict[str, float]] = {}

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: input should be returned unchanged
        self.assertEqual(result, input_feature_splits)
        self.assertIsNone(result[0].weights_or_none())

    def test_auto_collection_with_empty_mapping(self) -> None:
        # Setup: create input features with empty mapping
        input_features = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([3]),
        )
        input_feature_splits = [input_features]
        enabled_feature_score_auto_collection = True
        sharding_type_feature_score_mapping = {"table_wise": {}}

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: input should be returned without weights added
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].weights_or_none())

    def test_auto_collection_with_feature_scores(self) -> None:
        # Setup: create input features with feature score mapping
        input_features = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([2, 3]),
        )
        input_feature_splits = [input_features]
        enabled_feature_score_auto_collection = True
        sharding_type_feature_score_mapping = {
            "table_wise": {"feature_0": 1.5, "feature_1": 2.0}
        }

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: weights should be attached with correct scores
        self.assertEqual(len(result), 1)
        weights = result[0].weights_or_none()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape[0], 5)
        self.assertTrue(torch.allclose(weights[:2], torch.tensor([1.5, 1.5])))
        self.assertTrue(torch.allclose(weights[2:], torch.tensor([2.0, 2.0, 2.0])))

    def test_auto_collection_with_missing_feature_in_mapping(self) -> None:
        # Setup: create input features with one feature not in mapping
        input_features = KeyedJaggedTensor(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 2]),
        )
        input_feature_splits = [input_features]
        enabled_feature_score_auto_collection = True
        sharding_type_feature_score_mapping = {"table_wise": {"feature_0": 1.5}}

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: missing feature should get 0.0 score
        self.assertEqual(len(result), 1)
        weights = result[0].weights_or_none()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.shape[0], 3)
        self.assertEqual(weights[0].item(), 1.5)
        self.assertEqual(weights[1].item(), 0.0)
        self.assertEqual(weights[2].item(), 0.0)

    def test_auto_collection_with_multiple_feature_splits(self) -> None:
        # Setup: create multiple input feature splits
        input_features_1 = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.tensor([1, 2]),
            lengths=torch.tensor([2]),
        )
        input_features_2 = KeyedJaggedTensor(
            keys=["feature_1"],
            values=torch.tensor([3, 4, 5]),
            lengths=torch.tensor([3]),
        )
        input_feature_splits = [input_features_1, input_features_2]
        enabled_feature_score_auto_collection = True
        sharding_type_feature_score_mapping = {
            "sharding_1": {"feature_0": 1.0},
            "sharding_2": {"feature_1": 2.0},
        }

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: each split should have appropriate weights
        self.assertEqual(len(result), 2)
        weights_1 = result[0].weights_or_none()
        weights_2 = result[1].weights_or_none()
        self.assertIsNotNone(weights_1)
        self.assertIsNotNone(weights_2)
        self.assertTrue(torch.allclose(weights_1, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(weights_2, torch.tensor([2.0, 2.0, 2.0])))

    def test_auto_collection_preserves_device(self) -> None:
        # Setup: create input features on GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_features = KeyedJaggedTensor(
            keys=["feature_0"],
            values=torch.tensor([1, 2, 3], device=device),
            lengths=torch.tensor([3], device=device),
        )
        input_feature_splits = [input_features]
        enabled_feature_score_auto_collection = True
        sharding_type_feature_score_mapping = {"table_wise": {"feature_0": 1.5}}

        # Execute: run may_collect_feature_scores
        result = may_collect_feature_scores(
            input_feature_splits,
            enabled_feature_score_auto_collection,
            sharding_type_feature_score_mapping,
        )

        # Assert: weights should be on the same device as input
        self.assertEqual(len(result), 1)
        weights = result[0].weights_or_none()
        self.assertIsNotNone(weights)
        self.assertEqual(weights.device, device)


class EmbeddingBagConfigSupportTest(unittest.TestCase):
    def test_embedding_bag_config_with_auto_collection_enabled(self) -> None:
        # Setup: create EmbeddingBagConfig with auto collection enabled
        mock_embedding_bag_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.5, "feature_1": 2.0},
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_bag_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_bag_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: both flags are enabled and mapping contains feature scores
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertIn(ShardingType.TABLE_WISE.value, mapping)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 1.5, "feature_1": 2.0},
        )

    def test_embedding_bag_config_with_default_value(self) -> None:
        # Setup: create EmbeddingBagConfig with default value for missing features
        mock_embedding_bag_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1", "feature_2"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.5, "feature_1": 2.0},
                feature_score_default_value=0.5,
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_bag_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_bag_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: mapping contains explicit scores and default for feature_2
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 1.5, "feature_1": 2.0, "feature_2": 0.5},
        )

    def test_mixed_embedding_config_and_bag_config(self) -> None:
        # Setup: create both EmbeddingConfig and EmbeddingBagConfig with auto collection
        embedding_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_0": 1.0},
                enable_auto_feature_score_collection=True,
            ),
        )

        embedding_bag_config = EmbeddingBagConfig(
            name="table_1",
            embedding_dim=32,
            num_embeddings=50,
            feature_names=["feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={"feature_1": 2.0},
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param_0 = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_1 = torch.nn.Parameter(torch.randn(50, 32))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info_0 = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(embedding_config),
            param_sharding=mock_param_sharding,
            param=mock_param_0,
        )

        sharding_info_1 = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(embedding_bag_config),
            param_sharding=mock_param_sharding,
            param=mock_param_1,
        )

        embedding_configs = [embedding_config, embedding_bag_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info_0, sharding_info_1],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: mapping contains scores from both config types
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertIn(ShardingType.TABLE_WISE.value, mapping)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 1.0, "feature_1": 2.0},
        )

    def test_embedding_bag_config_without_virtual_table(self) -> None:
        # Setup: create EmbeddingBagConfig without virtual table
        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="table_0",
                embedding_dim=64,
                num_embeddings=100,
                feature_names=["feature_0"],
            ),
        ]
        sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_bag_configs, sharding_type_to_sharding_infos
        )

        # Assert: both flags should be False and mapping should be empty
        self.assertFalse(enable_weight_acc)
        self.assertFalse(enable_auto_collection)
        self.assertEqual(mapping, {})

    def test_embedding_bag_config_with_eviction_ttl_mins(self) -> None:
        # Setup: create EmbeddingBagConfig with positive eviction_ttl_mins
        mock_embedding_bag_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=64,
            num_embeddings=100,
            feature_names=["feature_0", "feature_1"],
            use_virtual_table=True,
            virtual_table_eviction_policy=FeatureScoreBasedEvictionPolicy(
                feature_score_mapping={},
                eviction_ttl_mins=60,
                enable_auto_feature_score_collection=True,
            ),
        )

        mock_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_param_sharding = Mock(spec=ParameterSharding)

        sharding_info = EmbeddingShardingInfo(
            embedding_config=_convert_to_table_config(mock_embedding_bag_config),
            param_sharding=mock_param_sharding,
            param=mock_param,
        )

        embedding_configs = [mock_embedding_bag_config]
        sharding_type_to_sharding_infos = {
            ShardingType.TABLE_WISE.value: [sharding_info],
        }

        # Execute: run create_sharding_type_to_feature_score_mapping
        (
            enable_weight_acc,
            enable_auto_collection,
            mapping,
        ) = create_sharding_type_to_feature_score_mapping(
            embedding_configs, sharding_type_to_sharding_infos
        )

        # Assert: all features get 0.0 score when eviction_ttl_mins is positive
        self.assertTrue(enable_weight_acc)
        self.assertTrue(enable_auto_collection)
        self.assertEqual(
            mapping[ShardingType.TABLE_WISE.value],
            {"feature_0": 0.0, "feature_1": 0.0},
        )
