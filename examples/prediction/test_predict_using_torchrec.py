#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import sys
import unittest

import torch

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import using the full module path
from torchrec.github.examples.prediction.predict_using_torchrec import (
    create_kjt_from_batch,
    DLRMRatingWrapper,
    RecommendationDataset,
    TorchRecDLRM,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

# TorchRec imports
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestRecommendationDataset(unittest.TestCase):
    """Test cases for the RecommendationDataset class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 50
        self.num_samples = 1000
        self.dataset = RecommendationDataset(
            num_users=self.num_users,
            num_items=self.num_items,
            num_samples=self.num_samples,
        )

    def test_init(self) -> None:
        """Test initialization of RecommendationDataset."""
        self.assertEqual(self.dataset.num_users, self.num_users)
        self.assertEqual(self.dataset.num_items, self.num_items)
        self.assertEqual(self.dataset.num_samples, self.num_samples)

        # Check tensor shapes
        self.assertEqual(self.dataset.user_ids.shape, (self.num_samples,))
        self.assertEqual(self.dataset.item_ids.shape, (self.num_samples,))
        self.assertEqual(self.dataset.ratings.shape, (self.num_samples,))
        self.assertEqual(self.dataset.user_categories.shape, (self.num_samples,))
        self.assertEqual(self.dataset.item_categories.shape, (self.num_samples,))
        self.assertEqual(self.dataset.dense_features.shape, (self.num_samples, 4))

    def test_len(self) -> None:
        """Test __len__ method."""
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem(self) -> None:
        """Test __getitem__ method."""
        item = self.dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn("user_id", item)
        self.assertIn("item_id", item)
        self.assertIn("user_category", item)
        self.assertIn("item_category", item)
        self.assertIn("dense_features", item)
        self.assertIn("rating", item)

        # Check tensor shapes for a single item
        self.assertEqual(item["dense_features"].shape, (4,))
        self.assertEqual(item["user_category"].shape, ())


class TestTorchRecDLRM(unittest.TestCase):
    """Test cases for the TorchRecDLRM class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedding_dim = 32
        self.dense_in_features = 4
        self.dense_arch_layer_sizes = [16, self.embedding_dim]
        self.over_arch_layer_sizes = [64, 32, 1]

        # Create embedding bag configs
        self.num_users = 100
        self.num_items = 50
        self.num_user_categories = 10
        self.num_item_categories = 20

        self.eb_configs = [
            EmbeddingBagConfig(
                name="user_id",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_users,
                feature_names=["user_id"],
            ),
            EmbeddingBagConfig(
                name="item_id",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_items,
                feature_names=["item_id"],
            ),
            EmbeddingBagConfig(
                name="user_category",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_user_categories,
                feature_names=["user_category"],
            ),
            EmbeddingBagConfig(
                name="item_category",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_item_categories,
                feature_names=["item_category"],
            ),
        ]

        # Create EmbeddingBagCollection
        self.device = torch.device("cpu")
        self.ebc = EmbeddingBagCollection(
            tables=self.eb_configs,
            device=self.device,
        )

        # Create model
        self.model = TorchRecDLRM(
            embedding_bag_collection=self.ebc,
            dense_in_features=self.dense_in_features,
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
        )

        # Test data
        self.batch_size = 8
        self.dense_features = torch.rand(self.batch_size, self.dense_in_features)

        # Create batch data
        self.batch = {
            "user_id": torch.randint(0, self.num_users, (self.batch_size,)),
            "item_id": torch.randint(0, self.num_items, (self.batch_size,)),
            "user_category": torch.randint(
                0, self.num_user_categories, (self.batch_size,)
            ),
            "item_category": torch.randint(
                0, self.num_item_categories, (self.batch_size,)
            ),
        }

        # Create KeyedJaggedTensor for sparse features
        self.sparse_features = create_kjt_from_batch(self.batch, self.device)

    def test_init(self) -> None:
        """Test initialization of TorchRecDLRM."""
        self.assertIsInstance(
            self.model.embedding_bag_collection, EmbeddingBagCollection
        )
        self.assertEqual(
            len(self.model.embedding_bag_collection.embedding_bag_configs()), 4
        )

        # Check embedding dimensions
        for config in self.model.embedding_bag_collection.embedding_bag_configs():
            self.assertEqual(config.embedding_dim, self.embedding_dim)

    def test_forward(self) -> None:
        """Test forward pass of TorchRecDLRM."""
        # Run forward pass
        output = self.model(self.dense_features, self.sparse_features)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())


class TestDLRMRatingWrapper(unittest.TestCase):
    """Test cases for the DLRMRatingWrapper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedding_dim = 32
        self.dense_in_features = 4
        self.device = torch.device("cpu")

        # Create embedding bag configs
        self.num_users = 100
        self.num_items = 50
        self.num_user_categories = 10
        self.num_item_categories = 20

        self.eb_configs = [
            EmbeddingBagConfig(
                name="user_id",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_users,
                feature_names=["user_id"],
            ),
            EmbeddingBagConfig(
                name="item_id",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_items,
                feature_names=["item_id"],
            ),
            EmbeddingBagConfig(
                name="user_category",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_user_categories,
                feature_names=["user_category"],
            ),
            EmbeddingBagConfig(
                name="item_category",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_item_categories,
                feature_names=["item_category"],
            ),
        ]

        # Create EmbeddingBagCollection
        self.ebc = EmbeddingBagCollection(
            tables=self.eb_configs,
            device=self.device,
        )

        # Create base model with the same architecture as in TestTorchRecDLRM
        self.base_model = TorchRecDLRM(
            embedding_bag_collection=self.ebc,
            dense_in_features=self.dense_in_features,
            dense_arch_layer_sizes=[16, self.embedding_dim],
            over_arch_layer_sizes=[64, 32, 1],
        )

        # Create wrapper
        self.model_wrapper = DLRMRatingWrapper(self.base_model).to(self.device)

        # Test data
        self.batch_size = 8
        self.dense_features = torch.rand(self.batch_size, self.dense_in_features)

        # Create batch data
        self.batch = {
            "user_id": torch.randint(0, self.num_users, (self.batch_size,)),
            "item_id": torch.randint(0, self.num_items, (self.batch_size,)),
            "user_category": torch.randint(
                0, self.num_user_categories, (self.batch_size,)
            ),
            "item_category": torch.randint(
                0, self.num_item_categories, (self.batch_size,)
            ),
        }

        # Create KeyedJaggedTensor for sparse features
        self.sparse_features = create_kjt_from_batch(self.batch, self.device)

    def test_forward(self) -> None:
        """Test forward pass of DLRMRatingWrapper."""
        # Run forward pass
        output = self.model_wrapper(self.dense_features, self.sparse_features)

        # Check output has the correct batch size
        self.assertEqual(output.numel(), self.batch_size)

        # Ensure output is 1D or 2D with second dimension of 1
        if output.dim() == 2:
            self.assertEqual(output.shape[1], 1)
            # Squeeze to make it 1D for further checks
            output = output.squeeze()

        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())

        # Since we're using random initialization, we can't guarantee exact output range
        # Just check that the output is finite
        self.assertTrue(torch.isfinite(output).all())


class TestCreateKJTFromBatch(unittest.TestCase):
    """Test cases for the create_kjt_from_batch function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 8
        self.device = torch.device("cpu")

        # Create batch data
        self.batch = {
            "user_id": torch.randint(0, 100, (self.batch_size,)),
            "item_id": torch.randint(0, 50, (self.batch_size,)),
            "user_category": torch.randint(0, 10, (self.batch_size,)),
            "item_category": torch.randint(0, 20, (self.batch_size,)),
        }

    def test_create_kjt_from_batch(self) -> None:
        """Test create_kjt_from_batch function."""
        kjt = create_kjt_from_batch(self.batch, self.device)

        # Check that it's a KeyedJaggedTensor
        self.assertIsInstance(kjt, KeyedJaggedTensor)

        # Check keys - use set comparison to avoid order issues
        self.assertEqual(
            set(kjt.keys()), {"user_id", "item_id", "user_category", "item_category"}
        )

        # Check values length
        self.assertEqual(kjt.values().shape[0], self.batch_size * 4)

        # Check lengths
        self.assertEqual(kjt.lengths().shape[0], self.batch_size * 4)
        self.assertTrue((kjt.lengths() == 1).all().item())


if __name__ == "__main__":
    unittest.main()
