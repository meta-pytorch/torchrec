#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest
from collections import Counter
from typing import List

import numpy as np
import torch
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestModelInput(unittest.TestCase):
    """Tests for ModelInput generation utilities."""

    def setUp(self) -> None:
        # Fix seeds for reproducibility
        # pyrefly: ignore [bad-argument-type, implicit-import]
        np.random.seed(42)
        torch.manual_seed(42)

        self.tables: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=64,
                num_embeddings=10000,
            ),
        ]
        self.batch_size = 1024
        self.mean_pooling_factor = 10

    def test_generate_power_law_alpha_none_uniform_distribution(self) -> None:
        """When power_law_alpha is None, indices should be uniformly distributed."""
        model_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=[],
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            power_law_alpha=None,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)

        # For uniform distribution over 10000 embeddings, ~1% should be in [0, 100)
        low_index_ratio = low_indices / total
        self.assertLess(low_index_ratio, 0.1)

    def test_generate_power_law_alpha_skewed_distribution(self) -> None:
        """When power_law_alpha is set, indices should follow a skewed distribution."""
        model_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=[],
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            power_law_alpha=1.2,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)

        # For power-law with alpha=1.2, a significant fraction should be in low indices
        low_index_ratio = low_indices / total
        self.assertGreater(low_index_ratio, 0.3)

    def test_generate_power_law_indices_within_valid_range(self) -> None:
        """Indices generated with power-law should be within [0, num_embeddings)."""
        num_embeddings = 1000
        tables = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=64,
                num_embeddings=num_embeddings,
            ),
        ]

        model_input = ModelInput.generate(
            tables=tables,
            weighted_tables=[],
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            power_law_alpha=1.1,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values()

        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))

    def test_generate_power_law_with_weighted_tables(self) -> None:
        """Power-law distribution should work with weighted tables."""
        weighted_tables = [
            EmbeddingBagConfig(
                name="weighted_table_0",
                feature_names=["weighted_feature_0"],
                embedding_dim=64,
                num_embeddings=10000,
            ),
        ]

        model_input = ModelInput.generate(
            tables=[],
            weighted_tables=weighted_tables,
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            power_law_alpha=1.2,
        )

        assert model_input.idscore_features is not None
        indices = model_input.idscore_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)
        low_index_ratio = low_indices / total

        self.assertGreater(low_index_ratio, 0.3)

    # =======================================================
    # Tests for ModelInput.generate() basic functionality
    # =======================================================

    def test_generate_returns_correct_structure(self) -> None:
        """ModelInput.generate should return a properly structured ModelInput."""
        model_input = ModelInput.generate(
            batch_size=8,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=16,
            pooling_avg=5,
        )

        self.assertIsInstance(model_input, ModelInput)
        self.assertEqual(model_input.float_features.shape, (8, 16))
        self.assertEqual(model_input.label.shape, (8,))
        self.assertIsInstance(model_input.idlist_features, KeyedJaggedTensor)
        self.assertIsNone(model_input.idscore_features)

    def test_generate_with_all_zeros(self) -> None:
        """When all_zeros=True, tensors should contain only zeros."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=8,
            all_zeros=True,
        )

        self.assertTrue(torch.all(model_input.float_features == 0))
        self.assertTrue(torch.all(model_input.label == 0))
        self.assertIsNotNone(model_input.idlist_features)
        self.assertTrue(torch.all(model_input.idlist_features.values() == 0))

    def test_generate_with_use_offsets(self) -> None:
        """When use_offsets=True, KJT should use offsets instead of lengths."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=self.tables,
            weighted_tables=[],
            use_offsets=True,
        )

        self.assertIsNotNone(model_input.idlist_features)
        kjt = model_input.idlist_features
        self.assertIsNotNone(kjt.offsets_or_none())
        self.assertIsNone(kjt.lengths_or_none())

    def test_generate_with_multiple_tables(self) -> None:
        """ModelInput.generate should handle multiple tables with multiple features."""
        tables: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0", "feature_1"],
                embedding_dim=64,
                num_embeddings=1000,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_2"],
                embedding_dim=32,
                num_embeddings=500,
            ),
        ]

        model_input = ModelInput.generate(
            batch_size=4,
            tables=tables,
            weighted_tables=[],
        )

        self.assertIsNotNone(model_input.idlist_features)
        self.assertEqual(
            model_input.idlist_features.keys(), ["feature_0", "feature_1", "feature_2"]
        )

    def test_generate_with_different_dtypes(self) -> None:
        """ModelInput.generate should respect custom dtype parameters."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=self.tables,
            weighted_tables=[],
            indices_dtype=torch.int32,
            lengths_dtype=torch.int32,
        )

        self.assertIsNotNone(model_input.idlist_features)
        kjt = model_input.idlist_features
        self.assertEqual(kjt.values().dtype, torch.int32)
        self.assertIsNotNone(kjt.lengths_or_none())
        self.assertEqual(kjt.lengths().dtype, torch.int32)

    def test_generate_with_tables_pooling(self) -> None:
        """ModelInput.generate should respect custom tables_pooling factors."""
        high_pooling = 50
        model_input = ModelInput.generate(
            batch_size=32,
            tables=self.tables,
            weighted_tables=[],
            tables_pooling=[high_pooling],
        )

        self.assertIsNotNone(model_input.idlist_features)
        kjt = model_input.idlist_features
        avg_pooling = kjt.values().numel() / 32
        # Average pooling should be close to the specified value (within 50% tolerance)
        self.assertGreater(avg_pooling, high_pooling * 0.5)
        self.assertLess(avg_pooling, high_pooling * 1.5)

    def test_generate_with_max_feature_lengths(self) -> None:
        """ModelInput.generate should respect max_feature_lengths constraint."""
        max_length = 3
        model_input = ModelInput.generate(
            batch_size=32,
            tables=self.tables,
            weighted_tables=[],
            pooling_avg=20,  # High pooling avg to ensure clipping is needed
            max_feature_lengths=[max_length],
        )

        self.assertIsNotNone(model_input.idlist_features)
        kjt = model_input.idlist_features
        lengths = kjt.lengths()
        self.assertTrue(torch.all(lengths <= max_length))

    def test_generate_without_tables(self) -> None:
        """ModelInput.generate should work when tables are empty or None."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=None,
            weighted_tables=None,
            num_float_features=8,
        )

        self.assertEqual(model_input.float_features.shape, (4, 8))
        self.assertEqual(model_input.label.shape, (4,))
        self.assertIsNone(model_input.idlist_features)
        self.assertIsNone(model_input.idscore_features)

    # =======================================================
    # Tests for ModelInput.size_in_bytes()
    # =======================================================

    def test_size_in_bytes_basic(self) -> None:
        """size_in_bytes should return positive integer reflecting memory usage."""
        model_input = ModelInput.generate(
            batch_size=16,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=32,
        )

        size = model_input.size_in_bytes()
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)

    def test_size_in_bytes_increases_with_batch_size(self) -> None:
        """size_in_bytes should increase with larger batch sizes."""
        model_input_small = ModelInput.generate(
            batch_size=8,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=16,
            pooling_avg=5,
            all_zeros=True,
        )
        model_input_large = ModelInput.generate(
            batch_size=32,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=16,
            pooling_avg=5,
            all_zeros=True,
        )

        size_small = model_input_small.size_in_bytes()
        size_large = model_input_large.size_in_bytes()
        self.assertGreater(size_large, size_small)

    def test_size_in_bytes_without_features(self) -> None:
        """size_in_bytes should work correctly without idlist/idscore features."""
        model_input = ModelInput.generate(
            batch_size=8,
            tables=None,
            weighted_tables=None,
            num_float_features=16,
        )

        size = model_input.size_in_bytes()
        # Size should be float_features + label size
        expected_size = (8 * 16 + 8) * 4  # float32 = 4 bytes
        self.assertEqual(size, expected_size)

    # =======================================================
    # Tests for ModelInput.to()
    # =======================================================

    def test_to_cpu_device(self) -> None:
        """ModelInput.to should transfer tensors to specified CPU device."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=8,
        )

        cpu_device = torch.device("cpu")
        result = model_input.to(device=cpu_device)

        self.assertEqual(result.float_features.device, cpu_device)
        self.assertEqual(result.label.device, cpu_device)
        self.assertIsNotNone(result.idlist_features)
        self.assertEqual(result.idlist_features.device(), cpu_device)

    def test_to_preserves_data(self) -> None:
        """ModelInput.to should preserve tensor values."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=8,
        )

        cpu_device = torch.device("cpu")
        result = model_input.to(device=cpu_device)

        self.assertTrue(torch.equal(result.float_features, model_input.float_features))
        self.assertTrue(torch.equal(result.label, model_input.label))

    def test_to_with_none_features(self) -> None:
        """ModelInput.to should handle None idlist/idscore features."""
        model_input = ModelInput.generate(
            batch_size=4,
            tables=None,
            weighted_tables=None,
            num_float_features=8,
        )

        cpu_device = torch.device("cpu")
        result = model_input.to(device=cpu_device)

        self.assertIsNone(result.idlist_features)
        self.assertIsNone(result.idscore_features)
        self.assertEqual(result.float_features.device, cpu_device)

    # =======================================================
    # Tests for ModelInput.generate_local_batches()
    # =======================================================

    def test_generate_local_batches_returns_correct_count(self) -> None:
        """generate_local_batches should return world_size batches."""
        world_size = 4
        local_batches = ModelInput.generate_local_batches(
            world_size=world_size,
            batch_size=8,
            tables=self.tables,
            weighted_tables=[],
        )

        self.assertEqual(len(local_batches), world_size)
        for batch in local_batches:
            self.assertIsInstance(batch, ModelInput)
            self.assertEqual(batch.float_features.shape[0], 8)

    def test_generate_local_batches_independent(self) -> None:
        """Each local batch should have independent random data."""
        local_batches = ModelInput.generate_local_batches(
            world_size=2,
            batch_size=8,
            tables=self.tables,
            weighted_tables=[],
        )

        # Check float_features are different between batches
        self.assertFalse(
            torch.equal(
                local_batches[0].float_features, local_batches[1].float_features
            )
        )

    # =======================================================
    # Tests for ModelInput.generate_global_and_local_batches()
    # =======================================================

    def test_generate_global_and_local_batches_structure(self) -> None:
        """generate_global_and_local_batches should return correct structure."""
        world_size = 2
        batch_size = 4
        global_input, local_inputs = ModelInput.generate_global_and_local_batches(
            world_size=world_size,
            batch_size=batch_size,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=8,
        )

        self.assertIsInstance(global_input, ModelInput)
        self.assertEqual(len(local_inputs), world_size)
        # Global batch should have world_size * batch_size rows
        self.assertEqual(global_input.float_features.shape[0], world_size * batch_size)
        # Each local batch should have batch_size rows
        for local_input in local_inputs:
            self.assertEqual(local_input.float_features.shape[0], batch_size)

    def test_generate_global_and_local_batches_consistent(self) -> None:
        """Global batch should be concatenation of local batches for float_features."""
        world_size = 2
        batch_size = 4
        global_input, local_inputs = ModelInput.generate_global_and_local_batches(
            world_size=world_size,
            batch_size=batch_size,
            tables=self.tables,
            weighted_tables=[],
            num_float_features=8,
        )

        # Float features should be concatenated
        expected_float_features = torch.cat([b.float_features for b in local_inputs])
        self.assertTrue(
            torch.equal(global_input.float_features, expected_float_features)
        )

        # Labels should be concatenated
        expected_labels = torch.cat([b.label for b in local_inputs])
        self.assertTrue(torch.equal(global_input.label, expected_labels))

    # =======================================================
    # Tests for ModelInput.create_standard_kjt()
    # =======================================================

    def test_create_standard_kjt_basic(self) -> None:
        """create_standard_kjt should create valid KeyedJaggedTensor."""
        kjt = ModelInput.create_standard_kjt(
            batch_size=8,
            tables=self.tables,
            pooling_avg=5,
            weighted=False,
        )

        self.assertIsInstance(kjt, KeyedJaggedTensor)
        self.assertEqual(kjt.keys(), ["feature_0"])
        self.assertEqual(kjt.values().numel(), kjt.lengths().sum().item())

    def test_create_standard_kjt_weighted(self) -> None:
        """create_standard_kjt with weighted=True should include weights."""
        kjt = ModelInput.create_standard_kjt(
            batch_size=8,
            tables=self.tables,
            pooling_avg=5,
            weighted=True,
        )

        self.assertIsNotNone(kjt.weights_or_none())
        self.assertEqual(kjt.weights().numel(), kjt.values().numel())

    def test_create_standard_kjt_with_offsets(self) -> None:
        """create_standard_kjt with use_offsets=True should use offsets."""
        kjt = ModelInput.create_standard_kjt(
            batch_size=8,
            tables=self.tables,
            pooling_avg=5,
            use_offsets=True,
        )

        self.assertIsNotNone(kjt.offsets_or_none())
        self.assertIsNone(kjt.lengths_or_none())

    # =======================================================
    # Tests for _generate_power_law_indices edge cases
    # =======================================================

    def test_generate_power_law_indices_alpha_zero_uniform(self) -> None:
        """When alpha=0.0, _generate_power_law_indices should return uniform distribution."""
        num_embeddings = 1000

        # Call the private method directly for edge case testing
        indices = ModelInput._generate_power_law_indices(
            alpha=0.0,
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))

        # Check distribution is roughly uniform
        counter = Counter(indices.tolist())
        low_indices = sum(counter.get(i, 0) for i in range(100))
        low_index_ratio = low_indices / len(indices)
        self.assertLess(low_index_ratio, 0.15)  # Should be ~10% for uniform

    def test_generate_power_law_indices_alpha_less_than_one(self) -> None:
        """Test _generate_power_law_indices with alpha < 1.0 (uses CDF method)."""
        num_embeddings = 1000

        indices = ModelInput._generate_power_law_indices(
            alpha=0.5,  # Less than 1.0
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))
        # Verify index 0 is actually generated (validates off-by-one fix)
        self.assertIn(0, indices.tolist())

    def test_generate_power_law_indices_alpha_equals_one(self) -> None:
        """Test _generate_power_law_indices with alpha == 1.0 (log-uniform, special case)."""
        num_embeddings = 1000

        # This test verifies that alpha=1.0 does not cause divide-by-zero
        indices = ModelInput._generate_power_law_indices(
            alpha=1.0,  # Special case: log-uniform
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))
        # Verify index 0 is actually generated (validates off-by-one fix)
        self.assertIn(0, indices.tolist())

        # For log-uniform (alpha=1), distribution should be moderately skewed
        counter = Counter(indices.tolist())
        low_indices = sum(counter.get(i, 0) for i in range(100))
        low_index_ratio = low_indices / len(indices)
        # Should be more than uniform (10%) but less than heavily skewed
        self.assertGreater(low_index_ratio, 0.1)

    def test_generate_power_law_indices_alpha_greater_than_one(self) -> None:
        """Test _generate_power_law_indices with alpha >= 1.0 (uses Pareto distribution)."""
        num_embeddings = 1000

        indices = ModelInput._generate_power_law_indices(
            alpha=1.5,  # Greater than 1.0
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))
        # Verify index 0 is actually generated (validates off-by-one fix)
        self.assertIn(0, indices.tolist())

        # Check distribution is skewed
        counter = Counter(indices.tolist())
        low_indices = sum(counter.get(i, 0) for i in range(100))
        low_index_ratio = low_indices / len(indices)
        self.assertGreater(low_index_ratio, 0.3)

    def test_generate_power_law_indices_num_embeddings_one(self) -> None:
        """When num_embeddings=1, all indices should be 0."""
        indices = ModelInput._generate_power_law_indices(
            alpha=1.5,
            num_indices=100,
            num_embeddings=1,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 100)
        self.assertTrue(torch.all(indices == 0))

    def test_generate_power_law_indices_alpha_near_one_from_below(self) -> None:
        """Test alpha very close to 1 from below uses log-uniform branch."""
        num_embeddings = 1000

        # alpha = 0.999 is within tolerance of 1.0, should use log-uniform
        indices = ModelInput._generate_power_law_indices(
            alpha=0.999,
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))
        # Should not have numerical overflow issues
        self.assertFalse(torch.any(torch.isnan(indices.float())))
        self.assertFalse(torch.any(torch.isinf(indices.float())))

    def test_generate_power_law_indices_alpha_near_one_from_above(self) -> None:
        """Test alpha very close to 1 from above uses log-uniform branch."""
        num_embeddings = 1000

        # alpha = 1.001 is within tolerance of 1.0, should use log-uniform
        indices = ModelInput._generate_power_law_indices(
            alpha=1.001,
            num_indices=10000,
            num_embeddings=num_embeddings,
            dtype=torch.int64,
            device=None,
        )

        self.assertEqual(indices.numel(), 10000)
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))
        # Should not have numerical overflow issues
        self.assertFalse(torch.any(torch.isnan(indices.float())))
        self.assertFalse(torch.any(torch.isinf(indices.float())))

    def test_generate_power_law_indices_negative_alpha_raises(self) -> None:
        """Negative alpha should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            ModelInput._generate_power_law_indices(
                alpha=-0.5,
                num_indices=100,
                num_embeddings=1000,
                dtype=torch.int64,
                device=None,
            )
        self.assertIn("alpha must be >= 0", str(context.exception))

    def test_generate_power_law_indices_zero_embeddings_raises(self) -> None:
        """num_embeddings=0 should raise ValueError."""
        with self.assertRaises(ValueError) as context:
            ModelInput._generate_power_law_indices(
                alpha=1.0,
                num_indices=100,
                num_embeddings=0,
                dtype=torch.int64,
                device=None,
            )
        self.assertIn("num_embeddings must be >= 1", str(context.exception))


if __name__ == "__main__":
    unittest.main()
