#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestModelInput(unittest.TestCase):
    """Tests for ModelInput generation utilities."""

    def setUp(self) -> None:
        # Fix seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        self.tables = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=64,
                num_embeddings=10000,
            ),
        ]
        self.batch_size = 1024
        self.mean_pooling_factor = 10

    def test_generate_zipf_alpha_none_uniform_distribution(self) -> None:
        """When zipf_alpha is None, indices should be uniformly distributed."""
        model_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=[],
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            zipf_alpha=None,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)

        # For uniform distribution over 10000 embeddings, ~1% should be in [0, 100)
        low_index_ratio = low_indices / total
        self.assertLess(low_index_ratio, 0.1)

    def test_generate_zipf_alpha_skewed_distribution(self) -> None:
        """When zipf_alpha is set, indices should follow a skewed distribution."""
        model_input = ModelInput.generate(
            tables=self.tables,
            weighted_tables=[],
            batch_size=self.batch_size,
            num_float_features=10,
            pooling_avg=self.mean_pooling_factor,
            zipf_alpha=1.2,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)

        # For Zipf with alpha=1.2, a significant fraction should be in low indices
        low_index_ratio = low_indices / total
        self.assertGreater(low_index_ratio, 0.3)

    def test_generate_zipf_indices_within_valid_range(self) -> None:
        """Indices generated with Zipf should be within [0, num_embeddings)."""
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
            zipf_alpha=1.1,
        )

        assert model_input.idlist_features is not None
        indices = model_input.idlist_features.values()

        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < num_embeddings))

    def test_generate_zipf_with_weighted_tables(self) -> None:
        """Zipf distribution should work with weighted tables."""
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
            zipf_alpha=1.2,
        )

        assert model_input.idscore_features is not None
        indices = model_input.idscore_features.values().tolist()

        counter = Counter(indices)
        low_indices = sum(counter.get(i, 0) for i in range(100))
        total = len(indices)
        low_index_ratio = low_indices / total

        self.assertGreater(low_index_ratio, 0.3)

    def test_generate_zipf_fallback_when_numpy_unavailable(self) -> None:
        """When numpy is unavailable, should fall back to uniform distribution."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):  # pyre-ignore[3]
            if name == "numpy":
                raise ImportError("Mocked numpy import failure")
            return original_import(name, *args, **kwargs)

        torch.manual_seed(42)
        with patch.object(builtins, "__import__", side_effect=mock_import):
            fallback_indices = ModelInput._generate_zipf_indices(
                zipf_alpha=1.2,
                num_indices=10000,
                num_embeddings=10000,
                dtype=torch.int64,
                device=None,
            )

        # Verify indices are within valid range
        self.assertTrue(torch.all(fallback_indices >= 0))
        self.assertTrue(torch.all(fallback_indices < 10000))

        # Verify fallback is uniform distribution, NOT Zipf
        # Zipf with alpha=1.2 would have >30% of indices in [0, 100)
        # Uniform distribution should have ~1% in [0, 100)
        fallback_counter = Counter(fallback_indices.tolist())
        fallback_low_indices = sum(fallback_counter.get(i, 0) for i in range(100))
        fallback_low_ratio = fallback_low_indices / len(fallback_indices)

        self.assertLess(
            fallback_low_ratio,
            0.1,
            "Fallback should be uniform (not Zipf which would have >30% in low indices)",
        )


if __name__ == "__main__":
    unittest.main()
