#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import List, Optional, Tuple

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from parameterized import param, parameterized
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.jagged_tensor_validator import validate_keyed_jagged_tensor


@st.composite
def valid_kjt_from_lengths_offsets_strategy(
    draw: st.DrawFn,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    keys = draw(st.lists(st.text(), min_size=1, max_size=10, unique=True))

    stride = draw(st.integers(1, 10))
    lengths = torch.tensor(
        draw(
            st.lists(
                st.integers(0, 20),
                min_size=len(keys) * stride,
                max_size=len(keys) * stride,
            )
        )
    )
    offsets = torch.cat((torch.tensor([0]), torch.cumsum(lengths, dim=0)))

    value_length = int(lengths.sum().item())
    values = torch.tensor(
        draw(
            st.lists(
                st.floats(0, 100),
                min_size=value_length,
                max_size=value_length,
            )
        )
    )
    weights_raw = draw(
        st.one_of(
            st.none(),
            st.lists(
                st.floats(0, 100),
                min_size=value_length,
                max_size=value_length,
            ),
        )
    )
    weights = torch.tensor(weights_raw) if weights_raw is not None else None

    return keys, values, weights, lengths, offsets


class TestJaggedTensorValidator(unittest.TestCase):
    INVALID_LENGTHS_OFFSETS_CASES = [
        param(
            expected_error_msg="lengths and offsets cannot be both empty",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=None,
            offsets=None,
        ),
        param(
            expected_error_msg="Expected lengths size to be 1 more than offsets size",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 5]),
        ),
        # Empty lengths is allowed but values must be empty as well
        param(
            expected_error_msg="Sum of lengths must equal the number of values",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([]),
            offsets=None,
        ),
        param(
            expected_error_msg="Sum of lengths must equal the number of values",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([3, 3, 2, 1]),
            offsets=None,
        ),
        param(
            expected_error_msg="offsets cannot be empty",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=None,
            offsets=torch.tensor([]),
        ),
        param(
            expected_error_msg="Expected first offset to be 0",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([1, 2, 4, 4, 6]),
        ),
        param(
            expected_error_msg="The last element of offsets must equal to the number of values",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 2, 4, 4, 6]),
        ),
        param(
            expected_error_msg="offsets is not equal to the cumulative sum of lengths",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 2, 3, 3, 5]),
        ),
    ]

    INVALID_KEYS_CASES = [
        param(
            expected_error_msg="keys must be unique",
            keys=["f1", "f1"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        ),
        param(
            expected_error_msg="keys is empty but lengths or offsets is not",
            keys=[],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        ),
        param(
            expected_error_msg="lengths size must be divisible by keys size",
            keys=["f1", "f2", "f3"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
        ),
    ]

    INVALID_WEIGHTS_CASES = [
        param(
            expected_error_msg="weights size must equal to values size",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
            weights=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        ),
        param(
            expected_error_msg="weights size must equal to values size",
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 2, 0, 2]),
            offsets=torch.tensor([0, 1, 3, 3, 5]),
            weights=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ),
    ]

    @parameterized.expand(
        [
            *INVALID_LENGTHS_OFFSETS_CASES,
            *INVALID_KEYS_CASES,
            *INVALID_WEIGHTS_CASES,
        ]
    )
    def test_invalid_keyed_jagged_tensor(
        self,
        expected_error_msg: str,
        keys: List[str],
        values: torch.Tensor,
        lengths: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            offsets=offsets,
            weights=weights,
        )

        with self.assertRaises(ValueError) as err:
            validate_keyed_jagged_tensor(kjt)
        self.assertIn(expected_error_msg, str(err.exception))

    # pyre-ignore[56]
    @given(valid_kjt_from_lengths_offsets_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=20)
    def test_valid_kjt_from_lengths(
        self,
        test_data: Tuple[
            List[str],
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        keys, values, weights, lengths, _ = test_data
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=keys, values=values, weights=weights, lengths=lengths
        )

        validate_keyed_jagged_tensor(kjt)

    # pyre-ignore[56]
    @given(valid_kjt_from_lengths_offsets_strategy())
    @settings(verbosity=Verbosity.verbose, max_examples=20)
    def test_valid_kjt_from_offsets(
        self,
        test_data: Tuple[
            List[str],
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        keys, values, weights, _, offsets = test_data
        kjt = KeyedJaggedTensor.from_offsets_sync(
            keys=keys, values=values, weights=weights, offsets=offsets
        )

        validate_keyed_jagged_tensor(kjt)

    def test_valid_empty_kjt(self) -> None:
        kjt = KeyedJaggedTensor.empty()

        result = validate_keyed_jagged_tensor(kjt)
        self.assertTrue(result)

    def test_feature_range_valid_values(self) -> None:
        # Setup: Create KJT with values within valid range and corresponding configs
        kjt = KeyedJaggedTensor(
            keys=["feature1", "feature2"],
            values=torch.tensor([0, 5, 10, 1, 2, 3]),
            lengths=torch.tensor([3, 3]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=20,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1", "feature2"],
            )
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Validation should return True for valid values
        self.assertTrue(result)

    def test_feature_range_boundary_values(self) -> None:
        # Setup: Create KJT with boundary values (0 and num_embeddings-1)
        kjt = KeyedJaggedTensor(
            keys=["feature1"],
            values=torch.tensor([0, 9]),
            lengths=torch.tensor([2]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            )
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Boundary values should be valid
        self.assertTrue(result)

    def test_feature_range_negative_value_returns_false(self) -> None:
        # Setup: Create KJT with negative value
        kjt = KeyedJaggedTensor(
            keys=["feature1"],
            values=torch.tensor([-1, 5, 10]),
            lengths=torch.tensor([3]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=20,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            )
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Validation should return False for out of range values
        self.assertFalse(result)

    def test_feature_range_value_exceeds_num_embeddings_returns_false(
        self,
    ) -> None:
        # Setup: Create KJT with value >= num_embeddings
        kjt = KeyedJaggedTensor(
            keys=["feature1"],
            values=torch.tensor([0, 5, 20]),
            lengths=torch.tensor([3]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=20,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            )
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Validation should return False for out of range values
        self.assertFalse(result)

    def test_feature_range_feature_not_in_config_returns_true(self) -> None:
        # Setup: Create KJT with feature not in any config
        kjt = KeyedJaggedTensor(
            keys=["feature1", "feature2"],
            values=torch.tensor([0, 5, 1, 2]),
            lengths=torch.tensor([2, 2]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=20,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            )
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Should return True since feature2 is just not in config (not invalid)
        self.assertTrue(result)

    def test_feature_range_multiple_tables_with_different_ranges(self) -> None:
        # Setup: Create KJT with features from different tables with different num_embeddings
        kjt = KeyedJaggedTensor(
            keys=["feature1", "feature2"],
            values=torch.tensor([0, 5, 10, 15, 25]),
            lengths=torch.tensor([3, 2]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=15,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            ),
            EmbeddingBagConfig(
                num_embeddings=30,
                embedding_dim=16,
                name="table2",
                feature_names=["feature2"],
            ),
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: feature1 has max 10 < 15, feature2 has max 25 < 30 - all valid
        self.assertTrue(result)

    def test_feature_range_multiple_features_one_out_of_range(self) -> None:
        # Setup: Create KJT with one feature in range and one out of range
        kjt = KeyedJaggedTensor(
            keys=["feature1", "feature2"],
            values=torch.tensor([0, 5, 10, 15, 25]),
            lengths=torch.tensor([3, 2]),
        )
        configs = [
            EmbeddingBagConfig(
                num_embeddings=15,
                embedding_dim=8,
                name="table1",
                feature_names=["feature1"],
            ),
            EmbeddingBagConfig(
                num_embeddings=20,
                embedding_dim=16,
                name="table2",
                feature_names=["feature2"],
            ),
        ]

        # Execute: Validate KJT with configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Should return False since feature2 has value 25 >= 20
        self.assertFalse(result)

    def test_feature_range_empty_configs(self) -> None:
        # Setup: Create KJT with no configs provided
        kjt = KeyedJaggedTensor(
            keys=["feature1"],
            values=torch.tensor([0, 5, 10]),
            lengths=torch.tensor([3]),
        )
        configs = []

        # Execute: Validate KJT with empty configs
        result = validate_keyed_jagged_tensor(kjt, configs)

        # Assert: Should return True since no configs means no range validation
        self.assertTrue(result)
