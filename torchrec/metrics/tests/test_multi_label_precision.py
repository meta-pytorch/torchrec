#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.metrics.multi_label_precision import (
    compute_multi_label_precision,
    MultiLabelPrecisionMetricComputation,
)


class MultiLabelPrecisionIntegerEncodingTest(unittest.TestCase):
    """Tests for MultiLabelPrecisionMetricComputation with integer-encoded inputs.

    The metric expects predictions and labels to be integer-encoded using
    LSB-first bit ordering. For example, with num_labels=3:
        - Integer 5 (binary 101) decodes to [1, 0, 1] (labels 0 and 2 are positive)
        - Integer 3 (binary 011) decodes to [1, 1, 0] (labels 0 and 1 are positive)
        - Integer 7 (binary 111) decodes to [1, 1, 1] (all labels are positive)
    """

    def _create_computation(
        self, num_labels: int = 1, label_names: list[str] | None = None
    ) -> MultiLabelPrecisionMetricComputation:
        """Helper to create a MultiLabelPrecisionMetricComputation instance."""
        return MultiLabelPrecisionMetricComputation(
            my_rank=0,
            batch_size=100,
            n_tasks=1,
            window_size=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            process_group=None,
            fused_update_limit=0,
            num_labels=num_labels,
            label_names=label_names,
        )

    def test_custom_label_names_creates_per_label_states(self) -> None:
        """Verify that custom label_names creates states for each label"""
        label_names = ["purchase", "add_to_cart", "view"]

        computation = self._create_computation(num_labels=3, label_names=label_names)

        # Verify that states exist for each label name
        for label_name in label_names:
            self.assertTrue(
                hasattr(computation, f"true_pos_sum_{label_name}"),
                f"Missing true_pos_sum_{label_name} state",
            )
            self.assertTrue(
                hasattr(computation, f"false_pos_sum_{label_name}"),
                f"Missing false_pos_sum_{label_name} state",
            )

    def test_default_label_names_generated_from_num_labels(self) -> None:
        """Verify that default label names are generated when not provided"""
        num_labels = 4

        computation = self._create_computation(num_labels=num_labels)

        # Verify default label names (label_0, label_1, etc.) are created
        for i in range(num_labels):
            expected_label = f"label_{i}"
            self.assertTrue(
                hasattr(computation, f"true_pos_sum_{expected_label}"),
                f"Missing true_pos_sum_{expected_label} state",
            )
            self.assertTrue(
                hasattr(computation, f"false_pos_sum_{expected_label}"),
                f"Missing false_pos_sum_{expected_label} state",
            )

    def test_compute_multi_label_precision_basic(self) -> None:
        """Test basic precision calculation: TP / (TP + FP)"""
        # Precision = 3 / (3 + 1) = 0.75
        true_positives = torch.tensor([3.0])
        false_positives = torch.tensor([1.0])

        result = compute_multi_label_precision(true_positives, false_positives)
        expected = torch.tensor([0.75])

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_compute_multi_label_precision_zero_denominator(self) -> None:
        """Test precision returns 0 when TP + FP = 0"""
        true_positives = torch.tensor([0.0])
        false_positives = torch.tensor([0.0])

        result = compute_multi_label_precision(true_positives, false_positives)
        expected = torch.tensor([0.0])

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_decode_integer_to_labels(self) -> None:
        """Test that integer decoding produces correct binary vectors"""
        computation = self._create_computation(num_labels=3)

        # Test: 5 (101) -> [1, 0, 1], 3 (011) -> [1, 1, 0]
        tensor = torch.tensor([5, 3])
        result = computation._decode_integer_to_labels(tensor, num_labels=3)

        expected = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_decode_integer_to_labels_single_label(self) -> None:
        """Test decoding with num_labels=1"""
        computation = self._create_computation(num_labels=1)

        tensor = torch.tensor([0, 1, 0, 1])
        result = computation._decode_integer_to_labels(tensor, num_labels=1)

        expected = torch.tensor([[0], [1], [0], [1]], dtype=torch.int32)
        torch.testing.assert_close(result, expected)

    def test_update_accumulates_tp_fp(self) -> None:
        """Test that update correctly accumulates TP and FP"""
        computation = self._create_computation(num_labels=2)

        # Integer-encoded: 2 samples, 2 labels (LSB first)
        # Sample 1: pred=3 (11) -> [1,1], label=1 (01) -> [1,0]
        #   -> label_0: TP=1, label_1: FP=1
        # Sample 2: pred=3 (11) -> [1,1], label=3 (11) -> [1,1]
        #   -> label_0: TP=1, label_1: TP=1
        predictions = torch.tensor([3, 3])
        labels = torch.tensor([1, 3])
        weights = torch.ones(2)

        computation.update(predictions=predictions, labels=labels, weights=weights)

        # label_0: TP=2, FP=0
        # label_1: TP=1, FP=1
        self.assertEqual(computation.true_pos_sum_label_0.item(), 2.0)
        self.assertEqual(computation.false_pos_sum_label_0.item(), 0.0)
        self.assertEqual(computation.true_pos_sum_label_1.item(), 1.0)
        self.assertEqual(computation.false_pos_sum_label_1.item(), 1.0)

    def test_update_with_weights(self) -> None:
        """Test that weights are applied correctly"""
        computation = self._create_computation(num_labels=2)

        # Sample 1: pred=3 (11) -> [1,1], label=1 (01) -> [1,0], weight=2.0
        # Sample 2: pred=3 (11) -> [1,1], label=2 (10) -> [0,1], weight=1.0
        predictions = torch.tensor([3, 3])
        labels = torch.tensor([1, 2])
        weights = torch.tensor([2.0, 1.0])

        computation.update(predictions=predictions, labels=labels, weights=weights)

        # label_0: weighted TP=2, weighted FP=1
        # label_1: weighted TP=1, weighted FP=2
        self.assertEqual(computation.true_pos_sum_label_0.item(), 2.0)
        self.assertEqual(computation.false_pos_sum_label_0.item(), 1.0)
        self.assertEqual(computation.true_pos_sum_label_1.item(), 1.0)
        self.assertEqual(computation.false_pos_sum_label_1.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
