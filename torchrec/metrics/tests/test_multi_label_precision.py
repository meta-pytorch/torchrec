#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from torchrec.metrics.multi_label_precision import (
    compute_multi_label_precision,
    MultiLabelPrecisionMetric,
    MultiLabelPrecisionMetricComputation,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecTaskInfo


class MultiLabelPrecisionMetricTest(unittest.TestCase):
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
            num_labels=num_labels,
            label_names=label_names,
        )

    def _create_metric(
        self,
        num_labels: int = 1,
        label_names: List[str] | None = None,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    ) -> MultiLabelPrecisionMetric:
        """Helper to create a MultiLabelPrecisionMetric instance."""
        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]
        return MultiLabelPrecisionMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=100,
            # pyre-fixme[6]: For 7th argument expected `Dict[str, Any]` but got `int`.
            num_labels=num_labels,
            # pyre-fixme[6]: For 8th argument expected `Dict[str, Any]` but got
            #  `Optional[List[str]]`.
            label_names=label_names,
        )

    def test_custom_label_names_creates_per_label_states(self) -> None:
        """Verify that custom label_names creates states for each label"""
        label_names = ["purchase", "add_to_cart", "view"]

        computation = self._create_computation(num_labels=3, label_names=label_names)

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

        predictions = torch.tensor([3, 3])
        labels = torch.tensor([1, 3])
        weights = torch.ones(2)

        computation.update(predictions=predictions, labels=labels, weights=weights)

        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.true_pos_sum_label_0.item(), 2.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.false_pos_sum_label_0.item(), 0.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.true_pos_sum_label_1.item(), 1.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.false_pos_sum_label_1.item(), 1.0)

    def test_update_with_weights(self) -> None:
        """Test that weights are applied correctly"""
        computation = self._create_computation(num_labels=2)

        predictions = torch.tensor([3, 3])
        labels = torch.tensor([1, 2])
        weights = torch.tensor([2.0, 1.0])

        computation.update(predictions=predictions, labels=labels, weights=weights)

        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.true_pos_sum_label_0.item(), 2.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.false_pos_sum_label_0.item(), 1.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.true_pos_sum_label_1.item(), 1.0)
        # pyre-fixme[29]: `Union[(self: TensorBase) -> Union[bool, float, int],
        #  Tensor, Module]` is not a function.
        self.assertEqual(computation.false_pos_sum_label_1.item(), 2.0)

    # =========================================================================
    # Per-label metric tests for different compute modes
    # =========================================================================

    def test_metric_unfused_mode_per_label_output(self) -> None:
        """Test MultiLabelPrecisionMetric in UNFUSED mode produces per-label metrics."""
        num_labels = 3
        label_names = ["click", "purchase", "view"]
        metric = self._create_metric(
            num_labels=num_labels,
            label_names=label_names,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        )

        # Batch 1: predictions=7 (all 1s), labels=5 (101 -> click=1, purchase=0, view=1)
        # click: TP=1, FP=0; purchase: TP=0, FP=1; view: TP=1, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([7])},
            labels={"test_task": torch.tensor([5])},
            weights={"test_task": torch.tensor([1.0])},
        )

        # Batch 2: predictions=3 (011), labels=3 (011 -> click=1, purchase=1, view=0)
        # click: TP=1, FP=0; purchase: TP=1, FP=0; view: TP=0, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([3])},
            labels={"test_task": torch.tensor([3])},
            weights={"test_task": torch.tensor([1.0])},
        )

        results = metric.compute()

        # Verify per-label results exist
        for label_name in label_names:
            lifetime_key = f"multi_label_precision-test_task|lifetime_multi_label_precision{label_name}"
            window_key = f"multi_label_precision-test_task|window_multi_label_precision{label_name}"
            self.assertIn(
                lifetime_key, results, f"Missing lifetime metric for {label_name}"
            )
            self.assertIn(
                window_key, results, f"Missing window metric for {label_name}"
            )

        # Verify precision values
        # click: TP=2, FP=0 -> precision=1.0
        # purchase: TP=1, FP=1 -> precision=0.5
        # view: TP=1, FP=0 -> precision=1.0
        click_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionclick"
        ]
        purchase_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionpurchase"
        ]
        view_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionview"
        ]

        self.assertAlmostEqual(click_precision.item(), 1.0, places=5)
        self.assertAlmostEqual(purchase_precision.item(), 0.5, places=5)
        self.assertAlmostEqual(view_precision.item(), 1.0, places=5)

    def test_metric_fused_mode_per_label_output(self) -> None:
        """Test MultiLabelPrecisionMetric in FUSED mode produces per-label metrics."""
        num_labels = 2
        metric = self._create_metric(
            num_labels=num_labels,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        # predictions=3 (11), labels=1 (01)
        # label_0: TP=1, FP=0; label_1: TP=0, FP=1
        metric.update(
            predictions={"test_task": torch.tensor([3])},
            labels={"test_task": torch.tensor([1])},
            weights={"test_task": torch.tensor([1.0])},
        )

        results = metric.compute()

        # Verify per-label results exist
        for i in range(num_labels):
            label_name = f"label_{i}"
            lifetime_key = f"multi_label_precision-test_task|lifetime_multi_label_precision{label_name}"
            self.assertIn(
                lifetime_key, results, f"Missing lifetime metric for {label_name}"
            )

        # label_0: precision=1.0 (TP=1, FP=0)
        # label_1: precision=0.0 (TP=0, FP=1)
        label0_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_0"
        ]
        label1_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_1"
        ]

        self.assertAlmostEqual(label0_precision.item(), 1.0, places=5)
        self.assertAlmostEqual(label1_precision.item(), 0.0, places=5)

    def test_metric_multiple_batches_accumulation(self) -> None:
        """Test that metric correctly accumulates states across multiple batches."""
        num_labels = 2
        metric = self._create_metric(num_labels=num_labels)

        # Batch 1: pred=3 (11), label=1 (01) -> label_0: TP=1, FP=0; label_1: TP=0, FP=1
        metric.update(
            predictions={"test_task": torch.tensor([3])},
            labels={"test_task": torch.tensor([1])},
            weights={"test_task": torch.tensor([1.0])},
        )

        # Batch 2: pred=3 (11), label=3 (11) -> label_0: TP=1, FP=0; label_1: TP=1, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([3])},
            labels={"test_task": torch.tensor([3])},
            weights={"test_task": torch.tensor([1.0])},
        )

        # Batch 3: pred=2 (10), label=0 (00) -> label_0: TP=0, FP=0; label_1: TP=0, FP=1
        metric.update(
            predictions={"test_task": torch.tensor([2])},
            labels={"test_task": torch.tensor([0])},
            weights={"test_task": torch.tensor([1.0])},
        )

        results = metric.compute()

        # label_0: TP=2, FP=0 -> precision=1.0
        # label_1: TP=1, FP=2 -> precision=0.333...
        label0_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_0"
        ]
        label1_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_1"
        ]

        self.assertAlmostEqual(label0_precision.item(), 1.0, places=4)
        self.assertAlmostEqual(label1_precision.item(), 1.0 / 3.0, places=4)

    def test_metric_with_batch_weights(self) -> None:
        """Test that sample weights are correctly applied in metric computation."""
        num_labels = 2
        metric = self._create_metric(num_labels=num_labels)

        # Two samples with different weights
        # Sample 1 (weight=2): pred=3 (11), label=1 (01) -> label_0: TP=2, FP=0; label_1: TP=0, FP=2
        # Sample 2 (weight=1): pred=3 (11), label=2 (10) -> label_0: TP=0, FP=1; label_1: TP=1, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([3, 3])},
            labels={"test_task": torch.tensor([1, 2])},
            weights={"test_task": torch.tensor([2.0, 1.0])},
        )

        results = metric.compute()

        # label_0: TP=2, FP=1 -> precision=2/3
        # label_1: TP=1, FP=2 -> precision=1/3
        label0_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_0"
        ]
        label1_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_1"
        ]

        self.assertAlmostEqual(label0_precision.item(), 2.0 / 3.0, places=4)
        self.assertAlmostEqual(label1_precision.item(), 1.0 / 3.0, places=4)

    def test_metric_window_vs_lifetime(self) -> None:
        """Test that window and lifetime metrics track separately."""
        num_labels = 1
        # Window size must be larger than batch size
        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]
        metric = MultiLabelPrecisionMetric(
            world_size=1,
            my_rank=0,
            batch_size=1,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=2,  # Only keep last 2 batches in window
            # pyre-fixme[6]: For 7th argument expected `Dict[str, Any]` but got `int`.
            num_labels=num_labels,
        )

        # Batch 1: pred=1, label=0 -> FP=1, TP=0
        metric.update(
            predictions={"test_task": torch.tensor([1])},
            labels={"test_task": torch.tensor([0])},
            weights={"test_task": torch.tensor([1.0])},
        )

        # Batch 2: pred=1, label=1 -> TP=1, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([1])},
            labels={"test_task": torch.tensor([1])},
            weights={"test_task": torch.tensor([1.0])},
        )

        # Batch 3: pred=1, label=1 -> TP=1, FP=0
        metric.update(
            predictions={"test_task": torch.tensor([1])},
            labels={"test_task": torch.tensor([1])},
            weights={"test_task": torch.tensor([1.0])},
        )

        results = metric.compute()

        # Lifetime: TP=2, FP=1 -> precision=2/3
        lifetime_precision = results[
            "multi_label_precision-test_task|lifetime_multi_label_precisionlabel_0"
        ]
        self.assertAlmostEqual(lifetime_precision.item(), 2.0 / 3.0, places=4)

        # Window should only have last 2 batches (batch 2 and 3): TP=2, FP=0 -> precision=1.0
        window_precision = results[
            "multi_label_precision-test_task|window_multi_label_precisionlabel_0"
        ]
        self.assertAlmostEqual(window_precision.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
