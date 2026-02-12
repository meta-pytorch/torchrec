#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Dict, Iterable, Optional, Type, Union

import torch
from torchrec.metrics.metrics_config import RecTaskInfo
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecMetricException
from torchrec.metrics.tensor_weighted_avg import get_mean, TensorWeightedAvgMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


WORLD_SIZE = 4
# pyrefly: ignore[bad-assignment]
METRIC_NAMESPACE: str = TensorWeightedAvgMetric._namespace.value
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTensorWeightedAvgMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute states for tensor weighted average.

        For TensorWeightedAvgMetric, we use the 'required_inputs_tensor' parameter
        which contains the actual tensor to compute the weighted average on.
        """

        if required_inputs_tensor is None:
            raise ValueError("required_inputs_tensor cannot be None")

        # Compute weighted sum and weighted num samples using the target tensor
        weighted_sum = (required_inputs_tensor * weights).sum(dim=-1)
        weighted_num_samples = weights.sum(dim=-1)

        return {
            "weighted_sum": weighted_sum,
            "weighted_num_samples": weighted_num_samples,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_mean(states["weighted_sum"], states["weighted_num_samples"])


class TensorWeightedAvgMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = TensorWeightedAvgMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION

    def test_tensor_weighted_avg_unfused(self) -> None:
        """Test TensorWeightedAvgMetric with UNFUSED_TASKS_COMPUTATION."""
        rec_metric_value_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            device=DEVICE,
        )

    def test_tensor_weighted_avg_fused(self) -> None:
        """Test TensorWeightedAvgMetric with FUSED_TASKS_COMPUTATION."""
        rec_metric_value_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            device=DEVICE,
        )

    def test_tensor_weighted_avg_single_task_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["single_task"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            device=DEVICE,
        )

    def test_tensor_weighted_avg_single_task_fused(self) -> None:
        """Test TensorWeightedAvgMetric with single task in FUSED_TASKS_COMPUTATION mode."""
        rec_metric_value_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["single_task"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            device=DEVICE,
        )


class TensorWeightedAvgGPUSyncTest(unittest.TestCase):
    """GPU synchronization tests for TensorWeightedAvgMetric."""

    def test_sync_tensor_weighted_avg_unfused(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
            device=DEVICE,
        )

    def test_sync_tensor_weighted_avg_fused(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=TensorWeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTensorWeightedAvgMetric,
            metric_name=METRIC_NAMESPACE,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
            device=DEVICE,
        )


class TensorWeightedAvgFunctionalityTest(unittest.TestCase):
    """Test basic functionality of TensorWeightedAvgMetric."""

    def test_tensor_weighted_avg_basic_functionality(self) -> None:

        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=True,
            )
        ]
        metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        self.assertIsNotNone(metric)
        self.assertEqual(len(metric._metrics_computations), 1)

        computation = metric._metrics_computations[0]
        # pyrefly: ignore[bad-index, missing-attribute]
        self.assertEqual(computation.tasks[0].tensor_name, "test_tensor")
        # pyrefly: ignore[bad-index, missing-attribute]
        self.assertTrue(computation.tasks[0].weighted)

    def test_tensor_weighted_avg_unweighted_task(self) -> None:

        # Create an unweighted task
        tasks = [
            RecTaskInfo(
                name="unweighted_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=False,
            )
        ]

        metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        computation = metric._metrics_computations[0]
        # pyrefly: ignore[bad-index, missing-attribute]
        self.assertEqual(computation.tasks[0].tensor_name, "test_tensor")
        # pyrefly: ignore[bad-index, missing-attribute]
        self.assertFalse(computation.tasks[0].weighted)

    def test_tensor_weighted_avg_unfused_required_inputs_validation(self) -> None:
        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=True,
            )
        ]

        metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=2,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        # Test that required inputs are correctly identified
        required_inputs = metric.get_required_inputs()
        self.assertIn("test_tensor", required_inputs)

        # Test update with missing required inputs should fail
        with self.assertRaisesRegex(RecMetricException, "required_inputs"):
            metric.update(
                predictions={"test_task": torch.tensor([0.1, 0.2])},
                labels={"test_task": torch.tensor([1.0, 0.0])},
                weights={"test_task": torch.tensor([1.0, 2.0])},
            )

    def test_tensor_weighted_avg_unfused_missing_tensor_name_init_error(self) -> None:
        with self.assertRaisesRegex(RecMetricException, "tensor_name"):
            TensorWeightedAvgMetric(
                world_size=1,
                my_rank=0,
                batch_size=2,
                tasks=[
                    RecTaskInfo(
                        name="test_task",
                        label_name="test_label",
                        prediction_name="test_pred",
                        weight_name="test_weight",
                        tensor_name=None,
                        weighted=True,
                    )
                ],
                compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                window_size=100,
            )

    def test_tensor_weighted_avg_unfused_missing_tensor_name_update_error(self) -> None:
        """Test error when tensor_name is missing in required_inputs for UNFUSED mode."""
        single_task = [
            RecTaskInfo(
                name="test_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=True,
            )
        ]

        single_metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=2,
            tasks=single_task,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        with self.assertRaisesRegex(RecMetricException, "test_tensor"):
            single_metric.update(
                predictions={"test_task": torch.tensor([0.1, 0.2])},
                labels={"test_task": torch.tensor([1.0, 0.0])},
                weights={"test_task": torch.tensor([1.0, 2.0])},
                required_inputs={"wrong_tensor_name": torch.tensor([1.0, 2.0])},
            )

    def test_tensor_weighted_avg_fused_conflicting_weighted_flags_error(self) -> None:
        with self.assertRaisesRegex(
            RecMetricException, "already registered as weighted"
        ):
            TensorWeightedAvgMetric(
                world_size=1,
                my_rank=0,
                batch_size=2,
                tasks=[
                    RecTaskInfo(
                        name="task1",
                        label_name="test_label",
                        prediction_name="test_pred",
                        weight_name="test_weight",
                        tensor_name="shared_tensor",
                        weighted=True,
                    ),
                    RecTaskInfo(
                        name="task2",
                        label_name="test_label",
                        prediction_name="test_pred",
                        weight_name="test_weight",
                        tensor_name="shared_tensor",
                        weighted=False,
                    ),
                ],
                compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
                window_size=100,
            )

    def test_tensor_weighted_unfused_avg_update(self) -> None:
        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=True,
            )
        ]

        metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        ).to(DEVICE)

        predictions = {"test_task": torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)}
        labels = {"test_task": torch.tensor([1.0, 0.0, 1.0, 0.0], device=DEVICE)}
        weights = {"test_task": torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)}
        required_inputs = {
            "test_tensor": torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEVICE)
        }

        metric.update(
            predictions=predictions,
            labels=labels,
            weights=weights,
            required_inputs=required_inputs,
        )

    def test_tensor_weighted_fused_avg_update(self) -> None:
        tasks = [
            RecTaskInfo(
                name="test_task",
                label_name="test_label",
                prediction_name="test_pred",
                weight_name="test_weight",
                tensor_name="test_tensor",
                weighted=True,
            )
        ]

        metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=tasks,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            window_size=100,
        ).to(DEVICE)

        predictions = {"test_task": torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)}
        labels = {"test_task": torch.tensor([1.0, 0.0, 1.0, 0.0], device=DEVICE)}
        weights = {"test_task": torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)}
        required_inputs = {
            "test_tensor": torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEVICE)
        }

        metric.update(
            predictions=predictions,
            labels=labels,
            weights=weights,
            required_inputs=required_inputs,
        )


def generate_tensor_model_outputs_cases() -> Iterable[Dict[str, torch.Tensor]]:
    """Generate test cases with known inputs and expected tensor weighted average outputs."""
    return [
        # Basic weighted case
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]], device=DEVICE),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]], device=DEVICE),
            "tensors": torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0]], device=DEVICE),
            "weights": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], device=DEVICE),
            # Expected: (2.0*0.1 + 4.0*0.2 + 6.0*0.3 + 8.0*0.4 + 10.0*0.5) / (0.1+0.2+0.3+0.4+0.5) = 11/1.5 = 7.3333
            "expected_tensor_weighted_avg": torch.tensor([7.3333], device=DEVICE),
        },
        # Uniform weights (should equal simple average)
        {
            "labels": torch.tensor([[1, 0, 1, 0]], device=DEVICE),
            "predictions": torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=DEVICE),
            "tensors": torch.tensor([[1.0, 3.0, 5.0, 7.0]], device=DEVICE),
            "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=DEVICE),
            # Expected: (1.0 + 3.0 + 5.0 + 7.0) / 4 = 16/4 = 4.0
            "expected_tensor_weighted_avg": torch.tensor([4.0], device=DEVICE),
        },
        # No weights (should default to uniform weights)
        {
            "labels": torch.tensor([[1, 0, 1]], device=DEVICE),
            "predictions": torch.tensor([[0.3, 0.7, 0.5]], device=DEVICE),
            "tensors": torch.tensor([[2.0, 8.0, 5.0]], device=DEVICE),
            # Expected: (2.0 + 8.0 + 5.0) / 3 = 15/3 = 5.0
            "expected_tensor_weighted_avg": torch.tensor([5.0], device=DEVICE),
        },
        # Single non-zero weight
        {
            "labels": torch.tensor([[1, 0, 1, 0]], device=DEVICE),
            "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=DEVICE),
            "tensors": torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=DEVICE),
            "weights": torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=DEVICE),
            # Expected: only third element contributes: 30.0/1.0 = 30.0
            "expected_tensor_weighted_avg": torch.tensor([30.0], device=DEVICE),
        },
        # All weights zero (should result in NaN)
        {
            "labels": torch.tensor([[1, 1, 1]], device=DEVICE),
            "predictions": torch.tensor([[0.2, 0.6, 0.8]], device=DEVICE),
            "tensors": torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE),
            "weights": torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE),
            "expected_tensor_weighted_avg": torch.tensor([float("nan")], device=DEVICE),
        },
        # Negative tensor values
        {
            "labels": torch.tensor([[1, 0, 1]], device=DEVICE),
            "predictions": torch.tensor([[0.1, 0.5, 0.9]], device=DEVICE),
            "tensors": torch.tensor([[-2.0, 4.0, -6.0]], device=DEVICE),
            "weights": torch.tensor([[0.5, 0.3, 0.2]], device=DEVICE),
            # Expected: (-2.0*0.5 + 4.0*0.3 + -6.0*0.2) / (0.5+0.3+0.2) = (-1.0 + 1.2 - 1.2) / 1.0 = -1.0
            "expected_tensor_weighted_avg": torch.tensor([-1.0], device=DEVICE),
        },
    ]


class TensorWeightedAvgValueTest(unittest.TestCase):
    """This set of tests verify the computation logic of tensor weighted avg in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @torch.no_grad()
    def _test_tensor_weighted_avg_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        tensors: torch.Tensor,
        weights: Optional[torch.Tensor],
        expected_tensor_weighted_avg: torch.Tensor,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[1]
        task_list = []

        predictions_dict: Dict[str, torch.Tensor] = {}
        labels_dict: Dict[str, torch.Tensor] = {}
        weights_dict: Optional[Dict[str, torch.Tensor]] = (
            {} if weights is not None else None
        )
        required_inputs_dict: Dict[str, torch.Tensor] = {}

        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
                tensor_name="test_tensor",
                weighted=True,
            )
            task_list.append(task_info)
            predictions_dict[task_info.name] = predictions[i]
            labels_dict[task_info.name] = labels[i]

            # Ensure tensor_name is not None before using as dict key
            tensor_name = task_info.tensor_name
            if tensor_name is not None:
                required_inputs_dict[tensor_name] = tensors[i]

            if weights is not None and weights_dict is not None:
                weights_dict[task_info.name] = weights[i]

        inputs: Dict[str, Union[Dict[str, torch.Tensor], None]] = {
            "predictions": predictions_dict,
            "labels": labels_dict,
            "weights": weights_dict,
            "required_inputs": required_inputs_dict,
        }
        tensor_weighted_avg = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        ).to(DEVICE)

        tensor_weighted_avg.update(**inputs)
        actual_tensor_weighted_avg = tensor_weighted_avg.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_tensor_weighted_avg = actual_tensor_weighted_avg[
                f"weighted_avg-{task.name}|window_weighted_avg"
            ]
            cur_expected_tensor_weighted_avg = expected_tensor_weighted_avg[
                task_id
            ].unsqueeze(dim=0)

            if cur_expected_tensor_weighted_avg.isnan().any():
                self.assertTrue(cur_actual_tensor_weighted_avg.isnan().any())
            else:
                torch.testing.assert_close(
                    cur_actual_tensor_weighted_avg,
                    cur_expected_tensor_weighted_avg,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    msg=f"Actual: {cur_actual_tensor_weighted_avg}, Expected: {cur_expected_tensor_weighted_avg}",
                )

    def test_tensor_weighted_avg_correctness(self) -> None:
        """Test tensor weighted average computation correctness with known values."""
        test_data = generate_tensor_model_outputs_cases()
        for inputs in test_data:
            try:
                # Extract and validate inputs
                labels = inputs["labels"]
                predictions = inputs["predictions"]
                tensors = inputs["tensors"]
                weights = inputs["weights"] if "weights" in inputs else None
                expected = inputs["expected_tensor_weighted_avg"]

                # Call helper with properly typed arguments
                self._test_tensor_weighted_avg_helper(
                    labels=labels,
                    predictions=predictions,
                    tensors=tensors,
                    weights=weights,
                    expected_tensor_weighted_avg=expected,
                )
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise

    def test_tensor_weighted_vs_unweighted_computation(self) -> None:
        """Test that weighted and unweighted computations produce different results when weights vary."""
        # Test data with non-uniform weights
        labels = torch.tensor([[1, 0, 1, 0]]).to(DEVICE)
        predictions = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(DEVICE)
        required_inputs_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).to(DEVICE)
        varying_weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).to(DEVICE)

        # Weighted: (1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4) / (0.1+0.2+0.3+0.4) = 3.0/1.0 = 3.0
        expected_weighted_avg = torch.tensor([3.0]).to(DEVICE)
        # Unweighted: (1.0 + 2.0 + 3.0 + 4.0) / 4 = 10.0/4 = 2.5
        expected_unweighted_avg = torch.tensor([2.5]).to(DEVICE)

        # Create weighted task
        weighted_task = RecTaskInfo(
            name="weighted_task",
            label_name="label",
            prediction_name="prediction",
            weight_name="weight",
            tensor_name="test_tensor",
            weighted=True,
        )

        # Create unweighted task
        unweighted_task = RecTaskInfo(
            name="unweighted_task",
            label_name="label",
            prediction_name="prediction",
            weight_name="weight",
            tensor_name="test_tensor",
            weighted=False,
        )
        # Test weighted computation
        weighted_metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=[weighted_task],
        ).to(DEVICE)

        weighted_metric.update(
            predictions={"weighted_task": predictions[0]},
            labels={"weighted_task": labels[0]},
            weights={"weighted_task": varying_weights[0]},
            required_inputs={"test_tensor": required_inputs_tensor[0]},
        )

        weighted_result = weighted_metric.compute()

        # Test unweighted computation
        unweighted_metric = TensorWeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=[unweighted_task],
        ).to(DEVICE)

        unweighted_metric.update(
            predictions={"unweighted_task": predictions[0]},
            labels={"unweighted_task": labels[0]},
            weights={"unweighted_task": varying_weights[0]},  # ignored
            required_inputs={"test_tensor": required_inputs_tensor[0]},
        )

        unweighted_result = unweighted_metric.compute()

        # Results should be different
        weighted_value = weighted_result[
            "weighted_avg-weighted_task|window_weighted_avg"
        ]
        unweighted_value = unweighted_result[
            "weighted_avg-unweighted_task|window_weighted_avg"
        ]

        torch.testing.assert_close(
            weighted_value,
            expected_weighted_avg,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            msg=f"Actual: {weighted_value}, Expected: {expected_weighted_avg}",
        )

        torch.testing.assert_close(
            unweighted_value,
            expected_unweighted_avg,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            msg=f"Actual: {unweighted_value}, Expected: {expected_unweighted_avg}",
        )
