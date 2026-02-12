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
from torch import no_grad
from torchrec.metrics.accuracy import AccuracyMetric, compute_accuracy
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    RecTaskInfo,
    sync_test_helper,
    TestMetric,
)


WORLD_SIZE = 4


class TestAccuracyMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        accuracy_sum = torch.sum(weights * ((predictions >= 0.5) == labels))
        return {
            "accuracy_sum": accuracy_sum,
            "weighted_num_samples": torch.sum(weights),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_accuracy(
            states["accuracy_sum"],
            states["weighted_num_samples"],
        )


class AccuracyMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = AccuracyMetric
    task_name: str = "accuracy"

    def test_accuracy_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AccuracyMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAccuracyMetric,
            metric_name=AccuracyMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_accuracy_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AccuracyMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestAccuracyMetric,
            metric_name=AccuracyMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_accuracy_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AccuracyMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestAccuracyMetric,
            metric_name=AccuracyMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class AccuracyMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of accuracy in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    def setUp(self) -> None:
        self.predictions = {"DefaultTask": None}
        self.weights = {"DefaultTask": None}
        self.labels = {"DefaultTask": None}
        self.batches = {
            "predictions": self.predictions,
            "weights": self.weights,
            "labels": self.labels,
        }
        self.accuracy = AccuracyMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_acc_perfect(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[1] * 5000 + [0] * 10000 + [1] * 5000]
        )

        expected_accuracy = torch.tensor([1], dtype=torch.double)
        self.accuracy.update(**self.batches)
        actual_accuracy = self.accuracy.compute()[
            "accuracy-DefaultTask|window_accuracy"
        ]
        torch.allclose(expected_accuracy, actual_accuracy)

    def test_calc_acc_zero(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[0] * 5000 + [1] * 10000 + [0] * 5000]
        )

        expected_accuracy = torch.tensor([0], dtype=torch.double)
        self.accuracy.update(**self.batches)
        actual_accuracy = self.accuracy.compute()[
            "accuracy-DefaultTask|window_accuracy"
        ]
        torch.allclose(expected_accuracy, actual_accuracy)

    def test_calc_accuracy_balanced(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.ones([1, 20000])

        expected_accuracy = torch.tensor([0.5], dtype=torch.double)
        self.accuracy.update(**self.batches)
        actual_accuracy = self.accuracy.compute()[
            "accuracy-DefaultTask|window_accuracy"
        ]
        torch.allclose(expected_accuracy, actual_accuracy)


def generate_model_outputs_cases() -> Iterable[Dict[str, Union[float, torch.Tensor]]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.3, 0.2, 0.5, 0.8, 0.7]]),
            "threshold": 0.6,
            "expected_accuracy": torch.tensor([0.28]),
        },
        # perfect_condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[1, 0, 0, 1, 1]]),
            "weights": torch.tensor([[1] * 5]),
            "threshold": 0.6,
            "expected_accuracy": torch.tensor([1.0]),
        },
        # inverse_prediction
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0, 1, 1, 0, 0]]),
            "weights": torch.tensor([[1] * 5]),
            "threshold": 0.1,
            "expected_accuracy": torch.tensor([0.0]),
        },
    ]


class ThresholdValueTest(unittest.TestCase):
    """This set of tests verify the computation logic of accuracy with a modified threshold
    in several cases that we know the computation results.
    """

    @no_grad()
    def _test_accuracy_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_accuracy: torch.Tensor,
        threshold: float,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[0]
        task_list = []
        inputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {
            "predictions": {},
            "labels": {},
            "weights": {},
        }
        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
            task_list.append(task_info)
            inputs["predictions"][task_info.name] = predictions[i]
            inputs["labels"][task_info.name] = labels[i]
            inputs["weights"][task_info.name] = weights[i]

        accuracy = AccuracyMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            threshold=threshold,  # threshold is one of the kwargs
        )
        accuracy.update(**inputs)
        actual_accuracy = accuracy.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_accuracy = actual_accuracy[
                f"accuracy-{task.name}|window_accuracy"
            ]
            cur_expected_accuracy = expected_accuracy[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_accuracy,
                cur_expected_accuracy,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                msg=f"Actual: {cur_actual_accuracy}, Expected: {cur_expected_accuracy}",
            )

    def test_accuracy(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_accuracy_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise


class AccuracyGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = AccuracyMetric
    task_name: str = "accuracy"

    def test_sync_accuracy(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=AccuracyMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAccuracyMetric,
            metric_name=AccuracyGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
