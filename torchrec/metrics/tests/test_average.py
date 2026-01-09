#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Optional, Type

import torch
from torchrec.metrics.average import AverageMetric, compute_average, get_average_states
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


class TestLabelAverageMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return get_average_states(labels, predictions, weights)

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_average(states["label_sum"], states["weighted_num_samples"])


class TestPredictionAverageMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return get_average_states(labels, predictions, weights)

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_average(states["prediction_sum"], states["weighted_num_samples"])


WORLD_SIZE = 4


class AverageMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = AverageMetric
    label_average_task_name: str = "label_average"
    prediction_average_task_name: str = "prediction_average"

    def test_label_average_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestLabelAverageMetric,
            metric_name=AverageMetricTest.label_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_label_average_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestLabelAverageMetric,
            metric_name=AverageMetricTest.label_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_label_average_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestLabelAverageMetric,
            metric_name=AverageMetricTest.label_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_prediction_average_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestPredictionAverageMetric,
            metric_name=AverageMetricTest.prediction_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_prediction_average_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestPredictionAverageMetric,
            metric_name=AverageMetricTest.prediction_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_prediction_average_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestPredictionAverageMetric,
            metric_name=AverageMetricTest.prediction_average_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class AverageGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = AverageMetric
    task_name: str = "label_average"

    def test_sync_label_average(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=AverageMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestLabelAverageMetric,
            metric_name=AverageGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


class AverageMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of Average metrics in several
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
        self.average = AverageMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_label_average(self) -> None:
        """Test label average computation"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.labels["DefaultTask"] = torch.Tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0] * 5])

        # Label average = (2 + 4 + 6 + 8 + 10) / 5 = 30 / 5 = 6.0
        expected_avg = torch.tensor([6.0], dtype=torch.double)
        self.average.update(**self.batches)
        actual_avg = self.average.compute()["average-DefaultTask|window_label_average"]
        self.assertTrue(torch.allclose(expected_avg, actual_avg, atol=1e-6))

    def test_calc_prediction_average(self) -> None:
        """Test prediction average computation"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.labels["DefaultTask"] = torch.Tensor([[2.0, 4.0, 6.0, 8.0, 10.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0] * 5])

        # Prediction average = (1 + 2 + 3 + 4 + 5) / 5 = 15 / 5 = 3.0
        expected_avg = torch.tensor([3.0], dtype=torch.double)
        self.average.update(**self.batches)
        actual_avg = self.average.compute()[
            "average-DefaultTask|window_prediction_average"
        ]
        self.assertTrue(torch.allclose(expected_avg, actual_avg, atol=1e-6))

    def test_calc_averages_with_weights(self) -> None:
        """Test label and prediction averages with non-uniform weights"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0]])
        self.labels["DefaultTask"] = torch.Tensor([[10.0, 20.0, 30.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0]])

        # Weighted label average = (10*1 + 20*2 + 30*3) / (1 + 2 + 3) = 140/6
        # Weighted prediction average = (1*1 + 2*2 + 3*3) / (1 + 2 + 3) = 14/6
        expected_label_avg = torch.tensor([140.0 / 6.0], dtype=torch.double)
        expected_pred_avg = torch.tensor([14.0 / 6.0], dtype=torch.double)
        self.average.update(**self.batches)
        actual_label_avg = self.average.compute()[
            "average-DefaultTask|window_label_average"
        ]
        actual_pred_avg = self.average.compute()[
            "average-DefaultTask|window_prediction_average"
        ]
        self.assertTrue(torch.allclose(expected_label_avg, actual_label_avg, atol=1e-6))
        self.assertTrue(torch.allclose(expected_pred_avg, actual_pred_avg, atol=1e-6))

    def test_calc_averages_zero_weights(self) -> None:
        """Test that zero weights return 0 average"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0]])
        self.labels["DefaultTask"] = torch.Tensor([[10.0, 20.0, 30.0]])
        self.weights["DefaultTask"] = torch.Tensor([[0.0, 0.0, 0.0]])

        expected_avg = torch.tensor([0.0], dtype=torch.double)
        self.average.update(**self.batches)
        actual_label_avg = self.average.compute()[
            "average-DefaultTask|window_label_average"
        ]
        actual_pred_avg = self.average.compute()[
            "average-DefaultTask|window_prediction_average"
        ]
        self.assertTrue(torch.allclose(expected_avg, actual_label_avg, atol=1e-6))
        self.assertTrue(torch.allclose(expected_avg, actual_pred_avg, atol=1e-6))
