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
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.mse import compute_mse, compute_rmse
from torchrec.metrics.nmse import compute_norm, get_norm_mse_states, NMSEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


WORLD_SIZE = 4


class TestNMSEMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return get_norm_mse_states(labels, predictions, weights)

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        mse = compute_mse(states["error_sum"], states["weighted_num_samples"])
        const_pred_mse = compute_mse(
            states["const_pred_error_sum"], states["weighted_num_samples"]
        )
        return compute_norm(mse, const_pred_mse)


class TestNRMSEMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return get_norm_mse_states(labels, predictions, weights)

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        rmse = compute_rmse(states["error_sum"], states["weighted_num_samples"])
        const_pred_rmse = compute_rmse(
            states["const_pred_error_sum"], states["weighted_num_samples"]
        )
        return compute_norm(rmse, const_pred_rmse)


class NMSEMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = NMSEMetric
    nmse_task_name: str = "nmse"
    nrmse_task_name: str = "nrmse"

    def test_nmse_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNMSEMetric,
            metric_name=NMSEMetricTest.nmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_nmse_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestNMSEMetric,
            metric_name=NMSEMetricTest.nmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_nmse_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestNMSEMetric,
            metric_name=NMSEMetricTest.nmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_nrmse_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNRMSEMetric,
            metric_name=NMSEMetricTest.nrmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_nrmse_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestNRMSEMetric,
            metric_name=NMSEMetricTest.nrmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_nrmse_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestNRMSEMetric,
            metric_name=NMSEMetricTest.nrmse_task_name,
            task_names=["t1", "t2"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class NMSEGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = NMSEMetric
    task_name: str = "nmse"

    def test_sync_nmse(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=NMSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNMSEMetric,
            metric_name=NMSEGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, Union[float, torch.Tensor]]]:
    return [
        # Perfect predictions - NMSE should be 0
        {
            "labels": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "predictions": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "weights": torch.tensor([[1.0] * 5]),
            "expected_nmse": torch.tensor([0.0]),
        },
        # Constant predictor (all 1.0) - NMSE should be 1.0
        {
            "labels": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            "predictions": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
            "weights": torch.tensor([[1.0] * 5]),
            "expected_nmse": torch.tensor([1.0]),
        },
        # Better than constant predictor
        {
            "labels": torch.tensor([[1.0, 2.0, 3.0]]),
            "predictions": torch.tensor([[1.5, 2.0, 2.5]]),
            "weights": torch.tensor([[1.0, 1.0, 1.0]]),
            "expected_nmse": torch.tensor([0.1]),
        },
        # With non-uniform weights
        {
            "labels": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "predictions": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "weights": torch.tensor([[0.5, 1.0, 1.5, 2.0]]),
            "expected_nmse": torch.tensor([0.0]),
        },
    ]


class NMSEMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of NMSE in several
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
        self.nmse = NMSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_nmse_perfect(self) -> None:
        """Test NMSE when predictions are perfect (NMSE should be 0)"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.labels["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0] * 5])

        expected_nmse = torch.tensor([0.0], dtype=torch.double)
        self.nmse.update(**self.batches)
        actual_nmse = self.nmse.compute()["nmse-DefaultTask|window_nmse"]
        self.assertTrue(torch.allclose(expected_nmse, actual_nmse, atol=1e-6))

    def test_calc_nmse_constant_predictor(self) -> None:
        """Test NMSE when predictions are all constant (NMSE should be 1.0)"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
        self.labels["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0] * 5])

        expected_nmse = torch.tensor([1.0], dtype=torch.double)
        self.nmse.update(**self.batches)
        actual_nmse = self.nmse.compute()["nmse-DefaultTask|window_nmse"]
        self.assertTrue(torch.allclose(expected_nmse, actual_nmse, atol=1e-6))

    def test_calc_nmse_better_than_baseline(self) -> None:
        """Test NMSE when predictions are better than baseline (NMSE should be < 1.0)"""
        self.predictions["DefaultTask"] = torch.Tensor([[1.5, 2.0, 2.5]])
        self.labels["DefaultTask"] = torch.Tensor([[1.0, 2.0, 3.0]])
        self.weights["DefaultTask"] = torch.Tensor([[1.0, 1.0, 1.0]])

        # Model MSE = ((1.5-1)^2 + (2-2)^2 + (2.5-3)^2) / 3 = (0.25 + 0 + 0.25) / 3 = 0.5/3
        # Baseline MSE = ((1-1)^2 + (1-2)^2 + (1-3)^2) / 3 = (0 + 1 + 4) / 3 = 5/3
        # NMSE = (0.5/3) / (5/3) = 0.5/5 = 0.1
        expected_nmse = torch.tensor([0.1], dtype=torch.double)
        self.nmse.update(**self.batches)
        actual_nmse = self.nmse.compute()["nmse-DefaultTask|window_nmse"]
        self.assertTrue(torch.allclose(expected_nmse, actual_nmse, atol=1e-6))


class NMSEThresholdValueTest(unittest.TestCase):
    """This set of tests verify the computation logic of NMSE with various scenarios."""

    @no_grad()
    def _test_nmse_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_nmse: torch.Tensor,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[0]
        task_list = []
        predictions_dict: Dict[str, torch.Tensor] = {}
        labels_dict: Dict[str, torch.Tensor] = {}
        weights_dict: Dict[str, torch.Tensor] = {}

        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
            task_list.append(task_info)
            predictions_dict[task_info.name] = predictions[i]
            labels_dict[task_info.name] = labels[i]
            weights_dict[task_info.name] = weights[i]

        nmse = NMSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        )
        nmse.update(
            predictions=predictions_dict,
            labels=labels_dict,
            weights=weights_dict,
        )
        actual_nmse = nmse.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_nmse = actual_nmse[f"nmse-{task.name}|window_nmse"]
            cur_expected_nmse = expected_nmse[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_nmse,
                cur_expected_nmse,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                msg=f"Actual: {cur_actual_nmse}, Expected: {cur_expected_nmse}",
            )

    def test_nmse_values(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                # pyrefly: ignore[bad-argument-type]
                self._test_nmse_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise
