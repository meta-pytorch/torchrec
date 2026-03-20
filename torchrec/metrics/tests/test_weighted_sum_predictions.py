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
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    RecTaskInfo,
    sync_test_helper,
    TestMetric,
)
from torchrec.metrics.weighted_sum_predictions import WeightedSumPredictionsMetric


WORLD_SIZE = 4


class TestWeightedSumPredictionsMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weighted_predictions_sum = torch.sum(
            weights * torch.nan_to_num(predictions, 0), dim=-1
        )
        return {
            "weighted_predictions_sum": weighted_predictions_sum,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return states["weighted_predictions_sum"]


class WeightedSumPredictionsMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = WeightedSumPredictionsMetric
    task_name: str = "weighted_sum_predictions"

    def test_precision_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedSumPredictionsMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedSumPredictionsMetric,
            metric_name=WeightedSumPredictionsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_precision_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedSumPredictionsMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedSumPredictionsMetric,
            metric_name=WeightedSumPredictionsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_precision_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedSumPredictionsMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestWeightedSumPredictionsMetric,
            metric_name=WeightedSumPredictionsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class SumWeightsGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = WeightedSumPredictionsMetric
    task_name: str = "weighted_sum_predictions"

    def test_sync_weighted_predictions_sum(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=WeightedSumPredictionsMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedSumPredictionsMetric,
            metric_name=SumWeightsGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, Optional[torch.Tensor]]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.5]]),
            "weights": torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]]),
            "expected_weighted_predictions_sum": torch.tensor([0.25]),
        },
        # no weight
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": None,
            "expected_weighted_predictions_sum": torch.tensor([3]),
        },
        # weights are 0.5
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": torch.tensor([[0.5] * 6]),
            "expected_weighted_predictions_sum": torch.tensor([1.5]),
        },
        # all weights are zero
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0] * 5]),
            "expected_weighted_predictions_sum": torch.tensor([0]),
        },
        # Multi tasks
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 1, 1]]),
            "predictions": torch.tensor(
                [[0.2, 0.6, 0.8, 0.4, 0.9], [0.2, 0.6, 0.8, 0.4, 0.9]]
            ),
            "weights": torch.tensor([[1] * 5, [2] * 5]),
            "expected_weighted_predictions_sum": torch.tensor([2.9, 5.8]),
        },
    ]


class WeightedSumPredictionsTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of weighted predictions sum in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @torch.no_grad()
    def _test_weighted_predictions_sum_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_weighted_predictions_sum: torch.Tensor,
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
            # pyre-ignore
            inputs["predictions"][task_info.name] = predictions[i]
            # pyre-ignore
            inputs["labels"][task_info.name] = labels[i]
            if weights is None:
                # pyre-ignore
                inputs["weights"] = None
            else:
                # pyre-ignore
                inputs["weights"][task_info.name] = weights[i]

        weighted_predictions_sum = WeightedSumPredictionsMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        )
        weighted_predictions_sum.update(**inputs)
        actual_weighted_predictions_sum = weighted_predictions_sum.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_weighted_predictions_sum = actual_weighted_predictions_sum[
                f"weighted_sum_predictions-{task.name}|window_weighted_sum_predictions"
            ][0]
            cur_expected_weighted_predictions_sum = expected_weighted_predictions_sum[
                task_id
            ]
            if cur_expected_weighted_predictions_sum.isnan().any():
                self.assertTrue(cur_actual_weighted_predictions_sum.isnan().any())
            else:
                torch.testing.assert_close(
                    cur_actual_weighted_predictions_sum,
                    cur_expected_weighted_predictions_sum,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    msg=f"Actual: {cur_actual_weighted_predictions_sum}, Expected: {cur_expected_weighted_predictions_sum}",
                )

    def test_weighted_predictions_sum(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                # pyre-ignore
                self._test_weighted_predictions_sum_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise
