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
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)
from torchrec.metrics.weighted_avg import get_mean, WeightedAvgMetric


WORLD_SIZE = 4


class TestWeightedAvgMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        return {
            "weighted_sum": (predictions * weights).sum(dim=-1),
            "weighted_num_samples": weights.sum(dim=-1),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_mean(states["weighted_sum"], states["weighted_num_samples"])


class WeightedAvgMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = WeightedAvgMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "weighted_avg"

    def test_weighted_avg_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_update_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, Optional[torch.Tensor]]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.6]]),
            "expected_weighted_avg": torch.tensor([0.74]),
        },
        # no weight
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": None,
            "expected_weighted_avg": torch.tensor([0.5]),
        },
        # all weights are zero
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0] * 5]),
            "expected_weighted_avg": torch.tensor([float("nan")]),
        },
    ]


class WeightedAvgValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of weighted avg in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @torch.no_grad()
    def _test_weighted_avg_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_weighted_avg: torch.Tensor,
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
            if weights is None:
                inputs["weights"] = None
            else:
                inputs["weights"][task_info.name] = weights[i]

        weighted_avg = WeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        )
        weighted_avg.update(**inputs)
        actual_weighted_avg = weighted_avg.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_weighted_avg = actual_weighted_avg[
                f"weighted_avg-{task.name}|window_weighted_avg"
            ]
            cur_expected_weighted_avg = expected_weighted_avg[task_id].unsqueeze(dim=0)
            if cur_expected_weighted_avg.isnan().any():
                self.assertTrue(cur_actual_weighted_avg.isnan().any())
            else:
                torch.testing.assert_close(
                    cur_actual_weighted_avg,
                    cur_expected_weighted_avg,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    msg=f"Actual: {cur_actual_weighted_avg}, Expected: {cur_expected_weighted_avg}",
                )

    def test_weighted_avg(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_weighted_avg_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise
