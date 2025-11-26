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
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.sum_weights import compute_weighted_sum, SumWeightsMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    RecTaskInfo,
    sync_test_helper,
    TestMetric,
)


WORLD_SIZE = 4


class TestSumWeightsMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        weighted_sum = torch.sum(weights, dim=-1)
        return {
            "weighted_sum": weighted_sum,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return states["weighted_sum"]


class SumWeightsMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = SumWeightsMetric
    task_name: str = "sum_weights"

    def test_precision_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=SumWeightsMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestSumWeightsMetric,
            metric_name=SumWeightsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_precision_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=SumWeightsMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestSumWeightsMetric,
            metric_name=SumWeightsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_precision_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=SumWeightsMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestSumWeightsMetric,
            metric_name=SumWeightsMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class SumWeightsGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = SumWeightsMetric
    task_name: str = "sum_weights"

    def test_sync_sum_weights(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=SumWeightsMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestSumWeightsMetric,
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
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.6]]),
            "expected_sum_weights": torch.tensor([1.0]),
        },
        # no weight
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": None,
            "expected_sum_weights": torch.tensor([6]),
        },
        # weights are 0.5
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": torch.tensor([[0.5] * 6]),
            "expected_sum_weights": torch.tensor([3]),
        },
        # all weights are zero
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0] * 5]),
            "expected_sum_weights": torch.tensor([0]),
        },
        # Multi tasks
        {
            "labels": torch.tensor([[[1, 1, 1, 1, 1]], [[1, 0, 0, 1, 1]]]),
            "predictions": torch.tensor(
                [[0.2, 0.6, 0.8, 0.4, 0.9], [0.2, 0.6, 0.8, 0.4, 0.9]]
            ),
            "weights": torch.tensor([[1] * 5, [2] * 5]),
            "expected_sum_weights": torch.tensor([5, 10]),
        },
    ]


class SumWeightsTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of num positive samples in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @torch.no_grad()
    def _test_sum_weights_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_sum_weights: torch.Tensor,
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

        sum_weights = SumWeightsMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        )
        sum_weights.update(**inputs)
        actual_sum_weights = sum_weights.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_sum_weights = actual_sum_weights[
                f"sum_weights-{task.name}|window_sum_weights"
            ][0]
            cur_expected_sum_weights = expected_sum_weights[task_id]
            if cur_expected_sum_weights.isnan().any():
                self.assertTrue(cur_actual_sum_weights.isnan().any())
            else:
                torch.testing.assert_close(
                    cur_actual_sum_weights,
                    cur_expected_sum_weights,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    msg=f"Actual: {cur_actual_sum_weights}, Expected: {cur_expected_sum_weights}",
                )

    def test_sum_weights(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                # pyre-ignore
                self._test_sum_weights_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise
