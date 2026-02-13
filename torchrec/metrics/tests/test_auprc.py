#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Iterable, List, Optional, Type, Union

import torch
from torch import no_grad
from torchrec.metrics.auprc import _compute_auprc_helper, AUPRCMetric
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
    RecMetricException,
    RecTaskInfo,
)
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


def compute_auprc(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return _compute_auprc_helper(predictions, labels, weights)


class TestAUPRCMetric(TestMetric):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
    ) -> None:
        super().__init__(
            world_size,
            rec_tasks,
            compute_lifetime_metric=False,
            local_compute_lifetime_metric=False,
        )

    @staticmethod
    def _aggregate(
        states: Dict[str, torch.Tensor], new_states: Dict[str, torch.Tensor]
    ) -> None:
        for k, v in new_states.items():
            if k not in states:
                states[k] = v.float().detach().clone()
            else:
                states[k] = torch.cat([states[k], v.float()])

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return {
            "predictions": predictions,
            "weights": weights,
            "labels": labels,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_auprc(states["predictions"], states["labels"], states["weights"])


WORLD_SIZE = 4


class AUPRCMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = AUPRCMetric
    task_name: str = "auprc"

    def test_unfused_auprc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUPRCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAUPRCMetric,
            metric_name=AUPRCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_auprc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUPRCMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestAUPRCMetric,
            metric_name=AUPRCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class AUPRCMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of AUPRC in several
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
        self.auprc = AUPRCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_auprc_perfect(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[1] * 5000 + [0] * 10000 + [1] * 5000]
        )

        expected_auprc = torch.tensor([1], dtype=torch.float)
        self.auprc.update(**self.batches)
        actual_auprc = self.auprc.compute()["auprc-DefaultTask|window_auprc"]
        torch.allclose(expected_auprc, actual_auprc)

    def test_calc_auprc_zero(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[0] * 5000 + [1] * 10000 + [0] * 5000]
        )

        expected_auprc = torch.tensor([0.3069], dtype=torch.float)
        self.auprc.update(**self.batches)
        actual_auprc = self.auprc.compute()["auprc-DefaultTask|window_auprc"]
        torch.allclose(expected_auprc, actual_auprc)

    def test_calc_auprc_balanced(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.ones([1, 20000])

        expected_auprc = torch.tensor([0.5], dtype=torch.float)
        self.auprc.update(**self.batches)
        actual_auprc = self.auprc.compute()["auprc-DefaultTask|window_auprc"]
        torch.allclose(expected_auprc, actual_auprc)


def generate_model_outputs_cases() -> Iterable[Dict[str, torch._tensor.Tensor]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_auprc": torch.tensor([0.5737]),
        },
        # perfect_condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[1, 0, 0, 1, 1]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 1, 0, 0, 1]),
            "expected_auprc": torch.tensor([1.0]),
        },
        # inverse_prediction
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0, 1, 1, 0, 0]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_auprc": torch.tensor([0.5833]),
        },
        # all_scores_the_same
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": torch.tensor([[1] * 6]),
            "grouping_keys": torch.tensor([1, 1, 1, 0, 0, 0]),
            "expected_auprc": torch.tensor([0.5]),
        },
        # one_class_in_input
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 0, 0, 1, 0]),
            "expected_auprc": torch.tensor([1.0]),
        },
        # one_group
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([1, 1, 1, 1, 1]),
            "expected_auprc": torch.tensor([0.8291]),
        },
        # two tasks
        {
            "labels": torch.tensor([[1, 0, 0, 1, 0], [1, 1, 1, 1, 0]]),
            "predictions": torch.tensor(
                [
                    [0.2281, 0.1051, 0.4885, 0.7740, 0.3097],
                    [0.4658, 0.3445, 0.6048, 0.6587, 0.5088],
                ]
            ),
            "weights": torch.tensor(
                [
                    [0.6334, 0.6937, 0.6631, 0.5078, 0.3570],
                    [0.2637, 0.2479, 0.2697, 0.6500, 0.7583],
                ]
            ),
            "grouping_keys": torch.tensor([0, 1, 0, 0, 1]),
            "expected_auprc": torch.tensor([0.3980, 0.6232]),
        },
    ]


class GroupedAUPRCValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of AUPRC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_grouped_auprc_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_auprc: torch.Tensor,
        grouping_keys: Optional[torch.Tensor] = None,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[0]
        task_list = []
        inputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {
            "predictions": {},
            "labels": {},
            "weights": {},
        }
        if grouping_keys is not None:
            inputs["required_inputs"] = {"grouping_keys": grouping_keys}
        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
            task_list.append(task_info)
            # pyrefly: ignore[unsupported-operation]
            inputs["predictions"][task_info.name] = predictions[i]
            # pyrefly: ignore[unsupported-operation]
            inputs["labels"][task_info.name] = labels[i]
            # pyrefly: ignore[unsupported-operation]
            inputs["weights"][task_info.name] = weights[i]

        auprc = AUPRCMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            grouped_auprc=True,
        )
        auprc.update(**inputs)
        actual_auprc = auprc.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_auprc = actual_auprc[f"auprc-{task.name}|window_grouped_auprc"]
            cur_expected_auprc = expected_auprc[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_auprc,
                cur_expected_auprc,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                msg=f"Actual: {cur_actual_auprc}, Expected: {cur_expected_auprc}",
            )

    def test_grouped_auprc(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_grouped_auprc_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise

    def test_misconfigured_grouped_auprc(self) -> None:
        with self.assertRaises(RecMetricException):
            self._test_grouped_auprc_helper(
                **{
                    "labels": torch.tensor([[1, 0, 0, 1, 1]]),
                    "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
                    "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
                    # no provided grouping_keys
                    "expected_auprc": torch.tensor([0.8291]),
                },
            )

    def test_required_input_for_grouped_auprc(self) -> None:
        auprc = AUPRCMetric(
            world_size=1,
            my_rank=0,
            batch_size=1,
            tasks=[
                RecTaskInfo(
                    name="Task:0",
                    label_name="label",
                    prediction_name="prediction",
                    weight_name="weight",
                )
            ],
            # pyrefly: ignore[bad-argument-type]
            grouped_auprc=True,
        )

        self.assertIn("grouping_keys", auprc.get_required_inputs())


class AUPRCGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = AUPRCMetric
    task_name: str = "auprc"

    def test_sync_auprc(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=AUPRCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAUPRCMetric,
            metric_name=AUPRCGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
