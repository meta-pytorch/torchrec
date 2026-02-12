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
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rauc import RAUCMetric
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


def compute_rauc(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    n = len(predictions)
    cnt = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if (labels[i] - labels[j]) * (predictions[i] - predictions[j]) >= 0:
                cnt += 1

    return torch.tensor(cnt / (n * (n - 1) / 2))


class TestRAUCMetric(TestMetric):
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
        return compute_rauc(states["predictions"], states["labels"], states["weights"])


WORLD_SIZE = 4


class RAUCMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = RAUCMetric
    task_name: str = "rauc"

    def test_unfused_rauc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=RAUCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestRAUCMetric,
            metric_name=RAUCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_rauc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=RAUCMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestRAUCMetric,
            metric_name=RAUCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class RAUCGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = RAUCMetric
    task_name: str = "rauc"

    def test_sync_rauc(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=RAUCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestRAUCMetric,
            metric_name=RAUCGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


class RAUCMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of RAUC in several
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
        self.rauc = RAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_rauc_perfect(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(100)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(100)] * 2]
        )
        self.weights["DefaultTask"] = torch.Tensor([[1] * 50 + [0] * 100 + [1] * 50])

        expected_rauc = torch.tensor([1], dtype=torch.float)
        self.rauc.update(**self.batches)
        actual_rauc = self.rauc.compute()["rauc-DefaultTask|window_rauc"]
        assert torch.allclose(expected_rauc, actual_rauc)

    def test_calc_rauc_zero(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(100)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor(
            [[-0.0001 * x for x in range(100)] * 2]
        )
        self.weights["DefaultTask"] = torch.Tensor([[0] * 50 + [1] * 100 + [0] * 50])

        expected_rauc = torch.tensor([0], dtype=torch.float)
        self.rauc.update(**self.batches)
        actual_rauc = self.rauc.compute()["rauc-DefaultTask|window_rauc"]
        assert torch.allclose(expected_rauc, actual_rauc)

    def test_calc_rauc_random(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor([[1, 2, 3, 4]])
        self.labels["DefaultTask"] = torch.Tensor([[2, 1, 4, 3]])
        self.weights["DefaultTask"] = torch.Tensor([[1, 1, 1, 1]])

        expected_rauc = torch.tensor([2.0 / 3], dtype=torch.float)
        self.rauc.update(**self.batches)
        actual_rauc = self.rauc.compute()["rauc-DefaultTask|window_rauc"]
        assert torch.allclose(expected_rauc, actual_rauc)

    def test_window_size_rauc(self) -> None:
        # for determinisitc batches
        torch.manual_seed(0)

        rauc = RAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            window_size=100,
            tasks=[DefaultTaskInfo],
        )

        # init states, so we expect 3 (state tensors) * 4 bytes (float)
        self.assertEqual(sum(rauc.get_memory_usage().values()), 12)

        # bs = 5
        self.labels["DefaultTask"] = torch.rand(5)
        self.predictions["DefaultTask"] = torch.rand(5)
        self.weights["DefaultTask"] = torch.rand(5)

        for _ in range(1000):
            rauc.update(**self.batches)

        # check memory, window size is 100, so we have upperbound of memory to expect
        # so with a 100 window size / tensors of size 5 = 20 tensors (per state) * 3 states * 20 bytes per tensor of size 5 = 1200 bytes
        self.assertEqual(sum(rauc.get_memory_usage().values()), 1200)
        # with bs 5, we expect 20 tensors per state, so 60 tensors
        self.assertEqual(len(rauc.get_memory_usage().values()), 60)

        assert torch.allclose(
            rauc.compute()["rauc-DefaultTask|window_rauc"],
            torch.tensor([0.5152], dtype=torch.float),
            atol=1e-4,
        )

        # test rauc memory usage with window size equal to incoming batch
        rauc = RAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            window_size=100,
            tasks=[DefaultTaskInfo],
        )

        self.labels["DefaultTask"] = torch.rand(100)
        self.predictions["DefaultTask"] = torch.rand(100)
        self.weights["DefaultTask"] = torch.rand(100)

        for _ in range(10):
            rauc.update(**self.batches)

        # passing in batch size == window size, we expect for each state just one tensor of size 400, sum to 1200 as previous
        self.assertEqual(sum(rauc.get_memory_usage().values()), 1200)
        self.assertEqual(len(rauc.get_memory_usage().values()), 3)

        assert torch.allclose(
            rauc.compute()["rauc-DefaultTask|window_rauc"],
            torch.tensor([0.5508], dtype=torch.float),
            atol=1e-4,
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, torch._tensor.Tensor]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_rauc": torch.tensor([0.3333]),
        },
        # perfect_condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[1, 0, 0, 1, 1]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 1, 0, 0, 1]),
            "expected_rauc": torch.tensor([1.0]),
        },
        # inverse_prediction
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0, 1, 1, 0, 0]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_rauc": torch.tensor([0.1667]),
        },
        # all_scores_the_same
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": torch.tensor([[1] * 6]),
            "grouping_keys": torch.tensor([1, 1, 1, 0, 0, 0]),
            "expected_rauc": torch.tensor([1.0]),
        },
        # one_class_in_input
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 0, 0, 1, 0]),
            "expected_rauc": torch.tensor([1.0]),
        },
        # one_group
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([1, 1, 1, 1, 1]),
            "expected_rauc": torch.tensor([0.6]),
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
            "expected_rauc": torch.tensor([0.8333, 0.5]),
        },
    ]


class GroupedRAUCValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of RAUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_grouped_rauc_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_rauc: torch.Tensor,
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
            inputs["predictions"][task_info.name] = predictions[i]
            inputs["labels"][task_info.name] = labels[i]
            inputs["weights"][task_info.name] = weights[i]

        rauc = RAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            grouped_rauc=True,
        )
        rauc.update(**inputs)
        actual_rauc = rauc.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_rauc = actual_rauc[f"rauc-{task.name}|window_grouped_rauc"]
            cur_expected_rauc = expected_rauc[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_rauc,
                cur_expected_rauc,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                msg=f"Actual: {cur_actual_rauc}, Expected: {cur_expected_rauc}",
            )

    def test_grouped_rauc(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_grouped_rauc_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise

    def test_misconfigured_grouped_rauc(self) -> None:
        with self.assertRaises(RecMetricException):
            self._test_grouped_rauc_helper(
                **{
                    "labels": torch.tensor([[1, 0, 0, 1, 1]]),
                    "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
                    "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
                    # no provided grouping_keys
                    "expected_rauc": torch.tensor([0.2419]),
                },
            )

    def test_required_input_for_grouped_Rauc(self) -> None:
        rauc = RAUCMetric(
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
            grouped_rauc=True,
        )

        self.assertIn("grouping_keys", rauc.get_required_inputs())
