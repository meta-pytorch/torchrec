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
from torchrec.metrics.auc import AUCMetric
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


def compute_auc(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    _, sorted_index = torch.sort(predictions, descending=True)
    sorted_labels = torch.index_select(labels, dim=0, index=sorted_index)
    sorted_weights = torch.index_select(weights, dim=0, index=sorted_index)
    cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
    cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
    auc = torch.where(
        cum_fp[-1] * cum_tp[-1] == 0,
        0.5,  # 0.5 is the no-signal default value for auc.
        torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
    )
    return auc


class TestAUCMetric(TestMetric):
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
        return compute_auc(states["predictions"], states["labels"], states["weights"])


WORLD_SIZE = 4


class AUCMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = AUCMetric
    task_name: str = "auc"

    def test_unfused_auc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAUCMetric,
            metric_name=AUCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_fused_auc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUCMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestAUCMetric,
            metric_name=AUCMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class AUCGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = AUCMetric
    task_name: str = "auc"

    def test_sync_auc(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=AUCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAUCMetric,
            metric_name=AUCGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


class AUCMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of AUC in several
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
        self.auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_auc_perfect(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[1] * 5000 + [0] * 10000 + [1] * 5000]
        )

        expected_auc = torch.tensor([1], dtype=torch.float)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)

    def test_calc_auc_zero(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[0] * 5000 + [1] * 10000 + [0] * 5000]
        )

        expected_auc = torch.tensor([0], dtype=torch.float)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)

    def test_calc_auc_balanced(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.ones([1, 20000])

        expected_auc = torch.tensor([0.5], dtype=torch.float)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)

    def test_calc_uneven_updates(self) -> None:
        auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=100,
            tasks=[DefaultTaskInfo],
        )

        expected_auc = torch.tensor([0.4464], dtype=torch.float)
        # first batch
        self.labels["DefaultTask"] = torch.tensor([1, 0, 0])
        self.predictions["DefaultTask"] = torch.tensor([0.2, 0.6, 0.8])
        self.weights["DefaultTask"] = torch.tensor([0.13, 0.2, 0.5])

        auc.update(**self.batches)
        # second batch
        self.labels["DefaultTask"] = torch.tensor([1, 1])
        self.predictions["DefaultTask"] = torch.tensor([0.4, 0.9])
        self.weights["DefaultTask"] = torch.tensor([0.8, 0.75])

        auc.update(**self.batches)
        multiple_batch = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, multiple_batch)

    def test_window_size_auc(self) -> None:
        # for determinisitc batches
        torch.manual_seed(0)

        auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            window_size=100,
            tasks=[DefaultTaskInfo],
        )

        # init states, so we expect 3 (state tensors) * 4 bytes (float)
        self.assertEqual(sum(auc.get_memory_usage().values()), 12)

        # bs = 5
        self.labels["DefaultTask"] = torch.rand(5)
        self.predictions["DefaultTask"] = torch.rand(5)
        self.weights["DefaultTask"] = torch.rand(5)

        for _ in range(1000):
            auc.update(**self.batches)

        # check memory, window size is 100, so we have upperbound of memory to expect
        # so with a 100 window size / tensors of size 5 = 20 tensors (per state) * 3 states * 20 bytes per tensor of size 5 = 1200 bytes
        self.assertEqual(sum(auc.get_memory_usage().values()), 1200)
        # with bs 5, we expect 20 tensors per state, so 60 tensors
        self.assertEqual(len(auc.get_memory_usage().values()), 60)

        torch.allclose(
            auc.compute()["auc-DefaultTask|window_auc"],
            torch.tensor([0.4859], dtype=torch.float),
        )

        # test auc memory usage with window size equal to incoming batch
        auc = AUCMetric(
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
            auc.update(**self.batches)

        # passing in batch size == window size, we expect for each state just one tensor of size 400, sum to 1200 as previous
        self.assertEqual(sum(auc.get_memory_usage().values()), 1200)
        self.assertEqual(len(auc.get_memory_usage().values()), 3)

        torch.allclose(
            auc.compute()["auc-DefaultTask|window_auc"],
            torch.tensor([0.4859], dtype=torch.float),
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, torch._tensor.Tensor]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_auc": torch.tensor([0.2419]),
        },
        # perfect_condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[1, 0, 0, 1, 1]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 1, 0, 0, 1]),
            "expected_auc": torch.tensor([1.0]),
        },
        # inverse_prediction
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0, 1, 1, 0, 0]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_auc": torch.tensor([0.0]),
        },
        # all_scores_the_same
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": torch.tensor([[1] * 6]),
            "grouping_keys": torch.tensor([1, 1, 1, 0, 0, 0]),
            "expected_auc": torch.tensor([0.5]),
        },
        # one_class_in_input
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[1] * 5]),
            "grouping_keys": torch.tensor([1, 0, 0, 1, 0]),
            "expected_auc": torch.tensor([0.5]),
        },
        # one_group
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([1, 1, 1, 1, 1]),
            "expected_auc": torch.tensor([0.4464]),
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
            "expected_auc": torch.tensor([0.4725, 0.25]),
        },
    ]


class GroupedAUCValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of AUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_grouped_auc_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_auc: torch.Tensor,
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

        auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            grouped_auc=True,
        )
        auc.update(**inputs)
        actual_auc = auc.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_auc = actual_auc[f"auc-{task.name}|window_grouped_auc"]
            cur_expected_auc = expected_auc[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_auc,
                cur_expected_auc,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                msg=f"Actual: {cur_actual_auc}, Expected: {cur_expected_auc}",
            )

    def test_grouped_auc(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_grouped_auc_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise

    def test_misconfigured_grouped_auc(self) -> None:
        with self.assertRaises(RecMetricException):
            self._test_grouped_auc_helper(
                **{
                    "labels": torch.tensor([[1, 0, 0, 1, 1]]),
                    "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
                    "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
                    # no provided grouping_keys
                    "expected_auc": torch.tensor([0.2419]),
                },
            )

    def test_required_input_for_grouped_auc(self) -> None:
        auc = AUCMetric(
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
            grouped_auc=True,
        )

        self.assertIn("grouping_keys", auc.get_required_inputs())
