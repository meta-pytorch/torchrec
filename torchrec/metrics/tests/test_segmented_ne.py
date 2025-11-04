#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Iterable, Union

import torch
from torch import no_grad
from torchrec.metrics.metrics_config import RecComputeMode
from torchrec.metrics.rec_metric import RecTaskInfo
from torchrec.metrics.segmented_ne import SegmentedNEMetric


class SegementedNEValueTest(unittest.TestCase):
    """
    This set of tests verify the computation logic of AUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_segemented_ne_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_ne: torch.Tensor,
        grouping_keys: torch.Tensor,
        grouping_key_tensor_name: str = "grouping_keys",
        cast_keys_to_int: bool = False,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
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
            inputs["required_inputs"] = {grouping_key_tensor_name: grouping_keys}
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
            # pyre-ignore
            inputs["weights"][task_info.name] = weights[i]

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            # pyre-ignore
            num_groups=max(2, torch.unique(grouping_keys)[-1].item() + 1),
            # pyre-ignore
            grouping_keys=grouping_key_tensor_name,
            # pyre-ignore
            cast_keys_to_int=cast_keys_to_int,
            compute_mode=compute_mode,
        )
        ne.update(**inputs)
        actual_ne = ne.compute()

        for task_id, task in enumerate(task_list):
            for label in [0, 1]:
                cur_actual_ne = actual_ne[
                    f"segmented_ne-{task.name}|lifetime_segmented_ne_{label}"
                ]
                cur_expected_ne = expected_ne[task_id][label]

                torch.testing.assert_close(
                    cur_actual_ne,
                    cur_expected_ne,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    equal_nan=True,
                    msg=f"Actual: {cur_actual_ne}, Expected: {cur_expected_ne}",
                )

    def test_grouped_ne(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in UNFUSED_TASKS_COMPUTATION mode",
                    inputs,
                )
                raise

            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in FUSED_TASKS_COMPUTATION mode",
                    inputs,
                )
                raise

            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in FUSED_TASKS_AND_STATES_COMPUTATION mode",
                    inputs,
                )
                raise


def generate_model_outputs_cases() -> Iterable[Dict[str, Any]]:
    return [
        # base condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
        },
        # one sided, edge case 1s
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([1, 1, 1, 1, 1]),
            "expected_ne": torch.tensor([[torch.nan, 1.3936]]),
        },
        # one sided, edge case 0s
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 0, 0, 0, 0]),
            "expected_ne": torch.tensor([[1.3936, torch.nan]]),
        },
        # three labels,
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1, 0]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9, 0.4]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75, 0.4]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 2, 2]),
            "expected_ne": torch.tensor([[3.1615, 1.8311, 0.3814]]),
        },
        # two tasks
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1], [1, 0, 0, 1, 1]]),
            "predictions": torch.tensor(
                [
                    [0.2, 0.6, 0.8, 0.4, 0.9],
                    [0.6, 0.2, 0.4, 0.8, 0.9],
                ]
            ),
            "weights": torch.tensor(
                [
                    [0.13, 0.2, 0.5, 0.8, 0.75],
                    [0.13, 0.2, 0.5, 0.8, 0.75],
                ]
            ),
            "grouping_keys": torch.tensor(
                [0, 1, 0, 1, 1]
            ),  # for this case, both tasks have same groupings
            "expected_ne": torch.tensor([[3.1615, 1.6004], [1.0034, 0.4859]]),
        },
        # Custom grouping key tensor name
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
            "grouping_key_tensor_name": "custom_key",
        },
        # Cast grouping keys to int32
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
            "grouping_key_tensor_name": "custom_key",
            "cast_keys_to_int": True,
        },
    ]
