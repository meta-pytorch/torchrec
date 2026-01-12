#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Optional, Union

import torch
from torch import no_grad
from torchrec.metrics.metrics_config import (
    RecComputeMode,
    RecTaskInfo,
    SessionMetricDef,
)
from torchrec.metrics.precision_session import PrecisionSessionMetric
from torchrec.metrics.rec_metric import RecMetricException


def generate_model_output_test1() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor(
            [[1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8]]
        ),
        "session": torch.tensor([[1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]]),
        "labels": torch.tensor(
            [[0.9, 0.1, 0.2, 0.3, 0.9, 0.9, 0.0, 0.9, 0.1, 0.4, 0.9, 0.1]]
        ),
        "weights": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        "expected_precision": torch.tensor([0.5]),
    }


def generate_model_output_test2() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor(
            [[1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8]]
        ),
        "session": torch.tensor([[1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]]),
        "labels": torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
        ),
        "weights": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        "expected_precision": torch.tensor([0.5]),
    }


def generate_model_output_with_no_positive_examples() -> (
    Dict[str, torch._tensor.Tensor]
):
    return {
        "predictions": torch.tensor(
            [[1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8, 1.0, 0.0, 0.51, 0.8]]
        ),
        "session": torch.tensor([[1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]]),
        "labels": torch.tensor([[0.0] * 12]),
        "weights": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        "expected_precision": torch.tensor([0.0]),
    }


def generate_model_output_with_no_positive_predictions() -> (
    Dict[str, torch._tensor.Tensor]
):
    return {
        "predictions": torch.tensor([[float("nan")] * 12]),
        "session": torch.tensor([[1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]]),
        "labels": torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
        ),
        "weights": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        ),
        "expected_precision": torch.tensor([float("nan")]),
    }


class PrecisionSessionValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of Precision in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_precision_session_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        session: torch.Tensor,
        expected_precision: torch.Tensor,
        run_ranking_of_labels: bool = False,
        precision_metric: Optional[PrecisionSessionMetric] = None,
    ) -> PrecisionSessionMetric:
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
                session_metric_def=SessionMetricDef(
                    session_var_name="session",
                    top_threshold=3,
                    run_ranking_of_labels=run_ranking_of_labels,
                ),
            )
            task_list.append(task_info)
            # pyre-ignore
            inputs["predictions"][task_info.name] = predictions[i]
            # pyre-ignore
            inputs["labels"][task_info.name] = labels[i]
            # pyre-ignore
            inputs["weights"][task_info.name] = weights[i]

        kwargs = {"required_inputs": {"session": session}}

        if precision_metric is None:
            precision_metric = PrecisionSessionMetric(
                world_size=1,
                my_rank=0,
                batch_size=batch_size,
                tasks=task_list,
            )
        precision_metric.update(
            predictions=inputs["predictions"],
            labels=inputs["labels"],
            weights=inputs["weights"],
            **kwargs,
        )
        actual_precision = precision_metric.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_precision = actual_precision[
                f"precision_session_level-{task.name}|lifetime_precision_session_level"
            ]
            cur_expected_precision = expected_precision[task_id].unsqueeze(dim=0)

            torch.testing.assert_close(
                cur_actual_precision,
                cur_expected_precision,
                atol=1e-4,
                rtol=1e-4,
                check_dtype=False,
                equal_nan=True,
                msg=f"Actual: {cur_actual_precision}, Expected: {cur_expected_precision}",
            )
        return precision_metric

    def test_precision_session_with_ranked_labels(self) -> None:
        test_data = generate_model_output_test1()
        try:
            self._test_precision_session_helper(
                run_ranking_of_labels=True, precision_metric=None, **test_data
            )
        except AssertionError:
            print("Assertion error caught with data set ", test_data)
            raise

    def test_precision_session_with_bool_labels(self) -> None:
        test_data = generate_model_output_test2()
        try:
            self._test_precision_session_helper(
                run_ranking_of_labels=False, precision_metric=None, **test_data
            )
        except AssertionError:
            print("Assertion error caught with data set ", test_data)
            raise

    def test_precision_session_with_no_positive_examples(self) -> None:
        test_data = generate_model_output_with_no_positive_examples()
        try:
            self._test_precision_session_helper(
                run_ranking_of_labels=False, precision_metric=None, **test_data
            )
        except AssertionError:
            print("Assertion error caught with data set ", test_data)
            raise

    def test_precision_session_with_no_positive_predictions(self) -> None:
        test_data = generate_model_output_with_no_positive_predictions()
        try:
            self._test_precision_session_helper(
                run_ranking_of_labels=False, precision_metric=None, **test_data
            )
        except AssertionError:
            print("Assertion error caught with data set ", test_data)
            raise

    def test_error_messages(self) -> None:
        task_info1 = RecTaskInfo(
            name="Task1",
            label_name="label1",
            prediction_name="prediction1",
            weight_name="weight1",
        )

        task_info2 = RecTaskInfo(
            name="Task2",
            label_name="label2",
            prediction_name="prediction2",
            weight_name="weight2",
            session_metric_def=SessionMetricDef(session_var_name="session"),
        )

        error_message1 = "Please, specify the session metric definition"
        with self.assertRaisesRegex(RecMetricException, error_message1):
            _ = PrecisionSessionMetric(
                world_size=1,
                my_rank=5,
                batch_size=100,
                tasks=[task_info1],
            )
        error_message2 = "Please, specify the top threshold"
        with self.assertRaisesRegex(RecMetricException, error_message2):
            _ = PrecisionSessionMetric(
                world_size=1,
                my_rank=5,
                batch_size=100,
                tasks=[task_info2],
            )

    def test_compute_mode_exception(self) -> None:
        task_info = RecTaskInfo(
            name="Task1",
            label_name="label1",
            prediction_name="prediction1",
            weight_name="weight1",
        )
        with self.assertRaisesRegex(
            RecMetricException,
            "Fused computation is not supported for precision session-level metrics",
        ):
            PrecisionSessionMetric(
                world_size=1,
                my_rank=0,
                batch_size=100,
                tasks=[task_info],
                compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            )

        with self.assertRaisesRegex(
            RecMetricException,
            "Fused computation is not supported for precision session-level metrics",
        ):
            PrecisionSessionMetric(
                world_size=1,
                my_rank=5,
                batch_size=100,
                tasks=[task_info],
                compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            )

    def test_tasks_input_propagation(self) -> None:
        task_info1 = RecTaskInfo(
            name="Task1",
            label_name="label1",
            prediction_name="prediction1",
            weight_name="weight1",
            session_metric_def=SessionMetricDef(
                session_var_name="session1",
                top_threshold=1,
                run_ranking_of_labels=True,
            ),
        )

        task_info2 = RecTaskInfo(
            name="Task2",
            label_name="label2",
            prediction_name="prediction2",
            weight_name="weight2",
            session_metric_def=SessionMetricDef(
                session_var_name="session2",
                top_threshold=2,
                run_ranking_of_labels=False,
            ),
        )

        precision_metric = PrecisionSessionMetric(
            world_size=1,
            my_rank=5,
            batch_size=100,
            tasks=[task_info1, task_info2],
        )

        # metrics checks
        self.assertSetEqual(
            precision_metric.get_required_inputs(), {"session1", "session2"}
        )
        self.assertTrue(len(precision_metric._tasks) == 2)
        self.assertTrue(precision_metric._tasks[0] == task_info1)
        self.assertTrue(precision_metric._tasks[1] == task_info2)

        # metrics_computations checks
        self.assertTrue(precision_metric._metrics_computations[0]._my_rank == 5)
        self.assertTrue(precision_metric._metrics_computations[1]._my_rank == 5)
        self.assertTrue(precision_metric._metrics_computations[0]._batch_size == 100)
        self.assertTrue(precision_metric._metrics_computations[1]._batch_size == 100)

        self.assertTrue(precision_metric._metrics_computations[0].top_threshold == 1)
        self.assertTrue(precision_metric._metrics_computations[1].top_threshold == 2)
        self.assertTrue(
            precision_metric._metrics_computations[0].session_var_name == "session1"
        )
        self.assertTrue(
            precision_metric._metrics_computations[1].session_var_name == "session2"
        )
        self.assertTrue(precision_metric._metrics_computations[0].run_ranking_of_labels)
        self.assertTrue(
            precision_metric._metrics_computations[1].run_ranking_of_labels is False
        )
