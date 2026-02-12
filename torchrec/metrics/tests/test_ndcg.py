#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import replace
from typing import Any, Dict, List

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo, RecComputeMode
from torchrec.metrics.ndcg import NDCGMetric, SESSION_KEY
from torchrec.metrics.test_utils import RecTaskInfo


WORLD_SIZE = 4
BATCH_SIZE = 10

DefaultTaskInfo0 = RecTaskInfo(
    name="DefaultTask0",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)

DefaultTaskInfo1 = RecTaskInfo(
    name="DefaultTask1",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)

DefaultTaskInfo2 = RecTaskInfo(
    name="DefaultTask2",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)


def get_test_case_single_session_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0]]),
        "expected_ndcg_exp": torch.tensor([0.1103]),
        "expected_ndcg_non_exp": torch.tensor([0.1522]),
    }


def get_test_case_multiple_sessions_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([0.6748]),
        "expected_ndcg_non_exp": torch.tensor([0.6463]),
    }


def get_test_case_all_labels_zero() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([2.5]),
        "expected_ndcg_non_exp": torch.tensor([2.5]),
    }


def get_test_case_another_multiple_sessions_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
    }


def get_test_case_at_k() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.6131 + 0.3869) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.6131 + 0.3869) / 2]),
    }


def get_test_case_remove_single_length_sessions() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 3, 4]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([0.3066]),
        "expected_ndcg_non_exp": torch.tensor([0.3066]),
    }


def get_test_case_negative_task() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.9, 0.5, 0.7, 0.6, 0.8, 0.9]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
    }


def get_test_case_scale_by_weights_tensor() -> Dict[str, torch.Tensor]:
    """
    For this test case,
    predictions * weights = [0.1, 0, 0, 0.4, 0.0, 0.0]
    labels * weights = [1, 0, 0, 1, 0, 0]
    So NDCG going to be perfect for both sessions.
    """
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "expected_ndcg_exp": torch.tensor([(1.0 + 1.0) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(1.0 + 1.0) / 2]),
    }


def get_test_case_multitask() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
            ]
        ),
        "session_ids": torch.tensor(
            [
                [1, 1, 1, 1, 1, 2, 2, 2],
                [1, 1, 1, 1, 1, 2, 2, 2],
                [1, 1, 1, 1, 1, 2, 2, 2],
            ]
        ),
        "labels": torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0],
            ]
        ),
        "weights": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
            ]
        ),
        "expected_ndcg_exp": torch.tensor([0.6748, 0.6748, 0.6748]),
        "expected_ndcg_non_exp": torch.tensor([0.6463, 0.6463, 0.6463]),
    }


class NDCGMetricValueTest(unittest.TestCase):
    def generate_metric(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo] = [DefaultTaskInfo],
        exponential_gain: bool = False,
        session_key: str = SESSION_KEY,
        k: int = -1,
        remove_single_length_sessions: bool = False,
        scale_by_weights_tensor: bool = False,
        report_ndcg_as_decreasing_curve: bool = True,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        **kwargs: Dict[str, Any],
    ) -> NDCGMetric:
        return NDCGMetric(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            # pyrefly: ignore[bad-argument-type]
            session_key=session_key,
            # pyrefly: ignore[bad-argument-type]
            exponential_gain=exponential_gain,
            # pyrefly: ignore[bad-argument-type]
            remove_single_length_sessions=remove_single_length_sessions,
            # pyrefly: ignore[bad-argument-type]
            scale_by_weights_tensor=scale_by_weights_tensor,
            # pyrefly: ignore[bad-argument-type]
            report_ndcg_as_decreasing_curve=report_ndcg_as_decreasing_curve,
            # pyrefly: ignore[bad-argument-type]
            k=k,
            compute_mode=compute_mode,
            # pyrefly: ignore[bad-argument-type]
            **kwargs,
        )

    def test_single_session_non_exp(self) -> None:
        """
        Test single session in a update.
        """
        model_output = get_test_case_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_single_session_exp(self) -> None:
        """
        Test single session in a update for exponential metric.
        """
        model_output = get_test_case_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=True,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multiple_sessions_non_exp(self) -> None:
        """
        Test multiple sessions in a single update.
        """
        model_output = get_test_case_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multiple_sessions_exp(self) -> None:
        model_output = get_test_case_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=True,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_negative_sessions(self) -> None:
        """
        Test sessions where all labels are 0.
        """
        model_output = get_test_case_all_labels_zero()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_negative_sessions_exp(self) -> None:
        """
        Test sessions where all labels are 0, for exponential gain.
        """
        model_output = get_test_case_all_labels_zero()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=True,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_another_multiple_sessions(self) -> None:
        """
        Test another multiple sessions in a single update.
        """
        model_output = get_test_case_another_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_another_multiple_sessions_exp(self) -> None:
        """
        Test another multiple sessions in a single update, for exponential gain.
        """
        model_output = get_test_case_another_multiple_sessions_within_batch()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=True,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_at_k(self) -> None:
        """
        Test NDCG @ K.
        """
        model_output = get_test_case_at_k()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
            k=2,
        )
        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_remove_single_length_sessions(self) -> None:
        """
        Test NDCG with removing single length sessions.
        """
        model_output = get_test_case_remove_single_length_sessions()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
            remove_single_length_sessions=True,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_apply_negative_task_mask(self) -> None:
        """
        Test NDCG with apply negative task mask.
        """
        model_output = get_test_case_negative_task()
        TempTaskInfo = replace(DefaultTaskInfo, is_negative_task=True)

        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[TempTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )

        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_case_report_as_increasing_ndcg_and_scale_by_weights_tensor(self) -> None:
        """
        Test NDCG with reporting as increasing NDCG and scaling by weights tensor correctly.
        """
        model_output = get_test_case_scale_by_weights_tensor()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            exponential_gain=False,
            session_key=SESSION_KEY,
            remove_single_length_sessions=True,
            scale_by_weights_tensor=True,
            report_ndcg_as_decreasing_curve=False,
        )

        metric.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )

        output = metric.compute()
        actual_metric = output[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multitask_non_exp(self) -> None:
        """
        Test NDCG with multiple tasks.
        """
        model_output = get_test_case_multitask()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2],
            exponential_gain=False,
            session_key=SESSION_KEY,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        metric.update(
            predictions=model_output["predictions"],
            labels=model_output["labels"],
            weights=model_output["weights"],
            required_inputs={SESSION_KEY: model_output["session_ids"]},
        )
        output = metric.compute()
        actual_metric = torch.stack(
            [
                output[f"ndcg-{task.name}|lifetime_ndcg"]
                for task in [DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2]
            ]
        )
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multitask_exp(self) -> None:
        """
        Test NDCG with multiple tasks.
        """
        model_output = get_test_case_multitask()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2],
            exponential_gain=True,
            session_key=SESSION_KEY,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        metric.update(
            predictions=model_output["predictions"],
            labels=model_output["labels"],
            weights=model_output["weights"],
            required_inputs={SESSION_KEY: model_output["session_ids"]},
        )
        output = metric.compute()
        actual_metric = torch.stack(
            [
                output[f"ndcg-{task.name}|lifetime_ndcg"]
                for task in [DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2]
            ]
        )
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multitask_exp_fused_tasks_and_states(self) -> None:
        """
        Test NDCG with multiple tasks.
        """
        model_output = get_test_case_multitask()
        metric = self.generate_metric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2],
            exponential_gain=True,
            session_key=SESSION_KEY,
            compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
        )

        metric.update(
            predictions=model_output["predictions"],
            labels=model_output["labels"],
            weights=model_output["weights"],
            required_inputs={SESSION_KEY: model_output["session_ids"]},
        )
        output = metric.compute()
        actual_metric = torch.stack(
            [
                output[f"ndcg-{task.name}|lifetime_ndcg"]
                for task in [DefaultTaskInfo0, DefaultTaskInfo1, DefaultTaskInfo2]
            ]
        )
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )
