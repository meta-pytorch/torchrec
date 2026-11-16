#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, Optional, Type

import torch
from torchrec.metrics.calibration import CalibrationMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


class TestCalibrationMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        calibration_num = torch.sum(predictions * weights)
        calibration_denom = torch.sum(labels * weights)
        num_samples = torch.tensor(labels.size()[0]).double()
        return {
            "calibration_num": calibration_num,
            "calibration_denom": calibration_denom,
            "num_samples": num_samples,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.where(
            states["calibration_denom"] <= 0.0,
            0.0,
            states["calibration_num"] / states["calibration_denom"],
        ).double()


WORLD_SIZE = 4


class CalibrationMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = CalibrationMetric
    task_name: str = "calibration"

    def test_calibration_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            metric_name=CalibrationMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_calibration_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            metric_name=CalibrationMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_calibration_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            metric_name=CalibrationMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )


class CalibrationGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = CalibrationMetric
    task_name: str = "calibration"

    def test_sync_calibration(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=CalibrationMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestCalibrationMetric,
            metric_name=CalibrationGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
