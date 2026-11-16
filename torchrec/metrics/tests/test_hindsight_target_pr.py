#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, Optional, Type

import torch
from torchrec.metrics.hindsight_target_pr import (
    compute_precision,
    compute_recall,
    compute_threshold_idx,
    HindsightTargetPRMetric,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


WORLD_SIZE = 4
THRESHOLD_GRANULARITY = 1000


class TestHindsightTargetPRMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        tp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        fp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
        for i, threshold in enumerate(thresholds):
            tp_sum[i] = torch.sum(weights * ((predictions >= threshold) * labels), -1)
            fp_sum[i] = torch.sum(
                weights * ((predictions >= threshold) * (1 - labels)), -1
            )
        return {
            "true_pos_sum": tp_sum,
            "false_pos_sum": fp_sum,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        threshold_idx = compute_threshold_idx(
            states["true_pos_sum"], states["false_pos_sum"], 0.5
        )
        return torch.Tensor(threshold_idx)


class TestHindsightTargetPrecisionMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        tp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        fp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
        for i, threshold in enumerate(thresholds):
            tp_sum[i] = torch.sum(weights * ((predictions >= threshold) * labels), -1)
            fp_sum[i] = torch.sum(
                weights * ((predictions >= threshold) * (1 - labels)), -1
            )
        return {
            "true_pos_sum": tp_sum,
            "false_pos_sum": fp_sum,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        threshold_idx = compute_threshold_idx(
            states["true_pos_sum"], states["false_pos_sum"], 0.5
        )
        return compute_precision(
            states["true_pos_sum"][threshold_idx],
            states["false_pos_sum"][threshold_idx],
        )


class TestHindsightTargetRecallMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        tp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        fp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        fn_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
        thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
        for i, threshold in enumerate(thresholds):
            tp_sum[i] = torch.sum(weights * ((predictions >= threshold) * labels), -1)
            fp_sum[i] = torch.sum(
                weights * ((predictions >= threshold) * (1 - labels)), -1
            )
            fn_sum[i] = torch.sum(weights * ((predictions <= threshold) * labels), -1)
        return {
            "true_pos_sum": tp_sum,
            "false_pos_sum": fp_sum,
            "false_neg_sum": fn_sum,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        threshold_idx = compute_threshold_idx(
            states["true_pos_sum"], states["false_pos_sum"], 0.5
        )
        return compute_recall(
            states["true_pos_sum"][threshold_idx],
            states["false_neg_sum"][threshold_idx],
        )


# Fused tests are not supported for this metric.
class TestHindsightTargetPRMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = HindsightTargetPRMetric
    pr_task_name: str = "hindsight_target_pr"
    precision_task_name: str = "hindsight_target_precision"
    recall_task_name: str = "hindsight_target_recall"

    def test_hindsight_target_precision_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=HindsightTargetPRMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestHindsightTargetPrecisionMetric,
            metric_name=TestHindsightTargetPRMetricTest.precision_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_hindsight_target_recall_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=HindsightTargetPRMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestHindsightTargetRecallMetric,
            metric_name=TestHindsightTargetPRMetricTest.recall_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )
