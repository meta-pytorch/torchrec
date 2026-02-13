#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, Optional, Type

import torch
from torchrec.metrics.multiclass_recall import (
    compute_multiclass_recall_at_k,
    get_multiclass_recall_states,
    MulticlassRecallMetric,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)

N_CLASSES = 4
WORLD_SIZE = 4


class TestMulticlassRecallMetric(TestMetric):
    n_classes: int = N_CLASSES

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        states = get_multiclass_recall_states(
            predictions, labels, weights, TestMulticlassRecallMetric.n_classes
        )
        return states

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_multiclass_recall_at_k(
            states["tp_at_k"],
            states["total_weights"],
        )


class MulticlassRecallMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = MulticlassRecallMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "multiclass_recall"

    def test_multiclass_recall_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            metric_name=MulticlassRecallMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            n_classes=N_CLASSES,
        )

    def test_multiclass_recall_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            metric_name=MulticlassRecallMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            n_classes=N_CLASSES,
        )

    def test_multiclass_recall_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            metric_name=MulticlassRecallMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            n_classes=N_CLASSES,
        )

        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            metric_name=MulticlassRecallMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
            n_classes=N_CLASSES,
        )


class MulticlassRecallGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = MulticlassRecallMetric
    task_name: str = "multiclass_recall"

    def test_sync_multiclass_recall(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            metric_name=MulticlassRecallGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
            # pyrefly: ignore[bad-argument-type]
            n_classes=N_CLASSES,
        )
