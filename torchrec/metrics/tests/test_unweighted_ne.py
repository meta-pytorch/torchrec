#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, Optional, Type

import torch
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)
from torchrec.metrics.unweighted_ne import (
    compute_cross_entropy,
    compute_ne,
    UnweightedNEMetric,
)


WORLD_SIZE = 4


class TestUnweightedNEMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Override the weights to be all ones
        weights = torch.ones_like(labels)
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestUnweightedNEMetric.eta
        )
        cross_entropy_sum = torch.sum(cross_entropy)
        weighted_num_samples = torch.sum(weights)
        pos_labels = torch.sum(weights * labels)
        neg_labels = torch.sum(weights * (1.0 - labels))
        return {
            "cross_entropy_sum": cross_entropy_sum,
            "weighted_num_samples": weighted_num_samples,
            "pos_labels": pos_labels,
            "neg_labels": neg_labels,
            "num_samples": torch.tensor(labels.size()).long(),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        allow_missing_label_with_zero_weight = False
        if not states["weighted_num_samples"].all():
            allow_missing_label_with_zero_weight = True

        return compute_ne(
            states["cross_entropy_sum"],
            states["weighted_num_samples"],
            pos_labels=states["pos_labels"],
            neg_labels=states["neg_labels"],
            eta=TestUnweightedNEMetric.eta,
            allow_missing_label_with_zero_weight=allow_missing_label_with_zero_weight,
        )


class UnweightedNEMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = UnweightedNEMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "unweighted_ne"

    def test_unweighted_ne_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_unweighted_ne_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_unweighted_ne_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_unweighted_ne_update_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
        )

    def test_unweighted_ne_zero_weights(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            zero_weights=True,
        )


class UnweightedNEGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = UnweightedNEMetric
    task_name: str = "unweighted_ne"

    def test_sync_unweighted_ne(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=UnweightedNEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestUnweightedNEMetric,
            metric_name=UnweightedNEGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
