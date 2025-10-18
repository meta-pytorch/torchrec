#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, Type

import torch
from torchrec.metrics.ne import (
    compute_cross_entropy,
    compute_logloss,
    compute_ne,
    NEMetric,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


WORLD_SIZE = 4


class TestNEMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestNEMetric.eta
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
            eta=TestNEMetric.eta,
            allow_missing_label_with_zero_weight=allow_missing_label_with_zero_weight,
        )


class TestLoglossMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestNEMetric.eta
        )
        cross_entropy_sum = torch.sum(cross_entropy)
        pos_labels = torch.sum(weights * labels, dim=-1)
        neg_labels = torch.sum(weights * (1.0 - labels), dim=-1)
        return {
            "cross_entropy_sum": cross_entropy_sum,
            "pos_labels": pos_labels,
            "neg_labels": neg_labels,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_logloss(
            states["cross_entropy_sum"],
            pos_labels=states["pos_labels"],
            neg_labels=states["neg_labels"],
            eta=TestLoglossMetric.eta,
        )


class NEMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = NEMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "ne"

    def test_ne_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
        )

    def test_ne_zero_weights(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            zero_weights=True,
        )

    def test_ne_allow_missing_label_with_zero_weight(self) -> None:
        eta = 1e-12
        ne = compute_ne(
            ce_sum=torch.rand(3),
            weighted_num_samples=torch.tensor([3, 0, 2]),
            pos_labels=torch.tensor([1, 0, 2]),
            neg_labels=torch.tensor([2, 0, 0]),
            eta=eta,
            allow_missing_label_with_zero_weight=True,
        )
        self.assertTrue(torch.all(~ne.isinf()))
        self.assertTrue(torch.all(~ne.isnan()))
        self.assertTrue(torch.equal(ne.eq(eta), torch.tensor([False, True, False])))

    _logloss_metric_test_helper: Callable[..., None] = partial(
        metric_test_helper, include_logloss=True
    )
    update_wrapper(_logloss_metric_test_helper, metric_test_helper)

    def test_logloss_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._logloss_metric_test_helper,
        )

    def test_logloss_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._logloss_metric_test_helper,
        )

    def test_logloss_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._logloss_metric_test_helper,
        )

    def test_logloss_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._logloss_metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._logloss_metric_test_helper,
            batch_window_size=10,
        )


class NEGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = NEMetric
    task_name: str = "ne"

    def test_sync_ne(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
