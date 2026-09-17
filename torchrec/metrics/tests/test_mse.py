#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, Set, Type

import torch
from torchrec.metrics.mse import compute_mse, compute_r_squared, compute_rmse, MSEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)


class TestMSEMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        error_sum = torch.sum(weights * torch.square(labels - predictions))
        return {
            "error_sum": error_sum,
            "weighted_num_samples": torch.sum(weights),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_mse(
            states["error_sum"],
            states["weighted_num_samples"],
        )


WORLD_SIZE = 4


class TestRMSEMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        error_sum = torch.sum(weights * torch.square(labels - predictions))
        return {
            "error_sum": error_sum,
            "weighted_num_samples": torch.sum(weights),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_rmse(
            states["error_sum"],
            states["weighted_num_samples"],
        )


class TestRSquaredMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predictions = predictions.double()
        error_sum = torch.sum(weights * torch.square(labels - predictions))
        return {
            "error_sum": error_sum,
            "weighted_num_samples": torch.sum(weights),
            "label_sum": torch.sum(weights * labels),
            "label_squared_sum": torch.sum(weights * torch.square(labels)),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_r_squared(
            states["error_sum"],
            states["weighted_num_samples"],
            states["label_sum"],
            states["label_squared_sum"],
        )


_r_squared_metric_test_helper: Callable[..., None] = partial(
    metric_test_helper, include_r_squared=True
)
update_wrapper(_r_squared_metric_test_helper, metric_test_helper)


class MSEMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = MSEMetric
    task_name: str = "mse"
    rmse_task_name: str = "rmse"
    r_squared_task_name: str = "r_squared"

    def test_mse_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMSEMetric,
            metric_name=MSEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_mse_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestMSEMetric,
            metric_name=MSEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_mse_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestMSEMetric,
            metric_name=MSEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_rmse_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestRMSEMetric,
            metric_name=MSEMetricTest.rmse_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_rmse_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestRMSEMetric,
            metric_name=MSEMetricTest.rmse_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_rmse_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestRMSEMetric,
            metric_name=MSEMetricTest.rmse_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_r_squared_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestRSquaredMetric,
            metric_name=MSEMetricTest.r_squared_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_r_squared_metric_test_helper,
        )

    def test_r_squared_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestRSquaredMetric,
            metric_name=MSEMetricTest.r_squared_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_r_squared_metric_test_helper,
        )

    def test_r_squared_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestRSquaredMetric,
            metric_name=MSEMetricTest.r_squared_task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_r_squared_metric_test_helper,
        )


class MSEGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = MSEMetric
    task_name: str = "mse"

    def test_sync_mse(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=MSEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMSEMetric,
            metric_name=MSEGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )


# The two states include_r_squared adds. The golden snapshot pins MSEMetric's
# default keys and has no row for this variant, so nothing else pins the delta.
_R_SQUARED_KEYS: Set[str] = {
    "_metrics_computations.0.label_squared_sum",
    "_metrics_computations.0.label_sum",
}


class MSECheckpointCompatibilityTest(unittest.TestCase):
    """Loading an r_squared checkpoint into a metric built without r_squared.

    Whether the load survives is not a property of the key sets, so the golden
    snapshot in `test_metric_fqn_backward_compatibility.py` cannot answer it.
    """

    @staticmethod
    def _build(include_r_squared: bool = False) -> MSEMetric:
        return MSEMetric(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=[RecTaskInfo(name="task1")],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
            include_r_squared=include_r_squared,
        )

    def test_r_squared_checkpoint_loads_into_default(self) -> None:
        """`mse.py` registers the r_squared states unconditionally and only
        toggles `persistent`. That leaves them in torchmetrics' `_defaults`, so
        a default-configured metric can still pop them off a checkpoint that
        has them. Moving the registration itself behind the flag moves no
        `state_dict` key, so the golden snapshot stays green.
        """
        default = self._build()
        variant = self._build(include_r_squared=True)

        default_keys = set(default.state_dict())
        variant_keys = set(variant.state_dict())
        # Both directions, both exact. Compared against the union instead, a
        # variant that stopped adding anything would still match, and the load
        # below would be feeding a module its own key set.
        self.assertEqual(
            default_keys - variant_keys,
            set(),
            "the variant drops keys the default has",
        )
        self.assertEqual(
            variant_keys - default_keys,
            _R_SQUARED_KEYS,
            "include_r_squared no longer adds exactly the r_squared states",
        )

        default.load_state_dict(variant.state_dict(), strict=True)

    def test_default_checkpoint_loads_into_r_squared(self) -> None:
        """The other direction: turning r_squared on and resuming.

        The older checkpoint carries none of the r_squared keys. torchmetrics
        pops by `_defaults`, which holds every registered state, so the load
        tolerates their absence instead of demanding them.
        """
        default = self._build()
        variant = self._build(include_r_squared=True)

        self.assertEqual(
            set(variant.state_dict()) - set(default.state_dict()),
            _R_SQUARED_KEYS,
            "include_r_squared no longer adds exactly the r_squared states",
        )

        variant.load_state_dict(default.state_dict(), strict=True)
