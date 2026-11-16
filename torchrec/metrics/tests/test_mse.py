#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, Set, Tuple, Type

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


# The two states include_r_squared adds. The golden snapshot records both key
# sets. The tests below check that a checkpoint loads strictly between them.
_R_SQUARED_KEYS: Set[str] = {
    "_metrics_computations.0.label_squared_sum",
    "_metrics_computations.0.label_sum",
}


class MSECheckpointCompatibilityTest(unittest.TestCase):
    """Loading a checkpoint across include_r_squared, in both directions.

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

    @staticmethod
    def _update(metric: MSEMetric) -> None:
        """One real batch, so the checkpoint holds measurements, not zeros."""
        generator = torch.Generator().manual_seed(0)
        metric.update(
            predictions={"task1": torch.rand(32, generator=generator)},
            labels={"task1": torch.rand(32, generator=generator)},
            weights={"task1": torch.rand(32, generator=generator)},
        )

    def _build_pair(self) -> Tuple[MSEMetric, MSEMetric]:
        default = self._build()
        variant = self._build(include_r_squared=True)
        default_keys = set(default.state_dict())
        variant_keys = set(variant.state_dict())
        # Exact in both directions: a variant that added nothing would make the
        # loads below same-config.
        self.assertEqual(default_keys - variant_keys, set())
        self.assertEqual(variant_keys - default_keys, _R_SQUARED_KEYS)
        return default, variant

    def _assert_loads(self, src: MSEMetric, dst: MSEMetric) -> None:
        """Strict-load one batch of src into dst. Shared states come back
        exactly, and states only dst has keep their defaults."""
        defaults = copy.deepcopy(dst.state_dict())
        self._update(src)
        saved = src.state_dict()
        # Load a copy: torchmetrics adopts the loaded tensors as its state.
        dst.load_state_dict(copy.deepcopy(saved), strict=True)
        loaded = dst.state_dict()
        self.assertEqual(set(loaded), set(defaults))
        for key, value in loaded.items():
            want = saved[key] if key in saved else defaults[key]
            torch.testing.assert_close(value, want, rtol=0, atol=0)

    def test_r_squared_checkpoint_loads_into_default(self) -> None:
        """`mse.py` registers the r_squared states unconditionally and only
        toggles `persistent`. That leaves them in torchmetrics' `_defaults`, so
        a default-configured metric can still pop them off a checkpoint that
        has them. Moving the registration itself behind the flag moves no
        `state_dict` key, so the golden snapshot stays green.
        """
        default, variant = self._build_pair()
        self._assert_loads(variant, default)

    def test_default_checkpoint_loads_into_r_squared(self) -> None:
        """The other direction: turning r_squared on and resuming.

        The older checkpoint carries none of the r_squared keys. torchmetrics
        restores only the states a checkpoint has and never reports the rest
        as missing, so the load tolerates their absence.
        """
        default, variant = self._build_pair()
        self._assert_loads(default, variant)
