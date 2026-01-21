#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Type, Union
from collections import OrderedDict
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from torch import Tensor
from torchrec.metrics.metrics_config import BatchSizeStage, DefaultTaskInfo
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
    RecMetricException,
    RecTaskInfo,
)
from torchrec.metrics.test_utils import (
    gen_test_batch,
    gen_test_tasks,
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)
from torchrec.metrics.tower_qps import TowerQPSMetric

WORLD_SIZE = 4
WARMUP_STEPS = 100
DURING_WARMUP_NSTEPS = 10
AFTER_WARMUP_NSTEPS = 120


TestRecMetricOutput = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


class TestTowerQPSMetric(TestMetric):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
    ) -> None:
        super().__init__(world_size, rec_tasks)

    # The abstract _get_states method in TestMetric has to be overwritten
    # For tower qps the time_lapse state is not generated from labels, predictions
    # or weights
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return {}

    @staticmethod
    def _reduce(states: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        reduced_states: Dict[str, torch.Tensor] = {}
        # Need to check if states is empty, because we only update the states after warmup
        if states:
            reduced_states["num_samples"] = torch.sum(
                torch.stack(states["num_samples"]), dim=0
            )
            reduced_states["time_lapse"] = torch.max(
                torch.stack(states["time_lapse"]), dim=0
            ).values
        return reduced_states

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "num_samples" not in states or "time_lapse" not in states:
            # This is to match the default 0.0 output from TowerQPSMetric if warmup is not done
            return torch.tensor(float("0.0"), dtype=torch.double)

        return torch.where(
            states["time_lapse"] <= 0.0,
            0.0,
            states["num_samples"] / states["time_lapse"],
        ).double()

    def compute(
        self,
        model_outs: List[Dict[str, torch.Tensor]],
        nsteps: int,
        batch_window_size: int,
        timestamps: Optional[List[float]],
    ) -> TestRecMetricOutput:
        assert timestamps is not None
        lifetime_states, window_states, local_lifetime_states, local_window_states = (
            {task_info.name: {} for task_info in self._rec_tasks} for _ in range(4)
        )
        for i in range(WARMUP_STEPS, nsteps):
            for task_info in self._rec_tasks:
                local_states = {
                    "num_samples": torch.tensor(
                        model_outs[i][task_info.label_name].shape[-1],
                        dtype=torch.long,
                    ),
                    "time_lapse": torch.tensor(
                        timestamps[i] - timestamps[i - 1], dtype=torch.double
                    ),
                }
                self._aggregate(local_lifetime_states[task_info.name], local_states)
                if nsteps - batch_window_size <= i:
                    self._aggregate(local_window_states[task_info.name], local_states)

        for task_info in self._rec_tasks:
            aggregated_lifetime_state = {}
            for k, v in local_lifetime_states[task_info.name].items():
                aggregated_lifetime_state[k] = [
                    torch.zeros_like(v) for _ in range(self.world_size)
                ]
                dist.all_gather(aggregated_lifetime_state[k], v)
            lifetime_states[task_info.name] = self._reduce(aggregated_lifetime_state)

            aggregated_window_state = {}
            for k, v in local_window_states[task_info.name].items():
                aggregated_window_state[k] = [
                    torch.zeros_like(v) for _ in range(self.world_size)
                ]
                dist.all_gather(aggregated_window_state[k], v)
            window_states[task_info.name] = self._reduce(aggregated_window_state)

        lifetime_metrics = {}
        window_metrics = {}
        local_lifetime_metrics = {}
        local_window_metrics = {}
        for task_info in self._rec_tasks:
            lifetime_metrics[task_info.name] = self._compute(
                lifetime_states[task_info.name]
            )
            window_metrics[task_info.name] = self._compute(
                window_states[task_info.name]
            )
            local_lifetime_metrics[task_info.name] = self._compute(
                local_lifetime_states[task_info.name]
            )
            local_window_metrics[task_info.name] = self._compute(
                local_window_states[task_info.name]
            )
        return (
            lifetime_metrics,
            window_metrics,
            local_lifetime_metrics,
            local_window_metrics,
        )


_test_tower_qps: Callable[..., None] = partial(
    metric_test_helper,
    is_time_dependent=True,
    time_dependent_metric={TowerQPSMetric: "torchrec.metrics.tower_qps"},
)
update_wrapper(_test_tower_qps, metric_test_helper)


class TowerQPSMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_size = 1
        self.batch_size = 256

    target_clazz: Type[RecMetric] = TowerQPSMetric
    task_names: str = "qps"

    def test_tower_qps_during_warmup_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
        )

    def test_tower_qps_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
            test_nsteps=DURING_WARMUP_NSTEPS,
        )

    def test_tower_qps_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_tower_qps_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_check_update_tower_qps_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=True,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_check_update_tower_qps_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=True,
            world_size=WORLD_SIZE,
            entry_point=_test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_warmup_checkpointing(self) -> None:
        warmup_steps = 5
        extra_steps = 2
        batch_size = 128
        qps = TowerQPSMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=[DefaultTaskInfo],
            warmup_steps=warmup_steps,
            compute_on_all_ranks=False,
            should_validate_update=False,
            window_size=200,
        )
        model_output = gen_test_batch(batch_size)
        for i in range(5):
            for _ in range(warmup_steps + extra_steps):
                qps.update(
                    predictions={"DefaultTask": model_output["prediction"]},
                    labels={"DefaultTask": model_output["label"]},
                    weights={"DefaultTask": model_output["weight"]},
                )
            self.assertEqual(
                qps._metrics_computations[0].warmup_examples,
                batch_size * warmup_steps * (i + 1),
            )
            self.assertEqual(
                qps._metrics_computations[0].num_examples,
                batch_size * (warmup_steps + extra_steps) * (i + 1),
            )
            # Mimic trainer crashing and loading a checkpoint.
            qps._metrics_computations[0]._steps = 0

    def test_mtml_empty_update(self) -> None:
        warmup_steps = 2
        extra_steps = 2
        batch_size = 128
        task_names = ["t1", "t2"]
        tasks = gen_test_tasks(task_names)
        qps = TowerQPSMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=tasks,
            warmup_steps=warmup_steps,
            compute_on_all_ranks=False,
            should_validate_update=False,
            window_size=200,
        )
        for step in range(warmup_steps + extra_steps):
            _model_output = [
                gen_test_batch(
                    label_name=task.label_name,
                    prediction_name=task.prediction_name,
                    weight_name=task.weight_name,
                    batch_size=batch_size,
                )
                for task in tasks
            ]
            model_output = {k: v for d in _model_output for k, v in d.items()}
            labels, predictions, weights, _ = parse_task_model_outputs(
                tasks, model_output
            )
            if step % 2 == 0:
                del labels["t1"]
            else:
                del labels["t2"]
            qps.update(predictions=predictions, labels=labels, weights=weights)
            self.assertEqual(
                qps._metrics_computations[0].num_examples, (step + 1) // 2 * batch_size
            )
            self.assertEqual(
                qps._metrics_computations[1].num_examples, (step + 2) // 2 * batch_size
            )

    def test_tower_qps_update_with_invalid_tensors(self) -> None:
        warmup_steps = 2
        batch_size = 128
        task_names = ["t1", "t2"]
        tasks = gen_test_tasks(task_names)
        qps = TowerQPSMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=tasks,
            warmup_steps=warmup_steps,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            compute_on_all_ranks=False,
            should_validate_update=True,
            window_size=200,
        )

        with self.assertRaisesRegex(
            RecMetricException,
            "Failed to convert labels to tensor for fused computation",
        ):
            qps.update(
                predictions=torch.ones(batch_size),
                labels={
                    "key_0": torch.rand(batch_size),
                    "key_1": torch.rand(batch_size),
                    "key_2": torch.rand(batch_size),
                },
                weights=torch.rand(batch_size),
            )

        with self.assertRaisesRegex(
            RecMetricException,
            "Failed to convert weights to tensor for fused computation",
        ):
            qps.update(
                predictions=torch.ones(batch_size),
                labels=torch.rand(batch_size),
                weights={
                    "key_0": torch.rand(batch_size),
                    "key_1": torch.rand(batch_size),
                    "key_2": torch.rand(batch_size),
                },
            )

    @patch("torchrec.metrics.tower_qps.time.monotonic")
    def test_batch_size_schedule(self, time_mock: Mock) -> None:

        def _gen_data_with_batch_size(
            batch_size: int,
        ) -> Dict[str, Union[Dict[str, Tensor], Tensor]]:
            return {
                "labels": {
                    "t1": torch.rand(batch_size),
                    "t2": torch.rand(batch_size),
                    "t3": torch.rand(batch_size),
                },
                "predictions": torch.ones(batch_size),
                "weights": torch.rand(batch_size),
            }

        batch_size_stages = [BatchSizeStage(256, 1), BatchSizeStage(512, None)]
        time_mock.return_value = 1
        batch_size = 256
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=batch_size,
            world_size=1,
            window_size=1000,
            batch_size_stages=batch_size_stages,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        data = _gen_data_with_batch_size(batch_size_stages[0].batch_size)
        metric.update(**data)  # pyre-ignore[6]

        self.assertEqual(
            metric.compute(),
            {
                "qps-t1|lifetime_qps": 0,
                "qps-t2|lifetime_qps": 0,
                "qps-t3|lifetime_qps": 0,
                "qps-t1|window_qps": 0,
                "qps-t2|window_qps": 0,
                "qps-t3|window_qps": 0,
                "qps-t1|total_examples": 256,
                "qps-t2|total_examples": 256,
                "qps-t3|total_examples": 256,
            },
        )

        data = _gen_data_with_batch_size(batch_size_stages[1].batch_size)
        metric.update(**data)  # pyre-ignore[6]

        self.assertEqual(
            metric.compute(),
            {
                "qps-t1|lifetime_qps": 0,
                "qps-t2|lifetime_qps": 0,
                "qps-t3|lifetime_qps": 0,
                "qps-t1|window_qps": 0,
                "qps-t2|window_qps": 0,
                "qps-t3|window_qps": 0,
                "qps-t1|total_examples": 768,
                "qps-t2|total_examples": 768,
                "qps-t3|total_examples": 768,
            },
        )

    def test_num_batch_without_batch_size_stages(self) -> None:
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        self.assertFalse(hasattr(metric, "num_batch"))

        metric.update(
            labels={
                "t1": torch.rand(self.batch_size),
                "t2": torch.rand(self.batch_size),
                "t3": torch.rand(self.batch_size),
            },
            predictions=torch.ones(self.batch_size),
            weights=torch.rand(self.batch_size),
        )
        state_dict: Dict[str, Any] = metric.state_dict()
        self.assertNotIn("num_batch", state_dict)

    def test_state_dict_load_module_lifecycle(self) -> None:
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )

        self.assertTrue(hasattr(metric, "_num_batch"))

        metric.update(
            labels={
                "t1": torch.rand(self.batch_size),
                "t2": torch.rand(self.batch_size),
                "t3": torch.rand(self.batch_size),
            },
            predictions=torch.ones(self.batch_size),
            weights=torch.rand(self.batch_size),
        )
        self.assertEqual(metric._num_batch, 1)
        state_dict = metric.state_dict()
        self.assertIn("num_batch", state_dict)
        self.assertEqual(state_dict["num_batch"].item(), metric._num_batch)

        new_metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )
        self.assertEqual(new_metric._num_batch, 0)
        new_metric.load_state_dict(state_dict)
        self.assertEqual(new_metric._num_batch, 1)

        state_dict = new_metric.state_dict()
        self.assertIn("num_batch", state_dict)
        self.assertEqual(state_dict["num_batch"].item(), new_metric._num_batch)

    def test_state_dict_hook_adds_key(self) -> None:
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(256, None)],
        )

        for _ in range(5):
            metric.update(
                labels={
                    "t1": torch.rand(self.batch_size),
                    "t2": torch.rand(self.batch_size),
                    "t3": torch.rand(self.batch_size),
                },
                predictions=torch.ones(self.batch_size),
                weights=torch.rand(self.batch_size),
            )
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix: str = "test_prefix_"
        metric.state_dict_hook(metric, state_dict, prefix, {})
        self.assertIn(f"{prefix}num_batch", state_dict)
        self.assertEqual(state_dict[f"{prefix}num_batch"].item(), 5)

    def test_state_dict_hook_no_batch_size_stages(self) -> None:
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            batch_size_stages=None,
        )
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix: str = "test_prefix_"
        metric.state_dict_hook(metric, state_dict, prefix, {})
        self.assertNotIn(f"{prefix}num_batch", state_dict)

    def test_load_state_dict_hook_restores_value(self) -> None:
        task_names = ["t1", "t2", "t3"]
        tasks = gen_test_tasks(task_names)
        metric = TowerQPSMetric(
            my_rank=0,
            tasks=tasks,
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_size=1000,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            batch_size_stages=[BatchSizeStage(256, 1), BatchSizeStage(512, None)],
        )
        state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        prefix: str = "test_prefix_"
        state_dict[f"{prefix}num_batch"] = torch.tensor(10, dtype=torch.long)
        metric.load_state_dict_hook(state_dict, prefix, {}, True, [], [], [])
        self.assertEqual(metric._num_batch, 10)
