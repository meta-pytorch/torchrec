#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List

import torch
from torchrec.metrics.auc import _state_reduction
from torchrec.metrics.cpu_comms_metric_module import CPUCommsRecMetricModule
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecMetricList
from torchrec.metrics.test_utils import gen_test_tasks
from torchrec.metrics.test_utils.mock_metrics import (
    assert_tensor_dict_equals,
    create_metric_states_dict,
    create_tensor_states,
    MockRecMetric,
)


class CPUCommsRecMetricModuleTest(unittest.TestCase):
    """
    Tests cloning rec metrics and loading snapshots into CPUCommsRecMetricModule.
    """

    def setUp(self) -> None:
        self.world_size = 2
        self.batch_size = 4
        self.my_rank = 0
        self.tasks = gen_test_tasks(["test_task"])

    def test_clone_rec_metrics_reference(self) -> None:
        """Tests cloned rec metrics upon initialization is a deep copy of the original."""

        mock_metric_1 = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
        )

        mock_metric_2 = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
        )

        rec_metrics = RecMetricList([mock_metric_1, mock_metric_2])

        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=rec_metrics,
        )

        original_metrics = rec_metrics.rec_metrics
        cloned_metrics = cpu_comms_module.rec_metrics.rec_metrics

        self.assertEqual(len(original_metrics), len(cloned_metrics))
        for original_metric, cloned_metric in zip(original_metrics, cloned_metrics):
            original_metric = cast(MockRecMetric, original_metric)
            cloned_metric = cast(MockRecMetric, cloned_metric)

            # Verify basic properties are preserved
            self.assertEqual(original_metric._world_size, cloned_metric._world_size)
            self.assertEqual(original_metric._my_rank, cloned_metric._my_rank)
            self.assertEqual(original_metric._batch_size, cloned_metric._batch_size)
            self.assertEqual(original_metric._compute_mode, cloned_metric._compute_mode)

            # State tensor names must be the same in order to load into the correct
            # state tensors.
            original_metric_states = set(
                original_metric.get_computation_states().keys()
            )
            cloned_metric_states = set(cloned_metric.get_computation_states().keys())
            self.assertSetEqual(original_metric_states, cloned_metric_states)

            # Cloned metric should have torchmetric.Metric's sync() disabled to prevent
            # unwanted distributed syncs. All syncs will be called via cpu_comms_module.
            self.assertTrue(cloned_metric.verify_sync_disabled())

    def test_load_metric_states(self) -> None:
        """
        Test loading metric states into a single metric computation.
        """

        initial_states = {
            "state_1": torch.tensor(1.0),
            "state_2": torch.tensor(2.0),
            "state_3": torch.tensor(3.0),
        }
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=initial_states,
        )

        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        metric_states = create_metric_states_dict(
            metric_prefix="test_prefix",
            computation_name="MockRecMetricComputation",
            metric_states={
                **initial_states,
                "ignored_key": torch.tensor(15.0),
            },
        )

        cloned_metric = cpu_comms_module.rec_metrics.rec_metrics[0]
        cloned_computation = cloned_metric._metrics_computations[0]

        cpu_comms_module._load_metric_states(
            "test_prefix", cloned_computation, metric_states
        )

        self.assertTrue(cloned_computation._update_called)
        self.assertIsNone(cloned_computation._computed)
        assert_tensor_dict_equals(
            cloned_metric.get_computation_states(),
            initial_states,
        )

    def test_snapshot_generation(self) -> None:
        """Test that original metrics and comms module loaded metrics produce the same snapshot."""

        original_states = {
            "state_1": torch.tensor(7.5),
            "state_2": torch.tensor(12.0),
            "state_3": torch.tensor(49.0),
        }

        original_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=original_states,
        )

        rec_metrics = RecMetricList([original_metric])
        original_snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=rec_metrics,
        )
        cpu_comms_module.load_local_metric_state_snapshot(original_snapshot)
        loaded_snapshot = MetricStateSnapshot.from_metrics(cpu_comms_module.rec_metrics)

        assert_tensor_dict_equals(
            original_snapshot.metric_states,
            loaded_snapshot.metric_states,
        )

    def test_load_metric_states_partial_load(self) -> None:
        """Test loading metric states when some keys are missing from the snapshot."""

        initial_states = {
            "state_1": torch.tensor(1.0),
            "state_2": torch.tensor(2.0),
            "state_3": torch.tensor(3.0),
        }
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=initial_states,
        )

        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=self.tasks,
            rec_metrics=RecMetricList([mock_metric]),
        )

        # Metric states only contains one of the initial keys
        metric_states = create_metric_states_dict(
            metric_prefix="test_prefix",
            computation_name="MockRecMetricComputation",
            metric_states={"state_1": torch.tensor(5.0)},
        )

        cloned_metric = cpu_comms_module.rec_metrics.rec_metrics[0]
        cloned_computation = cloned_metric._metrics_computations[0]

        cpu_comms_module._load_metric_states(
            "test_prefix", cloned_computation, metric_states
        )

        torch.testing.assert_close(
            cloned_metric.get_computation_states()["state_1"], torch.tensor(5.0)
        )
        self.assertFalse(
            torch.allclose(
                cloned_metric.get_computation_states()["state_2"], torch.tensor(2.0)
            )
        )
        self.assertFalse(
            torch.allclose(
                cloned_metric.get_computation_states()["state_2"], torch.tensor(3.0)
            )
        )

    def test_load_multiple_metrics_unfused(self) -> None:
        """Test handling multiple metrics and tasks together."""

        ne_tasks = gen_test_tasks(["task1", "task2", "task3"])
        auc_tasks = gen_test_tasks(["task4", "task5", "task6"])

        ne_states = create_tensor_states(["state_1", "state_2", "state_3"], n_tasks=1)
        mock_nes = [
            MockRecMetric(
                world_size=self.world_size,
                my_rank=self.my_rank,
                batch_size=self.batch_size,
                tasks=[task],
                initial_states=ne_states,
            )
            for task in ne_tasks
        ]

        auc_states = {
            "state_1": [torch.tensor([[1.0, 2.0]])],
            "state_2": [torch.tensor([[0.0, 1.0]])],
            "state_3": [torch.tensor([[4.0, 1.0]])],
        }
        mock_aucs = [
            MockRecMetric(
                world_size=self.world_size,
                my_rank=self.my_rank,
                batch_size=self.batch_size,
                tasks=[task],
                reduction_fn=_state_reduction,
                initial_states=auc_states,
                is_tensor_list=True,
            )
            for task in auc_tasks
        ]

        rec_metrics_list: List[RecMetric] = [*mock_nes, *mock_aucs]
        rec_metrics = RecMetricList(rec_metrics_list)
        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=ne_tasks + auc_tasks,
            rec_metrics=rec_metrics,
        )

        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)
        cpu_comms_module.load_local_metric_state_snapshot(snapshot)

        ne_states_dict = {}
        for task in ne_tasks:
            ne_states_dict.update(
                create_metric_states_dict(
                    metric_prefix=task.name,
                    computation_name="MockRecMetricComputation",
                    metric_states=ne_states,
                )
            )

        auc_states_dict = {}
        for task in auc_tasks:
            auc_states_dict.update(
                create_metric_states_dict(
                    metric_prefix=task.name,
                    computation_name="MockRecMetricComputation",
                    metric_states=auc_states,
                )
            )

        expected_metric_states = {**ne_states_dict, **auc_states_dict}
        self.assertEqual(len(cpu_comms_module.rec_metrics.rec_metrics), 6)
        actual_metric_states_dict = {}
        for task, metric in zip(
            cpu_comms_module.rec_tasks, cpu_comms_module.rec_metrics.rec_metrics
        ):
            metric = cast(MockRecMetric, metric)
            actual_metric_states_dict.update(
                create_metric_states_dict(
                    metric_prefix=task.name,
                    computation_name="MockRecMetricComputation",
                    metric_states=metric.get_computation_states(),
                )
            )
        assert_tensor_dict_equals(
            actual_metric_states_dict,
            expected_metric_states,
        )

    def test_load_multiple_metrics_fused(self) -> None:
        """Test handling multiple metrics and tasks together."""

        ne_tasks = gen_test_tasks(["task1", "task2", "task3"])

        ne_states = create_tensor_states(["state_1", "state_2", "state_3"], n_tasks=3)
        mock_ne = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=ne_tasks,
            initial_states=ne_states,
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        )

        rec_metrics = RecMetricList([mock_ne])
        cpu_comms_module = CPUCommsRecMetricModule(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rec_tasks=ne_tasks,
            rec_metrics=rec_metrics,
        )

        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)
        cpu_comms_module.load_local_metric_state_snapshot(snapshot)

        self.assertEqual(len(cpu_comms_module.rec_metrics.rec_metrics), 1)
        metric = cpu_comms_module.rec_metrics.rec_metrics[0]
        metric = cast(MockRecMetric, metric)
        assert_tensor_dict_equals(
            metric.get_computation_states(),
            ne_states,
        )


if __name__ == "__main__":
    unittest.main()
