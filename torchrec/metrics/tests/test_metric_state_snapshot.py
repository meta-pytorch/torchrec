#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.metrics.auc import _state_reduction
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.rec_metric import RecComputeMode, RecMetricList
from torchrec.metrics.test_utils import gen_test_tasks
from torchrec.metrics.test_utils.mock_metrics import (
    assert_tensor_dict_equals,
    create_metric_states_dict,
    create_tensor_list_states,
    create_tensor_states,
    MockRecMetric,
)
from torchrec.metrics.throughput import ThroughputMetric


class MetricStateSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_size = 2
        self.batch_size = 4
        self.my_rank = 0
        self.tasks = gen_test_tasks(["test_task"])

    def test_init(self) -> None:
        """Test basic initialization of MetricStateSnapshot."""
        initial_states = create_tensor_states(["test_state_tensor"])
        throughput_metric = ThroughputMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=10,
        )

        snapshot = MetricStateSnapshot(
            metric_states=initial_states,
            throughput_metric=throughput_metric,
        )

        assert_tensor_dict_equals(snapshot.metric_states, initial_states)
        self.assertEqual(snapshot.throughput_metric, throughput_metric)

    def test_from_metrics_initial_states(self) -> None:
        """Test creating snapshot from metric with tensor states."""
        initial_states = create_tensor_states(["test_state_tensor"])
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=initial_states,
            # pyrefly: ignore[bad-argument-type]
            reduction_fn=torch.sum,
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        expected_states = create_metric_states_dict(
            metric_prefix="test_task",
            computation_name="MockRecMetricComputation",
            metric_states=initial_states,
        )

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)

    def test_from_metrics_with_list_states(self) -> None:
        """Test creating snapshot from metric with list states (AUC-like)."""
        initial_states = {
            "predictions": [torch.tensor([[1.0, 2.0]])],
            "labels": [torch.tensor([[0.0, 1.0]])],
        }
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=initial_states,
            reduction_fn=_state_reduction,
            is_tensor_list=True,
        )
        mock_metric.append_to_computation_states(
            {
                "predictions": torch.tensor([[3.0]]),
                "labels": torch.tensor([[5.0]]),
            }
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        expected_states = create_metric_states_dict(
            metric_prefix="test_task",
            computation_name="MockRecMetricComputation",
            # concat initial with appended states.
            metric_states={
                "predictions": [torch.tensor([[1.0, 2.0, 3.0]])],
                "labels": [torch.tensor([[0.0, 1.0, 5.0]])],
            },
        )

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)

    def test_from_metrics_with_throughput(self) -> None:
        """
        Test creating snapshot with throughput metric.

        ThroughputMetric must be a deep copy of the original metric to prevent
        concurrent access to the same object.
        """
        initial_states = create_tensor_states(["test_state_tensor"])
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states=initial_states,
            reduction_fn="sum",
        )

        throughput_metric = ThroughputMetric(
            batch_size=4,
            world_size=2,
            window_seconds=10,
        )
        self.assertEqual(throughput_metric.total_examples, 0)
        # bump total examples to world_size * batch_size
        throughput_metric.update()
        self.assertEqual(throughput_metric.total_examples, 8)

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics, throughput_metric)

        expected_states = create_metric_states_dict(
            metric_prefix="test_task",
            computation_name="MockRecMetricComputation",
            metric_states=initial_states,
        )

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)
        self.assertNotEqual(snapshot.throughput_metric, throughput_metric)
        self.assertIsNotNone(snapshot.throughput_metric)
        self.assertEqual(snapshot.throughput_metric.total_examples, 8)

    def test_from_metrics_fused_tasks_state_tensors(self) -> None:
        """Test state tensors with FUSED_TASKS computation mode."""
        initial_states = {
            "cross_entropy_sum": torch.tensor(1.0),
            "weighted_num_samples": torch.tensor(2.0),
        }
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=gen_test_tasks(["task1", "task2"]),
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            initial_states=initial_states,
            reduction_fn="sum",
        )

        # Increment tensors
        mock_metric.add_to_computation_states(
            {
                "cross_entropy_sum": torch.tensor(3.0),
                "weighted_num_samples": torch.tensor(4.0),
            }
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        expected_states = create_metric_states_dict(
            metric_prefix="FUSED_TASKS_COMPUTATION",
            computation_name="MockRecMetricComputation",
            metric_states={
                "cross_entropy_sum": torch.tensor(4.0),
                "weighted_num_samples": torch.tensor(6.0),
            },
        )

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)

    def test_from_metrics_fused_tasks_state_tensor_lists(self) -> None:
        """Test state tensor lists with FUSED_TASKS computation mode."""

        initial_states = {
            "predictions": [
                torch.tensor(
                    [
                        [1.0],
                        [2.0],
                    ]
                )
            ],
            "labels": [
                torch.tensor(
                    [
                        [1.0],
                        [3.0],
                    ]
                )
            ],
        }
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=gen_test_tasks(["task3", "task4"]),
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            initial_states=initial_states,
            reduction_fn=_state_reduction,
            is_tensor_list=True,
        )

        # Append to tensor lists which will be concatenated with initial states
        mock_metric.append_to_computation_states(
            {
                "predictions": torch.tensor(
                    [
                        [3.0],
                        [4.0],
                    ]
                ),
                "labels": torch.tensor(
                    [
                        [2.0],
                        [4.0],
                    ]
                ),
            }
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        expected_states = create_metric_states_dict(
            metric_prefix="FUSED_TASKS_COMPUTATION",
            computation_name="MockRecMetricComputation",
            metric_states={
                "predictions": [
                    torch.tensor(
                        [
                            [1.0, 3.0],
                            [2.0, 4.0],
                        ]
                    )
                ],
                "labels": [
                    torch.tensor(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                        ]
                    )
                ],
            },
        )
        assert_tensor_dict_equals(snapshot.metric_states, expected_states)

    def test_from_metrics_multiple_metrics(self) -> None:
        """Test creating snapshot from multiple metrics."""
        tensor_states = create_tensor_states(["test_state_tensor"])
        tensor_list_states = create_tensor_list_states(["test_state_tensor_list"])

        task1 = gen_test_tasks(["task1"])
        task2 = gen_test_tasks(["task2"])

        tensor_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=task1,
            initial_states=tensor_states,
            reduction_fn="sum",
        )

        tensor_list_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=task2,
            initial_states=tensor_list_states,
            reduction_fn=_state_reduction,
            is_tensor_list=True,
        )

        rec_metrics = RecMetricList([tensor_metric, tensor_list_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        # Verify both metrics' states are captured
        expected_tensor_states = create_metric_states_dict(
            metric_prefix="task1",
            computation_name="MockRecMetricComputation",
            metric_states=tensor_states,
        )

        expected_tensor_list_states = create_metric_states_dict(
            metric_prefix="task2",
            computation_name="MockRecMetricComputation",
            metric_states=tensor_list_states,
        )

        expected_all_states = {**expected_tensor_states, **expected_tensor_list_states}
        assert_tensor_dict_equals(snapshot.metric_states, expected_all_states)

    def test_from_metrics_multiple_tasks(self) -> None:
        """Test creating snapshot from metric with multiple tasks."""
        multi_tasks = gen_test_tasks(["task1", "task2", "task3"])
        initial_states = {"cross_entropy_sum": torch.tensor(5.5)}

        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=multi_tasks,
            initial_states=initial_states,
            reduction_fn="sum",
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        expected_states = {}
        for task in ["task1", "task2", "task3"]:
            task_states = create_metric_states_dict(
                metric_prefix=task,
                computation_name="MockRecMetricComputation",
                metric_states=initial_states,
            )
            expected_states.update(task_states)

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)

    def test_from_metrics_no_throughput_metric(self) -> None:
        """Test creating snapshot with None throughput metric."""
        mock_metric = MockRecMetric(
            world_size=self.world_size,
            my_rank=self.my_rank,
            batch_size=self.batch_size,
            tasks=self.tasks,
            initial_states={"cross_entropy_sum": torch.tensor(1.0)},
            reduction_fn="sum",
        )

        rec_metrics = RecMetricList([mock_metric])
        snapshot = MetricStateSnapshot.from_metrics(rec_metrics)

        self.assertIsNone(snapshot.throughput_metric)

        expected_states = create_metric_states_dict(
            metric_prefix="test_task",
            computation_name="MockRecMetricComputation",
            metric_states={"cross_entropy_sum": torch.tensor(1.0)},
        )

        assert_tensor_dict_equals(snapshot.metric_states, expected_states)


if __name__ == "__main__":
    unittest.main()
