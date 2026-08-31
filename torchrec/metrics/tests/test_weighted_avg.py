#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, Iterable, Optional, Type, Union

import torch
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
    RecMetricComputation,
    RecTaskInfo,
    WindowBuffer,
)
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)
from torchrec.metrics.weighted_avg import get_mean, WeightedAvgMetric


WORLD_SIZE = 4


class TestWeightedAvgMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        return {
            "weighted_sum": (predictions * weights).sum(dim=-1),
            "weighted_num_samples": weights.sum(dim=-1),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return get_mean(states["weighted_sum"], states["weighted_num_samples"])


class WeightedAvgMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = WeightedAvgMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "weighted_avg"

    def test_weighted_avg_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_weighted_avg_update_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=WeightedAvgMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestWeightedAvgMetric,
            metric_name=WeightedAvgMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
        )


def generate_model_outputs_cases() -> Iterable[Dict[str, Optional[torch.Tensor]]]:
    return [
        # random_inputs
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.6]]),
            "expected_weighted_avg": torch.tensor([0.74]),
        },
        # no weight
        {
            "labels": torch.tensor([[1, 0, 1, 0, 1, 0]]),
            "predictions": torch.tensor([[0.5] * 6]),
            "weights": None,
            "expected_weighted_avg": torch.tensor([0.5]),
        },
        # all weights are zero
        {
            "labels": torch.tensor([[1, 1, 1, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0] * 5]),
            "expected_weighted_avg": torch.tensor([float("nan")]),
        },
    ]


class WeightedAvgValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of weighted avg in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @torch.no_grad()
    def _test_weighted_avg_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_weighted_avg: torch.Tensor,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[0]
        task_list = []
        inputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {
            "predictions": {},
            "labels": {},
            "weights": {},
        }
        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
            task_list.append(task_info)
            # pyrefly: ignore[unsupported-operation]
            inputs["predictions"][task_info.name] = predictions[i]
            # pyrefly: ignore[unsupported-operation]
            inputs["labels"][task_info.name] = labels[i]
            if weights is None:
                # pyrefly: ignore[unsupported-operation]
                inputs["weights"] = None
            else:
                # pyrefly: ignore[unsupported-operation]
                inputs["weights"][task_info.name] = weights[i]

        weighted_avg = WeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
        )
        weighted_avg.update(**inputs)
        actual_weighted_avg = weighted_avg.compute()

        for task_id, task in enumerate(task_list):
            cur_actual_weighted_avg = actual_weighted_avg[
                f"weighted_avg-{task.name}|window_weighted_avg"
            ]
            cur_expected_weighted_avg = expected_weighted_avg[task_id].unsqueeze(dim=0)
            if cur_expected_weighted_avg.isnan().any():
                self.assertTrue(cur_actual_weighted_avg.isnan().any())
            else:
                torch.testing.assert_close(
                    cur_actual_weighted_avg,
                    cur_expected_weighted_avg,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    msg=f"Actual: {cur_actual_weighted_avg}, Expected: {cur_expected_weighted_avg}",
                )

    def test_weighted_avg(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                # pyrefly: ignore[bad-argument-type]
                self._test_weighted_avg_helper(**inputs)
            except AssertionError:
                print("Assertion error caught with data set ", inputs)
                raise


def _window_buffer(metric: RecMetric, state_name: str) -> WindowBuffer:
    """The `WindowBuffer` backing one window state. No public accessor exists."""
    computation = cast(RecMetricComputation, metric._metrics_computations[0])
    buffers = computation._batch_window_buffers
    assert buffers is not None, "window metrics are disabled on this metric"
    return buffers[state_name]


class WindowSampleAccountingTest(unittest.TestCase):
    """Window eviction must count SAMPLES, not `update()` calls.

    ``RecMetricComputation._aggregate_window_state`` passes ``num_samples`` to
    ``WindowBuffer`` as the entry's ``size``, and the buffer evicts while
    ``_window_used_size > window_size``. In unfused mode ``labels`` is
    ``(n_tasks, n_samples)`` with ``n_tasks == 1``, so reading ``shape[0]``
    reports 1 sample per update regardless of batch size: a window whose
    ``window_size`` exceeds the update count then never evicts, and
    ``window_*`` covers an unbounded span of samples.

    Sibling metrics (``ne.py``, ``gauc.py``) already read ``shape[-1]``.
    """

    _BATCH = 100
    _WINDOW = 250  # 2 batches fit, the 3rd forces an eviction
    _UPDATES = 5

    def _run(
        self,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        n_tasks: int = 1,
        actual_batch: Optional[int] = None,
    ) -> tuple[RecMetric, int, int]:
        """Drive `_UPDATES` updates and return (metric, buffer entries, used samples).

        ``actual_batch`` decouples the tensor length actually fed from the ``batch_size``
        the metric is CONSTRUCTED with -- the divergence that `fused_update_limit` and
        batch-size stages produce in production.
        """
        tasks = [
            RecTaskInfo(
                name=f"t{i}",
                label_name=f"label{i}",
                prediction_name=f"prediction{i}",
                weight_name=f"weight{i}",
            )
            for i in range(n_tasks)
        ]
        metric = WeightedAvgMetric(
            world_size=1,
            my_rank=0,
            batch_size=self._BATCH,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=self._WINDOW,
        )
        fed = actual_batch if actual_batch is not None else self._BATCH
        for _ in range(self._UPDATES):
            metric.update(
                predictions={t.name: torch.rand(1, fed) for t in tasks},
                labels={t.name: torch.rand(1, fed) for t in tasks},
                weights={t.name: torch.ones(1, fed) for t in tasks},
            )
        buf = _window_buffer(metric, "window_weighted_sum")
        return metric, len(buf.buffers), buf._window_used_size

    def test_window_evicts_on_sample_count_not_update_count(self) -> None:
        _, n_entries, used = self._run()
        # samples: 5 x 100 = 500 > 250, so the window holds the newest 2 batches.
        # Reading shape[0] would report 1 sample/update -> used=5 <= 250 -> all 5 kept.
        self.assertEqual(n_entries, 2)
        self.assertEqual(used, 2 * self._BATCH)

    def test_window_evicts_on_sample_count_fused(self) -> None:
        """FUSED mode: `shape[0]` is n_tasks, not 1 -- still not the sample count.

        The unfused test alone cannot distinguish "reads shape[-1]" from "reads 1", since
        shape[0] happens to BE 1 there. Here shape[0] == 3, so a regression to shape[0]
        reports 3 samples/update -> used=15 <= 250 -> all 5 entries kept.
        """
        _, n_entries, used = self._run(
            compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION, n_tasks=3
        )
        self.assertEqual(n_entries, 2)
        self.assertEqual(used, 2 * self._BATCH)

    def test_oversized_update_keeps_one_entry_instead_of_emptying_window(self) -> None:
        """An update larger than the whole window must not empty the window.

        `WindowBuffer._aggregate_state_impl` evicts while `_window_used_size > _max_size`.
        Without the `len(self._buffers) > 1` guard the entry evicts itself, leaving
        `window_state == 0` and publishing `window_weighted_avg = 0/0 = NaN`.

        `RecMetric.__init__` rejects `window_size_local < batch_size`, but that check uses
        the CONFIGURED batch size while the buffer sizes by the ACTUAL tensor length; the
        two diverge under `fused_update_limit` (F concatenated batches) and under
        batch-size stages (which `RecMetric` discards). Reproduced here by feeding a
        longer tensor than the metric was constructed with.
        """
        metric, n_entries, used = self._run(actual_batch=self._WINDOW + 50)
        self.assertEqual(n_entries, 1)
        self.assertEqual(used, self._WINDOW + 50)
        window_value = metric.compute()["weighted_avg-t0|window_weighted_avg"]
        self.assertFalse(
            torch.isnan(window_value).any(),
            f"window_weighted_avg went NaN on an oversized update: {window_value}",
        )
