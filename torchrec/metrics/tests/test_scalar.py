#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import RecMetric, RecMetricComputation, WindowBuffer
from torchrec.metrics.scalar import ScalarMetric


WORLD_SIZE = 4
BATCH_SIZE = 10


class ScalarMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scalar = ScalarMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
        )

    def test_scalar(self) -> None:
        """
        Test scalar metric passes through each tensor as is
        """
        metric_to_log = torch.tensor([0.1])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )
        metric = self.scalar.compute()
        actual_metric = metric[f"scalar-{DefaultTaskInfo.name}|lifetime_scalar"]

        torch.testing.assert_close(
            actual_metric,
            metric_to_log,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {metric_to_log}",
        )

        # Pass through second tensor with different value
        # check we get the value back with no averaging or any differences

        metric_to_log = torch.tensor([0.9])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )
        metric = self.scalar.compute()
        actual_metric = metric[f"scalar-{DefaultTaskInfo.name}|lifetime_scalar"]

        torch.testing.assert_close(
            actual_metric,
            metric_to_log,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {metric_to_log}",
        )

    def test_scalar_window(self) -> None:
        """
        Test windowing of scalar metric gives average of previously reported values.
        """
        metric_to_log = torch.tensor([0.1])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )

        metric_to_log = torch.tensor([0.9])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )

        metric = self.scalar.compute()

        actual_window_metric = metric[f"scalar-{DefaultTaskInfo.name}|window_scalar"]

        expected_window_metric = torch.tensor([0.5])

        torch.testing.assert_close(
            actual_window_metric,
            expected_window_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_window_metric}, Expected: {expected_window_metric}",
        )


def _window_buffer(metric: RecMetric, state_name: str) -> WindowBuffer:
    """The `WindowBuffer` backing one window state. No public accessor exists."""
    computation = cast(RecMetricComputation, metric._metrics_computations[0])
    buffers = computation._batch_window_buffers
    assert buffers is not None, "window metrics are disabled on this metric"
    return buffers[state_name]


class WindowSampleAccountingTest(unittest.TestCase):
    """Window eviction must count SAMPLES, not `update()` calls.

    `ScalarMetricComputation.update` passes `num_samples` to `WindowBuffer` as the
    entry's `size`, and the buffer evicts while `_window_used_size > window_size`. In
    unfused mode `labels` is `(n_tasks, n_samples)` with `n_tasks == 1`, so reading
    `shape[0]` reports 1 sample per update regardless of batch size, and `window_scalar`
    covers an unbounded span.

    Sibling of the identical test in `test_weighted_avg.py` -- the three metrics that
    carried this bug are fixed atomically, so each one is pinned separately. Without a
    test here, `scalar.py` could silently regress to `shape[0]`.
    """

    _BATCH = 100
    _WINDOW = 250  # 2 batches fit, the 3rd forces an eviction
    _UPDATES = 5

    def test_window_evicts_on_sample_count_not_update_count(self) -> None:
        metric = ScalarMetric(
            world_size=1,
            my_rank=0,
            batch_size=self._BATCH,
            tasks=[DefaultTaskInfo],
            window_size=self._WINDOW,
        )
        for _ in range(self._UPDATES):
            value = torch.rand(1, self._BATCH)
            metric.update(
                labels={DefaultTaskInfo.name: value},
                predictions={DefaultTaskInfo.name: value},
                weights={DefaultTaskInfo.name: value},
            )
        buf = _window_buffer(metric, "window_labels")
        # samples: 5 x 100 = 500 > 250, so the window holds the newest 2 batches.
        # Reading shape[0] would report 1 sample/update -> used=5 <= 250 -> all 5 kept.
        self.assertEqual(len(buf.buffers), 2)
        self.assertEqual(buf._window_used_size, 2 * self._BATCH)
