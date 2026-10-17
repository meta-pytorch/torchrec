#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, List, Tuple

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricPrefix
from torchrec.metrics.ne import NEMetricComputation

WINDOW = 1000
WIDTH = 128
STEPS = 60


def _batches(seed: int) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Real columns per step vary; the label rate drifts, so windows that cover
    different spans of steps report different NE."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for step in range(STEPS):
        n = int(torch.randint(100, 121, (1,), generator=gen))
        rate = 0.1 + 0.3 * step / STEPS
        labels = (torch.rand(1, n, generator=gen) < rate).double()
        preds = (rate + 0.05 * torch.randn(1, n, generator=gen)).clamp(0.01, 0.99)
        weights = torch.rand(1, n, generator=gen).double() + 0.5
        out.append((n, preds.double(), labels, weights))
    return out


def _pad(t: torch.Tensor, value: float) -> torch.Tensor:
    return torch.cat([t, torch.full((1, WIDTH - t.size(1)), value, dtype=t.dtype)], 1)


def _ne(metric: NEMetricComputation) -> Dict[MetricPrefix, torch.Tensor]:
    return {
        r.metric_prefix: r.value for r in metric._compute() if r.name == MetricName.NE
    }


def _run(pad: bool, pass_real_count: bool) -> Dict[MetricPrefix, torch.Tensor]:
    metric = NEMetricComputation(
        my_rank=0, batch_size=WIDTH, n_tasks=1, window_size=WINDOW
    )
    for n, preds, labels, weights in _batches(seed=0):
        if pad:
            preds, labels, weights = _pad(preds, 0.5), _pad(labels, 0), _pad(weights, 0)
        kwargs: Dict[str, Any] = {"num_samples": n} if pass_real_count else {}
        metric.update(predictions=preds, labels=labels, weights=weights, **kwargs)
    return _ne(metric)


class WindowNumSamplesTest(unittest.TestCase):
    def test_zero_weight_padding_keeps_lifetime_exact(self) -> None:
        real = _run(pad=False, pass_real_count=False)
        padded = _run(pad=True, pass_real_count=False)
        torch.testing.assert_close(
            padded[MetricPrefix.LIFETIME], real[MetricPrefix.LIFETIME]
        )

    def test_padded_width_shrinks_the_window(self) -> None:
        # Counting pad columns spends the window budget faster, so fewer real
        # batches stay in it and window NE covers a different span.
        real = _run(pad=False, pass_real_count=False)
        padded = _run(pad=True, pass_real_count=False)
        gap = (padded[MetricPrefix.WINDOW] - real[MetricPrefix.WINDOW]).abs().item()
        self.assertGreater(gap, 1e-4)

    def test_real_count_restores_the_window(self) -> None:
        real = _run(pad=False, pass_real_count=False)
        fixed = _run(pad=True, pass_real_count=True)
        torch.testing.assert_close(
            fixed[MetricPrefix.WINDOW], real[MetricPrefix.WINDOW]
        )
        torch.testing.assert_close(
            fixed[MetricPrefix.LIFETIME], real[MetricPrefix.LIFETIME]
        )
