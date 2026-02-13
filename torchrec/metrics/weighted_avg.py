#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
)


def get_mean(value_sum: torch.Tensor, num_samples: torch.Tensor) -> torch.Tensor:
    return value_sum / num_samples


class WeightedAvgMetricComputation(RecMetricComputation):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "weighted_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "weighted_num_samples",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        num_samples = labels.shape[0]
        predictions = cast(torch.Tensor, predictions)
        weights = cast(torch.Tensor, weights)
        states = {
            "weighted_sum": (predictions * weights).sum(dim=-1),
            "weighted_num_samples": weights.sum(dim=-1),
        }
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.LIFETIME,
                value=get_mean(
                    cast(torch.Tensor, self.weighted_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
            ),
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.WINDOW,
                value=get_mean(
                    self.get_window_state("weighted_sum"),
                    self.get_window_state("weighted_num_samples"),
                ),
            ),
        ]


class WeightedAvgMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.WEIGHTED_AVG
    _computation_class: Type[RecMetricComputation] = WeightedAvgMetricComputation
