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


def compute_weighted_sum(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(weights, dim=-1)


def get_weighted_sum_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = torch.ones_like(labels)
    return {
        "weighted_sum": compute_weighted_sum(labels, predictions, weights),
    }


class SumWeightsMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for sum of weights.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "weighted_sum",
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
        states = get_weighted_sum_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.SUM_WEIGHTS,
                metric_prefix=MetricPrefix.LIFETIME,
                value=cast(torch.Tensor, self.weighted_sum),
            ),
            MetricComputationReport(
                name=MetricName.SUM_WEIGHTS,
                metric_prefix=MetricPrefix.WINDOW,
                value=self.get_window_state("weighted_sum"),
            ),
        ]
        return reports


class SumWeightsMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.SUM_WEIGHTS
    _computation_class: Type[RecMetricComputation] = SumWeightsMetricComputation
