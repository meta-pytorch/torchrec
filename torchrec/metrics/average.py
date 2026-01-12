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
    RecMetricException,
)


LABEL_SUM = "label_sum"
PREDICTION_SUM = "prediction_sum"
WEIGHTED_NUM_SAMPLES = "weighted_num_samples"


def compute_average(
    weighted_sum: torch.Tensor, weighted_num_samples: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        weighted_num_samples == 0.0, 0.0, weighted_sum / weighted_num_samples
    ).double()


def get_average_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {
        LABEL_SUM: torch.sum(labels * weights, dim=-1),
        PREDICTION_SUM: torch.sum(predictions * weights, dim=-1),
        WEIGHTED_NUM_SAMPLES: torch.sum(weights, dim=-1),
    }


class AverageMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Average metrics,
    including Label Average and Prediction Average.

    Label Average is the weighted average of labels.
    Prediction Average is the weighted average of predictions.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            LABEL_SUM,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            PREDICTION_SUM,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            WEIGHTED_NUM_SAMPLES,
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
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for AverageMetricComputation update"
            )
        num_samples = predictions.shape[-1]
        for state_name, state_value in get_average_states(
            labels, predictions, weights
        ).items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.LABEL_AVERAGE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_average(
                    cast(torch.Tensor, self.label_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
            ),
            MetricComputationReport(
                name=MetricName.PREDICTION_AVERAGE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_average(
                    cast(torch.Tensor, self.prediction_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
            ),
            MetricComputationReport(
                name=MetricName.LABEL_AVERAGE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_average(
                    self.get_window_state(LABEL_SUM),
                    self.get_window_state(WEIGHTED_NUM_SAMPLES),
                ),
            ),
            MetricComputationReport(
                name=MetricName.PREDICTION_AVERAGE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_average(
                    self.get_window_state(PREDICTION_SUM),
                    self.get_window_state(WEIGHTED_NUM_SAMPLES),
                ),
            ),
        ]


class AverageMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.AVERAGE
    _computation_class: Type[RecMetricComputation] = AverageMetricComputation
