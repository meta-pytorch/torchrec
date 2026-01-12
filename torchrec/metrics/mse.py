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


ERROR_SUM = "error_sum"
WEIGHTED_NUM_SAMPLES = "weighted_num_samples"
LABEL_SUM = "label_sum"
LABEL_SQUARED_SUM = "label_squared_sum"


def compute_mse(
    error_sum: torch.Tensor, weighted_num_samples: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        weighted_num_samples == 0.0, 0.0, error_sum / weighted_num_samples
    ).double()


def compute_rmse(
    error_sum: torch.Tensor, weighted_num_samples: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        weighted_num_samples == 0.0, 0.0, torch.sqrt(error_sum / weighted_num_samples)
    ).double()


def compute_r_squared(
    error_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    label_sum: torch.Tensor,
    label_squared_sum: torch.Tensor,
) -> torch.Tensor:
    total_sum = (
        label_squared_sum.double()
        - torch.square(label_sum.double()) / weighted_num_samples.double()
    )
    return torch.where(total_sum == 0.0, 1.0, 1.0 - error_sum / total_sum).double()


def compute_error_sum(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    predictions = predictions.double()
    return torch.sum(weights * torch.square(labels - predictions), dim=-1)


def get_mse_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        "error_sum": compute_error_sum(labels, predictions, weights),
        "weighted_num_samples": torch.sum(weights, dim=-1),
        "label_sum": torch.sum(weights * labels, dim=-1),
        "label_squared_sum": torch.sum(weights * torch.square(labels), dim=-1),
    }


class MSEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for MSE, i.e. Mean Squared Error.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self,
        *args: Any,
        include_r_squared: bool = False,
        **kwargs: Any,
    ) -> None:
        self._include_r_squared: bool = include_r_squared
        super().__init__(*args, **kwargs)
        self._add_state(
            "error_sum",
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
        self._add_state(
            "label_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=include_r_squared,
        )
        self._add_state(
            "label_squared_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=include_r_squared,
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
                "Inputs 'predictions' and 'weights' should not be None for MSEMetricComputation update"
            )
        states = get_mse_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.MSE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_mse(
                    cast(torch.Tensor, self.error_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
            ),
            MetricComputationReport(
                name=MetricName.RMSE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_rmse(
                    cast(torch.Tensor, self.error_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
            ),
            MetricComputationReport(
                name=MetricName.MSE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_mse(
                    self.get_window_state(ERROR_SUM),
                    self.get_window_state(WEIGHTED_NUM_SAMPLES),
                ),
            ),
            MetricComputationReport(
                name=MetricName.RMSE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_rmse(
                    self.get_window_state(ERROR_SUM),
                    self.get_window_state(WEIGHTED_NUM_SAMPLES),
                ),
            ),
        ]

        if self._include_r_squared:
            reports += [
                MetricComputationReport(
                    name=MetricName.R_SQUARED,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=compute_r_squared(
                        cast(torch.Tensor, self.error_sum),
                        cast(torch.Tensor, self.weighted_num_samples),
                        cast(torch.Tensor, self.label_sum),
                        cast(torch.Tensor, self.label_squared_sum),
                    ),
                ),
                MetricComputationReport(
                    name=MetricName.R_SQUARED,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_r_squared(
                        self.get_window_state(ERROR_SUM),
                        self.get_window_state(WEIGHTED_NUM_SAMPLES),
                        self.get_window_state(LABEL_SUM),
                        self.get_window_state(LABEL_SQUARED_SUM),
                    ),
                ),
            ]

        return reports


class MSEMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.MSE
    _computation_class: Type[RecMetricComputation] = MSEMetricComputation
