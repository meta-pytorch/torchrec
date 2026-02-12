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
WEIGHTED_NUM_PAIRS = "weighted_num_pairs"


def compute_xauc(
    error_sum: torch.Tensor, weighted_num_pairs: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        weighted_num_pairs == 0.0, 0.0, error_sum / weighted_num_pairs
    ).double()


def compute_error_sum(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    predictions = predictions.double()

    errors = []
    for predictions_i, labels_i, weights_i in zip(predictions, labels, weights):
        preds_x, preds_y = torch.meshgrid(predictions_i, predictions_i)
        labels_x, labels_y = torch.meshgrid(labels_i, labels_i)
        weights_x, weights_y = torch.meshgrid(weights_i, weights_i)
        weights_flag = weights_x * weights_y
        match = torch.logical_or(
            torch.logical_and(preds_x > preds_y, labels_x > labels_y),
            torch.logical_and(preds_x < preds_y, labels_x < labels_y),
        )
        match = (
            weights_flag
            * torch.logical_or(
                match, torch.logical_and(preds_x == preds_y, labels_x == labels_y)
            ).double()
        )
        errors.append(torch.sum(torch.triu(match, diagonal=1)).view(1))

    return torch.cat(errors)


def compute_weighted_num_pairs(weights: torch.Tensor) -> torch.Tensor:
    num_pairs = []
    for weight_i in weights:
        weights_x, weights_y = torch.meshgrid(weight_i, weight_i)
        weights_flag = weights_x * weights_y
        num_pairs.append(torch.sum(torch.triu(weights_flag, diagonal=1)).view(1))

    return torch.cat(num_pairs)


def get_xauc_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {
        "error_sum": compute_error_sum(labels, predictions, weights),
        "weighted_num_pairs": compute_weighted_num_pairs(weights),
    }


class XAUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for XAUC.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "error_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "weighted_num_pairs",
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
                "Inputs 'predictions' and 'weights' should not be None for XAUCMetricComputation update"
            )
        states = get_xauc_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.XAUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_xauc(
                    cast(torch.Tensor, self.error_sum),
                    cast(torch.Tensor, self.weighted_num_pairs),
                ),
            ),
            MetricComputationReport(
                name=MetricName.XAUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_xauc(
                    self.get_window_state(ERROR_SUM),
                    self.get_window_state(WEIGHTED_NUM_PAIRS),
                ),
            ),
        ]


class XAUCMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.XAUC
    _computation_class: Type[RecMetricComputation] = XAUCMetricComputation
