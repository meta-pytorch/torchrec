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


def compute_cross_entropy_positive(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    predictions = predictions.double()
    predictions.clamp_(min=eta, max=1 - eta)
    cross_entropy_positive = -weights * labels * torch.log2(predictions)
    return cross_entropy_positive


def _compute_cross_entropy_norm(
    mean_label: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    mean_label = mean_label.double()
    mean_label.clamp_(min=eta, max=1 - eta)
    return -pos_labels * torch.log2(mean_label) - neg_labels * torch.log2(
        1.0 - mean_label
    )


@torch.fx.wrap
def compute_ne_positive(
    ce_positive_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
    allow_missing_label_with_zero_weight: bool = False,
) -> torch.Tensor:
    if allow_missing_label_with_zero_weight and not weighted_num_samples.all():
        # If nan were to occur, return a dummy value instead of nan if
        # allow_missing_label_with_zero_weight is True
        return torch.tensor([eta])

    # Goes into this block if all elements in weighted_num_samples > 0
    weighted_num_samples = weighted_num_samples.double().clamp(min=eta)
    mean_label = pos_labels / weighted_num_samples
    ce_norm = _compute_cross_entropy_norm(mean_label, pos_labels, neg_labels, eta)
    return ce_positive_sum / ce_norm


def get_ne_positive_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor, eta: float
) -> Dict[str, torch.Tensor]:
    cross_entropy_positive = compute_cross_entropy_positive(
        labels,
        predictions,
        weights,
        eta,
    )
    return {
        "cross_entropy_positive_sum": torch.sum(cross_entropy_positive, dim=-1),
        "weighted_num_samples": torch.sum(weights, dim=-1),
        "pos_labels": torch.sum(weights * labels, dim=-1),
        "neg_labels": torch.sum(weights * (1.0 - labels), dim=-1),
    }


class NEPositiveMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for NE positive, i.e. Normalized Entropy where label = 1

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self,
        *args: Any,
        allow_missing_label_with_zero_weight: bool = False,
        **kwargs: Any,
    ) -> None:
        self._allow_missing_label_with_zero_weight: bool = (
            allow_missing_label_with_zero_weight
        )
        super().__init__(*args, **kwargs)
        self._add_state(
            "cross_entropy_positive_sum",
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
            "pos_labels",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "neg_labels",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self.eta = 1e-12

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
                "Inputs 'predictions' and 'weights' should not be None for NEMetricComputation update"
            )
        states = get_ne_positive_states(labels, predictions, weights, self.eta)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.NE_POSITIVE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_ne_positive(
                    cast(torch.Tensor, self.cross_entropy_positive_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                    cast(torch.Tensor, self.pos_labels),
                    cast(torch.Tensor, self.neg_labels),
                    self.eta,
                    self._allow_missing_label_with_zero_weight,
                ),
            ),
            MetricComputationReport(
                name=MetricName.NE_POSITIVE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_ne_positive(
                    self.get_window_state("cross_entropy_positive_sum"),
                    self.get_window_state("weighted_num_samples"),
                    self.get_window_state("pos_labels"),
                    self.get_window_state("neg_labels"),
                    self.eta,
                    self._allow_missing_label_with_zero_weight,
                ),
            ),
        ]
        return reports


class NEPositiveMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.NE_POSITIVE
    _computation_class: Type[RecMetricComputation] = NEPositiveMetricComputation
