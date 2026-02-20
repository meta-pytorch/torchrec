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


def compute_cross_entropy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    predictions = predictions.double()
    predictions.clamp_(min=eta, max=1 - eta)
    labels = labels.double()
    labels.clamp_(min=0.0, max=1.0)
    cross_entropy = -weights * labels * torch.log2(predictions) - weights * (
        1.0 - labels
    ) * torch.log2(1.0 - predictions)
    return cross_entropy


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
def compute_ne(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
    allow_missing_label_with_zero_weight: bool = False,
) -> torch.Tensor:
    clamped_weighted_num_samples = weighted_num_samples.double().clamp(min=eta)
    mean_label = pos_labels / clamped_weighted_num_samples
    ce_norm = _compute_cross_entropy_norm(mean_label, pos_labels, neg_labels, eta)
    ne = ce_sum / ce_norm
    if allow_missing_label_with_zero_weight and not weighted_num_samples.all():
        # If inf were to occur, return a dummy value instead.
        return torch.where(weighted_num_samples > 0, ne, eta)
    return ne


def compute_logloss(
    ce_sum: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    labels_sum = pos_labels + neg_labels
    labels_sum.clamp_(min=eta)
    return ce_sum / labels_sum


def get_ne_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor, eta: float
) -> Dict[str, torch.Tensor]:
    labels = labels.double().clamp(min=0.0, max=1.0)
    cross_entropy = compute_cross_entropy(
        labels,
        predictions,
        weights,
        eta,
    )
    return {
        "cross_entropy_sum": torch.sum(cross_entropy, dim=-1),
        "weighted_num_samples": torch.sum(weights, dim=-1),
        "pos_labels": torch.sum(weights * labels, dim=-1),
        "neg_labels": torch.sum(weights * (1.0 - labels), dim=-1),
    }


class NEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for NE, i.e. Normalized Entropy.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.

    Args:
        include_logloss (bool): return vanilla logloss as one of metrics results, on top of NE.
    """

    def __init__(
        self,
        *args: Any,
        include_logloss: bool = False,
        allow_missing_label_with_zero_weight: bool = False,
        **kwargs: Any,
    ) -> None:
        self._include_logloss: bool = include_logloss
        self._allow_missing_label_with_zero_weight: bool = (
            allow_missing_label_with_zero_weight
        )
        super().__init__(*args, **kwargs)
        self._add_state(
            "cross_entropy_sum",
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
        states = get_ne_states(labels, predictions, weights, self.eta)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.NE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_ne(
                    cast(torch.Tensor, self.cross_entropy_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                    cast(torch.Tensor, self.pos_labels),
                    cast(torch.Tensor, self.neg_labels),
                    self.eta,
                    self._allow_missing_label_with_zero_weight,
                ),
            ),
            MetricComputationReport(
                name=MetricName.NE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_ne(
                    self.get_window_state("cross_entropy_sum"),
                    self.get_window_state("weighted_num_samples"),
                    self.get_window_state("pos_labels"),
                    self.get_window_state("neg_labels"),
                    self.eta,
                    self._allow_missing_label_with_zero_weight,
                ),
            ),
        ]
        if self._include_logloss:
            reports += [
                MetricComputationReport(
                    name=MetricName.LOG_LOSS,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=compute_logloss(
                        cast(torch.Tensor, self.cross_entropy_sum),
                        cast(torch.Tensor, self.pos_labels),
                        cast(torch.Tensor, self.neg_labels),
                        self.eta,
                    ),
                ),
                MetricComputationReport(
                    name=MetricName.LOG_LOSS,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_logloss(
                        self.get_window_state("cross_entropy_sum"),
                        self.get_window_state("pos_labels"),
                        self.get_window_state("neg_labels"),
                        self.eta,
                    ),
                ),
            ]
        return reports


class NEMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.NE
    _computation_class: Type[RecMetricComputation] = NEMetricComputation
