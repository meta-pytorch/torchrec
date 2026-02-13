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


TARGET_PRECISION = "target_precision"
THRESHOLD_GRANULARITY = 1000


def compute_precision(
    num_true_positives: torch.Tensor, num_false_positives: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        num_true_positives + num_false_positives == 0.0,
        0.0,
        num_true_positives / (num_true_positives + num_false_positives).double(),
    )


def compute_recall(
    num_true_positives: torch.Tensor, num_false_negitives: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        num_true_positives + num_false_negitives == 0.0,
        0.0,
        num_true_positives / (num_true_positives + num_false_negitives),
    )


def compute_threshold_idx(
    num_true_positives: torch.Tensor,
    num_false_positives: torch.Tensor,
    target_precision: float,
) -> int:
    for i in range(THRESHOLD_GRANULARITY):
        if (
            compute_precision(num_true_positives[i], num_false_positives[i])
            >= target_precision
        ):
            return i

    return THRESHOLD_GRANULARITY - 1


def compute_true_pos_sum(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predictions = predictions.double()
    tp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
    thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
    for i, threshold in enumerate(thresholds):
        tp_sum[i] = torch.sum(weights * ((predictions >= threshold) * labels), -1)
    return tp_sum


def compute_false_pos_sum(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predictions = predictions.double()
    fp_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
    thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
    for i, threshold in enumerate(thresholds):
        fp_sum[i] = torch.sum(weights * ((predictions >= threshold) * (1 - labels)), -1)
    return fp_sum


def compute_false_neg_sum(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    predictions = predictions.double()
    fn_sum = torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double)
    thresholds = torch.linspace(0, 1, steps=THRESHOLD_GRANULARITY)
    for i, threshold in enumerate(thresholds):
        fn_sum[i] = torch.sum(weights * ((predictions <= threshold) * labels), -1)
    return fn_sum


def get_pr_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = torch.ones_like(predictions)
    return {
        "true_pos_sum": compute_true_pos_sum(labels, predictions, weights),
        "false_pos_sum": compute_false_pos_sum(labels, predictions, weights),
        "false_neg_sum": compute_false_neg_sum(labels, predictions, weights),
    }


class HindsightTargetPRMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Hingsight Target PR.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.

    Args:
        target_precision (float): If provided, computes the minimum threshold to achieve the target precision.
    """

    def __init__(
        self, *args: Any, target_precision: float = 0.5, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "true_pos_sum",
            torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "false_pos_sum",
            torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "false_neg_sum",
            torch.zeros(THRESHOLD_GRANULARITY, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._target_precision: float = target_precision

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None:
            raise RecMetricException(
                "Inputs 'predictions' should not be None for HindsightTargetPRMetricComputation update"
            )
        states = get_pr_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        true_pos_sum = cast(torch.Tensor, self.true_pos_sum)
        false_pos_sum = cast(torch.Tensor, self.false_pos_sum)
        false_neg_sum = cast(torch.Tensor, self.false_neg_sum)
        threshold_idx = compute_threshold_idx(
            true_pos_sum,
            false_pos_sum,
            self._target_precision,
        )
        window_threshold_idx = compute_threshold_idx(
            self.get_window_state("true_pos_sum"),
            self.get_window_state("false_pos_sum"),
            self._target_precision,
        )
        reports = [
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_PR,
                metric_prefix=MetricPrefix.LIFETIME,
                value=torch.Tensor(threshold_idx),
            ),
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_PR,
                metric_prefix=MetricPrefix.WINDOW,
                value=torch.Tensor(window_threshold_idx),
            ),
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_PRECISION,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_precision(
                    true_pos_sum[threshold_idx],
                    false_pos_sum[threshold_idx],
                ),
            ),
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_PRECISION,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_precision(
                    self.get_window_state("true_pos_sum")[window_threshold_idx],
                    self.get_window_state("false_pos_sum")[window_threshold_idx],
                ),
            ),
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_RECALL,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_recall(
                    true_pos_sum[threshold_idx],
                    false_neg_sum[threshold_idx],
                ),
            ),
            MetricComputationReport(
                name=MetricName.HINDSIGHT_TARGET_RECALL,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_recall(
                    self.get_window_state("true_pos_sum")[window_threshold_idx],
                    self.get_window_state("false_neg_sum")[window_threshold_idx],
                ),
            ),
        ]
        return reports


class HindsightTargetPRMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.HINDSIGHT_TARGET_PR
    _computation_class: Type[RecMetricComputation] = HindsightTargetPRMetricComputation
