#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Any, cast, Dict, List, Optional, Type

import torch
import torch.nn.functional as F
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


@torch.no_grad()
def compute_error_sum(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if not labels.is_floating_point():
        labels = labels.double()
    is_nan_sample = labels.isnan() | predictions.isnan()
    labels = labels.masked_fill(is_nan_sample, 0)
    predictions = predictions.masked_fill(is_nan_sample, 0)
    weights = weights.double().masked_fill(is_nan_sample, 0)

    order = _get_indices_sorted_by_prediction_asc_label_desc(
        labels=labels, predictions=predictions
    )
    sorted_labels = labels.gather(-1, order)
    sorted_predictions = predictions.gather(-1, order)
    sorted_weights = weights.gather(-1, order)

    weighted_num_strictly_increasing_label_pairs = (
        _get_weighted_num_strictly_increasing_label_pairs(
            labels=sorted_labels, weights=sorted_weights
        )
    )
    weighted_num_prediction_and_label_tied_pairs = (
        _get_weighted_num_prediction_and_label_tied_pairs(
            labels=sorted_labels, predictions=sorted_predictions, weights=sorted_weights
        )
    )
    return (
        weighted_num_strictly_increasing_label_pairs
        + weighted_num_prediction_and_label_tied_pairs
    )


def _get_indices_sorted_by_prediction_asc_label_desc(
    *, labels: torch.Tensor, predictions: torch.Tensor
) -> torch.Tensor:
    label_order = labels.argsort(dim=-1, descending=True)
    prediction_order = predictions.gather(-1, label_order).argsort(dim=-1, stable=True)
    return label_order.gather(-1, prediction_order)


def _get_weighted_num_strictly_increasing_label_pairs(
    *, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    num_samples = labels.shape[-1]
    block_size = math.ceil(math.sqrt(num_samples))
    labels = F.pad(labels, [0, block_size**2 - num_samples])
    weights = F.pad(weights, [0, block_size**2 - num_samples])
    return _get_weighted_num_strictly_increasing_label_pairs_across_blocks(
        labels=labels, weights=weights, block_size=block_size
    ) + _get_weighted_num_strictly_increasing_label_pairs_within_blocks(
        block_labels=labels.unflatten(-1, (block_size, block_size)),
        block_weights=weights.unflatten(-1, (block_size, block_size)),
    )


def _get_weighted_num_strictly_increasing_label_pairs_across_blocks(
    *, labels: torch.Tensor, weights: torch.Tensor, block_size: int
) -> torch.Tensor:
    sorted_block_labels, block_label_order = labels.unflatten(
        -1, (block_size, block_size)
    ).sort(dim=-1)
    block_weight_prefix_sums = F.pad(
        weights.unflatten(-1, (block_size, block_size))
        .gather(-1, block_label_order)
        .cumsum(-1),
        [1, 0],
    )

    num_smaller_labels_per_block = torch.searchsorted(
        sorted_block_labels, labels.unsqueeze(1).repeat(1, block_size, 1)
    )
    smaller_label_weight_per_block = block_weight_prefix_sums.gather(
        -1, num_smaller_labels_per_block
    )

    block_indices = torch.arange(block_size, device=labels.device)
    is_earlier_block = block_indices.unsqueeze(-1) < block_indices.repeat_interleave(
        block_size
    )
    strictly_increasing_label_pair_weights = torch.where(
        is_earlier_block, smaller_label_weight_per_block, 0
    ) * weights.unsqueeze(1)
    return strictly_increasing_label_pair_weights.sum(-1).sum(-1)


def _get_weighted_num_strictly_increasing_label_pairs_within_blocks(
    *, block_labels: torch.Tensor, block_weights: torch.Tensor
) -> torch.Tensor:
    is_strictly_increasing_pair = torch.triu(
        block_labels.unsqueeze(-1) < block_labels.unsqueeze(-2), diagonal=1
    )
    pair_weights = block_weights.unsqueeze(-1) * block_weights.unsqueeze(-2)
    strictly_increasing_pair_weights = torch.where(
        is_strictly_increasing_pair, pair_weights, 0
    )
    return strictly_increasing_pair_weights.sum(-1).flatten(1).sum(-1)


def _get_weighted_num_prediction_and_label_tied_pairs(
    *, labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    is_new_tie_group = torch.ones_like(labels, dtype=torch.bool)
    is_new_tie_group[:, 1:] = (labels[:, 1:] != labels[:, :-1]) | (
        predictions[:, 1:] != predictions[:, :-1]
    )
    positions = torch.arange(labels.shape[-1], device=labels.device)
    tie_group_starts = torch.where(is_new_tie_group, positions, 0).cummax(-1).values

    earlier_weight_sums = weights.cumsum(-1) - weights
    earlier_tied_weight_sums = earlier_weight_sums - earlier_weight_sums.gather(
        -1, tie_group_starts
    )
    return (weights * earlier_tied_weight_sums).sum(-1)


def compute_weighted_num_pairs(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.double()
    return (weights.sum(-1).square() - weights.square().sum(-1)) / 2


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
        self._get_xauc_states = self._maybe_compile(get_xauc_states)

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
        states = self._get_xauc_states(labels, predictions, weights)
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
