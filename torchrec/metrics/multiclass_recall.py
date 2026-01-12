#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, cast, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_config import RecComputeMode
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


logger: logging.Logger = logging.getLogger(__name__)


def compute_true_positives_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """
    Compute and return a list of weighted true positives (true predictions) at k. When k = 0,
    tp is counted when the 1st predicted class matches the label. When k = 1, tp is counted
    when either the 1st or 2nd predicted class matches the label.

    Args:
        predictions (Tensor): Tensor of label predictions with shape of (n_sample, n_class) or (n_task, n_sample, n_class).
        labels (Tensor): Tensor of ground truth labels with shape of (n_sample, ) or (n_task, n_sample).
        weights (Tensor): Tensor of weight on each sample, with shape of (n_sample, ) or (n_task, n_sample).
        n_classes (int): Number of classes.

    Output:
        true_positives_list (Tensor): Tensor of true positives with shape of (n_class, ) or (n_task, n_class).

    Examples:

        >>> predictions = torch.tensor([[0.9, 0.1, 0, 0, 0], [0.1, 0.2, 0.25, 0.15, 0.3], [0, 1.0, 0, 0, 0], [0, 0, 0.2, 0.7, 0.1]])
        >>> labels = torch.tensor([0, 3, 1, 2])
        >>> weights = torch.tensor([1, 0.25, 0.5, 0.25])
        >>> n_classes = 5
        >>> true_positives_list = compute_multiclass_k_sum(predictions, labels, n_classes)
        >>> true_positives_list
        tensor([1.5000, 1.7500, 1.7500, 2.0000, 2.0000])

    """
    ranks = torch.argsort(predictions, dim=-1, descending=True)
    true_positives = (
        torch.zeros(1, device=predictions.device)
        if predictions.ndim == 2
        else torch.zeros(predictions.shape[0], 1, device=predictions.device)
    )
    true_positives_list = torch.tensor([], device=predictions.device)

    for k in range(n_classes):
        mask = torch.unsqueeze(labels, dim=-1) == ranks[..., k : k + 1]
        mask = mask * torch.unsqueeze(weights, dim=-1)
        true_positives += mask.sum(dim=-2)
        true_positives_list = torch.cat((true_positives_list, true_positives), dim=-1)

    return true_positives_list


def compute_multiclass_recall_at_k(
    tp_at_k: torch.Tensor,
    total_weights: torch.Tensor,
) -> torch.Tensor:
    return tp_at_k / torch.unsqueeze(total_weights, dim=-1)


def get_multiclass_recall_states(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    n_classes: int,
) -> Dict[str, torch.Tensor]:
    true_positives_at_k_sum = compute_true_positives_at_k(
        predictions, labels, weights, n_classes
    )
    return {
        "tp_at_k": true_positives_at_k_sum,
        "total_weights": torch.sum(weights, dim=-1),
    }


class MulticlassRecallMetricComputation(RecMetricComputation):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._n_classes: int = kwargs.pop("number_of_classes")
        super().__init__(*args, **kwargs)
        self._add_state(
            "tp_at_k",
            torch.zeros(self._n_tasks, self._n_classes, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "total_weights",
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
                "Inputs 'predictions' and 'weights' should not be None for MulticlassRecallMetricComputation update"
            )
        states = get_multiclass_recall_states(
            predictions, labels, weights, self._n_classes
        )
        num_samples = predictions.shape[-2]
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.MULTICLASS_RECALL,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_multiclass_recall_at_k(
                    cast(torch.Tensor, self.tp_at_k),
                    cast(torch.Tensor, self.total_weights),
                ),
            ),
            MetricComputationReport(
                name=MetricName.MULTICLASS_RECALL,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_multiclass_recall_at_k(
                    self.get_window_state("tp_at_k"),
                    self.get_window_state("total_weights"),
                ),
            ),
        ]


class MulticlassRecallMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.MULTICLASS_RECALL
    _computation_class: Type[RecMetricComputation] = MulticlassRecallMetricComputation

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION:
            logging.warning(
                f"compute_mode FUSED_TASKS_AND_STATES_COMPUTATION can't support {self._namespace} yet "
                "because its states are not 1D Tensors. Only FUSED_TASKS_COMPUTATION will take effect."
            )
