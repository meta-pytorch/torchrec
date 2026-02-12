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
from torchrec.metrics.ne import compute_ne, get_ne_states
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


NUM_EXAMPLES = "num_examples"


class ServingNEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for NE over serving data only,
        i.e., excluding data with weight=0.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        self._add_state(
            NUM_EXAMPLES,
            torch.zeros(self._n_tasks, dtype=torch.long),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self.eta = 1e-12

    def _get_bucket_metric_states(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for BucketNEMetricComputation update"
            )

        if labels.nelement() == 0:
            return {
                "cross_entropy_sum": torch.zeros(self._n_tasks, dtype=torch.double),
                "weighted_num_samples": torch.zeros(self._n_tasks, dtype=torch.double),
                "pos_labels": torch.zeros(self._n_tasks, dtype=torch.double),
                "neg_labels": torch.zeros(self._n_tasks, dtype=torch.double),
            }

        return get_ne_states(
            labels=labels,
            predictions=predictions,
            weights=weights,
            eta=self.eta,
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
                "Inputs 'predictions' and 'weights' should not be None for ServingNEMetricComputation update"
            )

        states = get_ne_states(labels, predictions, weights, self.eta)

        num_samples = labels.shape[-1]
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

        num_examples_delta = torch.count_nonzero(weights, dim=-1)
        state_num_examples = getattr(self, NUM_EXAMPLES)
        state_num_examples += num_examples_delta

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.NE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_ne(
                    cast(torch.Tensor, self.cross_entropy_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                    cast(torch.Tensor, self.pos_labels),
                    cast(torch.Tensor, self.neg_labels),
                    self.eta,
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
                ),
            ),
            MetricComputationReport(
                name=MetricName.TOTAL_EXAMPLES,
                metric_prefix=MetricPrefix.DEFAULT,
                value=cast(torch.Tensor, self.num_examples).detach(),
            ),
        ]


class ServingNEMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.SERVING_NE
    _computation_class: Type[RecMetricComputation] = ServingNEMetricComputation
