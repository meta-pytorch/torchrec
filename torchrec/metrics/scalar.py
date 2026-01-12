#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
)


class ScalarMetricComputation(RecMetricComputation):
    """
    Metric that logs whatever value is given as the label.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "labels",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="mean",
            persistent=False,
        )
        self._add_state(
            "window_count",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="mean",
            persistent=False,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        num_samples = labels.shape[0]

        states = {
            "labels": labels.mean(dim=-1),
            "window_count": torch.ones(
                torch.Size([self._n_tasks]),
                device=labels.device,
                dtype=torch.double,
            ),  # put window count on the correct device
        }
        for state_name, state_value in states.items():
            setattr(self, state_name, state_value)
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.SCALAR,
                metric_prefix=MetricPrefix.LIFETIME,
                # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                value=self.labels,
            ),
            MetricComputationReport(
                name=MetricName.SCALAR,
                metric_prefix=MetricPrefix.WINDOW,
                # return the mean of the window state
                value=self.get_window_state("labels")
                / self.get_window_state("window_count"),
            ),
        ]


class ScalarMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.SCALAR
    _computation_class: Type[RecMetricComputation] = ScalarMetricComputation
