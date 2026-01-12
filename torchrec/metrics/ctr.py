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


CTR_NUM = "ctr_num"
CTR_DENOM = "ctr_denom"


def compute_ctr(ctr_num: torch.Tensor, ctr_denom: torch.Tensor) -> torch.Tensor:
    return torch.where(ctr_denom == 0.0, 0.0, ctr_num / ctr_denom).double()


def get_ctr_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {
        CTR_NUM: torch.sum(labels * weights, dim=-1),
        CTR_DENOM: torch.sum(weights, dim=-1),
    }


class CTRMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for CTR, i.e. Click Through Rate,
    which is the ratio between the predicted positive examples and the total examples.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            CTR_NUM,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            CTR_DENOM,
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
                "Inputs 'predictions' and 'weights' should not be None for CTRMetricComputation update"
            )
        num_samples = predictions.shape[-1]
        for state_name, state_value in get_ctr_states(
            labels, predictions, weights
        ).items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.CTR,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_ctr(
                    cast(torch.Tensor, self.ctr_num),
                    cast(torch.Tensor, self.ctr_denom),
                ),
            ),
            MetricComputationReport(
                name=MetricName.CTR,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_ctr(
                    self.get_window_state(CTR_NUM),
                    self.get_window_state(CTR_DENOM),
                ),
            ),
        ]


class CTRMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.CTR
    _computation_class: Type[RecMetricComputation] = CTRMetricComputation
