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


CALIBRATION_NUM = "calibration_num"
CALIBRATION_DENOM = "calibration_denom"


def compute_calibration(
    calibration_num: torch.Tensor, calibration_denom: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        calibration_denom <= 0.0, 0.0, calibration_num / calibration_denom
    ).double()


def get_calibration_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {
        CALIBRATION_NUM: torch.sum(predictions * weights, dim=-1),
        CALIBRATION_DENOM: torch.sum(labels * weights, dim=-1),
    }


class CalibrationMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Calibration, which is the
    ratio between the prediction and the labels (conversions).

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            CALIBRATION_NUM,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            CALIBRATION_DENOM,
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
                "Inputs 'predictions' and 'weights' should not be None for CalibrationMetricComputation update"
            )
        num_samples = predictions.shape[-1]
        for state_name, state_value in get_calibration_states(
            labels, predictions, weights
        ).items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.CALIBRATION,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_calibration(
                    cast(torch.Tensor, self.calibration_num),
                    cast(torch.Tensor, self.calibration_denom),
                ),
            ),
            MetricComputationReport(
                name=MetricName.CALIBRATION,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_calibration(
                    self.get_window_state(CALIBRATION_NUM),
                    self.get_window_state(CALIBRATION_DENOM),
                ),
            ),
        ]


class CalibrationMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.CALIBRATION
    _computation_class: Type[RecMetricComputation] = CalibrationMetricComputation
