#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Optional, Type

import torch
from torchrec.metrics.calibration import (
    CalibrationMetricComputation,
    get_calibration_states,
)
from torchrec.metrics.metrics_namespace import MetricNamespace
from torchrec.metrics.rec_metric import (
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


CALIBRATION_NUM = "calibration_num"
CALIBRATION_DENOM = "calibration_denom"


class RecalibratedCalibrationMetricComputation(CalibrationMetricComputation):
    r"""
    This class implements the RecMetricComputation for Calibration that is required to correctly estimate eval NE if negative downsampling was used during training.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self, *args: Any, recalibration_coefficient: float = 1.0, **kwargs: Any
    ) -> None:
        self._recalibration_coefficient: float = recalibration_coefficient
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

    def _recalibrate(
        self,
        predictions: torch.Tensor,
        calibration_coef: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if calibration_coef is not None:
            predictions = predictions / (
                predictions + (1.0 - predictions) / calibration_coef
            )
        return predictions

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
        predictions = self._recalibrate(
            predictions, self._recalibration_coefficient * torch.ones_like(predictions)
        )
        num_samples = predictions.shape[-1]
        for state_name, state_value in get_calibration_states(
            labels, predictions, weights
        ).items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)


class RecalibratedCalibrationMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.RECALIBRATED_CALIBRATION
    _computation_class: Type[RecMetricComputation] = (
        RecalibratedCalibrationMetricComputation
    )
