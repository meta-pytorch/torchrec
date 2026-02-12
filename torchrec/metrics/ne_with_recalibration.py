#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricNamespace
from torchrec.metrics.ne import get_ne_states, NEMetricComputation
from torchrec.metrics.rec_metric import (
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


class RecalibratedNEMetricComputation(NEMetricComputation):
    r"""
    This class implements the recalibration for NE that is required to correctly estimate eval NE if negative downsampling was used during training.

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
        recalibration_coefficient: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._recalibration_coefficient: float = recalibration_coefficient
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
                "Inputs 'predictions' and 'weights' should not be None for RecalibratedNEMetricComputation update"
            )

        predictions = self._recalibrate(
            predictions, self._recalibration_coefficient * torch.ones_like(predictions)
        )
        states = get_ne_states(labels, predictions, weights, self.eta)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)


class RecalibratedNEMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.RECALIBRATED_NE
    _computation_class: Type[RecMetricComputation] = RecalibratedNEMetricComputation
