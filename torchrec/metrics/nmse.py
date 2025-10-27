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
from torchrec.metrics.mse import (
    compute_error_sum,
    compute_mse,
    compute_rmse,
    ERROR_SUM,
    get_mse_states,
    MSEMetricComputation,
    WEIGHTED_NUM_SAMPES,
)
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricException,
)

CONST_PRED_ERROR_SUM = "const_pred_error_sum"


def compute_norm(
    model_error_sum: torch.Tensor, baseline_error_sum: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        baseline_error_sum == 0,
        torch.tensor(0.0),
        model_error_sum / baseline_error_sum,
    ).double()


def get_norm_mse_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        **get_mse_states(labels, predictions, weights),
        **(
            {
                CONST_PRED_ERROR_SUM: compute_error_sum(
                    labels, torch.ones_like(labels), weights
                )
            }
        ),
    }


class NMSEMetricComputation(MSEMetricComputation):
    r"""
    This class extends the MSEMetricComputation for normalization computation for L2 regression metrics.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            CONST_PRED_ERROR_SUM,
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
                "Inputs 'predictions' and 'weights' should not be None for NMSEMetricComputation update"
            )
        states = get_norm_mse_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        mse = compute_mse(
            cast(torch.Tensor, self.error_sum),
            cast(torch.Tensor, self.weighted_num_samples),
        )
        const_pred_mse = compute_mse(
            cast(torch.Tensor, self.const_pred_error_sum),
            cast(torch.Tensor, self.weighted_num_samples),
        )
        nmse = compute_norm(mse, const_pred_mse)

        rmse = compute_rmse(
            cast(torch.Tensor, self.error_sum),
            cast(torch.Tensor, self.weighted_num_samples),
        )
        const_pred_rmse = compute_rmse(
            cast(torch.Tensor, self.const_pred_error_sum),
            cast(torch.Tensor, self.weighted_num_samples),
        )
        nrmse = compute_norm(rmse, const_pred_rmse)

        window_mse = compute_mse(
            self.get_window_state(ERROR_SUM),
            self.get_window_state(WEIGHTED_NUM_SAMPES),
        )
        window_const_pred_mse = compute_mse(
            self.get_window_state(CONST_PRED_ERROR_SUM),
            self.get_window_state(WEIGHTED_NUM_SAMPES),
        )
        window_nmse = compute_norm(window_mse, window_const_pred_mse)

        window_rmse = compute_rmse(
            self.get_window_state(ERROR_SUM),
            self.get_window_state(WEIGHTED_NUM_SAMPES),
        )
        window_const_pred_rmse = compute_rmse(
            self.get_window_state(CONST_PRED_ERROR_SUM),
            self.get_window_state(WEIGHTED_NUM_SAMPES),
        )
        window_nrmse = compute_norm(window_rmse, window_const_pred_rmse)

        return [
            MetricComputationReport(
                name=MetricName.NMSE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=nmse,
            ),
            MetricComputationReport(
                name=MetricName.NRMSE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=nrmse,
            ),
            MetricComputationReport(
                name=MetricName.NMSE,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_nmse,
            ),
            MetricComputationReport(
                name=MetricName.NRMSE,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_nrmse,
            ),
        ]


class NMSEMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.NMSE
    _computation_class: Type[NMSEMetricComputation] = NMSEMetricComputation
