#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Any, cast, Dict, Optional

from torch import nn
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
    RecMetricComputation,
    RecMetricList,
)
from torchrec.metrics.throughput import ThroughputMetric


class MetricStateSnapshot:
    """
    Encapsulates both rec metrics reduced states and throughput metric snapshots
    for thread-safe CPU offloaded metric computation (updates and computes).
    """

    def __init__(
        self,
        metric_states: Dict[str, Any],
        throughput_metric: Optional[ThroughputMetric],
    ) -> None:
        """
        Args:
            metric_states (Dict[str, Any]): Reduced states from rec metrics
            throughput_metric (Optional[ThroughputMetric]): Deep copy of throughput metric
        """
        self.metric_states = metric_states
        self.throughput_metric = throughput_metric

    @classmethod
    def from_metrics(
        cls,
        rec_metrics: RecMetricList,
        throughput_metric: Optional[ThroughputMetric] = None,
    ) -> "MetricStateSnapshot":
        """
        Generate a MetricStateSnapshot before performing an all gather. This provides a consistent
        view of the local metric states without accessing the original references.

        Apply reductions BEFORE queuing to reduce memory footprint. For instance, AUC holds a list of
        tensors which can be reduced to a list of a single tensor. Only reduce lists for
        fused mode compatibility.
        """
        reduced_states: Dict[str, Any] = {}

        for metric in rec_metrics.rec_metrics:
            metric = cast(RecMetric, metric)
            compute_mode = metric._compute_mode
            if (
                compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
                or compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
            ):
                computation = metric._metrics_computations[0]
                _load_into_reduced_states(
                    compute_mode.name, computation, reduced_states
                )
            else:
                for task, computation in zip(
                    metric._tasks, metric._metrics_computations
                ):
                    _load_into_reduced_states(task.name, computation, reduced_states)

        # Snapshot throughput metric
        throughput_snapshot = None
        if throughput_metric:
            throughput_snapshot = copy.deepcopy(throughput_metric)

        return cls(
            metric_states=reduced_states,
            throughput_metric=throughput_snapshot,
        )


def _load_into_reduced_states(
    prefix: str,
    computation: nn.Module,
    reduced_states: Dict[str, Any],
) -> None:
    """
    Load the reduced states into the reduced_states dict.

    Args:
        prefix (str): prefix for the metric computation
        computation (nn.Module): metric computation
        reduced_states (Dict[str, Any]): reduced states dict to load into
    """
    computation = cast(RecMetricComputation, computation)
    computation_name = f"{prefix}_{computation.__class__.__name__}"

    for attr_name in computation._reductions:
        cache_key = f"{computation_name}_{attr_name}"
        original_value = getattr(computation, attr_name)
        reduction_fn = computation._reductions[attr_name]
        if callable(reduction_fn) and isinstance(original_value, list):
            reduced_value = reduction_fn(original_value)
        else:
            reduced_value = original_value

        reduced_states[cache_key] = reduced_value
