#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from typing import Any, cast, Dict

from torch import nn

from torch.profiler import record_function

from torchrec.metrics.metric_module import RecMetricModule
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.rec_metric import (
    RecComputeMode,
    RecMetric,
    RecMetricComputation,
    RecMetricList,
)

logger: logging.Logger = logging.getLogger(__name__)


class CPUCommsRecMetricModule(RecMetricModule):
    """
    A submodule of CPUOffloadedRecMetricModule.

    The comms module's main purposes are:
    1. All gather metric state tensors
    2. Load all gathered metric states
    3. Compute metrics

    This isolation allows CPUOffloadedRecMetricModule from having
    to concern about aggregated states and instead focus solely
    updating local state tensors and dumping snapshots to the comms module
    for metric aggregations.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        All arguments are the same as RecMetricModule
        """

        super().__init__(*args, **kwargs)

        rec_metrics_clone = self._clone_rec_metrics()
        self.rec_metrics: RecMetricList = rec_metrics_clone

        for metric in self.rec_metrics.rec_metrics:
            # Disable automatic sync for all metrics - handled manually via
            # RecMetricModule.get_pre_compute_states()
            metric = cast(RecMetric, metric)
            for computation in metric._metrics_computations:
                computation = cast(RecMetricComputation, computation)
                computation._to_sync = False

    def load_local_metric_state_snapshot(
        self, state_snapshot: MetricStateSnapshot
    ) -> None:
        """
        Load local metric states before all gather.
        MetricStateSnapshot provides already-reduced states.

        Args:
            state_snapshot (MetricStateSnapshot): a snapshot of metric states to load.
        """

        # Load states into comms module to be shared across ranks.

        with record_function("## CPUCommsRecMetricModule: load_snapshot ##"):
            for metric in self.rec_metrics.rec_metrics:
                metric = cast(RecMetric, metric)
                compute_mode = metric._compute_mode
                if (
                    compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
                    or compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
                ):
                    prefix = compute_mode.name
                    computation = metric._metrics_computations[0]
                    self._load_metric_states(
                        prefix, computation, state_snapshot.metric_states
                    )
                for task, computation in zip(
                    metric._tasks, metric._metrics_computations
                ):
                    self._load_metric_states(
                        task.name, computation, state_snapshot.metric_states
                    )

            if state_snapshot.throughput_metric is not None:
                self.throughput_metric = state_snapshot.throughput_metric

    def _load_metric_states(
        self, prefix: str, computation: nn.Module, metric_states: Dict[str, Any]
    ) -> None:
        """
        Load metric states after all gather.
        Uses aggregated states.
        """

        # All update() calls were done prior. Clear previous computed state.
        # Otherwise, we get warnings that compute() was called before
        # update() which is not the case.
        computation = cast(RecMetricComputation, computation)
        set_update_called(computation)
        computation._computed = None

        computation_name = f"{prefix}_{computation.__class__.__name__}"
        # Restore all cached states from reductions
        for attr_name in computation._reductions:
            cache_key = f"{computation_name}_{attr_name}"
            if cache_key in metric_states:
                cached_value = metric_states[cache_key]
                setattr(computation, attr_name, cached_value)

    def _clone_rec_metrics(self) -> RecMetricList:
        """
        Clone rec_metrics. We need to keep references to the original tasks
        and computation to load the state tensors. More importantly, we need to
        remove the references to the original metrics to prevent concurrent access
        from the update and compute threads.
        """

        cloned_metrics = []
        for metric in self.rec_metrics.rec_metrics:
            metric = cast(RecMetric, metric)
            cloned_metric = type(metric)(
                world_size=metric._world_size,
                my_rank=metric._my_rank,
                batch_size=metric._batch_size,
                tasks=metric._tasks,
                compute_mode=metric._compute_mode,
                # Standard initialization passes in the global window size. A RecMetric's
                # window size is set as the local window size.
                window_size=metric._window_size * metric._world_size,
                fused_update_limit=metric._fused_update_limit,
                compute_on_all_ranks=metric._metrics_computations[
                    0
                ]._compute_on_all_ranks,
                should_validate_update=metric._should_validate_update,
                # Process group should be none to prevent unwanted distributed syncs.
                # This is handled manually via RecMetricModule.get_pre_compute_states()
                process_group=None,
            )
            cloned_metrics.append(cloned_metric)

        return RecMetricList(cloned_metrics)


def set_update_called(computation: RecMetricComputation) -> None:
    """
    Set _update_called to True for RecMetricComputation.
    This is a workaround for torchmetrics 1.0.3+.
    """
    try:
        computation._update_called = True
    except AttributeError:
        # pyre-ignore
        computation._update_count = 1
