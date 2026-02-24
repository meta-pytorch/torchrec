#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import concurrent
from typing import Any, Dict

import torch
from torchrec.metrics.metric_module import MetricValue
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot


class MetricUpdateJob:
    """
    Encapsulates a metric update job for CPU processing.
    The model outputs are transferred to CPU (if needed) and used
    to update each metric's state tensors.
    """

    __slots__ = ["model_out", "kwargs"]

    def __init__(
        self,
        model_out: Dict[str, torch.Tensor],
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Args:
            model_out: intermediate model outputs (may be on GPU; DtoH
                transfer is handled by the processing thread)
            kwargs: additional arguments from the trainer platform
        """

        self.model_out: Dict[str, torch.Tensor] = model_out
        self.kwargs: Dict[str, Any] = kwargs


class MetricComputeJob:
    """
    Encapsulates metric compute job for CPU processing: perform an
    all gather across ranks, compute metrics, and return the result to be
    published.
    """

    __slots__ = ["future", "metric_state_snapshot"]

    def __init__(
        self,
        # pyrefly: ignore[implicit-import]
        future: concurrent.futures.Future[Dict[str, MetricValue]],
        metric_state_snapshot: MetricStateSnapshot,
    ) -> None:
        """
        Args:
            future: future to set the result of the compute job. Contains the computed metrics.
            metric_state_snapshot: snapshot of metric state tensors across all metrics types.
        """
        # pyrefly: ignore[implicit-import]
        self.future: concurrent.futures.Future[Dict[str, MetricValue]] = future
        self.metric_state_snapshot: MetricStateSnapshot = metric_state_snapshot


class SynchronizationMarker:
    """
    Represents the synchronization marker that is stored in the update queue. This is the point
    we want to synchronize across all ranks to compute metrics.
    When processed, this marker will convert to a MetricComputeJob in the compute queue.

    This separation of synchronization marker and compute job is so that the metric compute job
    accurately includes all of the metric jobs that were scheduled before it.
    """

    __slots__ = "future"

    def __init__(
        self,
        # pyrefly: ignore[implicit-import]
        future: concurrent.futures.Future[Dict[str, MetricValue]],
    ) -> None:
        """
        Args:
            future: future to set the result of the compute job. Passed to the MetricComputeJob.
        """
        # pyrefly: ignore[implicit-import]
        self.future: concurrent.futures.Future[Dict[str, MetricValue]] = future
