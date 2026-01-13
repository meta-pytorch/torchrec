#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Utility functions for handling MetricsOutput (Union[MetricsResult, MetricsFuture]) from
- RecMetricModule.compute()
- CPUOffloadedRecMetricModule.async_compute()
"""

import concurrent
import logging
from typing import Any, Callable, Dict, TypeVar

from torchrec.metrics.metric_module import MetricsFuture, MetricsOutput, MetricsResult

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def update_metrics_output(
    metrics_output: MetricsOutput,
    metrics_update: Dict[str, Any],
) -> MetricsOutput:
    """
    Updates metrics_output with specified dict.

    Args:
        metrics_output: Either metrics dict (sync) or Future (async)
        dict: Dict to update metrics_output with

    Returns:
        Updated metrics_output
    """

    if isinstance(metrics_output, concurrent.futures.Future):

        def _update_callback(future: MetricsFuture) -> None:
            future.result().update(metrics_update)

        metrics_output.add_done_callback(_update_callback)
    else:
        metrics_output.update(metrics_update)
    return metrics_output


def get_metrics_async(
    metrics_output: MetricsOutput,
    callback: Callable[[MetricsResult], T],
    *,
    on_error: Callable[[Exception], None] | None = None,
) -> T | None:
    """
    Register a callback to execute when metrics are ready.

    Preserves CPUOffloadedRecMetricModule's async benefits by executing callbacks when Future resolves,
    without blocking the critical training path.

    Args:
        metrics_output: Either metrics dict (sync from RecMetricModule) or Future (async from CPUOffloadedRecMetricModule)
        callback: Function to execute with resolved metrics
        on_error: Optional error handler for exceptions

    Returns:
        Result of callback if metrics are immediately available (Dict[str, MetricValue]),
        None if async (Future) - callback will be invoked later
    """

    # Asynchronous path
    if isinstance(metrics_output, concurrent.futures.Future):

        def on_complete(future: MetricsFuture) -> None:
            try:
                result = future.result()
                callback(result)
            except Exception as e:
                if on_error:
                    on_error(e)
                else:
                    logger.exception("Error in metrics callback")
                    raise

        metrics_output.add_done_callback(on_complete)
        return None
    else:
        # Synchronous path
        return callback(metrics_output)


def get_metrics_sync(
    metrics_output: MetricsOutput,
) -> MetricsResult:
    """
    Synchronously resolve MetricsOutput to MetricsResult.

    Use this when you need the actual metrics dict immediately (e.g., to modify it).
    For async handling, use get_metrics_async() instead.

    Args:
        metrics_output: Either metrics dict (sync) or Future (async)

    Returns:
        Resolved metrics dict

    Raises:
        Exception: Any exception from Future computation

    Example:
        >>> metrics_output = self.metrics.compute()
        >>> metrics_result = get_metrics_sync(metrics_output) # wait until metrics are ready
        >>> publish_metrics(metrics_result)
    """
    if isinstance(metrics_output, concurrent.futures.Future):
        return metrics_output.result()
    else:
        return metrics_output
