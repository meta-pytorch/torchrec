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

import concurrent.futures
import logging
from typing import Callable, Dict, Tuple, TypeVar

import torch
from torch.profiler import record_function
from torchrec.metrics.metric_module import (
    PublishableMetrics,
    PublishableMetricsFuture,
    PublishableMetricsOutput,
)

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def device_supports_async(device: torch.device) -> bool:
    """
    Check if the device is supported for asynchronous metric computation.

    Currently, only CUDA devices are supported.

    Args:
        device: The device to check

    Returns:
        True if device is supported (CUDA), False otherwise
    """
    return device.type == "cuda"


def transfer_tensors_to_cpu(
    tensors: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.cuda.Event | None]:
    """
    Move all tensors to CPU with non-blocking transfer unconditionally.
    Create a copy of model_out on CPU and return the copy. A cuda event
    is created to track when the copy is completed.

    Args:
        tensors: Dict of names to tensors

    Returns:
        Tuple of:
        - Dict with all tensors moved to CPU (non-blocking)
        - CUDA event to synchronize on before accessing values, or None if
          no CUDA tensors were present
    """
    source_device: torch.device | None = None
    for v in tensors.values():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            source_device = v.device
            break

    cpu_tensors = {
        k: v.to(device="cpu", non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in tensors.items()
    }

    if source_device is not None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(source_device))
    else:
        event = None
    return cpu_tensors, event


def update_metrics_output(
    metrics_output: PublishableMetricsOutput,
    metrics_update: PublishableMetrics,
) -> PublishableMetricsOutput:
    """
    Updates metrics_output with specified dict.

    When metrics_output is a Future (ZORM/async path), GPU tensors in metrics_update
    are moved to CPU with non-blocking transfer to prevent CUDA synchronization
    during later access (e.g., .item() calls in metrics publish path).

    For synchronous path (non-ZORM), tensors are left on their original device.

    Args:
        metrics_output: Either metrics dict (sync) or Future (async)
        metrics_update: Dict to update metrics_output with

    Returns:
        Updated metrics_output
    """
    if isinstance(metrics_output, concurrent.futures.Future):
        # Async path: initiate non blocking DtoH in advance
        cpu_metrics_update, _event = transfer_tensors_to_cpu(metrics_update)

        def _update_callback(future: PublishableMetricsFuture) -> None:
            if _event is not None:
                _event.synchronize()
            future.result().update(cpu_metrics_update)

        metrics_output.add_done_callback(_update_callback)
    else:
        # Sync path: update directly without DtoH
        metrics_output.update(metrics_update)
    return metrics_output


def get_metrics_async(
    metrics_output: PublishableMetricsOutput,
    callback: Callable[[PublishableMetrics], T],
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

    Example:
        >>> metrics_output = self.metrics.compute()
        >>> metrics_result = get_metrics_async(metrics_output, self._publish_metrics)
    """
    callback_name = getattr(callback, "__name__", "unknown_callback")

    # Asynchronous path
    # pyrefly: ignore[implicit-import]
    if isinstance(metrics_output, concurrent.futures.Future):

        def on_complete(future: PublishableMetricsFuture) -> None:
            try:
                result = future.result()
                with record_function(f"## {callback_name} ##"):
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
        with record_function(f"## {callback_name} ##"):
            return callback(metrics_output)


def get_metrics_sync(
    metrics_output: PublishableMetricsOutput,
) -> PublishableMetrics:
    """
    Synchronously resolve PublishableMetricsOutput to PublishableMetrics.

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
    # pyrefly: ignore[implicit-import]
    if isinstance(metrics_output, concurrent.futures.Future):
        return metrics_output.result()
    else:
        return metrics_output
