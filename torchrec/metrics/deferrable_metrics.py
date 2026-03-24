#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unified metrics type for sync and async metric access.

Thread safety: This class relies on CPython's GIL for safe mutation of
_data and _resolved from Future callback threads. The GIL ensures that
attribute assignment is atomic, so concurrent reads of is_resolved() or
resolve() will see a consistent state. If used outside CPython (e.g.,
free-threaded Python 3.13t), a threading.Lock would be needed.
"""

import logging
from concurrent.futures import Future
from typing import Any, Callable, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)


class DeferrableMetrics:
    """
    Metrics container that wraps either a resolved dict or a pending Future.

    Provides a unified interface for both sync and async metric access:
    - subscribe(callback): async access, always works
    - resolve(): sync access, blocks with warning if backed by Future
    - update(other): deferred merge
    - is_resolved(): check without blocking

    Does NOT implement Mapping. Callers must use subscribe() or resolve()
    to access metric values. This prevents silent blocking in the training
    path when backed by a Future.
    """

    def __init__(
        self,
        inner: dict[str, Any] | Future[dict[str, Any]],
    ) -> None:
        if isinstance(inner, Future):
            self._future: Future[dict[str, Any]] | None = inner
            self._data: dict[str, Any] = {}
            self._resolved: bool = False
        else:
            self._future = None
            self._data = dict(inner)
            self._resolved = True

    def subscribe(
        self,
        callback: Callable[[dict[str, Any]], Any],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Register a callback for when metrics are available.

        If already resolved, callback fires immediately (synchronously).
        If backed by Future, callback fires when Future completes.
        """
        if self._resolved:
            callback(self._data)
        elif self._future is not None:

            def _on_complete(f: Future[dict[str, Any]]) -> None:
                try:
                    result = f.result()
                    self._data = result
                    self._resolved = True
                    callback(self._data)
                except Exception as e:
                    if on_error:
                        on_error(e)
                    else:
                        raise

            self._future.add_done_callback(_on_complete)

    def resolve(self) -> dict[str, Any]:
        """Synchronously return the resolved metrics dict.

        If backed by a Future (ZORM active), blocks until resolved and logs a
        warning. Prefer subscribe() for async access in the training path to
        preserve ZORM's QPS benefit.
        """
        if not self._resolved:
            if self._future is not None:
                logger.warning(
                    "DeferrableMetrics.resolve() called on Future-backed metrics. "
                    "This blocks the training path and defeats ZORM's async benefit. "
                    "Use subscribe() for async access."
                )
                result = self._future.result()
                self._data = result
                self._resolved = True
            else:
                raise RuntimeError(
                    "DeferrableMetrics is in an invalid state: not resolved and no Future."
                )
        return self._data

    def update(self, other: "dict[str, Any] | DeferrableMetrics") -> None:
        """Merge additional key-value pairs.

        If already resolved, merges immediately.
        If backed by Future, defers the merge until Future resolves.
        If other is a DeferrableMetrics, subscribes to it for deferred merge.

        Ordering constraint: when backed by a Future, update() replaces the
        internal Future with a new merged Future. Any subscribe() callbacks
        registered *before* update() are attached to the original Future and
        will receive un-merged data. Call update() before subscribe() to
        ensure callbacks see the merged result.
        """
        if isinstance(other, DeferrableMetrics):
            if other.is_resolved():
                self.update(other.resolve())
            else:

                target_future = self._future

                def _propagate_error(e: Exception) -> None:
                    if target_future is not None and not target_future.done():
                        target_future.set_exception(e)

                other.subscribe(
                    lambda data: self.update(data),
                    on_error=_propagate_error,
                )
            return

        assert isinstance(other, dict)
        dict_other: dict[str, Any] = other

        if self._resolved:
            self._data.update(dict_other)
        elif self._future is not None:
            cpu_dict, event = transfer_tensors_to_cpu(dict_other)

            original_future = self._future
            merged_future: Future[dict[str, Any]] = Future()

            def _on_complete(f: Future[dict[str, Any]]) -> None:
                try:
                    if event is not None:
                        event.synchronize()
                    result = dict(f.result())
                    result.update(cpu_dict)
                    merged_future.set_result(result)
                except Exception as e:
                    merged_future.set_exception(e)

            original_future.add_done_callback(_on_complete)
            self._future = merged_future

    def is_resolved(self) -> bool:
        """Check if metrics are available without blocking."""
        return self._resolved

    def __bool__(self) -> bool:
        """Always True. Prevents ``metrics or {}`` from replacing DeferrableMetrics."""
        return True

    def __repr__(self) -> str:
        if self._resolved:
            return f"DeferrableMetrics(resolved, {len(self._data)} keys)"
        return "DeferrableMetrics(pending)"


def device_supports_async(device: torch.device) -> bool:
    """Check if a device supports non-blocking async transfers (CUDA events)."""
    return device.type == "cuda"


def transfer_tensors_to_cpu(
    tensors: dict[str, Any],
) -> Tuple[dict[str, Any], "torch.cuda.Event | None"]:
    """Transfer GPU tensors to CPU using non-blocking copies.

    Returns the CPU tensor dict and a CUDA event that tracks completion.
    For CPU-only inputs, returns the dict as-is with None event.
    Non-tensor values are preserved unchanged.

    Records the CUDA event on the source tensor's device stream (not the
    default device) to avoid metric corruption on non-rank-0 processes.
    """
    has_cuda = any(
        isinstance(v, torch.Tensor) and v.device.type == "cuda"
        for v in tensors.values()
    )
    if not has_cuda:
        return tensors, None

    # Detect source device from first CUDA tensor
    source_device: torch.device | None = None
    for v in tensors.values():
        if isinstance(v, torch.Tensor) and v.device.type == "cuda":
            source_device = v.device
            break

    cpu_tensors = {
        k: v.to(device="cpu", non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in tensors.items()
    }

    # Record event on the source tensor's device, not the default device
    assert source_device is not None
    with torch.cuda.device(source_device):
        event = torch.cuda.Event()
        event.record()

    return cpu_tensors, event
