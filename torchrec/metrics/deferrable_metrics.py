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

from concurrent.futures import Future
from typing import Any, Callable

import torch


def transfer_tensors_to_cpu(
    tensors: dict[str, Any],
) -> tuple[dict[str, Any], torch.cuda.Event | None]:
    """Move all CUDA tensors to CPU with non-blocking transfer.

    Returns a copy of the dict with CPU tensors plus a CUDA event to
    synchronize on before accessing values. The event is recorded on the
    source tensor's stream (not the default cuda:0 stream) to correctly
    handle multi-GPU setups.
    """
    source_device: torch.device | None = None
    for v in tensors.values():
        if isinstance(v, torch.Tensor) and v.is_cuda:
            source_device = v.device
            break

    cpu_tensors = {
        k: (
            v.to(device="cpu", non_blocking=True)
            if isinstance(v, torch.Tensor) and v.is_cuda
            else v
        )
        for k, v in tensors.items()
    }

    if source_device is not None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(source_device))
    else:
        event = None
    return cpu_tensors, event


def device_supports_async(device: torch.device) -> bool:
    """Check if the device supports asynchronous (non-blocking) DtoH transfer."""
    return device.type == "cuda"


class DeferrableMetrics:
    """
    Metrics container that wraps either a resolved dict or a pending Future.

    Provides a unified interface for both sync and async metric access:
    - subscribe(callback): async access, always works
    - resolve(): sync access, fails fast if backed by Future
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

        Fails fast with RuntimeError if backed by a Future.
        Only use in contexts where CPU-offloaded metrics are disabled
        (eval, inference, tests).
        """
        if not self._resolved:
            raise RuntimeError(
                "Cannot synchronously resolve DeferrableMetrics backed by a "
                "Future. Use subscribe() for async access, or ensure "
                "CPU-offloaded metrics are disabled for this code path."
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
