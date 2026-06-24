#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

"""Unified metrics type for sync and async metric access.

Thread safety: This class relies on CPython's GIL for safe mutation of
_data and _resolved from Future callback threads. The GIL ensures that
attribute assignment is atomic, so concurrent reads of is_resolved() or
resolve() will see a consistent state. If used outside CPython (e.g.,
free-threaded Python 3.13t), a threading.Lock would be needed.
"""

import logging
import traceback
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from typing import Any, Callable

import torch

try:
    # Guarded: TorchRec is packaged into inference paths without the logging
    # handler shim; an unconditional import would break those packages.
    from torchrec.distributed.logging_handlers import (
        EventLoggingHandler,
        TorchrecComponent,
    )
    from torchrec.distributed.logging_utils import EventType
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.metrics.deferrable_metrics.import_failure.logging_handlers"
    )

    from enum import Enum as _Enum
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from torchrec.distributed.logging_handlers import (
            EventLoggingHandler,
            TorchrecComponent,
        )
        from torchrec.distributed.logging_utils import EventType
    else:

        class TorchrecComponent(_Enum):
            REC_METRICS = "rec_metrics"

        class EventType(_Enum):
            INFO = "INFO"

        class EventLoggingHandler:
            @staticmethod
            def log_event(*args: object, **kwargs: object) -> None:
                pass


logger: logging.Logger = logging.getLogger(__name__)

_EVENT_NAME: str = "DeferrableMetrics.deferred_failure"
_ERROR_MESSAGE_MAX_LEN: int = 4096
_STACK_TRACE_MAX_LEN: int = 8192
_TRUNCATION_MARKER: str = "...[truncated]"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: max(0, n - len(_TRUNCATION_MARKER))] + _TRUNCATION_MARKER


def device_supports_async(device: torch.device) -> bool:
    """Check if a device supports non-blocking async transfers (CUDA events)."""
    return device.type == "cuda"


def transfer_tensors_to_cpu(
    tensors: dict[str, Any],
) -> tuple[dict[str, Any], torch.cuda.Event | None]:
    """Transfer GPU tensors to CPU using non-blocking copies.

    Returns the CPU tensor dict and a CUDA event that tracks completion.
    For CPU-only inputs, returns the dict as-is with None event.
    Non-tensor values are preserved unchanged.

    Records the CUDA event on the source tensor's device stream (not the
    default device) to avoid metric corruption on non-rank-0 processes.
    All tensors are assumed to be on the same device.
    """
    source_device: torch.device | None = None
    for v in tensors.values():
        if isinstance(v, torch.Tensor) and v.device.type == "cuda":
            source_device = v.device
            break
    if source_device is None:
        return tensors, None

    with torch.cuda.device(source_device):
        cpu_tensors: dict[str, Any] = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                dst = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
                dst.copy_(v, non_blocking=True)
                cpu_tensors[k] = dst
            else:
                cpu_tensors[k] = v
        event = torch.cuda.Event()
        event.record()

    return cpu_tensors, event


class DeferrableMetrics(Mapping[str, Any]):
    """
    Metrics container that wraps either a resolved dict or a pending Future.

    Provides a unified interface for both sync and async metric access:
    - subscribe(callback): async access, always works
    - resolve(): sync access, blocks if backed by Future
    - update(other): deferred merge
    - is_resolved(): check without blocking

    Implements Mapping[str, Any] so it is a drop-in replacement for
    Dict[str, MetricValue] at both type and runtime level. Dict-style
    access (__getitem__, __iter__, __len__) calls resolve() internally.

    Future-backed instances emit a `DeferrableMetrics.deferred_failure`
    INFO event to `torchrec_event_logging` if the Future raises. Success
    is already captured by the enclosing `RecMetricModule.compute` decorator,
    so no SUCCESS counterpart is emitted here. The done-callback fires once
    per Future regardless of whether resolve/subscribe is called, so
    unobserved futures still surface their failures.
    """

    _warned: bool = False

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

        if self._future is not None:
            self._future.add_done_callback(self._emit_failure_if_raised)

    def _emit_failure_if_raised(self, f: Future[dict[str, Any]]) -> None:
        # Runs on the Future's done-callback thread. Emit is synchronous to
        # match the local @event_logger convention; telemetry must never
        # raise into the caller.
        try:
            f.result()
        except BaseException as e:  # noqa: B036
            try:
                EventLoggingHandler.log_event(
                    component=TorchrecComponent.REC_METRICS.value,
                    event_name=_EVENT_NAME,
                    event_type=EventType.INFO,
                    metadata={"exception_type": type(e).__name__},
                    error_message=_truncate(str(e), _ERROR_MESSAGE_MAX_LEN),
                    stack_trace=_truncate(
                        "".join(
                            traceback.format_exception(type(e), e, e.__traceback__)
                        ),
                        _STACK_TRACE_MAX_LEN,
                    ),
                )
            except BaseException:  # noqa: B036
                pass

    def _warn_sync_access(self) -> None:
        """Log a warning once per process when dict-style access
        triggers resolve() on a Future-backed instance."""
        if not DeferrableMetrics._warned:
            DeferrableMetrics._warned = True
            logger.warning(
                "DeferrableMetrics: synchronous dict-style access "
                "is blocking on a Future. Use .subscribe() for async access.",
            )

    def subscribe(
        self,
        callback: Callable[[dict[str, Any]], None],
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

        If backed by a Future, blocks until the Future completes.
        Use subscribe() for non-blocking async access in training paths.
        """
        if not self._resolved:
            if self._future is not None:
                result = self._future.result()
                self._data = result
                self._resolved = True
            else:
                raise RuntimeError(
                    "DeferrableMetrics is in an invalid state: "
                    "not resolved and no Future."
                )
        return self._data

    def update(self, other: dict[str, Any] | DeferrableMetrics) -> None:
        """Merge additional key-value pairs.

        If already resolved, merges immediately.
        If backed by Future, defers the merge until Future resolves.
        If other is a DeferrableMetrics, subscribes to it for deferred merge.

        Any CUDA tensors in `other` are eagerly transferred to pinned CPU
        memory on the calling thread (truly async via pinned destination —
        caller is not blocked). The stored values are always CPU. This
        prevents per-tensor `.detach().cpu()` from running on the resolve
        thread (typically metric_compute), which would otherwise hold the
        GIL and stall backward.

        Ordering constraint: when backed by a Future, update() replaces the
        internal Future with a new merged Future. Any subscribe() callbacks
        registered *before* update() are attached to the original Future and
        will receive un-merged data. Call update() before subscribe() to
        ensure callbacks see the merged result.
        """
        if isinstance(other, DeferrableMetrics):
            if other.is_resolved():
                self.update(other._data)
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
        # Eagerly stage CUDA tensors to pinned CPU. Returns the input dict
        # unchanged with event=None when nothing is on CUDA.
        cpu_other, transfer_event = transfer_tensors_to_cpu(other)

        if self._resolved:
            if transfer_event is not None:
                transfer_event.synchronize()
            self._data.update(cpu_other)
        elif self._future is not None:
            original_future = self._future
            merged_future: Future[dict[str, Any]] = Future()

            def _on_complete(f: Future[dict[str, Any]]) -> None:
                try:
                    if transfer_event is not None:
                        transfer_event.synchronize()
                    result = dict(f.result())
                    result.update(cpu_other)
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

    def __setitem__(self, key: str, value: Any) -> None:
        self.resolve()[key] = value

    def __getitem__(self, key: str) -> Any:
        if not self._resolved:
            self._warn_sync_access()
        return self.resolve()[key]

    def __iter__(self) -> Iterator[str]:
        if not self._resolved:
            self._warn_sync_access()
        return iter(self.resolve())

    def __len__(self) -> int:
        if not self._resolved:
            self._warn_sync_access()
        return len(self.resolve())

    def __repr__(self) -> str:
        if self._resolved:
            return f"DeferrableMetrics(resolved, {len(self._data)} keys)"
        return "DeferrableMetrics(pending)"
