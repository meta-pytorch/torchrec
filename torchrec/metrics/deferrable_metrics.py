#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Unified metrics type for sync and async metric access.

Thread safety: This class relies on CPython's GIL for safe mutation of
_data, _resolved, and _callbacks from Future callback threads. The GIL
ensures that attribute assignment is atomic, so concurrent reads of
is_resolved() or resolve() will see a consistent state. If used outside
CPython (e.g., free-threaded Python 3.13t), a threading.Lock would be
needed.
"""

import logging
from concurrent.futures import Future
from typing import Any, Callable

logger: logging.Logger = logging.getLogger(__name__)


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
        self._callbacks: list[
            tuple[
                Callable[[dict[str, Any]], Any],
                Callable[[Exception], None] | None,
            ]
        ] = []
        if isinstance(inner, Future):
            self._future: Future[dict[str, Any]] | None = inner
            self._data: dict[str, Any] = {}
            self._resolved: bool = False
            inner.add_done_callback(self._on_resolve)
        else:
            self._future = None
            self._data = dict(inner)
            self._resolved = True

    def _on_resolve(self, f: Future[dict[str, Any]]) -> None:
        """Single resolution point. Only fires from the current self._future.

        When update() replaces self._future with a merged Future, stale
        callbacks from the original Future are ignored via identity check.
        This prevents the subscribe-before-update ordering bug where
        callbacks on the original Future would resolve with un-merged data.
        """
        if f is not self._future:
            return
        try:
            result = f.result()
        except Exception as e:
            for _, on_error in self._callbacks:
                if on_error:
                    on_error(e)
                else:
                    logger.warning(
                        "DeferrableMetrics Future failed with no error " "handler: %s",
                        e,
                    )
            return
        self._data = result
        self._resolved = True
        for cb, on_error in self._callbacks:
            try:
                cb(dict(self._data))
            except Exception as cb_error:
                if on_error:
                    on_error(cb_error)
                else:
                    logger.warning(
                        "DeferrableMetrics subscribe callback raised: %s",
                        cb_error,
                    )

    def subscribe(
        self,
        callback: Callable[[dict[str, Any]], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Register a callback for when metrics are available.

        If already resolved, callback fires immediately (synchronously)
        with a defensive copy.
        If backed by Future, callback fires when Future completes.

        Safe to call before or after update(). Callbacks always receive
        the fully-merged result regardless of registration order.
        """
        if self._resolved:
            callback(dict(self._data))
        else:
            self._callbacks.append((callback, on_error))

    def resolve(self) -> dict[str, Any]:
        """Synchronously return a copy of the resolved metrics dict.

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
        return dict(self._data)

    def update(self, other: "dict[str, Any] | DeferrableMetrics") -> None:
        """Merge additional key-value pairs.

        If already resolved, merges immediately.
        If backed by Future, defers the merge until Future resolves.
        If other is a DeferrableMetrics, subscribes to it for deferred merge.
        """
        if isinstance(other, DeferrableMetrics):
            if other.is_resolved():
                self.update(other.resolve())
            else:

                def _propagate_error(e: Exception) -> None:
                    if self._future is not None and not self._future.done():
                        self._future.set_exception(e)
                    else:
                        logger.warning(
                            "DeferrableMetrics.update() received error from "
                            "source but cannot propagate: %s",
                            e,
                        )

                other.subscribe(
                    lambda data: self.update(data),
                    on_error=_propagate_error,
                )
            return

        if not isinstance(other, dict):
            raise TypeError(
                f"update() requires a dict or DeferrableMetrics, "
                f"got {type(other).__name__}"
            )
        dict_other: dict[str, Any] = other

        if self._resolved:
            self._data.update(dict_other)
        elif self._future is not None:
            original_future = self._future
            merged_future: Future[dict[str, Any]] = Future()

            def _on_complete(f: Future[dict[str, Any]]) -> None:
                try:
                    result = dict(f.result())
                    result.update(dict_other)
                    merged_future.set_result(result)
                except Exception as e:
                    merged_future.set_exception(e)

            original_future.add_done_callback(_on_complete)
            self._future = merged_future
            merged_future.add_done_callback(self._on_resolve)

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
