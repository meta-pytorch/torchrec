#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import os
import pkgutil
import tempfile
import threading
import time
from typing import Any, Callable, Optional

import torch

logger: logging.Logger = logging.getLogger(__name__)

_PROFILE_STOP_GRACE_SEC: float = 30.0

# "module:function" resolved in the worker; the function takes (path, rank) and
# may return a URL for the uploaded trace.
TraceUploader = str


class WorkerProfiler:
    """Best-effort CPU profiler for the metric worker; its failures never reach
    metric processing.

    Either follows the trainer's profiler (start/stop control messages tagged
    with the trainer's update seq) or covers a fixed window of update seqs.
    A stop waits for computes requested inside the window so their compute
    thread work lands in the same trace.
    """

    def __init__(
        self,
        window: Optional[tuple[int, int]],
        rank: int,
        uploader: Optional[TraceUploader],
        on_uploaded: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_uploaded = on_uploaded
        self._fixed_window: Optional[tuple[int, int]] = (
            (window[0], window[0] + window[1]) if window else None
        )
        self._rank = rank
        self._uploader = uploader
        self._profiler: Optional[torch.profiler.profile] = None
        self._window_computes: list[concurrent.futures.Future[Any]] = []
        self._stop_deadline: Optional[float] = None
        self._exports: list[threading.Thread] = []

    @property
    def active(self) -> bool:
        return self._profiler is not None

    def on_seq(self, seq: int) -> None:
        """Called for every consumed update; a fixed window starts and stops at
        its bounds, and every other seq is a no-op."""
        if self._fixed_window is None:
            return
        start, stop = self._fixed_window
        if seq == start:
            self.start(seq)
        elif seq == stop:
            self.request_stop(seq)

    def start(self, seq: int) -> None:
        self.maybe_stop(force=True)
        if self._profiler is not None:
            return
        try:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                experimental_config=torch.profiler._ExperimentalConfig(
                    profile_all_threads=True
                ),
            )
            profiler.start()
        except Exception:
            logger.exception(f"metric worker profiler failed to start at seq {seq}")
            return
        self._profiler = profiler
        self._window_computes = []
        self._stop_deadline = None
        logger.info(f"metric worker profiler started at seq {seq}")

    def track_compute(self, future: concurrent.futures.Future[Any]) -> None:
        if self._profiler is not None and self._stop_deadline is None:
            self._window_computes.append(future)

    def request_stop(self, seq: int) -> None:
        if self._profiler is not None and self._stop_deadline is None:
            self._stop_deadline = time.monotonic() + _PROFILE_STOP_GRACE_SEC
            logger.info(
                f"metric worker profiler stop requested at seq {seq}; waiting for "
                f"{len(self._window_computes)} in-window computes"
            )

    def maybe_stop(self, force: bool = False) -> None:
        if self._profiler is None or self._stop_deadline is None:
            return
        pending = [f for f in self._window_computes if not f.done()]
        if pending and not force and time.monotonic() < self._stop_deadline:
            return
        profiler, self._profiler = self._profiler, None
        try:
            profiler.stop()
        except Exception:
            logger.exception("metric worker profiler failed to stop")
            return
        logger.info(
            f"metric worker profiler stopped ({len(pending)} computes still pending)"
        )
        export = threading.Thread(target=self._export, args=(profiler,), daemon=True)
        export.start()
        self._exports = [t for t in self._exports if t.is_alive()] + [export]

    def close(self, timeout: float) -> None:
        """Stops an active profile and waits up to `timeout` for trace exports."""
        if self._profiler is not None and self._stop_deadline is None:
            self._stop_deadline = time.monotonic()
        self.maybe_stop(force=True)
        deadline = time.monotonic() + timeout
        for export in self._exports:
            export.join(max(0.0, deadline - time.monotonic()))

    def _export(self, profiler: torch.profiler.profile) -> None:
        path = os.path.join(
            tempfile.gettempdir(),
            f"zorm_metric_worker_rank{self._rank}_{os.getpid()}_{time.time_ns()}"
            ".pt.trace.json.gz",
        )
        try:
            profiler.export_chrome_trace(path)
            if self._uploader is None:
                logger.info(f"metric worker trace written to {path}")
            else:
                self._upload(self._uploader, path)
        except Exception:
            logger.exception("metric worker trace export failed")

    def _upload(self, uploader: TraceUploader, path: str) -> None:
        # The uploaded copy is the trace; the local file is scratch either way.
        try:
            url = pkgutil.resolve_name(uploader)(path, self._rank)
        finally:
            os.remove(path)
        if url and self._on_uploaded is not None:
            self._on_uploaded(url)
