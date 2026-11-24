#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing
import multiprocessing.util
import os
import signal
import tempfile
import threading
import unittest
from multiprocessing.pool import Pool
from pathlib import Path
from types import FrameType

from torchrec.distributed.test_utils.multi_process import (
    _close_and_join_pool,
    _picklable_exception_wrapper,
    MultiProcessTestBase,
)


_RUNNER_TIMEOUT_SECONDS = 60
_WORLD_SIZE = 2


class _Abort(BaseException):
    """Derives from BaseException directly, like DCP's CheckpointException."""


def _raise(exception: BaseException) -> None:
    raise exception


def _abort_worker(rank: int, world_size: int) -> None:
    raise _Abort(f"boom on rank {rank}")


def _hold_worker_exit(
    rank: int,
    world_size: int,
    entered: list[threading.Event],
    release: threading.Event,
    start: threading.Barrier,
    marker_dir: str,
    fail_rank: int | None = None,
) -> None:
    # The barrier puts each rank on its own worker. The exit hook stands in for
    # the coverage save: it reports that the worker is exiting, waits until
    # every rank is exiting, then holds until the test releases it. `atexit`
    # does not run in pool workers.
    start.wait()

    def _exit_hook() -> None:
        entered[rank].set()
        for event in entered:
            event.wait()
        release.wait()
        Path(marker_dir, f"rank{rank}").touch()

    multiprocessing.util.Finalize(None, _exit_hook, exitpriority=10)
    if rank == fail_rank:
        raise RuntimeError(f"boom on rank {rank}")


class PicklableExceptionWrapperTest(unittest.TestCase):
    def test_reports_a_base_exception(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Worker raised _Abort: boom"):
            _picklable_exception_wrapper(_raise, _Abort("boom"))

    def test_interrupts_and_exits_propagate(self) -> None:
        for exception in (KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(exception=type(exception).__name__):
                with self.assertRaises(type(exception)):
                    _picklable_exception_wrapper(_raise, exception)


def _timed_out(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("the runner did not report the worker's BaseException")


class MultiProcessRunnerTest(MultiProcessTestBase):
    def test_worker_base_exception_fails_the_test(self) -> None:
        # The runner waits on its workers without a timeout, so a regression
        # would hang; the alarm turns that into a prompt failure. Leaving the
        # Pool's `with` block terminates the workers.
        previous = signal.signal(signal.SIGALRM, _timed_out)
        signal.alarm(_RUNNER_TIMEOUT_SECONDS)
        try:
            with self.assertRaisesRegex(
                AssertionError, "Worker raised _Abort: boom on rank"
            ):
                self._run_multi_process_test(callable=_abort_worker, world_size=2)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


class PoolShutdownTest(MultiProcessTestBase):
    def setUp(self) -> None:
        super().setUp()
        manager = multiprocessing.get_context(self._mp_init_mode).Manager()
        self.addCleanup(manager.shutdown)
        self._entered: list[threading.Event] = [
            manager.Event() for _ in range(_WORLD_SIZE)
        ]
        self._release: threading.Event = manager.Event()
        self._start: threading.Barrier = manager.Barrier(_WORLD_SIZE)
        marker_dir = tempfile.TemporaryDirectory()
        self.addCleanup(marker_dir.cleanup)
        self._marker_dir: str = marker_dir.name

    def _wait_until_all_exiting(self) -> None:
        for event in self._entered:
            self.assertTrue(event.wait(_RUNNER_TIMEOUT_SECONDS))

    def _markers(self) -> list[str]:
        return sorted(os.listdir(self._marker_dir))

    def _run_ranks(self, pool: Pool) -> None:
        pool.starmap(
            _hold_worker_exit,
            [
                (
                    rank,
                    _WORLD_SIZE,
                    self._entered,
                    self._release,
                    self._start,
                    self._marker_dir,
                )
                for rank in range(_WORLD_SIZE)
            ],
            chunksize=1,
        )

    def test_close_and_join_waits_for_worker_exit(self) -> None:
        ctx = multiprocessing.get_context(self._mp_init_mode)
        with ctx.Pool(processes=_WORLD_SIZE) as pool:
            self._run_ranks(pool)
            closer = threading.Thread(target=_close_and_join_pool, args=(pool,))
            closer.start()
            self._wait_until_all_exiting()
            # Both workers are held in their exit hooks.
            self.assertTrue(closer.is_alive())
            self._release.set()
            closer.join()
        self.assertEqual(self._markers(), ["rank0", "rank1"])

    def test_close_and_join_gives_up_after_timeout(self) -> None:
        ctx = multiprocessing.get_context(self._mp_init_mode)
        with ctx.Pool(processes=_WORLD_SIZE) as pool:
            self._run_ranks(pool)
            # The test never releases the workers, so they cannot exit.
            _close_and_join_pool(pool, timeout=0.2)
            self._wait_until_all_exiting()
        self.assertEqual(self._markers(), [])

    def test_failed_rank_still_lets_workers_exit(self) -> None:
        # The exit hooks still hold until every rank is exiting.
        self._release.set()
        with self.assertRaisesRegex(AssertionError, "boom on rank 1"):
            self._run_multi_process_test(
                callable=_hold_worker_exit,
                world_size=_WORLD_SIZE,
                entered=self._entered,
                release=self._release,
                start=self._start,
                marker_dir=self._marker_dir,
                fail_rank=1,
            )
        self.assertEqual(self._markers(), ["rank0", "rank1"])
