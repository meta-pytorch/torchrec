#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import signal
import unittest
from types import FrameType

from torchrec.distributed.test_utils.multi_process import (
    _picklable_exception_wrapper,
    MultiProcessTestBase,
)


_RUNNER_TIMEOUT_SECONDS = 60


class _Abort(BaseException):
    """Derives from BaseException directly, like DCP's CheckpointException."""


def _raise(exception: BaseException) -> None:
    raise exception


def _abort_worker(rank: int, world_size: int) -> None:
    raise _Abort(f"boom on rank {rank}")


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
