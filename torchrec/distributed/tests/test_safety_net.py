#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import sys
import threading
import unittest
from types import TracebackType
from typing import Optional
from unittest.mock import MagicMock, patch

import torchrec.distributed.safety_net as safety_net
from torchrec.distributed.logging_utils import EventType
from torchrec.distributed.safety_net import (
    _safe_get_rank,
    _should_log,
    install_torchrec_global_safety_net,
)


class _SafetyNetTestBase(unittest.TestCase):
    """Resets module-level state and restores excepthooks between tests."""

    def setUp(self) -> None:
        super().setUp()
        self._saved_sys_excepthook = sys.excepthook
        self._saved_threading_excepthook = threading.excepthook
        safety_net._installed = False
        safety_net._old_sys_excepthook = sys.excepthook
        safety_net._old_threading_excepthook = threading.excepthook
        safety_net._in_hook = threading.local()

    def tearDown(self) -> None:
        sys.excepthook = self._saved_sys_excepthook
        threading.excepthook = self._saved_threading_excepthook
        safety_net._installed = False
        safety_net._old_sys_excepthook = sys.excepthook
        safety_net._old_threading_excepthook = threading.excepthook
        safety_net._in_hook = threading.local()
        super().tearDown()


class InstallTest(_SafetyNetTestBase):
    def test_install_idempotent(self) -> None:
        original = sys.excepthook
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()
            after_first = sys.excepthook
            install_torchrec_global_safety_net()
            after_second = sys.excepthook

        self.assertIsNot(after_first, original)
        self.assertIs(after_first, after_second)

    def test_install_noop_when_jk_off(self) -> None:
        """Byte-exact off-path: sys.excepthook unchanged when JK is off."""
        original = sys.excepthook
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=False,
        ):
            install_torchrec_global_safety_net()
        self.assertIs(sys.excepthook, original)
        self.assertFalse(safety_net._installed)

    def test_install_passes_jk_default_false_and_switchval(self) -> None:
        # default=False keeps off-path on missing JK; switchval enables per-job override.
        with patch(
            "torchrec.distributed.safety_net._get_mast_job_name",
            return_value="aps-mygap4canary-20260516-abc123",
        ):
            with patch(
                "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
                return_value=False,
            ) as mock_jk:
                install_torchrec_global_safety_net()
        mock_jk.assert_called_once_with(
            "pytorch/torchrec:enable_torchrec_global_safety_net",
            default=False,
            switchval="aps-mygap4canary-20260516-abc123",
        )

    def test_install_safe_when_jk_check_itself_raises(self) -> None:
        original = sys.excepthook
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            side_effect=RuntimeError("JK unreachable"),
        ):
            install_torchrec_global_safety_net()
        self.assertIs(sys.excepthook, original)
        self.assertFalse(safety_net._installed)

    def test_install_serialized_across_concurrent_callers(self) -> None:
        """Two concurrent installs must produce one wrapper; smoking-gun
        is log_event call count == 1 (would be 2 if both chained)."""
        original_sys = sys.excepthook
        barrier = threading.Barrier(2)

        def _worker() -> None:
            with patch(
                "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
                return_value=True,
            ):
                barrier.wait(timeout=5.0)
                install_torchrec_global_safety_net()

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        self.assertIsNot(sys.excepthook, original_sys)
        self.assertIs(safety_net._old_sys_excepthook, original_sys)

        # JK patch must stay active at fire time (wrappers re-check JK).
        exc_type: Optional[type] = None
        exc_value: Optional[BaseException] = None
        exc_tb: Optional[TracebackType] = None
        try:
            raise RuntimeError("synthetic")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            with patch(
                "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
            ) as mock_log_event:
                sys.excepthook(exc_type, exc_value, exc_tb)
        self.assertEqual(mock_log_event.call_count, 1)

    def test_install_wraps_threading_excepthook_idempotently(self) -> None:
        original = threading.excepthook
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()
            after_first = threading.excepthook
            install_torchrec_global_safety_net()
            after_second = threading.excepthook

        self.assertIsNot(after_first, original)
        self.assertIs(after_first, after_second)


class ExcepthookFireTest(_SafetyNetTestBase):
    def _install_with_jk_on(self) -> None:
        # Persistent patcher — wrappers re-check JK on every fire.
        jk_patcher = patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        )
        jk_patcher.start()
        self.addCleanup(jk_patcher.stop)
        install_torchrec_global_safety_net()

    def _make_exc(self) -> tuple:
        try:
            raise RuntimeError("synthetic")
        except RuntimeError:
            return sys.exc_info()

    def test_uncaught_main_thread_exception_fires_failure_event(self) -> None:
        self._install_with_jk_on()
        exc_type, exc_value, exc_tb = self._make_exc()
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(exc_type, exc_value, exc_tb)

        mock_log_event.assert_called_once()
        kwargs = mock_log_event.call_args.kwargs
        self.assertEqual(kwargs["event_type"], EventType.FAILURE)
        self.assertEqual(kwargs["event_name"], "torchrec_uncaught_exception")
        self.assertEqual(kwargs["metadata"]["exception_type"], "RuntimeError")
        self.assertEqual(kwargs["error_message"], "synthetic")
        self.assertTrue(kwargs["stack_trace"])
        self.assertIn("thread_name", kwargs["metadata"])
        self.assertIn("rank", kwargs["metadata"])

    def test_uncaught_thread_exception_fires_failure_event(self) -> None:
        self._install_with_jk_on()
        exc_type, exc_value, exc_tb = self._make_exc()
        fake_thread = MagicMock()
        fake_thread.name = "TestWorker"
        args = threading.ExceptHookArgs([exc_type, exc_value, exc_tb, fake_thread])
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            threading.excepthook(args)

        mock_log_event.assert_called_once()
        kwargs = mock_log_event.call_args.kwargs
        self.assertEqual(kwargs["event_type"], EventType.FAILURE)
        self.assertEqual(kwargs["metadata"]["thread_name"], "TestWorker")

    def test_thread_excepthook_handles_none_thread(self) -> None:
        """thread=None per Python docs => thread_name='<unknown>', no crash."""
        self._install_with_jk_on()
        exc_type, exc_value, exc_tb = self._make_exc()
        args = threading.ExceptHookArgs([exc_type, exc_value, exc_tb, None])
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            threading.excepthook(args)

        mock_log_event.assert_called_once()
        self.assertEqual(
            mock_log_event.call_args.kwargs["metadata"]["thread_name"],
            "<unknown>",
        )

    def test_systemexit_zero_chained_but_not_logged(self) -> None:
        self._install_with_jk_on()
        exc = SystemExit(0)
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(SystemExit, exc, None)
        mock_log_event.assert_not_called()

    def test_systemexit_none_chained_but_not_logged(self) -> None:
        self._install_with_jk_on()
        exc = SystemExit()
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(SystemExit, exc, None)
        mock_log_event.assert_not_called()

    def test_systemexit_nonzero_is_logged(self) -> None:
        self._install_with_jk_on()
        try:
            raise SystemExit(1)
        except SystemExit:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(exc_type, exc_value, exc_tb)
        mock_log_event.assert_called_once()
        self.assertEqual(
            mock_log_event.call_args.kwargs["metadata"]["exception_type"],
            "SystemExit",
        )

    def test_keyboardinterrupt_chained_but_not_logged(self) -> None:
        self._install_with_jk_on()
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        mock_log_event.assert_not_called()

    def test_skips_non_torchrec_exception(self) -> None:
        """No torchrec frame => trainer harness owns reporting; skip emit."""
        self._install_with_jk_on()
        with patch(
            "torchrec.distributed.safety_net._torchrec_frame_in_traceback",
            return_value=None,
        ):
            with patch(
                "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
            ) as mock_log_event:
                sys.excepthook(RuntimeError, RuntimeError("user code"), None)
        mock_log_event.assert_not_called()

    def test_emits_torchrec_frame_in_metadata(self) -> None:
        self._install_with_jk_on()
        exc_type, exc_value, exc_tb = self._make_exc()
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(exc_type, exc_value, exc_tb)
        mock_log_event.assert_called_once()
        metadata = mock_log_event.call_args.kwargs["metadata"]
        self.assertIn("torchrec_frame", metadata)
        self.assertIn("/torchrec/", metadata["torchrec_frame"])

    def test_error_message_truncated(self) -> None:
        self._install_with_jk_on()
        exc_type: Optional[type] = None
        exc_value: Optional[BaseException] = None
        exc_tb: Optional[TracebackType] = None
        try:
            raise RuntimeError("x" * 10_000)
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        assert exc_type is not None and exc_value is not None
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
        ) as mock_log_event:
            sys.excepthook(exc_type, exc_value, exc_tb)
        error_message = mock_log_event.call_args.kwargs["error_message"]
        self.assertLessEqual(len(error_message), 4096)
        self.assertTrue(error_message.endswith("...[truncated]"))

    def test_stack_trace_truncated(self) -> None:
        # Mock format_exception to avoid depending on Python's frame compression.
        self._install_with_jk_on()
        exc_type, exc_value, exc_tb = self._make_exc()
        with patch(
            "torchrec.distributed.safety_net.traceback.format_exception",
            return_value=["x"] * 20_000,
        ):
            with patch(
                "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
            ) as mock_log_event:
                sys.excepthook(exc_type, exc_value, exc_tb)
        stack_trace = mock_log_event.call_args.kwargs["stack_trace"]
        self.assertLessEqual(len(stack_trace), 8192)
        self.assertTrue(stack_trace.endswith("...[truncated]"))


class ChainTest(_SafetyNetTestBase):
    def test_chain_preserves_old_sys_excepthook(self) -> None:
        old_hook_calls: list = []

        def fake_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            old_hook_calls.append((t, v))

        sys.excepthook = fake_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()

        exc = RuntimeError("test")
        with patch("torchrec.distributed.safety_net.EventLoggingHandler.log_event"):
            sys.excepthook(RuntimeError, exc, None)

        self.assertEqual(len(old_hook_calls), 1)
        self.assertIs(old_hook_calls[0][1], exc)

    def test_hook_never_raises_when_log_event_fails(self) -> None:
        """log_event raise must not propagate AND chain must still fire."""
        chain_called: list[bool] = [False]

        def fake_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            chain_called[0] = True

        sys.excepthook = fake_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()
        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event",
            side_effect=RuntimeError("Scuba down"),
        ):
            sys.excepthook(RuntimeError, RuntimeError("real"), None)
        self.assertTrue(
            chain_called[0],
            "chained excepthook must fire even when log_event raises",
        )

    def test_hook_never_raises_when_chain_fails(self) -> None:
        """Chained hook raise must not propagate AND must have been attempted."""
        chain_called: list[bool] = [False]

        def bad_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            chain_called[0] = True
            raise RuntimeError("chain explode")

        sys.excepthook = bad_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()
        with patch("torchrec.distributed.safety_net.EventLoggingHandler.log_event"):
            sys.excepthook(RuntimeError, RuntimeError("real"), None)
        self.assertTrue(
            chain_called[0],
            "chained excepthook must be attempted even when it raises",
        )

    def test_chain_fires_even_when_should_log_raises(self) -> None:
        """Pre-chain helper raising must not skip the chain (rank stderr)."""
        chain_called: list[bool] = [False]

        def fake_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            chain_called[0] = True

        sys.excepthook = fake_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()

        with patch(
            "torchrec.distributed.safety_net._should_log",
            side_effect=RuntimeError("simulated malformed exc_type"),
        ):
            try:
                raise RuntimeError("real")
            except RuntimeError:
                exc_type, exc_value, exc_tb = sys.exc_info()
            assert exc_type is not None and exc_value is not None
            sys.excepthook(exc_type, exc_value, exc_tb)

        self.assertTrue(
            chain_called[0],
            "chained excepthook must fire even when a pre-chain helper "
            "(_should_log) raises — otherwise rank-prefixed stderr is lost",
        )

    def test_jk_flipped_off_after_install_acts_as_kill_switch(self) -> None:
        """Flipping JK off mid-run stops emit but chain still fires."""
        jk_state = {"on": True}

        def jk_check_fn(name: str, default: bool = True, **kwargs: object) -> bool:
            return jk_state["on"]

        chain_called: list[bool] = [False]

        def fake_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            chain_called[0] = True

        sys.excepthook = fake_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            side_effect=jk_check_fn,
        ):
            install_torchrec_global_safety_net()
            self.assertTrue(safety_net._installed)

            jk_state["on"] = False

            try:
                raise RuntimeError("synthetic")
            except RuntimeError:
                exc_type, exc_value, exc_tb = sys.exc_info()
            assert exc_type is not None and exc_value is not None
            with patch(
                "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
            ) as mock_log_event:
                sys.excepthook(exc_type, exc_value, exc_tb)

            mock_log_event.assert_not_called()
            self.assertTrue(
                chain_called[0], "chain must fire even when JK is off mid-run"
            )

    def test_chain_fires_even_when_telemetry_and_logger_both_raise(self) -> None:
        """log_event + logger.exception both raising must not suppress chain."""
        chain_called: list[bool] = [False]

        def fake_old(t: type, v: BaseException, tb: Optional[TracebackType]) -> None:
            chain_called[0] = True

        sys.excepthook = fake_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()

        with patch(
            "torchrec.distributed.safety_net.EventLoggingHandler.log_event",
            side_effect=RuntimeError("scuba down"),
        ):
            with patch.object(
                safety_net.logger,
                "exception",
                side_effect=RuntimeError("logging handler down"),
            ):
                try:
                    raise RuntimeError("real")
                except RuntimeError:
                    exc_type, exc_value, exc_tb = sys.exc_info()
                assert exc_type is not None and exc_value is not None
                sys.excepthook(exc_type, exc_value, exc_tb)

        self.assertTrue(
            chain_called[0],
            "chained excepthook must fire even when telemetry + logger "
            "both raise — otherwise rank-prefixed stderr is lost",
        )

    def test_reentry_guard(self) -> None:
        """Recursive sys.excepthook re-entry must not loop; log_event fires once."""

        recursion_depth = {"value": 0}

        def recursive_old(
            t: type, v: BaseException, tb: Optional[TracebackType]
        ) -> None:
            recursion_depth["value"] += 1
            if recursion_depth["value"] < 3:
                sys.excepthook(t, v, tb)

        sys.excepthook = recursive_old
        with patch(
            "torchrec.distributed.safety_net.torch._utils_internal.justknobs_check",
            return_value=True,
        ):
            install_torchrec_global_safety_net()
            try:
                raise RuntimeError("real")
            except RuntimeError:
                exc_type, exc_value, exc_tb = sys.exc_info()
            assert exc_type is not None and exc_value is not None
            with patch(
                "torchrec.distributed.safety_net.EventLoggingHandler.log_event"
            ) as mock_log_event:
                sys.excepthook(exc_type, exc_value, exc_tb)

        self.assertEqual(mock_log_event.call_count, 1)


class HelperTest(_SafetyNetTestBase):
    def test_safe_get_rank_fallback_to_env_var(self) -> None:
        with patch.object(safety_net, "torch") as mock_torch:
            mock_torch.distributed.is_available.return_value = True
            mock_torch.distributed.is_initialized.return_value = False
            with patch.dict(os.environ, {"RANK": "3"}, clear=True):
                self.assertEqual(_safe_get_rank(), 3)

    def test_safe_get_rank_fallback_to_local_rank_env_var(self) -> None:
        with patch.object(safety_net, "torch") as mock_torch:
            mock_torch.distributed.is_available.return_value = True
            mock_torch.distributed.is_initialized.return_value = False
            with patch.dict(os.environ, {"LOCAL_RANK": "5"}, clear=True):
                self.assertEqual(_safe_get_rank(), 5)

    def test_safe_get_rank_fallback_to_minus_one(self) -> None:
        with patch.object(safety_net, "torch") as mock_torch:
            mock_torch.distributed.is_available.return_value = True
            mock_torch.distributed.is_initialized.return_value = False
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_safe_get_rank(), -1)

    def test_torchrec_frame_in_traceback_detects_real_torchrec_path(
        self,
    ) -> None:
        try:
            raise RuntimeError("test")
        except RuntimeError:
            _, _, tb = sys.exc_info()
        result = safety_net._torchrec_frame_in_traceback(tb)
        self.assertIsNotNone(result)
        # pyre-ignore[16]: result is not None per the assertion above
        self.assertIn("/torchrec/", result)

    def test_torchrec_frame_in_traceback_returns_none_for_non_torchrec(
        self,
    ) -> None:
        fake_frame = MagicMock()
        fake_frame.f_code.co_filename = "/usr/lib/python3.12/queue.py"
        fake_tb = MagicMock()
        fake_tb.tb_frame = fake_frame
        fake_tb.tb_lineno = 42
        fake_tb.tb_next = None
        self.assertIsNone(safety_net._torchrec_frame_in_traceback(fake_tb))

    def test_torchrec_frame_in_traceback_skips_user_code_passing_through_torchrec(
        self,
    ) -> None:
        """DMP.forward => user model code raise must attribute to user, not torchrec."""
        outer_frame = MagicMock()
        outer_frame.f_code.co_filename = (
            "/data/.../torchrec/distributed/model_parallel.py"
        )
        inner_frame = MagicMock()
        inner_frame.f_code.co_filename = "/data/users/foo/my_app/model.py"

        inner_tb = MagicMock()
        inner_tb.tb_frame = inner_frame
        inner_tb.tb_lineno = 99
        inner_tb.tb_next = None

        outer_tb = MagicMock()
        outer_tb.tb_frame = outer_frame
        outer_tb.tb_lineno = 300
        outer_tb.tb_next = inner_tb

        self.assertIsNone(safety_net._torchrec_frame_in_traceback(outer_tb))

    def test_torchrec_frame_in_traceback_attributes_torchrec_to_dep(
        self,
    ) -> None:
        """torchrec => dep raise (no user code in chain) attributes to torchrec."""
        torchrec_frame = MagicMock()
        torchrec_frame.f_code.co_filename = (
            "/data/.../torchrec/distributed/embedding_lookup.py"
        )
        dep_frame = MagicMock()
        dep_frame.f_code.co_filename = "/data/.../torch/nn/functional.py"

        innermost_tb = MagicMock()
        innermost_tb.tb_frame = dep_frame
        innermost_tb.tb_lineno = 555
        innermost_tb.tb_next = None

        torchrec_tb = MagicMock()
        torchrec_tb.tb_frame = torchrec_frame
        torchrec_tb.tb_lineno = 200
        torchrec_tb.tb_next = innermost_tb

        result = safety_net._torchrec_frame_in_traceback(torchrec_tb)
        self.assertIsNotNone(result)
        self.assertIn("embedding_lookup.py", result or "")
        self.assertIn("200", result or "")

    def test_torchrec_frame_in_traceback_skips_when_user_code_below_torchrec(
        self,
    ) -> None:
        """DMP.forward => user => torch.X: user frame in chain => skip."""
        torchrec_frame = MagicMock()
        torchrec_frame.f_code.co_filename = (
            "/data/.../torchrec/distributed/model_parallel.py"
        )
        user_frame = MagicMock()
        user_frame.f_code.co_filename = "/data/users/foo/my_app/model.py"
        dep_frame = MagicMock()
        dep_frame.f_code.co_filename = "/data/.../torch/nn/functional.py"

        innermost_tb = MagicMock()
        innermost_tb.tb_frame = dep_frame
        innermost_tb.tb_lineno = 555
        innermost_tb.tb_next = None

        user_tb = MagicMock()
        user_tb.tb_frame = user_frame
        user_tb.tb_lineno = 42
        user_tb.tb_next = innermost_tb

        torchrec_tb = MagicMock()
        torchrec_tb.tb_frame = torchrec_frame
        torchrec_tb.tb_lineno = 300
        torchrec_tb.tb_next = user_tb

        self.assertIsNone(safety_net._torchrec_frame_in_traceback(torchrec_tb))

    def test_is_torchrec_dependency(self) -> None:
        self.assertTrue(
            safety_net._is_torchrec_dependency("/data/.../torch/nn/functional.py")
        )
        self.assertTrue(
            safety_net._is_torchrec_dependency("/data/.../fbgemm/something.py")
        )
        self.assertTrue(
            safety_net._is_torchrec_dependency(
                "/data/.../deeplearning/fbgemm/embeddings.py"
            )
        )
        self.assertFalse(
            safety_net._is_torchrec_dependency("/data/users/foo/my_app/model.py")
        )
        self.assertFalse(
            safety_net._is_torchrec_dependency("/usr/lib/python3.12/queue.py")
        )

    def test_torchrec_frame_in_traceback_attributes_when_innermost_is_torchrec(
        self,
    ) -> None:
        """user => torchrec raise: innermost is torchrec, attribute to torchrec."""
        outer_frame = MagicMock()
        outer_frame.f_code.co_filename = "/data/users/foo/my_app/main.py"
        inner_frame = MagicMock()
        inner_frame.f_code.co_filename = (
            "/data/.../torchrec/distributed/embedding_lookup.py"
        )

        inner_tb = MagicMock()
        inner_tb.tb_frame = inner_frame
        inner_tb.tb_lineno = 123
        inner_tb.tb_next = None

        outer_tb = MagicMock()
        outer_tb.tb_frame = outer_frame
        outer_tb.tb_lineno = 50
        outer_tb.tb_next = inner_tb

        result = safety_net._torchrec_frame_in_traceback(outer_tb)
        self.assertIsNotNone(result)
        self.assertIn("embedding_lookup.py", result or "")
        self.assertIn("123", result or "")

    def test_truncate_under_limit_unchanged(self) -> None:
        s = "x" * 100
        self.assertEqual(safety_net._truncate(s, 4096), s)

    def test_truncate_over_limit_marked(self) -> None:
        s = "x" * 10_000
        out = safety_net._truncate(s, 4096)
        self.assertLessEqual(len(out), 4096)
        self.assertTrue(out.endswith("...[truncated]"))

    def test_get_mast_job_name_returns_empty_when_runtime_env_unavailable(
        self,
    ) -> None:
        """OSS / torch-package: fb-internal binding unavailable => ''."""
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop(
                "aiplatform.runtime_environment.runtime_environment_pybind", None
            )
            with patch(
                "builtins.__import__",
                side_effect=ImportError("fb-internal not available"),
            ):
                self.assertEqual(safety_net._get_mast_job_name(), "")

    def test_get_mast_job_name_returns_empty_when_lookup_raises(self) -> None:
        """Binding exists but lookup raises => still degrades to ''."""
        fake_re = MagicMock()
        fake_re.return_value.get_mast_job_name.side_effect = RuntimeError(
            "service down"
        )
        fake_module = MagicMock()
        fake_module.RuntimeEnvironment = fake_re
        with patch.dict(
            sys.modules,
            {"aiplatform.runtime_environment.runtime_environment_pybind": fake_module},
        ):
            self.assertEqual(safety_net._get_mast_job_name(), "")

    def test_should_log_filter(self) -> None:
        self.assertFalse(_should_log(KeyboardInterrupt, KeyboardInterrupt()))
        self.assertFalse(_should_log(SystemExit, SystemExit()))
        self.assertFalse(_should_log(SystemExit, SystemExit(0)))
        self.assertTrue(_should_log(SystemExit, SystemExit(1)))
        self.assertTrue(_should_log(SystemExit, SystemExit("error")))
        self.assertTrue(_should_log(RuntimeError, RuntimeError("x")))
        self.assertTrue(_should_log(ValueError, ValueError("x")))


if __name__ == "__main__":
    unittest.main()
