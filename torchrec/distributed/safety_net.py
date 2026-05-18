#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Process-global uncaught-exception safety net for torchrec.

Installs ``sys.excepthook`` and ``threading.excepthook`` wrappers that
emit a FAILURE event to ``torchrec_event_logging`` for torchrec-attributable
uncaught exceptions, then chain to the previously-installed hooks so
PyTorch's rank-prefixed stderr behavior is preserved.

JK-gated (``pytorch/torchrec:enable_torchrec_global_safety_net``,
default off). Off-path is "do not install" — byte-exact.

Known limitations:
- Pre-DMP exceptions (planner helpers without ``@event_logger``) miss.
- No SIGTERM/SIGINT/faulthandler coverage — owned by APFShutdownHandler
  and torchelastic ErrorHandler.
- Wrapper helper sitting between torchrec and a dep raise (frame in
  neither ``/torchrec/`` nor ``_TORCHREC_DEPENDENCY_MARKERS``) looks
  like user code and is skipped.
"""

import logging
import os
import sys
import threading
import traceback
from types import TracebackType
from typing import Callable, Optional, Type

import torch

try:
    # Defensive: torch-package / inference builds may strip the shim.
    from torchrec.distributed.logging_handlers import (
        EventLoggingHandler,
        TorchrecComponent,
    )
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.distributed.safety_net.import_failure.event_logging_handler"
    )

    from enum import Enum as _Enum
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from torchrec.distributed.logging_handlers import (
            EventLoggingHandler,
            TorchrecComponent,
        )
    else:

        class TorchrecComponent(_Enum):
            TRAIN_PIPELINE = "train_pipeline"

        class EventLoggingHandler:
            @staticmethod
            def log_event(*args: object, **kwargs: object) -> None:
                pass


from torchrec.distributed.logging_utils import EventType

logger: logging.Logger = logging.getLogger(__name__)

_SAFETY_NET_JK: str = "pytorch/torchrec:enable_torchrec_global_safety_net"
_UNCAUGHT_EVENT_NAME: str = "torchrec_uncaught_exception"

# Bound payload sizes so a pathological exception can't blow up the Scuba sample.
_ERROR_MESSAGE_MAX_LEN: int = 4096
_STACK_TRACE_MAX_LEN: int = 8192
_TRUNCATION_MARKER: str = "...[truncated]"

_TORCHREC_PATH_MARKER: str = "/torchrec/"

# Frames in these libs count as "torchrec called a dep that raised";
# anything else between the deepest torchrec frame and the leaf is user code.
_TORCHREC_DEPENDENCY_MARKERS: tuple = (
    "/torch/",
    "/fbgemm/",
    "/deeplearning/fbgemm/",
)

_installed: bool = False
# Without the lock, two concurrent DMP constructions can each capture
# sys.excepthook (one capturing the other's wrapper) and double-chain.
_install_lock: threading.Lock = threading.Lock()
_in_hook: threading.local = threading.local()
_old_sys_excepthook: Callable[
    [Type[BaseException], BaseException, Optional[TracebackType]], None
] = sys.excepthook
_old_threading_excepthook: Callable[["threading.ExceptHookArgs"], object] = (
    threading.excepthook
)
_component: str = TorchrecComponent.TRAIN_PIPELINE.value


# @dep=//aiplatform/runtime_environment:runtime_environment_pybind
# @dep=//aiplatform/runtime_environment:runtime_environment_pybind_types
# Annotations force autodeps to bundle the pybind module; without them
# the function-scope import silently fails in Buck binaries and per-job
# JK switchval targeting degrades to fleet-wide enablement.
def _get_mast_job_name() -> str:
    """Best-effort MAST job name for JK switchval. ``""`` when not on
    MAST or when the fb-internal binding is unavailable (OSS, torch-package)."""
    try:
        from aiplatform.runtime_environment.runtime_environment_pybind import (  # @manual
            RuntimeEnvironment,
        )

        return RuntimeEnvironment().get_mast_job_name() or ""
    except Exception:
        return ""


def _safe_get_rank() -> int:
    """Rank with fallback to torchelastic's RANK env var, then -1.
    The hook can fire before ``dist.init_process_group()``."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank_env is not None:
        try:
            return int(rank_env)
        except ValueError:
            pass
    return -1


def _should_log(exc_type: Type[BaseException], exc_value: BaseException) -> bool:
    """Skip KeyboardInterrupt and SystemExit(None|0); log SystemExit(nonzero)."""
    if issubclass(exc_type, KeyboardInterrupt):
        return False
    if issubclass(exc_type, SystemExit):
        code = getattr(exc_value, "code", None)
        if code is None or code == 0:
            return False
    return True


def _is_torchrec_dependency(filename: str) -> bool:
    return any(marker in filename for marker in _TORCHREC_DEPENDENCY_MARKERS)


def _torchrec_frame_in_traceback(
    exc_tb: Optional[TracebackType],
) -> Optional[str]:
    """Deepest torchrec ``"filename:lineno"`` if the failure is
    torchrec-attributable, else ``None``. Attributable iff every frame
    below the deepest torchrec frame is torchrec or a known dep — any
    other frame means the chain handed off to user code (skip)."""
    tb = exc_tb
    deepest_torchrec_tb: Optional[TracebackType] = None
    deepest_torchrec_frame: Optional[str] = None
    while tb is not None:
        try:
            filename = tb.tb_frame.f_code.co_filename
        except Exception:
            filename = ""
        if _TORCHREC_PATH_MARKER in filename:
            deepest_torchrec_tb = tb
            deepest_torchrec_frame = f"{filename}:{tb.tb_lineno}"
        tb = tb.tb_next

    if deepest_torchrec_tb is None:
        return None

    walker = deepest_torchrec_tb.tb_next
    while walker is not None:
        try:
            fn = walker.tb_frame.f_code.co_filename
        except Exception:
            fn = ""
        if _TORCHREC_PATH_MARKER not in fn and not _is_torchrec_dependency(fn):
            return None
        walker = walker.tb_next

    return deepest_torchrec_frame


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    keep = max(0, max_len - len(_TRUNCATION_MARKER))
    return s[:keep] + _TRUNCATION_MARKER


def _emit_failure_event(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
    thread_name: str,
) -> None:
    """Emit one FAILURE row for torchrec-attributable exceptions. May raise;
    caller owns the outer try/except."""
    torchrec_frame = _torchrec_frame_in_traceback(exc_tb)
    if torchrec_frame is None:
        return
    EventLoggingHandler.log_event(
        component=_component,
        event_name=_UNCAUGHT_EVENT_NAME,
        event_type=EventType.FAILURE,
        metadata={
            "exception_type": exc_type.__name__,
            "thread_name": thread_name,
            "rank": str(_safe_get_rank()),
            "torchrec_frame": torchrec_frame,
        },
        error_message=_truncate(str(exc_value), _ERROR_MESSAGE_MAX_LEN),
        stack_trace=_truncate(
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            _STACK_TRACE_MAX_LEN,
        ),
    )


def _jk_still_enabled() -> bool:
    """Per-fire JK recheck so flipping the JK off acts as a runtime kill
    switch on already-installed wrappers. Fail safe-off."""
    try:
        return torch._utils_internal.justknobs_check(
            _SAFETY_NET_JK, default=False, switchval=_get_mast_job_name()
        )
    except Exception:
        return False


def _safe_diag_log(msg: str) -> None:
    # logger.exception() can itself raise during interpreter teardown;
    # never let it skip the chain.
    try:
        logger.exception(msg)
    except BaseException:  # noqa: B036
        pass


def _safe_chain_sys(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    try:
        _old_sys_excepthook(exc_type, exc_value, exc_tb)
    except BaseException:  # noqa: B036
        _safe_diag_log("safety_net: chain failed")


def _safe_chain_threading(args: "threading.ExceptHookArgs") -> None:
    try:
        _old_threading_excepthook(args)
    except BaseException:  # noqa: B036
        _safe_diag_log("safety_net: chain failed")


def _run_telemetry_sys(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    # Telemetry must never skip the chain in the caller — wrap broadly.
    try:
        if _jk_still_enabled() and _should_log(exc_type, exc_value):
            try:
                _emit_failure_event(
                    exc_type,
                    exc_value,
                    exc_tb,
                    threading.current_thread().name,
                )
            except Exception:
                _safe_diag_log("safety_net: log_event failed")
    except BaseException:  # noqa: B036
        _safe_diag_log("safety_net: telemetry path raised")


def _run_telemetry_threading(args: "threading.ExceptHookArgs") -> None:
    try:
        exc_type = args.exc_type
        exc_value = args.exc_value
        exc_tb = args.exc_traceback
        thread = args.thread
        thread_name = thread.name if thread is not None else "<unknown>"
        # exc_value may be None per the threading.excepthook contract.
        if exc_value is not None and exc_type is not None:
            if _jk_still_enabled() and _should_log(exc_type, exc_value):
                try:
                    _emit_failure_event(exc_type, exc_value, exc_tb, thread_name)
                except Exception:
                    _safe_diag_log("safety_net: log_event failed")
    except BaseException:  # noqa: B036
        _safe_diag_log("safety_net: telemetry path raised")


def _sys_excepthook_wrapper(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: Optional[TracebackType],
) -> None:
    # Must never escape — corrupting process exit would hide the original.
    try:
        if getattr(_in_hook, "active", False):
            _safe_chain_sys(exc_type, exc_value, exc_tb)
            return
        _in_hook.active = True
        try:
            _run_telemetry_sys(exc_type, exc_value, exc_tb)
            _safe_chain_sys(exc_type, exc_value, exc_tb)
        finally:
            _in_hook.active = False
    except BaseException:  # noqa: B036
        pass


def _threading_excepthook_wrapper(args: "threading.ExceptHookArgs") -> None:
    try:
        if getattr(_in_hook, "active", False):
            _safe_chain_threading(args)
            return
        _in_hook.active = True
        try:
            _run_telemetry_threading(args)
            _safe_chain_threading(args)
        finally:
            _in_hook.active = False
    except BaseException:  # noqa: B036
        pass


def install_torchrec_global_safety_net(
    component: str = TorchrecComponent.TRAIN_PIPELINE.value,
) -> None:
    """Install the process-global excepthook safety net. Idempotent and
    JK-gated. Call from ``DistributedModelParallel.__init__`` so install
    runs after ``dist.init_process_group()`` and chains PyTorch's
    rank-prefixing ``_distributed_excepthook``.

    JK-off path is byte-exact (hooks unchanged) per the killswitch-fallback
    rule. Trade-off: flipping JK on after a job starts does not retroactively
    install on running processes — only DMP constructions that see JK=true
    install. ``_jk_still_enabled()`` recheck makes flipping JK off a
    runtime kill switch.
    """
    global _installed, _old_sys_excepthook, _old_threading_excepthook
    global _component
    # Double-checked locking — fast path avoids the lock once installed.
    if _installed:
        return
    try:
        # switchval=mast_job_name enables per-job admin-UI overrides.
        if not torch._utils_internal.justknobs_check(
            _SAFETY_NET_JK,
            default=False,
            switchval=_get_mast_job_name(),
        ):
            return
    except Exception:
        return

    with _install_lock:
        if _installed:
            return
        _component = component
        _old_sys_excepthook = sys.excepthook
        _old_threading_excepthook = threading.excepthook
        sys.excepthook = _sys_excepthook_wrapper
        threading.excepthook = _threading_excepthook_wrapper
        _installed = True
    logger.info(
        "torchrec safety net installed: sys.excepthook + "
        "threading.excepthook now route uncaught exceptions to "
        f"{_UNCAUGHT_EVENT_NAME} in torchrec_event_logging"
    )
