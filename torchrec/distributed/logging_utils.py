#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import contextlib
from enum import Enum, unique
from typing import Dict, Generator, Optional


@unique
class EventType(Enum):
    """Type of lifecycle event being logged."""

    START = "START"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INFO = "INFO"


@unique
class StackLayer(Enum):
    """Layer in the training stack where the event originates."""

    TORCHREC = "torchrec"
    FBGEMM = "fbgemm"
    FRAMEWORK = "framework"
    DPP = "dpp"


@unique
class OptimizationTechnique(Enum):
    """Training optimization techniques."""

    NONE = "none"
    EMO = "emo"
    ITEP = "itep"
    ALBT = "albt"
    SSD_OFFLOADING = "ssd_offloading"


@unique
class EventScope(Enum):
    """Scope of the logged event."""

    JOB = "job"
    TABLE = "table"
    TBE = "tbe"
    MODULE = "module"


class EventLoggingHandlerBase(abc.ABC):
    """Abstract base class for event logging handlers.

    Defines the interface that both the internal (Scuba-backed) and
    open-source (no-op) ``EventLoggingHandler`` implementations must satisfy.
    Functions that accept an event logging handler should type-hint against
    this base class.
    """

    batches_processed: int = 0

    @classmethod
    def update_batches_processed(cls, batches_processed: int) -> None:
        """Update the class-level batches processed counter."""
        cls.batches_processed = batches_processed

    @classmethod
    @abc.abstractmethod
    def log_event(
        cls,
        component: str,
        event_name: str,
        event_type: EventType,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None: ...

    @classmethod
    @abc.abstractmethod
    @contextlib.contextmanager
    def log_event_context(
        cls,
        component: str,
        event_name: str,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
    ) -> Generator[None, None, None]:
        """Context manager that logs START on entry and SUCCESS/FAILURE on exit.

        On entry, logs a START event. If the wrapped code block completes
        without raising, logs a SUCCESS event. If an exception is raised,
        logs a FAILURE event with the error message and stack trace
        extracted from the exception, then re-raises it.

        Args:
            component: Component name (e.g. ``"torchrec"``).
            event_name: Name identifying the event (e.g. ``"train_step"``).
            metadata: Optional key-value pairs to attach to the event records.
            add_wait_counter: If ``True``, a ``_WaitCounter`` is managed
                for the duration of the context.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def n_batch_log_event(
        cls,
        component: str,
        event_name: str,
        event_type: EventType,
        n: int = 1,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None: ...

    @classmethod
    @abc.abstractmethod
    @contextlib.contextmanager
    def n_batch_log_event_context(
        cls,
        component: str,
        event_name: str,
        n: int = 1,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
    ) -> Generator[None, None, None]: ...
