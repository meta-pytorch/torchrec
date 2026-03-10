#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from collections import defaultdict
from typing import Dict, Generator, Optional

from torchrec.distributed.logging_utils import EventLoggingHandlerBase, EventType


__all__: list[str] = []

UnfilteredLogger = "UnfilteredLogger"
SingleRankStaticLogger = "SingleRankStaticLogger"
AllRankStaticLogger = "AllRankStaticLogger"
CappedLogger = "CappedLogger"
Cap1Logger = "Cap1Logger"
Cap01Logger = "Cap01Logger"
MethodLogger = "MethodLogger"


class EventLoggingHandler(EventLoggingHandlerBase):
    """No-op event logging handler for open-source builds.

    This class can be used to add event logging implementation to Torchrec Components
    """

    @classmethod
    def log_event(
        cls,
        component: str,
        event_name: str,
        event_type: EventType,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        pass

    @classmethod
    @contextlib.contextmanager
    def log_event_context(
        cls,
        component: str,
        event_name: str,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
    ) -> Generator[None, None, None]:
        yield

    @classmethod
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
    ) -> None:
        pass

    @classmethod
    @contextlib.contextmanager
    def n_batch_log_event_context(
        cls,
        component: str,
        event_name: str,
        n: int = 1,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
    ) -> Generator[None, None, None]:
        yield


_log_handlers: dict[str, logging.Handler] = defaultdict(logging.NullHandler)
