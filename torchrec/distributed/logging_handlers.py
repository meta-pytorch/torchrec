#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from torchrec.distributed.logging_utils import (
    EventLoggingHandlerBase,
    EventScope,
    EventType,
    OptimizationTechnique,
    StackLayer,
)


__all__: list[str] = []

UnfilteredLogger = "UnfilteredLogger"
SingleRankStaticLogger = "SingleRankStaticLogger"
AllRankStaticLogger = "AllRankStaticLogger"
CappedLogger = "CappedLogger"
Cap1Logger = "Cap1Logger"
Cap01Logger = "Cap01Logger"
MethodLogger = "MethodLogger"

F = TypeVar("F", bound=Callable[..., Any])


class TorchrecComponent(Enum):
    """Enum representing different TorchRec components for event logging."""

    PLANNER = "planner"
    SHARDER = "sharder"
    TRAIN_PIPELINE = "train_pipeline"
    INPUT_DIST = "input_dist"
    OUTPUT_DIST = "output_dist"
    LOOKUP = "lookup"
    REC_METRICS = "rec_metrics"


class EventLoggingHandler(EventLoggingHandlerBase):
    """No-op event logging handler for open-source builds.

    This class can be used to add event logging implementation to Torchrec Components
    """

    @classmethod
    def event_logger(
        cls,
        component: TorchrecComponent,
        prefix: str = "",
        n: Optional[int] = None,
        add_wait_counter: bool = False,
    ) -> Callable[[F], F]:
        """
        Decorator that wraps a method with EventLoggingHandler.log_event_context
        or n_batch_log_event_context.

        The event name is constructed as "{prefix}{func.__qualname__}".

        Args:
            component: TorchrecComponent enum value for logging
            prefix: Optional prefix to prepend to the qualname (default: "")
            n: If provided, use n_batch_log_event_context to log only every
                n batches. If None (default), use log_event_context.
            add_wait_counter: If True, a _WaitCounter is managed for the
                duration of the decorated function. Defaults to False.

        Example::

            class MyPlanner:
                @EventLoggingHandler.event_logger(TorchrecComponent.PLANNER)
                def plan(self, module, sharders):
                    # This will log event_name="MyPlanner.plan"
                    ...

                @EventLoggingHandler.event_logger(TorchrecComponent.PLANNER, prefix="v2_")
                def collective_plan(self, module, sharders):
                    # This will log event_name="v2_MyPlanner.collective_plan"
                    ...

                @EventLoggingHandler.event_logger(TorchrecComponent.PLANNER, n=1000)
                def frequent_op(self, data):
                    # This will log only every 1000 batches
                    ...
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                event_name = f"{prefix}{func.__qualname__}"
                if n is None:
                    ctx = cls.log_event_context(
                        component=component.value,
                        event_name=event_name,
                        add_wait_counter=add_wait_counter,
                    )
                else:
                    ctx = cls.n_batch_log_event_context(
                        component=component.value,
                        event_name=event_name,
                        n=n,
                        add_wait_counter=add_wait_counter,
                    )
                with ctx:
                    return func(*args, **kwargs)

            return wrapper  # pyre-ignore[7]

        return decorator

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


class TrainingOptimizationLogger(EventLoggingHandler):
    """No-op training optimization logger for open-source builds."""

    @classmethod
    def log(
        cls,
        layer: StackLayer,
        event_name: str,
        event_type: EventType,
        technique: OptimizationTechnique,
        component: TorchrecComponent,
        event_scope: EventScope,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        pass

    @classmethod
    @contextlib.contextmanager
    def log_context(
        cls,
        layer: StackLayer,
        event_name: str,
        technique: OptimizationTechnique,
        component: TorchrecComponent,
        event_scope: EventScope,
        metadata: Optional[Dict[str, str]] = None,
        add_wait_counter: bool = False,
    ) -> Generator[None, None, None]:
        yield


def log_planning_result(
    planner_type: str,
    error_message: Optional[str] = None,
    **extra_metadata: str,
) -> None:
    """No-op OSS stub."""
    pass


def log_offloading_summary(best_plan: List, planner_type: str) -> None:  # type: ignore[type-arg]
    """No-op OSS stub."""
    pass


def log_storage_reservation(
    reservation_type: str,
    percentage: Optional[float] = None,
    dense_hbm_bytes: Optional[int] = None,
    kjt_hbm_bytes: Optional[int] = None,
    original_hbm_per_rank: int = 0,
    available_hbm_per_rank: int = 0,
    planner_type: str = "",
) -> None:
    """No-op OSS stub."""
    pass


def log_planner_config(metadata: Optional[Dict[str, str]] = None) -> None:
    """No-op OSS stub."""
    pass


def log_stats_match(
    table_name: str = "",
    table_height: int = 0,
    match_level: str = "",
    min_working_set: int = 0,
    recommended_cache_rows: int = 0,
    global_batch_size: int = 0,
    matched_height: Optional[int] = None,
) -> None:
    """No-op OSS stub."""
    pass


def log_clf_computed(
    table_name: str = "",
    table_height: int = 0,
    clf: float = 0.0,
) -> None:
    """No-op OSS stub."""
    pass


def log_cacheability_resolved(
    table_name: str = "",
    table_height: int = 0,
    cacheability: float = 0.0,
    expected_lookups: int = 0,
) -> None:
    """No-op OSS stub."""
    pass


def log_kernel_changed(
    table_name: str = "",
    action: str = "",
    reason: str = "",
    new_kernels: Optional[list] = None,  # type: ignore[type-arg]
    table_height: Optional[int] = None,
    cache_ratio: Optional[float] = None,
) -> None:
    """No-op OSS stub."""
    pass


def log_table_assignment(best_plan: List, planner_type: str = "") -> None:  # type: ignore[type-arg]
    """No-op OSS stub."""
    pass


def log_table_constraints(constraints: Optional[Dict] = None, planner_type: str = "") -> None:  # type: ignore[type-arg]
    """No-op OSS stub."""
    pass


def log_tbe_composition(grouped_configs: List, rank: int = 0) -> None:  # type: ignore[type-arg]
    """No-op OSS stub."""
    pass


_log_handlers: dict[str, logging.Handler] = defaultdict(logging.NullHandler)
