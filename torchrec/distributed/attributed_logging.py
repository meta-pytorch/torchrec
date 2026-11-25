#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attribute operation failures to an explicit batch owner."""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, cast, TypeVar

from torchrec.distributed.batch_attribution import attach_exception_attribution

F = TypeVar("F", bound=Callable[..., Any])
logger: logging.Logger = logging.getLogger(__name__)


def _make_owner_resolution_failure_logger(
    operation: str,
) -> Callable[[Exception], None]:
    warning_logged = False

    def warn(error: Exception) -> None:
        nonlocal warning_logged
        if warning_logged:
            return
        warning_logged = True
        try:
            logger.warning(
                "Failed to resolve batch attribution owner for %s; "
                "continuing without attribution",
                operation,
                exc_info=(type(error), error, error.__traceback__),
            )
        except Exception:
            pass

    return warn


def _find_owner_position(func: F, owner_kwarg: str | None) -> int | None:
    if owner_kwarg is None:
        return None
    parameters = inspect.signature(func).parameters
    if owner_kwarg not in parameters:
        raise TypeError(
            f"owner_kwarg {owner_kwarg!r} is not a parameter of the decorated function"
        )
    parameter = parameters[owner_kwarg]
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        return None
    if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
        raise TypeError(
            f"owner_kwarg {owner_kwarg!r} must name a positional-or-keyword "
            "or keyword-only parameter"
        )
    return list(parameters).index(owner_kwarg)


def _get_batch_attribution(
    owner: Callable[[Any], object | None] | None,
    owner_kwarg: str | None,
    owner_position: int | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    on_resolution_error: Callable[[Exception], None],
) -> tuple[object | None, int | None]:
    try:
        if owner_kwarg is not None:
            # An explicitly supplied keyword, including ``None``, is
            # authoritative. Fall back to its positional slot only when the
            # keyword is absent.
            context = kwargs.get(owner_kwarg)
            if (
                owner_kwarg not in kwargs
                and owner_position is not None
                and owner_position < len(args)
            ):
                context = args[owner_position]
        else:
            context = owner(args[0]) if owner is not None and args else None
        if context is None:
            return None, None
        return (
            getattr(context, "batch_metadata", None),
            getattr(context, "index", None),
        )
    except Exception as error:
        on_resolution_error(error)
        return None, None


def attributed_logger(
    operation: str,
    *,
    owner: Callable[[Any], object | None] | None = None,
    owner_kwarg: str | None = None,
) -> Callable[[F], F]:
    """Attribute synchronous failures to an explicit batch owner.

    Exactly one source is required: ``owner`` resolves from the first argument,
    while ``owner_kwarg`` names an owner argument. Owners expose
    ``batch_metadata`` and optionally ``index``; ``None`` metadata disables
    attribution. The wrapped call otherwise remains unchanged.

    Decorate asynchronous launches and waits separately. Failures surfacing
    after return, including deferred CUDA errors, are best-effort attributed at
    the later synchronization point. Owner-resolution failures warn once and do
    not affect the call.
    """

    if (owner is None) == (owner_kwarg is None):
        raise TypeError("exactly one attribution owner source is required")

    def decorate(func: F) -> F:
        if (
            inspect.iscoroutinefunction(func)
            or inspect.isgeneratorfunction(func)
            or inspect.isasyncgenfunction(func)
        ):
            raise TypeError("attributed_logger only supports synchronous callables")

        owner_position = _find_owner_position(func, owner_kwarg)
        # Owner-resolution diagnostics are emitted once per decorated
        # operation, independently of failures raised by the operation itself.
        warn_owner_resolution_failure = _make_owner_resolution_failure_logger(operation)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            metadata, batch_index = _get_batch_attribution(
                owner,
                owner_kwarg,
                owner_position,
                args,
                kwargs,
                warn_owner_resolution_failure,
            )
            try:
                return func(*args, **kwargs)
            except StopIteration:
                raise
            except Exception as error:
                if metadata is not None:
                    attach_exception_attribution(
                        error,
                        operation,
                        metadata,
                        batch_index=batch_index,
                    )
                raise

        return cast(F, wrapped)

    return decorate
