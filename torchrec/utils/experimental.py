#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, ParamSpec, Type, TypeVar, Union

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def experimental(
    obj: Union[Callable[P, R], Type[T]],
    feature: str | None = None,
    since: str | None = None,
) -> Union[Callable[P, R], Type[T]]:
    """
    Decorator that flags a function or class as *experimental*.

    The decorator emits a :class:`UserWarning` the first time the wrapped
    callable (or the constructor of a wrapped class) is invoked.  This is
    useful for APIs that may change or be removed without notice.

    Args:
        obj: The function, method, or class to be wrapped.
        feature: Optional explicit name for the experimental feature.
            Defaults to the name of the wrapped object.
        since: Optional version or date string (e.g. ``"1.2.0"`` or
            ``"2025-05-21"``) indicating when the feature became experimental.

    Returns:
        The same callable or class, wrapped so that it issues a warning once.

    Warning:
        The decorated API is **not stable**. Downstream code should not rely on
        its long-term availability or on the permanence of its current
        behavior.

    Example:
        >>> @experimental
        ... def fancy_new_op(x):
        ...     return x * 2
        >>> fancy_new_op(3)  # first call triggers a warning
        6

        >>> @experimental(feature="Hybrid 2D Parallel", since="1.2.0")
        ... class HybridDistributedModelParallel:
        ...     ...
    """
    tag: str = feature or obj.__name__
    message_parts: list[str] = [
        f"`{tag}` is *experimental* and may change or be removed without notice."
    ]
    if since:
        message_parts.insert(0, f"[since {since}]")
    warning_message: str = " ".join(message_parts)

    @functools.lru_cache(maxsize=1)
    def _issue_warning() -> None:
        warnings.warn(warning_message, UserWarning, stacklevel=3)

    if isinstance(obj, type):
        orig_init: Callable[..., None] = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args: Any, **kwargs: Any) -> Any:
            _issue_warning()
            return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        return obj
    else:

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _issue_warning()
            return obj(*args, **kwargs)

        # pyrefly: ignore[bad-return]
        return wrapper
