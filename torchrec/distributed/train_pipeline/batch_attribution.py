#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attribute failures in pipeline operations to the batches they worked on.

A stack wraps each reader item as ``Attributed(value, metadata)``. A pipeline
operation decorated with ``batch_operation`` runs on the raw value. If it
raises, the owner's metadata is attached to the original exception, which then
propagates unchanged: same object, type, message and traceback.
``get_batch_attribution`` reads it back.

Attribution is plain data under a string attribute, and carriers are recognized
by a class marker, not by ``isinstance``. Both therefore work across separate
copies of this module in one process, such as a copy loaded by torch.package.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    cast,
    ClassVar,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

_ATTRIBUTION_ATTR = "_torchrec_batch_attribution"
_CARRIER_MARKER = "_torchrec_batch_carrier"

# The owner of an operation: a parameter name, a positional index, or a
# callable that receives the operation's arguments and returns the batch.
Owner = Union[str, int, Callable[..., object]]


@dataclass(frozen=True, eq=False)
class Attributed(Generic[T]):
    """A batch paired with opaque metadata from its source.

    Not a ``Pipelineable`` on purpose: code that does not know about the carrier
    fails at its first use of the batch instead of carrying it further.
    """

    _torchrec_batch_carrier: ClassVar[bool] = True

    value: T
    metadata: object


def is_attributed(value: object) -> bool:
    return getattr(type(value), _CARRIER_MARKER, False) is True


def unwrap_attributed(value: Any) -> Any:
    """The raw batch inside a carrier, or the value itself."""
    return value.value if is_attributed(value) else value


def carry_attribution(value: Any, source: object) -> Any:
    """Pair a derived value with the source carrier's metadata. Raw source: raw value."""
    if not is_attributed(source) or value is None:
        return value
    return Attributed(unwrap_attributed(value), cast(Any, source).metadata)


def to_attributed(item: Any) -> Any:
    """Convert a reader envelope with ``batch`` and ``batch_metadata``; leave
    every other value unchanged."""
    if is_attributed(item):
        return item
    # @lint-ignore FIXIT AvoidHasattrEverywhere: structural envelope check.
    if hasattr(item, "batch") and hasattr(item, "batch_metadata"):
        return Attributed(item.batch, item.batch_metadata)
    return item


def get_batch_attribution(error: BaseException) -> Optional[Dict[str, object]]:
    """The attribution attached to ``error`` or to an exception in its cause chain.

    Returns ``{"operation": str, "metadata": object}`` or ``None``.
    """
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, _ATTRIBUTION_ATTR, None)
        if isinstance(value, dict):
            return value
        current = current.__cause__ or current.__context__
    return None


def _attach(error: BaseException, operation: str, metadata: object) -> None:
    # Never replace the original exception, even if it rejects attributes.
    try:
        if getattr(error, _ATTRIBUTION_ATTR, None) is None:  # the innermost wins
            setattr(
                error,
                _ATTRIBUTION_ATTR,
                {"operation": operation, "metadata": metadata},
            )
    except Exception:
        pass


def _owner_reader(
    func: Callable[..., Any], owner: Owner
) -> tuple[
    Callable[[tuple[Any, ...], Dict[str, Any]], object], Optional[int], Optional[str]
]:
    """Resolve the owner once, at decoration time. Returns a reader plus the
    position and name of the owner parameter (both None for a callable owner)."""
    if callable(owner):
        resolver = owner
        return (lambda args, kwargs: resolver(*args, **kwargs)), None, None

    params = list(inspect.signature(func).parameters.values())
    if isinstance(owner, int):
        index: Optional[int] = owner
        name: Optional[str] = params[owner].name if owner < len(params) else None
    else:
        names = [p.name for p in params]
        if owner not in names:
            raise TypeError(f"{func.__qualname__} has no parameter named {owner!r}")
        param = params[names.index(owner)]
        name = owner
        positional = param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        index = names.index(owner) if positional else None

    def read(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> object:
        if index is not None and index < len(args):
            return args[index]
        return kwargs.get(name) if name is not None else None

    return read, index, name


def batch_operation(
    operation: str,
    *,
    owner: Owner = "batch",
    propagate: bool = False,
) -> Callable[[F], F]:
    """Attribute failures of a synchronous operation to the batch it works on.

    ``owner`` names the batch: a parameter name, a positional index, or a
    callable that receives the operation's arguments, such as
    ``lambda self: self.batches[0]``. A callable owner is resolved before the
    call, so an operation that removes its batch still names it.

    When the owner is a carrier, an owner parameter is unwrapped before the
    call, and a failure attaches ``{"operation", "metadata"}`` to the original
    exception before re-raising it. With ``propagate=True``, a result that is
    not ``None`` is paired with the owner's metadata, so the decorated function
    must return the transformed batch. ``StopIteration`` passes through
    unchanged. A raw owner skips all of this.
    """

    def decorate(func: F) -> F:
        if (
            inspect.iscoroutinefunction(func)
            or inspect.isgeneratorfunction(func)
            or inspect.isasyncgenfunction(func)
        ):
            raise TypeError("batch_operation only supports synchronous callables")
        read_owner, index, name = _owner_reader(func, owner)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            batch = read_owner(args, kwargs)
            if not is_attributed(batch):
                return func(*args, **kwargs)
            raw = cast(Any, batch).value
            if index is not None and index < len(args):
                args = args[:index] + (raw,) + args[index + 1 :]
            elif name is not None and name in kwargs:
                kwargs = {**kwargs, name: raw}
            try:
                result = func(*args, **kwargs)
            except StopIteration:
                raise
            except Exception as error:
                _attach(error, operation, cast(Any, batch).metadata)
                raise
            return carry_attribution(result, batch) if propagate else result

        return cast(F, wrapped)

    return decorate
