#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pair training batches with opaque attribution metadata."""

from __future__ import annotations

from typing import Any, cast, ClassVar, Generic, TypeGuard, TypeVar

T = TypeVar("T")


_EXCEPTION_ATTRIBUTION_ATTRIBUTE = "_torchrec_exception_attribution"


class ExceptionAttribution:
    """The operation and metadata associated with an existing exception."""

    def __init__(
        self,
        operation: str,
        metadata: object,
        *,
        batch_index: int | None = None,
    ) -> None:
        self._operation = operation
        self._metadata = metadata
        self._batch_index = batch_index

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def metadata(self) -> object:
        return self._metadata

    @property
    def batch_index(self) -> int | None:
        return self._batch_index


class AttributedBatch(Generic[T]):
    """A training batch paired with opaque attribution metadata at ingress.

    Pipelines unwrap the carrier before processing the batch. TorchRec carries
    the metadata without interpreting its contents.

    This is a plain class because dataclasses with deferred annotations can
    fail when loaded in a separate ``torch.package`` namespace.
    """

    _torchrec_attributed_batch_carrier: ClassVar[bool] = True

    def __init__(self, value: T, metadata: object) -> None:
        self._value = value
        self._metadata = metadata

    @property
    def value(self) -> T:
        """The training batch being processed."""
        return self._value

    @property
    def metadata(self) -> object:
        """Opaque information that stays paired with the batch."""
        return self._metadata


def _is_attributed_batch(value: object) -> TypeGuard[AttributedBatch[Any]]:
    """Recognize a batch carrier loaded from any ``torch.package`` namespace.

    ``torch.package`` loads a private copy of this class, so an imported carrier
    is not an instance of the process-local ``AttributedBatch`` class. The stable
    marker keeps batch handling compatible across that package boundary.
    """
    return getattr(type(value), "_torchrec_attributed_batch_carrier", False) is True


def unwrap_attributed_batch(
    value: T | AttributedBatch[T],
) -> tuple[T, object | None]:
    """Return a raw batch and its metadata, passing raw batches through."""
    if _is_attributed_batch(value):
        return cast(AttributedBatch[T], value).value, value.metadata
    return cast(T, value), None


def get_exception_attribution(
    error: BaseException,
) -> ExceptionAttribution | None:
    """Return attribution attached to an exception or its explicit cause."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return None
        seen.add(current_id)
        try:
            attribution = getattr(
                current,
                _EXCEPTION_ATTRIBUTION_ATTRIBUTE,
                None,
            )
        except Exception:
            # Attribution is diagnostic-only and must not mask the operation failure.
            return None
        if attribution is not None:
            return cast(ExceptionAttribution, attribution)
        current = current.__cause__
    return None


def attach_exception_attribution(
    error: BaseException,
    operation: str,
    metadata: object,
    *,
    batch_index: int | None = None,
) -> ExceptionAttribution:
    """Attach attribution without replacing the exception or an inner attribution."""
    existing = get_exception_attribution(error)
    if existing is not None:
        return existing
    attribution = ExceptionAttribution(
        operation,
        metadata,
        batch_index=batch_index,
    )
    try:
        BaseException.__setattr__(
            error,
            _EXCEPTION_ATTRIBUTION_ATTRIBUTE,
            attribution,
        )
    except Exception:
        # Attribution is diagnostic-only and must not mask the operation failure.
        return attribution
    return attribution
