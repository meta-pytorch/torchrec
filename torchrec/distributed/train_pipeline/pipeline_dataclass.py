#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import enum
import inspect
from typing import TypeVar

from typing_extensions import dataclass_transform

T = TypeVar("T")

_MARKER = "__torchrec_pipeline_dataclass__"

# Hooks that can change what construction stores or what a field read returns. The
# pipeline replays the constructor and fx reads every field, so either would apply a
# transformation twice. `frozen=True` already rejects `__setattr__`/`__delattr__`.
_FORBIDDEN_MEMBERS = ("__init__", "__new__", "__post_init__", "__getattribute__")

_IMMUTABLE_DEFAULT_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    enum.Enum,
)


def is_pipeline_dataclass(cls: object) -> bool:
    """
    Whether `cls` itself was defined with `@pipeline_dataclass`. Subclasses do not
    inherit this: each must be decorated, and so validated, again.
    """
    return isinstance(cls, type) and getattr(cls, _MARKER, None) is cls


@dataclass_transform(
    frozen_default=True, eq_default=False, field_specifiers=(dataclasses.field,)
)
def pipeline_dataclass(cls: type[T]) -> type[T]:
    """
    Defines a frozen dataclass that the TorchRec train pipeline can rebuild from the
    batch, so a module taking it as input can still be pipelined.

    During sparse data dist the pipeline replays the constructor on the next batch and
    hands the rebuilt instance to the pipelined module; the model's own instance is not
    used. That is only correct if construction just assigns fields and the instance
    cannot change afterwards, so the class must:

    * declare only plain fields: no ``__init__``, ``__new__``, ``__post_init__``,
      ``__getattribute__``, or descriptors as field defaults;
    * have every field set by the constructor: no ``init=False`` fields or ``InitVar``;
    * use only immutable constants as defaults: no ``default_factory``;
    * use no custom metaclass, and inherit only from other pipeline dataclasses.

    Violations raise ``TypeError`` when the class is defined. Instances are frozen and
    compare by identity (``eq=False``).

    Immutability is shallow: fields cannot be reassigned, but tensors and containers
    held in them can still be modified in place. Whether a module mutates its inputs
    is the pipeline's postproc contract, not this type's.

    Example::

        @pipeline_dataclass
        class SparseArchIn:
            id_list_features: KeyedJaggedTensor
            timestamps: torch.Tensor
            max_len: int = 128
    """
    _check_class(cls)
    dc = dataclasses.dataclass(frozen=True, eq=False)(cls)
    _check_fields(dc)

    # Checked by identity in `is_pipeline_dataclass`.
    setattr(dc, _MARKER, dc)
    return dc


def _check_class(cls: type) -> None:
    name = cls.__qualname__

    if "__dataclass_fields__" in cls.__dict__:
        raise TypeError(f"{name}: use @pipeline_dataclass instead of @dataclass")

    for member in _FORBIDDEN_MEMBERS:
        if member in cls.__dict__:
            raise TypeError(
                f"{name}: a pipeline dataclass may not define `{member}`, because the "
                "train pipeline replays its constructor and reads every field"
            )

    if type(cls) is not type:
        raise TypeError(f"{name}: custom metaclasses are not supported")

    # Validated bases cannot carry forbidden members or unsafe fields.
    for base in cls.__bases__:
        if base is not object and not is_pipeline_dataclass(base):
            raise TypeError(
                f"{name}: base class {base.__qualname__} is not a pipeline dataclass"
            )


def _check_fields(dc: type) -> None:
    name = dc.__qualname__
    fields = dataclasses.fields(dc)

    for f in fields:
        if not f.init:
            raise TypeError(f"{name}: field `{f.name}` must not use init=False")

        if f.default_factory is not dataclasses.MISSING:
            raise TypeError(
                f"{name}: field `{f.name}` must not use default_factory; "
                "use an immutable default such as None or a tuple"
            )

        # Read the raw class attribute: `f.default` has already gone through a
        # descriptor's `__get__`, which would hide a descriptor in the field's slot.
        default = dc.__dict__.get(f.name, dataclasses.MISSING)
        if default is not dataclasses.MISSING and not _is_immutable_constant(default):
            raise TypeError(
                f"{name}: field `{f.name}` default must be an immutable constant, "
                f"got {type(default).__name__}"
            )

    # `InitVar`s are constructor parameters that `fields()` does not list.
    extra = set(inspect.signature(dc).parameters) - {f.name for f in fields}
    if extra:
        raise TypeError(
            f"{name}: constructor parameters {sorted(extra)} are not fields "
            "(InitVar is not supported)"
        )


def _is_immutable_constant(value: object) -> bool:
    if isinstance(value, tuple):
        return all(_is_immutable_constant(v) for v in value)
    return isinstance(value, _IMMUTABLE_DEFAULT_TYPES)
