#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from typing import Any, Dict, Optional

import torch
from torchrec.distributed.train_pipeline.pipeline_dataclass import (
    is_pipeline_dataclass,
    pipeline_dataclass,
)


class _PlainBase:
    x: int


class _AddsOne:
    """Data descriptor that transforms on write; reads as 2 on the class itself."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr = f"_{name}"

    def __get__(self, obj: object, objtype: Optional[type] = None) -> int:
        return 2 if obj is None else getattr(obj, self.attr)

    def __set__(self, obj: object, value: int) -> None:
        object.__setattr__(obj, self.attr, value + 1)


# Builds each case with `type()`, so a case is just a class namespace.
def _define(namespace: Dict[str, Any], bases: tuple[type, ...] = ()) -> type:
    return pipeline_dataclass(type("Inputs", bases, namespace))


class PipelineDataclassTest(unittest.TestCase):
    def test_is_frozen_dataclass(self) -> None:
        @pipeline_dataclass
        class Inputs:
            x: torch.Tensor
            features: Dict[str, torch.Tensor]
            name: Optional[str] = None

        inputs = Inputs(x=torch.ones(1), features={})
        self.assertTrue(dataclasses.is_dataclass(Inputs))
        self.assertTrue(is_pipeline_dataclass(Inputs))

        with self.assertRaises(dataclasses.FrozenInstanceError):
            # pyrefly: ignore[read-only]
            inputs.x = torch.zeros(1)
        self.assertIs(dataclasses.replace(inputs, name="a").x, inputs.x)

        # eq=False: compared by identity, never elementwise on tensors.
        self.assertNotEqual(inputs, Inputs(x=inputs.x, features={}))

    def test_plain_dataclass_is_not_pipeline_dataclass(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class Plain:
            x: int

        self.assertFalse(is_pipeline_dataclass(Plain))
        self.assertFalse(is_pipeline_dataclass(Plain(x=1)))

    def test_subclass_must_be_decorated(self) -> None:
        @pipeline_dataclass
        class Base:
            x: int

        class Undecorated(Base):
            pass

        @pipeline_dataclass
        class Decorated(Base):
            y: int = 0

        self.assertFalse(is_pipeline_dataclass(Undecorated))
        self.assertTrue(is_pipeline_dataclass(Decorated))
        self.assertEqual(Decorated(x=1, y=2).y, 2)

    def test_kw_only_and_tuple_defaults_are_allowed(self) -> None:
        @pipeline_dataclass
        class Inputs:
            x: int = dataclasses.field(kw_only=True)
            dims: tuple[int, ...] = (1, 2)

        self.assertEqual(Inputs(x=1).dims, (1, 2))

    def test_rejects_unsafe_definitions(self) -> None:
        int_field = {"__annotations__": {"x": int}}
        cases = {
            "__post_init__": {**int_field, "__post_init__": lambda self: None},
            "__init__": {**int_field, "__init__": lambda self, x: None},
            "__new__": {**int_field, "__new__": lambda cls, x: object.__new__(cls)},
            "init=False": {
                **int_field,
                "x": dataclasses.field(init=False, default=0),
            },
            "InitVar": {"__annotations__": {"x": dataclasses.InitVar[int]}},
            "default_factory": {
                "__annotations__": {"x": list},
                "x": dataclasses.field(default_factory=list),
            },
            "immutable constant, got Tensor": {
                "__annotations__": {"x": torch.Tensor},
                "x": torch.zeros(1),
            },
            # Passes a check on `field.default`, which has gone through `__get__`.
            "immutable constant, got _AddsOne": {
                "__annotations__": {"x": int},
                "x": _AddsOne(),
            },
            "__getattribute__": {
                **int_field,
                "__getattribute__": lambda self, name: object.__getattribute__(
                    self, name
                ),
            },
        }
        for message, namespace in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(TypeError, message):
                    _define(namespace)

    def test_rejects_plain_base(self) -> None:
        with self.assertRaisesRegex(TypeError, "not a pipeline dataclass"):
            _define({"__annotations__": {"y": int}}, bases=(_PlainBase,))

    def test_rejects_stacking_on_dataclass(self) -> None:
        with self.assertRaisesRegex(TypeError, "instead of @dataclass"):
            pipeline_dataclass(dataclasses.dataclass(type("Inputs", (), {})))
