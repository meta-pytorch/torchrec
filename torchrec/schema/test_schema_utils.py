#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Any, Dict, Tuple

from .utils import is_signature_compatible


def stable_test_func(
    a: int, b: float, *, c: int, d: float = 1.0, **kwargs: Dict[str, Any]
) -> int:
    return a


def stable_test_func_basic(
    a: int,
    b: float,
    c: bool = True,
    d: float = 1.0,
) -> bool:
    return c


class TestUtils(unittest.TestCase):
    def test_is_not_backwards_compatible(self) -> None:
        ## stable_test_func tests
        def test_func_positional_arg_removed(
            a: int, *, c: int, d: float = 1.0, **kwargs: Dict[str, Any]
        ) -> int:
            return a

        def test_func_positional_arg_added(
            a: int,
            b: float,
            z: float,
            *,
            c: int,
            d: float = 1.0,
            **kwargs: Dict[str, Any],
        ) -> int:
            return a

        def test_func_keyword_arg_removed(
            a: int, b: float, *, d: float = 1.0, **kwargs: Dict[str, Any]
        ) -> int:
            return a

        def test_func_var_kwargs_removed(
            a: int, b: float, z: float, *, d: float = 1.0
        ) -> int:
            return a

        def test_func_var_args_removed(
            a: int, b: float, z: float, d: float = 1.0, **kwargs: Dict[str, Any]
        ) -> int:
            return a

        # stable_test_func_basic tests
        def test_func_basic_keyword_or_pos_arg_shifted(
            a: int,
            b: float,
            d: float = 1.0,
            c: bool = True,
        ) -> bool:
            return c

        def test_func_basic_add_arg_in_middle(
            a: int,
            b: float,
            d: float = 1.0,
            z: float = 1.0,
            c: bool = True,
        ) -> bool:
            return c

        def test_func_basic_default_arg_changed(
            a: int,
            b: float,
            c: bool = True,
            d: float = 2.0,
        ) -> bool:
            return c

        def test_func_basic_default_arg_removed(
            a: int,
            b: float,
            c: bool,
            d: float = 1.0,
        ) -> bool:
            return c

        def test_func_basic_arg_type_change(
            a: int,
            b: bool,
            c: bool = True,
            d: float = 1.0,
        ) -> bool:
            return c

        def test_func_basic_return_type_changed(
            a: int,
            b: float,
            c: bool = True,
            d: float = 1.0,
        ) -> int:
            return a

        local_funcs = locals()
        for name, func in local_funcs.items():
            if name.startswith("test_func_basic"):
                self.assertFalse(
                    is_signature_compatible(
                        inspect.signature(stable_test_func_basic),
                        inspect.signature(func),
                    ),
                    f"{name} is backwards compatible with stable_test_func_basic when it shouldn't be.",
                )
            elif name.startswith("test_func"):
                self.assertFalse(
                    is_signature_compatible(
                        inspect.signature(stable_test_func), inspect.signature(func)
                    ),
                    f"{name} is not backwards compatible with stable_test_func when it shouldn't be.",
                )
            else:
                continue

    def test_is_backwards_compatible(self) -> None:
        # stable_test_func tests
        def test_func_keyword_arg_added(
            a: int,
            b: float,
            *,
            c: int,
            d: float = 1.0,
            e: float = 1.0,
            **kwargs: Dict[str, Any],
        ) -> int:
            return a

        def test_func_keyword_arg_added_in_middle(
            a: int,
            b: float,
            *,
            c: int,
            e: float = 1.0,
            d: float = 1.0,
            **kwargs: Dict[str, Any],
        ) -> int:
            return a

        def test_func_keyword_arg_shifted(
            a: int, b: float, *, d: float = 1.0, c: int, **kwargs: Dict[str, Any]
        ) -> int:
            return a

        # stable_test_func_basic tests
        def test_func_basic_add_arg_at_end(
            a: int,
            b: float,
            c: bool = True,
            d: float = 1.0,
            e: float = 1.0,
        ) -> bool:
            return c

        def test_func_basic_add_var_args_at_end(
            a: int,
            b: float,
            c: bool = True,
            d: float = 1.0,
            *args: Tuple[Any],
        ) -> bool:
            return c

        def test_func_basic_add_var_kwargs_at_end(
            a: int,
            b: float,
            c: bool = True,
            d: float = 1.0,
            **kwargs: Dict[str, Any],
        ) -> bool:
            return c

        local_funcs = locals()
        for name, func in local_funcs.items():
            if name.startswith("test_func_basic"):
                self.assertTrue(
                    is_signature_compatible(
                        inspect.signature(stable_test_func_basic),
                        inspect.signature(func),
                    ),
                    f"{name} is supposed to be backwards compatible with stable_test_func_basic",
                )
            elif name.startswith("test_func"):
                self.assertTrue(
                    is_signature_compatible(
                        inspect.signature(stable_test_func), inspect.signature(func)
                    ),
                    f"{name} is supposed to be backwards compatible with stable_test_func",
                )
            else:
                continue
