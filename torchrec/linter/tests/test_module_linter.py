#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import hypothesis.strategies as st
from hypothesis import given
from torchrec.linter import module_linter


def populate_parent_class_list(
    class_src: str,
    uses_LazyModuleExtensionMixin: bool,
) -> str:
    if uses_LazyModuleExtensionMixin:
        parent_class_list = "LazyModuleExtensionMixin, torch.nn.Module"
    else:
        parent_class_list = "torch.nn.Module"
    return class_src.replace("${parent_class_list}", parent_class_list)


class DocStringLinterTest(unittest.TestCase):
    def test_docstring_empty(self) -> None:
        src = ""
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 0)

    def test_docstring_no_modules(self) -> None:
        src = """
class A:
    pass
        """
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 0)

    #  `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(uses_LazyModuleExtensionMixin=st.booleans())
    def test_docstring_no_docstring(self, uses_LazyModuleExtensionMixin: bool) -> None:
        src = """
class F(${parent_class_list}):
    def __init__(self):
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 1)
        self.assertTrue(
            "No docstring found in a TorchRec module" in p.call_args_list[0][0][0]
        )

    #  `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(uses_LazyModuleExtensionMixin=st.booleans())
    def test_docstring_no_module_init(
        self, uses_LazyModuleExtensionMixin: bool
    ) -> None:
        src = """
class F(${parent_class_list}):
    \"""
    \"""
    def forward(self, net, z):
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 3)
        self.assertTrue(
            "No runnable example in a TorchRec module" in p.call_args_list[0][0][0]
        )
        self.assertTrue(
            "Missing required keywords from TorchRec module"
            in p.call_args_list[1][0][0]
        )
        self.assertTrue(
            "Missing docstring for forward function" in p.call_args_list[2][0][0]
        )

    #  `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(uses_LazyModuleExtensionMixin=st.booleans())
    def test_missing_args(self, uses_LazyModuleExtensionMixin: bool) -> None:
        src = """
class F(${parent_class_list}):
    \"""
    \"""
    def __init__(self, x, y='a', *arg, k='k'):
        pass

    def forward(self, net, z):
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 4)
        self.assertTrue(
            "No runnable example in a TorchRec module" in p.call_args_list[0][0][0]
        )
        self.assertTrue(
            "Missing required keywords from TorchRec module"
            in p.call_args_list[1][0][0]
        )
        self.assertTrue("Missing docstring descriptions" in p.call_args_list[2][0][0])
        self.assertTrue("['x']" in p.call_args_list[2][0][0])
        self.assertTrue("['y', 'k']" in p.call_args_list[2][0][0])
        self.assertTrue(
            "Missing docstring for forward function" in p.call_args_list[3][0][0]
        )

    #  `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(uses_LazyModuleExtensionMixin=st.booleans())
    def test_valid_module(self, uses_LazyModuleExtensionMixin: bool) -> None:
        src = """
class F(${parent_class_list}):
    \"""
    Blah.

    Args:
        x: Blah
        y: Blah. Default: "a"

    Example::

        pass
    \"""
    def __init__(self, x, y='a'):
        pass

    def forward(self, z):
        \"""
        Args:
            z: Blah

        Returns:
            None
        \"""
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 0)

    #  `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(uses_LazyModuleExtensionMixin=st.booleans())
    def test_num_ctor_args(self, uses_LazyModuleExtensionMixin: bool) -> None:
        # Case 1: TorchRec module has less than 5 ctor args -> pass
        src = """
class F(${parent_class_list}):
    \"""
    Blah.

    Args:
        x: Blah
        y: Blah. Default: "a"

    Example::

        pass
    \"""
    def __init__(self, x, y='a'):
        pass

    def forward(self, z):
        \"""
        Args:
            z: Blah

        Returns:
            None
        \"""
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 0)

        # Case 2: TorchRec module has more than 5 ctor args -> print error
        src = """
class F(${parent_class_list}):
    \"""
    Blah.

    Args:
        a: Blah
        b: Blah
        c: Blah
        d: Blah
        e: Blah
        f: Blah. Default: "f".

    Example::

        pass
    \"""
    def __init__(self, a, b, c, d, e, f='f'):
        pass

    def forward(self, z):
        \"""
        Args:
            z: Blah

        Returns:
            None
        \"""
        pass
        """
        src = populate_parent_class_list(src, uses_LazyModuleExtensionMixin)
        with patch("builtins.print") as pa, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(pa.call_count, 1)
        self.assertTrue(
            "TorchRec module has too many constructor arguments"
            in pa.call_args_list[0][0][0]
        )

        # Case 3: not a TorchRec module -> pass
        src = """
class F:
    \"""
    Blah.

    Args:
        a: Blah
        b: Blah
        c: Blah
        d: Blah
        e: Blah
        f: Blah. Default: "f".

    Example::

        pass
    \"""
    def __init__(self, x, y='a'):
        pass

    def forward(self, z):
        \"""
        Args:
            z: Blah

        Returns:
            None
        \"""
        pass
        """
        with patch("builtins.print") as p, patch(
            "torchrec.linter.module_linter.read_file", return_value=src
        ):
            module_linter.linter_one_file("a")

        self.assertEqual(p.call_count, 0)


if __name__ == "__main__":
    unittest.main()
