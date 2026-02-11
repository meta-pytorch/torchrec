#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import unittest
import warnings
from typing import Any

from torchrec.utils.experimental import experimental


class ExperimentalDecoratorTest(unittest.TestCase):
    """Tests for the experimental decorator."""

    def test_experimental_function_returns_correct_value(self) -> None:
        """Test that a decorated function returns the correct value."""

        @experimental
        def multiply(x: int, y: int) -> int:
            return x * y

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = multiply(3, 4)

        self.assertEqual(result, 12)

    def test_experimental_function_emits_warning_on_first_call(self) -> None:
        """Test that a decorated function emits a UserWarning on first call."""

        @experimental
        def add(x: int, y: int) -> int:
            return x + y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            add(1, 2)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("add", str(w[0].message))
            self.assertIn("experimental", str(w[0].message))

    def test_experimental_function_emits_warning_only_once(self) -> None:
        """Test that a decorated function emits warning only on the first call."""

        @experimental
        def subtract(x: int, y: int) -> int:
            return x - y

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            subtract(5, 3)
            subtract(10, 4)
            subtract(100, 50)

            self.assertEqual(len(w), 1)

    def test_experimental_function_with_feature_name(self) -> None:
        """Test that custom feature name appears in the warning message."""

        def dummy_func() -> None:
            pass

        wrapped = experimental(dummy_func, feature="Custom Feature Name")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped()

            self.assertEqual(len(w), 1)
            self.assertIn("Custom Feature Name", str(w[0].message))

    def test_experimental_function_with_since_version(self) -> None:
        """Test that since version appears in the warning message."""

        def versioned_func() -> None:
            pass

        wrapped = experimental(versioned_func, since="1.2.0")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped()

            self.assertEqual(len(w), 1)
            self.assertIn("[since 1.2.0]", str(w[0].message))

    def test_experimental_function_with_feature_and_since(self) -> None:
        """Test that both feature name and since version appear in warning."""

        def full_feature_func() -> None:
            pass

        wrapped = experimental(
            full_feature_func, feature="Hybrid 2D Parallel", since="1.2.0"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped()

            self.assertEqual(len(w), 1)
            message = str(w[0].message)
            self.assertIn("Hybrid 2D Parallel", message)
            self.assertIn("[since 1.2.0]", message)

    def test_experimental_class_returns_instance(self) -> None:
        """Test that a decorated class can be instantiated correctly."""

        @experimental
        class ExperimentalClass:
            def __init__(self, value: int) -> None:
                self.value = value

            def get_value(self) -> int:
                return self.value

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = ExperimentalClass(42)

        self.assertIsInstance(obj, ExperimentalClass)
        self.assertEqual(obj.get_value(), 42)

    def test_experimental_class_emits_warning_on_instantiation(self) -> None:
        """Test that a decorated class emits warning when instantiated."""

        @experimental
        class WarningClass:
            def __init__(self) -> None:
                pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WarningClass()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("WarningClass", str(w[0].message))

    def test_experimental_class_emits_warning_only_once(self) -> None:
        """Test that a decorated class emits warning only on first instantiation."""

        @experimental
        class SingleWarningClass:
            __slots__ = ("value",)

            def __init__(self, value: int = 0) -> None:
                self.value = value

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SingleWarningClass(1)
            SingleWarningClass(2)
            SingleWarningClass(3)

            self.assertEqual(len(w), 1)

    def test_experimental_class_with_feature_name(self) -> None:
        """Test that custom feature name appears in class warning."""

        class FeatureClass:
            pass

        WrappedClass = experimental(FeatureClass, feature="Feature Class")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WrappedClass()

            self.assertEqual(len(w), 1)
            self.assertIn("Feature Class", str(w[0].message))

    def test_experimental_preserves_function_name(self) -> None:
        """Test that the decorated function preserves its __name__ attribute."""

        @experimental
        def named_function() -> None:
            pass

        self.assertEqual(named_function.__name__, "named_function")

    def test_experimental_preserves_function_docstring(self) -> None:
        """Test that the decorated function preserves its docstring."""

        @experimental
        def documented_function() -> None:
            """This is a docstring."""
            pass

        self.assertEqual(documented_function.__doc__, "This is a docstring.")

    def test_experimental_function_with_kwargs(self) -> None:
        """Test that decorated function handles keyword arguments correctly."""

        @experimental
        def func_with_kwargs(a: int, b: int = 10, **kwargs: Any) -> int:
            return a + b + kwargs.get("c", 0)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = func_with_kwargs(1, b=2, c=3)

        self.assertEqual(result, 6)

    def test_experimental_function_with_args(self) -> None:
        """Test that decorated function handles *args correctly."""

        @experimental
        def sum_all(*args: int) -> int:
            return sum(args)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = sum_all(1, 2, 3, 4, 5)

        self.assertEqual(result, 15)

    def test_experimental_class_with_inheritance(self) -> None:
        """Test that decorated class can be subclassed."""

        @experimental
        class BaseExperimental:
            __slots__ = ("x",)

            def __init__(self, x: int) -> None:
                self.x = x

        class DerivedClass(BaseExperimental):
            __slots__ = ("y",)

            def __init__(self, x: int, y: int) -> None:
                super().__init__(x)
                self.y = y

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = DerivedClass(10, 20)

        self.assertEqual(obj.x, 10)
        self.assertEqual(obj.y, 20)

    def test_warning_message_format(self) -> None:
        """Test that warning message has the expected format."""

        def formatted_func() -> None:
            pass

        wrapped = experimental(formatted_func, feature="Test Feature", since="2.0.0")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wrapped()

            self.assertEqual(len(w), 1)
            message = str(w[0].message)
            self.assertIn("[since 2.0.0]", message)
            self.assertIn("`Test Feature`", message)
            self.assertIn("*experimental*", message)
            self.assertIn("may change or be removed without notice", message)

    def test_experimental_uses_functools_partial_correctly(self) -> None:
        """Test using functools.partial for decorator-style with arguments."""

        def my_func() -> str:
            return "result"

        custom_experimental = functools.partial(
            experimental, feature="Partial Feature", since="3.0.0"
        )
        wrapped = custom_experimental(my_func)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped()

            self.assertEqual(result, "result")
            self.assertEqual(len(w), 1)
            self.assertIn("Partial Feature", str(w[0].message))
            self.assertIn("[since 3.0.0]", str(w[0].message))


if __name__ == "__main__":
    unittest.main()
