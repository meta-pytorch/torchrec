#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any
from unittest import mock

from torchrec.distributed.logger import _get_input_from_func, _torchrec_method_logger


class TestLogger(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Mock torchrec_logger._get_msg_dict
        self.get_msg_dict_patcher = mock.patch(
            "torchrec.distributed.torchrec_logger._get_msg_dict"
        )
        self.mock_get_msg_dict = self.get_msg_dict_patcher.start()

        # Return a dictionary with func_name that can be modified by _get_input_from_func
        def mock_get_msg_dict_impl(func_name: str, **kwargs: Any) -> dict[str, Any]:
            return {"func_name": func_name}

        self.mock_get_msg_dict.side_effect = mock_get_msg_dict_impl

        # Mock _torchrec_logger
        self.logger_patcher = mock.patch("torchrec.distributed.logger._torchrec_logger")
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self) -> None:
        self.get_msg_dict_patcher.stop()
        self.logger_patcher.stop()
        super().tearDown()

    def test_get_input_from_func_no_args(self) -> None:
        """Test _get_input_from_func with a function that has no arguments."""

        def test_func() -> None:
            pass

        msg_dict = {"func_name": "test_func"}
        result = _get_input_from_func(test_func, msg_dict)
        self.assertEqual(result, "{}")

    def test_get_input_from_func_with_args(self) -> None:
        """Test _get_input_from_func with a function that has positional arguments."""

        def test_func(_a: int, _b: str) -> None:
            pass

        msg_dict = {"func_name": "test_func"}
        result = _get_input_from_func(test_func, msg_dict, 42, "hello")
        self.assertEqual(result, "{'_a': 42, '_b': 'hello'}")

    def test_get_input_from_func_with_kwargs(self) -> None:
        """Test _get_input_from_func with a function that has keyword arguments."""

        def test_func(_a: int = 0, _b: str = "default") -> None:
            pass

        msg_dict = {"func_name": "test_func"}
        result = _get_input_from_func(test_func, msg_dict, _b="world")
        self.assertEqual(result, "{'_a': 0, '_b': 'world'}")

    def test_get_input_from_func_with_args_and_kwargs(self) -> None:
        """Test _get_input_from_func with a function that has both positional and keyword arguments."""

        def test_func(
            _a: int, _b: str = "default", *_args: Any, **_kwargs: Any
        ) -> None:
            pass

        msg_dict = {"func_name": "test_func"}
        result = _get_input_from_func(
            test_func, msg_dict, 42, "hello", "extra", key="value"
        )
        self.assertIn("_a", result)
        self.assertIn("42", result)
        self.assertIn("_b", result)
        self.assertIn("hello", result)
        self.assertIn("_args", result)
        self.assertIn("extra", result)
        self.assertIn("_kwargs", result)
        self.assertIn("key", result)
        self.assertIn("value", result)

    def test_torchrec_method_logger_success(self) -> None:
        """Test _torchrec_method_logger with a successful function execution when logging is enabled."""
        # Create a mock function that returns a value
        mock_func = mock.MagicMock(return_value="result")
        mock_func.__name__ = "mock_func"

        # Apply the decorator
        decorated_func = _torchrec_method_logger()(mock_func)

        # Call the decorated function
        result = decorated_func(42, key="value")

        # Verify the result
        self.assertEqual(result, "result")

        # Verify that _get_msg_dict was called with the correct arguments
        self.mock_get_msg_dict.assert_called_once_with("mock_func", key="value")

        # Verify that the logger was called with the correct message
        self.mock_logger.debug.assert_called_once()
        msg_dict = self.mock_logger.debug.call_args[0][0]
        self.assertEqual(msg_dict["output"], "result")

    def test_torchrec_method_logger_exception(self) -> None:
        """Test _torchrec_method_logger with a function that raises an exception when logging is enabled."""
        # Create a mock function that raises an exception
        mock_func = mock.MagicMock(side_effect=ValueError("test error"))
        mock_func.__name__ = "mock_func"

        # Apply the decorator
        decorated_func = _torchrec_method_logger()(mock_func)

        # Call the decorated function and expect an exception
        with self.assertRaises(ValueError):
            decorated_func(42, key="value")

        # Verify that _get_msg_dict was called with the correct arguments
        self.mock_get_msg_dict.assert_called_once_with("mock_func", key="value")

        # Verify that the logger was called with the correct message
        self.mock_logger.error.assert_called_once()
        msg_dict = self.mock_logger.error.call_args[0][0]
        self.assertEqual(msg_dict["error"], "test error")

    def test_torchrec_method_logger_with_wrapper_kwargs(self) -> None:
        """Test _torchrec_method_logger with wrapper kwargs."""
        # Create a mock function that returns a value
        mock_func = mock.MagicMock(return_value="result")
        mock_func.__name__ = "mock_func"

        # Apply the decorator with wrapper kwargs
        decorated_func = _torchrec_method_logger(custom_kwarg="value")(mock_func)

        # Call the decorated function
        result = decorated_func(42, key="value")

        # Verify the result
        self.assertEqual(result, "result")

        # Verify that _get_msg_dict was called with the correct arguments
        self.mock_get_msg_dict.assert_called_once_with("mock_func", key="value")

        # Verify that the logger was called with the correct message
        self.mock_logger.debug.assert_called_once()
        msg_dict = self.mock_logger.debug.call_args[0][0]
        self.assertEqual(msg_dict["output"], "result")

    def test_torchrec_method_logger_constructor_with_args(self) -> None:
        """Test _torchrec_method_logger with a class constructor that has positional arguments."""

        class TestClass:
            @_torchrec_method_logger()
            def __init__(self, _a: int, _b: str) -> None:
                pass

        # Create an instance which will call __init__
        _ = TestClass(42, "hello")

        # Verify that the logger was called
        self.mock_logger.debug.assert_called_once()
        msg_dict = self.mock_logger.debug.call_args[0][0]
        # Verify that class name was prepended to function name
        self.assertEqual(msg_dict["func_name"], "TestClass.__init__")
        # Verify the input contains the arguments
        self.assertIn("_a", msg_dict["input"])
        self.assertIn("42", msg_dict["input"])
        self.assertIn("_b", msg_dict["input"])
        self.assertIn("hello", msg_dict["input"])

    def test_torchrec_method_logger_constructor_with_args_and_kwargs(self) -> None:
        """Test _torchrec_method_logger with a class constructor that has both positional and keyword arguments."""

        class TestClass:
            @_torchrec_method_logger()
            def __init__(
                self, _a: int, _b: str = "default", *_args: Any, **_kwargs: Any
            ) -> None:
                pass

        # Create an instance which will call __init__
        _ = TestClass(42, "hello", "extra", key="value")

        # Verify that the logger was called
        self.mock_logger.debug.assert_called_once()
        msg_dict = self.mock_logger.debug.call_args[0][0]
        # Verify that class name was prepended to function name
        self.assertEqual(msg_dict["func_name"], "TestClass.__init__")
        # Verify the input contains the arguments
        self.assertIn("_a", msg_dict["input"])
        self.assertIn("42", msg_dict["input"])
        self.assertIn("_b", msg_dict["input"])
        self.assertIn("hello", msg_dict["input"])
        self.assertIn("_args", msg_dict["input"])
        self.assertIn("extra", msg_dict["input"])
        self.assertIn("_kwargs", msg_dict["input"])
        self.assertIn("key", msg_dict["input"])
        self.assertIn("value", msg_dict["input"])


if __name__ == "__main__":
    unittest.main()
