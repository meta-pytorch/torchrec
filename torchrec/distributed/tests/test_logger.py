#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest
from typing import Any
from unittest import mock

import torch.distributed as dist
from torchrec.distributed.logger import (
    _get_input_from_func,
    _get_logging_handler,
    _get_msg_dict,
    _get_or_create_logger,
    _torchrec_method_logger,
    ARG_SIZE_LIMIT,
)
from torchrec.distributed.logging_handlers import _log_handlers, SingleRankStaticLogger


class TestMethodLogger(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Mock logger._get_msg_dict
        self.get_msg_dict_patcher = mock.patch(
            "torchrec.distributed.logger._get_msg_dict"
        )
        self.mock_get_msg_dict = self.get_msg_dict_patcher.start()

        # Return a dictionary with func_name that can be modified by _get_input_from_func
        def mock_get_msg_dict_impl(func_name: str, **kwargs: Any) -> dict[str, Any]:
            return {"func_name": func_name}

        self.mock_get_msg_dict.side_effect = mock_get_msg_dict_impl

        # Mock method_logger
        self.logger_patcher = mock.patch("torchrec.distributed.logger.method_logger")
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

    def test_get_input_from_func_truncates_large_args(self) -> None:
        """Test _get_input_from_func truncates arguments that exceed ARG_SIZE_LIMIT."""

        def test_func(_a: str) -> None:
            pass

        # Create a string larger than ARG_SIZE_LIMIT
        large_string = "x" * (ARG_SIZE_LIMIT + 100)
        msg_dict = {"func_name": "test_func"}

        result = _get_input_from_func(test_func, msg_dict, large_string)

        # Verify that the large argument was truncated with the appropriate message
        self.assertIn("Argument removed due to size limit", result)
        self.assertIn(f"Original size: {len(large_string)}", result)
        # Verify that the actual large string is NOT in the result
        self.assertNotIn(large_string, result)

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
        self.mock_logger.info.assert_called_once()
        msg_dict = self.mock_logger.info.call_args[0][0]
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
        self.mock_logger.info.assert_called_once()
        msg_dict = self.mock_logger.info.call_args[0][0]
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
        self.mock_logger.info.assert_called_once()
        msg_dict = self.mock_logger.info.call_args[0][0]
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
        self.mock_logger.info.assert_called_once()
        msg_dict = self.mock_logger.info.call_args[0][0]
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


class TestLoggerUtils(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Save the original _log_handlers to restore it after tests
        self.original_log_handlers = _log_handlers.copy()

        # Create a mock logging handler
        self.mock_handler = mock.MagicMock(spec=logging.Handler)
        _log_handlers[SingleRankStaticLogger] = self.mock_handler

    def tearDown(self) -> None:
        # Restore the original _log_handlers
        _log_handlers.clear()
        _log_handlers.update(self.original_log_handlers)
        super().tearDown()

    def test_get_logging_handler(self) -> None:
        """Test _get_logging_handler function."""
        # Test with SingleRankStaticLogger destination
        handler = _get_logging_handler(SingleRankStaticLogger)
        self.assertEqual(handler, self.mock_handler)

        # Test with custom destination
        custom_dest = "custom_dest"
        custom_handler = mock.MagicMock(spec=logging.Handler)
        _log_handlers[custom_dest] = custom_handler

        handler = _get_logging_handler(custom_dest)
        self.assertEqual(handler, custom_handler)

    @mock.patch("logging.getLogger")
    def test_get_or_create_logger(self, mock_get_logger: mock.MagicMock) -> None:
        """Test _get_or_create_logger function."""
        mock_logger = mock.MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger

        # Test with SingleRankStaticLogger destination
        logger = _get_or_create_logger(SingleRankStaticLogger)

        # Verify logger was created with the correct name
        logger_name = f"{SingleRankStaticLogger}-{self.mock_handler.__class__.__name__}"
        mock_get_logger.assert_called_once_with(logger_name)

        # Verify logger was configured correctly
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
        mock_logger.addHandler.assert_called_once_with(self.mock_handler)
        self.assertFalse(mock_logger.propagate)

        # Verify formatter was set on the handler
        self.mock_handler.setFormatter.assert_called_once()
        formatter = self.mock_handler.setFormatter.call_args[0][0]
        self.assertIsInstance(formatter, logging.Formatter)

        # Verify logger is returned
        self.assertEqual(logger, mock_logger)

    def test_get_msg_dict_without_dist(self) -> None:
        """Test _get_msg_dict function without dist initialized."""
        # Mock dist.is_initialized to return False
        with mock.patch.object(dist, "is_initialized", return_value=False):
            msg_dict = _get_msg_dict("test_func", kwarg1="val1")

            # Verify msg_dict contains only func_name
            self.assertEqual(len(msg_dict), 1)
            self.assertEqual(msg_dict["func_name"], "test_func")

    def test_get_msg_dict_with_dist(self) -> None:
        """Test _get_msg_dict function with dist initialized."""
        # Mock dist functions
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "get_world_size", return_value=4):
                with mock.patch.object(dist, "get_rank", return_value=2):
                    # Test with group in kwargs
                    mock_group = mock.MagicMock()
                    msg_dict = _get_msg_dict("test_func", group=mock_group)

                    # Verify msg_dict contains all expected keys
                    self.assertEqual(len(msg_dict), 4)
                    self.assertEqual(msg_dict["func_name"], "test_func")
                    self.assertEqual(msg_dict["group"], str(mock_group))
                    self.assertEqual(msg_dict["world_size"], "4")
                    self.assertEqual(msg_dict["rank"], "2")

    def test_get_msg_dict_with_process_group(self) -> None:
        """Test _get_msg_dict function with process_group in kwargs."""
        # Mock dist functions
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "get_world_size", return_value=8):
                with mock.patch.object(dist, "get_rank", return_value=3):
                    # Test with process_group in kwargs
                    mock_process_group = mock.MagicMock()
                    msg_dict = _get_msg_dict(
                        "test_func", process_group=mock_process_group
                    )

                    # Verify msg_dict contains all expected keys
                    self.assertEqual(msg_dict["func_name"], "test_func")
                    self.assertEqual(msg_dict["group"], str(mock_process_group))
                    self.assertEqual(msg_dict["world_size"], "8")
                    self.assertEqual(msg_dict["rank"], "3")
