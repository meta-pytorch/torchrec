#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest
from unittest import mock

import torch.distributed as dist
from torchrec.distributed.logging_handlers import _log_handlers
from torchrec.distributed.torchrec_logger import (
    _DEFAULT_DESTINATION,
    _get_logging_handler,
    _get_msg_dict,
    _get_or_create_logger,
)


class TestTorchrecLogger(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Save the original _log_handlers to restore it after tests
        self.original_log_handlers = _log_handlers.copy()

        # Create a mock logging handler
        self.mock_handler = mock.MagicMock(spec=logging.Handler)
        _log_handlers[_DEFAULT_DESTINATION] = self.mock_handler

        # Mock print function
        self.print_patcher = mock.patch("builtins.print")
        self.mock_print = self.print_patcher.start()

    def tearDown(self) -> None:
        # Restore the original _log_handlers
        _log_handlers.clear()
        _log_handlers.update(self.original_log_handlers)

        # Stop print patcher
        self.print_patcher.stop()

        super().tearDown()

    def test_get_logging_handler(self) -> None:
        """Test _get_logging_handler function."""
        # Test with default destination
        handler, name = _get_logging_handler()

        self.assertEqual(handler, self.mock_handler)
        self.assertEqual(
            name, f"{type(self.mock_handler).__name__}-{_DEFAULT_DESTINATION}"
        )

        # Test with custom destination
        custom_dest = "custom_dest"
        custom_handler = mock.MagicMock(spec=logging.Handler)
        _log_handlers[custom_dest] = custom_handler

        handler, name = _get_logging_handler(custom_dest)

        self.assertEqual(handler, custom_handler)
        self.assertEqual(name, f"{type(custom_handler).__name__}-{custom_dest}")

    @mock.patch("logging.getLogger")
    def test_get_or_create_logger(self, mock_get_logger: mock.MagicMock) -> None:
        """Test _get_or_create_logger function."""
        mock_logger = mock.MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger

        # Test with default destination
        _get_or_create_logger()

        # Verify logger was created with the correct name
        handler_name = f"{type(self.mock_handler).__name__}-{_DEFAULT_DESTINATION}"
        mock_get_logger.assert_called_once_with(f"torchrec-{handler_name}")

        # Verify logger was configured correctly
        mock_logger.setLevel.assert_called_once_with(logging.DEBUG)
        mock_logger.addHandler.assert_called_once_with(self.mock_handler)
        self.assertFalse(mock_logger.propagate)

        # Verify formatter was set on the handler
        self.mock_handler.setFormatter.assert_called_once()
        formatter = self.mock_handler.setFormatter.call_args[0][0]
        self.assertIsInstance(formatter, logging.Formatter)

        # Test with custom destination
        mock_get_logger.reset_mock()
        self.mock_handler.reset_mock()

        custom_dest = "custom_dest"
        custom_handler = mock.MagicMock(spec=logging.Handler)
        _log_handlers[custom_dest] = custom_handler

        _get_or_create_logger(custom_dest)

        # Verify logger was created with the correct name
        handler_name = f"{type(custom_handler).__name__}-{custom_dest}"
        mock_get_logger.assert_called_once_with(f"torchrec-{handler_name}")

        # Verify custom handler was used
        mock_logger.addHandler.assert_called_once_with(custom_handler)

    def test_get_msg_dict_without_dist(self) -> None:
        """Test _get_msg_dict function without dist initialized."""
        # Mock dist.is_initialized to return False
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            msg_dict = _get_msg_dict("test_func", kwarg1="val1")

            # Verify msg_dict contains only func_name
            self.assertEqual(len(msg_dict), 1)
            self.assertEqual(msg_dict["func_name"], "test_func")

    def test_get_msg_dict_with_dist(self) -> None:
        """Test _get_msg_dict function with dist initialized."""
        # Mock dist functions
        with mock.patch.multiple(
            dist,
            is_initialized=mock.MagicMock(return_value=True),
            get_world_size=mock.MagicMock(return_value=4),
            get_rank=mock.MagicMock(return_value=2),
        ):
            # Test with group in kwargs
            mock_group = mock.MagicMock()
            msg_dict = _get_msg_dict("test_func", group=mock_group)

            # Verify msg_dict contains all expected keys
            self.assertEqual(len(msg_dict), 4)
            self.assertEqual(msg_dict["func_name"], "test_func")
            self.assertEqual(msg_dict["group"], str(mock_group))
            self.assertEqual(msg_dict["world_size"], "4")
            self.assertEqual(msg_dict["rank"], "2")

            # Verify get_world_size and get_rank were called with the group
            dist.get_world_size.assert_called_once_with(mock_group)
            dist.get_rank.assert_called_once_with(mock_group)

            # Test with process_group in kwargs
            dist.get_world_size.reset_mock()
            dist.get_rank.reset_mock()

            mock_process_group = mock.MagicMock()
            msg_dict = _get_msg_dict("test_func", process_group=mock_process_group)

            # Verify msg_dict contains all expected keys
            self.assertEqual(msg_dict["group"], str(mock_process_group))

            # Verify get_world_size and get_rank were called with the process_group
            dist.get_world_size.assert_called_once_with(mock_process_group)
            dist.get_rank.assert_called_once_with(mock_process_group)


if __name__ == "__main__":
    unittest.main()
