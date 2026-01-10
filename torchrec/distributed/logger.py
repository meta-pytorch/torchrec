#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs

"""
TorchRec Distributed Logger Module.

This module provides logging utilities for TorchRec's distributed training components.
It includes a method decorator that automatically logs function inputs, outputs,
and exceptions for debugging and observability purposes.

Key Components:
    - _torchrec_method_logger: A decorator for logging method calls with full context
    - _get_input_from_func: A helper to extract and format function input arguments

Usage Example:
    @_torchrec_method_logger()
    def my_training_function(model, data, learning_rate=0.01):
        # Function implementation
        return result
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, TypeVar

import torchrec.distributed.torchrec_logger as torchrec_logger
from torchrec.distributed.torchrec_logging_handlers import TORCHREC_LOGGER_NAME
from typing_extensions import ParamSpec


# Module exports - intentionally empty as these are internal utilities
__all__: list[str] = []

# =============================================================================
# Module-Level Logger Initialization
# =============================================================================

# Global logger instance for TorchRec distributed operations.
# This logger is shared across all decorated functions in the distributed module.
global _torchrec_logger
_torchrec_logger = torchrec_logger._get_or_create_logger(TORCHREC_LOGGER_NAME)

# =============================================================================
# Type Variables for Generic Type Hints
# =============================================================================

# TypeVar for return type preservation in decorated functions
_T = TypeVar("_T")

# ParamSpec for preserving parameter types in decorated functions
_P = ParamSpec("_P")


# =============================================================================
# Method Logging Decorator
# =============================================================================


def _torchrec_method_logger(
    **wrapper_kwargs: Any,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:  # pyre-ignore
    """
    A method decorator that provides comprehensive logging for TorchRec functions.

    This decorator wraps functions to automatically log:
        - Function name and input arguments (on both success and failure)
        - Function output (on success, at DEBUG level)
        - Exception details (on failure, at ERROR level)

    The decorator is designed for observability in distributed training scenarios
    where debugging across multiple processes can be challenging.

    Args:
        **wrapper_kwargs: Additional keyword arguments for future extensibility.
            Currently unused but allows for backward-compatible additions.

    Returns:
        Callable: A decorator function that wraps the target function with logging.

    Example:
        @_torchrec_method_logger()
        def train_step(model, batch, optimizer):
            # Training logic here
            return loss

        # When called, this will log:
        # - DEBUG: func_name, input args, and output on success
        # - ERROR: func_name, input args, and error message on exception

    Note:
        - Logging failures within the decorator are caught and logged separately
          to prevent logging infrastructure issues from breaking the application.
        - The decorator preserves the original function's signature and docstring
          via functools.wraps.
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:  # pyre-ignore
        """
        Inner decorator that wraps the actual function.

        Args:
            func: The function to be wrapped with logging.

        Returns:
            Callable: The wrapped function with logging capabilities.
        """

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            """
            Wrapper function that executes the original function with logging.

            This wrapper:
                1. Creates a message dictionary with function metadata
                2. Executes the original function
                3. Logs success (DEBUG) or failure (ERROR) with context

            Args:
                *args: Positional arguments passed to the original function.
                **kwargs: Keyword arguments passed to the original function.

            Returns:
                The return value of the original function.

            Raises:
                Any exception raised by the original function is re-raised
                after logging.
            """
            # Initialize the log message dictionary with function name and kwargs
            msg_dict = torchrec_logger._get_msg_dict(func.__name__, **kwargs)

            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)

            except BaseException as error:
                # On exception: log error details and re-raise
                msg_dict["error"] = f"{error}"
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                _torchrec_logger.error(msg_dict)
                raise

            # On success: log function input and output at DEBUG level
            try:
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                msg_dict["output"] = str(result)
                _torchrec_logger.debug(msg_dict)

            except Exception as error:
                # Catch logging failures to prevent them from affecting the function
                logging.info(f"Torchrec logger: Failed in static logger: {error}")

            return result

        return wrapper

    return decorator


# =============================================================================
# Input Extraction Helper
# =============================================================================


def _get_input_from_func(
    func: Callable[_P, _T],
    msg_dict: Dict[str, Any],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> str:
    """
    Extract and format function input arguments for logging.

    This helper function uses Python's inspect module to extract all arguments
    passed to a function, including positional args, keyword args, and defaults.
    It handles special cases like constructor methods where the class name
    should be included in the function name.

    Args:
        func: The function whose inputs are being extracted.
        msg_dict: The message dictionary to update (modified in-place for
            constructor functions to prepend class name).
        *args: The positional arguments passed to the function.
        **kwargs: The keyword arguments passed to the function.

    Returns:
        str: A string representation of all input arguments as a dictionary.
            On error, returns an error message string.

    Example:
        For a function call `train(model, lr=0.01, epochs=10)`, this might return:
        "{'model': '<Model object>', 'lr': 0.01, 'epochs': 10}"

    Note:
        - Numeric types (int, float) are preserved as-is for readability
        - All other types are converted to their string representation
        - For __init__ methods, the class name is prepended to func_name in msg_dict
    """
    try:
        # Get the function's signature for parameter introspection
        signature = inspect.signature(func)

        # Bind the provided arguments to the function's parameters
        # bind_partial allows for missing arguments (useful for partial calls)
        bound_args = signature.bind_partial(*args, **kwargs)

        # Fill in any missing arguments with their default values
        bound_args.apply_defaults()

        # Initialize input_vars with parameter defaults
        input_vars = {
            param.name: param.default for param in signature.parameters.values()
        }

        # Update input_vars with actual argument values
        for key, value in bound_args.arguments.items():
            # Special handling for constructor methods (__init__)
            # Prepend the class name to the function name for better identification
            if key == "self" and func.__name__ == "__init__":
                msg_dict["func_name"] = (
                    f"{value.__class__.__name__}.{msg_dict['func_name']}"
                )

            # Preserve numeric types as-is, convert others to string
            # This improves readability for common numeric parameters like
            # learning rates, batch sizes, etc.
            if isinstance(value, (int, float)):
                input_vars[key] = value
            else:
                input_vars[key] = str(value)

        return str(input_vars)

    except Exception as error:
        # Log the error and return an error message instead of crashing
        logging.error(f"Torchrec Logger: Error in _get_input_from_func: {error}")
        return "Error in _get_input_from_func: " + str(error)
