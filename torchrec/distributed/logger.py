#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchRec Distributed Logger Module.

This module provides logging utilities for TorchRec distributed training,
including method decorators for automatic logging of function calls,
inputs, outputs, and errors. It integrates with various logging handlers
(single rank, all rank, capped, and method loggers) to provide flexible
logging capabilities for distributed training scenarios.

Key Components (for developers):
    - _torchrec_method_logger: Decorator for logging function calls with
      input/output tracking.
    - Pre-configured loggers: static_logger, all_rank_logger, capped_logger
    - static_logger: Logs only from rank 0 (default for most use cases)
    - all_rank_logger: Logs from all ranks
    - capped_logger: Rate-limited logging (10 messages per location)

Util Components (not intended for direct use by developers):
    - logger utils: _get_or_create_logger, _get_logging_handler
    - input utils: _get_msg_dict, _get_input_from_func

Example:
    @_torchrec_method_logger()
    def my_distributed_function(model, batch):
        # Your distributed training code
        return result

    # Logs from rank 0 only
    static_logger.info("Starting distributed training...")

    # Logs from all ranks
    all_rank_logger.debug("Starting distributed training...")

    # Logs with rate limiting
    capped_logger.info("Starting distributed training...")


Notes:
    - _torchrec_method_logger uses _method_logger.info under the hood
    - unfiltered_logger is the last resort logger for catching errors
    - All loggers should use info level by default, and above levels such
        as WARNING, ERROR, CRITICAL, FATAL when needed (mainly for alerts)
    - debug level is used for logger debugging purpose:
        - also log when hostname starts with devgpu/devvm
        - this is for the developer to verify the logger is working as expected

"""

# mypy: allow-untyped-defs

import functools
import inspect
import logging
from typing import Any, Callable, Dict, TypeVar

from torch import distributed as dist
from torchrec.distributed.logging_handlers import (
    _log_handlers,
    AllRankStaticLogger,
    Cap01Logger,
    Cap1Logger,
    CappedLogger,
    MethodLogger,
    SingleRankStaticLogger,
    UnfilteredLogger,
)
from typing_extensions import ParamSpec


# Module exports - intentionally empty as these are internal utilities
__all__: list[str] = []

# Some inputs like can get extremely large.
# Adding logic to limit the input size by truncating any args larger than this size
ARG_SIZE_LIMIT = 800000


def _get_or_create_logger(destination: str) -> logging.Logger:
    """
    Create or retrieve a logger with the specified destination handler.

    This function creates a new logger instance with a specific logging handler
    based on the destination parameter. The logger is configured with DEBUG level,
    a standard format, and propagation disabled to prevent duplicate log messages.

    Args:
        destination: A string identifier for the logging destination. Must be
            a key in the `_log_handlers` dictionary. Common values include:
            - SingleRankStaticLogger: Logs only from rank 0
            - AllRankStaticLogger: Logs from all ranks
            - CappedLogger: Logs with per-location rate limiting (10 times per location per rank)
            - Cap1Logger: Logs with per-location rate limiting (1 time per location per rank)
            - Cap01Logger: Logs with per-location rate limiting (1 time per location on rank 0 only)
            - MethodLogger: Logs method inputs/outputs

    Returns:
        logging.Logger: A configured logger instance with the appropriate handler.

    Note:
        The logger name is formatted as "{destination}-{handler_class_name}" to
        ensure uniqueness and traceability.
    """
    logging_handler = _get_logging_handler(destination)
    logger = logging.getLogger(f"{destination}-{logging_handler.__class__.__name__}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str,
) -> logging.Handler:
    """
    Retrieve a logging handler for the specified destination.

    Args:
        destination: A string identifier for the logging destination.
            Must be a key in the `_log_handlers` dictionary.

    Returns:
        logging.Handler: The logging handler associated with the destination.
            Returns a NullHandler if the destination is not found (default
            behavior of defaultdict).
    """
    log_handler = _log_handlers[destination]
    return log_handler


# pyrefly: ignore[unknown-name]
global unfiltered_logger, single_rank_logger, all_rank_logger, capped_logger, method_logger
unfiltered_logger = _get_or_create_logger(UnfilteredLogger)
method_logger = _get_or_create_logger(MethodLogger)

static_logger = _get_or_create_logger(SingleRankStaticLogger)
all_rank_logger = _get_or_create_logger(AllRankStaticLogger)
capped_logger = _get_or_create_logger(CappedLogger)
one_time_logger = _get_or_create_logger(Cap1Logger)
one_time_rank0_logger = _get_or_create_logger(Cap01Logger)


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
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """
    A method decorator that provides comprehensive logging for TorchRec functions.

    This decorator wraps functions to automatically log:
        - Function name and input arguments (on both success and failure)
        - Function output (on success, at DEBUG level)
        - Exception details (on failure, at ERROR level)

    The decorator is designed for observability in distributed training scenarios
    where debugging across multiple processes can be challenging.

    Uses ``Callable[..., _T]`` instead of ``Callable[_P, _T]`` (ParamSpec) to avoid
    a Pyre limitation where ParamSpec captures ``self`` and causes type errors in
    child class ``super().__init__()`` calls when the parent's ``__init__`` is
    decorated.

    Args:
        **wrapper_kwargs: Additional keyword arguments for future extensibility.
            Currently unused but allows for backward-compatible additions.

    Returns:
        Callable: A decorator function that wraps the target function with logging.

    Example:
        @_torchrec_method_logger()
        def __init__(self, model, optimizer):
            # init logic here
            pass

        # When called, this will log:
        # - DEBUG: func_name, input args, and output on success
        # - ERROR: func_name, input args, and error message on exception

    Note:
        - Logging failures within the decorator are caught and logged separately
          to prevent logging infrastructure issues from breaking the application.
        - The decorator preserves the original function's signature and docstring
          via functools.wraps.
    """

    def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
        """
        Inner decorator that wraps the actual function.

        Args:
            func: The function to be wrapped with logging.

        Returns:
            Callable: The wrapped function with logging capabilities.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
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
            msg_dict = _get_msg_dict(func.__name__, **kwargs)

            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)

            except BaseException as error:
                # On exception: log error details and re-raise
                msg_dict["error"] = f"{error}"
                # pyrefly: ignore[bad-argument-type]
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                method_logger.error(msg_dict)
                raise error

            # On success: log function input and output at DEBUG level
            try:
                # pyrefly: ignore[bad-argument-type]
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                msg_dict["output"] = str(result)
                method_logger.info(msg_dict)

            except Exception as error:
                # Catch logging failures to prevent them from affecting the function
                unfiltered_logger.fatal(
                    f"Torchrec logger: Failed in method_logger: {error}"
                )

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
            elif len(str(value)) > ARG_SIZE_LIMIT:
                input_vars[key] = (
                    f"Argument removed due to size limit. Original size: {len(str(value))}"
                )
            else:
                input_vars[key] = str(value)

        return str(input_vars)

    except Exception as error:
        # Log the error and return an error message instead of crashing
        logging.error(f"Torchrec Logger: Error in _get_input_from_func: {error}")
        return "Error in _get_input_from_func: " + str(error)


def _get_msg_dict(func_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Create a message dictionary for logging with function and distributed context.

    This helper function constructs a dictionary containing the function name and,
    if torch.distributed is initialized, adds distributed training context including
    process group information, world size, and rank.

    Args:
        func_name: The name of the function being logged.
        **kwargs: Additional keyword arguments that may contain distributed
            context. Specifically looks for:
            - "group": The process group to use for distributed info
            - "process_group": Alternative key for process group

    Returns:
        dict[str, Any]: A dictionary containing:
            - "func_name": The function name
            - "group": The process group (if dist is initialized)
            - "world_size": Total number of processes (if dist is initialized)
            - "rank": Current process rank (if dist is initialized)
            - "_get_msg_dict_error": Error message if an exception occurred

    Example:
        >>> msg_dict = _get_msg_dict("train_step", group=my_process_group)
        >>> # Returns: {"func_name": "train_step", "group": "...", "world_size": "8", "rank": "0"}
    """
    try:
        msg_dict = {
            "func_name": f"{func_name}",
        }
        if dist.is_initialized():
            group = kwargs.get("group") or kwargs.get("process_group")
            msg_dict["group"] = f"{group}"
            msg_dict["world_size"] = f"{dist.get_world_size(group)}"
            msg_dict["rank"] = f"{dist.get_rank(group)}"
        return msg_dict
    except Exception as error:
        logging.error(f"Torchrec Logger: Error in _get_msg_dict: {error}")
        return {"_get_msg_dict_error": str(error)}
