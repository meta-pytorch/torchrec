#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
import functools
import inspect
import logging
from typing import Any, Callable, Dict, TypeVar

import torchrec.distributed.torchrec_logger as torchrec_logger
from torchrec.distributed.torchrec_logging_handlers import TORCHREC_LOGGER_NAME
from typing_extensions import ParamSpec


__all__: list[str] = []

global _torchrec_logger
_torchrec_logger = torchrec_logger._get_or_create_logger(TORCHREC_LOGGER_NAME)

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _torchrec_method_logger(
    **wrapper_kwargs: Any,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:  # pyre-ignore
    """This method decorator logs the input, output, and exception of wrapped events."""

    def decorator(func: Callable[_P, _T]):  # pyre-ignore
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            msg_dict = torchrec_logger._get_msg_dict(func.__name__, **kwargs)
            try:
                # exceptions
                result = func(*args, **kwargs)
            except BaseException as error:
                msg_dict["error"] = f"{error}"
                ## Add function input to log message
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                _torchrec_logger.error(msg_dict)
                raise
            ## Add function input to log message
            try:
                msg_dict["input"] = _get_input_from_func(
                    func, msg_dict, *args, **kwargs
                )
                msg_dict["output"] = str(result)
                _torchrec_logger.debug(msg_dict)
            except Exception as error:
                logging.info(f"Torchrec logger: Failed in static logger: {error}")
            return result

        return wrapper

    return decorator


def _get_input_from_func(
    func: Callable[_P, _T],
    msg_dict: Dict[str, Any],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> str:
    try:
        signature = inspect.signature(func)
        bound_args = signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        input_vars = {
            param.name: param.default for param in signature.parameters.values()
        }
        for key, value in bound_args.arguments.items():
            if key == "self" and func.__name__ == "__init__":
                # Add class name to function name if the function is a constructor
                msg_dict["func_name"] = (
                    f"{value.__class__.__name__}.{msg_dict['func_name']}"
                )
            if isinstance(value, (int, float)):
                input_vars[key] = value
            else:
                input_vars[key] = str(value)
        return str(input_vars)
    except Exception as error:
        logging.error(f"Torchrec Logger: Error in _get_input_from_func: {error}")
        return "Error in _get_input_from_func: " + str(error)
