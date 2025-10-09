#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# mypy: allow-untyped-defs
import functools
import inspect
from typing import Any, Callable, TypeVar

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
                ## Add function input to log message
                msg_dict["input"] = _get_input_from_func(func, *args, **kwargs)
                # exceptions
                result = func(*args, **kwargs)
            except BaseException as error:
                msg_dict["error"] = f"{error}"
                _torchrec_logger.error(msg_dict)
                raise
            msg_dict["output"] = str(result)
            _torchrec_logger.debug(msg_dict)
            return result

        return wrapper

    return decorator


def _get_input_from_func(
    func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> str:
    signature = inspect.signature(func)
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    input_vars = {param.name: param.default for param in signature.parameters.values()}
    for key, value in bound_args.arguments.items():
        if isinstance(value, (int, float)):
            input_vars[key] = value
        else:
            input_vars[key] = str(value)
    return str(input_vars)
