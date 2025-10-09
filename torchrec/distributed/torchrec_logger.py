#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# mypy: allow-untyped-defs


import logging
from typing import Any

import torch.distributed as dist
from torchrec.distributed.logging_handlers import _log_handlers


__all__: list[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"torchrec-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


def _get_msg_dict(func_name: str, **kwargs: Any) -> dict[str, Any]:
    msg_dict = {
        "func_name": f"{func_name}",
    }
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict["group"] = f"{group}"
        msg_dict["world_size"] = f"{dist.get_world_size(group)}"
        msg_dict["rank"] = f"{dist.get_rank(group)}"
    return msg_dict
