#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Generator, Optional

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PruningLogBase(object):
    pass


class PruningLogger(ABC):
    @classmethod
    @abstractmethod
    @contextmanager
    def pruning_logger(
        cls,
        event: str,
        trainer: Optional[str] = None,
        publisher: Optional[str] = None,
    ) -> Generator[object, None, None]:
        pass


class PruningLoggerDefault(PruningLogger):
    """
    noop logger as a default
    """

    @classmethod
    @contextmanager
    def pruning_logger(
        cls,
        event: str,
        trainer: Optional[str] = None,
        publisher: Optional[str] = None,
    ) -> Generator[object, None, None]:
        yield SimpleNamespace()
