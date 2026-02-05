#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Any

from torch import optim
from torchrec.optim.keyed import KeyedOptimizer


class FusedOptimizer(KeyedOptimizer, abc.ABC):
    """
    Assumes that weight update is done during backward pass,
    thus step() is a no-op.
    """

    @abc.abstractmethod
    def step(self, closure: Any = None) -> None: ...

    @abc.abstractmethod
    def zero_grad(self, set_to_none: bool = False) -> None: ...

    def __repr__(self) -> str:
        return optim.Optimizer.__repr__(self)


class EmptyFusedOptimizer(FusedOptimizer):
    """
    Fused Optimizer class with no-op step and no parameters to optimize over
    """

    def __init__(self) -> None:
        super().__init__({}, {}, {})

    def step(self, closure: Any = None) -> None:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass


class FusedOptimizerModule(abc.ABC):
    """
    Module, which does weight update during backward pass.
    """

    @property
    @abc.abstractmethod
    def fused_optimizer(self) -> KeyedOptimizer: ...
