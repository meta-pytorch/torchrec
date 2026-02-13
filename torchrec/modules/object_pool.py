#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class ObjectPool(abc.ABC, torch.nn.Module, Generic[T]):
    """
    Interface for TensorPool and KeyedJaggedTensorPool

    Defines methods for lookup, update and obtaining pool size
    """

    @abc.abstractmethod
    def lookup(self, ids: torch.Tensor) -> T:
        pass

    @abc.abstractmethod
    def update(self, ids: torch.Tensor, values: T) -> None:
        pass

    @abc.abstractproperty
    # pyrefly: ignore[bad-return]
    def pool_size(self) -> int:
        pass
