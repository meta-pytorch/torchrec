#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import abstractmethod
from enum import Enum, unique

import torch
from torch import nn


class CacheMixin:
    """
    A mixin to allow modules that cache computation to clear the cache.
    """

    @abstractmethod
    def clear_cache(self) -> None: ...


class CopyMixIn:
    @abstractmethod
    def copy(self, device: torch.device) -> nn.Module: ...


class ModuleCopyMixin(CopyMixIn):
    """
    A mixin to allow modules to override copy behaviors in DMP.
    """

    def copy(self, device: torch.device) -> nn.Module:
        return self.to(device)


class ModuleNoCopyMixin(CopyMixIn):
    """
    A mixin to allow modules to override copy behaviors in DMP.
    """

    def copy(self, device: torch.device) -> nn.Module:
        return self


# moved DataType here to avoid circular import
# TODO: organize types and dependencies
@unique
class DataType(Enum):
    """
    Our fusion implementation supports only certain types of data
    so it makes sense to retrict in a non-fused version as well.
    """

    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT64 = "INT64"
    INT32 = "INT32"
    INT8 = "INT8"
    UINT8 = "UINT8"
    INT4 = "INT4"
    INT2 = "INT2"
    NFP8 = "NFP8"

    def __str__(self) -> str:
        return self.value
