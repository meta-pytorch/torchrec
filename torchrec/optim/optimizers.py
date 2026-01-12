#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Iterable, Iterator, Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


def in_backward_optimizer_filter(
    named_parameters: Iterator[Tuple[str, nn.Parameter]], include: bool = False
) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Filters named_parameters for whether they are or or not params that use
    the in_backward_optimizer.
    Note: This only supports the in_backward_optimizer from PT-D's API.
        The torchrec's equivalent API is deprecated and is not supported.
    Args:
    named_parameters(Iterator[Tuple[str, nn.Parameter]]): named_parameters
    include(bool): If true, only yields params with in_backward_optimizer. If false, returns the outside set
        Defaults to include params that are not in_backward (False)
    """
    for fqn, param in named_parameters:
        if hasattr(param, "_in_backward_optimizers") == include:
            yield fqn, param


class SGD(Optimizer):
    r"""
    Placeholder for SGD. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class LarsSGD(Optimizer):
    r"""
    Placeholder for LARS_SGD. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class LAMB(Optimizer):
    r"""
    Placeholder for LAMB. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class PartialRowWiseLAMB(Optimizer):
    r"""
    Placeholder for PartialRowWiseLAMB. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class Adam(Optimizer):
    r"""
    Placeholder for Adam. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class PartialRowWiseAdam(Optimizer):
    r"""
    Placeholder for PartialRowWiseAdam. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class Adagrad(Optimizer):
    r"""
    Placeholder for Adagrad. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError
