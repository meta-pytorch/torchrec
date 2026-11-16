#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Iterable, Type
from warnings import warn

import torch


def apply_optimizer_in_backward(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
    """
    NOTE: This API is deprecated. Please use Pytorch Distributed's _apply_optimizer_in_backward instead.

    Upon backwards(), parameters will fire the corresponding optimizer
    Each parameter will have the optimizer_class and optimizer_kwargs attached to
    _optimizer and _optimizer_kwargs.

    Note - gradients for these parameters will be set to None after backwards().
    This means that any other (non applied) optimizer over this parameter will be
    a no-op.

    Args:
        optimizer_class: Type[torch.optim.Optimizer]: Optimizer to apply to parameter
        params: Iterator[nn.Parameter]: parameters to apply optimizer state to
        optimizer_kwargs: Dict[str, Any]: kwargs to pass to optimizer constructor

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        param_2 = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})
        apply_optimizer_in_backward(torch.optim.Adam, param_2, {"lr": .04})

        print(param_1._optimizer, param_1._optimizer_kwargs)
        >> torch.optim.SGD, {"lr": .02}
    """

    from torch.distributed.optim import _apply_optimizer_in_backward

    warn(
        "This API is deprecated. Please use Pytorch Distributed's _apply_optimizer_in_backward API instead.",
        DeprecationWarning,
    )
    _apply_optimizer_in_backward(
        optimizer_class=optimizer_class,
        params=params,
        optimizer_kwargs=optimizer_kwargs,
    )
