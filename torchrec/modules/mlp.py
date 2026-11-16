#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torchrec.modules.activation import SwishLayerNorm
from torchrec.modules.utils import extract_module_or_tensor_callable


class Perceptron(torch.nn.Module):
    """
    Applies a linear transformation and activation.

    Args:
        in_size (int): number of elements in each input sample.
        out_size (int): number of elements in each output sample.
        bias (bool): if set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation.
            Default: torch.relu.
        device (Optional[torch.device]): default compute device.

    Example::

        batch_size = 3
        in_size = 40
        input = torch.randn(batch_size, in_size)

        out_size = 16
        perceptron = Perceptron(in_size, out_size, bias=True)

        output = perceptron(input)
        assert list(output) == [batch_size, out_size]
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        bias: bool = True,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._out_size = out_size
        self._in_size = in_size
        self._linear: nn.Linear = nn.Linear(
            self._in_size,
            self._out_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O) where O is number of elements per
                channel in each output sample (i.e. `out_size`).
        """
        return self._activation_fn(self._linear(input))


class MLP(torch.nn.Module):
    """
    Applies a stack of Perceptron modules sequentially (i.e. Multi-Layer Perceptron).

    Args:
        in_size (int): `in_size` of the input.
        layer_sizes (List[int]): `out_size` of each Perceptron module.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (str, Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation of
            each Perceptron module.
            If `activation` is a `str`, we currently only support the follow strings, as
            "relu", "sigmoid", and "swish_layernorm".
            If `activation` is a `Callable[[], torch.nn.Module]`, `activation()` will be
            called once per Perceptron module to generate the activation module for that
            Perceptron module, and the parameters won't be shared between those activation
            modules.
            One use case is when all the activation modules share the same constructor
            arguments, but don't share the actual module parameters.
            Default: torch.relu.
        device (Optional[torch.device]): default compute device.

    Example::

        batch_size = 3
        in_size = 40
        input = torch.randn(batch_size, in_size)

        layer_sizes = [16, 8, 4]
        mlp_module = MLP(in_size, layer_sizes, bias=True)
        output = mlp_module(input)
        assert list(output.shape) == [batch_size, layer_sizes[-1]]
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: Union[
            str,
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        activation_on_last: bool = True,
    ) -> None:
        super().__init__()

        if activation == "relu":
            activation = torch.relu
        elif activation == "sigmoid":
            activation = torch.sigmoid

        if not isinstance(activation, str):
            self._mlp: torch.nn.Module = torch.nn.Sequential(
                *[
                    Perceptron(
                        layer_sizes[i - 1] if i > 0 else in_size,
                        layer_sizes[i],
                        bias=bias,
                        activation=(
                            torch.nn.Identity()
                            if not activation_on_last and i == len(layer_sizes) - 1
                            else extract_module_or_tensor_callable(activation)
                        ),
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(len(layer_sizes))
                ]
            )
        else:
            if activation == "swish_layernorm":
                self._mlp: torch.nn.Module = torch.nn.Sequential(
                    *[
                        Perceptron(
                            layer_sizes[i - 1] if i > 0 else in_size,
                            layer_sizes[i],
                            bias=bias,
                            activation=(
                                torch.nn.Identity()
                                if not activation_on_last and i == len(layer_sizes) - 1
                                else SwishLayerNorm(layer_sizes[i], device=device)
                            ),
                            device=device,
                        )
                        for i in range(len(layer_sizes))
                    ]
                )
            else:
                assert (
                    ValueError
                ), "This MLP only support str version activation function of relu, sigmoid, and swish_layernorm"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor of shape (B, I) where I is number of elements
                in each input sample.

        Returns:
            torch.Tensor: tensor of shape (B, O) where O is `out_size` of the last Perceptron module.
        """
        return self._mlp(input)
