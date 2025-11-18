#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Sphinx Documentation Text (for user-facing classes only)

"""
CrossNet API
"""

from typing import Callable, Optional, Union

import torch


class CrossNet(torch.nn.Module):
    r"""
    `Cross Network <https://arxiv.org/abs/1708.05123>`_:

    Cross Net is a stack of "crossing" operations on a tensor of shape :math:`(*, N)`
    to the same shape, effectively creating :math:`N` learnable polynomical functions
    over the input tensor.

    In this module, the crossing operations are defined based on a full rank matrix
    (NxN), such that the crossing effect can cover all bits on each layer. On each layer
    l, the tensor is transformed into:

    .. math ::    x_{l+1} = x_0 * (W_l \cdot x_l + b_l) + x_l

    where :math:`W_l` is a square matrix :math:`(NxN)`, :math:`*` means element-wise
    multiplication, :math:`\cdot` means matrix multiplication.

    Args:
        in_features (int): the dimension of the input.
        num_layers (int): the number of layers in the module.

    Example::

        batch_size = 3
        num_layers = 2
        in_features = 10
        input = torch.randn(batch_size, in_features)
        dcn = CrossNet(num_layers=num_layers)
        output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers
        self.kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(torch.empty(in_features, in_features))
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: tensor with shape [batch_size, in_features].
        """
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            xl_w = torch.matmul(self.kernels[layer], x_l)  # (B, N, 1)
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            x_l = x_0 * (xl_w + self.bias[layer]) + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)


class LowRankCrossNet(torch.nn.Module):
    r"""
    Low Rank Cross Net is a highly efficient cross net. Instead of using full rank cross
    matrices (NxN) at each layer, it will use two kernels :math:`W (N x r)` and
    :math:`V (r x N)`, where `r << N`, to simplify the matrix multiplication.

    On each layer l, the tensor is transformed into:

    .. math::    x_{l+1} = x_0 * (W_l \cdot (V_l \cdot x_l) + b_l) + x_l

    where :math:`W_l` is either a vector, :math:`*` means element-wise multiplication,
    and :math:`\cdot` means matrix multiplication.

    NOTE:
        Rank `r` should be chosen smartly. Usually, we  expect `r < N/2` to have
        computational savings; we should expect :math:`r ~= N/4` to preserve the
        accuracy of the full rank cross net.

    Args:
        in_features (int): the dimension of the input.
        num_layers (int): the number of layers in the module.
        low_rank (int): the rank setup of the cross matrix (default = 1).
            Value must be always >= 1.

    Example::

        batch_size = 3
        num_layers = 2
        in_features = 10
        input = torch.randn(batch_size, in_features)
        dcn = LowRankCrossNet(num_layers=num_layers, low_rank=3)
        output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        low_rank: int = 1,
    ) -> None:
        super().__init__()
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._low_rank = low_rank
        self.W_kernels: torch.nn.ParameterList = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(in_features, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.ParameterList = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._low_rank, in_features)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.ParameterList = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: tensor with shape [batch_size, in_features].
        """

        x_0 = input
        x_l = x_0

        for layer in range(self._num_layers):
            x_l_v = torch.nn.functional.linear(x_l, self.V_kernels[layer])
            x_l_w = torch.nn.functional.linear(x_l_v, self.W_kernels[layer])
            x_l = x_0 * (x_l_w + self.bias[layer]) + x_l  # (B, N)

        return x_l


class VectorCrossNet(torch.nn.Module):
    r"""
    Vector Cross Network can be refered as
    `DCN-V1 <https://arxiv.org/pdf/1708.05123.pdf>`_.

    It is also a specialized low rank cross net, where rank=1. In this version, on each
    layer, instead of keeping two kernels W and V, we only keep one vector kernel W
    (Nx1). We use the dot operation to compute the "crossing" effect of the features,
    thus saving two matrix multiplications to further reduce computational cost and cut
    the number of learnable parameters.

    On each layer l, the tensor is transformed into

    .. math::    x_{l+1} = x_0 * (W_l . x_l) + b_l + x_l

    where :math:`W_l` is either a vector, :math:`*` means element-wise multiplication;
    :math:`.` means dot operations.

    Args:
        in_features (int): the dimension of the input.
        num_layers (int): the number of layers in the module.

    Example::

        batch_size = 3
        num_layers = 2
        in_features = 10
        input = torch.randn(batch_size, in_features)
        dcn = VectorCrossNet(num_layers=num_layers)
        output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers
        self.kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(torch.empty(in_features, 1))
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: tensor with shape [batch_size, in_features].
        """

        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            xl_w = torch.tensordot(
                x_l,
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                self.kernels[layer],
                dims=([1], [0]),
            )  # (B, 1, 1)
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            x_l = torch.matmul(x_0, xl_w) + self.bias[layer] + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)


class LowRankMixtureCrossNet(torch.nn.Module):
    r"""
    Low Rank Mixture Cross Net is a DCN V2 implementation from the `paper
    <https://arxiv.org/pdf/2008.13535.pdf>`_:

    `LowRankMixtureCrossNet` defines the learnable crossing parameter per layer as a
    low-rank matrix :math:`(N*r)` together with mixture of experts. Compared to
    `LowRankCrossNet`, instead of relying on one single expert to learn feature crosses,
    this module leverages such :math:`K` experts; each learning feature interactions in
    different subspaces, and adaptively combining the learned crosses using a gating
    mechanism that depends on input :math:`x`..

    On each layer l, the tensor is transformed into:

    .. math::    x_{l+1} = MoE({expert_i : i \in K_{experts}}) + x_l

    and each :math:`expert_i` is defined as:

    .. math::   expert_i = x_0 * (U_{li} \cdot g(C_{li} \cdot g(V_{li} \cdot x_l)) + b_l)

    where :math:`U_{li} (N, r)`, :math:`C_{li} (r, r)` and :math:`V_{li} (r, N)` are
    low-rank matrices, :math:`*` means element-wise multiplication, :math:`x` means
    matrix multiplication, and :math:`g()` is the non-linear activation function.

    When num_expert is 1, the gate evaluation and MOE will be skipped to save
    computation.

    Args:
        in_features (int): the dimension of the input.
        num_layers (int): the number of layers in the module.
        low_rank (int): the rank setup of the cross matrix (default = 1).
            Value must be always >= 1
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the non-linear activation function, used in defining experts.
            Default is relu.

    Example::

        batch_size = 3
        num_layers = 2
        in_features = 10
        input = torch.randn(batch_size, in_features)
        dcn = LowRankCrossNet(num_layers=num_layers, num_experts=5, low_rank=3)
        output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        num_experts: int = 1,
        low_rank: int = 1,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__()
        assert num_experts >= 1, "num_experts must be larger or equal to 1"
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._num_experts = num_experts
        self._low_rank = low_rank
        self._in_features = in_features
        self.U_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._in_features, self._low_rank
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._low_rank, self._in_features
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.zeros_(torch.empty(self._in_features, 1))
                )
                for i in range(self._num_layers)
            ]
        )
        self.gates: Optional[torch.nn.Module] = (
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(self._in_features, 1, bias=False)
                    for i in range(self._num_experts)
                ]
            )
            if self._num_experts > 1
            else None
        )

        self._activation = activation
        self.C_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._num_experts, self._low_rank, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: tensor with shape [batch_size, in_features].
        """

        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            # set up gating:
            if self._num_experts > 1:
                gating = []
                for i in range(self._num_experts):
                    # pyre-ignore[16]: `Optional` has no attribute `__getitem__`.
                    gating.append(self.gates[i](x_l.squeeze(2)))
                gating = torch.stack(gating, 1)  # (B, K, 1)

            # set up experts
            experts = []
            for i in range(self._num_experts):
                expert = torch.matmul(
                    # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                    self.V_kernels[layer][i],
                    x_l,
                )  # (B, r, 1)
                expert = torch.matmul(
                    # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                    self.C_kernels[layer][i],
                    self._activation(expert),
                )  # (B, r, 1)
                expert = torch.matmul(
                    # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                    self.U_kernels[layer][i],
                    self._activation(expert),
                )  # (B, N, 1)
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                expert = x_0 * (expert + self.bias[layer])  # (B, N, 1)
                experts.append(expert.squeeze(2))  # (B, N)
            experts = torch.stack(experts, 2)  # (B, N, K)

            if self._num_experts > 1:
                # MOE update
                moe = torch.matmul(
                    experts,
                    # pyre-ignore[61]: `gating` may not be initialized here.
                    torch.nn.functional.softmax(gating, 1),
                )  # (B, N, 1)
                x_l = moe + x_l  # (B, N, 1)
            else:
                x_l = experts + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)
