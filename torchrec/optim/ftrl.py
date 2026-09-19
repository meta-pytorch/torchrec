#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    r"""Implements the FTRL-Proximal algorithm (Follow The Regularized Leader).

    This is the dense counterpart of ``EmbOptimType.FTRL``, the fused FBGEMM TBE
    optimizer. It implements the same update, which matches TensorFlow's
    ``ApplyFtrlV2`` (with ``l2_shrinkage = 0``) and Alibaba x-deeplearning's
    ``FtrlUpdater``. Per coordinate, with state ``accum`` (the running sum of
    squared gradients) and ``linear``:

    .. code-block:: text

        new_accum = accum + g^2
        sigma_new = new_accum ^ (-learning_rate_power)
        sigma_old = accum     ^ (-learning_rate_power)
        linear   += g - (sigma_new - sigma_old) / lr * w
        quadratic = sigma_new / lr + 2 * l2_reg
        w         = |linear| > l1_reg ? (l1_reg * sgn(linear) - linear) / quadratic : 0
        accum     = new_accum

    Note that FTRL *recomputes* the parameter from ``(accum, linear)`` rather
    than incrementing it, so a parameter's initial value is discarded on its
    first step. This is inherent to the algorithm, and is why ``l1_reg`` can
    pin coordinates to exactly zero.

    The keyword names deliberately match the fused TBE's constructor arguments
    so that the same kwargs work for both, e.g. with
    :func:`~torchrec.optim.apply_optimizer_in_backward.apply_optimizer_in_backward`.

    Note that this implementation does not currently support sparse gradients.
    Unlike the fused kernel, which only visits rows present in a batch, this
    dense optimizer recomputes every row every step; rows whose state is still
    ``(0, 0)`` are therefore set to zero.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate, FTRL's ``alpha`` (default: 1e-2).
            Must be strictly positive -- the update divides by it.
        ftrl_learning_rate_power (float, optional): exponent applied to
            ``accum``; -0.5 gives the classic 1/sqrt schedule (default: -0.5)
        ftrl_l1_reg (float, optional): L1 regularization strength (default: 0.0)
        ftrl_l2_reg (float, optional): proximal L2 regularization strength.
            Enters as ``2 * l2`` in the denominator, not as a gradient-space
            weight decay (default: 0.0)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        ftrl_learning_rate_power: float = -0.5,
        ftrl_l1_reg: float = 0.0,
        ftrl_l2_reg: float = 0.0,
        **unused: Any,
    ) -> None:
        if not 0.0 < lr:
            raise ValueError(
                "Invalid learning rate: {} (FTRL divides by it)".format(lr)
            )
        if not 0.0 <= ftrl_l1_reg:
            raise ValueError("Invalid ftrl_l1_reg value: {}".format(ftrl_l1_reg))
        if not 0.0 <= ftrl_l2_reg:
            raise ValueError("Invalid ftrl_l2_reg value: {}".format(ftrl_l2_reg))

        defaults = dict(
            lr=lr,
            ftrl_learning_rate_power=ftrl_learning_rate_power,
            ftrl_l1_reg=ftrl_l1_reg,
            ftrl_l2_reg=ftrl_l2_reg,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                # Named to match the fused TBE's get_optimizer_state() slots.
                state["accum"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["linear"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["accum"].share_memory_()
                state["linear"].share_memory_()

    @torch.no_grad()
    # pyrefly: ignore[bad-override]
    def step(self, closure=None) -> torch.Tensor:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_accums = []
            state_linears = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_accums.append(state["accum"])
                    state_linears.append(state["linear"])
                    state_steps.append(state["step"])

            ftrl(
                params_with_grad,
                grads,
                state_accums,
                state_linears,
                state_steps,
                lr=group["lr"],
                learning_rate_power=group["ftrl_learning_rate_power"],
                l1_reg=group["ftrl_l1_reg"],
                l2_reg=group["ftrl_l2_reg"],
            )

        # pyrefly: ignore[bad-return]
        return loss


def ftrl(
    params: List[Tensor],
    grads: List[Tensor],
    state_accums: List[Tensor],
    state_linears: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    *,
    lr: float,
    learning_rate_power: float,
    l1_reg: float,
    l2_reg: float,
) -> None:
    r"""Functional API that performs the FTRL-Proximal computation.

    See :class:`~torchrec.optim.ftrl.FTRL` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    _single_tensor_ftrl(
        params,
        grads,
        state_accums,
        state_linears,
        state_steps,
        lr=lr,
        learning_rate_power=learning_rate_power,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
    )


def _single_tensor_ftrl(
    params: List[Tensor],
    grads: List[Tensor],
    state_accums: List[Tensor],
    state_linears: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    learning_rate_power: float,
    l1_reg: float,
    l2_reg: float,
) -> None:
    exponent = -learning_rate_power

    for param, grad, accum, linear, step_t in zip(
        params, grads, state_accums, state_linears, state_steps
    ):
        if grad.is_sparse:
            raise RuntimeError("FTRL cannot be used with sparse gradients")
        step_t += 1

        sigma_old = accum.pow(exponent)
        accum.add_(grad * grad)
        sigma_new = accum.pow(exponent)

        # NOTE: reads `param` before it is overwritten below.
        linear.add_(grad - (sigma_new - sigma_old) / lr * param)

        quadratic = sigma_new / lr + 2.0 * l2_reg
        param.copy_(
            torch.where(
                linear.abs() > l1_reg,
                (l1_reg * torch.sign(linear) - linear) / quadratic,
                torch.zeros_like(linear),
            )
        )
