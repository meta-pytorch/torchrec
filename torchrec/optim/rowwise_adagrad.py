#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import logging
from typing import Any, Dict, Iterable, List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger: logging.Logger = logging.getLogger(__name__)


class RowWiseAdagrad(Optimizer):
    r"""Implements Row wise Adagrad algorithm. This is an extension of the Adagrad algorithm
    https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py, for use with
    EmbeddingBag parameters, where we want the adaptive learning rate to be the same within an
    embedding row. Since we only need to store state for an embedding row, rather than every single
    parameter, we can have drastic memory savings (factor of embedding_dim).

    Note that this implementation does not currently support sparse gradients.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        *,
        maximize: bool = False,
        **unused,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            maximize=maximize,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = (
                    torch.full_like(p, init_value, memory_format=torch.preserve_format)
                    .mean(axis=1)
                    .view(-1, 1)
                )

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

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
                state["sum"].share_memory_()

    @torch.no_grad()
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
            state_sums = []
            state_steps = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state["sum"])
                    state_steps.append(state["step"])

            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        return loss


def adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs Adagrad algorithm computation.
    See :class:`~torch.optim.Adagrad` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    _single_tensor_adagrad(
        params,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        maximize=maximize,
    )


def _single_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    if weight_decay != 0 and len(state_steps) > 0 and state_steps[0].item() < 1.0:
        logger.warning(
            "Note that the weight decay mode of this optimizer may produce "
            "different results compared to the one by FBGEMM TBE. This is "
            "due to FBGEMM TBE rowwise adagrad is sparse, and will only "
            "update the optimizer states if that row has nonzero gradients."
        )

    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        if grad.is_sparse:
            raise RuntimeError("RowWise adagrad cannot be used with sparse gradients")
        # update step
        step_t += 1
        step = step_t.item()
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        state_sum += grad.pow(2).mean(axis=1).view(-1, 1)
        std = state_sum.sqrt().add_(eps)

        clr = lr / (1 + (step - 1) * lr_decay)

        param.addcdiv_(grad, std, value=-clr)
