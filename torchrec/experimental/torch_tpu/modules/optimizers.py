#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stateless functional optimizer library for compiling and running PyTorch training steps on TPU."""

import abc
import dataclasses
from typing import Any, Mapping, Optional

import torch
import torchrec
from torch.optim import adamw as torch_adamw, sgd as torch_sgd
from torch.utils import _pytree

KeyedOptimizer = torchrec.optim.keyed.KeyedOptimizer
CombinedOptimizer = torchrec.optim.keyed.CombinedOptimizer
in_backward_optimizer_filter = torchrec.optim.optimizers.in_backward_optimizer_filter

__all__ = [
    "AdamW",
    "Adagrad",
    "CombinedOptimizer",
    "FusedAdamw",
    "FusedSgd",
    "KeyedStatelessOptimizer",
    "Optimizer",
    "ParamGroup",
    "ReferenceAdamw",
    "ReferenceAdagrad",
    "ReferenceSgd",
    "SGD",
    "in_backward_optimizer_filter",
]


class Optimizer(abc.ABC):
    """Base class for functional optimizers."""

    @dataclasses.dataclass
    class ParamGroup(abc.ABC):
        """Base class for optimizer parameter groups and state."""

        params: dict[str, torch.Tensor]

        def state_dict(self) -> dict[str, Any]:
            """Serializes ParamGroup into a state dictionary."""
            return dataclasses.asdict(self)

        @classmethod
        def from_state_dict(cls, state_dict: dict[str, Any]) -> Any:
            """Deserializes a state dictionary into a ParamGroup."""
            return cls(**state_dict)

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    @abc.abstractmethod
    def init_param_group(
        self, params: dict[str, torch.Tensor] | torch.nn.Module
    ) -> Any:
        """Initializes optimizer state from model parameters or nn.Module."""
        pass

    @abc.abstractmethod
    def step(self, param_group: Any, grads: dict[str, torch.Tensor]) -> Any:
        """Performs optimizer step on param_group given grads and returns updated ParamGroup."""
        pass

    @abc.abstractmethod
    def state_dict(self, param_group: Any) -> dict[str, Any]:
        """Returns state dictionary matching PyTorch/TorchRec optimizer state_dict format."""
        pass

    @abc.abstractmethod
    def load_state_dict(
        self, param_group: Any, state_dict: dict[str, Any]
    ) -> Any:
        """Loads state dictionary matching PyTorch/TorchRec optimizer format and returns updated ParamGroup."""
        pass

    def __call__(self, param_group: Any, grads: dict[str, torch.Tensor]) -> Any:
        return self.step(param_group, grads)


ParamGroup = Optimizer.ParamGroup


class AdamW(Optimizer):
    """Base class for functional AdamW optimizers."""

    @dataclasses.dataclass
    class ParamGroup(Optimizer.ParamGroup):
        """Holds optimizer state for AdamW."""

        opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
            default_factory=dict
        )
        opt_state_v: dict[str, torch.Tensor] = dataclasses.field(
            default_factory=dict
        )
        opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
        extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        use_bfloat16_moments: bool = False,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.use_bfloat16_moments = use_bfloat16_moments

    def init_param_group(
        self, params: dict[str, torch.Tensor] | torch.nn.Module
    ) -> ParamGroup:
        """Initializes AdamW optimizer state from model parameters or nn.Module."""
        if isinstance(params, torch.nn.Module):
            params = {
                name: param.detach() for name, param in params.named_parameters()
            }
        if not params:
            return self.ParamGroup(params={})

        opt_state_m = {
            name: torch.zeros_like(
                param,
                dtype=torch.bfloat16 if self.use_bfloat16_moments else param.dtype,
                memory_format=torch.preserve_format,
                device=param.device,
            )
            for name, param in params.items()
        }
        opt_state_v = {
            name: torch.zeros_like(
                param,
                dtype=torch.bfloat16 if self.use_bfloat16_moments else param.dtype,
                memory_format=torch.preserve_format,
                device=param.device,
            )
            for name, param in params.items()
        }
        opt_steps = {
            name: torch.tensor(1.0, dtype=torch.float32, device=param.device)
            for name, param in params.items()
        }

        return self.ParamGroup(
            params=params,
            opt_state_m=opt_state_m,
            opt_state_v=opt_state_v,
            opt_steps=opt_steps,
        )

    def state_dict(self, param_group: ParamGroup) -> dict[str, Any]:
        """Serializes AdamW state in PyTorch/TorchRec optimizer state_dict format."""
        state = {}
        for name in param_group.params:
            s = {}
            if name in param_group.opt_steps:
                step = param_group.opt_steps[name]
                s["step"] = step.item() if isinstance(step, torch.Tensor) else step
            if name in param_group.opt_state_m:
                s["exp_avg"] = param_group.opt_state_m[name]
            if name in param_group.opt_state_v:
                s["exp_avg_sq"] = param_group.opt_state_v[name]
            state[name] = s
        return {
            "state": state,
            "param_groups": [{
                "lr": self.lr,
                "betas": (self.beta1, self.beta2),
                "eps": self.eps,
                "weight_decay": self.weight_decay,
                "params": list(param_group.params.keys()),
            }],
        }

    def load_state_dict(
        self, param_group: ParamGroup, state_dict: dict[str, Any]
    ) -> ParamGroup:
        """Loads state from a PyTorch/TorchRec state_dict into AdamW.ParamGroup."""
        if "param_groups" in state_dict and state_dict["param_groups"]:
            pg_conf = state_dict["param_groups"][0]
            if "lr" in pg_conf:
                self.lr = pg_conf["lr"]
            if "betas" in pg_conf:
                self.beta1, self.beta2 = pg_conf["betas"]
            if "eps" in pg_conf:
                self.eps = pg_conf["eps"]
            if "weight_decay" in pg_conf:
                self.weight_decay = pg_conf["weight_decay"]

        new_m = dict(param_group.opt_state_m)
        new_v = dict(param_group.opt_state_v)
        new_steps = dict(param_group.opt_steps)

        saved_state = state_dict.get("state", {})
        param_keys = list(param_group.params.keys())
        for idx_or_name, s in saved_state.items():
            # Support both string name keys and integer index keys from PyTorch
            name = (
                idx_or_name
                if isinstance(idx_or_name, str)
                else (
                    param_keys[idx_or_name] if idx_or_name < len(param_keys) else None
                )
            )
            if name is None or name not in param_group.params:
                continue
            param = param_group.params[name]
            if "exp_avg" in s:
                new_m[name] = (
                    s["exp_avg"]
                    if isinstance(s["exp_avg"], torch.Tensor)
                    else torch.tensor(s["exp_avg"], device=param.device)
                )
            if "exp_avg_sq" in s:
                new_v[name] = (
                    s["exp_avg_sq"]
                    if isinstance(s["exp_avg_sq"], torch.Tensor)
                    else torch.tensor(s["exp_avg_sq"], device=param.device)
                )
            if "step" in s:
                step_val = s["step"]
                new_steps[name] = (
                    step_val
                    if isinstance(step_val, torch.Tensor)
                    else torch.tensor(
                        float(step_val), dtype=torch.float32, device=param.device
                    )
                )

        return self.ParamGroup(
            params=param_group.params,
            opt_state_m=new_m,
            opt_state_v=new_v,
            opt_steps=new_steps,
            extra_state=param_group.extra_state,
        )


class ReferenceAdamw(AdamW):
    """Reference pure PyTorch functional implementation of AdamW."""

    def _reference_adamw_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        step: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes reference AdamW update returning (new_param, new_m, new_v)."""
        if self.weight_decay != 0.0:
            p_decayed = param * (1.0 - self.lr * self.weight_decay)
        else:
            p_decayed = param

        step_val = (
            step.to(torch.float32) if isinstance(step, torch.Tensor) else step
        )
        bias_correction1 = 1.0 - torch.pow(
            torch.tensor(self.beta1, device=param.device), step_val
        )
        bias_correction2 = 1.0 - torch.pow(
            torch.tensor(self.beta2, device=param.device), step_val
        )
        sqrt_bc2 = torch.sqrt(bias_correction2)
        step_size = (self.lr / bias_correction1) * sqrt_bc2
        scaled_eps = self.eps * sqrt_bc2

        new_m = (m * self.beta1 + grad * (1.0 - self.beta1)).to(m.dtype)
        new_v = (v * self.beta2 + (grad * grad) * (1.0 - self.beta2)).to(v.dtype)
        denom = torch.sqrt(new_v.to(param.dtype)) + scaled_eps
        new_p = p_decayed - (step_size * new_m.to(param.dtype)) / denom

        return new_p, new_m, new_v

    def step(
        self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
    ) -> AdamW.ParamGroup:
        """Reference pure PyTorch implementation of AdamW step."""
        new_params = {}
        new_m_dict = {}
        new_v_dict = {}
        new_steps_dict = {}
        for name, param in param_group.params.items():
            grad = grads.get(name)
            if grad is None:
                new_params[name] = param
                if name in param_group.opt_state_m:
                    new_m_dict[name] = param_group.opt_state_m[name]
                if name in param_group.opt_state_v:
                    new_v_dict[name] = param_group.opt_state_v[name]
                if name in param_group.opt_steps:
                    new_steps_dict[name] = param_group.opt_steps[name]
                continue
            step_tensor = param_group.opt_steps[name]
            m = param_group.opt_state_m[name]
            v = param_group.opt_state_v[name]
            new_p, new_m, new_v = self._reference_adamw_update(
                param=param,
                grad=grad,
                m=m,
                v=v,
                step=step_tensor,
            )
            new_params[name] = new_p
            new_m_dict[name] = new_m
            new_v_dict[name] = new_v
            new_steps_dict[name] = step_tensor + 1.0
        return AdamW.ParamGroup(
            params=new_params,
            opt_state_m=new_m_dict,
            opt_state_v=new_v_dict,
            opt_steps=new_steps_dict,
            extra_state=param_group.extra_state,
        )


class FusedAdamw(AdamW):
    """Fused AdamW implementation calling torch.optim.adamw.adamw directly with fused=True."""

    def step(
        self, param_group: AdamW.ParamGroup, grads: dict[str, torch.Tensor]
    ) -> AdamW.ParamGroup:
        """Fused AdamW step calling torch.optim.adamw.adamw directly."""
        param_keys = [
            k for k in param_group.params.keys() if grads.get(k) is not None
        ]
        if not param_keys:
            return param_group
        param_list = [param_group.params[k] for k in param_keys]
        grad_list = [grads[k] for k in param_keys]
        m_list = [param_group.opt_state_m[k] for k in param_keys]
        v_list = [param_group.opt_state_v[k] for k in param_keys]
        step_list = [param_group.opt_steps[k] for k in param_keys]

        torch_adamw.adamw(
            params=param_list,
            grads=grad_list,
            exp_avgs=m_list,
            exp_avg_sqs=v_list,
            max_exp_avg_sqs=[],
            state_steps=step_list,
            foreach=False,
            capturable=True,
            fused=True,
            amsgrad=False,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            maximize=False,
        )
        return param_group


class SGD(Optimizer):
    """Base class for functional SGD optimizers."""

    @dataclasses.dataclass
    class ParamGroup(Optimizer.ParamGroup):
        """Holds optimizer state for SGD."""

        opt_state_m: dict[str, torch.Tensor] = dataclasses.field(
            default_factory=dict
        )
        opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
        extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

    def init_param_group(
        self, params: dict[str, torch.Tensor] | torch.nn.Module
    ) -> ParamGroup:
        """Initializes SGD optimizer state from model parameters or nn.Module."""
        if isinstance(params, torch.nn.Module):
            params = {
                name: param.detach() for name, param in params.named_parameters()
            }
        if not params:
            return self.ParamGroup(params={})

        opt_state_m = (
            {
                name: torch.zeros_like(
                    param,
                    memory_format=torch.preserve_format,
                    device=param.device,
                )
                for name, param in params.items()
            }
            if self.momentum != 0.0
            else {}
        )
        opt_steps = {
            name: torch.tensor(1.0, dtype=torch.float32, device=param.device)
            for name, param in params.items()
        }

        return self.ParamGroup(
            params=params,
            opt_state_m=opt_state_m,
            opt_steps=opt_steps,
        )

    def state_dict(self, param_group: ParamGroup) -> dict[str, Any]:
        """Serializes SGD state in PyTorch/TorchRec optimizer state_dict format."""
        state = {}
        for name in param_group.params:
            s = {}
            if name in param_group.opt_steps:
                step = param_group.opt_steps[name]
                s["step"] = step.item() if isinstance(step, torch.Tensor) else step
            if name in param_group.opt_state_m:
                s["momentum_buffer"] = param_group.opt_state_m[name]
            state[name] = s
        return {
            "state": state,
            "param_groups": [{
                "lr": self.lr,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "weight_decay": self.weight_decay,
                "nesterov": self.nesterov,
                "params": list(param_group.params.keys()),
            }],
        }

    def load_state_dict(
        self, param_group: ParamGroup, state_dict: dict[str, Any]
    ) -> ParamGroup:
        """Loads state from a PyTorch/TorchRec state_dict into SGD.ParamGroup."""
        if "param_groups" in state_dict and state_dict["param_groups"]:
            pg_conf = state_dict["param_groups"][0]
            if "lr" in pg_conf:
                self.lr = pg_conf["lr"]
            if "momentum" in pg_conf:
                self.momentum = pg_conf["momentum"]
            if "dampening" in pg_conf:
                self.dampening = pg_conf["dampening"]
            if "weight_decay" in pg_conf:
                self.weight_decay = pg_conf["weight_decay"]
            if "nesterov" in pg_conf:
                self.nesterov = pg_conf["nesterov"]

        new_m = dict(param_group.opt_state_m)
        new_steps = dict(param_group.opt_steps)

        saved_state = state_dict.get("state", {})
        param_keys = list(param_group.params.keys())
        for idx_or_name, s in saved_state.items():
            # Support both string name keys and integer index keys from PyTorch
            name = (
                idx_or_name
                if isinstance(idx_or_name, str)
                else (
                    param_keys[idx_or_name] if idx_or_name < len(param_keys) else None
                )
            )
            if name is None or name not in param_group.params:
                continue
            param = param_group.params[name]
            if "momentum_buffer" in s:
                new_m[name] = (
                    s["momentum_buffer"]
                    if isinstance(s["momentum_buffer"], torch.Tensor)
                    else torch.tensor(s["momentum_buffer"], device=param.device)
                )
            if "step" in s:
                step_val = s["step"]
                new_steps[name] = (
                    step_val
                    if isinstance(step_val, torch.Tensor)
                    else torch.tensor(
                        float(step_val), dtype=torch.float32, device=param.device
                    )
                )

        return self.ParamGroup(
            params=param_group.params,
            opt_state_m=new_m,
            opt_steps=new_steps,
            extra_state=param_group.extra_state,
        )


class ReferenceSgd(SGD):
    """Reference pure PyTorch functional implementation of SGD."""

    def _reference_sgd_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor | None = None,
        step: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Computes reference SGD update returning (new_param, new_m)."""
        if self.weight_decay != 0.0:
            d_p = grad + self.weight_decay * param
        else:
            d_p = grad

        new_m = None
        if self.momentum != 0.0 and m is not None:
            if isinstance(step, torch.Tensor):
                new_m = torch.where(
                    step == 1.0, d_p, self.momentum * m + (1.0 - self.dampening) * d_p
                )
            else:
                new_m = (
                    d_p
                    if step == 1.0
                    else self.momentum * m + (1.0 - self.dampening) * d_p
                )
            if self.nesterov:
                d_p = d_p + self.momentum * new_m
            else:
                d_p = new_m

        new_p = param - self.lr * d_p
        return new_p, new_m

    def step(
        self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
    ) -> SGD.ParamGroup:
        """Reference pure PyTorch implementation of SGD step."""
        new_params = {}
        new_m_dict = {}
        new_steps_dict = {}
        for name, param in param_group.params.items():
            grad = grads.get(name)
            if grad is None:
                new_params[name] = param
                if name in param_group.opt_state_m:
                    new_m_dict[name] = param_group.opt_state_m[name]
                if name in param_group.opt_steps:
                    new_steps_dict[name] = param_group.opt_steps[name]
                continue
            step_tensor = param_group.opt_steps[name]
            m = param_group.opt_state_m.get(name, None)
            new_p, new_m = self._reference_sgd_update(
                param=param,
                grad=grad,
                m=m,
                step=step_tensor,
            )
            new_params[name] = new_p
            if new_m is not None:
                new_m_dict[name] = new_m
            new_steps_dict[name] = step_tensor + 1.0
        return SGD.ParamGroup(
            params=new_params,
            opt_state_m=new_m_dict,
            opt_steps=new_steps_dict,
            extra_state=param_group.extra_state,
        )


class FusedSgd(SGD):
    """Fused SGD implementation calling torch.optim.sgd.sgd directly with fused=True."""

    def step(
        self, param_group: SGD.ParamGroup, grads: dict[str, torch.Tensor]
    ) -> SGD.ParamGroup:
        """Fused SGD step calling torch.optim.sgd.sgd directly."""
        param_keys = [
            k for k in param_group.params.keys() if grads.get(k) is not None
        ]
        if not param_keys:
            return param_group
        param_list = [param_group.params[k] for k in param_keys]
        grad_list = [grads[k] for k in param_keys]
        m_list = (
            [param_group.opt_state_m[k] for k in param_keys]
            if self.momentum != 0.0
            else []
        )
        step_list = [param_group.opt_steps[k] for k in param_keys]

        torch_sgd.sgd(
            params=param_list,
            d_p_list=grad_list,
            momentum_buffer_list=m_list,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            lr=self.lr,
            dampening=self.dampening,
            nesterov=self.nesterov,
            maximize=False,
            foreach=False,
            fused=True,
        )
        torch.ops.aten._foreach_add_(step_list, 1.0)

        return param_group


class Adagrad(Optimizer):
    """Base class for functional Adagrad optimizers."""

    @dataclasses.dataclass
    class ParamGroup(Optimizer.ParamGroup):
        """Holds optimizer state for Adagrad."""

        opt_state_sum: dict[str, torch.Tensor] = dataclasses.field(
            default_factory=dict
        )
        opt_steps: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
        extra_state: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.1,
        eps: float = 1e-10,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.lr_decay = lr_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps

    def init_param_group(
        self, params: dict[str, torch.Tensor] | torch.nn.Module
    ) -> ParamGroup:
        """Initializes Adagrad optimizer state from model parameters or nn.Module."""
        if isinstance(params, torch.nn.Module):
            params = {
                name: param.detach() for name, param in params.named_parameters()
            }
        if not params:
            return self.ParamGroup(params={})

        opt_state_sum = {
            name: torch.full_like(
                param,
                self.initial_accumulator_value,
                dtype=param.dtype,
                memory_format=torch.preserve_format,
                device=param.device,
            )
            for name, param in params.items()
        }
        opt_steps = {
            name: torch.tensor(1.0, dtype=torch.float32, device=param.device)
            for name, param in params.items()
        }

        return self.ParamGroup(
            params=params,
            opt_state_sum=opt_state_sum,
            opt_steps=opt_steps,
        )

    def state_dict(self, param_group: ParamGroup) -> dict[str, Any]:
        """Serializes Adagrad state matching PyTorch/TorchRec optimizer format."""
        state = {}
        for name in param_group.params:
            s = {}
            if name in param_group.opt_steps:
                step = param_group.opt_steps[name]
                s["step"] = step.item() if isinstance(step, torch.Tensor) else step
            if name in param_group.opt_state_sum:
                s["sum"] = param_group.opt_state_sum[name]
            state[name] = s
        return {
            "state": state,
            "param_groups": [{
                "lr": self.lr,
                "lr_decay": self.lr_decay,
                "weight_decay": self.weight_decay,
                "initial_accumulator_value": self.initial_accumulator_value,
                "eps": self.eps,
                "params": list(param_group.params.keys()),
            }],
        }

    def load_state_dict(
        self, param_group: ParamGroup, state_dict: dict[str, Any]
    ) -> ParamGroup:
        """Loads state from a state_dict into Adagrad.ParamGroup."""
        if "param_groups" in state_dict and state_dict["param_groups"]:
            pg_conf = state_dict["param_groups"][0]
            if "lr" in pg_conf:
                self.lr = pg_conf["lr"]
            if "lr_decay" in pg_conf:
                self.lr_decay = pg_conf["lr_decay"]
            if "weight_decay" in pg_conf:
                self.weight_decay = pg_conf["weight_decay"]
            if "initial_accumulator_value" in pg_conf:
                self.initial_accumulator_value = pg_conf["initial_accumulator_value"]
            if "eps" in pg_conf:
                self.eps = pg_conf["eps"]

        new_sum = dict(param_group.opt_state_sum)
        new_steps = dict(param_group.opt_steps)

        saved_state = state_dict.get("state", {})
        param_keys = list(param_group.params.keys())
        for idx_or_name, s in saved_state.items():
            name = (
                idx_or_name
                if isinstance(idx_or_name, str)
                else (
                    param_keys[idx_or_name] if idx_or_name < len(param_keys) else None
                )
            )
            if name is None or name not in param_group.params:
                continue
            param = param_group.params[name]
            if "sum" in s:
                new_sum[name] = (
                    s["sum"]
                    if isinstance(s["sum"], torch.Tensor)
                    else torch.tensor(s["sum"], device=param.device)
                )
            if "step" in s:
                step_val = s["step"]
                new_steps[name] = (
                    step_val
                    if isinstance(step_val, torch.Tensor)
                    else torch.tensor(
                        float(step_val), dtype=torch.float32, device=param.device
                    )
                )

        return self.ParamGroup(
            params=param_group.params,
            opt_state_sum=new_sum,
            opt_steps=new_steps,
            extra_state=param_group.extra_state,
        )


class ReferenceAdagrad(Adagrad):
    """Reference pure PyTorch functional implementation of Adagrad."""

    def _reference_adagrad_update(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state_sum: torch.Tensor,
        step: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes reference Adagrad update returning (new_param, new_state_sum)."""
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * param

        if self.lr_decay != 0.0:
            step_num = step if isinstance(step, (int, float)) else step
            clr = self.lr / (1.0 + (step_num - 1.0) * self.lr_decay)
        else:
            clr = self.lr

        new_state_sum = state_sum + grad * grad
        std = torch.sqrt(new_state_sum) + self.eps
        new_param = param - clr * (grad / std)
        return new_param, new_state_sum

    def step(
        self, param_group: Adagrad.ParamGroup, grads: dict[str, torch.Tensor]
    ) -> Adagrad.ParamGroup:
        """Reference pure PyTorch implementation of Adagrad step."""
        new_params = {}
        new_sum_dict = {}
        new_steps_dict = {}
        for name, param in param_group.params.items():
            grad = grads.get(name)
            if grad is None:
                new_params[name] = param
                if name in param_group.opt_state_sum:
                    new_sum_dict[name] = param_group.opt_state_sum[name]
                if name in param_group.opt_steps:
                    new_steps_dict[name] = param_group.opt_steps[name]
                continue
            step_tensor = param_group.opt_steps[name]
            state_sum = param_group.opt_state_sum[name]
            new_p, new_sum = self._reference_adagrad_update(
                param=param,
                grad=grad,
                state_sum=state_sum,
                step=step_tensor,
            )
            new_params[name] = new_p
            new_sum_dict[name] = new_sum
            new_steps_dict[name] = step_tensor + 1.0
        return Adagrad.ParamGroup(
            params=new_params,
            opt_state_sum=new_sum_dict,
            opt_steps=new_steps_dict,
            extra_state=param_group.extra_state,
        )


# Register PyTree handlers for ParamGroup classes
def _adam_param_group_flatten(group: AdamW.ParamGroup):
    children = (
        group.params,
        group.opt_state_m,
        group.opt_state_v,
        group.opt_steps,
        group.extra_state,
    )
    return children, None


def _adam_param_group_unflatten(children, _context):
    return AdamW.ParamGroup(
        params=children[0],
        opt_state_m=children[1],
        opt_state_v=children[2],
        opt_steps=children[3],
        extra_state=children[4],
    )


_pytree.register_pytree_node(
    AdamW.ParamGroup,
    _adam_param_group_flatten,
    _adam_param_group_unflatten,
)


def _sgd_param_group_flatten(group: SGD.ParamGroup):
    children = (
        group.params,
        group.opt_state_m,
        group.opt_steps,
        group.extra_state,
    )
    return children, None


def _sgd_param_group_unflatten(children, _context):
    return SGD.ParamGroup(
        params=children[0],
        opt_state_m=children[1],
        opt_steps=children[2],
        extra_state=children[3],
    )


_pytree.register_pytree_node(
    SGD.ParamGroup,
    _sgd_param_group_flatten,
    _sgd_param_group_unflatten,
)


def _adagrad_param_group_flatten(group: Adagrad.ParamGroup):
    children = (
        group.params,
        group.opt_state_sum,
        group.opt_steps,
        group.extra_state,
    )
    return children, None


def _adagrad_param_group_unflatten(children, _context):
    return Adagrad.ParamGroup(
        params=children[0],
        opt_state_sum=children[1],
        opt_steps=children[2],
        extra_state=children[3],
    )


_pytree.register_pytree_node(
    Adagrad.ParamGroup,
    _adagrad_param_group_flatten,
    _adagrad_param_group_unflatten,
)


class KeyedStatelessOptimizer(KeyedOptimizer):
    """TorchRec KeyedOptimizer adapter wrapping a stateless functional optimizer.

    Provides seamless drop-in compatibility for TorchRec workflows (including
    TorchRec's CombinedOptimizer) while utilizing stateless functional optimizer
    implementations suitable for full-graph TPU compilation.
    """

    def __init__(
        self,
        params: Mapping[str, torch.Tensor] | torch.nn.Module,
        optimizer: Optimizer,
    ):
        self._functional_opt = optimizer
        if isinstance(params, torch.nn.Module):
            params_dict = {name: param for name, param in params.named_parameters()}
        else:
            params_dict = dict(params)
        self._param_group = self._functional_opt.init_param_group(params_dict)

        sd = self._functional_opt.state_dict(self._param_group)
        state_by_tensor = {
            param: sd["state"].get(name, {}) for name, param in params_dict.items()
        }
        param_groups_by_tensor = [{
            "params": list(params_dict.values()),
            **{k: v for k, v in sd["param_groups"][0].items() if k != "params"},
        }]
        super().__init__(
            params=params_dict,
            state=state_by_tensor,
            param_groups=param_groups_by_tensor,
        )

    @property
    def functional_optimizer(self) -> Optimizer:
        return self._functional_opt

    @property
    def param_group(self) -> Any:
        return self._param_group

    @param_group.setter
    def param_group(self, pg: Any) -> None:
        self._param_group = pg

    def step(
        self,
        closure: Any = None,
        grads: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Any:
        """Performs an optimization step.

        Args:
          closure: Optional evaluation closure for standard optimizer interface.
          grads: Optional dictionary of parameter gradients. If provided,
            functionally steps and updates the internal ParamGroup. If omitted,
            extracts .grad from registered parameters. In both cases, copies updated
            parameter values into underlying registered parameters.

        Returns:
          Updated ParamGroup.
        """
        if closure is not None:
            closure()

        if grads is not None:
            grads_dict = dict(grads)
        else:
            grads_dict = {
                name: param.grad
                for name, param in self._param_group.params.items()
                if getattr(param, "grad", None) is not None
            }

        if grads_dict:
            self._param_group = self._functional_opt.step(
                self._param_group, grads_dict
            )
            with torch.no_grad():
                for name, new_p in self._param_group.params.items():
                    if name in self.params:
                        self.params[name].copy_(new_p)

        return self._param_group

    def state_dict(self) -> dict[str, Any]:
        return self._functional_opt.state_dict(self._param_group)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._param_group = self._functional_opt.load_state_dict(
            self._param_group, dict(state_dict)
        )
        sd = self._functional_opt.state_dict(self._param_group)
        self.state = {
            param: sd["state"].get(name, {}) for name, param in self.params.items()
        }
        self.param_groups = [{
            "params": list(self.params.values()),
            **{k: v for k, v in sd["param_groups"][0].items() if k != "params"},
        }]
