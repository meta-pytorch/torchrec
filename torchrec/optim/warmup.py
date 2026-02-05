#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import math
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, List, Tuple

import torch
from torchrec.optim.keyed import KeyedOptimizer, OptimizerWrapper

logger: logging.Logger = logging.getLogger(__name__)


@unique
class WarmupPolicy(Enum):
    NONE = "none"
    LINEAR = "linear"
    CONSTANT = "constant"
    POLY = "poly"
    STEP = "step"
    INVSQRT = "inv_sqrt"  # inverse square root
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"


@dataclass
class WarmupStage:
    policy: WarmupPolicy = WarmupPolicy.LINEAR
    max_iters: int = 1
    value: float = 1.0
    lr_scale: float = 1.0
    # used as number denominator for iters in poly decay
    # default to max_iters if not set to value > 0
    # also used as stepsize in step decay
    # default to 1 if not set to value > 0
    decay_iters: int = -1
    sgdr_period: int = 1


def _lr_stages(stages: List[WarmupStage]) -> List[WarmupStage]:
    last_stage = WarmupStage(policy=WarmupPolicy.NONE, max_iters=1 << 63, value=1.0)
    if len(stages) == 0:
        return [last_stage]

    start_iter = 0
    for stage in stages:
        assert stage.max_iters > start_iter, (
            f"Max iter of the stage {stage} must be greater than the previous "
            f"max iter {start_iter}"
        )
        start_iter = stage.max_iters
        if stage.decay_iters <= 0:
            if stage.policy == WarmupPolicy.STEP:
                stage.decay_iters = 1
            else:
                stage.decay_iters = stage.max_iters
    return stages + [last_stage]


def _get_multiplier(stage: WarmupStage, iter: int) -> float:
    multiplier = 1.0
    if stage.policy == WarmupPolicy.LINEAR:
        multiplier = stage.value + (1.0 - stage.value) * iter / stage.max_iters
    elif stage.policy == WarmupPolicy.CONSTANT:
        multiplier = stage.value
    elif stage.policy == WarmupPolicy.POLY:
        multiplier = math.pow(1 - iter / stage.decay_iters, stage.value)
    elif stage.policy == WarmupPolicy.STEP:
        multiplier = math.pow(stage.value, iter // stage.decay_iters)
    elif stage.policy == WarmupPolicy.INVSQRT:
        multiplier = 1.0 / math.sqrt(iter)
    elif stage.policy == WarmupPolicy.COSINE_ANNEALING_WARM_RESTARTS:
        # SGDR: Stochastic Gradient Descent with Warm Restarts:
        # https://arxiv.org/abs/1608.03983.
        # Forgo period multiplier T_mult, as lr multiplier is a stateless
        # computation without knowledge of previous period size.
        eta_min = stage.value
        t_0 = stage.sgdr_period
        t_cur = iter % t_0
        cos_iter = 0.5 * (1 + math.cos(math.pi * t_cur / t_0))
        multiplier = eta_min + (1.0 - eta_min) * cos_iter
    return multiplier * stage.lr_scale


class WarmupOptimizer(OptimizerWrapper):
    """
    Adjusts learning rate according to the schedule.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap
        stages (List[WarmupStage]): stages to go through
        lr (float): initial learning rate
        lr_param (str): learning rate parameter in parameter group.
        param_name: Name of fake parameter to hold warmup state.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        stages: List[WarmupStage],
        lr: float = 0.1,
        lr_param: str = "lr",
        param_name: str = "__warmup",
    ) -> None:
        super().__init__(optimizer)
        self._stages: List[WarmupStage] = _lr_stages(stages)
        self._lr_param: str = lr_param
        self._lr: float = lr
        self._warmup_param: torch.nn.Parameter = torch.nn.Parameter()
        # pyre-ignore [16]
        self.params[param_name] = self._warmup_param
        # for fused optimizer we will do first backward() pass before calling step()
        self._set_lr(0, 0)

    def _set_lr(self, iter_: int, stage_id: int) -> None:
        lr = self._lr * _get_multiplier(self._stages[stage_id], iter_)
        for param_group in self.param_groups:
            # pyre-ignore [16]
            param_group[self._lr_param] = lr

    def _get_warmup_state(self) -> Tuple[int, int]:
        if self._warmup_param in self.state:
            iter_, stage_id = self.state[self._warmup_param]["warmup"].tolist()
        else:
            iter_ = 0
            stage_id = 0
        return iter_, stage_id

    def post_load_state_dict(self) -> None:
        iter_, stage_id = self._get_warmup_state()
        logger.info(f"Warmup Optimizer set to iteration {iter_}")
        self._set_lr(iter_, stage_id)

    def step(self, closure: Any = None) -> None:
        super().step(closure)
        iter_, stage_id = self._get_warmup_state()

        iter_ += 1
        if iter_ > self._stages[stage_id].max_iters and stage_id + 1 < len(
            self._stages
        ):
            stage_id += 1
            logger.info(
                "Warmup Optimizer finishing "
                f"{self._stages[stage_id - 1]} "
                "switching to "
                f"{self._stages[stage_id]}"
            )
        self._set_lr(iter_, stage_id)

        # pyre-ignore [16]
        self.state[self._warmup_param] = {
            "warmup": torch.tensor([iter_, stage_id], dtype=torch.long)
        }
