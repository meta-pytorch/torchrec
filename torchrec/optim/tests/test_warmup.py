#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import defaultdict
from typing import Any

import torch
from torch.autograd import Variable
from torchrec.optim.keyed import KeyedOptimizer
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage


class DummyKeyedOptimizer(KeyedOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.


class TestWarmupOptimizer(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        def get_optimizer() -> WarmupOptimizer:
            param_1_t = torch.tensor([1.0, 2.0])
            param_1 = Variable(param_1_t)
            keyed_optimizer = DummyKeyedOptimizer(
                {"param_1": param_1}, defaultdict(dict), [{"params": [param_1]}]
            )
            warmup_optimizer = WarmupOptimizer(
                keyed_optimizer,
                stages=[
                    WarmupStage(
                        WarmupPolicy.LINEAR, max_iters=100, value=1e-2, lr_scale=1
                    ),
                ],
            )
            warmup_optimizer.save_param_groups(True)
            return warmup_optimizer

        warmup_optimizer_1 = get_optimizer()
        num_iters = 10
        for _ in range(num_iters):
            warmup_optimizer_1.zero_grad()
            warmup_optimizer_1.step()

        param_state = list(warmup_optimizer_1.state.values())[0]
        self.assertEqual(
            param_state["warmup"].tolist()[0],
            num_iters,
        )

        warmup_optimizer_2 = get_optimizer()
        warmup_optimizer_2.step()
        warmup_optimizer_2.zero_grad()

        warmup_optimizer_2.save_param_groups(True)
        warmup_optimizer_2.load_state_dict(warmup_optimizer_1.state_dict())

        self.assertEqual(
            warmup_optimizer_1.state_dict()["param_groups"],
            warmup_optimizer_2.state_dict()["param_groups"],
        )
        torch.testing.assert_close(
            warmup_optimizer_1.state_dict()["state"]["__warmup"],
            warmup_optimizer_2.state_dict()["state"]["__warmup"],
        )
