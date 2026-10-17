#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import defaultdict
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import torch
from torch.autograd import Variable
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
)
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage


class DummyKeyedOptimizer(KeyedOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # pyrefly: ignore[bad-override]
    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.


class TestWarmupOptimizer(unittest.TestCase):
    def test_legacy_nested_checkpoint_resume(self) -> None:
        def make(
            optimizer_type: type[torch.optim.SGD] | type[torch.optim.AdamW],
        ) -> tuple[torch.Tensor, CombinedOptimizer, KeyedOptimizerWrapper]:
            p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
            kwargs = {"lr": 0.03, "weight_decay": 0.2}
            if optimizer_type is torch.optim.SGD:
                kwargs["momentum"] = 0.9
            keyed = KeyedOptimizerWrapper(
                {"weight": p}, lambda ps: optimizer_type(ps, **kwargs)
            )
            warmup = WarmupOptimizer(
                keyed,
                [WarmupStage(WarmupPolicy.LINEAR, max_iters=6, value=0.1)],
                lr=0.03,
            )
            return p, CombinedOptimizer([warmup]), keyed

        for optimizer_type in (torch.optim.SGD, torch.optim.AdamW):
            with self.subTest(optimizer=optimizer_type):
                p, source, _ = make(optimizer_type)
                for step in range(4):
                    p.grad = torch.tensor([0.2 * (step + 1), -0.3])
                    source.step()
                saved = deepcopy(source.state_dict())
                self.assertEqual(set(saved), {"state"})
                q, target, keyed = make(optimizer_type)
                target.init_state()
                with torch.no_grad():
                    q.copy_(p)
                with patch.object(keyed, "post_load_state_dict") as child_hook:
                    target.load_state_dict(saved)
                    child_hook.assert_not_called()
                for step in range(4, 9):
                    p.grad = torch.tensor([0.2 * (step + 1), -0.3])
                    q.grad = p.grad.clone()
                    source.step()
                    target.step()
                    torch.testing.assert_close(q, p, rtol=0, atol=0)
                    torch.testing.assert_close(
                        target.state_dict(), source.state_dict(), rtol=0, atol=0
                    )
                    self.assertEqual(
                        [group["lr"] for group in target.param_groups],
                        [group["lr"] for group in source.param_groups],
                    )

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

    def test_interpolate_policy(self) -> None:
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, defaultdict(dict), [{"params": [param_1], "lr": 1.0}]
        )
        warmup_optimizer = WarmupOptimizer(
            keyed_optimizer,
            stages=[
                WarmupStage(
                    WarmupPolicy.INTERPOLATE,
                    max_iters=200,
                    start_interpolating_iters=100,
                    value=0.01,
                    end_value=0.1,
                    lr_scale=1,
                ),
            ],
            lr=1.0,
        )

        # At iter 0 (before start_interpolating_iters=100), the formula gives:
        # multiplier = 0.01 + (0.1 - 0.01) * (0 - 100) / (200 - 100) = 0.01 - 0.09 = -0.08
        # But typically INTERPOLATE is used starting from start_interpolating_iters
        # Let's step to iteration 100 and verify
        for _ in range(100):
            warmup_optimizer.zero_grad()
            warmup_optimizer.step()

        # At iter 100: multiplier = 0.01 + (0.1 - 0.01) * (100 - 100) / (200 - 100) = 0.01
        lr_at_100 = list(warmup_optimizer.param_groups)[0]["lr"]
        self.assertAlmostEqual(lr_at_100, 0.01, places=5)

        # Step 50 more times to iter 150
        for _ in range(50):
            warmup_optimizer.zero_grad()
            warmup_optimizer.step()

        # At iter 150: multiplier = 0.01 + (0.1 - 0.01) * (150 - 100) / (200 - 100) = 0.01 + 0.045 = 0.055
        lr_at_150 = list(warmup_optimizer.param_groups)[0]["lr"]
        self.assertAlmostEqual(lr_at_150, 0.055, places=5)

        # Step 50 more times to iter 200
        for _ in range(50):
            warmup_optimizer.zero_grad()
            warmup_optimizer.step()

        # At iter 200: multiplier = 0.01 + (0.1 - 0.01) * (200 - 100) / (200 - 100) = 0.1
        lr_at_200 = list(warmup_optimizer.param_groups)[0]["lr"]
        self.assertAlmostEqual(lr_at_200, 0.1, places=5)
