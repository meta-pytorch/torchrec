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

    # pyre-ignore[2]
    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.


class TestGetMultiplier(unittest.TestCase):
    """Tests for the _get_multiplier function with TRANSFORMER policy."""

    def test_transformer_warmup_at_step_one(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=4000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=4000,
            lr_scale=1.0,
            warmup_steps=4000,
        )

        # Execute: Get multiplier at iteration 0 (step 1 internally)
        from torchrec.optim.warmup import _get_multiplier

        multiplier = _get_multiplier(stage, iter=0)

        # Assert: At step 1, multiplier should be min(1, 1/4000^1.5) ≈ 0.0000158
        # step^(-0.5) = 1^(-0.5) = 1.0
        # step * warm_steps^(-1.5) = 1 * 4000^(-1.5) ≈ 0.0000158
        expected = min(1.0, 1 * (4000 ** (-1.5)))
        self.assertAlmostEqual(multiplier, expected, places=8)
        self.assertLess(multiplier, 0.00002)

    def test_transformer_warmup_at_warmup_steps(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=4000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=4000,
            lr_scale=1.0,
            warmup_steps=4000,
        )

        # Execute: Get multiplier at iteration 3999 (step 4000 internally)
        from torchrec.optim.warmup import _get_multiplier

        multiplier = _get_multiplier(stage, iter=3999)

        # Assert: At step=warm_steps, both terms are equal
        # step^(-0.5) = 4000^(-0.5) ≈ 0.0158
        # step * warm_steps^(-1.5) = 4000 * 4000^(-1.5) ≈ 0.0158
        step = 4000
        expected = min(step ** (-0.5), step * (4000 ** (-1.5)))
        self.assertAlmostEqual(multiplier, expected, places=8)
        self.assertAlmostEqual(multiplier, 0.0158114, places=6)

    def test_transformer_warmup_after_warmup_steps(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=4000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=4000,
            lr_scale=1.0,
            warmup_steps=4000,
        )

        # Execute: Get multiplier at iteration 7999 (step 8000 internally)
        from torchrec.optim.warmup import _get_multiplier

        multiplier = _get_multiplier(stage, iter=7999)

        # Assert: After warmup, step^(-0.5) dominates (is smaller)
        # step^(-0.5) = 8000^(-0.5) ≈ 0.0112
        # step * warm_steps^(-1.5) = 8000 * 4000^(-1.5) ≈ 0.0316
        step = 8000
        inv_sqrt = step ** (-0.5)
        warmup_term = step * (4000 ** (-1.5))
        self.assertAlmostEqual(multiplier, inv_sqrt, places=8)
        self.assertLess(inv_sqrt, warmup_term)
        self.assertAlmostEqual(multiplier, 0.0111803, places=6)

    def test_transformer_warmup_with_lr_scale(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with lr_scale=2.0
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=4000,
            lr_scale=2.0,
            warmup_steps=4000,
        )

        # Execute: Get multiplier at iteration 3999 (step 4000 internally)
        from torchrec.optim.warmup import _get_multiplier

        multiplier = _get_multiplier(stage, iter=3999)

        # Assert: lr_scale is applied as a multiplier
        step = 4000
        base_multiplier = min(step ** (-0.5), step * (4000 ** (-1.5)))
        expected = base_multiplier * 2.0
        self.assertAlmostEqual(multiplier, expected, places=8)

    def test_transformer_warmup_formula_correctness(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=1000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=1000,
            lr_scale=1.0,
            warmup_steps=1000,
        )

        # Execute: Test multiple iterations to verify formula
        from torchrec.optim.warmup import _get_multiplier

        test_iters = [0, 99, 499, 999, 1999]  # steps 1, 100, 500, 1000, 2000
        for iter_val in test_iters:
            multiplier = _get_multiplier(stage, iter=iter_val)
            step = iter_val + 1

            # Assert: Multiplier matches the Transformer formula
            expected = min(step ** (-0.5), step * (1000 ** (-1.5)))
            self.assertAlmostEqual(
                multiplier,
                expected,
                places=8,
                msg=f"Failed at iteration {iter_val} (step {step})",
            )

    def test_transformer_warmup_monotonic_increase_during_warmup(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=1000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=1000,
            lr_scale=1.0,
            warmup_steps=1000,
        )

        # Execute: Get multipliers during warmup phase
        from torchrec.optim.warmup import _get_multiplier

        multipliers = [_get_multiplier(stage, iter=i) for i in range(0, 1000)]

        # Assert: Multipliers should increase monotonically during warmup
        for idx in range(len(multipliers) - 1):
            self.assertLess(
                multipliers[idx],
                multipliers[idx + 1],
                msg=f"Multiplier should increase at iteration {idx}",
            )

    def test_transformer_warmup_monotonic_decrease_after_warmup(self) -> None:
        # Setup: Create TRANSFORMER warmup stage with warm_steps=1000
        stage = WarmupStage(
            policy=WarmupPolicy.TRANSFORMER,
            max_iters=1000,
            lr_scale=1.0,
            warmup_steps=1000,
        )

        # Execute: Get multipliers after warmup phase
        from torchrec.optim.warmup import _get_multiplier

        multipliers = [_get_multiplier(stage, iter=i) for i in range(1000, 2000)]

        # Assert: Multipliers should decrease monotonically after warmup
        for i in range(len(multipliers) - 1):
            self.assertGreater(
                multipliers[i],
                multipliers[i + 1],
                msg=f"Multiplier should decrease at iteration {i + 1000}",
            )


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

    def test_transformer_warmup_integration(self) -> None:
        # Setup: Create optimizer with TRANSFORMER warmup policy
        param = Variable(torch.tensor([1.0, 2.0]))
        keyed_optimizer = DummyKeyedOptimizer(
            {"param": param}, defaultdict(dict), [{"params": [param]}]
        )

        base_lr = 0.001
        warm_steps = 100

        warmup_optimizer = WarmupOptimizer(
            keyed_optimizer,
            stages=[
                WarmupStage(
                    policy=WarmupPolicy.TRANSFORMER,
                    max_iters=100,  # Stage ends at iteration 100
                    lr_scale=1.0,
                    warmup_steps=100,
                ),
            ],
            lr=base_lr,
        )

        # Execute: Run optimizer through warmup steps
        learning_rates = []
        current_lr = 0.0
        for _ in range(100):  # Only iterate through the TRANSFORMER stage
            for param_group in warmup_optimizer.param_groups:
                current_lr = param_group["lr"]
            learning_rates.append(current_lr)
            warmup_optimizer.step()

        # Assert: Verify learning rate follows Transformer schedule during warmup
        # At step 1 (iteration 0)
        step_1 = 1
        expected_lr_1 = base_lr * min(step_1 ** (-0.5), step_1 * (warm_steps ** (-1.5)))
        self.assertAlmostEqual(learning_rates[0], expected_lr_1, places=10)

        # At step 50 (iteration 49) - mid-warmup
        step_50 = 50
        expected_lr_50 = base_lr * min(
            step_50 ** (-0.5), step_50 * (warm_steps ** (-1.5))
        )
        self.assertAlmostEqual(learning_rates[49], expected_lr_50, places=10)

        # At step 100 (iteration 99) - warmup completion
        step_100 = 100
        expected_lr_100 = base_lr * min(
            step_100 ** (-0.5), step_100 * (warm_steps ** (-1.5))
        )
        self.assertAlmostEqual(learning_rates[99], expected_lr_100, places=10)

        # Verify learning rate increases monotonically during warmup
        for idx in range(warm_steps - 1):
            self.assertLess(
                learning_rates[idx],
                learning_rates[idx + 1],
                msg=f"LR should increase during warmup at step {idx + 1}",
            )
            # Verify formula correctness at this step
            step = idx + 1
            expected_lr_at_idx = base_lr * min(
                step ** (-0.5), step * (warm_steps ** (-1.5))
            )
            self.assertAlmostEqual(
                learning_rates[idx],
                expected_lr_at_idx,
                places=10,
                msg=f"LR mismatch at step {step}",
            )

    def test_transformer_warmup_with_extended_stage(self) -> None:
        # Setup: Create optimizer with TRANSFORMER stage to test warmup and decay
        param = Variable(torch.tensor([1.0, 2.0]))
        keyed_optimizer = DummyKeyedOptimizer(
            {"param": param}, defaultdict(dict), [{"params": [param]}]
        )

        base_lr = 0.001
        # In the TRANSFORMER policy, max_iters acts as warm_steps in the formula
        max_iters = 8000  # Stage runs for 8000 iterations

        warmup_optimizer = WarmupOptimizer(
            keyed_optimizer,
            stages=[
                WarmupStage(
                    policy=WarmupPolicy.TRANSFORMER,
                    max_iters=max_iters,  # Stage runs for 8000 iterations
                    lr_scale=1.0,
                    warmup_steps=max_iters,
                ),
            ],
            lr=base_lr,
        )

        # Execute: Run optimizer through warmup and decay phases
        current_lr = 0.0
        learning_rates = []
        for _ in range(max_iters):
            for param_group in warmup_optimizer.param_groups:
                current_lr = param_group["lr"]
            learning_rates.append(current_lr)
            warmup_optimizer.step()

        # Assert: Verify the formula uses max_iters as warm_steps
        # At step 1, verify the formula: min(step^(-0.5), step * max_iters^(-1.5))
        step_1 = 1
        expected_lr_1 = base_lr * min(step_1 ** (-0.5), step_1 * (max_iters ** (-1.5)))
        self.assertAlmostEqual(
            learning_rates[0],
            expected_lr_1,
            places=10,
            msg=f"LR at step 1 should match formula with warm_steps={max_iters}",
        )

        # At step 4000, verify with max_iters=8000
        step_4000 = 4000
        expected_lr_4000 = base_lr * min(
            step_4000 ** (-0.5), step_4000 * (max_iters ** (-1.5))
        )
        self.assertAlmostEqual(
            learning_rates[3999],
            expected_lr_4000,
            places=10,
            msg=f"LR at step 4000 should match formula with warm_steps={max_iters}",
        )

        # At step max_iters (8000), both terms should be equal
        step_max = max_iters
        inv_sqrt = step_max ** (-0.5)
        warmup_term = step_max * (max_iters ** (-1.5))
        self.assertAlmostEqual(
            inv_sqrt,
            warmup_term,
            places=10,
            msg=f"At step={max_iters}, both formula terms should be equal",
        )

        expected_lr_max = base_lr * min(inv_sqrt, warmup_term)
        self.assertAlmostEqual(
            learning_rates[max_iters - 1],
            expected_lr_max,
            places=10,
            msg=f"LR at step {max_iters} should match formula",
        )

        # Verify learning rate increases before max_iters
        for idx in range(max_iters - 1):
            self.assertLess(
                learning_rates[idx],
                learning_rates[idx + 1],
                msg=f"LR should increase at step {idx + 1} (before max_iters={max_iters})",
            )
