#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
    OptimizerWrapper,
)


class TestKeyedZeroGrad(unittest.TestCase):
    def test_zero_grad_preserves_requested_update_semantics(self) -> None:
        for wrapper in ("keyed", "combined", "nested"):
            for set_to_none in (False, True, None):
                with self.subTest(wrapper=wrapper, set_to_none=set_to_none):
                    param = torch.nn.Parameter(torch.tensor([1.0]))
                    reference = torch.nn.Parameter(param.detach().clone())
                    optimizer: KeyedOptimizer = KeyedOptimizerWrapper(
                        {"weight": param},
                        lambda params: torch.optim.SGD(
                            params, lr=0.1, weight_decay=0.1
                        ),
                    )
                    if wrapper != "keyed":
                        optimizer = CombinedOptimizer([("dense", optimizer)])
                    if wrapper == "nested":
                        optimizer = OptimizerWrapper(optimizer)
                    reference_optimizer = torch.optim.SGD(
                        [reference], lr=0.1, weight_decay=0.1
                    )
                    param.grad = torch.ones_like(param)
                    reference.grad = torch.ones_like(reference)
                    if set_to_none is None:
                        optimizer.zero_grad()
                    else:
                        optimizer.zero_grad(set_to_none=set_to_none)
                    reference_optimizer.zero_grad(set_to_none=set_to_none is True)
                    self.assertEqual(param.grad is None, set_to_none is True)
                    if param.grad is not None:
                        torch.testing.assert_close(param.grad, torch.zeros_like(param))
                    optimizer.step()
                    reference_optimizer.step()
                    torch.testing.assert_close(param, reference)


if __name__ == "__main__":
    unittest.main()
