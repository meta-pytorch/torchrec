#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Tuple

import torch
import torchrec


def _ftrl_reference_step(
    weight: torch.Tensor,
    accum: torch.Tensor,
    linear: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
    learning_rate_power: float,
    l1_reg: float,
    l2_reg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent transcription of x-deeplearning's FtrlUpdater inner loop.

    Deliberately written as a scalar loop rather than reusing the optimizer's
    own vectorized code, so the test checks the algorithm and not just that the
    implementation equals itself.
    """
    weight_new = weight.clone()
    accum_new = accum.clone()
    linear_new = linear.clone()
    exponent = -learning_rate_power

    flat_w = weight_new.view(-1)
    flat_n = accum_new.view(-1)
    flat_z = linear_new.view(-1)
    flat_g = grad.reshape(-1)

    for i in range(flat_w.numel()):
        g = float(flat_g[i])
        n_old = float(flat_n[i])
        n_new = n_old + g * g
        sigma_new = n_new**exponent
        sigma_old = n_old**exponent
        z = float(flat_z[i]) + g - (sigma_new - sigma_old) / lr * float(flat_w[i])
        quadratic = sigma_new / lr + 2.0 * l2_reg
        if abs(z) > l1_reg:
            sgn = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
            flat_w[i] = (l1_reg * sgn - z) / quadratic
        else:
            flat_w[i] = 0.0
        flat_z[i] = z
        flat_n[i] = n_new

    return weight_new, accum_new, linear_new


class FTRLTest(unittest.TestCase):
    def test_optim(self) -> None:
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        # pyrefly: ignore[implicit-import]
        opt = torchrec.optim.FTRL(embedding_bag.parameters(), lr=0.1)
        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])
        embedding_bag_out = embedding_bag(index, offsets)
        opt.zero_grad()
        embedding_bag_out.sum().backward()
        opt.step()

    def test_matches_reference(self) -> None:
        """Multi-step agreement with a scalar transcription of the XDL kernel."""
        lr, power, l1, l2 = 0.5, -0.5, 1e-3, 1e-2
        torch.manual_seed(0)
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        # pyrefly: ignore[implicit-import]
        opt = torchrec.optim.FTRL(
            embedding_bag.parameters(),
            lr=lr,
            ftrl_learning_rate_power=power,
            ftrl_l1_reg=l1,
            ftrl_l2_reg=l2,
        )

        weight = embedding_bag.weight.detach().clone()
        accum = torch.zeros_like(weight)
        linear = torch.zeros_like(weight)

        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])
        for step in range(5):
            opt.zero_grad()
            (embedding_bag(index, offsets).sum() * (step + 1)).backward()
            grad = embedding_bag.weight.grad.detach().clone()
            opt.step()

            weight, accum, linear = _ftrl_reference_step(
                weight, accum, linear, grad, lr, power, l1, l2
            )
            torch.testing.assert_close(
                embedding_bag.weight.detach(),
                weight,
                atol=1e-6,
                rtol=1e-6,
                msg=f"weight mismatch at step {step}",
            )
            state = opt.state[embedding_bag.weight]
            torch.testing.assert_close(
                state["accum"], accum, atol=1e-6, rtol=1e-6, msg=f"accum @ {step}"
            )
            torch.testing.assert_close(
                state["linear"], linear, atol=1e-6, rtol=1e-6, msg=f"linear @ {step}"
            )

    def test_l1_pins_to_exact_zero(self) -> None:
        """A dominating L1 term must produce exactly 0.0, which is the point of
        using FTRL on embedding tables."""
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        # pyrefly: ignore[implicit-import]
        opt = torchrec.optim.FTRL(embedding_bag.parameters(), lr=0.1, ftrl_l1_reg=100.0)
        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])
        opt.zero_grad()
        embedding_bag(index, offsets).sum().backward()
        opt.step()

        self.assertTrue(
            torch.equal(embedding_bag.weight.detach(), torch.zeros(4, 4)),
            msg=f"expected all-zero weights, got {embedding_bag.weight}",
        )

    def test_zero_grad_is_idempotent(self) -> None:
        """FTRL recomputes the weight from (accum, linear); with g == 0 neither
        state moves, so the weight must not move either."""
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        # pyrefly: ignore[implicit-import]
        opt = torchrec.optim.FTRL(
            embedding_bag.parameters(), lr=0.5, ftrl_l1_reg=1e-3, ftrl_l2_reg=1e-3
        )
        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])

        opt.zero_grad()
        embedding_bag(index, offsets).sum().backward()
        opt.step()
        weight_before = embedding_bag.weight.detach().clone()

        opt.zero_grad()
        (embedding_bag(index, offsets).sum() * 0.0).backward()
        opt.step()

        torch.testing.assert_close(
            embedding_bag.weight.detach(), weight_before, atol=1e-7, rtol=1e-7
        )

    def test_l2_enters_denominator(self) -> None:
        """l2_reg scales the result by quad(0)/quad(l2) rather than acting as a
        gradient-space decay."""
        lr, l2 = 0.5, 0.25
        state_dict = {"weight": torch.zeros(4, 4)}

        bags = []
        for l2_reg in (0.0, l2):
            bag = torch.nn.EmbeddingBag(num_embeddings=4, embedding_dim=4, mode="sum")
            bag.load_state_dict(state_dict)
            # pyrefly: ignore[implicit-import]
            opt = torchrec.optim.FTRL(bag.parameters(), lr=lr, ftrl_l2_reg=l2_reg)
            index, offsets = torch.tensor([0]), torch.tensor([0, 1])
            opt.zero_grad()
            (bag(index, offsets).sum() * 0.8).backward()
            opt.step()
            bags.append(bag.weight.detach().clone())

        # First step from accum == 0 gives sigma_new == |g| == 0.8.
        sigma_new = 0.8
        ratio = (sigma_new / lr) / (sigma_new / lr + 2.0 * l2)
        torch.testing.assert_close(bags[1][0], bags[0][0] * ratio, atol=1e-6, rtol=1e-6)

    def test_invalid_args(self) -> None:
        params = torch.nn.EmbeddingBag(4, 4).parameters
        with self.assertRaises(ValueError):
            # pyrefly: ignore[implicit-import]
            torchrec.optim.FTRL(params(), lr=0.0)
        with self.assertRaises(ValueError):
            # pyrefly: ignore[implicit-import]
            torchrec.optim.FTRL(params(), lr=0.1, ftrl_l1_reg=-1.0)
        with self.assertRaises(ValueError):
            # pyrefly: ignore[implicit-import]
            torchrec.optim.FTRL(params(), lr=0.1, ftrl_l2_reg=-1.0)


if __name__ == "__main__":
    unittest.main()
