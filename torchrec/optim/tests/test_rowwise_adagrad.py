#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import torch
import torchrec


class RowWiseAdagradTest(unittest.TestCase):
    def test_optim(self) -> None:
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        # pyrefly: ignore[implicit-import]
        opt = torchrec.optim.RowWiseAdagrad(embedding_bag.parameters())
        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])
        embedding_bag_out = embedding_bag(index, offsets)
        opt.zero_grad()
        embedding_bag_out.sum().backward()
        opt.step()

    def test_optim_equivalence(self) -> None:
        # If rows are initialized to be the same and uniform, then RowWiseAdagrad and canonical Adagrad are identical
        rowwise_embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=4, embedding_dim=4, mode="sum"
        )
        state_dict = {
            "weight": torch.Tensor(
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
            )
        }
        rowwise_embedding_bag.load_state_dict(state_dict)
        embedding_bag.load_state_dict(state_dict)

        # pyrefly: ignore[implicit-import]
        row_wise_opt = torchrec.optim.RowWiseAdagrad(rowwise_embedding_bag.parameters())
        opt = torch.optim.Adagrad(embedding_bag.parameters())

        index, offsets = torch.tensor([0, 3]), torch.tensor([0, 1])

        for _ in range(5):
            row_wise_opt.zero_grad()
            opt.zero_grad()

            embedding_bag(index, offsets).sum().backward()
            opt.step()

            rowwise_embedding_bag(index, offsets).sum().backward()
            row_wise_opt.step()

            torch.testing.assert_close(
                embedding_bag.state_dict()["weight"],
                rowwise_embedding_bag.state_dict()["weight"],
            )
