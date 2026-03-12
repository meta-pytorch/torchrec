#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.optimizers import in_backward_optimizer_filter


class TestInBackwardOptimizerFilter(unittest.TestCase):
    def test_in_backward_optimizer_filter(self) -> None:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
                ),
            ]
        )
        apply_optimizer_in_backward(
            torch.optim.SGD,
            ebc.embedding_bags["t1"].parameters(),
            optimizer_kwargs={"lr": 1.0},
        )
        in_backward_params = dict(
            in_backward_optimizer_filter(ebc.named_parameters(), include=True)
        )
        non_in_backward_params = dict(
            in_backward_optimizer_filter(ebc.named_parameters(), include=False)
        )
        self.assertEqual(set(in_backward_params.keys()), {"embedding_bags.t1.weight"})
        self.assertEqual(
            set(non_in_backward_params.keys()), {"embedding_bags.t2.weight"}
        )
