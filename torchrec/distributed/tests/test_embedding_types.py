#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List

import torch
from torchrec.distributed.embedding_types import KJTList, ShardedEmbeddingModule
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionContext
from torchrec.distributed.types import Awaitable, LazyAwaitable

Out = Dict[str, torch.Tensor]
CompIn = KJTList
DistOut = List[torch.Tensor]
ShrdCtx = EmbeddingBagCollectionContext


class FakeShardedEmbeddingModule(ShardedEmbeddingModule[CompIn, DistOut, Out, ShrdCtx]):
    def __init__(self) -> None:
        # pyrefly: ignore[missing-attribute]
        super().__init__()
        self._lookups = [
            torch.nn.Module(),
            torch.nn.Module(),
        ]

    #  return value of `None`.
    # pyrefly: ignore[bad-return]
    def create_context(self) -> ShrdCtx:
        pass

    def input_dist(
        self,
        ctx: ShrdCtx,
        *input,
        **kwargs,
        #  return value of `None`.
        # pyrefly: ignore[bad-return]
    ) -> Awaitable[Awaitable[CompIn]]:
        pass

    # pyrefly: ignore[bad-return]
    def compute(self, ctx: ShrdCtx, dist_input: CompIn) -> DistOut:
        pass

    #  return value of `None`.
    # pyrefly: ignore[bad-return]
    def output_dist(self, ctx: ShrdCtx, output: DistOut) -> LazyAwaitable[Out]:
        pass


class TestShardedEmbeddingModule(unittest.TestCase):
    def test_train_mode(self) -> None:
        embedding_module = FakeShardedEmbeddingModule()
        for mode in [True, False]:
            with self.subTest(mode=mode):
                embedding_module.train(mode)
                self.assertEqual(embedding_module.training, mode)
                for lookup in embedding_module._lookups:
                    self.assertEqual(lookup.training, mode)
