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
from torchrec.distributed.embedding_types import (
    DPLookupAwaitable,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    NoWait,
    ParameterStorage,
    ShardingType,
)

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


class DPLookupAwaitableTest(unittest.TestCase):
    def test_lookup_is_deferred_and_started_once(self) -> None:
        calls = 0

        def lookup() -> Awaitable[int]:
            nonlocal calls
            calls += 1
            return NoWait(1)

        awaitable = DPLookupAwaitable(lookup)

        self.assertEqual(calls, 0)
        self.assertEqual(awaitable.wait(), 1)
        self.assertEqual(awaitable.wait(), 1)
        self.assertEqual(calls, 1)


class TestEmbeddingBagCollectionSharder(unittest.TestCase):
    def test_sharding_types(self) -> None:
        row_wise_types = {
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.GRID_SHARD.value,
        }
        cases = [
            ("cuda", True),
            ("mtia", True),
            ("xpu", True),
            ("cpu", False),
        ]
        sharder = EmbeddingBagCollectionSharder()
        for compute_device, has_row_wise in cases:
            with self.subTest(compute_device=compute_device):
                types = set(sharder.sharding_types(compute_device))
                if has_row_wise:
                    self.assertTrue(row_wise_types <= types)
                else:
                    self.assertFalse(row_wise_types & types)

    def test_storage_usage(self) -> None:
        tensor = torch.empty(10, 4, dtype=torch.float32)
        expected_bytes = tensor.numel() * 4
        cases = [
            ("cuda", ParameterStorage.HBM),
            ("mtia", ParameterStorage.HBM),
            ("xpu", ParameterStorage.HBM),
            ("cpu", ParameterStorage.DDR),
        ]
        sharder = EmbeddingBagCollectionSharder()
        for compute_device, storage in cases:
            with self.subTest(compute_device=compute_device):
                usage = sharder.storage_usage(
                    tensor, compute_device, EmbeddingComputeKernel.DENSE.value
                )
                self.assertEqual(usage, {storage.value: expected_bytes})
