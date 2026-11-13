#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torch
from torchrec.distributed.itep_embeddingbag import (
    ITEPEmbeddingBagCollectionSharder,
    ITEPEmbeddingCollectionSharder,
    ShardedITEPEmbeddingBagCollection,
    ShardedITEPEmbeddingCollection,
)


class _StopAfterInnerShard(Exception):
    pass


class ITEPShardersModuleFqnTest(unittest.TestCase):
    def test_ebc_sharder_forwards_module_fqn(self) -> None:
        with patch(
            "torchrec.distributed.itep_embeddingbag.ShardedITEPEmbeddingBagCollection"
        ) as sharded:
            ITEPEmbeddingBagCollectionSharder(ebc_sharder=MagicMock()).shard(
                MagicMock(),
                {},
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.itep_ebc",
            )
        self.assertEqual(sharded.call_args.kwargs["module_fqn"], "sparse.itep_ebc")

    def test_ec_sharder_forwards_module_fqn(self) -> None:
        with patch(
            "torchrec.distributed.itep_embeddingbag.ShardedITEPEmbeddingCollection"
        ) as sharded:
            ITEPEmbeddingCollectionSharder(ebc_sharder=MagicMock()).shard(
                MagicMock(),
                {},
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.itep_ec",
            )
        self.assertEqual(sharded.call_args.kwargs["module_fqn"], "sparse.itep_ec")

    def test_sharded_ebc_forwards_module_fqn_to_inner_sharder(self) -> None:
        inner_sharder = MagicMock()
        inner_sharder.shard.side_effect = _StopAfterInnerShard
        with self.assertRaises(_StopAfterInnerShard):
            ShardedITEPEmbeddingBagCollection(
                MagicMock(),
                {},
                ebc_sharder=inner_sharder,
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.itep_ebc",
            )
        self.assertEqual(
            inner_sharder.shard.call_args.kwargs["module_fqn"], "sparse.itep_ebc"
        )

    def test_sharded_ec_forwards_module_fqn_to_inner_sharder(self) -> None:
        inner_sharder = MagicMock()
        inner_sharder.shard.side_effect = _StopAfterInnerShard
        with self.assertRaises(_StopAfterInnerShard):
            ShardedITEPEmbeddingCollection(
                MagicMock(),
                {},
                ebc_sharder=inner_sharder,
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.itep_ec",
            )
        self.assertEqual(
            inner_sharder.shard.call_args.kwargs["module_fqn"], "sparse.itep_ec"
        )
