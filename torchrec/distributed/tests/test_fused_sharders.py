#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torch
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.fused_embedding import (
    FusedEmbeddingCollectionSharder,
    ShardedFusedEmbeddingCollection,
)
from torchrec.distributed.fused_embeddingbag import (
    FusedEmbeddingBagCollectionSharder,
    ShardedFusedEmbeddingBagCollection,
)


class FusedShardersModuleFqnTest(unittest.TestCase):
    def test_ebc_sharder_forwards_module_fqn(self) -> None:
        with patch(
            "torchrec.distributed.fused_embeddingbag.ShardedFusedEmbeddingBagCollection"
        ) as sharded:
            FusedEmbeddingBagCollectionSharder().shard(
                MagicMock(),
                {},
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.fused_ebc",
            )
        self.assertEqual(sharded.call_args.kwargs["module_fqn"], "sparse.fused_ebc")

    def test_ec_sharder_forwards_module_fqn(self) -> None:
        with patch(
            "torchrec.distributed.fused_embedding.ShardedFusedEmbeddingCollection"
        ) as sharded:
            FusedEmbeddingCollectionSharder().shard(
                MagicMock(),
                {},
                env=MagicMock(),
                device=torch.device("cpu"),
                module_fqn="sparse.fused_ec",
            )
        self.assertEqual(sharded.call_args.kwargs["module_fqn"], "sparse.fused_ec")

    def test_sharded_ebc_passes_module_fqn_to_parent(self) -> None:
        parent_kwargs: Dict[str, Any] = {}

        def fake_init(self: ShardedEmbeddingBagCollection, **kwargs: Any) -> None:
            parent_kwargs.update(kwargs)
            self._embedding_shardings = []
            self._lookups = []

        with patch(
            "torchrec.distributed.fused_embeddingbag.convert_optimizer_type_and_kwargs",
            return_value=(MagicMock(), {}),
        ), patch.object(ShardedEmbeddingBagCollection, "__init__", fake_init):
            ShardedFusedEmbeddingBagCollection(
                MagicMock(), {}, MagicMock(), module_fqn="sparse.fused_ebc"
            )
        self.assertEqual(parent_kwargs["module_fqn"], "sparse.fused_ebc")

    def test_sharded_ec_passes_module_fqn_to_parent(self) -> None:
        parent_kwargs: Dict[str, Any] = {}

        def fake_init(self: ShardedEmbeddingCollection, **kwargs: Any) -> None:
            parent_kwargs.update(kwargs)
            self._sharding_type_to_sharding = {}
            self._lookups = []

        with patch(
            "torchrec.distributed.fused_embedding.convert_optimizer_type_and_kwargs",
            return_value=(MagicMock(), {}),
        ), patch.object(ShardedEmbeddingCollection, "__init__", fake_init):
            ShardedFusedEmbeddingCollection(
                MagicMock(), {}, MagicMock(), module_fqn="sparse.fused_ec"
            )
        self.assertEqual(parent_kwargs["module_fqn"], "sparse.fused_ec")
