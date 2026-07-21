#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List

import torch
import torch.nn as nn
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.model_arch import extract_model_arch, model_arch_hash
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class ModelArchTest(unittest.TestCase):
    def _model(self, num_embeddings: int = 100) -> TestSparseNN:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=64,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(3)
        ]
        return TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

    def _sharders(self) -> List[ModuleSharder[nn.Module]]:
        return cast(List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()])

    def _sharders_with_opt(self, optimizer: str) -> List[ModuleSharder[nn.Module]]:
        return cast(
            List[ModuleSharder[nn.Module]],
            [
                EmbeddingBagCollectionSharder(
                    fused_params={"optimizer": optimizer, "learning_rate": 0.01}
                )
            ],
        )

    def test_extracts_tables_and_sharders(self) -> None:
        arch = extract_model_arch(self._model(), self._sharders())
        by_name = {t.name: t for t in arch.tables}
        for i in range(3):
            self.assertIn(f"table_{i}", by_name)
        self.assertEqual(by_name["table_0"].num_embeddings, 100)
        self.assertEqual(by_name["table_0"].embedding_dim, 64)
        self.assertEqual(by_name["table_0"].feature_names, ("feature_0",))
        self.assertIn("EmbeddingBagCollectionSharder", arch.sharder_types)
        # Each table records the sharder that shards it (enumerator-faithful).
        self.assertEqual(
            by_name["table_0"].sharder_type, "EmbeddingBagCollectionSharder"
        )
        # Tables are name-sorted for a deterministic, content-addressable order.
        self.assertEqual(
            [t.name for t in arch.tables], sorted(t.name for t in arch.tables)
        )

    def test_hash_is_stable_and_deterministic(self) -> None:
        h1 = model_arch_hash(extract_model_arch(self._model(), self._sharders()))
        h2 = model_arch_hash(extract_model_arch(self._model(), self._sharders()))
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_hash_changes_with_arch(self) -> None:
        base = model_arch_hash(
            extract_model_arch(self._model(num_embeddings=100), self._sharders())
        )
        bigger = model_arch_hash(
            extract_model_arch(self._model(num_embeddings=200), self._sharders())
        )
        self.assertNotEqual(base, bigger)

    def test_captures_sharder_optimizer(self) -> None:
        arch = extract_model_arch(self._model(), self._sharders_with_opt("adam"))
        self.assertEqual(len(arch.sharders), 1)
        sharder = arch.sharders[0]
        self.assertEqual(sharder.sharder_type, "EmbeddingBagCollectionSharder")
        self.assertEqual(sharder.optimizer, "adam")
        self.assertEqual(sharder.learning_rate, 0.01)

    def test_hash_changes_with_optimizer(self) -> None:
        # Same tables + same sharder type, different sparse optimizer -> different
        # hash (the optimizer changes the storage multiplier, so it must not collide).
        adam = model_arch_hash(
            extract_model_arch(self._model(), self._sharders_with_opt("adam"))
        )
        adagrad = model_arch_hash(
            extract_model_arch(
                self._model(), self._sharders_with_opt("rowwise_adagrad")
            )
        )
        self.assertNotEqual(adam, adagrad)
