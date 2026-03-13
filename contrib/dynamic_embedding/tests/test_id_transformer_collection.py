#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor
from torchrec_dynamic_embedding.id_transformer_collection import IDTransformerCollection


class TestIDTransformerCollection(unittest.TestCase):
    def testTransform(self):
        configs = [
            EmbeddingConfig(name="A", num_embeddings=8, embedding_dim=32),
            EmbeddingConfig(name="B", num_embeddings=8, embedding_dim=32),
        ]
        transformer_collection = IDTransformerCollection(configs)
        global_kjt = KeyedJaggedTensor(
            keys=["B", "A"],
            values=torch.tensor([1, 3, 2, 3, 4, 3, 2]),
            lengths=torch.tensor([4, 3]),
        )
        cache_kjt, fetch_handles = transformer_collection.transform(global_kjt)
        for handle in fetch_handles:
            handle.wait()
        self.assertEqual(cache_kjt.keys(), global_kjt.keys())
        self.assertTrue(torch.all(cache_kjt.lengths() == global_kjt.lengths()).item())
        self.assertTrue(
            torch.all(cache_kjt.values() == torch.tensor([0, 1, 2, 1, 0, 1, 2])).item()
        )

    def testFeatureNames(self):
        configs = [
            EmbeddingBagConfig(
                num_embeddings=8, embedding_dim=32, feature_names=["A", "B"]
            ),
            EmbeddingBagConfig(name="C", num_embeddings=8, embedding_dim=32),
        ]
        transformer_collection = IDTransformerCollection(configs)
        global_kjt = KeyedJaggedTensor(
            keys=["A", "C", "B"],
            values=torch.tensor([1, 3, 2, 3, 4, 3, 2, 3, 2, 1]),
            lengths=torch.tensor([4, 3, 3]),
        )
        cache_kjt, fetch_handles = transformer_collection.transform(global_kjt)
        for handle in fetch_handles:
            handle.wait()
        self.assertEqual(cache_kjt.keys(), global_kjt.keys())
        self.assertTrue(torch.all(cache_kjt.lengths() == global_kjt.lengths()).item())
        self.assertTrue(
            torch.all(
                cache_kjt.values() == torch.tensor([0, 1, 2, 1, 0, 1, 2, 1, 2, 0])
            ).item()
        )
