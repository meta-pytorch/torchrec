#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast

import torch
from torchrec.fx.tracer import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    FeatureProcessor,
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PositionWeightedModuleEmbeddingBagCollectionTest(unittest.TestCase):

    def generate_fp_ebc(self) -> FeatureProcessedEmbeddingBagCollection:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=8, num_embeddings=16, feature_names=["f2"]
                ),
            ],
            is_weighted=True,
        )
        feature_processors = {
            "f1": cast(FeatureProcessor, PositionWeightedModule(max_feature_length=10)),
            "f2": cast(FeatureProcessor, PositionWeightedModule(max_feature_length=5)),
        }
        return FeatureProcessedEmbeddingBagCollection(ebc, feature_processors)

    def test_position_weighted_module_ebc(self) -> None:
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        fp_ebc = self.generate_fp_ebc()

        pooled_embeddings = fp_ebc(features)
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 8, 16])

        fp_ebc_gm_script = torch.jit.script(symbolic_trace(fp_ebc))
        pooled_embeddings_gm_script = fp_ebc_gm_script(features)

        torch.testing.assert_close(
            pooled_embeddings_gm_script.values(), pooled_embeddings.values()
        )

        torch.testing.assert_close(
            pooled_embeddings_gm_script.offset_per_key(),
            pooled_embeddings.offset_per_key(),
        )

    def test_position_weighted_module_ebc_with_excessive_features(self) -> None:
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # 2   [8]   None    None
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8, 9, 9, 9]),
        )

        fp_ebc = self.generate_fp_ebc()

        pooled_embeddings = fp_ebc(features)
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 8, 16])

        # Test split method, FP then EBC
        fp, ebc = fp_ebc.split()
        fp_kjt = fp(features)
        pooled_embeddings_split = ebc(fp_kjt)

        self.assertEqual(pooled_embeddings_split.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings_split.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings_split.offset_per_key(), [0, 8, 16])


class PositionWeightedModuleCollectionEmbeddingBagCollectionTest(unittest.TestCase):
    def generate_fp_ebc(self) -> FeatureProcessedEmbeddingBagCollection:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=8, num_embeddings=16, feature_names=["f2"]
                ),
            ],
            is_weighted=True,
        )

        return FeatureProcessedEmbeddingBagCollection(
            ebc, PositionWeightedModuleCollection({"f1": 10, "f2": 10})
        )

    def test_position_weighted_collection_module_ebc(self) -> None:
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        fp_ebc = self.generate_fp_ebc()

        pooled_embeddings = fp_ebc(features)
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 8, 16])

        fp_ebc_gm_script = torch.jit.script(symbolic_trace(fp_ebc))
        pooled_embeddings_gm_script = fp_ebc_gm_script(features)

        torch.testing.assert_close(
            pooled_embeddings_gm_script.values(), pooled_embeddings.values()
        )

        torch.testing.assert_close(
            pooled_embeddings_gm_script.offset_per_key(),
            pooled_embeddings.offset_per_key(),
        )

        # Test split method, FP then EBC
        fp, ebc = fp_ebc.split()
        fp_kjt = fp(features)
        pooled_embeddings_split = ebc(fp_kjt)

        self.assertEqual(pooled_embeddings_split.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings_split.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings_split.offset_per_key(), [0, 8, 16])
