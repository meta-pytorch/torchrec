#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from functools import partial

import torch
import torch.fx
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbeddingBagCollectionTest(unittest.TestCase):
    def test_unweighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
            init_fn=partial(torch.nn.init.normal_, mean=0.0, std=1.5),
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
            init_fn=partial(torch.nn.init.uniform_, a=-0.036, b=0.036),
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

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

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (3, 7))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 7])

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])

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

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (3, 6))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6])

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (6, 7))
        self.assertEqual(pooled_embeddings.keys(), ["f1@t1", "f1@t2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 7])

    def test_weighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config], is_weighted=True)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 3, 4, 7]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.9, 0.1, 0.3, 0.4, 0.7]
            ),
        )

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (2, 10))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f3", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6, 10])

    def test_forward_with_meta_device(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            is_weighted=True,
            device=torch.device("meta"),
        )

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 3, 4, 7], device="meta"),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12], device="meta"),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.9, 0.1, 0.3, 0.4, 0.7],
                device="meta",
            ),
        )

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (2, 10))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f3", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6, 10])
        self.assertEqual(pooled_embeddings.values().device, torch.device("meta"))

    def test_fx(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config], is_weighted=True)

        gm = symbolic_trace(ebc)
        torch.jit.script(gm)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 3, 4, 7]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.9, 0.1, 0.3, 0.4, 0.7]
            ),
        )

        pooled_embeddings = gm(features)
        self.assertEqual(pooled_embeddings.values().size(), (2, 10))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f3", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6, 10])

    def test_scripting(self) -> None:
        config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        ebc = EmbeddingBagCollection(tables=[config])
        torch.jit.script(ebc)

    def test_duplicate_config_name_fails(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        with self.assertRaises(ValueError):
            EmbeddingBagCollection(tables=[eb1_config, eb2_config])

    def test_device(self) -> None:
        config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )

        # test with device from input
        ebc = EmbeddingBagCollection(tables=[config], device=torch.device("meta"))
        self.assertEqual(torch.device("meta"), ebc.embedding_bags["t1"].weight.device)
        self.assertEqual(torch.device("meta"), ebc.device)

        # test with device from context manager
        with torch.device("meta"):
            ebc = EmbeddingBagCollection(tables=[config])
        self.assertEqual(torch.device("meta"), ebc.embedding_bags["t1"].weight.device)
        self.assertEqual(torch.device("meta"), ebc.device)

        # test default device is cpu
        ebc = EmbeddingBagCollection(tables=[config])
        self.assertEqual(torch.device("cpu"), ebc.embedding_bags["t1"].weight.device)
        self.assertEqual(torch.device("cpu"), ebc.device)


class EmbeddingCollectionTest(unittest.TestCase):
    def test_forward(self) -> None:
        tb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f2"],
        )
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        ec = EmbeddingCollection(
            tables=[tb1_config, tb2_config],
        )

        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 3, 3, 4]),
        )

        sequence_embeddings = ec(
            features=id_list_features,
        )
        self.assertEqual(sequence_embeddings["f1"].values().size(), (3, 5))
        self.assertEqual(sequence_embeddings["f2@t1"].values().size(), (1, 5))
        self.assertEqual(sequence_embeddings["f2@t2"].values().size(), (1, 5))

    def test_indices(self) -> None:
        tb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ec = EmbeddingCollection(
            tables=[tb1_config, tb2_config],
            need_indices=True,
        )

        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 3, 3, 4]),
        )

        sequence_embeddings = ec(
            features=id_list_features,
        )
        torch.testing.assert_close(
            sequence_embeddings["f1"].weights(),
            torch.tensor([0, 1, 2]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sequence_embeddings["f2@t1"].weights(),
            torch.tensor([3]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sequence_embeddings["f2@t2"].weights(),
            torch.tensor([3]),
            rtol=0,
            atol=0,
        )

    def test_fx(self) -> None:
        tb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f1", "f2"],
        )
        tb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ec = EmbeddingCollection(
            tables=[tb1_config, tb2_config],
        )
        gm = symbolic_trace(ec)
        torch.jit.script(gm)

        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 3, 3, 4]),
        )

        sequence_embeddings = gm(
            features=id_list_features,
        )
        self.assertEqual(sequence_embeddings["f1"].values().size(), (3, 5))
        self.assertEqual(sequence_embeddings["f2@t1"].values().size(), (1, 5))
        self.assertEqual(sequence_embeddings["f2@t2"].values().size(), (1, 5))

    def test_scripting(self) -> None:
        config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        ec = EmbeddingCollection(tables=[config])
        torch.jit.script(ec)

    def test_device(self) -> None:
        config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        ec = EmbeddingCollection(
            tables=[config],
            device=torch.device("meta"),
        )
        self.assertEqual(torch.device("meta"), ec.embeddings["t1"].weight.device)

    def test_duplicate_config_name_fails(self) -> None:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        with self.assertRaises(ValueError):
            EmbeddingCollection(tables=[e1_config, e2_config])
