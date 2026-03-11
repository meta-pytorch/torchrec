#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.datasets.utils import Batch
from torchrec.models.dlrm import DLRMTrain
from torchrec.models.experimental.transformerdlrm import (
    DLRM_Transformer,
    InteractionTransformerArch,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class InteractionArchTransformerTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 8
        B = 10
        # multi-head attentions
        nhead = 8
        ntransformer_layers = 4
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionTransformerArch(
            num_sparse_features=F,
            embedding_dim=D,
            nhead=nhead,
            ntransformer_layers=ntransformer_layers,
        )
        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D * (F + 1)))

    def test_larger(self) -> None:
        D = 16
        B = 20
        # multi-head attentions
        nhead = 8
        ntransformer_layers = 4
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        inter_arch = InteractionTransformerArch(
            num_sparse_features=F,
            embedding_dim=D,
            nhead=nhead,
            ntransformer_layers=ntransformer_layers,
        )
        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        self.assertEqual(concat_dense.size(), (B, D * (F + 1)))

    def test_correctness_disabled_in_oss_compatibility(self) -> None:
        D = 4
        B = 3
        # multi-head attentions
        nhead = 4
        ntransformer_layers = 4
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        # place the manual_seed before the InteractionTransformerArch object to generate the same initialization random values in the Transformer
        torch.manual_seed(0)
        inter_arch = InteractionTransformerArch(
            num_sparse_features=F,
            embedding_dim=D,
            nhead=nhead,
            ntransformer_layers=ntransformer_layers,
        )
        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        self.assertEqual(concat_dense.size(), (B, D * (F + 1)))
        expected = torch.tensor(
            [
                [
                    -0.4411,
                    0.2487,
                    -1.2685,
                    1.4610,
                    1.3110,
                    0.5152,
                    -0.4960,
                    -1.3303,
                    -0.3962,
                    -0.0623,
                    -1.1371,
                    1.5956,
                    0.2431,
                    -1.6820,
                    0.5242,
                    0.9148,
                    1.3033,
                    0.6409,
                    -0.9577,
                    -0.9866,
                ],
                [
                    -1.0850,
                    -0.0366,
                    -0.4862,
                    1.6078,
                    1.1254,
                    -0.9989,
                    -0.9927,
                    0.8661,
                    -0.1704,
                    1.0223,
                    -1.5580,
                    0.7060,
                    -0.3081,
                    -1.3686,
                    0.2788,
                    1.3979,
                    0.0328,
                    1.5470,
                    -0.3670,
                    -1.2128,
                ],
                [
                    -1.5917,
                    -0.0995,
                    0.7302,
                    0.9609,
                    0.6606,
                    1.0238,
                    -0.1017,
                    -1.5827,
                    -0.6761,
                    -1.0771,
                    0.2262,
                    1.5269,
                    -0.5671,
                    -1.2114,
                    1.4503,
                    0.3281,
                    -0.6540,
                    -1.2925,
                    0.9134,
                    1.0331,
                ],
            ]
        )
        self.assertTrue(
            torch.allclose(
                concat_dense,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_numerical_stability_disabled_in_oss_compatibility(self) -> None:
        D = 4
        B = 3
        # multi-head attentions
        nhead = 4
        ntransformer_layers = 4
        keys = ["f1", "f2"]
        F = len(keys)
        torch.manual_seed(0)
        inter_arch = InteractionTransformerArch(
            num_sparse_features=F,
            embedding_dim=D,
            nhead=nhead,
            ntransformer_layers=ntransformer_layers,
        )
        dense_features = 10 * torch.rand(B, D)
        sparse_features = 10 * torch.rand(B, F, D)
        concat_dense = inter_arch(dense_features, sparse_features)
        expected = torch.LongTensor(
            [
                [0, 1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 1],
                [0, 0, 0, 1, -1, 0, 1, 0, 0, 0, 1, 0],
                [-1, 0, 0, 0, 0, 0, -1, 1, 0, 1, -1, 0],
            ]
        )
        torch.testing.assert_close(concat_dense.long(), expected, rtol=0, atol=0)


class DLRMTransformerTest(unittest.TestCase):
    def test_basic_disabled_in_oss_compatibility(self) -> None:
        torch.manual_seed(0)
        B = 2
        D = 8
        dense_in_features = 100
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = DLRM_Transformer(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )
        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))
        expected_logits = torch.tensor([[-0.2593], [-0.1043]])
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_one_sparse(self) -> None:
        B = 2
        D = 8
        dense_in_features = 100
        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config])
        sparse_nn = DLRM_Transformer(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )
        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    def test_no_sparse(self) -> None:
        ebc = EmbeddingBagCollection(tables=[])
        D_unused = 1
        with self.assertRaises(AssertionError):
            DLRM_Transformer(
                embedding_bag_collection=ebc,
                dense_in_features=100,
                dense_arch_layer_sizes=[20, D_unused],
                over_arch_layer_sizes=[5, 1],
            )


class DLRMTransformerTrainTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 2
        D = 8
        dense_in_features = 100
        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config])
        dlrm_module = DLRM_Transformer(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        dlrm = DLRMTrain(dlrm_module)
        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )
        batch = Batch(
            dense_features=features,
            sparse_features=sparse_features,
            labels=torch.randint(2, (B,)),
        )
        _, (_, logits, _) = dlrm(batch)
        self.assertEqual(logits.size(), (B,))
