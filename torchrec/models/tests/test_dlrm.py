#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import dataclass
from typing import List

import torch
from parameterized import parameterized
from torch import nn
from torch.testing import FileCheck  # @manual
from torchrec.datasets.utils import Batch
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.fx import symbolic_trace
from torchrec.ir.serializer import JsonSerializer
from torchrec.ir.utils import decapsulate_ir_modules, encapsulate_ir_modules
from torchrec.models.dlrm import (
    choose,
    DenseArch,
    DLRM,
    DLRM_DCN,
    DLRM_Projection,
    DLRMTrain,
    DLRMWrapper,
    InteractionArch,
    InteractionDCNArch,
    InteractionProjectionArch,
    SparseArch,
)
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class SparseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)

        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)

        keys = ["f1", "f2", "f3", "f4", "f5"]
        offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=keys,
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=offsets,
        )
        B = (len(offsets) - 1) // len(keys)

        sparse_features = sparse_arch(features)
        F = len(sparse_arch.sparse_feature_names)
        self.assertEqual(sparse_features.shape, (B, F, D))

        expected_values = torch.tensor(
            [
                [
                    [-0.4518, -0.0242, 0.3637],
                    [-0.4518, -0.0242, 0.3637],
                    [0.2940, -0.2385, -0.0074],
                ],
                [
                    [-0.5452, -0.0231, -0.1907],
                    [-0.5452, -0.0231, -0.1907],
                    [-0.3530, -0.5551, -0.3342],
                ],
            ]
        )

        self.assertTrue(
            torch.allclose(
                sparse_features,
                expected_values,
                rtol=1e-4,
                atol=1e-4,
            ),
        )

    def test_fx_and_shape(self) -> None:
        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)
        F = len(sparse_arch.sparse_feature_names)
        gm = symbolic_trace(sparse_arch)

        FileCheck().check("KeyedJaggedTensor").check("cat").run(gm.code)

        keys = ["f1", "f2", "f3", "f4", "f5"]
        offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=keys,
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=offsets,
        )
        B = (len(offsets) - 1) // len(keys)

        sparse_features = gm(features)
        self.assertEqual(sparse_features.shape, (B, F, D))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=D, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)

        gm = symbolic_trace(sparse_arch)
        torch.jit.script(gm)


class DenseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)
        B = 4
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        dense_embedded = dense_arch(torch.rand((B, in_features)))
        self.assertEqual(dense_embedded.size(), (B, D))

        expected = torch.tensor(
            [
                [0.2351, 0.1578, 0.2784],
                [0.1579, 0.1012, 0.2660],
                [0.2459, 0.2379, 0.2749],
                [0.2582, 0.2178, 0.2860],
            ]
        )
        self.assertTrue(
            torch.allclose(
                dense_embedded,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_fx_and_shape(self) -> None:
        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        gm = symbolic_trace(dense_arch)
        dense_embedded = gm(torch.rand((B, in_features)))
        self.assertEqual(dense_embedded.size(), (B, D))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        gm = symbolic_trace(dense_arch)
        scripted_gm = torch.jit.script(gm)
        dense_embedded = scripted_gm(torch.rand((B, in_features)))
        self.assertEqual(dense_embedded.size(), (B, D))


class InteractionArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        dense_features = torch.rand((B, D))

        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_fx_and_shape(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        gm = symbolic_trace(inter_arch)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = gm(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        gm = symbolic_trace(inter_arch)
        scripted_gm = torch.jit.script(gm)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = scripted_gm(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_correctness(self) -> None:
        D = 4
        B = 3
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)
        torch.manual_seed(0)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

        expected = torch.tensor(
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.2353,
                    1.0123,
                    1.1919,
                    0.7220,
                    0.3444,
                    0.7397,
                    0.4015,
                    1.5184,
                    0.8986,
                    1.2018,
                ],
                [
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    1.2787,
                    0.3275,
                    1.6734,
                    0.6325,
                    0.2089,
                    1.2982,
                    0.3977,
                    0.4200,
                    0.2475,
                    0.7834,
                ],
                [
                    0.4556,
                    0.6323,
                    0.3489,
                    0.4017,
                    0.8195,
                    1.1181,
                    1.0511,
                    0.4919,
                    1.6147,
                    1.0786,
                    0.4264,
                    1.3576,
                    0.5860,
                    0.6559,
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

    def test_numerical_stability(self) -> None:
        D = 3
        B = 6
        keys = ["f1", "f2"]
        F = len(keys)

        inter_arch = InteractionArch(num_sparse_features=F)
        torch.manual_seed(0)
        dense_features = torch.randint(0, 10, (B, D))

        sparse_features = torch.randint(0, 10, (B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)

        expected = torch.LongTensor(
            [
                [4, 9, 3, 61, 57, 63],
                [0, 3, 9, 84, 27, 45],
                [7, 3, 7, 34, 50, 25],
                [3, 1, 6, 21, 50, 91],
                [6, 9, 8, 125, 109, 74],
                [6, 6, 8, 18, 80, 21],
            ]
        )

        self.assertTrue(torch.equal(concat_dense, expected))


class DLRMTest(unittest.TestCase):
    def test_basic(self) -> None:
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
        sparse_nn = DLRM(
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

        expected_logits = torch.tensor([[-0.2593], [-0.2487]])
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
        sparse_nn = DLRM(
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
            DLRM(
                embedding_bag_collection=ebc,
                dense_in_features=100,
                dense_arch_layer_sizes=[20, D_unused],
                over_arch_layer_sizes=[5, 1],
            )

    def test_fx(self) -> None:
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
        sparse_nn = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        gm = symbolic_trace(sparse_nn)
        FileCheck().check("KeyedJaggedTensor").check("f2").run(gm.code)

        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )

        logits = gm(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
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
        sparse_nn = DLRM(
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

        sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )

        gm = symbolic_trace(sparse_nn)

        scripted_gm = torch.jit.script(gm)

        logits = scripted_gm(features, sparse_features)
        self.assertEqual(logits.size(), (B, 1))


class DLRMTrainTest(unittest.TestCase):
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
        dlrm_module = DLRM(
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


class InteractionProjectionArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        F1 = 2
        F2 = 2
        I1 = DenseArch(
            in_features=2 * D + D,
            layer_sizes=[2 * D, F1 * D],
        )
        I2 = DenseArch(
            in_features=2 * D + D,
            layer_sizes=[2 * D, F2 * D],
        )
        inter_arch = InteractionProjectionArch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )

        dense_features = torch.rand((B, D))

        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        F1 = 4
        F2 = 4
        I1 = DenseArch(
            in_features=4 * D + D,
            layer_sizes=[4 * D, F1 * D],  # F1 = 4
        )
        I2 = DenseArch(
            in_features=4 * D + D,
            layer_sizes=[4 * D, F2 * D],  # F2 = 4
        )
        inter_arch = InteractionProjectionArch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

    def test_correctness(self) -> None:
        D = 4
        B = 3
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        F1 = 5
        F2 = 5
        I1 = nn.Identity()
        I2 = nn.Identity()
        inter_arch = InteractionProjectionArch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )
        torch.manual_seed(0)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

        expected = torch.tensor(
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.5057,
                    0.6874,
                    0.5756,
                    0.8082,
                    0.6656,
                    0.5402,
                    0.3672,
                    0.5765,
                    0.8837,
                    0.2710,
                    0.7540,
                    0.9349,
                    0.7424,
                    1.0666,
                    0.7297,
                    1.3209,
                    1.2713,
                    1.2888,
                    1.9248,
                    0.9367,
                    0.4865,
                    0.7688,
                    0.9932,
                    1.3475,
                    0.8313,
                ],
                [
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    1.0706,
                    0.8757,
                    1.0621,
                    1.3669,
                    0.6122,
                    0.9342,
                    0.7316,
                    0.7294,
                    1.0603,
                    0.3866,
                    0.2011,
                    0.2153,
                    0.3768,
                    0.3638,
                    0.2154,
                    1.0712,
                    0.8293,
                    1.3000,
                    1.4564,
                    0.8369,
                    0.3655,
                    0.4440,
                    0.6148,
                    1.0776,
                    0.5871,
                ],
                [
                    0.4556,
                    0.6323,
                    0.3489,
                    0.4017,
                    0.7294,
                    1.3899,
                    0.9493,
                    0.6186,
                    0.7565,
                    0.9535,
                    1.5688,
                    0.8992,
                    0.7077,
                    1.0088,
                    1.1206,
                    1.9778,
                    1.1639,
                    0.8642,
                    1.1966,
                    1.1827,
                    1.8592,
                    1.3003,
                    0.9441,
                    1.1177,
                    0.4730,
                    0.7631,
                    0.4304,
                    0.3937,
                    0.3230,
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


class DLRM_ProjectionTest(unittest.TestCase):
    def test_basic(self) -> None:
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
        sparse_nn = DLRM_Projection(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
            interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
            over_arch_layer_sizes=[5, 1],
        )
        self.assertTrue(isinstance(sparse_nn.inter_arch, InteractionProjectionArch))

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

        expected_logits = torch.tensor([[-0.4603], [-0.4639]])
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_dense_size(self) -> None:
        torch.manual_seed(0)
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
        with self.assertRaises(ValueError):
            sparse_nn = DLRM_Projection(  # noqa
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D + 1],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
                over_arch_layer_sizes=[5, 1],
            )

    def test_interaction_size(self) -> None:
        torch.manual_seed(0)
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
        with self.assertRaises(ValueError):
            sparse_nn = DLRM_Projection(
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D + 1],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
                over_arch_layer_sizes=[5, 1],
            )

        with self.assertRaises(ValueError):
            sparse_nn = DLRM_Projection(  # noqa
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D + 1],
                over_arch_layer_sizes=[5, 1],
            )


class DLRM_ProjectionTrainTest(unittest.TestCase):
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
        dlrm_module = DLRM_Projection(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
            interaction_branch1_layer_sizes=[80, 40],
            interaction_branch2_layer_sizes=[80, 48],
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


class InteractionDCNArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        DCN = LowRankCrossNet(in_features=F * D + D, num_layers=1, low_rank=D)
        inter_arch = InteractionDCNArch(
            num_sparse_features=F,
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))

        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (F*D + D)
        self.assertEqual(concat_dense.size(), (B, F * D + D))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        DCN = LowRankCrossNet(in_features=F * D + D, num_layers=2, low_rank=D)
        inter_arch = InteractionDCNArch(
            num_sparse_features=F,
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (F*D+D)
        self.assertEqual(concat_dense.size(), (B, D * F + D))

    def test_correctness(self) -> None:
        D = 4
        B = 3
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        DCN = nn.Identity()
        inter_arch = InteractionDCNArch(
            num_sparse_features=F,
            crossnet=DCN,
        )
        torch.manual_seed(0)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (F*D + D)
        self.assertEqual(concat_dense.size(), (B, F * D + D))

        expected = torch.tensor(
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.0223,
                    0.1689,
                    0.2939,
                    0.5185,
                    0.6977,
                    0.8000,
                    0.1610,
                    0.2823,
                    0.6816,
                    0.9152,
                    0.3971,
                    0.8742,
                    0.4194,
                    0.5529,
                    0.9527,
                    0.0362,
                ],
                [
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    0.1852,
                    0.3734,
                    0.3051,
                    0.9320,
                    0.1759,
                    0.2698,
                    0.1507,
                    0.0317,
                    0.2081,
                    0.9298,
                    0.7231,
                    0.7423,
                    0.5263,
                    0.2437,
                    0.5846,
                    0.0332,
                ],
                [
                    0.4556,
                    0.6323,
                    0.3489,
                    0.4017,
                    0.1387,
                    0.2422,
                    0.8155,
                    0.7932,
                    0.2783,
                    0.4820,
                    0.8198,
                    0.9971,
                    0.6984,
                    0.5675,
                    0.8352,
                    0.2056,
                    0.5932,
                    0.1123,
                    0.1535,
                    0.2417,
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


class DLRM_DCNTest(unittest.TestCase):
    def test_basic(self) -> None:
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
        sparse_nn = DLRM_DCN(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            dcn_num_layers=2,
            dcn_low_rank_dim=8,
            over_arch_layer_sizes=[5, 1],
        )
        self.assertTrue(isinstance(sparse_nn.inter_arch, InteractionDCNArch))

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

        expected_logits = torch.tensor([[0.0455], [0.0408]])
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_dense_size(self) -> None:
        torch.manual_seed(0)
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
        with self.assertRaises(ValueError):
            sparse_nn = DLRM_DCN(  # noqa
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D + 1],
                dcn_num_layers=2,
                dcn_low_rank_dim=8,
                over_arch_layer_sizes=[5, 1],
            )


class DLRM_DCNTrainTest(unittest.TestCase):
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
        dlrm_dcn_module = DLRM_DCN(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
            dcn_num_layers=2,
            dcn_low_rank_dim=8,
        )
        dlrm = DLRMTrain(dlrm_dcn_module)

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


class DLRMExampleTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=100,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
            dense_features=features,
            sparse_features=sparse_features,
        )

        self.assertEqual(logits.size(), (B, 1))

    def test_export_serialization(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=100,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
            dense_features=features,
            sparse_features=sparse_features,
        )

        self.assertEqual(logits.size(), (B, 1))

        model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)

        ep = torch.export.export(
            model,
            (features, sparse_features),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        # Run forward on ExportedProgram
        ep_output = ep.module()(features, sparse_features)
        self.assertEqual(ep_output.size(), (B, 1))

        unflatten_model = torch.export.unflatten(ep)
        deserialized_model = decapsulate_ir_modules(unflatten_model, JsonSerializer)
        deserialized_logits = deserialized_model(features, sparse_features)

        self.assertEqual(deserialized_logits.size(), (B, 1))


class DLRMWrapperTest(unittest.TestCase):
    @dataclass
    class WrapperTestParams:
        # input parameters
        embedding_configs: List[EmbeddingBagConfig]
        sparse_feature_keys: List[str]
        sparse_feature_values: List[int]
        sparse_feature_offsets: List[int]
        # expected output parameters
        expected_output_size: tuple[int, ...]

    @parameterized.expand(
        [
            (
                "basic_with_multiple_features",
                WrapperTestParams(
                    embedding_configs=[
                        EmbeddingBagConfig(
                            name="t1",
                            embedding_dim=8,
                            num_embeddings=100,
                            feature_names=["f1", "f3"],
                        ),
                        EmbeddingBagConfig(
                            name="t2",
                            embedding_dim=8,
                            num_embeddings=100,
                            feature_names=["f2"],
                        ),
                    ],
                    sparse_feature_keys=["f1", "f3", "f2"],
                    sparse_feature_values=[1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3],
                    sparse_feature_offsets=[0, 2, 4, 6, 8, 10, 11],
                    expected_output_size=(2, 1),
                ),
            ),
            (
                "empty_sparse_features",
                WrapperTestParams(
                    embedding_configs=[
                        EmbeddingBagConfig(
                            name="t1",
                            embedding_dim=8,
                            num_embeddings=100,
                            feature_names=["f1"],
                        ),
                    ],
                    sparse_feature_keys=["f1"],
                    sparse_feature_values=[],
                    sparse_feature_offsets=[0, 0, 0],
                    expected_output_size=(2, 1),
                ),
            ),
        ]
    )
    def test_wrapper_functionality(
        self, _test_name: str, test_params: WrapperTestParams
    ) -> None:
        B = 2
        D = 8
        dense_in_features = 100

        ebc = EmbeddingBagCollection(tables=test_params.embedding_configs)

        dlrm_wrapper = DLRMWrapper(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        # Create ModelInput
        dense_features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=test_params.sparse_feature_keys,
            values=torch.tensor(test_params.sparse_feature_values, dtype=torch.long),
            offsets=torch.tensor(test_params.sparse_feature_offsets, dtype=torch.long),
        )

        model_input = ModelInput(
            float_features=dense_features,
            idlist_features=sparse_features,
            idscore_features=None,
            label=torch.rand((B,)),
        )

        # Test eval mode - should return just logits
        dlrm_wrapper.eval()
        logits = dlrm_wrapper(model_input)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.size(), test_params.expected_output_size)

        # Test training mode - should return (loss, logits) tuple
        dlrm_wrapper.train()
        result = dlrm_wrapper(model_input)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        loss, pred = result
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(pred, torch.Tensor)
        self.assertEqual(loss.size(), ())  # scalar loss
        self.assertEqual(pred.size(), test_params.expected_output_size)
