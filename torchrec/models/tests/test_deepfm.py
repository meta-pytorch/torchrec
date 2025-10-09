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
from torch.testing import FileCheck  # @manual
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.fx import symbolic_trace, Tracer
from torchrec.models.deepfm import (
    DenseArch,
    FMInteractionArch,
    SimpleDeepFMNN,
    SimpleDeepFMNNWrapper,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class DenseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)

        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(
            in_features=in_features, hidden_layer_size=10, embedding_dim=D
        )

        dense_arch_input = torch.rand((B, in_features))
        dense_embedded = dense_arch(dense_arch_input)
        self.assertEqual(dense_embedded.size(), (B, D))

        # check tracer compatibility
        gm = torch.fx.GraphModule(dense_arch, Tracer().trace(dense_arch))
        script = torch.jit.script(gm)
        script(dense_arch_input)


class FMInteractionArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)

        D = 3
        B = 3
        DI = 2
        keys = ["f1", "f2"]
        F = len(keys)
        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )
        inter_arch = FMInteractionArch(
            fm_in_features=D + D * F,
            sparse_feature_names=keys,
            deep_fm_dimension=DI,
        )
        inter_output = inter_arch(dense_features, embeddings)
        self.assertEqual(inter_output.size(), (B, D + DI + 1))

        # check output forward numerical accuracy
        expected_output = torch.Tensor(
            [
                [0.4963, 0.7682, 0.0885, 0.0000, 0.2646, 4.3660],
                [0.1320, 0.3074, 0.6341, 0.0000, 0.0834, 7.6417],
                [0.4901, 0.8964, 0.4556, 0.0000, 0.0671, 15.5230],
            ],
        )
        self.assertTrue(
            torch.allclose(
                inter_output,
                expected_output,
                rtol=1e-4,
                atol=1e-4,
            )
        )

        # check tracer compatibility
        gm = torch.fx.GraphModule(inter_arch, Tracer().trace(inter_arch))
        torch.jit.script(gm)


class SimpleDeepFMNNTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100
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

        features = torch.rand((B, num_dense_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )

        logits = deepfm_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    def test_no_sparse(self) -> None:
        ebc = EmbeddingBagCollection(tables=[])
        with self.assertRaises(AssertionError):
            SimpleDeepFMNN(
                num_dense_features=10,
                embedding_bag_collection=ebc,
                hidden_layer_size=20,
                deep_fm_dimension=5,
            )

    def test_fx(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )
        gm = symbolic_trace(deepfm_nn)
        FileCheck().check("KeyedJaggedTensor").check("f2").run(gm.code)

        features = torch.rand((B, num_dense_features))
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

    def test_fx_script(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100

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
        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )

        features = torch.rand((B, num_dense_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        deepfm_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )

        gm = symbolic_trace(deepfm_nn)

        scripted_gm = torch.jit.script(gm)

        logits = scripted_gm(features, sparse_features)
        self.assertEqual(logits.size(), (B, 1))


class SimpleDeepFMNNWrapperTest(unittest.TestCase):
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
        num_dense_features = 100

        ebc = EmbeddingBagCollection(tables=test_params.embedding_configs)

        deepfm_wrapper = SimpleDeepFMNNWrapper(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )

        # Create ModelInput
        dense_features = torch.rand((B, num_dense_features))
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
        deepfm_wrapper.eval()
        logits = deepfm_wrapper(model_input)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.size(), test_params.expected_output_size)

        # Test training mode - should return (loss, logits) tuple
        deepfm_wrapper.train()
        result = deepfm_wrapper(model_input)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        loss, pred = result
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(pred, torch.Tensor)
        self.assertEqual(loss.size(), ())  # scalar loss
        self.assertEqual(pred.size(), test_params.expected_output_size)


if __name__ == "__main__":
    unittest.main()
