#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchrec.fx import Tracer
from torchrec.modules.deepfm import DeepFM, FactorizationMachine


class TestDeepFM(unittest.TestCase):
    def test_deepfm_shape(self) -> None:

        batch_size = 3
        output_dim = 30
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
            torch.randn(batch_size, 3, 100),
            torch.randn(batch_size, 5, 120),
        ]
        in_features = 2 * 64 + 2 * 32 + 3 * 100 + 5 * 120
        dense_module = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
            torch.nn.ReLU(),
        )
        deepfm = DeepFM(dense_module=dense_module)

        deep_fm_output = deepfm(input_embeddings)

        self.assertEqual(list(deep_fm_output.shape), [batch_size, output_dim])

    def test_deepfm_with_lazy_shape(self) -> None:
        batch_size = 3
        output_dim = 30
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
            torch.randn(batch_size, 3, 100),
            torch.randn(batch_size, 5, 120),
        ]
        dense_module = torch.nn.Sequential(
            torch.nn.LazyLinear(output_dim),
            torch.nn.ReLU(),
        )
        deepfm = DeepFM(dense_module=dense_module)

        deep_fm_output = deepfm(input_embeddings)

        self.assertEqual(list(deep_fm_output.shape), [batch_size, output_dim])

    def test_deepfm_numerical_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        output_dim = 2
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
            torch.randn(batch_size, 3, 100),
            torch.randn(batch_size, 5, 120),
        ]
        in_features = 2 * 64 + 2 * 32 + 3 * 100 + 5 * 120
        dense_module = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim),
            torch.nn.ReLU(),
        )
        deepfm = DeepFM(dense_module=dense_module)

        output = deepfm(input_embeddings)

        expected_output = torch.Tensor(
            [
                [0.0896, 0.1182],
                [0.0675, 0.0972],
                [0.0764, 0.0199],
            ],
        )
        self.assertTrue(
            torch.allclose(
                output,
                expected_output,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_fx_script_deepfm(self) -> None:
        m = DeepFM(dense_module=torch.nn.Linear(4, 1))

        # dryrun to initialize the input
        m([torch.randn(2, 2, 2)])
        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)


class TestFM(unittest.TestCase):
    def test_fm_shape(self) -> None:

        batch_size = 3
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
            torch.randn(batch_size, 3, 100),
            torch.randn(batch_size, 5, 120),
        ]

        fm = FactorizationMachine()

        fm_output = fm(input_embeddings)

        self.assertEqual(list(fm_output.shape), [batch_size, 1])

    def test_fm_numerical_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
            torch.randn(batch_size, 3, 100),
            torch.randn(batch_size, 5, 120),
        ]
        fm = FactorizationMachine()

        output = fm(input_embeddings)

        expected_output = torch.Tensor(
            [
                [-577.5231],
                [752.7272],
                [-509.1023],
            ]
        )
        self.assertTrue(
            torch.allclose(
                output,
                expected_output,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_fx_script_fm(self) -> None:
        m = FactorizationMachine()
        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)
