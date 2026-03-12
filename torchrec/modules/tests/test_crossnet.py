#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torch.fx import GraphModule, Tracer
from torchrec.modules.crossnet import (
    CrossNet,
    LowRankCrossNet,
    LowRankMixtureCrossNet,
    VectorCrossNet,
)


# unit test for Full Rank CrossNet: CrossNet
class TestCrossNet(unittest.TestCase):
    def test_cross_net_numercial_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        num_layers = 20
        in_features = 2
        input = torch.randn(batch_size, in_features)

        # test using vector for crossing
        dcn = CrossNet(in_features=in_features, num_layers=num_layers)
        output = dcn(input)
        expected_output = torch.Tensor(
            [
                [2.4481, 2.2710],
                [-63.1721, -109.2410],
                [1.4030, 1.0054],
            ]
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_fx_script_cross_net(self) -> None:
        input = torch.randn(2, 3)
        dcn = CrossNet(in_features=3, num_layers=2)
        dcn(input)

        # dry-run to initialize lazy module
        gm = GraphModule(dcn, Tracer().trace(dcn))
        torch.jit.script(gm)


# unit test for Low Rank CrossNet: LowRankCrossNet
class TestLowRankCrossNet(unittest.TestCase):
    def test_cross_net_numercial_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        num_layers = 20
        in_features = 2
        input = torch.randn(batch_size, in_features)

        # test using vector for crossing
        dcn = LowRankCrossNet(
            in_features=in_features, num_layers=num_layers, low_rank=10
        )
        output = dcn(input)
        expected_output = torch.Tensor(
            [
                [-11.5000, -3.4863],
                [-0.2742, -0.3330],
                [249.6694, 117.3466],
            ]
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_fx_script_cross_net(self) -> None:
        input = torch.randn(2, 3)
        dcn = LowRankCrossNet(in_features=3, num_layers=2, low_rank=2)
        dcn(input)

        # dry-run to initialize lazy module
        gm = GraphModule(dcn, Tracer().trace(dcn))
        torch.jit.script(gm)


# unit test for Vector Version CrossNet: VectorCrossNet
class TestVectorCrossNet(unittest.TestCase):
    def test_cross_net_numercial_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        num_layers = 20
        in_features = 2
        input = torch.randn(batch_size, in_features)

        # test using vector for crossing
        dcn = VectorCrossNet(in_features=in_features, num_layers=num_layers)
        output = dcn(input)
        expected_output = torch.Tensor(
            [
                [1.8289e-04, -3.4827e-05],
                [-2.2084e02, 5.7615e01],
                [-1.3328e02, -1.7187e02],
            ]
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_fx_script_cross_net(self) -> None:
        input = torch.randn(2, 3)
        dcn = VectorCrossNet(in_features=3, num_layers=2)
        dcn(input)

        # dry-run to initialize lazy module
        gm = GraphModule(dcn, Tracer().trace(dcn))
        torch.jit.script(gm)


# unit test for Low Rank CrossNet with Mixture of Expert: LowRankMixtureCrossNet
class TestLowRankMixtureCrossNet(unittest.TestCase):
    def test_cross_net_numercial_forward(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        num_layers = 20
        in_features = 2
        input = torch.randn(batch_size, in_features)

        # test using vector for crossing
        dcn = LowRankMixtureCrossNet(
            in_features=in_features, num_layers=num_layers, num_experts=4, low_rank=10
        )
        output = dcn(input)
        expected_output = torch.Tensor(
            [
                [1.7045, -0.2848],
                [-2.5357, 0.5811],
                [-0.9467, -1.3091],
            ]
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_cross_net_numercial_forward_1_expert(self) -> None:
        torch.manual_seed(0)

        batch_size = 3
        num_layers = 20
        in_features = 2
        input = torch.randn(batch_size, in_features)

        # test using vector for crossing
        dcn = LowRankMixtureCrossNet(
            in_features=in_features, num_layers=num_layers, num_experts=1, low_rank=10
        )
        output = dcn(input)
        expected_output = torch.Tensor(
            [
                [3.9203, -0.2686],
                [-9.5767, 0.8621],
                [-2.5836, -1.8124],
            ]
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_fx_script_cross_net(self) -> None:
        input = torch.randn(2, 3)
        dcn = LowRankMixtureCrossNet(in_features=3, num_layers=2)
        dcn(input)

        # dry-run to initialize lazy module
        gm = GraphModule(dcn, Tracer().trace(dcn))
        torch.jit.script(gm)
