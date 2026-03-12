#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.fx import Tracer
from torchrec.modules.feature_processor import (
    PositionWeightedModule,
    PositionWeightedProcessor,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PositionWeightedModuleTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightedModule(features_max_length)

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
        features = features.to_dict()

        weighted_features = pw(features)

        self.assertEqual(weighted_features["f1"].weights().size(), (3,))
        self.assertEqual(weighted_features["f2"].weights().size(), (5,))

        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weights.f1"], 0, torch.tensor([0, 1, 0])
        )
        pw_f1 = weighted_features["f1"].weights().detach()
        torch.testing.assert_close(pw_f1_ref, pw_f1, rtol=1e-05, atol=1e-08)

        pw_f2_ref = torch.gather(
            pw.state_dict()["position_weights.f2"], 0, torch.tensor([0, 0, 0, 1, 2])
        )
        pw_f2 = weighted_features["f2"].weights().detach()
        torch.testing.assert_close(pw_f2_ref, pw_f2, rtol=1e-05, atol=1e-08)

    def test_fx_script_PositionWeightedModule(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightedModule(features_max_length)

        gm = torch.fx.GraphModule(pw, Tracer().trace(pw))
        torch.jit.script(gm)

    def test_populate_weights_PositionWeightedProcessor(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightedProcessor(features_max_length)

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

        weighted_features = pw(features)

        self.assertEqual(weighted_features["f1"].weights().size(), (3,))
        self.assertEqual(weighted_features["f2"].weights().size(), (5,))

        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weights.f1"], 0, torch.tensor([0, 1, 0])
        )
        pw_f1 = weighted_features["f1"].weights().detach()
        torch.testing.assert_close(pw_f1_ref, pw_f1, rtol=1e-05, atol=1e-08)

        pw_f2_ref = torch.gather(
            pw.state_dict()["position_weights.f2"], 0, torch.tensor([0, 0, 0, 1, 2])
        )
        pw_f2 = weighted_features["f2"].weights().detach()
        torch.testing.assert_close(pw_f2_ref, pw_f2, rtol=1e-05, atol=1e-08)

    def test_fx_script_PositionWeightedProcessor(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightedProcessor(features_max_length)

        gm = torch.fx.GraphModule(pw, Tracer().trace(pw))
        torch.jit.script(gm)
