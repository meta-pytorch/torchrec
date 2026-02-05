#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.distributed.utils import copy_to_device, init_parameters
from torchrec.fx.tracer import symbolic_trace
from torchrec.modules.feature_processor_ import (
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PositionWeightedModuleTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        pw = PositionWeightedModule(max_feature_length=10)

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

        jt = features["f1"]
        weighted_features = pw(jt)

        self.assertEqual(weighted_features.weights().size(), (3,))

        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weight"], 0, torch.tensor([0, 1, 0])
        )

        pw_f1 = weighted_features.weights().detach()
        self.assertTrue(torch.allclose(pw_f1_ref, pw_f1))

        position_weighted_module_gm = symbolic_trace(pw)
        position_weighted_module_gm_script = torch.jit.script(
            position_weighted_module_gm
        )

        weighted_features_gm_script = position_weighted_module_gm_script(jt)
        torch.testing.assert_close(
            weighted_features.values(), weighted_features_gm_script.values()
        )
        torch.testing.assert_close(
            weighted_features.lengths(), weighted_features_gm_script.lengths()
        )


class PositionWeightedCollectionModuleTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        position_weighted_module_collection = PositionWeightedModuleCollection(
            {"f1": 10, "f2": 10}
        )

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

        fp_kjt = position_weighted_module_collection(features)

        position_weighted_module_collection_gm = symbolic_trace(
            position_weighted_module_collection
        )
        position_weighted_module_collection_gm_script = torch.jit.script(
            position_weighted_module_collection_gm
        )
        fp_kjt_gm_script = position_weighted_module_collection_gm_script(features)

        torch.testing.assert_close(fp_kjt.values(), fp_kjt_gm_script.values())
        torch.testing.assert_close(fp_kjt.lengths(), fp_kjt_gm_script.lengths())
        torch.testing.assert_close(
            fp_kjt.length_per_key(), fp_kjt_gm_script.length_per_key()
        )

        empty_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[],
            values=torch.tensor([], dtype=torch.int32),
            lengths=torch.tensor([], dtype=torch.int32),
        )

        empty_fp_kjt = position_weighted_module_collection(empty_kjt)
        empty_fp_kjt_gm_script = position_weighted_module_collection_gm_script(
            empty_kjt
        )

        torch.testing.assert_close(
            empty_fp_kjt.values(), empty_fp_kjt_gm_script.values()
        )
        torch.testing.assert_close(
            empty_fp_kjt.lengths(), empty_fp_kjt_gm_script.lengths()
        )
        torch.testing.assert_close(
            empty_fp_kjt.length_per_key(), empty_fp_kjt_gm_script.length_per_key()
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    def test_copy(self) -> None:
        pwmc = PositionWeightedModuleCollection(
            max_feature_lengths={"f1": 10, "f2": 10},
            device=torch.device("cpu"),
        )

        self.assertTrue(
            all(param.device.type == "cpu" for param in pwmc.position_weights.values())
        )
        self.assertTrue(
            all(
                param.device.type == "cpu"
                for param in pwmc.position_weights_dict.values()
            )
        )

        res = copy_to_device(
            pwmc, current_device=torch.device("cpu"), to_device=torch.device("meta")
        )

        #  not a function.
        self.assertTrue(all(param.is_meta for param in res.position_weights.values()))
        self.assertTrue(
            #  is not a function.
            all(param.is_meta for param in res.position_weights_dict.values())
        )

    def test_to(self) -> None:
        pwmc = PositionWeightedModuleCollection(
            max_feature_lengths={"f1": 10, "f2": 10},
            device=torch.device("cpu"),
        )

        self.assertTrue(
            all(param.device.type == "cpu" for param in pwmc.position_weights.values())
        )
        self.assertTrue(
            all(
                param.device.type == "cpu"
                for param in pwmc.position_weights_dict.values()
            )
        )

        pwmc.to("meta")

        self.assertTrue(all(param.is_meta for param in pwmc.position_weights.values()))
        self.assertTrue(
            all(param.is_meta for param in pwmc.position_weights_dict.values())
        )

    def test_load_state_dict(self) -> None:
        values = torch.tensor([10, 11, 12, 20, 21, 22])
        lengths = torch.tensor([3, 3])
        kjt = KeyedJaggedTensor(
            keys=["feature1", "feature2"], values=values, lengths=lengths
        )

        # Step 1: Create module and observe initial state
        max_feature_lengths = {"feature1": 3, "feature2": 3}
        module = PositionWeightedModuleCollection(max_feature_lengths)

        # Before checkpoint loading, position_weights_dict is a element-wise reference of position_weights
        for f in ["feature1", "feature2"]:
            self.assertIs(
                module.position_weights[f],
                module.position_weights_dict[f],
            )

        output = module(kjt)
        expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertListEqual(output.weights().tolist(), expected)

        # Step 2: Simulate checkpoint loading with assign=True
        checkpoint = {
            "position_weights.feature1": torch.tensor([2.0, 3.0, 4.0]),
            "position_weights.feature2": torch.tensor([5.0, 6.0, 7.0]),
        }
        module.load_state_dict(checkpoint, strict=False, assign=True)

        # After checkpoint loading, position_weights_dict is a element-wise reference of position_weights
        for f in ["feature1", "feature2"]:
            self.assertIs(
                module.position_weights[f],
                module.position_weights_dict[f],
            )

        output = module(kjt)
        expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        self.assertListEqual(output.weights().tolist(), expected)
