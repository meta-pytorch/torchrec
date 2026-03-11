#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict

import torch
import torch.nn as nn
from torchrec.inference.modules import PredictModule, quantize_dense


class TestModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = nn.Linear(10, 1)
        self.linear1 = nn.Linear(1, 1)


class TestPredictModule(PredictModule):
    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.predict_module(*batch)


class PredictModulesTest(unittest.TestCase):
    def test_predict_module(self) -> None:
        module = TestModule()
        predict_module = TestPredictModule(module)

        module_state_dict = module.state_dict()
        predict_module_state_dict = predict_module.state_dict()

        self.assertEqual(module_state_dict.keys(), predict_module_state_dict.keys())

        for tensor0, tensor1 in zip(
            module_state_dict.values(), predict_module_state_dict.values()
        ):
            torch.testing.assert_close(tensor0, tensor1, rtol=0, atol=0)

    def test_dense_lowering(self) -> None:
        module = TestModule()
        predict_module = TestPredictModule(module)
        predict_module = quantize_dense(predict_module, torch.half)
        for param in predict_module.parameters():
            self.assertEqual(param.dtype, torch.half)
