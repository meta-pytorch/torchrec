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
from torchrec.modules.activation import SwishLayerNorm


class TestActivation(unittest.TestCase):
    def test_swish_takes_float(self) -> None:
        m = SwishLayerNorm([3, 4])
        input = torch.randn(2, 3, 4)
        output = m(input)
        norm = torch.nn.LayerNorm([3, 4])
        ref_output = input * torch.sigmoid(norm(input))
        torch.testing.assert_close(output, ref_output, rtol=1e-05, atol=1e-08)

    def test_fx_script_swish(self) -> None:
        m = SwishLayerNorm(10)

        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)
