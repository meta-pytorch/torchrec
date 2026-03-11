#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
import torch.fx
from torch.testing import FileCheck  # @manual
from torchrec.distributed.types import LazyAwaitable
from torchrec.fx import symbolic_trace


class TestTracer(unittest.TestCase):
    def test_trace_async_module(self) -> None:
        class NeedWait(LazyAwaitable[torch.Tensor]):
            def __init__(self, obj: torch.Tensor) -> None:
                super().__init__()
                self._obj = obj

            def _wait_impl(self) -> torch.Tensor:
                return self._obj + 3

        class MyAsyncModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input) -> LazyAwaitable[torch.Tensor]:
                return NeedWait(input + 2)

        # Test automated LazyAwaitable type `wait()`
        class AutoModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sparse = MyAsyncModule()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.add(self.sparse(input), input * 10)

        auto_model = AutoModel()
        auto_gm = symbolic_trace(auto_model)
        FileCheck().check("+ 2").check("NeedWait").check("* 10").run(auto_gm.code)

        input = torch.randn(3, 4)
        ref_out = auto_model(input)
        traced_out = auto_gm(input)
        torch.testing.assert_close(ref_out, traced_out, rtol=0, atol=0)
