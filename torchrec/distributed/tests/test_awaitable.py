#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.distributed.types import Awaitable


class AwaitableInstance(Awaitable[torch.Tensor]):
    def __init__(self) -> None:
        super().__init__()

    def _wait_impl(self) -> torch.Tensor:
        return torch.FloatTensor([1.0, 2.0, 3.0])


class AwaitableTests(unittest.TestCase):
    def test_callback(self) -> None:
        awaitable = AwaitableInstance()
        #  `(ret: Any) -> int`.
        awaitable.callbacks.append(lambda ret: 2 * ret)
        self.assertTrue(
            torch.allclose(awaitable.wait(), torch.FloatTensor([2.0, 4.0, 6.0]))
        )

    def test_callback_chained(self) -> None:
        awaitable = AwaitableInstance()
        #  `(ret: Any) -> int`.
        awaitable.callbacks.append(lambda ret: 2 * ret)
        awaitable.callbacks.append(lambda ret: ret**2)
        self.assertTrue(
            torch.allclose(awaitable.wait(), torch.FloatTensor([4.0, 16.0, 36.0]))
        )
