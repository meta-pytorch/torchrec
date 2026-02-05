#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.test_utils import skip_if_asan

# @manual=//torchrec/github/examples/retrieval:two_tower_retrieval_lib
from ..two_tower_retrieval import infer


class InferTest(unittest.TestCase):
    @skip_if_asan
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_infer_function(self) -> None:
        infer(
            embedding_dim=16,
            layer_sizes=[16],
            world_size=2,
        )
