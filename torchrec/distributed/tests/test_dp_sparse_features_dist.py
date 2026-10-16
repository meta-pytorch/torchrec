#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.distributed.sharding.dp_sharding import DpSparseFeaturesDist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class DpSparseFeaturesDistTest(unittest.TestCase):
    def test_returns_the_features_with_length_per_key_cached(self) -> None:
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.arange(7),
            lengths=torch.tensor([2, 0, 1, 3, 1, 0]),
        )
        self.assertIsNone(features.length_per_key_or_none())

        out = DpSparseFeaturesDist()(features).wait().wait()

        self.assertIs(out, features)
        self.assertEqual(out.length_per_key_or_none(), [3, 4])
        self.assertTrue(torch.equal(out.values(), torch.arange(7)))
