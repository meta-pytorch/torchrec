#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchPerFeatureTtlScorer,
    HashZchSingleTtlScorer,
)
from torchrec.sparse.jagged_tensor import JaggedTensor


class TestEvictionScorer(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires CUDA device",
    )
    def test_single_ttl_scorer(self) -> None:
        scorer = HashZchSingleTtlScorer(
            config=HashZchEvictionConfig(features=["f1"], single_ttl=24)
        )

        jt = JaggedTensor(
            values=torch.arange(0, 5, dtype=torch.int64),
            lengths=torch.tensor([2, 2, 1], dtype=torch.int64),
        )

        with patch("time.time") as mock_time:
            mock_time.return_value = 36000000  # hour 10000
            score = scorer.gen_score(jt, device=torch.device("cuda"))
            self.assertTrue(
                torch.equal(
                    score,
                    torch.tensor([10024, 10024, 10024, 10024, 10024], device="cuda"),
                ),
                f"{torch.unique(score)=}",
            )

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires CUDA device",
    )
    def test_per_feature_ttl_scorer(self) -> None:
        scorer = HashZchPerFeatureTtlScorer(
            config=HashZchEvictionConfig(
                features=["f1", "f2"], per_feature_ttl=[24, 48]
            )
        )

        jt = JaggedTensor(
            values=torch.arange(0, 5, dtype=torch.int64),
            lengths=torch.tensor([2, 2, 1], dtype=torch.int64),
            weights=torch.tensor([4, 1], dtype=torch.int64),
        )

        with patch("time.time") as mock_time:
            mock_time.return_value = 36000000  # hour 10000
            score = scorer.gen_score(jt, device=torch.device("cuda"))
            self.assertTrue(
                torch.equal(
                    score,
                    torch.tensor([10024, 10024, 10024, 10024, 10048], device="cuda"),
                ),
                f"{torch.unique(score)=}",
            )
