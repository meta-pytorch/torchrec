#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch
from torchrec.modules import hash_mc_evictions
from torchrec.modules.hash_mc_evictions import HashZchEvictionConfig
from torchrec.sparse.jagged_tensor import JaggedTensor


class HashMcEvictionsCompatibilityTest(unittest.TestCase):
    def test_scorers_without_torchscript(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "hash_mc_evictions_without_torchscript", hash_mc_evictions.__file__
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(torch.jit.__dict__), patch.dict(
            sys.modules, {spec.name: module}
        ):
            torch.jit.__dict__.pop("script", None)
            spec.loader.exec_module(module)

        config = module.HashZchEvictionConfig(features=["f1"], single_ttl=24)
        self.assertEqual(config.features, ["f1"])
        self.assertIsNone(config.per_feature_ttl)
        single_ttl_scorer = module.HashZchSingleTtlScorer(config)
        per_feature_scorer = module.HashZchPerFeatureTtlScorer(
            module.HashZchEvictionConfig(
                features=["f1", "f2"], per_feature_ttl=[24, 48]
            )
        )
        feature = JaggedTensor(
            values=torch.arange(5),
            lengths=torch.tensor([2, 2, 1]),
            weights=torch.tensor([4, 1]),
        )
        with patch("time.time", return_value=36000000):
            torch.testing.assert_close(
                single_ttl_scorer.gen_score(feature, torch.device("cpu")),
                torch.tensor([10024] * 5, dtype=torch.int32),
            )
            torch.testing.assert_close(
                per_feature_scorer.gen_score(feature, torch.device("cpu")),
                torch.tensor([10024, 10024, 10024, 10024, 10048], dtype=torch.int32),
            )
            self.assertEqual(single_ttl_scorer.gen_threshold(), 10000)
            self.assertEqual(per_feature_scorer.gen_threshold(), 10000)

    @unittest.skipUnless(hasattr(torch.jit, "script"), "TorchScript is unavailable")
    def test_config_remains_scriptable(self) -> None:
        @torch.jit.script
        def get_ttl(config: HashZchEvictionConfig) -> int:
            ttl = config.single_ttl
            return 0 if ttl is None else ttl

        self.assertEqual(get_ttl(HashZchEvictionConfig(["f1"], single_ttl=24)), 24)
        self.assertEqual(get_ttl(HashZchEvictionConfig(["f1"])), 0)
