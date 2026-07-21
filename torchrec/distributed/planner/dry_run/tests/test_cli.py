#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchrec.distributed.planner.dry_run import cli
from torchrec.distributed.planner.dry_run.types import DryRunResult

# Small synthetic model so the real OSS planner runs quickly offline.
_SMALL_MODEL_ARGS = [
    "--num-tables",
    "2",
    "--num-embeddings",
    "100",
    "--embedding-dim",
    "16",
]


class DryRunCLITest(unittest.TestCase):
    def test_run_returns_dry_run_result_per_sku(self) -> None:
        args = cli.build_arg_parser().parse_args(
            ["--sku-list", "H100,GB200", "--world-size", "2", *_SMALL_MODEL_ARGS]
        )
        results = cli.run(args)
        self.assertEqual(set(results), {"H100", "GB200"})
        for sku in ("H100", "GB200"):
            self.assertIsInstance(results[sku], DryRunResult)
            self.assertTrue(results[sku].success, results[sku].planner_failure_reason)
            self.assertTrue(results[sku].request_fingerprint)

    def test_main_prints_report_and_exits_zero(self) -> None:
        rc = cli.main(["--sku-list", "H100", "--world-size", "2", *_SMALL_MODEL_ARGS])
        self.assertEqual(rc, 0)

    def test_local_world_size_defaults_to_world_size(self) -> None:
        args = cli.build_arg_parser().parse_args(
            ["--sku-list", "H100", "--world-size", "4", *_SMALL_MODEL_ARGS]
        )
        model, sharders = cli.build_model_and_sharders(args)
        request = cli.build_request(args, model, sharders)
        self.assertEqual(request.local_world_size, 4)
        self.assertEqual(request.sku_list, ["H100"])
