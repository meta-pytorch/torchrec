#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import MagicMock

from torchrec.distributed.planner.perf_models import NoopPerfModel, NoopStorageModel
from torchrec.distributed.planner.types import (
    Perf,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)


class TestPerfModels(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = Topology(world_size=2, compute_device="cuda")
        self.tables = [
            ShardingOption(
                name=MagicMock(),
                tensor=MagicMock(),
                module=MagicMock(),
                input_lengths=MagicMock(),
                batch_size=MagicMock(),
                sharding_type=MagicMock(),
                partition_by=MagicMock(),
                compute_kernel=MagicMock(),
                shards=[
                    Shard(
                        size=MagicMock(),
                        offset=MagicMock(),
                        rank=rank,
                        perf=Perf(
                            fwd_compute=2 - rank,
                            fwd_comms=0,
                            bwd_compute=0,
                            bwd_comms=0,
                        ),
                        storage=Storage(hbm=100 * (rank + 1), ddr=0),
                    ),
                ],
            )
            for rank in range(2)
        ]

    def test_noop_perf_model(self) -> None:
        perf_model = NoopPerfModel(self.topology)
        perf_rating = perf_model.rate(self.tables)
        self.assertEqual(perf_rating, 2)

    def test_noop_storage_model(self) -> None:
        perf_model = NoopStorageModel(self.topology)
        perf_rating = perf_model.rate(self.tables)
        self.assertEqual(perf_rating, 200)
