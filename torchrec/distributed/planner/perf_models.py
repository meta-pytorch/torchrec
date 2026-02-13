#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, List

from torchrec.distributed.planner.types import (
    Perf,
    PerfModel,
    ShardingOption,
    Storage,
    Topology,
)


class NoopPerfModel(PerfModel):
    """
    A no-op model that returns the maximum perf among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        perfs = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyrefly: ignore[bad-index, unsupported-operation]
                perfs[shard.rank] += cast(Perf, shard.perf).total

        return max(perfs)


class NoopStorageModel(PerfModel):
    """
    A no-op model that returns the maximum hbm usage among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        hbms = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyrefly: ignore[bad-index, unsupported-operation]
                hbms[shard.rank] += cast(Storage, shard.storage).hbm

        return max(hbms)
