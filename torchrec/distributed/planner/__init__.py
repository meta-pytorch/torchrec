#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Planner

The planner provides the specifications necessary for a module to be sharded,
considering the possible options to build an optimized plan.

The features includes:
    - generating all possible sharding options.
    - estimating perf and storage for every shard.
    - estimating peak memory usage to eliminate sharding plans that might OOM.
    - customizability for parameter constraints, partitioning, proposers, or performance
      modeling.
    - automatically building and selecting an optimized sharding plan.
"""

from torchrec.distributed.planner.planners import (  # noqa  # noqa
    EmbeddingPlannerBase,
    EmbeddingShardingPlanner,
)
from torchrec.distributed.planner.types import ParameterConstraints, Topology  # noqa
from torchrec.distributed.planner.utils import bytes_to_gb, sharder_name  # noqa
