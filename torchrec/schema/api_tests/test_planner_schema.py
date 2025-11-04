#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Dict, List, Optional, Union

import torch.distributed as dist
from torch import nn
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner, SortBy
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.proposers import GreedyProposer
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    Partitioner,
    PerfModel,
    Proposer,
    ShardEstimator,
    ShardingOption,
    ShardingPlan,
    Stats,
    StorageReservation,
    Topology,
)
from torchrec.distributed.types import ModuleSharder, PipelineType
from torchrec.schema.utils import is_signature_compatible


class StableEmbeddingShardingPlanner:
    def __init__(
        self,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
    ) -> None:
        pass

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlan:
        return ShardingPlan(plan={})

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        return ShardingPlan(plan={})


class StableEmbeddingEnumerator:
    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
    ) -> None:
        pass

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        return []

    def populate_estimates(self, sharding_options: List[ShardingOption]) -> None:
        pass


class StableGreedyPerfPartitioner:
    def __init__(
        self, sort_by: SortBy = SortBy.STORAGE, balance_modules: bool = False
    ) -> None:
        pass

    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        return []


class StableHeuristicalStorageReservation:
    def __init__(
        self,
        percentage: float,
        parameter_multiplier: float = 6.0,
        dense_tensor_estimate: Optional[int] = None,
    ) -> None:
        pass

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        return Topology(world_size=0, compute_device="cuda")


class StableGreedyProposer:
    def __init__(self, use_depth: bool = True, threshold: Optional[int] = None) -> None:
        pass

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        pass

    def propose(self) -> Optional[List[ShardingOption]]:
        return []

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        pass


class StableEmbeddingPerfEstimator:
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        is_inference: bool = False,
    ) -> None:
        pass

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        pass


class StableEmbeddingStorageEstimator:
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        pipeline_type: PipelineType = PipelineType.NONE,
        run_embedding_at_peak_memory: bool = False,
        is_inference: bool = False,
    ) -> None:
        pass

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        pass


class TestPlanner(unittest.TestCase):
    def test_planner(self) -> None:
        stable_planner_funcs = inspect.getmembers(
            StableEmbeddingShardingPlanner, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_planner_funcs:
            self.assertTrue(
                getattr(EmbeddingShardingPlanner, func_name, None) is not None
            )
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(EmbeddingShardingPlanner, func_name)),
                )
            )

    def test_enumerator(self) -> None:
        stable_enumerator_funcs = inspect.getmembers(
            StableEmbeddingEnumerator, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_enumerator_funcs:
            self.assertTrue(getattr(EmbeddingEnumerator, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(EmbeddingEnumerator, func_name)),
                )
            )

    def test_partitioner(self) -> None:
        stable_partitioner_funcs = inspect.getmembers(
            StableGreedyPerfPartitioner, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_partitioner_funcs:
            self.assertTrue(getattr(GreedyPerfPartitioner, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(GreedyPerfPartitioner, func_name)),
                )
            )

    def test_storage_reservation(self) -> None:
        stable_storage_reservation_funcs = inspect.getmembers(
            StableHeuristicalStorageReservation, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_storage_reservation_funcs:
            self.assertTrue(
                getattr(HeuristicalStorageReservation, func_name, None) is not None
            )
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(
                        getattr(HeuristicalStorageReservation, func_name)
                    ),
                )
            )

    def test_proposer(self) -> None:
        stable_proposer_funcs = inspect.getmembers(
            StableGreedyProposer, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_proposer_funcs:
            self.assertTrue(getattr(GreedyProposer, func_name, None) is not None)
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(GreedyProposer, func_name)),
                )
            )

    def test_perf_estimator(self) -> None:
        stable_perf_estimator_funcs = inspect.getmembers(
            StableEmbeddingPerfEstimator, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_perf_estimator_funcs:
            self.assertTrue(
                getattr(EmbeddingPerfEstimator, func_name, None) is not None
            )
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(EmbeddingPerfEstimator, func_name)),
                )
            )

    def test_storage_estimator(self) -> None:
        stable_storage_estimator_funcs = inspect.getmembers(
            StableEmbeddingStorageEstimator, predicate=inspect.isfunction
        )

        for func_name, stable_func in stable_storage_estimator_funcs:
            self.assertTrue(
                getattr(EmbeddingStorageEstimator, func_name, None) is not None
            )
            self.assertTrue(
                is_signature_compatible(
                    inspect.signature(stable_func),
                    inspect.signature(getattr(EmbeddingStorageEstimator, func_name)),
                )
            )
