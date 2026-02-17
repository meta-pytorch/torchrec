#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List, Optional, Type
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE, DEFAULT_PERF_ESTIMATOR
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.estimator import EmbeddingPerfEstimatorFactory
from torchrec.distributed.planner.proposers import (
    DynamicProgrammingProposer,
    EmbeddingOffloadScaleupProposer,
    GreedyProposer,
    GridSearchProposer,
    proposers_to_proposals_list,
    UniformProposer,
)
from torchrec.distributed.planner.shard_estimators import (
    _calculate_storage_specific_sizes,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    Perf,
    Proposer,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    CacheAlgorithm,
    CacheParams,
    CacheStatistics,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class MockProposer(Proposer):
    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        pass

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        pass

    def propose(self) -> Optional[List[ShardingOption]]:
        pass


class MockCacheStatistics(CacheStatistics):
    def __init__(self, expected_lookups: int, cacheability: float) -> None:
        self._expected_lookups = expected_lookups
        self._cacheability = cacheability

    @property
    def expected_lookups(self) -> int:
        return self._expected_lookups

    def expected_miss_rate(self, clf: float) -> float:
        return clf

    @property
    def cacheability(self) -> float:
        return self._cacheability


# Mocking _calculate_storage_specific_sizes to skip cache aux state accounting for
# simpler testing
def mock_calculate_storage_specific_sizes(
    storage: int,
    shape: torch.Size,
    shard_sizes: List[List[int]],
    sharding_type: str,
    optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
    is_inference: bool = False,
    clf: Optional[float] = None,
) -> List[int]:
    return _calculate_storage_specific_sizes(
        storage, shape, shard_sizes, sharding_type, optimizer_class, is_inference, None
    )


def make_sharding_option(
    name: str,
    raw_size: int,
    clf: Optional[float],
    perf: Optional[Perf] = None,
) -> ShardingOption:
    """
    Convenience factory method for creating a sharding option with a single shard.
    """
    return ShardingOption(
        name=name,
        tensor=torch.zeros(1),
        # pyrefly: ignore[bad-argument-type]
        module=("model", None),
        input_lengths=[],
        batch_size=8,
        sharding_type="row_wise",
        partition_by="DEVICE",
        compute_kernel="fused" if clf is None else "fused_uvm_caching",
        shards=[
            Shard(
                size=[1, raw_size],
                offset=[0, 0],
                perf=(
                    Perf(
                        fwd_compute=0,
                        fwd_comms=0,
                        bwd_compute=0,
                        bwd_comms=0,
                    )
                    if perf is None
                    else perf
                ),
            )
        ],
        cache_params=(
            None
            if clf is None
            else CacheParams(
                load_factor=clf,
            )
        ),
    )


class TestProposers(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = EmbeddingEnumerator(topology=topology, batch_size=BATCH_SIZE)
        self.greedy_proposer = GreedyProposer()
        self.uniform_proposer = UniformProposer()
        self.grid_search_proposer = GridSearchProposer()
        self.dynamic_programming_proposer = DynamicProgrammingProposer()
        self._sharding_types = [x.value for x in ShardingType]
        self.maxDiff = None

    def test_greedy_two_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]
        """
        GRID_SHARD only is available if specified by user in parameter constraints, however,
        adding parameter constraints does not work because of the non deterministic nature of
        _filter_sharding_types (set & set) operation when constraints are present. This means
        the greedy proposer will have a different order of sharding types on each test invocation
        which we cannot have a harcoded "correct" answer for. We mock the call to _filter_sharding_types
        to ensure the order of the sharding types list is always the same.
        """
        self.enumerator._filter_sharding_types = MagicMock(
            return_value=self._sharding_types
        )

        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        self.greedy_proposer.load(search_space)

        # simulate first five iterations:
        output = []
        for _ in range(5):
            proposal = cast(List[ShardingOption], self.greedy_proposer.propose())
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf.total for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.greedy_proposer.feedback(partitionable=True)

        expected_output = [
            [
                ("table_0", "row_wise", "fused"),
                ("table_1", "row_wise", "fused"),
            ],
            [
                ("table_0", "table_row_wise", "fused"),
                ("table_1", "row_wise", "fused"),
            ],
            [
                ("table_0", "grid_shard", "fused"),
                ("table_1", "row_wise", "fused"),
            ],
            [
                ("table_1", "row_wise", "fused"),
                ("table_0", "data_parallel", "dense"),
            ],
            [
                ("table_1", "table_row_wise", "fused"),
                ("table_0", "data_parallel", "dense"),
            ],
        ]

        self.assertEqual(expected_output, output)

        # Test threshold for early_stopping
        self.greedy_proposer._threshold = 10
        self.greedy_proposer.load(search_space)

        # With early stopping, after berf_perf_rating is assigned, after 10 iterations with
        # consecutive worse perf_rating, the returned proposal should be None.
        proposal = None
        for i in range(13):
            proposal = cast(List[ShardingOption], self.greedy_proposer.propose())
            self.greedy_proposer.feedback(partitionable=True, perf_rating=100 + i)
        self.assertEqual(self.greedy_proposer._best_perf_rating, 100)
        self.assertEqual(proposal, None)

    def test_uniform_three_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        mock_ebc_sharder = cast(
            ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder()
        )
        # TODO update this test for CW and TWCW sharding
        mock_ebc_sharder.sharding_types = MagicMock(
            return_value=[
                ShardingType.DATA_PARALLEL.value,
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]
        )

        self.maxDiff = None

        search_space = self.enumerator.enumerate(
            module=model, sharders=[mock_ebc_sharder]
        )
        self.uniform_proposer.load(search_space)

        output = []
        proposal = self.uniform_proposer.propose()
        while proposal:
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf.total for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.uniform_proposer.feedback(partitionable=True)
            proposal = self.uniform_proposer.propose()

        expected_output = [
            [
                (
                    "table_1",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_2",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_3",
                    "data_parallel",
                    "dense",
                ),
            ],
            [
                (
                    "table_1",
                    "table_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "table_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "table_wise",
                    "fused",
                ),
            ],
            [
                (
                    "table_1",
                    "row_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "row_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "row_wise",
                    "fused",
                ),
            ],
            [
                (
                    "table_1",
                    "table_row_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "table_row_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "table_row_wise",
                    "fused",
                ),
            ],
        ]

        self.assertEqual(expected_output, output)

    def test_grid_search_three_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        """
        GRID_SHARD only is available if specified by user in parameter constraints, however,
        adding parameter constraints does not work because of the non deterministic nature of
        _filter_sharding_types (set & set) operation when constraints are present, we mock the
        call to _filter_sharding_types to ensure the order of the sharding types list is always
        the same.
        """
        self.enumerator._filter_sharding_types = MagicMock(
            return_value=self._sharding_types
        )
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )

        """
        All sharding types but DP will have 3 possible compute kernels after pruning:
            - fused
            - fused_uvm_caching
            - fused_uvm
        DP will have 1 possible compute kernel: dense
        So the total number of pruned options will be:
            (num_sharding_types - 1) * 3 + 1 = 19
        """
        num_pruned_options = (len(ShardingType) - 1) * 3 + 1
        self.grid_search_proposer.load(search_space)
        for (
            sharding_options
        ) in self.grid_search_proposer._sharding_options_by_fqn.values():
            # number of sharding types after pruning is number of sharding types * 3
            # 3 compute kernels fused/dense, fused_uvm_caching, fused_uvm
            self.assertEqual(len(sharding_options), num_pruned_options)

        num_proposals = 0
        proposal = self.grid_search_proposer.propose()
        while proposal:
            self.grid_search_proposer.feedback(partitionable=True)
            proposal = self.grid_search_proposer.propose()
            num_proposals += 1

        self.assertEqual(num_pruned_options ** len(tables), num_proposals)

    def test_dynamic_programming_three_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )

        self.dynamic_programming_proposer.load(search_space)

        num_proposals = 0
        proposal = self.dynamic_programming_proposer.propose()
        GB = 1024 * 1024 * 1024
        storage_constraint = Topology(
            world_size=2, compute_device="cuda", hbm_cap=100 * GB, ddr_cap=1000 * GB
        )
        while proposal:
            self.dynamic_programming_proposer.feedback(
                partitionable=True, storage_constraint=storage_constraint
            )
            proposal = self.dynamic_programming_proposer.propose()
            num_proposals += 1
        self.assertEqual(2, num_proposals)

    def test_get_scalable_sharding_options(self) -> None:
        def make_so(
            name: str, clf: Optional[float], stats: Optional[CacheStatistics]
        ) -> ShardingOption:
            so = make_sharding_option(name, 1, clf)
            if clf:
                assert so.cache_params
                so.cache_params.stats = stats
            return so

        proposal = [
            make_so("fused", None, None),
            make_so("caching-no-stats", 0.5, None),
            make_so(
                "caching-stats",
                0.5,
                MockCacheStatistics(expected_lookups=1, cacheability=0.42),
            ),
            make_so(
                "caching-stats-no-data",
                0,
                MockCacheStatistics(expected_lookups=0, cacheability=0),
            ),
        ]
        got = EmbeddingOffloadScaleupProposer.get_scalable_sharding_options(proposal)
        want = [proposal[-2]]
        self.assertEqual(got, want)

    def test_allocate_budget(self) -> None:
        model = torch.tensor([[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]])
        got = EmbeddingOffloadScaleupProposer.clf_to_bytes(
            model, torch.tensor([0, 0.5, 1])
        )
        torch.testing.assert_close(got, torch.tensor([0, 4, 9], dtype=torch.float64))

        # Scenario 1, enough budget to scale everything to 1.0
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        fused_hbm_ceiling = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1.0)
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 100_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([2, 2, 2]),
        )
        torch.testing.assert_close(got, torch.tensor([1.0, 1.0, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        ).item()
        self.assertLess(increase, budget)

        # Scenario 2, limited budget, uniform scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        fused_hbm_ceiling = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1.0)
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 10_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([2, 2, 2]),
        )
        torch.testing.assert_close(got, torch.tensor([0.26667, 0.26667, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(increase, budget)

        # Scenario 3, limited budget, skewed scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        fused_hbm_ceiling = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1.0)
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 10_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([2, 4, 2]),
        )
        # increase is twice as much for table 2 (started at 0.1)
        torch.testing.assert_close(
            got, torch.tensor([0.1 + 0.11111, 0.1 + 2 * 0.11111, 1.0])
        )
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(int(increase), budget)

        # Scenario 4, multi-pass scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        fused_hbm_ceiling = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1.0)
        mins = torch.tensor([0.1, 0.3, 0.5])
        budget = 50_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([1, 2, 100]),
        )
        torch.testing.assert_close(got, torch.tensor([0.56667, 1.0, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(increase, budget)

        # Scenario 5, prefetch overhead causing early promotion
        # like scenario 4, but we set fused size to 80%, which saves enough memory
        # to promote all 3 to HBM inside the same budget.
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        fused_hbm_ceiling = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1.0) * 0.8
        )
        mins = torch.tensor([0.1, 0.3, 0.5])
        budget = 50_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            fused_hbm_ceiling=fused_hbm_ceiling,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([1, 2, 100]),
        )
        torch.testing.assert_close(got, torch.tensor([1.0, 1.0, 1.0]))
        self.assertLessEqual(
            fused_hbm_ceiling.sum().item(),
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum().item()
            + budget,
        )

    # pyrefly: ignore[implicit-import]
    @unittest.mock.patch(
        "torchrec.distributed.planner.shard_estimators._calculate_storage_specific_sizes",
        side_effect=mock_calculate_storage_specific_sizes,
    )
    def test_scaleup(self, _) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2_000_000,
                embedding_dim=10000,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(4)
        ]

        # Place first three tables into cache, 4th table leave on hbm. table_1 has a
        # larger cacheability score so budget should be skewed to scaling table_1 more
        # than table_0. table_2 is a deprecated feature we have no stats for (so
        # expected_lookups 0), we want to see that left at its original load factor,
        # i.e. doesn't participate in scaleup.
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.COLUMN_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MockCacheStatistics(expected_lookups=2, cacheability=0.2),
                ),
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.COLUMN_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MockCacheStatistics(expected_lookups=2, cacheability=0.5),
                ),
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.COLUMN_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.002,
                    stats=MockCacheStatistics(expected_lookups=0, cacheability=0.5),
                ),
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.COLUMN_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED.value],
                cache_params=CacheParams(),
            ),
        }

        GB = 1024 * 1024 * 1024
        storage_constraint = Topology(
            world_size=2, compute_device="cuda", hbm_cap=100 * GB, ddr_cap=1000 * GB
        )
        # Ignoring table_2, the remainder require 224GB if all placed on HBM. We only
        # have 200GB, so we can't promote both uvm tables. Initial plan needs uses 90GB,
        # with 110GB of available budget.
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        enumerator = EmbeddingEnumerator(
            topology=storage_constraint,
            batch_size=BATCH_SIZE,
            constraints=constraints,
            estimator=[
                EmbeddingPerfEstimatorFactory.create(
                    DEFAULT_PERF_ESTIMATOR,
                    topology=storage_constraint,
                    constraints=constraints,
                ),
                EmbeddingStorageEstimator(
                    topology=storage_constraint,
                    constraints=constraints,
                ),
            ],
        )
        search_space = enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        proposer = EmbeddingOffloadScaleupProposer()
        proposer.load(search_space, enumerator=enumerator)

        proposal = proposer.propose()
        best_plan = None
        best_perf = 1e99
        proposals = -1
        while proposal is not None:
            proposals += 1
            mem = sum(so.total_storage.hbm for so in proposal)
            # simple perf model, assume partitioner gives a lowest score around 150GB of memory.
            perf = abs(mem - (150 * GB))
            plan = {
                "mem": mem,
                "proposal": [
                    (
                        candidate.name,
                        candidate.compute_kernel,
                        (
                            candidate.cache_params.load_factor
                            if candidate.cache_params
                            else None
                        ),
                    )
                    for candidate in proposal
                ],
            }
            if perf < best_perf:
                best_plan = plan
                best_perf = perf
            proposer.feedback(
                partitionable=True,
                plan=proposal,
                perf_rating=perf,
                storage_constraint=storage_constraint,
            )
            proposal = proposer.propose()

        self.assertEqual(proposals, 16)
        self.assertEqual(
            best_plan,
            {
                # 146GB, close to target of 150GB
                "mem": 157178896800,
                # table 1 has been scaled up 2.5x more than table 0 (vs original 0.1)
                # which aligns with their different cacheability scores
                # table_2 has been left alone (deprecated feature, expected zero lookups in stats)
                "proposal": [
                    ("table_0", "fused_uvm_caching", 0.3173336386680603),
                    ("table_1", "fused_uvm_caching", 0.6433340907096863),
                    ("table_2", "fused_uvm_caching", 0.002),
                    ("table_3", "fused", None),
                ],
            },
        )

    def test_promote_high_prefetch_overheaad_table_to_hbm(self) -> None:
        def expect_sharding_option(
            so: List[ShardingOption],
            names: List[str],
            total_hbms: List[int],
            total_ddrs: List[int],
            kernels: List[str],
        ) -> None:
            for s, name, hbm, ddr, kernel in zip(
                so, names, total_hbms, total_ddrs, kernels
            ):
                self.assertEqual(s.name, name)
                self.assertEqual(s.total_storage.hbm, hbm)
                self.assertEqual(s.total_storage.ddr, ddr)
                self.assertEqual(s.compute_kernel, kernel)

        def mock_storage_estimator_func(so: List[ShardingOption]) -> None:
            # This mock storage estimator will give table2 a consistent penalty
            # for using UVM caching
            for s in so:
                size = s.shards[0].size[0] * s.shards[0].size[1]
                if s.compute_kernel == "fused_uvm_caching":
                    assert s.cache_params
                    assert s.cache_params.load_factor
                    penalty = 50 if s.name == "table-2" else 0
                    s.shards[0].storage = Storage(
                        ddr=size,
                        hbm=int(size * s.cache_params.load_factor) + penalty,
                    )
                else:
                    s.shards[0].storage = Storage(
                        ddr=0,
                        hbm=size,
                    )

        mock_enumerator = MagicMock()
        mock_enumerator.populate_estimates.side_effect = mock_storage_estimator_func

        # Run1: table-2 is getting penalized but HBM is still lower when offloaded.
        # so it won't change after promotion
        p = EmbeddingOffloadScaleupProposer()
        proposal1 = [
            make_sharding_option("table-1", 50, None),
            make_sharding_option("table-2", 90, 0.3),
            make_sharding_option("table-3", 100, 0.5),
        ]
        mock_storage_estimator_func(proposal1)
        p.load(search_space=proposal1, enumerator=mock_enumerator)
        expect_sharding_option(
            p.starting_proposal,
            ["table-1", "table-2", "table-3"],
            [50, 77, 50],
            [0, 90, 100],
            ["fused", "fused_uvm_caching", "fused_uvm_caching"],
        )

        # Run1: table-2 will have higher HBM after penalty (90 * 0.8 + 50 > 90). So promote it.
        p = EmbeddingOffloadScaleupProposer()
        proposal2 = [
            make_sharding_option("table-1", 50, None),
            make_sharding_option("table-2", 90, 0.8),
            make_sharding_option("table-3", 100, 0.9),
        ]
        mock_storage_estimator_func(proposal2)
        p.load(search_space=proposal2, enumerator=mock_enumerator)
        expect_sharding_option(
            p.starting_proposal,
            ["table-1", "table-2", "table-3"],
            [50, 90, 90],
            [0, 0, 100],
            ["fused", "fused", "fused_uvm_caching"],
        )

    # pyrefly: ignore[implicit-import]
    @unittest.mock.patch(
        "torchrec.distributed.planner.shard_estimators._calculate_storage_specific_sizes",
        side_effect=mock_calculate_storage_specific_sizes,
    )
    def test_budget_shrink(self, _) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2_000_000,
                embedding_dim=10000,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        constraints = {
            "table_0": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MockCacheStatistics(expected_lookups=2, cacheability=0.2),
                ),
            ),
        }

        GB = 1024 * 1024 * 1024
        storage_constraint = Topology(
            world_size=1, compute_device="cuda", hbm_cap=100 * GB, ddr_cap=1000 * GB
        )
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        enumerator = EmbeddingEnumerator(
            topology=storage_constraint,
            batch_size=BATCH_SIZE,
            constraints=constraints,
            estimator=[
                EmbeddingPerfEstimatorFactory.create(
                    DEFAULT_PERF_ESTIMATOR,
                    topology=storage_constraint,
                    constraints=constraints,
                ),
                EmbeddingStorageEstimator(
                    topology=storage_constraint,
                    constraints=constraints,
                ),
            ],
        )
        search_space = enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        proposer = EmbeddingOffloadScaleupProposer()
        proposer.load(search_space, enumerator=enumerator)

        proposal = proposer.propose()
        best_plan = None
        best_perf = 1e99
        proposals = -1
        initial_mem = None
        # pyrefly: ignore[bad-assignment]
        while proposal is not None:
            proposals += 1
            mem = sum(so.total_storage.hbm for so in proposal)
            if initial_mem is None:
                initial_mem = mem
            # Budget given constraints:
            # unscaled plan=7.47 GB, cache scale up budget=92.53 GB, peak scale up budget need=67.06 GB, exploring plans of size [7.47, 74.53] GB
            #
            # Simple perf model, assume partitioner gives a lowest score at 7.9GB, and
            # anything larger than 8GB fails to partition. This is very hard to hit when
            # exploring the larger [7.47, 100] range with limited iterations without
            # shrinkage.
            perf = abs(mem - (7.9 * GB))
            partitionable = mem < 8 * GB
            if perf < best_perf:
                best_plan = mem
                best_perf = perf
            proposer.feedback(
                partitionable=partitionable,
                plan=proposal,
                perf_rating=perf if partitionable else None,
                storage_constraint=storage_constraint,
            )
            proposal = proposer.propose()

        self.assertEqual(proposals, 16)
        self.assertNotEqual(initial_mem, best_plan, "couldn't find a better plan")
        # goal is 7.9, we get very close
        self.assertEqual(best_plan, 7.9028974287211895 * GB)

    def test_proposers_to_proposals_list(self) -> None:
        def make_mock_proposal(name: str) -> List[ShardingOption]:
            return [
                ShardingOption(
                    name=name,
                    tensor=torch.zeros(1),
                    # pyrefly: ignore[bad-argument-type]
                    module=("model", None),
                    input_lengths=[],
                    batch_size=8,
                    sharding_type="row_wise",
                    partition_by="DEVICE",
                    compute_kernel="fused",
                    shards=[],
                )
            ]

        mock_proposer_1 = MockProposer()
        mock_proposer_1_sharding_options = [
            make_mock_proposal("p1so1"),
            make_mock_proposal("p1so2"),
            make_mock_proposal("p1so1"),
            None,
        ]
        mock_proposer_1.propose = MagicMock(
            side_effect=mock_proposer_1_sharding_options
        )

        mock_proposer_2 = MockProposer()
        mock_proposer_2_sharding_options = [
            make_mock_proposal("p2so1"),
            make_mock_proposal("p2so1"),
            make_mock_proposal("p1so2"),
            make_mock_proposal("p2so2"),
            None,
        ]
        mock_proposer_2.propose = MagicMock(
            side_effect=mock_proposer_2_sharding_options
        )

        mock_proposer_3 = MockProposer()
        mock_proposer_3_sharding_options = [
            make_mock_proposal("p3so1"),
            make_mock_proposal("p2so1"),
            make_mock_proposal("p3so2"),
            None,
        ]
        mock_proposer_3.propose = MagicMock(
            side_effect=mock_proposer_3_sharding_options
        )

        proposers_list: List[Proposer] = [
            mock_proposer_1,
            mock_proposer_2,
            mock_proposer_3,
        ]

        proposals_list = proposers_to_proposals_list(proposers_list, search_space=[])
        proposals_list_names = []

        for sharding_option in proposals_list:
            proposals_list_names.append(sharding_option[0].name)

        expected_list_names = ["p1so1", "p1so2", "p2so1", "p2so2", "p3so1", "p3so2"]

        self.assertEqual(proposals_list_names, expected_list_names)

    def test_embedding_offload_scaleup_proposer_uses_fused_kernel_when_possible(
        self,
    ) -> None:
        def mock_storage_estimator_func(so: List[ShardingOption]) -> None:
            # This mock storage estimator will give all tables a penalty
            # for using UVM caching.
            for s in so:
                size = s.shards[0].size[0] * s.shards[0].size[1]
                if s.compute_kernel == "fused_uvm_caching":
                    assert s.cache_params
                    assert s.cache_params.load_factor
                    penalty = 50
                    s.shards[0].storage = Storage(
                        ddr=size,
                        hbm=int(size * s.cache_params.load_factor) + penalty,
                    )
                else:
                    s.shards[0].storage = Storage(
                        ddr=0,
                        hbm=size,
                    )

        mock_enumerator = MagicMock()
        mock_enumerator.populate_estimates.side_effect = mock_storage_estimator_func

        p = EmbeddingOffloadScaleupProposer()
        search_space = [
            # We should pick the first option since it has a fused_uvm_caching kernel
            # even though it doesn't have the best perf.
            make_sharding_option("table-1", 5000, 0.1, Perf(9, 9, 9, 9)),
            make_sharding_option("table-1", 5000, None, Perf(1, 1, 1, 1)),
            make_sharding_option("table-1", 1, None, Perf(1, 1, 1, 1)),
            # Neither option has fused_uvm_caching, so we should pick the one with best perf.
            make_sharding_option("table-2", 10, None, Perf(9, 9, 9, 9)),
            make_sharding_option("table-2", 50, None, Perf(1, 1, 1, 1)),
        ]

        mock_storage_estimator_func(search_space)
        p.load(search_space=search_space, enumerator=mock_enumerator)

        self.assertEqual(p.starting_proposal[0].name, "table-1")
        self.assertEqual(p.starting_proposal[0].compute_kernel, "fused_uvm_caching")
        self.assertEqual(p.starting_proposal[0].total_storage.hbm, 550)
        self.assertEqual(p.starting_proposal[0].total_storage.ddr, 5000)

        self.assertEqual(p.starting_proposal[1].name, "table-2")
        self.assertEqual(p.starting_proposal[1].compute_kernel, "fused")
        self.assertEqual(p.starting_proposal[1].total_storage.hbm, 50)
        self.assertEqual(p.starting_proposal[1].total_storage.ddr, 0)

    # pyrefly: ignore[implicit-import]
    @unittest.mock.patch("torchrec.distributed.planner.proposers.logger")
    def test_build_proposal_from_sharding_options(self, mock_logger: MagicMock) -> None:
        table_4_sharding_option = make_sharding_option("table-4", 1, 0.1)
        assert table_4_sharding_option.cache_params  # appease pyre
        table_4_sharding_option.cache_params.algorithm = CacheAlgorithm.LFU

        sharding_options_by_fqn = {
            # Case 1: Only one option, use it even though it isn't fused_uvm_caching. Don't log anything.
            "table-1": [make_sharding_option("table-1", 1, None)],
            # Case 2: Multiple options, 1+ with fused_uvm_caching, use the first with fused_uvm_caching. Log warning.
            "table-2": [
                make_sharding_option("table-2", 1, None),
                make_sharding_option("table-2", 1, 0.1),
                make_sharding_option("table-2", 1, None),
                make_sharding_option("table-2", 1, 0.1),
            ],
            # Case 3: Multiple options, none with fused_uvm_caching, use the first one. Log warning.
            "table-3": [
                make_sharding_option("table-3", 1, None),
                make_sharding_option("table-3", 1, None),
            ],
            # Case 4: One option, but it's using LFU cache constraints. Use it, but log error.
            "table-4": [table_4_sharding_option],
        }

        proposer = EmbeddingOffloadScaleupProposer()
        proposal = proposer._build_proposal_from_sharding_options(
            sharding_options_by_fqn
        )

        self.assertEqual(len(proposal), 4)
        self.assertEqual(mock_logger.warning.call_count, 2)
        self.assertEqual(mock_logger.error.call_count, 1)

        # Case 1
        self.assertEqual(proposal[0], sharding_options_by_fqn["table-1"][0])

        # Case 2
        self.assertEqual(proposal[1], sharding_options_by_fqn["table-2"][1])
        self.assertRegex(
            mock_logger.warning.call_args_list[0].args[0],
            r"^EmbeddingOffloadScaleupProposer - ignored \d+ sharding options for table table-2",
        )

        # Case 3
        self.assertEqual(proposal[2], sharding_options_by_fqn["table-3"][0])
        self.assertRegex(
            mock_logger.warning.call_args_list[1].args[0],
            r"^EmbeddingOffloadScaleupProposer - ignored \d+ sharding options for table table-3",
        )

        # Case 4
        self.assertEqual(proposal[3], sharding_options_by_fqn["table-4"][0])
        self.assertIn(
            "EmbeddingOffloadScaleupProposer - proposer only supports LRU cache algorithm",
            mock_logger.error.call_args_list[0].args[0],
        )


if __name__ == "__main__":
    unittest.main()
