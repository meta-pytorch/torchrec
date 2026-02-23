#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, List, Optional

import torch
from torch import nn
from torchrec import EmbeddingBagCollection, EmbeddingConfig
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.perf_models import NoopPerfModel
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner, extract_plan
from torchrec.distributed.planner.proposers import EmbeddingOffloadScaleupProposer
from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PlanLoader,
    PlannerError,
    PlannerErrorType,
    Shard,
    ShardingOption,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    EmbeddingModuleShardingPlan,
    KeyValueParams,
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TWvsRWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TestEmbeddingShardingPlanner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.planner = EmbeddingShardingPlanner(topology=self.topology)

    def test_tw_rank_assignment(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[bad-argument-type, missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[TWSharder()])
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]
        for rank_list in ranks:
            for rank in rank_list:
                self.assertTrue(0 <= rank <= 1, f"Rank {rank} not in [0,1]")

    def test_tw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[bad-argument-type, missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0], [0], [1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_hidden_rw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[bad-argument-type, missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0], [0, 1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_never_fit(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10000000,
                embedding_dim=10000000,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        with self.assertRaises(PlannerError) as context:
            # pyrefly: ignore[bad-argument-type, missing-argument]
            self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        self.assertEqual(
            context.exception.error_type, PlannerErrorType.INSUFFICIENT_STORAGE
        )

        # since it has negative storage_constraint
        self.assertEqual(self.planner._num_proposals, 0)

    def test_fail_then_rerun(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=4096,
                embedding_dim=128,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        with self.assertRaises(PlannerError) as context:
            # pyrefly: ignore[bad-argument-type, missing-argument]
            self.planner.plan(module=model, sharders=[TWSharder()])
        self.assertEqual(
            context.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS
        )

        # pyrefly: ignore[bad-argument-type, missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0, 1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_no_sharders(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[])

        self.assertEqual(sharding_plan, ShardingPlan({}))


class TestEmbeddingShardingPlannerWithConstraints(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.constraints = {
            "table_0": ParameterConstraints(
                enforce_hbm=True,
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                ),
                feature_names=self.tables[0].feature_names,
            ),
            "table_1": ParameterConstraints(
                enforce_hbm=False,
                stochastic_rounding=True,
                feature_names=self.tables[1].feature_names,
            ),
            "table_2": ParameterConstraints(
                bounds_check_mode=BoundsCheckMode.FATAL,
                feature_names=self.tables[2].feature_names,
            ),
            "table_3": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.1,
                    reserved_memory=1.0,
                    precision=DataType.FP16,
                ),
                feature_names=self.tables[3].feature_names,
            ),
        }
        self.planner = EmbeddingShardingPlanner(
            topology=self.topology, constraints=self.constraints
        )

    def test_fused_paramters_from_constraints(self) -> None:
        model = TestSparseNN(tables=self.tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=get_default_sharders())

        expected_fused_params = {
            "table_0": (
                CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=None,
                    reserved_memory=None,
                    precision=None,
                ),
                True,
                None,
                None,
            ),
            "table_1": (None, False, True, None),
            "table_2": (None, None, None, BoundsCheckMode.FATAL),
            "table_3": (
                CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.1,
                    reserved_memory=1.0,
                    precision=DataType.FP16,
                ),
                None,
                None,
                None,
            ),
        }

        table_names = ["table_" + str(i) for i in range(4)]
        for table in table_names:
            parameter_sharding = cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            )[table]
            self.assertEqual(
                (
                    parameter_sharding.cache_params,
                    parameter_sharding.enforce_hbm,
                    parameter_sharding.stochastic_rounding,
                    parameter_sharding.bounds_check_mode,
                ),
                expected_fused_params[table],
            )

    def test_passing_info_through_constraints(self) -> None:
        model = TestSparseNN(tables=self.tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[missing-argument]
        _ = self.planner.plan(module=model, sharders=get_default_sharders())

        best_plan: Optional[List[ShardingOption]] = self.planner._best_plan
        self.assertIsNotNone(best_plan)

        for table, constraint, sharding_option in zip(
            self.tables, self.constraints.values(), best_plan
        ):
            self.assertEqual(table.name, sharding_option.name)

            self.assertEqual(table.feature_names, sharding_option.feature_names)
            self.assertEqual(table.feature_names, constraint.feature_names)

            self.assertEqual(constraint.cache_params, sharding_option.cache_params)
            self.assertEqual(constraint.enforce_hbm, sharding_option.enforce_hbm)
            self.assertEqual(
                constraint.stochastic_rounding, sharding_option.stochastic_rounding
            )
            self.assertEqual(
                constraint.bounds_check_mode, sharding_option.bounds_check_mode
            )
            self.assertEqual(constraint.is_weighted, sharding_option.is_weighted)


class TestEmbeddingShardingHashPlannerContextInputs(unittest.TestCase):

    def setUp(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        module = EmbeddingBagCollection(
            tables=[eb_config],
            is_weighted=False,
            device=torch.device(
                "meta"
            ),  # Using meta device for now since only getting search space
        )
        sharders = [EmbeddingBagCollectionSharder()]

        self.topology = Topology(
            local_world_size=8,
            world_size=1,
            compute_device="cuda",
        )
        self.batch_size = 128
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=self.batch_size
        )
        # pyrefly: ignore[bad-argument-type]
        self.enumerator.enumerate(module, sharders)

        self.storage_reservation = HeuristicalStorageReservation(percentage=0.15)
        self.perf_model = NoopPerfModel(topology=self.topology)
        self.constraints = {"table1": ParameterConstraints()}

        self.storage_reservation.reserve(
            topology=self.topology,
            batch_size=self.batch_size,
            module=module,
            # pyrefly: ignore[bad-argument-type]
            sharders=sharders,
            constraints=self.constraints,
        )

    def test_hash_equality(self) -> None:
        planner1 = EmbeddingShardingPlanner(
            topology=self.topology,
            batch_size=self.batch_size,
            enumerator=self.enumerator,
            storage_reservation=self.storage_reservation,
            performance_model=self.perf_model,
            constraints=self.constraints,
        )

        planner2 = EmbeddingShardingPlanner(
            topology=self.topology,
            batch_size=self.batch_size,
            enumerator=self.enumerator,
            storage_reservation=self.storage_reservation,
            performance_model=self.perf_model,
            constraints=self.constraints,
        )

        self.assertEqual(
            planner1.hash_planner_context_inputs(),
            planner2.hash_planner_context_inputs(),
            "Hashes should be equal for identical planners",
        )

    def test_hash_inequality(self) -> None:
        planner1 = EmbeddingShardingPlanner(
            topology=self.topology,
            batch_size=self.batch_size,
            enumerator=self.enumerator,
            storage_reservation=self.storage_reservation,
            performance_model=self.perf_model,
            constraints=self.constraints,
        )

        different_topology = Topology(
            local_world_size=8,
            world_size=2,  # Different world size
            compute_device="cuda",
        )

        planner2 = EmbeddingShardingPlanner(
            topology=different_topology,  # Different topology
            batch_size=self.batch_size * 2,  # Different batch size
            enumerator=self.enumerator,
            storage_reservation=self.storage_reservation,
            performance_model=self.perf_model,
            constraints=self.constraints,
        )

        self.assertNotEqual(
            planner1.hash_planner_context_inputs(),
            planner2.hash_planner_context_inputs(),
            "Hashes should be different for different planners",
        )


class AutoSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [
            k.value
            for k in EmbeddingComputeKernel
            if k is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL
        ]


class TestAutoPlannerWithScaleupProposer(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2,
            hbm_cap=1024 * 1024 * 2,
            compute_device=compute_device,
        )
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.constraints = {
            f"table_{i}": ParameterConstraints(
                # Just needs to be non-None for ScaleupProposer to work.
                cache_params=CacheParams(algorithm=CacheAlgorithm.LRU),
            )
            for i in range(4)
        }
        self.planner = EmbeddingShardingPlanner(
            topology=self.topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=self.constraints,
        )

    def test_auto_sharder_solution(self) -> None:
        model = TestSparseNN(tables=self.tables, sparse_device=torch.device("meta"))
        # pyrefly: ignore[bad-argument-type, missing-argument]
        sharding_plan = self.planner.plan(module=model, sharders=[AutoSharder()])
        expected_ranks = [[0, 1], [0, 1], [0, 1], [0, 1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]
        compute_kernels = {
            param_shard.compute_kernel
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        }

        self.assertEqual(sorted(expected_ranks), sorted(ranks))
        self.assertSetEqual(
            {EmbeddingComputeKernel.FUSED_UVM_CACHING.value}, compute_kernels
        )

    def test_planner_with_virtual_table(self) -> None:
        table_count = 4
        tables = [
            EmbeddingConfig(
                num_embeddings=1_125_899_902_955_520,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                use_virtual_table=True,
                total_num_buckets=3_991_680,
            )
            for i in range(table_count // 2)
        ] + [
            EmbeddingConfig(
                num_embeddings=100_000,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(table_count // 2, table_count)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        constraints = {
            **{
                f"table_{i}": ParameterConstraints(
                    sharding_types=["row_wise"],
                    compute_kernels=["dram_virtual_table"],
                )
                for i in range(table_count // 2)
            },
            **{
                f"table_{i}": ParameterConstraints(
                    cache_params=CacheParams(algorithm=CacheAlgorithm.LRU)
                )
                for i in range(table_count // 2, table_count)
            },
        }

        topology = Topology(
            world_size=2,
            hbm_cap=1024 * 1024 * 1024 * 2,
            ddr_cap=1024 * 1024 * 1024 * 256,
            compute_device="cuda",
        )

        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=constraints,
        )

        # pyrefly: ignore[missing-argument]
        sharding_plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[EmbeddingCollectionSharder()],
        )

        for table_index in range(4):
            # pyrefly: ignore[bad-index]
            shards = sharding_plan.plan["sparse.ec"][
                f"table_{table_index}"
            ].sharding_spec.shards
            self.assertEqual(len(shards), 2)
            self.assertEqual(shards[0].shard_offsets, [0, 0])
            self.assertEqual(
                shards[0].shard_sizes,
                [562949951477760 if table_index < 2 else 50_000, 64],
            )
            self.assertEqual(
                shards[1].shard_offsets,
                [562949951477760 if table_index < 2 else 50_000, 0],
            )
            self.assertEqual(
                shards[1].shard_sizes,
                [562949951477760 if table_index < 2 else 50_000, 64],
            )
        stats: List[str] = cast(EmbeddingStats, planner._stats[0])._stats_table
        # L1 cache size is 64GB per shard and L2 cache size is 128MB per shard per table
        self.assertTrue(
            any(
                "dram_virtual_table: HBM: 0.001 GB, DDR: 0.0 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any(
                "fused_uvm_caching: HBM: 0.011 GB, DDR: 0.048 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any("Max HBM: 0.006 GB on ranks [0, 1]" in line for line in stats)
        )
        self.assertTrue(
            any("Max HBM: 0.006 GB on ranks [0, 1]" in line for line in stats)
        )

        constraints = {
            **{
                f"table_{i}": ParameterConstraints(
                    sharding_types=["row_wise"],
                    compute_kernels=["dram_virtual_table"],
                    key_value_params=KeyValueParams(
                        l2_cache_size=64, max_l1_cache_size=128
                    ),
                )
                for i in range(table_count // 2)
            },
            **{
                f"table_{i}": ParameterConstraints(
                    cache_params=CacheParams(algorithm=CacheAlgorithm.LRU),
                )
                for i in range(table_count // 2, table_count)
            },
        }

        topology = Topology(
            world_size=2,
            hbm_cap=1024 * 1024 * 1024 * 2,
            ddr_cap=1024 * 1024 * 1024 * 256,
            compute_device="cuda",
        )

        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=constraints,
        )
        # pyrefly: ignore[missing-argument]
        sharding_plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[EmbeddingCollectionSharder()],
        )

        expected_ranks = [[0, 1], [0, 1], [0, 1], [0, 1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ec"]
            ).values()
        ]
        compute_kernels = {
            param_shard.compute_kernel
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ec"]
            ).values()
        }
        self.assertEqual(sorted(expected_ranks), sorted(ranks))
        self.assertSetEqual(
            {
                EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            },
            compute_kernels,
        )

        for table_index in range(4):
            # pyrefly: ignore[bad-index]
            shards = sharding_plan.plan["sparse.ec"][
                f"table_{table_index}"
            ].sharding_spec.shards
            self.assertEqual(len(shards), 2)
            self.assertEqual(shards[0].shard_offsets, [0, 0])
            self.assertEqual(
                shards[0].shard_sizes,
                [562949951477760 if table_index < 2 else 50_000, 64],
            )
            self.assertEqual(
                shards[1].shard_offsets,
                [562949951477760 if table_index < 2 else 50_000, 0],
            )
            self.assertEqual(
                shards[1].shard_sizes,
                [562949951477760 if table_index < 2 else 50_000, 64],
            )
        stats: List[str] = cast(EmbeddingStats, planner._stats[0])._stats_table
        # L1 cache size is 64GB per shard and L2 cache size is 128MB per shard per table
        self.assertTrue(
            any(
                "dram_virtual_table: HBM: 0.501 GB, DDR: 0.0 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any(
                "fused_uvm_caching: HBM: 0.011 GB, DDR: 0.048 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any("Max HBM: 0.256 GB on ranks [0, 1]" in line for line in stats)
        )
        self.assertTrue(
            any("Min HBM: 0.256 GB on ranks [0, 1]" in line for line in stats)
        )

        constraints = {
            **{
                f"table_{i}": ParameterConstraints(
                    sharding_types=["row_wise"],
                    compute_kernels=["dram_virtual_table"],
                    key_value_params=KeyValueParams(
                        l2_cache_size=64, max_l1_cache_size=128
                    ),
                )
                for i in range(table_count // 2)
            },
            **{
                f"table_{i}": ParameterConstraints(
                    cache_params=CacheParams(algorithm=CacheAlgorithm.LRU),
                )
                for i in range(table_count // 2, table_count)
            },
        }

        topology = Topology(
            world_size=2,
            hbm_cap=1024 * 1024 * 1024 * 2,
            ddr_cap=1024 * 1024 * 1024 * 256,
            compute_device="cuda",
        )

        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=constraints,
        )
        # pyrefly: ignore[missing-argument]
        sharding_plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[EmbeddingCollectionSharder()],
        )

        expected_ranks = [[0, 1], [0, 1], [0, 1], [0, 1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ec"]
            ).values()
        ]
        compute_kernels = {
            param_shard.compute_kernel
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ec"]
            ).values()
        }
        self.assertEqual(sorted(expected_ranks), sorted(ranks))
        self.assertSetEqual(
            {
                EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            },
            compute_kernels,
        )

        tables = [
            EmbeddingConfig(
                num_embeddings=10000,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                use_virtual_table=True,
                total_num_buckets=10,
            )
            for i in range(table_count // 2)
        ] + [
            EmbeddingConfig(
                num_embeddings=100_000,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(table_count // 2, table_count)
        ]

        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=constraints,
        )

        #  L1 cache size > size of embedding table * default cache load factor

        # pyrefly: ignore[missing-argument]
        sharding_plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[EmbeddingCollectionSharder()],
        )
        for table_index in range(4):
            # pyrefly: ignore[bad-index]
            shards = sharding_plan.plan["sparse.ec"][
                f"table_{table_index}"
            ].sharding_spec.shards
            self.assertEqual(len(shards), 2)
            self.assertEqual(shards[0].shard_offsets, [0, 0])
            self.assertEqual(
                shards[0].shard_sizes,
                [5000 if table_index < 2 else 50_000, 64],
            )
            self.assertEqual(
                shards[1].shard_offsets,
                [5000 if table_index < 2 else 50_000, 0],
            )
            self.assertEqual(
                shards[1].shard_sizes,
                [5000 if table_index < 2 else 50_000, 64],
            )
        stats: List[str] = cast(EmbeddingStats, planner._stats[0])._stats_table
        # L1 cache size of 64GB > size of embedding table * cache load factor. We use the smaller value.
        # L2 cache size is 128MB per shard per table
        self.assertTrue(
            any(
                "dram_virtual_table: HBM: 0.002 GB, DDR: 0.0 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any(
                "fused_uvm_caching: HBM: 0.011 GB, DDR: 0.048 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any("Max HBM: 0.007 GB on ranks [0, 1]" in line for line in stats)
        )
        self.assertTrue(
            any("Min HBM: 0.007 GB on ranks [0, 1]" in line for line in stats)
        )

        # Override cache load factor
        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=EmbeddingOffloadScaleupProposer(),
            constraints=constraints,
        )
        # pyrefly: ignore[missing-argument]
        sharding_plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                EmbeddingCollectionSharder(fused_params={"cache_load_factor": 0.5})
            ],
        )
        for table_index in range(4):
            # pyrefly: ignore[bad-index]
            shards = sharding_plan.plan["sparse.ec"][
                f"table_{table_index}"
            ].sharding_spec.shards
            self.assertEqual(len(shards), 2)
            self.assertEqual(shards[0].shard_offsets, [0, 0])
            self.assertEqual(
                shards[0].shard_sizes,
                [5000 if table_index < 2 else 50_000, 64],
            )
            self.assertEqual(
                shards[1].shard_offsets,
                [5000 if table_index < 2 else 50_000, 0],
            )
            self.assertEqual(
                shards[1].shard_sizes,
                [5000 if table_index < 2 else 50_000, 64],
            )
        stats: List[str] = cast(EmbeddingStats, planner._stats[0])._stats_table
        # L1 cache size of 64GB > size of embedding table * cache load factor. We use the smaller value.
        # L2 cache size is 128MB per shard per table
        self.assertTrue(
            any(
                "dram_virtual_table: HBM: 0.005 GB, DDR: 0.0 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any(
                "fused_uvm_caching: HBM: 0.027 GB, DDR: 0.048 GB" in line
                for line in stats
            )
        )
        self.assertTrue(
            any("Max HBM: 0.016 GB on ranks [0, 1]" in line for line in stats)
        )
        self.assertTrue(
            any("Min HBM: 0.016 GB on ranks [0, 1]" in line for line in stats)
        )


class MockPlanLoader(PlanLoader):
    """Mock PlanLoader implementation for testing."""

    def __init__(
        self,
        loaded_sharding_options: Optional[Dict[int, ShardingOption]] = None,
        context_hash: Optional[str] = None,
        plan_id: str = "test_plan_123",
    ) -> None:
        self._loaded_sharding_options = loaded_sharding_options
        self._context_hash = context_hash
        self._plan_id = plan_id

    def load(self) -> Optional[Dict[int, ShardingOption]]:
        return self._loaded_sharding_options

    def plan_context_hash(self) -> Optional[str]:
        return self._context_hash

    def get_plan_id(self) -> str:
        return self._plan_id


class TestPlanLoaderIntegration(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)  # Reduced to 2 tables for simplicity
        ]
        self.constraints = {
            "table_0": ParameterConstraints(
                enforce_hbm=True,
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                ),
                feature_names=self.tables[0].feature_names,
            ),
            "table_1": ParameterConstraints(
                enforce_hbm=False,
                stochastic_rounding=True,
                feature_names=self.tables[1].feature_names,
            ),
        }
        self.model = TestSparseNN(
            tables=self.tables, sparse_device=torch.device("meta")
        )

    def test_plan_loader_with_valid_plan(self) -> None:
        """Test EmbeddingShardingPlanner with PlanLoader that provides a valid plan."""
        # First, create a planner without loader to generate a baseline plan
        baseline_planner = EmbeddingShardingPlanner(
            topology=self.topology, constraints=self.constraints
        )
        # pyrefly: ignore[missing-argument]
        baseline_plan = baseline_planner.plan(
            module=self.model, sharders=get_default_sharders()
        )

        # Extract the best plan from baseline planner
        best_plan = baseline_planner._best_plan
        self.assertIsNotNone(best_plan)

        # Create loaded sharding options map from the best plan
        loaded_sharding_options = {}
        for so in best_plan:
            # Modify the shards to simulate a loaded plan with different shard assignments
            modified_shards = [
                Shard(
                    size=shard.size,
                    offset=shard.offset,
                    storage=shard.storage,
                    perf=shard.perf,
                    rank=(
                        1 - shard.rank if shard.rank is not None else None
                    ),  # Flip ranks
                )
                for shard in so.shards
            ]
            loaded_so = ShardingOption(
                name=so.name,
                tensor=so.tensor,
                module=so.module,
                input_lengths=so.input_lengths,
                batch_size=so.batch_size,
                compute_kernel=so.compute_kernel,
                sharding_type=so.sharding_type,
                partition_by=so.partition_by,
                shards=modified_shards,
                cache_params=so.cache_params,
                enforce_hbm=so.enforce_hbm,
                stochastic_rounding=so.stochastic_rounding,
                bounds_check_mode=so.bounds_check_mode,
                feature_names=so.feature_names,
            )
            loaded_sharding_options[so.storage_hash()] = loaded_so

        # Create mock plan loader with matching context hash
        context_hash = baseline_planner.hash_planner_context_inputs_str()
        mock_loader = MockPlanLoader(
            loaded_sharding_options=loaded_sharding_options,
            context_hash=context_hash,
        )

        # Create planner with plan loader
        planner_with_loader = EmbeddingShardingPlanner(
            topology=self.topology,
            constraints=self.constraints,
            plan_loader=mock_loader,
        )

        # Plan with loader should use the loaded plan
        # pyrefly: ignore[missing-argument]
        loaded_plan = planner_with_loader.plan(
            module=self.model, sharders=get_default_sharders()
        )

        # Verify the plan was loaded (should have flipped rank assignments)
        self.assertIsNotNone(loaded_plan)
        self.assertEqual(len(loaded_plan.plan), len(baseline_plan.plan))

        # Check that ranks were actually flipped in the loaded plan
        for module_name, module_plan in loaded_plan.plan.items():
            baseline_module_plan = baseline_plan.plan[module_name]
            for param_name, param_sharding in cast(
                EmbeddingModuleShardingPlan, module_plan
            ).items():
                baseline_param_sharding = cast(
                    EmbeddingModuleShardingPlan, baseline_module_plan
                )[param_name]
                # The ranks should be different (flipped) from baseline
                self.assertNotEqual(param_sharding.ranks, baseline_param_sharding.ranks)

    def test_plan_loader_with_context_mismatch(self) -> None:
        """Test EmbeddingShardingPlanner with PlanLoader that has mismatched context hash."""
        # Create mock plan loader with different context hash
        mock_loader = MockPlanLoader(
            loaded_sharding_options={},
            context_hash="mismatched_hash",
        )

        # Create planner with plan loader
        planner_with_loader = EmbeddingShardingPlanner(
            topology=self.topology,
            constraints=self.constraints,
            plan_loader=mock_loader,
        )

        # Planning should raise PlannerError due to context mismatch
        with self.assertRaises(PlannerError) as context:
            # pyrefly: ignore[missing-argument]
            planner_with_loader.plan(module=self.model, sharders=get_default_sharders())

        self.assertEqual(
            context.exception.error_type,
            PlannerErrorType.PLANNER_INPUT_CONTEXT_MISMATCH,
        )
        self.assertIn("planner input mismatch", str(context.exception))

    def test_plan_loader_with_no_loaded_options(self) -> None:
        """Test EmbeddingShardingPlanner with PlanLoader that returns no loaded options."""
        # First get the correct context hash
        baseline_planner = EmbeddingShardingPlanner(
            topology=self.topology, constraints=self.constraints
        )
        # pyrefly: ignore[missing-argument]
        baseline_planner.plan(module=self.model, sharders=get_default_sharders())
        context_hash = baseline_planner.hash_planner_context_inputs_str()

        # Create mock plan loader with no loaded options but matching context
        mock_loader = MockPlanLoader(
            loaded_sharding_options=None,
            context_hash=context_hash,
        )

        # Create planner with plan loader
        planner_with_loader = EmbeddingShardingPlanner(
            topology=self.topology,
            constraints=self.constraints,
            plan_loader=mock_loader,
        )

        # Planning should succeed and generate a new plan (no loading)
        # pyrefly: ignore[missing-argument]
        loaded_plan = planner_with_loader.plan(
            module=self.model, sharders=get_default_sharders()
        )

        # Verify a plan was generated
        self.assertIsNotNone(loaded_plan)
        self.assertTrue(len(loaded_plan.plan) > 0)


class TestExtractPlan(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.constraints = {
            "table_0": ParameterConstraints(
                enforce_hbm=True,
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                ),
                feature_names=self.tables[0].feature_names,
            ),
            "table_1": ParameterConstraints(
                enforce_hbm=False,
                stochastic_rounding=True,
                feature_names=self.tables[1].feature_names,
            ),
            "table_2": ParameterConstraints(
                bounds_check_mode=BoundsCheckMode.FATAL,
                feature_names=self.tables[2].feature_names,
            ),
            "table_3": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.1,
                    reserved_memory=1.0,
                    precision=DataType.FP16,
                ),
                feature_names=self.tables[3].feature_names,
            ),
        }
        self.planner = EmbeddingShardingPlanner(
            topology=self.topology, constraints=self.constraints
        )
        self.model = TestSparseNN(
            tables=self.tables, sparse_device=torch.device("meta")
        )
        # pyrefly: ignore[missing-argument]
        self.sharding_plan = self.planner.plan(
            module=self.model, sharders=get_default_sharders()
        )

    def _create_loaded_sharding_options_map(
        self, best_plan: List[ShardingOption]
    ) -> Dict[int, ShardingOption]:
        """Creates a loaded sharding options map from enumerated sharding options."""
        loaded_map = {}
        for so in best_plan:
            sharding_options = ShardingOption(
                name=so.name,
                tensor=so.tensor,
                module=so.module,
                input_lengths=so.input_lengths,
                sharding_type=so.sharding_type,
                batch_size=so.batch_size,
                partition_by=so.partition_by,
                compute_kernel=so.compute_kernel,
                shards=so.shards,
                is_pooled=so.is_pooled,
                feature_names=so.feature_names,
                cache_params=so.cache_params,
            )

            loaded_map[so.storage_hash()] = sharding_options

        return loaded_map

    def test_extract_plan_success(self) -> None:
        """Test successful extraction of plan."""
        # pyrefly: ignore[missing-attribute]
        enumerated_plan = self.planner._enumerator.last_stored_search_space
        best_plan = none_throws(self.planner._best_plan)
        loaded_sharding_options = self._create_loaded_sharding_options_map(best_plan)

        result = extract_plan(enumerated_plan, loaded_sharding_options)

        self.assertEqual(len(result), len(best_plan))

        for i, result_so in enumerate(result):
            expected_so = best_plan[i]
            self.assertEqual(result_so.name, expected_so.name)
            self.assertEqual(result_so.tensor.shape, expected_so.tensor.shape)
            self.assertEqual(result_so.tensor.dtype, expected_so.tensor.dtype)
            self.assertEqual(result_so.tensor.device, expected_so.tensor.device)
            self.assertEqual(result_so.module, expected_so.module)
            self.assertEqual(result_so.input_lengths, expected_so.input_lengths)
            self.assertEqual(result_so.batch_size, expected_so.batch_size)
            self.assertEqual(result_so.compute_kernel, expected_so.compute_kernel)
            self.assertEqual(result_so.sharding_type, expected_so.sharding_type)
            self.assertEqual(result_so.partition_by, expected_so.partition_by)
            self.assertEqual(result_so.shards, expected_so.shards)
            self.assertEqual(result_so.is_pooled, expected_so.is_pooled)
            self.assertEqual(result_so.feature_names, expected_so.feature_names)
            self.assertEqual(result_so.cache_params, expected_so.cache_params)

    def test_extract_plan_duplicate_storage_hash_error(self) -> None:
        """Test extract_plan failure when duplicate storage hashes exist."""
        # Create search space with duplicate storage hashes by modifying sharding options
        # to have the same storage hash
        # pyrefly: ignore[missing-attribute]
        enumerated_plan = self.planner._enumerator.last_stored_search_space
        best_plan = none_throws(self.planner._best_plan)
        loaded_sharding_options = self._create_loaded_sharding_options_map(best_plan)

        # Create a search space with duplicate storage hashes by duplicating first option
        duplicate_search_space = [
            enumerated_plan[0],
            enumerated_plan[0],
        ]  # Same option twice

        with self.assertRaises(PlannerError) as context:
            extract_plan(duplicate_search_space, loaded_sharding_options)

        self.assertEqual(
            context.exception.error_type, PlannerErrorType.PLAN_LOADING_FAILED
        )
        self.assertIn("Found a duplicate storage hash", str(context.exception))

    def test_extract_plan_empty_search_space(self) -> None:
        """Test extract_plan with empty search space."""
        result = extract_plan([], {})
        self.assertEqual(result, [])

    def test_extract_plan_empty_loaded_options(self) -> None:
        """Test extract_plan with empty loaded options but non-empty search space."""
        # pyrefly: ignore[missing-attribute]
        enumerated_plan = self.planner._enumerator.last_stored_search_space

        # When loaded options is empty, extract_plan should return empty list
        # This is actually the correct behavior - no matching options means no extracted options
        result = extract_plan(enumerated_plan, {})
        self.assertEqual(result, [])

    def test_extract_plan_excess_loaded_options(self) -> None:
        """Test extract_plan when loaded options contain more entries than search space."""
        # pyrefly: ignore[missing-attribute]
        enumerated_plan = self.planner._enumerator.last_stored_search_space
        best_plan = none_throws(self.planner._best_plan)
        loaded_sharding_options = self._create_loaded_sharding_options_map(best_plan)

        extra_so = ShardingOption(
            name="extra_table",
            tensor=torch.tensor([1, 2, 3], device=torch.device("meta")),
            module=("extra_table.test", torch.nn.Module()),
            input_lengths=[100],
            batch_size=128,
            compute_kernel="fused",
            sharding_type=ShardingType.TABLE_WISE.value,
            partition_by="uniform",
            shards=[Shard(size=[100, 64], offset=[0, 0])],
            feature_names=["extra_feature"],
        )
        loaded_sharding_options[99999] = extra_so  # Arbitrary hash that won't match

        with self.assertRaises(PlannerError) as context:
            extract_plan(enumerated_plan, loaded_sharding_options)

        self.assertEqual(
            context.exception.error_type, PlannerErrorType.PLAN_LOADING_FAILED
        )
        self.assertIn("not all search space is covered", str(context.exception))

    def test_extract_plan_properties_preservation(self) -> None:
        """Test that extract_plan preserves all non-shard properties from search space."""
        # pyrefly: ignore[missing-attribute]
        enumerated_plan = self.planner._enumerator.last_stored_search_space
        best_plan = none_throws(self.planner._best_plan)
        loaded_sharding_options = self._create_loaded_sharding_options_map(best_plan)

        # Modify loaded options to have different shards but keep other properties
        for loaded_so in loaded_sharding_options.values():
            # Change the shard data to verify only shards are updated
            loaded_so.shards = [
                Shard(size=[200, 128], offset=[0, 0], rank=0)  # Different shard
            ]

        result = extract_plan(enumerated_plan, loaded_sharding_options)

        # Verify that result has search space properties but loaded shards
        for result_so in result:
            # Find the matching search space option by storage hash
            search_so = next(
                so
                for so in enumerated_plan
                if so.storage_hash() == result_so.storage_hash()
            )
            loaded_so = loaded_sharding_options[result_so.storage_hash()]

            # Properties from search space should be preserved
            self.assertEqual(result_so.name, search_so.name)
            self.assertEqual(result_so.compute_kernel, search_so.compute_kernel)
            self.assertEqual(result_so.sharding_type, search_so.sharding_type)
            self.assertEqual(result_so.batch_size, search_so.batch_size)
            self.assertEqual(result_so.feature_names, search_so.feature_names)

            # Shards should come from loaded options
            self.assertEqual(result_so.shards, loaded_so.shards)
            self.assertEqual(len(result_so.shards), 1)
            self.assertEqual(result_so.shards[0].size, [200, 128])
