#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import cast

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.proposers import UniformProposer
from torchrec.distributed.planner.storage_reservations import (
    FixedAbsoluteStorageReservation,
)
from torchrec.distributed.planner.tests.uvm_cache_padding_test_utils import (
    _hbm_delta_by_shard,
    _mixed_dimension_tables,
    _shard_key,
    _uvm_cache_constraints,
    CWUVMCachingSharder,
    enable_uvm_cache_planner_knobs,
    FixedRankPartitioner,
    RWUVMCachingSharder,
    TWUVMCachingSharder,
    uvm_cache_padding_knob,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PlannerError,
    ShardingOption,
    Topology,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ModuleSharder, ShardingType
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig


def _plan_without_padding(
    tables: list[EmbeddingBagConfig],
    constraints: dict[str, ParameterConstraints],
    topology: Topology,
    sharder: ModuleSharder[nn.Module],
    ranks_by_table: dict[str, list[int]] | None = None,
) -> list[ShardingOption]:
    planner = EmbeddingShardingPlanner(
        topology=topology,
        proposer=UniformProposer() if ranks_by_table is not None else None,
        partitioner=(
            FixedRankPartitioner(ranks_by_table) if ranks_by_table is not None else None
        ),
        constraints=constraints,
        storage_reservation=FixedAbsoluteStorageReservation(0),
    )
    with uvm_cache_padding_knob(False):
        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[sharder],
        )
    return planner.get_selected_options()


class TestUVMCachePaddingGrouping(unittest.TestCase):
    def setUp(self) -> None:
        enable_uvm_cache_planner_knobs(self)

    def test_table_prefetch_true_overrides_global_without_padding(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=1000,
                embedding_dim=8,
                name="table_small",
                feature_names=["feature_small"],
            ),
            EmbeddingBagConfig(
                num_embeddings=1000,
                embedding_dim=160,
                name="table_large",
                feature_names=["feature_large"],
            ),
        ]
        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.5,
                    prefetch_pipeline=True,
                ),
            )
            for table in tables
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])
        topology = Topology(
            world_size=1,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        sharder = TWUVMCachingSharder(
            fused_params={
                "cache_load_factor": 0.5,
                "prefetch_pipeline": False,
            }
        )

        planner.plan(
            module=model,
            sharders=[cast(ModuleSharder[torch.nn.Module], sharder)],
        )

        selected_options = planner.get_selected_options()
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_options,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], sharder),
            ),
        )
        self.assertEqual(2, len(selected_options))
        for option in selected_options:
            self.assertTrue(none_throws(option.cache_params).prefetch_pipeline)
            for shard in option.shards:
                self.assertGreater(shard.cache_weight_bytes, 0)
                self.assertEqual(
                    0,
                    hbm_delta_by_shard[_shard_key(option.name, shard)],
                )

    def test_table_prefetch_false_overrides_global_prefetch_for_padding(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=1000,
                embedding_dim=8,
                name="table_small",
                feature_names=["feature_small"],
            ),
            EmbeddingBagConfig(
                num_embeddings=1000,
                embedding_dim=160,
                name="table_large",
                feature_names=["feature_large"],
            ),
        ]
        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[ShardingType.TABLE_WISE.value],
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.5,
                    prefetch_pipeline=False,
                ),
            )
            for table in tables
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])
        topology = Topology(
            world_size=1,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        sharder = TWUVMCachingSharder(
            fused_params={
                "cache_load_factor": 0.5,
                "prefetch_pipeline": True,
            }
        )

        planner.plan(
            module=model,
            sharders=[cast(ModuleSharder[torch.nn.Module], sharder)],
        )

        selected_option_list = planner.get_selected_options()
        selected_options = {option.name: option for option in selected_option_list}
        small_option = selected_options["table_small"]
        small_shard = small_option.shards[0]
        large_shard = selected_options["table_large"].shards[0]
        expected_small_padding = small_shard.cache_weight_bytes * (160 // 8 - 1)
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_option_list,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], sharder),
            ),
        )

        self.assertEqual(
            False,
            none_throws(small_option.cache_params).prefetch_pipeline,
        )
        self.assertEqual(
            expected_small_padding,
            hbm_delta_by_shard[_shard_key("table_small", small_shard)],
        )
        self.assertEqual(
            0,
            hbm_delta_by_shard[_shard_key("table_large", large_shard)],
        )

    def test_uvm_cache_dimension_padding_is_rank_local(self) -> None:
        tables = _mixed_dimension_tables(
            [
                ("small_r0", 8),
                ("large_r0", 16),
                ("small_r1", 8),
                ("large_r1", 160),
            ]
        )
        constraints = _uvm_cache_constraints(tables, ShardingType.TABLE_WISE.value)
        topology = Topology(
            world_size=2,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        ranks_by_table = {
            "small_r0": [0],
            "large_r0": [0],
            "small_r1": [1],
            "large_r1": [1],
        }
        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=UniformProposer(),
            partitioner=FixedRankPartitioner(ranks_by_table),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
        )

        selected_option_list = planner.get_selected_options()
        options = {option.name: option for option in selected_option_list}
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_option_list,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], TWUVMCachingSharder()),
                ranks_by_table,
            ),
        )
        for rank, suffix in ((0, "r0"), (1, "r1")):
            small_option = options[f"small_{suffix}"]
            small = small_option.shards[0]
            large = options[f"large_{suffix}"].shards[0]
            cache_dtype_bytes = small_option.tensor.element_size()
            cached_rows, remainder = divmod(
                small.cache_weight_bytes,
                small.size[1] * cache_dtype_bytes,
            )
            expected_padding = (
                cached_rows * (large.size[1] - small.size[1]) * cache_dtype_bytes
            )
            self.assertEqual(rank, small.rank)
            self.assertEqual(rank, large.rank)
            self.assertEqual(0, remainder)
            self.assertGreater(cached_rows, 0)
            self.assertGreater(expected_padding, 0)
            self.assertEqual(
                expected_padding,
                hbm_delta_by_shard[_shard_key(f"small_{suffix}", small)],
            )
            self.assertEqual(
                0,
                hbm_delta_by_shard[_shard_key(f"large_{suffix}", large)],
            )

        required_by_rank = {0: 0, 1: 0}
        padding_by_rank = {0: 0, 1: 0}
        for option in options.values():
            shard = option.shards[0]
            rank = none_throws(shard.rank)
            required_by_rank[rank] += none_throws(shard.storage).hbm
            padding_by_rank[rank] += hbm_delta_by_shard[_shard_key(option.name, shard)]
        hbm_cap = required_by_rank[1] - 1
        self.assertLessEqual(required_by_rank[0], hbm_cap)
        self.assertLessEqual(required_by_rank[1] - padding_by_rank[1], hbm_cap)

        constrained_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=2,
                hbm_cap=hbm_cap,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            proposer=UniformProposer(),
            partitioner=FixedRankPartitioner(ranks_by_table),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with self.assertRaises(PlannerError) as context:
            constrained_planner.plan(
                module=TestSparseNN(tables=tables, weighted_tables=[]),
                sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
            )
        self.assertIn("rank 1:", str(context.exception))
        self.assertNotIn("rank 0:", str(context.exception))

    def test_row_wise_uvm_cache_dimension_padding_uses_shard_dimensions(self) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        constraints = _uvm_cache_constraints(tables, ShardingType.ROW_WISE.value)
        topology = Topology(
            world_size=2,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        ranks_by_table = {"table_small": [0, 1], "table_large": [0, 1]}
        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=UniformProposer(),
            partitioner=FixedRankPartitioner(ranks_by_table),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], RWUVMCachingSharder())],
        )

        selected_option_list = planner.get_selected_options()
        options = {option.name: option for option in selected_option_list}
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_option_list,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], RWUVMCachingSharder()),
                ranks_by_table,
            ),
        )
        cache_dtype_bytes = options["table_small"].tensor.element_size()
        small_shards = sorted(
            options["table_small"].shards, key=lambda shard: none_throws(shard.rank)
        )
        large_shards = sorted(
            options["table_large"].shards, key=lambda shard: none_throws(shard.rank)
        )
        self.assertEqual([0, 1], [shard.rank for shard in small_shards])
        self.assertEqual([[500, 8], [500, 8]], [shard.size for shard in small_shards])
        for small, large in zip(small_shards, large_shards):
            cached_rows, remainder = divmod(
                small.cache_weight_bytes,
                small.size[1] * cache_dtype_bytes,
            )
            expected_padding = (
                cached_rows * (large.size[1] - small.size[1]) * cache_dtype_bytes
            )
            self.assertEqual(0, remainder)
            self.assertGreater(cached_rows, 0)
            self.assertGreater(expected_padding, 0)
            self.assertEqual(
                expected_padding,
                hbm_delta_by_shard[_shard_key("table_small", small)],
            )
            self.assertEqual(
                0,
                hbm_delta_by_shard[_shard_key("table_large", large)],
            )

    def test_column_wise_uvm_cache_dimension_padding_uses_shard_dimensions(
        self,
    ) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        constraints = _uvm_cache_constraints(
            tables,
            ShardingType.COLUMN_WISE.value,
            min_partitions={"table_small": 4, "table_large": 40},
        )
        topology = Topology(
            world_size=2,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        ranks_by_table = {
            "table_small": [0, 1],
            "table_large": [0, 1, 0, 1],
        }
        planner = EmbeddingShardingPlanner(
            topology=topology,
            proposer=UniformProposer(),
            partitioner=FixedRankPartitioner(ranks_by_table),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], CWUVMCachingSharder())],
        )

        selected_option_list = planner.get_selected_options()
        options = {option.name: option for option in selected_option_list}
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_option_list,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], CWUVMCachingSharder()),
                ranks_by_table,
            ),
        )
        cache_dtype_bytes = options["table_small"].tensor.element_size()
        small_shards = sorted(
            options["table_small"].shards, key=lambda shard: none_throws(shard.rank)
        )
        large_shards = sorted(
            options["table_large"].shards, key=lambda shard: none_throws(shard.rank)
        )
        self.assertEqual([0, 1], [shard.rank for shard in small_shards])
        self.assertEqual([[1000, 4], [1000, 4]], [shard.size for shard in small_shards])
        self.assertEqual(
            [[1000, 40], [1000, 40], [1000, 40], [1000, 40]],
            [shard.size for shard in large_shards],
        )
        self.assertEqual([0, 0, 1, 1], [shard.rank for shard in large_shards])
        for small in small_shards:
            large_dimension = max(
                shard.size[1] for shard in large_shards if shard.rank == small.rank
            )
            cached_rows, remainder = divmod(
                small.cache_weight_bytes,
                small.size[1] * cache_dtype_bytes,
            )
            expected_padding = (
                cached_rows * (large_dimension - small.size[1]) * cache_dtype_bytes
            )
            self.assertEqual(0, remainder)
            self.assertGreater(cached_rows, 0)
            self.assertGreater(expected_padding, 0)
            self.assertEqual(
                expected_padding,
                hbm_delta_by_shard[_shard_key("table_small", small)],
            )
        for large in large_shards:
            self.assertEqual(
                0,
                hbm_delta_by_shard[_shard_key("table_large", large)],
            )

    def test_use_one_tbe_per_table_avoids_uvm_cache_dimension_padding(self) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        constraints = _uvm_cache_constraints(tables, ShardingType.TABLE_WISE.value)
        topology = Topology(
            world_size=1,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        sharder = TWUVMCachingSharder(
            fused_params={
                "cache_load_factor": 0.5,
                "use_one_tbe_per_table": True,
            }
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], sharder)],
        )

        selected_options = planner.get_selected_options()
        hbm_delta_by_shard = _hbm_delta_by_shard(
            selected_options,
            _plan_without_padding(
                tables,
                constraints,
                topology,
                cast(ModuleSharder[nn.Module], sharder),
            ),
        )
        for option in selected_options:
            shard = option.shards[0]
            self.assertGreater(shard.cache_weight_bytes, 0)
            self.assertEqual(
                0,
                hbm_delta_by_shard[_shard_key(option.name, shard)],
            )
