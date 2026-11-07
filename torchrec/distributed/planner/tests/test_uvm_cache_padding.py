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
    _hbm_by_table,
    _mixed_dimension_tables,
    _total_hbm,
    _uvm_cache_constraints,
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
from torchrec.distributed.types import (
    CacheParams,
    DataType,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestUVMCachePadding(unittest.TestCase):
    def setUp(self) -> None:
        enable_uvm_cache_planner_knobs(self)

    def test_uvm_cache_accounts_for_dominant_dimension_padding(self) -> None:
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
                cache_params=CacheParams(load_factor=0.5),
            )
            for table in tables
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharder = TWUVMCachingSharder(fused_params={"cache_load_factor": 0.5})
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

        planner.plan(
            module=model,
            sharders=[cast(ModuleSharder[torch.nn.Module], sharder)],
        )
        selected_option_list = planner.get_selected_options()
        selected_options = {option.name: option for option in selected_option_list}
        small_shard = selected_options["table_small"].shards[0]

        expected_small_padding = small_shard.cache_weight_bytes * (160 // 8 - 1)
        baseline_planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with uvm_cache_padding_knob(False):
            baseline_planner.plan(
                module=model,
                sharders=[cast(ModuleSharder[torch.nn.Module], sharder)],
            )
        padded_hbm_by_table = _hbm_by_table(selected_option_list)
        baseline_hbm_by_table = _hbm_by_table(baseline_planner.get_selected_options())
        self.assertEqual(
            expected_small_padding,
            padded_hbm_by_table["table_small"][0]
            - baseline_hbm_by_table["table_small"][0],
        )
        self.assertEqual(
            baseline_hbm_by_table["table_large"],
            padded_hbm_by_table["table_large"],
        )

        padded_hbm = _total_hbm(selected_option_list)
        self.assertEqual(
            expected_small_padding,
            padded_hbm - _total_hbm(baseline_planner.get_selected_options()),
        )

        constrained_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=padded_hbm - 1,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with self.assertRaisesRegex(
            PlannerError,
            r"UVM cache dimension padding causes per-rank HBM to exceed capacity: "
            r"rank \d+: required \d+ bytes after UVM cache dimension padding, "
            r"available \d+ bytes",
        ):
            constrained_planner.plan(
                module=model,
                sharders=[cast(ModuleSharder[torch.nn.Module], sharder)],
            )

    def test_uvm_cache_dimension_padding_uses_cache_precision(self) -> None:
        tables = _mixed_dimension_tables(
            [("table_small", 8), ("table_large", 160)],
            data_type=DataType.FP16,
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=_uvm_cache_constraints(
                tables,
                ShardingType.TABLE_WISE.value,
                cache_precision=DataType.FP32,
            ),
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
        )

        selected_option_list = planner.get_selected_options()
        options = {option.name: option for option in selected_option_list}
        small_option = options["table_small"]
        small = small_option.shards[0]
        large = options["table_large"].shards[0]
        cached_rows = 500
        cache_dtype_bytes = 4
        self.assertEqual(2, small_option.tensor.element_size())
        self.assertEqual(cached_rows * 8 * cache_dtype_bytes, small.cache_weight_bytes)
        self.assertEqual(
            cached_rows * 160 * cache_dtype_bytes,
            large.cache_weight_bytes,
        )
        baseline_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=_uvm_cache_constraints(
                tables,
                ShardingType.TABLE_WISE.value,
                cache_precision=DataType.FP32,
            ),
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with uvm_cache_padding_knob(False):
            baseline_planner.plan(
                module=TestSparseNN(tables=tables, weighted_tables=[]),
                sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
            )
        padded_hbm_by_table = _hbm_by_table(selected_option_list)
        baseline_hbm_by_table = _hbm_by_table(baseline_planner.get_selected_options())
        self.assertEqual(
            cached_rows * (160 - 8) * cache_dtype_bytes,
            padded_hbm_by_table["table_small"][0]
            - baseline_hbm_by_table["table_small"][0],
        )
        self.assertEqual(
            baseline_hbm_by_table["table_large"],
            padded_hbm_by_table["table_large"],
        )

    def test_enforce_hbm_accounts_for_uvm_cache_dimension_padding(self) -> None:
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
                cache_params=CacheParams(load_factor=0.5),
                enforce_hbm=True,
            )
            for table in tables
        }
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )

        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[torch.nn.Module], TWUVMCachingSharder())],
        )

        selected_option_list = planner.get_selected_options()
        selected_options = {option.name: option for option in selected_option_list}
        self.assertTrue(all(option.enforce_hbm for option in selected_options.values()))
        small_shard = selected_options["table_small"].shards[0]
        expected_small_padding = small_shard.cache_weight_bytes * (160 // 8 - 1)
        self.assertGreater(small_shard.cache_weight_bytes, 0)
        baseline_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with uvm_cache_padding_knob(False):
            baseline_planner.plan(
                module=TestSparseNN(tables=tables, weighted_tables=[]),
                sharders=[cast(ModuleSharder[torch.nn.Module], TWUVMCachingSharder())],
            )
        padded_hbm_by_table = _hbm_by_table(selected_option_list)
        baseline_hbm_by_table = _hbm_by_table(baseline_planner.get_selected_options())
        self.assertEqual(
            expected_small_padding,
            padded_hbm_by_table["table_small"][0]
            - baseline_hbm_by_table["table_small"][0],
        )
        self.assertEqual(
            baseline_hbm_by_table["table_large"],
            padded_hbm_by_table["table_large"],
        )

    def test_uvm_cache_dimension_padding_requires_knob(self) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        constraints = _uvm_cache_constraints(tables, ShardingType.TABLE_WISE.value)
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharder = cast(ModuleSharder[torch.nn.Module], TWUVMCachingSharder())

        def planned_hbm(hbm_cap: int = 1024**3) -> int:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=1,
                    hbm_cap=hbm_cap,
                    ddr_cap=1024**3,
                    compute_device="cuda",
                ),
                constraints=constraints,
                storage_reservation=FixedAbsoluteStorageReservation(0),
            )
            planner.plan(module=model, sharders=[sharder])
            return _total_hbm(planner.get_selected_options())

        padded_hbm = planned_hbm()
        with uvm_cache_padding_knob(False):
            unpadded_hbm = planned_hbm()
            # Without padding, a budget below the padded total still fits.
            self.assertEqual(unpadded_hbm, planned_hbm(hbm_cap=padded_hbm - 1))
        self.assertGreater(padded_hbm, unpadded_hbm)

    def test_empty_row_wise_shard_counts_toward_dominant_dimension(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2000,
                embedding_dim=8,
                name="table_small",
                feature_names=["feature_small"],
            ),
            EmbeddingBagConfig(
                num_embeddings=1,
                embedding_dim=160,
                name="table_wide",
                feature_names=["feature_wide"],
            ),
        ]
        constraints = _uvm_cache_constraints(tables, ShardingType.ROW_WISE.value)
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharder = cast(ModuleSharder[nn.Module], RWUVMCachingSharder())
        topology = Topology(
            world_size=2,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        ranks_by_table = {"table_small": [0, 1], "table_wide": [0, 1]}

        def selected_options() -> dict[str, ShardingOption]:
            planner = EmbeddingShardingPlanner(
                topology=topology,
                proposer=UniformProposer(),
                partitioner=FixedRankPartitioner(ranks_by_table),
                constraints=constraints,
                storage_reservation=FixedAbsoluteStorageReservation(0),
            )
            planner.plan(module=model, sharders=[sharder])
            return {option.name: option for option in planner.get_selected_options()}

        padded = selected_options()
        with uvm_cache_padding_knob(False):
            unpadded = selected_options()

        wide_shard_on_rank_1 = padded["table_wide"].shards[1]
        self.assertEqual(1, wide_shard_on_rank_1.rank)
        self.assertEqual([0, 160], wide_shard_on_rank_1.size)
        self.assertEqual(0, wide_shard_on_rank_1.cache_weight_bytes)
        for padded_shard, unpadded_shard in zip(
            padded["table_small"].shards, unpadded["table_small"].shards
        ):
            # The empty wide shard on rank 1 still widens that rank's cache rows.
            self.assertEqual(
                padded_shard.cache_weight_bytes * (160 // 8 - 1),
                none_throws(padded_shard.storage).hbm
                - none_throws(unpadded_shard.storage).hbm,
            )
        for padded_shard, unpadded_shard in zip(
            padded["table_wide"].shards, unpadded["table_wide"].shards
        ):
            self.assertEqual(
                none_throws(unpadded_shard.storage).hbm,
                none_throws(padded_shard.storage).hbm,
            )
