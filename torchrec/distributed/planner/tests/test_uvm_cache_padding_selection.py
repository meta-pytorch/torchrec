#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from typing import cast

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.storage_reservations import (
    FixedAbsoluteStorageReservation,
)
from torchrec.distributed.planner.tests.uvm_cache_padding_test_utils import (
    _hbm_by_table,
    _loaded_options_without_cache_metadata,
    _mixed_dimension_tables,
    _total_hbm,
    _uvm_cache_constraints,
    enable_uvm_cache_planner_knobs,
    FixedRankPartitioner,
    KernelSequenceProposer,
    MockPlanLoader,
    TWUVMCachingAndUVMSharder,
    TWUVMCachingSharder,
    uvm_cache_padding_knob,
)
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType


class TestUVMCachePaddingSelection(unittest.TestCase):
    def setUp(self) -> None:
        enable_uvm_cache_planner_knobs(self)

    def test_loaded_plan_recomputes_padding_from_current_estimates(self) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        constraints = _uvm_cache_constraints(tables, ShardingType.TABLE_WISE.value)
        topology = Topology(
            world_size=2,
            hbm_cap=1024**3,
            ddr_cap=1024**3,
            compute_device="cuda",
        )
        baseline_planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            partitioner=FixedRankPartitioner({"table_small": [0], "table_large": [0]}),
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        sharder = cast(ModuleSharder[nn.Module], TWUVMCachingSharder())

        baseline_planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[sharder],
        )

        baseline_options = baseline_planner.get_selected_options()
        baseline_hbm = _total_hbm(baseline_options)
        unpadded_planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            partitioner=FixedRankPartitioner({"table_small": [0], "table_large": [0]}),
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with uvm_cache_padding_knob(False):
            unpadded_planner.plan(
                module=TestSparseNN(tables=tables, weighted_tables=[]),
                sharders=[sharder],
            )
        unpadded_options = unpadded_planner.get_selected_options()
        small_padding = baseline_hbm - _total_hbm(unpadded_options)
        self.assertGreater(small_padding, 0)
        loaded_map = _loaded_options_without_cache_metadata(unpadded_options)
        self.assertTrue(
            all(
                shard.cache_weight_bytes == 0
                for option in loaded_map.values()
                for shard in option.shards
            )
        )
        for option in loaded_map.values():
            for shard in option.shards:
                shard.rank = 1

        loaded_planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
            plan_loader=MockPlanLoader(
                loaded_sharding_options=loaded_map,
                context_hash=baseline_planner.hash_planner_context_inputs_str(),
            ),
        )
        loaded_planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[sharder],
        )

        loaded_options = loaded_planner.get_selected_options()
        self.assertEqual(baseline_hbm, _total_hbm(loaded_options))
        self.assertEqual(
            _hbm_by_table(baseline_options),
            _hbm_by_table(loaded_options),
        )
        self.assertEqual(
            {1},
            {shard.rank for option in loaded_options for shard in option.shards},
        )
        loaded_small_shard = next(
            option.shards[0]
            for option in loaded_options
            if option.name == "table_small"
        )
        self.assertGreater(loaded_small_shard.cache_weight_bytes, 0)

    def test_loaded_padding_rejection_falls_back_to_feasible_proposal(self) -> None:
        tables = _mixed_dimension_tables([("table_small", 8), ("table_large", 160)])
        cached_constraints = _uvm_cache_constraints(
            tables, ShardingType.TABLE_WISE.value
        )
        calibration_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=cached_constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        calibration_planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
        )
        calibrated_options = calibration_planner.get_selected_options()
        padded_hbm = _total_hbm(calibrated_options)
        unpadded_calibration_planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=1024**3,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            constraints=cached_constraints,
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        with uvm_cache_padding_knob(False):
            unpadded_calibration_planner.plan(
                module=TestSparseNN(tables=tables, weighted_tables=[]),
                sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingSharder())],
            )
        unpadded_calibrated_options = (
            unpadded_calibration_planner.get_selected_options()
        )
        padding = padded_hbm - _total_hbm(unpadded_calibrated_options)
        self.assertGreater(padding, 0)

        kernels = [
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            EmbeddingComputeKernel.FUSED_UVM.value,
        ]
        proposer = KernelSequenceProposer(
            [
                [("table_small", kernels[0]), ("table_large", kernels[0])],
                [("table_small", kernels[1]), ("table_large", kernels[0])],
            ]
        )
        hbm_cap = padded_hbm - 1
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=1,
                hbm_cap=hbm_cap,
                ddr_cap=1024**3,
                compute_device="cuda",
            ),
            proposer=proposer,
            constraints=_uvm_cache_constraints(
                tables,
                ShardingType.TABLE_WISE.value,
                compute_kernels=kernels,
            ),
            plan_loader=MockPlanLoader(
                loaded_sharding_options=_loaded_options_without_cache_metadata(
                    unpadded_calibrated_options
                )
            ),
            storage_reservation=FixedAbsoluteStorageReservation(0),
        )
        planner.plan(
            module=TestSparseNN(tables=tables, weighted_tables=[]),
            sharders=[cast(ModuleSharder[nn.Module], TWUVMCachingAndUVMSharder())],
        )

        first_unpadded_hbm = _total_hbm(proposer.proposals[0])
        self.assertLessEqual(first_unpadded_hbm, hbm_cap)
        self.assertGreater(first_unpadded_hbm + padding, hbm_cap)
        self.assertEqual([False, True], proposer.partitionable_feedback)
        selected = {option.name: option for option in planner.get_selected_options()}
        self.assertEqual(
            EmbeddingComputeKernel.FUSED_UVM.value,
            selected["table_small"].compute_kernel,
        )
        self.assertEqual(
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            selected["table_large"].compute_kernel,
        )
        self.assertLessEqual(_total_hbm(list(selected.values())), hbm_cap)
