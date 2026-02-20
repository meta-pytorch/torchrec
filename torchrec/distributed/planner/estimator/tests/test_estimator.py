#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for the estimator module - main EmbeddingPerfEstimator and evaluators.

Tests the core estimator classes:
- EmbeddingShardingPerfEvaluator base class methods
- Sharding-specific evaluators (TableWise, RowWise, etc.)
- EmbeddingPerfEstimatorFactory registration and creation
- EmbeddingPerfEstimatorV2 estimate method
"""

import unittest
from typing import Optional
from unittest.mock import MagicMock

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import HBM_MEM_BW
from torchrec.distributed.planner.estimator.annotations import (
    bwd_coefficient,
    forward_compute,
    fwd_coefficient,
    prefetch_coefficient,
)
from torchrec.distributed.planner.estimator.estimator import (
    ColumnWiseEvaluator,
    compute_block_usage_penalty,
    DataParallelEvaluator,
    EmbeddingPerfEstimatorFactory,
    EmbeddingPerfEstimatorV2,
    get_embedding_perf_sharding_evaluator,
    RowWiseEvaluator,
    TableRowWiseEvaluator,
    TableWiseEvaluator,
    TRAINING_EVALUATORS,
)
from torchrec.distributed.planner.estimator.types import (
    HardwarePerfConfig,
    PerfCoefficient,
    PrefetchCoefficients,
    ShardPerfContext,
)
from torchrec.distributed.planner.types import CollectiveType, GeneralizedCommsBandwidth
from torchrec.distributed.types import ShardingType


class MockCommsBandwidth(GeneralizedCommsBandwidth):
    """Mock bandwidth class for testing."""

    def __init__(self, bw: float = 1000.0) -> None:
        self._bw = bw

    def get_bw(
        self,
        local_world_size: int,
        world_size: int,
        collective_type: CollectiveType,
    ) -> float:
        return self._bw

    @property
    def intra_host_bw(self) -> float:
        return self._bw

    @property
    def inter_host_bw(self) -> float:
        return self._bw


def create_test_context(
    sharding_type: str = ShardingType.TABLE_WISE.value,
    compute_kernel: str = EmbeddingComputeKernel.FUSED.value,
    hash_size: int = 10000,
    emb_dim: int = 128,
    batch_sizes: Optional[list] = None,  # pyre-ignore[24]
    num_poolings: Optional[list] = None,  # pyre-ignore[24]
    input_lengths: Optional[list] = None,  # pyre-ignore[24]
    world_size: int = 8,
    local_world_size: int = 4,
    device_bw: float = HBM_MEM_BW,
    is_pooled: bool = True,
    is_weighted: bool = False,
    comms_bw: float = 1000.0,
) -> ShardPerfContext:
    """Helper to create a ShardPerfContext for testing."""
    return ShardPerfContext(
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
        hash_size=hash_size,
        emb_dim=emb_dim,
        batch_sizes=batch_sizes or [64],
        num_poolings=num_poolings or [1.0],
        input_lengths=input_lengths or [10.0],
        world_size=world_size,
        local_world_size=local_world_size,
        device_bw=device_bw,
        hbm_to_ddr_mem_bw=device_bw,
        is_pooled=is_pooled,
        is_weighted=is_weighted,
        comms_bandwidths=MockCommsBandwidth(comms_bw),
        table_data_type_size=4.0,
        output_data_type_size=4.0,
        input_data_type_size=8.0,
        fwd_a2a_comm_data_type_size=4.0,
        bwd_a2a_comm_data_type_size=4.0,
        fwd_sr_comm_data_type_size=4.0,
        bwd_sr_comm_data_type_size=4.0,
    )


class ComputeBlockUsagePenaltyTest(unittest.TestCase):
    """Tests for compute_block_usage_penalty helper function."""

    def test_full_block_no_penalty(self) -> None:
        """Test that emb_dim >= 128 has no penalty."""
        penalty = compute_block_usage_penalty(128)
        self.assertEqual(penalty, 1.0)
        penalty = compute_block_usage_penalty(256)
        self.assertEqual(penalty, 1.0)

    def test_half_block_penalty(self) -> None:
        """Test that 64 <= emb_dim < 128 gets half block penalty."""
        penalty = compute_block_usage_penalty(64)
        self.assertGreater(penalty, 1.0)

    def test_quarter_block_penalty(self) -> None:
        """Test that emb_dim < 64 gets quarter block penalty."""
        penalty = compute_block_usage_penalty(32)
        self.assertGreater(penalty, compute_block_usage_penalty(64))


class GetEmbeddingPerfShardingEvaluatorTest(unittest.TestCase):
    """Tests for get_embedding_perf_sharding_evaluator function."""

    def test_returns_table_wise_evaluator(self) -> None:
        """Test that TABLE_WISE returns TableWiseEvaluator."""
        evaluator = get_embedding_perf_sharding_evaluator(ShardingType.TABLE_WISE.value)
        self.assertIsInstance(evaluator, TableWiseEvaluator)

    def test_returns_row_wise_evaluator(self) -> None:
        """Test that ROW_WISE returns RowWiseEvaluator."""
        evaluator = get_embedding_perf_sharding_evaluator(ShardingType.ROW_WISE.value)
        self.assertIsInstance(evaluator, RowWiseEvaluator)

    def test_returns_table_row_wise_evaluator(self) -> None:
        """Test that TABLE_ROW_WISE returns TableRowWiseEvaluator."""
        evaluator = get_embedding_perf_sharding_evaluator(
            ShardingType.TABLE_ROW_WISE.value
        )
        self.assertIsInstance(evaluator, TableRowWiseEvaluator)

    def test_returns_column_wise_evaluator(self) -> None:
        """Test that COLUMN_WISE returns ColumnWiseEvaluator."""
        evaluator = get_embedding_perf_sharding_evaluator(
            ShardingType.COLUMN_WISE.value
        )
        self.assertIsInstance(evaluator, ColumnWiseEvaluator)

    def test_returns_data_parallel_evaluator(self) -> None:
        """Test that DATA_PARALLEL returns DataParallelEvaluator."""
        evaluator = get_embedding_perf_sharding_evaluator(
            ShardingType.DATA_PARALLEL.value
        )
        self.assertIsInstance(evaluator, DataParallelEvaluator)

    def test_unknown_sharding_returns_table_wise(self) -> None:
        """Test that unknown sharding type returns TableWiseEvaluator as default."""
        evaluator = get_embedding_perf_sharding_evaluator("unknown_sharding")
        self.assertIsInstance(evaluator, TableWiseEvaluator)


class TableWiseEvaluatorTest(unittest.TestCase):
    """Tests for TableWiseEvaluator."""

    def setUp(self) -> None:
        self.evaluator = TableWiseEvaluator()
        self.config = HardwarePerfConfig()

    def test_use_min_dim_for_lookup_default_true(self) -> None:
        """Test that TABLE_WISE uses min dim for lookup by default."""
        self.assertTrue(self.evaluator.use_min_dim_for_lookup(self.config))

    def test_use_block_usage_penalty_default_true(self) -> None:
        """Test that TABLE_WISE uses block usage penalty by default."""
        self.assertTrue(self.evaluator.use_block_usage_penalty(self.config))

    def test_compute_perf_returns_perf_object(self) -> None:
        """Test that compute_perf returns a Perf object with all fields."""
        ctx = create_test_context(sharding_type=ShardingType.TABLE_WISE.value)
        perf = self.evaluator.compute_perf(ctx, self.config)
        self.assertIsNotNone(perf)
        self.assertIsInstance(perf.fwd_compute, float)
        self.assertIsInstance(perf.bwd_compute, float)
        self.assertIsInstance(perf.fwd_comms, float)
        self.assertIsInstance(perf.bwd_comms, float)

    def test_get_batch_inputs_divisor_is_one(self) -> None:
        """Test that TABLE_WISE batch inputs divisor is 1 (no division)."""
        ctx = create_test_context()
        divisor = self.evaluator.get_batch_inputs_divisor(ctx)
        self.assertEqual(divisor, 1)

    def test_fwd_compute_is_positive(self) -> None:
        """Test that forward compute time is positive."""
        ctx = create_test_context(sharding_type=ShardingType.TABLE_WISE.value)
        fwd_compute = self.evaluator.compute_fwd_comp(ctx, self.config)
        self.assertGreater(fwd_compute, 0.0)

    def test_bwd_compute_is_positive(self) -> None:
        """Test that backward compute time is positive."""
        ctx = create_test_context(sharding_type=ShardingType.TABLE_WISE.value)
        bwd_compute = self.evaluator.compute_bwd_comp(ctx, self.config)
        self.assertGreater(bwd_compute, 0.0)


class RowWiseEvaluatorTest(unittest.TestCase):
    """Tests for RowWiseEvaluator."""

    def setUp(self) -> None:
        self.evaluator = RowWiseEvaluator()
        self.config = HardwarePerfConfig()

    def test_batch_inputs_divisor_is_world_size(self) -> None:
        """Test that ROW_WISE divides batch inputs by world_size."""
        ctx = create_test_context(world_size=8)
        divisor = self.evaluator.get_batch_inputs_divisor(ctx)
        self.assertEqual(divisor, 8)

    def test_prefetch_divisor_is_world_size(self) -> None:
        """Test that ROW_WISE prefetch divisor is world_size."""
        ctx = create_test_context(world_size=8)
        divisor = self.evaluator.get_prefetch_divisor(ctx)
        self.assertEqual(divisor, 8)

    def test_use_min_dim_for_lookup_default_false(self) -> None:
        """Test that ROW_WISE does not use min dim for lookup by default."""
        self.assertFalse(self.evaluator.use_min_dim_for_lookup(self.config))

    def test_use_block_usage_penalty_default_false(self) -> None:
        """Test that ROW_WISE does not use block usage penalty by default."""
        self.assertFalse(self.evaluator.use_block_usage_penalty(self.config))

    def test_compute_perf_returns_perf_object(self) -> None:
        """Test that compute_perf returns a Perf object."""
        ctx = create_test_context(sharding_type=ShardingType.ROW_WISE.value)
        perf = self.evaluator.compute_perf(ctx, self.config)
        self.assertIsNotNone(perf)


class TableRowWiseEvaluatorTest(unittest.TestCase):
    """Tests for TableRowWiseEvaluator."""

    def setUp(self) -> None:
        self.evaluator = TableRowWiseEvaluator()
        self.config = HardwarePerfConfig()

    def test_batch_inputs_divisor_is_local_world_size(self) -> None:
        """Test that TABLE_ROW_WISE divides batch inputs by local_world_size."""
        ctx = create_test_context(world_size=8, local_world_size=4)
        divisor = self.evaluator.get_batch_inputs_divisor(ctx)
        self.assertEqual(divisor, 4)

    def test_prefetch_divisor_is_local_world_size(self) -> None:
        """Test that TABLE_ROW_WISE prefetch divisor is local_world_size."""
        ctx = create_test_context(world_size=8, local_world_size=4)
        divisor = self.evaluator.get_prefetch_divisor(ctx)
        self.assertEqual(divisor, 4)

    def test_compute_perf_returns_perf_object(self) -> None:
        """Test that compute_perf returns a Perf object."""
        ctx = create_test_context(sharding_type=ShardingType.TABLE_ROW_WISE.value)
        perf = self.evaluator.compute_perf(ctx, self.config)
        self.assertIsNotNone(perf)


class ColumnWiseEvaluatorTest(unittest.TestCase):
    """Tests for ColumnWiseEvaluator."""

    def setUp(self) -> None:
        self.evaluator = ColumnWiseEvaluator()
        self.config = HardwarePerfConfig()

    def test_compute_perf_returns_perf_object(self) -> None:
        """Test that compute_perf returns a Perf object."""
        ctx = create_test_context(sharding_type=ShardingType.COLUMN_WISE.value)
        perf = self.evaluator.compute_perf(ctx, self.config)
        self.assertIsNotNone(perf)


class DataParallelEvaluatorTest(unittest.TestCase):
    """Tests for DataParallelEvaluator."""

    def setUp(self) -> None:
        self.evaluator = DataParallelEvaluator()
        self.config = HardwarePerfConfig()

    def test_fwd_comms_is_zero(self) -> None:
        """Test that DATA_PARALLEL has no forward communication."""
        ctx = create_test_context(sharding_type=ShardingType.DATA_PARALLEL.value)
        fwd_comms = self.evaluator._default_fwd_comms(ctx, self.config)
        self.assertEqual(fwd_comms, 0.0)

    def test_input_dist_comms_is_zero(self) -> None:
        """Test that DATA_PARALLEL has no input distribution communication."""
        ctx = create_test_context(sharding_type=ShardingType.DATA_PARALLEL.value)
        input_dist_comms = self.evaluator.compute_input_dist_comms(ctx, self.config)
        self.assertEqual(input_dist_comms, 0.0)

    def test_compute_perf_returns_perf_object(self) -> None:
        """Test that compute_perf returns a Perf object."""
        ctx = create_test_context(sharding_type=ShardingType.DATA_PARALLEL.value)
        perf = self.evaluator.compute_perf(ctx, self.config)
        self.assertIsNotNone(perf)


class EmbeddingPerfEstimatorFactoryTest(unittest.TestCase):
    """Tests for EmbeddingPerfEstimatorFactory."""

    def test_create_unknown_hardware_raises(self) -> None:
        """Test that creating with unknown hardware raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            EmbeddingPerfEstimatorFactory.create(
                hardware_name="nonexistent_hardware_xyz",
                topology=MagicMock(),
            )
        self.assertIn("Unknown hardware", str(ctx.exception))

    def test_list_registered(self) -> None:
        """Test that list_registered returns a list."""
        registered = EmbeddingPerfEstimatorFactory.list_registered()
        self.assertIsInstance(registered, list)

    def test_register_hardware(self) -> None:
        """Test registering a new hardware config."""

        @EmbeddingPerfEstimatorFactory.register("test_hardware_for_unit_test")
        class TestHardwareConfig(HardwarePerfConfig):
            name = "test_hardware_for_unit_test"

        self.assertIn(
            "test_hardware_for_unit_test",
            EmbeddingPerfEstimatorFactory.list_registered(),
        )

    def test_create_with_config(self) -> None:
        """Test creating estimator with a specific config instance."""
        config = HardwarePerfConfig()
        config.name = "custom_config"
        estimator = EmbeddingPerfEstimatorFactory.create_with_config(
            config=config,
            topology=MagicMock(),
        )
        self.assertIsInstance(estimator, EmbeddingPerfEstimatorV2)
        self.assertEqual(estimator.config.name, "custom_config")


class EmbeddingPerfEstimatorV2Test(unittest.TestCase):
    """Tests for EmbeddingPerfEstimatorV2."""

    def test_config_property(self) -> None:
        """Test that config property returns the config."""
        config = HardwarePerfConfig()
        config.name = "test_config"
        estimator = EmbeddingPerfEstimatorV2(
            topology=MagicMock(),
            config=config,
        )
        self.assertEqual(estimator.config.name, "test_config")

    def test_is_inference_property(self) -> None:
        """Test is_inference property."""
        estimator = EmbeddingPerfEstimatorV2(
            topology=MagicMock(),
            is_inference=True,
        )
        self.assertTrue(estimator.is_inference)

        estimator2 = EmbeddingPerfEstimatorV2(
            topology=MagicMock(),
            is_inference=False,
        )
        self.assertFalse(estimator2.is_inference)

    def test_default_config_created_when_none(self) -> None:
        """Test that default HardwarePerfConfig is created when none provided."""
        estimator = EmbeddingPerfEstimatorV2(
            topology=MagicMock(),
        )
        self.assertIsInstance(estimator.config, HardwarePerfConfig)


class EvaluatorCustomMethodTest(unittest.TestCase):
    """Tests for evaluator with custom annotation methods."""

    def test_custom_forward_compute_annotation(self) -> None:
        """Test that @forward_compute annotation is used when present."""

        class CustomConfig(HardwarePerfConfig):
            name = "custom_fwd"

            @forward_compute(sharding_type="table_wise")
            def compute_fwd(self, ctx: ShardPerfContext) -> float:
                return 999.0

        config = CustomConfig()
        evaluator = TableWiseEvaluator()
        ctx = create_test_context(sharding_type=ShardingType.TABLE_WISE.value)

        fwd_compute = evaluator.compute_fwd_comp(ctx, config)
        self.assertEqual(fwd_compute, 999.0)

    def test_custom_coefficient_annotation(self) -> None:
        """Test that @fwd_coefficient and @bwd_coefficient annotations are used."""

        class CoefficientConfig(HardwarePerfConfig):
            name = "coeff_config"

            @fwd_coefficient(sharding_type="table_wise")
            def tw_fwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=10.0)

            @bwd_coefficient(sharding_type="table_wise")
            def tw_bwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=20.0)

        config = CoefficientConfig()
        coeffs = config.get_coefficients_for_sharding(ShardingType.TABLE_WISE.value)
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 10.0)
        self.assertIsNotNone(coeffs.bwd)
        self.assertEqual(coeffs.bwd.lookup_size_multiplier, 20.0)


class EvaluatorBandwidthPriorityTest(unittest.TestCase):
    """Tests for bandwidth priority in evaluators."""

    def test_device_bw_uses_ctx_value(self) -> None:
        """Test that device_bw comes from ctx.device_bw."""

        class ConfigWithDeviceBW(HardwarePerfConfig):
            name = "with_device_bw"
            device_bw = 5000.0

        config = ConfigWithDeviceBW()
        evaluator = TableWiseEvaluator()
        ctx = create_test_context(device_bw=1000.0)

        # device_bw comes from context, not config
        device_bw = evaluator._get_device_bw(ctx, config)
        self.assertEqual(device_bw, 1000.0)

    def test_device_bw_uses_ctx_when_config_none(self) -> None:
        """Test that ctx.device_bw is used when config.device_bw is None."""
        config = HardwarePerfConfig()
        evaluator = TableWiseEvaluator()
        ctx = create_test_context(device_bw=2000.0)

        device_bw = evaluator._get_device_bw(ctx, config)
        self.assertEqual(device_bw, 2000.0)


class EvaluatorPrefetchTest(unittest.TestCase):
    """Tests for prefetch compute in evaluators."""

    def test_default_prefetch_comp_no_cache_fetches(self) -> None:
        """Test default prefetch compute when no cache fetches."""
        evaluator = TableWiseEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context()
        # Without FUSED_UVM_CACHING kernel or caching_ratio, expected_cache_fetches is 0

        prefetch = evaluator.compute_prefetch_comp(ctx, config)
        self.assertEqual(prefetch, 0.0)

    def test_prefetch_without_caching_kernel_returns_zero(self) -> None:
        """Test prefetch compute returns 0 without FUSED_UVM_CACHING kernel."""
        evaluator = TableWiseEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context(compute_kernel=EmbeddingComputeKernel.FUSED.value)
        # Prefetch only applies to FUSED_UVM_CACHING kernel

        prefetch = evaluator.compute_prefetch_comp(ctx, config)
        self.assertEqual(prefetch, 0.0)

    def test_prefetch_with_caching_kernel(self) -> None:
        """Test prefetch compute with FUSED_UVM_CACHING kernel and cache stats."""
        evaluator = TableWiseEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context(
            compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        )
        # Set required cache stats for prefetch computation
        ctx.caching_ratio = 0.5
        ctx.cache_stats_expected_lookups = 100.0
        ctx.expected_miss_rate = 0.1

        prefetch = evaluator.compute_prefetch_comp(ctx, config)
        self.assertGreaterEqual(prefetch, 0.0)

    def test_prefetch_divisor_applied_for_row_wise(self) -> None:
        """Test that prefetch divisor is applied for ROW_WISE (world_size)."""
        evaluator = RowWiseEvaluator()
        ctx = create_test_context(world_size=8)

        divisor = evaluator.get_prefetch_divisor(ctx)
        self.assertEqual(divisor, 8)

    def test_prefetch_divisor_applied_for_table_row_wise(self) -> None:
        """Test that prefetch divisor is applied for TABLE_ROW_WISE (local_world_size)."""
        evaluator = TableRowWiseEvaluator()
        ctx = create_test_context(world_size=8, local_world_size=4)

        divisor = evaluator.get_prefetch_divisor(ctx)
        self.assertEqual(divisor, 4)

    def test_prefetch_divisor_one_for_table_wise(self) -> None:
        """Test that prefetch divisor is 1 for TABLE_WISE (no division)."""
        evaluator = TableWiseEvaluator()
        ctx = create_test_context()

        divisor = evaluator.get_prefetch_divisor(ctx)
        self.assertEqual(divisor, 1)


class TrainingEvaluatorsRegistryTest(unittest.TestCase):
    """Tests for TRAINING_EVALUATORS registry."""

    def test_all_sharding_types_have_evaluators(self) -> None:
        """Test that all standard sharding types have evaluators."""
        expected_types = [
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.DATA_PARALLEL.value,
        ]
        for sharding_type in expected_types:
            self.assertIn(sharding_type, TRAINING_EVALUATORS)

    def test_evaluators_are_correct_types(self) -> None:
        """Test that evaluators in registry are correct types."""
        self.assertIsInstance(
            TRAINING_EVALUATORS[ShardingType.TABLE_WISE.value], TableWiseEvaluator
        )
        self.assertIsInstance(
            TRAINING_EVALUATORS[ShardingType.ROW_WISE.value], RowWiseEvaluator
        )
        self.assertIsInstance(
            TRAINING_EVALUATORS[ShardingType.TABLE_ROW_WISE.value],
            TableRowWiseEvaluator,
        )
        self.assertIsInstance(
            TRAINING_EVALUATORS[ShardingType.COLUMN_WISE.value], ColumnWiseEvaluator
        )
        self.assertIsInstance(
            TRAINING_EVALUATORS[ShardingType.DATA_PARALLEL.value], DataParallelEvaluator
        )


class EvaluatorComputePerfIntegrationTest(unittest.TestCase):
    """Integration tests for compute_perf across evaluators."""

    def test_table_wise_perf_values_are_reasonable(self) -> None:
        """Test that TABLE_WISE perf values are non-negative."""
        evaluator = TableWiseEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context(sharding_type=ShardingType.TABLE_WISE.value)

        perf = evaluator.compute_perf(ctx, config)
        self.assertGreaterEqual(perf.fwd_compute, 0.0)
        self.assertGreaterEqual(perf.bwd_compute, 0.0)
        self.assertGreaterEqual(perf.fwd_comms, 0.0)
        self.assertGreaterEqual(perf.bwd_comms, 0.0)
        self.assertGreaterEqual(perf.prefetch_compute, 0.0)

    def test_row_wise_perf_values_are_reasonable(self) -> None:
        """Test that ROW_WISE perf values are non-negative."""
        evaluator = RowWiseEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context(sharding_type=ShardingType.ROW_WISE.value)

        perf = evaluator.compute_perf(ctx, config)
        self.assertGreaterEqual(perf.fwd_compute, 0.0)
        self.assertGreaterEqual(perf.bwd_compute, 0.0)
        self.assertGreaterEqual(perf.fwd_comms, 0.0)
        self.assertGreaterEqual(perf.bwd_comms, 0.0)

    def test_data_parallel_has_zero_fwd_comms(self) -> None:
        """Test that DATA_PARALLEL has zero forward comms."""
        evaluator = DataParallelEvaluator()
        config = HardwarePerfConfig()
        ctx = create_test_context(sharding_type=ShardingType.DATA_PARALLEL.value)

        perf = evaluator.compute_perf(ctx, config)
        self.assertEqual(perf.fwd_comms, 0.0)


if __name__ == "__main__":
    unittest.main()
