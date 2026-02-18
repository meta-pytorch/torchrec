#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for the types module in the estimator package.

Tests the core dataclasses and config classes:
- PerfCoefficient, EstimatorPerfCoefficients, PrefetchCoefficients
- PerfCoefficientConfig with get_coefficients() method
- HardwarePerfConfig with coefficient lookup and validation
- ShardPerfContext computed properties
"""

import unittest

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.estimator.annotations import (
    bwd_coefficient,
    fwd_coefficient,
    prefetch_coefficient,
    supported_sharding_types,
)
from torchrec.distributed.planner.estimator.types import (
    EstimatorPerfCoefficients,
    HardwarePerfConfig,
    PerfCoefficient,
    PerfCoefficientConfig,
    PrefetchCoefficients,
    ShardPerfContext,
)
from torchrec.distributed.types import ShardingType


class PerfCoefficientTest(unittest.TestCase):
    """Tests for the PerfCoefficient frozen dataclass."""

    def test_default_values(self) -> None:
        """Test default coefficient values."""
        coeff = PerfCoefficient()
        self.assertEqual(coeff.input_read_size_multiplier, 1.0)
        self.assertEqual(coeff.lookup_size_multiplier, 1.0)
        self.assertEqual(coeff.embedding_output_multiplier, 1.0)
        self.assertEqual(coeff.hash_size_multiplier, 0.0)

    def test_custom_values(self) -> None:
        """Test creating coefficient with custom values."""
        coeff = PerfCoefficient(
            input_read_size_multiplier=2.0,
            lookup_size_multiplier=1.5,
            embedding_output_multiplier=0.5,
            hash_size_multiplier=0.1,
        )
        self.assertEqual(coeff.input_read_size_multiplier, 2.0)
        self.assertEqual(coeff.lookup_size_multiplier, 1.5)
        self.assertEqual(coeff.embedding_output_multiplier, 0.5)
        self.assertEqual(coeff.hash_size_multiplier, 0.1)

    def test_frozen(self) -> None:
        """Test that PerfCoefficient is frozen (immutable)."""
        coeff = PerfCoefficient()
        with self.assertRaises(AttributeError):
            coeff.input_read_size_multiplier = 2.0  # pyre-ignore[41]


class EstimatorPerfCoefficientsTest(unittest.TestCase):
    """Tests for EstimatorPerfCoefficients frozen dataclass."""

    def test_default_values(self) -> None:
        """Test default fwd coefficient and bwd is None (for FB estimator compatibility)."""
        coeffs = EstimatorPerfCoefficients()
        self.assertEqual(coeffs.fwd.input_read_size_multiplier, 1.0)
        # bwd is None by default to support FB estimators using fwd_compute * bwd_compute_multiplier
        self.assertIsNone(coeffs.bwd)

    def test_custom_fwd_bwd(self) -> None:
        """Test creating with custom forward and backward coefficients."""
        fwd = PerfCoefficient(lookup_size_multiplier=2.0)
        bwd = PerfCoefficient(lookup_size_multiplier=3.0)
        coeffs = EstimatorPerfCoefficients(fwd=fwd, bwd=bwd)
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 2.0)
        self.assertIsNotNone(coeffs.bwd)
        self.assertEqual(coeffs.bwd.lookup_size_multiplier, 3.0)


class PrefetchCoefficientsTest(unittest.TestCase):
    """Tests for PrefetchCoefficients frozen dataclass."""

    def test_default_values(self) -> None:
        """Test default prefetch coefficient values."""
        prefetch = PrefetchCoefficients()
        self.assertEqual(prefetch.expected_num_lookups_coefficient, 0.0)
        self.assertEqual(prefetch.expected_num_unique_lookups_coefficient, 0.0)
        self.assertEqual(prefetch.expected_size_cache_fetches_coefficient, 0.0)

    def test_custom_values(self) -> None:
        """Test creating prefetch coefficients with custom values."""
        prefetch = PrefetchCoefficients(
            expected_num_lookups_coefficient=1.5,
            expected_num_unique_lookups_coefficient=2.0,
            expected_size_cache_fetches_coefficient=0.8,
        )
        self.assertEqual(prefetch.expected_num_lookups_coefficient, 1.5)
        self.assertEqual(prefetch.expected_num_unique_lookups_coefficient, 2.0)
        self.assertEqual(prefetch.expected_size_cache_fetches_coefficient, 0.8)


class PerfCoefficientConfigTest(unittest.TestCase):
    """Tests for PerfCoefficientConfig with get_coefficients() method."""

    def test_get_coefficients_returns_default_when_no_specific(self) -> None:
        """Test that get_coefficients returns default when no specific coefficients."""
        config = PerfCoefficientConfig()
        coeffs = config.get_coefficients(ShardingType.TABLE_WISE.value)
        self.assertEqual(coeffs.fwd.input_read_size_multiplier, 1.0)
        # bwd is None by default - estimator will use fwd_compute * bwd_compute_multiplier approach
        self.assertIsNone(coeffs.bwd)

    def test_get_coefficients_returns_sharding_specific(self) -> None:
        """Test that get_coefficients returns sharding-specific coefficients."""
        table_wise_coeffs = EstimatorPerfCoefficients(
            fwd=PerfCoefficient(lookup_size_multiplier=2.0),
            bwd=PerfCoefficient(lookup_size_multiplier=3.0),
        )
        config = PerfCoefficientConfig(table_wise=table_wise_coeffs)
        coeffs = config.get_coefficients(ShardingType.TABLE_WISE.value)
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 2.0)
        self.assertIsNotNone(coeffs.bwd)
        self.assertEqual(coeffs.bwd.lookup_size_multiplier, 3.0)

    def test_get_coefficients_kernel_override_takes_priority(self) -> None:
        """Test that kernel-specific overrides take priority over sharding type."""
        table_wise_coeffs = EstimatorPerfCoefficients(
            fwd=PerfCoefficient(lookup_size_multiplier=2.0),
        )
        kernel_coeffs = EstimatorPerfCoefficients(
            fwd=PerfCoefficient(lookup_size_multiplier=5.0),
        )
        config = PerfCoefficientConfig(
            table_wise=table_wise_coeffs,
            by_kernel={("table_wise", "fused"): kernel_coeffs},
        )
        coeffs = config.get_coefficients(ShardingType.TABLE_WISE.value, "fused")
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 5.0)
        coeffs = config.get_coefficients(ShardingType.TABLE_WISE.value)
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 2.0)

    def test_get_coefficients_all_sharding_types(self) -> None:
        """Test get_coefficients works for all standard sharding types."""
        config = PerfCoefficientConfig(
            table_wise=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=1.0)
            ),
            row_wise=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=2.0)
            ),
            table_row_wise=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=3.0)
            ),
            column_wise=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=4.0)
            ),
            data_parallel=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=5.0)
            ),
        )
        self.assertEqual(
            config.get_coefficients(
                ShardingType.TABLE_WISE.value
            ).fwd.lookup_size_multiplier,
            1.0,
        )
        self.assertEqual(
            config.get_coefficients(
                ShardingType.ROW_WISE.value
            ).fwd.lookup_size_multiplier,
            2.0,
        )
        self.assertEqual(
            config.get_coefficients(
                ShardingType.TABLE_ROW_WISE.value
            ).fwd.lookup_size_multiplier,
            3.0,
        )
        self.assertEqual(
            config.get_coefficients(
                ShardingType.COLUMN_WISE.value
            ).fwd.lookup_size_multiplier,
            4.0,
        )
        self.assertEqual(
            config.get_coefficients(
                ShardingType.DATA_PARALLEL.value
            ).fwd.lookup_size_multiplier,
            5.0,
        )

    def test_get_coefficients_case_insensitive(self) -> None:
        """Test that sharding type lookup is case-insensitive."""
        config = PerfCoefficientConfig(
            table_wise=EstimatorPerfCoefficients(
                fwd=PerfCoefficient(lookup_size_multiplier=2.0)
            ),
        )
        coeffs = config.get_coefficients("TABLE_WISE")
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 2.0)


class HardwarePerfConfigTest(unittest.TestCase):
    """Tests for HardwarePerfConfig class."""

    def test_default_name(self) -> None:
        """Test default hardware config name."""
        config = HardwarePerfConfig()
        self.assertEqual(config.name, "default")

    def test_is_sharding_type_supported_all_by_default(self) -> None:
        """Test that all sharding types are supported by default."""
        config = HardwarePerfConfig()
        self.assertTrue(
            config.is_sharding_type_supported(ShardingType.TABLE_WISE.value)
        )
        self.assertTrue(config.is_sharding_type_supported(ShardingType.ROW_WISE.value))
        self.assertTrue(
            config.is_sharding_type_supported(ShardingType.COLUMN_WISE.value)
        )

    def test_is_sharding_type_supported_with_restriction(self) -> None:
        """Test sharding type support with explicit restrictions."""

        @supported_sharding_types("table_wise", "row_wise")
        class RestrictedConfig(HardwarePerfConfig):
            name = "restricted"

        config = RestrictedConfig()
        self.assertTrue(
            config.is_sharding_type_supported(ShardingType.TABLE_WISE.value)
        )
        self.assertTrue(config.is_sharding_type_supported(ShardingType.ROW_WISE.value))
        self.assertFalse(
            config.is_sharding_type_supported(ShardingType.COLUMN_WISE.value)
        )

    def test_validate_sharding_type_raises_for_unsupported(self) -> None:
        """Test validate_sharding_type raises ValueError for unsupported types."""

        @supported_sharding_types("table_wise")
        class LimitedConfig(HardwarePerfConfig):
            name = "limited"

        config = LimitedConfig()
        config.validate_sharding_type(ShardingType.TABLE_WISE.value)
        with self.assertRaises(ValueError) as ctx:
            config.validate_sharding_type(ShardingType.ROW_WISE.value)
        self.assertIn("row_wise", str(ctx.exception))

    def test_get_coefficients_for_sharding_with_annotations(self) -> None:
        """Test get_coefficients_for_sharding uses annotation-based coefficients."""

        class AnnotatedConfig(HardwarePerfConfig):
            name = "annotated"

            @fwd_coefficient(sharding_type="table_wise")
            def table_wise_fwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=5.0)

            @bwd_coefficient(sharding_type="table_wise")
            def table_wise_bwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=6.0)

        config = AnnotatedConfig()
        coeffs = config.get_coefficients_for_sharding(ShardingType.TABLE_WISE.value)
        self.assertEqual(coeffs.fwd.lookup_size_multiplier, 5.0)
        self.assertIsNotNone(coeffs.bwd)
        self.assertEqual(coeffs.bwd.lookup_size_multiplier, 6.0)

    def test_coefficients_property_builds_from_annotations(self) -> None:
        """Test that coefficients property builds config from annotations."""

        class ConfigWithAnnotations(HardwarePerfConfig):
            name = "with_annotations"

            @fwd_coefficient(sharding_type="row_wise")
            def row_wise_fwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=3.0)

            @prefetch_coefficient()
            def get_prefetch(self) -> PrefetchCoefficients:
                return PrefetchCoefficients(expected_num_lookups_coefficient=1.0)

        config = ConfigWithAnnotations()
        perf_config = config.coefficients
        self.assertIsNotNone(perf_config.row_wise)
        self.assertEqual(
            perf_config.row_wise.fwd.lookup_size_multiplier,  # pyre-ignore[16]
            3.0,
        )
        self.assertIsNotNone(perf_config.prefetch)
        self.assertEqual(perf_config.prefetch.expected_num_lookups_coefficient, 1.0)

    def test_get_device_bw_priority_kernel_specific(self) -> None:
        """Test get_device_bw priority: kernel-specific takes precedence."""

        class ConfigWithKernelBW(HardwarePerfConfig):
            name = "kernel_bw"
            kernel_device_bandwidths = {("cuda", "fused"): 1000.0}
            device_bw = 500.0

        config = ConfigWithKernelBW()
        bw = config.get_device_bw(
            compute_device="cuda",
            compute_kernel="fused",
            hbm_mem_bw=900.0,
            ddr_mem_bw=100.0,
            hbm_to_ddr_mem_bw=200.0,
        )
        self.assertEqual(bw, 1000.0)

    def test_get_device_bw_priority_device_bw_override(self) -> None:
        """Test get_device_bw priority: device_bw override when no kernel match."""

        class ConfigWithDeviceBW(HardwarePerfConfig):
            name = "device_bw"
            device_bw = 500.0

        config = ConfigWithDeviceBW()
        bw = config.get_device_bw(
            compute_device="cuda",
            compute_kernel="dense",
            hbm_mem_bw=900.0,
            ddr_mem_bw=100.0,
            hbm_to_ddr_mem_bw=200.0,
        )
        self.assertEqual(bw, 500.0)

    def test_get_device_bw_uses_kernel_bw_lookup_fallback(self) -> None:
        """Test get_device_bw falls back to kernel_bw_lookup."""
        config = HardwarePerfConfig()
        bw = config.get_device_bw(
            compute_device="cuda",
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            hbm_mem_bw=900.0,
            ddr_mem_bw=100.0,
            hbm_to_ddr_mem_bw=200.0,
        )
        self.assertEqual(bw, 900.0)

    def test_get_device_bw_invalid_kernel_raises(self) -> None:
        """Test get_device_bw raises for invalid compute kernel."""
        config = HardwarePerfConfig()
        with self.assertRaises(ValueError) as ctx:
            config.get_device_bw(
                compute_device="cuda",
                compute_kernel="invalid_kernel",
                hbm_mem_bw=900.0,
                ddr_mem_bw=100.0,
                hbm_to_ddr_mem_bw=200.0,
            )
        self.assertIn("invalid_kernel", str(ctx.exception))

    def test_post_process_perfs_returns_unchanged_by_default(self) -> None:
        """Test post_process_perfs returns input unchanged by default."""
        from torchrec.distributed.planner.types import Perf

        config = HardwarePerfConfig()
        perfs = [
            Perf(fwd_compute=1.0, fwd_comms=2.0, bwd_compute=3.0, bwd_comms=4.0),
            Perf(fwd_compute=5.0, fwd_comms=6.0, bwd_compute=7.0, bwd_comms=8.0),
        ]
        shard_sizes = [[100, 64], [200, 64]]
        result = config.post_process_perfs(
            perfs, shard_sizes, ShardingType.TABLE_WISE.value
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].fwd_compute, 1.0)
        self.assertEqual(result[1].fwd_compute, 5.0)


class ShardPerfContextTest(unittest.TestCase):
    """Tests for ShardPerfContext dataclass."""

    def test_default_values(self) -> None:
        """Test ShardPerfContext default values."""
        ctx = ShardPerfContext()
        self.assertEqual(ctx.sharding_type, "")
        self.assertEqual(ctx.compute_kernel, "")
        self.assertEqual(ctx.hash_size, 0)
        self.assertEqual(ctx.emb_dim, 0)
        self.assertEqual(ctx.world_size, 1)
        self.assertEqual(ctx.local_world_size, 1)
        self.assertFalse(ctx.is_inference)
        self.assertFalse(ctx.is_weighted)
        self.assertTrue(ctx.is_pooled)

    def test_num_hosts_property(self) -> None:
        """Test num_hosts computed property."""
        ctx = ShardPerfContext(world_size=8, local_world_size=4)
        self.assertEqual(ctx.num_hosts, 2)
        ctx2 = ShardPerfContext(world_size=4, local_world_size=4)
        self.assertEqual(ctx2.num_hosts, 1)

    def test_batch_inputs_property(self) -> None:
        """Test batch_inputs computed property."""
        ctx = ShardPerfContext(
            input_lengths=[10.0, 20.0],
            num_poolings=[1.0, 2.0],
            batch_sizes=[32, 32],
        )
        self.assertEqual(ctx.batch_inputs, 1600.0)

    def test_batch_outputs_property_pooled(self) -> None:
        """Test batch_outputs for pooled embeddings."""
        ctx = ShardPerfContext(
            input_lengths=[10.0, 20.0],
            num_poolings=[1.0, 2.0],
            batch_sizes=[32, 32],
            is_pooled=True,
        )
        self.assertEqual(ctx.batch_outputs, 96.0)

    def test_batch_outputs_property_unpooled(self) -> None:
        """Test batch_outputs for unpooled embeddings (same as batch_inputs)."""
        ctx = ShardPerfContext(
            input_lengths=[10.0, 20.0],
            num_poolings=[1.0, 2.0],
            batch_sizes=[32, 32],
            is_pooled=False,
        )
        self.assertEqual(ctx.batch_outputs, ctx.batch_inputs)

    def test_is_uvm_caching_property(self) -> None:
        """Test is_uvm_caching computed property."""
        ctx = ShardPerfContext(compute_kernel=EmbeddingComputeKernel.FUSED.value)
        self.assertFalse(ctx.is_uvm_caching)
        ctx2 = ShardPerfContext(
            compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        )
        self.assertTrue(ctx2.is_uvm_caching)

    def test_custom_initialization(self) -> None:
        """Test ShardPerfContext with custom initialization."""
        ctx = ShardPerfContext(
            sharding_type=ShardingType.TABLE_WISE.value,
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            hash_size=10000,
            emb_dim=128,
            batch_sizes=[64],
            num_poolings=[1.0],
            input_lengths=[50.0],
            world_size=8,
            local_world_size=4,
            is_inference=True,
            is_weighted=True,
            is_pooled=False,
        )
        self.assertEqual(ctx.sharding_type, ShardingType.TABLE_WISE.value)
        self.assertEqual(ctx.compute_kernel, EmbeddingComputeKernel.FUSED.value)
        self.assertEqual(ctx.hash_size, 10000)
        self.assertEqual(ctx.emb_dim, 128)
        self.assertEqual(ctx.world_size, 8)
        self.assertTrue(ctx.is_inference)
        self.assertTrue(ctx.is_weighted)
        self.assertFalse(ctx.is_pooled)


class HardwarePerfConfigIntegrationTest(unittest.TestCase):
    """Integration tests for HardwarePerfConfig with annotations."""

    def test_full_config_with_all_annotation_types(self) -> None:
        """Test a full config class with all annotation types."""

        @supported_sharding_types("table_wise", "row_wise")
        class FullConfig(HardwarePerfConfig):
            name = "full_config"

            @fwd_coefficient(sharding_type="table_wise")
            def tw_fwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=1.5)

            @bwd_coefficient(sharding_type="table_wise")
            def tw_bwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=2.0)

            @fwd_coefficient(sharding_type="row_wise")
            def rw_fwd(self) -> PerfCoefficient:
                return PerfCoefficient(lookup_size_multiplier=1.2)

            @prefetch_coefficient()
            def prefetch(self) -> PrefetchCoefficients:
                return PrefetchCoefficients(
                    expected_num_lookups_coefficient=0.5,
                    expected_num_unique_lookups_coefficient=0.3,
                )

        config = FullConfig()

        self.assertTrue(config.is_sharding_type_supported("table_wise"))
        self.assertTrue(config.is_sharding_type_supported("row_wise"))
        self.assertFalse(config.is_sharding_type_supported("column_wise"))

        tw_coeffs = config.get_coefficients_for_sharding("table_wise")
        self.assertEqual(tw_coeffs.fwd.lookup_size_multiplier, 1.5)
        self.assertIsNotNone(tw_coeffs.bwd)
        self.assertEqual(tw_coeffs.bwd.lookup_size_multiplier, 2.0)

        rw_coeffs = config.get_coefficients_for_sharding("row_wise")
        self.assertEqual(rw_coeffs.fwd.lookup_size_multiplier, 1.2)

        perf_config = config.coefficients
        self.assertIsNotNone(perf_config.prefetch)
        self.assertEqual(perf_config.prefetch.expected_num_lookups_coefficient, 0.5)


if __name__ == "__main__":
    unittest.main()
