#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for the annotations module.

These tests cover the decorators and helper functions used in the estimator package:
- Coefficient decorators: @fwd_coefficient, @bwd_coefficient, @prefetch_coefficient
- Helper functions: get_fwd_coefficient, get_bwd_coefficient, get_prefetch_coefficient
- Other decorators: @device_bw, @output_write_size, @supported_sharding_types
"""

import unittest

from torchrec.distributed.planner.estimator.annotations import (
    bwd_coefficient,
    device_bw,
    fwd_coefficient,
    get_bwd_coefficient,
    get_fwd_coefficient,
    get_output_write_size,
    get_prefetch_coefficient,
    output_write_size,
    prefetch_coefficient,
    supported_sharding_types,
)


class FwdCoefficientDecoratorTest(unittest.TestCase):
    """Tests for @fwd_coefficient decorator and get_fwd_coefficient helper."""

    def test_fwd_coefficient_decorator_marks_method(self) -> None:
        """Test that @fwd_coefficient decorator marks method with correct attributes."""

        class TestConfig:
            @fwd_coefficient(sharding_type="table_wise")
            def get_table_wise_fwd(self) -> object:
                return {"emb_lookup": 1.5}

        config = TestConfig()
        method = config.get_table_wise_fwd

        self.assertTrue(hasattr(method, "_is_fwd_coefficient"))
        self.assertTrue(method._is_fwd_coefficient)  # pyre-ignore[16]
        self.assertTrue(hasattr(method, "_coefficient_sharding_type"))
        self.assertEqual(
            method._coefficient_sharding_type, "table_wise"  # pyre-ignore[16]
        )

    def test_fwd_coefficient_decorator_with_multiple_sharding_types(self) -> None:
        """Test @fwd_coefficient decorator with multiple sharding types."""

        class TestConfig:
            @fwd_coefficient(sharding_type=["table_wise", "row_wise"])
            def get_fwd(self) -> object:
                return {"emb_lookup": 2.0}

        config = TestConfig()
        method = config.get_fwd

        self.assertTrue(method._is_fwd_coefficient)  # pyre-ignore[16]
        self.assertEqual(
            method._coefficient_sharding_type,  # pyre-ignore[16]
            ["table_wise", "row_wise"],
        )

    def test_fwd_coefficient_decorator_no_sharding_type(self) -> None:
        """Test @fwd_coefficient decorator with no sharding type (applies to all)."""

        class TestConfig:
            @fwd_coefficient()
            def get_fwd_all(self) -> object:
                return {"emb_lookup": 1.0}

        config = TestConfig()
        method = config.get_fwd_all

        self.assertTrue(method._is_fwd_coefficient)  # pyre-ignore[16]
        self.assertIsNone(method._coefficient_sharding_type)  # pyre-ignore[16]

    def test_get_fwd_coefficient_returns_coefficient_for_matching_sharding_type(
        self,
    ) -> None:
        """Test get_fwd_coefficient returns the coefficient for matching sharding type."""

        class TestConfig:
            @fwd_coefficient(sharding_type="table_wise")
            def get_table_wise_fwd(self) -> object:
                return {"emb_lookup": 1.5, "hash_size": 0.1}

        config = TestConfig()
        result = get_fwd_coefficient(config, "table_wise")

        self.assertIsNotNone(result)
        self.assertEqual(result["emb_lookup"], 1.5)  # pyre-ignore[16]

    def test_get_fwd_coefficient_returns_none_for_non_matching_sharding_type(
        self,
    ) -> None:
        """Test get_fwd_coefficient returns None when sharding type doesn't match."""

        class TestConfig:
            @fwd_coefficient(sharding_type="table_wise")
            def get_table_wise_fwd(self) -> object:
                return {"emb_lookup": 1.5}

        config = TestConfig()
        result = get_fwd_coefficient(config, "row_wise")

        self.assertIsNone(result)

    def test_get_fwd_coefficient_returns_none_when_no_decorator(self) -> None:
        """Test get_fwd_coefficient returns None when no @fwd_coefficient decorator."""

        class TestConfig:
            def some_method(self) -> object:
                return {"emb_lookup": 1.0}

        config = TestConfig()
        result = get_fwd_coefficient(config, "table_wise")

        self.assertIsNone(result)

    def test_get_fwd_coefficient_matches_any_when_no_sharding_type_specified(
        self,
    ) -> None:
        """Test get_fwd_coefficient returns coefficient when no sharding_type filter."""

        class TestConfig:
            @fwd_coefficient()  # No sharding_type = matches all
            def get_fwd_all(self) -> object:
                return {"emb_lookup": 1.0}

        config = TestConfig()

        # Should match any sharding type
        result_tw = get_fwd_coefficient(config, "table_wise")
        result_rw = get_fwd_coefficient(config, "row_wise")

        self.assertIsNotNone(result_tw)
        self.assertIsNotNone(result_rw)


class BwdCoefficientDecoratorTest(unittest.TestCase):
    """Tests for @bwd_coefficient decorator and get_bwd_coefficient helper."""

    def test_bwd_coefficient_decorator_marks_method(self) -> None:
        """Test that @bwd_coefficient decorator marks method with correct attributes."""

        class TestConfig:
            @bwd_coefficient(sharding_type="row_wise")
            def get_row_wise_bwd(self) -> object:
                return {"emb_lookup": 2.0}

        config = TestConfig()
        method = config.get_row_wise_bwd

        self.assertTrue(hasattr(method, "_is_bwd_coefficient"))
        self.assertTrue(method._is_bwd_coefficient)  # pyre-ignore[16]
        self.assertTrue(hasattr(method, "_coefficient_sharding_type"))
        self.assertEqual(
            method._coefficient_sharding_type, "row_wise"  # pyre-ignore[16]
        )

    def test_get_bwd_coefficient_returns_coefficient_for_matching_sharding_type(
        self,
    ) -> None:
        """Test get_bwd_coefficient returns the coefficient for matching sharding type."""

        class TestConfig:
            @bwd_coefficient(sharding_type="row_wise")
            def get_row_wise_bwd(self) -> object:
                return {"emb_lookup": 3.0}

        config = TestConfig()
        result = get_bwd_coefficient(config, "row_wise")

        self.assertIsNotNone(result)
        self.assertEqual(result["emb_lookup"], 3.0)  # pyre-ignore[16]

    def test_get_bwd_coefficient_returns_none_for_non_matching_sharding_type(
        self,
    ) -> None:
        """Test get_bwd_coefficient returns None when sharding type doesn't match."""

        class TestConfig:
            @bwd_coefficient(sharding_type="row_wise")
            def get_row_wise_bwd(self) -> object:
                return {"emb_lookup": 3.0}

        config = TestConfig()
        result = get_bwd_coefficient(config, "table_wise")

        self.assertIsNone(result)


class PrefetchCoefficientDecoratorTest(unittest.TestCase):
    """Tests for @prefetch_coefficient decorator and get_prefetch_coefficient helper."""

    def test_prefetch_coefficient_decorator_marks_method(self) -> None:
        """Test that @prefetch_coefficient decorator marks method correctly."""

        class TestConfig:
            @prefetch_coefficient()
            def get_prefetch(self) -> object:
                return {
                    "expected_num_lookups_coefficient": 0.5,
                    "expected_num_unique_lookups_coefficient": 0.3,
                }

        config = TestConfig()
        method = config.get_prefetch

        self.assertTrue(hasattr(method, "_is_prefetch_coefficient"))
        self.assertTrue(method._is_prefetch_coefficient)  # pyre-ignore[16]

    def test_get_prefetch_coefficient_returns_coefficients(self) -> None:
        """Test get_prefetch_coefficient returns prefetch coefficients."""

        class TestConfig:
            @prefetch_coefficient()
            def get_prefetch(self) -> object:
                return {
                    "expected_num_lookups_coefficient": 0.5,
                    "expected_num_unique_lookups_coefficient": 0.3,
                }

        config = TestConfig()
        result = get_prefetch_coefficient(config)

        self.assertIsNotNone(result)
        self.assertEqual(
            result["expected_num_lookups_coefficient"], 0.5  # pyre-ignore[16]
        )

    def test_get_prefetch_coefficient_returns_none_when_no_decorator(self) -> None:
        """Test get_prefetch_coefficient returns None when no decorator."""

        class TestConfig:
            def some_method(self) -> object:
                return {}

        config = TestConfig()
        result = get_prefetch_coefficient(config)

        self.assertIsNone(result)


class DeviceBwDecoratorTest(unittest.TestCase):
    """Tests for @device_bw class decorator."""

    def test_device_bw_decorator_sets_general_bandwidth(self) -> None:
        """Test that @device_bw sets general device bandwidth."""

        @device_bw(bandwidth=100.0)
        class TestConfig:
            pass

        config = TestConfig()

        self.assertTrue(hasattr(config, "device_bw"))
        self.assertEqual(config.device_bw, 100.0)  # pyre-ignore[16]

    def test_device_bw_decorator_with_specific_device_and_kernel(self) -> None:
        """Test @device_bw decorator with specific device and compute kernel."""

        @device_bw(bandwidth=200.0, device="cuda", compute_kernel="fused")
        class TestConfig:
            pass

        config = TestConfig()

        self.assertTrue(hasattr(config, "kernel_device_bandwidths"))
        self.assertIn(
            ("cuda", "fused"), config.kernel_device_bandwidths  # pyre-ignore[16]
        )
        self.assertEqual(
            config.kernel_device_bandwidths[("cuda", "fused")],  # pyre-ignore[16]
            200.0,
        )

    def test_device_bw_decorator_stacking(self) -> None:
        """Test multiple @device_bw decorators can be stacked."""

        @device_bw(bandwidth=200.0, device="cuda", compute_kernel="fused")
        @device_bw(bandwidth=150.0, device="cuda", compute_kernel="dense")
        class TestConfig:
            pass

        config = TestConfig()

        self.assertIn(
            ("cuda", "fused"), config.kernel_device_bandwidths  # pyre-ignore[16]
        )
        self.assertIn(
            ("cuda", "dense"), config.kernel_device_bandwidths  # pyre-ignore[16]
        )
        self.assertEqual(
            config.kernel_device_bandwidths[("cuda", "fused")],  # pyre-ignore[16]
            200.0,
        )
        self.assertEqual(
            config.kernel_device_bandwidths[("cuda", "dense")],  # pyre-ignore[16]
            150.0,
        )


class OutputWriteSizeDecoratorTest(unittest.TestCase):
    """Tests for @output_write_size decorator and get_output_write_size helper."""

    def test_output_write_size_decorator_marks_method(self) -> None:
        """Test that @output_write_size decorator marks method correctly."""

        class TestConfig:
            @output_write_size(sharding_type="table_wise")
            def custom_output_write(self, ctx: object) -> float:
                return 100.0

        config = TestConfig()
        method = config.custom_output_write

        self.assertTrue(hasattr(method, "_is_custom_output_write_size"))
        self.assertTrue(method._is_custom_output_write_size)  # pyre-ignore[16]

    def test_get_output_write_size_returns_method_for_matching_sharding_type(
        self,
    ) -> None:
        """Test get_output_write_size returns method for matching sharding type."""

        class TestConfig:
            @output_write_size(sharding_type="table_wise")
            def custom_output_write(self, ctx: object) -> float:
                return 100.0

        config = TestConfig()
        method = get_output_write_size(config, "table_wise")

        self.assertIsNotNone(method)

    def test_get_output_write_size_returns_none_for_non_matching(self) -> None:
        """Test get_output_write_size returns None for non-matching sharding type."""

        class TestConfig:
            @output_write_size(sharding_type="table_wise")
            def custom_output_write(self, ctx: object) -> float:
                return 100.0

        config = TestConfig()
        method = get_output_write_size(config, "row_wise")

        self.assertIsNone(method)


class SupportedShardingTypesDecoratorTest(unittest.TestCase):
    """Tests for @supported_sharding_types class decorator."""

    def test_supported_sharding_types_sets_class_attribute(self) -> None:
        """Test that @supported_sharding_types sets _supported_sharding_types."""

        @supported_sharding_types("table_wise", "row_wise")
        class TestConfig:
            pass

        config = TestConfig()

        self.assertTrue(hasattr(config, "_supported_sharding_types"))
        self.assertIn("table_wise", config._supported_sharding_types)  # pyre-ignore[16]
        self.assertIn("row_wise", config._supported_sharding_types)  # pyre-ignore[16]
        self.assertNotIn(
            "column_wise", config._supported_sharding_types  # pyre-ignore[16]
        )


class MultipleCoefficientDecoratorTest(unittest.TestCase):
    """Tests for configs with multiple coefficient decorators."""

    def test_config_with_both_fwd_and_bwd_coefficients(self) -> None:
        """Test config that has both fwd and bwd coefficient decorators."""

        class TestConfig:
            @fwd_coefficient(sharding_type="table_wise")
            def get_fwd_table_wise(self) -> object:
                return {"emb_lookup": 1.0}

            @bwd_coefficient(sharding_type="table_wise")
            def get_bwd_table_wise(self) -> object:
                return {"emb_lookup": 2.0}

            @fwd_coefficient(sharding_type="row_wise")
            def get_fwd_row_wise(self) -> object:
                return {"emb_lookup": 1.5}

        config = TestConfig()

        # Get fwd coefficients
        fwd_tw = get_fwd_coefficient(config, "table_wise")
        fwd_rw = get_fwd_coefficient(config, "row_wise")
        bwd_tw = get_bwd_coefficient(config, "table_wise")
        bwd_rw = get_bwd_coefficient(config, "row_wise")

        self.assertIsNotNone(fwd_tw)
        self.assertEqual(fwd_tw["emb_lookup"], 1.0)  # pyre-ignore[16]

        self.assertIsNotNone(fwd_rw)
        self.assertEqual(fwd_rw["emb_lookup"], 1.5)  # pyre-ignore[16]

        self.assertIsNotNone(bwd_tw)
        self.assertEqual(bwd_tw["emb_lookup"], 2.0)  # pyre-ignore[16]

        # No bwd for row_wise
        self.assertIsNone(bwd_rw)

    def test_config_with_prefetch_and_fwd_coefficients(self) -> None:
        """Test config that has both fwd and prefetch coefficient decorators."""

        class TestConfig:
            @fwd_coefficient(sharding_type="table_wise")
            def get_fwd(self) -> object:
                return {"emb_lookup": 1.0}

            @prefetch_coefficient()
            def get_prefetch(self) -> object:
                return {"expected_num_lookups_coefficient": 0.5}

        config = TestConfig()

        fwd = get_fwd_coefficient(config, "table_wise")
        prefetch = get_prefetch_coefficient(config)

        self.assertIsNotNone(fwd)
        self.assertIsNotNone(prefetch)
        self.assertEqual(fwd["emb_lookup"], 1.0)  # pyre-ignore[16]
        self.assertEqual(
            prefetch["expected_num_lookups_coefficient"], 0.5  # pyre-ignore[16]
        )
