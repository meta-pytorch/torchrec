#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict

import torch.nn as nn
from torchrec.distributed.planner.dry_run.types import (
    CacheConfig,
    DryRunRequest,
    DryRunResult,
    SkuOverride,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    ShardingPlanRequest,
    ShardingPlanResult,
)


class DryRunRequestTest(unittest.TestCase):
    def _create_request(self, **kwargs: Any) -> DryRunRequest:
        defaults: Dict[str, Any] = {
            "model": nn.Linear(10, 10),
            "sharders": [],
            "sku_list": ["H100"],
            "training_framework": "apf",
            "world_size": 8,
            "local_world_size": 8,
            "batch_size": 512,
        }
        defaults.update(kwargs)
        return DryRunRequest(**defaults)

    def test_valid_request_construction(self) -> None:
        override = SkuOverride(hbm_gb=80.0, ddr_gb=512.0)
        cache = CacheConfig(enabled=True, ttl_seconds=3600, manifold_bucket="test")
        request = self._create_request(
            pod_size=8,
            hbm_gb=80.0,
            ddr_gb=512.0,
            per_sku_overrides={"H100": override},
            cache_config=cache,
        )
        self.assertEqual(request.world_size, 8)
        self.assertEqual(request.hbm_gb, 80.0)
        self.assertIsNotNone(request.per_sku_overrides)
        self.assertIsNotNone(request.cache_config)
        self.assertEqual(request.cache_config.enabled, True)

    def test_empty_sku_list_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sku_list must not be empty"):
            self._create_request(sku_list=[])

    def test_invalid_single_field_rejected(self) -> None:
        cases = [
            ({"world_size": 0}, "world_size must be positive"),
            ({"world_size": -1}, "world_size must be positive"),
            ({"local_world_size": 0}, "local_world_size must be positive"),
            ({"batch_size": 0}, "batch_size must be positive"),
            ({"hbm_gb": -1.0}, "hbm_gb must be non-negative"),
            ({"ddr_gb": -10.0}, "ddr_gb must be non-negative"),
            ({"pod_size": 0}, "pod_size must be positive"),
            ({"pod_size": -1}, "pod_size must be positive"),
        ]
        for overrides, expected_msg in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, expected_msg):
                    self._create_request(**overrides)

    def test_cross_field_validation(self) -> None:
        with self.subTest("local exceeds world"):
            with self.assertRaisesRegex(
                ValueError, "local_world_size.*must not exceed world_size"
            ):
                self._create_request(world_size=4, local_world_size=8)
        with self.subTest("world not divisible by local"):
            with self.assertRaisesRegex(
                ValueError, "world_size.*must be divisible by local_world_size"
            ):
                self._create_request(world_size=10, local_world_size=3)

    def test_zero_hbm_gb_allowed(self) -> None:
        request = self._create_request(hbm_gb=0.0)
        self.assertEqual(request.hbm_gb, 0.0)

    def test_model_callable_factory(self) -> None:
        constructed = nn.Linear(10, 10)
        factory = lambda: constructed  # noqa: E731
        request = self._create_request(model=factory)
        self.assertNotIsInstance(request.model, nn.Module)
        # pyre-ignore[29]: factory is callable
        self.assertIs(request.model(), constructed)

    def test_model_module_vs_factory_disambiguation(self) -> None:
        module = nn.Linear(10, 10)
        factory = lambda: module  # noqa: E731
        req_module = self._create_request(model=module)
        req_factory = self._create_request(model=factory)
        self.assertIsInstance(req_module.model, nn.Module)
        self.assertNotIsInstance(req_factory.model, nn.Module)

    def test_multi_sku_with_per_sku_overrides(self) -> None:
        request = self._create_request(
            sku_list=["H100", "GB200"],
            per_sku_overrides={
                "H100": SkuOverride(hbm_gb=80.0),
                "GB200": SkuOverride(hbm_gb=192.0),
            },
        )
        self.assertEqual(len(request.sku_list), 2)
        assert request.per_sku_overrides is not None
        self.assertEqual(request.per_sku_overrides["H100"].hbm_gb, 80.0)
        self.assertEqual(request.per_sku_overrides["GB200"].hbm_gb, 192.0)

    def test_fingerprint_deterministic(self) -> None:
        req = self._create_request()
        self.assertEqual(req.fingerprint("H100"), req.fingerprint("H100"))

    def test_fingerprint_differs_by_sku(self) -> None:
        req = self._create_request(sku_list=["H100", "GB200"])
        self.assertNotEqual(req.fingerprint("H100"), req.fingerprint("GB200"))

    def test_fingerprint_differs_by_config(self) -> None:
        req1 = self._create_request(batch_size=512)
        req2 = self._create_request(batch_size=1024)
        self.assertNotEqual(req1.fingerprint("H100"), req2.fingerprint("H100"))

    def test_fingerprint_includes_per_sku_override(self) -> None:
        req_no_override = self._create_request()
        req_with_override = self._create_request(
            per_sku_overrides={"H100": SkuOverride(hbm_gb=80.0)},
        )
        self.assertNotEqual(
            req_no_override.fingerprint("H100"),
            req_with_override.fingerprint("H100"),
        )

    def test_fingerprint_differs_by_constraints(self) -> None:
        # Constraints change the resulting plan, so they must change the
        # fingerprint. They participate via the composed request_hash; before
        # that composition, constraint-only differences collided.
        req1 = self._create_request()
        req2 = self._create_request(
            constraints={"t1": ParameterConstraints(min_partition=8)},
        )
        self.assertNotEqual(req1.fingerprint("H100"), req2.fingerprint("H100"))

    def test_fingerprint_differs_by_launcher_hardware(self) -> None:
        req1 = self._create_request(launcher_hardware="ZIONEX")
        req2 = self._create_request(launcher_hardware="GRANDTETON")
        self.assertNotEqual(req1.fingerprint("H100"), req2.fingerprint("H100"))

    def test_fingerprint_length(self) -> None:
        req = self._create_request()
        self.assertEqual(len(req.fingerprint("H100")), 16)

    def test_inherits_sharding_plan_request(self) -> None:
        req = self._create_request()
        self.assertIsInstance(req, ShardingPlanRequest)

    def test_empty_training_framework_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "training_framework is required for dry-run"
        ):
            self._create_request(training_framework="")

    def test_base_fields_accessible(self) -> None:
        req = self._create_request(
            launcher_hardware="ZIONEX",
        )
        self.assertEqual(req.launcher_hardware, "ZIONEX")
        self.assertEqual(req.training_framework, "apf")


class SkuOverrideTest(unittest.TestCase):
    def test_negative_value_rejected(self) -> None:
        cases = [
            ({"hbm_gb": -1.0}, "hbm_gb must be non-negative"),
            ({"ddr_gb": -512.0}, "ddr_gb must be non-negative"),
            ({"intra_host_bw": -100.0}, "intra_host_bw must be non-negative"),
            ({"inter_host_bw": -50.0}, "inter_host_bw must be non-negative"),
        ]
        for kwargs, expected_msg in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, expected_msg):
                    SkuOverride(**kwargs)

    def test_zero_values_allowed(self) -> None:
        override = SkuOverride(hbm_gb=0.0, ddr_gb=0.0)
        self.assertEqual(override.hbm_gb, 0.0)
        self.assertEqual(override.ddr_gb, 0.0)


class CacheConfigTest(unittest.TestCase):
    def test_enabled_requires_manifold_bucket(self) -> None:
        with self.assertRaisesRegex(ValueError, "manifold_bucket is required"):
            CacheConfig(enabled=True, manifold_bucket="")

    def test_enabled_with_bucket_valid(self) -> None:
        config = CacheConfig(
            enabled=True, ttl_seconds=3600, manifold_bucket="dry_run_cache"
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.manifold_bucket, "dry_run_cache")

    def test_disabled_without_bucket_valid(self) -> None:
        config = CacheConfig(enabled=False)
        self.assertFalse(config.enabled)
        self.assertEqual(config.manifold_bucket, "")

    def test_invalid_ttl_rejected(self) -> None:
        for ttl in (0, -1):
            with self.subTest(ttl=ttl):
                with self.assertRaisesRegex(ValueError, "ttl_seconds must be positive"):
                    CacheConfig(ttl_seconds=ttl)


class DryRunResultTest(unittest.TestCase):
    def _create_success_result(self, **kwargs: Any) -> DryRunResult:
        defaults: Dict[str, Any] = {
            "sku": "H100",
            "success": True,
            "sharding_plan": None,
            "planner_failure_reason": None,
            "estimated_max_hbm_bytes": 40 * 1024**3,
            "estimated_max_ddr_bytes": 200 * 1024**3,
            "request_fingerprint": "abc123def456ab00",
        }
        defaults.update(kwargs)
        return DryRunResult(**defaults)

    def _create_failure_result(self, **kwargs: Any) -> DryRunResult:
        defaults: Dict[str, Any] = {
            "sku": "H100",
            "success": False,
            "sharding_plan": None,
            "planner_failure_reason": "OOM_HBM",
            "estimated_max_hbm_bytes": 90 * 1024**3,
            "estimated_max_ddr_bytes": 200 * 1024**3,
            "request_fingerprint": "abc123def456ab00",
        }
        defaults.update(kwargs)
        return DryRunResult(**defaults)

    def test_empty_sku_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sku must not be empty"):
            self._create_success_result(sku="")

    def test_empty_request_fingerprint_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "request_fingerprint must not be empty"
        ):
            self._create_success_result(request_fingerprint="")

    def test_negative_hbm_bytes_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "estimated_max_hbm_bytes must be non-negative"
        ):
            self._create_success_result(estimated_max_hbm_bytes=-1)

    def test_negative_ddr_bytes_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "estimated_max_ddr_bytes must be non-negative"
        ):
            self._create_success_result(estimated_max_ddr_bytes=-1)

    def test_success_with_failure_reason_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "planner_failure_reason must be None when success is True"
        ):
            self._create_success_result(planner_failure_reason="OOM_HBM")

    def test_failure_without_reason_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "planner_failure_reason is required when success is False"
        ):
            self._create_failure_result(planner_failure_reason=None)

    def test_negative_estimated_qps_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "estimated_qps must be non-negative"):
            self._create_success_result(estimated_qps=-1.0)

    def test_negative_critical_path_ms_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "critical_path_ms must be non-negative"
        ):
            self._create_success_result(critical_path_ms=-0.5)

    def test_validation_warnings_tuple(self) -> None:
        result = self._create_success_result()
        self.assertEqual(result.validation_warnings, ())
        warnings = ("high HBM usage", "close to DDR limit", "untested SKU")
        result_with = self._create_success_result(validation_warnings=warnings)
        self.assertEqual(len(result_with.validation_warnings), 3)
        self.assertIn("untested SKU", result_with.validation_warnings)

    def test_manifold_url(self) -> None:
        # Uses the inherited sharding_plan_manifold_url (no DryRunResult-specific
        # manifold_url field).
        result = self._create_success_result()
        self.assertIsNone(result.sharding_plan_manifold_url)
        url = "manifold://tbe_benchmarking/tree/dry_run/plans/abc.json"
        result_with = self._create_success_result(sharding_plan_manifold_url=url)
        self.assertEqual(result_with.sharding_plan_manifold_url, url)

    def test_inherits_sharding_plan_result(self) -> None:
        result = self._create_success_result()
        self.assertIsInstance(result, ShardingPlanResult)
