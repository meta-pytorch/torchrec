#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import unittest

from torchrec.distributed.planner.dry_run.protocols import CacheMetadata, PlanCache
from torchrec.distributed.types import ShardingPlan


class MockPlanCache:
    """Test implementation of PlanCache."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[ShardingPlan, CacheMetadata]] = {}

    def get(
        self, request_fingerprint: str
    ) -> tuple[ShardingPlan, CacheMetadata] | None:
        return self._store.get(request_fingerprint)

    def put(
        self,
        request_fingerprint: str,
        plan: ShardingPlan,
        metadata: CacheMetadata,
    ) -> None:
        self._store[request_fingerprint] = (plan, metadata)


class CacheMetadataTest(unittest.TestCase):
    def test_defaults(self) -> None:
        meta = CacheMetadata()
        self.assertEqual(meta.created_by, "")
        self.assertEqual(meta.sku, "")
        self.assertEqual(meta.world_size, 0)
        self.assertEqual(meta.extra, {})

    def test_with_all_fields(self) -> None:
        meta = CacheMetadata(
            created_by="dry_run_service",
            sku="H100",
            world_size=16,
            extra={"version": "2"},
        )
        self.assertEqual(meta.created_by, "dry_run_service")
        self.assertEqual(meta.sku, "H100")
        self.assertEqual(meta.world_size, 16)
        self.assertEqual(meta.extra, {"version": "2"})

    def test_frozen(self) -> None:
        meta = CacheMetadata()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            meta.created_by = "x"  # pyre-ignore[41]

    def test_equality(self) -> None:
        m1 = CacheMetadata(created_by="a", sku="H100")
        m2 = CacheMetadata(created_by="a", sku="H100")
        self.assertEqual(m1, m2)


class PlanCacheTest(unittest.TestCase):
    def test_runtime_checkable(self) -> None:
        cache = MockPlanCache()
        self.assertIsInstance(cache, PlanCache)

    def test_get_miss(self) -> None:
        cache = MockPlanCache()
        self.assertIsNone(cache.get("nonexistent"))

    def test_put_and_get(self) -> None:
        cache = MockPlanCache()
        plan = ShardingPlan(plan={})
        meta = CacheMetadata(created_by="test", sku="H100", world_size=8)
        cache.put("hash123", plan, meta)
        result = cache.get("hash123")
        self.assertIsNotNone(result)
        # get() returns (plan, metadata) so a hit is self-describing.
        assert result is not None
        self.assertIs(result[0], plan)
        self.assertIs(result[1], meta)

    def test_put_overwrites(self) -> None:
        cache = MockPlanCache()
        plan1 = ShardingPlan(plan={})
        plan2 = ShardingPlan(plan={})
        meta = CacheMetadata()
        cache.put("h", plan1, meta)
        cache.put("h", plan2, meta)
        # plan1 and plan2 compare equal (both empty), so assert identity to prove
        # the second put actually replaced the first rather than relying on ==.
        hit = cache.get("h")
        assert hit is not None
        self.assertIs(hit[0], plan2)
        self.assertIsNot(hit[0], plan1)

    def test_non_conforming_class_not_instance(self) -> None:
        class NoPlanCache:
            pass

        self.assertNotIsInstance(NoPlanCache(), PlanCache)

    def test_class_with_get_put_is_instance(self) -> None:
        class HasGetPut:
            def get(
                self, request_fingerprint: str
            ) -> tuple[ShardingPlan, CacheMetadata] | None:
                return None

            def put(
                self,
                request_fingerprint: str,
                plan: ShardingPlan,
                metadata: CacheMetadata,
            ) -> None:
                pass

        self.assertIsInstance(HasGetPut(), PlanCache)
