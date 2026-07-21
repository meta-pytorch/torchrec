#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torchrec.distributed.types import ShardingPlan


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata associated with a cached sharding plan entry."""

    # Identity of the process or user that created this cache entry
    created_by: str = ""
    # Hardware SKU this cached plan was generated for
    sku: str = ""
    # World size the cached plan was generated with
    world_size: int = 0
    # Additional key-value metadata for debugging and tracing. Note: frozen=True
    # only blocks rebinding fields; this dict is still mutable in place, so treat
    # it as read-only after construction.
    extra: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class PlanCache(Protocol):
    """Protocol for caching and retrieving sharding plans by request fingerprint.

    Implementations may back the cache with in-memory stores, Manifold,
    or other persistent storage.

    The ``request_fingerprint`` key is exactly the value returned by
    ``DryRunRequest.fingerprint(sku)`` (``request_hash`` + the SKU + that SKU's
    override) -- there is no extra runtime state -- so the cache keys on the same
    per-(request, SKU) identity carried by ``DryRunResult.request_fingerprint``.
    """

    def get(
        self, request_fingerprint: str
    ) -> tuple[ShardingPlan, CacheMetadata] | None:
        """Return the cached ``(plan, metadata)`` for a fingerprint, or None.

        The metadata is returned alongside the plan so a cache hit can be
        attributed (which SKU/world_size it was generated for, who created it)
        without a second lookup.
        """
        ...

    def put(
        self,
        request_fingerprint: str,
        plan: ShardingPlan,
        metadata: CacheMetadata,
    ) -> None: ...
