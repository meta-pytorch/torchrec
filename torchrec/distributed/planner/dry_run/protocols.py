#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

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
    # Additional key-value metadata for debugging and tracing
    extra: Dict[str, str] = field(default_factory=dict)


@runtime_checkable
class PlanCache(Protocol):
    """Protocol for caching and retrieving sharding plans by context hash.

    Implementations may back the cache with in-memory stores, Manifold,
    or other persistent storage.
    """

    def get(self, context_hash: str) -> Optional[ShardingPlan]: ...

    def put(
        self,
        context_hash: str,
        plan: ShardingPlan,
        metadata: CacheMetadata,
    ) -> None: ...
