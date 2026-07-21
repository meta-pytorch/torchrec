#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional

from torchrec.distributed.planner.types import ShardingPlanRequest, ShardingPlanResult


@dataclass(frozen=True)
class SkuOverride:
    """Per-SKU hardware overrides for dry-run topology building.

    HARDWARE_CAPABILITIES registry values can be stale vs MAST runtime.
    Overrides enable what-if exploration and stricter dry runs for APS,
    and support TC_ANY multi-SKU validation with per-SKU tuning.
    """

    # Override HBM capacity (GB) when registry values are stale or for what-if testing
    hbm_gb: Optional[float] = None
    # Override DDR capacity (GB) — DDR varies inversely with HBM on many SKUs
    ddr_gb: Optional[float] = None
    # Override intra-host bandwidth (GB/s) for NVLink/NVSwitch topology modeling
    intra_host_bw: Optional[float] = None
    # Override inter-host bandwidth (GB/s) for cross-node communication modeling
    inter_host_bw: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("hbm_gb", "ddr_gb", "intra_host_bw", "inter_host_bw"):
            val = getattr(self, name)
            if val is not None and val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for dry-run plan caching.

    Same topology+model params produce the same sharding plan, so caching
    avoids re-running the expensive ILP solver on repeated requests.
    """

    # Master switch — disabled by default to avoid stale plans during development
    enabled: bool = False
    # Cache TTL — None means entries never expire; set for production use
    ttl_seconds: Optional[int] = None
    # Manifold bucket for persistent cross-session caching
    manifold_bucket: str = ""

    def __post_init__(self) -> None:
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {self.ttl_seconds}")
        if self.enabled and not self.manifold_bucket:
            raise ValueError("manifold_bucket is required when caching is enabled")


@dataclass(frozen=True)
class DryRunRequest(ShardingPlanRequest):
    """Immutable request for dry-run sharding plan validation across SKUs.

    Extends ShardingPlanRequest with dry-run-specific fields for multi-SKU
    validation, per-SKU hardware overrides, plan caching, and safety margins.
    The planner runs offline (no GPU required) against each SKU in sku_list.

    sku_list vs the inherited launcher_hardware -- why both exist:
        - launcher_hardware (base ShardingPlanRequest): the single hardware the
          job will actually launch on. It may be a *concrete* type ("ZIONEX",
          "GRANDTETON") or an *abstract/fungible* umbrella ("TC_ANY"), which
          means "could land on any of several SKUs".
        - sku_list (here): the *concrete* set of SKUs to validate offline. When
          launcher_hardware is abstract (e.g. TC_ANY), it expands one-to-many
          into the candidate SKUs to check; when concrete it maps one-to-one.
        They are different layers, not duplicates: launcher_hardware is the
        upstream launch intent (one value, possibly abstract), while sku_list is
        the dry-run sweep set (concrete, possibly many) the orchestrator iterates
        over to produce one result per SKU. The abstract->concrete expansion is a
        framework/resolver concern, so both are kept: the base carries the launch
        intent; the dry-run request carries the resolved candidates.
    """

    # Concrete SKUs to validate offline, one ShardingPlanResult per SKU. This is
    # the resolved/expanded form of the inherited launcher_hardware (see the
    # class docstring): an abstract launcher_hardware like "TC_ANY" expands to
    # many SKUs here; a concrete one maps to a single-element list. Enables
    # multi-SKU (TC_ANY) and cartesian (AVT) validation.
    sku_list: List[str] = field(default_factory=list)
    # Per-SKU hardware overrides for topology tuning
    per_sku_overrides: Optional[Dict[str, SkuOverride]] = None
    # Plan caching configuration
    cache_config: Optional[CacheConfig] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.sku_list:
            raise ValueError("sku_list must not be empty")
        if not self.training_framework:
            raise ValueError("training_framework is required for dry-run requests")
        if self.per_sku_overrides:
            unknown = set(self.per_sku_overrides) - set(self.sku_list)
            if unknown:
                raise ValueError(
                    "per_sku_overrides references SKUs not in sku_list: "
                    f"{sorted(unknown)}"
                )

    def fingerprint(self, sku: str) -> str:
        """Stable hash identifying a single (request, SKU) planning unit.

        Composes the request-level identity (ShardingPlanRequest.request_hash)
        with the SKU and its per-SKU override. request_hash already covers every
        base planner-affecting param -- world_size, local_world_size, batch_size,
        pod_size, hbm_gb, ddr_gb, training_framework, launcher_hardware,
        constraints, and the full PlannerConfig (including storage_reservation_policy
        and storage_reservation_percentage) -- so this method only adds the
        SKU-specific delta. Because storage reservation participates in
        request_hash, two requests that differ only in storage-reservation policy
        (which changes the plan) get different fingerprints -- no cache collision.
        Building on request_hash (rather than re-listing the base params) keeps the
        two in sync and ensures constraints/launcher_hardware/storage-reservation
        participate in the hash; omitting them previously let configs that produce
        different plans collide.

        Used as the correlation key between request configs and results — enables
        TUO iterative validation (same SKU, different configs), plan cache keys,
        and debugging which config produced which result.
        """
        if sku not in self.sku_list:
            raise ValueError(f"sku {sku!r} is not in sku_list {self.sku_list}")
        override = None
        if self.per_sku_overrides and sku in self.per_sku_overrides:
            # asdict so a newly added SkuOverride field automatically enters the
            # hash; hand-listing fields would silently drop it -> cache collisions.
            override = asdict(self.per_sku_overrides[sku])
        parts = {
            "request_hash": self.request_hash,
            "sku": sku,
            "override": override,
        }
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DryRunResult(ShardingPlanResult):
    """Immutable result of dry-run sharding plan validation for a single SKU.

    Extends ShardingPlanResult with the per-SKU request fingerprint. The
    uploaded-plan URL reuses the inherited sharding_plan_manifold_url; the
    request-level identity reuses the inherited request_hash (by content) and
    request_id (by instance).

    Invariant: ``request_fingerprint == request.fingerprint(sku)`` -- the SKU the
    fingerprint encodes must match the top-level ``sku``. Build via
    ``DryRunResult.from_result(request, sku, base_result)``, which guarantees it;
    direct construction is allowed but must uphold it.
    """

    # Per-(request, SKU) key from DryRunRequest.fingerprint(sku); keys the
    # per-SKU results map and correlates TUO iterative validation runs.
    # Distinct from the inherited request-level fields: it composes request_hash
    # with the SKU and per-SKU override, so each SKU's result gets a unique key
    # (the request-level request_hash/request_id are the same across SKUs).
    # Effectively required: the "" default exists only because the base class has
    # defaulted fields (so every subclass field must default too); __post_init__
    # rejects an empty value, so callers must always supply a real fingerprint.
    request_fingerprint: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.request_fingerprint:
            raise ValueError("request_fingerprint must not be empty")

    @classmethod
    def from_result(
        cls,
        request: DryRunRequest,
        sku: str,
        result: ShardingPlanResult,
    ) -> "DryRunResult":
        """Wrap a base ShardingPlanResult as a DryRunResult for ``sku``.

        The sanctioned constructor: it derives ``request_fingerprint =
        request.fingerprint(sku)`` and forces ``sku=sku``, so the top-level SKU
        and the SKU encoded in the fingerprint cannot disagree -- the mismatch is
        unrepresentable through this path.
        """
        base_fields = {
            f.name: getattr(result, f.name) for f in fields(ShardingPlanResult)
        }
        base_fields["sku"] = sku
        return cls(**base_fields, request_fingerprint=request.fingerprint(sku))
