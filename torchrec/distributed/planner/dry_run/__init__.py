#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torchrec.distributed.planner.dry_run.api import DryRunOrchestrator  # noqa: F401
from torchrec.distributed.planner.dry_run.protocols import (  # noqa: F401
    CacheMetadata,
    PlanCache,
    PlannerExecutor,
)
from torchrec.distributed.planner.dry_run.types import (  # noqa: F401
    CacheConfig,
    DryRunRequest,
    DryRunResult,
    SkuOverride,
)

# Explicit public API: marks these names as first-class re-exports for strict
# type checkers (mypy --no-implicit-reexport, pyright strict), which otherwise
# treat imported-but-unassigned names as private to this module.
__all__ = [
    "CacheConfig",
    "CacheMetadata",
    "DryRunOrchestrator",
    "DryRunRequest",
    "DryRunResult",
    "PlanCache",
    "PlannerExecutor",
    "SkuOverride",
]
