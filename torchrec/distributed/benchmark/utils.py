#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging
import os
import platform
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)


def get_gpu_type() -> str:
    """Return the GPU device name, or 'N/A' if CUDA is unavailable."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


def get_cpu_type() -> str:
    """Return the CPU model string."""
    return platform.processor() or platform.machine()


def dump_benchmark_result(
    result: Any,
    output_dir: str,
    world_size: int,
) -> None:
    """Write benchmark result to a JSON file in *output_dir*.

    The file is named ``torchrec_benchmark_<short_name>_<rank>.json`` and
    contains all metrics from ``to_dict()``, plus hardware and source info.

    Args:
        result: A ``BenchmarkResult`` instance (typed as ``Any`` to avoid a
            circular dependency with ``base``).
        output_dir: Directory where the JSON file is written.
        world_size: Number of ranks in the distributed run.
    """
    data: dict[str, object] = {
        "short_name": result.short_name,
        "rank": result.rank,
        "world_size": world_size,
        "gpu_type": get_gpu_type(),
        "cpu_type": get_cpu_type(),
        "metrics": result.to_dict(),
    }

    path = os.path.join(
        output_dir,
        f"torchrec_benchmark_{result.short_name}_{result.rank}.json",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    logger.info(f"Benchmark result written to {path}")
