#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Per-hardware tuning knobs for the Triton TBE backward kernels.

TBE does no tensor-core work, so the kernel bodies are hardware portable and
every target runs the same Triton source. Only the values below differ per
target, and hardware-specific execution features (currently CLC, which needs
TLX on Blackwell) are additive flags rather than separate kernel bodies.

To tune a new target, add a `TbeBackwardConfig` and a branch in
`resolve_backward_config`; nothing in the kernels changes.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

# Historical grid constants, kept as named defaults so each target states its
# own value explicitly rather than inheriting a magic number.
_CUDA_BASE_GRID = 24576  # 192 SMs * 32 warps/SM * 4 waves
_AMD_BASE_GRID = 65536  # 256 CUs * 64 warps/CU * 4 waves


@dataclass(frozen=True)
class TbeBackwardConfig:
    name: str

    # Programs launched for each backward tier. All three kernels self-distribute
    # work with while-loops, so these are throughput knobs and not correctness
    # constraints. Larger grids give the static partition more, smaller chunks to
    # balance across, which matters because segment lengths within one batch span
    # several orders of magnitude.
    short_run_programs: int
    long_run_fused_programs: int
    long_run_accum_programs: int

    # Segment length at which a run leaves the short-run tier for the long-run
    # (split + atomically accumulate) tier.
    long_run_threshold: int

    # Rows gathered per iteration by the short-run kernels. Each buffered row
    # keeps BLOCK_SIZE 64-bit addresses live, so this trades memory-level
    # parallelism against register footprint, and therefore occupancy.
    short_run_buffer_size_unweighted: int
    short_run_buffer_size_weighted: int

    # Same knob, same trade-off, for the long-run grad-accumulation kernels.
    # These kept the historical 8/16 when the short-run widths were tuned down,
    # and paid the same register cliff: on Blackwell the unweighted long-run
    # kernel sat at 158 registers/thread and 17.6% occupancy at width 8, versus
    # 62 registers and 44.7% at width 2 -- with byte-identical memory traffic
    # and an identical global-load request count, so the entire delta is
    # latency hiding.
    long_run_accum_buffer_size_unweighted: int
    long_run_accum_buffer_size_weighted: int

    num_warps: int

    # Split the short-run tier into one launch per next_pow2(D) bucket so each
    # launch sizes BLOCK_SIZE to its own rows instead of the global max.
    # Portable: plain Triton, no hardware feature needed.
    enable_dim_bucketing: bool

    # Cluster Launch Control: Blackwell-only work stealing. Purely additive on
    # top of the portable path; also requires TLX to be importable.
    allow_clc: bool


# Blackwell (B200 / GB200). Measured on GB200 (triton-beta) against the
# heavy-table-sharing GEM-v6 shape. The short-run grid saturates at 32x the base
# grid -- 0.97x CUDA TBE on the portable path, 1.01x with CLC layered on -- and
# the buffer sizes sit at a true interior optimum: narrower starves memory-level
# parallelism, wider spills occupancy (width 8 costs 184 registers/thread and
# 12.5% occupancy, versus 64 and 49.9% at width 2).
_BLACKWELL = TbeBackwardConfig(
    name="blackwell",
    short_run_programs=32 * _CUDA_BASE_GRID,
    long_run_fused_programs=32 * _CUDA_BASE_GRID,
    long_run_accum_programs=_CUDA_BASE_GRID,
    long_run_threshold=256,
    short_run_buffer_size_unweighted=2,
    short_run_buffer_size_weighted=4,
    long_run_accum_buffer_size_unweighted=2,
    long_run_accum_buffer_size_weighted=4,
    num_warps=1,
    enable_dim_bucketing=True,
    allow_clc=True,
)

# Hopper (H100). UNTUNED: retains the historical grid. The Blackwell sweep found
# the base grid badly undersized for large run counts (1.66x slower than 32x on
# the portable path), so `short_run_programs` is the most likely win from a
# tuning pass here.
_HOPPER = TbeBackwardConfig(
    name="hopper",
    short_run_programs=_CUDA_BASE_GRID,
    long_run_fused_programs=_CUDA_BASE_GRID,
    long_run_accum_programs=_CUDA_BASE_GRID,
    long_run_threshold=256,
    short_run_buffer_size_unweighted=2,
    short_run_buffer_size_weighted=4,
    long_run_accum_buffer_size_unweighted=2,
    long_run_accum_buffer_size_weighted=4,
    num_warps=1,
    enable_dim_bucketing=True,
    allow_clc=False,
)

# CDNA3 (MI300X). UNTUNED: retains the historical AMD grid.
_MI300X = TbeBackwardConfig(
    name="mi300x",
    short_run_programs=_AMD_BASE_GRID,
    long_run_fused_programs=_AMD_BASE_GRID,
    long_run_accum_programs=_AMD_BASE_GRID,
    long_run_threshold=256,
    short_run_buffer_size_unweighted=2,
    short_run_buffer_size_weighted=4,
    long_run_accum_buffer_size_unweighted=2,
    long_run_accum_buffer_size_weighted=4,
    num_warps=1,
    enable_dim_bucketing=True,
    allow_clc=False,
)

# CDNA4 (MI350X). UNTUNED: currently identical to MI300X.
_MI350X = TbeBackwardConfig(
    name="mi350x",
    short_run_programs=_AMD_BASE_GRID,
    long_run_fused_programs=_AMD_BASE_GRID,
    long_run_accum_programs=_AMD_BASE_GRID,
    long_run_threshold=256,
    short_run_buffer_size_unweighted=2,
    short_run_buffer_size_weighted=4,
    long_run_accum_buffer_size_unweighted=2,
    long_run_accum_buffer_size_weighted=4,
    num_warps=1,
    enable_dim_bucketing=True,
    allow_clc=False,
)

# Used when the device matches no known target: historical grid, no add-ons.
_PORTABLE_DEFAULT = TbeBackwardConfig(
    name="portable_default",
    short_run_programs=_CUDA_BASE_GRID,
    long_run_fused_programs=_CUDA_BASE_GRID,
    long_run_accum_programs=_CUDA_BASE_GRID,
    long_run_threshold=256,
    short_run_buffer_size_unweighted=2,
    short_run_buffer_size_weighted=4,
    long_run_accum_buffer_size_unweighted=2,
    long_run_accum_buffer_size_weighted=4,
    num_warps=1,
    enable_dim_bucketing=True,
    allow_clc=False,
)


@lru_cache(maxsize=None)
def _resolve_by_key(is_hip: bool, arch_key: str) -> TbeBackwardConfig:
    """Map a device identity to a config. Cached: the mapping is static."""
    if is_hip:
        if "gfx95" in arch_key:
            return _MI350X
        if "gfx94" in arch_key:
            return _MI300X
        return _PORTABLE_DEFAULT
    # CUDA compute capability major version: 9 = Hopper, 10/11 = Blackwell.
    if arch_key.startswith(("10", "11")):
        return _BLACKWELL
    if arch_key.startswith("9"):
        return _HOPPER
    return _PORTABLE_DEFAULT


def resolve_backward_config(
    device: torch.device | int | None = None,
) -> TbeBackwardConfig:
    """Pick the tuning config for the device the TBE weights live on."""
    if torch.version.hip is not None:
        props = torch.cuda.get_device_properties(device)
        return _resolve_by_key(True, getattr(props, "gcnArchName", ""))
    major, _minor = torch.cuda.get_device_capability(device)
    return _resolve_by_key(False, str(major))
