#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Launcher for torchrec distributed benchmarks.

Selects *which* benchmark to run and *how* to run it, then dispatches to the
per-rank runner via the shared process-runner entry points:

- ``--mode local``  -> :func:`process_runner.run_local_multi_process_func`, which
  spawns ``--world-size`` local worker processes on this host (each routed through
  the single-process path).
- ``--mode remote`` -> :func:`process_runner.run_single_process_func`, the
  one-rank-per-process path. In this mode the binary is the per-rank entry point a
  torchrun/MAST job launches on each rank; ``RANK`` / ``WORLD_SIZE`` /
  ``LOCAL_RANK`` / rendezvous endpoint come from the environment (so ``--world-size``
  is ignored).

Both paths create + handshake the process group and inject a live
``SingleProcessContext`` (``ctx``), plus this rank's ``rank`` / ``world_size``,
into the selected benchmark's ``benchmark_runner``.

Examples:
    # local: spawn 2 ranks on this host and run the primitive benchmark
    buck2 run @fbcode//mode/opt \\
        fbcode//torchrec/distributed/test_utils/benchmark_utils:benchmark_launcher -- \\
        --mode=local --benchmark=primitive --world-size=2

    # remote: invoked per-rank by torchrun/MAST (rendezvous env preset)
    buck2 run @fbcode//mode/opt \\
        fbcode//torchrec/distributed/test_utils/benchmark_utils:benchmark_launcher -- \\
        --mode=remote --benchmark=module
"""

import argparse
import logging
from typing import Callable, Dict

import torch
from torchrec.distributed.benchmark import benchmark_module, benchmark_primitive
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    run_single_process_func,
)

logger: logging.Logger = logging.getLogger(__name__)

# Registry of available benchmarks: name -> per-rank runner. Each runner has the
# signature ``benchmark_runner(ctx, rank, world_size, **kwargs)`` and is invoked
# once per rank with the live context injected by the process runner.
_BENCHMARKS: Dict[str, Callable[..., None]] = {
    "primitive": benchmark_primitive.benchmark_runner,
    "module": benchmark_module.benchmark_runner,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["local", "remote"],
        default="local",
        help="local: spawn workers on this host (run_local_multi_process_func); "
        "remote: run this process as a single torchrun/MAST-placed rank "
        "(run_single_process_func). Default: local.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(_BENCHMARKS),
        required=True,
        help="which benchmark runner to launch.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="number of ranks to spawn in local mode (ignored in remote mode, "
        "where world size comes from the environment). Default: 2.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="process-group backend (defaults to nccl on GPU, gloo otherwise).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    runner: Callable[..., None] = _BENCHMARKS[args.benchmark]

    if args.mode == "local":
        # Validate we have enough GPUs for the requested world_size before
        # spawning workers, so the job fails fast in the launcher instead of
        # inside each spawned rank. ``torch.cuda`` is the device API for both
        # NVIDIA (CUDA) and AMD (HIP/ROCm) GPUs, so this works on either backend.
        # Only enforced on GPU hosts; CPU/gloo runs have no per-rank device to
        # contend for. (Remote mode gets world size and device placement from
        # torchrun/MAST, so this check does not apply.)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            assert device_count >= args.world_size, (
                "Insufficient GPUs for the requested world_size: "
                f"world_size={args.world_size}, available GPUs={device_count}"
            )
        logger.info(
            f"launching benchmark={args.benchmark} mode=local "
            f"world_size={args.world_size} backend={args.backend}"
        )
        run_local_multi_process_func(
            runner,
            world_size=args.world_size,
            backend=args.backend,
        )
    else:  # remote
        logger.info(
            f"launching benchmark={args.benchmark} mode=remote backend={args.backend}"
        )
        run_single_process_func(
            runner,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
