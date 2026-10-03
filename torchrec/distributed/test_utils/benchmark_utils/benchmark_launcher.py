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

``--benchmark=all`` runs every benchmark back to back in one dispatch, reusing the
same hosts and process group.

Examples:
    # local: spawn 2 ranks on this host and run the KJT (sparse index) A2A benchmark
    python -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \\
        --mode=local --benchmark=primitive --name=kjt_a2a --world-size=2

    # local: run the row-wise sharded EBC benchmark through the train pipeline
    python -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \\
        --mode=local --benchmark=module --name=row_wise --pipeline=true --world-size=2

    # local: run every benchmark
    python -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \\
        --mode=local --benchmark=all --world-size=2

    # remote: invoked per-rank by torchrun/MAST (rendezvous env preset)
    python -m torchrec.distributed.test_utils.benchmark_utils.benchmark_launcher \\
        --mode=remote --benchmark=all
"""

import argparse
import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchrec.distributed.benchmark import benchmark_ebc, benchmark_primitive
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    run_single_process_func,
    SingleProcessContext,
)

logger: logging.Logger = logging.getLogger(__name__)

# Registry of available benchmarks: name -> per-rank runner. Each runner has the
# signature ``benchmark_runner(ctx, rank, world_size, **kwargs)`` and is invoked
# once per rank with the live context injected by the process runner. Runners may
# return a per-rank result (e.g. ``BenchmarkResult``) or ``None``; the launcher
# ignores the value, so the registry is typed ``Callable[..., Any]``.
_BENCHMARKS: Dict[str, Callable[..., Any]] = {
    "primitive": benchmark_primitive.benchmark_runner,
    "module": benchmark_ebc.benchmark_runner,
}

# Valid ``--name`` values for each entry in ``_BENCHMARKS``.
_AVAILABLE_NAMES: Dict[str, Callable[[], List[str]]] = {
    "primitive": benchmark_primitive.available_primitives,
    "module": benchmark_ebc.available_sharding_types,
}

# Special ``--benchmark`` token; selects :func:`_run_all_benchmarks`.
RUN_ALL: str = "all"

# torchrun/torchelastic inject args like ``--local-rank`` into the worker argv;
# LOCAL_RANK is read from the environment instead, so these must not be forwarded
# as benchmark kwargs.
_TORCHRUN_INJECTED_KEYS: frozenset[str] = frozenset({"local_rank"})


def _coerce(value: str) -> Any:
    """Best-effort cast a CLI string to ``int``, then ``float``, else leave as ``str``."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_forwarded_kwargs(unknown: List[str]) -> Dict[str, Any]:
    """Parse leftover ``--key=value`` / ``--key value`` args into benchmark kwargs.

    Unrecognized args are benchmark-specific options (e.g. ``--batch_size``,
    ``--dim``) forwarded verbatim to the selected ``benchmark_runner``. Keys are
    normalized (leading ``--`` stripped, ``-`` -> ``_``) and values are coerced to
    ``int``/``float`` when possible. torchrun-injected keys (see
    ``_TORCHRUN_INJECTED_KEYS``) are dropped.
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        body = tok[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            i += 1
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            key, value = body, unknown[i + 1]
            i += 2
        else:
            key, value = body, "true"
            i += 1
        key = key.replace("-", "_")
        if key in _TORCHRUN_INJECTED_KEYS:
            continue
        kwargs[key] = _coerce(value)
    return kwargs


def _run_and_reclaim(
    runner: Callable[..., Any],
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    results: List[Any],
    **kwargs: Any,
) -> None:
    """Run one benchmark into ``results``, then free its memory for the next run."""
    try:
        results.append(runner(ctx, rank, world_size, **kwargs))
    except Exception:
        logger.exception(
            "rank=%d %s name=%s pipeline=%s failed",
            rank,
            runner.__module__,
            kwargs.get("name"),
            kwargs.get("pipeline"),
        )
        raise
    # Modules and buffers sit in reference cycles, so without an explicit collection
    # a finished run stays resident and inflates every later run's peak memory.
    gc.collect()
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def _run_all_benchmarks(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> List[Any]:
    """Per-rank runner for ``--benchmark=all``: every primitive, then every sharding
    type both unpipelined and pipelined.

    The remaining ``kwargs`` go to every run, so a shared name such as ``batch_size``
    (which means different things to each benchmark) applies to all of them.
    """
    kwargs.pop("pipeline", None)
    results: List[Any] = []

    for name in benchmark_primitive.available_primitives():
        _run_and_reclaim(
            benchmark_primitive.benchmark_runner,
            ctx,
            rank,
            world_size,
            results,
            name=name,
            **kwargs,
        )

    for name in benchmark_ebc.available_sharding_types():
        for pipeline in (False, True):
            _run_and_reclaim(
                benchmark_ebc.benchmark_runner,
                ctx,
                rank,
                world_size,
                results,
                name=name,
                pipeline=pipeline,
                **kwargs,
            )

    return results


def _resolve_runner(benchmark: str) -> Callable[..., Any]:
    """Return the per-rank runner for a ``--benchmark`` selection."""
    if benchmark == RUN_ALL:
        return _run_all_benchmarks
    return _BENCHMARKS[benchmark]


def _check_name(benchmark: str, name: Optional[str]) -> None:
    """Validate that ``name`` picks exactly one run of ``benchmark``."""
    if benchmark == RUN_ALL:
        if name is not None:
            raise ValueError(f"--name must be omitted with --benchmark={RUN_ALL}")
        return
    available = _AVAILABLE_NAMES[benchmark]()
    if name not in available:
        raise ValueError(
            f"--benchmark={benchmark} requires --name to be one of {available}, "
            f"got {name!r}"
        )


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared benchmark-selection and dispatch arguments on ``parser``.

    Exposed so internal wrappers can build their CLI on top of the SAME arguments
    this launcher uses -- then dispatch through :func:`run_benchmark` -- instead of
    re-declaring the arguments and drifting whenever the benchmark framework changes.
    """
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
        choices=sorted([*_BENCHMARKS, RUN_ALL]),
        required=True,
        help="which benchmark runner to launch, or 'all' to run every one of them "
        "back to back in this one job (sharing its hosts and process group).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "which benchmark of --benchmark to run. primitive: "
            + ", ".join(benchmark_primitive.available_primitives())
            + ". module: "
            + ", ".join(benchmark_ebc.available_sharding_types())
            + ". Required unless --benchmark=all, which runs every one of them."
        ),
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
    # Named after the ``profile_dir`` kwarg the benchmark runners consume, so the flag,
    # the forwarded kwarg and the runner argument are all one name. Accept both
    # spellings so callers using the hyphen (--profile-dir) and the underscore
    # (--profile_dir) convention both resolve to the same destination.
    parser.add_argument(
        "--profile-dir",
        "--profile_dir",
        dest="profile_dir",
        type=str,
        default="",
        help="directory to write structured results (torchrec_benchmark_*.json) "
        "and any traces/snapshots into. Empty (default) disables result dumping. "
        "Each rank writes its own per-rank file here.",
    )


def _parse_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser(description=__doc__)
    add_benchmark_args(parser)
    # Use parse_known_args so argparse does not fail (exit code 2) on the leftover
    # args: these are either torchrun/fb.dist.ddp injections (e.g. --local-rank,
    # dropped) or benchmark-specific options (e.g. --batch_size, --dim) that we
    # forward to the selected benchmark_runner.
    args, unknown = parser.parse_known_args()
    extra_kwargs = parse_forwarded_kwargs(unknown)
    if extra_kwargs:
        logger.info("forwarding benchmark kwargs: %s", extra_kwargs)
    return args, extra_kwargs


def run_benchmark(args: argparse.Namespace, extra_kwargs: Dict[str, Any]) -> None:
    """Dispatch to the selected benchmark runner for the requested mode.

    ``args`` must carry the attributes added by :func:`add_benchmark_args`
    (``mode`` / ``benchmark`` / ``name`` / ``world_size`` / ``backend`` /
    ``profile_dir``); ``extra_kwargs`` are benchmark-specific options forwarded
    verbatim to the runner.

    Raises:
        ValueError: if ``name`` does not pick exactly one run of ``benchmark`` (see
            :func:`_check_name`).
    """
    _check_name(args.benchmark, args.name)
    runner: Callable[..., Any] = _resolve_runner(args.benchmark)

    runner_kwargs: Dict[str, Any] = dict(extra_kwargs)
    runner_kwargs["profile_dir"] = args.profile_dir
    if args.name is not None:
        runner_kwargs["name"] = args.name

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
            **runner_kwargs,
        )
    else:  # remote
        logger.info(
            f"launching benchmark={args.benchmark} mode=remote backend={args.backend}"
        )
        run_single_process_func(
            runner,
            backend=args.backend,
            **runner_kwargs,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, extra_kwargs = _parse_args()
    run_benchmark(args, extra_kwargs)


if __name__ == "__main__":
    main()
