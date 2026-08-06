#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for primitive ops.

``benchmark_runner`` is the per-rank entry point. It is invoked once per rank by the process
runner (``process_runner.run_single_process_func`` /
``process_runner.run_local_multi_process_func``), which owns process group init +
handshake and injects a live ``SingleProcessContext`` (``ctx``) plus this rank's
``rank`` and ``world_size``. The runner must therefore use ``ctx.device`` /
``ctx.pg`` directly rather than creating its own context.

Multiple primitive benchmarks live in this file. ``benchmark_runner`` selects which one
to run via the ``primitive`` flag (see ``_BENCHMARKS``); each benchmark measures latency
only -- outputs are not checked for correctness. The first is ``kjt_a2a``, the All-to-All
performance of ``KJTAllToAll`` (the ``KeyedJaggedTensor`` A2A collective from
``dist_data.py``). The second is ``kt_a2a``, the All-to-All performance of
``PooledEmbeddingsAllToAll`` -- the dense pooled-embedding (``KeyedTensor``) collective
``output_dist`` uses to redistribute real embedding outputs. Unlike ``kjt_a2a`` it
exchanges float tensors rather than sparse indices. The third is ``reduce_scatter``, the
reduce-scatter performance of ``PooledEmbeddingsReduceScatter`` (from ``dist_data.py``) --
the collective ``output_dist`` uses instead of the A2A for row-wise / table-row-wise
sharding, summing each rank's partial pooled embeddings and scattering the batch dimension.

A follow-up launcher binary will call ``runner`` explicitly with options to run on
MAST or locally.
"""

import logging
import socket
from typing import Any, Callable, Dict, List, Optional

import torch
from torchrec.distributed.benchmark.base import benchmark_func, BenchmarkResult
from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
)
from torchrec.distributed.test_utils.process_runner import SingleProcessContext
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _build_splits(num_features: int, world_size: int) -> List[int]:
    """Distribute ``num_features`` keys across ``world_size`` ranks as evenly as possible.

    The remainder is spread over the first ranks, so the result sums to ``num_features``
    and has exactly ``world_size`` entries (the contract ``KJTAllToAll`` enforces).
    """
    base, rem = divmod(num_features, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def _make_input_kjt(
    num_features: int,
    num_values: int,
    batch_size: int,
    values_dtype: torch.dtype,
    device: torch.device,
) -> KeyedJaggedTensor:
    """Build a ``KeyedJaggedTensor`` whose ``values`` tensor is exactly ``num_values`` long.

    ``num_values`` is spread uniformly across the ``num_features * batch_size`` length
    slots (the leftover from integer division is added to the first slots) so the per-slot
    jagged lengths sum to ``num_values``. Content is random -- only the A2A transport size
    matters for this benchmark, not the values themselves.
    """
    keys: List[str] = [f"f_{i}" for i in range(num_features)]

    num_slots = num_features * batch_size
    per_slot, rem = divmod(num_values, num_slots)
    lengths = torch.full((num_slots,), per_slot, dtype=torch.int32, device=device)
    if rem > 0:
        lengths[:rem] += 1

    # Vocab bound is arbitrary -- A2A moves the raw bytes regardless of index values.
    values = torch.randint(
        0, 1_000_000, (num_values,), dtype=values_dtype, device=device
    )

    return KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)


def _run_a2a(
    _batch_inputs: List[Any],
    *,
    a2a: KJTAllToAll,
    kjt: KeyedJaggedTensor,
) -> None:
    """One measured iteration: full two-stage KJT A2A, then touch the output.

    Rank alignment against the straggler effect is handled by ``PerfWrapper`` (it
    barriers before each iteration, outside the timing window); this function only
    runs the collective. ``KJTAllToAll`` returns a two-stage awaitable -- the first
    ``wait()`` exchanges splits, the second exchanges the tensors. ``out.values()``
    only returns the values tensor handle: no data read, no host sync (and ``wait()``
    on CUDA does not block the host either -- the collective runs async on the stream).
    The input KJT is reused across iterations: ``_wait_impl`` does not free input storage
    (only the explicit ``clear_inputs()`` does), which we never call.

    So we ``torch.cuda.synchronize()`` at the end to actually block the host on the
    collective inside the measured region; that is what makes the wall-clock timer reflect
    end-to-end collective latency (GPU-event timing is unaffected either way).
    """
    out = a2a(kjt).wait().wait()
    values = out.values()
    if values.is_cuda:
        torch.cuda.synchronize(values.device)


def _benchmark_kjt_a2a(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> BenchmarkResult:
    """``KJTAllToAll`` (A2A) benchmark.

    Builds an input ``KeyedJaggedTensor`` of a configurable size, then measures the
    latency of running it through ``KJTAllToAll`` over ``ctx.pg``. Correctness of the
    redistributed output is intentionally not verified.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: benchmark options:
            num_features (int): total number of KJT keys across all ranks (split across
                ranks via ``KJTAllToAll`` splits). Default 256.
            num_values (int): total length of the ``values`` tensor -- the headline size
                knob (with the default, ``50M * 8B (int64) ~= 400 MB`` of transport per
                rank). Default 50_000_000.
            batch_size (int): stride; the lengths tensor has ``num_features * batch_size``
                entries. Default 32 * 1024.
            values_dtype (torch.dtype): dtype of the ``values`` tensor. Default int64.
            num_benchmarks (int): number of measured iterations. Default 20.
            num_profiles (int): number of profiled iterations (requires profile_dir).
                Default 0.
            profile_dir (str): directory for chrome traces; empty disables profiling.
            name (str): human-readable benchmark name. Default "kjt_a2a".

    Returns:
        This rank's ``BenchmarkResult``.
    """
    num_features: int = int(kwargs.get("num_features", 256))
    num_values: int = int(kwargs.get("num_values", 50_000_000))
    batch_size: int = int(kwargs.get("batch_size", 32 * 1024))
    values_dtype: torch.dtype = kwargs.get("values_dtype", torch.int64)
    num_benchmarks: int = int(kwargs.get("num_benchmarks", 20))
    num_profiles: int = int(kwargs.get("num_profiles", 0))
    profile_dir: str = str(kwargs.get("profile_dir", ""))
    name: str = str(kwargs.get("name", "kjt_a2a"))

    pg: Optional[torch.distributed.ProcessGroup] = ctx.pg
    assert pg is not None, "ctx.pg must be initialized by the process runner"

    splits = _build_splits(num_features, world_size)
    assert sum(splits) == num_features
    assert len(splits) == world_size
    if num_features < world_size:
        logger.warning(
            "num_features (%d) < world_size (%d): some ranks receive no features.",
            num_features,
            world_size,
        )

    kjt = _make_input_kjt(
        num_features=num_features,
        num_values=num_values,
        batch_size=batch_size,
        values_dtype=values_dtype,
        device=ctx.device,
    )
    a2a = KJTAllToAll(pg=pg, splits=splits)

    logger.info(
        "rank=%d local_rank=%d host=%s running KJT A2A benchmark: num_features=%d "
        "num_values=%d batch_size=%d splits=%s device=%s",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        num_features,
        num_values,
        batch_size,
        splits,
        ctx.device,
    )

    result = benchmark_func(
        name=name,
        rank=rank,
        world_size=world_size,
        func_to_benchmark=_run_a2a,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs={"a2a": a2a, "kjt": kjt},
        num_profiles=num_profiles,
        num_benchmarks=num_benchmarks,
        profile_dir=profile_dir,
        device_type=ctx.device.type,
        pg=pg,
    )

    if rank == 0:
        logger.info("KJT A2A benchmark result:\n%s", result)

    return result


def _make_pooled_embeddings(
    batch_size: int,
    dim: int,
    values_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build this rank's dense pooled-embedding input for ``PooledEmbeddingsAllToAll``.

    Returns a ``[batch_size, dim]`` tensor holding the *global* batch (every rank's rows)
    but only this rank's local embedding width. The A2A scatters the batch dimension and
    gathers the width, so each rank ends up with
    ``[batch_size // world_size, dim * world_size]``. Content is random -- only the A2A
    transport size matters for this benchmark, not the values themselves.
    """
    return torch.rand((batch_size, dim), dtype=values_dtype, device=device)


def _run_pooled_a2a(
    _batch_inputs: List[Any],
    *,
    a2a: PooledEmbeddingsAllToAll,
    local_embs: torch.Tensor,
) -> None:
    """One measured iteration: full pooled-embedding A2A, then touch the output.

    Rank alignment against the straggler effect is handled by ``PerfWrapper`` (it
    barriers before each iteration, outside the timing window); this function only
    runs the collective. ``PooledEmbeddingsAllToAll`` returns a single-stage awaitable
    (unlike the two-stage ``KJTAllToAll``) -- one ``wait()`` exchanges the dense tensors
    and yields the redistributed output. ``numel()`` only reads the output's shape
    metadata: no data read, no kernel, and -- like ``wait()`` on CUDA -- no host sync
    (the collective runs async on the stream). The input tensor is reused across
    iterations.

    So we ``torch.cuda.synchronize()`` at the end to actually block the host on the
    collective inside the measured region; that is what makes the wall-clock timer reflect
    end-to-end collective latency (GPU-event timing is unaffected either way).
    """
    out = a2a(local_embs).wait()
    out.numel()
    if local_embs.is_cuda:
        torch.cuda.synchronize(local_embs.device)


def _benchmark_kt_a2a(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> BenchmarkResult:
    """``PooledEmbeddingsAllToAll`` (dense pooled-embedding / ``KeyedTensor`` A2A) benchmark.

    Builds a dense pooled-embedding input tensor of a configurable size, then measures the
    latency of redistributing it through ``PooledEmbeddingsAllToAll`` over ``ctx.pg`` --
    the collective ``output_dist`` uses to exchange real embedding outputs (float tensors)
    rather than sparse indices. Correctness of the redistributed output is intentionally
    not verified.

    The embedding width is assumed balanced across ranks (``dim_sum_per_rank`` is
    ``[dim] * world_size``), matching balanced table-wise sharding.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: benchmark options:
            batch_size (int): global batch size (number of rows in the input) -- must be
                divisible by ``world_size`` (``PooledEmbeddingsAllToAll`` scatters the
                batch evenly). Default 32 * 1024.
            dim (int): per-rank embedding width; ``dim_sum_per_rank`` is
                ``[dim] * world_size``. The headline transport size is
                ``batch_size * dim`` (with the defaults, ``32768 * 3072 * 4B ~= 400 MB``
                of float32 per rank, comparable to ``kjt_a2a``). Default 3072.
            values_dtype (torch.dtype): dtype of the embedding tensor; must be a floating
                dtype. Default float32.
            num_benchmarks (int): number of measured iterations. Default 20.
            num_profiles (int): number of profiled iterations (requires profile_dir).
                Default 0.
            profile_dir (str): directory for chrome traces; empty disables profiling.
            name (str): human-readable benchmark name. Default "kt_a2a".

    Returns:
        This rank's ``BenchmarkResult``.
    """
    batch_size: int = int(kwargs.get("batch_size", 32 * 1024))
    dim: int = int(kwargs.get("dim", 3072))
    values_dtype: torch.dtype = kwargs.get("values_dtype", torch.float32)
    num_benchmarks: int = int(kwargs.get("num_benchmarks", 20))
    num_profiles: int = int(kwargs.get("num_profiles", 0))
    profile_dir: str = str(kwargs.get("profile_dir", ""))
    name: str = str(kwargs.get("name", "kt_a2a"))

    pg: Optional[torch.distributed.ProcessGroup] = ctx.pg
    assert pg is not None, "ctx.pg must be initialized by the process runner"
    assert batch_size % world_size == 0, (
        f"batch_size ({batch_size}) must be divisible by world_size ({world_size}): "
        "PooledEmbeddingsAllToAll scatters the global batch evenly across ranks."
    )

    dim_sum_per_rank = [dim] * world_size

    local_embs = _make_pooled_embeddings(
        batch_size=batch_size,
        dim=dim,
        values_dtype=values_dtype,
        device=ctx.device,
    )
    a2a = PooledEmbeddingsAllToAll(
        pg=pg,
        dim_sum_per_rank=dim_sum_per_rank,
        device=ctx.device,
    )

    logger.info(
        "rank=%d local_rank=%d host=%s running KT A2A benchmark: batch_size=%d "
        "dim=%d dim_sum_per_rank=%s device=%s",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        batch_size,
        dim,
        dim_sum_per_rank,
        ctx.device,
    )

    result = benchmark_func(
        name=name,
        rank=rank,
        world_size=world_size,
        func_to_benchmark=_run_pooled_a2a,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs={"a2a": a2a, "local_embs": local_embs},
        num_profiles=num_profiles,
        num_benchmarks=num_benchmarks,
        profile_dir=profile_dir,
        device_type=ctx.device.type,
        pg=pg,
    )

    if rank == 0:
        logger.info("KT A2A benchmark result:\n%s", result)

    return result


def _make_reduce_scatter_input(
    batch_size: int,
    dim: int,
    values_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build this rank's input for ``PooledEmbeddingsReduceScatter``.

    Returns a ``[batch_size, dim]`` tensor standing in for this rank's *partial* pooled
    embeddings over the full global batch. reduce-scatter sums these partials across ranks
    and scatters the batch dimension, so each rank ends up with
    ``[batch_size // world_size, dim]`` -- its slice of the reduced result. Content is
    random -- only the transport size matters for this benchmark, not the values.
    """
    return torch.rand((batch_size, dim), dtype=values_dtype, device=device)


def _run_reduce_scatter(
    _batch_inputs: List[Any],
    *,
    rs: PooledEmbeddingsReduceScatter,
    local_embs: torch.Tensor,
) -> None:
    """One measured iteration: full reduce-scatter, then touch the output.

    Rank alignment against the straggler effect is handled by ``PerfWrapper`` (it barriers
    before each iteration, outside the timing window); this function only runs the
    collective. ``PooledEmbeddingsReduceScatter`` returns a single-stage awaitable -- one
    ``wait()`` sums each rank's ``local_embs`` across the group and scatters the batch
    dimension, yielding this rank's ``[batch_size // world_size, dim]`` slice. ``numel()``
    only reads the output's shape metadata: no data read, no kernel, and -- like ``wait()``
    on CUDA -- no host sync (the collective runs async on the stream). The input tensor is
    reused across iterations.

    So we ``torch.cuda.synchronize()`` at the end to actually block the host on the
    collective inside the measured region; that is what makes the wall-clock timer reflect
    end-to-end collective latency (GPU-event timing is unaffected either way).
    """
    out = rs(local_embs).wait()
    out.numel()
    if local_embs.is_cuda:
        torch.cuda.synchronize(local_embs.device)


def _benchmark_reduce_scatter(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> BenchmarkResult:
    """``PooledEmbeddingsReduceScatter`` (dense pooled-embedding reduce-scatter) benchmark.

    Builds a dense pooled-embedding input tensor of a configurable size, then measures the
    latency of reducing-and-scattering it through ``PooledEmbeddingsReduceScatter`` (the
    ``dist_data.py`` module, the same one row-wise / table-row-wise / grid sharding
    instantiate for ``output_dist``) over ``ctx.pg`` -- each rank holds partial pooled sums
    for the global batch that must be summed across ranks and scattered back to each rank's
    local batch slice. Correctness of the reduced output is intentionally not verified.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: benchmark options:
            batch_size (int): global batch size (rows of the input) -- must be divisible by
                ``world_size`` (reduce-scatter splits the batch evenly across ranks).
                Default 32 * 1024.
            dim (int): embedding width. The headline transport size is ``batch_size * dim``
                (with the defaults, ``32768 * 3072 * 4B ~= 400 MB`` of float32 per rank,
                comparable to ``kt_a2a``). Default 3072.
            values_dtype (torch.dtype): dtype of the embedding tensor; must be a floating
                dtype. Default float32.
            num_benchmarks (int): number of measured iterations. Default 20.
            num_profiles (int): number of profiled iterations (requires profile_dir).
                Default 0.
            profile_dir (str): directory for chrome traces; empty disables profiling.
            name (str): human-readable benchmark name. Default "reduce_scatter".

    Returns:
        This rank's ``BenchmarkResult``.
    """
    batch_size: int = int(kwargs.get("batch_size", 32 * 1024))
    dim: int = int(kwargs.get("dim", 3072))
    values_dtype: torch.dtype = kwargs.get("values_dtype", torch.float32)
    num_benchmarks: int = int(kwargs.get("num_benchmarks", 20))
    num_profiles: int = int(kwargs.get("num_profiles", 0))
    profile_dir: str = str(kwargs.get("profile_dir", ""))
    name: str = str(kwargs.get("name", "reduce_scatter"))

    pg: Optional[torch.distributed.ProcessGroup] = ctx.pg
    assert pg is not None, "ctx.pg must be initialized by the process runner"
    assert batch_size % world_size == 0, (
        f"batch_size ({batch_size}) must be divisible by world_size ({world_size}): "
        "PooledEmbeddingsReduceScatter scatters the global batch evenly across ranks."
    )

    local_embs = _make_reduce_scatter_input(
        batch_size=batch_size,
        dim=dim,
        values_dtype=values_dtype,
        device=ctx.device,
    )
    rs = PooledEmbeddingsReduceScatter(pg)

    logger.info(
        "rank=%d local_rank=%d host=%s running reduce-scatter benchmark: batch_size=%d "
        "dim=%d device=%s",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        batch_size,
        dim,
        ctx.device,
    )

    result = benchmark_func(
        name=name,
        rank=rank,
        world_size=world_size,
        func_to_benchmark=_run_reduce_scatter,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs={"rs": rs, "local_embs": local_embs},
        num_profiles=num_profiles,
        num_benchmarks=num_benchmarks,
        profile_dir=profile_dir,
        device_type=ctx.device.type,
        pg=pg,
    )

    if rank == 0:
        logger.info("reduce-scatter benchmark result:\n%s", result)

    return result


# Registry of available primitive benchmarks, keyed by the ``primitive`` flag.
# Add new primitive benchmarks here -- each is called as
# ``fn(ctx, rank, world_size, **kwargs)`` and returns a per-rank ``BenchmarkResult``.
_BENCHMARKS: Dict[str, Callable[..., BenchmarkResult]] = {
    "kjt_a2a": _benchmark_kjt_a2a,
    "kt_a2a": _benchmark_kt_a2a,
    "reduce_scatter": _benchmark_reduce_scatter,
}


def available_primitives() -> List[str]:
    """Return the sorted names of registered primitive benchmarks.

    These are the valid values for the ``name`` flag consumed by
    :func:`benchmark_runner` (and surfaced as the launcher's ``--name`` choices).
    """
    return sorted(_BENCHMARKS)


def benchmark_runner(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> BenchmarkResult:
    """Per-rank primitive benchmark entry point.

    Selects which primitive benchmark to run via the ``name`` flag and dispatches to
    it. The injected ``ctx`` (device + process group), ``rank`` and ``world_size`` plus
    the remaining ``kwargs`` are forwarded to the selected benchmark.

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: ``name`` (str) selects the benchmark (default ``"kjt_a2a"``); the
            rest are forwarded to the selected benchmark (see its docstring).

    Returns:
        This rank's ``BenchmarkResult``.
    """
    name: str = str(kwargs.pop("name", "kjt_a2a"))
    if name not in _BENCHMARKS:
        raise ValueError(
            f"unknown primitive benchmark {name!r}; "
            f"available: {sorted(_BENCHMARKS)}"
        )

    logger.info(
        "rank=%d local_rank=%d host=%s selected primitive benchmark %r",
        rank,
        ctx.local_rank,
        socket.gethostname(),
        name,
    )
    return _BENCHMARKS[name](ctx, rank, world_size, **kwargs)
