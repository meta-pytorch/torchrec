#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Per-rank process runner for torchrec distributed benchmarks/tests.

This module provides two layers that share the same per-rank entry path:

1. Single-process (one-rank-per-process) entry point --
   :func:`run_single_process_func` / :class:`SingleProcessContext`. Assumes the
   current process **is already** a single rank that torchrun/torchelastic placed
   (possibly on a remote host), with the standard rendezvous env vars populated.
   It validates those env vars, binds this process to its local GPU
   (``cuda:LOCAL_RANK`` -- the global rank is NOT a valid device index on
   multi-host jobs), initializes the default process group via ``env://``
   rendezvous and barriers (handshake) so every rank has joined before any work
   runs, runs the provided runner for this rank, then tears the process group(s)
   down.

2. Local (single-host) launcher -- :func:`run_local_multi_process_func`. Unlike
   ``multi_process.run_multi_process_func`` (which spawns a local pool and lets
   each worker build its own ``MultiProcessContext``), this launcher synthesizes
   the torchrun rendezvous env vars for localhost and hands each spawned worker to
   :func:`run_single_process_func`. That makes the local run use the SAME per-rank
   entry path as a torchrun/MAST job; the only difference is that here the
   rendezvous env is synthesized for localhost instead of provided by torchrun.

In both cases the runner must NOT create its own context -- it should consume the
injected ``ctx.device`` / ``ctx.pg`` directly.
"""

import logging
import multiprocessing
import os
import traceback
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.comm import _CROSS_PG, _INTRA_PG
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


# Env vars torchrun / torchelastic populate in every worker. RANK / WORLD_SIZE /
# LOCAL_RANK / LOCAL_WORLD_SIZE describe this process' place in the (possibly
# multi-host) topology; MASTER_ADDR / MASTER_PORT are the env:// rendezvous
# endpoint used by ``init_process_group``. All are required to bind the local
# device and handshake the process group.
REQUIRED_ENV_VARS: Tuple[str, ...] = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


def check_required_env_vars() -> None:
    """Validate that every torchrun rendezvous env var is set.

    Raises:
        RuntimeError: if any of :data:`REQUIRED_ENV_VARS` is missing, listing the
            missing names. A missing var means this process was not launched by
            torchrun/torchelastic, so the single-process path cannot run -- the
            caller should use the local spawn path
            (:func:`run_local_multi_process_func`) instead.
    """
    missing = [name for name in REQUIRED_ENV_VARS if name not in os.environ]
    if missing:
        raise RuntimeError(
            "single-process entry point requires torchrun/torchelastic rendezvous "
            f"env vars, but the following are missing: {missing}. "
            "Launch this process via torch.distributed.run (e.g. torchx on MAST), "
            "or use run_local_multi_process_func for the local spawn path."
        )


class SingleProcessContext:
    """Context manager that owns this rank's device + default process group.

    On ``__enter__`` it (re)initializes the default process group using ``env://``
    rendezvous and barriers so every rank has handshaked before the body runs. On
    ``__exit__`` it destroys the torchrec intra/cross groups (if created during
    the run) and the default group.

    Attributes:
        rank: global rank of this process (from ``RANK``).
        world_size: global world size (from ``WORLD_SIZE``).
        local_rank: local rank on this host (from ``LOCAL_RANK``); selects the GPU.
        device: the bound ``torch.device`` (``cuda:LOCAL_RANK`` or ``cpu``).
        pg: the default process group, available after ``__enter__``.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        disable_cuda_tf_32: bool = True,
    ) -> None:
        self.rank: int = int(os.environ["RANK"])
        self.world_size: int = int(os.environ["WORLD_SIZE"])
        self.local_rank: int = int(os.environ["LOCAL_RANK"])
        self.disable_cuda_tf_32 = disable_cuda_tf_32
        # Default to nccl on GPU hosts, gloo otherwise.
        self.backend: str = backend or ("nccl" if torch.cuda.is_available() else "gloo")

        if torch.cuda.is_available():
            # Bind to the LOCAL device index, not the global rank: on a multi-host
            # job rank 10 on an 8-GPU host would map to a nonexistent cuda:10.
            self.device: torch.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            if self.disable_cuda_tf_32:
                # Capture the prior TF32 flags so __exit__ restores exactly what
                # was set before, rather than assuming both were enabled.
                self._prior_cudnn_allow_tf32: bool = torch.backends.cudnn.allow_tf32
                self._prior_matmul_allow_tf32: bool = (
                    torch.backends.cuda.matmul.allow_tf32
                )
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
        else:
            self.device: torch.device = torch.device("cpu")

        self.pg: Optional[dist.ProcessGroup] = None

    def __enter__(self) -> "SingleProcessContext":
        # torchrun already exported RANK / WORLD_SIZE / LOCAL_RANK /
        # MASTER_ADDR / MASTER_PORT, so env:// rendezvous needs no overrides.
        # Start from a clean slate in case a prior group was left initialized.
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(backend=self.backend)
        self.pg = dist.group.WORLD

        # Handshake: make sure every rank has joined the group before the runner
        # issues any collectives. init_process_group already rendezvouses, but an
        # explicit barrier guarantees all ranks are past setup and surfaces a
        # rendezvous/topology mismatch here rather than deep inside the runner.
        # device_ids is only meaningful for nccl; a gloo-only backend must not
        # receive it.
        if torch.cuda.is_available() and "nccl" in self.backend:
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()
        logger.info(
            f"single_process handshake complete: rank={self.rank} "
            f"world_size={self.world_size} local_rank={self.local_rank} "
            f"device={self.device} backend={self.backend}"
        )
        return self

    # pyre-ignore[2]
    def __exit__(self, exc_type, exc_instance, traceback) -> None:
        if _INTRA_PG is not None:
            dist.destroy_process_group(_INTRA_PG)
        if _CROSS_PG is not None:
            dist.destroy_process_group(_CROSS_PG)
        if self.pg is not None:
            dist.destroy_process_group(self.pg)
        if torch.cuda.is_available() and self.disable_cuda_tf_32:
            torch.backends.cudnn.allow_tf32 = self._prior_cudnn_allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = self._prior_matmul_allow_tf32


def run_single_process_func(
    func: Callable[..., Any],
    *,
    backend: Optional[str] = None,
    use_deterministic_algorithms: bool = False,
    disable_cuda_tf_32: bool = True,
    **kwargs: Any,
) -> Any:
    """Entry point: handshake this rank, then run ``func`` for it.

    Validates the torchrun env vars, applies the standard determinism settings,
    binds the local GPU, initializes + barriers the default process group, and
    then invokes ``func`` with the live :class:`SingleProcessContext` injected as
    ``ctx`` (plus this rank's ``rank`` and ``world_size``) into ``kwargs``. The
    process group is already initialized + handshaked when ``func`` runs, so
    ``func`` must NOT create its own context -- it should use ``ctx.device`` /
    ``ctx.pg`` directly. The context (and its device + group) is owned and torn
    down here.

    Args:
        func: the per-rank runner. Called as ``func(ctx=..., rank=...,
            world_size=..., **kwargs)``; its return value is returned verbatim.
        backend: process-group backend; defaults to ``nccl`` on GPU, ``gloo``
            otherwise.
        use_deterministic_algorithms: toggle ``torch.use_deterministic_algorithms``.
            Defaults to ``False`` -- deterministic kernels (e.g. embedding/scatter
            backward) are significantly slower, so benchmarks must keep this off to
            measure representative QPS.
        disable_cuda_tf_32: disable TF32 matmul/cudnn during the run.
        **kwargs: forwarded to ``func``.

    Returns:
        Whatever ``func`` returns (this rank's result only).
    """
    check_required_env_vars()

    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    if torch.cuda.is_available() and use_deterministic_algorithms:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        with SingleProcessContext(
            backend=backend,
            disable_cuda_tf_32=disable_cuda_tf_32,
        ) as ctx:
            kwargs["ctx"] = ctx
            kwargs["rank"] = ctx.rank
            kwargs["world_size"] = ctx.world_size
            return func(**kwargs)
    finally:
        torch.use_deterministic_algorithms(False)


def _set_local_rendezvous_env(rank: int, world_size: int) -> None:
    """Export this worker's torchrun-style rendezvous env vars.

    On a single host the local rank/size equal the global rank/size, so the same
    env contract used by torchrun (and consumed by ``run_single_process_func``)
    can be synthesized here. ``MASTER_ADDR`` / ``MASTER_PORT`` are set once by the
    parent and inherited by the spawned children.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)


# pyre-ignore[2]
def _worker_entry(args) -> Any:
    """Worker entry: set this rank's rendezvous env, then run the single-process
    path. Defined at module scope so it is picklable by ``spawn``.

    ``multiprocessing.Pool`` pickles a worker's exception (with its traceback) to
    ship it back to the parent. If the exception value or any frame locals in the
    traceback reference unpicklable objects (e.g. modules held alive by
    ``torch._dynamo`` / ``torch.compile`` errors), pickling fails with
    ``MaybeEncodingError`` and the original failure is lost. Convert any exception
    to a plain ``RuntimeError`` carrying a stringified traceback so the real
    failure always reaches the parent.
    """
    func, rank, world_size, backend, use_deterministic_algorithms, kwargs = args
    _set_local_rendezvous_env(rank=rank, world_size=world_size)
    try:
        return run_single_process_func(
            func,
            backend=backend,
            use_deterministic_algorithms=use_deterministic_algorithms,
            **kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            f"Worker raised {type(e).__name__}: {e}\n{traceback.format_exc()}"
        ) from None


def run_local_multi_process_func(
    func: Callable[..., Any],
    *,
    world_size: int = 2,
    backend: Optional[str] = None,
    use_deterministic_algorithms: bool = False,
    multiprocessing_method: str = "spawn",
    **kwargs: Any,
) -> List[Any]:
    """Spawn ``world_size`` local workers, each routed through the single-process
    (torchrun) entry point.

    Sets the localhost rendezvous endpoint once in the parent (inherited by the
    children), then spawns one worker per rank. Each worker synthesizes its
    per-rank rendezvous env and calls :func:`run_single_process_func`, which
    initializes + handshakes the process group and injects the live
    ``SingleProcessContext`` (``ctx``), ``rank`` and ``world_size`` into ``func``.

    Args:
        func: the per-rank runner. Invoked as ``func(ctx=..., rank=...,
            world_size=..., **kwargs)``; its return value is collected per rank.
        world_size: number of local worker processes (ranks) to spawn.
        backend: process-group backend forwarded to ``run_single_process_func``
            (defaults to ``nccl`` on GPU, ``gloo`` otherwise).
        use_deterministic_algorithms: forwarded to ``run_single_process_func``.
            Defaults to ``False`` so benchmarks measure representative QPS.
        multiprocessing_method: ``multiprocessing`` start method (``"spawn"``).
        **kwargs: forwarded verbatim to ``func``.

    Returns:
        A list of per-rank results, indexed by rank.
    """
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    if world_size == 1:
        # Single-rank job: no pool needed, run this process as the sole rank.
        _set_local_rendezvous_env(rank=0, world_size=1)
        return [
            run_single_process_func(
                func,
                backend=backend,
                use_deterministic_algorithms=use_deterministic_algorithms,
                **kwargs,
            )
        ]

    ctx = multiprocessing.get_context(multiprocessing_method)

    # One arg tuple per rank; kwargs are copied so workers never share state.
    args_list = [
        (func, rank, world_size, backend, use_deterministic_algorithms, kwargs.copy())
        for rank in range(world_size)
    ]

    with ctx.Pool(processes=world_size) as pool:
        results = pool.map(_worker_entry, args_list)

    return results
