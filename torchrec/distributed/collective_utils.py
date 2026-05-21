#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This file contains utilities for constructing collective based control flows.
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, cast, List, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist

_USE_COLLECTIVE_VALIDATION: bool = False
_INITIALIZED: bool = False

logger: logging.Logger = logging.getLogger(__name__)


def is_leader(pg: Optional[dist.ProcessGroup], leader_rank: int = 0) -> bool:
    """
    Checks if the current processs is the leader.

    Args:
        pg (Optional[dist.ProcessGroup]): the process's rank within the pg is used to
            determine if the process is the leader. pg being None implies that the
            process is the only member in the group (e.g. a single process program).
        leader_rank (int): the definition of leader (defaults to 0). The caller can
            override it with a context-specific definition.
    """
    if pg is None:
        return leader_rank == 0
    return pg.rank() == leader_rank


T = TypeVar("T")


def invoke_on_rank_and_broadcast_result(
    pg: dist.ProcessGroup,
    rank: int,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Invokes a function on the designated rank and broadcasts the result to all
    members within the group.

    Example::

        id = invoke_on_rank_and_broadcast_result(pg, 0, allocate_id)
    """
    if pg.rank() == rank:
        res = func(*args, **kwargs)
        object_list = [res]
    else:
        object_list = [None]
    if pg.size() > 1:
        dist.broadcast_object_list(object_list, rank, group=pg)
    return cast(T, object_list[0])


def run_on_leader(pg: dist.ProcessGroup, rank: int):
    def callable(func: Callable[..., T]) -> T:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return invoke_on_rank_and_broadcast_result(pg, rank, func, *args, **kwargs)

        # pyrefly: ignore[bad-return]
        return wrapped

    return callable


def create_on_rank_and_share_result(
    pg: dist.ProcessGroup,
    rank: int,
    creator: Callable[..., T],
    extractor: Callable[[T], List[Optional[torch.Tensor]]],
    constructor: Callable[[List[Optional[torch.Tensor]]], T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Invokes ``creator`` on the designated rank, moves the resulting CPU tensors
    into POSIX shared memory (``/dev/shm``), and reconstructs the result on all
    other ranks by mapping the same shared memory regions — no data is copied.

    This is safe to call repeatedly (e.g. inside a benchmark loop) because
    ``broadcast_object_list`` is a collective that synchronises all ranks on
    every call, unlike ``store.set`` / ``store.get`` which can return stale
    values after the first iteration.

    Note:
        Shared memory (``/dev/shm``) is host-local, so ``pg`` should be an
        **intra-node** process group — one whose members all reside on the same
        host. Use ``torchrec.distributed.comm.intra_and_cross_node_pg()`` to
        obtain such a group. For multi-host jobs, each host independently runs
        one creator (e.g. local rank 0) while the remaining ranks on that host
        map the same shared memory regions.

    Args:
        pg: The process group whose members participate. Should be an
            intra-node (host-local) group so that all members can access the
            same ``/dev/shm`` regions.
        rank: The rank within ``pg`` that runs ``creator``.
        creator: ``creator(*args, **kwargs) -> T`` — called only on ``rank``.
        extractor: ``extractor(result) -> List[Optional[Tensor]]`` — pulls the
            tensors out of the creator's return value so they can be shared.
            ``None`` entries are preserved as-is (not shared).
        constructor: ``constructor(List[Optional[Tensor]]) -> T`` — rebuilds
            ``T`` from shared-memory tensors (and ``None`` placeholders) on
            every non-creator rank.
        *args, **kwargs: Forwarded to ``creator``.

    Returns:
        On the creator rank the original ``T``; on every other rank a ``T``
        whose tensors point to the same physical shared memory.

    Example::

        from torchrec.distributed.comm import intra_and_cross_node_pg

        intra_pg, _cross_pg = intra_and_cross_node_pg(device, backend="gloo")

        # Each host runs the creator on local rank 0; other ranks on the
        # same host map the shared memory — no cross-host coordination.
        tensor = create_on_rank_and_share_result(
            intra_pg, 0,
            creator=lambda: torch.full((1024, 1024), 42.0),
            extractor=lambda t: [t],
            constructor=lambda ts: ts[0],
        )
    """
    _ShmMeta = Tuple[bytes, bytes, int, torch.Size, Tuple[int, ...], int, torch.dtype]

    res: Optional[T] = None
    if pg.rank() == rank:
        res = creator(*args, **kwargs)
        tensors = extractor(res)
        metadata: List[Optional[_ShmMeta]] = []
        for t in tensors:
            if t is None:
                metadata.append(None)
                continue
            assert t.device.type == "cpu", (
                f"create_on_rank_and_share_result only supports CPU tensors, "
                f"got tensor on {t.device}"
            )
            # Call _share_filename_cpu_() directly — it moves data to
            # filename-based shared memory and returns the metadata in one
            # step. Calling share_memory_() first would add an extra copy
            # (fd-based shm → filename-based shm).
            manager_handle, storage_handle, size = (
                t.untyped_storage()._share_filename_cpu_()
            )
            metadata.append(
                (
                    bytes(manager_handle),
                    bytes(storage_handle),
                    int(size),
                    t.shape,
                    tuple(t.stride()),
                    int(t.storage_offset()),
                    t.dtype,
                )
            )
        object_list: List[Optional[List[Optional[_ShmMeta]]]] = [metadata]
    else:
        object_list = [None]

    if pg.size() > 1:
        dist.broadcast_object_list(object_list, rank, group=pg)

    if res is not None:
        return res

    metadata = cast(List[Optional[_ShmMeta]], object_list[0])
    shared_tensors: List[Optional[torch.Tensor]] = []
    for entry in metadata:
        if entry is None:
            shared_tensors.append(None)
            continue
        (
            manager_handle,
            storage_handle,
            size,
            shape,
            stride,
            storage_offset,
            dtype,
        ) = entry
        shared_storage = torch.UntypedStorage._new_shared_filename_cpu(
            manager_handle, storage_handle, size
        )
        tensor = torch.empty([], dtype=dtype).set_(
            shared_storage, storage_offset, shape, stride
        )
        shared_tensors.append(tensor)

    return constructor(shared_tensors)


def _resolve_enablement_on_leader() -> bool:
    """Determine if collective validation is enabled on rank 0.

    Fail-closed on JK errors so a flaky JK service can't hang peers waiting in
    broadcast_object_list.
    """
    enable_via_env = os.environ.get("TORCHREC_VALIDATE_COLLECTIVES", "")
    if enable_via_env == "0":
        return False
    elif enable_via_env == "1":
        return True
    try:
        return torch._utils_internal.justknobs_check(
            "pytorch/torchrec:enable_collective_validation", default=False
        )
    except Exception:
        logger.exception(
            "Failed to check JK for collective validation; defaulting to disabled"
        )
        return False


def init_collective_validation(pg: dist.ProcessGroup) -> None:
    """Broadcast collective validation flag from rank 0 to all ranks in pg.

    Call once on the world PG during DistributedModelParallel.__init__
    so all ranks participate. Uses both JK and env var to determine if validation
    is enabled.
    """
    global _USE_COLLECTIVE_VALIDATION, _INITIALIZED
    if _INITIALIZED:
        return  # prevents double-DMP / JK initialization
    _USE_COLLECTIVE_VALIDATION = invoke_on_rank_and_broadcast_result(
        pg=pg,
        func=_resolve_enablement_on_leader,
        rank=0,
    )
    _INITIALIZED = True
    logger.info(f"Collective validation initialized: {_USE_COLLECTIVE_VALIDATION}")


def validate_collectives_enabled() -> bool:
    return _USE_COLLECTIVE_VALIDATION
