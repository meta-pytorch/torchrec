#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# NOTE: Do NOT add `from __future__ import annotations` here.
# This module is loaded inside a torch.package at model-publish time. Combining
# PEP 563 (string annotations) with @dataclass on Python 3.12 hits
# https://github.com/python/cpython/issues/115258 — `dataclass._is_type` does
# `sys.modules.get(cls.__module__).__dict__`, which returns None for
# torch.package-synthetic module names ("<torch_package_N>.…") and crashes with
# AttributeError. Keeping annotations as runtime objects avoids that path.

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from torchrec.distributed.comm import get_local_size

logger: logging.Logger = logging.getLogger(__name__)

# NOTE: Do NOT add static `import torchcomms` / `from torchcomms import ...` here
# (or `from caffe2.torch.distributed.fb.sharded_relay_process_group import ...`).
# This module is reachable from `torchrec.distributed.model_parallel`, which the
# Module Factory packager pulls into a torch.package. torch.package's static
# dependency analyzer follows ALL `import` / `from ... import` statements
# (even when wrapped in try/except), and `torchcomms._comms` /
# `torchcomms._comms_rcclx` are compiled C-extensions with no `__file__`,
# which causes packaging to fail with "Module had no __file__ defined".
# Routing the imports through `importlib.import_module(<string>)` keeps these
# names out of the AST, so the static analyzer never sees them.


def _try_dynamic_import(module_path: str, attr: str) -> Any:
    """Dynamic import that is invisible to torch.package's static analyzer."""
    try:
        return getattr(importlib.import_module(module_path), attr, None)
    except ImportError:
        return None


def _get_fused_sharded_relay_multi_group_cls() -> Any:
    return _try_dynamic_import(
        "caffe2.torch.distributed.fb.sharded_relay_process_group",
        "FusedShardedRelayMultiGroup",
    )


def _get_torchcomms_new_comm() -> Any:
    return _try_dynamic_import("torchcomms", "new_comm")


@dataclass
class ShardedRelayState:
    """
    Runtime state for fused sharded relay multi-group allreduce.

    Holds all configuration and communicator handles needed to perform
    phase-synchronized sharded relay allreduce calls. Created once during
    setup and reused across allreduce calls.

    A single RCCLX communicator is shared across all sparse groups. The
    phase-synchronized batched API handles all groups in one call.

    Flat-concat allreduce design
    ----------------------------
    Instead of making N serial allreduce_multi_group calls (one per embedding
    table), all tensors for the active group are packed into a single flat
    buffer, ONE fused call is made for all 4 groups simultaneously, and results
    are unpacked back into the original tensors. This matches the intended
    usage of the C++ ncclShardedRelayMultiGroupAllReduceImpl kernel.

    Buffer aliasing for helper groups
    ---------------------------------
    With phase-synchronized execution, all groups are processed simultaneously,
    so each helper group MUST have its own buffer (no aliasing across groups).
    Helper buffers are sized to nActiveRanks × chunkSize (two-slot passthrough),
    which is much smaller than the full per-group total.  Across the 3 helper
    groups per rank (in a 4-group topology), the total helper memory is
    6 × chunkSize per rank.

    Caches
    ------
    _active_flat_cache : per-dtype grow-only flat buffer for the active group.
        Used as the pack/allreduce/unpack buffer. Keyed by dtype so weights
        (bf16) and optimizer states (fp32) each have their own buffer.

    _helper_flat_cache : per-(group_idx, dtype) grow-only flat scratch buffer
        for helper groups. Sized to the passthrough minimum
        (nActiveRanks × chunkSize) for each group. Each helper group has its
        own buffer because all groups are processed simultaneously under
        phase-sync.

    _flat_metadata_cache : per-(annotation+dtype) cached allgather results.
        Stores per_group_total_counts (total elements per group). Populated
        on the first call and reused forever — embedding table dimensions are
        fixed for the entire training run, so the allgather never needs to
        repeat.
    """

    # Single FusedShardedRelayMultiGroup for all sparse groups at once.
    fused: Any
    intra_node_pytorch_pg: dist.ProcessGroup | None
    local_rank: int
    sparse_group_size: int
    my_sparse_group: int
    num_sparse_groups: int
    local_size: int
    precomputed_active_ranks: list[list[int]]
    # Single RCCLX comm — held for cleanup via finalize().
    _rcclx_comm: Any = field(default=None)
    # Grow-only flat buffer for the active group: dtype → tensor.
    # Packed before allreduce, unpacked after.
    _active_flat_cache: dict[torch.dtype, torch.Tensor] = field(default_factory=dict)
    # Grow-only flat scratch buffer for helper groups: (group_idx, dtype) → tensor.
    # Each helper group has its own buffer (no aliasing) because phase-sync
    # processes all groups simultaneously.
    _helper_flat_cache: dict[tuple[int, torch.dtype], torch.Tensor] = field(
        default_factory=dict
    )
    # Cached allgather metadata: (annotation + str(dtype)) → per_group_total_counts.
    # Never invalidated — embedding table sizes are fixed throughout training.
    _flat_metadata_cache: dict[str, list[int]] = field(default_factory=dict)


def _validate_sharded_relay_preconditions(
    use_inter_host_allreduce: bool,
    sharding_group_size: int,
    local_size: int,
) -> bool:
    """Return True if all preconditions for sharded relay are met.

    Logs a warning and returns False otherwise. This factoring keeps
    setup_sharded_relay below the C901 complexity threshold by moving the
    chain of guard returns into a single linear validation step.
    """
    if use_inter_host_allreduce:
        logger.warning(
            "[TorchRec 2D Parallel] Sharded relay is NOT supported with "
            "use_inter_host_allreduce=True (replica_pg spans multiple nodes). "
            "Disabling sharded relay mode."
        )
        return False

    # The RCCLX C++ kernel (buildShardedRelayRankConfig) requires exactly 2
    # active ranks per group.  Any other sharding_group_size is unsupported.
    if sharding_group_size != 2:
        logger.warning(
            f"[TorchRec 2D Parallel] Sharded relay requires sharding_group_size=2, "
            f"but got sharding_group_size={sharding_group_size}. "
            "Disabling sharded relay mode."
        )
        return False

    if local_size < 4:
        logger.warning(
            f"[TorchRec 2D Parallel] Sharded relay requires at least 4 GPUs "
            f"per node, but local_size={local_size}. "
            "Disabling sharded relay mode."
        )
        return False

    if local_size // sharding_group_size == 0:
        logger.warning(
            f"[TorchRec 2D Parallel] Invalid configuration: "
            f"num_sparse_groups=0 (local_size={local_size}, "
            f"sparse_group_size={sharding_group_size}). "
            "Disabling sharded relay mode."
        )
        return False

    return True


def _create_intra_node_rcclx_comm(
    global_rank: int,
    local_rank: int,
    local_size: int,
    my_node_idx: int,
) -> Any | None:
    """Create the shared 8-rank intra-node RCCLX comm.

    Returns the comm object or None on failure. Wraps the env-var override
    needed to make ``new_comm`` create an intra-node (not world-size) comm.
    """
    import os

    torchcomms_new_comm = _get_torchcomms_new_comm()
    if torchcomms_new_comm is None:
        logger.warning(
            "[TorchRec 2D Parallel] Intra-node RCCLX comm not available. "
            "Disabling sharded relay mode."
        )
        return None

    global_store = dist.distributed_c10d._get_default_store()
    if global_store is None:
        logger.warning(
            "[TorchRec 2D Parallel] No default store available for RCCLX "
            "comm creation. Disabling sharded relay mode."
        )
        return None

    device = torch.device(f"cuda:{local_rank}")
    comm_name_base = f"sharded_relay_node{my_node_idx}"

    orig_tc_rank = os.environ.get("TORCHCOMM_RANK")
    orig_tc_size = os.environ.get("TORCHCOMM_SIZE")
    try:
        # Override rank/size so new_comm creates an 8-rank comm (not 64-rank).
        os.environ["TORCHCOMM_RANK"] = str(local_rank)
        os.environ["TORCHCOMM_SIZE"] = str(local_size)
        group_store = dist.PrefixStore(f"rcclx_intra_{my_node_idx}", global_store)
        rcclx_comm = torchcomms_new_comm(
            backend="rcclx",
            device=device,
            name=comm_name_base,
            store=group_store,
        )
    finally:
        if orig_tc_rank is None:
            os.environ.pop("TORCHCOMM_RANK", None)
        else:
            os.environ["TORCHCOMM_RANK"] = orig_tc_rank
        if orig_tc_size is None:
            os.environ.pop("TORCHCOMM_SIZE", None)
        else:
            os.environ["TORCHCOMM_SIZE"] = orig_tc_size

    if rcclx_comm is None:
        logger.warning(
            "[TorchRec 2D Parallel] new_comm() returned None for "
            "intra-node RCCLX comm. "
            "Disabling sharded relay mode."
        )
        return None

    logger.info(
        f"[TorchRec 2D Parallel] Intra-node RCCLX comm created "
        f"(single shared comm for all groups): "
        f"global_rank={global_rank}, "
        f"node_idx={my_node_idx}, "
        f"local_rank_in_comm={rcclx_comm.get_rank()}, "
        f"comm_size={rcclx_comm.get_size()}"
    )
    return rcclx_comm


def _create_intra_node_pytorch_pg(
    global_rank: int,
    local_size: int,
    num_nodes: int,
    my_node_idx: int,
) -> dist.ProcessGroup | None:
    """Create intra-node PyTorch ProcessGroups for allgather metadata.

    IMPORTANT: ``dist.new_group()`` is COLLECTIVE — every rank must call it
    for every node's group, even if only one node ends up using its own.
    Returns this rank's intra-node PG, or None if creation failed.
    """
    intra_node_pytorch_pg: dist.ProcessGroup | None = None
    try:
        for node_idx in range(num_nodes):
            node_ranks = list(range(node_idx * local_size, (node_idx + 1) * local_size))
            pg = dist.new_group(ranks=node_ranks)
            if node_idx == my_node_idx:
                intra_node_pytorch_pg = pg
        if intra_node_pytorch_pg is not None:
            logger.info(
                f"[TorchRec 2D Parallel] Created intra-node ProcessGroup: "
                f"global_rank={global_rank}, node_idx={my_node_idx}, "
                f"pg_size={local_size}"
            )
    except Exception as e:
        logger.warning(
            f"[TorchRec 2D Parallel] Failed to create intra-node ProcessGroup: {e}. "
            "Will fall back to local tensor sizes only."
        )
        return None
    return intra_node_pytorch_pg


def setup_sharded_relay(
    global_rank: int,
    world_size: int,
    use_inter_host_allreduce: bool,
    sharding_group_size: int = 2,
) -> ShardedRelayState | None:
    """
    Set up fused sharded relay for 2D sparse parallelism.

    Creates the RCCLX communicators and FusedShardedRelayMultiGroup needed
    for phase-synchronized multi-group allreduce. Returns None if any
    precondition fails (disabling sharded relay).

    Args:
        global_rank: Global rank of this process across all nodes.
        world_size: Total number of ranks (all nodes combined).
        use_inter_host_allreduce: If True, sharded relay is not supported.
        sharding_group_size: Number of active ranks per sparse group. The
            underlying C++ kernel requires exactly 2; any other value disables
            sharded relay.

    Returns:
        ShardedRelayState on success, None to disable sharded relay.
    """
    fused_cls = _get_fused_sharded_relay_multi_group_cls()
    if fused_cls is None:
        logger.warning(
            "[TorchRec 2D Parallel] FusedShardedRelayMultiGroup not available. "
            "Disabling sharded relay mode."
        )
        return None

    local_size = get_local_size(world_size)
    local_rank = global_rank % local_size

    if not _validate_sharded_relay_preconditions(
        use_inter_host_allreduce, sharding_group_size, local_size
    ):
        return None

    sparse_group_size = sharding_group_size
    num_sparse_groups = local_size // sparse_group_size

    logger.info(
        f"[TorchRec 2D Parallel] sparse_group_size={sparse_group_size}, "
        f"num_sparse_groups={num_sparse_groups}, local_size={local_size}"
    )

    my_node_idx = global_rank // local_size
    num_nodes = world_size // local_size

    logger.info(
        f"[TorchRec 2D Parallel] Setting up fused sharded relay: "
        f"global_rank={global_rank}, num_sparse_groups={num_sparse_groups}, "
        f"num_nodes={num_nodes}"
    )

    try:
        rcclx_comm = _create_intra_node_rcclx_comm(
            global_rank, local_rank, local_size, my_node_idx
        )
        if rcclx_comm is None:
            return None
        intra_node_pytorch_pg = _create_intra_node_pytorch_pg(
            global_rank, local_size, num_nodes, my_node_idx
        )
    except Exception as e:
        logger.warning(
            f"[TorchRec 2D Parallel] Failed to create RCCLX comm: {e}. "
            "Disabling sharded relay mode."
        )
        return None

    # Build per-group active ranks list using LOCAL ranks
    all_active_ranks_list: list[list[int]] = [
        list(range(g * sparse_group_size, (g + 1) * sparse_group_size))
        for g in range(num_sparse_groups)
    ]

    # Create ONE FusedShardedRelayMultiGroup for ALL sparse groups at once.
    try:
        fused = fused_cls(
            rcclx_comm=rcclx_comm,
            world_size=local_size,
            rank=local_rank,
            all_active_ranks=all_active_ranks_list,
        )
        logger.info(
            f"[TorchRec 2D Parallel] Created FusedShardedRelayMultiGroup "
            f"(single instance for all {num_sparse_groups} groups): "
            f"global_rank={global_rank}, local_rank={local_rank}, "
            f"all_active_ranks={all_active_ranks_list}"
        )
    except Exception as e:
        logger.warning(
            f"[TorchRec 2D Parallel] Failed to create FusedShardedRelayMultiGroup: {e}. "
            "Disabling sharded relay mode."
        )
        return None

    return ShardedRelayState(
        fused=fused,
        intra_node_pytorch_pg=intra_node_pytorch_pg,
        local_rank=local_rank,
        sparse_group_size=sparse_group_size,
        my_sparse_group=local_rank // sparse_group_size,
        num_sparse_groups=num_sparse_groups,
        local_size=local_size,
        precomputed_active_ranks=all_active_ranks_list,
        _rcclx_comm=rcclx_comm,
    )


def cleanup_sharded_relay(state: ShardedRelayState) -> None:
    """
    Properly finalize RCCLX communicators to avoid thread cleanup issues.

    Cleanup order:
    1. Clear the FusedShardedRelayMultiGroup instance (it holds a comm reference)
    2. Finalize the shared intra-node RCCLX comm
    """
    state.fused = None

    if state._rcclx_comm is not None:
        try:
            state._rcclx_comm.finalize()
        except Exception as e:
            logger.warning(f"[TorchRec 2D Parallel] Error finalizing RCCLX comm: {e}")
    state._rcclx_comm = None


def _get_active_flat_buf(
    state: ShardedRelayState,
    total: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a grow-only flat buffer for the active group.

    Reuses the cached buffer if it is large enough; reallocates (and caches
    the new larger buffer) only when the total element count has grown.
    """
    existing = state._active_flat_cache.get(dtype)
    if existing is None or existing.numel() < total or existing.device != device:
        state._active_flat_cache[dtype] = torch.empty(total, dtype=dtype, device=device)
    buf = state._active_flat_cache[dtype]
    return buf if buf.numel() == total else buf.narrow(0, 0, total)


def _get_helper_flat_buf(
    state: ShardedRelayState,
    group_idx: int,
    total: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a grow-only flat scratch buffer for a specific helper group.

    Keyed by (group_idx, dtype) because phase-sync processes all groups
    simultaneously — each helper group needs its own buffer.  Weights (bf16)
    and optimizer states (fp32) have separate buffers and do not evict each
    other across training steps.

    The buffer is sized to ``total``, which should be
    _passthrough_helper_size(per_group_total_counts[g], ...).
    """
    key = (group_idx, dtype)
    existing = state._helper_flat_cache.get(key)
    if existing is None or existing.numel() < total or existing.device != device:
        state._helper_flat_cache[key] = torch.empty(total, dtype=dtype, device=device)
    buf = state._helper_flat_cache[key]
    return buf if buf.numel() == total else buf.narrow(0, 0, total)


def _passthrough_helper_size(
    total_g: int,
    sparse_group_size: int,
    num_chunks: int,
) -> int:
    """Compute the passthrough helper buffer size for one group.

    Returns the allocation size (in elements) for the two-slot passthrough
    helper buffer:

        min(total_g, sparse_group_size × chunk_aligned)

    where chunk_aligned = (total_g // num_chunks) rounded down to
    CHUNK_ALIGN_ELEMENTS = 128 elements, falling back to total_g when
    total_g < num_chunks × 128.

    This is the canonical Python-side mirror of the C++ ``minRequired``
    formula in TorchCommRCCLX.cpp.
    """
    CHUNK_ALIGN_ELEMENTS = 128
    chunk = total_g // num_chunks
    chunk_aligned = (chunk // CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS
    if chunk_aligned == 0:
        chunk_aligned = total_g
    return min(total_g, sparse_group_size * chunk_aligned)


def allreduce_tensors_with_sharded_relay(
    state: ShardedRelayState,
    tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    annotation: str,
    op: dist.ReduceOp = dist.ReduceOp.AVG,
) -> None:
    """
    Perform allreduce using the fused sharded relay algorithm.

    Flat-concat approach — one fused call per dtype
    ------------------------------------------------
    For each dtype present in ``tensors_dict``:

    1. **Pack**: copy all of the active group's tensors for this dtype into a
       single contiguous flat buffer (device-to-device HBM copy).

    2. **Metadata** (first call only): run one ``dist.all_gather`` to learn
       the total flat size for each group, then cache the result permanently.
       Embedding table dimensions are fixed throughout training, so this
       allgather never needs to repeat.

    3. **Build group tensors**: active group → flat pack buffer; each helper
       group → grow-only flat scratch buffer sized to that group's total.

    4. **One fused call**: ``allreduce_multi_group`` with 4 big flat buffers,
       one per sparse group.  All groups execute in lockstep phases
       (phase-synchronized), eliminating XGMI link contention.

    5. **Unpack**: copy allreduced values from the flat buffer back into each
       original tensor (device-to-device HBM copy).

    The two HBM copies (pack + unpack) add ~0.3 ms for ~200 MB of data at
    1.3 TB/s — negligible compared to eliminating ~100 serial kernel launches.

    Args:
        state: Sharded relay runtime state (pre-computed invariants + comms).
        tensors_dict: Tensors to allreduce, grouped by dtype.
        annotation: Profiling annotation string for record_function.
        op: Reduction op to apply. Only ``ReduceOp.AVG`` and ``ReduceOp.SUM``
            are supported by the underlying RCCLX kernel; other values will
            be rejected by the backend.
    """
    sparse_group_size = state.sparse_group_size
    my_sparse_group = state.my_sparse_group
    num_sparse_groups = state.num_sparse_groups
    local_size = state.local_size
    precomputed_active_ranks = state.precomputed_active_ranks

    with record_function(f"{annotation}_fused_sharded_relay"):
        for dtype, my_tensor_list in tensors_dict.items():
            if not my_tensor_list:
                continue

            my_total = sum(t.numel() for t in my_tensor_list)
            if my_total == 0:
                continue

            device = my_tensor_list[0].device

            # --- Step 1: Pack ---
            # torch.cat(out=) writes all tensors into the pre-allocated flat
            # buffer in a single fused CUDA kernel, replacing N individual
            # copy_() calls (N = number of embedding tables, typically 101 for
            # BM-FM).  Each copy_() incurs a separate kernel launch (~1-5μs on
            # AMD); fusing them into one eliminates ~100 launches for pack.
            active_flat = _get_active_flat_buf(state, my_total, dtype, device)
            torch.cat(
                [t.flatten() for t in my_tensor_list],
                out=active_flat,
            )

            # --- Step 2: Metadata (allgather once, cache forever) ---
            meta_key = annotation + str(dtype)
            per_group_total_counts: list[int]

            if meta_key not in state._flat_metadata_cache:
                if state.intra_node_pytorch_pg is not None:
                    my_total_tensor = torch.tensor(
                        [my_total], dtype=torch.int64, device=device
                    )
                    all_totals_list = [
                        torch.zeros(1, dtype=torch.int64, device=device)
                        for _ in range(local_size)
                    ]
                    dist.all_gather(
                        all_totals_list,
                        my_total_tensor,
                        group=state.intra_node_pytorch_pg,
                    )
                    per_group_total_counts = [
                        int(all_totals_list[g * sparse_group_size].item())
                        for g in range(num_sparse_groups)
                    ]
                else:
                    logger.warning(
                        "[TorchRec 2D Parallel] no intra_node_pytorch_pg! "
                        "Assuming all groups have the same total element count."
                    )
                    per_group_total_counts = [my_total] * num_sparse_groups

                state._flat_metadata_cache[meta_key] = per_group_total_counts
                logger.info(
                    f"[TorchRec 2D Parallel] flat allreduce metadata cached: "
                    f"annotation={annotation!r}, dtype={dtype}, "
                    f"per_group_total_counts={per_group_total_counts}"
                )
            else:
                per_group_total_counts = state._flat_metadata_cache[meta_key]

            # --- Step 3: Build group tensor list ---
            group_tensors: list[torch.Tensor] = []
            group_sizes: list[int] = []

            # Compute num_chunks for passthrough helper size (mirrors C++).
            num_chunks = (local_size - sparse_group_size) + 1

            for g in range(num_sparse_groups):
                if g == my_sparse_group:
                    group_tensors.append(active_flat)
                    group_sizes.append(my_total)
                else:
                    total_g = per_group_total_counts[g]
                    helper_size_g = _passthrough_helper_size(
                        total_g, sparse_group_size, num_chunks
                    )
                    helper_buf = _get_helper_flat_buf(
                        state, g, helper_size_g, dtype, device
                    )
                    group_tensors.append(helper_buf)
                    group_sizes.append(total_g)  # full count goes to the kernel

            # --- Step 4: ONE fused call — all groups, all data, phase-synchronized ---
            state.fused.allreduce_multi_group(
                tensors=group_tensors,
                num_groups=num_sparse_groups,
                per_group_sizes=group_sizes,
                all_active_ranks=precomputed_active_ranks,
                op=op,
                skip_validation=True,
            )

            # --- Step 5: Unpack ---
            # torch._foreach_copy_ dispatches all N copies as a single batched
            # operation, replacing N individual copy_() calls.  The split()
            # produces views (no allocation), so this is still a pure HBM copy
            # but with a single kernel launch instead of N.
            slices = active_flat.split([t.numel() for t in my_tensor_list])
            torch._foreach_copy_(
                my_tensor_list,
                [s.view(t.shape) for s, t in zip(slices, my_tensor_list)],
            )
