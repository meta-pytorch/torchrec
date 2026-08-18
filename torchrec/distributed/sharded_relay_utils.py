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
    # Grow-only flat OUTPUT buffer for the active group (reduce-scatter
    # out-of-place path): dtype → tensor. Holds recv_count elements; unused on
    # the in-place path (where the output aliases the input's local block).
    _active_output_flat_cache: dict[torch.dtype, torch.Tensor] = field(
        default_factory=dict
    )
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
            if node_idx == my_node_idx and isinstance(pg, dist.ProcessGroup):
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
    _passthrough_helper_size(count_g, ...) for the count the kernel receives for
    this group -- the group total for allreduce/reduce-scatter, the per-segment
    count for all-to-all.
    """
    key = (group_idx, dtype)
    existing = state._helper_flat_cache.get(key)
    if existing is None or existing.numel() < total or existing.device != device:
        state._helper_flat_cache[key] = torch.empty(total, dtype=dtype, device=device)
    buf = state._helper_flat_cache[key]
    return buf if buf.numel() == total else buf.narrow(0, 0, total)


def _passthrough_helper_size(
    count_g: int,
    sparse_group_size: int,
    num_chunks: int,
) -> int:
    """Compute the passthrough helper buffer size for one group.

    ``count_g`` is the per-group element count that the KERNEL receives for this
    group, which is not always the caller's group total. Allreduce and
    reduce-scatter hand the kernel their group total, so they pass that;
    all-to-all hands it segmentCounts[g], so it passes the per-segment count. The
    kernel derives its chunking from whichever count it was given, so the helper
    size has to be derived from that same value -- passing a group total where the
    kernel sees a segment count (or the reverse) mis-sizes the buffer.

    ``num_chunks`` must equal the kernel's numHelpers + 1.

    Returns the allocation size (in elements) for the two-slot passthrough
    helper buffer:

        min(count_g, sparse_group_size × chunk_aligned)

    where chunk_aligned = (count_g // num_chunks) rounded down to
    CHUNK_ALIGN_ELEMENTS = 128 elements, falling back to count_g when
    count_g < num_chunks × 128.

    This is the canonical Python-side mirror of the C++ ``minRequired``
    formula in TorchCommRCCLX.cpp.
    """
    CHUNK_ALIGN_ELEMENTS = 128
    chunk = count_g // num_chunks
    chunk_aligned = (chunk // CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS
    if chunk_aligned == 0:
        chunk_aligned = count_g
    return min(count_g, sparse_group_size * chunk_aligned)


def _get_active_output_flat_buf(
    state: ShardedRelayState,
    total: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a grow-only flat OUTPUT buffer for the active group.

    Used by the reduce-scatter out-of-place path to receive the reduced
    output block (recv_count elements). Separate from _active_flat_cache so
    that the input (2 x recv_count) and output (recv_count) buffers do not
    evict each other.
    """
    existing = state._active_output_flat_cache.get(dtype)
    if existing is None or existing.numel() < total or existing.device != device:
        state._active_output_flat_cache[dtype] = torch.empty(
            total, dtype=dtype, device=device
        )
    buf = state._active_output_flat_cache[dtype]
    return buf if buf.numel() == total else buf.narrow(0, 0, total)


def _pack_into_flat(tensor_list: list[torch.Tensor], flat: torch.Tensor) -> None:
    """Copy each tensor in ``tensor_list`` into successive slices of ``flat``.

    Batched device-to-device pack: ``torch.cat(out=)`` writes all tensors into
    the destination in a single fused kernel, replacing N individual copy_()
    calls (N = number of embedding tables, ~101 for BM-FM). Each copy_() incurs
    a separate kernel launch (~1-5us on AMD); fusing them eliminates ~100
    launches per pack. Fills the first ``sum(numel)`` elements of ``flat``
    (which the caller may size larger); tensors must be contiguous.
    """
    if not tensor_list:
        return
    total = sum(t.numel() for t in tensor_list)
    dst = flat if flat.numel() == total else flat.narrow(0, 0, total)
    torch.cat([t.flatten() for t in tensor_list], out=dst)


def _unpack_from_flat(flat: torch.Tensor, tensor_list: list[torch.Tensor]) -> None:
    """Scatter successive slices of ``flat`` back into each tensor.

    Inverse of :func:`_pack_into_flat`. ``torch._foreach_copy_`` dispatches all
    N copies as a single batched op; ``split()`` yields views (no allocation),
    so this is still a pure HBM copy but with one launch instead of N.
    """
    if not tensor_list:
        return
    total = sum(t.numel() for t in tensor_list)
    src = flat if flat.numel() == total else flat.narrow(0, 0, total)
    slices = src.split([t.numel() for t in tensor_list])
    torch._foreach_copy_(
        tensor_list, [s.view(t.shape) for s, t in zip(slices, tensor_list)]
    )


def reduce_scatter_tensors_with_sharded_relay(
    state: ShardedRelayState,
    input_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    output_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    annotation: str,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    in_place: bool = False,
) -> None:
    """
    Perform reduce-scatter using the fused sharded relay algorithm.

    Reduce-scatter analogue of allreduce_tensors_with_sharded_relay. For each
    dtype, the active group's input tensors are packed into a single flat send
    buffer (holding nActiveRanks x recv_count elements: block[i] is the slice
    destined for active index i), ONE fused call reduce-scatters all groups
    simultaneously (phase-synchronized, no XGMI contention), and the reduced
    output block (recv_count elements) is unpacked into the caller's output
    tensors.

    In-place vs out-of-place
    ------------------------
    ``in_place`` controls the internal flat output buffer for the active group:

    - ``in_place=True``: the kernel writes the output into the input flat
      buffer's local-contribution block (a view at ownBlockOffset). This
      exercises the kernel's in-place path and avoids a separate output
      allocation.
    - ``in_place=False``: a separate grow-only flat output buffer is used,
      exercising the kernel's out-of-place path.

    Either way, results are unpacked into ``output_tensors_dict``. (Reduce-scatter
    is inherently out-of-place at the tensor level since the input is
    nActiveRanks x larger than the output; ``in_place`` only selects the internal
    buffer strategy / kernel code path.)

    Args:
        state: Sharded relay runtime state (pre-computed invariants + comms).
        input_tensors_dict: Send tensors grouped by dtype. The active group's
            tensors for a dtype must total nActiveRanks x recv_count elements.
        output_tensors_dict: Receive tensors grouped by dtype. The active
            group's tensors for a dtype must total recv_count elements.
        annotation: Profiling annotation string for record_function.
        op: Reduction op (ReduceOp.SUM or ReduceOp.AVG).
        in_place: Select the internal output-buffer strategy / kernel path.
    """
    sparse_group_size = state.sparse_group_size
    my_sparse_group = state.my_sparse_group
    num_sparse_groups = state.num_sparse_groups
    local_size = state.local_size
    local_rank = state.local_rank
    precomputed_active_ranks = state.precomputed_active_ranks

    # This rank's index among its group's active ranks (0 or 1 for size-2).
    my_active_index = local_rank % sparse_group_size

    with record_function(f"{annotation}_fused_sharded_relay_reduce_scatter"):
        for dtype, my_input_list in input_tensors_dict.items():
            if not my_input_list:
                continue

            my_input_total = sum(t.numel() for t in my_input_list)
            if my_input_total == 0:
                continue

            if my_input_total % sparse_group_size != 0:
                raise ValueError(
                    f"reduce_scatter_tensors_with_sharded_relay: active input total "
                    f"({my_input_total}) for dtype {dtype} must be divisible by "
                    f"sparse_group_size ({sparse_group_size})."
                )
            my_recv_count = my_input_total // sparse_group_size

            my_output_list = output_tensors_dict.get(dtype, [])
            my_output_total = sum(t.numel() for t in my_output_list)
            if my_output_total != my_recv_count:
                raise ValueError(
                    f"reduce_scatter_tensors_with_sharded_relay: active output total "
                    f"({my_output_total}) for dtype {dtype} must equal recv_count "
                    f"({my_recv_count} = input_total/{sparse_group_size})."
                )

            device = my_input_list[0].device

            # --- Step 1: Metadata (allgather recv counts once, cache forever) ---
            meta_key = "rs_" + annotation + str(dtype)
            per_group_recv_counts: list[int]

            if meta_key not in state._flat_metadata_cache:
                if state.intra_node_pytorch_pg is not None:
                    my_recv_tensor = torch.tensor(
                        [my_recv_count], dtype=torch.int64, device=device
                    )
                    all_recv_list = [
                        torch.zeros(1, dtype=torch.int64, device=device)
                        for _ in range(local_size)
                    ]
                    dist.all_gather(
                        all_recv_list,
                        my_recv_tensor,
                        group=state.intra_node_pytorch_pg,
                    )
                    per_group_recv_counts = [
                        int(all_recv_list[g * sparse_group_size].item())
                        for g in range(num_sparse_groups)
                    ]
                else:
                    logger.warning(
                        "[TorchRec 2D Parallel] no intra_node_pytorch_pg! "
                        "Assuming all groups have the same recv count."
                    )
                    per_group_recv_counts = [my_recv_count] * num_sparse_groups

                state._flat_metadata_cache[meta_key] = per_group_recv_counts
                logger.info(
                    f"[TorchRec 2D Parallel] flat reduce-scatter metadata cached: "
                    f"annotation={annotation!r}, dtype={dtype}, "
                    f"per_group_recv_counts={per_group_recv_counts}"
                )
            else:
                per_group_recv_counts = state._flat_metadata_cache[meta_key]

            # --- Step 3: Build per-group tensors ---
            # Each group contributes ONE contiguous tensor. The active group's
            # per-table inputs are packed into a single flat send buffer (the
            # single-segment zero-copy fast path); helper groups pass a single
            # passthrough scratch buffer. The reduced output block is unpacked
            # into my_output_list after the call.
            input_group_tensors: list[torch.Tensor] = []
            output_group_tensors: list[torch.Tensor] = []

            unpack_flat: torch.Tensor | None = None

            num_chunks = (local_size - sparse_group_size) + 1

            for g in range(num_sparse_groups):
                if g == my_sparse_group:
                    in_flat = _get_active_flat_buf(state, my_input_total, dtype, device)
                    _pack_into_flat(my_input_list, in_flat)
                    input_group_tensors.append(in_flat)
                    if in_place:
                        # In-place: the output block is the owned-block view
                        # inside the input flat (offset
                        # my_active_index * recv_count), so
                        # activeOutSegPtr == activeInSegPtr + ownBlockOffset.
                        out_view = in_flat.narrow(
                            0, my_active_index * my_recv_count, my_recv_count
                        )
                        output_group_tensors.append(out_view)
                        unpack_flat = out_view
                    else:
                        out_flat = _get_active_output_flat_buf(
                            state, my_recv_count, dtype, device
                        )
                        output_group_tensors.append(out_flat)
                        unpack_flat = out_flat
                else:
                    recv_count_g = per_group_recv_counts[g]
                    helper_size_g = _passthrough_helper_size(
                        recv_count_g, sparse_group_size, num_chunks
                    )
                    helper_buf = _get_helper_flat_buf(
                        state, g, helper_size_g, dtype, device
                    )
                    # Helper uses one two-slot scratch buffer for send and recv.
                    input_group_tensors.append(helper_buf)
                    output_group_tensors.append(helper_buf)

            # --- Step 4: ONE fused call — all groups, phase-synchronized ---
            state.fused.reduce_scatter_multi_group(
                input_tensors=input_group_tensors,
                output_tensors=output_group_tensors,
                num_groups=num_sparse_groups,
                per_group_recv_counts=per_group_recv_counts,
                all_active_ranks=precomputed_active_ranks,
                op=op,
                skip_validation=True,
            )

            # --- Step 5: Unpack the active-group result into caller tensors ---
            if unpack_flat is not None:
                _unpack_from_flat(unpack_flat, my_output_list)


def allreduce_tensors_with_sharded_relay(
    state: ShardedRelayState,
    tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    annotation: str,
    op: dist.ReduceOp | dist.ReduceOp.RedOpType = dist.ReduceOp.AVG,
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


def all_to_all_tensors_with_sharded_relay(
    state: ShardedRelayState,
    input_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    output_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    annotation: str,
) -> None:
    """
    Perform all-to-all using the fused sharded relay algorithm.

    All-to-all analogue of reduce_scatter_tensors_with_sharded_relay. For each
    dtype, the active group's input tensors are packed into a single flat send
    buffer (holding nActiveRanks x segment_count elements:
    input = [sendSeg[0]|sendSeg[1]]), ONE fused call all-to-alls all groups
    simultaneously (phase-synchronized, no XGMI contention), and the transposed
    output (output = [recvSeg[0]|recvSeg[1]]) is unpacked into the caller's
    output tensors.

    All-to-all performs NO reduction (no op) and is OUT-OF-PLACE ONLY: a separate
    flat output buffer is always used (distinct from the input flat buffer), as
    required by the underlying collective.

    Args:
        state: Sharded relay runtime state (pre-computed invariants + comms).
        input_tensors_dict: Send tensors grouped by dtype. The active group's
            tensors for a dtype must total nActiveRanks x segment_count elements.
        output_tensors_dict: Receive tensors grouped by dtype. The active group's
            tensors for a dtype must total the same number of elements as the
            input (nActiveRanks x segment_count).
        annotation: Profiling annotation string for record_function.
    """
    sparse_group_size = state.sparse_group_size
    my_sparse_group = state.my_sparse_group
    num_sparse_groups = state.num_sparse_groups
    local_size = state.local_size
    precomputed_active_ranks = state.precomputed_active_ranks

    with record_function(f"{annotation}_fused_sharded_relay_all_to_all"):
        for dtype, my_input_list in input_tensors_dict.items():
            if not my_input_list:
                continue

            my_input_total = sum(t.numel() for t in my_input_list)
            if my_input_total == 0:
                continue

            if my_input_total % sparse_group_size != 0:
                raise ValueError(
                    f"all_to_all_tensors_with_sharded_relay: active input total "
                    f"({my_input_total}) for dtype {dtype} must be divisible by "
                    f"sparse_group_size ({sparse_group_size})."
                )
            my_segment_count = my_input_total // sparse_group_size

            my_output_list = output_tensors_dict.get(dtype, [])
            my_output_total = sum(t.numel() for t in my_output_list)
            if my_output_total != my_input_total:
                raise ValueError(
                    f"all_to_all_tensors_with_sharded_relay: active output total "
                    f"({my_output_total}) for dtype {dtype} must equal the input "
                    f"total ({my_input_total})."
                )

            device = my_input_list[0].device

            # --- Step 1: Metadata (allgather segment counts once, cache) ---
            meta_key = "a2a_" + annotation + str(dtype)
            per_group_segment_counts: list[int]

            if meta_key not in state._flat_metadata_cache:
                if state.intra_node_pytorch_pg is not None:
                    my_seg_tensor = torch.tensor(
                        [my_segment_count], dtype=torch.int64, device=device
                    )
                    all_seg_list = [
                        torch.zeros(1, dtype=torch.int64, device=device)
                        for _ in range(local_size)
                    ]
                    dist.all_gather(
                        all_seg_list,
                        my_seg_tensor,
                        group=state.intra_node_pytorch_pg,
                    )
                    per_group_segment_counts = [
                        int(all_seg_list[g * sparse_group_size].item())
                        for g in range(num_sparse_groups)
                    ]
                else:
                    logger.warning(
                        "[TorchRec 2D Parallel] no intra_node_pytorch_pg! "
                        "Assuming all groups have the same segment count."
                    )
                    per_group_segment_counts = [my_segment_count] * num_sparse_groups

                state._flat_metadata_cache[meta_key] = per_group_segment_counts
                logger.info(
                    f"[TorchRec 2D Parallel] flat all-to-all metadata cached: "
                    f"annotation={annotation!r}, dtype={dtype}, "
                    f"per_group_segment_counts={per_group_segment_counts}"
                )
            else:
                per_group_segment_counts = state._flat_metadata_cache[meta_key]

            # --- Step 3: Build per-group tensors ---
            # Each group contributes ONE contiguous tensor. All-to-all is
            # out-of-place only: the active group's per-table inputs are packed
            # into a single flat send buffer and a separate flat output buffer
            # receives the transposed result (single-segment fast path); helper
            # groups pass a single passthrough scratch buffer. The output is
            # unpacked into my_output_list after the call.
            input_group_tensors: list[torch.Tensor] = []
            output_group_tensors: list[torch.Tensor] = []

            unpack_flat: torch.Tensor | None = None

            num_chunks = (local_size - sparse_group_size) + 1

            for g in range(num_sparse_groups):
                if g == my_sparse_group:
                    in_flat = _get_active_flat_buf(state, my_input_total, dtype, device)
                    _pack_into_flat(my_input_list, in_flat)
                    out_flat = _get_active_output_flat_buf(
                        state, my_input_total, dtype, device
                    )
                    input_group_tensors.append(in_flat)
                    output_group_tensors.append(out_flat)
                    unpack_flat = out_flat
                else:
                    # _passthrough_helper_size's first argument is the per-group
                    # count the KERNEL receives, not this caller's group total.
                    # All-to-all hands the kernel segmentCounts[g], so it derives
                    # chunkSize = align_down(seg / numChunks, 128) from the
                    # segment count and each helper stages nActiveRanks slots of
                    # that -- exactly what this returns. (The allreduce path
                    # passes its group total for the same reason: that is the
                    # count its kernel sees.) num_chunks matches the kernel's
                    # numHelpers + 1.
                    seg_g = per_group_segment_counts[g]
                    helper_size_g = _passthrough_helper_size(
                        seg_g, sparse_group_size, num_chunks
                    )
                    helper_buf = _get_helper_flat_buf(
                        state, g, helper_size_g, dtype, device
                    )
                    # Helper uses one two-slot scratch buffer for send and recv.
                    input_group_tensors.append(helper_buf)
                    output_group_tensors.append(helper_buf)

            # --- Step 4: ONE fused call — all groups, phase-synchronized ---
            state.fused.all_to_all_multi_group(
                input_tensors=input_group_tensors,
                output_tensors=output_group_tensors,
                num_groups=num_sparse_groups,
                per_group_segment_counts=per_group_segment_counts,
                all_active_ranks=precomputed_active_ranks,
                skip_validation=True,
            )
            # --- Step 5: Unpack the active-group result into caller tensors ---
            if unpack_flat is not None:
                _unpack_from_flat(unpack_flat, my_output_list)


def all_gather_tensors_with_sharded_relay(
    state: ShardedRelayState,
    input_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    output_tensors_dict: dict[torch.dtype, list[torch.Tensor]],
    annotation: str,
    in_place: bool = False,
) -> None:
    """
    Perform all-gather using the fused sharded relay algorithm.

    All-gather analogue of reduce_scatter_tensors_with_sharded_relay (its dual).
    For each dtype, the active group's input tensors are packed into a flat send
    buffer (send_count elements: this rank's contribution), ONE fused call
    all-gathers all groups simultaneously (phase-synchronized, no XGMI
    contention), and the gathered output (nActiveRanks x send_count elements,
    output[i x send_count] from active index i) is unpacked into the caller's
    output tensors.

    All-gather performs NO reduction (no op). ``in_place`` controls the internal
    flat input buffer for the active group:

    - ``in_place=True``: the input is packed into the active rank's own slot of
      the flat output buffer (a view at myActiveIndex x send_count), exercising
      the kernel's in-place path (sendbuff == recvbuff + rank x send_count).
    - ``in_place=False``: a separate flat input buffer is used.

    Either way, results are unpacked into ``output_tensors_dict``.

    Args:
        state: Sharded relay runtime state (pre-computed invariants + comms).
        input_tensors_dict: Send tensors grouped by dtype. The active group's
            tensors for a dtype must total send_count elements.
        output_tensors_dict: Receive tensors grouped by dtype. The active
            group's tensors for a dtype must total nActiveRanks x send_count
            elements.
        annotation: Profiling annotation string for record_function.
        in_place: Select the internal input-buffer strategy / kernel path.
    """
    sparse_group_size = state.sparse_group_size
    my_sparse_group = state.my_sparse_group
    num_sparse_groups = state.num_sparse_groups
    local_size = state.local_size
    local_rank = state.local_rank
    precomputed_active_ranks = state.precomputed_active_ranks

    # This rank's index among its group's active ranks (0 or 1 for size-2).
    my_active_index = local_rank % sparse_group_size

    with record_function(f"{annotation}_fused_sharded_relay_all_gather"):
        for dtype, my_input_list in input_tensors_dict.items():
            if not my_input_list:
                continue

            my_send_count = sum(t.numel() for t in my_input_list)
            if my_send_count == 0:
                continue

            my_output_list = output_tensors_dict.get(dtype, [])
            my_output_total = sum(t.numel() for t in my_output_list)
            if my_output_total != sparse_group_size * my_send_count:
                raise ValueError(
                    f"all_gather_tensors_with_sharded_relay: active output total "
                    f"({my_output_total}) for dtype {dtype} must equal "
                    f"sparse_group_size x send_count "
                    f"({sparse_group_size} x {my_send_count})."
                )

            device = my_input_list[0].device

            # --- Step 1: Metadata (allgather send counts once, cache) ---
            meta_key = "ag_" + annotation + str(dtype)
            per_group_send_counts: list[int]

            if meta_key not in state._flat_metadata_cache:
                if state.intra_node_pytorch_pg is not None:
                    my_send_tensor = torch.tensor(
                        [my_send_count], dtype=torch.int64, device=device
                    )
                    all_send_list = [
                        torch.zeros(1, dtype=torch.int64, device=device)
                        for _ in range(local_size)
                    ]
                    dist.all_gather(
                        all_send_list,
                        my_send_tensor,
                        group=state.intra_node_pytorch_pg,
                    )
                    per_group_send_counts = [
                        int(all_send_list[g * sparse_group_size].item())
                        for g in range(num_sparse_groups)
                    ]
                else:
                    logger.warning(
                        "[TorchRec 2D Parallel] no intra_node_pytorch_pg! "
                        "Assuming all groups have the same send count."
                    )
                    per_group_send_counts = [my_send_count] * num_sparse_groups

                state._flat_metadata_cache[meta_key] = per_group_send_counts
                logger.info(
                    f"[TorchRec 2D Parallel] flat all-gather metadata cached: "
                    f"annotation={annotation!r}, dtype={dtype}, "
                    f"per_group_send_counts={per_group_send_counts}"
                )
            else:
                per_group_send_counts = state._flat_metadata_cache[meta_key]

            # --- Step 3: Build per-group tensors ---
            # Each group contributes ONE contiguous tensor. The active group's
            # per-table inputs are packed into a single flat send buffer and a
            # single flat output buffer receives the gathered result
            # (single-segment fast path); helper groups pass a single
            # passthrough scratch buffer. Out-of-place packs the input into a
            # separate flat send buffer; in-place packs the input into the
            # active rank's own slot of the output flat. The gathered output is
            # unpacked into my_output_list after the call.
            input_group_tensors: list[torch.Tensor] = []
            output_group_tensors: list[torch.Tensor] = []

            unpack_flat: torch.Tensor | None = None

            num_chunks = (local_size - sparse_group_size) + 1

            for g in range(num_sparse_groups):
                if g == my_sparse_group:
                    out_flat = _get_active_output_flat_buf(
                        state, my_output_total, dtype, device
                    )
                    if in_place:
                        # In-place: input aliases the active rank's own output
                        # slot (offset my_active_index * send_count), so
                        # activeInSegPtr == activeOutSegPtr + m*sendCount.
                        in_view = out_flat.narrow(
                            0, my_active_index * my_send_count, my_send_count
                        )
                        _pack_into_flat(my_input_list, in_view)
                        input_group_tensors.append(in_view)
                    else:
                        in_flat = _get_active_flat_buf(
                            state, my_send_count, dtype, device
                        )
                        _pack_into_flat(my_input_list, in_flat)
                        input_group_tensors.append(in_flat)
                    output_group_tensors.append(out_flat)
                    unpack_flat = out_flat
                else:
                    send_g = per_group_send_counts[g]
                    helper_size_g = _passthrough_helper_size(
                        send_g, sparse_group_size, num_chunks
                    )
                    helper_buf = _get_helper_flat_buf(
                        state, g, helper_size_g, dtype, device
                    )
                    # Helper uses one two-slot scratch buffer for send and recv.
                    input_group_tensors.append(helper_buf)
                    output_group_tensors.append(helper_buf)

            # --- Step 4: ONE fused call — all groups, phase-synchronized ---
            state.fused.all_gather_multi_group(
                input_tensors=input_group_tensors,
                output_tensors=output_group_tensors,
                num_groups=num_sparse_groups,
                per_group_send_counts=per_group_send_counts,
                all_active_ranks=precomputed_active_ranks,
                skip_validation=True,
            )

            # --- Step 5: Unpack the active-group result into caller tensors ---
            if unpack_flat is not None:
                _unpack_from_flat(unpack_flat, my_output_list)
