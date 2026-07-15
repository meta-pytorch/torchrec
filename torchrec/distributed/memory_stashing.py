#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.autograd.profiler import record_function
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.logging_handlers import log_ems_event
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)

# Trunk size for the bulk stash (D2H) / restore (H2D) copies. ~32 MiB trunks
# keep the copy engine yielding often enough for the main stream's small copies
# to interleave, at a negligible bandwidth cost (see chunked copy design).
_STASH_CHUNK_SIZE_BYTES: int = 32 * 1024**2


def _tensor_size_text(tensor: Union[List[torch.Tensor], torch.Tensor]) -> str:
    """Return a human-readable size string (KB / MB / GB) for a tensor.

    Uses a KB→MB→GB ladder so small stashed groups (e.g. VLE tables under a
    shrunk ``max_ind_range``) are still legible in the ``record_function``
    trace labels for size validation.
    """
    if isinstance(tensor, list):
        size_bytes = sum(t.numel() * t.element_size() for t in tensor)
    else:
        size_bytes = tensor.numel() * tensor.element_size()
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.2f} KB"
    size_mb = size_kb / 1024
    if size_mb < 1024:
        return f"{size_mb:.2f} MB"
    return f"{size_mb / 1024:.2f} GB"


def chunked_copy_(
    dst: torch.Tensor,
    src: torch.Tensor,
    chunk_size_bytes: int = 512 * 1024**2,  # 512 MiB
    non_blocking: bool = True,
    dummy_compute: bool = True,
) -> None:
    """Copy ``src`` into ``dst`` in byte-bounded chunks to ease copy-engine contention.

    A single bulk ``copy_`` (an H2D restore or a D2H stash) saturates the CUDA
    copy engine until it finishes, serializing every same-direction copy from
    other streams behind it (see the ``copy_engine_execution_order`` study).
    Splitting the transfer into chunks of at most ``chunk_size_bytes`` bytes —
    with a tiny dummy compute op enqueued between chunks — breaks the back-to-back
    readiness of consecutive same-stream copies, so the copy engine yields
    between chunks and lets other streams slip their (typically small) copies
    into the gaps. This reduces, but does not eliminate, the contention latency.

    The copy runs on the current CUDA stream; wrap the call in
    ``with torch.cuda.stream(...)`` to target a specific stream.

    Args:
        dst: Destination tensor, written in place.
        src: Source tensor with the same number of elements and dtype as ``dst``.
        chunk_size_bytes: Maximum size of each chunk in bytes (default 512 MiB).
            Values <= 0 disable chunking and fall back to a single un-chunked
            ``copy_``.
        non_blocking: Passed through to each per-chunk ``copy_`` for async DMA.
        dummy_compute: If True, enqueue a tiny ``add_`` between chunks so the
            copy engine yields to other streams. The study shows chunking alone
            does not yield — under current-stream-first scheduling consecutive
            same-stream copies run back-to-back as one uninterrupted block.

    Note:
        ``dst`` and ``src`` must be contiguous so that flat 1-D slicing aliases
        their storage; non-contiguous tensors fall back to a single ``copy_``.
    """
    if dst.numel() != src.numel():
        raise ValueError(
            f"chunked_copy_ size mismatch: dst has {dst.numel()} elements, "
            f"src has {src.numel()}"
        )

    numel = dst.numel()

    # Fall back to a plain copy when chunking is disabled, the tensor is empty,
    # or the layout cannot be safely flattened into an aliasing view.
    if (
        chunk_size_bytes <= 0
        or numel == 0
        or not dst.is_contiguous()
        or not src.is_contiguous()
    ):
        dst.copy_(src, non_blocking=non_blocking)
        return

    chunk_elems = max(1, chunk_size_bytes // dst.element_size())
    if chunk_elems >= numel:
        dst.copy_(src, non_blocking=non_blocking)
        return

    dst_flat = dst.view(-1)
    src_flat = src.view(-1)

    # Reused single-element scratch tensor for the inter-chunk dummy op, so we
    # do not allocate per chunk. Allocate it directly on the GPU side of the
    # transfer (dst for H2D, src for D2H).
    # Citrine C3: create the scratch tensor directly on device, not on CPU.
    cuda_device = dst.device if dst.is_cuda else src.device if src.is_cuda else None
    dummy = (
        torch.ones(1, device=cuda_device)
        if dummy_compute and cuda_device is not None
        else None
    )

    for start in range(0, numel, chunk_elems):
        end = min(start + chunk_elems, numel)
        dst_flat[start:end].copy_(src_flat[start:end], non_blocking=non_blocking)
        # Break same-stream copy readiness so the copy engine checks other
        # streams before the next chunk runs (no dummy op after the last chunk).
        if dummy is not None and end < numel:
            dummy.add_(1)


class MemoryStashingManager:
    """
    Class-level manager that holds shared resources (streams, threads, etc.)
    for all memory-stashing use cases.

    All state is stored in class variables and accessed via classmethods,
    so there is no need to instantiate or manage a singleton instance.

    Call ``reset()`` to tear down and release all resources (e.g. between
    training runs or in tests).
    """

    _host_to_device_stream: Optional[torch.cuda.Stream] = None
    _device_to_host_stream: Optional[torch.cuda.Stream] = None
    _embedding_weight_restore_callbacks: List[Callable[..., None]] = []
    _optimizer_state_restore_callbacks: List[Callable[..., None]] = []
    _emo_cache_restore_callbacks: List[Callable[..., None]] = []
    _stash_executor: Optional[ThreadPoolExecutor] = None
    _delay_stash: bool = False
    _pending_stash_callbacks: List[Callable[[Optional[torch.Tensor]], None]] = []
    # Event names already emitted to the structured event logger. Used to emit
    # each EMS telemetry event at most once per process. Required because
    # TrainingOptimizationLogger.log() has no built-in dedup and several call
    # sites run on every batch; without this guard they would flood the event
    # logger. Intentionally NOT cleared in reset() to avoid re-flooding.
    _logged_event_keys: set[str] = set()
    # Trunk size (bytes) for the bulk D2H stash / H2D restore copies, used as
    # ``chunked_copy_``'s ``chunk_size_bytes``. Defaults to
    # ``_STASH_CHUNK_SIZE_BYTES``; override via ``set_trunk_size``.
    _stash_chunk_size_bytes: int = _STASH_CHUNK_SIZE_BYTES

    @classmethod
    def _log_ems_once(
        cls, event_name: str, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Emit an EMS event to the structured event logger at most once.

        EMO ``lxu_cache`` stash/restore events are routed through here too —
        they are memory-stashing operations, so they share the EMS technique
        and stay distinguishable via their ``emo_`` event-name prefix.
        """
        if event_name in cls._logged_event_keys:
            return
        cls._logged_event_keys.add(event_name)
        log_ems_event(event_name, metadata)

    # DCP staging-boundary redirect map (criterion 1: no CUDA invalid-argument).
    # Maps a *freed* GPU StorageImpl identity (``untyped_storage()._cdata``) to
    # ``(full_slab_cpu_buffer, base_storage_offset, orphaned_slab_storage,
    # d2h_done_event)``. The checkpoint stager may
    # capture a view of ``weights_dev`` BEFORE the forward-pass stash swaps it to
    # CPU + ``resize_(0)``s the GPU storage; its later ``copy_`` would then read
    # a 0-byte CUDA storage at a non-zero ``storage_offset`` -> unmapped address
    # -> ``cudaErrorInvalidValue``. ``staged_cpu_view_for`` lets the stager copy
    # from the CPU buffer instead. The key is the StorageImpl pointer, which is
    # SHARED by every aliasing view and SURVIVES ``resize_(0)`` (the StorageImpl
    # is reused, only its DataPtr is freed), so a single registration covers all
    # captured siblings regardless of capture-vs-stash ordering.
    _staged_storage_buffers: Dict[
        int,
        Tuple[
            torch.Tensor,
            Union[int, torch.SymInt],
            torch.UntypedStorage,
            torch.cuda.Event,
        ],
    ] = {}

    @classmethod
    def thread_submit(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a function to the shared thread pool executor."""
        if cls._stash_executor is None:
            cls._stash_executor = ThreadPoolExecutor(max_workers=1)
        return cls._stash_executor.submit(fn, *args, **kwargs)

    @classmethod
    def h2d_stream(cls) -> torch.cuda.Stream:
        """Return the shared CUDA stream for H2D transfers."""
        assert cls._host_to_device_stream is not None
        return cls._host_to_device_stream

    @classmethod
    def d2h_stream(cls) -> torch.cuda.Stream:
        """Return the shared CUDA stream for D2H transfers."""
        assert cls._device_to_host_stream is not None
        return cls._device_to_host_stream

    @classmethod
    def is_enabled(cls) -> bool:
        """Return whether memory stashing streams have been initialized."""
        return cls._device_to_host_stream is not None

    @classmethod
    def set_streams(
        cls,
        host_to_device_stream: Optional[torch.cuda.Stream] = None,
        device_to_host_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Set the shared CUDA streams for H2D and D2H transfers."""
        cls._host_to_device_stream = host_to_device_stream
        if device_to_host_stream is None:
            cls._device_to_host_stream = host_to_device_stream
        else:
            cls._device_to_host_stream = device_to_host_stream
        logger.info("MemoryStashingManager: streams initialized")
        cls._log_ems_once("ems_streams_initialized")

    @classmethod
    def set_trunk_size(cls, size_bytes: int) -> None:
        """Set the trunk size (in bytes) for bulk stash/restore copies.

        Controls how ``_stash_tensors`` splits each bulk D2H stash and H2D
        restore via ``chunked_copy_``. Smaller trunks let the copy engine yield
        to the main stream's small copies more often, at a small bandwidth cost;
        values <= 0 disable chunking and fall back to a single ``copy_``.
        """
        cls._stash_chunk_size_bytes = size_bytes

    @classmethod
    def set_delay_stash(cls, delay: bool) -> None:
        """Enable or disable delayed stashing.

        When enabled, ``stash_embedding_weights`` collects tensors and records
        a sync event but does NOT execute the D2H copy or free HBM.  The
        caller must later invoke ``execute_pending_stashes`` to perform the
        actual stash and register restore callbacks.
        """
        cls._delay_stash = delay

    @classmethod
    def execute_pending_stashes(cls) -> None:
        """Execute all pending (delayed) stash operations.

        For each pending stash, invokes the deferred callable which performs
        the D2H copy, ``resize_(0)`` to free HBM, and registers the restore
        callback so that a subsequent ``restore_embedding_weights`` call can
        bring the data back.
        """
        for cb in cls._pending_stash_callbacks:
            cb(None)
        cls._pending_stash_callbacks.clear()

    @classmethod
    def reset(cls) -> None:
        """Release all resources."""
        logger.info("MemoryStashingManager: resetting all resources")
        cls._host_to_device_stream = None
        cls._device_to_host_stream = None
        cls._embedding_weight_restore_callbacks.clear()
        cls._optimizer_state_restore_callbacks.clear()
        cls._emo_cache_restore_callbacks.clear()
        cls._delay_stash = False
        cls._stash_chunk_size_bytes = _STASH_CHUNK_SIZE_BYTES
        cls._pending_stash_callbacks.clear()
        cls._staged_storage_buffers.clear()
        if cls._stash_executor is not None:
            cls._stash_executor.shutdown(wait=False)
            cls._stash_executor = None

    @classmethod
    def _drop_staging_redirect(cls, storages: List[torch.UntypedStorage]) -> None:
        """Remove DCP-staging redirect entries for the given (restored) slabs.

        Called from ``restore`` once ``weights_dev`` points back to valid CUDA
        storage: any view the stager captures from here on is GPU-valid, so the
        redirect is no longer needed. Popping keeps ``_staged_storage_buffers``
        from growing across checkpoints and releases the pinned CPU buffers.
        """
        for storage in storages:
            cls._staged_storage_buffers.pop(storage._cdata, None)

    @classmethod
    def staged_cpu_view_for(cls, obj: torch.Tensor) -> Optional[torch.Tensor]:
        """Return a CPU view to copy from if ``obj`` aliases a stashed slab.

        Called by the DCP staging boundary (``_staging_utils.create_and_copy_tensor``)
        when it is about to copy a CUDA tensor whose storage has been freed
        (``untyped_storage().size() == 0``) by a forward-pass stash. Looks up the
        original GPU StorageImpl (shared by all aliasing views, stable across
        ``resize_(0)``) and reconstructs the matching slice of the pinned CPU
        buffer so the stager performs a valid CPU->CPU copy instead of faulting
        on the unmapped device address. Returns ``None`` if ``obj`` is not a
        known stashed slab (then the caller copies from ``obj`` as usual).
        """
        entry = cls._staged_storage_buffers.get(obj.untyped_storage()._cdata)
        if entry is None:
            return None
        cpu_buffer, base_storage_offset, _slab_storage, d2h_event = entry
        cpu_offset = obj.storage_offset() - base_storage_offset
        if cpu_offset < 0 or cpu_offset + obj.numel() > cpu_buffer.numel():
            return None
        # Ensure the async D2H that fills cpu_buffer has completed before the
        # stager reads it (host-side wait; near-zero cost when already done).
        d2h_event.synchronize()
        view = torch.empty(0, dtype=cpu_buffer.dtype, device="cpu")
        view.set_(cpu_buffer.untyped_storage(), cpu_offset, obj.shape, obj.stride())
        return view

    @classmethod
    def restore_embedding_weights(
        cls,
        _grad: Optional[torch.Tensor] = None,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Pop and call all embedding weight restore callbacks in reverse order."""
        cls._log_ems_once(
            "ems_restore_embedding_weights_callbacks",
            {"num_callbacks": str(len(cls._embedding_weight_restore_callbacks))},
        )
        while cls._embedding_weight_restore_callbacks:
            cls._embedding_weight_restore_callbacks.pop()(None, sync_event)

    @classmethod
    def restore_optimizer_state(
        cls,
        _grad: Optional[torch.Tensor] = None,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Pop and call all optimizer state restore callbacks in reverse order."""
        cls._log_ems_once(
            "ems_restore_optimizer_state_callbacks",
            {"num_callbacks": str(len(cls._optimizer_state_restore_callbacks))},
        )
        while cls._optimizer_state_restore_callbacks:
            cls._optimizer_state_restore_callbacks.pop()(None, sync_event)

    @classmethod
    def _stash_tensors(
        cls,
        tensors: List[torch.Tensor],
        label: str = "tensor",
        sync_event: Optional[torch.cuda.Event] = None,
        delay: bool = False,
    ) -> Tuple[
        Callable[[Optional[torch.Tensor]], None],
        Callable[[Optional[torch.Tensor]], None],
        Callable[[Optional[torch.Tensor]], None],
    ]:
        """
        Stash a list of CUDA tensors from HBM to CPU asynchronously.

        Core implementation shared by ``stash_embedding_weights`` and
        ``stash_optimizer_state``.  Wraps the D2H copy and HBM free into an
        ``execute_stash`` callable.  In non-delayed mode the callable is
        invoked immediately; in delayed mode it is returned for the caller
        to invoke later.

        Args:
            tensors: CUDA tensors to stash.  Non-CUDA / empty tensors are
                silently skipped.
            label: Human-readable label used in ``record_function`` profiling
                annotations (e.g. ``"embedding"`` or ``"optimizer state"``).
            sync_event: Optional CUDA event recorded on the caller's stream.
                When provided, the D2H stream waits on this event instead of
                calling ``torch.cuda.current_stream()``.  This is required
                when ``_stash_tensors`` runs on a background thread, where
                ``current_stream()`` would return the wrong (idle) stream.
            delay: If True, do NOT execute the stash immediately.  The
                returned ``execute_stash`` callable must be invoked later
                to perform the actual D2H copy and free HBM.  A sync event
                is recorded automatically so that the deferred execution
                waits for the correct GPU work.

        Returns:
            A tuple of three callback functions:
            - await_restore: Pauses current stream awaiting restore completion
            - restore: Retrieves stashed data from CPU back to HBM asynchronously
            - execute_stash: Performs the D2H copy and frees HBM.  Accepts an
              optional ``torch.Tensor`` so it can be registered as a backward
              hook.  In non-delayed mode it has already been called and is
              safe to call again (no-op on empty tensor list).
        """
        # (hbm_tensor ref, cpu_buffer, original_storage_size)
        stash_data: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        # Original CUDA storages saved before .data= so restore can
        # re-point tensors (and their views) back to the same storage.
        _orig_storages: List[torch.UntypedStorage] = []
        _orig_storage_offsets: List[Union[int, torch.SymInt]] = []

        # Restore events populated by ``restore`` and consumed by ``await_restore``
        restore_events: List[torch.cuda.Event] = []

        # For delayed mode, capture the current stream state now so
        # execute_stash can correctly synchronize when called later.
        if delay and sync_event is None:
            sync_event = torch.cuda.Event()
            sync_event.record(torch.cuda.current_stream())

        def execute_stash(_grad: Optional[torch.Tensor] = None) -> None:
            """Perform the D2H copy and free HBM."""
            d2h_stream = cls.d2h_stream()

            # Ensure all operations on the caller's stream complete before we
            # start copying — prevents reading while still being written.
            # When sync_event is provided (delayed or background thread), use
            # it instead of current_stream() which may have advanced or be the
            # wrong stream.
            if sync_event is not None:
                d2h_stream.wait_event(sync_event)
            else:
                d2h_stream.wait_stream(torch.cuda.current_stream())

            size_text = _tensor_size_text(tensors)
            cls._log_ems_once(
                f"ems_stash_summary_{label.replace(' ', '_')}",
                {"label": label, "num_tensors": str(len(tensors)), "size": size_text},
            )

            # Start async copy from HBM to CPU
            with record_function(f"stash {label} to host ({size_text})"):
                with torch.cuda.stream(d2h_stream):
                    for tensor in tensors:
                        # Create pinned CPU buffer for efficient async DMA transfer
                        cpu_buffer = torch.empty(
                            tensor.shape,
                            dtype=tensor.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        # Chunk the bulk D2H copy so the copy engine yields
                        # between trunks, letting other streams' small copies
                        # interleave instead of waiting for the whole transfer.
                        chunked_copy_(
                            cpu_buffer,
                            tensor,
                            chunk_size_bytes=cls._stash_chunk_size_bytes,
                        )
                        # Tell the caching allocator this tensor is in use on
                        # d2h_stream so the bytes are not reused by default-
                        # stream allocations until the async D2H copy completes.
                        # Without this, resize_(0) below would let the allocator
                        # immediately reuse the HBM, corrupting the in-flight
                        # copy and producing NaN after a few batches.
                        tensor.record_stream(d2h_stream)
                        orig_storage_size = tensor.untyped_storage().size()
                        stash_data.append((tensor, cpu_buffer, orig_storage_size))
                        # Register the (still-CUDA) slab's StorageImpl so the DCP
                        # stager can redirect to this CPU buffer if it captured a
                        # view of weights_dev before the ``tensor.data = cpu_buffer``
                        # swap below. Keyed on the StorageImpl (shared by all
                        # aliasing siblings, survives the resize_(0) free) so one
                        # entry covers every captured shard. Dropped on restore.
                        # Pin the orphaned StorageImpl in the registry value
                        # itself (not just via _orig_storages) so its _cdata key
                        # cannot be reused by the allocator for a NEW storage
                        # while a redirect entry is live (ABA hazard -> would copy
                        # from a stale CPU buffer). resize_(0) below frees the
                        # DataPtr, so this pins the 0-byte shell, not HBM.
                        slab_storage = tensor.untyped_storage()
                        # Record D2H completion so a checkpoint-time redirect can
                        # wait for the async copy that fills cpu_buffer before
                        # reading it (else it could read a not-yet-complete buffer).
                        d2h_done = torch.cuda.Event()
                        d2h_done.record()
                        cls._staged_storage_buffers[slab_storage._cdata] = (
                            cpu_buffer,
                            tensor.storage_offset(),
                            slab_storage,
                            d2h_done,
                        )

                    # Save original CUDA storage refs and offsets before
                    # freeing, so restore can re-point tensors (and views
                    # sharing the same storage) back to the original storage.
                    for tensor, _, _ in stash_data:
                        _orig_storages.append(tensor.untyped_storage())
                        _orig_storage_offsets.append(tensor.storage_offset())

                    # Two-pass: free HBM storage only after all copies are
                    # enqueued.  Tensors may share the same underlying storage
                    # (e.g. views into an all-to-all output buffer), so freeing
                    # one tensor's storage before copying another would
                    # invalidate the shared data.
                    seen_storage_ptrs: set = set()
                    for tensor, _, _ in stash_data:
                        storage_ptr = tensor.untyped_storage().data_ptr()
                        if storage_ptr not in seen_storage_ptrs:
                            seen_storage_ptrs.add(storage_ptr)
                            tensor.untyped_storage().resize_(0)

                for tensor, cpu_buffer, _ in stash_data:
                    tensor.data = cpu_buffer

        def restore(
            _grad: Optional[torch.Tensor] = None,
            sync_event: Optional[torch.cuda.Event] = None,
        ) -> None:
            """Restore tensors from CPU to HBM asynchronously."""
            restore_events.clear()
            h2d_stream = cls.h2d_stream()

            size_text = _tensor_size_text([hbm_ref for hbm_ref, _, _ in stash_data])
            cls._log_ems_once(
                f"ems_restore_summary_{label.replace(' ', '_')}",
                {
                    "label": label,
                    "num_tensors": str(len(stash_data)),
                    "size": size_text,
                },
            )
            try:
                with record_function(f"restore {label} on device ({size_text})"):
                    for (
                        (hbm_ref, _cpu_buf, orig_storage_size),
                        orig_stor,
                        orig_offset,
                    ) in zip(stash_data, _orig_storages, _orig_storage_offsets):
                        # Re-allocate the original CUDA storage to its full size
                        # (which may be larger than this tensor if storage is
                        # shared).  For shared storage, the first resize allocates
                        # the full buffer; subsequent resizes for sibling views
                        # are no-ops since the storage is already large enough.
                        cur_size = orig_stor.size()
                        if cur_size < orig_storage_size:
                            orig_stor.resize_(orig_storage_size)
                        ref = torch.tensor(
                            [], dtype=hbm_ref.dtype, device=torch.cuda.current_device()
                        )
                        ref.set_(
                            orig_stor, orig_offset, hbm_ref.shape, hbm_ref.stride()
                        )
                        hbm_ref.data = ref

                # Ensure h2d_stream waits for the caller's stream before any
                # copies.  When called from a background thread the default
                # stream may differ from the main thread's, so all GPU work
                # must happen on h2d_stream.
                if sync_event is not None:
                    h2d_stream.wait_event(sync_event)
                else:
                    h2d_stream.wait_stream(torch.cuda.current_stream())

                with record_function(f"restore {label} from host ({size_text})"):
                    for hbm_ref, cpu_buf, _ in stash_data:
                        # Use a temporary tensor viewing the same storage to
                        # bypass autograd.  copy_() on hbm_ref directly would
                        # increment its version counter, causing "modified by
                        # inplace operation" errors during backward.
                        tmp = torch.tensor(
                            [], dtype=hbm_ref.dtype, device=hbm_ref.device
                        )
                        tmp.set_(
                            hbm_ref.untyped_storage(),
                            storage_offset=hbm_ref.storage_offset(),
                            size=hbm_ref.shape,
                            stride=hbm_ref.stride(),
                        )
                        with torch.cuda.stream(h2d_stream):
                            # Chunk the bulk H2D restore so the copy engine yields
                            # between trunks (see chunked_copy_).
                            chunked_copy_(
                                tmp,
                                cpu_buf,
                                chunk_size_bytes=cls._stash_chunk_size_bytes,
                            )
                        # Tell the caching allocator this storage is in use on
                        # h2d_stream so the bytes cannot be reused by main-stream
                        # allocations before the H2D copy completes.
                        hbm_ref.record_stream(h2d_stream)

                    # Only need to record the last event — the stream is ordered.
                    restore_event = torch.cuda.Event()
                    restore_event.record(h2d_stream)
                    restore_events.append(restore_event)
            finally:
                # Always drop this cycle's DCP-staging redirect entries (even if
                # the H2D loop raised) so the registry can't leak across cycles.
                # weights_dev is re-pointed to valid CUDA storage above, so future
                # captured views are GPU-valid and need no redirect.
                cls._drop_staging_redirect(_orig_storages)

        def await_restore(_grad: Optional[torch.Tensor] = None) -> None:
            """Pause current stream awaiting restore completion."""
            for restore_event in restore_events:
                torch.cuda.current_stream().wait_event(restore_event)

        if not delay:
            execute_stash()

            def _noop_stash(_grad: Optional[torch.Tensor] = None) -> None:
                pass

            return await_restore, restore, _noop_stash

        return await_restore, restore, execute_stash

    @classmethod
    def stash_embedding_weights(
        cls,
        lookup: nn.Module,
        caller: str = "",
    ) -> Optional[
        Tuple[
            Callable[[Optional[torch.Tensor]], None],
            Callable[[Optional[torch.Tensor]], None],
            Callable[[Optional[torch.Tensor]], None],
        ]
    ]:
        """
        Stash embedding weights from HBM to CPU asynchronously.

        Starts an async copy of embedding weights from GPU (HBM) to CPU
        (pinned memory) using the shared D2H stream. HBM is freed immediately
        using ``record_stream`` to let the caching allocator handle stream
        ordering — no background thread or CPU-blocking synchronize needed.

        Args:
            lookup: A lookup module (e.g., GroupedPooledEmbeddingsLookup) containing
                embedding modules with weights to stash.

        Returns:
            A tuple of three callback functions, or None if no tensors qualify
            for stashing (e.g., all tables have ``stash_weights=False``):
            - await_restore: Pauses current stream awaiting restore completion
            - restore: Retrieves stashed data from CPU back to HBM asynchronously
            - execute_stash: Performs the actual D2H copy and frees HBM.  Takes
              an optional ``torch.Tensor`` argument so it can be registered as
              a backward hook.  In immediate mode this is a no-op (stash
              already happened).  In delayed mode (``set_delay_stash(True)``),
              this callable is auto-appended to the pending list and should be
              triggered later via ``execute_pending_stashes()``.

        Usage:
            >>> await_restore, restore, execute_stash = MemoryStashingManager.stash_embedding_weights(lookup)
            >>> # In immediate mode: stash already happened, execute_stash is no-op
            >>> # In delayed mode: call execute_pending_stashes() later to trigger

        Note:
            - Uses pinned CPU memory for efficient async transfers
            - Uses d2h_stream for stash (GPU->CPU) and h2d_stream for restore (CPU->GPU)
            - record_stream tells the caching allocator not to reuse memory until
              the D2H copy completes, avoiding GIL deadlocks from background threads
            - restore starts async copy, await_restore blocks GPU stream until complete
        """

        # Handle DDP wrapper - unwrap to get the actual module
        module = lookup.module if hasattr(lookup, "module") else lookup

        # Early return if module doesn't have embedding modules
        if not hasattr(module, "_emb_modules"):
            cls._log_ems_once("ems_stash_embedding_weights_skipped")
            return None

        # Collect CUDA embedding weight tensors from TBE groups marked for stashing
        tensors: List[torch.Tensor] = []
        # pyrefly: ignore[not-iterable]
        for emb_module in module._emb_modules:
            if not hasattr(emb_module, "_emb_module"):
                continue
            # Check if this TBE group is marked for stashing via per-table config.
            # If _config is a GroupedEmbeddingConfig, only stash TBEs where at
            # least one table has stash_weights=True. Otherwise (e.g., in tests
            # with mock objects), stash all TBEs.
            config = getattr(emb_module, "_config", None)
            if isinstance(config, GroupedEmbeddingConfig):
                should_stash = any(
                    getattr(t, "stash_weights", False) for t in config.embedding_tables
                )
                if not should_stash:
                    continue
            inner = emb_module._emb_module
            if not hasattr(inner, "weights_dev"):
                continue
            weights_dev = inner.weights_dev
            if weights_dev is None or not weights_dev.is_cuda:
                continue
            tensors.append(weights_dev)

        tensor_size = _tensor_size_text(tensors) if tensors else "0.00 Bytes"
        cls._log_ems_once(
            "ems_stash_embedding_weights_collected",
            {
                "caller": caller,
                "lookup": type(lookup).__name__,
                "module": type(module).__name__,
                "num_tensors": str(len(tensors)),
                "tensor_size": tensor_size,
            },
        )

        if not tensors:
            return None

        if cls._delay_stash:
            await_restore, restore, execute_stash = cls._stash_tensors(
                tensors, label="embedding", delay=True
            )

            # Wrap execute_stash to also register the restore callback,
            # which only works after stash_data is populated.
            def _deferred_stash(_grad: Optional[torch.Tensor] = None) -> None:
                execute_stash(None)
                cls._embedding_weight_restore_callbacks.append(restore)

            cls._pending_stash_callbacks.append(_deferred_stash)
            return await_restore, restore, _deferred_stash

        await_restore, restore, execute_stash = cls._stash_tensors(
            tensors, label="embedding"
        )
        cls._embedding_weight_restore_callbacks.append(restore)
        return await_restore, restore, execute_stash

    @classmethod
    def restore_emo_cache(
        cls,
        _grad: Optional[torch.Tensor] = None,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Pop and call all EMO cache restore callbacks."""
        cls._log_ems_once(
            "emo_restore_cache_callbacks",
            {"num_callbacks": str(len(cls._emo_cache_restore_callbacks))},
        )
        while cls._emo_cache_restore_callbacks:
            cls._emo_cache_restore_callbacks.pop()(None, sync_event)

    @classmethod
    def stash_emo_cache(
        cls,
        lookup: nn.Module,
        features: KeyedJaggedTensor,
    ) -> Optional[
        Tuple[
            Callable[[Optional[torch.Tensor]], None],
            Callable[[Optional[torch.Tensor]], None],
        ]
    ]:
        """
        Stash EMO cache (lxu_cache_weights) for MANAGED_CACHING TBE modules.

        For tables using ``fused_uvm_caching`` (EMO), the actual HBM consumer
        is ``lxu_cache_weights``, not ``weights_dev``. This method:

        1. Flushes dirty cache lines back to ``weights_uvm`` (the authoritative
           copy in host DDR).
        2. Frees ``lxu_cache_weights`` HBM via ``resize_(0)``.
        3. Returns restore callbacks that re-allocate the cache, reset cache
           state, and re-prefetch the current batch rows so the TBE backward
           (fused backward+optimizer) can update the correct cache slots.

        Batch_i+1 prefetch is handled separately by the pipeline via
        ``_restore_and_prefetch_embeddings`` which calls
        ``sharded_module.prefetch()`` after this restore completes.

        Args:
            lookup: A lookup module (e.g., GroupedPooledEmbeddingsLookup)
                containing TBE modules with MANAGED_CACHING tables.
            features: The KeyedJaggedTensor used in the current forward pass.
                Captured for re-prefetch at restore time.

        Returns:
            A tuple of (await_restore, restore) callbacks, or None if no
            EMO caches were stashed.
        """
        # Handle DDP wrapper
        module = lookup.module if hasattr(lookup, "module") else lookup

        if not hasattr(module, "_emb_modules"):
            return None

        feature_splits = getattr(module, "_feature_splits", None)
        if feature_splits is None:
            return None

        features_by_group = features.split(feature_splits)

        # Collect EMO TBE modules and their features
        # Each entry: (tbe_inner, group_features)
        emo_stash_data: List[Tuple[Any, KeyedJaggedTensor]] = []

        # pyre-ignore[16]: _emb_modules is dynamically set by lookup modules
        for emb_module, group_features in zip(module._emb_modules, features_by_group):
            config = getattr(emb_module, "_config", None)
            if not isinstance(config, GroupedEmbeddingConfig):
                continue

            has_emo = any(
                t.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING
                for t in config.embedding_tables
            )
            if not has_emo:
                continue

            inner = getattr(emb_module, "_emb_module", None)
            if inner is None:
                continue

            if (
                not hasattr(inner, "lxu_cache_weights")
                or inner.lxu_cache_weights.numel() == 0
            ):
                continue

            emo_stash_data.append((inner, group_features))

        if not emo_stash_data:
            return None

        with record_function(f"## stash_emo_cache ({len(emo_stash_data)} TBEs) ##"):
            orig_cache_sizes: List[int] = []
            for inner, _group_features in emo_stash_data:
                inner.flush()
                orig_cache_sizes.append(
                    inner.lxu_cache_weights.untyped_storage().size()
                )
                inner.lxu_cache_weights.untyped_storage().resize_(0)

        total_freed_mb = sum(orig_cache_sizes) / (1024**2)
        cls._log_ems_once(
            "emo_stash_cache",
            {
                "num_caches": str(len(emo_stash_data)),
                "freed_mb": f"{total_freed_mb:.2f}",
            },
        )

        def restore(
            _grad: Optional[torch.Tensor] = None,
            sync_event: Optional[torch.cuda.Event] = None,
        ) -> None:
            """Restore EMO caches and re-prefetch batch_i for backward."""
            with record_function(
                f"## restore_emo_cache ({len(emo_stash_data)} TBEs) ##"
            ):
                for (tbe, group_features), orig_cache_size in zip(
                    emo_stash_data, orig_cache_sizes
                ):
                    # Step 1: Re-allocate cache HBM
                    tbe.lxu_cache_weights.untyped_storage().resize_(orig_cache_size)

                    # Step 2: Invalidate all cache slots and reset LRU/LFU
                    tbe.reset_cache_states()

                    # Step 3: Clear stale prefetch state
                    tbe.lxu_cache_locations_list.clear()
                    if hasattr(tbe, "timesteps_prefetched"):
                        tbe.timesteps_prefetched.clear()

                    # Step 4: Reset locking counter for prefetch_pipeline
                    if (
                        hasattr(tbe, "lxu_cache_locking_counter")
                        and tbe.lxu_cache_locking_counter is not None
                        and tbe.lxu_cache_locking_counter.numel() > 0
                    ):
                        tbe.lxu_cache_locking_counter.fill_(0)

                    # Step 5: Re-prefetch batch_i rows for backward.
                    # With prefetch_pipeline=True, this locks batch_i slots.
                    tbe.prefetch(
                        indices=group_features.values(),
                        offsets=group_features.offsets(),
                    )

                    # Step 6: Consume batch_i's prefetch state — pop cache
                    # locations for backward and the corresponding timestep
                    # so they stay in sync with lxu_cache_locations_list.
                    if tbe.lxu_cache_locations_list:
                        tbe.lxu_cache_locations = tbe.lxu_cache_locations_list.pop(0)
                    if (
                        hasattr(tbe, "timesteps_prefetched")
                        and tbe.timesteps_prefetched
                    ):
                        tbe.timesteps_prefetched.pop(0)

        def await_restore(_grad: Optional[torch.Tensor] = None) -> None:
            """No-op: re-prefetch runs synchronously on the current stream."""
            pass

        cls._emo_cache_restore_callbacks.append(restore)
        return await_restore, restore

    @classmethod
    def stash_optimizer_state(
        cls,
        optimizer: torch.optim.Optimizer,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> Tuple[
        Callable[[Optional[torch.Tensor]], None],
        Callable[[Optional[torch.Tensor]], None],
    ]:
        """
        Stash optimizer state tensors from HBM to CPU asynchronously.

        Starts an async copy of optimizer state tensors from GPU (HBM) to CPU
        (pinned memory) using the shared D2H stream. HBM is freed immediately
        using ``record_stream`` to let the caching allocator handle stream
        ordering — no background thread or CPU-blocking synchronize needed.
        Returns two callback functions for the restore phase.

        This method works with any optimizer that stores state tensors in
        ``optimizer.state``, including Shampoo (with nested
        ShampooKroneckerFactors), Adam, SGD with momentum, etc.

        Args:
            optimizer: A PyTorch optimizer containing state tensors to stash.

        Returns:
            A tuple of two callback functions:
            - await_restore: Pauses current stream awaiting restore completion
            - restore: Retrieves stashed data from CPU back to HBM asynchronously

        Usage:
            >>> await_restore, restore = MemoryStashingManager.stash_optimizer_state(optimizer)
            >>> # ... HBM is freed via record_stream (allocator reclaims after D2H completes) ...
            >>> # Before optimizer.step():
            >>> restore()  # Start async restore from CPU to HBM
            >>> await_restore()  # Wait for restore to complete before using state
            >>> optimizer.step()  # Now optimizer state is available

        Note:
            - Uses pinned CPU memory for efficient async transfers
            - Uses d2h_stream for stash (GPU->CPU) and h2d_stream for restore (CPU->GPU)
            - record_stream tells the caching allocator not to reuse memory until
              the D2H copy completes, avoiding GIL deadlocks from background threads
            - restore starts async copy, await_restore blocks GPU stream until complete
            - Only CUDA tensors are stashed; CPU tensors and non-tensor state are left unchanged
            - Supports nested structures like Shampoo's ShampooKroneckerFactors
        """
        # Collect all CUDA tensors from optimizer state
        tensors: List[torch.Tensor] = []
        for _param, state_dict in optimizer.state.items():
            if not isinstance(state_dict, dict):
                continue
            for _state_key, state_value in state_dict.items():
                tensors.extend(_collect_cuda_tensors_from_value(state_value))

        cls._log_ems_once(
            "ems_stash_optimizer_state_collected",
            {
                "optimizer": type(optimizer).__name__,
                "num_tensors": str(len(tensors)),
            },
        )

        await_restore, restore, _execute_stash = cls._stash_tensors(
            tensors, label="optimizer state", sync_event=sync_event
        )
        cls._optimizer_state_restore_callbacks.append(restore)
        return await_restore, restore

    @classmethod
    def stash_optimizer_state_threaded(
        cls,
        optimizer: torch.optim.Optimizer,
    ) -> "Future[Tuple[Callable[[Optional[torch.Tensor]], None], Callable[[Optional[torch.Tensor]], None]]]":
        """
        Submit ``stash_optimizer_state`` to a background thread.

        Records a CUDA event on the caller's current stream before submitting,
        so the D2H stream in the background thread correctly waits for all
        prior GPU work (e.g. ``optimizer.step()``) to complete.

        The returned ``Future`` resolves to the same
        ``(await_restore, restore)`` tuple as ``stash_optimizer_state``.
        Call ``.result()`` to block until the stash is done and retrieve the
        callbacks.
        """
        sync_event = torch.cuda.Event()
        sync_event.record(torch.cuda.current_stream())
        device = torch.cuda.current_device()

        def _work() -> Tuple[
            Callable[[Optional[torch.Tensor]], None],
            Callable[[Optional[torch.Tensor]], None],
        ]:
            torch.cuda.set_device(device)
            return cls.stash_optimizer_state(optimizer, sync_event=sync_event)

        return cls.thread_submit(_work)

    @classmethod
    def restore_optimizer_state_threaded(cls) -> "Future[None]":
        """
        Submit ``restore_optimizer_state`` to a background thread.

        Records a CUDA event on the caller's current stream before submitting,
        so the H2D stream in the background thread correctly waits for all
        prior GPU work to complete before starting the CPU-to-GPU copies.

        The returned ``Future`` resolves to ``None`` once all restore
        callbacks have been executed.  The caller must still invoke
        ``await_restore`` on the main thread afterwards to make the default
        stream wait for the H2D copies to finish.
        """
        cls._log_ems_once("ems_restore_optimizer_state_threaded")
        sync_event = torch.cuda.Event()
        sync_event.record(torch.cuda.current_stream())
        device = torch.cuda.current_device()

        def _work() -> None:
            torch.cuda.set_device(device)
            cls.restore_optimizer_state(None, sync_event)

        return cls.thread_submit(_work)

    @classmethod
    def restore_embedding_weights_threaded(cls) -> "Future[None]":
        """
        Submit ``restore_embedding_weights`` to a background thread.

        Records a CUDA event on the caller's current stream before submitting,
        so the H2D stream in the background thread correctly waits for all
        prior GPU work to complete before starting the CPU-to-GPU copies.

        The returned ``Future`` resolves to ``None`` once all restore
        callbacks have been executed.  The caller must still invoke
        ``await_restore`` on the main thread afterwards to make the default
        stream wait for the H2D copies to finish.
        """
        cls._log_ems_once("ems_restore_embedding_weights_threaded")
        sync_event = torch.cuda.Event()
        sync_event.record(torch.cuda.current_stream())
        device = torch.cuda.current_device()

        def _work() -> None:
            torch.cuda.set_device(device)
            cls.restore_embedding_weights(sync_event=sync_event)

        return cls.thread_submit(_work)


def _collect_cuda_tensors_from_value(
    value: Any,
    min_size_bytes: int = 1024 * 1024,  # 1MB default threshold
) -> List[torch.Tensor]:
    """
    Recursively collect CUDA tensors from a value.

    Handles nested structures like:
    - Direct tensors
    - Sharded optimizer state (ShardedTensor / DTensor), unwrapped to their
      local shard tensor(s)
    - Tuples/lists of tensors (e.g., Shampoo's factor_matrices)
    - Objects with tensor attributes (e.g., ShampooKroneckerFactors)
    - Nested dicts

    Args:
        value: The value to extract tensors from.
        min_size_bytes: Minimum tensor size in bytes to include. Tensors smaller
            than this threshold are skipped to avoid overhead of stashing small
            tensors. Default is 1MB.

    Returns:
        List of CUDA tensors found in the value that meet the size threshold.
    """
    tensors: List[torch.Tensor] = []

    if isinstance(value, ShardedTensor):
        for shard in value.local_shards():
            tensors.extend(
                _collect_cuda_tensors_from_value(shard.tensor, min_size_bytes)
            )
    elif isinstance(value, DTensor):
        tensors.extend(
            _collect_cuda_tensors_from_value(value.to_local(), min_size_bytes)
        )
    elif isinstance(value, torch.Tensor):
        if value.is_cuda and value.numel() > 0:
            tensor_size_bytes = value.numel() * value.element_size()
            if tensor_size_bytes >= min_size_bytes:
                tensors.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(_collect_cuda_tensors_from_value(item, min_size_bytes))
    elif isinstance(value, dict):
        for v in value.values():
            tensors.extend(_collect_cuda_tensors_from_value(v, min_size_bytes))
    elif hasattr(value, "__dataclass_fields__"):
        # Handle dataclass-like objects (e.g., ShampooKroneckerFactors)
        for field_name in value.__dataclass_fields__:
            field_value = getattr(value, field_name, None)
            if field_value is not None:
                tensors.extend(
                    _collect_cuda_tensors_from_value(field_value, min_size_bytes)
                )
    elif hasattr(value, "__dict__"):
        # Handle generic objects with attributes
        for attr_value in value.__dict__.values():
            tensors.extend(_collect_cuda_tensors_from_value(attr_value, min_size_bytes))

    return tensors
