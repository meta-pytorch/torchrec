#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.logger import capped_logger, one_time_rank0_logger
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def _tensor_size_text(tensor: Union[List[torch.Tensor], torch.Tensor]) -> str:
    """Return a human-readable size string (MB or GB) for a tensor."""
    if isinstance(tensor, list):
        size_mb = sum(t.numel() * t.element_size() for t in tensor) / (1024**2)
    else:
        size_mb = tensor.numel() * tensor.element_size() / (1024**2)
    if size_mb < 200:
        return f"{size_mb:.2f} MB"
    return f"{size_mb / 1024:.2f} GB"


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
        one_time_rank0_logger.info("MemoryStashingManager: streams initialized")

    @classmethod
    def reset(cls) -> None:
        """Release all resources."""
        logger.info("MemoryStashingManager: resetting all resources")
        cls._host_to_device_stream = None
        cls._device_to_host_stream = None
        cls._embedding_weight_restore_callbacks.clear()
        cls._optimizer_state_restore_callbacks.clear()
        cls._emo_cache_restore_callbacks.clear()
        if cls._stash_executor is not None:
            cls._stash_executor.shutdown(wait=False)
            cls._stash_executor = None

    @classmethod
    def restore_embedding_weights(
        cls,
        _grad: Optional[torch.Tensor] = None,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Pop and call all embedding weight restore callbacks in reverse order."""
        one_time_rank0_logger.info(
            f"restore_embedding_weights: invoking {len(cls._embedding_weight_restore_callbacks)} callbacks"
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
        one_time_rank0_logger.info(
            f"restore_optimizer_state: invoking {len(cls._optimizer_state_restore_callbacks)} callbacks"
        )
        while cls._optimizer_state_restore_callbacks:
            cls._optimizer_state_restore_callbacks.pop()(None, sync_event)

    @classmethod
    def _stash_tensors(
        cls,
        tensors: List[torch.Tensor],
        label: str = "tensor",
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> Tuple[
        Callable[[Optional[torch.Tensor]], None],
        Callable[[Optional[torch.Tensor]], None],
    ]:
        """
        Stash a list of CUDA tensors from HBM to CPU asynchronously.

        Core implementation shared by ``stash_embedding_weights`` and
        ``stash_optimizer_state``.  Starts an async copy of each tensor from
        GPU (HBM) to pinned CPU memory using the shared D2H stream, then frees
        HBM via ``record_stream`` + ``resize_(0)``.  Returns two callback
        functions for the restore phase.

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

        Returns:
            A tuple of two callback functions:
            - await_restore: Pauses current stream awaiting restore completion
            - restore: Retrieves stashed data from CPU back to HBM asynchronously
        """
        # (hbm_tensor ref, cpu_buffer, original_storage_size)
        stash_data: List[Tuple[torch.Tensor, torch.Tensor, int]] = []

        # Restore events populated by ``restore`` and consumed by ``await_restore``
        restore_events: List[torch.cuda.Event] = []

        d2h_stream = cls.d2h_stream()

        # Ensure all operations on the caller's stream complete before we
        # start copying — prevents reading while still being written.
        # When sync_event is provided (background thread), use it instead
        # of current_stream() which would return the wrong stream.
        if sync_event is not None:
            d2h_stream.wait_event(sync_event)
        else:
            d2h_stream.wait_stream(torch.cuda.current_stream())

        size_text = _tensor_size_text(tensors)
        capped_logger.info(f"stash {label}: {len(tensors)} tensors, total {size_text}")

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
                    cpu_buffer.copy_(tensor, non_blocking=True)
                    orig_storage_size = tensor.untyped_storage().size()
                    stash_data.append((tensor, cpu_buffer, orig_storage_size))

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

        def restore(
            _grad: Optional[torch.Tensor] = None,
            sync_event: Optional[torch.cuda.Event] = None,
        ) -> None:
            """Restore tensors from CPU to HBM asynchronously."""
            restore_events.clear()
            h2d_stream = cls.h2d_stream()

            size_text = _tensor_size_text([cpu_buf for _, cpu_buf, _ in stash_data])
            capped_logger.info(
                f"restore {label}: {len(stash_data)} tensors, total {size_text}"
            )
            with record_function(f"restore {label} on device ({size_text})"):
                for hbm_ref, _cpu_buf, orig_storage_size in stash_data:
                    # Re-allocate HBM storage to the original size (which
                    # may be larger than this tensor if storage is shared).
                    # For shared storage, the first resize allocates the
                    # full buffer; subsequent resizes for sibling views are
                    # no-ops since the storage is already large enough.
                    cur_size = hbm_ref.untyped_storage().size()
                    if cur_size < orig_storage_size:
                        hbm_ref.untyped_storage().resize_(orig_storage_size)

            # Ensure h2d_stream waits for the caller's stream before any
            # allocations.  When called from a background thread the default
            # stream may differ from the main thread's, so all GPU work
            # (including resize_ allocations) must happen on h2d_stream.
            if sync_event is not None:
                h2d_stream.wait_event(sync_event)
            else:
                h2d_stream.wait_stream(torch.cuda.current_stream())

            with record_function(f"restore {label} from host ({size_text})"):
                for hbm_ref, cpu_buf, _orig_storage_size in stash_data:

                    # Copy data back using a temporary tensor to bypass
                    # autograd.  copy_() on the original tensor would
                    # increment its version counter, causing "modified by
                    # inplace operation" errors during backward.  A fresh
                    # tensor viewing the same storage has its own version
                    # counter, avoiding the issue.
                    #
                    # Use hbm_ref.storage_offset() to place data at the
                    # correct position within potentially shared storage.
                    tmp = torch.tensor([], dtype=hbm_ref.dtype, device=hbm_ref.device)
                    tmp.set_(
                        hbm_ref.untyped_storage(),
                        storage_offset=hbm_ref.storage_offset(),
                        size=hbm_ref.shape,
                        stride=hbm_ref.stride(),
                    )
                    with torch.cuda.stream(h2d_stream):
                        tmp.copy_(cpu_buf, non_blocking=True)

                # only need to record the last event
                restore_event = torch.cuda.Event()
                restore_event.record(h2d_stream)
                restore_events.append(restore_event)

        def await_restore(_grad: Optional[torch.Tensor] = None) -> None:
            """Pause current stream awaiting restore completion."""
            for restore_event in restore_events:
                torch.cuda.current_stream().wait_event(restore_event)

        return await_restore, restore

    @classmethod
    def stash_embedding_weights(
        cls,
        lookup: nn.Module,
    ) -> Optional[
        Tuple[
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
        Returns two callback functions for the restore phase.

        Args:
            lookup: A lookup module (e.g., GroupedPooledEmbeddingsLookup) containing
                embedding modules with weights to stash.

        Returns:
            A tuple of two callback functions, or None if no tensors were stashed:
            - await_restore: Pauses current stream awaiting restore completion
            - restore: Retrieves stashed data from CPU back to HBM asynchronously

        Usage:
            >>> await_restore, restore = MemoryStashingManager.stash_embedding_weights(lookup)
            >>> # ... HBM is freed via record_stream (allocator reclaims after D2H completes) ...
            >>> # Before backward (or next forward):
            >>> restore()  # Start async restore from CPU to HBM
            >>> await_restore()  # Wait for restore to complete before using weights

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
            capped_logger.info(
                "stash_embedding_weights: no _emb_modules found, skipping"
            )
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

        capped_logger.info(
            f"stash_embedding_weights: lookup={type(lookup).__name__}, "
            f"module={type(module).__name__}, "
            f"collected {len(tensors)} weight tensors"
        )

        if not tensors:
            return None

        await_restore, restore = cls._stash_tensors(tensors, label="embedding")
        cls._embedding_weight_restore_callbacks.append(restore)
        return await_restore, restore

    @classmethod
    def restore_emo_cache(
        cls,
        _grad: Optional[torch.Tensor] = None,
        sync_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """Pop and call all EMO cache restore callbacks."""
        one_time_rank0_logger.info(
            f"restore_emo_cache: invoking {len(cls._emo_cache_restore_callbacks)} callbacks"
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
        capped_logger.info(
            f"stash_emo_cache: freed {len(emo_stash_data)} EMO caches, "
            f"total {total_freed_mb:.2f} MB"
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

        one_time_rank0_logger.info(
            f"stash_optimizer_state: optimizer={type(optimizer).__name__}, "
            f"collected {len(tensors)} state tensors"
        )

        await_restore, restore = cls._stash_tensors(
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
        one_time_rank0_logger.info(
            "restore_optimizer_state_threaded: submitting to background thread"
        )
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
        one_time_rank0_logger.info(
            "restore_embedding_weights_threaded: submitting to background thread"
        )
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

    if isinstance(value, torch.Tensor):
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
