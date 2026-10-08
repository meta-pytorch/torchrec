#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, unique
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.logger import one_time_rank0_logger
from torchrec.distributed.memory_stashing import MemoryStashingManager
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.train_pipeline.backward_injection import (
    FirstGradTensorFinder,
    InjectionSite,
    InjectionTargetType,
    OutputDistTensorFinder,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    CPUEmbeddingTrainPipelineContext,
    In,
    Out,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    CPUEmbeddingPipelinedForward,
    PipelinedForward,
    PrefetchPipelinedForward,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline.types import PipelineState
from torchrec.distributed.train_pipeline.utils import (
    _batch_tensor_size,
    _init_inplace_copy_worker as _bind_worker_to_device,
    _override_input_dist_forwards,
    _rewrite_model,
    _start_data_dist,
    _to_device,
    _wait_for_batch,
    FutureDeque,
    prefetch_embeddings,
    use_context_for_postprocs,
)
from torchrec.distributed.types import LazyNoWait, ShardingType
from torchrec.sparse.jagged_tensor import KeyedTensor

try:
    from torchrec.distributed.logging_handlers import (
        log_ems_config,
        log_inplace_copy_batch,
    )
    from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.distributed.train_pipeline.experimental_pipelines.import_failure.logging_handlers"
    )

    DATA_TYPE_NUM_BITS: dict = {}  # type: ignore[no-redef]

    def log_ems_config(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def log_inplace_copy_batch(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass


logger: logging.Logger = logging.getLogger(__name__)


class TrainEvalHybridPipelineBase(TrainPipelineSparseDist[In, Out]):
    """
    A hybrid pipeline that supports both training and evaluation modes in a single
    pipelined execution flow.

    This class extends `TrainPipelineSparseDist` to enable seamless switching between
    training and evaluation within the same pipeline. It is particularly useful for
    scenarios where you need to interleave training and evaluation batches without
    the overhead of switching between separate pipelines.

    Key Features:
        - Supports both training and evaluation modes via the model's training flag.
        - Conditionally executes backward pass and optimizer step only during training.
        - Maintains the same pipelining benefits (overlapping data transfer, sparse
          data distribution, and forward pass) for both modes.
        - Uses model.training state (set by set_eval_mode()) to determine whether
          to run backward/optimizer for each batch.

    Pipeline Stages (inherited from TrainPipelineSparseDist):
        - Stage 3: Forward/Backward/Optimizer (current batch)
        - Stage 2: Sparse data distribution (next batch)
        - Stage 1: Device transfer (batch i+2)

    Eval Draining:
        When the eval data iterator is exhausted, the pipeline enters a draining
        state where it continues processing remaining eval batches already in the
        pipeline queue without fetching new data. Once the queue is empty, the
        pipeline resets its state and raises StopIteration to signal eval completion.
    """

    _draining_eval: bool = False

    def copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        """
        Retrieve batch from dataloader and move to device.

        Returns:
            Tuple of (batch, context).
        """
        context = self._create_context()
        with record_function(f"## copy_batch_to_gpu {context.index} ##"):
            # pyrefly: ignore [bad-argument-type]
            with self._stream_context(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)

                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch, context

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if self._state == PipelineState.UNKNOWN:
            return super()._next_batch(dataloader_iter)
        return self._next_batch_on_cpu

    def _prepare_pipeline_step(self, dataloader_iter: Iterator[In]) -> None:
        """
        Reset flags, pre-fetch next batch, and check for empty pipeline.

        Handles draining mode transitions for eval: when the eval data iterator
        is exhausted, enters draining mode to process remaining queued batches.
        When the pipeline is fully drained, resets state and raises StopIteration.

        Raises:
            StopIteration: When pipeline is empty (all batches processed or
                eval draining complete).
        """
        # Only reset exhaustion/draining flags when in training mode.
        # During eval, we want to preserve these flags so the pipeline
        # can drain remaining eval batches after the eval iter exhausts.
        if self._model.training:
            self._dataloader_exhausted = False
            self._draining_eval = False

        # Pre-fetch next batch unless we're draining (no more data to fetch)
        if not self._draining_eval:
            self._next_batch_on_cpu = TrainPipelineSparseDist._next_batch(
                self, dataloader_iter
            )
            if self._next_batch_on_cpu is None and not self._model.training:
                # Eval data exhausted. Enter draining mode to process
                # remaining eval batches already in the pipeline queue.
                self._draining_eval = True

        # Pipeline is empty — either all batches processed or draining complete
        if not self.batches:
            if self._draining_eval:
                # Pipeline fully drained after eval. Reset state so
                # fill_pipeline re-initializes on the next progress() call.
                self._state = PipelineState.UNKNOWN
                self._draining_eval = False
            raise StopIteration

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        Execute one step of the pipelined train/eval loop.

        This method processes one batch through the full pipeline while overlapping
        operations for subsequent batches. It conditionally executes backward pass
        and optimizer step based on the model's training mode.

        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
            - batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt
                          (expecting input_dist completed)
            - batches[1]: next batch, for input_dist (expecting copied to device)
            - batches[2]: i+2 batch, for copy_batch_to_gpu
                          (expecting non-exhausted dataloader iter)

        Args:
            dataloader_iter: Iterator yielding input batches from the dataloader.

        Returns:
            Out: The output from the forward pass of the current batch (batches[0]).

        Raises:
            StopIteration: When all batches have been processed (pipeline is empty),
                or when eval draining completes (all remaining eval batches processed).
        """

        self._state = PipelineState.UNKNOWN
        # Attach the model just in case the user forgets to call it, especially when the user
        # pauses the pipeline.progress and detaches the model for other purposes.
        if not self._model_attached:
            self.attach(self._model)

        # Fill the pipeline is only needed for the beginning when the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)
        self._state = PipelineState.IDLE

        self._prepare_pipeline_step(dataloader_iter)

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        is_curr_training = self._model.training

        # Zero gradients only when model is in training mode
        if is_curr_training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # Wait for batches[0] being available on device, this should always be completed since
        # the input_dist of batches[0] has been invoked in previous iter. TODO: fact check
        self._wait_for_batch()

        if self._clear_data_dist_inputs:
            self.clear_sparse_data_dist_inputs(self.contexts[0])

        # Start sparse data distribution for the next batch (overlapped with current forward)
        if len(self.batches) >= 2:
            # Invoke splits all_to_all comms (first part of input_dist)
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # Batch i+2: load data and copy to GPU (skip when draining — no more data)
        if not self._draining_eval:
            self.enqueue_batch(dataloader_iter)

        # Forward pass for current batch
        if self.batches[0] is None:
            # Pipeline drained: batch was None from exhausted dataloader.
            # No input_dist was staged, so forward would fail. Stop here.
            self.dequeue_batch()
            raise StopIteration

        if is_curr_training:
            with record_function(f"## forward {self.contexts[0].index} ##"):
                self._state = PipelineState.CALL_FWD
                losses, output = self._model_fwd(self.batches[0])
        else:
            with record_function(f"## eval {self.contexts[0].index} ##"):
                with torch.no_grad():
                    self._state = PipelineState.CALL_FWD
                    losses, output = self._model_fwd(self.batches[0])

        # Complete sparse data distribution for the next batch
        if len(self.batches) >= 2:
            # Invoke data (values, lengths, etc.) all_to_all comms (second part of input_dist)
            self.wait_sparse_data_dist(self.contexts[1])

        # Execute backward and optimizer step only when model is in training mode
        if is_curr_training:
            # Backward pass
            self._state = PipelineState.CALL_BWD
            self._backward(losses)

            # Sync embeddings if configured (for distributed model parallel)
            self.sync_embeddings(
                self._model,
                self._dmp_collection_sync_interval_batches,
                self.contexts[0],
            )

            # Optimizer step (weight update)
            with record_function(f"## optimizer {self.contexts[0].index} ##"):
                self._optimizer.step()

        # Remove processed batch from the pipeline
        self.dequeue_batch()
        return output


class EvalPipelineCPUSparse(TrainPipelineSparseDist[In, Out]):
    """
    3-stage pipelined eval pipeline for CPU embeddings + GPU dense forward.

    Designed for DATA_PARALLEL sharding where all embedding tables are on CPU
    and dense model runs on GPU. The 3 stages overlap CPU and GPU work:

    - **Stage 1** (CPU): ``_sparse_forward`` — merged input_dist + compute +
      output_dist. For DP sharding there are no cross-rank comms, so this runs
      entirely on CPU.
    - **Stage 2** (memcpy stream): ``copy_data_to_gpu`` — async DMA of pinned
      embedding outputs + dense features to GPU.
    - **Stage 3** (GPU default stream): ``_dense_forward`` — model forward
      where ``CPUEmbeddingPipelinedForward`` returns pre-copied GPU embeddings.

    Pipeline overlap (steady state)::

        progress() returning output for batch N:
          Stage 3: dense_forward(batch N)       ← GPU (async kernel launch)
          Stage 1: sparse_forward(batch N+1)    ← CPU (overlaps with GPU above)
          Stage 2: copy_to_gpu(batch N+1)       ← memcpy stream (async DMA)

    Pre-allocated pinned memory buffers are used for embedding outputs to enable
    truly async CPU→GPU DMA transfers via ``non_blocking=True``.

    Args:
        model: The model to pipeline.
        optimizer: The optimizer (unused for eval, but required by base class).
        device: The GPU device for dense forward.
        apply_jit: Whether to apply ``torch.jit.script`` to non-pipelined modules.
    """

    # pyrefly: ignore [bad-override]
    _pipelined_forward_type = CPUEmbeddingPipelinedForward

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        pipeline_postproc: bool = False,
        enable_inplace_copy_batch: bool = False,
        multi_thread: bool = False,
        pipeline_depth: int = 2,
        free_features_storage_early: bool = False,
        clear_data_dist_inputs: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            context_type=CPUEmbeddingTrainPipelineContext,
            pipeline_postproc=pipeline_postproc,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
            clear_data_dist_inputs=clear_data_dist_inputs,
        )
        assert pipeline_depth in (1, 2), "pipeline_depth must be 1 or 2"
        self._pipeline_depth = pipeline_depth
        # Pre-allocated pinned memory buffers keyed by module FQN
        self._pinned_buffers: Dict[str, torch.Tensor] = {}

        # Override parent types: batches are never None in this pipeline
        # pyrefly: ignore [bad-override]
        self.batches: Deque[In] = deque()
        # pyrefly: ignore [bad-override]
        self.contexts: Deque[CPUEmbeddingTrainPipelineContext] = deque()

        # Optional thread pool for running Stage 1 (sparse forward) in background
        self._multi_thread = multi_thread
        self._sparse_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1) if multi_thread else None
        )
        self._sparse_future: Optional[Future[In]] = None

    # pyrefly: ignore[bad-override]
    def _pipeline_model(
        self,
        batch: In,
        context: TrainPipelineContext,
        pipelined_forward: Type[
            CPUEmbeddingPipelinedForward
        ] = CPUEmbeddingPipelinedForward,
    ) -> None:
        """
        Model surgery + bootstrap input_dist.

        Overrides the base ``_pipeline_model`` to avoid calling
        ``self.start_sparse_data_dist`` (which depends on ``self.contexts[0]``
        existing). Instead, runs input_dist inline and waits for splits.
        """
        (
            self._pipelined_modules,
            self._model,
            self._original_forwards,
            self._pipelined_postprocs,
            _,
        ) = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=None,  # No cross-rank comms for DP CPU eval
            default_stream=None,
            batch=batch,
            apply_jit=self._apply_jit,
            # pyrefly: ignore[bad-argument-type]
            pipelined_forward=pipelined_forward,
            pipeline_postproc=self._pipeline_postproc,
        )
        # Bootstrap: run input_dist once to initialize KJT dist state
        _start_data_dist(self._pipelined_modules, batch, context)
        # Wait for splits → tensor awaitables
        for names, awaitable in context.fused_splits_awaitables:
            for name, request in zip(names, awaitable.wait()):
                context.input_dist_tensors_requests[name] = request
        context.fused_splits_awaitables.clear()
        # Override KJT dist forwards with fused versions (requires initialized dists)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

    def _get_or_alloc_pinned(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Get or allocate a pre-allocated pinned memory buffer matching tensor."""
        buf = self._pinned_buffers.get(name)
        if buf is None or buf.shape != tensor.shape or buf.dtype != tensor.dtype:
            self._pinned_buffers[name] = torch.empty_like(tensor, pin_memory=True)
        return self._pinned_buffers[name]

    def _embedding_lookup_and_pin(
        self, context: CPUEmbeddingTrainPipelineContext
    ) -> None:
        """
        Run ``compute_and_output_dist`` for each pipelined module on CPU,
        then copy embedding values into pre-allocated pinned memory buffers.

        After this, ``context.embedding_a2a_requests`` holds ``LazyNoWait``
        wrappers around pinned-memory ``KeyedTensor`` / ``Dict[str, JaggedTensor]``.
        """
        with record_function(f"## embedding_lookup_and_pin {context.index} on CPU ##"):
            for module in self._pipelined_modules:
                # pyrefly: ignore[missing-attribute]
                name = module.forward.name
                module_ctx = context.module_contexts[name]
                kjt = context.input_dist_tensors_requests.pop(name).wait()
                output = module.compute_and_output_dist(module_ctx, kjt)
                embeddings = output.wait()  # CPU result

                # Copy values into pre-allocated pinned buffer for async DMA
                if isinstance(embeddings, KeyedTensor):
                    values = embeddings.values()
                    pinned = self._get_or_alloc_pinned(name, values)
                    pinned.copy_(values)
                    embeddings._values = pinned

                context.embedding_a2a_requests[name] = LazyNoWait(embeddings)

    def _sparse_forward(
        self, batch: In, context: CPUEmbeddingTrainPipelineContext
    ) -> In:
        """
        Stage 1: merged input_dist + compute + output_dist on CPU.

        For DP sharding there are no cross-rank comms — splits awaitables
        resolve immediately. Results are stored in pinned memory for Stage 2.
        """
        with record_function(f"## sparse_forward {context.index} on CPU ##"):
            # input_dist
            with use_context_for_postprocs(self._pipelined_postprocs, context):
                _start_data_dist(self._pipelined_modules, batch, context)
            # wait splits → tensor awaitables
            for names, awaitable in context.fused_splits_awaitables:
                for name, request in zip(names, awaitable.wait()):
                    context.input_dist_tensors_requests[name] = request
            context.fused_splits_awaitables.clear()
            # compute + output_dist + pin
            self._embedding_lookup_and_pin(context)

            if self._pipeline_depth == 1:
                return self.copy_data_to_gpu(batch, context)
            else:
                return batch

    def _inplace_copy_to_gpu(
        self, batch: In, context: CPUEmbeddingTrainPipelineContext
    ) -> In:
        """
        Async inplace copy of pinned embeddings + dense features to GPU.

        GPU buffers are pre-allocated on the default stream, then filled via
        ``non_blocking`` inplace copies on the memcpy stream. This avoids
        allocating GPU memory on the memcpy stream and enables truly async DMA.
        """
        with record_function(f"## inplace_copy_to_gpu {context.index} ##"):
            # Pre-allocate GPU embedding buffers on the default stream
            gpu_embedding_buffers: Dict[str, Tuple[Any, Optional[torch.Tensor]]] = {}
            for name in context.embedding_a2a_requests:
                awaitable = context.embedding_a2a_requests[name]
                assert isinstance(awaitable, LazyNoWait)
                embeddings = awaitable._obj
                if isinstance(embeddings, KeyedTensor):
                    gpu_values = torch.empty_like(
                        embeddings.values(), device=self._device
                    )
                    gpu_embedding_buffers[name] = (embeddings, gpu_values)
                else:
                    gpu_embedding_buffers[name] = (embeddings, None)

            # Copy batch dense features to GPU (pre-alloc + inplace copy on memcpy stream)
            # pyrefly: ignore[bad-assignment]
            batch = batch.to(
                self._device,
                non_blocking=True,
                # pyrefly: ignore[unexpected-keyword]
                data_copy_stream=self._memcpy_stream,
                # pyrefly: ignore[unexpected-keyword]
                dense_only=True,
            )

            # Inplace copy pinned embeddings to pre-allocated GPU buffers on memcpy stream
            # pyrefly: ignore[bad-argument-type]
            with self._stream_context(self._memcpy_stream):
                if self._memcpy_stream:
                    self._memcpy_stream.wait_stream(
                        torch.get_device_module(self._device).current_stream()
                    )
                for name, (embeddings, gpu_values) in gpu_embedding_buffers.items():
                    if gpu_values is not None:
                        gpu_values.copy_(embeddings.values(), non_blocking=True)
                        context.gpu_embedding_outputs[name] = KeyedTensor(
                            keys=embeddings.keys(),
                            length_per_key=embeddings.length_per_key(),
                            values=gpu_values,
                        )
                    else:
                        context.gpu_embedding_outputs[name] = embeddings.to(
                            device=self._device, non_blocking=True
                        )
        return batch

    def _copy_to_gpu(self, batch: In, context: CPUEmbeddingTrainPipelineContext) -> In:
        """
        Stage 2: async copy pinned embeddings + dense features to GPU.

        Uses the memcpy stream with ``non_blocking=True`` for async DMA.
        Dense features are copied via ``batch.to(..., dense_only=True)``.
        Embedding outputs (pinned) are copied via ``KeyedTensor.to(device)``.
        """
        with record_function(f"## copy_to_gpu {context.index} ##"):
            # pyrefly: ignore[bad-argument-type]
            with self._stream_context(self._memcpy_stream):
                # Copy dense features to GPU (non_blocking)
                # pyrefly: ignore[bad-assignment]
                batch = batch.to(
                    self._device,
                    non_blocking=True,
                    # pyrefly: ignore[unexpected-keyword]
                    data_copy_stream=self._memcpy_stream,
                    # pyrefly: ignore[unexpected-keyword]
                    dense_only=True,
                )
                # Copy pinned embedding outputs to GPU (async DMA)
                for name in list(context.embedding_a2a_requests.keys()):
                    awaitable = context.embedding_a2a_requests[name]
                    # pyrefly: ignore[missing-attribute]
                    embeddings = awaitable.wait()
                    gpu_embeddings = embeddings.to(
                        device=self._device, non_blocking=True
                    )
                    context.gpu_embedding_outputs[name] = gpu_embeddings
        return batch

    def copy_data_to_gpu(
        self, batch: In, context: CPUEmbeddingTrainPipelineContext
    ) -> In:
        """Stage 2: copy pinned embeddings + dense features to GPU."""
        if self._enable_inplace_copy_batch:
            return self._inplace_copy_to_gpu(batch, context)
        else:
            return self._copy_to_gpu(batch, context)

    # pyrefly: ignore[bad-override]
    def enqueue_batch(self, dataloader_iter: Iterator[In]) -> bool:
        """
        Load a batch from the dataloader, create context, and append to
        pipeline deques. The batch stays on CPU at this point.
        """
        batch = self._next_batch(dataloader_iter)
        if batch is None:
            return False
        context = self._create_context()
        assert isinstance(context, CPUEmbeddingTrainPipelineContext)
        context.dense_gpu_device = self._device.type

        self.batches.append(batch)
        self.contexts.append(context)
        self._batch_count += 1
        return True

    def _dense_forward(
        self, batch: In, context: CPUEmbeddingTrainPipelineContext
    ) -> Out:
        """
        Stage 3: GPU dense forward.

        Waits for the memcpy stream (Stage 2) to complete, then runs model
        forward. ``CPUEmbeddingPipelinedForward`` intercepts embedding module
        calls and returns the pre-copied GPU embeddings from ``context``.
        """
        with record_function(f"## dense_forward {context.index} ##"):
            if self._memcpy_stream:
                torch.get_device_module(self._device).current_stream().wait_stream(
                    self._memcpy_stream
                )
            self._set_module_context(context)
            with torch.no_grad():
                _, output = self._model_fwd(batch)
        return output

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        Cold start: load first batch, run model surgery, then Stage 1 + Stage 2.

        After this, ``batches[0]`` is ready for Stage 3 (dense forward).
        """
        # pipeline is already filled with max capacity (2)
        if len(self.batches) >= 2:
            return

        # executes last batch in pipeline, when there is only one batch in the pipeline
        # TODO: this _execute_all_batches doesn't really work here D43546239. it will
        # just throw an exception at copy_to_gpu when the dataloader is exhausted
        if self.batches and self._execute_all_batches:
            return

        # batch i, data (batch) and context
        if not self.enqueue_batch(dataloader_iter):
            logger.info("fill_pipeline: failed to load batch i")
            return

        batch = self.batches[0]
        context = self.contexts[0]

        # First-time model surgery + bootstrap input_dist
        self._pipeline_model(batch, context, self._pipelined_forward_type)
        # context now has input_dist_tensors_requests populated

        # Stage 1 (remaining): compute + output_dist + pin
        self._embedding_lookup_and_pin(context)
        self.batches[0] = self.copy_data_to_gpu(batch, context)

        if self._pipeline_depth == 2:
            # batch i+1
            if not self.enqueue_batch(dataloader_iter):
                logger.info("fill_pipeline: failed to load batch i+1")
                return

            self.batches[1] = self._sparse_forward(self.batches[1], self.contexts[1])

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        3-stage pipelined progress with CPU/GPU overlap.

        Each call processes one batch through Stage 3 (GPU dense forward)
        while preparing the next batch through Stages 1+2 (CPU sparse forward
        + async copy). This overlaps CPU sparse work with GPU dense compute.
        """
        self.fill_pipeline(dataloader_iter)
        if not self.batches:
            one_time_rank0_logger.info(
                f"training stopped at {self._batch_count} batches"
            )
            raise StopIteration

        # Wait for any pending sparse forward from previous iteration
        if self._sparse_future is not None:
            self.batches[-1] = self._sparse_future.result()
            self._sparse_future = None

        # Stage 3: dense forward for current batch (GPU, async kernel launch)
        output = self._dense_forward(
            self.batches[0],
            self.contexts[0],
        )

        # Stage 2: copy to GPU (must be on main thread for stream ops)
        if len(self.batches) > 1 and self._pipeline_depth == 2:
            if self._memcpy_stream:
                self._memcpy_stream.wait_stream(
                    torch.get_device_module(self._device).current_stream()
                )
            self.batches[1] = self.copy_data_to_gpu(self.batches[1], self.contexts[1])

        # Dequeue current batch
        self.dequeue_batch()

        # While GPU computes, load and prepare next batch:
        if self.enqueue_batch(dataloader_iter):
            batch = self.batches[-1]
            context = self.contexts[-1]
            if self._sparse_executor is not None:
                # Stage 1: submit sparse forward to background thread
                self._sparse_future = self._sparse_executor.submit(
                    self._sparse_forward, batch, context
                )
            else:
                # Stage 1: sparse forward on CPU (overlaps with GPU dense forward above)
                self.batches[-1] = self._sparse_forward(batch, context)

        return output


class TrainPipelineSparseDistT(TrainPipelineSparseDist[In, Out]):
    """
    Extends TrainPipelineSparseDist by running the inplace H2D copy (_to_device) in a
    background thread so the CPU is not blocked while submitting non-blocking copy
    operations to the memcpy stream.

    The background result is resolved lazily before the batch is actually consumed
    (in fill_pipeline before _init_pipelined_modules, and in progress before
    start_sparse_data_dist).

    All other pipeline behaviour is identical to TrainPipelineSparseDist.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
        clear_data_dist_inputs: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward=False,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
            clear_data_dist_inputs=clear_data_dist_inputs,
        )
        self._copy_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self.batches: Deque[Optional[In]] = cast(Deque[Optional[In]], FutureDeque())

    def copy_batch_to_gpu(
        self, dataloader_iter: Iterator[In]
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        context = self._create_context()
        with record_function(f"## copy_batch_to_gpu {context.index} ##"):
            batch = self._next_batch(dataloader_iter)
            if batch is not None:

                def _copy_work() -> In:
                    # pyrefly: ignore [bad-argument-type]
                    with self._stream_context(self._memcpy_stream):
                        return _to_device(batch, self._device, True)

                future_batch = self._copy_executor.submit(_copy_work)
                return cast(In, future_batch), context
            elif not self._execute_all_batches:
                logger.info(
                    "copy_batch_to_gpu: raising StopIteration for None Batch (execute_all_batches=False)"
                )
                raise StopIteration
            else:
                logger.info(
                    "copy_batch_to_gpu: returning None batch (execute_all_batches=True)"
                )
            return batch, context

    def inplace_copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        context = self._create_context()
        with record_function(f"## inplace_copy_batch_to_gpu {context.index} ##"):
            batch = self._next_batch(dataloader_iter)
            if batch is not None:
                if not self._inplace_copy_batch_size_logged:
                    self._inplace_copy_batch_size_logged = True
                    size = _batch_tensor_size(batch)

                    log_inplace_copy_batch(size)
                future_batch = self._copy_executor.submit(
                    _to_device,
                    batch,
                    self._device,
                    True,
                    self._memcpy_stream,
                )
                # Return the CPU batch as placeholder; _resolve_copy_future
                # will replace it in self.batches before consumption.
                return cast(In, future_batch), context
            elif not self._execute_all_batches:
                logger.info(
                    "inplace_copy_batch_to_gpu: raising StopIteration for None Batch (execute_all_batches=False)"
                )
                raise StopIteration
            else:
                logger.info(
                    "inplace_copy_batch_to_gpu: returning None batch (execute_all_batches=True)"
                )
            return batch, context


class TrainPipelineSparseDistBwdOpt(TrainPipelineSparseDist[In, Out]):
    """
    Extends TrainPipelineSparseDist by moving the optimizer step into the backward
    pass via OutputDistTensorFinder backward hook injection. This overlaps the optimizer
    computation with backward all-to-all communication, improving training throughput.

    The explicit optimizer.step() in progress() is removed; instead, the optimizer
    fires during backward when the output distribution tensor's gradient is computed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        site_fqn: str,
        sharding_type: ShardingType = ShardingType.TABLE_WISE,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
        clear_data_dist_inputs: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
            clear_data_dist_inputs=clear_data_dist_inputs,
        )
        self._output_dist_site = InjectionSite(
            fqn=site_fqn,
            tensor_finder=OutputDistTensorFinder(sharding_type=sharding_type),
            target_type=InjectionTargetType.ACTIVATION,
        )

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        super()._pipeline_model(batch, context, pipelined_forward)

        def work(pipeline: Any) -> None:
            with record_function(f"## optimizer {pipeline.contexts[0].index} ##"):
                pipeline._optimizer.step()

        self.register_backward_hook(self._output_dist_site, work)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
            batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt (expecting input_dist)
            batches[1]: next batch, for input_dist (expecting copied to device)
            batches[2]: i+2 batch, for copy_batch_to_gpu (expecting non-exhausted dataloader iter)
        """

        self._state = PipelineState.IDLE
        # attach the model just in case the user forgets to call it, especially when the user
        # pauses the pipeline.progress and detach the model for other purpose.
        if not self._model_attached:
            self.attach(self._model)

        # fill the pipeline is only needed for the beginning when the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)

        # here is the expected stop after exhausting all batches
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # wait for batches[0] being available on device, this should always be completed since
        # the input_dist of batches[0] has be invoked in previous iter. TODO: fact check
        self._wait_for_batch()

        if self._clear_data_dist_inputs:
            self.clear_sparse_data_dist_inputs(self.contexts[0])

        if len(self.batches) >= 2:
            # invoke splits all_to_all comms (first part of input_dist)
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        if not self._enqueue_batch_after_forward:
            # batch i+2: load data and copy to gpu, the dataload iter will first exhaust here
            self.enqueue_batch(dataloader_iter)

        # forward
        with record_function(f"## forward {self.contexts[0].index} ##"):
            self._state = PipelineState.CALL_FWD
            losses, output = self._model_fwd(self.batches[0])

        if self._enqueue_batch_after_forward:
            # batch i+2: load data and copy to gpu, the dataload iter will first exhaust here.
            # Start this step after the forward of batch i, so that the H2D copy doesn't compete
            # for pcie bandwidth with embedding lookup from UVM/UVM_CACHING.
            self.enqueue_batch(dataloader_iter)

        if len(self.batches) >= 2:
            # invoke data (values, lengths, etc.) all_to_all comms (second part of input_dist)
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            self._state = PipelineState.CALL_BWD
            self._backward(losses)

            self.sync_embeddings(
                self._model,
                self._dmp_collection_sync_interval_batches,
                self.contexts[0],
            )

        self.dequeue_batch()
        return output


class TrainPipelineSparseDistOptStash(TrainPipelineSparseDist[In, Out]):
    """
    Extends TrainPipelineSparseDist by stashing optimizer state to CPU after
    optimizer.step() and restoring it via backward hook injection before the
    next optimizer.step().

    This frees HBM occupied by optimizer state (e.g. Shampoo's Kronecker
    factors) between optimizer steps, making it available for forward/backward
    computation. The restore is triggered during the backward pass at the
    specified OutputDistTensorFinder site, overlapping the CPU->GPU transfer with backward
    all-to-all communication.

    Timeline per iteration:
        forward -> backward [ restore_optimizer_state at output dist site ] ->
        optimizer.step() -> stash_optimizer_state
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        site_fqn: str,
        sharding_type: ShardingType = ShardingType.TABLE_WISE,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
        clear_data_dist_inputs: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
            clear_data_dist_inputs=clear_data_dist_inputs,
        )
        self._output_dist_site = InjectionSite(
            fqn=site_fqn,
            tensor_finder=OutputDistTensorFinder(sharding_type=sharding_type),
        )
        # Set up shared CUDA streams for memory stashing
        MemoryStashingManager.set_streams(
            self._memcpy_stream,  # pyrefly: ignore[bad-argument-type]
            torch.cuda.Stream(device=device),
        )
        self._await_restore: Callable[..., None] = lambda: None
        self._stash_future: Optional[
            Future[Tuple[Callable[..., None], Callable[..., None]]]
        ] = None
        self._restore_future: Optional[Future[None]] = None

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        super()._pipeline_model(batch, context, pipelined_forward)

        def work(_pipeline: Any) -> None:
            with record_function("## restore_optimizer_state ##"):
                self._restore_future = (
                    MemoryStashingManager.restore_optimizer_state_threaded()
                )

        self.register_backward_hook(self._output_dist_site, work)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._state = PipelineState.IDLE
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(dataloader_iter)

        if not self.batches:
            raise StopIteration

        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        self._wait_for_batch()

        if self._clear_data_dist_inputs:
            self.clear_sparse_data_dist_inputs(self.contexts[0])

        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        if not self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        # forward
        with record_function(f"## forward {self.contexts[0].index} ##"):
            self._state = PipelineState.CALL_FWD
            losses, output = self._model_fwd(self.batches[0])

        if self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # Wait for the previous iteration's background stash to complete
            # before backward, because the backward hook calls
            # restore_optimizer_state which does resize_(storage_size) on the
            # same tensors that stash does resize_(0) on.
            if self._stash_future is not None:
                self._await_restore, _ = self._stash_future.result()
                self._stash_future = None

            # backward (restore_optimizer_state fires via hook)
            self._state = PipelineState.CALL_BWD
            self._backward(losses)

            self.sync_embeddings(
                self._model,
                self._dmp_collection_sync_interval_batches,
                self.contexts[0],
            )

            # optimizer step, then stash state back to CPU
            with record_function(f"## optimizer {self.contexts[0].index} ##"):
                if self._restore_future is not None:
                    self._restore_future.result()
                    self._restore_future = None
                self._await_restore()
                self._optimizer.step()

            with record_function("## stash_optimizer_state ##"):
                self._stash_future = (
                    MemoryStashingManager.stash_optimizer_state_threaded(
                        self._optimizer
                    )
                )

        self.dequeue_batch()
        return output


class TrainPipelineSparseDistEmbStash(TrainPipelineSparseDist[In, Out]):
    """
    Extends TrainPipelineSparseDist by restoring stashed embedding weights
    during backward via an InjectionSite backward hook at the specified module
    (e.g., over-arch).

    The stashing itself is done inside the sharded embedding modules
    (embeddingbag.py / embedding.py) immediately after the lookup forward.
    This pipeline registers a restore hook at the injection site so that
    weights are restored before backward reaches the sparse modules.

    Timeline per iteration:
        forward [stash happens inside lookup] -> backward
        [restore_embedding_weights at injection site] -> optimizer.step()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        site_fqn: Union[str, InjectionSite],
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
        delay_stash: bool = False,
        stash_site_fqn: Optional[str] = None,
        clear_data_dist_inputs: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
            clear_data_dist_inputs=clear_data_dist_inputs,
        )
        if isinstance(site_fqn, str):
            self._injection_site = InjectionSite(
                fqn=site_fqn,
                tensor_finder=FirstGradTensorFinder(),
                target_type=InjectionTargetType.PARAM_GRAD,
            )
        else:
            self._injection_site = site_fqn
        self._delay_stash = delay_stash
        self._stash_site_fqn = stash_site_fqn
        self._stash_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        MemoryStashingManager.set_streams(
            self._memcpy_stream,  # pyrefly: ignore[bad-argument-type]
            torch.cuda.Stream(device=device),
        )
        if delay_stash:
            MemoryStashingManager.set_delay_stash(True)

        self._log_ems_config()

    def _log_ems_config(self) -> None:
        """Emit an ems_config event with table-level stash info."""
        try:
            stash_table_names: list[str] = []
            total_stash_bytes_unsharded: int = 0
            base_model = self._model
            if isinstance(base_model, DistributedModelParallel):
                base_model = base_model.module
            for module in base_model.modules():
                embedding_configs = getattr(module, "embedding_configs", None)
                if embedding_configs is not None:
                    for ec in embedding_configs:
                        if getattr(ec, "stash_weights", False):
                            stash_table_names.append(ec.name)
                            bits = DATA_TYPE_NUM_BITS.get(ec.data_type)
                            if bits is not None:
                                total_stash_bytes_unsharded += (
                                    ec.num_embeddings * ec.embedding_dim * bits // 8
                                )
                            else:
                                logger.warning(
                                    f"Skipping stash bytes calculation for {ec.name}: "
                                    f"unknown data_type {ec.data_type}"
                                )
            log_ems_config(
                metadata={
                    "num_stash_tables": str(len(stash_table_names)),
                    "stash_table_names": ",".join(stash_table_names),
                    "total_stash_bytes_unsharded": str(total_stash_bytes_unsharded),
                    "pipeline_type": type(self).__name__,
                },
            )
        except Exception:
            logger.debug("EMS config logging failed", exc_info=True)

    def _try_hook_stash(self) -> None:
        """Register a forward pre-hook on ``stash_site_fqn`` to execute
        pending stashes right before that module's forward pass.

        This follows the same pattern used by
        ``TrainPipelineCustomizedOrderSparseDist`` to hook SDD steps into
        specific module forwards.  The hook frees HBM (D2H copy +
        ``resize_(0)``) just before peak-memory modules run, avoiding PCIe
        contention with ``start_sparse_data_dist``'s H2D copy.
        """
        if self._stash_site_fqn is None or self._stash_hook_handle is not None:
            return

        model = self._model
        if isinstance(model, DistributedModelParallel):
            model = model.module

        target = model.get_submodule(self._stash_site_fqn)

        def _execute_stash_hook(module: torch.nn.Module, args: Any) -> None:
            with record_function("## execute_pending_stashes ##"):
                MemoryStashingManager.execute_pending_stashes()

        self._stash_hook_handle = target.register_forward_pre_hook(_execute_stash_hook)
        logger.info(
            f"MemoryStashingManager: hooked execute_pending_stashes "
            f"as forward pre-hook on {self._stash_site_fqn}"
        )

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        super()._pipeline_model(batch, context, pipelined_forward)

        if self._delay_stash and self._stash_site_fqn is None:
            # No forward hook site specified — fall back to executing
            # pending stashes via backward hook at the injection site.
            def execute_work(_pipeline: Any) -> None:
                with record_function("## execute_pending_stashes ##"):
                    MemoryStashingManager.execute_pending_stashes()

            self.register_backward_hook(self._injection_site, execute_work)

        # Hook stash execution into the target module's forward if configured.
        if self._delay_stash:
            self._try_hook_stash()

        def work(_pipeline: Any) -> None:
            with record_function("## restore_embedding_weights ##"):
                MemoryStashingManager.restore_embedding_weights()

        self.register_backward_hook(self._injection_site, work)


class TrainPipelinePrefetchEMS(TrainPipelineSparseDistEmbStash[In, Out]):
    """
    Extends TrainPipelineSparseDistEmbStash with unified EMO + EMS support.

    For tables using ``fused_uvm_caching`` (EMO), the standard embedding
    stash is a no-op because ``weights_dev`` is empty — the actual HBM
    consumer is ``lxu_cache_weights``. This pipeline enables time-sharing
    HBM between the embedding cache and dense compute by:

    1. After forward: flushing dirty cache lines to ``weights_uvm``, then
       freeing ``lxu_cache_weights`` HBM via ``resize_(0)`` on the D2H
       stream (overlaps with dense forward).
    2. During dense forward/backward: cache HBM is available for activations.
    3. Before backward: restoring the cache and re-prefetching on the main
       stream.

    Timeline per iteration::

        default stream:   forward ── restore+prefetch ── backward ── opt
        d2h stream:            stash (flush+free) ──┘
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        site_fqn: Union[str, InjectionSite],
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            site_fqn=site_fqn,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=context_type,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
        )

    def _restore_and_prefetch(self) -> None:
        """
        Restore stashed embedding weights and EMO caches (batch_i), then
        prefetch batch_i+1 for the next forward pass.

        Runs on the main stream before backward.
        """
        # Step 1: Restore stashed weights and EMO caches (re-prefetch batch_i)
        MemoryStashingManager.restore_embedding_weights()
        MemoryStashingManager.restore_emo_cache()

        # Step 2: Prefetch batch_i+1 for next forward
        if len(self.batches) < 2:
            return

        context_next = self.contexts[1]
        for sharded_module in self._pipelined_modules:
            forward = sharded_module.forward
            # pyre-ignore[16]: _name is set by PipelinedForward
            name = forward._name
            if name not in context_next.input_dist_tensors_requests:
                continue

            # Peek at the awaitable — materialize if needed, but don't pop.
            # Store back as LazyNoWait so PipelinedForward.__call__ can
            # still consume it in the next iteration.
            request = context_next.input_dist_tensors_requests[name]
            if isinstance(request, LazyNoWait):
                data = request._obj
            else:
                data = request.wait()
                context_next.input_dist_tensors_requests[name] = LazyNoWait(data)

            module_context = context_next.module_contexts.get(name)
            sharded_module.prefetch(ctx=module_context, dist_input=data)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._state = PipelineState.IDLE
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(dataloader_iter)

        if not self.batches:
            raise StopIteration

        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        self._wait_for_batch()

        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        if not self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        # forward — stash_emo_cache fires inside ShardedEBC.compute()
        # on the d2h stream (overlaps with dense forward)
        with record_function(f"## forward {self.contexts[0].index} ##"):
            self._state = PipelineState.CALL_FWD
            losses, output = self._model_fwd(self.batches[0])

        if self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        # Restore + prefetch on main stream before backward.
        with record_function("## restore_and_prefetch ##"):
            self._restore_and_prefetch()

        if self._model.training:
            self._state = PipelineState.CALL_BWD
            self._backward(losses)

            self.sync_embeddings(
                self._model,
                self._dmp_collection_sync_interval_batches,
                self.contexts[0],
            )

            with record_function(f"## optimizer {self.contexts[0].index} ##"):
                self._optimizer.step()

        self.dequeue_batch()
        return output


@unique
class EvalPipelineStage(Enum):
    """
    Relocatable stages of ``EvalPipelinePrefetchSparseDist``.

    Each names work done for a batch *behind* the one being evaluated, so it can
    be moved off its default slot in ``progress()`` and onto a module forward
    hook. See ``EvalPipelinePrefetchSparseDist.hook_stage``.
    """

    # H2D copy of the newest batch (i+3)
    ENQUEUE_BATCH = "enqueue_batch"
    # splits all2all for batch i+2
    START_SPARSE_DATA_DIST = "start_sparse_data_dist"
    # tensor all2all for batch i+1
    WAIT_SPARSE_DATA_DIST = "wait_sparse_data_dist"
    # embedding cache prefetch for batch i+1
    PREFETCH = "prefetch"

    @classmethod
    def from_name(cls, name: str) -> "EvalPipelineStage":
        """Resolve a stage from its config string, listing the valid names on typo."""
        try:
            return cls(name)
        except ValueError:
            valid = ", ".join(stage.value for stage in cls)
            raise ValueError(
                f"Unknown pipeline stage {name!r}. Valid stages: {valid}"
            ) from None


@dataclass
class _StageHookSpec:
    """Where a relocated stage should run."""

    fqn: str
    prepend: bool = False


# Within one step a stage may only run once its producer has. START_SPARSE_DATA_DIST
# maps to ENQUEUE_BATCH rather than to a consumer of its all2all: the batch it
# distributes has to exist first, while the a2a it starts is awaited by the *next*
# step's WAIT_SPARSE_DATA_DIST, not this one's.
_STAGE_PREREQUISITE: Dict[EvalPipelineStage, EvalPipelineStage] = {
    EvalPipelineStage.START_SPARSE_DATA_DIST: EvalPipelineStage.ENQUEUE_BATCH,
    EvalPipelineStage.PREFETCH: EvalPipelineStage.WAIT_SPARSE_DATA_DIST,
}

# Install order for hooks landing on the same module. PyTorch fires forward hooks
# in registration order, so without a canonical order the caller's dict ordering
# (i.e. yaml key order) would silently decide whether the dependency chain holds.
_STAGE_ORDER: List[EvalPipelineStage] = [
    EvalPipelineStage.ENQUEUE_BATCH,
    EvalPipelineStage.START_SPARSE_DATA_DIST,
    EvalPipelineStage.WAIT_SPARSE_DATA_DIST,
    EvalPipelineStage.PREFETCH,
]


class EvalPipelinePrefetchSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    Eval-only 4-stage pipeline with embedding cache prefetching.

    This is the eval counterpart of ``PrefetchTrainPipelineSparseDist``: it keeps
    the dedicated prefetch stage that hides UVM/UVM_CACHING lookup latency, but
    drops everything that only exists to serve training. It is **not** usable for training — no
    gradients are ever produced.

    Pipeline stages (3 batches in flight)::

        batch i     forward                       default stream
        batch i+1   wait input_dist + prefetch    data_dist / prefetch streams
        batch i+2   H2D device transfer           memcpy stream
        batch i+2   start input_dist (splits a2a) data_dist stream

    Eval-specific tuning versus the training pipeline:

    - The forward runs under ``torch.no_grad()``, so no autograd graph is built.
    - Eval can be far more aggressive than training about *what* it hides behind
      the forward. Weights are frozen, so batch i+1's sparse work is ordered
      against nothing: both input_dist and the cache prefetch are pushed into the
      window of batch i's forward.
    - That is also why the forward for batch i is issued **before** the
      CPU-blocking ``wait_sparse_data_dist``/``prefetch`` for batch i+1. Those two
      calls stall the CPU thread on a collective, and the GPU only runs what the
      CPU already enqueued. Training absorbs the stall with the previous step's
      backward and optimizer kernels; eval has no such backlog, so the forward has
      to be launched first or the GPU idles through the stall.
    - ``execute_all_batches`` is forced to ``True``: eval metrics are wrong if the
      tail batches left in the pipeline are dropped when the dataloader ends.
    - 2D DMP sync is disabled — there are no weight updates to sync.

    Entry invariant of ``progress()``: ``batches[0]`` has been prefetched and
    ``batches[1]``'s input_dist has been started, either by ``fill_pipeline`` or
    by the tail of the previous iteration.

    Relocating stages:
        The default slots above put every sparse stage *after* the dense forward
        returns, which is the right choice when the forward is one long opaque
        block. When it is not -- e.g. a dense arch with a natural gap partway
        through -- ``hook_stage`` moves an individual stage onto a module's
        forward hook so it is issued mid-forward instead::

            pipeline.hook_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST, "dense")
            pipeline.hook_stage(EvalPipelineStage.PREFETCH, "dense")

        This mirrors the ``SDDStepsOrder`` mechanism in
        ``TrainPipelineCustomizedOrderSparseDist``, narrowed to the stages this
        pipeline owns. Stages left unhooked keep their default slots, so mixing
        the two can invert the dependency order -- see ``hook_stage``.

    NOTE: prefetching keeps two batches' embedding lookups alive at once, on top
    of 3 in-flight batches. Peak HBM is higher than ``EvalPipelineSparseDist``;
    size the eval batch accordingly.

    Args:
        stage_hooks (Optional[Dict[str, str]]): stage name -> module FQN, applied
            as if ``hook_stage`` had been called for each entry. Exists so a
            config/yaml can place stages without building the pipeline in Python;
            equivalent to the programmatic API but without the ``prepend`` knob.
            Stage names are the ``EvalPipelineStage`` values, e.g.
            ``{"wait_sparse_data_dist": "dense", "prefetch": "dense"}``.
            Default: None.
        async_prefetch (bool): dispatch the prefetch stage on a background thread
            so its CPU work runs concurrently with the dense forward rather than
            serializing in front of it. See the section below. Default: False.

    Overlapping the prefetch:
        Relocating the prefetch with ``hook_stage`` moves *where* in the forward
        the stage runs, but it still runs on the forward's own thread, so the
        forward stops for its whole duration. That is only worth paying when the
        step is GPU-bound and the stall is covered by kernels already enqueued.
        It is not covered when the step is CPU-dispatch-bound -- the queue drains
        within microseconds of the thread stopping, and the stall shows up one
        for one in step time.

        ``async_prefetch=True`` hands the stage to a single background worker
        instead, so the forward keeps issuing kernels while the prefetch
        dispatches::

            main thread:     ... dense_forward(i) ...............
            prefetch worker:      wait+populate cache for (i+1)

        The stage is a good candidate for this because it is dispatch, not GPU
        work, and because it issues no collectives -- see ``_submit_prefetch``
        for the invariants that make it safe. The join happens at the top of the
        next ``progress()``, i.e. before the forward that reads the cache lines.

        The ceiling is the stage's own wall time, and the GIL is what decides how
        much of it is actually recovered: both threads are part Python, and only
        the C++ op bodies release the GIL. Measure on end-to-end throughput at a
        matched example count, not on trace step time.

    Example:
        >>> model.eval()
        >>> pipeline = EvalPipelinePrefetchSparseDist(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device=torch.device("cuda:0"),
        ... )
        >>> while True:
        ...     try:
        ...         output = pipeline.progress(dataloader_iter)
        ...     except StopIteration:
        ...         break
    """

    # The PipelinedForward class that is used in _rewrite_model
    # pyrefly: ignore [bad-override]
    _pipelined_forward_type = PrefetchPipelinedForward

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        apply_jit: bool = False,
        pipeline_postproc: bool = True,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        enable_inplace_copy_batch: bool = False,
        free_features_storage_early: bool = False,
        enqueue_batch_after_forward: bool = False,
        stage_hooks: Optional[Dict[str, str]] = None,
        async_prefetch: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=True,
            apply_jit=apply_jit,
            context_type=PrefetchTrainPipelineContext,
            pipeline_postproc=pipeline_postproc,
            custom_model_fwd=custom_model_fwd,
            dmp_collection_sync_interval_batches=None,
            enqueue_batch_after_forward=enqueue_batch_after_forward,
            enable_inplace_copy_batch=enable_inplace_copy_batch,
            free_features_storage_early=free_features_storage_early,
        )
        self._prefetch_stream: Optional[torch.Stream] = (
            (torch.get_device_module(device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )
        self._default_stream: Optional[torch.Stream] = (
            (torch.get_device_module(self._device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )
        stage_hooks = stage_hooks or {}

        self._async_prefetch = async_prefetch
        self._prefetch_device = device
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._start_prefetch_executor()
        self._prefetch_future: Optional[Future[None]] = None

        # Stages relocated out of their default slot in progress() and into a
        # module forward hook. See hook_stage().
        self._stage_hooks: Dict[EvalPipelineStage, _StageHookSpec] = {}
        self._stage_hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._stage_ran: Dict[EvalPipelineStage, bool] = {}
        self._stage_hook_missed_warned: bool = False
        # progress() stashes the live iterator so a hooked ENQUEUE_BATCH, which
        # fires from inside the model forward, can still pull the next batch.
        self._active_dataloader_iter: Optional[Iterator[In]] = None

        # Config-driven equivalent of calling hook_stage() for each entry, so a
        # benchmark yaml can place stages without constructing the pipeline in
        # Python. Bad stage names raise here rather than silently doing nothing.
        for stage_name, module_fqn in stage_hooks.items():
            self.hook_stage(EvalPipelineStage.from_name(stage_name), module_fqn)

    def hook_stage(
        self,
        stage: "EvalPipelineStage",
        module_fqn: str,
        prepend: bool = False,
    ) -> None:
        """
        Relocate a pipeline stage into a module's forward hook.

        By default every stage runs at a fixed point in ``progress()`` (see the
        class docstring). Hooking a stage removes it from that slot and runs it
        instead when ``module_fqn``'s forward completes, i.e. partway through the
        dense forward of batch i. That lets callers place the sparse work of
        batch i+1 wherever the dense timeline has a gap, without subclassing.

        Ordering is the caller's responsibility. The stages form a dependency
        chain per batch::

            ENQUEUE_BATCH -> START_SPARSE_DATA_DIST -> WAIT_SPARSE_DATA_DIST
                          -> PREFETCH

        A stage left unhooked keeps its default slot, so mixing hooked and
        unhooked stages can invert that order. ``_run_stage`` raises if a stage
        runs before its prerequisite has run in the same step.

        Args:
            stage: which stage to relocate.
            module_fqn: dotted path of the module to hook, relative to the
                unwrapped model (e.g. ``"sparse_arch"``, ``"over.overarch"``).
                Resolved when the pipeline is first attached; an unknown path
                raises then, not here.
            prepend: run this hook before other forward hooks on that module.

        Raises:
            RuntimeError: if called after the pipeline has already run, since
                hooks are installed during the one-time model surgery.

        Example:
            >>> pipeline.hook_stage(
            ...     EvalPipelineStage.PREFETCH, "sparse_arch"
            ... )
        """
        if self._pipelined_modules:
            raise RuntimeError(
                "hook_stage() must be called before the first progress(); "
                "stage hooks are installed during model surgery."
            )
        self._stage_hooks[stage] = _StageHookSpec(fqn=module_fqn, prepend=prepend)

    def _stage_is_hooked(self, stage: "EvalPipelineStage") -> bool:
        return stage in self._stage_hooks

    def _run_stage(self, stage: "EvalPipelineStage") -> None:
        """
        Execute one pipeline stage against the batches currently in flight.

        Reads ``self.batches``/``self.contexts`` at call time rather than taking
        them as arguments, so the same body works from the default slot in
        ``progress()`` and from a forward hook firing mid-forward.
        """
        self._check_stage_preconditions(stage)
        # Marked before the body: a stage whose guard finds too few batches in
        # flight has still had its turn, and its consumers may run this step.
        self._stage_ran[stage] = True
        self._dispatch_stage(stage)

    def _check_stage_preconditions(self, stage: "EvalPipelineStage") -> None:
        """
        Reject a hooked ``stage`` that is about to run at an illegal point.

        Only hooked stages are checked. Unhooked ones run from the fixed sequence
        in ``progress()``, which already follows the dependency chain.
        """
        if not self._stage_is_hooked(stage):
            return

        prerequisite = _STAGE_PREREQUISITE.get(stage)
        if prerequisite is not None and not self._stage_ran.get(prerequisite, False):
            raise RuntimeError(
                f"{stage.value} ran before its prerequisite {prerequisite.value} "
                "in the same step. Re-order the stage hooks: a hooked stage runs "
                "at its module's forward, which may precede an unhooked stage's "
                "default slot."
            )
        self._check_hook_site_not_too_early(stage)

    def _dispatch_stage(self, stage: "EvalPipelineStage") -> None:
        """Run ``stage``'s body, if enough batches are in flight for it to have work."""
        if stage is EvalPipelineStage.ENQUEUE_BATCH:
            dataloader_iter = self._active_dataloader_iter
            if dataloader_iter is not None:
                self.enqueue_batch(dataloader_iter)
        elif stage is EvalPipelineStage.START_SPARSE_DATA_DIST:
            if len(self.batches) >= 3:
                self.start_sparse_data_dist(self.batches[2], self.contexts[2])
        elif len(self.batches) >= 2:
            self._dispatch_next_batch_stage(stage)

    def _dispatch_next_batch_stage(self, stage: "EvalPipelineStage") -> None:
        """Run the body of a stage that acts on batch i+1's context."""
        context = cast(PrefetchTrainPipelineContext, self.contexts[1])
        if stage is EvalPipelineStage.WAIT_SPARSE_DATA_DIST:
            self.wait_sparse_data_dist(context)
        elif stage is EvalPipelineStage.PREFETCH:
            self._submit_prefetch(context)

    def _check_hook_site_not_too_early(self, stage: "EvalPipelineStage") -> None:
        """
        Reject a hook site that fires before batch i's sparse modules have run.

        ``get_submodule`` only proves an FQN exists, not that it is a sane place
        to hook. A module that runs *before* a pipelined sparse module is not:
        ``_prefetch`` would evict cache lines batch i's own lookup has yet to
        read.

        ``module_input_post_prefetch`` is the signal -- ``_prefetch`` fills one
        entry per pipelined module and each module's forward pops its own, so a
        non-empty dict means some sparse module has not run yet.
        """
        if stage is not EvalPipelineStage.PREFETCH:
            return
        if not self.contexts:
            return
        context_0 = cast(PrefetchTrainPipelineContext, self.contexts[0])
        pending = sorted(context_0.module_input_post_prefetch)
        if pending:
            spec = self._stage_hooks[stage]
            raise RuntimeError(
                f"Stage {stage.value} is hooked on {spec.fqn!r}, which runs before "
                f"the sparse modules {pending} of the current batch. Running it "
                "there would disturb the embedding cache and module contexts that "
                "batch's own lookup still needs. Hook a module that runs after the "
                "sparse arch."
            )

    def _maybe_run_stage(self, stage: "EvalPipelineStage") -> None:
        """Run ``stage`` in its default slot unless it has been hooked away."""
        if not self._stage_is_hooked(stage):
            self._run_stage(stage)

    def _try_hook_stages(self) -> None:
        """
        Install the forward hooks requested via ``hook_stage``.

        Called once after model surgery. FQNs resolve against the unwrapped
        model so callers write user-facing paths like ``"sparse_arch"`` rather
        than the DMP-prefixed ones.
        """
        if not self._stage_hooks or self._stage_hook_handles:
            return

        model = self._model
        if isinstance(model, DistributedModelParallel):
            model = model.module

        pipelined = {id(module) for module in self._pipelined_modules}

        for stage in _STAGE_ORDER:
            spec = self._stage_hooks.get(stage)
            if spec is None:
                continue
            # Unknown FQN raises AttributeError here rather than silently never
            # firing.
            target = model.get_submodule(spec.fqn)

            if id(target) in pipelined:
                raise ValueError(
                    f"Cannot hook stage {stage.value} on {spec.fqn!r}: that is a "
                    "pipelined sparse module. Its forward is rewritten to return "
                    "an awaitable, and running batch i+1's sparse work from "
                    "inside it would mutate the embedding cache and module "
                    "context that batch i's own lookup is still using. Hook a "
                    "dense module that runs after it instead."
                )

            def _hook(
                module: torch.nn.Module,
                args: Any,
                output: Any,
                stage: "EvalPipelineStage" = stage,
            ) -> None:
                # Guard against a module invoked more than once per forward:
                # the stage's work is only valid once per step.
                if self._stage_ran.get(stage, False):
                    return
                with record_function(f"## stage_hook {stage.value} ##"):
                    self._run_stage(stage)

            self._stage_hook_handles.append(
                target.register_forward_hook(_hook, prepend=spec.prepend)
            )
            logger.info(f"Hooked pipeline stage {stage.value} on {spec.fqn}")

    def _remove_stage_hooks(self) -> None:
        """Uninstall the stage hooks, so ``_try_hook_stages`` can install again."""
        for handle in self._stage_hook_handles:
            handle.remove()
        self._stage_hook_handles.clear()

    def detach(self) -> torch.nn.Module:
        """
        Detach the model, taking the stage hooks off with the model surgery.

        Left installed they would keep firing on the returned model: a standalone
        forward would run batch i+1's sparse work against whatever contexts the
        pipeline still holds. They also close over ``self``, so the model would
        pin this pipeline -- and the in-flight batches ``detach`` deliberately
        keeps -- for as long as the caller holds the model.

        Any backgrounded prefetch is joined first, for the same reason: the
        caller is about to run the returned model itself. Its executor is also
        shut down so a detached pipeline does not retain a worker thread;
        ``attach`` starts a fresh one when asynchronous prefetch is enabled.

        Returns the original model.
        """
        self._join_prefetch()
        executor = self._prefetch_executor
        if executor is not None:
            executor.shutdown(wait=True)
            self._prefetch_executor = None
        self._remove_stage_hooks()
        return super().detach()

    def attach(
        self, model: Optional[torch.nn.Module] = None, sparse_dist: bool = True
    ) -> None:
        """
        Re-attach the model and reinstall the stage hooks ``detach`` removed.

        ``fill_pipeline`` is the usual install site, but it returns early while
        batches are still in flight -- exactly the state ``detach`` preserves so
        ``progress`` can resume -- so the hooks are reinstalled here instead.
        """
        super().attach(model, sparse_dist)
        self._start_prefetch_executor()
        # An empty context deque means super() deferred model surgery to the next
        # fill_pipeline, which installs the hooks itself once the FQNs resolve.
        if self._pipelined_modules:
            self._try_hook_stages()

    def _start_prefetch_executor(self) -> None:
        """Start the async prefetch worker when requested and not already live."""
        if not self._async_prefetch or self._prefetch_executor is not None:
            return
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="eval_prefetch",
            initializer=_bind_worker_to_device,
            initargs=(self._prefetch_device,),
        )

    def _warn_on_missed_stage_hooks(self) -> None:
        """
        Warn once if a hooked stage never fired during a step.

        A stage silently skipped means its module was not reached in the
        forward, which usually points at a wrong FQN and would otherwise show up
        only as a much later correctness or hang symptom.
        """
        if self._stage_hook_missed_warned:
            return
        missed = [
            stage.value
            for stage in self._stage_hooks
            if not self._stage_ran.get(stage, False)
        ]
        if missed:
            self._stage_hook_missed_warned = True
            logger.warning(
                f"Stage hooks {missed} did not fire this step; their modules were "
                "not reached during forward. Check the FQNs passed to hook_stage()."
            )

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        Cold start: prime 2 batches so ``progress`` meets its entry invariant.

        ``batches[0]`` gets model surgery, a completed input_dist and a prefetch;
        ``batches[1]`` gets its input_dist started.
        """
        # pipeline is already filled: fill_pipeline primes 2 batches, and progress
        # holds 2 at its start (it enqueues a 3rd, then dequeues one at the end).
        if len(self.batches) >= 2:
            return

        # draining the tail after the dataloader is exhausted, nothing to prime
        if self.batches:
            return

        with torch.no_grad():
            # batch i
            if not self.enqueue_batch(dataloader_iter):
                logger.info("fill_pipeline: failed to load batch i")
                return

            self._init_pipelined_modules(
                cast(In, self.batches[0]),
                self.contexts[0],
                # pyrefly: ignore [bad-argument-type]
                self._pipelined_forward_type,
            )
            # Model surgery is done, so hook FQNs can now be resolved.
            self._try_hook_stages()
            self.wait_sparse_data_dist(self.contexts[0])
            self._prefetch(cast(PrefetchTrainPipelineContext, self.contexts[0]))

            # batch i+1
            if not self.enqueue_batch(dataloader_iter):
                logger.info("fill_pipeline: failed to load batch i+1")
                return

            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """
        Runs the eval forward for batch i while prefetching batch i+1.

        Args:
            dataloader_iter: iterator producing eval batches.

        Returns:
            Model output for batch i.

        Raises:
            StopIteration: once the dataloader is exhausted and every batch left
                in the pipeline has been evaluated.
        """
        self._state = PipelineState.IDLE
        # batch i's prefetch may still be dispatching on the worker, and the
        # forward below is what consumes it.
        self._join_prefetch()
        # attach the model just in case the user forgets to call it, especially when the
        # user pauses the pipeline.progress and detaches the model for other purposes.
        if not self._model_attached:
            self.attach(self._model)

        with torch.no_grad():
            self.fill_pipeline(dataloader_iter)

            # expected stop after every batch has been evaluated
            if not self.batches:
                one_time_rank0_logger.info(
                    f"eval stopped at {self._batch_count} batches"
                )
                raise StopIteration

            self._set_module_context(self.contexts[0])

            # Hooked stages fire from inside the forward below and read these.
            self._stage_ran = {}
            self._active_dataloader_iter = dataloader_iter

            # batch i's prefetch was issued on the prefetch stream; the forward
            # below reads its output on the current stream.
            with record_function(f"## wait_for_batch {self.contexts[0].index} ##"):
                _wait_for_batch(cast(In, self.batches[0]), self._prefetch_stream)

            if not self._enqueue_batch_after_forward:
                # batch i+3: load data and copy to gpu, the dataloader iter will
                # first exhaust here
                self._maybe_run_stage(EvalPipelineStage.ENQUEUE_BATCH)

            # Forward batch i FIRST. It only depends on prefetch(i), which is
            # already done, so launching it before the CPU-blocking waits below
            # keeps the GPU busy while the CPU drains the input_dist awaitables.
            with record_function(f"## eval {self.contexts[0].index} ##"):
                self._state = PipelineState.CALL_FWD
                _, output = self._model_fwd(self.batches[0])

            if self._enqueue_batch_after_forward:
                # Deferring the H2D copy until after the forward keeps it off the
                # PCIe path used by the embedding prefetch below.
                self._maybe_run_stage(EvalPipelineStage.ENQUEUE_BATCH)

            # Free batch i's prefetched embeddings (already consumed by the
            # forward) so the caching allocator can reuse them for batch i+1.
            context_0 = cast(PrefetchTrainPipelineContext, self.contexts[0])
            context_0.module_input_post_prefetch.clear()
            context_0.module_contexts_post_prefetch.clear()

            # complete input_dist (tensor a2a) for i+1, then prefetch its
            # embeddings on the prefetch stream, overlapping the forward above.
            self._maybe_run_stage(EvalPipelineStage.WAIT_SPARSE_DATA_DIST)
            self._maybe_run_stage(EvalPipelineStage.PREFETCH)

            # start input_dist for i+2 last, so its splits a2a overlaps with
            # the next iteration's early phases
            self._maybe_run_stage(EvalPipelineStage.START_SPARSE_DATA_DIST)

            self._warn_on_missed_stage_hooks()
            self._active_dataloader_iter = None

            self.dequeue_batch()
            return output

    def _submit_prefetch(self, context: PrefetchTrainPipelineContext) -> None:
        """
        Runs the prefetch for ``context``'s batch, on a worker thread under
        ``async_prefetch`` and inline otherwise.

        The stage is CPU dispatch, not GPU work: it resolves the input_dist
        awaitables and issues the per-table cache-populate kernels. Run inline it
        stops the forward for its whole duration, and on a dispatch-bound eval
        step the GPU has nothing enqueued to cover the stall.

        Two properties make the hand-off safe, and both are load-bearing:

        - The stage issues no collectives. Every input_dist all2all is started by
          ``wait_sparse_data_dist``, which stays on the main thread;
          ``KJTAllToAllTensorsAwaitable.wait()`` only waits on the ``dist.Work``
          that call already issued. So the per-rank order of collectives on the
          process group is unchanged and the worker cannot interleave into it.
        - It touches no module the concurrent forward touches. It acts on batch
          i+1's context, and ``_check_hook_site_not_too_early`` already rejects
          hook sites that run before batch i's sparse modules, so no pipelined
          module's forward is in flight when the worker calls ``prefetch()``.

        The worker enters the prefetch stream itself -- the current stream is
        thread-local, so a fresh thread would otherwise dispatch onto the default
        stream -- and the main thread picks the GPU-side dependency back up
        through the ``_wait_for_batch(..., self._prefetch_stream)`` it already
        does at the top of the next step.
        """
        executor = self._prefetch_executor
        if executor is None:
            self._prefetch(context)
            return
        self._prefetch_future = executor.submit(self._prefetch, context)

    def _join_prefetch(self) -> None:
        """
        Blocks until a backgrounded prefetch has finished dispatching.

        Has to happen before anything reads what the prefetch produced: the
        forward that consumes the cache lines, and ``detach``, which hands the
        model back to a caller who may run it standalone. Re-raises on the main
        thread whatever the worker threw.
        """
        future = self._prefetch_future
        if future is None:
            return
        self._prefetch_future = None
        with record_function("## prefetch_join ##"):
            future.result()

    def _prefetch(self, context: PrefetchTrainPipelineContext) -> None:
        """
        Prefetches embeddings for ``context``'s batch on the prefetch stream.

        Args:
            context: prefetch context holding the completed input dist requests
                and module contexts for the batch to prefetch.
        """
        context.module_input_post_prefetch.clear()
        context.module_contexts_post_prefetch.clear()

        with record_function(f"## sharded_module_prefetch {context.index} ##"):
            # pyrefly: ignore [bad-argument-type]
            with self._stream_context(self._prefetch_stream):
                prefetch_embeddings(
                    context,
                    self._pipelined_modules,
                    self._device,
                    # pyrefly: ignore [bad-argument-type]
                    self._stream_context,
                    self._data_dist_stream,
                    self._default_stream,
                )
