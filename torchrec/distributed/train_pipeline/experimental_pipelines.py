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
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.logger import one_time_rank0_logger
from torchrec.distributed.memory_stashing import MemoryStashingManager
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
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    CPUEmbeddingPipelinedForward,
    PipelinedForward,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline.types import PipelineState
from torchrec.distributed.train_pipeline.utils import (
    _override_input_dist_forwards,
    _rewrite_model,
    _start_data_dist,
    _to_device,
    FutureDeque,
    use_context_for_postprocs,
)
from torchrec.distributed.types import LazyNoWait, ShardingType
from torchrec.sparse.jagged_tensor import KeyedTensor

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
        )
        if isinstance(site_fqn, str):
            self._injection_site = InjectionSite(
                fqn=site_fqn,
                tensor_finder=FirstGradTensorFinder(),
                target_type=InjectionTargetType.PARAM_GRAD,
            )
        else:
            self._injection_site = site_fqn

        MemoryStashingManager.set_streams(
            self._memcpy_stream,  # pyrefly: ignore[bad-argument-type]
            torch.cuda.Stream(device=device),
        )

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        super()._pipeline_model(batch, context, pipelined_forward)

        def work(_pipeline: Any) -> None:
            with record_function("## restore_embedding_weights ##"):
                MemoryStashingManager.restore_embedding_weights()

        self.register_backward_hook(self._injection_site, work)
