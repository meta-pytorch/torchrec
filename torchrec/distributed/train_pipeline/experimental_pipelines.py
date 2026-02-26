#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, cast, Deque, Iterator, Optional, Tuple, Type, Union

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.memory_stashing import MemoryStashingManager
from torchrec.distributed.train_pipeline.backward_injection import (
    InjectionSite,
    OutputDistSite,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    In,
    Out,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import PipelinedForward
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline.types import PipelineState
from torchrec.distributed.train_pipeline.utils import _to_device, FutureDeque
from torchrec.distributed.types import ShardingType

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
        - Supports both training and evaluation modes via the `is_training` flag.
        - Conditionally executes backward pass and optimizer step only during training.
        - Maintains the same pipelining benefits (overlapping data transfer, sparse
          data distribution, and forward pass) for both modes.

    Pipeline Stages (inherited from TrainPipelineSparseDist):
        - Stage 3: Forward/Backward/Optimizer (current batch)
        - Stage 2: Sparse data distribution (next batch)
        - Stage 1: Device transfer (batch i+2)

    Note:
        The `is_training` flag is set per-batch via the context, allowing fine-grained
        control over which batches trigger gradient computation and weight updates.
    """

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if self._state == PipelineState.UNKNOWN:
            return super()._next_batch(dataloader_iter)
        return self._next_batch_on_cpu

    def progress(self, dataloader_iter: Iterator[In], is_training: bool = True) -> Out:
        """
        Execute one step of the pipelined train/eval loop.

        This method processes one batch through the full pipeline while overlapping
        operations for subsequent batches. It conditionally executes backward pass
        and optimizer step based on both the model's training mode and the per-batch
        `is_training` flag.

        For TrainPipelineSparseDist, we assume the max pipelined batches == 3 (capacity):
            - batches[0]: current batch, for emb_lookup, output_dist, and fwd/bwd/opt
                          (expecting input_dist completed)
            - batches[1]: next batch, for input_dist (expecting copied to device)
            - batches[2]: i+2 batch, for copy_batch_to_gpu
                          (expecting non-exhausted dataloader iter)

        Args:
            dataloader_iter: Iterator yielding input batches from the dataloader.
            is_training: Whether the *newly enqueued* batch (batch i+2) should be
                processed in training mode. Defaults to True.

        Returns:
            Out: The output from the forward pass of the current batch (batches[0]).

        Raises:
            StopIteration: When all batches have been processed (pipeline is empty).

        Note:
            The backward/optimizer execution depends on BOTH:
            1. `self._model.training` - the model's overall training mode
            2. `self.contexts[0].is_training` - the per-batch training flag set during enqueue

            This allows fine-grained control where the model can be in training mode
            but specific batches can skip gradient computation (e.g., for evaluation batches
            interleaved with training batches).
        """

        self._state = PipelineState.UNKNOWN
        # Attach the model just in case the user forgets to call it, especially when the user
        # pauses the pipeline.progress and detaches the model for other purposes.
        if not self._model_attached:
            self.attach(self._model)

        # Fill the pipeline is only needed for the beginning when the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)
        self._state = PipelineState.IDLE

        self._next_batch_on_cpu = TrainPipelineSparseDist._next_batch(
            self, dataloader_iter
        )
        if self._next_batch_on_cpu is None and not is_training:
            # stop current progress if eval iter is exhausted
            # training iter will continue
            raise StopIteration

        for context in self.contexts:
            # this only fills the context loaded from fill_pipeline
            if not hasattr(context, "is_training"):
                logger.info(
                    f"initial fill batch-{context.index} with {'train' if is_training else 'eval'}"
                )
                # pyrefly: ignore [missing-attribute]
                context.is_training = is_training

        # Here is the expected stop after exhausting all batches
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        # Zero gradients only when model is in training mode
        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        # Wait for batches[0] being available on device, this should always be completed since
        # the input_dist of batches[0] has been invoked in previous iter. TODO: fact check
        self._wait_for_batch()

        # Start sparse data distribution for the next batch (overlapped with current forward)
        if len(self.batches) >= 2:
            # Invoke splits all_to_all comms (first part of input_dist)
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # pyrefly: ignore [missing-attribute]
        is_curr_training = self._model.training and self.contexts[0].is_training

        # Batch i+2: load data and copy to GPU, the dataloader iter will first exhaust here
        if self.enqueue_batch(dataloader_iter):
            # pyrefly: ignore [missing-attribute]
            self.contexts[-1].is_training = is_training

        # Forward pass for current batch
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

        # Execute backward and optimizer step only if:
        # 1. Model is in training mode (self._model.training)
        # 2. Current batch is marked for training (self.contexts[0].is_training)
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
                        return _to_device(cast(In, batch), self._device, True)

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
    pass via OutputDistSite backward hook injection. This overlaps the optimizer
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
        )
        self._output_dist_site = OutputDistSite(
            fqn=site_fqn, sharding_type=sharding_type
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
    specified OutputDistSite, overlapping the CPU->GPU transfer with backward
    all-to-all communication.

    Timeline per iteration:
        forward -> backward [ restore_optimizer_state at OutputDistSite ] ->
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
        )
        self._output_dist_site = OutputDistSite(
            fqn=site_fqn, sharding_type=sharding_type
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
        )
        if isinstance(site_fqn, str):
            self._injection_site = InjectionSite(fqn=site_fqn)
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
