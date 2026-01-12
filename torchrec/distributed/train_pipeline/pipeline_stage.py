#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from torch.profiler import record_function
from torch.utils.hooks import RemovableHandle
from torchrec.distributed.dist_data import KJTAllToAllTensorsAwaitable
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
    In,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import (
    BaseForward,
    InSyncEmbeddingPipelinedForward,
    PipelinedForward,
    PrefetchPipelinedForward,
)
from torchrec.distributed.train_pipeline.tracing import PipelinedPostproc
from torchrec.distributed.train_pipeline.utils import (
    _override_input_dist_forwards,
    _pipeline_detach_model,
    _prefetch_embeddings,
    _rewrite_model,
    _start_data_dist,
    _start_embedding_lookup,
    use_context_for_postprocs,
)
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

logger: logging.Logger = logging.getLogger(__name__)

StageOut = TypeVar("StageOut", bound=Pipelineable)
RunnableType = Callable[..., StageOut]
StageOutputWithEvent = Tuple[Optional[StageOut], Optional[torch.Event]]


@dataclass
class PipelineStage:
    """
    A pipeline stage represents a transform to an input that is independent of the
    backwards() of the model. Examples include batch H2D transfer, GPU postproc, or
    gradient-less model processing.

    Args:
        name (str): Name of the stage.
        runnable (Callable[In, Out]): Function that performs a gradient-less
            transform.
        stream (torch.cuda.streams.Stream): Stream to run on. Often each stage has a
            unique stream, but having different pipelines share a stream provides more
            synchronization semantics.
        fill_callback (Optional[Callable[[], None]])) - optional step to run after the main
            runnable during filling the pipeline
        data_exhausted_callback (Optional[Callable[[], None]])) - optional callback to run
            when data is ehxausted
    """

    name: str
    runnable: RunnableType
    stream: torch.Stream
    fill_callback: Optional[Callable[[], None]] = None
    data_exhausted_callback: Optional[Callable[[], None]] = None


class SparseDataDistUtil(Generic[In]):
    """
    Helper class exposing methods for sparse data dist and prefetch pipelining.
    Currently used for `StagedTrainPipeline` pipeline stages.\n
    Specifying `embedding_lookup_stream` makes `StagedTrainPipeline` semi-synchronous.

    Args:
        model (torch.nn.Module): Model to pipeline
        data_dist_stream (torch.cuda.Stream): Stream on which to run sparse data dist.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
        prefetch_stream (Optional[torch.cuda.Stream]): Stream on which model prefetch runs
            Defaults to `None`. This needs to be passed in to enable prefetch pipelining.
        pipeline_postproc (bool): whether to pipeline postproc modules. Defaults to `False`.
        embedding_lookup_stream (Optional[torch.cuda.Stream]): Stream on which embedding lookup runs
            Defaults to `None`. This needs to be passed in to enable embedding lookup pipelining.

    Example::
        # With prefetch pipeline:
        sdd = SparseDataDistUtil(
            model=model,
            data_dist_stream=torch.cuda.Stream(),
            prefetch_stream=torch.cuda.Stream(), <-- required to enable prefetch pipeline
        )
        pipeline = [
            PipelineStage(
                name="data_copy",
                runnable=lambda batch, context: batch.to(
                    self._device, non_blocking=True
                ),
                stream=torch.cuda.Stream(),
            ),
            PipelineStage(
                name="start_sparse_data_dist",
                runnable=sdd.start_sparse_data_dist,
                stream=sdd.data_dist_stream,
                fill_callback=sdd.wait_sdd_fill_callback,
            ),
            PipelineStage(
                name="prefetch",
                runnable=sdd.prefetch,
                stream=sdd.prefetch_stream,
                fill_callback=sdd.load_prefetch,
            ),
        ]

        # With embedding lookup pipeline:
        sdd = SparseDataDistUtil(
            model=model,
            data_dist_stream=torch.cuda.Stream(),
            embedding_lookup_stream=torch.cuda.Stream(), <-- required to enable embedding lookup pipeline
        )
        pipeline = [
            PipelineStage(
                name="data_copy",
                runnable=lambda batch, context: batch.to(
                    self._device, non_blocking=True
                ),
                stream=torch.cuda.Stream(),
            ),
            PipelineStage(
                name="start_sparse_data_dist",
                runnable=sdd.start_sparse_data_dist,
                stream=sdd.data_dist_stream,
                fill_callback=sdd.wait_sdd_fill_callback,
            ),
            PipelineStage(
                name="start_embedding_lookup",
                runnable=sdd.start_embedding_lookup,
                stream=sdd.embedding_lookup_stream,
            ),
        ]

        return StagedTrainPipeline(pipeline_stages=pipeline)
    """

    _TRAIN_CONTEXT_VERSION = 1
    # Convenience flag to perform additional assertions on contexts
    # to make sure contexts are advancing correctly.
    _WITH_CONTEXT_ASSERTIONS = False

    def __init__(
        self,
        model: torch.nn.Module,
        data_dist_stream: torch.Stream,
        apply_jit: bool = False,
        prefetch_stream: Optional[torch.Stream] = None,
        pipeline_postproc: bool = False,
        embedding_lookup_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.data_dist_stream = data_dist_stream
        self.apply_jit = apply_jit
        self.prefetch_stream = prefetch_stream
        self.embedding_lookup_stream = embedding_lookup_stream
        self._next_index: int = 0
        self._contexts: Deque[TrainPipelineContext] = deque()
        self.initialized = False
        self._pipelined_modules: List[ShardedModule] = []
        self._pipelined_postprocs: List[PipelinedPostproc] = []
        self.fwd_hook: Optional[RemovableHandle] = None
        self._device: torch.device = data_dist_stream.device

        assert not (
            self._with_prefetch and self._with_embedding_lookup
        ), "Cannot enable both prefetch and embedding lookup at the same time. Prefetch is redundant with embedding lookup."

        self._stream_context: Callable[
            [Optional[torch.Stream]], torch.cuda.StreamContext
        ] = (
            torch.get_device_module(self._device).stream
            if self._device.type in ["cuda", "mtia"]
            else torch.cuda.stream
        )

        # pyre-ignore
        self._original_forwards: List[Callable[..., Any]] = []
        self._original_kjt_dist_forwards: List[
            Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
        ] = []

        self._pipelined_forward: Type[BaseForward[TrainPipelineContext]]
        if self._with_prefetch:
            self._pipelined_forward = cast(
                Type[BaseForward[TrainPipelineContext]], PrefetchPipelinedForward
            )
        elif self._with_embedding_lookup:
            self._pipelined_forward = cast(
                Type[BaseForward[TrainPipelineContext]], InSyncEmbeddingPipelinedForward
            )
        else:
            self._pipelined_forward = PipelinedForward

        self._default_stream: Optional[torch.Stream] = (
            (torch.get_device_module(self._device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )
        # When data iterator is exhausted, contexts should continue advancing until
        # reaching the end (i.e. no longer called from the StagedTrainingPipeline)
        # however normal invariants no longer apply (e.g. module_contexts might be empty
        # before prefetch stage). Currently, all actions (`prefetch`, `start/wait_sparse_data_dist`)
        # tolerate lack of data from the previous stage - so context assertions are mostly
        # correctness invariant. However, if that changes, having invariants monitored/enforced
        # during exhastion phase might become necessary.
        self._exhausting_mode = False
        self._pipeline_postproc = pipeline_postproc

    @property
    def _with_prefetch(self) -> bool:
        return self.prefetch_stream is not None

    @property
    def _with_embedding_lookup(self) -> bool:
        return self.embedding_lookup_stream is not None

    def _is_reattaching(self) -> bool:
        return len(self._contexts) > 0

    def should_assert_context_invariants(self, ctx: TrainPipelineContext) -> bool:
        return (
            self._WITH_CONTEXT_ASSERTIONS
            and self.initialized
            and not self._exhausting_mode
            and (
                ctx.index is not None and ctx.index >= 0
            )  # "fake contexts" to support pipeline initialization
        )

    # === Debugging helpers === #
    @property
    def _have_pipelined_modules(self) -> bool:
        return len(self._pipelined_modules) > 0

    @property
    def _have_pipelined_postprocs(self) -> bool:
        return len(self._pipelined_postprocs) > 0

    def _pipelined_modules_fqns(self) -> Set[str]:
        return {module.forward._name for module in self._pipelined_modules}

    def _pipelined_postprocs_fqns(self) -> Set[str]:
        return {module._fqn for module in self._pipelined_postprocs}

    # === Debugging helpers === #

    # ==== Context management === #
    # In short: version=1 contexts essentially represent "passing of time"
    # and have one-to-one correspondence to batches. "Monolithic" torchrec pipelines
    # (e.g. TrainPipelineSparseDist) explicitly manage batches and contexts together
    # (see TrainPipelineSparseDist.enqueue_batch), however StagedTrainPipeline abstracts
    # that away + supports stages that don't require contexts (in fact, SDD is the only one)
    # So we just manage contexts and batches together in lockstep - via _advance_context calls.
    #
    # Essentially, StagedTrainPipeline during a single `progress` call runs each stage
    # for a different batch, keeping the stage outputs in a `_stage_outputs` list, and
    # advancing the list at the beginning of the `progress`.
    # Tricky part is that SparseDataDistUtil might be participating in TWO stages:
    # * "main" with start_data_dist -> wait_data_dist pair for `runnable` and `fill_callback`
    # * "prefetch" with prefetch -> load_prefetch for `runnable` and `fill_callback` (optional)
    # * "embedding_lookup" with start_embedding_lookup for `runnable` (optional)
    #
    # For this to work, we:
    # (1) need to manage contexts in a lockstep with batch advancing through stages (_advance_context)
    # (2) perform various actions (start dist, wait dist, etc.) against the correct contexts
    #    ("named" contexts below and how they are used in start/wait sparse_dist, prefetch, etc.)
    # (3) set contexts for the _pipelined_modules and _pipelined_postprocs to the "current batch context"
    #       for the model to run correctly (_set_module_context)
    #
    # SDD Util uses two or three contexts, depending on if prefetch or embedding_lookup is enabled
    # * context[0] is always the "current batch" context - used for model forward (outside this class)
    # * context[1] is used for prefetch/embedding_lookup if either is set, and start/wait_sparse_data_dist if not
    # * context[2] is used for start/wait_sparse_data_dist if prefetch or embedding_lookup is set

    def _create_context(self, index: int) -> TrainPipelineContext:
        version = self._TRAIN_CONTEXT_VERSION
        if self._with_prefetch:
            return PrefetchTrainPipelineContext(index=index, version=version)
        if self._with_embedding_lookup:
            return EmbeddingTrainPipelineContext(index=index, version=version)
        return TrainPipelineContext(index=index, version=version)

    def _add_context(self) -> None:
        self._contexts.append(self._create_context(self._next_index))
        self._next_index += 1

    def _advance_context(self) -> None:
        self._assert_contexts_count()
        self._contexts.popleft()
        self._add_context()
        self._set_module_context(self._context_for_model_forward())

    def _set_module_context(self, context: TrainPipelineContext) -> None:
        for module in self._pipelined_modules:
            module.forward.set_context(context)

        for postproc_module in self._pipelined_postprocs:
            # This ensures that next iter model fwd uses cached results
            postproc_module.set_context(context)

    def _assert_contexts_count(self) -> None:
        if not self._WITH_CONTEXT_ASSERTIONS:
            return
        contexts_len = len(self._contexts)
        expected = 3 if (self._with_prefetch or self._with_embedding_lookup) else 2
        assert (
            contexts_len == expected
        ), f"Expected to have {expected} contexts, but had {contexts_len}"

    # ====== "Named" contexts - to make it clearer which contexts are used for which operation ====== #
    # This is purely convenience methods, feel free to remove if they get in the way
    def _current_context(self) -> TrainPipelineContext:
        return self._contexts[0]

    def _assert_input_dist_tensors(
        self, context: TrainPipelineContext, expected_fqns: Set[str]
    ) -> None:
        specified_keys = context.input_dist_tensors_requests.keys()
        assert (
            specified_keys == expected_fqns
        ), f"Context(idx:{context.index}).input_dist_tensors_requests {specified_keys} != pipelined modules fqns {expected_fqns}"

    def _assert_module_contexts(
        self, context: TrainPipelineContext, expected_fqns: Set[str]
    ) -> None:
        specified_keys = context.module_contexts.keys()
        assert (
            specified_keys == expected_fqns
        ), f"Context(idx:{context.index}).module_contexts {specified_keys} != pipelined modules fqns {expected_fqns}"

    def _assert_module_contexts_post_prefetch(
        self, context: PrefetchTrainPipelineContext, expected_fqns: Set[str]
    ) -> None:
        specified_keys = context.module_contexts_post_prefetch.keys()
        assert (
            specified_keys == expected_fqns
        ), f"Context(idx:{context.index}).module_contexts_post_prefetch {specified_keys} != pipelined modules fqns {expected_fqns}"

    def _assert_module_input_post_prefetch(
        self, context: PrefetchTrainPipelineContext, expected_fqns: Set[str]
    ) -> None:
        specified_keys = context.module_input_post_prefetch.keys()
        assert (
            specified_keys == expected_fqns
        ), f"Context(idx:{context.index}).module_input_post_prefetch {specified_keys} != pipelined modules fqns {expected_fqns}"

    def _assert_embedding_a2a_requests(
        self, context: EmbeddingTrainPipelineContext, expected_fqns: Set[str]
    ) -> None:
        specified_keys = context.embedding_a2a_requests.keys()
        assert (
            specified_keys == expected_fqns
        ), f"Context(idx:{context.index}).embedding_a2a_requests {specified_keys} != pipelined modules fqns {expected_fqns}"

    def _context_for_model_forward(self) -> TrainPipelineContext:
        ctx = self._current_context()
        if self.should_assert_context_invariants(ctx):
            target_fqns = self._pipelined_modules_fqns()
            if self._with_prefetch:
                assert isinstance(ctx, PrefetchTrainPipelineContext)
                self._assert_module_input_post_prefetch(ctx, target_fqns)
                self._assert_module_contexts_post_prefetch(ctx, target_fqns)
            elif self._with_embedding_lookup:
                assert isinstance(ctx, EmbeddingTrainPipelineContext)
                self._assert_embedding_a2a_requests(ctx, target_fqns)
            else:
                self._assert_input_dist_tensors(ctx, target_fqns)
                self._assert_module_contexts(ctx, target_fqns)
        return ctx

    def _start_dist_context(self) -> TrainPipelineContext:
        if self._with_prefetch or self._with_embedding_lookup:
            ctx = self._contexts[2]
        else:
            ctx = self._contexts[1]

        return ctx

    def _wait_dist_context(self) -> TrainPipelineContext:
        # Note: see comment on the forward_hook in _initialize method
        ctx = self._start_dist_context()
        if self.should_assert_context_invariants(ctx):
            if self._have_pipelined_modules:
                assert (
                    len(ctx.fused_splits_awaitables) > 0
                ), f"fused_splits_awaitables was empty on {ctx.index=} - was start_sparse_data_dist called?"
        return ctx

    def _prefetch_context(self) -> PrefetchTrainPipelineContext:
        ctx = self._contexts[1]
        assert isinstance(
            ctx, PrefetchTrainPipelineContext
        ), "Pass prefetch_stream into SparseDataDistUtil to use prefetch_context()"
        if self.should_assert_context_invariants(ctx):
            target_fqns = self._pipelined_modules_fqns()
            self._assert_input_dist_tensors(ctx, target_fqns)
            self._assert_module_contexts(ctx, target_fqns)
        return ctx

    def _embedding_lookup_context(self) -> EmbeddingTrainPipelineContext:
        ctx = self._contexts[1]
        assert isinstance(
            ctx, EmbeddingTrainPipelineContext
        ), "Pass embedding_lookup_stream into SparseDataDistUtil to use embedding_lookup_context()"
        if self.should_assert_context_invariants(ctx):
            target_fqns = self._pipelined_modules_fqns()
            self._assert_embedding_a2a_requests(ctx, target_fqns)
        return ctx

    # ====== End "Named" contexts ====== #

    # === End context management === #

    def detach(self) -> torch.nn.Module:
        """
        Removes sparse data dist (SDD) pipelining from model forward and input dist.
        Modifies existing model in place and returns the model.

        detach() can be called at any point, and inflight batches do not need to be
        flushed before calling it. Calling pipeline.progress() will re-attach the model
        to the pipeline and the pipeline will progress normally from the point it was
        detached (i.e. inflight batches will be kept when calling detach).

        While the model is detached, it is equivalent to the model before passing to
        the pipeline, so forward and backward passes, and optimizer updates can be
        carried out normally.
        """
        if self.initialized:
            assert self.fwd_hook is not None
            self.fwd_hook.remove()

            _pipeline_detach_model(
                model=self.model,
                pipelined_modules=self._pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
                pipelined_postprocs=self._pipelined_postprocs,
            )

        self.initialized = False
        return self.model

    def _initialize_or_reattach(self, batch: In) -> None:
        # Step 0: Handle differences between initialization and reattaching
        if self._is_reattaching():
            # if reattaching, contexts are already there, so we want to use
            # the current context for model forward - as if continuing to run normally
            context_for_rewrite = self._current_context()
        else:
            # if initializing, no contexts are present, so we add them:
            if self._with_prefetch or self._with_embedding_lookup:
                self._contexts.append(self._create_context(-2))  # throwaway context
            self._contexts.append(self._create_context(-1))  # throwaway context
            self._add_context()  # actual context to be used for everything in the initial iteration
            context_for_rewrite = self._contexts[-1]

        self._assert_contexts_count()

        # Step 1: Pipeline input dist in trec sharded modules
        (
            self._pipelined_modules,
            self.model,
            self._original_forwards,
            self._pipelined_postprocs,
            _,
        ) = _rewrite_model(
            model=self.model,
            context=context_for_rewrite,
            dist_stream=self.data_dist_stream,
            batch=batch,
            apply_jit=self.apply_jit,
            pipelined_forward=self._pipelined_forward,
            pipeline_postproc=self._pipeline_postproc,
            default_stream=self._default_stream,
        )
        # Setting the stage for the first batch
        # initialize input dist
        _start_data_dist(self._pipelined_modules, batch, self._start_dist_context())
        # so we can override input dist forwards
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

        # Step 2: Register post-forward hook to wait SDD and advance contexts
        def forward_hook(
            module: torch.nn.Module,
            input: Union[torch.Tensor, Tuple[torch.Tensor]],
            output: Union[torch.Tensor, Tuple[torch.Tensor]],
        ) -> None:
            # Note: tricky part - a bit delicate choreography between
            # StagedPipeline and this class
            # (see https://github.com/meta-pytorch/torchrec/pull/2239 for details)
            # wait_dist need to be called as post_forward hook
            # at the end of the batch N, so that the data is awaited
            # before start of the next batch.
            self.wait_sparse_data_dist()
            # _advance_context should be called after wait_sparse_data_dist,
            # but before start_data_dist for the next batch
            # which means right here, and nowhere else
            self._advance_context()
            # ... this can be made more explicit by adding dedicated hooks for "batch start"/"batch end" events
            # to the StagedPipeline, PipelineStage and this class, but hook seems to be doing an adequate job for now

        self.fwd_hook = self.model.register_forward_hook(forward_hook)

        self.initialized = True

    def wait_sdd_fill_callback(self) -> None:
        """
        Used by StagedTrainPipeline during only during initial pipeline filling.

        At that part, model.forward is not executed, so forward hook is not called.
        """
        self.wait_sparse_data_dist()
        self._advance_context()

    def data_exhausted_callback(self) -> None:
        """
        Called by StagedTrainPipeline when all batches were processed.
        """
        self._exhausting_mode = True

    def start_sparse_data_dist(self, batch: In) -> In:
        if not self.initialized:
            self._initialize_or_reattach(batch)

        ctx = self._start_dist_context()
        with record_function(f"## start_sparse_data_dist {ctx.index} ##"):
            with use_context_for_postprocs(self._pipelined_postprocs, ctx):
                _start_data_dist(self._pipelined_modules, batch, ctx)

        return batch

    def wait_sparse_data_dist(self) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        ctx = self._wait_dist_context()
        with record_function(f"## wait_sparse_data_dist {ctx.index} ##"):
            with self._stream_context(self.data_dist_stream):
                for names, awaitable in ctx.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        ctx.input_dist_tensors_requests[name] = request
        # these won't be used by the rest of the pipeline, so just deleting them to free
        # the memory they occupy
        ctx.input_dist_splits_requests.clear()
        ctx.fused_splits_awaitables.clear()

    def prefetch(self, batch: In) -> In:
        """
        Waits for input dist to finish, then prefetches data.
        """
        assert isinstance(
            self._prefetch_context(), PrefetchTrainPipelineContext
        ), "Pass prefetch_stream into SparseDataDistUtil to use prefetch() as a stage"
        ctx: PrefetchTrainPipelineContext = self._prefetch_context()

        with self._stream_context(self.prefetch_stream):
            data_per_pipelined_module = _prefetch_embeddings(
                batch,
                ctx,
                self._pipelined_modules,
                self._device,
                self._stream_context,
                self.data_dist_stream,
                self._default_stream,
            )
            # TODO (eugenykolpakov): investigate if these can be moved outside of the `with stream_context(...)`  block
            # This might impact memory fragmentation (since CUDA caching allocator is stream-aware),
            # so need to check how memory behaves with different streams
            for sharded_module in self._pipelined_modules:
                forward = sharded_module.forward
                data = data_per_pipelined_module[forward._name]
                ctx.module_input_post_prefetch[forward._name] = data
                ctx.module_contexts_post_prefetch[forward._name] = (
                    ctx.module_contexts.pop(forward._name)
                )
        return batch

    def load_prefetch(self) -> None:
        """
        DEPRECATED: exists for backward compatibility
        """
        # Version=0 did
        # module_input_post_prefetch = module_input_post_prefetch_for_next_batch
        # module_contexts_post_prefetch = module_contexts_post_prefetch_for_next_batch
        # with version=1, there's nothing to do - they are managed at a context level,
        # so this is essentially done by _advance_context + prefetch above
        pass

    def start_embedding_lookup(self, batch: In) -> In:
        """
        Initiates embedding lookup on the embedding_lookup_stream after data distribution.
        This enables pipelining of embedding lookup operations independently from the main
        model forward pass.
        """
        context = self._embedding_lookup_context()
        with record_function(f"## start_embedding_lookup {context.index} ##"):
            current_stream = torch.get_device_module(self._device).current_stream()
            with self._stream_context(self.embedding_lookup_stream):
                for module in self._pipelined_modules:
                    _start_embedding_lookup(
                        module,
                        context,
                        source_stream=self.data_dist_stream,
                        target_stream=current_stream,
                        stream_context=self._stream_context,
                    )
        return batch

    def start_sparse_data_dist_stage(
        self,
        name: str = "start_sparse_data_dist",
        runnable: Optional[RunnableType] = None,
    ) -> PipelineStage:
        return PipelineStage(
            name=name,
            runnable=runnable or self.start_sparse_data_dist,
            stream=self.data_dist_stream,
            fill_callback=self.wait_sdd_fill_callback,
            data_exhausted_callback=self.data_exhausted_callback,
        )

    def start_embedding_lookup_stage(
        self,
        name: str = "start_embedding_lookup",
        runnable: Optional[RunnableType] = None,
    ) -> PipelineStage:
        assert (
            self.embedding_lookup_stream is not None
        ), "embedding_lookup_stream is not set"
        return PipelineStage(
            name=name,
            runnable=runnable or self.start_embedding_lookup,
            stream=self.embedding_lookup_stream,
        )
