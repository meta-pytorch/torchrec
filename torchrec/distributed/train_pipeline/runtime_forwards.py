#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import logging
import os
import weakref
from contextlib import nullcontext
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import distributed as dist
from torch.profiler import record_function

# This is required to support older torch package exports that do not contain
# _check_int_overflow. During repackaging the
# old PackageImporter resolves dist_data from the frozen snapshot which may
# predate this helper.
try:
    from torchrec.distributed.dist_data import (
        _check_int_overflow,
        _get_kjt_input_split_tensors,
        _list_to_device,
    )
except ImportError:
    torch._C._log_api_usage_once(
        "torchrec.distributed.train_pipeline.import_failure._check_int_overflow"
    )

    def _check_int_overflow(*args: object, **kwargs: object) -> bool:  # type: ignore[misc]
        return False

    def _get_kjt_input_split_tensors(  # type: ignore[misc]
        *args: object,
        **kwargs: object,
    ) -> Optional[List[torch.Tensor]]:
        return None

    def _list_to_device(  # type: ignore[misc]
        data: List[int],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.tensor(data, device=device, dtype=dtype)


from torchrec.distributed.embedding_sharding import KJTSplitsAllToAllMeta
from torchrec.distributed.model_parallel import ShardedModule

# This is a safety measure against torch package issues. PEC pulls in
# torchrec modules (dist_data, modules.utils, ...) that older packages freeze
# at a revision predating PEC's dependencies, so this import can fail during
# repackaging. PEC training never runs from such a package, so falling back is
# safe: ShardedPECEmbeddingCollection is bound to the empty tuple, which makes
# the isinstance() check in PECPipelinedForward always False and routes every
# module through the inherited InSync path.
try:
    from torchrec.distributed.pec_embedding import (
        PECEmbeddingCollectionContext,
        ShardedPECEmbeddingCollection,
    )
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.distributed.train_pipeline.import_failure.pec_embedding"
    )
    PECEmbeddingCollectionContext = None  # type: ignore[misc,assignment]
    ShardedPECEmbeddingCollection = ()  # type: ignore[misc,assignment]

from torchrec.distributed.train_pipeline.pipeline_context import (
    CPUEmbeddingTrainPipelineContext,
    EmbeddingTrainPipelineContext,
    PECTrainPipelineContext,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.types import CallArgs
from torchrec.distributed.types import Awaitable, LazyAwaitable, LazyNoWait
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable

logger: logging.Logger = logging.getLogger(__name__)

TForwardContext = TypeVar("TForwardContext", bound=TrainPipelineContext)

EmbeddingModuleRetType = Union[Dict[str, JaggedTensor], KeyedTensor]


class BaseForward(Generic[TForwardContext]):
    def __init__(
        self,
        name: str,
        args: CallArgs,
        module: ShardedModule,
        context: TForwardContext,
        stream: Optional[torch.Stream] = None,
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._stream = stream
        self._device: torch.device = stream.device if stream else torch.device("cuda")

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> CallArgs:
        return self._args

    def set_context(self, context: TForwardContext) -> None:
        self._context = context

    def get_context(self) -> TForwardContext:
        return self._context

    def _write_with_reused_input_dist_if_requested(
        self,
        ctx: Multistreamable,
        inputs: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        if kwargs.get("write") is not True:
            return
        features = inputs[0] if inputs else kwargs.get("features")
        if not isinstance(features, KeyedJaggedTensor):
            raise ValueError("write=True requires a KeyedJaggedTensor forward input")
        write_with_reused_input_dist = getattr(
            self._module, "write_with_reused_input_dist", None
        )
        if not callable(write_with_reused_input_dist):
            raise ValueError(
                "write=True requires a sharded module with routing reuse support"
            )
        write_with_reused_input_dist(features, ctx)


# Set when TORCHREC_PIPELINED_FWD_EVENT_REUSE_CHECK sees a mismatch.
_dist_event_reuse_disabled = False


def _dist_wait_is_launch_free(request: Awaitable[Any], stream: torch.Stream) -> bool:
    """
    True only if `request.wait()`, run with `stream` current, enqueues no kernel,
    copy or stream wait: a callback-free KJTListAwaitable whose members are
    callback-free NoWait KJTs (with no device-side stride_per_key_per_rank to read
    back) or KJTAllToAllTensorsAwaitables prefetched on `stream`, whose wait only
    host-synchronizes the prefetch event. Must run before `wait()`, which consumes
    the prefetch. Freeing the pinned length_per_key buffer in that wait can still
    record a caching-host-allocator event on `stream`; no stream waits on it.
    """
    try:
        from torchrec.distributed.dist_data import KJTAllToAllTensorsAwaitable
        from torchrec.distributed.embedding_sharding import KJTListAwaitable
        from torchrec.distributed.types import NoWait
    except ImportError:
        return False
    if type(request) is not KJTListAwaitable or request.callbacks:
        return False
    for awaitable in request.awaitables:
        if awaitable.callbacks:
            return False
        if type(awaitable) is NoWait:
            kjt = awaitable._obj
            if type(kjt) is not KeyedJaggedTensor:
                return False
            stride_per_key_per_rank = kjt._stride_per_key_per_rank
            if (
                kjt.variable_stride_per_key()
                and stride_per_key_per_rank is not None
                and stride_per_key_per_rank.device.type != "cpu"
            ):
                return False
        elif type(awaitable) is KJTAllToAllTensorsAwaitable:
            prefetched = getattr(awaitable, "_prefetched", None)
            if awaitable._workers == 1 or prefetched is None or prefetched[4] != stream:
                return False
        else:
            return False
    return True


class PipelinedForward(BaseForward[TrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSparseDist

    With TORCHREC_PIPELINED_FWD_EVENT_REUSE=1 the current stream waits on the
    dist-stream event that an earlier pipelined forward of the same batch recorded
    when `request.wait()` enqueues nothing and the request was already pending at
    that record; otherwise it records a new event, as `wait_stream` does.
    Requests that cannot be hashed or weakly referenced never reuse an event.

    The reuse requires that, between that record and the reusing wait, nothing
    other than these launch-free waits enqueues work (kernels, copies, in-place
    writes) on the dist stream or makes the dist stream wait on another stream.
    TrainPipelineSparseDist meets this: its dist-stream work
    (start_sparse_data_dist, wait_sparse_data_dist,
    clear_sparse_data_dist_inputs) runs outside the model forward.
    """

    _use_main_stream_for_dist_init = False
    _dist_event_reuse_logged = False
    _dist_event_reuse_check_batches = 3

    def _dist_ready_events(
        self,
    ) -> Optional[Dict[torch.Stream, Tuple[torch.Event, "weakref.WeakSet[Any]"]]]:
        """
        Per dist stream, the last event recorded for this batch and the requests
        pending at that record; None when the event reuse is off for this call.
        """
        if (
            torch.compiler.is_compiling()
            or os.environ.get("TORCHREC_PIPELINED_FWD_EVENT_REUSE", "1") != "1"
            or _dist_event_reuse_disabled
            or self._stream is None
            or self._use_main_stream_for_dist_init
        ):
            return None
        index = self._context.index
        if index is None:
            return None
        state = getattr(self._context, "_pipelined_fwd_dist_ready_events", None)
        if state is None or state[0] != index:
            state = (index, {})
            # pyrefly: ignore[missing-attribute]
            self._context._pipelined_fwd_dist_ready_events = state
        return state[1]

    def _wait_dist_ready(
        self,
        ready_events: Dict[torch.Stream, Tuple[torch.Event, "weakref.WeakSet[Any]"]],
        reusable: Optional[torch.Event],
        data: Multistreamable,
    ) -> None:
        stream = cast(torch.Stream, self._stream)
        cur_stream = torch.get_device_module(self._device).current_stream()
        if reusable is None:
            pending: "weakref.WeakSet[Any]" = weakref.WeakSet()
            for r in self._context.input_dist_tensors_requests.values():
                if isinstance(r, Awaitable):
                    try:
                        pending.add(r)
                    except TypeError:
                        pass
            event = stream.record_event()
            cur_stream.wait_event(event)
            ready_events[stream] = (event, pending)
        else:
            cur_stream.wait_event(reusable)
            if (
                cast(int, self._context.index) < self._dist_event_reuse_check_batches
                and os.environ.get("TORCHREC_PIPELINED_FWD_EVENT_REUSE_CHECK", "1")
                == "1"
            ):
                self._check_dist_ready(data)
        if not self._dist_event_reuse_logged:
            self._dist_event_reuse_logged = True
            logger.info(
                f"PipelinedForward {self._name}: dist-stream event "
                f"{'recorded' if reusable is None else 'reused'} at batch "
                f"{self._context.index} (TORCHREC_PIPELINED_FWD_EVENT_REUSE)"
            )

    def _check_dist_ready(self, data: Multistreamable) -> None:
        """
        Compares the bytes of the KJT tensors as the current stream reads them
        right after the reused event with their bytes after a device synchronize,
        so NaNs with equal bit patterns compare equal.
        """
        global _dist_event_reuse_disabled
        tensors = [
            t
            for kjt in cast(Iterable[KeyedJaggedTensor], data)
            for t in (
                kjt.values(),
                kjt.lengths_or_none(),
                kjt.offsets_or_none(),
                kjt.weights_or_none(),
            )
            if t is not None
        ]
        seen = [t.clone() for t in tensors]
        torch.get_device_module(self._device).synchronize()
        if all(
            torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
            for a, b in zip(seen, tensors)
        ):
            logger.info(
                f"PipelinedForward {self._name}: dist-stream event reuse check "
                f"passed at batch {self._context.index} ({len(tensors)} tensors)"
            )
        else:
            _dist_event_reuse_disabled = True
            logger.error(
                f"PipelinedForward {self._name}: dist-stream event reuse check "
                f"failed at batch {self._context.index}; event reuse disabled"
            )

    def __call__(self, *input, **kwargs) -> Awaitable:
        if self._name not in self._context.input_dist_tensors_requests:
            available_names = list(self._context.input_dist_tensors_requests.keys())
            raise AssertionError(
                f"Invalid PipelinedForward usage, input_dist of {self._name} "
                f"is not available, probably consumed by others. "
                f"This typically means the embedding module '{self._name}' is "
                f"being invoked more than once in the model's forward pass, "
                f"or the pipeline context was not properly populated for this "
                f"iteration. Each embedding module (EBC, EC, etc.) can only be "
                f"invoked once per forward pass when using pipelined training. "
                f"Available input_dist names: {available_names}."
            )
        # we made a basic assumption that an embedding module (EBC, EC, etc.) should only be evoked only
        # once in the model's forward pass. For more details: https://github.com/meta-pytorch/torchrec/pull/3294
        request = self._context.input_dist_tensors_requests.pop(self._name)
        assert isinstance(request, Awaitable)
        ready_events = self._dist_ready_events()
        reusable: Optional[torch.Event] = None
        if ready_events is not None:
            stream = cast(torch.Stream, self._stream)
            recorded = ready_events.get(stream)
            if recorded is not None:
                try:
                    was_pending = request in recorded[1]
                except TypeError:
                    was_pending = False
                if was_pending and _dist_wait_is_launch_free(request, stream):
                    reusable = recorded[0]
        with record_function("## runtime_forward_assemble_KJT ##"):
            stream_ctx = (
                nullcontext()
                if self._use_main_stream_for_dist_init
                else torch.get_device_module(self._device).stream(self._stream)
            )
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with stream_ctx:
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        ctx = self._context.module_contexts.pop(self._name)

        if self._stream is not None:
            if ready_events is None:
                torch.get_device_module(self._device).current_stream().wait_stream(
                    self._stream
                )
            else:
                self._wait_dist_ready(ready_events, reusable, data)
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            ctx.record_stream(cur_stream)

        output = self._module.compute_and_output_dist(ctx, data)
        self._write_with_reused_input_dist_if_requested(ctx, input, kwargs)
        return output


class EmbeddingPipelinedForward(BaseForward[EmbeddingTrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSemiSync
    """

    def _wait_for_lookup_stream(self, cur_stream: torch.Stream) -> None:
        if self._stream is not None:
            cur_stream.wait_stream(self._stream)

    def __call__(
        self,
        *input,
        **kwargs,
    ) -> Union[
        Awaitable[EmbeddingModuleRetType],
        Tuple[
            Awaitable[EmbeddingModuleRetType], Awaitable[Optional[KeyedJaggedTensor]]
        ],
    ]:
        if self._name not in self._context.embedding_a2a_requests:
            available_names = list(self._context.embedding_a2a_requests.keys())
            raise AssertionError(
                f"Invalid PipelinedForward usage, input_dist of {self._name} "
                f"is not available, probably consumed by others. "
                f"Each embedding module (EBC, EC, etc.) can only be invoked "
                f"once per forward pass when using pipelined training. "
                f"Available embedding_a2a names: {available_names}."
            )
        # we made a basic assumption that an embedding module (EBC, EC, etc.) should only be evoked only
        # once in the model's forward pass. For more details: https://github.com/meta-pytorch/torchrec/pull/3294

        ctx = self._context.module_contexts.pop(self._name)
        cur_stream = torch.get_device_module(self._device).current_stream()

        if self._stream is not None:
            self._wait_for_lookup_stream(cur_stream)
            ctx.record_stream(cur_stream)

        awaitable = self._context.embedding_a2a_requests.pop(self._name)
        # in case of MC modules
        # pyrefly: ignore[unsafe-overlap]
        is_mc_module: bool = isinstance(awaitable, Iterable)
        remapped_kjts: Optional[KeyedJaggedTensor] = None

        if is_mc_module:
            embeddings = awaitable[0].wait()
            remapped_kjts = awaitable[1].wait()
        else:
            assert isinstance(awaitable, Awaitable)
            embeddings = (
                awaitable.wait()
            )  # trigger awaitable manually for type checking

        self._write_with_reused_input_dist_if_requested(ctx, input, kwargs)

        # pyrefly: ignore[bad-argument-type]
        self.detach_embeddings(embeddings=embeddings, cur_stream=cur_stream)
        if self._context.defer_pipeline_stream_fences and remapped_kjts is not None:
            remapped_kjts.record_stream(cur_stream)

        if is_mc_module:
            # pyrefly: ignore[bad-argument-type]
            return (LazyNoWait(embeddings), LazyNoWait(remapped_kjts))
        else:
            # pyrefly: ignore[bad-argument-type]
            return LazyNoWait(embeddings)

    def detach_embeddings(
        self,
        embeddings: Union[Dict[str, JaggedTensor], KeyedTensor],
        cur_stream: torch.Stream,
    ) -> None:
        """
        detach the grad from embeddings so that the backward/opt of the embeddings
        won't be invoked by loss.backward(). Instead, there is a dedicated embedding_backward
        call in semi-sync pipeline progress.
        """
        tensors = []
        detached_tensors = []
        # in case of EC, embeddings are Dict[str, JaggedTensor]
        if isinstance(embeddings, Dict):
            for jt in embeddings.values():
                assert isinstance(jt, JaggedTensor)
                tensor = jt.values()
                detached_tensor = tensor.detach().requires_grad_()
                detached_tensor.retain_grad()
                jt._values = detached_tensor
                tensors.append(tensor)
                detached_tensors.append(detached_tensor)
            self._context.embedding_tensors.append(tensors)
            self._context.embedding_features.append(list(embeddings.keys()))
            self._context.detached_embedding_tensors.append(detached_tensors)
        else:
            # in case of EBC, embeddings are KeyedTensor
            assert isinstance(embeddings, KeyedTensor)
            # pyrefly: ignore[bad-argument-type]
            embeddings.record_stream(cur_stream)
            tensor = embeddings.values()
            detached_tensor = tensor.detach().requires_grad_()
            detached_tensor.retain_grad()
            embeddings._values = detached_tensor
            tensors.append(tensor)
            detached_tensors.append(detached_tensor)
            self._context.embedding_tensors.append(tensors)
            """
            KeyedTensor is returned by EmbeddingBagCollections and its variants
            KeyedTensor holds dense data from multiple features and .values()
            returns a single concatenated dense tensor. To ensure that
            context.embedding_tensors[i] has the same length as
            context.embedding_features[i], we pass in a list with a single item:
            a list containing all the embedding feature names.
            """
            self._context.embedding_features.append([list(embeddings.keys())])
            self._context.detached_embedding_tensors.append(detached_tensors)


class CPUEmbeddingPipelinedForward(EmbeddingPipelinedForward):
    """
    Pipeline used in CPU eval with 3-stage pipelining.

    Returns pre-copied GPU embeddings from ``ctx.gpu_embedding_outputs``,
    which are populated by Stage 2 (``_copy_to_gpu``) of ``EvalPipelineCPUSparse``.
    """

    # pyrefly: ignore
    def __call__(self, *input, **kwargs) -> Union[
        Awaitable[EmbeddingModuleRetType],
        Tuple[
            Awaitable[EmbeddingModuleRetType], Awaitable[Optional[KeyedJaggedTensor]]
        ],
    ]:
        ctx = self._context
        assert isinstance(ctx, CPUEmbeddingTrainPipelineContext)
        assert self._name in ctx.gpu_embedding_outputs, (
            f"GPU embeddings for {self._name} not available. "
            f"Ensure Stage 2 (copy_to_gpu) ran before dense_forward."
        )
        embeddings = ctx.gpu_embedding_outputs.pop(self._name)
        return LazyNoWait(embeddings)


class InSyncEmbeddingPipelinedForward(EmbeddingPipelinedForward):
    """
    This pipeline is used in TrainPipelineFusedSparseDist
    """

    def _wait_for_lookup_stream(self, cur_stream: torch.Stream) -> None:
        if not self._context.defer_pipeline_stream_fences:
            super()._wait_for_lookup_stream(cur_stream)
            return
        event = self._context.embedding_lookup_events.pop(self._name, None)
        if event is None:
            raise RuntimeError(
                f"Missing lookup completion event for pipelined module {self._name}"
            )
        cur_stream.wait_event(event)

    def detach_embeddings(
        self,
        embeddings: Union[Dict[str, JaggedTensor], KeyedTensor],
        cur_stream: torch.Stream,
    ) -> None:
        if not self._context.defer_pipeline_stream_fences:
            return
        if isinstance(embeddings, dict):
            for jagged_tensor in embeddings.values():
                # JaggedTensor's annotation is CUDA-specific, but the tensor
                # operation dispatches on the actual stream device.
                # pyrefly: ignore[bad-argument-type]
                jagged_tensor.record_stream(cur_stream)
        else:
            # pyrefly: ignore[bad-argument-type]
            embeddings.record_stream(cur_stream)


class InSyncPrefetchEmbeddingPipelinedForward(InSyncEmbeddingPipelinedForward):
    """Returns a hoisted lookup's pending awaitable without waiting on it.

    Deferring the wait until the embeddings reach a torch function preserves the
    graph break and real-tensor layout guards required by ``torch.compile``.
    """

    def __call__(self, *input, **kwargs) -> Union[
        LazyAwaitable[Multistreamable],
        Tuple[LazyAwaitable[KeyedTensor], LazyAwaitable[Optional[KeyedJaggedTensor]]],
    ]:
        if self._name not in self._context.embedding_a2a_requests:
            available_names = list(self._context.embedding_a2a_requests.keys())
            raise AssertionError(
                f"Invalid PipelinedForward usage, embedding lookup of {self._name} "
                f"is not available, probably consumed by others. "
                f"Each embedding module (EBC, EC, etc.) can only be invoked "
                f"once per forward pass when using pipelined training. "
                f"Available embedding_a2a names: {available_names}."
            )
        return self._context.embedding_a2a_requests.pop(self._name)


class PECPipelinedForward(InSyncEmbeddingPipelinedForward):
    """Shared in-sync pipelined forward for PEC and non-PEC modules.

    Both retrieve embeddings computed one stage ahead — nothing is computed
    inline. PEC modules (ShardedPECEmbeddingCollection) merge their precomputed
    OL/NOL partitions via merge_partitioned_embeddings; non-PEC modules use the
    inherited InSyncEmbeddingPipelinedForward path (embedding_a2a_requests, with
    a no-op detach so loss.backward flows through to the fused-TBE update).

    Used by TrainPipelinePEC.
    """

    # pyre-ignore[2,3]
    def __call__(self, *input, **kwargs):
        if not isinstance(self._module, ShardedPECEmbeddingCollection):
            return super().__call__(*input, **kwargs)

        ctx = self._context
        assert isinstance(ctx, PECTrainPipelineContext)
        name = self._name

        pec_ctx = cast(PECEmbeddingCollectionContext, ctx.module_contexts[name])

        # OL/NOL awaitables and forward contexts are consumed by this forward.
        ol_awaitables = ctx.pec_ol_awaitables.pop(name)
        nol_awaitables = ctx.pec_nol_awaitables.pop(name)
        forward_ctxs = ctx.pec_forward_ctxs.pop(name)

        return self._module.merge_partitioned_embeddings(
            pec_ctx,
            ol_awaitables,
            nol_awaitables,
            forward_ctxs,
        )


class PrefetchPipelinedForward(BaseForward[PrefetchTrainPipelineContext]):
    """
    This pipeline is used in PrefetchTrainPipelineSparseDist
    OR in TrainPipelineCustomizedOrderSparseDist, when prefetch is enabled but pipeline_embedding_lookup_fwd is disabled
    """

    def __init__(
        self,
        name: str,
        args: CallArgs,
        module: ShardedModule,
        context: PrefetchTrainPipelineContext,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__(
            name=name,
            args=args,
            module=module,
            context=context,
            stream=prefetch_stream,
        )

    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name in self._context.module_input_post_prefetch
        ), "Invalid PrefetchPipelinedForward usage, please do not directly call model.forward()"
        data = self._context.module_input_post_prefetch.pop(self._name)
        ctx = self._context.module_contexts_post_prefetch.pop(self._name)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(ctx, data)


class PrefetchEmbeddingPipelinedForward(PrefetchPipelinedForward):
    """
    This pipeline is used in TrainPipelineCustomizedOrderSparseDist when
    prefetch is enabled and pipelined_sprase_lookup_fwd is enabled
    compute_and_output_dist for batch N is called at the end of step N - 1
    """

    def __init__(
        self,
        name: str,
        args: CallArgs,
        module: ShardedModule,
        context: PrefetchTrainPipelineContext,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__(
            name=name,
            args=args,
            module=module,
            context=context,
            prefetch_stream=prefetch_stream,
        )
        self._compute_and_output_dist_awaitable: Optional[
            Awaitable[Multistreamable]
        ] = None

    def compute_and_output_dist(self) -> None:
        assert (
            self._name in self._context.module_input_post_prefetch
        ), "Invalid PrefetchEmbeddingPipelinedForward usage, please do not directly call model.forward()"
        data = self._context.module_input_post_prefetch.pop(self._name)
        ctx = self._context.module_contexts_post_prefetch.pop(self._name)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx.record_stream(cur_stream)

        self._compute_and_output_dist_awaitable = self._module.compute_and_output_dist(
            ctx, data
        )

    def __call__(self, *input, **kwargs) -> Awaitable:
        awaitable = self._compute_and_output_dist_awaitable
        if not awaitable:
            raise Exception(
                "compute_and_output_dist must be called before __call__",
            )
        # Keeping it would keep its embeddings alive until the next batch.
        self._compute_and_output_dist_awaitable = None
        return awaitable


class KJTAllToAllForward:
    def __init__(
        self, pg: dist.ProcessGroup, splits: List[int], stagger: int = 1
    ) -> None:
        self._pg = pg
        self._splits = splits
        self._stagger = stagger
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))

    def __call__(self, input: KeyedJaggedTensor) -> KJTSplitsAllToAllMeta:
        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]
            device = input.values().device
            input_split_tensors = _get_kjt_input_split_tensors(input, self._splits)
            input_splits = (
                input.dist_splits(self._splits) if input_split_tensors is None else []
            )
            splits_tensors = (
                [_list_to_device(splits, device) for splits in input_splits]
                if input_split_tensors is None
                else input_split_tensors
            )
            if not input.variable_stride_per_key():
                input_stride = input.stride()
                splits_tensors.append(
                    _list_to_device(
                        [input_stride] * self._pg.size(),
                        device,
                        splits_tensors[0].dtype,
                    )
                )
                # Log corrupted input stride before it enters the
                # AllToAll pipeline.
                _check_int_overflow(
                    "KJTAllToAllForward",
                    [input_stride],
                    "input stride (BEFORE AllToAll)",
                    rank=rank,
                    world_size=self._pg.size(),
                    local_keys=local_keys,
                    splits=self._splits,
                )

            # Log all input split tensors being sent into AllToAll.
            # If these are already corrupted, the bug is upstream.
            for idx, splits in enumerate(input_splits):
                _check_int_overflow(
                    "KJTAllToAllForward",
                    splits,
                    f"input_splits[{idx}] (BEFORE AllToAll)",
                    rank=rank,
                    world_size=self._pg.size(),
                    local_keys=local_keys,
                )

            return KJTSplitsAllToAllMeta(
                pg=self._pg,
                _input=input,
                splits=self._splits,
                splits_tensors=splits_tensors,
                input_splits=input_splits,
                input_tensors=input.dist_tensors(),
                labels=input.dist_labels(),
                keys=local_keys,
                device=device,
                stagger=self._stagger,
                input_splits_are_tensors=input_split_tensors is not None,
            )
