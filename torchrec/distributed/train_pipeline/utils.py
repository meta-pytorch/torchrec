#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import contextlib
import copy
import logging
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import AbstractContextManager
from threading import Event, Thread
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
)

import torch
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten
from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListAwaitable,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.embedding_types import KJTList

try:
    from torchrec.distributed.logger import (
        LazyStr,
        one_time_logger,
        one_time_rank0_logger,
    )
except Exception:
    # Safety measure against torch package issues: old packages may not have
    # LazyStr/one_time_logger in their archived torchrec.distributed.logger
    torch._C._log_api_usage_once(
        "torchrec.distributed.train_pipeline.utils.import_failure.logger"
    )

    class LazyStr:  # pyre-ignore[11]: Annotation for no-op fallback
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def __str__(self) -> str:
            return self._fn()

    one_time_logger = logging.getLogger(__name__)
    one_time_rank0_logger = logging.getLogger(__name__)
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
    In,
    Out,  # noqa
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.postproc import PipelinedPostproc
from torchrec.distributed.train_pipeline.runtime_forwards import (
    BaseForward,
    CPUEmbeddingPipelinedForward,
    EmbeddingPipelinedForward,
    InSyncEmbeddingPipelinedForward,
    KJTAllToAllForward,
    PipelinedForward,
    PrefetchPipelinedForward,
    TForwardContext,
)
from torchrec.distributed.train_pipeline.tracing import (
    _get_leaf_module_names,
    NodeArgsHelper,
    Tracer,
)
from torchrec.distributed.train_pipeline.types import CallArgs  # noqa
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable, Pipelineable

logger: logging.Logger = logging.getLogger(__name__)


def _batch_tensor_size(batch: Any) -> int:
    """Compute total tensor storage size in bytes for a batch.

    Uses pytree to flatten the batch into leaf tensors, then sums
    element_size() * numel() for each. All operations are O(1) tensor
    metadata reads — no data is accessed.
    """
    leaves, _ = tree_flatten(batch)
    return sum(
        leaf.element_size() * leaf.numel()
        for leaf in leaves
        if isinstance(leaf, torch.Tensor)
    )


def _to_device(
    batch: In,
    device: torch.device,
    non_blocking: bool,
    data_copy_stream: Optional[torch.Stream] = None,
) -> In:
    assert isinstance(
        batch, (torch.Tensor, Pipelineable)
    ), f"{type(batch)} must implement Pipelineable interface"
    if data_copy_stream is not None:
        return cast(
            In,
            batch.to(
                device=device,
                non_blocking=non_blocking,
                # pyrefly: ignore[unexpected-keyword]
                data_copy_stream=data_copy_stream,
            ),
        )
    else:
        return cast(
            In,
            batch.to(
                device=device,
                non_blocking=non_blocking,
            ),
        )


def _wait_for_batch(
    batch: In, stream: Optional[torch.Stream], record_stream: bool = True
) -> None:
    """
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tenosrs. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.
    """
    if stream is None:
        return

    # batch is loaded/processed in the given stream, but will be used in the current
    device = stream.device
    curr_stream = torch.get_device_module(device).current_stream()

    # current stream needs to wait for the given stream to complete
    curr_stream.wait_stream(stream)

    # record_stream is needed when the batch is created (allocated) in the given stream
    # but used by another stream (e.g., the current stream), however, when the batch is
    # created in the current stream (in-place copy), we don't need to call
    if record_stream:
        assert isinstance(
            batch, (torch.Tensor, Multistreamable)
        ), f"{type(batch)} must implement Multistreamable interface"
        batch.record_stream(curr_stream)


def _wait_for_events(
    batch: In,
    context: TrainPipelineContext,
    stream: Optional[torch.Stream],
) -> None:
    """
    Wait for any outstanding events for a given context
    """

    for event in context.events:
        event.wait()
    context.events.clear()
    if stream:
        assert isinstance(
            batch, (torch.Tensor, Multistreamable)
        ), f"{type(batch)} must implement Multistreamable interface"
        batch.record_stream(stream)


def _clear_releasable_inputs(context: TrainPipelineContext) -> None:
    """Clear deferred KJT feature storage from a context's module contexts.

    Called after all pipelined modules' input_dist calls are complete, so it's
    safe to free the original batch KJT tensor storage. Each module has already
    permuted and extracted its features into independent tensors.
    """
    # module contexts for the current batch are in module_contexts_next_batch or module_contexts.
    contexts_dict = (
        context.module_contexts_next_batch
        if context.version == 0
        else context.module_contexts
    )
    size = 0
    for module_ctx in contexts_dict.values():
        early_released: list[KeyedJaggedTensor] = getattr(
            module_ctx, "early_releasable_inputs", []
        )
        for kjt in early_released:
            size += kjt.clear_storage()
        early_released.clear()
    if size > 0:

        def get_kjt_size() -> str:
            msg: str = f"clear_releasable_inputs {size / 1024**3:.2f} GB"
            logger.info(msg)
            return msg

        one_time_logger.info(LazyStr(get_kjt_size))


def _clear_input_dist_tensors(context: TrainPipelineContext) -> None:
    """Free input tensor storage for all in-flight KJT AllToAll operations.

    Iterates over input_dist_tensors_requests and calls clear_inputs() on each
    KJTAllToAllTensorsAwaitable. This waits for the AllToAll collectives to
    complete and frees the input tensor storage early, before _wait_impl() is
    called during the model forward pass.
    """
    for request in context.input_dist_tensors_requests.values():
        if isinstance(request, KJTListAwaitable):
            for awaitable in request.awaitables:
                if isinstance(awaitable, KJTAllToAllTensorsAwaitable):
                    awaitable.clear_inputs()


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: Pipelineable,
    context: TrainPipelineContext,
) -> None:
    if context.version == 0:
        context.input_dist_splits_requests.clear()
        context.module_contexts_next_batch.clear()
        context.fused_splits_awaitables.clear()

    for module in pipelined_modules:
        forward = module.forward
        assert isinstance(
            forward,
            (
                PipelinedForward,
                PrefetchPipelinedForward,
                EmbeddingPipelinedForward,
                InSyncEmbeddingPipelinedForward,
                CPUEmbeddingPipelinedForward,
            ),
        )

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args, kwargs = forward.args.build_args_kwargs(batch)
        args, kwargs = module.preprocess_input(args, kwargs)

        # Start input distribution.
        module_ctx = module.create_context()
        if context.version == 0:
            context.module_contexts_next_batch[forward.name] = module_ctx
        else:
            context.module_contexts[forward.name] = module_ctx
        context.input_dist_splits_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )

    # All pipelined modules' input_dist calls are complete. Safe to clear deferred
    # KJT feature storage now — each sharded module has already permuted and extracted
    # its features, so the original input batch KJT storage can be cleared to free up HBM.
    _clear_releasable_inputs(context)

    _fuse_input_dist_splits(context)


def _start_embedding_lookup(
    module: ShardedModule,
    context: EmbeddingTrainPipelineContext,
    source_stream: Optional[torch.Stream],
    target_stream: Optional[torch.Stream],
    stream_context: Callable[..., AbstractContextManager[Any, Any]],
) -> None:
    # pyrefly: ignore[missing-attribute]
    module_context = context.module_contexts[module.forward.name]
    with stream_context(source_stream):
        # pyrefly: ignore[missing-attribute]
        kjt = context.input_dist_tensors_requests[module.forward.name].wait()

    if target_stream is not None:
        kjt.record_stream(target_stream)
        module_context.record_stream(target_stream)
    output_dist_out = module.compute_and_output_dist(module_context, kjt)
    # pyrefly: ignore[missing-attribute]
    context.embedding_a2a_requests[module.forward.name] = output_dist_out


def _fuse_input_dist_splits(context: TrainPipelineContext) -> None:
    names_per_pg = defaultdict(list)
    for name, request in context.input_dist_splits_requests.items():
        pg = None
        if isinstance(request, KJTListSplitsAwaitable):
            for awaitable in request.awaitables:
                if isinstance(awaitable, KJTSplitsAllToAllMeta):
                    pg = awaitable.pg
                    break
        names_per_pg[pg].append(name)

    for pg, names in names_per_pg.items():
        context.fused_splits_awaitables.append(
            (
                names,
                FusedKJTListSplitsAwaitable(
                    # pyrefly: ignore[bad-argument-type]
                    requests=[
                        context.input_dist_splits_requests[name] for name in names
                    ],
                    contexts=[
                        (
                            context.module_contexts_next_batch[name]
                            if context.version == 0
                            else context.module_contexts[name]
                        )
                        for name in names
                    ],
                    pg=pg,
                ),
            )
        )


def _jit_modules(module: torch.nn.Module, path: str, optional: bool = True) -> bool:
    sharded_children = set()
    for name, child in module.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _jit_modules(child, curr_path + ".", optional)
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, child in module.named_children():
            if name not in sharded_children:
                try:
                    jit_child = torch.jit.script(child)
                    setattr(module, name, jit_child)
                    logger.info(f"jit.script applied to {path + name}.")
                except Exception as error:
                    if not optional:
                        raise
                    else:
                        logger.info(
                            f"Warning: failed to jit.script {path + name}: {error}."
                        )

    return len(sharded_children) > 0


def _pipeline_detach_model(
    model: torch.nn.Module,
    pipelined_modules: List[ShardedModule],
    original_forwards: List[Callable[..., Any]],
    original_kjt_dist_forwards: List[
        Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
    ],
    pipelined_postprocs: List[PipelinedPostproc],
) -> None:
    # Replace pipelined module forward and input dist forward with original forward
    kjt_dists = []
    for mod, original_fwd in zip(pipelined_modules, original_forwards):
        mod.forward = original_fwd

        for _, child_module in mod.named_modules():
            if not hasattr(child_module, "_input_dists"):
                continue
            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    kjt_dists.append(input_dist._dist)
    assert len(kjt_dists) == len(
        original_kjt_dist_forwards
    ), f"Number of KJT dists ({len(kjt_dists)}) does not match number of kjt dist forwards provided ({len(original_kjt_dist_forwards)})"

    for kjt_dist, original_kjt_dist_fwd in zip(
        kjt_dists,
        original_kjt_dist_forwards,
    ):
        kjt_dist.forward = original_kjt_dist_fwd

    # Get underlying nn.Module
    if isinstance(model, DistributedModelParallel):
        model = model.module

    # Replace pipelined postproc modules with original postproc modules
    for postproc_mod in pipelined_postprocs:
        setattr(model, postproc_mod.fqn, postproc_mod.postproc_module)


def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TForwardContext,
    dist_stream: Optional[torch.Stream],
    batch: Optional[In] = None,
    apply_jit: bool = False,
    pipelined_forward: Type[BaseForward[TrainPipelineContext]] = PipelinedForward,
    pipeline_postproc: bool = False,
    default_stream: Optional[torch.Stream] = None,
) -> Tuple[
    List[ShardedModule],
    torch.nn.Module,
    List[Callable[..., Any]],
    List[PipelinedPostproc],
    List[str],
]:
    """
    This is a very important util function used by TorchRec's sparse-dist (and others) train pipeline.

    The high-level idea of the sparse-dist train pipeline is to extract the forward calls of the sharded
    modules (e.g., ShardedEBC, ShardedEC, etc.) from the model's forward call, so that the sparse-dist
    pipeline can apply some optimization technique like overlapping the comms (i.e., input_dist) with
    compute (e.g., dense-forward, emb-lookup, etc.). And this "extraction of sharded forward" is done by
    this `_rewrite_model` util function.

    currently the `_rewrite_model` function uses fx tracer to capture the graph of the sharded model,
    and find the "call_module" nodes for sharded modules.

    theoretically the ShardedModule takes a KJT as the only input (EBC, EC, etc.), it calls `_get_node_args`
    to
    """
    input_model = model
    # Get underlying sharded model (nn.Module) from DistributedModelParallel
    #   which will not be wrapped in DDP, FSDP, DMP, or any other parallelism wrappers.
    if isinstance(model, DistributedModelParallel):
        model = model.module

    # Collect a list of sharded modules.
    sharded_modules: Dict[str, ShardedModule] = {}  # fqn -> ShardedModule
    for name, m in model.named_modules():
        if isinstance(m, ShardedModule):
            sharded_modules[name] = m

    ## Trace a model. for more: https://pytorch.org/docs/stable/fx.html
    concrete_args = {}
    """
    concrete_args allows you to partially specialize your function, whether it’s to remove
    control flow or data structures.
    """

    # special handling of placeholder, adding meta/label to the PH node
    if batch:
        if hasattr(batch, "to_proxy"):
            # for some special models, it requires using "input" as the key for input
            # pyrefly: ignore[missing-attribute]
            concrete_args["inputs"] = copy.copy(batch).to_proxy()
        elif hasattr(batch, "to_proxy_tuple"):
            # when the model is pre-fx traced or dynamo exported, the inputs are already flattened,
            # and therefore we use tuple as concrete args that fx.trace will automatically match
            # with the argument names. We pass in the model for the caller side to customize the batch
            concrete_args = batch.to_proxy_tuple(model)

    tracer = Tracer(leaf_modules=_get_leaf_module_names(model))

    # When a compiled (torch.compile) module contains ShardedModules, the FX
    # tracer will trace into it (non-leaf), which triggers dynamo's eval-frame
    # hook. If error_on_nested_fx_trace is True, dynamo rejects the nested FX
    # trace and raises an error.  Temporarily patch the config to False so the
    # FX tracer can operate on compiled models regardless of the global setting.
    if torch._utils_internal.justknobs_check(
        "pytorch/torchrec:killswitch_rewrite_model_patch_nested_fx_trace",
    ):
        with torch._dynamo.config.patch(error_on_nested_fx_trace=False):
            graph = tracer.trace(model, concrete_args=concrete_args)
    else:
        graph = tracer.trace(model, concrete_args=concrete_args)

    # Select sharded modules, which are top-level in the forward call graph,
    # i.e. don't have input transformations, i.e. rely only on 'builtins.getattr'.
    pipelined_forwards = []
    original_forwards = []

    non_pipelined_sharded_modules = []

    args_helper = NodeArgsHelper(
        model, context, pipeline_postproc, default_stream, dist_stream
    )

    logger.info(
        f"pipeline_postproc is {'enabled' if pipeline_postproc else 'disabled'}"
    )
    for node in graph.nodes:
        # only work on the call_module node which is also a sharded module
        if node.op != "call_module" or node.target not in sharded_modules:
            continue

        total_num_args = len(node.args) + len(node.kwargs)
        # only work on node with input(s), we don't expect zero input count for sharded module
        if total_num_args == 0:
            logger.warning(f"Module '{node.target}' is a ShardedModule with zero input")
            continue

        # List[ArgInfo]: for rebuilding the input arguments, while the num verifies if missing any
        arg_info_list, num_found = args_helper.get_node_args(node)

        if num_found == total_num_args:
            logger.info(f"Module '{node.target}' will be pipelined")
            child = sharded_modules[node.target]
            original_forwards.append(child.forward)
            # Set pipelining flag on the child module
            child.is_pipelined = True
            # pyrefly: ignore[bad-assignment]
            child.forward = pipelined_forward(
                node.target,
                arg_info_list,
                child,
                context,
                dist_stream,
            )
            pipelined_forwards.append(child)
        else:
            logger.warning(
                f"Module '{node.target}' will NOT be pipelined, due to input modifications"
            )
            one_time_rank0_logger.warning(
                f"Module '{node.target}' will NOT be pipelined, due to input modifications"
            )
            non_pipelined_sharded_modules.append(node.target)

    # JIT script unsharded modules if applicable.
    if apply_jit:
        graph_model = torch.fx.GraphModule(model, graph)
        _jit_modules(graph_model, "")
        if isinstance(input_model, DistributedModelParallel):
            input_model.module = graph_model

    if non_pipelined_sharded_modules:
        logger.warning(
            "Sharded modules were not pipelined: %s. "
            + "This should be fixed for pipelining to work to the full extent.",
            ", ".join(non_pipelined_sharded_modules),
        )

    return (
        pipelined_forwards,
        input_model,
        original_forwards,
        list(args_helper.pipelined_postprocs),
        non_pipelined_sharded_modules,
    )


def _override_input_dist_forwards(
    pipelined_modules: List[ShardedModule],
) -> List[Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]]:
    """
    Overrides each input dist forward to support fusing the splits collective.
    NOTE: this can only be called after the input dists are initialized.
    """
    original_kjt_dist_forwards = []
    for module in pipelined_modules:
        for child_fqn, child_module in module.named_modules():
            if hasattr(child_module, "_has_uninitialized_input_dist"):
                assert (
                    not child_module._has_uninitialized_input_dist
                ), f"{child_fqn} has uninitialized input dist"

            if not hasattr(child_module, "_input_dists"):
                continue

            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    assert isinstance(input_dist._dist, KJTAllToAll)
                    original_kjt_dist_forwards.append(input_dist._dist.forward)
                    # pyrefly: ignore[bad-assignment]
                    input_dist._dist.forward = KJTAllToAllForward(
                        pg=input_dist._dist._pg,
                        splits=input_dist._dist._splits,
                        stagger=input_dist._dist._stagger,
                    )
    # pyrefly: ignore[bad-return]
    return original_kjt_dist_forwards


def get_h2d_func(batch: In, device: torch.device) -> Pipelineable:
    return batch.to(device, non_blocking=True)


class DataLoadingThread(Thread, Generic[In]):
    def __init__(
        self,
        device: torch.device,
        dataloader_iter: Iterator[In],
        to_device_non_blocking: bool,
        memcpy_stream_priority: int = 0,
        memcpy_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__(name="DataLoadingThread")
        self._stop: bool = False
        self.daemon = True  # Mark as daemon thread so that Python will not wait for it at shutdown.
        self._dataloader_iter = dataloader_iter
        self._buffer_empty_event: Event = Event()
        self._buffer_filled_event: Event = Event()
        if memcpy_stream is None:
            self._memcpy_stream: Optional[torch.Stream] = (
                torch.get_device_module(device).Stream(priority=memcpy_stream_priority)
                if device.type in ["cuda", "mtia"]
                else None
            )
        else:
            self._memcpy_stream = memcpy_stream
        self._device = device
        self._to_device_non_blocking = to_device_non_blocking
        self._buffered: Optional[In] = None
        self._buffer_empty_event.set()
        one_time_rank0_logger.info(
            f"{self.__class__.__name__} created with device={device}, "
            f"to_device_non_blocking={to_device_non_blocking}, "
            f"memcpy_stream_priority={memcpy_stream_priority}, "
            f"memcpy_stream={'provided' if memcpy_stream is not None else 'auto-created' if self._memcpy_stream is not None else 'None'}"
        )

    def run(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_available():
            # set the current device the same as the one used in the main thread
            torch.cuda.set_device(self._device)
        elif self._device.type == "mtia" and torch.mtia.is_available():
            # set the current device the same as the one used in the main thread
            torch.mtia.set_device(self._device)

        while not self._stop:
            self._buffer_empty_event.wait()
            # Set the filled event to unblock progress() and return.
            if self._stop:
                self._buffer_filled_event.set()
                return
            with record_function("## load_batch ##"):
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    self._stop = True
                    self._buffer_filled_event.set()
                    return
            with record_function("## copy_batch_to_gpu ##"):
                with torch.get_device_module(self._device).stream(self._memcpy_stream):
                    self._buffered = cast(
                        In,
                        batch.to(
                            self._device, non_blocking=self._to_device_non_blocking
                        ),
                    )
                self._buffer_empty_event.clear()
                self._buffer_filled_event.set()

    def stop(self) -> None:
        logger.info(f"{self.__class__.__name__}: Stopping data loading thread...")
        self._stop = True
        # Unblock any thread that are waiting for these events.
        self._buffer_filled_event.set()
        self._buffer_empty_event.set()
        logger.info(f"{self.__class__.__name__}: Data loading thread stopped.")

    def get_next_batch(self, none_throws: bool = False) -> Optional[In]:
        """
        Get the next batch from the buffer if threading is enabled, otherwise
        call load_next_batch directly.

        This function is not thread safe. We assume this is only invoked from
        the main thread in the training loop.
        """
        self._buffer_filled_event.wait()
        batch = self._buffered
        if batch is None:
            if none_throws:
                raise StopIteration
            return None
        self._buffered = None
        self._buffer_filled_event.clear()
        self._buffer_empty_event.set()
        return batch


def prefetch_embeddings(
    context: PrefetchTrainPipelineContext,
    pipelined_modules: List[ShardedModule],
    device: torch.device,
    stream_context: Callable[[Optional[torch.Stream]], torch.cuda.StreamContext],
    data_dist_stream: Optional[torch.Stream],
    forward_stream: Optional[torch.Stream],
) -> None:
    """
    Prefetches embeddings for the given batch.

    This function processes each sharded module by:
    1. Retrieving and waiting for the input distribution request for the module
    2. Ensuring proper stream synchronization for the resulting data
    3. Initiating the prefetch operation for the module

    Args:
        context: The prefetch pipeline context containing state information
        pipelined_modules: List of sharded modules to process
        device: The device to use for computation
        stream_context: Context manager for stream operations
        data_dist_stream: Stream used for data distribution operations
        forward_stream: Default stream for operations
    """

    if data_dist_stream is None:
        return

    cur_stream = torch.get_device_module(device).current_stream()

    for sharded_module in pipelined_modules:
        forward = sharded_module.forward
        assert isinstance(forward, PrefetchPipelinedForward)

        assert forward._name in context.input_dist_tensors_requests
        request = context.input_dist_tensors_requests.pop(forward._name)
        assert isinstance(request, Awaitable)

        with record_function(f"## _prefetch_embeddings {context.index} ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with stream_context(data_dist_stream):
                dist_input = request.wait()
                assert isinstance(
                    dist_input, (torch.Tensor, Multistreamable)
                ), f"{type(dist_input)} must implement Multistreamable interface"

        # Ensure prefetch stream waits for data_dist stream's GPU work
        # before launching prefetch kernels that read the dist output.
        cur_stream.wait_stream(data_dist_stream)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        module_context = context.module_contexts[forward._name]

        dist_input.record_stream(cur_stream)
        module_context.record_stream(cur_stream)
        if forward_stream is not None:
            dist_input.record_stream(forward_stream)
            module_context.record_stream(forward_stream)

        sharded_module.prefetch(
            ctx=module_context,
            dist_input=dist_input,
            forward_stream=forward_stream,
        )
        context.module_input_post_prefetch[forward._name] = dist_input
        context.module_contexts_post_prefetch[forward._name] = (
            context.module_contexts.pop(forward._name)
        )


def _prefetch_embeddings(
    batch: In,
    context: PrefetchTrainPipelineContext,
    pipelined_modules: List[ShardedModule],
    device: torch.device,
    stream_context: Callable[[Optional[torch.Stream]], torch.cuda.StreamContext],
    data_dist_stream: Optional[torch.Stream],
    default_stream: Optional[torch.Stream],
) -> Dict[str, KJTList]:
    data_per_sharded_module = {}
    for sharded_module in pipelined_modules:
        forward = sharded_module.forward
        assert isinstance(forward, PrefetchPipelinedForward)
        assert forward._name in context.input_dist_tensors_requests
        request = context.input_dist_tensors_requests.pop(forward._name)
        assert isinstance(request, Awaitable)
        with record_function(f"## _prefetch_embeddings {context.index} ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with stream_context(data_dist_stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        module_context = context.module_contexts[forward._name]
        if data_dist_stream is not None:
            torch.get_device_module(device).current_stream().wait_stream(
                data_dist_stream
            )
            cur_stream = torch.get_device_module(device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            if default_stream:
                data.record_stream(default_stream)

            module_context.record_stream(cur_stream)
            if default_stream:
                module_context.record_stream(default_stream)

        sharded_module.prefetch(
            ctx=module_context,
            dist_input=data,
            forward_stream=default_stream,
        )
        data_per_sharded_module[forward._name] = data
    return data_per_sharded_module


@contextlib.contextmanager
def use_context_for_postprocs(
    pipelined_postprocs: List[PipelinedPostproc],
    next_batch_context: TrainPipelineContext,
) -> Generator[None, None, None]:
    """
    Temporarily set pipelined postproc context for next iter to populate cache.
    """
    # Save original context for model fwd
    original_contexts = [p.get_context() for p in pipelined_postprocs]

    # Temporarily set context for next iter to populate cache
    for postproc_mod in pipelined_postprocs:
        postproc_mod.set_context(next_batch_context)

    yield

    # Restore context for model fwd
    for module, context in zip(pipelined_postprocs, original_contexts):
        module.set_context(context)


class FutureDeque(deque):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        one_time_rank0_logger.info("Creating FutureDeque")

    def __getitem__(self, index: Any) -> Any:
        item = super().__getitem__(index)
        if isinstance(item, Future):
            resolved = item.result()
            super().__setitem__(index, resolved)
            return resolved
        return item

    def pop(self) -> Any:
        item = super().pop()
        if isinstance(item, Future):
            return item.result()
        return item

    def popleft(self) -> Any:
        item = super().popleft()
        if isinstance(item, Future):
            return item.result()
        return item
