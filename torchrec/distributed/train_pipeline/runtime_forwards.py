#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
from typing import Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torch import distributed as dist
from torch.profiler import record_function
from torchrec.distributed.embedding_sharding import KJTSplitsAllToAllMeta
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.pipeline_context import (
    EmbeddingTrainPipelineContext,
    PrefetchTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.types import CallArgs
from torchrec.distributed.types import Awaitable, LazyNoWait
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


class PipelinedForward(BaseForward[TrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSparseDist
    """

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name in self._context.input_dist_tensors_requests
        ), f"Invalid PipelinedForward usage, input_dist of {self._name} is not available, probably consumed by others"
        # we made a basic assumption that an embedding module (EBC, EC, etc.) should only be evoked only
        # once in the model's forward pass. For more details: https://github.com/meta-pytorch/torchrec/pull/3294
        request = self._context.input_dist_tensors_requests.pop(self._name)
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.get_device_module(self._device).stream(self._stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        ctx = self._context.module_contexts.pop(self._name)

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


class EmbeddingPipelinedForward(BaseForward[EmbeddingTrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSemiSync
    """

    def __call__(
        self,
        # pyre-ignore
        *input,
        # pyre-ignore
        **kwargs,
    ) -> Union[
        Awaitable[EmbeddingModuleRetType],
        Tuple[
            Awaitable[EmbeddingModuleRetType], Awaitable[Optional[KeyedJaggedTensor]]
        ],
    ]:
        assert (
            self._name in self._context.embedding_a2a_requests
        ), f"Invalid PipelinedForward usage, input_dist of {self._name} is not available, probably consumed by others"
        # we made a basic assumption that an embedding module (EBC, EC, etc.) should only be evoked only
        # once in the model's forward pass. For more details: https://github.com/meta-pytorch/torchrec/pull/3294

        ctx = self._context.module_contexts.pop(self._name)
        cur_stream = torch.get_device_module(self._device).current_stream()

        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            ctx.record_stream(cur_stream)

        awaitable = self._context.embedding_a2a_requests.pop(self._name)
        # in case of MC modules
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

        self.detach_embeddings(embeddings=embeddings, cur_stream=cur_stream)

        if is_mc_module:
            return (LazyNoWait(embeddings), LazyNoWait(remapped_kjts))
        else:
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
            # pyre-fixme[6]: For 1st argument expected `Stream` but got `Stream`.
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


class InSyncEmbeddingPipelinedForward(EmbeddingPipelinedForward):
    """
    This pipeline is used in TrainPipelineFusedSparseDist
    """

    def detach_embeddings(
        self,
        embeddings: Union[Dict[str, JaggedTensor], KeyedTensor],
        cur_stream: torch.Stream,
    ) -> None:
        # doing nothing
        pass


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

    # pyre-ignore [2, 24]
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

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        if not self._compute_and_output_dist_awaitable:
            raise Exception(
                "compute_and_output_dist must be called before __call__",
            )
        return self._compute_and_output_dist_awaitable


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
            input_splits = input.dist_splits(self._splits)
            device = input.values().device
            splits_tensors = [
                torch.tensor(splits, device=device) for splits in input_splits
            ]
            if not input.variable_stride_per_key():
                splits_tensors.append(
                    torch.tensor([input.stride()] * self._pg.size(), device=device)
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
            )
