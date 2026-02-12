#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torchrec.distributed.embedding_sharding import FusedKJTListSplitsAwaitable
from torchrec.distributed.types import Awaitable, LazyAwaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable, Pipelineable

logger: logging.Logger = logging.getLogger(__name__)


In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")


@dataclass
class TrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        input_dist_splits_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        input_dist_tensors_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the tensors awaitable stage, which occurs after calling `wait()`
            on the splits awaitable.
        output_dist_embeddings_requests (Dict[str, Awaitable[Any]]): Stores output dist
            awaitables for pipelined modules, keyed by module FQN. Used for registering
            backward hooks on the output dist tensors during the backward pass.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
        module_contexts_next_batch (Dict[str, Multistreamable]): Stores module contexts
            from the input dist for the next batch. (only for version 0)
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
        event: Optional[torch.cuda.Event]: Event to record the completion of this stage
        index: Optional[int]: Index of the current batch.
        version: int = 0; support for backward compatiblity
    """

    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    output_dist_embeddings_requests: Dict[str, Awaitable[Any]] = field(
        default_factory=dict
    )
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )  # deprecated: to support legacy code
    fused_splits_awaitables: List[Tuple[List[str], FusedKJTListSplitsAwaitable]] = (
        field(default_factory=list)
    )
    events: List[torch.Event] = field(default_factory=list)
    postproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = (
        0  # 1 is current version, 0 is deprecated but supported for backward compatibility
    )


@dataclass
class PrefetchTrainPipelineContext(TrainPipelineContext):
    module_input_post_prefetch: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_post_prefetch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_input_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_contexts_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )


@dataclass
class EmbeddingTrainPipelineContext(TrainPipelineContext):
    embedding_a2a_requests: Dict[
        str,
        Union[
            LazyAwaitable[Multistreamable],
            # ManagedCollisionEC/EBC returns tuple of awaitables
            Tuple[
                LazyAwaitable[KeyedTensor], LazyAwaitable[Optional[KeyedJaggedTensor]]
            ],
        ],
    ] = field(default_factory=dict)
    embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)
    embedding_features: List[List[Union[str, List[str]]]] = field(default_factory=list)
    detached_embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)
