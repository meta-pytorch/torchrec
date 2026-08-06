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
    module_input_post_prefetch: Dict[str, Multistreamable | torch.Tensor] = field(
        default_factory=dict
    )
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


@dataclass
class PECTrainPipelineContext(EmbeddingTrainPipelineContext):
    """Context for TrainPipelinePEC.

    Non-PEC precomputed embeddings use the inherited embedding_a2a_requests
    (+ embedding_tensors etc. read by the InSync forward path). PEC module
    contexts and raw input_dist requests use the inherited module_contexts /
    input_dist_tensors_requests — PEC modules go through the shared SDD. The
    fields below are PEC-only; each dict is keyed by module FQN.
    """

    # PEC features after input_dist (KJTList, one KJT per sharding group), waited
    # inside overlap_dist. Held one batch so the next batch's overlap_dist can use
    # it as prev.
    pec_dist_inputs: Dict[str, Any] = field(default_factory=dict)

    # List[ForwardPartitionContext] / List[BackwardPartitionContext] from
    # overlap_dist, one entry per sharding group.
    pec_forward_ctxs: Dict[str, Any] = field(default_factory=dict)
    pec_backward_ctxs: Dict[str, Any] = field(default_factory=dict)

    # OL/NOL output-dist awaitables from compute (List per sharding group),
    # consumed by merge in forward.
    pec_ol_awaitables: Dict[str, Any] = field(default_factory=dict)
    pec_nol_awaitables: Dict[str, Any] = field(default_factory=dict)

    # The previous batch's deferred NOL grad inputs, attached at the end of the
    # previous batch's progress: (module_ctx, nol_features_per_group). This
    # batch's progress runs the NOL grad A2A + apply from it. module_ctx carries
    # the nol_grad_apply applier (set during the prev batch's backward); both it
    # and the nol_features stay alive because this context holds the refs.
    pec_deferred_nol_grad: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPUEmbeddingTrainPipelineContext(EmbeddingTrainPipelineContext):
    dense_gpu_device: str = field(default_factory=str)
    # Populated by Stage 2 (copy_to_gpu): embedding results already on GPU, keyed by module FQN
    gpu_embedding_outputs: Dict[str, Any] = field(default_factory=dict)
