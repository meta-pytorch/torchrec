#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.pec_embedding import (
    BackwardPartitionContext,
    ForwardPartitionContext,
    ShardedPECEmbeddingCollection,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    PECTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import PECPipelinedForward
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.streamable import Pipelineable

In = Pipelineable
Out = object


class TrainPipelinePEC(TrainPipelineSparseDist[In, Out]):
    """Compute-ahead training pipeline for models with PEC EmbeddingCollections."""

    # pyrefly: ignore [bad-override]
    _pipelined_forward_type = PECPipelinedForward

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=PECTrainPipelineContext,
            custom_model_fwd=custom_model_fwd,
        )
        self._pec_modules: Dict[str, ShardedPECEmbeddingCollection] = {}
        # Reusable events ordering data-dist-stream work before the main-stream
        # compute that consumes it: _input_dist_event for non-PEC input-dist
        # waits -> embedding lookup; _overlap_dist_event for overlap_dist ->
        # NOL compute.
        self._input_dist_event: torch.cuda.Event = torch.cuda.Event()
        self._overlap_dist_event: torch.cuda.Event = torch.cuda.Event()
        # Deferred NOL grad is carried on PECTrainPipelineContext.pec_deferred_nol_grad,
        # attached to the next batch's context -- no pipeline-level stash.

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Any = PECPipelinedForward,  # pyre-ignore[2]
    ) -> None:
        # Base _pipeline_model has no early return -- it always runs _rewrite_model
        # and (re)populates _pipelined_modules -- so rebuilding _pec_modules right
        # after super() is always valid and stays in sync whenever the model is
        # (re)pipelined. (Override _pipeline_model, NOT _init_pipelined_modules,
        # which DOES early-return when already pipelined and would skip this build.)
        super()._pipeline_model(batch, context, pipelined_forward)

        self._pec_modules = {
            m.forward.name: m  # pyre-ignore[16]
            for m in self._pipelined_modules
            if isinstance(m, ShardedPECEmbeddingCollection)
        }

    @staticmethod
    def _record_stream_forward_ctx(
        ctx: ForwardPartitionContext, stream: torch.cuda.streams.Stream
    ) -> None:
        ctx.ol_features.record_stream(stream)
        ctx.nol_features.record_stream(stream)
        ctx.permute.record_stream(stream)

    @staticmethod
    def _record_stream_backward_ctx(
        ctx: BackwardPartitionContext, stream: torch.cuda.streams.Stream
    ) -> None:
        ctx.ol_features.record_stream(stream)
        ctx.nol_features.record_stream(stream)
        ctx.ol_permute.record_stream(stream)
        ctx.nol_permute.record_stream(stream)
