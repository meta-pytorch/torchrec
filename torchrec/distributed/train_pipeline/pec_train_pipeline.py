#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.pec_embedding import (
    BackwardPartitionContext,
    ForwardPartitionContext,
    OverlapDistOutput,
    PECEmbeddingCollectionContext,
    ShardedPECEmbeddingCollection,
)
from torchrec.distributed.train_pipeline.pipeline_context import (
    PECTrainPipelineContext,
    TrainPipelineContext,
)
from torchrec.distributed.train_pipeline.runtime_forwards import PECPipelinedForward
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.types import LazyAwaitable
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

    def _wait_pec_overlap_results(
        self,
        results: List[LazyAwaitable[OverlapDistOutput]],
        default_stream: torch.cuda.streams.Stream,
    ) -> Tuple[List[ForwardPartitionContext], List[BackwardPartitionContext]]:
        """Waits the per-group overlap awaitables and records them on default_stream.

        overlap_dist returns one OverlapDistOutput per sharding group; the
        returned lists stay aligned with the per-group order of ec._lookups
        (which merge / grad_dist zip against). A group contributes to forward
        only when forward is present (absent for the last batch) and to backward
        only when backward is present (absent for the first batch).

        Args:
            results: one OverlapDistOutput awaitable per sharding group.
            default_stream: stream the partition tensors are consumed on.

        Returns:
            (forward_ctxs, backward_ctxs) as per-group lists.
        """
        fwd_ctxs: List[ForwardPartitionContext] = []
        bwd_ctxs: List[BackwardPartitionContext] = []
        for result in results:
            fwd_ctx, bwd_ctx = result.wait()
            if fwd_ctx is not None:
                self._record_stream_forward_ctx(fwd_ctx, default_stream)
                self._precache_length_per_key(fwd_ctx)
                fwd_ctxs.append(fwd_ctx)
            if bwd_ctx is not None:
                self._record_stream_backward_ctx(bwd_ctx, default_stream)
                self._precache_length_per_key(bwd_ctx)
                bwd_ctxs.append(bwd_ctx)
        return fwd_ctxs, bwd_ctxs

    @staticmethod
    def _precache_length_per_key(ctx: object) -> None:
        """Forces length_per_key / offset_per_key on a partition ctx's OL/NOL KJTs.

        split_kjt_by_values_mask builds these KJTs bare, so the first access
        (the OL/NOL compute lookup and the grad-apply re-lookup) would otherwise
        run _maybe_compute_length_per_key -> a .tolist() device sync. We are
        inside overlap_dist's data-dist stream context here, so doing it now syncs
        only on the short local split rather than draining the main-stream
        forward/backward queue on the critical path; the cached lists persist on
        the KJT objects (including across the NOL deferral) for the consumers.
        """
        ctx.ol_features.sync()  # pyre-ignore[16]
        ctx.nol_features.sync()  # pyre-ignore[16]

    @staticmethod
    def _pec_module_ctx(
        ctx: PECTrainPipelineContext, name: str
    ) -> PECEmbeddingCollectionContext:
        """Returns name's PEC module context.

        module_contexts is typed as Multistreamable; PEC modules always store a
        PECEmbeddingCollectionContext, so the cast is localized here.
        """
        return cast(PECEmbeddingCollectionContext, ctx.module_contexts[name])

    def _pec_overlap_dist(
        self,
        current_ctx: Optional[PECTrainPipelineContext],
        prev_ctx: Optional[PECTrainPipelineContext],
    ) -> None:
        """Runs overlap_dist for current_ctx against prev_ctx, for every PEC module.

        Mirrors ShardedPECEmbeddingCollection.overlap_dist's three positions,
        selected by which context is None:
          - first batch:  current_ctx set, prev_ctx None -> forward only
          - normal batch: both set                       -> forward + backward
          - finalize:     current_ctx None, prev_ctx set -> backward only (the
            last batch has no successor, so all its values are NOL)

        Forward partition contexts attach to current_ctx; backward partition
        contexts attach to prev_ctx (backward of batch N is defined by its
        overlap with N+1). When current_ctx is set, its PEC features are waited
        from the inherited input_dist_tensors_requests (popped so the base
        forward path won't touch them) and stashed in pec_dist_inputs to serve
        as the next batch's prev.

        The mask AllToAlls launch on the data-dist stream (owned here); the
        produced tensors are recorded against the default compute stream they
        are consumed on. On completion it records _overlap_dist_event so a
        main-stream consumer (NOL compute) can order against the produced
        forward contexts via wait_event.

        Args:
            current_ctx: context of the batch being distributed (None at finalize).
            prev_ctx: context of the previous batch (None for the first batch).
        """
        default_stream = torch.get_device_module(self._device).current_stream()
        # pyrefly: ignore [bad-argument-type]
        with self._stream_context(self._data_dist_stream):
            for name, pec in self._pec_modules.items():
                dist_input = None
                current_module_ctx = None
                if current_ctx is not None:
                    dist_input = current_ctx.input_dist_tensors_requests.pop(
                        name
                    ).wait()
                    current_ctx.pec_dist_inputs[name] = dist_input
                    current_module_ctx = self._pec_module_ctx(current_ctx, name)

                prev_module_ctx = None
                prev_dist_input = None
                if prev_ctx is not None:
                    prev_module_ctx = self._pec_module_ctx(prev_ctx, name)
                    prev_dist_input = prev_ctx.pec_dist_inputs[name]

                results = pec.overlap_dist(
                    ctx=current_module_ctx,
                    dist_input=dist_input,
                    prev_ctx=prev_module_ctx,
                    prev_dist_input=prev_dist_input,
                )
                fwd_ctxs, bwd_ctxs = self._wait_pec_overlap_results(
                    results, default_stream
                )

                if current_ctx is not None:
                    current_ctx.pec_forward_ctxs[name] = fwd_ctxs
                if bwd_ctxs and prev_ctx is not None:
                    prev_ctx.pec_backward_ctxs[name] = bwd_ctxs

            # Mark overlap_dist's data-dist work complete so main-stream
            # consumers (NOL compute) can order against it via wait_event.
            self._overlap_dist_event.record(self._data_dist_stream)

    def _compute_and_output_dist(self, context: PECTrainPipelineContext) -> None:
        """Runs compute + output_dist for non-PEC pipelined modules.

        The input-dist AllToAlls are waited on the data-dist stream; the main
        stream is then ordered after them via the data-dist event and runs the
        embedding lookups, storing each output awaitable in
        context.embedding_a2a_requests (consumed by the inherited InSync forward
        path). The waited KJTs / module contexts are recorded on the main stream
        so the allocator does not reuse their data-dist memory mid-lookup. PEC
        modules are skipped -- their OL/NOL compute is done manually (see
        _pec_ol_compute / _pec_nol_compute).

        Args:
            context: context whose input_dist tensors are looked up.
        """
        main_stream = torch.get_device_module(self._device).current_stream()
        non_pec_modules = [
            module
            for module in self._pipelined_modules
            if not isinstance(module, ShardedPECEmbeddingCollection)
        ]

        # Wait the input-dist AllToAlls on the data-dist stream.
        waited_kjts: Dict[str, Any] = {}
        # pyrefly: ignore [bad-argument-type]
        with self._stream_context(self._data_dist_stream):
            for module in non_pec_modules:
                name = module.forward.name  # pyre-ignore[16]
                waited_kjts[name] = context.input_dist_tensors_requests[name].wait()

        # Order the main stream after the waits, then look up on the main stream.
        self._input_dist_event.record(self._data_dist_stream)
        main_stream.wait_event(self._input_dist_event)

        for module in non_pec_modules:
            name = module.forward.name  # pyre-ignore[16]
            kjt = waited_kjts[name]
            kjt.record_stream(main_stream)
            module_ctx = context.module_contexts[name]
            module_ctx.record_stream(main_stream)
            context.embedding_a2a_requests[name] = module.compute_and_output_dist(
                module_ctx, kjt
            )

    def _pec_ol_compute(self, context: PECTrainPipelineContext) -> None:
        """Computes + output_dist the OL partition for every PEC module.

        Runs in the same progress as forward (fresh weights). Stores per-group
        awaitables in context.pec_ol_awaitables, consumed by merge in forward.

        Args:
            context: context holding the forward partition contexts.
        """
        for name, pec in self._pec_modules.items():
            fwd_ctxs: List[ForwardPartitionContext] = context.pec_forward_ctxs[name]
            context.pec_ol_awaitables[name] = pec.compute_and_output_dist_in_partition(
                self._pec_module_ctx(context, name),
                [fc.ol_features for fc in fwd_ctxs],
                [fc.splits for fc in fwd_ctxs],
                is_overlapped=True,
            )

    def _pec_nol_compute(self, context: PECTrainPipelineContext) -> None:
        """Computes + output_dist the NOL partition for every PEC module.

        Runs one batch ahead (stale weights are safe for NOL). Stores per-group
        awaitables in context.pec_nol_awaitables, consumed by merge in forward.

        Args:
            context: context holding the forward partition contexts.
        """
        for name, pec in self._pec_modules.items():
            fwd_ctxs: List[ForwardPartitionContext] = context.pec_forward_ctxs[name]
            context.pec_nol_awaitables[name] = pec.compute_and_output_dist_in_partition(
                self._pec_module_ctx(context, name),
                [fc.nol_features for fc in fwd_ctxs],
                [fc.splits for fc in fwd_ctxs],
                is_overlapped=False,
            )
