#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: Do NOT add `from __future__ import annotations` here.
# This module is loaded inside a torch.package at model-publish time. Combining
# PEP 563 (string annotations) with @dataclass on Python 3.12 hits
# https://github.com/python/cpython/issues/115258 — `dataclass._is_type` does
# `sys.modules.get(cls.__module__).__dict__`, which returns None for
# torch.package-synthetic module names ("<torch_package_N>.…") and crashes with
# AttributeError. Keeping annotations as runtime objects avoids that path.
# This is also why PECGradientApply is declared before PECAll2AllSeqInfo: the
# fields referencing it must be real annotations, not string forward refs.

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.pec_collision_handlers import OverlapSplits
from torchrec.distributed.types import (
    Awaitable,
    QuantizedCommCodec,
    QuantizedCommCodecs,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _grad_dist(
    grad: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    embedding_dim: int,
    pg: dist.ProcessGroup,
    async_op: bool = True,
    codec: QuantizedCommCodec | None = None,
    qcomm_ctx: Any = None,
) -> Tuple[dist.Work, torch.Tensor]:
    """Starts gradient reverse AllToAll. Shared by the OL and NOL partitions.

    Args:
        grad: gradient tensor to send.
        input_splits: per-rank send counts (number of rows).
        output_splits: per-rank receive counts (number of rows).
        embedding_dim: embedding dimension.
        pg: process group.
        async_op: whether to run asynchronously.
        codec: backward communication codec, if configured.
        qcomm_ctx: per-partition context shared by encoding and decoding.

    Returns:
        (work_handle, output_buffer), in wire format when quantized.
    """
    output_split_sizes = [s * embedding_dim for s in output_splits]
    input_split_sizes = [s * embedding_dim for s in input_splits]
    grad = grad.contiguous().view(-1)
    if codec is not None:
        grad = codec.encode(grad, qcomm_ctx)
        input_split_sizes = [
            codec.calc_quantized_size(s, qcomm_ctx) for s in input_split_sizes
        ]
        output_split_sizes = [
            codec.calc_quantized_size(s, qcomm_ctx) for s in output_split_sizes
        ]

    output_buffer = torch.empty(
        sum(output_split_sizes),
        device=grad.device,
        dtype=grad.dtype,
    )

    req = dist.all_to_all_single(
        output_buffer,
        grad,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
        async_op=async_op,
    )
    assert req is not None

    return req, output_buffer


def _recat_gradient(
    grad: torch.Tensor,
    grad_recat: torch.Tensor,
    grad_recat_lengths: torch.Tensor,
) -> torch.Tensor:
    """Restores feature order from the source-rank order returned by AllToAll."""
    permute = (
        torch.ops.fbgemm.permute_1D_sparse_data
        if grad_recat_lengths.dim() == 1
        else torch.ops.fbgemm.permute_2D_sparse_data
    )
    _, reordered, _ = permute(
        grad_recat,
        grad_recat_lengths,
        grad.flatten(),
        None,
        grad.numel(),
    )
    return reordered.view_as(grad)


class PECGradientApply:
    """Handles gradient AllToAll and application to TBE for one partition.

    Created by PECAll2AllSeqWait.backward — one for OL, one for NOL.
    Short-lived: exists for one batch, consumed by apply().

    Lifecycle:
        1. __init__: holds the re-split gradient
        2. dist(): starts async gradient AllToAll
        3. apply(): waits AllToAll, re-lookups features in TBE, calls
           embs.backward(grad) to push gradient through TBE kernel

    dist() is called by the pipeline via grad_dist for both partitions —
    OL before the optimizer step, NOL deferred to the next batch.
    """

    def __init__(
        self,
        grad: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        # pyrefly: ignore [missing-attribute]
        pg: dist.ProcessGroup,
        codecs: QuantizedCommCodecs | None = None,
    ) -> None:
        self._grad = grad
        self._input_splits = input_splits
        self._output_splits = output_splits
        self._pg = pg
        self._codec = codecs.backward if codecs is not None else None
        self._qcomm_ctx: Any = None
        self._work: dist.Work | None = None
        self._grad_buffer: torch.Tensor | None = None

    def dist(self) -> None:
        """Starts async gradient reverse AllToAll. Idempotent — no-op if already started."""
        if self._work is not None:
            return

        embedding_dim = self._grad.shape[1]

        if self._codec is not None:
            self._qcomm_ctx = self._codec.create_context()

        self._work, self._grad_buffer = _grad_dist(
            self._grad,
            input_splits=self._input_splits,
            output_splits=self._output_splits,
            embedding_dim=embedding_dim,
            pg=self._pg,
            codec=self._codec,
            qcomm_ctx=self._qcomm_ctx,
        )
        self._grad = None  # type: ignore[assignment]

    def apply(
        self,
        features: KeyedJaggedTensor,
        lookup_fn: Callable[[KeyedJaggedTensor], torch.Tensor],
        embedding_dim: int,
        grad_recat: torch.Tensor | None = None,
        grad_recat_lengths: torch.Tensor | None = None,
    ) -> None:
        """Waits AllToAll and applies gradient to TBE via re-lookup + backward."""
        assert self._work is not None
        assert self._grad_buffer is not None

        self._work.wait()
        grad = self._grad_buffer

        if self._codec is not None:
            grad = self._codec.decode(grad, self._qcomm_ctx)

        if features.values().numel() > 0:
            if grad_recat is not None:
                assert grad_recat_lengths is not None
                grad = _recat_gradient(grad, grad_recat, grad_recat_lengths)

            with torch.enable_grad():
                embs = lookup_fn(features).view(-1, embedding_dim)
                if embs.numel() > 0:
                    embs.backward(grad.view(-1, embedding_dim))
        self._work = None
        self._grad_buffer = None
        self._qcomm_ctx = None


@dataclass
class PECAll2AllSeqInfo:
    """Metadata for PEC backward gradient re-split and distribution.

    Created by merge_partitioned_embeddings during forward. Backward fields
    are populated before loss.backward() from the independently waited
    backward overlap result. PECAll2AllSeqWait reads them during backward to
    re-split gradients and create PECGradientApply instances.

    Attributes:
        forward_permute: Permute tensor for merging [ol, nol] → original order.
        backward_ol_permute: Original-order indices for OL gradient split.
        backward_nol_permute: Original-order indices for NOL gradient split.
        backward_splits: Per-rank OL/NOL split sizes for gradient AllToAll.
        pg: Process group for gradient AllToAll.
        codecs: Sequence embedding communication codecs for this sharding.
        ol_grad_apply: OL gradient applier (set by PECAll2AllSeqWait.backward).
            Pipeline calls dist() via grad_dist.
        nol_grad_apply: NOL gradient applier (set by PECAll2AllSeqWait.backward).
            Pipeline calls dist() via grad_dist.
    """

    forward_permute: torch.Tensor
    backward_ol_permute: torch.Tensor | None = None
    backward_nol_permute: torch.Tensor | None = None
    backward_splits: OverlapSplits | None = None
    pg: dist.ProcessGroup | None = None
    codecs: QuantizedCommCodecs | None = None

    # Gradient appliers — set by PECAll2AllSeqWait.backward
    ol_grad_apply: PECGradientApply | None = None
    nol_grad_apply: PECGradientApply | None = None


class PECAll2AllSeqWait(torch.autograd.Function):
    """PEC gradient intercept — merges embeddings in forward, re-splits
    gradients in backward.

    Forward: merges OL/NOL embeddings using forward_permute.

    Backward: splits gradient into OL/NOL via backward_ol_permute and
    backward_nol_permute (pre-composed with recat, so index_select
    produces rank-major order directly). Creates PECGradientApply
    instances for each partition. Both OL and NOL dist() are started by
    the pipeline via grad_dist (not here), so gradient communication is
    fully pipeline-controlled.

    Gradient propagation stops here (returns None) — PECGradientApply
    handles the AllToAll wait and TBE backward application.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        autograd_ctx: PECAll2AllSeqInfo,
        ol_embs: torch.Tensor,
        nol_embs: torch.Tensor,
        grad_anchor: torch.Tensor,
    ) -> torch.Tensor:
        ctx.autograd_ctx = autograd_ctx  # pyre-ignore[16]

        merged_embs = torch.cat([ol_embs, nol_embs], dim=0)
        return torch.index_select(merged_embs, 0, autograd_ctx.forward_permute)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[None, None, None, None]:
        autograd_ctx: PECAll2AllSeqInfo = ctx.autograd_ctx  # pyre-ignore[16]
        splits = autograd_ctx.backward_splits

        assert autograd_ctx.backward_ol_permute is not None
        assert autograd_ctx.backward_nol_permute is not None
        assert splits is not None
        assert autograd_ctx.pg is not None

        # Split gradient into ol/nol. Permute indices are pre-composed
        # with recat, so index_select produces rank-major order directly.
        ol_grad = grad_output.index_select(0, autograd_ctx.backward_ol_permute)
        nol_grad = grad_output.index_select(0, autograd_ctx.backward_nol_permute)

        # Create OL/NOL appliers; the pipeline starts both AllToAlls via
        # grad_dist (OL before optimizer, NOL deferred to the next batch).
        autograd_ctx.ol_grad_apply = PECGradientApply(
            ol_grad,
            input_splits=splits.input_splits[0],
            output_splits=splits.output_splits[0],
            pg=autograd_ctx.pg,
            codecs=autograd_ctx.codecs,
        )

        autograd_ctx.nol_grad_apply = PECGradientApply(
            nol_grad,
            input_splits=splits.input_splits[1],
            output_splits=splits.output_splits[1],
            pg=autograd_ctx.pg,
            codecs=autograd_ctx.codecs,
        )

        return (None, None, None, None)


class PECGradUpdateAwaitable(Awaitable[None]):
    """Awaitable that applies gradient to TBE on wait.

    Wraps a PECGradientApply instance. On wait(), calls apply() which
    waits on the AllToAll and pushes gradient through TBE's backward kernel
    via re-lookup. The caller-provided context installs the originating batch's
    learning rates during gradient application and restores the current rates.
    """

    def __init__(
        self,
        applier: PECGradientApply,
        features: KeyedJaggedTensor,
        lookup_fn: Callable[[KeyedJaggedTensor], torch.Tensor],
        embedding_dim: int,
        learning_rate_context: Callable[[], AbstractContextManager[None]],
        grad_recat: torch.Tensor | None,
        grad_recat_lengths: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self._applier = applier
        self._features = features
        self._lookup_fn = lookup_fn
        self._embedding_dim = embedding_dim
        self._learning_rate_context = learning_rate_context
        self._grad_recat = grad_recat
        self._grad_recat_lengths = grad_recat_lengths

    def _wait_impl(self) -> None:
        with self._learning_rate_context():
            self._applier.apply(
                self._features,
                self._lookup_fn,
                self._embedding_dim,
                self._grad_recat,
                self._grad_recat_lengths,
            )
