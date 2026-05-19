#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.pec_collision_handlers import OverlapSplits
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class PECAll2AllSeqInfo:
    """Metadata for PEC backward gradient re-split and distribution.

    Created by merge_partitioned_embeddings during forward. When
    backward_ctxs is passed to merge, backward fields are populated
    automatically. Read by PECAll2AllSeqWait during backward to re-split
    gradients and create PECGradientApply instances.

    Attributes:
        forward_permute: Permute tensor for merging [ol, nol] → original order.
        backward_ol_permute: Original-order indices for OL gradient split.
        backward_nol_permute: Original-order indices for NOL gradient split.
        backward_splits: Per-rank OL/NOL split sizes for gradient AllToAll.
        pg: Process group for gradient AllToAll.
        ol_grad_apply: OL gradient applier (set by PECAll2AllSeqWait.backward).
            dist() already called.
        nol_grad_apply: NOL gradient applier (set by PECAll2AllSeqWait.backward).
            Pipeline calls dist() later.
    """

    forward_permute: torch.Tensor
    backward_ol_permute: torch.Tensor | None = None
    backward_nol_permute: torch.Tensor | None = None
    backward_splits: OverlapSplits | None = None
    pg: dist.ProcessGroup | None = None

    # Gradient appliers — set by PECAll2AllSeqWait.backward
    ol_grad_apply: "PECGradientApply | None" = None
    nol_grad_apply: "PECGradientApply | None" = None


def _grad_dist(
    grad: torch.Tensor,
    input_splits: List[int],
    output_splits: List[int],
    embedding_dim: int,
    pg: dist.ProcessGroup,
    async_op: bool = True,
) -> Tuple[dist.Work, torch.Tensor]:
    """Starts gradient reverse AllToAll. Shared by OL (autograd) and NOL (pipeline).

    Args:
        grad: gradient tensor to send.
        input_splits: per-rank send counts (number of rows).
        output_splits: per-rank receive counts (number of rows).
        embedding_dim: embedding dimension.
        pg: process group.
        async_op: whether to run asynchronously.

    Returns:
        (work_handle, output_buffer) of shape [sum(output_splits) * embedding_dim].
    """
    output_split_sizes = [s * embedding_dim for s in output_splits]
    input_split_sizes = [s * embedding_dim for s in input_splits]

    output_buffer = torch.empty(
        sum(output_split_sizes),
        device=grad.device,
        dtype=grad.dtype,
    )

    req = dist.all_to_all_single(
        output_buffer,
        grad.contiguous().view(-1),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=pg,
        async_op=async_op,
    )
    assert req is not None

    return req, output_buffer


class PECAll2AllSeqWait(torch.autograd.Function):
    """PEC gradient intercept — merges embeddings in forward, re-splits
    gradients in backward.

    Forward: merges OL/NOL embeddings using forward_permute.

    Backward: splits gradient into OL/NOL via backward_ol_permute and
    backward_nol_permute (pre-composed with recat, so index_select
    produces rank-major order directly). Creates PECGradientApply
    instances for each partition — OL dist() starts immediately,
    NOL dist() is deferred to the pipeline.

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

        # OL: create applier and start AllToAll immediately
        autograd_ctx.ol_grad_apply = PECGradientApply(
            ol_grad,
            input_splits=splits.input_splits[0],
            output_splits=splits.output_splits[0],
            pg=autograd_ctx.pg,
        )
        autograd_ctx.ol_grad_apply.dist()

        # NOL: create applier, pipeline calls dist() later
        autograd_ctx.nol_grad_apply = PECGradientApply(
            nol_grad,
            input_splits=splits.input_splits[1],
            output_splits=splits.output_splits[1],
            pg=autograd_ctx.pg,
        )

        return (None, None, None, None)


class PECGradientApply:
    """Handles gradient AllToAll and application to TBE for one partition.

    Created by PECAll2AllSeqWait.backward — one for OL, one for NOL.
    Short-lived: exists for one batch, consumed by apply().

    Lifecycle:
        1. __init__: holds the re-split gradient
        2. dist(): starts async gradient AllToAll
        3. apply(): waits AllToAll, re-lookups features in TBE, calls
           embs.backward(grad) to push gradient through TBE kernel

    For OL: dist() is called in PECAll2AllSeqWait.backward (immediate).
    For NOL: dist() is called by the pipeline (deferred).
    """

    def __init__(
        self,
        grad: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        pg: dist.ProcessGroup,
    ) -> None:
        self._grad = grad
        self._input_splits = input_splits
        self._output_splits = output_splits
        self._pg = pg
        self._work: dist.Work | None = None
        self._grad_buffer: torch.Tensor | None = None

    def dist(self) -> None:
        """Starts async gradient reverse AllToAll. Idempotent — no-op if already started."""
        if self._work is not None:
            return

        embedding_dim = self._grad.shape[1]
        self._work, self._grad_buffer = _grad_dist(
            self._grad,
            input_splits=self._input_splits,
            output_splits=self._output_splits,
            embedding_dim=embedding_dim,
            pg=self._pg,
        )
        self._grad = None  # type: ignore[assignment]

    def apply(
        self,
        features: KeyedJaggedTensor,
        lookup_fn: Callable[[KeyedJaggedTensor], torch.Tensor],
        embedding_dim: int,
    ) -> None:
        """Waits AllToAll and applies gradient to TBE via re-lookup + backward."""
        assert self._work is not None
        assert self._grad_buffer is not None

        self._work.wait()

        if features.values().numel() > 0:
            with torch.enable_grad():
                embs = lookup_fn(features).view(-1, embedding_dim)
                if embs.numel() > 0:
                    embs.backward(self._grad_buffer.view(-1, embedding_dim))
        self._work = None
        self._grad_buffer = None


class PECGradUpdateAwaitable(Awaitable[None]):
    """Awaitable that applies gradient to TBE on wait.

    Wraps a PECGradientApply instance. On wait(), calls apply() which
    waits on the AllToAll and pushes gradient through TBE's backward kernel
    via re-lookup.
    """

    def __init__(
        self,
        applier: PECGradientApply,
        features: KeyedJaggedTensor,
        lookup_fn: Callable[[KeyedJaggedTensor], torch.Tensor],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._applier = applier
        self._features = features
        self._lookup_fn = lookup_fn
        self._embedding_dim = embedding_dim

    def _wait_impl(self) -> None:
        self._applier.apply(self._features, self._lookup_fn, self._embedding_dim)
