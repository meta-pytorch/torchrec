#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.pec_collision_handlers import (
    CollisionPermutation,
    CollisionSplits,
)


@dataclass
class PECAll2AllSeqInfo:
    """Metadata for PEC backward gradient re-split and distribution.

    Created by merge_partitioned_embeddings during forward. Read by
    PECAll2AllSeqWait during backward to re-split and distribute gradients.

    After backward, the pipeline reads the gradient outputs below and is
    responsible for waiting on the OL AllToAll and applying both OL/NOL
    gradients to TBE. A dedicated gradient manager class for wait + apply
    will be added in a future diff.

    Attributes:
        permutation: CollisionPermutation containing forward_permute (used in
            forward merge) and backward_ol_permute/backward_nol_permute
            (pre-composed original-order indices for splitting gradients
            in rank-major order — no separate recat needed).
        backward_splits: Previous batch's forward CollisionSplits, reused as
            the gradient AllToAll split sizes. None until set by pipeline.
        pg: Process group for gradient AllToAll.

    Gradient outputs (set by PECAll2AllSeqWait.backward):
        ol_grad_work: Async work handle for OL gradient AllToAll.
            Pipeline must wait before applying.
        ol_grad_local: Output buffer for OL gradient AllToAll.
        nol_grad: NOL gradient tensor. Pipeline starts its own deferred
            AllToAll using backward_splits.[input|output]_splits[1].
    """

    permutation: CollisionPermutation
    pg: dist.ProcessGroup | None = None
    backward_splits: CollisionSplits | None = None

    # Gradient outputs — set by PECAll2AllSeqWait.backward, read by pipeline.
    # Pipeline is responsible for wait + apply (future: gradient manager class).
    ol_grad_work: dist.Work | None = None
    ol_grad_local: torch.Tensor | None = None
    nol_grad: torch.Tensor | None = None


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
    """PEC gradient intercept — merges embeddings in forward, re-splits gradients in backward.

    Forward: merges OL/NOL embeddings using forward_permute from backward_ctx.permutation.

    Backward: permutes gradient to bucketized order, splits by backward
    overlap mask (using pre-computed ol/nol indices from backward_permute),
    starts async OL gradient AllToAll (stored on backward_ctx for pipeline to wait),
    saves NOL gradient on backward_ctx for deferred processing.

    Gradient propagation stops here (returns None) — the pipeline applies
    gradients to TBE separately.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        backward_ctx: PECAll2AllSeqInfo,
        ol_embs: torch.Tensor,
        nol_embs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.backward_ctx = backward_ctx  # pyre-ignore[16]

        merged_embs = torch.cat([ol_embs, nol_embs], dim=0)
        return torch.index_select(
            merged_embs, 0, backward_ctx.permutation.forward_permute
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[None, None, None]:
        backward_ctx: PECAll2AllSeqInfo = ctx.backward_ctx  # pyre-ignore[16]
        perm = backward_ctx.permutation
        splits = backward_ctx.backward_splits

        assert perm.backward_ol_permute is not None
        assert perm.backward_nol_permute is not None
        assert splits is not None
        assert backward_ctx.pg is not None

        _, embedding_dim = grad_output.shape

        # Split gradient into ol/nol. Permute indices are pre-composed
        # with recat, so index_select produces rank-major order directly.
        ol_grad = grad_output.index_select(0, perm.backward_ol_permute)
        nol_grad = grad_output.index_select(0, perm.backward_nol_permute)

        # 3. Start async reverse AllToAll for OL grad
        ol_grad_work, ol_grad_local = _grad_dist(
            ol_grad,
            input_splits=splits.input_splits[0],
            output_splits=splits.output_splits[0],
            embedding_dim=embedding_dim,
            pg=backward_ctx.pg,
        )
        backward_ctx.ol_grad_work = ol_grad_work
        backward_ctx.ol_grad_local = ol_grad_local

        # 4. Save NOL grad for pipeline (deferred AllToAll).
        # Pipeline reads NOL splits from backward_ctx.backward_splits directly.
        backward_ctx.nol_grad = nol_grad

        return (None, None, None)
