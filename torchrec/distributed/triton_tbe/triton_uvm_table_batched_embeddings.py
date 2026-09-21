#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from torchrec.distributed.triton_tbe.triton_table_batched_embeddings import (
    TritonTableBatchedEmbeddingBags,
)


class TritonUVMTableBatchedEmbeddingBags(TritonTableBatchedEmbeddingBags):
    """Forward-only Triton TBE with weights allocated in CUDA managed memory."""

    def __init__(
        self,
        *args: Any,
        uvm_host_mapped: bool = False,
        **kwargs: Any,
    ) -> None:
        self.uvm_host_mapped = uvm_host_mapped
        super().__init__(*args, **kwargs)

    def _allocate_weight(
        self,
        total_weight_size: int,
        weights_precision: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if device.type != "cuda":
            raise ValueError("Triton UVM TBE requires a CUDA device")
        reference = torch.empty(0, dtype=weights_precision, device=device)
        if self.uvm_host_mapped:
            return torch.ops.fbgemm.new_unified_tensor(
                reference,
                [total_weight_size],
                is_host_mapped=True,
            )
        return torch.ops.fbgemm.new_managed_tensor(reference, [total_weight_size])

    def forward(
        self,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        per_sample_weights: torch.Tensor | None = None,
        batch_size_per_feature_per_rank: list[list[int]] | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return super().forward(
                indices,
                offsets,
                per_sample_weights,
                batch_size_per_feature_per_rank,
            )


class TritonUVMCappedTableBatchedEmbeddingBags(TritonUVMTableBatchedEmbeddingBags):
    """Forward-only Triton UVM TBE with bounded forward launches."""

    def __init__(
        self,
        *args: Any,
        forward_block_limit: int = 0,
        vbe_forward_block_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        if forward_block_limit < 0:
            raise ValueError("forward_block_limit must be non-negative")
        if vbe_forward_block_limit < 0:
            raise ValueError("vbe_forward_block_limit must be non-negative")
        if forward_block_limit == 0 and vbe_forward_block_limit == 0:
            raise ValueError("capped UVM TBE requires a positive forward block limit")
        super().__init__(*args, **kwargs)
        self.forward_block_limit = forward_block_limit
        self.vbe_forward_block_limit = vbe_forward_block_limit
