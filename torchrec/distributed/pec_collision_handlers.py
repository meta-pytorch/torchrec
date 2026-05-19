#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import TensorAllToAllValuesAwaitable
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.pec_embedding_modules import OverlappingCheckerType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class OverlapMasks:
    """Overlap masks from detect_overlap for a single sharding group.

    These masks are calculated after input dist and serve as input to mask_dist,
    which permutes and AllToAll's them to produce post-dist masks.

    Attributes:
        forward_overlap_mask: Bool tensor, shape [num_values]. True at
            position i means the i-th value in the current batch was also
            present in the previous batch. None when there is no current
            batch (last-batch finalization).
        backward_overlap_mask: Bool tensor, shape [num_prev_values]. True
            at position i means the i-th value from the previous batch is
            also present in the current batch. None when there is no
            previous batch (first batch).
    """

    forward_overlap_mask: torch.Tensor | None
    backward_overlap_mask: torch.Tensor | None


class OverlapChecker(abc.ABC):
    """Interface for overlap detection between consecutive batches."""

    @abc.abstractmethod
    def check(
        self,
        current_remapped: torch.Tensor,
        prev_remapped: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (forward_mask, backward_mask)."""
        ...


class BooleanOverlapChecker(OverlapChecker):
    """Boolean mask checker with reusable scratch buffer.

    Semantically stateless — buffer is zeroed before every use.
    """

    def __init__(self, device: torch.device, mask_size: int) -> None:
        self._seen_buffer = torch.zeros(
            mask_size,
            dtype=torch.bool,
            device=device,
        )

    def check(
        self,
        current_remapped: torch.Tensor,
        prev_remapped: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._seen_buffer.zero_()
        self._seen_buffer[prev_remapped] = True
        forward_mask = self._seen_buffer[current_remapped].clone()

        self._seen_buffer.zero_()
        self._seen_buffer[current_remapped] = True
        backward_mask = self._seen_buffer[prev_remapped].clone()

        return forward_mask, backward_mask


def split_kjt_by_values_mask(
    features: KeyedJaggedTensor,
    overlap_mask: torch.Tensor,
) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """Splits KJT into overlapped and nonoverlapped partitions by bool mask over
    KJT values tensor.

    Args:
        features: input KJT (after input dist)
        overlap_mask: bool tensor of shape [num_values], True = overlapped

    Returns:
        (overlapped_kjt, nonoverlapped_kjt) with correct per-feature lengths
    """
    lengths = features.lengths()
    values = features.values()
    weights = features.weights_or_none()

    lengths_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

    nol_indicator = (~overlap_mask).to(lengths.dtype)
    nol_lengths = torch.ops.fbgemm.segment_sum_csr(
        1,
        lengths_cumsum,
        nol_indicator,
    )

    ol_weights = None
    nol_weights = None

    if weights is not None:
        ol_weights = weights[overlap_mask]
        nol_weights = weights[~overlap_mask]

    overlapped_kjt = KeyedJaggedTensor(
        keys=features.keys(),
        values=values[overlap_mask],
        weights=ol_weights,
        lengths=lengths - nol_lengths,
    )
    nonoverlapped_kjt = KeyedJaggedTensor(
        keys=features.keys(),
        values=values[~overlap_mask],
        weights=nol_weights,
        lengths=nol_lengths,
    )
    return overlapped_kjt, nonoverlapped_kjt


@dataclass
class OverlapSplits:
    """Per-rank split sizes for overlapped/nonoverlapped partitions.

    input_splits: (ol_per_rank, nol_per_rank) — how many values this rank
        sends to each other rank per partition.
    output_splits: (ol_received, nol_received) — how many values this rank
        receives from each other rank per partition.
    """

    input_splits: Tuple[List[int], List[int]]
    output_splits: Tuple[List[int], List[int]]


class OverlapHandler(abc.ABC):
    """Interface for PEC overlap detection and distribution handlers.

    Subclasses implement sharding-specific logic. The sharded module
    calls these methods in a sharding-agnostic orchestration.
    """

    @abc.abstractmethod
    def remap_kjt_values(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """Remaps KJT values to globally-unique keys across tables.

        Sharding-specific: offset computation depends on how rows
        are distributed across shards.
        """
        ...

    @abc.abstractmethod
    def detect_overlap(
        self,
        current_remapped: torch.Tensor | None,
        prev_remapped: torch.Tensor | None,
    ) -> OverlapMasks:
        """Detects overlap between current and previous remapped values."""
        ...

    @abc.abstractmethod
    def mask_dist(
        self,
        features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
        overlap_mask: torch.Tensor,
    ) -> MaskDistAwaitable:
        """Distributes overlap mask via AllToAll.

        Sends the overlap mask using same splits from input_dist (reuses
        sharding_ctx splits). Returns MaskDistAwaitable which holds both the
        AllToAll awaitable and the pre-AllToAll permuted mask (needed by
        compute_splits).
        """
        ...

    @abc.abstractmethod
    def compute_splits(
        self,
        post_dist_mask: torch.Tensor,
        pre_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> OverlapSplits:
        """Computes per-rank OL/NOL splits from masks.

        Args:
            post_dist_mask: mask after mask_dist AllToAll
            pre_dist_mask: mask before mask_dist AllToAll
            sharding_ctx: provides total per-rank splits as segment boundaries
        """
        ...

    @abc.abstractmethod
    def compute_forward_permute(
        self,
        post_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> torch.Tensor:
        """Computes forward permute for merging [ol, nol] → original order."""
        ...

    @abc.abstractmethod
    def compute_backward_permutes(
        self,
        post_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes (ol_permute, nol_permute) to partition backward gradient.

        Returns original-order indices pre-composed with recat,
        in rank-major order for AllToAll alignment.
        """
        ...


class MaskDistAwaitable(Awaitable[Tuple[torch.Tensor, torch.Tensor]]):
    """Wraps TensorAllToAllValuesAwaitable for mask AllToAll.

    On wait, returns (post_dist_mask, pre_dist_mask):
    - post_dist_mask: bool mask after AllToAll
    - pre_dist_mask: bool mask before AllToAll (needed by compute_splits)
    """

    def __init__(
        self,
        awaitable: TensorAllToAllValuesAwaitable,
        pre_dist_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self._awaitable = awaitable
        self._pre_dist_mask = pre_dist_mask

    def _wait_impl(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._awaitable.wait().bool(), self._pre_dist_mask


class RWOverlapHandler(OverlapHandler):
    """Row-wise overlap handler for PEC."""

    def __init__(
        self,
        device: torch.device,
        grouped_emb_configs: List[GroupedEmbeddingConfig],
        table_name_to_config: Dict[str, EmbeddingConfig],
        process_group: dist.ProcessGroup,
        checker_type: OverlappingCheckerType = OverlappingCheckerType.BOOLEAN,
    ) -> None:
        assert (
            checker_type == OverlappingCheckerType.BOOLEAN
        ), f"Only BOOLEAN checker is supported, got {checker_type}"

        self._device = device
        self._pg = process_group
        self._feature_to_table_offset: Dict[str, int] = {}
        self._input_features_offset: torch.Tensor = torch.empty(
            0,
            dtype=torch.int64,
            device=device,
        )

        mask_size = 0
        for conf in grouped_emb_configs:
            for emb_table in conf.embedding_tables:
                for fname in table_name_to_config[emb_table.name].feature_names:
                    self._feature_to_table_offset[fname] = mask_size
                mask_size += emb_table.local_rows

        self._checker = BooleanOverlapChecker(device, mask_size)

    def remap_kjt_values(self, features: KeyedJaggedTensor) -> torch.Tensor:
        """Adds per-table row offsets to make values unique within the shard.

        In RW sharding, values from different tables may collide (e.g.,
        table_0 row 3 and table_1 row 3 are both value 3). This adds
        cumulative local_rows offsets per table so each value maps to a
        unique position in the checker's boolean buffer.

        Example: 2 tables with local_rows=8 each, features=[f0, f1].
          table_0 offset = 0, table_1 offset = 8.
          values = [0, 1, 2, 3] with keys=[f0, f1], lengths=[2, 2]
          → remapped = [0+0, 1+0, 2+8, 3+8] = [0, 1, 10, 11]

        The offset tensor is lazily initialized on the first call and
        reused for subsequent batches.
        """
        if self._input_features_offset.numel() == 0:
            offset_list = [
                self._feature_to_table_offset[fname] for fname in features.keys()
            ]
            self._input_features_offset = torch.tensor(
                offset_list,
                device=self._device,
                dtype=torch.int64,
            ).reshape(1, -1)

        per_feat_lengths = features.lengths().view(-1, features.stride()).sum(1)
        offsets = torch.repeat_interleave(
            self._input_features_offset,
            per_feat_lengths.view(-1),
            dim=1,
        )
        return features.values() + offsets.view(-1)

    def detect_overlap(
        self,
        current_remapped: torch.Tensor | None,
        prev_remapped: torch.Tensor | None,
    ) -> OverlapMasks:
        """Detects value overlap between current and previous batches.

        First batch (prev None): no overlap possible, returns all-False
        forward mask so all values go to the NOL partition.

        Last batch (current None): no next batch to overlap with, returns
        all-True backward mask so all previous values go through the OL
        gradient path.

        Normal batch: runs the checker to produce per-value forward and
        backward masks.
        """
        if prev_remapped is None:
            assert current_remapped is not None
            return OverlapMasks(
                forward_overlap_mask=torch.zeros(
                    current_remapped.numel(),
                    dtype=torch.bool,
                    device=self._device,
                ),
                backward_overlap_mask=None,
            )

        if current_remapped is None:
            return OverlapMasks(
                forward_overlap_mask=None,
                backward_overlap_mask=torch.ones(
                    prev_remapped.numel(),
                    dtype=torch.bool,
                    device=self._device,
                ),
            )

        fwd_mask, bwd_mask = self._checker.check(current_remapped, prev_remapped)
        return OverlapMasks(
            forward_overlap_mask=fwd_mask,
            backward_overlap_mask=bwd_mask,
        )

    def mask_dist(
        self,
        features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
        overlap_mask: torch.Tensor,
    ) -> MaskDistAwaitable:
        """Permutes mask from feature-major to rank-major order via recat, then
        AllToAll.

        In RW sharding, input_dist recats features from [f0_r0, f0_r1,
        f1_r0, f1_r1] (feature-major) to [f0_r0, f1_r0, f0_r1, f1_r1]
        (rank-major). The mask must follow the same recat before AllToAll
        so it aligns with the per-rank splits in sharding_ctx.
        """
        assert sharding_ctx.sparse_features_recat is not None

        recat = torch.ops.fbgemm.invert_permute(sharding_ctx.sparse_features_recat)
        mask_int = overlap_mask.to(torch.int32)

        _, pre_dist_mask, _ = torch.ops.fbgemm.permute_2D_sparse_data(
            recat,
            features.lengths().view(recat.shape[0], -1),
            mask_int,
            None,
            mask_int.numel(),
        )

        values_awaitable = TensorAllToAllValuesAwaitable(
            self._pg,
            pre_dist_mask,
            input_splits=torch.tensor(
                sharding_ctx.output_splits,
                dtype=torch.int32,
            ),
            output_splits=torch.tensor(
                sharding_ctx.input_splits,
                dtype=torch.int32,
            ),
            device=pre_dist_mask.device,
        )

        return MaskDistAwaitable(values_awaitable, pre_dist_mask)

    def compute_splits(
        self,
        post_dist_mask: torch.Tensor,
        pre_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> OverlapSplits:
        """Segments the mask by per-rank splits to get OL/NOL counts.

        Uses sharding_ctx.input_splits as segment boundaries for
        post_dist_mask, and sharding_ctx.output_splits as segment
        boundaries for pre_dist_mask. segment_sum_csr counts True
        values per segment to produce OL counts; NOL = total - OL.

        Example: world_size=2, post_dist_mask = [T, F, T, T, F, F]
          input_splits = [3, 3] → segments [T,F,T] and [T,F,F]
          ol_input = [2, 1], nol_input = [1, 2]
          → input_splits = ([2, 1], [1, 2])
        """
        device = post_dist_mask.device

        # Input splits (what this rank sends): from post-dist mask
        input_total = torch.tensor(
            sharding_ctx.input_splits,
            device=device,
            dtype=torch.int64,
        )
        input_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(input_total)
        ol_input = torch.ops.fbgemm.segment_sum_csr(
            1,
            input_cumsum,
            post_dist_mask.long(),
        )
        nol_input = input_total - ol_input

        # Output splits (what this rank receives): from pre-dist mask
        output_total = torch.tensor(
            sharding_ctx.output_splits,
            device=device,
            dtype=torch.int64,
        )
        output_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(output_total)
        ol_output = torch.ops.fbgemm.segment_sum_csr(
            1,
            output_cumsum,
            pre_dist_mask.long(),
        )
        nol_output = output_total - ol_output

        return OverlapSplits(
            input_splits=(ol_input.tolist(), nol_input.tolist()),
            output_splits=(ol_output.tolist(), nol_output.tolist()),
        )

    def compute_forward_permute(
        self,
        post_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> torch.Tensor:
        """Builds a permutation that merges [OL, NOL] back to original order.

        After OL and NOL embeddings are AllToAll'd back separately, they
        arrive as [ol_embs, nol_embs]. This computes a permutation that
        maps each position to its slot in [ol, nol] order, composed with
        unbucketize_permute_tensor to undo input_dist bucketization.

        Example:
          Assume world_size = 2, hash_size = 6
          - KJT values: [5, 3, 0, 1, 2, 4]
          - bucketized: [0, 1, 2, 5, 3, 4]
          - upt: [3, 4, 0, 1, 2, 5]

          Assume overlaps:
          - rank 0: [0, 1, 2] -> [F, T, T]
          - rank 1: [5, 3, 4] -> [T, F, F]

          post_dist_mask = [F, T, T, T, F, F] in bucketized order
          ol_rank  = cumsum([0,1,1,1,0,0]) - 1 = [-1, 0, 1, 2, 2, 2]
          nol_rank = cumsum([1,0,0,0,1,1]) - 1 + 3 = [3, 3, 3, 3, 4, 5]
          bucket_to_merged = [3 (nol), 0(ol), 1(ol), 2(ol), 4(nol), 5(nol)]
          merged_order = [1, 2, 5] (ol) + [0, 3, 4] (nol)

          bucket_to_merged will map merged_order back to bucketized order above:
            [1, 2, 5, 0, 3, 4] -> [0, 1, 2, 5, 3, 4]
          And then upt can unbucketize to the original order.

        """
        assert sharding_ctx.unbucketize_permute_tensor is not None
        upt = sharding_ctx.unbucketize_permute_tensor

        num_ol = post_dist_mask.sum()
        ol_rank = post_dist_mask.long().cumsum(0) - 1
        nol_rank = (~post_dist_mask).long().cumsum(0) - 1 + num_ol
        bucket_to_merged = torch.where(post_dist_mask, ol_rank, nol_rank)

        return bucket_to_merged[upt]

    def compute_backward_permutes(
        self,
        post_dist_mask: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes indices to split the backward gradient into OL and NOL.

        During backward, gradients arrive in original (pre-input-dist)
        order. We need to select OL and NOL gradients separately for
        their respective AllToAlls. The selected indices must be in
        rank-major (bucketized) order to align with AllToAll splits.

        recat = invert_permute(upt) maps original_pos → bucket_pos.
        We find which bucket positions are OL vs NOL from the mask,
        then index into recat so index_select produces gradients in
        rank-major order directly.

        Example:
          Same setup as compute_forward_permute:
          - KJT values: [5, 3, 0, 1, 2, 4]
          - bucketized: [0, 1, 2, 5, 3, 4]
          - upt: [3, 4, 0, 1, 2, 5]
          - post_dist_mask = [F, T, T, T, F, F] in bucketized order

          recat = invert_permute(upt) = [2, 3, 4, 0, 1, 5]
          ol_bucket_indices  = where(mask)  = [1, 2, 3]
          nol_bucket_indices = where(~mask) = [0, 4, 5]
          ol_permute  = recat[[1, 2, 3]] = [3, 4, 0]
          nol_permute = recat[[0, 4, 5]] = [2, 1, 5]

          Usage: grad_output is in original order [5, 3, 0, 1, 2, 4].
            ol_grad  = grad_output.index_select(0, ol_permute)
                     = grad_output[[3, 4, 0]] -> grads for [1, 2, 5]
            nol_grad = grad_output.index_select(0, nol_permute)
                     = grad_output[[2, 1, 5]] -> grads for [0, 3, 4]
          Both are in rank-major order, ready for gradient AllToAll.

        """
        assert sharding_ctx.unbucketize_permute_tensor is not None

        recat = torch.ops.fbgemm.invert_permute(sharding_ctx.unbucketize_permute_tensor)
        (ol_bucket_indices,) = torch.where(post_dist_mask)
        (nol_bucket_indices,) = torch.where(~post_dist_mask)

        return recat[ol_bucket_indices], recat[nol_bucket_indices]


def create_overlap_handler(
    sharding_type: str,
    device: torch.device,
    grouped_emb_configs: List[GroupedEmbeddingConfig],
    table_name_to_config: Dict[str, EmbeddingConfig],
    process_group: dist.ProcessGroup,
    checker_type: OverlappingCheckerType,
) -> OverlapHandler:
    """Creates an overlap handler for the given sharding type."""
    if sharding_type == "row_wise":
        return RWOverlapHandler(
            device=device,
            grouped_emb_configs=grouped_emb_configs,
            table_name_to_config=table_name_to_config,
            process_group=process_group,
            checker_type=checker_type,
        )
    raise ValueError(
        f"PEC overlap detection does not support sharding type: {sharding_type}"
    )
