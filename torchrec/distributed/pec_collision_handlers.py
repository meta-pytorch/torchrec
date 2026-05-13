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
from torchrec.distributed.dist_data import (
    SplitsAllToAllAwaitable,
    TensorAllToAllValuesAwaitable,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import Awaitable
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.pec_embedding_modules import OverlappingCheckerType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class CollisionResult:
    """Output of detect_collisions for a single sharding group.

    Contains overlap masks and remapped feature values. Remapped feature
    values are KJT values shifted by per-table cumulative offsets so that
    values from different tables occupy non-overlapping ranges, allowing a
    single boolean mask to track overlap across all tables.

    Example with two tables (table_0: 8 local rows, table_1: 8 local rows)::

        table_0 offset = 0,  table_1 offset = 8
        Raw values:      [3, 5]  (table_0)    [2, 7]  (table_1)
        Remapped values: [3, 5]                [10, 15]

    The pipeline saves `remapped_feature_values` from each batch and sets it
    on the next batch's context as `prev_remapped_feature_values`.

    Attributes:
        forward_overlap_mask: Bool tensor, shape [num_values]. True at
            position i means the i-th value in the current batch was also
            present in the previous batch. Used to split the current batch
            into overlapped (prioritized) and non-overlapped partitions.
            All False for the first batch.
        backward_overlap_mask: Bool tensor, shape [num_prev_values]. True
            at position i means the i-th value from the previous batch is
            also present in the current batch. Used to re-split the previous
            batch's gradients during backward. None for the first batch.
        remapped_feature_values: Tensor, shape [num_values]. Current batch's
            feature values shifted by per-table cumulative offsets.
    """

    forward_overlap_mask: torch.Tensor
    backward_overlap_mask: torch.Tensor | None
    remapped_feature_values: torch.Tensor


class BooleanMaskChecker:
    """Boolean mask checker for overlap detection.

    Maintains a boolean tensor of size `mask_size` (total local rows across
    all tables). Each position corresponds to a remapped feature value.
    `update` marks positions as seen; `check_overlap` checks them.
    """

    def __init__(self, device: torch.device, mask_size: int) -> None:
        self._device = device
        self._seen_values = torch.zeros(
            mask_size,
            dtype=torch.bool,
            device=device,
        )

    def reset_mask(self) -> None:
        self._seen_values.zero_()

    def update(self, remapped_feature_values: torch.Tensor) -> None:
        """Marks values as seen. Resets first (only current batch matters)."""
        self.reset_mask()
        self._seen_values[remapped_feature_values] = True

    def check_overlap(
        self,
        remapped_feature_values: torch.Tensor,
    ) -> torch.Tensor:
        """Returns bool mask: True where values overlap with previously-seen values."""
        return self._seen_values[remapped_feature_values]


def split_features_by_values_mask(
    features: KeyedJaggedTensor,
    overlap_mask: torch.Tensor,
) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """Splits KJT into overlapped and nonoverlapped partitions by bool mask.

    Uses fbgemm.asynchronous_complete_cumsum + fbgemm.segment_sum_csr to
    compute per-segment lengths for each partition efficiently.

    Args:
        features: input KJT (after input dist)
        overlap_mask: bool tensor of shape [num_values], True = overlapped

    Returns:
        (overlapped_kjt, nonoverlapped_kjt) with correct per-feature lengths
    """
    lengths = features.lengths()
    values = features.values()
    weights = features.weights_or_none()

    # Compute per-segment nonoverlapped counts via segment_sum_csr
    lengths_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

    # 1 = nonoverlapped, 0 = overlapped
    nol_indicator = (~overlap_mask).to(lengths.dtype)
    nol_lengths = torch.ops.fbgemm.segment_sum_csr(
        1,
        lengths_cumsum,
        nol_indicator,
    )

    # Split weights if present
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
class CollisionPermutation:
    """Result of permute_dist for a single sharding group.

    Attributes:
        forward_permute: For batch i — merges [ol, nol] embeddings back to
            original order. Nonoverlapped values don't conflict with batch
            i-1, so their lookup can start early (overlapped with i-1's
            backward). Overlapped values must wait for i-1's gradient update.
            Usage: index_select(cat([ol_embs, nol_embs]), 0, forward_permute)

        backward_ol_permute: Original-order indices of batch i-1's
            overlapped values, in rank-major order. Used to extract OL
            gradients via index_select. The OL gradient update must
            complete before batch i's ol lookup. None for the first batch.

        backward_nol_permute: Original-order indices of batch i-1's
            nonoverlapped values, in rank-major order. NOL gradient update
            can be deferred into batch i. None for the first batch.
    """

    forward_permute: torch.Tensor
    backward_ol_permute: torch.Tensor | None = None
    backward_nol_permute: torch.Tensor | None = None


def _compute_permute_from_mask(
    bool_mask: torch.Tensor,
    unbucketize_permute_tensor: torch.Tensor,
) -> torch.Tensor:
    """Computes a permute tensor from a received overlap mask and upt.

    The mask is in bucketized order (shard 0 values first, then shard 1, etc.).
    Builds a mapping from original position to position in merged [ol, nol],
    composed with unbucketize_permute_tensor.

    Example: bool_mask = [T, F, T, T, F, F]
      ol_rank  = cumsum([1,0,1,1,0,0]) - 1 = [0, 0, 1, 2, 2, 2]
      nol_rank = cumsum([0,1,0,0,1,1]) - 1 + 3 = [2, 3, 3, 3, 4, 5]
      bucket_to_merged = where(mask, ol_rank, nol_rank) = [0, 3, 1, 2, 4, 5]
    """
    num_ol = bool_mask.sum()

    ol_rank = bool_mask.long().cumsum(0) - 1
    nol_rank = (~bool_mask).long().cumsum(0) - 1 + num_ol
    bucket_to_merged = torch.where(bool_mask, ol_rank, nol_rank)

    return bucket_to_merged[unbucketize_permute_tensor]


class CollisionPermuteAwaitable(Awaitable[CollisionPermutation]):
    """RW-specific awaitable that produces CollisionPermutation from received masks.

    On wait(), receives forward (and optionally backward) masks via AllToAll,
    then builds permute tensors by composing with unbucketize_permute_tensor.
    """

    def __init__(
        self,
        fwd_awaitable: TensorAllToAllValuesAwaitable,
        fwd_unbucketize_permute: torch.Tensor,
        bwd_awaitable: TensorAllToAllValuesAwaitable | None = None,
        bwd_unbucketize_permute: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self._fwd_awaitable = fwd_awaitable
        self._fwd_unbucketize_permute = fwd_unbucketize_permute
        self._bwd_awaitable = bwd_awaitable
        self._bwd_unbucketize_permute = bwd_unbucketize_permute

    def _wait_impl(self) -> CollisionPermutation:
        fwd_mask = self._fwd_awaitable.wait().bool()
        forward_permute = _compute_permute_from_mask(
            fwd_mask, self._fwd_unbucketize_permute
        )

        backward_ol_permute = None
        backward_nol_permute = None

        if self._bwd_awaitable is not None:
            assert self._bwd_unbucketize_permute is not None
            bwd_mask = self._bwd_awaitable.wait().bool()

            # Pre-compose recat into indices so backward can index_select
            # directly from original-order gradient without a separate
            # recat step. The indices are in ascending bucketized order
            # (rank-major), which aligns with AllToAll splits.
            recat = torch.ops.fbgemm.invert_permute(self._bwd_unbucketize_permute)
            (ol_bucket_indices,) = torch.where(bwd_mask)
            (nol_bucket_indices,) = torch.where(~bwd_mask)

            backward_ol_permute = recat[ol_bucket_indices]
            backward_nol_permute = recat[nol_bucket_indices]

        return CollisionPermutation(
            forward_permute=forward_permute,
            backward_ol_permute=backward_ol_permute,
            backward_nol_permute=backward_nol_permute,
        )


@dataclass
class CollisionSplits:
    """Per-rank split sizes for overlapped/nonoverlapped partitions.

    input_splits: [ol_per_rank, nol_per_rank] — how many values this rank
        sends to each other rank per partition.
    output_splits: [ol_received, nol_received] — how many values this rank
        receives from each other rank per partition.
    """

    input_splits: List[List[int]]
    output_splits: List[List[int]]


class CollisionSplitsAwaitable(Awaitable[CollisionSplits]):
    """Resolves to CollisionSplits for a single sharding group.

    Wraps a SplitsAllToAllAwaitable that exchanges nonoverlapped per-rank
    counts. On wait(), derives overlapped received splits from
    total - nol_received, then assembles CollisionSplits.
    """

    def __init__(
        self,
        ol_input_splits: List[int],
        nol_input_splits: List[int],
        splits_awaitable: SplitsAllToAllAwaitable,
        total_input_splits: List[int],
    ) -> None:
        super().__init__()
        self._ol_input_splits = ol_input_splits
        self._nol_input_splits = nol_input_splits
        self._splits_awaitable = splits_awaitable
        self._total_input_splits = total_input_splits

    def _wait_impl(self) -> CollisionSplits:
        result = self._splits_awaitable.wait()
        nol_received = result[0]
        ol_received = [
            self._total_input_splits[r] - nol_received[r]
            for r in range(len(nol_received))
        ]
        return CollisionSplits(
            input_splits=[self._ol_input_splits, self._nol_input_splits],
            output_splits=[ol_received, nol_received],
        )


class CollisionHandlerBase(abc.ABC):
    """Interface for PEC collision detection handlers.

    Subclasses implement sharding-specific overlap detection logic.
    """

    @abc.abstractmethod
    def detect_collisions(
        self,
        features: KeyedJaggedTensor,
        prev_remapped_feature_values: torch.Tensor | None = None,
    ) -> CollisionResult:
        """Detects overlapping IDs between current and previous batch.

        Args:
            features: current batch features (after input dist)
            prev_remapped_feature_values: previous batch's remapped_feature_values, None for first batch

        Returns:
            CollisionResult with forward/backward masks and remapped_feature_values
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets internal state (e.g. at epoch boundary)."""
        ...

    @abc.abstractmethod
    def permute_dist(
        self,
        # Forward: current batch
        features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
        forward_overlap_mask: torch.Tensor,
        # Backward: previous batch (all None for first batch)
        prev_features: KeyedJaggedTensor | None = None,
        prev_sharding_ctx: SequenceShardingContext | None = None,
        backward_overlap_mask: torch.Tensor | None = None,
    ) -> Awaitable[CollisionPermutation]:
        """Distributes forward and backward partition permutations via AllToAll.

        Sends forward overlap mask (current batch) and optionally backward
        overlap mask (previous batch) from sharding ranks back to the original
        rank. Both AllToAlls are fired together for batching.

        Args:
            features: current batch features after input_dist
            sharding_ctx: current batch's sharding context
            forward_overlap_mask: current batch's overlap mask
            prev_features: previous batch's features (None for first batch)
            prev_sharding_ctx: previous batch's sharding context
            backward_overlap_mask: previous batch's backward mask

        Returns:
            Awaitable resolving to CollisionPermutation.
        """
        ...

    @abc.abstractmethod
    def split_dist(
        self,
        nonoverlapped_features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
    ) -> Awaitable[CollisionSplits]:
        """Exchanges per-rank overlapped/nonoverlapped split sizes via AllToAll.

        Computes how many nonoverlapped values this rank sends to each peer
        rank, derives the overlapped counts from the total, and starts a
        SplitsAllToAll. Returns an awaitable that resolves to CollisionSplits.

        Args:
            nonoverlapped_features: nonoverlapped partition KJT
            sharding_ctx: sharding context from input_dist

        Returns:
            Awaitable resolving to CollisionSplits with per-rank send/receive
            counts for each partition.
        """
        ...


class RWCollisionHandler(CollisionHandlerBase):
    """Row-wise collision handler for PEC.

    Uses cumulative local_rows across all tables as the offset space for
    BooleanMaskChecker. Each feature's IDs are offset by their table's
    cumulative row position so that IDs from different tables don't collide.
    """

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

        self._checker = BooleanMaskChecker(device, mask_size)

    def _remap_feature_values(self, features: KeyedJaggedTensor) -> torch.Tensor:
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

    def detect_collisions(
        self,
        features: KeyedJaggedTensor,
        prev_remapped_feature_values: torch.Tensor | None = None,
    ) -> CollisionResult:
        remapped_feature_values = self._remap_feature_values(features)

        if prev_remapped_feature_values is not None:
            forward_overlap_mask = self._checker.check_overlap(remapped_feature_values)
            self._checker.update(remapped_feature_values)
            backward_overlap_mask = self._checker.check_overlap(
                prev_remapped_feature_values
            )
        else:
            forward_overlap_mask = torch.zeros(
                features.values().numel(),
                dtype=torch.bool,
                device=self._device,
            )
            backward_overlap_mask = None
            self._checker.update(remapped_feature_values)

        return CollisionResult(
            forward_overlap_mask=forward_overlap_mask,
            backward_overlap_mask=backward_overlap_mask,
            remapped_feature_values=remapped_feature_values,
        )

    def reset(self) -> None:
        self._checker.reset_mask()

    def _compute_nonoverlapped_per_rank(
        self,
        nonoverlapped_features: KeyedJaggedTensor,
        batch_size_per_rank: List[int],
    ) -> torch.Tensor:
        """Computes per-rank nonoverlapped value counts for RW sharding.

        After RW input_dist, the KJT lengths are in feature-major order.
        We transpose to batch-major, then use segment_sum_csr with
        batch_size_per_rank as segment boundaries to sum lengths across
        all features for each rank's batch elements.
        """
        batch_size_cumsum = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(
                batch_size_per_rank,
                device=self._device,
                dtype=torch.int64,
            )
        )
        num_features = len(nonoverlapped_features.keys())
        lengths_batch_major = (
            nonoverlapped_features.lengths().view(num_features, -1).t().flatten()
        )
        return torch.ops.fbgemm.segment_sum_csr(
            num_features, batch_size_cumsum, lengths_batch_major
        )

    def split_dist(
        self,
        nonoverlapped_features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
    ) -> CollisionSplitsAwaitable:
        """Computes per-rank ol/nol split sizes and exchanges via AllToAll."""
        assert sharding_ctx.batch_size_per_rank is not None
        nol = self._compute_nonoverlapped_per_rank(
            nonoverlapped_features, sharding_ctx.batch_size_per_rank
        )
        ol = (
            torch.tensor(
                sharding_ctx.output_splits,
                device=nol.device,
                dtype=nol.dtype,
            )
            - nol
        )

        splits_awaitable = SplitsAllToAllAwaitable([nol], self._pg)

        return CollisionSplitsAwaitable(
            ol_input_splits=ol.tolist(),
            nol_input_splits=nol.tolist(),
            splits_awaitable=splits_awaitable,
            total_input_splits=sharding_ctx.input_splits,
        )

    def _mask_dist(
        self,
        features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
        overlap_mask: torch.Tensor,
    ) -> TensorAllToAllValuesAwaitable:
        """Permutes mask to rank-major order and starts reverse AllToAll."""
        assert sharding_ctx.sparse_features_recat is not None

        recat = torch.ops.fbgemm.invert_permute(sharding_ctx.sparse_features_recat)
        mask_int = overlap_mask.to(torch.int32)

        _, permuted_mask, _ = torch.ops.fbgemm.permute_2D_sparse_data(
            recat,
            features.lengths().view(recat.shape[0], -1),
            mask_int,
            None,
            mask_int.numel(),
        )

        return TensorAllToAllValuesAwaitable(
            self._pg,
            permuted_mask,
            input_splits=torch.tensor(
                sharding_ctx.output_splits,
                dtype=torch.int32,
            ),
            output_splits=torch.tensor(
                sharding_ctx.input_splits,
                dtype=torch.int32,
            ),
            device=permuted_mask.device,
        )

    def permute_dist(
        self,
        features: KeyedJaggedTensor,
        sharding_ctx: SequenceShardingContext,
        forward_overlap_mask: torch.Tensor,
        prev_features: KeyedJaggedTensor | None = None,
        prev_sharding_ctx: SequenceShardingContext | None = None,
        backward_overlap_mask: torch.Tensor | None = None,
    ) -> CollisionPermuteAwaitable:
        """Fires forward (and optionally backward) mask AllToAlls together."""
        assert sharding_ctx.unbucketize_permute_tensor is not None

        fwd_awaitable = self._mask_dist(
            features,
            sharding_ctx,
            forward_overlap_mask,
        )

        bwd_awaitable = None
        bwd_unbucketize_permute = None

        if backward_overlap_mask is not None:
            assert prev_features is not None
            assert prev_sharding_ctx is not None
            assert prev_sharding_ctx.unbucketize_permute_tensor is not None

            bwd_awaitable = self._mask_dist(
                prev_features,
                prev_sharding_ctx,
                backward_overlap_mask,
            )
            bwd_unbucketize_permute = prev_sharding_ctx.unbucketize_permute_tensor

        return CollisionPermuteAwaitable(
            fwd_awaitable=fwd_awaitable,
            fwd_unbucketize_permute=sharding_ctx.unbucketize_permute_tensor,
            bwd_awaitable=bwd_awaitable,
            bwd_unbucketize_permute=bwd_unbucketize_permute,
        )


def create_collision_handler(
    sharding_type: str,
    device: torch.device,
    grouped_emb_configs: List[GroupedEmbeddingConfig],
    table_name_to_config: Dict[str, EmbeddingConfig],
    process_group: dist.ProcessGroup,
    checker_type: OverlappingCheckerType,
) -> CollisionHandlerBase:
    """Creates a collision handler for the given sharding type."""
    if sharding_type == "row_wise":
        return RWCollisionHandler(
            device=device,
            grouped_emb_configs=grouped_emb_configs,
            table_name_to_config=table_name_to_config,
            process_group=process_group,
            checker_type=checker_type,
        )
    raise ValueError(
        f"PEC collision detection does not support sharding type: {sharding_type}"
    )
