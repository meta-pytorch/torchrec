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
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
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
        """Mark values as seen. Resets first (only current batch matters)."""
        self.reset_mask()
        self._seen_values[remapped_feature_values] = True

    def check_overlap(
        self,
        remapped_feature_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return bool mask: True where values overlap with previously-seen values."""
        return self._seen_values[remapped_feature_values]


def split_features_by_values_mask(
    features: KeyedJaggedTensor,
    overlap_mask: torch.Tensor,
) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """Split KJT into overlapped and nonoverlapped partitions by bool mask.

    Uses fbgemm.asynchronous_complete_cumsum + fbgemm.segment_sum_csr to
    compute per-segment lengths for each partition efficiently.

    Args:
        features: input KJT (after input dist, on sharder side)
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


class CollisionHandlerBase(abc.ABC):
    """Interface for PEC collision detection handlers.

    Subclasses implement sharding-specific overlap detection logic.
    The pipeline passes prev_remapped_feature_values as an argument to detect_collisions
    rather than storing cross-batch state internally.
    """

    @abc.abstractmethod
    def detect_collisions(
        self,
        features: KeyedJaggedTensor,
        prev_remapped_feature_values: torch.Tensor | None = None,
    ) -> CollisionResult:
        """Detect overlapping IDs between current and previous batch.

        Args:
            features: current batch features (after input dist)
            prev_remapped_feature_values: previous batch's remapped_feature_values, None for first batch

        Returns:
            CollisionResult with forward/backward masks and remapped_feature_values
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. at epoch boundary)."""
        ...

    @abc.abstractmethod
    def compute_nonoverlapped_per_rank(
        self,
        nonoverlapped_features: KeyedJaggedTensor,
        batch_size_per_rank: List[int],
    ) -> torch.Tensor:
        """Computes per-rank nonoverlapped value counts.

        Used by collision_split_dist to compute per-rank split sizes for
        the nonoverlapped partition. The overlapped splits are derived from
        the total (output_splits - nonoverlapped).

        Args:
            nonoverlapped_features: nonoverlapped partition KJT
            batch_size_per_rank: number of batch elements from each rank

        Returns:
            Tensor of shape [world_size] with per-rank nonoverlapped value counts
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
        checker_type: OverlappingCheckerType = OverlappingCheckerType.BOOLEAN,
    ) -> None:
        assert (
            checker_type == OverlappingCheckerType.BOOLEAN
        ), f"Only BOOLEAN checker is supported, got {checker_type}"
        self._device = device
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

    def compute_nonoverlapped_per_rank(
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


def create_collision_handler(
    sharding_type: str,
    device: torch.device,
    grouped_emb_configs: List[GroupedEmbeddingConfig],
    table_name_to_config: Dict[str, EmbeddingConfig],
    checker_type: OverlappingCheckerType,
) -> CollisionHandlerBase:
    """Factory function to create a collision handler for a given sharding type."""
    if sharding_type == "row_wise":
        return RWCollisionHandler(
            device=device,
            grouped_emb_configs=grouped_emb_configs,
            table_name_to_config=table_name_to_config,
            checker_type=checker_type,
        )
    raise ValueError(
        f"PEC collision detection does not support sharding type: {sharding_type}"
    )
