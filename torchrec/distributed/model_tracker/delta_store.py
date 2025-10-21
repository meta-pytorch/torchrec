#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Dict, List, Optional

import torch
from torchrec.distributed.model_tracker.types import (
    DeltaRows,
    EmbdUpdateMode,
    IndexedLookup,
)
from torchrec.distributed.utils import none_throws


def _compute_unique_rows(
    ids: List[torch.Tensor],
    states: Optional[List[torch.Tensor]],
    mode: EmbdUpdateMode,
) -> DeltaRows:
    r"""
    To calculate unique ids and embeddings
    """
    if mode == EmbdUpdateMode.NONE:
        assert states is None, f"{mode=} == EmbdUpdateMode.NONE but received embeddings"
        unique_ids = torch.cat(ids).unique(return_inverse=False)
        return DeltaRows(ids=unique_ids, states=None)
    else:
        assert (
            states is not None
        ), f"{mode=} != EmbdUpdateMode.NONE but received no embeddings"

        cat_ids = torch.cat(ids)
        cat_states = torch.cat(states)

        if mode == EmbdUpdateMode.LAST:
            cat_ids = cat_ids.flip(dims=[0])
            cat_states = cat_states.flip(dims=[0])

        # Get unique ids and inverse mapping (each element's index in unique_ids).
        unique_ids, inverse = cat_ids.unique(sorted=False, return_inverse=True)

        # Create a tensor of original indices. This will be used to find first occurrences of ids.
        all_indices = torch.arange(cat_ids.size(0), device=cat_ids.device)

        # Initialize tensor for first occurrence indices (filled with a high value).
        first_occurrence = torch.full(
            (unique_ids.size(0),),
            cat_ids.size(0),
            dtype=torch.int64,
            device=cat_ids.device,
        )

        # Scatter indices using inverse mapping and reduce with "amin" to get first or last (if reversed) occurrence per unique id.
        first_occurrence = first_occurrence.scatter_reduce(
            0, inverse, all_indices, reduce="amin"
        )

        # Use first occurrence indices to select corresponding embedding row.
        unique_states = cat_states[first_occurrence]
        return DeltaRows(ids=unique_ids, states=unique_states)


class DeltaStore(ABC):
    """
    DeltaStore is an abstract base class that defines the interface for storing and managing
    local delta (row) updates for embeddings/states across various batches during training.

    Implementations should maintain a representation of requested ids and embeddings/states,
    providing a way to compact and get delta updates for each embedding table.

    The class supports different embedding update modes (NONE, FIRST, LAST) to determine
    how to handle duplicate ids when compacting or retrieving embeddings.
    """

    @abstractmethod
    def __init__(self, embdUpdateMode: EmbdUpdateMode = EmbdUpdateMode.NONE) -> None:
        pass

    @abstractmethod
    def append(
        self,
        batch_idx: int,
        fqn: str,
        ids: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        """
        Append a batch of ids and states to the store for a specific table.

        Args:
            batch_idx: The batch index
            table_fqn: The fully qualified name of the table
            ids: The tensor of ids to append
            states: Optional tensor of states to append
        """
        pass

    @abstractmethod
    def delete(self, up_to_idx: Optional[int] = None) -> None:
        """
        Delete all idx from the store up to `up_to_idx`

        Args:
            up_to_idx: Optional index up to which to delete lookups
        """
        pass

    @abstractmethod
    def compact(self, start_idx: int, end_idx: int) -> None:
        """
        Compact (ids, embeddings) in batch index range from start_idx to end_idx.

        Args:
            start_idx: The starting batch index
            end_idx: The ending batch index
        """
        pass

    @abstractmethod
    def get_unique(self, from_idx: int = 0) -> Dict[str, DeltaRows]:
        """
        Return all unique/delta ids per table from the Delta Store.

        Args:
            from_idx: The batch index from which to get deltas

        Returns:
            A dictionary mapping table FQNs to their delta rows
        """
        pass


class DeltaStoreTrec(DeltaStore):
    """
    DeltaStoreTrec is a concrete implementation of DeltaStore that stores and manages
    local delta (row) updates for embeddings/states across various batches during training,
    designed to be used with TorchRecs ModelDeltaTracker.

    It maintains a CUDA in-memory representation of requested ids and embeddings/states,
    providing a way to compact and get delta updates for each embedding table.

    The class supports different embedding update modes (NONE, FIRST, LAST) to determine
    how to handle duplicate ids when compacting or retrieving embeddings.
    """

    def __init__(self, embdUpdateMode: EmbdUpdateMode = EmbdUpdateMode.NONE) -> None:
        super().__init__(embdUpdateMode)
        self.embdUpdateMode = embdUpdateMode
        self.per_fqn_lookups: Dict[str, List[IndexedLookup]] = {}

    def append(
        self,
        batch_idx: int,
        fqn: str,
        ids: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        table_fqn_lookup = self.per_fqn_lookups.get(fqn, [])
        table_fqn_lookup.append(
            IndexedLookup(batch_idx=batch_idx, ids=ids, states=states)
        )
        self.per_fqn_lookups[fqn] = table_fqn_lookup

    def delete(self, up_to_idx: Optional[int] = None) -> None:
        """
        Delete all idx from the store up to `up_to_idx`
        """
        if up_to_idx is None:
            # If up_to_idx is None, delete all lookups
            self.per_fqn_lookups = {}
        else:
            # lookups are sorted by idx.
            up_to_idx = none_throws(up_to_idx)
            for table_fqn, lookups in self.per_fqn_lookups.items():
                # remove all lookups up to up_to_idx
                self.per_fqn_lookups[table_fqn] = [
                    lookup for lookup in lookups if lookup.batch_idx >= up_to_idx
                ]

    def compact(self, start_idx: int, end_idx: int) -> None:
        r"""
        Compact (ids, embeddings) in batch index range from start_idx, curr_batch_idx.
        """
        assert (
            start_idx < end_idx
        ), f"start_idx {start_idx} must be smaller then end_idx, but got {end_idx}"

        new_per_fqn_lookups: Dict[str, List[IndexedLookup]] = {}
        for table_fqn, lookups in self.per_fqn_lookups.items():
            indexices = [h.batch_idx for h in lookups]
            index_l = bisect_left(indexices, start_idx)
            index_r = bisect_left(indexices, end_idx)
            lookups_to_compact = lookups[index_l:index_r]
            if len(lookups_to_compact) <= 1:
                new_per_fqn_lookups[table_fqn] = lookups
                continue
            ids = [lookup.ids for lookup in lookups_to_compact]
            states = (
                [none_throws(lookup.states) for lookup in lookups_to_compact]
                if self.embdUpdateMode != EmbdUpdateMode.NONE
                else None
            )
            delta_rows = _compute_unique_rows(
                ids=ids, states=states, mode=self.embdUpdateMode
            )
            new_per_fqn_lookups[table_fqn] = (
                lookups[:index_l]
                + [
                    IndexedLookup(
                        batch_idx=start_idx,
                        ids=delta_rows.ids,
                        states=delta_rows.states,
                    )
                ]
                + lookups[index_r:]
            )
        self.per_fqn_lookups = new_per_fqn_lookups

    def get_unique(self, from_idx: int = 0) -> Dict[str, DeltaRows]:
        r"""
        Return all unique/delta ids per table from the Delta Store.
        """

        delta_per_table_fqn: Dict[str, DeltaRows] = {}
        for table_fqn, lookups in self.per_fqn_lookups.items():
            compact_ids = [
                lookup.ids for lookup in lookups if lookup.batch_idx >= from_idx
            ]
            compact_states = (
                [
                    none_throws(lookup.states)
                    for lookup in lookups
                    if lookup.batch_idx >= from_idx
                ]
                if self.embdUpdateMode != EmbdUpdateMode.NONE
                else None
            )

            delta_per_table_fqn[table_fqn] = _compute_unique_rows(
                ids=compact_ids, states=compact_states, mode=self.embdUpdateMode
            )
        return delta_per_table_fqn
