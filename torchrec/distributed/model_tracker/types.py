#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


@dataclass
class IndexedLookup:
    r"""
    Data class for storing per batch lookedup ids and embeddings or optimizer states.
    """

    batch_idx: int
    ids: torch.Tensor
    states: Optional[torch.Tensor]


@dataclass
class DeltaRows:
    r"""
    Data class as an interface for returning and storing compacted ids and embeddings or optimizer states.
    compact(List[IndexedLookup]) -> DeltaRows
    """

    ids: torch.Tensor
    states: Optional[torch.Tensor]


class TrackingMode(Enum):
    r"""
    Tracking mode for ``ModelDeltaTracker``.

    Enums:
        ID_ONLY:    Tracks row IDs only, providing a lightweight option for monitoring.
        EMBEDDING:  Tracks both row IDs and their corresponding embedding values,
                    enabling precise top-k result calculations. However, this option comes
                    with increased memory usage.
        MOMENTUM_LAST:  Tracks both row IDs and their corresponding momentum values. This mode
                        supports approximate top-k delta-row selection.
        MOMENTUM_DIFF: Tracks both row IDs and their corresponding momentum difference values.
        ROWWISE_ADAGRAD: Tracks both row IDs and their corresponding rowwise adagrad states.
    """

    ID_ONLY = "id_only"
    EMBEDDING = "embedding"
    MOMENTUM_LAST = "momentum_last"
    MOMENTUM_DIFF = "momentum_diff"
    ROWWISE_ADAGRAD = "rowwise_adagrad"


class EmbdUpdateMode(Enum):
    r"""
    To identify which embedding value to store while tracking.

    Enums:
        NONE: Used for id only mode when we aren't tracking the embeddings.
        FIRST: Stores the earlier embedding value for each id. Useful for checkpoint/snapshot.
        LAST: Stores the latest embedding value for each id. Used for some opmtimizer state modes.
    """

    NONE = "none"
    FIRST = "first"
    LAST = "last"


@dataclass
class ModelTrackerConfig:
    r"""
    Configuration for ``ModelDeltaTracker``.

    Args:
        tracking_mode (TrackingMode): tracking mode for the delta tracker.
        consumers (Optional[List[str]]): list of consumers for the delta tracker.
        delete_on_read (bool): whether to delete the compacted data after get_delta method is called.

    """

    tracking_mode: TrackingMode = TrackingMode.ID_ONLY
    consumers: Optional[List[str]] = None
    delete_on_read: bool = True
    auto_compact: bool = False
    fqns_to_skip: List[str] = field(default_factory=list)
