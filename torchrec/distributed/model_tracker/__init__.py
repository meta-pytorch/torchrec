#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Torchrec Model Tracker

The model tracker module provides functionality to track and retrieve unique IDs and
embeddings for supported modules during training. This is useful for identifying and
retrieving the latest delta or unique rows for a model, which can help compute topk
or to stream updated embeddings from predictors to trainers during online training.

Key features include:
- Tracking unique IDs and embeddings for supported modules
- Support for multiple consumers with independent tracking
- Configurable tracking modes (ID_ONLY, EMBEDDING)
- Compaction of tracked data to reduce memory usage
"""

from torchrec.distributed.model_tracker.delta_store import DeltaStore as DeltaStore
from torchrec.distributed.model_tracker.model_delta_tracker import (
    ModelDeltaTracker as ModelDeltaTracker,
    SUPPORTED_MODULES as SUPPORTED_MODULES,
)
from torchrec.distributed.model_tracker.types import (
    IndexedLookup as IndexedLookup,
    ModelTrackerConfigs as ModelTrackerConfigs,
    Trackers as Trackers,
    TrackingMode as TrackingMode,
    UniqueRows as UniqueRows,
    UpdateMode as UpdateMode,
)
