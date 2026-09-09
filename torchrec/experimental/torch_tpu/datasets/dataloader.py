#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading utilities for TPU SparseCore."""

import dataclasses
from typing import Any, Generator

from torchrec.datasets.utils import Batch
from torchrec.experimental.torch_tpu.datasets.input_preprocessing import (
    KeyedSparseCorePreprocessedInput,
    SparseCoreInputPreprocessor,
)

dataclass = dataclasses.dataclass


@dataclass
class SparseCoreBatch(Batch):
    """Batch containing preprocessed sparse features for SparseCore."""

    sparse_features: KeyedSparseCorePreprocessedInput

    @classmethod
    def from_batch(
        cls, batch: Batch, preprocessor: SparseCoreInputPreprocessor
    ) -> "SparseCoreBatch":
        """Creates a SparseCoreBatch from a standard Batch by preprocessing sparse features on CPU."""
        preprocessed_cpu = preprocessor(batch.sparse_features)
        return cls(
            dense_features=batch.dense_features,
            sparse_features=preprocessed_cpu,
            labels=batch.labels,
        )


class SparseCoreDataLoader:
    """Wrapper around a PyTorch DataLoader to perform CPU preprocessing eagerly."""

    def __init__(self, dataloader: Any, preprocessor: SparseCoreInputPreprocessor):
        self.dataloader = dataloader
        self.preprocessor = preprocessor

    def __iter__(self) -> Generator[SparseCoreBatch, None, None]:
        for batch in self.dataloader:
            yield SparseCoreBatch.from_batch(batch, self.preprocessor)

    def __len__(self) -> int:
        return len(self.dataloader)
