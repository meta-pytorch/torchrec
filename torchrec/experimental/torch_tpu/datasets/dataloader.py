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
import queue
import threading
from typing import Any, Generator, Optional

import torch
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

    def to(self, device: torch.device, non_blocking: bool = False) -> "SparseCoreBatch":
        """Moves all batch tensors to the specified target device."""
        return SparseCoreBatch(
            dense_features=(
                self.dense_features.to(device, non_blocking=non_blocking)
                if self.dense_features is not None
                else None
            ),
            sparse_features=self.sparse_features.to(device, non_blocking=non_blocking),
            labels=(
                self.labels.to(device, non_blocking=non_blocking)
                if self.labels is not None
                else None
            ),
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


class PrefetchDataLoader:
    """Runs CPU preprocessing in a background thread and performs non-blocking H2D transfers."""

    def __init__(
        self,
        dataloader: Any,
        preprocessor: SparseCoreInputPreprocessor,
        device: torch.device,
        max_steps: Optional[int] = None,
        queue_size: int = 3,
    ):
        self.dataloader = dataloader
        self.preprocessor = preprocessor
        self.device = device
        self.max_steps = max_steps
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        try:
            for step, item in enumerate(self.dataloader):
                if (
                    self.max_steps is not None and step >= self.max_steps
                ) or self._stop_event.is_set():
                    break
                if isinstance(item, Batch):
                    preprocessed = SparseCoreBatch.from_batch(item, self.preprocessor)
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    dense_features, labels, kjt = item
                    preprocessed_features = self.preprocessor(kjt)
                    preprocessed = (dense_features, labels, preprocessed_features)
                else:
                    raise ValueError(f"Unsupported batch type: {type(item)}")
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(preprocessed, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:  # pylint: disable=broad-exception-caught
            while not self._stop_event.is_set():
                try:
                    self.queue.put(e, timeout=0.1)
                    break
                except queue.Full:
                    continue
        finally:
            self.queue.put(None)

    def __iter__(self) -> Generator[Any, None, None]:
        while True:
            item = self.queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, SparseCoreBatch):
                yield item.to(self.device, non_blocking=True)
            else:
                dense_features, labels, preprocessed_features = item
                if dense_features is not None:
                    dense_features = dense_features.to(self.device, non_blocking=True)
                if labels is not None:
                    labels = labels.to(self.device, non_blocking=True)
                preprocessed_features = preprocessed_features.to(
                    self.device, non_blocking=True
                )
                yield dense_features, labels, preprocessed_features

    def stop(self) -> None:
        self._stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def __enter__(self) -> "PrefetchDataLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
