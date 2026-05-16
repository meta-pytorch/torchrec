#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import sys
from typing import cast, Iterator, List, Optional

import torch
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _RandomRecBatch:
    generator: Optional[torch.Generator]

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_sizes: List[int],
        ids_per_features: List[int],
        num_dense: int,
        manual_seed: Optional[int] = None,
        num_generated_batches: int = 10,
        num_batches: Optional[int] = None,
        *,
        min_ids_per_features: Optional[List[int]] = None,
    ) -> None:

        self.keys = keys
        self.keys_length: int = len(keys)
        self.batch_size = batch_size
        self.hash_sizes = hash_sizes
        self.ids_per_features = ids_per_features
        self.min_ids_per_features: List[int] = (
            min_ids_per_features
            if min_ids_per_features
            else [0] * len(ids_per_features)
        )
        self.num_dense = num_dense
        self.num_batches = num_batches
        self.num_generated_batches = num_generated_batches

        if manual_seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(manual_seed)
        else:
            self.generator = None

        self._generated_batches: List[Batch] = [
            self._generate_batch() for _ in range(num_generated_batches)
        ]
        self.batch_index = 0

    def __iter__(self) -> "_RandomRecBatch":
        self.batch_index = 0
        return self

    def __next__(self) -> Batch:
        if self.batch_index == self.num_batches:
            raise StopIteration
        if self.num_generated_batches >= 0:
            batch = self._generated_batches[
                self.batch_index % len(self._generated_batches)
            ]
        else:
            batch = self._generate_batch()
        self.batch_index += 1
        return batch

    def _generate_batch(self) -> Batch:

        values = []
        lengths = []
        for key_idx, _ in enumerate(self.keys):
            hash_size = self.hash_sizes[key_idx]
            min_num_ids = self.min_ids_per_features[key_idx]
            max_num_ids = self.ids_per_features[key_idx]
            length = torch.randint(
                min_num_ids,
                max_num_ids + 1,
                (self.batch_size,),
                dtype=torch.int32,
                generator=self.generator,
            )
            value = torch.randint(
                0, hash_size, (cast(int, length.sum()),), generator=self.generator
            )
            lengths.append(length)
            values.append(value)

        sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=self.keys,
            values=torch.cat(values),
            lengths=torch.cat(lengths),
        )

        dense_features = torch.randn(
            self.batch_size,
            self.num_dense,
            generator=self.generator,
        )
        labels = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size,),
            generator=self.generator,
        )

        batch = Batch(
            dense_features=dense_features,
            sparse_features=sparse_features,
            labels=labels,
        )
        return batch


class RandomRecDataset(IterableDataset[Batch]):
    """
    Random iterable dataset used to generate batches for recommender systems
    (RecSys). Currently produces unweighted sparse features only. TODO: Add
    weighted sparse features.

    Args:
        keys (List[str]): List of feature names for sparse features.
        batch_size (int): batch size.
        hash_size (Optional[int]): Max sparse id value. All sparse IDs will be taken
            modulo this value.
        hash_sizes (Optional[List[int]]): Max sparse id value per feature in keys. Each
            sparse ID will be taken modulo the corresponding value from this argument. Note, if this is used, hash_size will be ignored.
        ids_per_feature (Optional[int]): Number of IDs per sparse feature per sample.
        ids_per_features (Optional[List[int]]): Number of IDs per sparse feature per sample in each key. Note, if this is used, ids_per_feature will be ignored.
        num_dense (int): Number of dense features.
        manual_seed (int): Seed for deterministic behavior.
        num_batches: (Optional[int]): Num batches to generate before raising StopIteration
        num_generated_batches int: Num batches to cache. If num_batches > num_generated batches, then we will cycle to the first generated batch.
                                   If this value is negative, batches will be generated on the fly.
        min_ids_per_feature (Optional[int]): Minimum number of IDs per features.
        min_ids_per_features (Optional[List[int]]): Minimum number of IDs per sparse feature per sample in each key. Note, if this is used, min_ids_per_feature will be ignored.

    Example::

        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100_000,
            ids_per_feature=1,
            num_dense=13,
        ),
        example = next(iter(dataset))
    """

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_size: Optional[int] = None,
        hash_sizes: Optional[List[int]] = None,
        ids_per_feature: Optional[int] = None,
        ids_per_features: Optional[List[int]] = None,
        num_dense: int = 50,
        manual_seed: Optional[int] = None,
        num_batches: Optional[int] = None,
        num_generated_batches: int = 10,
        min_ids_per_feature: Optional[int] = None,
        min_ids_per_features: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if hash_sizes is None:
            hash_sizes = [hash_size if hash_size else 100] * len(keys)

        assert hash_sizes is not None
        assert len(hash_sizes) == len(
            keys
        ), "length of hash_sizes must be equal to the number of keys"

        if ids_per_features is None:
            ids_per_features = [ids_per_feature if ids_per_feature else 2] * len(keys)

        assert ids_per_features is not None

        if min_ids_per_features is None:
            min_ids_per_feature = (
                min_ids_per_feature
                if min_ids_per_feature is not None
                else ids_per_feature
            )
            min_ids_per_features = [
                min_ids_per_feature if min_ids_per_feature else 0
            ] * len(keys)

        assert len(ids_per_features) == len(
            keys
        ), "length of ids_per_features must be equal to the number of keys"

        self.batch_generator = _RandomRecBatch(
            keys=keys,
            batch_size=batch_size,
            hash_sizes=hash_sizes,
            ids_per_features=ids_per_features,
            num_dense=num_dense,
            manual_seed=manual_seed,
            num_batches=None,
            num_generated_batches=num_generated_batches,
            min_ids_per_features=min_ids_per_features,
        )
        self.num_batches: int = cast(
            int, num_batches if num_batches is not None else sys.maxsize
        )

    def __iter__(self) -> Iterator[Batch]:
        return itertools.islice(iter(self.batch_generator), self.num_batches)

    def __len__(self) -> int:
        return self.num_batches
