#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import unittest

from hypothesis import given, settings, strategies as st
from torchrec.datasets.random import RandomRecDataset


class RandomDataLoader(unittest.TestCase):
    def test_hash_per_feature_ids_per_feature(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_sizes=[100, 200],
            ids_per_features=[100, 200],
            min_ids_per_features=[100, 200],
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 100)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 200)

    def test_hash_ids_per_feature(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_features=[100, 200],
            min_ids_per_features=[100, 200],
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 100)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 200)

    @given(
        batch_size=st.sampled_from([2048, 4096, 8192]),
    )
    @settings(max_examples=3, deadline=5000)  # expected runtime <=500ms
    def test_large_batch_size_deadline(self, batch_size: int) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=batch_size,
            ids_per_features=[10, 20],
            hash_size=100,
            num_dense=5,
        )
        iterator = iter(dataset)
        for _ in range(5):
            next(iterator)

    def test_hash_ids(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            min_ids_per_feature=50,
            ids_per_feature=50,
            num_dense=5,
        )

        example = next(iter(dataset))
        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 50)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 50)

    def test_on_fly_batch_generation(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_feature=50,
            min_ids_per_feature=50,
            num_dense=5,
            num_generated_batches=-1,
        )

        it = iter(dataset)

        example = next(it)
        example = next(it)
        example = next(it)
        example = next(it)

        dense = example.dense_features
        self.assertEqual(dense.shape, (16, 5))

        labels = example.labels
        self.assertEqual(labels.shape, (16,))

        sparse = example.sparse_features
        self.assertEqual(sparse.stride(), 16)

        feat1 = sparse["feat1"].to_dense()
        self.assertEqual(len(feat1), 16)
        for batch in feat1:
            self.assertEqual(len(batch), 50)

        feat2 = sparse["feat2"].to_dense()
        self.assertEqual(len(feat2), 16)
        for batch in feat2:
            self.assertEqual(len(batch), 50)

    # We want RandomRecDataset to support len() and
    # itertools.chain() so the random dataloader can
    # run the same code as real dataset dataloaders
    # and substitute when wanted without issue.
    def test_len_and_itertools_chain(self) -> None:
        dataset = RandomRecDataset(
            keys=["feat1", "feat2"],
            batch_size=16,
            hash_size=100,
            ids_per_feature=50,
            num_dense=5,
            num_generated_batches=-1,
            num_batches=5,
        )
        self.assertEqual(len(dataset), 5)
        it = itertools.chain(iter(dataset), iter(dataset))
        for _ in range(10):
            next(it)
        self.assertRaises(StopIteration, lambda: next(it))
