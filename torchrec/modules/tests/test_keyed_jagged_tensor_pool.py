#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
from torchrec.modules.keyed_jagged_tensor_pool import KeyedJaggedTensorPool
from torchrec.modules.object_pool_lookups import (
    _assert_valid_key_lengths,
    TensorJaggedIndexSelectLookup,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class KeyedJaggedTensorPoolTest(unittest.TestCase):
    def test_update_lookup(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        values_dtype = torch.int64

        keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=values_dtype,
            device=device,
        )

        # init global state is
        # 4         8
        # f1       f2
        # [3,3] .  [13,13,13]
        # [2,2] .  [12,12]
        # [1] .    [11]
        # [4]      [14,14,14,14]

        keyed_jagged_tensor_pool.update(
            ids=torch.tensor([2, 0, 1, 3], device=device),
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.tensor(
                    [1, 3, 3, 2, 2, 4, 11, 13, 13, 13, 12, 12, 14, 14, 14, 14],
                    dtype=values_dtype,
                    device=device,
                ),
                lengths=torch.tensor(
                    [1, 2, 2, 1, 1, 3, 2, 4], dtype=torch.int, device=device
                ),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([2, 0], device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [[1], [3, 3]],
        #     "f2": [[11], [13, 13, 13]]
        # })

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor(
                [1, 3, 3, 11, 13, 13, 13],
                dtype=values_dtype,
                device=torch.device("cpu"),
            ),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor(
                [1, 2, 1, 3],
                dtype=torch.int,
                device=torch.device("cpu"),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([1, 3, 0, 2], device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [[2, 2], [4], [3, 3], [1]],
        #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
        # })

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor(
                [2, 2, 4, 3, 3, 1, 12, 12, 14, 14, 14, 14, 13, 13, 13, 11],
                dtype=values_dtype,
                device=torch.device("cpu"),
            ),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor(
                [2, 1, 2, 1, 2, 4, 3, 1],
                dtype=torch.int,
                device=torch.device("cpu"),
            ),
        )

    def test_input_permute(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        values_dtype = torch.int32

        keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=values_dtype,
            device=device,
        )

        # init global state is
        # 4         8
        # f1       f2               f3
        # [3,3] .  [13,13,13]       [23]
        # [2,2] .  [12,12]          [22, 22, 22]
        # [1] .    [11]             [21, 21]
        # [4]      [14,14,14,14]    []

        keyed_jagged_tensor_pool.update(
            ids=torch.tensor([2, 0, 1, 3], device=device),
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f2", "f3", "f1"],
                values=torch.tensor(
                    [
                        11,
                        13,
                        13,
                        13,
                        12,
                        12,
                        14,
                        14,
                        14,
                        14,
                        21,
                        21,
                        23,
                        22,
                        22,
                        22,
                        1,
                        3,
                        3,
                        2,
                        2,
                        4,
                    ],
                    dtype=values_dtype,
                    device=device,
                ),
                lengths=torch.tensor(
                    [1, 3, 2, 4, 2, 1, 3, 0, 1, 2, 2, 1], dtype=torch.int, device=device
                ),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([2, 0], device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [[1], [3, 3]],
        #     "f2": [[11], [13, 13, 13]]
        # })

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor(
                [1, 3, 3, 11, 13, 13, 13],
                dtype=values_dtype,
                device=torch.device("cpu"),
            ),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor(
                [1, 2, 1, 3],
                dtype=torch.int,
                device=torch.device("cpu"),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([1, 3, 0, 2], device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [[2, 2], [4], [3, 3], [1]],
        #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
        # })

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor(
                [2, 2, 4, 3, 3, 1, 12, 12, 14, 14, 14, 14, 13, 13, 13, 11],
                dtype=values_dtype,
                device=torch.device("cpu"),
            ),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor(
                [2, 1, 2, 1, 2, 4, 3, 1],
                dtype=torch.int,
                device=torch.device("cpu"),
            ),
        )

    def test_conflict(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        values_dtype = torch.int32

        keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=values_dtype,
            device=device,
        )

        # input is
        # ids    f1       f2
        # 2      [1]      [11]
        # 0      [3,3] .  [13,13,13]
        # 2      [2,2]    [12,12]
        # 3      [4]      [14,14,14,14]

        keyed_jagged_tensor_pool.update(
            ids=torch.tensor([2, 0, 2, 3], device=device),
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.tensor(
                    [1, 3, 3, 2, 2, 4, 11, 13, 13, 13, 12, 12, 14, 14, 14, 14],
                    dtype=values_dtype,
                    device=device,
                ),
                lengths=torch.tensor(
                    [1, 2, 2, 1, 1, 3, 2, 4], dtype=torch.int, device=device
                ),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([2, 0], device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [[2,2], [3, 3]],
        #     "f2": [[12, 12], [13, 13, 13]]
        # })

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor(
                [2, 2, 3, 3, 12, 12, 13, 13, 13],
                dtype=values_dtype,
                device=torch.device("cpu"),
            ),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor(
                [2, 2, 2, 3],
                dtype=torch.int,
                device=torch.device("cpu"),
            ),
        )

    def test_empty_lookup(
        self,
    ) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        values_dtype = torch.int32

        keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=values_dtype,
            device=device,
        )

        # init global state is
        # 4         8
        # f1       f2
        # [3,3] .  [13,13,13]
        # [2,2] .  [12,12]
        # [1] .    [11]
        # [4]      [14,14,14,14]

        ids = torch.tensor([2, 0, 1, 3], device=device)
        keyed_jagged_tensor_pool.update(
            ids=ids,
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.tensor(
                    [1, 3, 3, 2, 2, 4, 11, 13, 13, 13, 12, 12, 14, 14, 14, 14],
                    dtype=values_dtype,
                    device=device,
                ),
                lengths=torch.tensor(
                    [1, 2, 2, 1, 1, 3, 2, 4], dtype=torch.int, device=device
                ),
            ),
        )

        kjt = keyed_jagged_tensor_pool.lookup(
            ids=torch.tensor([], dtype=ids.dtype, device=device),
        )

        # expected values
        # KeyedJaggedTensor({
        #     "f1": [],
        #     "f2": [],
        # })

        self.assertEqual(kjt.keys(), ["f1", "f2"])

        torch.testing.assert_close(
            kjt.values().cpu(),
            torch.tensor([], dtype=values_dtype, device=torch.device("cpu")),
        )

        torch.testing.assert_close(
            kjt.lengths().cpu(),
            torch.tensor([], dtype=torch.int, device=torch.device("cpu")),
        )


class KeyedJaggedTensorPoolLengthValidationTest(unittest.TestCase):
    """Regression coverage for T273509522.

    A corrupted / desynced row-wise update all-to-all delivers out-of-range key
    lengths (negative or +/-2**31 overflow) that used to be written silently
    into ``_key_lengths`` and only crash much later in ``lookup()`` as a
    negative dimension in ``jagged_index_select``. ``update()`` must now reject
    them at the write site.
    """

    def test_assert_valid_key_lengths_accepts_valid(self) -> None:
        feature_max_lengths_t = torch.tensor([2, 4], dtype=torch.int32)
        # In-range lengths (including 0 and the per-feature max) must pass.
        _assert_valid_key_lengths(
            torch.tensor([[1, 3], [2, 0], [0, 4]], dtype=torch.int32),
            feature_max_lengths_t,
        )
        # An empty batch is a no-op (e.g. a rank that received nothing).
        _assert_valid_key_lengths(
            torch.zeros((0, 2), dtype=torch.int32), feature_max_lengths_t
        )

    def test_assert_valid_key_lengths_rejects_negative(self) -> None:
        feature_max_lengths_t = torch.tensor([2, 4], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "out-of-range key lengths"):
            _assert_valid_key_lengths(
                torch.tensor([[1, -7], [2, 0]], dtype=torch.int32),
                feature_max_lengths_t,
            )

    def test_assert_valid_key_lengths_rejects_over_max(self) -> None:
        feature_max_lengths_t = torch.tensor([2, 4], dtype=torch.int32)
        # f1 length 5 exceeds its configured max of 2.
        with self.assertRaisesRegex(RuntimeError, "out-of-range key lengths"):
            _assert_valid_key_lengths(
                torch.tensor([[5, 1]], dtype=torch.int32), feature_max_lengths_t
            )

    def test_assert_valid_key_lengths_rejects_overflow_magnitude(self) -> None:
        # The actual bug signature: lengths at the +/-2**31 int32 boundary.
        feature_max_lengths_t = torch.tensor([2, 4], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "out-of-range key lengths"):
            _assert_valid_key_lengths(
                torch.tensor([[2_115_462_454, -2_133_429_463]], dtype=torch.int32),
                feature_max_lengths_t,
            )

    def test_update_rejects_corrupted_lengths(self) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        lookup = TensorJaggedIndexSelectLookup(
            pool_size=4,
            values_dtype=torch.int64,
            feature_max_lengths={"f1": 2, "f2": 4},
            is_weighted=False,
            device=device,
        )
        # One id, two features: f1 length 5 exceeds its configured max of 2 -
        # exactly the kind of out-of-range length a corrupted update all-to-all
        # would deliver. update() must reject it instead of storing garbage.
        corrupted = JaggedTensor(
            values=torch.arange(6, dtype=torch.int64, device=device),
            lengths=torch.tensor([5, 1], dtype=torch.int, device=device),
        )
        with self.assertRaisesRegex(RuntimeError, "out-of-range key lengths"):
            lookup.update(torch.tensor([0], device=device), corrupted)

    def test_update_accepts_valid_lengths(self) -> None:
        device = (
            torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda:0")
        )
        lookup = TensorJaggedIndexSelectLookup(
            pool_size=4,
            values_dtype=torch.int64,
            feature_max_lengths={"f1": 2, "f2": 4},
            is_weighted=False,
            device=device,
        )
        # In-range update must not raise (guards against false positives).
        valid = JaggedTensor(
            values=torch.arange(3, dtype=torch.int64, device=device),
            lengths=torch.tensor([2, 1], dtype=torch.int, device=device),
        )
        lookup.update(torch.tensor([0], device=device), valid)
