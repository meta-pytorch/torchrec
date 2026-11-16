#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.modules.keyed_jagged_tensor_pool import KeyedJaggedTensorPool
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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
