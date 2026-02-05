#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import torch
import torch.utils._pytree as pytree
from torch.testing import FileCheck
from torchrec.fx import symbolic_trace
from torchrec.sparse.jagged_tensor import (
    ComputeJTDictToKJT,
    JaggedTensor,
    jt_is_equal,
    KeyedJaggedTensor,
)

torch.fx.wrap("len")


class TestJaggedTensor(unittest.TestCase):
    def test_equality(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        lengths = torch.IntTensor([1, 0, 2, 3])
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        """
        JaggedTensor representation from above values
        # [[1.0], [], [2.0, 3.0], [4.0, 5.0, 6.0]]
        """
        # JT equality, from different construction methods
        jt = JaggedTensor(values=values, lengths=lengths)
        dense_values = torch.Tensor(
            [[1.0, 11.0, 12.0], [9.0, 23.0, 11.0], [2.0, 3.0, 55.0], [4.0, 5.0, 6.0]]
        )
        jt_1 = JaggedTensor.from_dense_lengths(
            values=dense_values, lengths=torch.IntTensor([1, 0, 2, 3])
        )
        self.assertTrue(jt_is_equal(jt, jt_1))

        # Different values
        jt = JaggedTensor(
            values=torch.Tensor([2.0, 10.0, 11.0, 42.0, 3.0, 99.0]), lengths=lengths
        )
        self.assertFalse(jt_is_equal(jt, jt_1))

        # Different lengths
        jt = JaggedTensor(values=values, lengths=torch.IntTensor([1, 1, 0, 4]))
        self.assertFalse(jt_is_equal(jt, jt_1))

        # Including weights
        """
        # values: [[1.0], [], [2.0, 3.0], [4.0, 5.0, 6.0]]
        # weights: [[0.1], [], [0.2, 0.3], [0.4, 0.5 ,0.6]]
        """
        jt = JaggedTensor(values=values, lengths=lengths, weights=weights)

        dense_weights = torch.Tensor(
            [[0.1, 1.1, 1.2], [0.9, 2.3, 1.1], [0.2, 0.3, 5.5], [0.4, 0.5, 0.6]]
        )
        jt_1 = JaggedTensor.from_dense_lengths(
            values=dense_values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
            weights=dense_weights,
        )

        self.assertTrue(jt_is_equal(jt, jt_1))

        # Different weights
        jt = JaggedTensor(
            values=values,
            lengths=lengths,
            weights=torch.Tensor([1.4, 0.2, 3.2, 0.4, 42.0, 0.6]),
        )
        self.assertFalse(jt_is_equal(jt, jt_1))

        # from dense, equal lengths
        values_for_dense = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([2.0, 3.0]),
            torch.Tensor([4.0, 5.0, 6.0]),
        ]
        weights_for_dense = [
            torch.Tensor([0.1]),
            torch.Tensor(),
            torch.Tensor([0.2, 0.3]),
            torch.Tensor([0.4, 0.5, 0.6]),
        ]

        jt = JaggedTensor.from_dense(
            values=values_for_dense,
            weights=weights_for_dense,
        )

        self.assertTrue(jt_is_equal(jt, jt_1))

        # from dense, unequal lengths
        values_for_dense = [
            torch.Tensor([1.0]),
            torch.Tensor([3.0, 10.0, 42.0]),
            torch.Tensor([2.0, 3.0]),
            torch.Tensor([4.0, 5.0, 6.0]),
        ]
        weights_for_dense = [
            torch.Tensor([0.1]),
            torch.Tensor([0.3, 1.1, 4.2]),
            torch.Tensor([0.2, 0.3]),
            torch.Tensor([0.4, 0.5, 0.6]),
        ]

        jt = JaggedTensor.from_dense(
            values=values_for_dense,
            weights=weights_for_dense,
        )
        self.assertFalse(jt_is_equal(jt, jt_1))

        # wrong type
        jt = "not a jagged tensor"
        self.assertFalse(jt_is_equal(jt, jt_1))

    def test_str(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        j_1d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        jt_str = str(j_1d)
        expected_str = (
            "JaggedTensor({\n    [[1.0], [], [2.0, 3.0], [4.0, 5.0, 6.0]]\n})\n"
        )
        self.assertEqual(expected_str, jt_str)

        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        j_2d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        jt_str = str(j_2d)
        expected_str = "JaggedTensor({\n    [[[1.0, 2.0, 3.0]], [], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], []]\n})\n"
        self.assertEqual(expected_str, jt_str)

        values = torch.Tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0]],
            ]
        )
        j_3d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        with self.assertRaises(ValueError):
            jt_str = str(j_3d)

    def test_from_dense_lengths(self) -> None:
        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        j0 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(j0.weights_or_none() is None)

        traced_from_dense_lengths = torch.fx.symbolic_trace(
            JaggedTensor.from_dense_lengths
        )
        traced_j0 = traced_from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        self.assertTrue(torch.equal(traced_j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(
                traced_j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0])
            )
        )

    def test_from_dense_lengths_weighted(self) -> None:
        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        weights = 12.0 - values

        j1 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([2, 0, 1, 1]),
            weights=weights,
        )
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([1.0, 2.0, 7.0, 10.0])))
        self.assertTrue(torch.equal(j1.weights(), torch.Tensor([11.0, 10.0, 5.0, 2.0])))

        traced_from_dense_lengths = torch.fx.symbolic_trace(
            JaggedTensor.from_dense_lengths
        )
        traced_j1 = traced_from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([2, 0, 1, 1]),
            weights=weights,
        )
        self.assertTrue(torch.equal(traced_j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(
            torch.equal(traced_j1.values(), torch.Tensor([1.0, 2.0, 7.0, 10.0]))
        )
        self.assertTrue(
            torch.equal(traced_j1.weights(), torch.Tensor([11.0, 10.0, 5.0, 2.0]))
        )

    def test_from_dense(self) -> None:
        values = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]

        weights = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]
        j0 = JaggedTensor.from_dense(
            values=values,
        )
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(j0.weights_or_none() is None)
        j1 = JaggedTensor.from_dense(
            values=values,
            weights=weights,
        )
        self.assertTrue(torch.equal(j1.offsets(), torch.IntTensor([0, 1, 1, 3, 6])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_from_dense_device(self) -> None:
        device = torch.device("cuda", index=0)
        values = [
            torch.tensor([1.0], device=device),
            torch.tensor([7.0, 8.0], device=device),
            torch.tensor([10.0, 11.0, 12.0], device=device),
        ]

        j0 = JaggedTensor.from_dense(
            values=values,
        )
        self.assertEqual(j0.values().device, device)
        self.assertEqual(j0.lengths().device, device)
        self.assertEqual(j0.offsets().device, device)

    def test_to_dense(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        torch_list = jt.to_dense()
        expected_list = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([]),
            torch.tensor([3.0]),
            torch.tensor([4.0]),
            torch.tensor([5.0]),
            torch.tensor([6.0, 7.0, 8.0]),
        ]
        for t0, expected_t0 in zip(torch_list, expected_list):
            self.assertTrue(torch.equal(t0, expected_t0))

    def test_to_dense_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            weights=weights,
            offsets=offsets,
        )
        weights_list = jt.to_dense_weights()
        expected_weights_list = [
            torch.tensor([0.1, 0.2]),
            torch.tensor([]),
            torch.tensor([0.3]),
            torch.tensor([0.4]),
            torch.tensor([0.5]),
            torch.tensor([0.6, 0.7, 0.8]),
        ]
        for t0, expected_t0 in zip(weights_list, expected_weights_list):
            self.assertTrue(torch.equal(t0, expected_t0))

        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        weights_list = jt.to_dense_weights()
        self.assertIsNone(weights_list)

    def test_to_padded_dense(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).type(
            torch.float32
        )
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        t0 = jt.to_padded_dense()
        self.assertEqual(t0.dtype, torch.float32)
        t0_value = [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 7.0, 8.0],
        ]
        expected_t0 = torch.tensor(t0_value).type(torch.float32)
        self.assertTrue(torch.equal(t0, expected_t0))

        t1 = jt.to_padded_dense(desired_length=2, padding_value=10.0)
        self.assertEqual(t1.dtype, torch.float32)
        t1_value = [
            [1.0, 2.0],
            [10.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
            [6.0, 7.0],
        ]
        expected_t1 = torch.tensor(t1_value).type(torch.float32)
        self.assertTrue(torch.equal(t1, expected_t1))

        values = torch.Tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ).type(torch.int64)
        jt = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        t2 = jt.to_padded_dense(desired_length=3)
        self.assertEqual(t2.dtype, torch.int64)
        t2_value = [
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
        expected_t2 = torch.tensor(t2_value).type(torch.int64)
        self.assertTrue(torch.equal(t2, expected_t2))

    def test_to_padded_dense_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).type(
            torch.float64
        )
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            weights=weights,
            offsets=offsets,
        )
        t0_weights = jt.to_padded_dense_weights()
        expected_t0_weights = [
            [0.1, 0.2, 0.0],
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.6, 0.7, 0.8],
        ]

        expected_t0_weights = torch.tensor(expected_t0_weights)
        self.assertTrue(torch.equal(t0_weights, expected_t0_weights))

        t1_weights = jt.to_padded_dense_weights(desired_length=2, padding_value=1.0)
        expected_t1_weights = [
            [0.1, 0.2],
            [1.0, 1.0],
            [0.3, 1.0],
            [0.4, 1.0],
            [0.5, 1.0],
            [0.6, 0.7],
        ]
        expected_t1_weights = torch.tensor(expected_t1_weights)
        self.assertTrue(torch.equal(t1_weights, expected_t1_weights))

        values = torch.Tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ).type(torch.int64)
        weights = torch.Tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ]
        )
        jt = JaggedTensor(
            values=values,
            weights=weights,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        t2_weights = jt.to_padded_dense_weights(desired_length=3)
        expected_t2_weights = [
            [
                [0.1, 0.2, 0.3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
        expected_t2_weights = torch.tensor(expected_t2_weights)
        self.assertTrue(torch.equal(t2_weights, expected_t2_weights))

        jt = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        t3_weights = jt.to_padded_dense_weights(desired_length=3)
        self.assertIsNone(t3_weights)

    def test_key_lookup(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_split(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )
        j0, j1 = jag_tensor.split([1, 1])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), ["index_0"])
        self.assertEqual(j1.keys(), ["index_1"])
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_length_vs_offset(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3])

        j_offset = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        j_lens = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        self.assertTrue(torch.equal(j_offset.lengths(), j_lens.lengths()))
        self.assertTrue(torch.equal(j_offset.offsets(), j_lens.offsets().int()))

        stride_per_key_per_rank = [[3], [5]]
        j_offset = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        j_lens = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )
        self.assertTrue(torch.equal(j_offset.lengths(), j_lens.lengths()))
        self.assertTrue(torch.equal(j_offset.offsets(), j_lens.offsets().int()))

    def test_empty(self) -> None:
        jt = JaggedTensor.empty(values_dtype=torch.int64)

        self.assertTrue(torch.equal(jt.values(), torch.tensor([], dtype=torch.int64)))
        self.assertTrue(torch.equal(jt.offsets(), torch.tensor([], dtype=torch.int32)))

        jt_from_script = torch.jit.script(JaggedTensor.empty)()
        self.assertEqual(jt_from_script.to_dense(), [])

    def test_2d(self) -> None:
        values = torch.Tensor([[i * 0.5, i * 1.0, i * 1.5] for i in range(1, 4)])
        offsets = torch.IntTensor([0, 2, 2, 3])

        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertTrue(torch.equal(jt.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(
            torch.equal(
                jt.values(),
                torch.Tensor(
                    [
                        [0.5, 1.0, 1.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 3.0, 4.5],
                    ],
                ),
            )
        )

    def test_float_lengths_offsets_throws(self) -> None:
        values = torch.rand((7, 3))
        lengths = torch.tensor([3.0, 4.0])
        offsets = torch.tensor([0.0, 3.0, 7.0])

        with self.assertRaises(AssertionError):
            JaggedTensor(values=values, lengths=lengths)
        with self.assertRaises(AssertionError):
            JaggedTensor(values=values, offsets=offsets)

    def test_to(self) -> None:
        j = JaggedTensor(
            offsets=torch.tensor([0, 2, 2, 3]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            weights=torch.tensor([5.0, 10.0, 15.0]),
        )
        j2 = j.to(device=torch.device("cpu"))
        self.assertTrue(torch.equal(j.offsets(), j2.offsets()))
        self.assertTrue(torch.equal(j.lengths(), j2.lengths()))
        self.assertTrue(torch.equal(j.values(), j2.values()))
        self.assertTrue(torch.equal(j.weights(), j2.weights()))

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_record_stream(self) -> None:
        j = JaggedTensor(
            offsets=torch.tensor([0, 2, 2, 3]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            weights=torch.tensor([5.0, 10.0, 15.0]),
        ).to(torch.device("cuda"))
        j.record_stream(torch.cuda.current_stream())

    def test_string_basic(self) -> None:
        values = torch.Tensor([1.0])
        offsets = torch.IntTensor([0, 1])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """JaggedTensor({\n    [[1.0]]\n})\n""",
        )

    def test_string_values(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            "JaggedTensor({\n    [[1.0, 2.0], [], [3.0], "
            "[4.0], [5.0], [6.0, 7.0, 8.0]]\n})\n",
        )

    def test_string_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
            weights=weights,
        )

        self.assertEqual(
            str(jag_tensor),
            'JaggedTensor({\n    "values": [[1.0, 2.0], [], [3.0], '
            '[4.0], [5.0], [6.0, 7.0, 8.0]],\n    "weights": '
            "[[1.0, 0.5], [], [1.5], [1.0], [0.5], [1.0, 1.0, 1.5]]\n})\n",
        )

    def test_pytree(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        j0 = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        elems, spec = pytree.tree_flatten(j0)
        j1 = pytree.tree_unflatten(elems, spec)

        self.assertTrue(torch.equal(j0.lengths(), j1.lengths()))
        self.assertIsNone(j0.weights_or_none())
        self.assertIsNone(j1.weights_or_none())
        self.assertTrue(torch.equal(j0.values(), j1.values()))

        values = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]
        weights = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]
        j0 = JaggedTensor.from_dense(
            values=values,
            weights=weights,
        )
        elems, spec = pytree.tree_flatten(j0)
        j1 = pytree.tree_unflatten(elems, spec)

        self.assertTrue(torch.equal(j0.lengths(), j1.lengths()))
        self.assertTrue(torch.equal(j0.weights(), j1.weights()))
        self.assertTrue(torch.equal(j0.values(), j1.values()))

    def test_compute_jt_dict_to_kjt_module(self) -> None:
        compute_jt_dict_to_kjt = ComputeJTDictToKJT()
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        jag_tensor_dict = jag_tensor.to_dict()
        kjt = compute_jt_dict_to_kjt(jag_tensor_dict)
        j0 = kjt["index_0"]
        j1 = kjt["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5, 1.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_from_jt_dict(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        jag_tensor_dict = jag_tensor.to_dict()
        kjt = KeyedJaggedTensor.from_jt_dict(jag_tensor_dict)
        j0 = kjt["index_0"]
        j1 = kjt["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5, 1.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_from_jt_dict_vb(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        stride_per_key_per_rank = [[2], [4]]

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )
        jag_tensor_dict = jag_tensor.to_dict()
        kjt = KeyedJaggedTensor.from_jt_dict(jag_tensor_dict)
        j0 = kjt["index_0"]
        j1 = kjt["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.5, 1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_to_dict_compute_offsets_false(self) -> None:
        # Setup: KJT with two keys and standard stride
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["f1", "f2"]
        lengths = torch.IntTensor([2, 0, 1, 1, 1, 3])

        kjt = KeyedJaggedTensor(values=values, keys=keys, lengths=lengths)

        # Execute: call to_dict with compute_offsets=False
        jt_dict = kjt.to_dict(compute_offsets=False)

        # Assert: offsets_or_none() should be None for each JaggedTensor
        self.assertIsNone(jt_dict["f1"].offsets_or_none())
        self.assertIsNone(jt_dict["f2"].offsets_or_none())
        # Lengths should still be available
        self.assertTrue(
            torch.equal(jt_dict["f1"].lengths(), torch.IntTensor([2, 0, 1]))
        )
        self.assertTrue(
            torch.equal(jt_dict["f2"].lengths(), torch.IntTensor([1, 1, 3]))
        )

    def test_to_dict_compute_offsets_false_variable_stride(self) -> None:
        # Setup: KJT with variable stride per key (reusing test_from_jt_dict_vb data)
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["f1", "f2"]
        lengths = torch.IntTensor([2, 0, 1, 1, 1, 3])
        stride_per_key_per_rank = [[2], [4]]

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        # Execute: call to_dict with compute_offsets=False
        jt_dict = kjt.to_dict(compute_offsets=False)

        # Assert: offsets_or_none() should be None for each JaggedTensor
        self.assertIsNone(jt_dict["f1"].offsets_or_none())
        self.assertIsNone(jt_dict["f2"].offsets_or_none())
        # Lengths should still be available
        self.assertTrue(torch.equal(jt_dict["f1"].lengths(), torch.IntTensor([2, 0])))
        self.assertTrue(
            torch.equal(jt_dict["f2"].lengths(), torch.IntTensor([1, 1, 1, 3]))
        )


class TestJaggedTensorTracing(unittest.TestCase):
    def test_jagged_tensor(self) -> None:
        class ModuleCreateAndAccessJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: int) -> int:
                features = JaggedTensor(
                    values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                    weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                    offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
                )
                return (
                    features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        class ModuleUseJaggedTensorAsInputAndOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: JaggedTensor) -> JaggedTensor:
                return JaggedTensor(
                    input.values(),
                    input.weights(),
                    lengths=input.lengths(),
                    offsets=input.offsets(),
                )

        class ModuleUseJaggedTensorAsInput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: JaggedTensor) -> int:
                return (
                    input.values().numel()
                    + input.weights().numel()
                    + input.lengths().numel()
                    + input.offsets().numel()
                )

        class ModuleUseJaggedTensorAsOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                values: torch.Tensor,
                weights: torch.Tensor,
                lengths: torch.Tensor,
            ) -> JaggedTensor:
                return JaggedTensor(values, weights, lengths)

        # Case 1: JaggedTensor is only used as an output of the root module.
        m = ModuleUseJaggedTensorAsOutput()
        gm = symbolic_trace(m)
        FileCheck().check("JaggedTensor").check("return jagged_tensor").run(gm.code)

        values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        weights = torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        lengths = torch.tensor([0, 2, 2, 3, 4, 5, 8])

        ref_jt = m(values, weights, lengths)
        traced_jt = gm(values, weights, lengths)

        self.assertTrue(torch.equal(traced_jt.values(), ref_jt.values()))
        self.assertTrue(torch.equal(traced_jt.weights(), ref_jt.weights()))
        self.assertTrue(torch.equal(traced_jt.lengths(), ref_jt.lengths()))

        # Case 2: JaggedTensor is only used as an input of the root module.
        m = ModuleUseJaggedTensorAsInput()
        gm = symbolic_trace(m)
        FileCheck().check("values()").check("numel()").check("weights").check(
            "lengths"
        ).check("offsets").run(gm.code)

        input = JaggedTensor(
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out, traced_out)

        # Case 3: JaggedTensor is used as both an input and an output of the root module.
        m = ModuleUseJaggedTensorAsInputAndOutput()
        gm = symbolic_trace(m)
        FileCheck().check("values()").check("weights").check("lengths").check(
            "offsets"
        ).check("JaggedTensor").run(gm.code)

        ref_out = m(input)
        traced_out = gm(input)
        self.assertTrue(torch.equal(traced_out.values(), ref_out.values()))
        self.assertTrue(torch.equal(traced_out.weights(), ref_out.weights()))
        self.assertTrue(torch.equal(traced_out.lengths(), ref_out.lengths()))

        # Case 4: JaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 29").check_not("JaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)

    def test_size_in_bytes(self) -> None:
        # Test with values only (float32 = 4 bytes per element)
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        lengths = torch.IntTensor([1, 0, 2, 3])
        jt = JaggedTensor(values=values, lengths=lengths)

        # Expected: 6 float32 values (24 bytes) + 4 int32 lengths (16 bytes)
        expected_size = 6 * 4 + 4 * 4
        self.assertEqual(jt.size_in_bytes(), expected_size)

    def test_size_in_bytes_with_weights(self) -> None:
        # Test with values and weights (both float32 = 4 bytes per element)
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        lengths = torch.IntTensor([1, 0, 2, 3])
        jt = JaggedTensor(values=values, lengths=lengths, weights=weights)

        # Expected: 6 float32 values (24 bytes) + 6 float32 weights (24 bytes) + 4 int32 lengths (16 bytes)
        expected_size = 6 * 4 + 6 * 4 + 4 * 4
        self.assertEqual(jt.size_in_bytes(), expected_size)

    def test_size_in_bytes_with_offsets(self) -> None:
        # Test with offsets instead of lengths
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        offsets = torch.IntTensor([0, 1, 1, 3, 6])
        jt = JaggedTensor(values=values, offsets=offsets)

        # Expected: 6 float32 values (24 bytes) + 5 int32 offsets (20 bytes)
        expected_size = 6 * 4 + 5 * 4
        self.assertEqual(jt.size_in_bytes(), expected_size)

    def test_size_in_bytes_different_dtypes(self) -> None:
        # Test with different dtypes (float64 = 8 bytes per element)
        values = torch.DoubleTensor([1.0, 2.0, 3.0, 4.0])
        lengths = torch.LongTensor([1, 3])

        jt = JaggedTensor(values=values, lengths=lengths)

        # Expected: 4 float64 values (32 bytes) + 2 int64 lengths (16 bytes)
        expected_size = 4 * 8 + 2 * 8
        self.assertEqual(jt.size_in_bytes(), expected_size)

    def test_empty_like_basic(self) -> None:
        # Setup: Create a weighted JaggedTensor
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        lengths = torch.tensor([2, 2, 3, 1])

        jt = JaggedTensor(
            values=values,
            lengths=lengths,
            weights=weights,
        )

        # Execute: Create an empty_like JaggedTensor
        empty_jt = JaggedTensor.empty_like(jt)

        # Assert: Verify the empty JT has the same dtypes but empty tensors
        self.assertEqual(empty_jt.device(), jt.device())
        self.assertEqual(empty_jt.values().dtype, jt.values().dtype)
        self.assertEqual(empty_jt.values().numel(), 0)
        self.assertEqual(empty_jt.lengths().numel(), 0)
        self.assertEqual(empty_jt.lengths().dtype, jt.lengths().dtype)
        self.assertIsNotNone(empty_jt.weights_or_none())
        self.assertEqual(empty_jt.weights().numel(), 0)
        self.assertEqual(empty_jt.weights().dtype, jt.weights().dtype)

    def test_empty_like_without_weights(self) -> None:
        # Setup: Create a non-weighted JaggedTensor
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        lengths = torch.tensor([2, 2])

        jt = JaggedTensor(
            values=values,
            lengths=lengths,
        )

        # Execute: Create an empty_like JaggedTensor
        empty_jt = JaggedTensor.empty_like(jt)

        # Assert: Verify the empty JT has no weights
        self.assertEqual(empty_jt.values().numel(), 0)
        self.assertEqual(empty_jt.lengths().numel(), 0)
        self.assertIsNone(empty_jt.weights_or_none())

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_empty_like_with_device(self) -> None:
        # Setup: Create a JaggedTensor on CPU
        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.1, 0.2, 0.3])
        lengths = torch.tensor([3])

        jt = JaggedTensor(
            values=values,
            lengths=lengths,
            weights=weights,
        )

        # Execute: Create an empty_like JaggedTensor on CUDA device (cross-device)
        empty_jt = JaggedTensor.empty_like(jt, device=torch.device("cuda"))

        # Assert: Verify the empty JT preserves structure and is on the correct device
        self.assertEqual(empty_jt.device().type, "cuda")
        self.assertEqual(empty_jt.values().dtype, jt.values().dtype)
        self.assertEqual(empty_jt.weights().dtype, jt.weights().dtype)
        # Assert: Verify tensors are allocated on CUDA
        self.assertTrue(empty_jt.values().is_cuda)
        self.assertTrue(empty_jt.lengths().is_cuda)
        self.assertTrue(empty_jt.weights().is_cuda)
        # Assert: Verify tensor sizes match the original
        self.assertEqual(empty_jt.values().size(), jt.values().size())
        self.assertEqual(empty_jt.weights().size(), jt.weights().size())
        self.assertEqual(empty_jt.lengths().size(), jt.lengths().size())

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_basic(self) -> None:
        # Setup: Create source JT on CPU and destination JT on CUDA
        source_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        source_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        source_lengths = torch.tensor([2, 2])

        source_jt = JaggedTensor(
            values=source_values,
            lengths=source_lengths,
            weights=source_weights,
        )

        dest_values = torch.zeros(4, device=torch.device("cuda"))
        dest_weights = torch.zeros(4, device=torch.device("cuda"))
        dest_lengths = torch.zeros(2, dtype=torch.int64, device=torch.device("cuda"))

        dest_jt = JaggedTensor(
            values=dest_values,
            lengths=dest_lengths,
            weights=dest_weights,
        )

        # Execute: Copy source JT (CPU) to destination JT (CUDA)
        result_jt = dest_jt.copy_(source_jt)

        # Assert: Verify the destination JT has the source values
        self.assertTrue(torch.equal(result_jt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_jt.weights().cpu(), source_weights))
        self.assertTrue(torch.equal(result_jt.lengths().cpu(), source_lengths))
        # Verify it returns the same object (in-place operation)
        self.assertIs(result_jt, dest_jt)
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_jt.values().is_cuda)
        self.assertTrue(result_jt.weights().is_cuda)
        self.assertTrue(result_jt.lengths().is_cuda)

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_without_weights(self) -> None:
        # Setup: Create source JT on CPU and destination JT on CUDA without weights
        source_values = torch.tensor([5.0, 6.0, 7.0])
        source_lengths = torch.tensor([1, 2])

        source_jt = JaggedTensor(
            values=source_values,
            lengths=source_lengths,
        )

        dest_values = torch.zeros(3, device=torch.device("cuda"))
        dest_lengths = torch.zeros(2, dtype=torch.int64, device=torch.device("cuda"))

        dest_jt = JaggedTensor(
            values=dest_values,
            lengths=dest_lengths,
        )

        # Execute: Copy source JT (CPU) to destination JT (CUDA)
        result_jt = dest_jt.copy_(source_jt)

        # Assert: Verify the destination JT has the source values
        self.assertTrue(torch.equal(result_jt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_jt.lengths().cpu(), source_lengths))
        self.assertIsNone(result_jt.weights_or_none())
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_jt.values().is_cuda)
        self.assertTrue(result_jt.lengths().is_cuda)

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_with_offsets(self) -> None:
        # Setup: Create source JT on CPU and destination JT on CUDA with offsets
        source_values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        source_offsets = torch.tensor([0, 2, 5])

        source_jt = JaggedTensor(
            values=source_values,
            offsets=source_offsets,
        )

        dest_values = torch.zeros(5, device=torch.device("cuda"))
        dest_offsets = torch.zeros(3, dtype=torch.int64, device=torch.device("cuda"))

        dest_jt = JaggedTensor(
            values=dest_values,
            offsets=dest_offsets,
        )

        # Execute: Copy source JT (CPU) to destination JT (CUDA)
        result_jt = dest_jt.copy_(source_jt)

        # Assert: Verify the destination JT has the source values and offsets
        self.assertTrue(torch.equal(result_jt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_jt.offsets().cpu(), source_offsets))
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_jt.values().is_cuda)
        self.assertTrue(result_jt.offsets().is_cuda)

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_non_blocking(self) -> None:
        # Setup: Create source JT on CPU and destination JT on CUDA
        source_values = torch.tensor([1.0, 2.0, 3.0])
        source_lengths = torch.tensor([3])

        source_jt = JaggedTensor(
            values=source_values,
            lengths=source_lengths,
        )

        dest_values = torch.zeros(3, device=torch.device("cuda"))
        dest_lengths = torch.zeros(1, dtype=torch.int64, device=torch.device("cuda"))

        dest_jt = JaggedTensor(
            values=dest_values,
            lengths=dest_lengths,
        )

        # Execute: Copy source JT (CPU) to destination JT (CUDA) with non_blocking=True
        result_jt = dest_jt.copy_(source_jt, non_blocking=True)

        # Synchronize to ensure the non-blocking copy completes before checking
        torch.cuda.synchronize()

        # Assert: Verify the copy succeeded
        self.assertTrue(torch.equal(result_jt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_jt.lengths().cpu(), source_lengths))
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_jt.values().is_cuda)
        self.assertTrue(result_jt.lengths().is_cuda)
