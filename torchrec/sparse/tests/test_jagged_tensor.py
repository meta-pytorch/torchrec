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

    def test_to_dense_stacked(self) -> None:
        """Test to_dense_stacked method for various cases."""
        # Test case 1: All sequences have length 1 (ultra-fast path)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        lengths = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        jt = JaggedTensor(values=values, lengths=lengths)

        dense_stacked = jt.to_dense_stacked()
        expected = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.assertTrue(torch.equal(dense_stacked, expected))

        # Test with 2D values - length 1 sequences
        values_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        lengths = torch.tensor([1, 1, 1], dtype=torch.int32)
        jt_2d = JaggedTensor(values=values_2d, lengths=lengths)

        dense_stacked_2d = jt_2d.to_dense_stacked()
        expected_2d = values_2d.view(3, 2)  # Should match original shape
        self.assertTrue(torch.equal(dense_stacked_2d, expected_2d))

        # Test case 2: All sequences have same uniform length (fast path)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        lengths = torch.tensor([2, 2, 2], dtype=torch.int32)
        jt = JaggedTensor(values=values, lengths=lengths)

        dense_stacked = jt.to_dense_stacked()
        # Compare with to_padded_dense to ensure consistency
        expected_padded = jt.to_padded_dense()
        self.assertTrue(torch.equal(dense_stacked, expected_padded))

        # Test case 3: Variable lengths (fallback path)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.tensor([2, 0, 3, 1, 2], dtype=torch.int32)
        jt = JaggedTensor(values=values, lengths=lengths)

        dense_stacked = jt.to_dense_stacked()
        # Should be equivalent to to_padded_dense for variable lengths
        expected_padded = jt.to_padded_dense()
        self.assertTrue(torch.equal(dense_stacked, expected_padded))

        # Test case 4: Empty tensor
        empty_values = torch.tensor([], dtype=torch.float32)
        empty_lengths = torch.tensor([], dtype=torch.int32)
        jt_empty = JaggedTensor(values=empty_values, lengths=empty_lengths)

        dense_stacked_empty = jt_empty.to_dense_stacked()
        # Empty case should result in empty tensor with correct shape
        self.assertEqual(dense_stacked_empty.numel(), 0)

        # Test case 5: Single batch with multiple elements
        values = torch.tensor([10.0, 20.0, 30.0])
        lengths = torch.tensor([3], dtype=torch.int32)
        jt_single = JaggedTensor(values=values, lengths=lengths)

        dense_stacked_single = jt_single.to_dense_stacked()
        expected_single = torch.tensor([[10.0, 20.0, 30.0]])
        self.assertTrue(torch.equal(dense_stacked_single, expected_single))

        # Test case 6: Mix of empty and non-empty sequences (variable length)
        values = torch.tensor([1.0, 2.0, 3.0])
        lengths = torch.tensor([2, 0, 1], dtype=torch.int32)
        jt_mixed = JaggedTensor(values=values, lengths=lengths)

        dense_stacked_mixed = jt_mixed.to_dense_stacked()
        expected_mixed = jt_mixed.to_padded_dense()
        self.assertTrue(torch.equal(dense_stacked_mixed, expected_mixed))

        # Test case 7: Performance comparison - ensure to_dense_stacked matches to_padded_dense
        # for correctness on various shapes
        values = torch.randn(100)
        lengths = torch.randint(1, 10, (20,), dtype=torch.int32)
        # Adjust lengths to match values size
        total_needed = values.size(0)
        current_sum = int(lengths.sum().item())
        if current_sum != total_needed:
            # Adjust the last length to make the sum correct
            lengths[-1] = lengths[-1] + (total_needed - current_sum)

        jt_large = JaggedTensor(values=values, lengths=lengths)

        dense_stacked_large = jt_large.to_dense_stacked()
        expected_padded_large = jt_large.to_padded_dense()
        self.assertTrue(torch.equal(dense_stacked_large, expected_padded_large))

        # Test case 8: Different data types
        values_int = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        lengths_int = torch.tensor([2, 2, 2], dtype=torch.int32)
        jt_int = JaggedTensor(values=values_int, lengths=lengths_int)

        dense_stacked_int = jt_int.to_dense_stacked()
        self.assertEqual(dense_stacked_int.dtype, torch.int64)
        expected_int = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
        self.assertTrue(torch.equal(dense_stacked_int, expected_int))

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
