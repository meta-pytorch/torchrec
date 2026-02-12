#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import List, Tuple

import torch
import torch.utils._pytree as pytree
from torch.testing import FileCheck
from torchrec.fx import symbolic_trace
from torchrec.sparse.jagged_tensor import (
    ComputeKJTToJTDict,
    JaggedTensor,
    KeyedJaggedTensor,
    kjt_is_equal,
)
from torchrec.test_utils import skip_if_asan_class

torch.fx.wrap("len")


class TestKeyedJaggedTensor(unittest.TestCase):
    def test_key_lookup(self) -> None:
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
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]

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

    def test_key_lookup_vb(self) -> None:
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
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]

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

    def test_to_dict(self) -> None:
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
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

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

    def test_pytree_kjt(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        stride_per_key_per_rank = [[2], [4]]
        inverse_indices = torch.tensor([[0, 1, 0], [0, 0, 0]])

        kjt_0 = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=(keys, inverse_indices),
        )
        elems, spec = pytree.tree_flatten(kjt_0)
        kjt_1 = pytree.tree_unflatten(elems, spec)

        self.assertTrue(torch.equal(kjt_0.values(), kjt_1.values()))
        self.assertIsNone(kjt_0.lengths_or_none())
        self.assertIsNone(kjt_1.lengths_or_none())
        self.assertTrue(torch.equal(kjt_0.weights(), kjt_1.weights()))
        self.assertTrue(torch.equal(kjt_0.offsets(), kjt_1.offsets()))
        self.assertEqual(kjt_0.keys(), kjt_1.keys())
        self.assertEqual(
            kjt_0.stride_per_key_per_rank(), kjt_1.stride_per_key_per_rank()
        )
        self.assertEqual(kjt_0.inverse_indices()[0], kjt_1.inverse_indices()[0])
        self.assertTrue(
            torch.equal(kjt_0.inverse_indices()[1], kjt_1.inverse_indices()[1])
        )

        kjt_0 = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        elems, spec = pytree.tree_flatten(kjt_0)

        # Simulate missing stride_per_key_per_rank and inverse_indices
        spec = pytree.TreeSpec(
            type=spec.type,
            context=spec.context,
            children_specs=spec.children_specs[:4],
        )
        kjt_1 = pytree.tree_unflatten(elems[:4], spec)

        self.assertTrue(torch.equal(kjt_0.values(), kjt_1.values()))
        self.assertIsNone(kjt_0.lengths_or_none())
        self.assertIsNone(kjt_1.lengths_or_none())
        self.assertTrue(torch.equal(kjt_0.weights(), kjt_1.weights()))
        self.assertTrue(torch.equal(kjt_0.offsets(), kjt_1.offsets()))
        self.assertEqual(kjt_0.keys(), kjt_1.keys())
        self.assertTrue(len(kjt_0.stride_per_key_per_rank()) == 0)
        self.assertTrue(len(kjt_1.stride_per_key_per_rank()) == 0)
        self.assertIsNone(kjt_0.inverse_indices_or_none())
        self.assertIsNone(kjt_1.inverse_indices_or_none())

        kjt_0 = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        elems, spec = pytree.tree_flatten(kjt_0)

        # Simulate extra fields
        with self.assertRaises(ValueError):
            kjt_1 = pytree.tree_unflatten(elems + elems, spec)

    def test_to_dict_vb(self) -> None:
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
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

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

    def test_empty(self) -> None:
        keys = ["index_0"]
        values = torch.tensor([])
        lengths = torch.tensor([])
        offsets = torch.tensor([])

        kjt_0 = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        j0 = kjt_0["index_0"]
        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))

        keys = ["index_1"]
        kjt_1 = KeyedJaggedTensor(keys=keys, values=values, offsets=offsets)
        j1 = kjt_1["index_1"]

        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        combined_kjt = KeyedJaggedTensor.concat([kjt_0, kjt_1])
        j0 = combined_kjt["index_0"]
        j1 = combined_kjt["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        kjt_2 = KeyedJaggedTensor.empty()
        self.assertEqual(kjt_2.to_dict(), {})

        kjt_from_script = torch.jit.script(KeyedJaggedTensor.empty)()
        kjt_like = torch.jit.script(KeyedJaggedTensor.empty_like)(kjt_from_script)
        self.assertEqual(kjt_from_script.to_dict(), {})
        self.assertEqual(kjt_like.to_dict(), {})

    def test_empty_to_dict(self) -> None:
        keys = ["index_0", "index_1"]
        values = torch.tensor([])
        lengths = torch.tensor([[], []])
        length_per_key = [0, 0]

        jag_tensor = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, length_per_key=length_per_key
        )
        jag_tensor_dict = jag_tensor.to_dict()
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            keys=keys, values=values, lengths=lengths
        )
        jag_tensor_dict = jag_tensor.to_dict()
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

    def test_split(self) -> None:
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
        j0, j1 = jag_tensor.split([1, 1])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), ["index_0"])
        self.assertEqual(j1.keys(), ["index_1"])
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

    def test_empty_vb(self) -> None:
        keys = ["index_0"]
        values = torch.tensor([])
        lengths = torch.tensor([])
        stride_per_key_per_rank = [[]]

        kjt_0 = KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )
        self.assertTrue(torch.equal(kjt_0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(kjt_0.values(), torch.Tensor([])))
        self.assertEqual(kjt_0.stride(), 0)

    def test_split_vb(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        keys = ["index_0", "index_1", "index_2", "index_3"]
        lengths = torch.IntTensor([2, 0, 1, 1, 1, 3, 0, 2])
        stride_per_key_per_rank = [[3], [0], [1], [4]]
        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )
        j0, j1, j2 = jag_tensor.split([1, 1, 2])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), ["index_0"])
        self.assertEqual(j1.keys(), ["index_1"])
        self.assertEqual(j2.keys(), ["index_2", "index_3"])
        self.assertEqual(j0.stride(), 4)
        self.assertEqual(j1.stride(), 4)
        self.assertEqual(j2.stride(), 4)
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))
        self.assertTrue(torch.equal(j2.lengths(), torch.IntTensor([1, 1, 3, 0, 2])))
        self.assertTrue(
            torch.equal(j2.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        )

        j0, j1, j2, j3 = jag_tensor.split([0, 3, 0, 1])
        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), [])
        self.assertEqual(j1.keys(), ["index_0", "index_1", "index_2"])
        self.assertEqual(j2.keys(), [])
        self.assertEqual(j3.keys(), ["index_3"])
        self.assertEqual(j0.stride(), 4)
        self.assertEqual(j1.stride(), 4)
        self.assertEqual(j2.stride(), 4)
        self.assertEqual(j3.stride(), 4)
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([1.0, 2.0, 3.0, 4.0])))
        self.assertTrue(torch.equal(j2.lengths(), torch.IntTensor([])))
        self.assertTrue(torch.equal(j2.values(), torch.Tensor([])))
        self.assertTrue(torch.equal(j3.lengths(), torch.IntTensor([1, 3, 0, 2])))
        self.assertTrue(
            torch.equal(j3.values(), torch.Tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
        )

    def test_zero_split(self) -> None:
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
        j0, j1 = jag_tensor.split([0, 2])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), [])
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertEqual(j0.stride(), 3)

        self.assertEqual(j1.keys(), ["index_0", "index_1"])
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1, 1, 3])))
        self.assertTrue(torch.equal(j1.weights(), weights))
        self.assertTrue(torch.equal(j1.values(), values))
        self.assertEqual(j1.stride(), 3)

    def test_permute_w_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)
        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor([3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.weights(),
                torch.Tensor([1.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            ),
        )

    def test_permute(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor([3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_permute_vb(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([1, 0, 1, 3, 0, 1, 0, 2, 0])
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 5, 6, 8],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor([2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 3, 0, 1, 1, 0, 0, 2, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_permute_vb_duplicate(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([1, 0, 1, 3, 0, 1, 0, 2, 0])
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 1, 0, 0, 2, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_1", "index_0", "index_0", "index_2", "index_2"],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor(
                    [
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        1.0,
                        1.0,
                        7.0,
                        8.0,
                        7.0,
                        8.0,
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 3, 0, 1, 1, 3, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0, 2, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_permute_duplicates(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2, 1, 1]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_0", "index_2", "index_1", "index_1"],
        )
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8, 11, 14],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor(
                    [
                        3.0,
                        4.0,
                        5.0,
                        1.0,
                        2.0,
                        6.0,
                        7.0,
                        8.0,
                        3.0,
                        4.0,
                        5.0,
                        3.0,
                        4.0,
                        5.0,
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_concat(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        keys = ["index_0", "index_1", "index_2"]
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0, 0, 1, 0])

        kjt_expected = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )
        kjt_actual = KeyedJaggedTensor.concat(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    values=values[:4],
                    keys=keys[:1],
                    lengths=lengths[:4],
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    values=values[4:],
                    keys=keys[1:],
                    lengths=lengths[4:],
                ),
            ],
        )
        self.assertTrue(torch.equal(kjt_expected.lengths(), kjt_actual.lengths()))
        self.assertTrue(torch.equal(kjt_expected.offsets(), kjt_actual.offsets()))
        self.assertTrue(torch.equal(kjt_expected.values(), kjt_actual.values()))
        self.assertListEqual(kjt_expected._length_per_key, kjt_actual._length_per_key)

    def test_concat_fxable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, inputs: List[KeyedJaggedTensor]) -> KeyedJaggedTensor:
                return KeyedJaggedTensor.concat(inputs)

        m = MyModule()

        # input
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        keys = ["index_0", "index_1", "index_2"]
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0, 0, 1, 0])
        kjt_1 = KeyedJaggedTensor.from_lengths_sync(
            values=values[:4],
            keys=keys[:1],
            lengths=lengths[:4],
        )
        kjt_2 = KeyedJaggedTensor.from_lengths_sync(
            values=values[4:],
            keys=keys[1:],
            lengths=lengths[4:],
        )
        inputs = [kjt_1, kjt_2]

        # ensure that symbolic tracing works
        gm = torch.fx.symbolic_trace(m)
        kjt_expected = m(inputs)
        kjt_actual = gm(inputs)

        self.assertTrue(torch.equal(kjt_expected.lengths(), kjt_actual.lengths()))
        self.assertTrue(torch.equal(kjt_expected.offsets(), kjt_actual.offsets()))
        self.assertTrue(torch.equal(kjt_expected.values(), kjt_actual.values()))
        self.assertListEqual(kjt_expected._length_per_key, kjt_actual._length_per_key)

    def test_length_vs_offset(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3])

        j_offset = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )

        j_lens = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
        )

        self.assertTrue(torch.equal(j_offset.lengths(), j_lens.lengths()))
        # TO DO: T88149179
        self.assertTrue(torch.equal(j_offset.offsets(), j_lens.offsets().int()))

    def test_2d(self) -> None:
        values = torch.Tensor([[i * 0.5, i * 1.0, i * 1.5] for i in range(1, 9)])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        j = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            weights=weights,
            keys=keys,
            offsets=offsets,
        )
        j_0 = j["index_0"]

        self.assertTrue(torch.equal(j_0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(
            torch.equal(
                j_0.values(),
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
        keys = ["f1", "f2"]
        # torch.Tensor([3, 4]) also fails
        #  param but got `Type[float]`.
        lengths = torch.tensor([3, 4], dtype=float)
        #  param but got `Type[float]`.
        offsets = torch.tensor([0, 3, 7], dtype=float)

        with self.assertRaises(AssertionError):
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys, values=values, lengths=lengths
            )
        with self.assertRaises(AssertionError):
            KeyedJaggedTensor.from_offsets_sync(
                keys=keys, values=values, offsets=offsets
            )

    def test_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedJaggedTensor) -> torch.Tensor:
                values = input["any"].values()
                return values

        m = MyModule()
        torch.jit.script(m)

    def test_to(self) -> None:
        j = KeyedJaggedTensor.from_offsets_sync(
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
            values=torch.arange(8),
            weights=torch.arange(8 * 10),
            keys=["index_0", "index_1"],
        )
        j2 = j.to(device=torch.device("cpu"))
        self.assertTrue(torch.equal(j.offsets(), j2.offsets()))
        self.assertTrue(torch.equal(j.lengths(), j2.lengths()))
        self.assertTrue(torch.equal(j.values(), j2.values()))
        self.assertTrue(torch.equal(j.weights(), j2.weights()))

    def test_string_none(self) -> None:
        jag_tensor = KeyedJaggedTensor(
            [],
            torch.Tensor(),
        )

        self.assertEqual(
            str(jag_tensor),
            """KeyedJaggedTensor()\n""",
        )

    def test_string_basic(self) -> None:
        values = torch.Tensor([1.0])
        keys = ["key"]
        offsets = torch.IntTensor([0, 1])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """KeyedJaggedTensor({\n    "key": [[1.0]]\n})\n""",
        )

    def test_string_values(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            'KeyedJaggedTensor({\n    "index_0": [[1.0, 2.0], [], [3.0]],\n'
            '    "index_1": [[4.0], [5.0], [6.0, 7.0, 8.0]]\n})\n',
        )

    def test_string_weights(self) -> None:
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

        self.assertEqual(
            str(jag_tensor),
            'KeyedJaggedTensor({\n    "index_0": {\n'
            '        "values": [[1.0, 2.0], [], [3.0]],\n'
            '        "weights": [[1.0, 0.5], [], [1.5]]\n'
            '    },\n    "index_1": {\n'
            '        "values": [[4.0], [5.0], [6.0, 7.0, 8.0]],\n'
            '        "weights": [[1.0], [0.5], [1.0, 1.0, 1.5]]\n    }\n})\n',
        )

    def test_string_vb(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        stride_per_key_per_rank = [[1, 1], [1, 3]]

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        self.assertEqual(
            str(jag_tensor),
            'KeyedJaggedTensor({\n    "index_0": {\n        '
            '"values": [[1.0, 2.0], []],\n        '
            '"weights": [[1.0, 0.5], []]\n    },\n    '
            '"index_1": {\n        '
            '"values": [[3.0], [4.0], [5.0], [6.0, 7.0, 8.0]],\n        '
            '"weights": [[1.5], [1.0], [0.5], [1.0, 1.0, 1.5]]\n    }\n})\n',
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_record_stream(self) -> None:
        j = KeyedJaggedTensor.from_offsets_sync(
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
            values=torch.arange(8),
            weights=torch.arange(8 * 10),
            keys=["index_0", "index_1"],
        ).to(torch.device("cuda"))
        j.record_stream(torch.cuda.current_stream())

    def test_equality(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3])
        """
        KJT looks like, represented from the inputs above
        #              0         1        2         3    <-- dim_1
        # "index_0"   None  [1.0, 2.0]  None     [3.0]
        # "index_1"   [4.0]    [5.0]    None [1.0, 1.0, 1.5]
        #   ^
        #  dim_0
        """
        kt = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        kt_2 = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        kt_3 = KeyedJaggedTensor(
            values=values,
            keys=["index_1", "index_0"],
            offsets=offsets,
        )

        kt_4 = KeyedJaggedTensor(
            values=torch.Tensor([10.0, 4.0, 2.0, 5.0, 2.0, 6.0, 9.0, 8.0]),
            keys=keys,
            lengths=lengths,
        )

        kt_5 = KeyedJaggedTensor(
            values=values,
            keys=["index_0"],
            offsets=offsets,
        )

        weighted_kt = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )

        self.assertTrue(kjt_is_equal(kt, kt_2))  # base check
        self.assertFalse(kjt_is_equal(kt, kt_3))  # different order of keys
        self.assertFalse(kjt_is_equal(kt, kt_4))  # different values
        self.assertFalse(kjt_is_equal(kt, kt_5))  # different keys
        self.assertFalse(kjt_is_equal(kt, weighted_kt))  # different weights

        # Different lengths
        lengths = torch.IntTensor([1, 2, 3, 4, 5, 6, 7, 8])
        lengths_2 = torch.IntTensor([8, 7, 6, 5, 4, 3, 2, 1])
        kt_length_1 = KeyedJaggedTensor.from_lengths_sync(
            values=values, keys=keys, lengths=lengths
        )
        kt_length_2 = KeyedJaggedTensor.from_lengths_sync(
            values=values, keys=keys, lengths=lengths_2
        )
        self.assertFalse(kjt_is_equal(kt_length_1, kt_length_2))

        # Different offsets
        offsets_2 = torch.IntTensor([8, 4, 1, 5, 0, 1, 2, 1, 2])
        kt_offset_1 = KeyedJaggedTensor.from_offsets_sync(
            values=values, keys=keys, offsets=offsets
        )
        kt_offset_2 = KeyedJaggedTensor.from_offsets_sync(
            values=values, keys=keys, offsets=offsets_2
        )
        self.assertFalse(kjt_is_equal(kt_offset_1, kt_offset_2))

        # Different length_per_key and offset_per_key
        length_per_key_1 = [4, 4]
        length_per_key_2 = [3, 5]
        offset_per_key_1 = [0, 4]
        offset_per_key_2 = [0, 3]
        kt_lpk_opk_1 = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            length_per_key=length_per_key_1,
            offset_per_key=offset_per_key_1,
        )
        kt_lpk_opk_2 = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            length_per_key=length_per_key_2,
            offset_per_key=offset_per_key_2,
        )
        self.assertFalse(kjt_is_equal(kt_lpk_opk_1, kt_lpk_opk_2))

        # None values in optional fields
        kt_none_fields = KeyedJaggedTensor(values=values, keys=keys, offsets=offsets)
        kt_some_fields = KeyedJaggedTensor(
            values=values, keys=keys, offsets=offsets, lengths=lengths, weights=weights
        )
        self.assertFalse(kjt_is_equal(kt_none_fields, kt_some_fields))

        # Empty KeyedJaggedTensor
        kt_empty = KeyedJaggedTensor(
            values=torch.Tensor([]), keys=[], offsets=torch.IntTensor([])
        )
        self.assertTrue(kjt_is_equal(kt_empty, kt_empty))
        self.assertFalse(kjt_is_equal(kt, kt_empty))

        # Non-KeyedJaggedTensor input
        non_kjt_input = "not a KeyedJaggedTensor instance"
        self.assertFalse(kjt_is_equal(kt, non_kjt_input))

    def test_meta_device_compatibility(self) -> None:
        keys = ["index_0", "index_1", "index_2", "index_3"]
        lengths = torch.tensor(
            [2, 0, 1, 1, 1, 3, 0, 2],
            device=torch.device("meta"),
        )
        offsets = torch.tensor(
            [0, 2, 2, 3, 4, 5, 8, 8, 10],
            device=torch.device("meta"),
        )
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            device=torch.device("meta"),
        )
        weights = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            device=torch.device("meta"),
        )
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
        )

        kjt.sync()
        kjt.unsync()

        jt_dict = kjt.to_dict()
        kjt = KeyedJaggedTensor.from_jt_dict(jt_dict)

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=keys, values=values, weights=weights, lengths=lengths
        )

        kjt = KeyedJaggedTensor.from_offsets_sync(
            keys=keys, values=values, weights=weights, offsets=offsets
        )

        # test empty keys case
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[],
            values=torch.tensor([], device=torch.device("meta")),
            lengths=torch.tensor([], device=torch.device("meta")),
        )

    def test_vbe_kjt_stride(self) -> None:
        inverse_indices = torch.tensor([[0, 1, 0], [0, 0, 0]])
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([5, 6, 7, 1, 2, 3, 0, 1]),
            lengths=torch.tensor([3, 3, 2]),
            stride_per_key_per_rank=[[2], [1]],
            inverse_indices=(["f1", "f2"], inverse_indices),
        )

        self.assertEqual(kjt.stride(), inverse_indices.shape[-1])

    def test_empty_like_basic(self) -> None:
        # Setup: Create a weighted KJT with explicit stride
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.tensor([0, 2, 2, 3, 4, 5, 8])

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )

        # Execute: Create an empty_like KJT
        empty_kjt = KeyedJaggedTensor.empty_like(kjt)

        # Assert: Verify the empty KJT has the same structure but empty tensors
        self.assertEqual(empty_kjt.keys(), [])
        self.assertEqual(empty_kjt.device(), kjt.device())
        self.assertEqual(empty_kjt.values().dtype, kjt.values().dtype)
        self.assertEqual(empty_kjt.values().numel(), 0)
        self.assertEqual(empty_kjt.lengths().numel(), 0)
        self.assertEqual(empty_kjt.weights().numel(), 0)
        self.assertEqual(empty_kjt.weights().dtype, kjt.weights().dtype)
        # Assert: Verify stride is preserved
        self.assertEqual(empty_kjt.stride(), kjt.stride())
        # Assert: Verify stride_per_key_per_rank is None (since original KJT doesn't have it)
        self.assertIsNone(empty_kjt._stride_per_key_per_rank)

    def test_empty_like_without_weights(self) -> None:
        # Setup: Create a non-weighted KJT with explicit stride
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        keys = ["index_0"]
        lengths = torch.tensor([2, 2])

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        # Execute: Create an empty_like KJT
        empty_kjt = KeyedJaggedTensor.empty_like(kjt)

        # Assert: Verify the empty KJT has no weights
        self.assertEqual(empty_kjt.keys(), [])
        self.assertEqual(empty_kjt.values().numel(), 0)
        self.assertEqual(empty_kjt.lengths().numel(), 0)
        self.assertIsNone(empty_kjt.weights_or_none())
        # Assert: Verify stride is preserved
        self.assertEqual(empty_kjt.stride(), kjt.stride())
        # Assert: Verify stride_per_key_per_rank is None (since original KJT doesn't have it)
        self.assertIsNone(empty_kjt._stride_per_key_per_rank)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_empty_like_with_device(self) -> None:
        # Setup: Create a KJT on CPU
        values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.1, 0.2, 0.3])
        keys = ["index_0"]
        lengths = torch.tensor([3])

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
        )

        # Execute: Create an empty_like KJT on CUDA device (cross-device)
        empty_kjt = KeyedJaggedTensor.empty_like(kjt, device=torch.device("cuda"))

        # Assert: Verify the empty KJT preserves structure and is on the correct device
        self.assertEqual(empty_kjt.keys(), kjt.keys())
        self.assertEqual(empty_kjt.device().type, "cuda")
        self.assertEqual(empty_kjt.values().dtype, kjt.values().dtype)
        self.assertEqual(empty_kjt.weights().dtype, kjt.weights().dtype)
        # Assert: Verify tensors are allocated on CUDA
        self.assertTrue(empty_kjt.values().is_cuda)
        self.assertTrue(empty_kjt.lengths().is_cuda)
        self.assertTrue(empty_kjt.weights().is_cuda)
        # Assert: Verify tensor sizes match the original
        self.assertEqual(empty_kjt.values().size(), kjt.values().size())
        self.assertEqual(empty_kjt.weights().size(), kjt.weights().size())
        self.assertEqual(empty_kjt.lengths().size(), kjt.lengths().size())

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_empty_like_with_device_and_inverse_indices(self) -> None:
        # Setup: Create a KJT with inverse_indices on CPU
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        keys = ["index_0", "index_1"]
        lengths = torch.tensor([3, 3])
        stride_per_key_per_rank = [[1], [1]]
        inverse_indices = torch.tensor([[0, 1, 0], [0, 0, 0]])

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
            stride_per_key_per_rank=stride_per_key_per_rank,
            inverse_indices=(keys, inverse_indices),
        )

        # Execute: Create an empty_like KJT on CUDA device (cross-device)
        empty_kjt = KeyedJaggedTensor.empty_like(kjt, device=torch.device("cuda"))

        # Assert: Verify the empty KJT preserves structure and is on the correct device
        self.assertEqual(empty_kjt.keys(), kjt.keys())
        self.assertEqual(empty_kjt.device().type, "cuda")
        self.assertEqual(empty_kjt.values().dtype, kjt.values().dtype)
        self.assertEqual(empty_kjt.weights().dtype, kjt.weights().dtype)
        # Assert: Verify tensors are allocated on CUDA
        self.assertTrue(empty_kjt.values().is_cuda)
        self.assertTrue(empty_kjt.lengths().is_cuda)
        self.assertTrue(empty_kjt.weights().is_cuda)
        # Assert: Verify tensor sizes match the original
        self.assertEqual(empty_kjt.values().size(), kjt.values().size())
        self.assertEqual(empty_kjt.weights().size(), kjt.weights().size())
        self.assertEqual(empty_kjt.lengths().size(), kjt.lengths().size())
        # Assert: Verify stride_per_key_per_rank is preserved
        self.assertEqual(
            empty_kjt.stride_per_key_per_rank(), kjt.stride_per_key_per_rank()
        )
        # Assert: Verify inverse_indices are preserved and on CUDA
        self.assertIsNotNone(empty_kjt.inverse_indices_or_none())
        empty_inverse_indices = empty_kjt.inverse_indices()
        kjt_inverse_indices = kjt.inverse_indices()
        self.assertEqual(empty_inverse_indices[0], kjt_inverse_indices[0])
        self.assertTrue(empty_inverse_indices[1].is_cuda)
        self.assertEqual(empty_inverse_indices[1].size(), kjt_inverse_indices[1].size())

    def test_empty_like_with_stride_per_key_per_rank(self) -> None:
        # Setup: Create a KJT with variable stride per key
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        keys = ["index_0", "index_1"]
        lengths = torch.tensor([2, 1, 3])
        stride_per_key_per_rank = [[2], [1]]

        kjt = KeyedJaggedTensor(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        # Execute: Create an empty_like KJT
        empty_kjt = KeyedJaggedTensor.empty_like(kjt)

        # Assert: Verify stride per key per rank is preserved
        self.assertEqual(empty_kjt.keys(), [])
        self.assertTrue(empty_kjt.variable_stride_per_key())
        self.assertEqual(empty_kjt.values().numel(), 0)
        # Assert: Verify stride_per_key_per_rank is correctly preserved
        self.assertEqual(
            empty_kjt.stride_per_key_per_rank(), kjt.stride_per_key_per_rank()
        )
        self.assertIsNotNone(empty_kjt._stride_per_key_per_rank)
        self.assertTrue(
            torch.equal(
                empty_kjt._stride_per_key_per_rank,
                kjt._stride_per_key_per_rank,
            )
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_basic(self) -> None:
        # Setup: Create source KJT on CPU and destination KJT on CUDA
        source_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        source_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        source_lengths = torch.tensor([2, 2])
        keys = ["index_0"]

        source_kjt = KeyedJaggedTensor(
            values=source_values,
            keys=keys,
            lengths=source_lengths,
            weights=source_weights,
        )

        dest_values = torch.zeros(4, device=torch.device("cuda"))
        dest_weights = torch.zeros(4, device=torch.device("cuda"))
        dest_lengths = torch.zeros(2, dtype=torch.int32, device=torch.device("cuda"))

        dest_kjt = KeyedJaggedTensor(
            values=dest_values,
            keys=keys,
            lengths=dest_lengths,
            weights=dest_weights,
        )

        # Execute: Copy source KJT (CPU) to destination KJT (CUDA)
        result_kjt = dest_kjt.copy_(source_kjt)

        # Assert: Verify the destination KJT has the source values
        self.assertTrue(torch.equal(result_kjt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_kjt.weights().cpu(), source_weights))
        self.assertTrue(torch.equal(result_kjt.lengths().cpu(), source_lengths))
        # Verify it returns the same object (in-place operation)
        self.assertIs(result_kjt, dest_kjt)
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_kjt.values().is_cuda)
        self.assertTrue(result_kjt.weights().is_cuda)
        self.assertTrue(result_kjt.lengths().is_cuda)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_without_weights(self) -> None:
        # Setup: Create source KJT on CPU and destination KJT on CUDA without weights
        source_values = torch.tensor([5.0, 6.0, 7.0])
        source_lengths = torch.tensor([1, 2])
        keys = ["index_0"]

        source_kjt = KeyedJaggedTensor(
            values=source_values,
            keys=keys,
            lengths=source_lengths,
        )

        dest_values = torch.zeros(3, device=torch.device("cuda"))
        dest_lengths = torch.zeros(2, dtype=torch.int32, device=torch.device("cuda"))

        dest_kjt = KeyedJaggedTensor(
            values=dest_values,
            keys=keys,
            lengths=dest_lengths,
        )

        # Execute: Copy source KJT (CPU) to destination KJT (CUDA)
        result_kjt = dest_kjt.copy_(source_kjt)

        # Assert: Verify the destination KJT has the source values
        self.assertTrue(torch.equal(result_kjt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_kjt.lengths().cpu(), source_lengths))
        self.assertIsNone(result_kjt.weights_or_none())
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_kjt.values().is_cuda)
        self.assertTrue(result_kjt.lengths().is_cuda)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_with_offsets(self) -> None:
        # Setup: Create source KJT on CPU and destination KJT on CUDA with offsets
        source_values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        source_offsets = torch.tensor([0, 2, 5])
        keys = ["index_0", "index_1"]

        source_kjt = KeyedJaggedTensor(
            values=source_values,
            keys=keys,
            offsets=source_offsets,
        )

        dest_values = torch.zeros(5, device=torch.device("cuda"))
        dest_offsets = torch.zeros(3, dtype=torch.int32, device=torch.device("cuda"))

        dest_kjt = KeyedJaggedTensor(
            values=dest_values,
            keys=keys,
            offsets=dest_offsets,
        )

        # Execute: Copy source KJT (CPU) to destination KJT (CUDA)
        result_kjt = dest_kjt.copy_(source_kjt)

        # Assert: Verify the destination KJT has the source values and offsets
        self.assertTrue(torch.equal(result_kjt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_kjt.offsets().cpu(), source_offsets))
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_kjt.values().is_cuda)
        self.assertTrue(result_kjt.offsets().is_cuda)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_copy_non_blocking(self) -> None:
        # Setup: Create source KJT on CPU and destination KJT on CUDA
        source_values = torch.tensor([1.0, 2.0, 3.0])
        source_lengths = torch.tensor([3])
        keys = ["index_0"]

        source_kjt = KeyedJaggedTensor(
            values=source_values,
            keys=keys,
            lengths=source_lengths,
        )

        dest_values = torch.zeros(3, device=torch.device("cuda"))
        dest_lengths = torch.zeros(1, dtype=torch.int32, device=torch.device("cuda"))

        dest_kjt = KeyedJaggedTensor(
            values=dest_values,
            keys=keys,
            lengths=dest_lengths,
        )

        # Execute: Copy source KJT (CPU) to destination KJT (CUDA) with non_blocking=True
        result_kjt = dest_kjt.copy_(source_kjt, non_blocking=True)

        # Assert: Verify the copy succeeded
        self.assertTrue(torch.equal(result_kjt.values().cpu(), source_values))
        self.assertTrue(torch.equal(result_kjt.lengths().cpu(), source_lengths))
        # Assert: Verify tensors are on CUDA
        self.assertTrue(result_kjt.values().is_cuda)
        self.assertTrue(result_kjt.lengths().is_cuda)


class TestKeyedJaggedTensorScripting(unittest.TestCase):
    def test_scriptable_forward(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedJaggedTensor) -> KeyedJaggedTensor:
                input["any"].values()
                input.dist_labels()
                input.dist_splits([1, 2])
                return KeyedJaggedTensor.dist_init(
                    keys=input.keys(),
                    tensors=input.dist_tensors(),
                    variable_stride_per_key=False,
                    num_workers=2,
                    recat=torch.tensor([]),
                    stride_per_rank=[2, 3],
                )

        m = MyModule()
        torch.jit.script(m)

    def test_scriptable_split(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedJaggedTensor) -> List[KeyedJaggedTensor]:
                return input.split([1, 0, 1])

        m = MyModule()
        torch.jit.script(m)

    def test_scriptable_init(self) -> None:
        def create_kjt() -> KeyedJaggedTensor:
            return KeyedJaggedTensor.from_offsets_sync(
                values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                weights=torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                keys=["index_0", "index_1"],
                offsets=torch.tensor([0, 0, 2, 2, 3, 4, 5, 5, 8], dtype=torch.int32),
            )

        def create_vb_kjt() -> KeyedJaggedTensor:
            return KeyedJaggedTensor.from_offsets_sync(
                values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                weights=torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                keys=["index_0", "index_1"],
                offsets=torch.tensor([0, 0, 2, 2, 3, 4, 5, 5, 8], dtype=torch.int32),
                stride_per_key_per_rank=[[2], [4]],
            )

        # assert that we can script KJT creation
        torch.jit.script(create_kjt)
        torch.jit.script(create_vb_kjt)

    def test_scriptable_empty(self) -> None:
        def create_empty() -> KeyedJaggedTensor:
            return KeyedJaggedTensor.empty()

        def create_empty_weighted() -> KeyedJaggedTensor:
            return KeyedJaggedTensor.empty(is_weighted=True)

        # assert that we can script KJT creation
        torch.jit.script(create_empty)
        torch.jit.script(create_empty_weighted)


class TestKeyedJaggedTensorTracingScripting(unittest.TestCase):
    def test_jit_tracable(self) -> None:
        # This module will simply go through the constructor of the
        # KeyedJaggedTensor to construct it with multiple different batch sizes
        class MyModule(torch.nn.Module):
            def forward(
                self, offsets: torch.Tensor, values: torch.Tensor, weights: torch.Tensor
            ) -> torch.Tensor:
                j = KeyedJaggedTensor.from_offsets_sync(
                    offsets=offsets,
                    values=values,
                    weights=weights,
                    keys=["index_0", "index_1"],
                )
                return j["index_0"].offsets()

        sample_2 = (
            torch.tensor([0, 2, 2]),
            torch.arange(2),
            torch.arange(2 * 10),
        )
        sample_6 = (
            torch.tensor([0, 2, 2, 3, 4, 6, 8]),
            torch.arange(8),
            torch.arange(8 * 10),
        )
        m = MyModule()
        model_eager_traced: torch.jit.ScriptModule = torch.jit.trace(
            m, sample_2, strict=False
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_2), torch.tensor([0, 2]))
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_6), torch.tensor([0, 2, 2, 3]))
        )

    def test_create_and_access_keyed_jagged_tensor(self) -> None:
        class ModuleCreateAndAccessKeyedJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: int) -> int:
                features = KeyedJaggedTensor.from_offsets_sync(
                    values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                    weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                    keys=["index_0", "index_1"],
                    offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
                )
                return (
                    len(features.keys())
                    + features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        # Case 4: KeyedJaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessKeyedJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 35").check_not("KeyedJaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)
        torch.jit.script(gm)

    def test_create_and_access_empty_keyed_jagged_tensor(self) -> None:
        class ModuleCreateAndAccessEmptyKeyedJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: int) -> int:
                features = KeyedJaggedTensor.empty(is_weighted=True)
                return (
                    len(features.keys())
                    + features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        # Case 4: KeyedJaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessEmptyKeyedJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 1").check_not("KeyedJaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)
        torch.jit.script(gm)

    def test_traceable_empty_like(self) -> None:
        class ModuleCreateAndAccessEmptyLikeKeyedJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, kjt: KeyedJaggedTensor) -> int:
                features = KeyedJaggedTensor.empty_like(kjt)
                return (
                    len(features.keys())
                    + features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        # Case 4: KeyedJaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessEmptyLikeKeyedJaggedTensor()
        kjt = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
        )
        gm = symbolic_trace(m)
        ref_out = m(kjt)
        traced_out = gm(kjt)
        self.assertEqual(ref_out, traced_out)
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_input_and_output(self) -> None:
        class ModuleUseKeyedJaggedTensorAsInputAndOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, input: KeyedJaggedTensor
            ) -> Tuple[KeyedJaggedTensor, int]:
                output = KeyedJaggedTensor(
                    input.keys(),
                    input.values(),
                    input.weights(),
                    lengths=input.lengths(),
                    offsets=input.offsets(),
                )
                return output, output.stride()

        # Case 3: KeyedJaggedTensor is used as both an input and an output of the root module.
        m = ModuleUseKeyedJaggedTensorAsInputAndOutput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check("keys()").check("values()").check(
            "stride"
        ).run(gm.code)
        input = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out[1], traced_out[1])
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_input(self) -> None:
        class ModuleUseKeyedJaggedTensorAsInput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: KeyedJaggedTensor) -> int:
                return (
                    len(input.keys())
                    + input.values().numel()
                    + input.weights().numel()
                    + input.lengths().numel()
                    + input.offsets().numel()
                )

        # Case 2: KeyedJaggedTensor is only used as an input of the root module.
        m = ModuleUseKeyedJaggedTensorAsInput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check("keys()").check("len").check(
            "values()"
        ).check("numel()").run(gm.code)

        input = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out, traced_out)
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_output(self) -> None:
        class ModuleUseKeyedJaggedTensorAsOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                keys: List[str],
                values: torch.Tensor,
                weights: torch.Tensor,
                lengths: torch.Tensor,
            ) -> Tuple[KeyedJaggedTensor, int]:
                output = KeyedJaggedTensor(keys, values, weights, lengths)
                return output, output.stride()

        # Case 1: KeyedJaggedTensor is only used as an output of the root module.
        m = ModuleUseKeyedJaggedTensorAsOutput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check(
            "return (keyed_jagged_tensor,"
        ).run(gm.code)

        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        lengths = torch.IntTensor([2, 0, 1, 1, 1, 3])

        ref_out = m(keys, values, weights, lengths)
        traced_out = gm(keys, values, weights, lengths)

        self.assertEqual(ref_out[1], traced_out[1])
        self.assertTrue(torch.equal(traced_out[0].offsets(), ref_out[0].offsets()))
        torch.jit.script(gm)


class TestComputeKJTToJTDict(unittest.TestCase):
    def test_key_lookup(self) -> None:
        m = ComputeKJTToJTDict()
        input = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
            stride_per_key_per_rank=[[0, 2], [3, 3]],
        )

        out = m(input)

        i0 = out["index_0"]
        self.assertTrue(torch.equal(i0._values, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(i0._weights, torch.tensor([1.0, 0.5])))
        self.assertTrue(torch.equal(i0._lengths, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(i0._offsets, torch.tensor([0, 0, 2])))

        i1 = out["index_1"]
        self.assertTrue(
            torch.equal(i1._values, torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
        )
        self.assertTrue(
            torch.equal(i1._weights, torch.tensor([1.5, 1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(torch.equal(i1._lengths, torch.tensor([0, 1, 1, 1, 0, 3])))
        self.assertTrue(torch.equal(i1._offsets, torch.tensor([0, 0, 1, 2, 3, 3, 6])))


@skip_if_asan_class
class TestKeyedJaggedTensorGPU(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.cuda.current_device()

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([0, 2, 0, 1, 1, 1, 0, 3, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )
        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(), [1, 1, 1, 0, 2, 0, 0, 3, 0]
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_vb(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([1, 0, 1, 3, 0, 1, 0, 2, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 5, 6, 8],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(), [1, 3, 0, 1, 1, 0, 0, 2, 0]
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_vb_duplicate(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([1, 0, 1, 3, 0, 1, 0, 2, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 1, 0, 0, 2, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_1", "index_0", "index_0", "index_2", "index_2"],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values().cpu(),
                torch.Tensor(
                    [
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        1.0,
                        1.0,
                        7.0,
                        8.0,
                        7.0,
                        8.0,
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths().cpu(),
                torch.IntTensor([1, 3, 0, 1, 1, 3, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0, 2, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_duplicates(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([0, 2, 0, 1, 1, 1, 0, 3, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2, 1, 1]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_0", "index_2", "index_1", "index_1"],
        )
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8, 11, 14],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [
                3.0,
                4.0,
                5.0,
                1.0,
                2.0,
                6.0,
                7.0,
                8.0,
                3.0,
                4.0,
                5.0,
                3.0,
                4.0,
                5.0,
            ],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(),
            [1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)
