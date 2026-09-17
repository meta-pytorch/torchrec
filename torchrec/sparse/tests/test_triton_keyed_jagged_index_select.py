#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest
from typing import Optional, TYPE_CHECKING

import torch
from parameterized import parameterized

_TESTS_SUPPORTED = torch.cuda.is_available() and sys.version_info < (3, 15)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    from torchrec.sparse.triton_keyed_jagged_index_select import (
        triton_keyed_jagged_index_select_dim1,
    )

if _TESTS_SUPPORTED:
    try:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    except OSError:
        pass


def _inputs(
    dtype: torch.dtype,
    has_weights: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    lengths = torch.tensor(
        [0, 2, 1, 3, 4, 1, 2, 0],
        dtype=torch.int64,
        device="cuda",
    )
    offsets = torch.zeros(9, dtype=torch.int64, device="cuda")
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    if dtype.is_floating_point:
        values = torch.randn(int(offsets[-1]), dtype=dtype, device="cuda")
    else:
        values = torch.randint(1000, (int(offsets[-1]),), dtype=dtype, device="cuda")
    indices = torch.tensor([3, 1, 3, 0, 2, 1], dtype=torch.int64, device="cuda")
    weights = torch.randn_like(values, dtype=torch.float32) if has_weights else None
    return values, lengths, offsets, indices, weights


@unittest.skipUnless(_TESTS_SUPPORTED, "CUDA and Python below 3.15 are required")
class TritonKeyedJaggedIndexSelectTest(unittest.TestCase):
    @parameterized.expand(
        (
            ("int32", torch.int32, False, False),
            ("int64_with_sum", torch.int64, False, True),
            ("float32_weights", torch.float32, True, False),
            ("float16_weights_with_sum", torch.float16, True, True),
        )
    )
    def test_matches_fbgemm(
        self,
        _name: str,
        dtype: torch.dtype,
        has_weights: bool,
        provide_sum: bool,
    ) -> None:
        values, lengths, offsets, indices, weights = _inputs(dtype, has_weights)
        selected_lengths = lengths.view(2, 4)[:, indices].flatten()
        selected_lengths_sum = (
            int(selected_lengths.sum().item()) if provide_sum else None
        )
        actual = triton_keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            4,
            weights,
            selected_lengths_sum,
        )
        expected = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            4,
            weights,
            selected_lengths_sum,
        )
        self.assertEqual(len(actual), 3)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        if has_weights:
            assert actual[2] is not None and len(expected) == 3
            torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        else:
            self.assertIsNone(actual[2])

    def test_backward_matches_fbgemm(self) -> None:
        values, lengths, offsets, indices, _weights = _inputs(torch.float32, False)
        actual_values = values.detach().clone().requires_grad_(True)
        expected_values = values.detach().clone().requires_grad_(True)
        selected_lengths_sum = int(lengths.view(2, 4)[:, indices].sum().item())

        actual = triton_keyed_jagged_index_select_dim1(
            actual_values,
            lengths,
            offsets,
            indices,
            4,
            None,
            selected_lengths_sum,
        )
        expected = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            expected_values,
            lengths,
            offsets,
            indices,
            4,
            None,
            selected_lengths_sum,
        )
        grad_values = torch.randn_like(actual[0])
        actual_grad = torch.autograd.grad(actual[0], actual_values, grad_values)[0]
        expected_grad = torch.autograd.grad(expected[0], expected_values, grad_values)[
            0
        ]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-6)

    def test_compile(self) -> None:
        values, lengths, offsets, indices, _weights = _inputs(torch.float32, False)
        selected_lengths_sum = int(lengths.view(2, 4)[:, indices].sum().item())
        compiled = torch.compile(
            triton_keyed_jagged_index_select_dim1,
            backend="aot_eager",
            fullgraph=True,
        )
        actual = compiled(
            values,
            lengths,
            offsets,
            indices,
            4,
            None,
            selected_lengths_sum,
        )
        expected = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            4,
            None,
            selected_lengths_sum,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_empty_values(self) -> None:
        values = torch.empty(0, device="cuda")
        lengths = torch.zeros(2, dtype=torch.int64, device="cuda")
        offsets = torch.zeros(3, dtype=torch.int64, device="cuda")
        indices = torch.tensor([1, 0, 1], device="cuda")
        actual = triton_keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            2,
        )
        self.assertEqual(actual[0].numel(), 0)
        torch.testing.assert_close(
            actual[1],
            torch.zeros(3, dtype=torch.int64, device="cuda"),
        )

    def test_empty_indices(self) -> None:
        values = torch.arange(3, device="cuda")
        lengths = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        offsets = torch.tensor([0, 1, 3], dtype=torch.int64, device="cuda")
        indices = torch.empty(0, dtype=torch.int64, device="cuda")
        actual = triton_keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            2,
        )
        self.assertEqual(actual[0].numel(), 0)
        self.assertEqual(actual[1].numel(), 0)

    def test_large_segment_metadata_fallback(self) -> None:
        values = torch.tensor([7], device="cuda")
        lengths = torch.ones(1, dtype=torch.int64, device="cuda")
        offsets = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        indices = torch.zeros(4097, dtype=torch.int64, device="cuda")
        actual = triton_keyed_jagged_index_select_dim1(
            values,
            lengths,
            offsets,
            indices,
            1,
            selected_lengths_sum=4097,
        )
        self.assertEqual(actual[0].tolist(), [7] * 4097)
        self.assertEqual(actual[1].tolist(), [1] * 4097)

    def test_rejects_cpu_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            triton_keyed_jagged_index_select_dim1(
                torch.empty(0),
                torch.zeros(2, dtype=torch.int64),
                torch.zeros(3, dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                2,
            )

    def test_rejects_invalid_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide lengths"):
            triton_keyed_jagged_index_select_dim1(
                torch.empty(0, device="cuda"),
                torch.zeros(3, dtype=torch.int64, device="cuda"),
                torch.zeros(4, dtype=torch.int64, device="cuda"),
                torch.tensor([0], dtype=torch.int64, device="cuda"),
                2,
            )

    def test_rejects_invalid_tensor_metadata(self) -> None:
        values, lengths, offsets, indices, _weights = _inputs(torch.float32, False)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths,
                offsets,
                indices.repeat_interleave(2)[::2],
                4,
            )
        with self.assertRaisesRegex(ValueError, "lengths must"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths.to(torch.float32),
                offsets,
                indices,
                4,
            )
        with self.assertRaisesRegex(ValueError, "offsets must"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths,
                offsets.to(torch.float32),
                indices,
                4,
            )
        with self.assertRaisesRegex(ValueError, "indices must"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths,
                offsets,
                indices.to(torch.float32),
                4,
            )
        with self.assertRaisesRegex(ValueError, "one more"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths,
                offsets[:-1],
                indices,
                4,
            )
        with self.assertRaisesRegex(ValueError, "match values"):
            triton_keyed_jagged_index_select_dim1(
                values,
                lengths,
                offsets,
                indices,
                4,
                torch.empty(1, device="cuda"),
            )
