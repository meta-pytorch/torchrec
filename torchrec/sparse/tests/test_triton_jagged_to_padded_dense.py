#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest
from typing import TYPE_CHECKING

import torch
from parameterized import parameterized

_TESTS_SUPPORTED = torch.cuda.is_available() and sys.version_info < (3, 15)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    from torchrec.sparse.triton_jagged_to_padded_dense import (
        triton_jagged_to_padded_dense,
    )

if _TESTS_SUPPORTED:
    try:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    except OSError:
        pass


def _make_invalid_inputs(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, int, str]:
    if case == "cpu":
        return torch.empty(2), torch.tensor([0, 2]), 2, "CUDA tensors"
    if case == "different_device":
        return torch.empty(2, device="cuda"), torch.tensor([0, 2]), 2, "same device"
    if case == "non_contiguous":
        return (
            torch.empty(4, 2, device="cuda").t(),
            torch.tensor([0, 2], device="cuda"),
            2,
            "contiguous",
        )
    if case == "values_dtype":
        return (
            torch.empty(2, dtype=torch.int64, device="cuda"),
            torch.tensor([0, 2], device="cuda"),
            2,
            "values must have",
        )
    if case == "offsets_dtype":
        return (
            torch.empty(2, device="cuda"),
            torch.tensor([0.0, 2.0], device="cuda"),
            2,
            "offsets must have",
        )
    if case == "empty_offsets":
        return (
            torch.empty(0, device="cuda"),
            torch.empty(0, dtype=torch.int64, device="cuda"),
            2,
            "at least one element",
        )
    return (
        torch.empty(2, device="cuda"),
        torch.tensor([0, 2], device="cuda"),
        -1,
        "nonnegative",
    )


@unittest.skipUnless(_TESTS_SUPPORTED, "CUDA and Python below 3.15 are required")
class TritonJaggedToPaddedDenseTest(unittest.TestCase):
    @parameterized.expand(
        (
            ("fp32_scalar", torch.float32, 1, 0.0),
            ("fp16_vector", torch.float16, 7, -1.0),
            ("bf16_vector", torch.bfloat16, 32, 2.5),
        )
    )
    def test_matches_fbgemm(
        self,
        _name: str,
        dtype: torch.dtype,
        dim: int,
        padding_value: float,
    ) -> None:
        lengths = torch.tensor([0, 1, 3, 5], device="cuda", dtype=torch.int64)
        offsets = torch.zeros(5, device="cuda", dtype=torch.int64)
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        values = torch.randn(int(offsets[-1]), dim, device="cuda", dtype=dtype)
        if dim == 1:
            values = values.flatten()

        actual = triton_jagged_to_padded_dense(
            values,
            offsets,
            max_length=3,
            padding_value=padding_value,
        )
        expected = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [3],
            padding_value,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_matches_fbgemm(self) -> None:
        lengths = torch.tensor([1, 4, 2], device="cuda", dtype=torch.int32)
        offsets = torch.zeros(4, device="cuda", dtype=torch.int32)
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        actual_values = torch.randn(
            int(offsets[-1]), 5, device="cuda", requires_grad=True
        )
        expected_values = actual_values.detach().clone().requires_grad_(True)
        actual = triton_jagged_to_padded_dense(actual_values, offsets, 3)
        expected = torch.ops.fbgemm.jagged_to_padded_dense(
            expected_values,
            [offsets],
            [3],
            0.0,
        )
        grad_output = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, actual_values, grad_output)[0]
        expected_grad = torch.autograd.grad(expected, expected_values, grad_output)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    @parameterized.expand(
        (
            ("empty_batch", [0], 3),
            ("zero_max_length", [0, 0], 0),
        )
    )
    def test_empty_output(
        self,
        _name: str,
        offset_values: list[int],
        max_length: int,
    ) -> None:
        values = torch.empty(0, 4, device="cuda")
        offsets = torch.tensor(offset_values, device="cuda", dtype=torch.int64)
        actual = triton_jagged_to_padded_dense(values, offsets, max_length)
        expected = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [max_length],
            0.0,
        )
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @parameterized.expand(
        (
            ("cpu",),
            ("different_device",),
            ("non_contiguous",),
            ("values_dtype",),
            ("offsets_dtype",),
            ("empty_offsets",),
            ("negative_max_length",),
        )
    )
    def test_rejects_invalid_input(self, case: str) -> None:
        values, offsets, max_length, message = _make_invalid_inputs(case)
        with self.assertRaisesRegex(ValueError, message):
            triton_jagged_to_padded_dense(values, offsets, max_length)

    def test_compile(self) -> None:
        values = torch.randn(7, 4, device="cuda")
        offsets = torch.tensor([0, 2, 3, 7], device="cuda")
        compiled = torch.compile(
            triton_jagged_to_padded_dense,
            backend="aot_eager",
            fullgraph=True,
        )
        actual = compiled(values, offsets, 3, -1.0)
        expected = torch.ops.fbgemm.jagged_to_padded_dense(
            values,
            [offsets],
            [3],
            -1.0,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
