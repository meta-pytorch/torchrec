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
    from torchrec.sparse.triton_quantized_comm import (
        triton_float_to_fused8bitrowwise_quantized,
        triton_fused8bitrowwise_quantized_to_float,
    )

if _TESTS_SUPPORTED:
    try:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    except OSError:
        pass


def _make_invalid_quantize_input(case: str) -> tuple[torch.Tensor, str]:
    if case == "cpu":
        return torch.empty(2, 4), "CUDA tensor"
    if case == "non_contiguous":
        return torch.empty(4, 2, device="cuda").t(), "contiguous"
    if case == "wrong_dtype":
        return torch.empty(2, 4, dtype=torch.float16, device="cuda"), "torch.float32"
    if case == "scalar":
        return torch.empty((), device="cuda"), "at least one dimension"
    return torch.empty(1, 4097, device="cuda"), "cannot exceed 4096"


def _make_invalid_dequantize_input(case: str) -> tuple[torch.Tensor, str]:
    if case == "cpu":
        return torch.empty(2, 12, dtype=torch.uint8), "CUDA tensor"
    if case == "non_contiguous":
        return torch.empty(12, 2, dtype=torch.uint8, device="cuda").t(), "contiguous"
    if case == "wrong_dtype":
        return torch.empty(2, 12, device="cuda"), "torch.uint8"
    if case == "scalar":
        return (
            torch.empty((), dtype=torch.uint8, device="cuda"),
            "at least one dimension",
        )
    columns = 4 if case == "too_narrow" else 10
    return (
        torch.empty(2, columns, dtype=torch.uint8, device="cuda"),
        "multiple of four and at least eight",
    )


@unittest.skipUnless(_TESTS_SUPPORTED, "CUDA and Python below 3.15 are required")
class TritonQuantizedCommTest(unittest.TestCase):
    @parameterized.expand(
        (
            ("single_row", 1, 32),
            ("narrow_unaligned", 20, 31),
            ("wide_unaligned", 21, 33),
            ("many_rows", 4096, 32),
            ("medium_width", 257, 128),
            ("wide", 257, 1024),
            ("one_column", 20, 1),
            ("two_columns", 20, 2),
        )
    )
    def test_matches_fbgemm(self, _name: str, num_rows: int, num_columns: int) -> None:
        input = torch.randn(
            num_rows,
            num_columns,
            dtype=torch.float32,
            device="cuda",
        )
        actual_quantized = triton_float_to_fused8bitrowwise_quantized(input)
        expected_quantized = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input)
        payload_columns = (num_columns + 3) // 4 * 4
        torch.testing.assert_close(
            actual_quantized[:, :num_columns],
            expected_quantized[:, :num_columns],
            rtol=0,
            atol=1,
        )
        torch.testing.assert_close(
            actual_quantized[:, payload_columns:].view(torch.float32),
            expected_quantized[:, payload_columns:].view(torch.float32),
            rtol=1e-6,
            atol=1e-8,
        )

        actual = triton_fused8bitrowwise_quantized_to_float(expected_quantized)
        expected = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(expected_quantized)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_constant_rows(self) -> None:
        input = torch.full((37, 32), 2.5, device="cuda")
        quantized = triton_float_to_fused8bitrowwise_quantized(input)
        output = triton_fused8bitrowwise_quantized_to_float(quantized)
        torch.testing.assert_close(output, input, rtol=0, atol=0)

    def test_empty_rows(self) -> None:
        input = torch.empty(0, 32, device="cuda")
        quantized = triton_float_to_fused8bitrowwise_quantized(input)
        self.assertEqual(quantized.shape, (0, 40))
        self.assertEqual(
            triton_fused8bitrowwise_quantized_to_float(quantized).shape,
            input.shape,
        )

    def test_empty_columns(self) -> None:
        input = torch.empty(3, 0, device="cuda")
        quantized = triton_float_to_fused8bitrowwise_quantized(input)
        self.assertEqual(quantized.shape, (3, 8))
        torch.testing.assert_close(quantized, torch.zeros_like(quantized))

    @parameterized.expand(
        (("cpu",), ("non_contiguous",), ("wrong_dtype",), ("scalar",), ("too_wide",))
    )
    def test_quantize_rejects_invalid_input(self, case: str) -> None:
        input, message = _make_invalid_quantize_input(case)
        with self.assertRaisesRegex(ValueError, message):
            triton_float_to_fused8bitrowwise_quantized(input)

    @parameterized.expand(
        (
            ("cpu",),
            ("non_contiguous",),
            ("wrong_dtype",),
            ("scalar",),
            ("too_narrow",),
            ("unaligned",),
        )
    )
    def test_dequantize_rejects_invalid_input(self, case: str) -> None:
        input, message = _make_invalid_dequantize_input(case)
        with self.assertRaisesRegex(ValueError, message):
            triton_fused8bitrowwise_quantized_to_float(input)

    def test_compile(self) -> None:
        input = torch.randn(64, 32, device="cuda")
        quantize = torch.compile(
            triton_float_to_fused8bitrowwise_quantized,
            backend="aot_eager",
            fullgraph=True,
        )
        dequantize = torch.compile(
            triton_fused8bitrowwise_quantized_to_float,
            backend="aot_eager",
            fullgraph=True,
        )
        quantized = quantize(input)
        actual = dequantize(quantized)
        expected = torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(input)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
