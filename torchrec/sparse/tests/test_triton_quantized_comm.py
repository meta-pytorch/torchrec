#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys
import unittest

import torch
from fbgemm_gpu.quantize_utils import (
    bf16_to_fp32,
    fp32_to_bf16_with_clamp,
    fp32_to_mx4,
    mx4_to_float,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from parameterized import parameterized
from torchrec.sparse.triton_quantized_comm import (
    triton_bfloat16_quantized_to_float,
    triton_float_to_bfloat16_quantized,
    triton_float_to_fused8bitrowwise_quantized,
    triton_float_to_mx4_quantized,
    triton_fused8bitrowwise_quantized_to_float,
    triton_mx4_quantized_to_float,
)

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


@unittest.skipIf(
    not torch.cuda.is_available() or sys.version_info >= (3, 15),
    "CUDA and Python below 3.15 are required",
)
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

    def test_bfloat16_matches_qcomm(self) -> None:
        input = torch.tensor(
            [
                -float("inf"),
                -1.0,
                0.0,
                1.0,
                float("inf"),
                float("nan"),
            ],
            device="cuda",
        )
        actual_quantized = triton_float_to_bfloat16_quantized(input)
        expected_quantized = fp32_to_bf16_with_clamp(input)
        torch.testing.assert_close(
            actual_quantized,
            expected_quantized,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            triton_bfloat16_quantized_to_float(actual_quantized),
            bf16_to_fp32(expected_quantized),
            rtol=0,
            atol=0,
            equal_nan=True,
        )

    def test_mx4_matches_qcomm(self) -> None:
        input = torch.randn(257, 127, device="cuda")
        actual_quantized = triton_float_to_mx4_quantized(input)
        expected_quantized = fp32_to_mx4(input)
        torch.testing.assert_close(actual_quantized, expected_quantized, rtol=0, atol=0)
        for output_dtype, sparse_type in (
            (torch.float32, SparseType.FP32),
            (torch.bfloat16, SparseType.BF16),
        ):
            with self.subTest(output_dtype=output_dtype):
                torch.testing.assert_close(
                    triton_mx4_quantized_to_float(
                        actual_quantized,
                        output_dtype=output_dtype,
                    ),
                    mx4_to_float(expected_quantized, output_dtype=sparse_type),
                    rtol=0,
                    atol=0,
                )

    def test_other_formats_compile(self) -> None:
        input = torch.randn(64, 128, device="cuda")
        bf16_roundtrip = torch.compile(
            lambda value: triton_bfloat16_quantized_to_float(
                triton_float_to_bfloat16_quantized(value)
            ),
            backend="aot_eager",
            fullgraph=True,
        )
        mx4_roundtrip = torch.compile(
            lambda value: triton_mx4_quantized_to_float(
                triton_float_to_mx4_quantized(value)
            ),
            backend="aot_eager",
            fullgraph=True,
        )
        self.assertEqual(bf16_roundtrip(input).shape, input.shape)
        self.assertEqual(mx4_roundtrip(input).shape, input.shape)

    def test_other_formats_empty(self) -> None:
        input = torch.empty(0, 128, device="cuda")
        bfloat16 = triton_float_to_bfloat16_quantized(input)
        self.assertEqual(bfloat16.shape, input.shape)
        self.assertEqual(
            triton_bfloat16_quantized_to_float(bfloat16).shape, input.shape
        )
        mx4 = triton_float_to_mx4_quantized(input)
        self.assertEqual(mx4.shape, (0, 68))
        self.assertEqual(triton_mx4_quantized_to_float(mx4).shape, input.shape)
