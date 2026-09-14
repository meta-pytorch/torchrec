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

_TESTS_SUPPORTED = torch.cuda.is_available() and sys.version_info < (3, 15)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    from torchrec.sparse.triton_pack_segments import triton_pack_segments

if _TESTS_SUPPORTED:
    try:
        torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    except OSError:
        pass


@unittest.skipUnless(_TESTS_SUPPORTED, "CUDA and Python below 3.15 are required")
class TritonPackSegmentsTest(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for lengths_dtype in (torch.int32, torch.int64):
                with self.subTest(dtype=dtype, lengths_dtype=lengths_dtype):
                    lengths = torch.tensor(
                        [0, 1, 7, 3, 9], device="cuda", dtype=lengths_dtype
                    )
                    input = torch.randn(
                        20,
                        3,
                        5,
                        device="cuda",
                        dtype=dtype,
                        requires_grad=True,
                    )
                    reference_input = input.detach().clone().requires_grad_()

                    output = triton_pack_segments(input, lengths, 6)
                    expected = torch.ops.fbgemm.pack_segments(
                        reference_input, lengths, 6
                    )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)

                    grad_output = torch.randn_like(output)
                    actual_grad = torch.autograd.grad(output, input, grad_output)[0]
                    expected_grad = torch.autograd.grad(
                        expected, reference_input, grad_output
                    )[0]
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

    def test_empty_output(self) -> None:
        input = torch.empty(0, 8, device="cuda")
        lengths = torch.empty(0, dtype=torch.int64, device="cuda")
        output = triton_pack_segments(input, lengths, 10)
        self.assertEqual(output.shape, (0, 10, 8))

    def test_zero_total_length(self) -> None:
        lengths = torch.zeros(4, dtype=torch.int64, device="cuda")
        input = torch.empty(0, 8, device="cuda", requires_grad=True)
        reference_input = input.detach().clone().requires_grad_()

        output = triton_pack_segments(input, lengths, 10)
        expected = torch.ops.fbgemm.pack_segments(reference_input, lengths, 10)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

        grad_output = torch.randn_like(output)
        actual_grad = torch.autograd.grad(output, input, grad_output)[0]
        expected_grad = torch.autograd.grad(expected, reference_input, grad_output)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_compile_forward_and_backward(self) -> None:
        lengths = torch.tensor([2, 5, 3, 8], device="cuda", dtype=torch.int64)
        input = torch.randn(
            int(lengths.sum().item()),
            16,
            device="cuda",
            requires_grad=True,
        )
        reference_input = input.detach().clone().requires_grad_()
        compiled = torch.compile(
            triton_pack_segments,
            backend="aot_eager",
            fullgraph=True,
        )

        output = compiled(input, lengths, 6)
        expected = torch.ops.fbgemm.pack_segments(reference_input, lengths, 6)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        grad_output = torch.randn_like(output)
        actual_grad = torch.autograd.grad(output, input, grad_output)[0]
        expected_grad = torch.autograd.grad(expected, reference_input, grad_output)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
