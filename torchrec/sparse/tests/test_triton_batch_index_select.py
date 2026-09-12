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
from unittest import mock

import torch

_TESTS_SUPPORTED = torch.cuda.is_available() and sys.version_info < (3, 15)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    import torchrec.sparse.triton_batch_index_select as triton_batch_index_select_module
    from torchrec.sparse.triton_batch_index_select import triton_batch_index_select_dim0


def _reference(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    input_splits = inputs.split(
        [rows * columns for rows, columns in zip(input_rows, input_columns)]
    )
    index_splits = indices.split(batch_size)
    outputs = [
        input_part.view(rows, columns).index_select(0, index_part)
        for input_part, index_part, rows, columns in zip(
            input_splits, index_splits, input_rows, input_columns
        )
    ]
    return torch.cat(outputs, dim=1).flatten()


@unittest.skipUnless(_TESTS_SUPPORTED, "CUDA and Python below 3.15 are required")
class TritonBatchIndexSelectTest(unittest.TestCase):
    def _inputs(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, int, list[int], list[int]]:
        batch_size = 13
        input_rows = [17, 23, 31, 37, 41]
        input_columns = [12, 16, 48, 112, 160]
        inputs = torch.cat(
            [
                torch.randn(rows, columns, device="cuda", dtype=dtype).flatten()
                for rows, columns in zip(input_rows, input_columns)
            ]
        ).requires_grad_(True)
        index_parts = []
        for rows in input_rows:
            index_part = torch.randint(0, rows, (batch_size,), device="cuda")
            index_part[1] = index_part[0]
            index_parts.append(index_part)
        indices = torch.cat(index_parts)
        return inputs, indices, batch_size, input_rows, input_columns

    def test_forward_and_backward(self) -> None:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs, indices, batch_size, input_rows, input_columns = self._inputs(
                    dtype
                )
                reference_inputs = inputs.detach().clone().requires_grad_()
                output = triton_batch_index_select_dim0(
                    inputs, indices, batch_size, input_rows, input_columns
                )
                expected = _reference(
                    reference_inputs,
                    indices,
                    batch_size,
                    input_rows,
                    input_columns,
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

                grad_output = torch.randn_like(output)
                actual_grad = torch.autograd.grad(output, inputs, grad_output)[0]
                expected_grad = torch.autograd.grad(
                    expected, reference_inputs, grad_output
                )[0]
                # Duplicate indices make the atomic accumulation order nondeterministic.
                tolerance = 16 * torch.finfo(dtype).eps
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=0, atol=tolerance
                )

    def test_compile_forward_and_backward(self) -> None:
        inputs, indices, batch_size, input_rows, input_columns = self._inputs(
            torch.float32
        )
        reference_inputs = inputs.detach().clone().requires_grad_()
        compiled = torch.compile(
            triton_batch_index_select_dim0,
            backend="aot_eager",
            fullgraph=True,
        )
        output = compiled(inputs, indices, batch_size, input_rows, input_columns)
        expected = _reference(
            reference_inputs, indices, batch_size, input_rows, input_columns
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        grad_output = torch.randn_like(output)
        actual_grad = torch.autograd.grad(output, inputs, grad_output)[0]
        expected_grad = torch.autograd.grad(expected, reference_inputs, grad_output)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-5)

    def test_large_batch_launch_grid(self) -> None:
        batch_size = 65536
        input_rows = [2]
        input_columns = [160]
        inputs = torch.randn(320, device="cuda")
        indices = torch.randint(0, 2, (batch_size,), device="cuda")

        output = triton_batch_index_select_dim0(
            inputs, indices, batch_size, input_rows, input_columns
        )
        expected = _reference(inputs, indices, batch_size, input_rows, input_columns)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_sorted_reduction_backward(self) -> None:
        batch_size = 64
        input_rows = [17, 23]
        input_columns = [512, 512]
        inputs = torch.randn(
            sum(rows * columns for rows, columns in zip(input_rows, input_columns)),
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        indices = torch.cat(
            [
                torch.zeros(batch_size, device="cuda", dtype=torch.int64),
                torch.arange(batch_size, device="cuda") % input_rows[1],
            ]
        )
        reference_inputs = inputs.detach().clone().requires_grad_()

        with mock.patch.object(
            triton_batch_index_select_module,
            "_SORTED_REDUCTION_MIN_OUTPUT_ELEMENTS",
            0,
        ):
            output = triton_batch_index_select_dim0(
                inputs, indices, batch_size, input_rows, input_columns
            )
            grad_output = torch.randn_like(output)
            actual_grad = torch.autograd.grad(output, inputs, grad_output)[0]

        expected = _reference(
            reference_inputs,
            indices,
            batch_size,
            input_rows,
            input_columns,
        )
        expected_grad = torch.autograd.grad(expected, reference_inputs, grad_output)[0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=0,
            atol=batch_size * torch.finfo(inputs.dtype).eps,
        )

    def test_specialized_backward_paths(self) -> None:
        cases = (
            ("high_duplication", 1024, [64] * 32, [128] * 32, False),
            ("mostly_unique", 4096, [3072] * 4, [256] * 4, True),
        )
        for name, batch_size, input_rows, input_columns, cover_all_rows in cases:
            with self.subTest(name=name):
                inputs = torch.randn(
                    sum(
                        rows * columns
                        for rows, columns in zip(input_rows, input_columns)
                    ),
                    device="cuda",
                    dtype=torch.float16,
                    requires_grad=True,
                )
                index_parts = []
                for rows in input_rows:
                    if cover_all_rows:
                        index_part = torch.cat(
                            [
                                torch.arange(rows, device="cuda"),
                                torch.randint(
                                    0,
                                    rows,
                                    (batch_size - rows,),
                                    device="cuda",
                                ),
                            ]
                        )[torch.randperm(batch_size, device="cuda")]
                    else:
                        index_part = torch.randint(
                            0, rows, (batch_size,), device="cuda"
                        )
                    index_parts.append(index_part)
                indices = torch.cat(index_parts)
                reference_inputs = inputs.detach().clone().requires_grad_()

                output = triton_batch_index_select_dim0(
                    inputs, indices, batch_size, input_rows, input_columns
                )
                expected = _reference(
                    reference_inputs,
                    indices,
                    batch_size,
                    input_rows,
                    input_columns,
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                grad_output = torch.randn_like(output)
                actual_grad = torch.autograd.grad(output, inputs, grad_output)[0]
                expected_grad = torch.autograd.grad(
                    expected, reference_inputs, grad_output
                )[0]
                torch.testing.assert_close(
                    actual_grad,
                    expected_grad,
                    rtol=0,
                    atol=max(16, 2 * batch_size // min(input_rows))
                    * torch.finfo(inputs.dtype).eps,
                )
