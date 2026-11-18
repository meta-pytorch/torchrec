#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from parameterized import parameterized
from torchrec.modules.segment_permute import permute_segments_gather_backward
from torchrec.modules.utils import _permute_tensor_by_segments


def _run(
    use_gather: bool,
    values: torch.Tensor,
    segment_sizes: torch.Tensor,
    recat: torch.Tensor,
    grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_size = int(segment_sizes[recat].sum().item())
    x = values.detach().clone().requires_grad_(True)
    if use_gather:
        out = permute_segments_gather_backward(x, segment_sizes, recat, output_size)
    else:
        out, _ = _permute_tensor_by_segments(
            x, segment_sizes, recat, weights=None, output_size=output_size
        )
    out.backward(grad)
    grad_input = x.grad
    assert grad_input is not None
    return out.detach(), grad_input


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class SegmentPermuteTest(unittest.TestCase):
    def _inputs(
        self, num_segments: int, max_size: int, recat: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device="cuda").manual_seed(0)
        segment_sizes = torch.randint(
            0, max_size, (num_segments,), device="cuda", generator=gen
        )
        values = torch.randn(
            int(segment_sizes.sum().item()), device="cuda", generator=gen
        ).to(dtype)
        output_size = int(segment_sizes[recat].sum().item())
        grad = torch.randn(output_size, device="cuda", generator=gen).to(dtype)
        return values, segment_sizes, grad

    @parameterized.expand(
        [
            ("bf16_small", torch.bfloat16, 17, 300),
            ("fp32_small", torch.float32, 17, 300),
            ("bf16_long_segments", torch.bfloat16, 64, 40000),
            ("fp16_many_segments", torch.float16, 2048, 600),
        ]
    )
    def test_permutation_is_bit_identical(
        self, _name: str, dtype: torch.dtype, num_segments: int, max_size: int
    ) -> None:
        recat = torch.randperm(num_segments, device="cuda")
        values, segment_sizes, grad = self._inputs(num_segments, max_size, recat, dtype)
        ref_out, ref_grad = _run(False, values, segment_sizes, recat, grad)
        out, grad_input = _run(True, values, segment_sizes, recat, grad)
        self.assertTrue(torch.equal(out, ref_out))
        self.assertTrue(torch.equal(grad_input, ref_grad))

    def test_repeated_and_missing_segments_match(self) -> None:
        # Segment 0 is selected three times, segments 3 and 5 never.
        recat = torch.tensor([0, 2, 0, 1, 4, 0, 6], device="cuda")
        values, segment_sizes, grad = self._inputs(7, 5000, recat, torch.float32)
        ref_out, ref_grad = _run(False, values, segment_sizes, recat, grad)
        out, grad_input = _run(True, values, segment_sizes, recat, grad)
        self.assertTrue(torch.equal(out, ref_out))
        torch.testing.assert_close(grad_input, ref_grad)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(segment_sizes)
        for missing in (3, 5):
            begin, end = int(offsets[missing]), int(offsets[missing + 1])
            self.assertTrue(torch.all(grad_input[begin:end] == 0))

    def test_backward_is_deterministic(self) -> None:
        recat = torch.tensor([3, 0, 3, 1, 2, 3], device="cuda")
        values, segment_sizes, grad = self._inputs(4, 50000, recat, torch.bfloat16)
        _, first = _run(True, values, segment_sizes, recat, grad)
        for _ in range(3):
            _, again = _run(True, values, segment_sizes, recat, grad)
            self.assertTrue(torch.equal(first, again))

    def test_empty(self) -> None:
        segment_sizes = torch.zeros(4, dtype=torch.int64, device="cuda")
        recat = torch.randperm(4, device="cuda")
        values = torch.empty(0, device="cuda")
        grad = torch.empty(0, device="cuda")
        out, grad_input = _run(True, values, segment_sizes, recat, grad)
        self.assertEqual(out.numel(), 0)
        self.assertEqual(grad_input.numel(), 0)
