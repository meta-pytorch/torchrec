#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Optional, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import assume, given, settings
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)


class QuantizationCommCodecTest(unittest.TestCase):
    @settings(deadline=4000)
    @given(
        comm_precisions_loss_scale=st.sampled_from(
            [
                (CommType.FP32, None),
                (CommType.FP16, None),
                (CommType.FP16, 4.0),
                (CommType.BF16, None),
                (CommType.FP8, None),
                (CommType.INT8, None),
            ]
        ),
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
        row_dim=st.sampled_from([-1, 4, 8, 16, 32]),
    )
    def test_quantized_comm_codec(
        self,
        comm_precisions_loss_scale: Tuple[CommType, Optional[float]],
        row_size: int,
        col_size: int,
        rand_seed: int,
        row_dim: int,
    ) -> None:

        (comm_precision, loss_scale) = comm_precisions_loss_scale

        if comm_precision == CommType.FP8:
            if row_dim > 0:
                assume((col_size * row_size) % row_dim == 0)
            assume(col_size % 4 == 0)

        torch.manual_seed(rand_seed)
        shape = (row_size, col_size)
        input_tensor = torch.rand(shape, requires_grad=True)
        cur_row_dim = None

        if (
            comm_precision == CommType.FP8
            and torch.cuda.device_count() != 0
            and row_dim > 0
        ):
            cur_row_dim = row_dim
            input_tensor = input_tensor.view(-1).cuda()

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=comm_precision,
                fp8_quantize_dim=cur_row_dim,
            )
        )
        ctx = quant_codec.forward.create_context()
        if comm_precision == CommType.INT8:
            # pyrefly: ignore[missing-attribute]
            assume(row_size * col_size % ctx.row_dim == 0)
            input_tensor = input_tensor.view(-1)

        quant_tensor = quant_codec.forward.encode(input_tensor, ctx)
        output_tensor = quant_codec.forward.decode(quant_tensor, ctx)

        rtol = 0.005
        atol = 0.005
        if comm_precision == CommType.FP8:
            rtol = 0.05
            atol = 0.05

        torch.testing.assert_close(
            input_tensor.detach().cpu(),
            output_tensor.detach().cpu(),
            rtol=rtol,
            atol=atol,
        )

    def test_fp8_rowwise_padding(self) -> None:
        """Verify FP8 rowwise padding handles non-aligned dim_sum per rank.

        Reproduces the production crash:
            RuntimeError: input_len N is not a multiple of row dim 256
        which occurs when B_local * D_rank_sum % row_dim != 0 during
        FP8-quantized AllToAll with small eval batch sizes and column-based
        sharding that produces irregular D_rank_sum values.
        """
        row_dim = 256
        batch_size = 16
        # Non-aligned dims: 259 % 256 = 3, 131 % 256 = 131
        # Without padding: 16 * 259 = 4144, 4144 % 256 = 48 → crash
        dim_sum_per_rank = [259, 131]
        my_rank = 0
        dim_sum = dim_sum_per_rank[my_rank]

        input_tensor = torch.rand((batch_size, dim_sum), requires_grad=False)

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                fp8_quantize_dim=row_dim,
            )
        )
        ctx = quant_codec.forward.create_context()

        # padded_size pads 259 → 512, 131 → 256 (next multiples of 256)
        padded_dim, padding_size = quant_codec.forward.padded_size(
            input_tensor, dim_sum_per_rank, my_rank, ctx
        )

        self.assertEqual(padded_dim, 512)
        self.assertEqual(padding_size, 253)
        # pyrefly: ignore[missing-attribute]
        self.assertIsNotNone(ctx.padded_dim_sum_per_rank)
        # pyrefly: ignore[missing-attribute]
        self.assertEqual(ctx.padded_dim_sum_per_rank, [512, 256])
        # pyrefly: ignore[missing-attribute]
        self.assertTrue(all(d % row_dim == 0 for d in ctx.padded_dim_sum_per_rank))

        # calc_quantized_size must not crash with padded dims
        # Without padding, calc_quantized_size(16 * 259 = 4144) would assert
        # because 4144 % 256 = 48.
        # pyrefly: ignore[missing-attribute]
        for padded_d in ctx.padded_dim_sum_per_rank:
            size = quant_codec.forward.calc_quantized_size(batch_size * padded_d, ctx)
            self.assertGreater(size, 0)

    def test_fp8_rowwise_2d_tensor_padding(self) -> None:
        """Verify 2D tensor is padded along dim 1 before FP8 encoding.

        Without F.pad(input_embeddings, (0, padding_size)), the 2D tensor
        has D_local_sum not aligned to row_dim. encode() calls
        view(-1, row_dim) which fails, or produces a tensor whose size
        doesn't match the padded split sizes (Split sizes mismatch error).
        """
        row_dim = 256
        batch_size = 16
        dim_sum = 259  # not a multiple of 256
        dim_sum_per_rank = [dim_sum, 131]
        my_rank = 0

        input_tensor = torch.rand((batch_size, dim_sum), requires_grad=False)

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                fp8_quantize_dim=row_dim,
            )
        )
        ctx = quant_codec.forward.create_context()

        _, padding_size = quant_codec.forward.padded_size(
            input_tensor, dim_sum_per_rank, my_rank, ctx
        )
        self.assertGreater(padding_size, 0)

        # Pad the tensor (the fix)
        padded_tensor = torch.nn.functional.pad(input_tensor, (0, padding_size))
        self.assertEqual(padded_tensor.shape[1], 512)  # 259 + 253 = 512
        self.assertEqual(padded_tensor.shape[1] % row_dim, 0)

        # Total elements must align for view(-1, row_dim) in _quantize_tensor
        self.assertEqual(padded_tensor.numel() % row_dim, 0)

        # Without padding, total elements would NOT align
        self.assertNotEqual(input_tensor.numel() % row_dim, 0)

    def test_fp8_rowwise_variable_batch_split_padding(self) -> None:
        """Verify per-split padding for variable batch FP8 quantized AllToAll.

        In the variable batch path, splits are sum(batch_size_j * emb_dim_j)
        per rank. Each split must be padded individually to a multiple of
        row_dim. The sum of individually quantized padded splits must equal
        the quantized size of the total padded tensor.
        """
        row_dim = 256
        # Real failing input_len values from production
        original_splits = [549408, 555760, 558000, 562048]

        # None are multiples of 256
        for split in original_splits:
            self.assertNotEqual(split % row_dim, 0)

        # Pad each split individually
        padded_splits = [
            split + (row_dim - split % row_dim) % row_dim for split in original_splits
        ]

        # All padded splits must be multiples of row_dim
        for ps in padded_splits:
            self.assertEqual(ps % row_dim, 0)

        # Padded total >= original total
        self.assertGreaterEqual(sum(padded_splits), sum(original_splits))

        # calc_quantized_size must not crash on padded splits
        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                fp8_quantize_dim=row_dim,
            )
        )
        ctx = quant_codec.forward.create_context()
        for ps in padded_splits:
            size = quant_codec.forward.calc_quantized_size(ps, ctx)
            self.assertGreater(size, 0)

        # Critical invariant: sum of individually quantized splits ==
        # quantized size of total. If this fails, AllToAll crashes with
        # "Split sizes doesn't match total dim 0 size".
        q_splits = [
            quant_codec.forward.calc_quantized_size(ps, ctx) for ps in padded_splits
        ]
        q_total = quant_codec.forward.calc_quantized_size(sum(padded_splits), ctx)
        self.assertEqual(sum(q_splits), q_total)

    def test_fp8_rowwise_no_padding_crashes(self) -> None:
        """Without FP8 padding, calc_quantized_size crashes on non-aligned dims."""
        row_dim = 256
        batch_size = 16
        dim_sum = 259  # 259 % 256 = 3, not aligned

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.FP8,
                fp8_quantize_dim=row_dim,
            )
        )
        ctx = quant_codec.forward.create_context()

        # Directly calling calc_quantized_size without padding must fail
        unpadded_input_len = batch_size * dim_sum  # 16 * 259 = 4144
        self.assertNotEqual(unpadded_input_len % row_dim, 0)
        with self.assertRaises((AssertionError, RuntimeError)):
            quant_codec.forward.calc_quantized_size(unpadded_input_len, ctx)

    @settings(deadline=4000)
    @given(
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
    )
    def test_mx4_comm_codec(
        self,
        row_size: int,
        col_size: int,
        rand_seed: int,
    ) -> None:

        torch.manual_seed(rand_seed)
        shape = (row_size, col_size)
        input_tensor = torch.rand(shape, requires_grad=False) * 2 - 1

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=CommType.MX4,
            )
        )
        dim_sum_per_rank = [shape[1]]
        ctx = quant_codec.forward.create_context()

        rank = 0
        quant_codec.forward.padded_size(input_tensor, dim_sum_per_rank, rank, ctx)
        quant_tensor = quant_codec.forward.encode(input_tensor, ctx)
        output_tensor = quant_codec.forward.decode(quant_tensor, ctx)
        # pyrefly: ignore[missing-attribute]
        output_tensor = output_tensor.view(shape[0], ctx.padded_dim_sum_per_rank[rank])
        output_tensor = output_tensor[:, : shape[1]]

        rtol = 0.1
        atol = 0.15

        torch.testing.assert_close(
            input_tensor.detach().cpu(),
            output_tensor.detach().cpu(),
            rtol=rtol,
            atol=atol,
        )
