#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import torch
from torch.overrides import TorchFunctionMode
from torchrec.distributed.shards_wrapper import (
    merge_cw_local_shards_by_view,
    merging_cw_local_shards_by_view,
    view_cw_local_shards,
)


def _reference(shards: List[torch.Tensor], dim: int) -> torch.Tensor:
    # What the sharded modules' pre-load hooks compute for a rank's CW shards.
    return torch.cat([s.view(-1) for s in shards], dim=0).view(-1, dim)


def _fused_layout(
    rows: int, widths: List[int], dtype: torch.dtype = torch.float16
) -> List[torch.Tensor]:
    # A fused TBE stores a table's same-rank CW shards back to back in one buffer.
    flat = torch.randn(rows * sum(widths)).to(dtype)
    shards, start = [], 0
    for w in widths:
        shards.append(flat[start : start + rows * w].view(rows, w))
        start += rows * w
    return shards


def _not_aliasable() -> List[List[torch.Tensor]]:
    # Inputs that must not be aliased even inside the opt-in. Each passes every alias
    # check but one: the second shard sits at the first's offset + numel.
    f1, f2 = torch.randn(40), torch.randn(40)
    flat = torch.randn(40)
    strided = torch.randn(80)
    complex_flat = torch.tensor([1 + 2j, 3 + 4j])
    neg_flat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gap = torch.randn(41)
    quant = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0, 3.0, 4.0]), 1.0, 0, torch.quint8
    )
    quant[2:].view(1, 2).copy_(
        torch.quantize_per_tensor(torch.tensor([[6.0, 8.0]]), 2.0, 0, torch.quint8)
    )
    buf = bytearray(16)

    class _Sub(torch.Tensor):
        pass

    a, b = _fused_layout(rows=5, widths=[4, 4], dtype=torch.float32)
    return [
        [f1[:20].view(5, 4), f2[20:40].view(5, 4)],  # other storage
        [  # same address, smaller storage
            torch.frombuffer(buf, dtype=torch.float32, count=2).view(1, 2),
            torch.frombuffer(buf, dtype=torch.float32)[2:].view(1, 2),
        ],
        [flat[:20].view(5, 4), flat.view(torch.int32)[20:40].view(5, 4)],  # dtype
        [complex_flat[:1].view(1, 1), complex_flat[1:].conj().view(1, 1)],  # conj
        [neg_flat[:1], torch._neg_view(neg_flat[1:])],  # negative view
        [a.as_subclass(_Sub), b.as_subclass(_Sub)],  # tensor subclass
        [gap[:20].view(5, 4), gap[21:41].view(5, 4)],  # a one-element gap
        [quant[:2].view(1, 2), quant[2:].view(1, 2)],  # quantized, two quantizers
        [b, a],  # out of memory order
        [strided[0:40:2].view(5, 4), strided[20:60:2].view(5, 4)],  # strided
        [torch.randn(24).view(6, 4), torch.randn(48).view(6, 8)],  # unequal, apart
    ]


class _RecordTorchCalls(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[object] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(func)
        return func(*args, **(kwargs or {}))


class OutsideOptInTest(unittest.TestCase):
    """Without the opt-in, the helper does nothing at all."""

    def test_returns_none_without_a_torch_call(self) -> None:
        shards = _fused_layout(rows=5, widths=[4, 4], dtype=torch.float32)
        self.assertFalse(merging_cw_local_shards_by_view())
        with _RecordTorchCalls() as rec:
            self.assertIsNone(view_cw_local_shards(shards, 4))
        self.assertEqual(rec.calls, [])


class OptInTest(unittest.TestCase):
    """Inside merge_cw_local_shards_by_view(), adjacent shards are aliased."""

    def test_adjacent_shards_alias(self) -> None:
        for widths, rows in (([132, 132], 50), ([8, 8, 8], 50), ([4], 50), ([4, 8], 6)):
            shards = _fused_layout(rows=rows, widths=widths)
            dim = widths[0]
            with merge_cw_local_shards_by_view():
                merged = view_cw_local_shards(shards, dim)
            assert merged is not None
            _assert_same(self, merged, _reference(shards, dim))
            self.assertEqual(merged.data_ptr(), shards[0].data_ptr())
            self.assertEqual(merged.shape, (rows * sum(widths) // dim, dim))

    def test_adjacent_shards_with_leading_offset(self) -> None:
        flat = torch.randn(10 + 2 * 3 * 4)
        shards = [flat[10:22].view(3, 4), flat[22:34].view(3, 4)]
        with merge_cw_local_shards_by_view():
            merged = view_cw_local_shards(shards, 4)
        assert merged is not None
        _assert_same(self, merged, _reference(shards, 4))
        self.assertEqual(merged.data_ptr(), shards[0].data_ptr())

    def test_other_inputs_return_none(self) -> None:
        for shards in _not_aliasable():
            with merge_cw_local_shards_by_view():
                self.assertIsNone(view_cw_local_shards(shards, shards[0].shape[-1]))

    def test_sizes_a_view_cannot_take_return_none(self) -> None:
        # Shapes the caller's own final view(-1, dim) rejects: the caller then raises.
        cases = [
            (_fused_layout(rows=3, widths=[1, 1], dtype=torch.float32), 4),
            (_fused_layout(rows=5, widths=[4, 4], dtype=torch.float32), 0),
            (_fused_layout(rows=5, widths=[4, 4], dtype=torch.float32), -1),
            ([], 4),
        ]
        for shards, dim in cases:
            with merge_cw_local_shards_by_view():
                self.assertIsNone(view_cw_local_shards(shards, dim))

    def test_view_errors_are_raised_first(self) -> None:
        # The caller's own view(-1) errors are raised the same way.
        base = torch.arange(8.0, requires_grad=True)
        a = base[:4].view(2, 2)
        with torch.no_grad():
            b = base[4:].view(2, 2)
            base.add_(1)
        flat = torch.randn(40)
        cases = [
            ([a, b], 2, RuntimeError),
            ([flat[:20].view(4, 5).t(), flat[20:40].view(5, 4)], 4, RuntimeError),
            (
                [torch.eye(4).to_sparse(), torch.eye(4).to_sparse()],
                4,
                NotImplementedError,
            ),
        ]
        for shards, dim, err in cases:
            with self.assertRaises(err):
                _reference(shards, dim)
            with merge_cw_local_shards_by_view():
                with self.assertRaises(err):
                    view_cw_local_shards(shards, dim)

    def test_storageless_batched_tensor_returns_none(self) -> None:
        results: List[bool] = []

        def merge(t: torch.Tensor) -> torch.Tensor:
            results.append(view_cw_local_shards([t[:2], t[2:]], 2) is None)
            return t * 1

        with merge_cw_local_shards_by_view():
            torch.vmap(merge)(torch.arange(8.0).reshape(2, 4))
        self.assertEqual(results, [True])

    def test_alias_load_is_a_self_copy(self) -> None:
        # The eval reload path: the merged tensor is copied back into the table's own
        # parameter, which already holds the same bytes.
        shards = _fused_layout(rows=6, widths=[4, 4], dtype=torch.float32)
        param = shards[0].as_strided((12, 4), (4, 1))
        before = param.clone()
        with merge_cw_local_shards_by_view():
            merged = view_cw_local_shards(shards, 4)
            assert merged is not None
            param.copy_(merged)
        self.assertTrue(torch.equal(param, before))

    def test_opt_in_ends_with_the_block(self) -> None:
        shards = _fused_layout(rows=5, widths=[4, 4], dtype=torch.float32)
        with self.assertRaises(KeyError):
            with merge_cw_local_shards_by_view():
                raise KeyError("leave the block by an exception")
        for _ in range(2):
            self.assertFalse(merging_cw_local_shards_by_view())
            self.assertIsNone(view_cw_local_shards(shards, 4))
            with merge_cw_local_shards_by_view():
                with merge_cw_local_shards_by_view():
                    pass
                self.assertTrue(merging_cw_local_shards_by_view())
                inner = view_cw_local_shards(shards, 4)
            assert inner is not None
            self.assertEqual(inner.data_ptr(), shards[0].data_ptr())


def _assert_same(test: unittest.TestCase, got: torch.Tensor, ref: torch.Tensor) -> None:
    # torch.equal alone ignores dtype, so compare it and the shape explicitly.
    test.assertEqual((got.dtype, got.shape), (ref.dtype, ref.shape))
    test.assertTrue(torch.equal(got, ref))
