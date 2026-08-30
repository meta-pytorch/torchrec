#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from torchrec.distributed.maglev.input_dist import (
    flatten_to_tensors,
    input_data_dist,
    input_size_dist,
    unflatten_from_tensors,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class _Carrier:
    name: str  # non-tensor leaf
    activation: torch.Tensor
    extras: List[torch.Tensor]
    pair: Tuple[torch.Tensor, int]  # mixed tensor + non-tensor
    named: Dict[str, torch.Tensor]
    features: KeyedJaggedTensor


def _kjt(values: torch.Tensor) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=["f1", "f2"],
        values=values,
        lengths=torch.tensor([1, 2, 1, 2], dtype=torch.int64),
    )


def _example() -> _Carrier:
    return _Carrier(
        name="carrier",
        activation=torch.zeros(2, 3),
        extras=[torch.zeros(4), torch.zeros(5)],
        pair=(torch.zeros(1), 7),
        named={"a": torch.zeros(2), "b": torch.zeros(3)},
        features=_kjt(torch.zeros(6, dtype=torch.int64)),
    )


def _instance() -> _Carrier:
    return _Carrier(
        name="carrier",
        activation=torch.randn(2, 3),
        extras=[torch.randn(4), torch.randn(5)],
        pair=(torch.randn(1), 7),
        named={"a": torch.randn(2), "b": torch.randn(3)},
        features=_kjt(torch.randint(0, 10, (6,), dtype=torch.int64)),
    )


class InputDistTest(unittest.TestCase):
    def test_flatten_count(self) -> None:
        tensors = flatten_to_tensors(_instance())
        # activation(1) + extras(2) + pair tensor(1) + named(2) + kjt values+lengths(2)
        self.assertEqual(len(tensors), 8)
        for t in tensors:
            self.assertIsInstance(t, torch.Tensor)

    def test_roundtrip(self) -> None:
        obj = _instance()
        rebuilt = unflatten_from_tensors(flatten_to_tensors(obj), _example())

        self.assertIsInstance(rebuilt, _Carrier)
        self.assertEqual(rebuilt.name, "carrier")
        torch.testing.assert_close(rebuilt.activation, obj.activation)
        self.assertEqual(len(rebuilt.extras), 2)
        for got, want in zip(rebuilt.extras, obj.extras):
            torch.testing.assert_close(got, want)
        torch.testing.assert_close(rebuilt.pair[0], obj.pair[0])
        self.assertEqual(rebuilt.pair[1], 7)  # non-tensor leaf from example
        torch.testing.assert_close(rebuilt.named["a"], obj.named["a"])
        torch.testing.assert_close(rebuilt.named["b"], obj.named["b"])
        self.assertEqual(rebuilt.features.keys(), ["f1", "f2"])
        torch.testing.assert_close(rebuilt.features.values(), obj.features.values())
        torch.testing.assert_close(rebuilt.features.lengths(), obj.features.lengths())

    def test_non_tensor_comes_from_example(self) -> None:
        obj = _instance()
        example = _example()
        example.name = "from_example"
        rebuilt = unflatten_from_tensors(flatten_to_tensors(obj), example)
        # ``name`` is a non-tensor leaf, so it is taken from the example.
        self.assertEqual(rebuilt.name, "from_example")

    def test_dict_of_kjt_roundtrip(self) -> None:
        obj = {
            "a": _kjt(torch.randint(0, 10, (6,), dtype=torch.int64)),
            "b": _kjt(torch.randint(0, 10, (6,), dtype=torch.int64)),
        }
        example = {
            "a": _kjt(torch.zeros(6, dtype=torch.int64)),
            "b": _kjt(torch.zeros(6, dtype=torch.int64)),
        }
        tensors = flatten_to_tensors(obj)
        # Two KJTs, each contributing values + lengths.
        self.assertEqual(len(tensors), 4)

        rebuilt = unflatten_from_tensors(tensors, example)
        self.assertEqual(list(rebuilt.keys()), ["a", "b"])
        for key in ("a", "b"):
            torch.testing.assert_close(rebuilt[key].values(), obj[key].values())
            torch.testing.assert_close(rebuilt[key].lengths(), obj[key].lengths())
            self.assertEqual(rebuilt[key].keys(), ["f1", "f2"])

    def test_too_few_tensors_raises(self) -> None:
        tensors = flatten_to_tensors(_instance())
        with self.assertRaises(ValueError):
            unflatten_from_tensors(tensors[:-1], _example())

    def test_too_many_tensors_raises(self) -> None:
        tensors = flatten_to_tensors(_instance())
        with self.assertRaises(ValueError):
            unflatten_from_tensors(tensors + [torch.zeros(1)], _example())


_DENSE_DIM = 4
_NUM_FEATURES = 2


@dataclass
class _Batch:
    dense: torch.Tensor
    features: KeyedJaggedTensor
    label: torch.Tensor
    extras: List[torch.Tensor]
    meta: Dict[str, torch.Tensor]
    ids: torch.Tensor  # int32: a third dtype bucket, with a trailing dim
    tag: str  # non-tensor leaf: must come from the example


def _make_kjt(src: int, dst: int, batch_size: int) -> KeyedJaggedTensor:
    g = torch.Generator().manual_seed(1009 * src + dst + 1)
    lengths = torch.randint(
        0, 3, (_NUM_FEATURES * batch_size,), generator=g, dtype=torch.int64
    )
    values = torch.randint(
        0, 100, (int(lengths.sum()),), generator=g, dtype=torch.int64
    )
    return KeyedJaggedTensor(keys=["f1", "f2"], values=values, lengths=lengths)


def _make_carrier(src: int, dst: int) -> _Batch:
    """Deterministic carrier rank ``src`` sends to rank ``dst``.

    Content and batch size are pure functions of ``(src, dst)`` so any rank can
    reconstruct the expected value. Batch size varies with both to exercise the
    per-item / per-slot variable dim-0 sizing; ``extras`` mixes two different
    dim-0 sizes on purpose. Dtypes span three buckets -- float32 (``dense`` /
    ``label`` / ``extras`` / ``meta``), int64 (the KJT), and int32 (``ids``) -- so
    ``input_data_dist``'s dtype bucketing runs three fused all-to-alls.
    """
    batch_size = 1 + src + dst
    dense = torch.full((batch_size, _DENSE_DIM), float(10 * src + dst))
    label = torch.arange(batch_size, dtype=torch.float32) + src
    extras = [
        torch.full((batch_size,), float(src)),
        torch.full((batch_size + 1,), float(dst)),
    ]
    meta = {"m": torch.full((batch_size, 2), float(src - dst))}
    ids = (torch.arange(batch_size * 3, dtype=torch.int32) + (10 * src + dst)).reshape(
        batch_size, 3
    )
    return _Batch(
        dense=dense,
        features=_make_kjt(src, dst, batch_size),
        label=label,
        extras=extras,
        meta=meta,
        ids=ids,
        tag="batch",
    )


class InputDistInProcessTest(unittest.TestCase):
    """Single-rank coverage for size exchange and CUDA data distribution."""

    def setUp(self) -> None:
        super().setUp()
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )

    def tearDown(self) -> None:
        dist.destroy_process_group()
        super().tearDown()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_input_size_dist_copies_into_fused_gpu_buffers(self) -> None:
        pg = dist.group.WORLD
        assert pg is not None
        device = torch.device("cuda")
        memcpy_stream = torch.cuda.Stream(device=device)
        send = [_make_carrier(src=0, dst=0)]
        example = _make_carrier(src=0, dst=0)
        source_tensors = flatten_to_tensors(send[0])

        (
            in_bufs,
            in_splits_by_bucket,
            size_awaitable,
            example_flat,
            copy_done_event,
        ) = input_size_dist(send, example, pg, device, memcpy_stream)
        recv_sizes = size_awaitable.wait()
        copy_done_event.synchronize()

        self.assertEqual(recv_sizes[0], [tensor.shape[0] for tensor in source_tensors])
        self.assertEqual(len(example_flat), len(source_tensors))
        expected_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
        for tensor in source_tensors:
            expected_by_dtype.setdefault(tensor.dtype, []).append(tensor.reshape(-1))
        self.assertEqual(len(in_bufs), len(expected_by_dtype))
        for in_buf, in_splits, parts in zip(
            in_bufs,
            in_splits_by_bucket,
            expected_by_dtype.values(),
        ):
            self.assertEqual(in_buf.device.type, "cuda")
            self.assertEqual(in_splits, [sum(part.numel() for part in parts)])
            torch.testing.assert_close(in_buf.cpu(), torch.cat(parts))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @patch("torchrec.distributed.maglev.input_dist.dist.all_to_all_single")
    def test_input_data_dist_uses_pp_stream(self, all_to_all_single: Any) -> None:
        pg = dist.group.WORLD
        assert pg is not None
        device = torch.device("cuda")
        pp_data_dist_stream = torch.cuda.Stream(device=device)
        source = _make_carrier(src=0, dst=0)
        source_tensors = flatten_to_tensors(source)
        tensors_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
        for tensor in source_tensors:
            tensors_by_dtype.setdefault(tensor.dtype, []).append(tensor.reshape(-1))
        in_bufs = [torch.cat(parts).to(device) for parts in tensors_by_dtype.values()]
        in_splits_by_bucket = [[in_buf.numel()] for in_buf in in_bufs]
        recv_sizes = [[tensor.shape[0] for tensor in source_tensors]]

        copy_done_event = torch.cuda.Event()
        copy_done_event.record(torch.cuda.current_stream(device))

        def _all_to_all(
            out_buf: torch.Tensor,
            in_buf: torch.Tensor,
            out_splits: List[int],
            in_splits: List[int],
            group: dist.ProcessGroup,
            async_op: bool,
        ) -> None:
            self.assertFalse(async_op)
            self.assertEqual(torch.cuda.current_stream(device), pp_data_dist_stream)
            self.assertEqual(out_splits, in_splits)
            out_buf.copy_(in_buf)

        all_to_all_single.side_effect = _all_to_all

        (received,) = input_data_dist(
            in_bufs,
            in_splits_by_bucket,
            recv_sizes,
            source,
            source_tensors,
            pg,
            pp_data_dist_stream=pp_data_dist_stream,
            copy_done_event=copy_done_event,
        ).wait()

        for expected, actual in zip(source_tensors, flatten_to_tensors(received)):
            torch.testing.assert_close(actual.cpu(), expected)
