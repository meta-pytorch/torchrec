#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.maglev.input_dist import (
    flatten_to_tensors,
    input_data_dist,
    input_dist,
    input_size_dist,
    unflatten_from_tensors,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
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


def _run_all_to_all(rank: int, world_size: int) -> None:
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo") as ctx:
        pg = ctx.pg
        assert pg is not None

        send = [_make_carrier(src=rank, dst=dst) for dst in range(world_size)]
        # Structure/dtype template; its sizes and content are irrelevant.
        example = _make_carrier(src=0, dst=0)
        example.tag = "from_example"

        # gloo stands in for nccl here: all_to_all_single is backend-agnostic, so
        # the same group drives both the size and the data phase in the test.
        recv = input_dist(send, example, pg_gloo=pg, pg_nccl=pg).wait()

        assert (
            len(recv) == world_size
        ), f"expected {world_size} carriers, got {len(recv)}"
        for src in range(world_size):
            expected = _make_carrier(src=src, dst=rank)
            got = recv[src]
            torch.testing.assert_close(got.dense, expected.dense)
            torch.testing.assert_close(got.label, expected.label)
            torch.testing.assert_close(got.ids, expected.ids)
            assert got.ids.dtype == torch.int32
            assert len(got.extras) == 2
            for g_t, e_t in zip(got.extras, expected.extras):
                torch.testing.assert_close(g_t, e_t)
            torch.testing.assert_close(got.meta["m"], expected.meta["m"])
            assert got.features.keys() == ["f1", "f2"]
            torch.testing.assert_close(
                got.features.values(), expected.features.values()
            )
            torch.testing.assert_close(
                got.features.lengths(), expected.features.lengths()
            )
            # non-tensor leaf comes from the example, not the sender.
            assert got.tag == "from_example", got.tag


class InputAllToAllTest(MultiProcessTestBase):
    def test_all_to_all_roundtrip(self) -> None:
        self._run_multi_process_test(
            callable=_run_all_to_all,
            world_size=4,
        )


class InputDistInProcessTest(unittest.TestCase):
    """In-process (single-rank, gloo) coverage of the dist entry points.

    A ``world_size == 1`` all-to-all is the identity (``send[0]`` -> self), so this
    exercises ``input_size_dist`` / ``input_data_dist`` / ``input_dist`` and their
    awaitables in the test process (unlike the multi-process test, whose spawned
    workers run outside coverage) without needing more than one rank.
    """

    def setUp(self) -> None:
        super().setUp()
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )

    def tearDown(self) -> None:
        dist.destroy_process_group()
        super().tearDown()

    def test_input_dist_single_rank(self) -> None:
        pg = dist.group.WORLD
        assert pg is not None
        send = [_make_carrier(src=0, dst=0)]
        example = _make_carrier(src=0, dst=0)
        example.tag = "from_example"

        recv = input_dist(send, example, pg_gloo=pg, pg_nccl=pg).wait()

        self.assertEqual(len(recv), 1)
        got = recv[0]
        torch.testing.assert_close(got.dense, send[0].dense)
        torch.testing.assert_close(got.label, send[0].label)
        torch.testing.assert_close(got.ids, send[0].ids)
        self.assertEqual(got.ids.dtype, torch.int32)
        torch.testing.assert_close(got.features.values(), send[0].features.values())
        torch.testing.assert_close(got.features.lengths(), send[0].features.lengths())
        self.assertEqual(got.tag, "from_example")

    def test_size_then_data_phase(self) -> None:
        pg = dist.group.WORLD
        assert pg is not None
        send = [_make_carrier(src=0, dst=0)]
        example = _make_carrier(src=0, dst=0)

        flat_send, recv_sizes = input_size_dist(send, pg).wait()
        # One carrier -> one row; sizes match the flattened tensors' dim-0.
        self.assertEqual(len(recv_sizes), 1)
        self.assertEqual(recv_sizes[0], [t.shape[0] for t in flat_send[0]])

        recv = input_data_dist(flat_send, recv_sizes, example, pg).wait()
        self.assertEqual(len(recv), 1)
        torch.testing.assert_close(recv[0].dense, send[0].dense)
