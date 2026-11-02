#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import pickle
import unittest
from typing import Any

import torch
import torchrec.metrics.cpu_offloaded_metric_ipc as ipc
from torchrec.metrics.cpu_offloaded_metric_ipc import (
    ALIGNMENT_BYTES,
    build_slot_layout,
    header_ints,
    header_timestamp,
    HEADER_TIMESTAMP_END,
    load_counter,
    read_header,
    recv,
    send,
    set_codec,
    SharedUpdateRing,
    SLOT_HEADER_BYTES,
    slot_views,
    store_counter,
)


class SlotLayoutTest(unittest.TestCase):
    def test_header_fits_and_keeps_payload_aligned(self) -> None:
        self.assertLessEqual(HEADER_TIMESTAMP_END, SLOT_HEADER_BYTES)
        self.assertEqual(SLOT_HEADER_BYTES % ALIGNMENT_BYTES, 0)
        layout = build_slot_layout(0, [("x", torch.ones(3))])
        self.assertGreaterEqual(layout.specs[0].offset, SLOT_HEADER_BYTES)

    def test_dtype_groups_are_contiguous_and_round_trip(self) -> None:
        items = [
            ("a", torch.arange(5, dtype=torch.float32)),
            ("b", torch.arange(3, dtype=torch.int64)),
            ("c", torch.ones(2, 2, dtype=torch.float32)),
            ("d", torch.tensor([1.5], dtype=torch.float64)),
        ]
        layout = build_slot_layout(7, items)
        self.assertEqual([spec.key for spec in layout.specs], ["a", "b", "c", "d"])
        self.assertEqual(
            [group.spec_indices for group in layout.groups], [(0, 2), (1,), (3,)]
        )
        for group in layout.groups:
            self.assertEqual(
                layout.specs[group.spec_indices[0]].offset % ALIGNMENT_BYTES, 0
            )
        self.assertEqual(layout.total_bytes % ALIGNMENT_BYTES, 0)

        ring = SharedUpdateRing(num_slots=2, slot_bytes=layout.total_bytes)
        buffer = ring.data[1]
        torch._foreach_copy_(slot_views(buffer, layout), [t for _, t in items])
        header_ints(buffer).copy_(torch.tensor([11, layout.layout_id]))
        header_timestamp(buffer).fill_(3.25)

        self.assertEqual(read_header(buffer), (11, 7, 3.25))
        for (_, expected), actual in zip(items, slot_views(buffer, layout)):
            torch.testing.assert_close(actual, expected)
        self.assertTrue(ring.data.is_shared())


class PicklerTest(unittest.TestCase):
    def test_tensors_travel_by_value_and_keep_state_dict_metadata(self) -> None:
        state = torch.nn.Linear(3, 2).state_dict()
        reader, writer = multiprocessing.Pipe(duplex=False)
        send(writer, {"state": state, "count": 3})
        restored = recv(reader)

        self.assertEqual(restored["count"], 3)
        self.assertIn("_metadata", vars(state))
        self.assertEqual(vars(restored["state"]), vars(state))
        for key, value in state.items():
            torch.testing.assert_close(restored["state"][key], value)
            self.assertFalse(restored["state"][key].is_shared())

    def test_shared_ring_tensors_stay_shared(self) -> None:
        ring = SharedUpdateRing(num_slots=2, slot_bytes=64)
        reader, writer = multiprocessing.Pipe(duplex=False)
        send(writer, ring)
        received = recv(reader)

        ring.data[0, 0] = 7
        self.assertEqual(int(received.data[0, 0]), 7)
        store_counter(received.published, 5)
        self.assertEqual(load_counter(ring.published), 5)


class SharedUpdateRingTest(unittest.TestCase):
    def test_non_cpu_default_device_keeps_ring_on_cpu(self) -> None:
        with torch.device("meta"):
            ring = SharedUpdateRing(num_slots=2, slot_bytes=65)

        self.assertEqual(ring.slot_bytes, 128)
        for tensor in (ring.data, ring.published, ring.consumed):
            self.assertEqual(tensor.device, torch.device("cpu"))
            self.assertTrue(tensor.is_shared())
            self.assertEqual(torch.count_nonzero(tensor), 0)
        store_counter(ring.published, 7)
        self.assertEqual(load_counter(ring.published), 7)

    def test_non_cpu_default_device_keeps_replacement_counters(self) -> None:
        ring = SharedUpdateRing(num_slots=2, slot_bytes=64)
        store_counter(ring.published, 7)
        store_counter(ring.consumed, 3)
        with torch.device("meta"):
            replacement = SharedUpdateRing(
                num_slots=4,
                slot_bytes=128,
                counters=(ring.published, ring.consumed),
            )

        self.assertEqual(replacement.data.device, torch.device("cpu"))
        self.assertTrue(replacement.data.is_shared())
        self.assertIs(replacement.published, ring.published)
        self.assertIs(replacement.consumed, ring.consumed)
        self.assertEqual(load_counter(replacement.consumed), 3)


class CounterTest(unittest.TestCase):
    def test_store_then_load_shared_counter(self) -> None:
        ring = SharedUpdateRing(num_slots=2, slot_bytes=64)
        self.assertEqual(load_counter(ring.published), 0)
        store_counter(ring.published, 7)
        self.assertEqual(load_counter(ring.published), 7)
        self.assertEqual(int(ring.published[0]), 7)
        self.assertEqual(load_counter(ring.consumed), 0)


class PipeCodecTest(unittest.TestCase):
    def test_send_recv_round_trip(self) -> None:
        reader, writer = multiprocessing.Pipe(duplex=False)
        payload = {"layout": build_slot_layout(0, [("x", torch.ones(1))]), "n": 2}
        send(writer, payload)
        restored = recv(reader)
        self.assertEqual(restored["layout"], payload["layout"])
        self.assertEqual(restored["n"], 2)

    def test_set_codec_replaces_pickling(self) -> None:
        calls: list[str] = []

        def dumps(obj: Any) -> bytes:
            calls.append("dumps")
            return pickle.dumps(obj)

        def loads(data: bytes) -> Any:
            calls.append("loads")
            return pickle.loads(data)

        reader, writer = multiprocessing.Pipe(duplex=False)
        default = ipc._codec
        set_codec((dumps, loads))
        try:
            send(writer, 7)
            self.assertEqual(recv(reader), 7)
        finally:
            set_codec(default)
        self.assertEqual(calls, ["dumps", "loads"])
