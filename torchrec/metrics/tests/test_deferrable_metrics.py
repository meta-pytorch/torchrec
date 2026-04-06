#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Mapping
from concurrent.futures import Future
from unittest.mock import patch

import torch
from torchrec.metrics.deferrable_metrics import (
    DeferrableMetrics,
    device_supports_async,
    transfer_tensors_to_cpu,
)


class TestDeferrableMetrics(unittest.TestCase):
    def setUp(self) -> None:
        DeferrableMetrics._warned = False

    def test_resolved_basic(self) -> None:
        data = {"a": 1, "b": 2}
        dm = DeferrableMetrics(data)
        self.assertTrue(dm.is_resolved())
        self.assertEqual(dm.resolve(), {"a": 1, "b": 2})

    def test_resolved_subscribe(self) -> None:
        data = {"x": 10}
        dm = DeferrableMetrics(data)
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        self.assertEqual(received, [{"x": 10}])

    def test_resolved_update_dict(self) -> None:
        dm = DeferrableMetrics({"a": 1})
        dm.update({"b": 2, "c": 3})
        self.assertEqual(dm.resolve(), {"a": 1, "b": 2, "c": 3})

    def test_resolved_update_deferrable(self) -> None:
        dm1 = DeferrableMetrics({"a": 1})
        dm2 = DeferrableMetrics({"b": 2})
        dm1.update(dm2)
        self.assertEqual(dm1.resolve(), {"a": 1, "b": 2})

    def test_future_backed_is_not_resolved(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        self.assertFalse(dm.is_resolved())

    def test_future_backed_resolve_blocks(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        f.set_result({"val": 42})
        self.assertEqual(dm.resolve(), {"val": 42})
        self.assertTrue(dm.is_resolved())

    def test_future_backed_subscribe(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        self.assertEqual(received, [])
        f.set_result({"val": 42})
        self.assertEqual(received, [{"val": 42}])
        self.assertTrue(dm.is_resolved())

    def test_future_backed_subscribe_error(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        errors: list[Exception] = []
        dm.subscribe(
            lambda d: None,
            on_error=lambda e: errors.append(e),
        )
        f.set_exception(ValueError("boom"))
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)
        self.assertEqual(str(errors[0]), "boom")

    def test_future_backed_update_deferred(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        dm.update({"extra": 99})
        self.assertFalse(dm.is_resolved())
        f.set_result({"original": 1})
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        self.assertEqual(received, [{"original": 1, "extra": 99}])

    def test_bool_always_true(self) -> None:
        self.assertTrue(bool(DeferrableMetrics({})))
        self.assertTrue(bool(DeferrableMetrics({"a": 1})))
        f: Future[dict] = Future()
        self.assertTrue(bool(DeferrableMetrics(f)))

    def test_or_pattern(self) -> None:
        dm = DeferrableMetrics({})
        result = dm or {}
        self.assertIsInstance(result, DeferrableMetrics)
        self.assertIs(result, dm)

    def test_mapping_getitem(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        self.assertEqual(dm["a"], 1)
        self.assertEqual(dm["b"], 2)
        with self.assertRaises(KeyError):
            _ = dm["nonexistent"]

    def test_mapping_iter(self) -> None:
        dm = DeferrableMetrics({"x": 10, "y": 20})
        self.assertEqual(set(dm), {"x", "y"})

    def test_mapping_len(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(dm), 3)

    def test_mapping_items_keys_values(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        self.assertEqual(dict(dm.items()), {"a": 1, "b": 2})
        self.assertEqual(set(dm.keys()), {"a", "b"})
        self.assertEqual(set(dm.values()), {1, 2})

    def test_mapping_isinstance(self) -> None:
        dm = DeferrableMetrics({"a": 1})
        self.assertIsInstance(dm, Mapping)

    def test_dict_update_from_deferrable(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        d: dict[str, int] = {"c": 3}
        d.update(dm)
        self.assertEqual(d, {"a": 1, "b": 2, "c": 3})

    def test_dict_unpacking(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        result = {**dm}
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_in_operator(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        self.assertIn("a", dm)
        self.assertNotIn("c", dm)

    def test_get_method(self) -> None:
        dm = DeferrableMetrics({"a": 1})
        self.assertEqual(dm.get("a"), 1)
        self.assertIsNone(dm.get("missing"))
        self.assertEqual(dm.get("missing", 42), 42)

    def test_warn_sync_access_on_future_backed(self) -> None:
        f: Future[dict] = Future()
        f.set_result({"a": 1})
        dm = DeferrableMetrics(f)
        with patch("torchrec.metrics.deferrable_metrics.logger") as mock_logger:
            _ = dm["a"]
            mock_logger.warning.assert_called_once()

    def test_warn_once_per_process(self) -> None:
        f1: Future[dict] = Future()
        f1.set_result({"a": 1})
        dm1 = DeferrableMetrics(f1)
        f2: Future[dict] = Future()
        f2.set_result({"b": 2})
        dm2 = DeferrableMetrics(f2)
        with patch("torchrec.metrics.deferrable_metrics.logger") as mock_logger:
            _ = dm1["a"]
            _ = dm2["b"]
            _ = list(dm1)
            _ = len(dm2)
            self.assertEqual(mock_logger.warning.call_count, 1)

    def test_no_warn_on_resolved(self) -> None:
        dm = DeferrableMetrics({"a": 1})
        with patch("torchrec.metrics.deferrable_metrics.logger") as mock_logger:
            _ = dm["a"]
            _ = list(dm)
            _ = len(dm)
            mock_logger.warning.assert_not_called()

    def test_repr(self) -> None:
        dm_resolved = DeferrableMetrics({"a": 1, "b": 2})
        self.assertEqual(repr(dm_resolved), "DeferrableMetrics(resolved, 2 keys)")
        f: Future[dict] = Future()
        dm_pending = DeferrableMetrics(f)
        self.assertEqual(repr(dm_pending), "DeferrableMetrics(pending)")

    def test_empty_resolved(self) -> None:
        dm = DeferrableMetrics({})
        self.assertTrue(dm.is_resolved())
        self.assertEqual(dm.resolve(), {})
        self.assertEqual(len(dm), 0)

    def test_update_deferrable_to_deferrable(self) -> None:
        f1: Future[dict] = Future()
        f2: Future[dict] = Future()
        dm1 = DeferrableMetrics(f1)
        dm2 = DeferrableMetrics(f2)
        dm1.update(dm2)
        f1.set_result({"a": 1})
        received_1: list[dict] = []
        dm1.subscribe(lambda d: received_1.append(d))
        self.assertEqual(received_1[0], {"a": 1})
        f2.set_result({"b": 2})
        self.assertIn("b", dm1.resolve())

    def test_update_deferrable_error_propagation(self) -> None:
        f1: Future[dict] = Future()
        f2: Future[dict] = Future()
        dm1 = DeferrableMetrics(f1)
        dm2 = DeferrableMetrics(f2)
        dm1.update(dm2)
        f1.set_result({"a": 1})
        errors: list[Exception] = []
        dm1.subscribe(lambda d: None, on_error=lambda e: errors.append(e))
        f2.set_exception(ValueError("upstream failed"))
        self.assertEqual(len(errors), 0)

    def test_update_deferrable_error_propagation_pending(self) -> None:
        f1: Future[dict] = Future()
        f2: Future[dict] = Future()
        dm1 = DeferrableMetrics(f1)
        dm2 = DeferrableMetrics(f2)
        dm1.update(dm2)
        errors: list[Exception] = []
        dm1.subscribe(lambda d: None, on_error=lambda e: errors.append(e))
        f2.set_exception(ValueError("upstream failed"))
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)

    def test_defensive_copy(self) -> None:
        original = {"a": 1}
        dm = DeferrableMetrics(original)
        original["a"] = 999
        self.assertEqual(dm.resolve()["a"], 1)

    def test_subscribe_no_on_error_future_raises(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        f.set_exception(ValueError("boom"))
        self.assertEqual(received, [])
        self.assertFalse(dm.is_resolved())

    def test_mapping_equality(self) -> None:
        dm = DeferrableMetrics({"a": 1, "b": 2})
        self.assertEqual(dm, {"a": 1, "b": 2})
        self.assertNotEqual(dm, {"a": 1})
        self.assertEqual(dm, DeferrableMetrics({"a": 1, "b": 2}))

    def test_resolve_future_exception_propagates(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        f.set_exception(ValueError("compute failed"))
        with self.assertRaises(ValueError):
            dm.resolve()
        self.assertFalse(dm.is_resolved())

    def test_future_backed_update_dict_with_dtoh(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        dm.update({"cpu_val": torch.tensor([1.0])})
        f.set_result({"original": torch.tensor([2.0])})
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        self.assertEqual(len(received), 1)
        torch.testing.assert_close(received[0]["original"], torch.tensor([2.0]))
        torch.testing.assert_close(received[0]["cpu_val"], torch.tensor([1.0]))
        self.assertEqual(received[0]["cpu_val"].device.type, "cpu")


class TransferTensorsToCpuTest(unittest.TestCase):

    def test_cpu_tensors_passthrough(self) -> None:
        tensors: dict[str, torch.Tensor] = {
            "predictions": torch.tensor([1.0, 2.0]),
            "labels": torch.tensor([0.0, 1.0]),
        }
        cpu_tensors, event = transfer_tensors_to_cpu(tensors)
        self.assertEqual(len(cpu_tensors), 2)
        self.assertIsNone(event)
        for key, tensor in cpu_tensors.items():
            self.assertEqual(tensor.device.type, "cpu")
            torch.testing.assert_close(tensor, tensors[key])

    def test_non_tensor_values_preserved(self) -> None:
        # pyre-ignore[6]
        tensors: dict[str, torch.Tensor] = {
            "predictions": torch.tensor([1.0]),
            "name": "task1",
            "count": 42,
        }
        cpu_tensors, event = transfer_tensors_to_cpu(tensors)
        self.assertEqual(cpu_tensors["name"], "task1")
        self.assertEqual(cpu_tensors["count"], 42)
        torch.testing.assert_close(cpu_tensors["predictions"], torch.tensor([1.0]))

    def test_empty_dict(self) -> None:
        cpu_tensors, event = transfer_tensors_to_cpu({})
        self.assertEqual(len(cpu_tensors), 0)
        self.assertIsNone(event)


class DeviceSupportsAsyncTest(unittest.TestCase):

    def test_cuda_device_supported(self) -> None:
        self.assertTrue(device_supports_async(torch.device("cuda")))

    def test_cuda_indexed_device_supported(self) -> None:
        self.assertTrue(device_supports_async(torch.device("cuda:1")))

    def test_cpu_device_not_supported(self) -> None:
        self.assertFalse(device_supports_async(torch.device("cpu")))
