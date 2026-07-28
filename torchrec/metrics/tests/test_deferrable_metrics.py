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
from typing import Any
from unittest.mock import patch

import torch
from torchrec.metrics.deferrable_metrics import (
    _get_metric_dtoh_stream,
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

    def test_future_backed_update_dict_merges(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        dm.update({"cpu_val": "extra"})
        f.set_result({"original": "data"})
        received: list[dict] = []
        dm.subscribe(lambda d: received.append(d))
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["original"], "data")
        self.assertEqual(received[0]["cpu_val"], "extra")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_update_resolved_cuda_tensors_moved_to_cpu(self) -> None:
        # Use fp32-exact values (powers of 2) to avoid precision-based assertion noise.
        dm = DeferrableMetrics({"a": torch.tensor(1.0)})
        dm.update({"loss": torch.tensor(0.5, device="cuda")})
        resolved = dm.resolve()
        self.assertEqual(resolved["loss"].device.type, "cpu")
        self.assertEqual(resolved["loss"].item(), 0.5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_update_future_backed_cuda_tensors_moved_to_cpu(self) -> None:
        f: Future[dict[str, Any]] = Future()
        dm = DeferrableMetrics(f)
        dm.update({"loss": torch.tensor(0.25, device="cuda")})
        f.set_result({"window_ne": torch.tensor(0.5)})
        resolved = dm.resolve()
        self.assertEqual(resolved["loss"].device.type, "cpu")
        self.assertEqual(resolved["loss"].item(), 0.25)
        self.assertEqual(resolved["window_ne"].item(), 0.5)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_update_bit_identical_to_per_tensor_cpu_extraction(self) -> None:
        # Reference: per-tensor .cpu() (what publish would do today).
        torch.manual_seed(0)
        gpu_dict: dict[str, torch.Tensor] = {
            f"task_{i}:loss": torch.rand((), device="cuda") for i in range(30)
        }
        expected = {k: v.cpu().item() for k, v in gpu_dict.items()}

        f: Future[dict[str, Any]] = Future()
        dm = DeferrableMetrics(f)
        dm.update(gpu_dict)
        f.set_result({})
        actual = {k: v.item() for k, v in dm.resolve().items()}

        for k in expected:
            self.assertEqual(expected[k], actual[k])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_update_does_not_mutate_caller_dict(self) -> None:
        caller_dict: dict[str, Any] = {"loss": torch.tensor(1.0, device="cuda")}
        dm = DeferrableMetrics({"a": 1})
        dm.update(caller_dict)
        # Caller's dict still has the GPU tensor — we made a defensive copy.
        self.assertEqual(caller_dict["loss"].device.type, "cuda")

    def test_update_resolved_cpu_only_unchanged(self) -> None:
        # No CUDA tensors → transfer_tensors_to_cpu short-circuits (no event).
        dm = DeferrableMetrics({"a": torch.tensor(1.0)})
        dm.update({"b": torch.tensor(2.0), "name": "task1"})
        resolved = dm.resolve()
        self.assertEqual(resolved["b"].item(), 2.0)
        self.assertEqual(resolved["name"], "task1")


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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cuda_transfer_uses_dedicated_stream(self) -> None:
        """Regression test: transfer_tensors_to_cpu must enqueue copies on the
        dedicated DtoH stream, not the caller's current (default) stream.
        Default-stream DtoH serializes with training kernels and NCCL
        collectives, causing severe regressions (validated in
        fbsource//fbcode/torchrec/distributed/benchmark/yaml/stress/zorm_stream_repro.yml
        — without this guard, NCCL SendRecv inflates 15x and probe stalls
        reach 1.3 seconds).
        """
        device = torch.device("cuda:0")
        default_stream = torch.cuda.current_stream(device)
        dtoh_stream = _get_metric_dtoh_stream(device)
        self.assertNotEqual(default_stream, dtoh_stream)

        # Block the dedicated stream with a long synthetic op BEFORE calling
        # transfer_tensors_to_cpu. The returned event must be ordered behind
        # this blocker because it was recorded on the same (dedicated) stream.
        # If a future change regressed to recording on the default stream,
        # the event would be independent of dtoh_stream's queue and
        # event.query() would return True immediately (default stream idle).
        with torch.cuda.stream(dtoh_stream):
            torch.cuda._sleep(int(1e8))  # ~100ms of GPU stall on dtoh_stream
        tensors: dict[str, Any] = {"x": torch.randn(128, device=device)}

        _cpu, event = transfer_tensors_to_cpu(tensors)

        self.assertIsNotNone(event)
        self.assertFalse(
            event.query(),
            "event must be pending behind the blocker on the dedicated stream; "
            "if event.query() is True the copy was queued on the wrong stream",
        )
        dtoh_stream.synchronize()
        self.assertTrue(event.query())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_mixed_cpu_and_cuda_tensors(self) -> None:
        """Regression: a dict mixing CPU and CUDA tensors must not crash.
        Previously the copy loop guarded only on isinstance(v, Tensor) and
        called v.record_stream (CUDA-only) on the CPU tensors too, raising
        NotImplementedError. CPU tensors must pass through untouched while the
        CUDA tensor is transferred to CPU."""
        cpu_tensor = torch.tensor([1.0, 2.0])
        tensors: dict[str, Any] = {
            "cuda_val": torch.tensor([0.5], device="cuda"),
            "cpu_val": cpu_tensor,
            "name": "task1",
        }
        cpu_tensors, event = transfer_tensors_to_cpu(tensors)
        self.assertIsNotNone(event)
        # CUDA tensor transferred to CPU.
        self.assertEqual(cpu_tensors["cuda_val"].device.type, "cpu")
        # CPU tensor passed through as the same object (not copied/pinned).
        self.assertIs(cpu_tensors["cpu_val"], cpu_tensor)
        # Non-tensor preserved.
        self.assertEqual(cpu_tensors["name"], "task1")
        if event is not None:
            event.synchronize()
        torch.testing.assert_close(cpu_tensors["cuda_val"], torch.tensor([0.5]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_cuda_transfer_destination_is_pinned(self) -> None:
        """The CPU destination must be pinned so cudaMemcpyAsync is truly
        non-blocking on the host. Without pinning, the driver stages
        through an internal pinned buffer and the host call blocks."""
        device = torch.device("cuda:0")
        tensors: dict[str, Any] = {"x": torch.randn(128, device=device)}
        cpu_tensors, _event = transfer_tensors_to_cpu(tensors)
        self.assertTrue(cpu_tensors["x"].is_pinned())


class DeviceSupportsAsyncTest(unittest.TestCase):

    def test_cuda_device_supported(self) -> None:
        self.assertTrue(device_supports_async(torch.device("cuda")))

    def test_cuda_indexed_device_supported(self) -> None:
        self.assertTrue(device_supports_async(torch.device("cuda:1")))

    def test_cpu_device_not_supported(self) -> None:
        self.assertFalse(device_supports_async(torch.device("cpu")))
