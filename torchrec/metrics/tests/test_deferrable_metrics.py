#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from concurrent.futures import Future

from torchrec.metrics.deferrable_metrics import DeferrableMetrics


class TestDeferrableMetrics(unittest.TestCase):
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

    def test_future_backed_resolve_raises(self) -> None:
        f: Future[dict] = Future()
        dm = DeferrableMetrics(f)
        with self.assertRaisesRegex(RuntimeError, "Cannot synchronously resolve"):
            dm.resolve()

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

    def test_not_a_mapping(self) -> None:
        dm = DeferrableMetrics({"a": 1})
        with self.assertRaisesRegex(TypeError, ""):
            _ = dm["a"]  # pyrefly: ignore
        with self.assertRaisesRegex(TypeError, ""):
            len(dm)  # pyrefly: ignore
        with self.assertRaisesRegex(TypeError, ""):
            list(dm)  # pyrefly: ignore

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
