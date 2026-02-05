#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Tests for metrics_output_util - utilities for handling PublishableMetricsOutput (dict or Future).

These tests focus on our utility functions' behavior, not on testing Python's
concurrent.futures module which is already well-tested.
"""

import unittest
from concurrent.futures import Future
from typing import Callable

import torch
from torchrec.metrics.metric_module import PublishableMetrics, PublishableMetricsFuture
from torchrec.metrics.metrics_output_util import (
    get_metrics_async,
    get_metrics_sync,
    update_metrics_output,
)


class GetMetricsAsyncTest(unittest.TestCase):
    """Tests that get_metrics_async correctly dispatches between sync/async paths."""

    def setUp(self) -> None:
        self.metrics: PublishableMetrics = {"loss": torch.tensor(0.5)}
        self.received_metrics: PublishableMetrics | None = None
        self.received_error: Exception | None = None

    def _callback(self, metrics: PublishableMetrics) -> None:
        self.received_metrics = metrics

    def _error_handler(self, error: Exception) -> None:
        self.received_error = error

    def test_synchronous_dict_path(self) -> None:
        result = get_metrics_async(self.metrics, self._callback)

        self.assertIsNone(result)
        self.assertIs(self.received_metrics, self.metrics)

    def test_synchronous_dict_returns_callback_value(self) -> None:
        result = get_metrics_async(self.metrics, lambda metrics: "success")
        self.assertEqual(result, "success")

    def test_asynchronous_future_path(self) -> None:
        future: PublishableMetricsFuture = Future()

        get_metrics_async(future, self._callback)
        self.assertIsNone(self.received_metrics)

        future.set_result(self.metrics)
        self.assertEqual(self.received_metrics, self.metrics)

    def test_error_handler_receives_callback_exceptions(self) -> None:
        """Error handler receives exceptions raised by callbacks."""
        future: PublishableMetricsFuture = Future()

        def failing_callback(metrics: PublishableMetrics) -> None:
            raise ValueError("callback failed")

        get_metrics_async(future, failing_callback, on_error=self._error_handler)
        future.set_result(self.metrics)

        self.assertIsInstance(self.received_error, ValueError)

    def test_error_handler_receives_future_exceptions(self) -> None:
        """Error handler receives exceptions from Future resolution."""
        future: PublishableMetricsFuture = Future()

        get_metrics_async(future, self._callback, on_error=self._error_handler)
        future.set_exception(RuntimeError("computation failed"))

        self.assertIsInstance(self.received_error, RuntimeError)


class GetMetricsSyncTest(unittest.TestCase):
    """Tests that get_metrics_sync correctly handles both dict and Future inputs."""

    def setUp(self) -> None:
        self.metrics: PublishableMetrics = {"loss": torch.tensor(0.5)}

    def test_dict_returns_immediately(self) -> None:
        result = get_metrics_sync(self.metrics)
        self.assertEqual(result, self.metrics)

    def test_future_blocks_until_resolved(self) -> None:
        future: PublishableMetricsFuture = Future()
        future.set_result(self.metrics)

        result = get_metrics_sync(future)
        self.assertEqual(result, self.metrics)

    def test_future_exception_propagates(self) -> None:
        """Exceptions from Future are propagated to caller."""
        future: PublishableMetricsFuture = Future()
        future.set_exception(RuntimeError("failed"))

        with self.assertRaises(RuntimeError):
            get_metrics_sync(future)


class MultipleCallbacksTest(unittest.TestCase):
    """Tests that multiple callbacks can be attached and all execute correctly."""

    def setUp(self) -> None:
        self.sample_data: PublishableMetrics = {
            "metric_a": torch.tensor(1.0),
            "metric_b": torch.tensor(2.0),
        }
        self.callback_executions: dict[str, bool] = {
            "callback_1": False,
            "callback_2": False,
            "callback_3": False,
        }
        self.extracted_value: float | None = None

    def _make_tracking_callback(
        self, name: str
    ) -> Callable[[PublishableMetrics], None]:
        """Factory for callbacks that track execution."""

        def callback(metrics: PublishableMetrics) -> None:
            self.callback_executions[name] = True

        return callback

    def _value_extraction_callback(self, metrics: PublishableMetrics) -> None:
        """Callback that extracts a value from metrics."""
        metric = metrics.get("metric_a")
        if isinstance(metric, torch.Tensor):
            self.extracted_value = metric.item()

    def test_multiple_callbacks_on_future(self) -> None:
        """Multiple callbacks attached to same Future all execute when resolved."""
        future: Future[PublishableMetrics] = Future()

        # Attach multiple callbacks to the same Future
        get_metrics_async(future, self._make_tracking_callback("callback_1"))
        get_metrics_async(future, self._make_tracking_callback("callback_2"))
        get_metrics_async(future, self._make_tracking_callback("callback_3"))
        get_metrics_async(future, self._value_extraction_callback)

        # Verify no callbacks executed yet
        self.assertEqual(
            self.callback_executions,
            {"callback_1": False, "callback_2": False, "callback_3": False},
        )
        self.assertIsNone(self.extracted_value)

        # Resolve the Future
        future.set_result(self.sample_data)

        # Verify all callbacks executed
        self.assertEqual(
            self.callback_executions,
            {"callback_1": True, "callback_2": True, "callback_3": True},
        )
        self.assertIsNotNone(self.extracted_value)
        self.assertAlmostEqual(self.extracted_value, 1.0, places=5)

    def test_multiple_callbacks_on_dict(self) -> None:
        """Multiple callbacks with dict input all execute immediately."""
        get_metrics_async(self.sample_data, self._make_tracking_callback("callback_1"))
        get_metrics_async(self.sample_data, self._make_tracking_callback("callback_2"))
        get_metrics_async(self.sample_data, self._value_extraction_callback)

        # Verify all callbacks executed immediately
        self.assertEqual(
            self.callback_executions,
            {"callback_1": True, "callback_2": True, "callback_3": False},
        )
        self.assertIsNotNone(self.extracted_value)
        self.assertAlmostEqual(self.extracted_value, 1.0, places=5)


class UpdateMetricsOutputTest(unittest.TestCase):
    """Tests for update_metrics_output chaining behavior."""

    def test_update_dict_sync(self) -> None:
        metrics: PublishableMetrics = {"loss": torch.tensor(0.5)}

        result = update_metrics_output(metrics, {"exception": "error"})

        self.assertIs(result, metrics)
        self.assertIn("exception", metrics)
        self.assertEqual(metrics["exception"], "error")

    def test_update_future(self) -> None:
        future: PublishableMetricsFuture = Future()
        metrics: PublishableMetrics = {"loss": torch.tensor(0.5)}

        result = update_metrics_output(future, {"exception": "error"})

        self.assertIs(result, future)
        future.set_result(metrics)

        self.assertIn("exception", metrics)
        self.assertDictEqual(
            metrics,
            {
                "loss": torch.tensor(0.5),
                "exception": "error",
            },
        )

    def test_update_future_then_read(self) -> None:
        future: PublishableMetricsFuture = Future()
        metrics: PublishableMetrics = {"loss": torch.tensor(0.5)}
        captured_metrics: PublishableMetrics | None = None

        # Chain: update first, then read
        update_metrics_output(future, {"exception": "error"})

        def capture_callback(m: PublishableMetrics) -> None:
            nonlocal captured_metrics
            captured_metrics = m

        get_metrics_async(future, capture_callback)
        future.set_result(metrics)

        self.assertDictEqual(
            # pyre-fixme[6]: For 1st argument expected `Mapping[Any, object]` but
            #  got `None`.
            captured_metrics,
            {"loss": torch.tensor(0.5), "exception": "error"},
        )
