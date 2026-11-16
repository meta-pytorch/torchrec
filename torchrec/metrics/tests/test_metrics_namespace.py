#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
import unittest

from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    MetricName,
    MetricNamespace,
    MetricPrefix,
    task_wildcard_metrics_pattern,
)


class MetricNamespaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = "abc"

    def test_compose_metric_key(self) -> None:
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.THROUGHPUT,
                self.task,
                MetricName.THROUGHPUT,
                MetricPrefix.LIFETIME,
            ),
            f"throughput-{self.task}|lifetime_throughput",
        )
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.THROUGHPUT,
                self.task,
                MetricName.THROUGHPUT,
                MetricPrefix.WINDOW,
            ),
            f"throughput-{self.task}|window_throughput",
        )
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.THROUGHPUT,
                self.task,
                MetricName.THROUGHPUT,
            ),
            f"throughput-{self.task}|throughput",
        )

    def test_task_wildcard_metric_pattern(self) -> None:
        pattern = task_wildcard_metrics_pattern(
            MetricNamespace.THROUGHPUT,
            MetricName.THROUGHPUT,
            MetricPrefix.LIFETIME,
        )
        key1 = compose_metric_key(
            MetricNamespace.THROUGHPUT,
            "t1",
            MetricName.THROUGHPUT,
            MetricPrefix.LIFETIME,
        )
        key2 = compose_metric_key(
            MetricNamespace.THROUGHPUT,
            "t2",
            MetricName.THROUGHPUT,
            MetricPrefix.LIFETIME,
        )
        key3 = compose_metric_key(
            MetricNamespace.THROUGHPUT,
            "t3",
            MetricName.THROUGHPUT,
            MetricPrefix.WINDOW,
        )
        self.assertTrue(re.search(pattern, key1))
        self.assertTrue(re.search(pattern, key2))
        self.assertFalse(re.search(pattern, key3))
