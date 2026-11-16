#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch

logger: logging.Logger = logging.getLogger(__name__)


class PercentileLogger:
    """
    Used to log percentiles of a metric over a window of samples.
    """

    def __init__(
        self, metric_name: str, window_size: int = 1000, log_interval: int = 100
    ) -> None:
        """
        Args:
            metric_name: name of the metric to log.
            window_size: number of samples to keep in the window.
            log_interval: number of samples between logging events.
        """
        self.metric_name = metric_name
        self.window_size = window_size
        self.values: torch.Tensor = torch.zeros(window_size)
        self.index = 0
        self.count = 0
        self.log_interval = log_interval
        self.update_count = 0

    def add(self, value: float) -> None:
        self.values[self.index] = value
        self.index = (self.index + 1) % self.window_size
        self.count = min(self.count + 1, self.window_size)
        self.update_count += 1

        if self.update_count % self.log_interval == 0:
            self.log_percentiles()

    def log_percentiles(self) -> None:
        if self.count == 0:
            return

        active_values = self.values[: self.count]
        p50 = torch.quantile(active_values, 0.50).item()
        p90 = torch.quantile(active_values, 0.90).item()
        p99 = torch.quantile(active_values, 0.99).item()
        p999 = torch.quantile(active_values, 0.999).item()

        logger.info(
            f"{self.metric_name} percentiles: "
            f"p50={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}, p999={p999:.2f}, "
            f"samples={self.count}"
        )
