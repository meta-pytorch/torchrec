#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.metrics.metric_module import generate_metric_module, RecMetricModule
from torchrec.metrics.metrics_config import (
    MetricsConfig,
    RecMetricDef,
    RecMetricEnum,
    RecTaskInfo,
)


# Mapping from string names to RecMetricEnum values for CLI usage
_METRIC_NAME_MAP = {e.value: e for e in RecMetricEnum}


@dataclass
class RecMetricConfig:
    """
    Configuration for including RecMetrics in the benchmark loop.

    When enabled, a RecMetricModule is created and its update()/compute()
    calls are included in the benchmark timing, reflecting the overhead
    that metrics add to a real training pipeline.

    Args:
        enable_metrics (bool): Whether to include RecMetricModule
            update/compute in the benchmark loop. Default is False.
        metrics (List[str]): List of metric names to compute.
            Valid values: "ne", "auc", "calibration", "mse", "mae", etc.
            (see RecMetricEnum for all options). Default is ["ne"].
        compute_interval (int): How often to call compute() in batches.
            Default is 10 (once per benchmark iteration with default
            num_batches=10).
        window_size (int): Window size for windowed metric computation.
            Default is 10_000_000.
        task_name (str): Name of the metric task. Default is "DefaultTask".
    """

    enable_metrics: bool = False
    metrics: List[str] = field(default_factory=lambda: ["ne"])
    compute_interval: int = 10
    window_size: int = 10_000_000
    task_name: str = "DefaultTask"

    def generate_metric_module(
        self,
        batch_size: int,
        world_size: int,
        rank: int,
        device: torch.device,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Optional[RecMetricModule]:
        """
        Generate a RecMetricModule based on this configuration.

        Returns None if enable_metrics is False.

        Args:
            batch_size: Batch size for the benchmark.
            world_size: Number of distributed ranks.
            rank: Current rank.
            device: Device to place metric tensors on.
            process_group: Distributed process group for metric sync.

        Returns:
            A RecMetricModule if metrics are enabled, None otherwise.
        """
        if not self.enable_metrics:
            return None

        task = RecTaskInfo(
            name=self.task_name,
            label_name="label",
            prediction_name="prediction",
            weight_name="weight",
        )

        rec_metrics: Dict[RecMetricEnum, RecMetricDef] = {}
        for metric_name in self.metrics:
            if metric_name not in _METRIC_NAME_MAP:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    # pyrefly: ignore[no-matching-overload]
                    f"Valid options: {sorted(_METRIC_NAME_MAP.keys())}"
                )
            rec_metrics[_METRIC_NAME_MAP[metric_name]] = RecMetricDef(
                rec_tasks=[task],
                window_size=self.window_size,
            )

        metrics_config = MetricsConfig(
            rec_tasks=[task],
            rec_metrics=rec_metrics,
            compute_interval_steps=self.compute_interval,
        )

        module = generate_metric_module(
            metric_class=RecMetricModule,
            metrics_config=metrics_config,
            batch_size=batch_size,
            world_size=world_size,
            my_rank=rank,
            state_metrics_mapping={},
            device=device,
            process_group=process_group,
        )

        return module
