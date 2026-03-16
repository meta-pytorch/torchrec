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
    RecComputeMode,
    RecMetricDef,
    RecMetricEnum,
    RecTaskInfo,
)


# Mapping from string names to RecMetricEnum values for CLI usage
_METRIC_NAME_MAP = {e.value: e for e in RecMetricEnum}

_REC_COMPUTE_MODE_MAP: Dict[str, RecComputeMode] = {
    "unfused": RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    "fused": RecComputeMode.FUSED_TASKS_COMPUTATION,
    "fused_states": RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
}


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
        num_tasks (int): Number of metric tasks. Each task gets unique
            prediction/label/weight keys in model_out. Default is 1.
        rec_compute_mode (str): Computation mode for RecMetrics.
            "unfused" (default), "fused", or "fused_states".
    """

    enable_metrics: bool = False
    metrics: List[str] = field(default_factory=lambda: ["ne"])
    compute_interval: int = 10
    window_size: int = 10_000_000
    num_tasks: int = 1
    rec_compute_mode: str = "unfused"

    def _generate_tasks(self) -> List[RecTaskInfo]:
        if self.num_tasks == 1:
            return [
                RecTaskInfo(
                    name="DefaultTask",
                    label_name="label",
                    prediction_name="prediction",
                    weight_name="weight",
                )
            ]
        return [
            RecTaskInfo(
                name=f"task_{i}",
                label_name=f"label_{i}",
                prediction_name=f"prediction_{i}",
                weight_name=f"weight_{i}",
            )
            for i in range(self.num_tasks)
        ]

    def generate_model_output(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic per-task model output tensors for metric update."""
        model_out: Dict[str, torch.Tensor] = {}
        for task in self._generate_tasks():
            model_out[task.prediction_name] = torch.rand(batch_size, device=device)
            model_out[task.label_name] = torch.rand(batch_size, device=device)
            model_out[task.weight_name] = torch.ones(batch_size, device=device)
        return model_out

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
        """
        if not self.enable_metrics:
            return None

        if self.rec_compute_mode not in _REC_COMPUTE_MODE_MAP:
            raise ValueError(
                f"Unknown rec_compute_mode '{self.rec_compute_mode}'. "
                f"Valid options: {sorted(_REC_COMPUTE_MODE_MAP.keys())}"
            )
        compute_mode = _REC_COMPUTE_MODE_MAP[self.rec_compute_mode]

        tasks = self._generate_tasks()

        rec_metrics: Dict[RecMetricEnum, RecMetricDef] = {}
        for metric_name in self.metrics:
            if metric_name not in _METRIC_NAME_MAP:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    f"Valid options: {sorted(_METRIC_NAME_MAP.keys())}"
                )
            rec_metrics[_METRIC_NAME_MAP[metric_name]] = RecMetricDef(
                rec_tasks=tasks,
                window_size=self.window_size,
            )

        metrics_config = MetricsConfig(
            rec_tasks=tasks,
            rec_metrics=rec_metrics,
            rec_compute_mode=compute_mode,
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
