#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch
from torch import distributed as dist
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecComputeMode,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
    RecTaskInfo,
)


class OutputMetricComputation(RecMetricComputation):
    """
    Metric that logs whatever model outputs are given in kwargs
    TODO - make this generic metric that can be used for any model output tensor
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "latest_imp",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=False,
        )
        self._add_state(
            "total_latest_imp",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=False,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        required_list = ["latest_imp", "total_latest_imp"]
        if "required_inputs" not in kwargs or not all(
            item in kwargs["required_inputs"] for item in required_list
        ):
            raise RecMetricException(
                "OutputMetricComputation requires 'latest_imp' and 'total_latest_imp' in kwargs"
            )
        states = {
            "latest_imp": kwargs["required_inputs"]["latest_imp"]
            .float()
            .mean(dim=-1, dtype=torch.double),
            "total_latest_imp": kwargs["required_inputs"]["total_latest_imp"]
            .float()
            .mean(dim=-1, dtype=torch.double),
        }

        for state_name, state_value in states.items():
            setattr(self, state_name, state_value)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.OUTPUT,
                metric_prefix=MetricPrefix.DEFAULT,
                # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                value=self.latest_imp,
                description="_latest_imp",
            ),
            MetricComputationReport(
                name=MetricName.OUTPUT,
                metric_prefix=MetricPrefix.DEFAULT,
                # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                #  `Union[Tensor, Module]`.
                value=self.total_latest_imp,
                description="_total_latest_imp",
            ),
        ]


class OutputMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.OUTPUT
    _computation_class: Type[RecMetricComputation] = OutputMetricComputation

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            process_group=process_group,
            **kwargs,
        )
        self._required_inputs.add("latest_imp")
        self._required_inputs.add("total_latest_imp")
