#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Set, Type, Union

import torch
from torchrec.metrics.metrics_config import RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


def get_mean(value_sum: torch.Tensor, num_samples: torch.Tensor) -> torch.Tensor:
    return value_sum / num_samples


class TensorWeightedAvgMetricComputation(RecMetricComputation):
    """
    This class implements the RecMetricComputation for tensor weighted average.

    It is a sibling to WeightedAvgMetricComputation, but it computes the weighted average of a tensor
    passed in as a required input instead of the predictions tensor.

    FUSED_TASKS_COMPUTATION:
        This class requires all target tensors from tasks to be stacked together in RecMetrics._update().
        During TensorWeightedAvgMetricComputation.update(), the weighted sum and weighted num samples are
        computed per stacked tensor.
    """

    def __init__(
        self,
        *args: Any,
        tasks: List[RecTaskInfo],
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tasks: List[RecTaskInfo] = tasks

        for task in self.tasks:
            if task.tensor_name is None:
                raise RecMetricException(
                    "TensorWeightedAvgMetricComputation expects all tasks to have tensor_name, but got None."
                )

        self._add_state(
            "weighted_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "weighted_num_samples",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._description = description

        self.weighted_mask: torch.Tensor = torch.tensor(
            [task.weighted for task in self.tasks]
        ).unsqueeze(dim=-1)

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:

        target_tensor: torch.Tensor

        if "required_inputs" not in kwargs:
            raise RecMetricException(
                "TensorWeightedAvgMetricComputation expects 'required_inputs' to exist."
            )
        else:
            if len(self.tasks) > 1:
                # In FUSED mode, RecMetric._update() always creates "target_tensor" for the stacked tensor.
                # Note that RecMetric._update() only stacks if the tensor_name exists in kwargs["required_inputs"].
                target_tensor = cast(
                    torch.Tensor,
                    kwargs["required_inputs"]["target_tensor"],
                )
            elif len(self.tasks) == 1:
                # UNFUSED_TASKS_COMPUTATION
                tensor_name = self.tasks[0].tensor_name
                if tensor_name not in kwargs["required_inputs"]:
                    raise RecMetricException(
                        f"TensorWeightedAvgMetricComputation expects required_inputs to contain target tensor {self.tasks[0].tensor_name}"
                    )
                else:
                    target_tensor = cast(
                        torch.Tensor,
                        kwargs["required_inputs"][tensor_name],
                    )

        num_samples = labels.shape[0]
        weights = cast(torch.Tensor, weights)
        self.weighted_mask = self.weighted_mask.to(target_tensor.device)

        # Vectorized computation using masks
        weighted_values = torch.where(
            self.weighted_mask, target_tensor * weights, target_tensor
        )

        weighted_counts = torch.where(
            self.weighted_mask, weights, torch.ones_like(weights)
        )

        # Sum across batch dimension to Shape(n_tasks,)
        weighted_sum = weighted_values.sum(dim=-1)
        weighted_num_samples = weighted_counts.sum(dim=-1)

        # Update states
        states = {
            "weighted_sum": weighted_sum,
            "weighted_num_samples": weighted_num_samples,
        }
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.LIFETIME,
                value=get_mean(
                    cast(torch.Tensor, self.weighted_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
                description=self._description,
            ),
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.WINDOW,
                value=get_mean(
                    self.get_window_state("weighted_sum"),
                    self.get_window_state("weighted_num_samples"),
                ),
                description=self._description,
            ),
        ]


class TensorWeightedAvgMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.WEIGHTED_AVG
    _computation_class: Type[RecMetricComputation] = TensorWeightedAvgMetricComputation

    def _get_task_kwargs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Dict[str, Any]:
        all_tasks = (
            [task_config] if isinstance(task_config, RecTaskInfo) else task_config
        )
        return {
            "tasks": all_tasks,
        }

    def _get_task_required_inputs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Set[str]:
        """
        Returns the required inputs for the task.

        FUSED_TASKS_COMPUTATION:
            - Given two tasks with the same tensor_name, assume the same tensor reference
            - For a given tensor_name, assume all tasks have the same weighted flag
        """
        all_tasks = (
            [task_config] if isinstance(task_config, RecTaskInfo) else task_config
        )

        required_inputs: dict[str, bool] = {}
        for task in all_tasks:
            if task.tensor_name is not None:
                if (
                    task.tensor_name in required_inputs
                    and task.weighted is not required_inputs[task.tensor_name]
                ):
                    existing_weighted_flag = required_inputs[task.tensor_name]
                    raise RecMetricException(
                        f"This target tensor was already registered as weighted={existing_weighted_flag}. "
                        f"This target tensor cannot be re-registered with weighted={task.weighted}"
                    )
                else:
                    required_inputs[str(task.tensor_name)] = task.weighted

        return set(required_inputs.keys())
