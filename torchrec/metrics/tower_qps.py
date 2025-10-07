#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import time
from typing import Any, cast, Dict, List, Optional, OrderedDict, Type

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.utils import none_throws

from torchrec.metrics.metrics_config import BatchSizeStage, RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
    RecModelOutput,
)


WARMUP_STEPS = 100

NUM_EXAMPLES = "num_examples"
WARMUP_EXAMPLES = "warmup_examples"
TIME_LAPSE = "time_lapse"


def _compute_tower_qps(
    num_examples: torch.Tensor, time_lapse: torch.Tensor
) -> torch.Tensor:
    return torch.where(time_lapse <= 0.0, 0.0, num_examples / time_lapse).double()


def _max_reduction(state: torch.Tensor) -> torch.Tensor:
    return torch.max(state, dim=0).values


class TowerQPSMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for tower QPS.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    _previous_ts: float
    _steps: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._warmup_steps: int = kwargs.pop("warmup_steps")
        super().__init__(*args, **kwargs)
        self._add_state(
            NUM_EXAMPLES,
            torch.zeros(self._n_tasks, dtype=torch.long),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            WARMUP_EXAMPLES,
            torch.zeros(self._n_tasks, dtype=torch.long),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            TIME_LAPSE,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx=_max_reduction,
            persistent=True,
        )
        self._previous_ts = 0
        self._steps = 0

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        self._steps += 1
        num_examples_scalar = labels.shape[-1]
        num_examples = torch.tensor(num_examples_scalar, dtype=torch.long)
        self_num_examples = getattr(self, NUM_EXAMPLES)
        self_num_examples += num_examples
        ts = time.monotonic()
        if self._steps <= self._warmup_steps:
            self_warmup_examples = getattr(self, WARMUP_EXAMPLES)
            self_warmup_examples += num_examples
            if self._steps == self._warmup_steps:
                self._previous_ts = ts
        else:
            self._aggregate_window_state(
                NUM_EXAMPLES, num_examples, num_examples_scalar
            )
            time_lapse = torch.tensor(ts - self._previous_ts, dtype=torch.double)
            self_time_lapse = getattr(self, TIME_LAPSE)
            self_time_lapse += time_lapse
            self._aggregate_window_state(TIME_LAPSE, time_lapse, num_examples_scalar)
            self._previous_ts = ts

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.TOWER_QPS,
                metric_prefix=MetricPrefix.LIFETIME,
                value=_compute_tower_qps(
                    cast(torch.Tensor, self.num_examples)
                    - cast(torch.Tensor, self.warmup_examples),
                    cast(torch.Tensor, self.time_lapse),
                ),
            ),
            MetricComputationReport(
                name=MetricName.TOWER_QPS,
                metric_prefix=MetricPrefix.WINDOW,
                value=_compute_tower_qps(
                    self.get_window_state(NUM_EXAMPLES),
                    self.get_window_state(TIME_LAPSE),
                ),
            ),
            MetricComputationReport(
                name=MetricName.TOTAL_EXAMPLES,
                metric_prefix=MetricPrefix.DEFAULT,
                value=cast(torch.Tensor, self.num_examples).detach(),
            ),
        ]


class TowerQPSMetric(RecMetric):
    r"""
    TowerQPSMetric defines the tower QPS metric.
    Tower QPS's formula is training example count / time
    where training example count  = sum(examples for trainer 1, ... examples for trainer n)
    and time = max(time for trainer 1, ... time for trainer n)
    It's mostly used for cases where there's no fixed batch size
    For example for Pyper MTML models, given the same input, different tasks may have
    different numbers of examples to process

    Args:
        world_size (int): the number of trainers.
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        tasks (List[RecTaskInfo]): the information of the model tasks.
        compute_mode (RecComputeMode): the computation mode. See RecComputeMode.
        window_size (int): the window size for the window metric.
        fused_update_limit (int): the maximum number of updates to be fused.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        For world_size = 4, suppose we have 1 step after warmup
        predictions = [
            [0.8033, 0.0662, 0.7559],
            [0.1821, 0.9652, 0.4602],
            [0.8545, 0.4758, 0.2220],
            [0.1021, 0.2469, 0.7259],
        ],
        previous_ts = [278.94, 312.16, 286.96, 291.43]
        ts = [281.35, 316.45, 289.47, 295.55]

        num_examples = [3, 3, 3, 3]
        time_lapse = [2.41, 4.29, 2.51, 4.12]

        tower_qps = torch.sum(num_examples) / torch.max(time_lapse) = 2.80
    """

    _namespace: MetricNamespace = MetricNamespace.TOWER_QPS
    _computation_class: Type[RecMetricComputation] = TowerQPSMetricComputation

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        warmup_steps: int = WARMUP_STEPS,
        batch_size_stages: Optional[List[BatchSizeStage]] = None,
        **kwargs: Any,
    ) -> None:
        if fused_update_limit > 0:
            raise RecMetricException("Fused update is not supported for tower QPS")

        kwargs["warmup_steps"] = warmup_steps

        super().__init__(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            process_group=process_group,
            **kwargs,
        )

        self._batch_size = batch_size
        self._world_size = world_size
        self._batch_size_stages: Optional[List[BatchSizeStage]] = copy.deepcopy(
            batch_size_stages
        )

        if self._batch_size_stages is not None:
            self._num_batch: int = 0

        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)
        self.register_state_dict_post_hook(self.state_dict_hook)

    def update(
        self,
        *,
        predictions: Optional[RecModelOutput],
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        if self._batch_size_stages is not None:
            self._num_batch += 1
            self._batch_size = self._get_batch_size()
        with torch.no_grad():
            if self._compute_mode in [
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]:
                if not isinstance(labels, torch.Tensor):
                    try:
                        labels = torch.stack(
                            [labels[task.name] for task in self._tasks]
                        )
                    except Exception as e:
                        raise RecMetricException(
                            f"Failed to convert labels to tensor for fused computation: {e}"
                        )
                labels = labels.view(-1, self._batch_size)
                if self._should_validate_update:
                    # Set the default value to be all True. When weights is None, it's considered
                    # to be a valid input, and we'll use the default value
                    has_valid_weights = torch.ones(
                        len(self._tasks),
                        dtype=torch.bool,
                        device=self._metrics_computations[0].has_valid_update.device,
                    )
                    if weights is not None:
                        if not isinstance(weights, torch.Tensor):
                            try:
                                weights = torch.stack(
                                    [weights[task.name] for task in self._tasks]
                                )
                            except Exception as e:
                                raise RecMetricException(
                                    f"Failed to convert weights to tensor for fused computation: {e}"
                                )
                        has_valid_weights = torch.gt(
                            torch.count_nonzero(
                                weights.view(-1, self._batch_size), dim=-1
                            ),
                            0,
                        )

                    if torch.any(has_valid_weights):
                        self._metrics_computations[0].update(
                            predictions=None, labels=labels, weights=None
                        )
                        self._metrics_computations[0].has_valid_update.logical_or_(
                            has_valid_weights
                        )
                else:
                    self._metrics_computations[0].update(
                        predictions=None, labels=labels, weights=None
                    )
            else:
                for task, metric_ in zip(self._tasks, self._metrics_computations):
                    if task.name not in labels:
                        continue
                    # pyre-fixme[6]: For 1st argument expected `Union[None,
                    #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any,
                    #  ...]]` but got `str`.
                    task_labels = labels[task.name].view(1, -1)
                    if self._should_validate_update:
                        has_valid_weights = torch.ones(
                            1,
                            dtype=torch.bool,
                            # pyre-fixme[6]: For 3rd argument expected `Union[None,
                            #  int, str, device]` but got `Union[device, Tensor,
                            #  Module]`.
                            device=metric_.has_valid_update.device,
                        )
                        if weights is not None and task.name in weights:
                            has_valid_weights = torch.gt(
                                torch.count_nonzero(
                                    # pyre-fixme[6]: For 1st argument expected
                                    #  `Union[None, List[typing.Any], int, slice,
                                    #  Tensor, typing.Tuple[typing.Any, ...]]` but got
                                    #  `str`.
                                    weights[task.name].view(1, -1),
                                    dim=-1,
                                ),
                                0,
                            )
                        if has_valid_weights[0]:
                            # pyre-fixme[29]: `Union[(self: TensorBase, other:
                            #  Tensor) -> Tensor, Tensor, Module]` is not a function.
                            metric_.has_valid_update.logical_or_(has_valid_weights)
                        else:
                            continue
                    # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                    metric_.update(
                        predictions=None,
                        labels=task_labels,
                        weights=None,
                    )

    def _get_batch_size(self) -> int:
        if not self._batch_size_stages:
            return self._batch_size

        batch_size_stages = none_throws(self._batch_size_stages)
        while self._batch_size_stages:
            stage = self._batch_size_stages[0]
            if stage.max_iters is None:
                assert len(batch_size_stages) == 1
                return stage.batch_size
            if stage.max_iters < self._num_batch:
                batch_size_stages.pop(0)
                continue
            return stage.batch_size
        raise AssertionError("Unreachable, batch_size_stages should always has 1 item")

    @staticmethod
    def state_dict_hook(
        module: nn.Module,
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> None:
        """
        The state dict hook and load state dict hook exist to ensure we load num_batch for a metric with
        batch_size_stages set. The reason we do this apporach as opposted to saving num_batch as a buffer
        is in some cases we are accessing the value from a CPU workload where the tensors are on GPU. This
        incurs a device to head call, which is expensive and blocking.
        """
        if module._batch_size_stages is not None:
            num_batch_key = f"{prefix}num_batch"
            state_dict[num_batch_key] = torch.tensor(
                module._num_batch, dtype=torch.long
            )

    def load_state_dict_hook(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        key = f"{prefix}num_batch"
        if key in state_dict and self._batch_size_stages is not None:
            num_batch_tensor = state_dict.pop(key)
            self._num_batch = int(num_batch_tensor.item())
