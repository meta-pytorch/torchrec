#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import abc
import inspect
import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function
from torchmetrics import Metric
from torchrec.distributed.types import get_tensor_size_bytes
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    MetricNameBase,
    MetricNamespaceBase,
    MetricPrefix,
)
from torchrec.pt2.utils import pt2_compile_callable


RecModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class MetricComputationReport:
    name: MetricNameBase
    metric_prefix: MetricPrefix
    value: torch.Tensor
    description: Optional[str] = None


DefaultValueT = TypeVar("DefaultValueT")
ComputeIterType = Iterator[
    Tuple[RecTaskInfo, MetricNameBase, torch.Tensor, MetricPrefix, str]
]

MAX_BUFFER_COUNT = 1000


class RecMetricException(Exception):
    pass


class WindowBuffer:
    def __init__(self, max_size: int, max_buffer_count: int) -> None:
        self._max_size: int = max_size
        self._max_buffer_count: int = max_buffer_count

        self._buffers: Deque[torch.Tensor] = deque(maxlen=max_buffer_count)
        self._used_sizes: Deque[int] = deque(maxlen=max_buffer_count)
        self._window_used_size = 0

    def aggregate_state(
        self, window_state: torch.Tensor, curr_state: torch.Tensor, size: int
    ) -> None:
        def remove(window_state: torch.Tensor) -> None:
            window_state -= self._buffers.popleft()
            self._window_used_size -= self._used_sizes.popleft()

        if len(self._buffers) == self._buffers.maxlen:
            remove(window_state)

        self._buffers.append(curr_state)
        self._used_sizes.append(size)
        window_state += curr_state
        self._window_used_size += size

        while self._window_used_size > self._max_size:
            remove(window_state)

    @property
    def buffers(self) -> Deque[torch.Tensor]:
        return self._buffers


class RecMetricComputation(Metric, abc.ABC):
    r"""The internal computation class template.
    A metric implementation should overwrite `update()` and `compute()`. These two
    APIs focus on the actual mathematical meaning of the metric, without detailed
    knowledge of model output and task information.

    Args:
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        n_tasks (int): the number tasks this communication object will have to compute.
        window_size (int): the window size for the window metric.
        compute_on_all_ranks (bool): whether to compute metrics on all ranks. This is
            necessary if the non-leader rank wants to consume the metrics results.
        should_validate_update (bool): whether to check the inputs of `update()` and
            skip the update if the inputs are invalid. Invalid inputs include the case
            where all examples have 0 weights for a batch.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.
    """

    _batch_window_buffers: Optional[Dict[str, WindowBuffer]]

    def __init__(
        self,
        my_rank: int,
        batch_size: int,
        n_tasks: int,
        window_size: int,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        process_group: Optional[dist.ProcessGroup] = None,
        fused_update_limit: int = 0,
        allow_missing_label_with_zero_weight: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        metric_init_signature = inspect.signature(Metric.__init__)
        if "fuse_state_tensors" in metric_init_signature.parameters:
            kwargs["fuse_state_tensors"] = (
                True
                if compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
                else False
            )
        if "compute_mode" in metric_init_signature.parameters:
            kwargs["compute_mode"] = compute_mode
        super().__init__(
            # pyrefly: ignore[bad-keyword-argument]
            process_group=process_group,
            *args,
            **kwargs,
        )

        self._my_rank = my_rank
        self._n_tasks = n_tasks
        self._batch_size = batch_size
        self._window_size = window_size
        self._compute_on_all_ranks = compute_on_all_ranks
        self._should_validate_update = should_validate_update
        if self._window_size > 0:
            self._batch_window_buffers = {}
        else:
            self._batch_window_buffers = None
        if self._should_validate_update:
            self._add_state(
                "has_valid_update",
                torch.zeros(self._n_tasks, dtype=torch.uint8),
                add_window_state=False,
                dist_reduce_fx=lambda x: torch.any(x, dim=0).byte(),
                persistent=True,
            )
        self._compute_mode: RecComputeMode = compute_mode

    @staticmethod
    def get_window_state_name(state_name: str) -> str:
        return f"window_{state_name}"

    def get_window_state(self, state_name: str) -> torch.Tensor:
        return getattr(self, self.get_window_state_name(state_name))

    def _add_state(
        self, name: str, default: DefaultValueT, add_window_state: bool, **kwargs: Any
    ) -> None:
        """
        name (str): the name of this state. The state will be accessible
            with `self.THE_NAME_YOU_DEFINE`.
        default (DefaultValueT): the initial value of this state. The most common
            initial value is `torch.zeros(self._n_tasks, dtype=torch.float)`, but
            users need to check the math formula to decide what is the correct
            initial value for the metric. Note the `self._n_tasks` in the above
            code. As a metric may handle multiple tasks at the same time, the
            highest dimension of a state should be `self._n_tasks`.
        add_window_state (bool): when this is True, a `window_{name}` state will
            be created to record the window state information for this state.
        dist_reduce_fx (str): the reduction function when aggregating the local
            state. For example, tower_qps uses “sum” to aggregate the total
            trained examples.
        persistent (bool): set this to True if you want to save/checkpoint the
            metric and this state is required to compute the checkpointed metric.
        """
        #  param but got `DefaultValueT`.
        # pyrefly: ignore[bad-argument-type]
        super().add_state(name, default, **kwargs)
        if add_window_state:
            if self._batch_window_buffers is None:
                raise RuntimeError(
                    "Users is adding a window state while window metric is disabled."
                )
            kwargs["persistent"] = False
            window_state_name = self.get_window_state_name(name)
            # Avoid pyre error
            assert isinstance(default, torch.Tensor)
            super().add_state(window_state_name, default.detach().clone(), **kwargs)

            self._batch_window_buffers[window_state_name] = WindowBuffer(
                max_size=self._window_size,
                max_buffer_count=MAX_BUFFER_COUNT,
            )

    # disable pt2 compile for this function as it would exceed the recompilation limit
    @torch.compiler.disable
    def _aggregate_window_state(
        self, state_name: str, state: torch.Tensor, num_samples: int
    ) -> None:
        if self._batch_window_buffers is None:
            raise RuntimeError(
                "Users is adding a window state while window metric is disabled."
            )
        window_state_name = self.get_window_state_name(state_name)
        assert self._batch_window_buffers is not None
        self._batch_window_buffers[window_state_name].aggregate_state(
            getattr(self, window_state_name), curr_state=state, size=num_samples
        )

    @abc.abstractmethod
    # pyrefly: ignore[bad-override]
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:  # pragma: no cover
        pass

    @abc.abstractmethod
    def _compute(self) -> List[MetricComputationReport]:  # pragma: no cover
        pass

    def pre_compute(self) -> None:
        r"""If a metric need to do some work before `compute()`, the metric
        has to override this `pre_compute()`. One possible usage is to do
        some pre-processing of the local state before `compute()` as TorchMetric
        wraps `RecMetricComputation.compute()` and will do the global aggregation
        before `RecMetricComputation.compute()` is called.
        """
        return

    def compute(self) -> List[MetricComputationReport]:
        with record_function(f"## {self.__class__.__name__}:compute ##"):
            if self._my_rank == 0 or self._compute_on_all_ranks:
                return self._compute()
            else:
                return []

    def local_compute(self) -> List[MetricComputationReport]:
        return self._compute()

    def reset(self) -> None:
        super().reset()
        if self._batch_window_buffers is not None:
            self._batch_window_buffers = {
                name: WindowBuffer(
                    max_size=self._window_size,
                    max_buffer_count=MAX_BUFFER_COUNT,
                )
                for name in self._batch_window_buffers
            }


class RecMetric(nn.Module, abc.ABC):
    r"""The main class template to implement a recommendation metric.
    This class contains the recommendation tasks information (RecTaskInfo) and
    the actual computation object (RecMetricComputation). RecMetric processes
    all the information related to RecTaskInfo and models, and passes the required
    signals to the computation object, allowing the implementation of
    RecMetricComputation to focus on the mathematical meaning.

    A new metric that inherits RecMetric must override the following attributes
    in its own `__init__()`: `_namespace` and `_metrics_computations`. No other
    methods should be overridden.

    Args:
        world_size (int): the number of trainers.
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        tasks (List[RecTaskInfo]): the information of the model tasks.
        compute_mode (RecComputeMode): the computation mode. See RecComputeMode.
        window_size (int): the window size for the window metric.
        fused_update_limit (int): the maximum number of updates to be fused.
        compute_on_all_ranks (bool): whether to compute metrics on all ranks. This
            is necessary if the non-leader rank wants to consume global metrics result.
        should_validate_update (bool): whether to check the inputs of `update()` and
            skip the update if the inputs are invalid. Invalid inputs include the case
            where all examples have 0 weights for a batch.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.

    Example::

        ne = NEMetric(
            world_size=4,
            my_rank=0,
            batch_size=128,
            tasks=DefaultTaskInfo,
        )
    """

    _computation_class: Type[RecMetricComputation]
    _namespace: MetricNamespaceBase
    _metrics_computations: nn.ModuleList

    _tasks: List[RecTaskInfo]
    _window_size: int
    _tasks_iter: Callable[[str], ComputeIterType]
    _update_buffers: Dict[str, List[RecModelOutput]]
    _default_weights: Dict[Tuple[int, ...], torch.Tensor]

    _required_inputs: Set[str]

    PREDICTIONS: str = "predictions"
    LABELS: str = "labels"
    WEIGHTS: str = "weights"

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
        torch._C._log_api_usage_once(
            f"torchrec.metrics.rec_metric.{self.__class__.__name__}"
        )
        # TODO(stellaya): consider to inherit from TorchMetrics.Metric or
        # TorchMetrics.MetricCollection.
        if (
            compute_mode
            in [
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]
            and fused_update_limit > 0
        ):
            raise ValueError(
                "The fused tasks computation and the fused update cannot be set at the same time"
            )
        super().__init__()
        self._world_size = world_size
        self._my_rank = my_rank
        self._window_size = math.ceil(window_size / world_size)
        self._batch_size = batch_size
        self._metrics_computations = nn.ModuleList()
        self._tasks = tasks
        self._compute_mode = compute_mode
        self._fused_update_limit = fused_update_limit
        self._should_validate_update = should_validate_update
        self._default_weights = {}
        self._required_inputs = set()
        self._update_buffers = {
            self.PREDICTIONS: [],
            self.LABELS: [],
            self.WEIGHTS: [],
        }
        #  Dict[str, Any]]`.
        # pyrefly: ignore[bad-assignment]
        self.enable_pt2_compile: bool = kwargs.get("enable_pt2_compile", False)
        # we need to remove the enable_pt2_compile from kwargs to avoid Metric object being initialized with it
        if "enable_pt2_compile" in kwargs:
            del kwargs["enable_pt2_compile"]

        #  Dict[str, Any]]`.
        # pyrefly: ignore[bad-assignment]
        self._should_clone_update_inputs: bool = kwargs.get(
            "should_clone_update_inputs", False
        )
        if "should_clone_update_inputs" in kwargs:
            del kwargs["should_clone_update_inputs"]

        # Pop batch_size_stages from kwargs so it doesn't cause type conflicts
        # when subclasses pass **kwargs: Dict[str, Any] to super().__init__().
        # TowerQPSMetric declares its own explicit batch_size_stages parameter.
        kwargs.pop("batch_size_stages", None)

        if self._window_size < self._batch_size:
            raise ValueError(
                f"Local window size must be larger than batch size. Got local window size {self._window_size} and batch size {self._batch_size}."
            )

        if compute_mode in [
            RecComputeMode.FUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
        ]:
            task_per_metric = len(self._tasks)
            self._tasks_iter = self._fused_tasks_iter
        else:
            task_per_metric = 1
            self._tasks_iter = self._unfused_tasks_iter

        for task_config in (
            [self._tasks]
            if compute_mode
            in [
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]
            else self._tasks
        ):
            # pyrefly: ignore[unsupported-operation]
            kwargs["fused_update_limit"] = fused_update_limit
            # This Pyre error seems to be Pyre's bug as it can be inferred by mypy
            # according to https://github.com/python/mypy/issues/3048.
            metric_computation = self._computation_class(
                my_rank=my_rank,
                batch_size=batch_size,
                n_tasks=task_per_metric,
                window_size=self._window_size,
                compute_on_all_ranks=compute_on_all_ranks,
                should_validate_update=self._should_validate_update,
                compute_mode=compute_mode,
                process_group=process_group,
                # pyrefly: ignore[bad-argument-type]
                **{**kwargs, **self._get_task_kwargs(task_config)},
            )
            required_inputs = self._get_task_required_inputs(task_config)

            self._metrics_computations.append(metric_computation)
            self._required_inputs.update(required_inputs)

    def _get_task_kwargs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Dict[str, Any]:
        return {}

    def _get_task_required_inputs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Set[str]:
        return set()

    # TODO(stellaya): Refactor the _[fused, unfused]_tasks_iter methods and replace the
    # compute_scope str input with an enum
    def _fused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        assert len(self._metrics_computations) == 1
        # pyrefly: ignore[not-callable]
        self._metrics_computations[0].pre_compute()
        for metric_report in getattr(
            self._metrics_computations[0], compute_scope + "compute"
        )():
            # pyrefly: ignore[no-matching-overload]
            for task, metric_value, has_valid_update in zip(
                self._tasks,
                metric_report.value,
                (
                    self._metrics_computations[0].has_valid_update
                    if self._should_validate_update
                    else itertools.repeat(1)
                ),  # has_valid_update > 0 means the update is valid
            ):
                # The attribute has_valid_update is a tensor whose length equals to the
                # number of tasks. Each value in it is corresponding to whether a task
                # has valid updates or not.
                # If for a task there's no valid updates, the calculated metric_value
                # will be meaningless, so we mask it with the default value, i.e. 0.
                valid_metric_value = (
                    metric_value
                    if has_valid_update > 0
                    else torch.zeros_like(metric_value)
                )
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value, metric_report.description

    def _unfused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        """
        For each task, we generate an associated RecMetricComputation object for it.
        This would mean in the states of each RecMetricComputation object, the n_tasks dimension is 1.
        """
        for task, metric_computation in zip(self._tasks, self._metrics_computations):
            # pyrefly: ignore[not-callable]
            metric_computation.pre_compute()
            for metric_report in getattr(
                metric_computation, compute_scope + "compute"
            )():
                # The attribute has_valid_update is a tensor with only 1 value
                # corresponding to whether the task has valid updates or not.
                # If there's no valid update, the calculated metric_report.value
                # will be meaningless, so we mask it with the default value, i.e. 0.
                valid_metric_value = (
                    metric_report.value
                    if not self._should_validate_update
                    # pyrefly: ignore[bad-index]
                    or metric_computation.has_valid_update[0] > 0
                    else torch.zeros_like(metric_report.value)
                )
                # ultimately compute result comes here, and is then written to tensorboard, for fused tasks we need to know the metric prefix val and description
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value, metric_report.description

    def _fuse_update_buffers(self) -> Dict[str, RecModelOutput]:
        def fuse(outputs: List[RecModelOutput]) -> RecModelOutput:
            assert len(outputs) > 0
            if isinstance(outputs[0], torch.Tensor):
                return torch.cat(cast(List[torch.Tensor], outputs))
            else:
                task_outputs: Dict[str, List[torch.Tensor]] = defaultdict(list)
                for output in outputs:
                    assert isinstance(output, dict)
                    for task_name, tensor in output.items():
                        task_outputs[task_name].append(tensor)
                return {
                    name: torch.cat(tensors) for name, tensors in task_outputs.items()
                }

        ret: Dict[str, RecModelOutput] = {}
        for key, output_list in self._update_buffers.items():
            if len(output_list) > 0:
                ret[key] = fuse(output_list)
            else:
                assert key == self.WEIGHTS
            output_list.clear()
        return ret

    def _check_fused_update(self, force: bool) -> None:
        if self._fused_update_limit <= 0:
            return
        if len(self._update_buffers[self.PREDICTIONS]) == 0:
            return
        if (
            not force
            and len(self._update_buffers[self.PREDICTIONS]) < self._fused_update_limit
        ):
            return
        fused_arguments = self._fuse_update_buffers()
        self._update(
            predictions=fused_arguments[self.PREDICTIONS],
            labels=fused_arguments[self.LABELS],
            weights=fused_arguments.get(self.WEIGHTS, None),
        )

    def _create_default_weights(self, predictions: torch.Tensor) -> torch.Tensor:
        weights = self._default_weights.get(predictions.size(), None)
        if weights is None:
            weights = torch.ones_like(predictions)
            self._default_weights[predictions.size()] = weights
        return weights

    def _check_nonempty_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.gt(torch.count_nonzero(weights, dim=-1), 0)

    def clone_update_inputs(
        self,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> tuple[
        RecModelOutput, RecModelOutput, Optional[RecModelOutput], Dict[str, Any]
    ]:
        def clone_rec_model_output(
            rec_model_output: RecModelOutput,
        ) -> RecModelOutput:
            if isinstance(rec_model_output, torch.Tensor):
                return rec_model_output.clone()
            else:
                return {k: v.clone() for k, v in rec_model_output.items()}

        predictions = clone_rec_model_output(predictions)
        labels = clone_rec_model_output(labels)
        if weights is not None:
            weights = clone_rec_model_output(weights)

        if "required_inputs" in kwargs:
            kwargs["required_inputs"] = {
                k: v.clone() for k, v in kwargs["required_inputs"].items()
            }

        return predictions, labels, weights, kwargs

    def _update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        with torch.no_grad():
            if self._should_clone_update_inputs:
                predictions, labels, weights, kwargs = self.clone_update_inputs(
                    predictions, labels, weights, **kwargs
                )

            if self._compute_mode in [
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]:
                task_names = [task.name for task in self._tasks]

                if not isinstance(predictions, torch.Tensor):
                    predictions = torch.stack(
                        [predictions[task_name] for task_name in task_names]
                    )

                if not isinstance(labels, torch.Tensor):
                    labels = torch.stack(
                        [labels[task_name] for task_name in task_names]
                    )
                if weights is not None and not isinstance(weights, torch.Tensor):
                    weights = torch.stack(
                        [weights[task_name] for task_name in task_names]
                    )

                assert isinstance(predictions, torch.Tensor) and isinstance(
                    labels, torch.Tensor
                )

                # Metrics such as TensorWeightedAvgMetric will have tensors that we also need to stack.
                # Stack in task order: (n_tasks, batch_size)
                if "required_inputs" in kwargs:
                    target_tensors: list[torch.Tensor] = []
                    for task in self._tasks:
                        if (
                            task.tensor_name
                            and task.tensor_name in kwargs["required_inputs"]
                        ):
                            target_tensors.append(
                                kwargs["required_inputs"][task.tensor_name]
                            )

                    if target_tensors:
                        stacked_tensor = torch.stack(target_tensors)

                        # Reshape the stacked_tensor to size([len(self._tasks), self._batch_size])
                        stacked_tensor = stacked_tensor.view(len(self._tasks), -1)
                        assert isinstance(stacked_tensor, torch.Tensor)
                        kwargs["required_inputs"]["target_tensor"] = stacked_tensor

                predictions = (
                    # Reshape the predictions to size([len(self._tasks), self._batch_size])
                    predictions.view(len(self._tasks), -1)
                    if predictions.dim() == labels.dim()
                    # predictions.dim() == labels.dim() + 1 for multiclass models
                    else predictions.view(len(self._tasks), -1, predictions.size()[-1])
                )
                labels = labels.view(len(self._tasks), -1)
                if weights is None:
                    weights = self._create_default_weights(predictions)
                else:
                    assert isinstance(weights, torch.Tensor)
                    weights = weights.view(len(self._tasks), -1)
                if self._should_validate_update:
                    # has_valid_weights is a tensor of bool whose length equals to the number
                    # of tasks. Each value in it is corresponding to whether the weights
                    # are valid, i.e. are set to non-zero values for that task in this update.
                    # If has_valid_weights are Falses for all the tasks, we just ignore this
                    # update.
                    has_valid_weights = self._check_nonempty_weights(weights)
                    if torch.any(has_valid_weights):
                        # pyrefly: ignore[not-callable]
                        self._metrics_computations[0].update(
                            predictions=predictions,
                            labels=labels,
                            weights=weights,
                            **kwargs,
                        )
                        # pyrefly: ignore[not-callable]
                        self._metrics_computations[0].has_valid_update.logical_or_(
                            has_valid_weights
                        )
                else:
                    # pyrefly: ignore[not-callable]
                    self._metrics_computations[0].update(
                        predictions=predictions,
                        labels=labels,
                        weights=weights,
                        **kwargs,
                    )
            else:
                for task, metric_ in zip(self._tasks, self._metrics_computations):
                    if task.name not in predictions:
                        continue
                    #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any,
                    #  ...]]` but got `str`.
                    # pyrefly: ignore[bad-index]
                    if torch.numel(predictions[task.name]) == 0:
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        assert torch.numel(labels[task.name]) == 0
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        assert weights is None or torch.numel(weights[task.name]) == 0
                        continue
                    task_predictions = (
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        predictions[task.name].view(1, -1)
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        if predictions[task.name].dim() == labels[task.name].dim()
                        # predictions[task.name].dim() == labels[task.name].dim() + 1 for multiclass models
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        else predictions[task.name].view(
                            1,
                            -1,
                            predictions[
                                # pyrefly: ignore[bad-index]
                                task.name
                                #  List[typing.Any], int, slice, Tensor,
                                #  typing.Tuple[typing.Any, ...]]` but got `str`.
                            ].size()[-1],
                        )
                    )
                    #  List[typing.Any], int, slice, Tensor, typing.Tuple[typing.Any,
                    #  ...]]` but got `str`.
                    # pyrefly: ignore[bad-index]
                    task_labels = labels[task.name].view(1, -1)
                    if weights is None:
                        task_weights = self._create_default_weights(task_predictions)
                    else:
                        #  List[typing.Any], int, slice, Tensor,
                        #  typing.Tuple[typing.Any, ...]]` but got `str`.
                        # pyrefly: ignore[bad-index]
                        task_weights = weights[task.name].view(1, -1)
                    if self._should_validate_update:
                        # has_valid_weights is a tensor with only 1 value corresponding to
                        # whether the weights are valid, i.e. are set to non-zero values for
                        # the task in this update.
                        # If has_valid_update[0] is False, we just ignore this update.
                        has_valid_weights = self._check_nonempty_weights(task_weights)
                        if has_valid_weights[0]:
                            #  Tensor) -> Tensor, Module, Tensor]` is not a function.
                            # pyrefly: ignore[not-callable]
                            metric_.has_valid_update.logical_or_(has_valid_weights)
                        else:
                            continue
                    if "required_inputs" in kwargs:
                        # Expand scalars to match the shape of the predictions
                        kwargs["required_inputs"] = {
                            k: (
                                v.view(task_labels.size())
                                if v.numel() > 1
                                else v.expand(task_labels.size())
                            )
                            for k, v in kwargs["required_inputs"].items()
                        }
                    # pyrefly: ignore[not-callable]
                    metric_.update(
                        predictions=task_predictions,
                        labels=task_labels,
                        weights=task_weights,
                        **kwargs,
                    )

    @pt2_compile_callable
    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        with record_function(f"## {self.__class__.__name__}:update ##"):
            if self._fused_update_limit > 0:
                self._update_buffers[self.PREDICTIONS].append(predictions)
                self._update_buffers[self.LABELS].append(labels)
                if weights is not None:
                    self._update_buffers[self.WEIGHTS].append(weights)
                self._check_fused_update(force=False)
            else:
                self._update(
                    predictions=predictions, labels=labels, weights=weights, **kwargs
                )

    # The implementation of compute is very similar to local_compute, but compute overwrites
    # the abstract method compute in torchmetrics.Metric, which is wrapped by _wrap_compute
    @pt2_compile_callable
    def compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix, description in self._tasks_iter(
            ""
        ):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix, description
            )
            ret[metric_key] = metric_value
        return ret

    def local_compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix, description in self._tasks_iter(
            "local_"
        ):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix, description
            )
            ret[metric_key] = metric_value
        return ret

    def sync(self) -> None:
        for computation in self._metrics_computations:
            # pyrefly: ignore[not-callable]
            computation.sync()

    def unsync(self) -> None:
        for computation in self._metrics_computations:
            if computation._is_synced:
                # pyrefly: ignore[not-callable]
                computation.unsync()

    def reset(self) -> None:
        for computation in self._metrics_computations:
            # pyrefly: ignore[not-callable]
            computation.reset()

    def get_memory_usage(self) -> Dict[torch.Tensor, int]:
        r"""Estimates the memory of the rec metric instance's
        underlying tensors; returns the map of tensor to size
        """
        tensor_map = {}
        attributes_q = deque(self.__dict__.values())
        while attributes_q:
            attribute = attributes_q.popleft()
            if isinstance(attribute, torch.Tensor):
                tensor_map[attribute] = get_tensor_size_bytes(attribute)
            elif isinstance(attribute, WindowBuffer):
                attributes_q.extend(attribute.buffers)
            elif isinstance(attribute, Mapping):
                attributes_q.extend(attribute.values())
            elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
                attributes_q.extend(attribute)
            elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
                attributes_q.extend(attribute.__dict__.values())
        return tensor_map

    # pyrefly: ignore[bad-override]
    def state_dict(
        self,
        destination: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # We need to flush the cached output to ensure checkpointing correctness.
        self._check_fused_update(force=True)
        # pyrefly: ignore[no-matching-overload]
        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return self._metrics_computations.state_dict(
            destination=destination,
            prefix=f"{prefix}_metrics_computations.",
            keep_vars=keep_vars,
        )

    def get_required_inputs(self) -> Set[str]:
        return self._required_inputs


class RecMetricList(nn.Module):
    """
    A list module to encapulate multiple RecMetric instances and provide the
    same interfaces as RecMetric.

    Args:
        rec_metrics (List[RecMetric]: the list of the input RecMetrics.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        ne = NEMetric(
                 world_size=4,
                 my_rank=0,
                 batch_size=128,
                 tasks=DefaultTaskInfo
             )
        metrics = RecMetricList([ne])
    """

    rec_metrics: nn.ModuleList
    required_inputs: Optional[List[str]]

    def __init__(self, rec_metrics: List[RecMetric]) -> None:
        # TODO(stellaya): consider to inherit from TorchMetrics.MetricCollection.
        # The prequsite to use MetricCollection is that RecMetric inherits from
        # TorchMetrics.Metric or TorchMetrics.MetricCollection

        super().__init__()
        self.rec_metrics = nn.ModuleList(rec_metrics)
        self.required_inputs = (
            list(
                set().union(
                    *[rec_metric.get_required_inputs() for rec_metric in rec_metrics]
                )
            )
            or None
        )

    def __len__(self) -> int:
        return len(self.rec_metrics)

    def __getitem__(self, idx: int) -> nn.Module:
        return self.rec_metrics[idx]

    def get_required_inputs(self) -> Optional[List[str]]:
        return self.required_inputs

    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: RecModelOutput,
        **kwargs: Dict[str, Any],
    ) -> None:
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            metric.update(
                predictions=predictions, labels=labels, weights=weights, **kwargs
            )

    def compute(self) -> Dict[str, torch.Tensor]:
        ret = {}
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            ret.update(metric.compute())
        return ret

    def local_compute(self) -> Dict[str, torch.Tensor]:
        ret = {}
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            ret.update(metric.local_compute())
        return ret

    def sync(self) -> None:
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            metric.sync()

    def unsync(self) -> None:
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            metric.unsync()

    def reset(self) -> None:
        for metric in self.rec_metrics:
            # pyrefly: ignore[not-callable]
            metric.reset()
