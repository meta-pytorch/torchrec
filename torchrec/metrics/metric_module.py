#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import abc
import concurrent
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh
from torch.profiler import record_function
from torchrec.metrics.accuracy import AccuracyMetric
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.auprc import AUPRCMetric
from torchrec.metrics.cali_free_ne import CaliFreeNEMetric
from torchrec.metrics.calibration import CalibrationMetric
from torchrec.metrics.calibration_with_recalibration import (
    RecalibratedCalibrationMetric,
)
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.hindsight_target_pr import HindsightTargetPRMetric
from torchrec.metrics.mae import MAEMetric
from torchrec.metrics.metrics_config import (
    BatchSizeStage,
    MetricsConfig,
    RecMetricEnum,
    RecMetricEnumBase,
    RecTaskInfo,
    StateMetricEnum,
    validate_batch_size_stages,
)
from torchrec.metrics.metrics_namespace import (
    compose_customized_metric_key,
    compose_metric_namespace,
    MetricNamespace,
)
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.multiclass_recall import MulticlassRecallMetric
from torchrec.metrics.ndcg import NDCGMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.ne_positive import NEPositiveMetric
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric
from torchrec.metrics.output import OutputMetric
from torchrec.metrics.precision import PrecisionMetric
from torchrec.metrics.precision_session import PrecisionSessionMetric
from torchrec.metrics.rauc import RAUCMetric
from torchrec.metrics.rec_metric import RecMetric, RecMetricException, RecMetricList
from torchrec.metrics.recall import RecallMetric
from torchrec.metrics.recall_session import RecallSessionMetric
from torchrec.metrics.scalar import ScalarMetric
from torchrec.metrics.segmented_ne import SegmentedNEMetric
from torchrec.metrics.serving_calibration import ServingCalibrationMetric
from torchrec.metrics.serving_ne import ServingNEMetric
from torchrec.metrics.tensor_weighted_avg import TensorWeightedAvgMetric
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.metrics.tower_qps import TowerQPSMetric
from torchrec.metrics.unweighted_ne import UnweightedNEMetric
from torchrec.metrics.weighted_avg import WeightedAvgMetric
from torchrec.metrics.xauc import XAUCMetric
from torchrec.utils.experimental import experimental


logger: logging.Logger = logging.getLogger(__name__)

REC_METRICS_MAPPING: Dict[RecMetricEnumBase, Type[RecMetric]] = {
    RecMetricEnum.NE: NEMetric,
    RecMetricEnum.NE_POSITIVE: NEPositiveMetric,
    RecMetricEnum.SEGMENTED_NE: SegmentedNEMetric,
    RecMetricEnum.RECALIBRATED_NE: RecalibratedNEMetric,
    RecMetricEnum.RECALIBRATED_CALIBRATION: RecalibratedCalibrationMetric,
    RecMetricEnum.CTR: CTRMetric,
    RecMetricEnum.CALIBRATION: CalibrationMetric,
    RecMetricEnum.AUC: AUCMetric,
    RecMetricEnum.AUPRC: AUPRCMetric,
    RecMetricEnum.RAUC: RAUCMetric,
    RecMetricEnum.MSE: MSEMetric,
    RecMetricEnum.MAE: MAEMetric,
    RecMetricEnum.MULTICLASS_RECALL: MulticlassRecallMetric,
    RecMetricEnum.WEIGHTED_AVG: WeightedAvgMetric,
    RecMetricEnum.TOWER_QPS: TowerQPSMetric,
    RecMetricEnum.RECALL_SESSION_LEVEL: RecallSessionMetric,
    RecMetricEnum.PRECISION_SESSION_LEVEL: PrecisionSessionMetric,
    RecMetricEnum.ACCURACY: AccuracyMetric,
    RecMetricEnum.NDCG: NDCGMetric,
    RecMetricEnum.XAUC: XAUCMetric,
    RecMetricEnum.SCALAR: ScalarMetric,
    RecMetricEnum.PRECISION: PrecisionMetric,
    RecMetricEnum.RECALL: RecallMetric,
    RecMetricEnum.SERVING_NE: ServingNEMetric,
    RecMetricEnum.SERVING_CALIBRATION: ServingCalibrationMetric,
    RecMetricEnum.OUTPUT: OutputMetric,
    RecMetricEnum.TENSOR_WEIGHTED_AVG: TensorWeightedAvgMetric,
    RecMetricEnum.CALI_FREE_NE: CaliFreeNEMetric,
    RecMetricEnum.UNWEIGHTED_NE: UnweightedNEMetric,
    RecMetricEnum.HINDSIGHT_TARGET_PR: HindsightTargetPRMetric,
}


T = TypeVar("T")

# Label used for emitting model metrics to the coresponding trainer publishers.
MODEL_METRIC_LABEL: str = "model"


MetricValue = Union[torch.Tensor, float]


class StateMetric(abc.ABC):
    """
    The interface of state metrics for a component (e.g., optimizer, qat).
    """

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, MetricValue]:
        pass


class RecMetricModule(nn.Module):
    r"""
    For the current recommendation models, we assume there will be three
    types of metrics, 1.) RecMetric, 2.) Throughput, 3.) StateMetric.

    RecMetric is a metric that is computed from the model outputs (labels,
    predictions, weights).

    Throughput is being a standalone type as its unique characteristic, time-based.

    StateMetric is a metric that is computed based on a model componenet
    (e.g., Optimizer) internal logic.

    Args:
        batch_size (int): batch size used by this trainer.
        world_size (int): the number of trainers.
        rec_tasks (Optional[List[RecTaskInfo]]): the information of the model tasks.
        rec_metrics (Optional[RecMetricList]): the list of the RecMetrics.
        throughput_metric (Optional[ThroughputMetric]): the ThroughputMetric.
        state_metrics (Optional[Dict[str, StateMetric]]): the dict of StateMetrics.
        compute_interval_steps (int): the intervals between two compute calls in the unit of batch number

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example:
        >>> config = dataclasses.replace(
        >>>     DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
        >>> )
        >>>
        >>> metricModule = generate_metric_module(
        >>>     metric_class=RecMetricModule,
        >>>     metrics_config=config,
        >>>     batch_size=128,
        >>>     world_size=64,
        >>>     my_rank=0,
        >>>     state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
        >>>     device=torch.device("cpu"),
        >>>     pg=dist.new_group([0]),
        >>> )
    """

    batch_size: int
    world_size: int
    rec_tasks: List[RecTaskInfo]
    rec_metrics: RecMetricList
    throughput_metric: Optional[ThroughputMetric]
    state_metrics: Dict[str, StateMetric]
    oom_count: int
    compute_count: int
    last_compute_time: float

    # TODO(chienchin): Reorganize the argument to directly accept a MetricsConfig.
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        rec_tasks: Optional[List[RecTaskInfo]] = None,
        rec_metrics: Optional[RecMetricList] = None,
        throughput_metric: Optional[ThroughputMetric] = None,
        state_metrics: Optional[Dict[str, StateMetric]] = None,
        compute_interval_steps: int = 100,
        min_compute_interval: float = 0.0,
        max_compute_interval: float = float("inf"),
    ) -> None:
        super().__init__()
        self.rec_tasks = rec_tasks if rec_tasks else []
        self.rec_metrics = rec_metrics if rec_metrics else RecMetricList([])
        self.throughput_metric = throughput_metric
        self.state_metrics = state_metrics if state_metrics else {}
        self.trained_batches: int = 0
        self.batch_size = batch_size
        self.world_size = world_size
        self.oom_count = 0
        self.compute_count = 0

        self.compute_interval_steps = compute_interval_steps
        self.min_compute_interval = min_compute_interval
        self.max_compute_interval = max_compute_interval
        if self.min_compute_interval == 0.0 and self.max_compute_interval == float(
            "inf"
        ):
            self.min_compute_interval = -1.0
            self.max_compute_interval = -1.0
        else:
            if self.max_compute_interval <= 0.0:
                raise ValueError("Max compute interval should not be smaller than 0.0.")
            if self.min_compute_interval < 0.0:
                raise ValueError("Min compute interval should not be smaller than 0.0.")
        self.register_buffer(
            "_compute_interval_steps",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )
        self.last_compute_time = -1.0

    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        r"""the internal update function to parse the model output.
        Override this function if the implementation cannot support
        the model output format.
        """
        if self.rec_metrics and self.rec_tasks:
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                self.rec_tasks, model_out, self.get_required_inputs()
            )
            if required_inputs:
                kwargs["required_inputs"] = required_inputs

            self.rec_metrics.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                **kwargs,
            )

    def update(self, model_out: Dict[str, torch.Tensor], **kwargs: Any) -> None:
        r"""update() is called per batch, usually right after forward() to
        update the local states of metrics based on the model_output.

        Throughput.update() is also called due to the implementation sliding window
        throughput.
        """
        with record_function("## RecMetricModule:update ##"):
            self._update_rec_metrics(model_out, **kwargs)
            if self.throughput_metric:
                self.throughput_metric.update()
            self.trained_batches += 1

    def _adjust_compute_interval(self) -> None:
        """
        Adjust the compute interval (in batches) based on the first two time
        elapsed between the first two compute().
        """
        if self.last_compute_time > 0 and self.min_compute_interval >= 0:
            now = time.time()
            interval = now - self.last_compute_time
            if not (self.max_compute_interval >= interval >= self.min_compute_interval):
                per_step_time = interval / self.compute_interval_steps

                assert (
                    self.max_compute_interval != float("inf")
                    or self.min_compute_interval != 0.0
                ), (
                    "The compute time interval is "
                    f"[{self.max_compute_interval}, {self.min_compute_interval}]. "
                    "Something is not correct of this range. __init__() should have "
                    "captured this earlier."
                )
                if self.max_compute_interval == float("inf"):
                    # The `per_step_time` is not perfectly measured -- each
                    # step training time can vary. Since max_compute_interval
                    # is set to infinite, adding 1.0 to the `min_compute_interval`
                    # increase the chance that the final compute interval is
                    # indeed larger than `min_compute_interval`.
                    self._compute_interval_steps[0] = int(
                        (self.min_compute_interval + 1.0) / per_step_time
                    )
                elif self.min_compute_interval == 0.0:
                    # Similar to the above if, subtracting 1.0 from
                    # `max_compute_interval` to compute `_compute_interval_steps`
                    # can increase the chance that the final compute interval
                    # is indeed smaller than `max_compute_interval`
                    offset = 0.0 if self.max_compute_interval <= 1.0 else 1.0
                    self._compute_interval_steps[0] = int(
                        (self.max_compute_interval - offset) / per_step_time
                    )
                else:
                    self._compute_interval_steps[0] = int(
                        (self.max_compute_interval + self.min_compute_interval)
                        / 2
                        / per_step_time
                    )
                dist.all_reduce(self._compute_interval_steps, op=dist.ReduceOp.MAX)
            self.compute_interval_steps = int(self._compute_interval_steps.item())
            self.min_compute_interval = -1.0
            self.max_compute_interval = -1.0
        self.last_compute_time = time.time()

    def should_compute(self) -> bool:
        return self.trained_batches % self.compute_interval_steps == 0

    def compute(self) -> Dict[str, MetricValue]:
        r"""compute() is called when the global metrics are required, usually
        right before logging the metrics results to the data sink.
        """
        self.compute_count += 1
        ret: Dict[str, MetricValue] = {}
        with record_function("## RecMetricModule:compute ##"):
            if self.rec_metrics:
                self._adjust_compute_interval()
                ret.update(self.rec_metrics.compute())
            if self.throughput_metric:
                ret.update(self.throughput_metric.compute())
            if self.state_metrics:
                for namespace, component in self.state_metrics.items():
                    ret.update(
                        {
                            f"{compose_customized_metric_key(namespace, metric_name)}": metric_value
                            for metric_name, metric_value in component.get_metrics().items()
                        }
                    )
        return ret

    def local_compute(self) -> Dict[str, MetricValue]:
        r"""local_compute() is called when per-trainer metrics are required. It's
        can be used for debugging. Currently only rec_metrics is supported.
        """
        ret: Dict[str, MetricValue] = {}
        if self.rec_metrics:
            ret.update(self.rec_metrics.local_compute())
        return ret

    def sync(self) -> None:
        self.rec_metrics.sync()

    def unsync(self) -> None:
        self.rec_metrics.unsync()

    def reset(self) -> None:
        self.rec_metrics.reset()

    def get_required_inputs(self) -> Optional[List[str]]:
        return self.rec_metrics.get_required_inputs()

    def _get_throughput_metric_states(
        self, metric: ThroughputMetric
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        states = {}
        # this doesn't use `state_dict` as some buffers are not persistent
        for name, buf in metric.named_buffers():
            states[name] = buf
        return {metric._metric_name.value: states}

    def _get_metric_states(
        self,
        metric: RecMetric,
        world_size: int,
        process_group: Union[dist.ProcessGroup, DeviceMesh],
    ) -> Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        result = defaultdict(dict)
        for task, computation in zip(metric._tasks, metric._metrics_computations):
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `items`.
            for state_name, reduction_fn in computation._reductions.items():
                tensor_or_list: Union[List[torch.Tensor], torch.Tensor] = getattr(
                    computation, state_name
                )

                if isinstance(tensor_or_list, list):
                    gathered = _all_gather_tensor_list(
                        tensor_or_list, world_size, process_group
                    )
                else:
                    gathered = torch.stack(
                        _all_gather_tensor(tensor_or_list, world_size, process_group)
                    )
                reduced = (
                    reduction_fn(gathered) if reduction_fn is not None else gathered
                )
                result[task.name][state_name] = reduced

        return result

    @experimental
    def get_pre_compute_states(
        self, pg: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None
    ) -> Dict[str, Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]]:
        """
        This function returns the states per rank for each metric to be saved. The states are are aggregated by the state defined reduction_function.
        This can be optionall disabled by setting ``reduce_metrics`` to False. The output on each rank is identical.

        Each metric has N number of tasks associated with it. This is reflected in the metric state, where the size of the tensor is
        typically (n_tasks, 1). Depending on the `RecComputeMode` the metric is in, the number of tasks can be 1 or len(tasks).

        The output of the data is defined as nested dictionary, a dict of ``metric._namespace`` each mapping to a dict of tasks and their states and associated tensors:
            metric : str -> { task : {state : tensor or list[tensor]} }

        This differs from the state dict such that the metric states are gathered to all ranks within the process group and the reduction function is
        applied to them. Typical state dict exposes just the metric states that live on the rank it's called from.

        Args:
            pg (Optional[Union[dist.ProcessGroup, DeviceMesh]]): the process group to use for all gather, defaults to WORLD process group.

        Returns:
            Dict[str, Dict[str, Dict[str, torch.Tensor]]]: the states for each metric to be saved
        """
        pg = pg if pg is not None else dist.group.WORLD
        process_group: dist.ProcessGroup = (  # pyre-ignore[9]
            pg.get_group(mesh_dim="shard") if isinstance(pg, DeviceMesh) else pg
        )

        aggregated_states = {}
        world_size = dist.get_world_size(
            process_group
        )  # Under 2D parallel context, this should be sharding world size

        for metric in self.rec_metrics.rec_metrics:
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `value`.
            aggregated_states[metric._namespace.value] = self._get_metric_states(
                # pyre-fixme[6]: For 1st argument expected `RecMetric` but got `Module`.
                metric,
                world_size,
                process_group,
            )

        # throughput metric requires special handling, since it's not a RecMetric
        throughput_metric = self.throughput_metric
        if throughput_metric is not None:
            aggregated_states[throughput_metric._namespace.value] = (
                self._get_throughput_metric_states(throughput_metric)
            )

        return aggregated_states

    @experimental
    def load_pre_compute_states(
        self,
        source: Dict[
            str, Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]
        ],
    ) -> None:
        """
        Load states from ``get_pre_compute_states``. This is called on every rank, no collectives are called in this function.

        Args:
            source (Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]): the source states to load from. This
                is the output of ``get_pre_compute_states``.

        Returns:
            None
        """
        for metric in self.rec_metrics.rec_metrics:
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `value`.
            states = source[metric._namespace.value]
            for task, metric_computation in zip(
                # pyre-fixme[6]: For 1st argument expected `Iterable[_T1]` but got
                #  `Union[Module, Tensor]`.
                # pyre-fixme[6]: For 2nd argument expected `Iterable[_T2]` but got
                #  `Union[Module, Tensor]`.
                metric._tasks,
                # pyre-fixme[6]: For 2nd argument expected `Iterable[_T2]` but got
                #  `Union[Module, Tensor]`.
                metric._metrics_computations,
            ):
                state = states[task.name]
                for attr, tensor in state.items():
                    setattr(metric_computation, attr, tensor)

        if self.throughput_metric is not None:
            states = source[self.throughput_metric._namespace.value][
                self.throughput_metric._metric_name.value  # pyre-ignore[16]
            ]
            for name, buf in self.throughput_metric.named_buffers():  # pyre-ignore[16]
                buf.copy_(states[name])

    def shutdown(self) -> None:
        logger.info("Initiating graceful shutdown...")

    def async_compute(
        self, future: concurrent.futures.Future[Dict[str, MetricValue]]
    ) -> None:
        raise RecMetricException("async_compute is not supported in RecMetricModule")


def _generate_rec_metrics(
    metrics_config: MetricsConfig,
    world_size: int,
    my_rank: int,
    batch_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> RecMetricList:
    rec_metrics = []
    for metric_enum, metric_def in metrics_config.rec_metrics.items():

        kwargs: Dict[str, Any] = {}
        if metric_def and metric_def.arguments is not None:
            kwargs = metric_def.arguments

        kwargs["enable_pt2_compile"] = metrics_config.enable_pt2_compile
        kwargs["should_clone_update_inputs"] = metrics_config.should_clone_update_inputs

        rec_tasks: List[RecTaskInfo] = []
        if metric_def.rec_tasks and metric_def.rec_task_indices:
            raise ValueError(
                "Only one of RecMetricDef.rec_tasks and RecMetricDef.rec_task_indices "
                "should be specified."
            )
        if metric_def.rec_tasks:
            rec_tasks = metric_def.rec_tasks
        elif metric_def.rec_task_indices:
            rec_tasks = [
                metrics_config.rec_tasks[idx] for idx in metric_def.rec_task_indices
            ]
        else:
            raise ValueError(
                "One of RecMetricDef.rec_tasks and RecMetricDef.rec_task_indices "
                "should be a non-empty list"
            )

        rec_metrics.append(
            REC_METRICS_MAPPING[metric_enum](
                world_size=world_size,
                my_rank=my_rank,
                batch_size=batch_size,
                tasks=rec_tasks,
                compute_mode=metrics_config.rec_compute_mode,
                window_size=metric_def.window_size,
                fused_update_limit=metrics_config.fused_update_limit,
                compute_on_all_ranks=metrics_config.compute_on_all_ranks,
                should_validate_update=metrics_config.should_validate_update,
                process_group=process_group,
                **kwargs,
            )
        )
    return RecMetricList(rec_metrics)


STATE_METRICS_NAMESPACE_MAPPING: Dict[StateMetricEnum, MetricNamespace] = {
    StateMetricEnum.OPTIMIZERS: MetricNamespace.OPTIMIZERS,
    StateMetricEnum.MODEL_CONFIGURATOR: MetricNamespace.MODEL_CONFIGURATOR,
}


def _generate_state_metrics(
    metrics_config: MetricsConfig,
    state_metrics_mapping: Dict[StateMetricEnum, StateMetric],
) -> Dict[str, StateMetric]:
    state_metrics: Dict[str, StateMetric] = {}
    for metric_enum in metrics_config.state_metrics:
        metric_namespace: Optional[MetricNamespace] = (
            STATE_METRICS_NAMESPACE_MAPPING.get(metric_enum, None)
        )
        if metric_namespace is None:
            raise ValueError(f"Unknown StateMetrics {metric_enum}")
        full_namespace = compose_metric_namespace(
            metric_namespace, str(metric_namespace)
        )
        state_metrics[full_namespace] = state_metrics_mapping[metric_enum]
    return state_metrics


def generate_metric_module(
    metric_class: Type[RecMetricModule],
    metrics_config: MetricsConfig,
    batch_size: int,
    world_size: int,
    my_rank: int,
    state_metrics_mapping: Dict[StateMetricEnum, StateMetric],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
    batch_size_stages: Optional[List[BatchSizeStage]] = None,
) -> RecMetricModule:
    rec_metrics = _generate_rec_metrics(
        metrics_config, world_size, my_rank, batch_size, process_group
    )
    """
    Batch_size_stages currently only used by ThroughputMetric to ensure total_example correct so
    different training jobs have aligned mertics.
    TODO: update metrics other than ThroughputMetric if it has dependency on batch_size
    """
    validate_batch_size_stages(batch_size_stages)

    if metrics_config.throughput_metric:
        throughput_metric = ThroughputMetric(
            batch_size=batch_size,
            world_size=world_size,
            window_seconds=metrics_config.throughput_metric.window_size,
            warmup_steps=metrics_config.throughput_metric.warmup_steps,
            batch_size_stages=batch_size_stages,
        )
    else:
        throughput_metric = None
    state_metrics = _generate_state_metrics(metrics_config, state_metrics_mapping)
    metrics = metric_class(
        batch_size=batch_size,
        world_size=world_size,
        rec_tasks=metrics_config.rec_tasks,
        rec_metrics=rec_metrics,
        throughput_metric=throughput_metric,
        state_metrics=state_metrics,
        compute_interval_steps=metrics_config.compute_interval_steps,
        min_compute_interval=metrics_config.min_compute_interval,
        max_compute_interval=metrics_config.max_compute_interval,
    )
    metrics.to(device)
    return metrics


def _all_gather_tensor(
    tensor: torch.Tensor,
    world_size: int,
    pg: Union[dist.ProcessGroup, DeviceMesh],
) -> List[torch.Tensor]:
    """All-gather a single tensor and return the gathered list."""
    out = [torch.empty_like(tensor) for _ in range(world_size)]  # pragma: no cover
    dist.all_gather(out, tensor, group=pg)
    return out


def _all_gather_tensor_list(
    tensors: List[torch.Tensor],
    world_size: int,
    pg: Union[dist.ProcessGroup, DeviceMesh],
) -> List[torch.Tensor]:
    """All-gather every tensor in a list and flatten the result."""
    gathered: List[torch.Tensor] = []  # pragma: no cover
    for t in tensors:
        gathered.extend(_all_gather_tensor(t, world_size, pg))
    return gathered
