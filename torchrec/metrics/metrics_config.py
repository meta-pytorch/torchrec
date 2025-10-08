#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from torchrec.metrics.metrics_namespace import StrValueMixin


class RecMetricEnumBase(StrValueMixin, Enum):
    pass


class RecMetricEnum(RecMetricEnumBase):
    NE = "ne"
    NE_POSITIVE = "ne_positive"
    RECALIBRATED_NE = "recalibrated_ne"
    RECALIBRATED_CALIBRATION = "recalibrated_calibration"
    SEGMENTED_NE = "segmented_ne"
    LOG_LOSS = "log_loss"
    CTR = "ctr"
    AUC = "auc"
    AUPRC = "auprc"
    RAUC = "rauc"
    CALIBRATION = "calibration"
    MSE = "mse"
    MAE = "mae"
    MULTICLASS_RECALL = "multiclass_recall"
    RECALL_SESSION_LEVEL = "recall_session_level"
    PRECISION_SESSION_LEVEL = "precision_session_level"
    WEIGHTED_AVG = "weighted_avg"
    TOWER_QPS = "tower_qps"
    ACCURACY = "accuracy"
    NDCG = "ndcg"
    XAUC = "xauc"
    SCALAR = "scalar"
    PRECISION = "precision"
    RECALL = "recall"
    SERVING_NE = "serving_ne"
    SERVING_CALIBRATION = "serving_calibration"
    OUTPUT = "output"
    TENSOR_WEIGHTED_AVG = "tensor_weighted_avg"
    CALI_FREE_NE = "cali_free_ne"
    UNWEIGHTED_NE = "unweighted_ne"
    HINDSIGHT_TARGET_PR = "hindsight_target_pr"


@dataclass(unsafe_hash=True, eq=True)
class SessionMetricDef:
    # hyperparameters required for session level metrics
    # session_var_name: name of session tensor in the model_out
    # top_threshold: predictiones ranked in top "top_threshold" are considered as positive
    # run_ranking_of_labels: if True, labels are also ranked as predictions
    session_var_name: str
    top_threshold: Optional[int] = None
    run_ranking_of_labels: bool = False


@dataclass(unsafe_hash=True, eq=True)
class RecTaskInfo:
    name: str = "DefaultTask"
    label_name: str = "label"
    prediction_name: str = "prediction"
    weight_name: str = "weight"
    session_metric_def: Optional[SessionMetricDef] = (
        None  # used for session level metrics
    )
    is_negative_task: bool = False
    tensor_name: Optional[str] = None
    weighted: bool = True


class RecComputeMode(Enum):
    """This Enum lists the supported computation modes for RecMetrics.

    FUSED_TASKS_COMPUTATION indicates that RecMetrics will fuse the computation
        for multiple tasks of the same metric. This can be used by modules where the
        outputs of all the tasks are vectorized.
    FUSED_TASKS_AND_STATES_COMPUTATION fuse both the tasks (same as FUSED_TASKS_COMPUTATION)
        and states (e.g. calibration_num and calibration_denom for caliration) of the
        same metric. This currently only supports 1D state tensors (e.g. when all state
        tensors are of the same (n_tasks) shape).
    """

    FUSED_TASKS_COMPUTATION = 1
    UNFUSED_TASKS_COMPUTATION = 2
    FUSED_TASKS_AND_STATES_COMPUTATION = 3


_DEFAULT_WINDOW_SIZE = 10_000_000
_DEFAULT_THROUGHPUT_WINDOW_SECONDS = 100
_DEFAULT_THROUGHPUT_WARMUP_STEPS = 100


@dataclass
class RecMetricDef:
    """The dataclass that defines a RecMetric.

    Args:
        rec_tasks (List[RecTaskInfo]): this and next fields specify the RecTask
            information. ``rec_tasks`` specifies the RecTask information while
            ``rec_task_indices`` represents the indices that point to the
            RecTask information stored in the parent ``MetricsConfig``. Only one
            of the two fields should be specified.
        rec_task_indices (List[int]): see the doscstring of ``rec_tasks``.
        window_size (int): the window size for this metric. Note that this is global window size.
            The local window size is window_size / world_size, and must be larger than batch size.
        arguments (Optional[Dict[str, Any]]): any propritary arguments to be used
            by this Metric.
    """

    rec_tasks: List[RecTaskInfo] = field(default_factory=list)
    rec_task_indices: List[int] = field(default_factory=list)
    window_size: int = _DEFAULT_WINDOW_SIZE
    arguments: Optional[Dict[str, Any]] = None


class StateMetricEnum(StrValueMixin, Enum):
    OPTIMIZERS = "optimizers"
    MODEL_CONFIGURATOR = "model_configurator"


@dataclass
class ThroughputDef:
    window_size: int = _DEFAULT_THROUGHPUT_WINDOW_SECONDS
    warmup_steps: int = _DEFAULT_THROUGHPUT_WARMUP_STEPS


@dataclass
class MetricsConfig:
    """The dataclass that lists all the configurations to be used by the
    MetricModule.

    Args:
        rec_tasks (List[RecTaskInfo]): the list of RecTasks that will be shared
            by all the metrics.
        rec_metrics (Dict[RecMetricEnum, RecMetricDef]): the confiurations of
            the RecMetric objects.
        throughput_metric: (Optional[ThroughputDef]): the configurations of the ThroughputMetric
            object.
        rec_compute_mode (RecComputeMode): the computation mode for the
            RecMetric objects. This will be applied to all the RecMetric
            objects defined by ``rec_metrics``.
        fused_update_limit (int): the maximum updates that can be fused. The
            default is 0 which means no fusion. Setting this field to 1 is
            logically identical to 0.  If this field ii larger than 1,
            RecMetrics will perform the actual update every ``update()`` calls.
        state_metrics (List[StateMetricEnum]): indicates what state_metrics
            will be enabled.
        compute_interval_steps(int): computing metrics every step can be
            expsensive. This field is used to specify the computation interval
            in batch count. `should_compute()` return True if the current
            trained batch count match the setting.
        min_compute_interval(float): minimum compute interval in seconds.
            If this value is set (should be larger than 0), MetricModule will
            adjust `compute_interval_steps` after the second compute() is called.
        max_compute_interval(float): maximum compute interval in seconds.
            If this value is set (should be larger than 0), MetricModule will
            adjust `compute_interval_steps` after the second compute() is called.
        compute_on_all_ranks (bool): whether to compute rec metrics on all ranks.
            If False, only compute on rank 0.
        should_validate_update (bool): whether to check the inputs of update() and skip
            update if the inputs are invalid. Invalid inputs include the case where all
            examples have 0 weights for a batch.
        enable_pt2_compile (bool): whether to enable PT2 compilation for metrics.
        should_clone_update_inputs (bool): whether to clone the inputs of update(). This
            prevents CUDAGraph error on overwritting tensor outputs by subsequent runs.
    """

    rec_tasks: List[RecTaskInfo] = field(default_factory=list)
    rec_metrics: Dict[RecMetricEnum, RecMetricDef] = field(default_factory=dict)
    throughput_metric: Optional[ThroughputDef] = None
    rec_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    fused_update_limit: int = 0
    state_metrics: List[StateMetricEnum] = field(default_factory=list)
    compute_interval_steps: int = 100
    min_compute_interval: float = 0.0
    max_compute_interval: float = float("inf")
    compute_on_all_ranks: bool = False
    should_validate_update: bool = False
    enable_pt2_compile: bool = False
    should_clone_update_inputs: bool = False

    def __post_init__(self) -> None:
        for metric_enum, metric_def in self.rec_metrics.items():
            if metric_def.rec_task_indices:
                if self.rec_tasks is None:
                    raise ValueError(
                        f"rec_task_indices {metric_def.rec_task_indices} is specified, but rec_tasks is None for metric {metric_enum}"
                    )
                for idx in metric_def.rec_task_indices:
                    if idx >= len(self.rec_tasks):
                        raise ValueError(
                            f"rec_task_indices {idx} is out of range of {len(self.rec_tasks)} tasks for metric {metric_enum}"
                        )


DefaultTaskInfo = RecTaskInfo(
    name="DefaultTask",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)


def _create_default_metrics_config() -> MetricsConfig:
    return MetricsConfig(
        rec_tasks=[DefaultTaskInfo],
        rec_metrics={
            RecMetricEnum.NE: RecMetricDef(
                rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
            ),
        },
        throughput_metric=ThroughputDef(),
        state_metrics=[],
    )


def _create_empty_metrics_config() -> MetricsConfig:
    # Explicitly specifying the empty fields to avoid any mistakes cased by simply
    # relying on the Python default values, e.g., MetricConfig().
    return MetricsConfig(
        rec_tasks=[],
        rec_metrics={},
        throughput_metric=None,
        state_metrics=[],
    )


DefaultMetricsConfig: MetricsConfig = _create_default_metrics_config()
EmptyMetricsConfig: MetricsConfig = _create_empty_metrics_config()


@dataclass
class BatchSizeStage:
    """
    BatchSizeStage class for defining the variable batch size stage.
    For a List[BatchSizeStage], the max_iter should be in ascending order, and the last one should have max_iter=None
    Attributes
    ----------
        batch_size(int): A multiple of base_batch_size
        max_iter(int): The maximum number of iterations for the stage.
                       When previous BatchSizeStage.max_iters < iter <= max_iters, the stage is effective.
                       Max_iter is the absolute train iteration count, not the relative count within each stage
    """

    batch_size: int = 0
    max_iters: Optional[int] = 0


def validate_batch_size_stages(
    batch_size_stages: Optional[List[BatchSizeStage]],
) -> None:
    if not batch_size_stages:
        return

    if len(batch_size_stages) == 0:
        raise ValueError("Batch size stages should not be empty")

    if batch_size_stages[-1].max_iters is not None:
        raise ValueError(
            f"Batch size stages last stage should have max_iters = None, but get {batch_size_stages[-1].max_iters}"
        )
