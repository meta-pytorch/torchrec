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
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    cast,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh
from torch.profiler import record_function

try:
    # This is a safety measure against torch package issues for when
    # Torchrec is included in the inference side model code. We should
    # remove this once we are sure all model side packages have the required
    # dependencies
    from torchrec.distributed.logging_handlers import (
        EventLoggingHandler,
        TorchrecComponent,
    )
except Exception:
    torch._C._log_api_usage_once(
        "torchrec.metrics.metric_module.import_failure.logging_handlers"
    )

    from enum import Enum as _Enum
    from typing import Callable, TYPE_CHECKING

    if TYPE_CHECKING:
        from torchrec.distributed.logging_handlers import (
            EventLoggingHandler,
            TorchrecComponent,
        )
    else:

        class TorchrecComponent(_Enum):
            REC_METRICS = "rec_metrics"

        class EventLoggingHandler:
            @staticmethod
            def event_logger(*args: object, **kwargs: object) -> Callable:
                def decorator(func: Callable) -> Callable:
                    return func

                return decorator

            @staticmethod
            def log_event(*args: object, **kwargs: object) -> None:
                pass


from torchrec.metrics.accuracy import AccuracyMetric
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.auprc import AUPRCMetric
from torchrec.metrics.average import AverageMetric
from torchrec.metrics.cali_free_ne import CaliFreeNEMetric
from torchrec.metrics.calibration import CalibrationMetric
from torchrec.metrics.calibration_with_recalibration import (
    RecalibratedCalibrationMetric,
)
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.deferrable_metrics import DeferrableMetrics
from torchrec.metrics.hindsight_target_pr import HindsightTargetPRMetric
from torchrec.metrics.mae import MAEMetric
from torchrec.metrics.metrics_config import (
    BatchSizeStage,
    LOSS_DENOM_SUFFIX,
    LossAggregation,
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
from torchrec.metrics.multi_label_precision import MultiLabelPrecisionMetric
from torchrec.metrics.multiclass_recall import MulticlassRecallMetric
from torchrec.metrics.ndcg import NDCGMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.ne_positive import NEPositiveMetric
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric
from torchrec.metrics.nmse import NMSEMetric
from torchrec.metrics.num_missing_labels import NumMissingLabelsMetric
from torchrec.metrics.num_positive_samples import NumPositiveSamplesMetric
from torchrec.metrics.output import OutputMetric
from torchrec.metrics.precision import PrecisionMetric
from torchrec.metrics.precision_session import PrecisionSessionMetric
from torchrec.metrics.rauc import RAUCMetric
from torchrec.metrics.rec_metric import (
    RecMetric,
    RecMetricException,
    RecMetricList,
    RecMetricValidationError,
)
from torchrec.metrics.recall import RecallMetric
from torchrec.metrics.recall_session import RecallSessionMetric
from torchrec.metrics.scalar import ScalarMetric
from torchrec.metrics.segmented_ne import SegmentedNEMetric
from torchrec.metrics.serving_calibration import ServingCalibrationMetric
from torchrec.metrics.serving_ne import ServingNEMetric
from torchrec.metrics.session_pairwise_auc import SessionPairwiseAUCMetric
from torchrec.metrics.sum_weights import SumWeightsMetric
from torchrec.metrics.tensor_weighted_avg import TensorWeightedAvgMetric
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.metrics.tower_qps import TowerQPSMetric
from torchrec.metrics.unweighted_ne import UnweightedNEMetric
from torchrec.metrics.weighted_avg import WeightedAvgMetric
from torchrec.metrics.weighted_sum_predictions import WeightedSumPredictionsMetric
from torchrec.metrics.xauc import XAUCMetric

logger: logging.Logger = logging.getLogger(__name__)


class LossAggregationScope(Enum):
    """What span a published ``:loss`` value covers.

    The policy this module applies, not a description of the trainer: three sites read it
    -- accumulate across the span, publish the recombined value, and decide which loss keys
    the caller recombines rather than the worker.
    """

    # One value per reader batch, published as it arrives. The pre-existing path.
    PER_READER_BATCH = "per_reader_batch"
    # Values accumulated across the reader batches that feed one optimizer step and
    # published once, recombined, when that step completes.
    PER_OPTIMIZER_STEP = "per_optimizer_step"


@dataclass(frozen=True)
class _InputTensorSummary:
    dtype: str
    shape: tuple[int, ...]
    numel: int
    nan_count: int
    positive_infinity_count: int
    negative_infinity_count: int
    mixed_infinity_count: int


@dataclass(frozen=True)
class _InvalidInputRecord:
    metric_name: str
    namespace: str
    task_name: str
    tensor_kind: str
    allow_nan: bool
    summary: _InputTensorSummary


def _summarize_input_tensor(tensor: torch.Tensor) -> _InputTensorSummary:
    """Synchronously summarize one tensor for opt-in diagnostics."""
    nan_count = 0
    positive_infinity_count = 0
    negative_infinity_count = 0
    mixed_infinity_count = 0
    if tensor.is_floating_point():
        counts = torch.stack(
            [
                torch.isnan(tensor).sum(),
                torch.isposinf(tensor).sum(),
                torch.isneginf(tensor).sum(),
            ]
        ).tolist()
        nan_count = int(counts[0])
        positive_infinity_count = int(counts[1])
        negative_infinity_count = int(counts[2])
    elif tensor.is_complex():
        nan_mask = torch.isnan(tensor)
        positive_infinity_mask = torch.isposinf(tensor.real) | torch.isposinf(
            tensor.imag
        )
        negative_infinity_mask = torch.isneginf(tensor.real) | torch.isneginf(
            tensor.imag
        )
        non_nan_mask = ~nan_mask
        counts = torch.stack(
            [
                nan_mask.sum(),
                (positive_infinity_mask & ~negative_infinity_mask & non_nan_mask).sum(),
                (negative_infinity_mask & ~positive_infinity_mask & non_nan_mask).sum(),
                (positive_infinity_mask & negative_infinity_mask & non_nan_mask).sum(),
            ]
        ).tolist()
        nan_count = int(counts[0])
        positive_infinity_count = int(counts[1])
        negative_infinity_count = int(counts[2])
        mixed_infinity_count = int(counts[3])

    return _InputTensorSummary(
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        numel=tensor.numel(),
        nan_count=nan_count,
        positive_infinity_count=positive_infinity_count,
        negative_infinity_count=negative_infinity_count,
        mixed_infinity_count=mixed_infinity_count,
    )


def _format_input_summary(rank: int, record: _InvalidInputRecord) -> str:
    summary = record.summary
    return (
        f"rank={rank} {record.tensor_kind} shape={list(summary.shape)} "
        f"dtype={summary.dtype} numel={summary.numel} nan={summary.nan_count} "
        f"+inf={summary.positive_infinity_count} "
        f"-inf={summary.negative_infinity_count} "
        f"mixed_inf={summary.mixed_infinity_count}"
    )


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
    RecMetricEnum.NMSE: NMSEMetric,
    RecMetricEnum.AVERAGE: AverageMetric,
    RecMetricEnum.MULTI_LABEL_PRECISION: MultiLabelPrecisionMetric,
    RecMetricEnum.NUM_MISSING_LABELS: NumMissingLabelsMetric,
    RecMetricEnum.WEIGHTED_SUM_PREDICTIONS: WeightedSumPredictionsMetric,
    RecMetricEnum.NUM_POSITIVE_SAMPLES: NumPositiveSamplesMetric,
    RecMetricEnum.SUM_WEIGHTS: SumWeightsMetric,
    # Append new metrics so the indices of existing metrics remain stable in
    # RecMetricModule checkpoints built from REC_METRICS_MAPPING order.
    RecMetricEnum.SESSION_PAIRWISE_AUC: SessionPairwiseAUCMetric,
}


T = TypeVar("T")

# Label used for emitting model metrics to the corresponding trainer publishers.
MODEL_METRIC_LABEL: str = "model"


MetricValue = Union[torch.Tensor, float]
MetricState = Union[torch.Tensor, List[torch.Tensor]]
PreComputeStates = Dict[str, Dict[str, Dict[str, MetricState]]]
MetricsResult = Dict[str, MetricValue]
# pyrefly: ignore[implicit-import]
MetricsFuture = concurrent.futures.Future[MetricsResult]
MetricsOutput = Union[MetricsResult, MetricsFuture, DeferrableMetrics]

# Looser types for publishing metrics with additional metadata
PublishableMetrics = Dict[str, Any]
# pyrefly: ignore[implicit-import]
PublishableMetricsFuture = concurrent.futures.Future[PublishableMetrics]
PublishableMetricsOutput = Union[
    PublishableMetrics, PublishableMetricsFuture, DeferrableMetrics
]


class StateMetric(abc.ABC):
    """
    The interface of state metrics for a component (e.g., optimizer, qat).
    """

    @abc.abstractmethod
    def get_metrics(self) -> MetricsResult:
        pass


class _StagedLossDelta(NamedTuple):
    """The prospective result of accumulating one micro-batch's loss keys.

    Every field is the accumulator's fully-resolved REPLACEMENT value (not an increment),
    so applying it is a plain assignment with nothing left to compute and nothing that can
    raise. Why the staging split exists: ``_LossAccumulator``.
    """

    sums: Dict[str, torch.Tensor]
    key_counts: Dict[str, int]
    denom_sums: Dict[str, torch.Tensor]
    ratio_keys: Set[str]
    # Keys whose "no denominator declared" warning has not been emitted yet.
    warn_keys: Set[str]
    # Whether this micro-batch carried any loss key at all (drives the call count).
    saw_loss: bool


class _LossAccumulator:
    """One optimizer step's per-key loss accumulation, from staging to recombination.

    Six pieces of mutable state plus the per-key aggregation overrides, held together
    because they are written by ``commit``, cleared by ``reset`` and read by
    ``reduced_losses`` as a unit -- and because every one of them has to be restored by
    hand after an unpickle that skipped ``__init__``.

    The staging/committing split is the contract, not an implementation detail. The
    accumulation rules are validated per key while walking ``model_out``, so a key that
    violates the denominator contract would otherwise raise after earlier keys had
    already been folded in, leaving the accumulators holding half a micro-batch.
    Separating the fallible walk from the infallible apply makes the accumulation
    all-or-nothing, and lets a caller that must stay rank-symmetric decide what to do
    with the failure instead of inheriting a partial mutation.
    """

    def __init__(
        self, aggregation: Optional[Dict[str, LossAggregation]] = None
    ) -> None:
        self.sums: Dict[str, torch.Tensor] = {}
        # Calls that saw *any* loss key. Read only as the per-key fallback in _reduce.
        self.count: int = 0
        # Per-KEY contribution count -- the divisors reduction actually uses. Counted per
        # key rather than per call because a key emitted by only j of the K micro-batches
        # must be divided by j, not by K.
        self.key_counts: Dict[str, int] = {}
        # Sum of per-micro-batch denominators, for producer-declared ratio keys.
        self.denom_sums: Dict[str, torch.Tensor] = {}
        # Keys that took the ratio path this step. Tracked separately from the override
        # map because the ratio kind is declared by the PRODUCER (a denominator was
        # emitted), not by config.
        self.ratio_keys: Set[str] = set()
        self.aggregation: Dict[str, LossAggregation] = (
            dict(aggregation) if aggregation else {}
        )
        self._warned_keys: Set[str] = set()

    def set_aggregation(
        self, aggregation: Optional[Dict[str, LossAggregation]]
    ) -> None:
        self.aggregation = dict(aggregation) if aggregation else {}

    def reset(self) -> None:
        self.sums.clear()
        self.count = 0
        self.key_counts.clear()
        self.denom_sums.clear()
        self.ratio_keys.clear()

    def accumulate(self, model_out: Dict[str, torch.Tensor]) -> None:
        """Stage and commit in one call, for callers with nothing to do on failure."""
        self.commit(self.stage(model_out))

    def stage(self, model_out: Dict[str, torch.Tensor]) -> _StagedLossDelta:
        """Validate and compute this micro-batch's delta WITHOUT mutating state.

        Every branch reads committed state and writes only into the returned delta, so
        it is safe to call and discard. Raises ``RecMetricException`` if the producer
        violates the denominator contract.
        """
        sums: Dict[str, torch.Tensor] = {}
        key_counts: Dict[str, int] = {}
        denom_sums: Dict[str, torch.Tensor] = {}
        ratio_keys: Set[str] = set()
        warn_keys: Set[str] = set()
        has_loss = False
        for k, v in model_out.items():
            if not (k.endswith(":loss") or k == "loss"):
                continue
            has_loss = True
            override = self.aggregation.get(k)
            denom = model_out.get(f"{k}{LOSS_DENOM_SUFFIX}")

            if override is LossAggregation.MERGEABLE_RATIO and denom is None:
                # Fail closed: the override asserts that this producer emits a
                # denominator, so a call-count mean would apply the wrong divisor.
                raise RecMetricException(
                    f"Loss key '{k}' is declared MERGEABLE_RATIO but the micro-batch "
                    f"emitted no '{k}{LOSS_DENOM_SUFFIX}'. A MERGEABLE_RATIO producer "
                    "must emit its effective guarded denominator on every micro-batch."
                )

            contribution = v.detach()
            use_ratio = denom is not None and override not in (
                LossAggregation.NON_MERGEABLE,
                LossAggregation.SUM,
            )
            if use_ratio != (k in self.ratio_keys) and k in self.sums:
                # The producer emitted a denominator for some micro-batches of this
                # step but not others, so `sums[k]` would mix `l*d` terms with raw `l`
                # terms and no divisor is correct for the result.
                raise RecMetricException(
                    f"Loss key '{k}' emitted '{k}{LOSS_DENOM_SUFFIX}' on some "
                    "micro-batches of an optimizer step but not others. A producer must "
                    "emit its denominator on every micro-batch or on none."
                )
            if use_ratio:
                # Guarded by `denom is not None` inside `use_ratio`.
                # pyrefly: ignore[missing-attribute]
                denom = denom.detach()
                contribution = contribution * denom
                if k in self.denom_sums:
                    denom_sums[k] = self.denom_sums[k] + denom
                else:
                    denom_sums[k] = denom.clone()
                ratio_keys.add(k)
            elif override is None and k not in self._warned_keys:
                warn_keys.add(k)

            if k in self.sums:
                sums[k] = self.sums[k] + contribution
                key_counts[k] = self.key_counts[k] + 1
            else:
                sums[k] = contribution.clone()
                key_counts[k] = 1

        return _StagedLossDelta(
            sums=sums,
            key_counts=key_counts,
            denom_sums=denom_sums,
            ratio_keys=ratio_keys,
            warn_keys=warn_keys,
            saw_loss=has_loss,
        )

    def commit(self, delta: _StagedLossDelta) -> None:
        """Apply a staged delta. Contains no fallible operation by construction."""
        for k in delta.warn_keys:
            self._warned_keys.add(k)
            logger.warning(
                f"Loss key '{k}' emitted no '{LOSS_DENOM_SUFFIX}' companion, so "
                "under gradient accumulation it is reported as an unweighted mean "
                "over micro-batches. That is today's behaviour, but it is only exact "
                "when the per-micro-batch denominators are equal. Have the loss "
                "module emit its effective guarded denominator, or declare the key "
                "in MetricsConfig.loss_aggregation to acknowledge the approximation."
            )
        self.sums.update(delta.sums)
        self.key_counts.update(delta.key_counts)
        self.denom_sums.update(delta.denom_sums)
        self.ratio_keys.update(delta.ratio_keys)
        if delta.saw_loss:
            self.count += 1

    def reduced_losses(self) -> Dict[str, torch.Tensor]:
        """Every accumulated key, recombined over this step's micro-batches.

        For ``LossAggregation.SUM`` the returned tensor IS the accumulator, so an in-place
        write on either side is visible to the other. Clone to own it.
        """
        return {key: self._reduce(key, value) for key, value in self.sums.items()}

    def _reduce(self, key: str, accumulated: torch.Tensor) -> torch.Tensor:
        if self.aggregation.get(key) is LossAggregation.SUM:
            return accumulated
        # `.get` rather than `[]`: commit writes `sums` and `key_counts` together, so
        # this only keeps a future divergence from becoming a KeyError.
        count = self.key_counts.get(key, self.count)
        if key in self.ratio_keys:
            denom = self.denom_sums[key]
            # Sync-free 0/0 guard: sum(d) == 0 only when every micro was empty, and each
            # l_i was 0 then, so only the division needs guarding. `torch.where`, not
            # `clamp(min=1)` -- the global-denominator branch publishes a MEAN count that
            # can legitimately be a fraction below 1, which clamping would rescale.
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            return accumulated / safe_denom
        return accumulated / count if count else accumulated


class RecMetricModule(nn.Module):
    r"""
    For the current recommendation models, we assume there will be three
    types of metrics, 1.) RecMetric, 2.) Throughput, 3.) StateMetric.

    RecMetric is a metric that is computed from the model outputs (labels,
    predictions, weights).

    Throughput is being a standalone type as its unique characteristic, time-based.

    StateMetric is a metric that is computed based on a model componenet
    (e.g., Optimizer) internal logic.

    **Optimizer step** is used here in one sense only: one application of the optimizer to
    the model weights. A trainer may feed it several reader batches -- accumulating their
    gradients first -- in which case one optimizer step spans several batches. Where a
    quantity is per reader batch this docstring says *reader batch*, never *step*.

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
    _compute_interval_steps: torch.Tensor

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
        loss_aggregation: Optional[Dict[str, LossAggregation]] = None,
    ) -> None:
        super().__init__()
        self.rec_tasks = rec_tasks if rec_tasks else []
        # pyrefly: ignore[not-callable]
        self.rec_metrics = rec_metrics if rec_metrics else RecMetricList([])
        self.throughput_metric = throughput_metric
        self.state_metrics = state_metrics if state_metrics else {}
        self.trained_batches: int = 0
        self.batch_size = batch_size
        self.world_size = world_size
        self.oom_count = 0
        self.compute_count = 0
        self._debug_mode: bool = False
        self._debug_rank: int = 0
        self._loss_acc: _LossAccumulator = _LossAccumulator(loss_aggregation)
        # Span each published ``:loss`` value covers. Armed from config by
        # set_under_micro_batching(), never by call history, so every model that takes one
        # optimizer step per reader batch keeps the pre-existing publication path.
        self._loss_aggregation_scope: LossAggregationScope = (
            LossAggregationScope.PER_READER_BATCH
        )

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

        self._register_load_state_dict_pre_hook(self.load_state_dict_hook)

    def _configure_debug_mode(
        self,
        debug_mode: bool,
        my_rank: int,
    ) -> None:
        self._debug_mode = debug_mode
        self._debug_rank = my_rank

    def _build_invalid_input_records(
        self,
        labels: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
    ) -> List[_InvalidInputRecord]:
        records: List[_InvalidInputRecord] = []
        summaries: Dict[tuple[str, str], Optional[_InputTensorSummary]] = {}
        inputs = (
            (RecMetric.LABELS, labels),
            (RecMetric.PREDICTIONS, predictions),
            (RecMetric.WEIGHTS, weights),
        )
        for metric_module in self.rec_metrics.rec_metrics:
            metric = cast(RecMetric, metric_module)
            namespace = str(metric._namespace.value)
            for task in metric._tasks:
                for tensor_kind, task_tensors in inputs:
                    if tensor_kind in metric.ignored_input_values:
                        continue
                    key = (task.name, tensor_kind)
                    if key not in summaries:
                        tensor = task_tensors.get(task.name)
                        summaries[key] = (
                            _summarize_input_tensor(tensor)
                            if tensor is not None
                            else None
                        )
                    summary = summaries[key]
                    if summary is None:
                        continue
                    allow_nan = tensor_kind in metric.allowed_nan_inputs
                    has_infinity = any(
                        (
                            summary.positive_infinity_count,
                            summary.negative_infinity_count,
                            summary.mixed_infinity_count,
                        )
                    )
                    if (allow_nan or summary.nan_count == 0) and not has_infinity:
                        continue
                    records.append(
                        _InvalidInputRecord(
                            metric_name=type(metric).__name__,
                            namespace=namespace,
                            task_name=task.name,
                            tensor_kind=tensor_kind,
                            allow_nan=allow_nan,
                            summary=summary,
                        )
                    )
        return records

    def _format_input_validation_error(
        self,
        records: List[_InvalidInputRecord],
    ) -> str:
        lines = [
            "RecMetric validation failed at input: "
            f"rank={self._debug_rank} trained_batches={self.trained_batches}"
        ]
        for record in records:
            policy = (
                f"NaN {record.tensor_kind} allowed; "
                f"{record.tensor_kind} infinities rejected"
                if record.allow_nan
                else f"finite {record.tensor_kind} required"
            )
            lines.extend(
                (
                    f"metric={record.metric_name} namespace={record.namespace} "
                    f"task={record.task_name}",
                    _format_input_summary(self._debug_rank, record),
                    f"policy={policy}",
                )
            )
        return "\n".join(lines)

    def _validate_rec_metric_inputs(
        self,
        labels: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
    ) -> None:
        if not self._debug_mode:
            return
        records = self._build_invalid_input_records(labels, predictions, weights)
        if records:
            raise RecMetricValidationError(self._format_input_validation_error(records))

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # torch.load restores a module via __new__ + __setstate__, skipping __init__, so an
        # instance pickled before these attributes existed comes back without them. Restore
        # per-instance defaults here -- never a mutable CLASS attribute, which would alias
        # the accumulator across every instance.
        super().__setstate__(state)
        if not hasattr(self, "_loss_acc"):
            self._loss_acc = _LossAccumulator()
        if not hasattr(self, "_loss_aggregation_scope"):
            self._loss_aggregation_scope = LossAggregationScope.PER_READER_BATCH

    def set_loss_aggregation(
        self, loss_aggregation: Optional[Dict[str, LossAggregation]]
    ) -> None:
        """Override how specific ``:loss`` keys recombine across micro-batches.

        Only needed for exceptions — the normal path is producer-declared via an
        emitted ``:loss_denom``. See ``LossAggregation``.
        """
        self._loss_acc.set_aggregation(loss_aggregation)

    def set_under_micro_batching(self, value: bool) -> None:
        """Declare whether this job accumulates several reader batches per optimizer step.

        Gates the accumulation chain, which must stay unreachable otherwise: ``update()``
        runs for every model, so a live chain would give every one-batch-per-step consumer
        published ``<task>:loss`` keys. Config-driven, never history-driven -- eval skips
        the micro-batch loop, so a latched scope would change the eval's own key set.
        """
        self._loss_aggregation_scope = (
            LossAggregationScope.PER_OPTIMIZER_STEP
            if value
            else LossAggregationScope.PER_READER_BATCH
        )

    @property
    def loss_aggregation_scope(self) -> LossAggregationScope:
        """The span each published ``:loss`` value covers."""
        return getattr(
            self, "_loss_aggregation_scope", LossAggregationScope.PER_READER_BATCH
        )

    @property
    def under_micro_batching(self) -> bool:
        """Boolean view of the scope, for a caller's capability guard."""
        return self.loss_aggregation_scope is LossAggregationScope.PER_OPTIMIZER_STEP

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
        """Remove _trained_batches key for backward compatibility."""
        key = f"{prefix}_trained_batches"
        if key in state_dict:
            state_dict.pop(key)
            logger.warning(
                f"Removed key '{key}' from state_dict for backward compatibility"
            )

    def _prepare_model_out_for_metrics(
        self, model_out: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Prepare model outputs before task and required-input parsing.

        Subclasses can override this hook when their model-output format needs
        normalization. CPU-offloaded modules run it on the worker after transfer
        to CPU, keeping preparation off the trainer critical path.
        """
        return model_out

    def _update_rec_metrics(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        r"""the internal update function to parse the model output.
        Override this function if the implementation cannot support
        the model output format.
        """
        # pyrefly: ignore[not-callable]
        if self.rec_metrics and self.rec_tasks:
            model_out = self._prepare_model_out_for_metrics(model_out)
            labels, predictions, weights, required_inputs = parse_task_model_outputs(
                self.rec_tasks, model_out, self.get_required_inputs()
            )
            self._validate_rec_metric_inputs(labels, predictions, weights)
            if required_inputs:
                kwargs["required_inputs"] = required_inputs

            self.rec_metrics.update(
                predictions=predictions,
                labels=labels,
                weights=weights,
                **kwargs,
            )

    @EventLoggingHandler.event_logger(
        TorchrecComponent.REC_METRICS, n=1000, add_wait_counter=True
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
            # update() runs for every model, so the accumulation chain must stay
            # unreachable unless the scope actually spans several reader batches.
            if self.loss_aggregation_scope is LossAggregationScope.PER_OPTIMIZER_STEP:
                self._accumulate_loss_metrics(model_out)

    def update_micro_batch(
        self, model_out: Dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        """Like ``update()``, but does not advance the optimizer-step counter.

        For the intermediate micro-batches (0..K-2); the closing one takes ``update()``.
        Throughput IS still advanced: every micro-batch is a real reader batch of the full
        configured ``batch_size`` (K does not divide it), so skipping the call would
        under-report QPS by K x. Losses accumulate so ``compute()`` covers all K.
        """
        if self.loss_aggregation_scope is not LossAggregationScope.PER_OPTIMIZER_STEP:
            raise RecMetricException(
                "update_micro_batch() called while the loss scope spans a single reader "
                "batch. Accumulation would run but compute() would never publish it, so "
                "the loss would build up unread. Arm the scope with "
                "set_under_micro_batching(True), or call update() instead."
            )
        with record_function("## RecMetricModule:update_micro_batch ##"):
            self._update_rec_metrics(model_out, **kwargs)
            if self.throughput_metric:
                self.throughput_metric.update()
            self._accumulate_loss_metrics(model_out)

    def reset_loss_metrics(self) -> None:
        """Reset the per-step loss accumulators, before any update for that step.

        Cleared at step boundaries rather than in ``compute()``: with
        ``compute_interval_steps > 1`` that keeps the published loss the most recent
        step's, matching the behaviour of reading it straight off the model output.
        Recombination is per the key's declared ``LossAggregation``.
        """
        self._loss_acc.reset()

    def _accumulate_loss_metrics(self, model_out: Dict[str, torch.Tensor]) -> None:
        """Accumulate this micro-batch's per-task loss values into the current step.

        Validation and accumulation interleave per key, so a raise part-way leaves the
        earlier keys already written. Recombination rules: ``_LossAccumulator``.
        """
        self._loss_acc.accumulate(model_out)

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

    @EventLoggingHandler.event_logger(
        TorchrecComponent.REC_METRICS, add_wait_counter=True, n=1000
    )
    def compute(self) -> DeferrableMetrics:
        r"""compute() is called when the global metrics are required, usually
        right before logging the metrics results to the data sink.
        """
        self.compute_count += 1
        ret: MetricsResult = {}
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
            # Publishing the recombined per-step losses is reachable only once the scope
            # spans several reader batches, so every one-batch-per-step model keeps the
            # pre-existing key set -- there the caller supplies ``<task>:loss`` itself.
            if self.loss_aggregation_scope is LossAggregationScope.PER_OPTIMIZER_STEP:
                ret.update(self._loss_acc.reduced_losses())
        return DeferrableMetrics(ret)

    def compute_throughput(self) -> DeferrableMetrics:
        r"""Compute throughput metrics without cross-rank collectives."""
        ret: MetricsResult = {}
        if self.throughput_metric:
            ret.update(self.throughput_metric.compute())
        return DeferrableMetrics(ret)

    def local_compute(self) -> DeferrableMetrics:
        r"""local_compute() is called when per-trainer metrics are required. It's
        can be used for debugging. Currently only rec_metrics is supported.
        """
        ret: MetricsResult = {}
        if self.rec_metrics:
            ret.update(self.rec_metrics.local_compute())
        return DeferrableMetrics(ret)

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
        # pyrefly: ignore[bad-return]
        return {metric._metric_name.value: states}

    def _get_metric_states(
        self,
        metric: RecMetric,
        world_size: int,
        process_group: dist.ProcessGroup,
    ) -> Dict[str, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        Gather metric states from all ranks and apply reduction.

        For list states (e.g., AUC predictions/labels):
        - Concatenate local tensors into single tensor for efficient transport
        - Single all_gather instead of N all_gathers
        - Apply reduction once on gathered tensors

        For tensor states (e.g., NE cross_entropy_sum):
        - Stack gathered tensors
        - Apply reduction

        This approach works with ANY reduction function (no associativity requirement).
        """
        result = defaultdict(dict)
        for task, computation in zip(metric._tasks, metric._metrics_computations):
            #  `items`.
            # pyrefly: ignore[missing-attribute]
            for state_name, reduction_fn in computation._reductions.items():
                with record_function(f"## RecMetricModule: {state_name} all gather ##"):
                    tensor_or_list: Union[List[torch.Tensor], torch.Tensor] = getattr(
                        computation, state_name
                    )

                    if isinstance(tensor_or_list, list):
                        if len(tensor_or_list) == 0:
                            result[task.name][state_name] = []
                            continue

                        # Concatenate local tensors into single tensor for transport
                        # This reduces N all_gathers to 1 all_gather
                        local_concat = torch.cat(tensor_or_list, dim=-1)

                        # Single all_gather for the concatenated tensor
                        gathered_list = _all_gather_tensor(
                            local_concat, world_size, process_group
                        )

                        # Apply reduction once on the gathered tensors (one per rank)
                        if reduction_fn is not None:
                            reduced = reduction_fn(gathered_list)
                        else:
                            reduced = gathered_list
                    else:
                        # Single tensor case
                        gathered = torch.stack(
                            _all_gather_tensor(
                                tensor_or_list, world_size, process_group
                            )
                        )
                        if reduction_fn is not None:
                            reduced = reduction_fn(gathered)
                        else:
                            reduced = gathered

                    result[task.name][state_name] = reduced

        return result

    def _get_rec_metric_states(
        self, pg: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None
    ) -> PreComputeStates:
        pg = pg if pg is not None else dist.group.WORLD
        process_group: dist.ProcessGroup = (
            # pyrefly: ignore[bad-assignment]
            pg.get_group(mesh_dim="shard")
            if isinstance(pg, DeviceMesh)
            else pg
        )

        aggregated_states: PreComputeStates = {}
        world_size = dist.get_world_size(
            process_group
        )  # Under 2D parallel context, this should be sharding world size

        for metric in self.rec_metrics.rec_metrics:
            #  `value`.
            # pyrefly: ignore[missing-attribute]
            aggregated_states[metric._namespace.value] = self._get_metric_states(
                # pyrefly: ignore[bad-argument-type]
                metric,
                world_size,
                process_group,
            )

        return aggregated_states

    def get_pre_compute_states(
        self, pg: Optional[Union[dist.ProcessGroup, DeviceMesh]] = None
    ) -> PreComputeStates:
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
        aggregated_states = self._get_rec_metric_states(pg)

        # throughput metric requires special handling, since it's not a RecMetric
        throughput_metric = self.throughput_metric
        if throughput_metric is not None:
            # Merge in case there are rec metric namespaces that overlap with throughput metric namespace
            # pyrefly: ignore [no-matching-overload]
            aggregated_states.setdefault(throughput_metric._namespace.value, {}).update(
                self._get_throughput_metric_states(throughput_metric)
            )

        return aggregated_states

    def _load_rec_metric_states(self, source: PreComputeStates) -> None:
        for metric in self.rec_metrics.rec_metrics:
            #  `value`.
            # pyrefly: ignore[missing-attribute]
            states = source[metric._namespace.value]
            # pyrefly: ignore[no-matching-overload]
            for task, metric_computation in zip(
                #  `Union[Module, Tensor]`.
                #  `Union[Module, Tensor]`.
                # pyrefly: ignore
                metric._tasks,
                #  `Union[Module, Tensor]`.
                # pyrefly: ignore
                metric._metrics_computations,
            ):
                state = states[task.name]
                for attr, tensor in state.items():
                    setattr(metric_computation, attr, tensor)

    def load_pre_compute_states(self, source: PreComputeStates) -> None:
        """
        Load states from ``get_pre_compute_states``. This is called on every rank, no collectives are called in this function.

        Args:
            source (PreComputeStates): the source states to load from. This is the output of ``get_pre_compute_states``.

        Returns:
            None
        """
        self._load_rec_metric_states(source)

        if self.throughput_metric is not None:
            # pyrefly: ignore[bad-index]
            states = source[self.throughput_metric._namespace.value][
                # pyrefly: ignore[bad-index]
                self.throughput_metric._metric_name.value
            ]
            for name, buf in self.throughput_metric.named_buffers():
                # pyrefly: ignore[bad-argument-type]
                buf.copy_(states[name])

    def shutdown(self) -> None:
        logger.info("Initiating graceful shutdown...")

    def async_compute(self) -> DeferrableMetrics:
        raise RecMetricException("async_compute is not supported in RecMetricModule")


def _generate_rec_metrics(
    metrics_config: MetricsConfig,
    world_size: int,
    my_rank: int,
    batch_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
    batch_size_stages: Optional[List[BatchSizeStage]] = None,
) -> RecMetricList:
    rec_metrics = []
    for metric_enum, metric_def in metrics_config.rec_metrics.items():

        kwargs: Dict[str, Any] = {}
        if metric_def and metric_def.arguments is not None:
            kwargs = metric_def.arguments

        kwargs["enable_pt2_compile"] = metrics_config.enable_pt2_compile
        kwargs["should_clone_update_inputs"] = metrics_config.should_clone_update_inputs
        kwargs["batch_size_stages"] = batch_size_stages

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
    module_kwargs: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False,
) -> RecMetricModule:
    rec_metrics = _generate_rec_metrics(
        metrics_config,
        world_size,
        my_rank,
        batch_size,
        process_group,
        batch_size_stages,
    )
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
        **(module_kwargs if module_kwargs else {}),
    )
    metrics._configure_debug_mode(
        debug_mode,
        my_rank,
    )
    # Applied post-construction rather than as a constructor kwarg: RecMetricModule
    # subclasses declare explicit __init__ signatures and forward a fixed argument
    # list, so a new kwarg here would be a TypeError for every one of them. getattr,
    # not attribute access: MetricsConfig is widely subclassed and an instance can
    # predate this field.
    metrics.set_loss_aggregation(getattr(metrics_config, "loss_aggregation", None))
    # Config-driven scope selection. getattr for the same reason as the line above:
    # MetricsConfig is widely subclassed and an instance can predate this field.
    metrics.set_under_micro_batching(
        getattr(metrics_config, "num_micro_batches_per_step", 1) > 1
    )
    metrics.to(device)
    return metrics


def _all_gather_tensor(
    tensor: torch.Tensor,
    world_size: int,
    pg: dist.ProcessGroup,
) -> List[torch.Tensor]:
    """All-gather a single tensor and return the gathered list."""
    out = [torch.empty_like(tensor) for _ in range(world_size)]  # pragma: no cover
    dist.all_gather(out, tensor, group=pg)
    return out
