#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import sys
import threading
import time
import traceback
import unittest
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, cast, Iterable, Mapping
from unittest.mock import patch

import torch
from parameterized import parameterized
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.deferrable_metrics import DeferrableMetrics
from torchrec.metrics.metric_module import REC_METRICS_MAPPING, RecMetricModule
from torchrec.metrics.metric_state_snapshot import MetricStateSnapshot
from torchrec.metrics.metrics_config import (
    RecComputeMode,
    RecMetricEnum,
    RecTaskInfo,
    SessionMetricDef,
)
from torchrec.metrics.rec_metric import (
    RecMetric,
    RecMetricComputation,
    RecMetricException,
    RecMetricList,
)
from torchrec.metrics.test_utils import gen_test_tasks


class _InputKind(Enum):
    BINARY = "binary"
    MISSING_LABELS = "missing_labels"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    SCALAR = "scalar"


class _FailurePhase(Enum):
    CONSTRUCTION = "construction"
    UPDATE = "update"
    CHECKPOINT = "checkpoint"


_ALL_COMPUTE_MODES = (
    RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    RecComputeMode.FUSED_TASKS_COMPUTATION,
    RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
)
_UNFUSED_ONLY = (RecComputeMode.UNFUSED_TASKS_COMPUTATION,)
_MULTI_METRIC_CASE = (
    RecMetricEnum.NE,
    RecMetricEnum.AUC,
    RecMetricEnum.TENSOR_WEIGHTED_AVG,
)
_SHARD_COUNT = 4


@dataclass(frozen=True)
class _ExpectedDifferences:
    initial_states: frozenset[str]
    initial_outputs: frozenset[str]
    sentinel_states: frozenset[str]
    sentinel_outputs: frozenset[str]


@dataclass(frozen=True)
class _MetricCase:
    input_kind: _InputKind
    supported_modes: tuple[RecComputeMode, ...]
    equivalent_modes: tuple[RecComputeMode, ...]
    expected_differences: _ExpectedDifferences | None = None


class _DeterministicTime:
    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            self._value += 1.0
            return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@dataclass(frozen=True)
class _UnsupportedExpectation:
    exception_type: type[Exception]
    message: str | None = None
    exception_args: tuple[object, ...] | None = None
    traceback_location: tuple[str, str] | None = None
    phase: _FailurePhase = _FailurePhase.UPDATE


_TOWER_QPS_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset({"time_lapse", "window_time_lapse"}),
    initial_outputs=frozenset({"local_lifetime_qps", "local_window_qps"}),
    sentinel_states=frozenset({"time_lapse", "window_time_lapse"}),
    sentinel_outputs=frozenset({"local_lifetime_qps", "local_window_qps"}),
)
_RECALL_SESSION_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset(
        {
            "num_false_neg",
            "num_true_pos",
            "window_num_false_neg",
            "window_num_true_pos",
        }
    ),
    initial_outputs=frozenset(
        {
            "local_lifetime_recall_session_level",
            "local_window_recall_session_level",
        }
    ),
    sentinel_states=frozenset(
        {
            "num_false_neg",
            "num_true_pos",
            "window_num_false_neg",
            "window_num_true_pos",
        }
    ),
    sentinel_outputs=frozenset(
        {
            "local_lifetime_recall_session_level",
            "local_window_recall_session_level",
        }
    ),
)
_PRECISION_SESSION_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset(
        {
            "num_false_pos",
            "num_true_pos",
            "window_num_false_pos",
            "window_num_true_pos",
        }
    ),
    initial_outputs=frozenset(
        {
            "local_lifetime_precision_session_level",
            "local_window_precision_session_level",
        }
    ),
    sentinel_states=frozenset(
        {
            "num_false_pos",
            "num_true_pos",
            "window_num_false_pos",
            "window_num_true_pos",
        }
    ),
    sentinel_outputs=frozenset(
        {
            "local_lifetime_precision_session_level",
            "local_window_precision_session_level",
        }
    ),
)
_NDCG_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset(
        {"num_sessions", "sum_ndcg", "window_num_sessions", "window_sum_ndcg"}
    ),
    initial_outputs=frozenset({"local_lifetime_ndcg", "local_window_ndcg"}),
    sentinel_states=frozenset(
        {"num_sessions", "sum_ndcg", "window_num_sessions", "window_sum_ndcg"}
    ),
    sentinel_outputs=frozenset({"local_lifetime_ndcg", "local_window_ndcg"}),
)
_XAUC_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset(
        {
            "error_sum",
            "weighted_num_pairs",
            "window_error_sum",
            "window_weighted_num_pairs",
        }
    ),
    initial_outputs=frozenset({"local_lifetime_xauc", "local_window_xauc"}),
    sentinel_states=frozenset(
        {
            "error_sum",
            "weighted_num_pairs",
            "window_error_sum",
            "window_weighted_num_pairs",
        }
    ),
    sentinel_outputs=frozenset({"local_lifetime_xauc", "local_window_xauc"}),
)
_SCALAR_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset({"labels", "window_labels", "window_window_count"}),
    initial_outputs=frozenset({"local_lifetime_scalar"}),
    sentinel_states=frozenset({"window_labels", "window_window_count"}),
    sentinel_outputs=frozenset({"local_window_scalar"}),
)
_OUTPUT_DIFFERENCES = _ExpectedDifferences(
    initial_states=frozenset({"latest_imp", "total_latest_imp"}),
    initial_outputs=frozenset(
        {"local_output_latest_imp", "local_output_total_latest_imp"}
    ),
    sentinel_states=frozenset(),
    sentinel_outputs=frozenset(),
)

_METRIC_CASES: dict[RecMetricEnum, _MetricCase] = {
    RecMetricEnum.NE: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.NE_POSITIVE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.SEGMENTED_NE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.RECALIBRATED_NE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.RECALIBRATED_CALIBRATION: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.CTR: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.CALIBRATION: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.AUC: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.AUPRC: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.RAUC: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.MSE: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.MAE: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.MULTICLASS_RECALL: _MetricCase(
        _InputKind.MULTICLASS,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.WEIGHTED_AVG: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.TOWER_QPS: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        (),
        _TOWER_QPS_DIFFERENCES,
    ),
    RecMetricEnum.RECALL_SESSION_LEVEL: _MetricCase(
        _InputKind.BINARY,
        _UNFUSED_ONLY,
        (),
        _RECALL_SESSION_DIFFERENCES,
    ),
    RecMetricEnum.PRECISION_SESSION_LEVEL: _MetricCase(
        _InputKind.BINARY,
        _UNFUSED_ONLY,
        (),
        _PRECISION_SESSION_DIFFERENCES,
    ),
    RecMetricEnum.ACCURACY: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.NDCG: _MetricCase(
        _InputKind.BINARY,
        _UNFUSED_ONLY,
        (),
        _NDCG_DIFFERENCES,
    ),
    RecMetricEnum.XAUC: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        (),
        _XAUC_DIFFERENCES,
    ),
    RecMetricEnum.SCALAR: _MetricCase(
        _InputKind.SCALAR,
        _ALL_COMPUTE_MODES,
        (),
        _SCALAR_DIFFERENCES,
    ),
    RecMetricEnum.PRECISION: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.RECALL: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.SERVING_NE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.SERVING_CALIBRATION: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.OUTPUT: _MetricCase(
        _InputKind.BINARY,
        _UNFUSED_ONLY,
        (),
        _OUTPUT_DIFFERENCES,
    ),
    RecMetricEnum.TENSOR_WEIGHTED_AVG: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.CALI_FREE_NE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.UNWEIGHTED_NE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.HINDSIGHT_TARGET_PR: _MetricCase(
        _InputKind.BINARY,
        _UNFUSED_ONLY,
        _UNFUSED_ONLY,
    ),
    RecMetricEnum.NMSE: _MetricCase(
        _InputKind.BINARY, _ALL_COMPUTE_MODES, _ALL_COMPUTE_MODES
    ),
    RecMetricEnum.AVERAGE: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.MULTI_LABEL_PRECISION: _MetricCase(
        _InputKind.MULTILABEL,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.NUM_MISSING_LABELS: _MetricCase(
        _InputKind.MISSING_LABELS,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.WEIGHTED_SUM_PREDICTIONS: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.NUM_POSITIVE_SAMPLES: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
    RecMetricEnum.SUM_WEIGHTS: _MetricCase(
        _InputKind.BINARY,
        _ALL_COMPUTE_MODES,
        _ALL_COMPUTE_MODES,
    ),
}

_UNSUPPORTED_EXPECTATIONS: dict[RecMetricEnum, _UnsupportedExpectation] = {
    RecMetricEnum.RECALL_SESSION_LEVEL: _UnsupportedExpectation(
        exception_type=RecMetricException,
        message="Fused computation is not supported for recall session-level metrics",
        phase=_FailurePhase.CONSTRUCTION,
    ),
    RecMetricEnum.PRECISION_SESSION_LEVEL: _UnsupportedExpectation(
        exception_type=RecMetricException,
        message="Fused computation is not supported for precision session-level metrics",
        phase=_FailurePhase.CONSTRUCTION,
    ),
    RecMetricEnum.NDCG: _UnsupportedExpectation(
        exception_type=AssertionError,
        exception_args=(),
        traceback_location=("torchrec/metrics/ndcg.py", "_validate_model_outputs"),
    ),
    RecMetricEnum.OUTPUT: _UnsupportedExpectation(
        exception_type=TypeError,
        message="0-d tensor",
        phase=_FailurePhase.CHECKPOINT,
    ),
    RecMetricEnum.HINDSIGHT_TARGET_PR: _UnsupportedExpectation(
        exception_type=RuntimeError,
        message="size=[]",
    ),
}


@dataclass(frozen=True)
class _MetricCheckpoint:
    states: dict[str, Any]
    state_names: dict[str, str]
    outputs: dict[str, Any]


@dataclass(frozen=True)
class _ArmResult:
    initial: _MetricCheckpoint
    after_sentinel: _MetricCheckpoint


class RecMetricBatchingBehaviorTest(unittest.TestCase):
    K: int = 10
    BATCH_SIZE: int = 2
    TASK_COUNT: int = 2
    WINDOW_SIZE: int = 1000
    ASYNC_TIMEOUT_SECONDS: float = 30.0
    maxDiff: int | None = None

    def _tasks(self) -> list[RecTaskInfo]:
        return [
            replace(
                task,
                session_metric_def=SessionMetricDef(
                    session_var_name="session",
                    top_threshold=1,
                    run_ranking_of_labels=False,
                ),
            )
            for task in gen_test_tasks(
                [f"task{index}" for index in range(1, self.TASK_COUNT + 1)]
            )
        ]

    def _arguments(self, metric_enum: RecMetricEnum) -> dict[str, Any]:
        if metric_enum == RecMetricEnum.MULTICLASS_RECALL:
            return {"number_of_classes": 2}
        if metric_enum == RecMetricEnum.SEGMENTED_NE:
            return {"num_groups": 2}
        if metric_enum == RecMetricEnum.TOWER_QPS:
            return {"warmup_steps": 0}
        if metric_enum == RecMetricEnum.MULTI_LABEL_PRECISION:
            return {"num_labels": 2, "label_names": ["a", "b"]}
        if metric_enum == RecMetricEnum.AUC:
            return {"grouped_auc": True}
        if metric_enum == RecMetricEnum.AUPRC:
            return {"grouped_auprc": True}
        if metric_enum == RecMetricEnum.RAUC:
            return {"grouped_rauc": True}
        if metric_enum == RecMetricEnum.HINDSIGHT_TARGET_PR:
            return {"target_precision": 0.0}
        return {}

    def _make_metrics(
        self,
        metric_enums: list[RecMetricEnum],
        compute_mode: RecComputeMode,
        tasks: list[RecTaskInfo],
    ) -> RecMetricList:
        metrics: list[RecMetric] = []
        for metric_enum in metric_enums:
            metric_class = cast(type[RecMetric], REC_METRICS_MAPPING[metric_enum])
            metrics.append(
                metric_class(
                    world_size=1,
                    my_rank=0,
                    batch_size=self.BATCH_SIZE,
                    tasks=tasks,
                    compute_mode=compute_mode,
                    window_size=self.WINDOW_SIZE,
                    should_validate_update=True,
                    **self._arguments(metric_enum),
                )
            )
        return RecMetricList(metrics)

    def _make_module(
        self,
        metric_enums: list[RecMetricEnum],
        compute_mode: RecComputeMode,
        tasks: list[RecTaskInfo],
        update_batch_size: int | None,
    ) -> RecMetricModule:
        kwargs: dict[str, Any] = {
            "batch_size": self.BATCH_SIZE,
            "world_size": 1,
            "rec_tasks": tasks,
            "rec_metrics": self._make_metrics(metric_enums, compute_mode, tasks),
        }
        if update_batch_size is None:
            return RecMetricModule(**kwargs)
        return CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"),
            update_batch_size=update_batch_size,
            **kwargs,
        )

    def _base_predictions_and_labels(
        self, input_kind: _InputKind, batch_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_indices = torch.arange(self.BATCH_SIZE)
        if input_kind == _InputKind.MULTICLASS:
            labels = (sample_indices + batch_index) % 2
            confidence = torch.linspace(
                0.6, 0.9, steps=self.BATCH_SIZE, dtype=torch.double
            )
            predictions = torch.empty((self.BATCH_SIZE, 2), dtype=torch.double)
            predictions[:, 0] = torch.where(labels == 0, confidence, 1.0 - confidence)
            predictions[:, 1] = 1.0 - predictions[:, 0]
            return predictions, labels
        if input_kind == _InputKind.MULTILABEL:
            # Multi-label predictions and labels are two-bit integer masks.
            predictions = (sample_indices + batch_index) % 4
            return predictions, (predictions + 1) % 4
        if input_kind == _InputKind.SCALAR:
            return (
                torch.zeros(self.BATCH_SIZE, dtype=torch.double),
                batch_index + torch.arange(self.BATCH_SIZE, dtype=torch.double) / 2.0,
            )
        predictions = torch.tensor(
            [
                (
                    0.9 - 0.01 * (batch_index % 20)
                    if sample_index % 2 == 0
                    else 0.1 + 0.01 * (batch_index % 20)
                )
                for sample_index in range(self.BATCH_SIZE)
            ],
            dtype=torch.double,
        )
        if input_kind == _InputKind.MISSING_LABELS:
            labels = torch.zeros(self.BATCH_SIZE, dtype=torch.double)
            labels[0] = float("nan") if batch_index % 2 == 0 else 1.0
            return predictions, labels
        return predictions, torch.tensor(
            [
                float((batch_index + sample_index) % 2 == 0)
                for sample_index in range(self.BATCH_SIZE)
            ],
            dtype=torch.double,
        )

    def _task_inputs(
        self,
        input_kind: _InputKind,
        batch_index: int,
        task_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions, labels = self._base_predictions_and_labels(input_kind, batch_index)
        if task_index % 2 == 1:
            if input_kind == _InputKind.MULTICLASS:
                predictions, labels = predictions.flip(-1), 1 - labels
            elif input_kind == _InputKind.MULTILABEL:
                predictions, labels = (predictions + 1) % 4, (labels + 2) % 4
            elif input_kind == _InputKind.MISSING_LABELS:
                predictions, labels = 1 - predictions, labels.flip(0)
            elif input_kind == _InputKind.SCALAR:
                labels = labels + 10.0
            else:
                predictions, labels = 1 - predictions, 1 - labels
        weights = (
            torch.arange(1, self.BATCH_SIZE + 1, dtype=torch.double)
            + batch_index
            + 10.0 * task_index
        )
        return predictions, labels, weights

    def _make_batch(
        self,
        input_kind: _InputKind,
        tasks: list[RecTaskInfo],
        batch_index: int,
    ) -> dict[str, torch.Tensor]:
        model_out: dict[str, torch.Tensor] = {}
        for task_index, task in enumerate(tasks):
            predictions, labels, weights = self._task_inputs(
                input_kind, batch_index, task_index
            )
            model_out[task.prediction_name] = predictions
            model_out[task.label_name] = labels
            model_out[task.weight_name] = weights
            tensor_name = cast(str, task.tensor_name)
            model_out[tensor_name] = (
                10.0 * torch.arange(1, self.BATCH_SIZE + 1, dtype=torch.double)
                + batch_index
                + 100.0 * task_index
            )
        model_out["grouping_keys"] = torch.full((self.BATCH_SIZE,), batch_index % 2)
        model_out["session_id"] = torch.ones(self.BATCH_SIZE, dtype=torch.int64)
        model_out["session"] = torch.ones(self.BATCH_SIZE, dtype=torch.int64)
        model_out["latest_imp"] = (
            torch.arange(1, self.BATCH_SIZE + 1, dtype=torch.double)
            + 10.0 * batch_index
        )
        model_out["total_latest_imp"] = (
            torch.arange(1, self.BATCH_SIZE + 1, dtype=torch.double)
            + 100.0 * batch_index
        )
        return model_out

    def _resolve_metrics(
        self, metrics: DeferrableMetrics, operation: str
    ) -> dict[str, Any]:
        completed = threading.Event()
        result: dict[str, Any] | None = None
        error: Exception | None = None

        def on_success(value: dict[str, Any]) -> None:
            nonlocal result
            result = value
            completed.set()

        def on_error(value: Exception) -> None:
            nonlocal error
            error = value
            completed.set()

        metrics.subscribe(on_success, on_error)
        if not completed.wait(self.ASYNC_TIMEOUT_SECONDS):
            raise TimeoutError(f"Timed out waiting for {operation}")
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError(f"{operation} completed without a result")
        return result

    def _metric_state_names(self, module: RecMetricModule) -> dict[str, str]:
        state_names: dict[str, str] = {}
        for metric_value in module.rec_metrics.rec_metrics:
            metric = cast(RecMetric, metric_value)
            prefixes = (
                [task.name for task in metric._tasks]
                if metric._compute_mode == RecComputeMode.UNFUSED_TASKS_COMPUTATION
                else [metric._compute_mode.name]
            )
            for prefix, computation_value in zip(
                prefixes, metric._metrics_computations
            ):
                computation = cast(RecMetricComputation, computation_value)
                computation_name = f"{prefix}_{computation.__class__.__name__}"
                for state_name in computation._reductions:
                    state_names[f"{computation_name}_{state_name}"] = state_name
        return state_names

    def _checkpoint(self, module: RecMetricModule) -> _MetricCheckpoint:
        if isinstance(module, CPUOffloadedRecMetricModule):
            # The public throughput future is the worker-queue barrier.
            self._resolve_metrics(module.compute_throughput(), "update barrier")
        outputs = dict(self._resolve_metrics(module.local_compute(), "local compute"))
        return _MetricCheckpoint(
            states=MetricStateSnapshot.from_metrics(module.rec_metrics).metric_states,
            state_names=self._metric_state_names(module),
            outputs=outputs,
        )

    def _process_updates(
        self,
        module: RecMetricModule,
        input_kind: _InputKind,
        tasks: list[RecTaskInfo],
    ) -> _ArmResult:
        for batch_index in range(self.K):
            module.update(self._make_batch(input_kind, tasks, batch_index))
        initial = self._checkpoint(module)
        module.update(self._make_batch(input_kind, tasks, self.K))
        return _ArmResult(initial, self._checkpoint(module))

    def _shutdown_module(
        self,
        module: RecMetricModule,
        active_exception: BaseException | None,
        expected_worker_error: Exception | None = None,
    ) -> None:
        if not isinstance(module, CPUOffloadedRecMetricModule):
            return
        try:
            # Shutdown enqueues a distributed compute; isolate test teardown.
            with patch.object(module, "_process_metric_compute_job", return_value={}):
                module.shutdown()
        except Exception as shutdown_error:
            if (
                shutdown_error is expected_worker_error
                or shutdown_error is active_exception
            ):
                return
            if active_exception is not None:
                raise active_exception from shutdown_error
            raise

    def _run_module(
        self,
        module: RecMetricModule,
        input_kind: _InputKind,
        tasks: list[RecTaskInfo],
    ) -> _ArmResult:
        try:
            return self._process_updates(module, input_kind, tasks)
        finally:
            self._shutdown_module(module, sys.exc_info()[1])

    def _run_arm(
        self,
        metric_enums: list[RecMetricEnum],
        input_kind: _InputKind,
        compute_mode: RecComputeMode,
        update_batch_size: int | None,
    ) -> _ArmResult:
        tasks = self._tasks()
        module = self._make_module(metric_enums, compute_mode, tasks, update_batch_size)
        if RecMetricEnum.TOWER_QPS not in metric_enums:
            return self._run_module(module, input_kind, tasks)
        with patch("torchrec.metrics.tower_qps.time", _DeterministicTime()):
            return self._run_module(module, input_kind, tasks)

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(
                right, torch.Tensor
            ):
                return False
            low_precision = left.dtype in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            } or right.dtype in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }
            try:
                torch.testing.assert_close(
                    left,
                    right,
                    rtol=1e-5 if low_precision else 1e-6,
                    atol=1e-6 if low_precision else 1e-8,
                    equal_nan=True,
                )
            except AssertionError:
                return False
            return True
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                return False
            return set(left) == set(right) and all(
                self._values_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not isinstance(left, (list, tuple)) or not isinstance(
                right, (list, tuple)
            ):
                return False
            return len(left) == len(right) and all(
                self._values_equal(left_value, right_value)
                for left_value, right_value in zip(left, right)
            )
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    def _short_repr(self, value: Any) -> str:
        representation = repr(value)
        if len(representation) <= 200:
            return representation
        return f"{representation[:197]}..."

    def _difference_details(
        self, left: Mapping[str, Any], right: Mapping[str, Any]
    ) -> dict[str, str]:
        differences: dict[str, str] = {}
        for key in sorted(set(left) | set(right)):
            if (
                key in left
                and key in right
                and self._values_equal(left[key], right[key])
            ):
                continue
            reference_value = (
                self._short_repr(left[key]) if key in left else "<missing>"
            )
            candidate_value = (
                self._short_repr(right[key]) if key in right else "<missing>"
            )
            differences[key] = (
                f"reference={reference_value}, candidate={candidate_value}"
            )
        return differences

    def _checkpoint_differences(
        self, reference: _MetricCheckpoint, candidate: _MetricCheckpoint
    ) -> str | None:
        state_differences = self._difference_details(reference.states, candidate.states)
        output_differences = self._difference_details(
            reference.outputs, candidate.outputs
        )
        if not state_differences and not output_differences:
            return None
        return f"states={state_differences}, outputs={output_differences}"

    def _arm_differences(
        self, reference: _ArmResult, candidate: _ArmResult
    ) -> str | None:
        initial = self._checkpoint_differences(reference.initial, candidate.initial)
        sentinel = self._checkpoint_differences(
            reference.after_sentinel, candidate.after_sentinel
        )
        if initial is None and sentinel is None:
            return None
        return f"initial=({initial}), sentinel=({sentinel})"

    def _difference_key_signature(
        self, reference: _ArmResult, candidate: _ArmResult
    ) -> _ExpectedDifferences:
        def state_keys(
            left: _MetricCheckpoint, right: _MetricCheckpoint
        ) -> frozenset[str]:
            state_names = left.state_names | right.state_names
            return frozenset(
                state_names[key]
                for key in self._difference_details(left.states, right.states)
            )

        def output_keys(
            left: Mapping[str, Any], right: Mapping[str, Any]
        ) -> frozenset[str]:
            return frozenset(
                key.partition("|")[2] or key
                for key in self._difference_details(left, right)
            )

        return _ExpectedDifferences(
            initial_states=state_keys(reference.initial, candidate.initial),
            initial_outputs=output_keys(
                reference.initial.outputs, candidate.initial.outputs
            ),
            sentinel_states=state_keys(
                reference.after_sentinel, candidate.after_sentinel
            ),
            sentinel_outputs=output_keys(
                reference.after_sentinel.outputs, candidate.after_sentinel.outputs
            ),
        )

    def _metric_behavior_failures(
        self,
        metric_enum: RecMetricEnum,
        case: _MetricCase,
        compute_mode: RecComputeMode,
    ) -> dict[str, str]:
        metric_name = metric_enum.value
        standard = self._run_arm([metric_enum], case.input_kind, compute_mode, None)
        zorm_k1 = self._run_arm([metric_enum], case.input_kind, compute_mode, 1)
        failures: dict[str, str] = {}
        k1_difference = self._arm_differences(standard, zorm_k1)
        if k1_difference is not None:
            failures[f"{metric_name}/zorm_k1"] = k1_difference

        zorm_k10 = self._run_arm([metric_enum], case.input_kind, compute_mode, self.K)
        k10_difference = self._arm_differences(standard, zorm_k10)
        if compute_mode in case.equivalent_modes:
            if k10_difference is not None:
                failures[f"{metric_name}/zorm_k10"] = k10_difference
            return failures

        actual_differences = self._difference_key_signature(standard, zorm_k10)
        if actual_differences != case.expected_differences:
            failures[f"{metric_name}/zorm_k10"] = (
                f"expected={case.expected_differences}, actual={actual_differences}"
            )
        return failures

    def _behavior_failures(
        self,
        compute_mode: RecComputeMode,
        metric_cases: Iterable[tuple[RecMetricEnum, _MetricCase]],
    ) -> dict[str, str]:
        failures: dict[str, str] = {}
        for metric_enum, case in metric_cases:
            if compute_mode not in case.supported_modes:
                continue
            with self.subTest(
                metric=metric_enum.value,
                compute_mode=compute_mode.name,
            ):
                failures.update(
                    self._metric_behavior_failures(metric_enum, case, compute_mode)
                )
        return failures

    def _run_construction_failure(
        self,
        metric_enum: RecMetricEnum,
        compute_mode: RecComputeMode,
        tasks: list[RecTaskInfo],
        update_batch_size: int | None,
        expectation: _UnsupportedExpectation,
        context: str,
    ) -> Exception:
        try:
            module = self._make_module(
                [metric_enum], compute_mode, tasks, update_batch_size
            )
        except expectation.exception_type as expected_error:
            return expected_error
        self._shutdown_module(module, None)
        self.fail(context)

    def _run_update_failure(
        self,
        module: RecMetricModule,
        input_kind: _InputKind,
        tasks: list[RecTaskInfo],
        update_count: int,
        expectation: _UnsupportedExpectation,
        context: str,
    ) -> Exception:
        if not isinstance(module, CPUOffloadedRecMetricModule):
            for batch_index in range(update_count):
                error = self._capture_if_raised(
                    lambda batch_index=batch_index: module.update(
                        self._make_batch(input_kind, tasks, batch_index)
                    ),
                    expectation,
                )
                if error is not None:
                    return error
            self.fail(context)
        for batch_index in range(update_count):
            error = self._capture_if_raised(
                lambda batch_index=batch_index: module.update(
                    self._make_batch(input_kind, tasks, batch_index)
                ),
                expectation,
            )
            if error is not None:
                return error
        module.update_thread.join(timeout=self.ASYNC_TIMEOUT_SECONDS)
        if module.update_thread.is_alive():
            self.fail("Timed out waiting for the update worker to fail")
        return self._capture_expected_error(
            lambda: module.update(self._make_batch(input_kind, tasks, update_count)),
            expectation,
            context,
        )

    def _run_unsupported_case(
        self,
        metric_enum: RecMetricEnum,
        input_kind: _InputKind,
        compute_mode: RecComputeMode,
        update_batch_size: int | None,
        expectation: _UnsupportedExpectation,
    ) -> Exception:
        tasks = self._tasks()
        context = f"{metric_enum.value} unexpectedly supported {compute_mode.name}"
        if expectation.phase == _FailurePhase.CONSTRUCTION:
            return self._run_construction_failure(
                metric_enum,
                compute_mode,
                tasks,
                update_batch_size,
                expectation,
                context,
            )

        module = self._make_module(
            [metric_enum], compute_mode, tasks, update_batch_size
        )
        expected_error: Exception | None = None
        try:
            update_count = 1 if update_batch_size == 1 else self.K
            if expectation.phase == _FailurePhase.UPDATE:
                expected_error = self._run_update_failure(
                    module,
                    input_kind,
                    tasks,
                    update_count,
                    expectation,
                    context,
                )
            else:
                for batch_index in range(update_count):
                    module.update(self._make_batch(input_kind, tasks, batch_index))
                expected_error = self._capture_expected_error(
                    lambda: self._checkpoint(module), expectation, context
                )
            return expected_error
        finally:
            self._shutdown_module(module, sys.exc_info()[1], expected_error)

    def _capture_if_raised(
        self,
        operation: Callable[[], object],
        expectation: _UnsupportedExpectation,
    ) -> Exception | None:
        try:
            operation()
        except expectation.exception_type as expected_error:
            return expected_error
        return None

    def _capture_expected_error(
        self,
        operation: Callable[[], object],
        expectation: _UnsupportedExpectation,
        failure_message: str,
    ) -> Exception:
        try:
            operation()
        except expectation.exception_type as expected_error:
            return expected_error
        self.fail(failure_message)

    def _assert_matches_expectation(
        self, error: Exception, expectation: _UnsupportedExpectation
    ) -> None:
        self.assertIsInstance(error, expectation.exception_type)
        if expectation.message is not None:
            self.assertIn(expectation.message, str(error))
        if expectation.exception_args is not None:
            self.assertEqual(expectation.exception_args, error.args)
        if expectation.traceback_location is not None:
            filename, function_name = expectation.traceback_location
            frames = traceback.extract_tb(error.__traceback__)
            self.assertTrue(
                any(
                    frame.filename.endswith(filename) and frame.name == function_name
                    for frame in frames
                ),
                traceback.format_list(frames),
            )

    def _assert_unsupported_metric(
        self,
        metric_enum: RecMetricEnum,
        case: _MetricCase,
        compute_mode: RecComputeMode,
        expectation: _UnsupportedExpectation,
    ) -> None:
        for update_batch_size in (None, 1, self.K):
            arm = (
                "standard"
                if update_batch_size is None
                else f"zorm_k{update_batch_size}"
            )
            with self.subTest(
                metric=metric_enum.value,
                compute_mode=compute_mode.name,
                arm=arm,
            ):
                error = self._run_unsupported_case(
                    metric_enum,
                    case.input_kind,
                    compute_mode,
                    update_batch_size,
                    expectation,
                )
                self._assert_matches_expectation(error, expectation)

    def _assert_unsupported_cases(
        self,
        compute_mode: RecComputeMode,
        metric_cases: Iterable[tuple[RecMetricEnum, _MetricCase]],
    ) -> None:
        unsupported_cases = [
            (metric_enum, case)
            for metric_enum, case in metric_cases
            if compute_mode not in case.supported_modes
        ]
        missing_expectations = [
            metric_enum.value
            for metric_enum, _case in unsupported_cases
            if metric_enum not in _UNSUPPORTED_EXPECTATIONS
        ]
        self.assertEqual([], missing_expectations)
        for metric_enum, case in unsupported_cases:
            self._assert_unsupported_metric(
                metric_enum,
                case,
                compute_mode,
                _UNSUPPORTED_EXPECTATIONS[metric_enum],
            )

    def _metric_cases_for_shard(
        self, shard_index: int
    ) -> list[tuple[RecMetricEnum, _MetricCase]]:
        metric_cases = sorted(
            _METRIC_CASES.items(),
            key=lambda item: item[0].value,
        )
        return metric_cases[shard_index::_SHARD_COUNT]

    def _assert_mode_behavior(
        self, compute_mode: RecComputeMode, shard_index: int
    ) -> None:
        metric_cases = self._metric_cases_for_shard(shard_index)
        self._assert_unsupported_cases(compute_mode, metric_cases)
        self.assertEqual({}, self._behavior_failures(compute_mode, metric_cases))

    def test_registry_covers_every_metric_and_compute_mode(self) -> None:
        self.assertEqual(set(REC_METRICS_MAPPING), set(_METRIC_CASES))
        self.assertEqual(set(RecComputeMode), set(_ALL_COMPUTE_MODES))
        for shard_index in range(_SHARD_COUNT):
            with self.subTest(shard_index=shard_index):
                self.assertTrue(self._metric_cases_for_shard(shard_index))
        unsupported_metrics = {
            metric_enum
            for metric_enum, case in _METRIC_CASES.items()
            if set(case.supported_modes) != set(_ALL_COMPUTE_MODES)
        }
        self.assertEqual(unsupported_metrics, set(_UNSUPPORTED_EXPECTATIONS))
        for metric_enum, case in _METRIC_CASES.items():
            with self.subTest(metric=metric_enum.value):
                self.assertLessEqual(
                    set(case.equivalent_modes), set(case.supported_modes)
                )
                has_non_equivalent_mode = bool(
                    set(case.supported_modes) - set(case.equivalent_modes)
                )
                self.assertEqual(
                    has_non_equivalent_mode,
                    case.expected_differences is not None,
                )

    @parameterized.expand(
        [
            (compute_mode.name.lower(), compute_mode)
            for compute_mode in _ALL_COMPUTE_MODES
        ]
    )
    def test_three_arm_behavior_with_multiple_metrics(
        self,
        _name: str,
        compute_mode: RecComputeMode,
    ) -> None:
        metric_enums = list(_MULTI_METRIC_CASE)
        standard = self._run_arm(metric_enums, _InputKind.BINARY, compute_mode, None)
        zorm_k1 = self._run_arm(metric_enums, _InputKind.BINARY, compute_mode, 1)
        self.assertIsNone(self._arm_differences(standard, zorm_k1), "zorm_k1")

        zorm_k10 = self._run_arm(metric_enums, _InputKind.BINARY, compute_mode, self.K)
        k10_difference = self._arm_differences(standard, zorm_k10)
        self.assertIsNone(k10_difference, "zorm_k10")

    @parameterized.expand(
        [
            (
                f"{compute_mode.name.lower()}_{shard_index}",
                compute_mode,
                shard_index,
            )
            for compute_mode in _ALL_COMPUTE_MODES
            for shard_index in range(_SHARD_COUNT)
        ]
    )
    def test_three_arm_behavior(
        self,
        _name: str,
        compute_mode: RecComputeMode,
        shard_index: int,
    ) -> None:
        self._assert_mode_behavior(compute_mode, shard_index)
