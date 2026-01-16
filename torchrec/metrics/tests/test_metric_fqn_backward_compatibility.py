#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
These tests compares current metric FQNs against a golden baseline and fails if:
1. Any state_dict key is REMOVED (breaks loading old checkpoints into new code)
2. Any state_dict key is ADDED (breaks loading old checkpoints in DCP clients
   unless allow_partial_load=True, which most production trainers don't use)

## How to Fix Breaking Changes

If you need to add a new buffer/state to a metric:
1. Consider if it can be non-persistent (won't appear in state_dict)
2. If it must be persistent, coordinate with the trainers team to enable
   allow_partial_load for metrics, OR add a migration path
3. Update the golden snapshot with --update-golden after confirming the change
   won't break production training jobs

To update the golden snapshot after intentional changes:
    python -m torchrec.metrics.tests.test_metric_fqn_backward_compatibility --update-golden
"""

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
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
from torchrec.metrics.gauc import GAUCMetric
from torchrec.metrics.hindsight_target_pr import HindsightTargetPRMetric
from torchrec.metrics.mae import MAEMetric
from torchrec.metrics.metric_module import RecMetricModule
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.multi_label_precision import MultiLabelPrecisionMetric
from torchrec.metrics.multiclass_recall import MulticlassRecallMetric
from torchrec.metrics.ndcg import NDCGMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.ne_positive import NEPositiveMetric
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric
from torchrec.metrics.nmse import NMSEMetric
from torchrec.metrics.output import OutputMetric
from torchrec.metrics.precision import PrecisionMetric
from torchrec.metrics.precision_session import PrecisionSessionMetric
from torchrec.metrics.rauc import RAUCMetric
from torchrec.metrics.rec_metric import RecMetric, RecMetricList
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


# Path to the golden snapshot file
GOLDEN_SNAPSHOT_PATH = Path(__file__).parent / "metric_fqn_golden_snapshot.json"


def create_test_task(
    task_name: str = "test_task",
    with_tensor_name: bool = False,
    with_session_metric_def: bool = False,
) -> RecTaskInfo:
    from torchrec.metrics.metrics_config import SessionMetricDef

    session_metric_def = None
    if with_session_metric_def:
        session_metric_def = SessionMetricDef(
            session_var_name=f"{task_name}-session",
            top_threshold=1,
            run_ranking_of_labels=False,
        )

    return RecTaskInfo(
        name=task_name,
        label_name=f"{task_name}-label",
        prediction_name=f"{task_name}-prediction",
        weight_name=f"{task_name}-weight",
        tensor_name=f"{task_name}-tensor" if with_tensor_name else None,
        session_metric_def=session_metric_def,
    )


def extract_state_dict_keys(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    task_names: Optional[List[str]] = None,
    use_tensor_task: bool = False,
    use_session_task: bool = False,
    **kwargs: Any,
) -> List[str]:
    if task_names is None:
        task_names = ["test_task"]

    tasks = [
        create_test_task(
            name,
            with_tensor_name=use_tensor_task,
            with_session_metric_def=use_session_task,
        )
        for name in task_names
    ]

    metric = metric_class(
        world_size=1,
        my_rank=0,
        batch_size=32,
        tasks=tasks,
        compute_mode=compute_mode,
        window_size=100,
        fused_update_limit=0,
        **kwargs,
    )

    state_dict = metric.state_dict()
    return sorted(state_dict.keys())


def extract_named_buffer_fqns(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    task_names: Optional[List[str]] = None,
    use_tensor_task: bool = False,
    use_session_task: bool = False,
    **kwargs: Any,
) -> Tuple[List[str], List[str]]:
    if task_names is None:
        task_names = ["test_task"]

    tasks = [
        create_test_task(
            name,
            with_tensor_name=use_tensor_task,
            with_session_metric_def=use_session_task,
        )
        for name in task_names
    ]

    metric = metric_class(
        world_size=1,
        my_rank=0,
        batch_size=32,
        tasks=tasks,
        compute_mode=compute_mode,
        window_size=100,
        fused_update_limit=0,
        **kwargs,
    )

    all_buffer_fqns = [name for name, _ in metric.named_buffers()]
    state_dict_keys = set(metric.state_dict().keys())

    persistent = []
    non_persistent = []

    for fqn in all_buffer_fqns:
        is_persistent = any(
            fqn == key or key.endswith(fqn) or fqn in key for key in state_dict_keys
        )
        if is_persistent:
            persistent.append(fqn)
        else:
            non_persistent.append(fqn)

    return sorted(persistent), sorted(non_persistent)


def get_metric_snapshot_key(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode,
    variant: str = "",
) -> str:
    key = f"{metric_class.__name__}_{compute_mode.name}"
    if variant:
        key = f"{key}_{variant}"
    return key


# List of metrics to test with their configurations
# Format: (metric_class, compute_modes_to_test, extra_kwargs, variants)
# ThroughputMetric is excluded as it's not a RecMetric subclass (it's nn.Module)
METRICS_TO_TEST: List[
    Tuple[Type[RecMetric], List[RecComputeMode], Dict[str, Any], List[str]]
] = [
    # Core metrics with persistent state
    (NEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        NEMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"include_logloss": True},
        ["with_logloss"],
    ),
    (CalibrationMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (CTRMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (MSEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (MAEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (WeightedAvgMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AccuracyMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (PrecisionMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RecallMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (TowerQPSMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NMSEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AverageMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (HindsightTargetPRMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NDCGMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (XAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ScalarMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        MultiLabelPrecisionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"num_labels": 1},
        [""],
    ),
    # Metrics with non-persistent state (AUC family)
    (AUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AUPRCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (GAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    # TensorWeightedAvgMetric requires tensor_name in tasks
    (
        TensorWeightedAvgMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_tensor_task": True},
        [""],
    ),
    (CaliFreeNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NEPositiveMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ServingNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (UnweightedNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RecalibratedNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ServingCalibrationMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        RecalibratedCalibrationMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {},
        [""],
    ),
    (OutputMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    # MulticlassRecallMetric requires number_of_classes
    (
        MulticlassRecallMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"number_of_classes": 3},
        [""],
    ),
    # SegmentedNEMetric requires num_groups and grouping_keys
    (
        SegmentedNEMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"num_groups": 2, "grouping_keys": "test_task-grouping"},
        [""],
    ),
    # Session-level metrics require session_metric_def in tasks
    (
        PrecisionSessionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_session_task": True},
        [""],
    ),
    (
        RecallSessionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_session_task": True},
        [""],
    ),
    # FUSED mode tests
    (NEMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
    (CalibrationMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
    (WeightedAvgMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
]


def generate_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Generate a complete golden snapshot of all metrics.

    Returns:
        Dictionary mapping metric keys to their FQN information.
    """
    snapshot: Dict[str, Dict[str, Any]] = {}

    for metric_class, compute_modes, kwargs, variants in METRICS_TO_TEST:
        for compute_mode in compute_modes:
            for variant in variants:
                try:
                    key = get_metric_snapshot_key(metric_class, compute_mode, variant)
                    state_dict_keys = extract_state_dict_keys(
                        metric_class, compute_mode, **kwargs
                    )
                    persistent_fqns, non_persistent_fqns = extract_named_buffer_fqns(
                        metric_class, compute_mode, **kwargs
                    )

                    snapshot[key] = {
                        "metric_class": metric_class.__name__,
                        "compute_mode": compute_mode.name,
                        "variant": variant,
                        "state_dict_keys": state_dict_keys,
                        "persistent_buffer_fqns": persistent_fqns,
                        "non_persistent_buffer_fqns": non_persistent_fqns,
                    }
                except Exception as e:
                    print(f"Warning: Failed to generate snapshot for {key}: {e}")
                    continue

    return snapshot


def load_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    if not GOLDEN_SNAPSHOT_PATH.exists():
        return {}
    with open(GOLDEN_SNAPSHOT_PATH, "r") as f:
        return json.load(f)


def save_golden_snapshot(snapshot: Dict[str, Dict[str, Any]]) -> None:
    with open(GOLDEN_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")


class MetricFQNBackwardCompatibilityTest(unittest.TestCase):
    """
    Test suite for metric FQN backward compatibility.

    These tests ensure that changes to metrics don't break checkpoint loading
    by verifying that:
    1. No persistent buffer FQNs are removed
    2. No state_dict keys are removed

    New additions are allowed as long as they have proper load_state_dict hooks
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_golden_snapshot()
        if not cls.golden_snapshot:
            print("No golden snapshot found. Generating initial snapshot...")
            cls.golden_snapshot = generate_golden_snapshot()
            save_golden_snapshot(cls.golden_snapshot)
            print(f"Golden snapshot saved to {GOLDEN_SNAPSHOT_PATH}")
        cls.current_snapshot = None

    def _check_metric_compatibility(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode,
        variant: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Check a single metric for backward compatibility with DCP CheckpointClient.

        This test simulates what happens when a new model version (with potentially
        new state_dict keys) tries to load an old checkpoint:

        1. REMOVED keys: Old checkpoint has keys that new model doesn't expect.
           - This typically works fine (load_state_dict ignores extra keys)
           - But indicates the metric structure changed unexpectedly

        2. ADDED keys: New model has keys that old checkpoint doesn't have.
           - This BREAKS DCP CheckpointClient! The client validates that ALL model
             FQNs exist in the checkpoint metadata before loading.
           - Raises InvalidParamQualNameException at the DCP level.

        Raises assertion error if incompatible changes are detected.
        """
        key = get_metric_snapshot_key(metric_class, compute_mode, variant)

        if key not in self.golden_snapshot:
            self.skipTest(
                f"No golden snapshot for {key}. Run with --update-golden to create."
            )

        baseline = self.golden_snapshot[key]

        current_state_dict_keys = set(
            extract_state_dict_keys(metric_class, compute_mode, **kwargs)
        )
        current_persistent_fqns, _ = extract_named_buffer_fqns(
            metric_class, compute_mode, **kwargs
        )
        current_persistent_fqns_set = set(current_persistent_fqns)

        baseline_state_dict_keys = set(baseline["state_dict_keys"])
        baseline_persistent_fqns = set(baseline["persistent_buffer_fqns"])

        removed_keys = baseline_state_dict_keys - current_state_dict_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}. "
                "This will cause old checkpoints to fail loading. "
                "If this is intentional, update the golden snapshot."
            )

        removed_fqns = baseline_persistent_fqns - current_persistent_fqns_set
        if removed_fqns:
            self.fail(
                f"BREAKING CHANGE in {key}: persistent buffer FQNs removed: {sorted(removed_fqns)}. "
                "This will cause old checkpoints to fail loading. "
                "If this is intentional, update the golden snapshot."
            )

        added_keys = current_state_dict_keys - baseline_state_dict_keys
        added_fqns = current_persistent_fqns_set - baseline_persistent_fqns

        if added_keys or added_fqns:
            has_backward_compat = self._verify_backward_compatibility(
                metric_class, compute_mode, added_keys, **kwargs
            )

            if not has_backward_compat:
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}.\n"
                    "This will cause DCP CheckpointClient to fail with InvalidParamQualNameException "
                    "when loading old checkpoints (the client validates that ALL model FQNs exist "
                    "in checkpoint metadata before loading).\n\n"
                    "To fix:\n"
                    "1. Make the new state non-persistent (use persistent=False in add_state)\n"
                    "2. OR coordinate with trainers to enable allow_partial_load for metrics\n"
                    "3. OR run with --update-golden if this change is intentional and coordinated"
                )

    def _verify_backward_compatibility(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode,
        added_keys: Set[str],
        **kwargs: Any,
    ) -> bool:
        """
        Check if added state_dict keys are backward compatible.

        Note: While torchmetrics.Metric handles missing keys gracefully at the
        PyTorch load_state_dict level (keeping default values), this does NOT
        help at the DCP CheckpointClient level. The DCP client validates FQNs
        BEFORE calling load_state_dict, so it fails before PyTorch hooks run.

        For added keys, we always return False to require explicit acknowledgment:
        1. Confirm the change is intentional
        2. Verify coordination with trainer teams
        3. Update the golden snapshot with --update-golden

        Returns:
            True if no added keys (backward compatible)
            False if there are added keys (requires developer acknowledgment)
        """
        # If there are added keys, require user to update golden snapshot
        return len(added_keys) == 0

    def test_ne_metric_unfused(self) -> None:
        self._check_metric_compatibility(
            NEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_ne_metric_fused(self) -> None:
        self._check_metric_compatibility(
            NEMetric, RecComputeMode.FUSED_TASKS_COMPUTATION
        )

    def test_ne_metric_with_logloss(self) -> None:
        self._check_metric_compatibility(
            NEMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            variant="with_logloss",
            include_logloss=True,
        )

    def test_calibration_metric_unfused(self) -> None:
        self._check_metric_compatibility(
            CalibrationMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_calibration_metric_fused(self) -> None:
        self._check_metric_compatibility(
            CalibrationMetric, RecComputeMode.FUSED_TASKS_COMPUTATION
        )

    def test_ctr_metric(self) -> None:
        self._check_metric_compatibility(
            CTRMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_mse_metric(self) -> None:
        self._check_metric_compatibility(
            MSEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_mae_metric(self) -> None:
        self._check_metric_compatibility(
            MAEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_weighted_avg_metric_unfused(self) -> None:
        self._check_metric_compatibility(
            WeightedAvgMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_weighted_avg_metric_fused(self) -> None:
        self._check_metric_compatibility(
            WeightedAvgMetric, RecComputeMode.FUSED_TASKS_COMPUTATION
        )

    def test_accuracy_metric(self) -> None:
        self._check_metric_compatibility(
            AccuracyMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_precision_metric(self) -> None:
        self._check_metric_compatibility(
            PrecisionMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_recall_metric(self) -> None:
        self._check_metric_compatibility(
            RecallMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_tower_qps_metric(self) -> None:
        self._check_metric_compatibility(
            TowerQPSMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_nmse_metric(self) -> None:
        self._check_metric_compatibility(
            NMSEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_average_metric(self) -> None:
        self._check_metric_compatibility(
            AverageMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_hindsight_target_pr_metric(self) -> None:
        self._check_metric_compatibility(
            HindsightTargetPRMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_auc_metric(self) -> None:
        self._check_metric_compatibility(
            AUCMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_auprc_metric(self) -> None:
        self._check_metric_compatibility(
            AUPRCMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_rauc_metric(self) -> None:
        self._check_metric_compatibility(
            RAUCMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_gauc_metric(self) -> None:
        self._check_metric_compatibility(
            GAUCMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_ndcg_metric(self) -> None:
        self._check_metric_compatibility(
            NDCGMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_xauc_metric(self) -> None:
        self._check_metric_compatibility(
            XAUCMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_scalar_metric(self) -> None:
        self._check_metric_compatibility(
            ScalarMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_tensor_weighted_avg_metric(self) -> None:
        self._check_metric_compatibility(
            TensorWeightedAvgMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            variant="",
            use_tensor_task=True,
        )

    def test_cali_free_ne_metric(self) -> None:
        self._check_metric_compatibility(
            CaliFreeNEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_ne_positive_metric(self) -> None:
        self._check_metric_compatibility(
            NEPositiveMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_serving_ne_metric(self) -> None:
        self._check_metric_compatibility(
            ServingNEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_unweighted_ne_metric(self) -> None:
        self._check_metric_compatibility(
            UnweightedNEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_recalibrated_ne_metric(self) -> None:
        self._check_metric_compatibility(
            RecalibratedNEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_serving_calibration_metric(self) -> None:
        self._check_metric_compatibility(
            ServingCalibrationMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_recalibrated_calibration_metric(self) -> None:
        self._check_metric_compatibility(
            RecalibratedCalibrationMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_output_metric(self) -> None:
        self._check_metric_compatibility(
            OutputMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    def test_multiclass_recall_metric(self) -> None:
        self._check_metric_compatibility(
            MulticlassRecallMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            number_of_classes=3,
        )

    def test_multi_label_precision_metric(self) -> None:
        self._check_metric_compatibility(
            MultiLabelPrecisionMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            num_labels=1,
        )

    def test_segmented_ne_metric(self) -> None:
        self._check_metric_compatibility(
            SegmentedNEMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            num_groups=2,
            grouping_keys="test_task-grouping",
        )

    def test_precision_session_metric(self) -> None:
        self._check_metric_compatibility(
            PrecisionSessionMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            use_session_task=True,
        )

    def test_recall_session_metric(self) -> None:
        self._check_metric_compatibility(
            RecallSessionMetric,
            RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            use_session_task=True,
        )


class MetricStateSnapshotTest(unittest.TestCase):
    """
    Test suite for verifying metric state snapshot capabilities.

    These tests verify that metrics properly register their states
    and can be serialized/deserialized correctly.
    """

    def _test_metric_state_roundtrip(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        **kwargs: Any,
    ) -> None:
        tasks = [create_test_task("task1")]

        original = metric_class(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=100,
            **kwargs,
        )

        initial_state = original.state_dict()

        restored = metric_class(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=100,
            **kwargs,
        )

        restored.load_state_dict(initial_state, strict=True)

        restored_state = restored.state_dict()
        self.assertEqual(
            set(initial_state.keys()),
            set(restored_state.keys()),
            f"State dict keys mismatch for {metric_class.__name__}",
        )

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key} in {metric_class.__name__}",
            )

    def test_ne_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(NEMetric)

    def test_calibration_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(CalibrationMetric)

    def test_mse_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(MSEMetric)

    def test_weighted_avg_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(WeightedAvgMetric)


# Golden snapshot keys for ThroughputMetric and RecMetricModule
THROUGHPUT_SNAPSHOT_KEY = "ThroughputMetric"
THROUGHPUT_WITH_STAGES_SNAPSHOT_KEY = "ThroughputMetric_with_batch_size_stages"
REC_METRIC_MODULE_SNAPSHOT_KEY = "RecMetricModule"
REC_METRIC_MODULE_WITH_THROUGHPUT_KEY = "RecMetricModule_with_throughput"


class ThroughputMetricBackwardCompatibilityTest(unittest.TestCase):
    """
    Test suite for ThroughputMetric FQN backward compatibility.

    ThroughputMetric is an nn.Module (not RecMetric) with its own state_dict structure.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_golden_snapshot()

    def _create_throughput_metric(
        self, with_batch_size_stages: bool = False
    ) -> ThroughputMetric:
        from torchrec.metrics.metrics_config import BatchSizeStage

        batch_size_stages = None
        if with_batch_size_stages:
            batch_size_stages = [
                BatchSizeStage(batch_size=32, max_iters=100),
                BatchSizeStage(batch_size=64, max_iters=None),
            ]

        return ThroughputMetric(
            batch_size=32,
            world_size=1,
            window_seconds=100,
            warmup_steps=10,
            batch_size_stages=batch_size_stages,
        )

    def _get_state_dict_keys(self, with_batch_size_stages: bool = False) -> List[str]:
        metric = self._create_throughput_metric(with_batch_size_stages)
        return sorted(metric.state_dict().keys())

    def test_throughput_metric_compatibility(self) -> None:
        key = THROUGHPUT_SNAPSHOT_KEY

        if key not in self.golden_snapshot:
            current_keys = self._get_state_dict_keys(with_batch_size_stages=False)
            self.golden_snapshot[key] = {
                "metric_class": "ThroughputMetric",
                "variant": "",
                "state_dict_keys": current_keys,
                "persistent_buffer_fqns": [],
                "non_persistent_buffer_fqns": [],
            }
            save_golden_snapshot(self.golden_snapshot)
            return

        baseline = self.golden_snapshot[key]
        current_keys = set(self._get_state_dict_keys(with_batch_size_stages=False))
        baseline_keys = set(baseline["state_dict_keys"])

        removed_keys = baseline_keys - current_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}. "
                "This will cause old checkpoints to fail loading."
            )

        added_keys = current_keys - baseline_keys
        if added_keys:
            if not self._verify_backward_compatibility(added_keys):
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}. "
                    "Implement load_state_dict hooks to handle missing keys."
                )

    def test_throughput_metric_with_stages_compatibility(self) -> None:
        key = THROUGHPUT_WITH_STAGES_SNAPSHOT_KEY

        if key not in self.golden_snapshot:
            current_keys = self._get_state_dict_keys(with_batch_size_stages=True)
            self.golden_snapshot[key] = {
                "metric_class": "ThroughputMetric",
                "variant": "with_batch_size_stages",
                "state_dict_keys": current_keys,
                "persistent_buffer_fqns": [],
                "non_persistent_buffer_fqns": [],
            }
            save_golden_snapshot(self.golden_snapshot)
            return

        baseline = self.golden_snapshot[key]
        current_keys = set(self._get_state_dict_keys(with_batch_size_stages=True))
        baseline_keys = set(baseline["state_dict_keys"])

        removed_keys = baseline_keys - current_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}. "
                "This will cause old checkpoints to fail loading."
            )

        added_keys = current_keys - baseline_keys
        if added_keys:
            if not self._verify_backward_compatibility(added_keys, with_stages=True):
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}. "
                    "Implement load_state_dict hooks to handle missing keys."
                )

    def _verify_backward_compatibility(
        self, added_keys: Set[str], with_stages: bool = False
    ) -> bool:
        try:
            metric = self._create_throughput_metric(with_batch_size_stages=with_stages)
            state_dict = metric.state_dict()
            old_checkpoint = {
                k: v for k, v in state_dict.items() if k not in added_keys
            }

            fresh_metric = self._create_throughput_metric(
                with_batch_size_stages=with_stages
            )
            fresh_metric.load_state_dict(old_checkpoint, strict=True)
            return True
        except Exception:
            return False

    def test_throughput_metric_state_roundtrip(self) -> None:
        metric = self._create_throughput_metric()
        initial_state = metric.state_dict()

        fresh_metric = self._create_throughput_metric()
        fresh_metric.load_state_dict(initial_state, strict=True)

        restored_state = fresh_metric.state_dict()
        self.assertEqual(set(initial_state.keys()), set(restored_state.keys()))

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key}",
            )


class RecMetricModuleBackwardCompatibilityTest(unittest.TestCase):
    """
    Test suite for RecMetricModule FQN backward compatibility.

    RecMetricModule is a container that holds RecMetrics, ThroughputMetric, and StateMetrics.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load the golden snapshot."""
        cls.golden_snapshot = load_golden_snapshot()

    def _create_rec_metric_module(
        self, with_throughput: bool = False, with_metrics: bool = True
    ) -> RecMetricModule:
        tasks = [create_test_task("task1")]

        rec_metrics = None
        if with_metrics:
            ne_metric = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=32,
                tasks=tasks,
                compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                window_size=100,
            )
            rec_metrics = RecMetricList([ne_metric])

        throughput_metric = None
        if with_throughput:
            throughput_metric = ThroughputMetric(
                batch_size=32,
                world_size=1,
                window_seconds=100,
                warmup_steps=10,
            )

        return RecMetricModule(
            batch_size=32,
            world_size=1,
            rec_tasks=tasks,
            rec_metrics=rec_metrics,
            throughput_metric=throughput_metric,
        )

    def _get_state_dict_keys(
        self, with_throughput: bool = False, with_metrics: bool = True
    ) -> List[str]:
        module = self._create_rec_metric_module(
            with_throughput=with_throughput, with_metrics=with_metrics
        )
        return sorted(module.state_dict().keys())

    def test_rec_metric_module_compatibility(self) -> None:
        key = REC_METRIC_MODULE_SNAPSHOT_KEY

        if key not in self.golden_snapshot:
            current_keys = self._get_state_dict_keys(with_throughput=False)
            self.golden_snapshot[key] = {
                "metric_class": "RecMetricModule",
                "variant": "",
                "state_dict_keys": current_keys,
                "persistent_buffer_fqns": [],
                "non_persistent_buffer_fqns": [],
            }
            save_golden_snapshot(self.golden_snapshot)
            return

        baseline = self.golden_snapshot[key]
        current_keys = set(self._get_state_dict_keys(with_throughput=False))
        baseline_keys = set(baseline["state_dict_keys"])

        removed_keys = baseline_keys - current_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}."
            )

        added_keys = current_keys - baseline_keys
        if added_keys:
            if not self._verify_backward_compatibility(
                added_keys, with_throughput=False
            ):
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}. "
                    "Implement load_state_dict hooks to handle missing keys."
                )

    def test_rec_metric_module_with_throughput_compatibility(self) -> None:
        key = REC_METRIC_MODULE_WITH_THROUGHPUT_KEY

        if key not in self.golden_snapshot:
            current_keys = self._get_state_dict_keys(with_throughput=True)
            self.golden_snapshot[key] = {
                "metric_class": "RecMetricModule",
                "variant": "with_throughput",
                "state_dict_keys": current_keys,
                "persistent_buffer_fqns": [],
                "non_persistent_buffer_fqns": [],
            }
            save_golden_snapshot(self.golden_snapshot)
            return

        baseline = self.golden_snapshot[key]
        current_keys = set(self._get_state_dict_keys(with_throughput=True))
        baseline_keys = set(baseline["state_dict_keys"])

        removed_keys = baseline_keys - current_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}."
            )

        added_keys = current_keys - baseline_keys
        if added_keys:
            if not self._verify_backward_compatibility(
                added_keys, with_throughput=True
            ):
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}. "
                    "Implement load_state_dict hooks to handle missing keys."
                )

    def _verify_backward_compatibility(
        self, added_keys: Set[str], with_throughput: bool = False
    ) -> bool:
        try:
            module = self._create_rec_metric_module(with_throughput=with_throughput)
            state_dict = module.state_dict()
            old_checkpoint = {
                k: v for k, v in state_dict.items() if k not in added_keys
            }

            fresh_module = self._create_rec_metric_module(
                with_throughput=with_throughput
            )
            fresh_module.load_state_dict(old_checkpoint, strict=True)
            return True
        except Exception:
            return False

    def test_rec_metric_module_state_roundtrip(self) -> None:
        module = self._create_rec_metric_module(with_throughput=True)
        initial_state = module.state_dict()

        fresh_module = self._create_rec_metric_module(with_throughput=True)
        fresh_module.load_state_dict(initial_state, strict=True)

        restored_state = fresh_module.state_dict()
        self.assertEqual(set(initial_state.keys()), set(restored_state.keys()))

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key}",
            )

    def test_rec_metric_module_backward_compat_trained_batches(self) -> None:
        module = self._create_rec_metric_module()
        state_dict = module.state_dict()

        state_dict["_trained_batches"] = torch.tensor(100)

        fresh_module = self._create_rec_metric_module()
        fresh_module.load_state_dict(state_dict, strict=False)


class MetricCoverageTest(unittest.TestCase):
    """
    Test that ensures all RecMetric subclasses are covered by backward compatibility tests.

    This test will FAIL if a new metric is added to torchrec but not added to METRICS_TO_TEST.
    When adding a new metric, users must add it to METRICS_TO_TEST in this file.
    """

    # Metrics that are intentionally excluded from testing (with reason)
    EXCLUDED_METRICS: Dict[str, str] = {
        # Add metrics here that should be excluded, with a reason
        # e.g., "SomeMetric": "deprecated, will be removed in next release",
    }

    def test_all_recmetrics_are_covered(self) -> None:
        import importlib
        import pkgutil

        import torchrec.metrics

        discovered_metrics: Set[str] = set()

        package_path = torchrec.metrics.__path__
        for _, module_name, _ in pkgutil.iter_modules(package_path):
            try:
                module = importlib.import_module(f"torchrec.metrics.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, RecMetric)
                        and attr is not RecMetric
                        and not attr_name.startswith("_")
                    ):
                        discovered_metrics.add(attr_name)
            except ImportError:
                continue

        covered_metrics: Set[str] = {
            metric_class.__name__ for metric_class, _, _, _ in METRICS_TO_TEST
        }

        missing_metrics = (
            discovered_metrics - covered_metrics - set(self.EXCLUDED_METRICS.keys())
        )

        if missing_metrics:
            self.fail(
                f"The following RecMetric subclasses are not covered by backward "
                f"compatibility tests: {sorted(missing_metrics)}.\n\n"
                f"To fix this:\n"
                f"1. Add the metric to METRICS_TO_TEST in this file\n"
                f"2. Add a corresponding test_<metric>_metric() method\n"
                f"3. Run the tests to generate the golden snapshot\n\n"
                f"If the metric should be excluded, add it to EXCLUDED_METRICS with a reason."
            )


class DCPCheckpointClientSimulationTest(unittest.TestCase):
    """
    Test suite that simulates DCP CheckpointClient FQN validation behavior.

    This test explicitly simulates what happens at the DCP level when loading
    checkpoints. The DCP CheckpointClient (in aiplatform/modelstore) validates
    that ALL FQNs in the model's state_dict exist in the checkpoint metadata
    BEFORE calling load_state_dict.

    This is different from PyTorch's load_state_dict behavior:
    - PyTorch: Fails if checkpoint has keys model doesn't expect (with strict=True)
    - DCP: Fails if MODEL has keys that CHECKPOINT doesn't have

    When a new buffer is added to a metric:
    - PyTorch load_state_dict: May work (torchmetrics handles missing keys)
    - DCP CheckpointClient: FAILS with InvalidParamQualNameException

    See: aiplatform/modelstore/experimental/DCP/planners/planner_utils.py
    Function: is_loading_param_fqn_in_cp_metadata()
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_golden_snapshot()

    def _simulate_dcp_fqn_validation(
        self,
        model_fqns: Set[str],
        checkpoint_fqns: Set[str],
        allow_partial_load: bool = False,
    ) -> Tuple[bool, Set[str]]:
        """
        Simulate DCP CheckpointClient FQN validation.

        This mirrors the logic in is_loading_param_fqn_in_cp_metadata():
        - For each FQN in the model's state_dict
        - Check if it exists in the checkpoint metadata
        - If not, and allow_partial_load=False, raise InvalidParamQualNameException

        Args:
            model_fqns: FQNs from the current model's state_dict
            checkpoint_fqns: FQNs from the saved checkpoint (golden snapshot)
            allow_partial_load: If True, skip validation for missing FQNs

        Returns:
            Tuple of (would_succeed, missing_fqns)
        """
        missing_in_checkpoint = model_fqns - checkpoint_fqns

        if allow_partial_load:
            # DCP logs a warning but continues
            return True, missing_in_checkpoint

        # DCP raises InvalidParamQualNameException
        would_succeed = len(missing_in_checkpoint) == 0
        return would_succeed, missing_in_checkpoint

    def test_ne_metric_dcp_validation(self) -> None:
        key = get_metric_snapshot_key(
            NEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )
        if key not in self.golden_snapshot:
            self.skipTest(f"No golden snapshot for {key}")

        current_fqns = set(
            extract_state_dict_keys(NEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION)
        )
        checkpoint_fqns = set(self.golden_snapshot[key]["state_dict_keys"])

        would_succeed, missing = self._simulate_dcp_fqn_validation(
            current_fqns, checkpoint_fqns, allow_partial_load=False
        )

        if not would_succeed:
            self.fail(
                f"DCP CheckpointClient would fail for NEMetric!\n"
                f"Model has FQNs not in checkpoint: {sorted(missing)}\n"
                f"This simulates InvalidParamQualNameException at load time."
            )

    def test_rec_metric_module_dcp_validation(self) -> None:
        key = REC_METRIC_MODULE_SNAPSHOT_KEY
        if key not in self.golden_snapshot:
            self.skipTest(f"No golden snapshot for {key}")

        tasks = [create_test_task("task1")]
        ne_metric = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )
        rec_metrics = RecMetricList([ne_metric])

        module = RecMetricModule(
            batch_size=32,
            world_size=1,
            rec_tasks=tasks,
            rec_metrics=rec_metrics,
        )

        current_fqns = set(module.state_dict().keys())
        checkpoint_fqns = set(self.golden_snapshot[key]["state_dict_keys"])

        would_succeed, missing = self._simulate_dcp_fqn_validation(
            current_fqns, checkpoint_fqns, allow_partial_load=False
        )

        if not would_succeed:
            self.fail(
                f"DCP CheckpointClient would fail for RecMetricModule!\n"
                f"Model has FQNs not in checkpoint: {sorted(missing)}\n"
                f"This simulates InvalidParamQualNameException at load time.\n\n"
                f"To fix: Update golden snapshot with --update-golden after "
                f"coordinating with trainer teams."
            )


def update_golden_snapshot() -> None:
    print("Generating golden snapshot...")
    snapshot = generate_golden_snapshot()
    save_golden_snapshot(snapshot)
    print(f"Golden snapshot saved to {GOLDEN_SNAPSHOT_PATH}")
    print(f"Total metrics captured: {len(snapshot)}")
    for key in sorted(snapshot.keys()):
        info = snapshot[key]
        print(
            f"  - {key}: {len(info['state_dict_keys'])} state_dict keys, "
            f"{len(info['persistent_buffer_fqns'])} persistent buffers"
        )


if __name__ == "__main__":
    if "--update-golden" in sys.argv or os.environ.get("UPDATE_GOLDEN_SNAPSHOT"):
        if "--update-golden" in sys.argv:
            sys.argv.remove("--update-golden")
        update_golden_snapshot()
    else:
        unittest.main()
