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

To update the golden snapshot after intentional changes, from fbcode:
    buck2 run //torchrec/metrics/tests:update_metric_fqn_golden_snapshot -- --update-golden
"""

import contextlib
import copy
import inspect
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

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
from torchrec.metrics.cpu_comms_metric_module import CPUCommsRecMetricModule
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.gauc import GAUCMetric
from torchrec.metrics.hindsight_target_pr import HindsightTargetPRMetric
from torchrec.metrics.mae import MAEMetric
from torchrec.metrics.metric_module import RecMetricModule
from torchrec.metrics.metrics_config import BatchSizeStage, RecComputeMode, RecTaskInfo
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.multi_label_precision import MultiLabelPrecisionMetric
from torchrec.metrics.multiclass_recall import MulticlassRecallMetric
from torchrec.metrics.ndcg import NDCGMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.ne_positive import NEPositiveMetric
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric
from torchrec.metrics.nmse import NMSEMetric
from torchrec.metrics.noop_metric_module import NoOpMetricModule
from torchrec.metrics.num_missing_labels import NumMissingLabelsMetric
from torchrec.metrics.num_positive_samples import NumPositiveSamplesMetric
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
from torchrec.metrics.sum_weights import SumWeightsMetric
from torchrec.metrics.tensor_weighted_avg import TensorWeightedAvgMetric
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.metrics.tower_qps import TowerQPSMetric
from torchrec.metrics.unweighted_ne import UnweightedNEMetric
from torchrec.metrics.weighted_avg import WeightedAvgMetric
from torchrec.metrics.weighted_sum_predictions import WeightedSumPredictionsMetric
from torchrec.metrics.xauc import XAUCMetric
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.keyed_jagged_tensor_pool import KeyedJaggedTensorPool
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    LFU_EvictionPolicy,
    LRU_EvictionPolicy,
    MCHEvictionPolicy,
    MCHManagedCollisionModule,
)
from torchrec.modules.tensor_pool import TensorPool


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


def build_metric(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    task_names: Optional[List[str]] = None,
    use_tensor_task: bool = False,
    use_session_task: bool = False,
    **kwargs: Any,
) -> RecMetric:
    """One metric under the fixed configuration every golden entry is taken at."""
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

    return metric_class(
        world_size=1,
        my_rank=0,
        batch_size=32,
        tasks=tasks,
        compute_mode=compute_mode,
        window_size=100,
        fused_update_limit=0,
        **kwargs,
    )


def extract_state_dict_keys(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    **kwargs: Any,
) -> List[str]:
    return sorted(
        build_metric(metric_class, compute_mode, **kwargs).state_dict().keys()
    )


def get_metric_snapshot_key(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode,
    variant: str = "",
) -> str:
    key = f"{metric_class.__name__}_{compute_mode.name}"
    if variant:
        key = f"{key}_{variant}"
    return key


def load_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    if not GOLDEN_SNAPSHOT_PATH.exists():
        return {}
    with open(GOLDEN_SNAPSHOT_PATH, "r") as f:
        return json.load(f)


def load_required_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    """The golden snapshot, or raise.

    An empty file makes the orphan check pass by having nothing to compare.
    Regenerating against one would make every entry look new to the gates.
    """
    snapshot = load_golden_snapshot()
    if not snapshot:
        raise RuntimeError(
            f"{GOLDEN_SNAPSHOT_PATH} is missing or empty. It is checked in, so "
            "restore it from source control rather than writing a new one. A "
            "regenerated baseline authorizes whatever the code produces today."
        )
    return snapshot


def save_golden_snapshot(snapshot: Dict[str, Dict[str, Any]]) -> None:
    with open(GOLDEN_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")


# Fixture values for the golden cases.
_THROUGHPUT_BATCH_SIZE_STAGES: List[BatchSizeStage] = [
    BatchSizeStage(batch_size=32, max_iters=100),
    BatchSizeStage(batch_size=64, max_iters=None),
]


def _make_throughput_metric(
    batch_size_stages: Optional[List[BatchSizeStage]] = None,
) -> ThroughputMetric:
    return ThroughputMetric(
        batch_size=32,
        world_size=1,
        window_seconds=100,
        warmup_steps=10,
        batch_size_stages=batch_size_stages,
    )


def _make_rec_metric_module(
    throughput_metric: Optional[ThroughputMetric] = None,
) -> RecMetricModule:
    return RecMetricModule(
        **_module_fixture_kwargs(),
        throughput_metric=throughput_metric,
    )


@dataclass(frozen=True)
class _GoldenCase:
    """One golden snapshot: which entry it is, and how to build it.

    The builder is self-contained, so a case cannot be paired with another
    case's configuration. `stable_id` is written by hand rather than derived
    from the class, so moving a file does not silently change the key and
    abandon the entry it used to match.
    """

    # Snake_case, not the class name: this is the golden file's key, so it
    # must survive a class rename.
    stable_id: str
    expected_class: Type[torch.nn.Module]
    variant: str
    build: Callable[[], torch.nn.Module]

    @property
    def key(self) -> str:
        return f"{self.stable_id}_{self.variant}" if self.variant else self.stable_id

    @classmethod
    def for_metric(
        cls,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        key_suffix: str = "",
        **kwargs: Any,
    ) -> "_GoldenCase":
        """One case for a RecMetric configuration.

        The id comes from get_metric_snapshot_key rather than a second copy of
        its formula, so the two cannot drift. Named key_suffix rather than
        variant, because every other keyword here reaches the metric's own
        constructor and a metric could one day take a variant.
        """
        return cls(
            stable_id=get_metric_snapshot_key(metric_class, compute_mode),
            expected_class=metric_class,
            variant=key_suffix,
            build=partial(build_metric, metric_class, compute_mode, **kwargs),
        )


_REC_METRIC_MODULE_DEFAULT = _GoldenCase(
    "rec_metric_module", RecMetricModule, "", _make_rec_metric_module
)

# Each stable_id is the golden file's key. It has to survive a rename of the
# class beside it, which is why it is snake_case and not a class name.
_CORE_SCHEMA_CASES: Tuple[_GoldenCase, ...] = (
    _GoldenCase("throughput_metric", ThroughputMetric, "", _make_throughput_metric),
    _GoldenCase(
        "throughput_metric",
        ThroughputMetric,
        "with_batch_size_stages",
        lambda: _make_throughput_metric(_THROUGHPUT_BATCH_SIZE_STAGES),
    ),
    _REC_METRIC_MODULE_DEFAULT,
    _GoldenCase(
        "rec_metric_module",
        RecMetricModule,
        "with_throughput",
        lambda: _make_rec_metric_module(_make_throughput_metric()),
    ),
)


def _module_fixture_kwargs() -> Dict[str, Any]:
    """Construction args shared by every RecMetricModule case.

    One real metric, because an empty RecMetricList produces an empty
    state_dict. Shared so the enrolled modules keep describing comparable
    shapes.
    """
    tasks = [create_test_task("task1")]
    return {
        "batch_size": 32,
        "world_size": 1,
        "rec_tasks": tasks,
        "rec_metrics": RecMetricList(
            [
                NEMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=32,
                    tasks=tasks,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                    window_size=100,
                )
            ]
        ),
    }


class _InertThread:
    """A Thread that never runs.

    This file only reads a module's state_dict, so CPUOffloadedRecMetricModule's
    workers do nothing here. It constructs its threads directly, with no seam to
    pass a substitute through, which is why this one arrives by patch. Without
    it the module spawns two real workers that then need shutting down.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: Optional[float] = None) -> None:
        pass


def _make_cpu_offloaded_module() -> CPUOffloadedRecMetricModule:
    with unittest.mock.patch(
        "torchrec.metrics.cpu_offloaded_metric_module.threading.Thread",
        new=_InertThread,
    ):
        return CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"), **_module_fixture_kwargs()
        )


def _make_mch_module(policy: MCHEvictionPolicy) -> MCHManagedCollisionModule:
    return MCHManagedCollisionModule(
        zch_size=64,
        device=torch.device("cpu"),
        eviction_interval=2,
        eviction_policy=policy,
    )


def _make_hash_zch_module(**overrides: Any) -> HashZchManagedCollisionModule:
    return HashZchManagedCollisionModule(
        zch_size=64,
        device=torch.device("cpu"),
        total_num_buckets=4,
        **overrides,
    )


# Not RecMetric subclasses, so _discover_all_recmetric_subclasses cannot see
# them. A row here is the only thing that enrolls one.
_MODULE_CASES: Tuple[_GoldenCase, ...] = (
    _GoldenCase("noop_metric_module", NoOpMetricModule, "", NoOpMetricModule),
    _GoldenCase(
        "cpu_comms_rec_metric_module",
        CPUCommsRecMetricModule,
        "",
        lambda: CPUCommsRecMetricModule(**_module_fixture_kwargs()),
    ),
    # state_dict() reads the comms tree and load_state_dict() writes the offloaded
    # one, but both carry the same keys, so this pins the shape and not which tree
    # a load reached. test_cpu_offloaded_metric_module covers the load direction.
    _GoldenCase(
        "cpu_offloaded_rec_metric_module",
        CPUOffloadedRecMetricModule,
        "",
        _make_cpu_offloaded_module,
    ),
    # torchrec.modules containers with real buffers, so a key here can actually
    # go missing on a load. Buffer names come from the eviction policy, and
    # DistanceLFU is the union of the other two, so each policy needs its own row.
    _GoldenCase(
        "mch_managed_collision_module",
        MCHManagedCollisionModule,
        "",
        lambda: _make_mch_module(DistanceLFU_EvictionPolicy()),
    ),
    _GoldenCase(
        "mch_managed_collision_module",
        MCHManagedCollisionModule,
        "lfu",
        lambda: _make_mch_module(LFU_EvictionPolicy()),
    ),
    _GoldenCase(
        "mch_managed_collision_module",
        MCHManagedCollisionModule,
        "lru",
        lambda: _make_mch_module(LRU_EvictionPolicy()),
    ),
    _GoldenCase(
        "tensor_pool",
        TensorPool,
        "",
        lambda: TensorPool(
            pool_size=16, dim=4, dtype=torch.float, device=torch.device("cpu")
        ),
    ),
    # enable_uvm adds two more shapes, both needing CUDA. This target is CPU
    # only, and those shapes cannot load today anyway.
    _GoldenCase(
        "keyed_jagged_tensor_pool",
        KeyedJaggedTensorPool,
        "",
        lambda: KeyedJaggedTensorPool(
            pool_size=16,
            feature_max_lengths={"f1": 2},
            device=torch.device("cpu"),
        ),
    ),
    # is_weighted adds a "weights" key; both directions are pinned.
    _GoldenCase(
        "keyed_jagged_tensor_pool",
        KeyedJaggedTensorPool,
        "weighted",
        lambda: KeyedJaggedTensorPool(
            pool_size=16,
            feature_max_lengths={"f1": 2},
            is_weighted=True,
            device=torch.device("cpu"),
        ),
    ),
    _GoldenCase(
        "hash_zch_managed_collision_module",
        HashZchManagedCollisionModule,
        "",
        _make_hash_zch_module,
    ),
    # persist_hash_zch_bucket=False drops the bucket buffer, for warm-loading
    # checkpoints written before it existed. Both shapes are real.
    _GoldenCase(
        "hash_zch_managed_collision_module",
        HashZchManagedCollisionModule,
        "without_bucket_buffer",
        lambda: _make_hash_zch_module(persist_hash_zch_bucket=False),
    ),
    # With no eviction policy the metadata buffer is None, which state_dict
    # leaves out. Production always sets a policy, so pin that shape too.
    _GoldenCase(
        "hash_zch_managed_collision_module",
        HashZchManagedCollisionModule,
        "with_eviction",
        lambda: _make_hash_zch_module(
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(features=[], single_ttl=1),
        ),
    ),
    # track_id_freq adds _hash_zch_runtime_meta, which write_runtime_meta_dim
    # also adds. The two are mutually exclusive, so one case covers the key.
    _GoldenCase(
        "hash_zch_managed_collision_module",
        HashZchManagedCollisionModule,
        "with_runtime_meta",
        lambda: _make_hash_zch_module(track_id_freq=True),
    ),
    # Eviction and runtime metadata compose. Only whether a policy is set
    # changes the key set, not which one, so pin the pair.
    _GoldenCase(
        "hash_zch_managed_collision_module",
        HashZchManagedCollisionModule,
        "with_eviction_and_runtime_meta",
        lambda: _make_hash_zch_module(
            eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
            eviction_config=HashZchEvictionConfig(features=[], single_ttl=-1),
            track_id_freq=True,
        ),
    ),
)


# Every RecMetric configuration a golden entry is taken at.
_METRIC_SPECS: Tuple[_GoldenCase, ...] = (
    # Core metrics with persistent state
    # There is no include_logloss row: it only gates which metrics _compute
    # reports, so its entry was identical to the default row below.
    _GoldenCase.for_metric(NEMetric),
    _GoldenCase.for_metric(CalibrationMetric),
    _GoldenCase.for_metric(CTRMetric),
    _GoldenCase.for_metric(MSEMetric),
    _GoldenCase.for_metric(
        MSEMetric,
        key_suffix="with_r_squared",
        include_r_squared=True,
    ),
    _GoldenCase.for_metric(MAEMetric),
    _GoldenCase.for_metric(WeightedAvgMetric),
    _GoldenCase.for_metric(AccuracyMetric),
    _GoldenCase.for_metric(PrecisionMetric),
    _GoldenCase.for_metric(RecallMetric),
    _GoldenCase.for_metric(TowerQPSMetric),
    _GoldenCase.for_metric(NMSEMetric),
    _GoldenCase.for_metric(AverageMetric),
    _GoldenCase.for_metric(HindsightTargetPRMetric),
    _GoldenCase.for_metric(NDCGMetric),
    _GoldenCase.for_metric(XAUCMetric),
    _GoldenCase.for_metric(ScalarMetric),
    _GoldenCase.for_metric(MultiLabelPrecisionMetric, num_labels=1),
    # Metrics with non-persistent state (AUC family)
    _GoldenCase.for_metric(AUCMetric),
    _GoldenCase.for_metric(AUPRCMetric),
    _GoldenCase.for_metric(RAUCMetric),
    _GoldenCase.for_metric(GAUCMetric),
    # TensorWeightedAvgMetric requires tensor_name in tasks
    _GoldenCase.for_metric(TensorWeightedAvgMetric, use_tensor_task=True),
    _GoldenCase.for_metric(CaliFreeNEMetric),
    _GoldenCase.for_metric(NEPositiveMetric),
    _GoldenCase.for_metric(ServingNEMetric),
    _GoldenCase.for_metric(UnweightedNEMetric),
    _GoldenCase.for_metric(RecalibratedNEMetric),
    _GoldenCase.for_metric(ServingCalibrationMetric),
    _GoldenCase.for_metric(RecalibratedCalibrationMetric),
    _GoldenCase.for_metric(OutputMetric),
    # MulticlassRecallMetric requires number_of_classes
    _GoldenCase.for_metric(MulticlassRecallMetric, number_of_classes=3),
    # SegmentedNEMetric requires num_groups and grouping_keys
    _GoldenCase.for_metric(
        SegmentedNEMetric,
        num_groups=2,
        grouping_keys="test_task-grouping",
    ),
    # Session-level metrics require session_metric_def in tasks
    _GoldenCase.for_metric(PrecisionSessionMetric, use_session_task=True),
    _GoldenCase.for_metric(RecallSessionMetric, use_session_task=True),
    # FUSED mode tests
    _GoldenCase.for_metric(NEMetric, RecComputeMode.FUSED_TASKS_COMPUTATION),
    _GoldenCase.for_metric(CalibrationMetric, RecComputeMode.FUSED_TASKS_COMPUTATION),
    _GoldenCase.for_metric(WeightedAvgMetric, RecComputeMode.FUSED_TASKS_COMPUTATION),
    # New utility metrics
    _GoldenCase.for_metric(NumMissingLabelsMetric),
    _GoldenCase.for_metric(NumPositiveSamplesMetric),
    _GoldenCase.for_metric(SumWeightsMetric),
    _GoldenCase.for_metric(WeightedSumPredictionsMetric),
)


_SCHEMA_CASES: Tuple[_GoldenCase, ...] = (
    _CORE_SCHEMA_CASES + _MODULE_CASES + _METRIC_SPECS
)


def _validate_case_entry(
    case: _GoldenCase, entry: Dict[str, Any], current_keys: Set[str]
) -> None:
    """Compare one case against its golden entry. Raises on any disagreement."""
    baseline_keys = set(entry["state_dict_keys"])

    removed = baseline_keys - current_keys
    if removed:
        raise AssertionError(
            f"BREAKING CHANGE in {case.key}: state_dict keys removed: "
            f"{sorted(removed)}."
        )

    added = current_keys - baseline_keys
    if added:
        raise AssertionError(
            f"BREAKING CHANGE in {case.key}: state_dict keys added: "
            f"{sorted(added)}.\n"
            "DCP validates every model FQN against checkpoint metadata before "
            "load_state_dict runs, so the planner rejects a checkpoint written "
            "before these keys existed. Loading one in process proves nothing: "
            "state torchmetrics holds by setattr is invisible to the strict "
            "check and still demanded by the planner."
        )


class GoldenCaseTest(unittest.TestCase):
    """Compares each golden case against its recorded entry.

    The entry records the state_dict keys and nothing else, so that is the
    whole comparison. A case pointed at another class is caught by the type
    assertion in _check_case, not by anything stored in the file.
    """

    golden_snapshot: Dict[str, Dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_required_golden_snapshot()

    def _check_case(self, case: _GoldenCase) -> None:
        module = case.build()
        self.assertIs(type(module), case.expected_class)

        if case.key not in self.golden_snapshot:
            self.fail(
                f"No golden entry for {case.key}. Run with --update-golden "
                "to create it. Writing one here would bless whatever the "
                "code currently produces, and concurrent tests writing the "
                "whole file at once corrupt it."
            )

        _validate_case_entry(
            case, self.golden_snapshot[case.key], set(module.state_dict())
        )

    def test_golden_cases(self) -> None:
        for case in _SCHEMA_CASES:
            with self.subTest(case.key):
                self._check_case(case)

    def test_key_drift_is_rejected(self) -> None:
        """Checks that the drift check works.

        Neither failure path runs in a green suite. This makes both run.
        """
        case = _GoldenCase("probe", torch.nn.Module, "", torch.nn.Module)
        for label, baseline, expected in (
            ("removed", ["alpha", "beta", "gamma"], "state_dict keys removed"),
            ("added", ["alpha"], "state_dict keys added"),
        ):
            with self.subTest(label):
                entry = {"state_dict_keys": baseline}
                with self.assertRaisesRegex(AssertionError, expected):
                    _validate_case_entry(case, entry, {"alpha", "beta"})

    def test_regeneration_reproduces_the_checked_in_file(self) -> None:
        """The file must equal what --update-golden writes, entry for entry.

        Compared as three questions rather than one, because a single
        assertEqual over the whole file reports a truncated dump that names
        nothing. An entry nobody generates is dead weight that still reads as
        coverage, so it gets its own answer.
        """
        generated = generate_schema_case_entries()
        checked_in = load_golden_snapshot()

        self.assertEqual(
            sorted(set(checked_in) - set(generated)),
            [],
            "golden entries that no case generates",
        )
        self.assertEqual(
            sorted(set(generated) - set(checked_in)),
            [],
            "cases with no golden entry",
        )
        for key in sorted(set(generated) & set(checked_in)):
            with self.subTest(key):
                self.assertEqual(generated[key], checked_in[key])

    def test_one_class_per_stable_id(self) -> None:
        """A stable id names one class, however many variants it has.

        The key carries the variant, so two classes could share an id and still
        keep distinct keys. Anything that maps id to class would then silently
        keep whichever row it read last.
        """
        classes_by_id: Dict[str, Set[Type[torch.nn.Module]]] = {}
        for case in _SCHEMA_CASES:
            classes_by_id.setdefault(case.stable_id, set()).add(case.expected_class)

        for stable_id, classes in classes_by_id.items():
            with self.subTest(stable_id):
                self.assertEqual(
                    len(classes),
                    1,
                    f"{stable_id} is claimed by "
                    f"{sorted(c.__name__ for c in classes)}.",
                )

    def test_cases_for_same_id_have_distinct_key_sets(self) -> None:
        """Every case under one id must snapshot a different key set.

        The unnamed case counts too, so this catches a named variant whose
        builder forgot the argument that distinguishes it. The key and the
        stored metadata both agree with themselves in that case; only the
        resulting shape disagrees.
        """
        by_id: Dict[str, List[FrozenSet[str]]] = {}
        for case in _SCHEMA_CASES:
            by_id.setdefault(case.stable_id, []).append(
                frozenset(case.build().state_dict().keys())
            )

        for stable_id, entries in by_id.items():
            with self.subTest(stable_id):
                self.assertEqual(
                    len(set(entries)),
                    len(entries),
                    f"{stable_id} has {len(entries)} cases but "
                    f"{len(set(entries))} distinct key sets. One builder is carrying "
                    "another's configuration.",
                )


class RecMetricModuleBackwardCompatibilityTest(unittest.TestCase):
    """Load checks for RecMetricModule that are not golden comparisons."""

    def test_rec_metric_module_backward_compat_trained_batches(self) -> None:
        module = _make_rec_metric_module()
        state_dict = module.state_dict()

        state_dict["_trained_batches"] = torch.tensor(100)

        fresh_module = _make_rec_metric_module()
        # strict=True matches production. Under strict=False this test passes
        # even without the pop hook.
        fresh_module.load_state_dict(state_dict, strict=True)


class GoldenRegenerationTest(unittest.TestCase):
    """The gates on the real generate-validate-save path.

    The tests above call the comparison helpers directly, which says nothing
    about whether `update_golden_snapshot` consults them or writes anyway. Each
    case here doctors a baseline, runs the whole path against a temporary file,
    and requires that file to come back untouched.

    The patch redirects where the snapshot is read and written. It replaces a
    path, not a dependency, so the file I/O under test is real.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Building every case is the slow part, so pay it once and reuse the
        # result as the baseline each case then doctors.
        cls.current: Dict[str, Dict[str, Any]] = generate_schema_case_entries()

    @contextlib.contextmanager
    def _baseline(self, snapshot: Dict[str, Dict[str, Any]]) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
            with unittest.mock.patch(f"{__name__}.GOLDEN_SNAPSHOT_PATH", path):
                yield path

    def _assert_refused(self, baseline: Dict[str, Dict[str, Any]], reason: str) -> None:
        with self._baseline(baseline) as path:
            before = path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)) as cm:
                update_golden_snapshot()
            self.assertIn(reason, str(cm.exception))
            self.assertEqual(path.read_bytes(), before, "the golden was rewritten")

    def _entry_holding_keys(self, baseline: Dict[str, Dict[str, Any]]) -> str:
        """An entry with at least one state key.

        Six entries legitimately have none, the AUC family among them, and they
        sort first. Doctoring one of those changes nothing and the gate has
        nothing to catch.
        """
        for key in sorted(baseline):
            if baseline[key]["state_dict_keys"]:
                return key
        self.fail("no golden entry holds a state key")

    def test_an_empty_baseline_is_refused(self) -> None:
        self._assert_refused({}, "restore it from source control")

    def test_a_changed_entry_is_refused(self) -> None:
        """Both directions, because they break different loads.

        An added key is rejected by the planner before load_state_dict runs.
        A removed key is still carried by older checkpoints and read back by a
        checkpoint-derived load.
        """
        for label, doctor in (
            # Drop a key from the baseline, so the live code looks like it
            # added one.
            ("added", lambda keys: keys[1:]),
            ("removed", lambda keys: keys + ["a_key_the_code_no_longer_produces"]),
        ):
            with self.subTest(label):
                baseline = copy.deepcopy(type(self).current)
                key = self._entry_holding_keys(baseline)
                baseline[key]["state_dict_keys"] = doctor(
                    baseline[key]["state_dict_keys"]
                )
                self._assert_refused(baseline, "change the state_dict keys")

    def test_a_dropped_entry_is_refused(self) -> None:
        baseline = copy.deepcopy(type(self).current)
        baseline["AnEntryTheCodeNoLongerProduces"] = {"state_dict_keys": []}
        self._assert_refused(baseline, "remove golden entries")

    def test_a_brand_new_entry_is_written(self) -> None:
        """A new case has no baseline to break, so it needs no acknowledgement.

        Named, not searched for: most key sets are shared by several entries,
        so asking whether one appears anywhere would accept a sibling's.
        """
        baseline = copy.deepcopy(type(self).current)
        key = self._entry_holding_keys(baseline)
        fresh = baseline.pop(key)
        with self._baseline(baseline) as path:
            update_golden_snapshot()
            written = json.loads(path.read_text())
        self.assertIn(key, written)
        self.assertEqual(written[key], fresh)

    def test_an_unchanged_baseline_is_rewritten_identically(self) -> None:
        """Byte-identical, so regeneration never churns the file on its own.

        The other cases parse what was written, which says nothing about
        indent, key order, or the trailing newline.
        """
        with self._baseline(type(self).current) as path:
            before = path.read_bytes()
            update_golden_snapshot()
            self.assertEqual(path.read_bytes(), before)


class MetricCoverageTest(unittest.TestCase):
    """
    Test that ensures all RecMetric subclasses are covered by backward compatibility tests.

    This test will FAIL if a new metric is added to torchrec but not added to
    _METRIC_SPECS. When adding a new metric, users must add it to _METRIC_SPECS
    in this file.
    """

    # Metrics that are intentionally excluded from testing (with reason)
    EXCLUDED_METRICS: Dict[str, str] = {
        # Class names, with a reason, e.g.
        # "SomeMetric": "deprecated, removed next release",
    }

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_all_recmetrics_are_covered(self) -> None:
        discovered_metrics: Set[str] = {
            cls.__name__ for cls in _discover_all_recmetric_subclasses()
        }

        covered_metrics: Set[str] = {
            spec.expected_class.__name__ for spec in _METRIC_SPECS
        }

        # A metric the walk stops finding drops out of the check below without
        # failing it. That happens when torchrec.metrics is reorganized, which
        # is a change nobody reviewing this file sees.
        self.assertEqual(
            covered_metrics - discovered_metrics,
            set(),
            "METRICS_TO_TEST names classes that discovery does not find",
        )

        missing_metrics = (
            discovered_metrics - covered_metrics - set(self.EXCLUDED_METRICS.keys())
        )

        if missing_metrics:
            self.fail(
                f"The following RecMetric subclasses are not covered by backward "
                f"compatibility tests: {sorted(missing_metrics)}.\n\n"
                f"To fix this:\n"
                f"1. Add a _METRIC_SPECS row. The check runs from the table, "
                f"so the row is the whole registration.\n"
                f"2. Run with --update-golden to write its entry.\n\n"
                f"If the metric should be excluded, add it to EXCLUDED_METRICS with a reason."
            )


# Cross-config state_dict tests: detect config-dependent keys and verify
# cross-config loads succeed with strict=True. New conditional-state params
# must be added to KNOWN_CONDITIONAL_STATE with a proper always-pop hook.

_BATCH_SIZE_STAGES_ALTERNATIVE: List[BatchSizeStage] = [
    BatchSizeStage(batch_size=256, max_iters=1),
    BatchSizeStage(batch_size=512, max_iters=None),
]

_BASE_RECMETRIC_PARAMS: Set[str] = {
    "self",
    "args",
    "kwargs",
    "world_size",
    "my_rank",
    "batch_size",
    "tasks",
    "compute_mode",
    "window_size",
    "fused_update_limit",
    "compute_on_all_ranks",
    "should_validate_update",
    "process_group",
    "enable_pt2_compile",
    "should_clone_update_inputs",
}

_BASE_COMPUTATION_PARAMS: Set[str] = {
    "self",
    "args",
    "kwargs",
    "my_rank",
    "batch_size",
    "n_tasks",
    "tasks",
    "window_size",
    "compute_on_all_ranks",
    "should_validate_update",
    "compute_mode",
    "process_group",
    "fused_update_limit",
    "allow_missing_label_with_zero_weight",
    "session_metric_def",
}

_PARAM_ALTERNATIVES: Dict[str, List[Any]] = {
    "batch_size_stages": [_BATCH_SIZE_STAGES_ALTERNATIVE],
    "description": ["test_description"],
    "is_negative_task_mask": [[True]],
    "label_names": [["label_a", "label_b"]],
}


def _get_metric_specific_params(
    metric_cls: Type[RecMetric],
) -> Dict[str, inspect.Parameter]:
    """Get non-base params from metric and computation class signatures."""
    params: Dict[str, inspect.Parameter] = {}

    for name, param in inspect.signature(metric_cls.__init__).parameters.items():
        if name not in _BASE_RECMETRIC_PARAMS:
            params[name] = param

    comp_cls = getattr(metric_cls, "_computation_class", None)
    if comp_cls is not None:
        for name, param in inspect.signature(comp_cls.__init__).parameters.items():
            if name not in _BASE_COMPUTATION_PARAMS and name not in params:
                params[name] = param

    return params


def _generate_alternatives(
    param: inspect.Parameter,
) -> List[Any]:
    """Auto-generate alternative values for a param based on its default."""
    name = param.name
    default = param.default

    if name in _PARAM_ALTERNATIVES:
        return _PARAM_ALTERNATIVES[name]

    if default is inspect.Parameter.empty:
        return []

    if isinstance(default, bool):
        return [not default]
    if isinstance(default, int):
        return [default + 1]
    if isinstance(default, float):
        return [default + 1.0]
    if isinstance(default, str):
        return [default + "_alt"]
    return []


# Params that produce different state_dict keys. Uses strings (includes ThroughputMetric/nn.Module).
KNOWN_CONDITIONAL_STATE: Set[Tuple[str, str]] = {
    ("MSEMetric", "include_r_squared"),
    ("MultiLabelPrecisionMetric", "label_names"),
    ("MultiLabelPrecisionMetric", "num_labels"),
    ("ThroughputMetric", "batch_size_stages"),
    ("TowerQPSMetric", "batch_size_stages"),
}

# Params that do NOT affect state_dict keys. Every non-base param must be here
# or in KNOWN_CONDITIONAL_STATE.
KNOWN_SAFE_PARAMS: Set[Tuple[str, str]] = {
    ("AUCMetric", "apply_bin"),
    ("AUCMetric", "grouped_auc"),
    ("AUPRCMetric", "grouped_auprc"),
    ("AUPRCMetric", "max_prediction"),
    ("AUPRCMetric", "min_prediction"),
    ("AUPRCMetric", "num_bins"),
    ("AccuracyMetric", "threshold"),
    ("HindsightTargetPRMetric", "target_precision"),
    ("NDCGMetric", "exponential_gain"),
    ("NDCGMetric", "is_negative_task_mask"),
    ("NDCGMetric", "k"),
    ("NDCGMetric", "remove_single_length_sessions"),
    ("NDCGMetric", "report_ndcg_as_decreasing_curve"),
    ("NDCGMetric", "scale_by_weights_tensor"),
    ("NDCGMetric", "session_key"),
    ("NEMetric", "include_logloss"),
    ("PrecisionMetric", "threshold"),
    ("RAUCMetric", "grouped_rauc"),
    ("RecalibratedCalibrationMetric", "recalibration_coefficient"),
    ("RecalibratedNEMetric", "include_logloss"),
    ("RecalibratedNEMetric", "recalibration_coefficient"),
    ("RecallMetric", "threshold"),
    ("SegmentedNEMetric", "cast_keys_to_int"),
    ("SegmentedNEMetric", "grouping_keys"),
    ("SegmentedNEMetric", "include_logloss"),
    ("SegmentedNEMetric", "num_groups"),  # changes tensor shapes, not key names
    ("TensorWeightedAvgMetric", "description"),
    ("TowerQPSMetric", "warmup_steps"),
}

# Subset with proper always-pop hooks (cross-config load tests use these).
RECMETRIC_CONDITIONAL_STATE: Dict[Tuple[Type[RecMetric], str], List[Any]] = {
    (MSEMetric, "include_r_squared"): [True],
    (TowerQPSMetric, "batch_size_stages"): [_BATCH_SIZE_STAGES_ALTERNATIVE],
}

_RECMETRIC_COMMON_KWARGS: Dict[str, Any] = {
    "world_size": 1,
    "my_rank": 0,
    "batch_size": 32,
    "compute_mode": RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    "window_size": 100,
}

_THROUGHPUT_COMMON_KWARGS: Dict[str, Any] = {
    "batch_size": 32,
    "world_size": 1,
    "window_seconds": 100,
    "warmup_steps": 10,
}


_cached_recmetric_subclasses: Optional[Set[Type[RecMetric]]] = None


def _discover_all_recmetric_subclasses() -> Set[Type[RecMetric]]:
    """Every RecMetric subclass in torchrec.metrics.

    A module that would not import used to be skipped in silence. All 51
    modules import today, so a failure here is news rather than noise.
    """
    global _cached_recmetric_subclasses
    if _cached_recmetric_subclasses is not None:
        return _cached_recmetric_subclasses

    import importlib
    import pkgutil

    import torchrec.metrics

    subclasses: Set[Type[RecMetric]] = set()
    failures: List[str] = []
    for _, module_name, _ in pkgutil.iter_modules(torchrec.metrics.__path__):
        try:
            module = importlib.import_module(f"torchrec.metrics.{module_name}")
        except Exception as e:
            failures.append(
                f"  torchrec.metrics.{module_name}: {type(e).__name__}: {e}"
            )
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, RecMetric)
                and attr is not RecMetric
                and not attr_name.startswith("_")
            ):
                subclasses.add(attr)

    if failures:
        detail = "\n".join(failures)
        raise RuntimeError(
            f"Could not import every torchrec.metrics module:\n{detail}\n"
            "A module skipped here hides its metrics from the coverage check, "
            "which asserts discovery is a subset of the table. Anything that "
            "shrinks discovery makes that assertion easier to satisfy."
        )

    _cached_recmetric_subclasses = subclasses
    return subclasses


def _get_default_keys_cached(
    metric_cls: Type[RecMetric],
    cls_name: str,
    default_keys_cache: Optional[Dict[str, Set[str]]] = None,
) -> Optional[Set[str]]:
    """Get default state_dict keys for a metric, using cache if available."""
    if default_keys_cache is not None and cls_name in default_keys_cache:
        return default_keys_cache[cls_name]
    try:
        default_keys = set(extract_state_dict_keys(metric_cls))
    except (TypeError, ValueError, KeyError, RecMetricException):
        return None
    if default_keys_cache is not None:
        default_keys_cache[cls_name] = default_keys
    return default_keys


def _probe_alternatives(
    metric_cls: Type[RecMetric],
    param_name: str,
    alternatives: List[Any],
    default_keys: Set[str],
) -> Optional[str]:
    """Probe alternative param values. Returns 'misclassified', 'unprobed', or None."""
    tested_any = False
    for alt_value in alternatives:
        try:
            variant_keys = set(
                extract_state_dict_keys(metric_cls, **{param_name: alt_value})
            )
            tested_any = True
        except (TypeError, ValueError, KeyError, RecMetricException):
            continue
        if default_keys != variant_keys:
            return "misclassified"
    if not tested_any:
        return "unprobed"
    return None


def _classify_known_safe_param(
    cls_name: str,
    param_name: str,
    metrics_by_name: Dict[str, Type[RecMetric]],
    default_keys_cache: Optional[Dict[str, Set[str]]] = None,
) -> Optional[str]:
    """Classify a KNOWN_SAFE_PARAMS entry. Returns category or None if verified safe."""
    metric_cls = metrics_by_name.get(cls_name)
    if metric_cls is None:
        return "stale"
    params = _get_metric_specific_params(metric_cls)
    if param_name not in params:
        return "stale"
    alternatives = _generate_alternatives(params[param_name])
    if not alternatives:
        return "unprobed"
    default_keys = _get_default_keys_cached(metric_cls, cls_name, default_keys_cache)
    if default_keys is None:
        return "unprobed"
    return _probe_alternatives(metric_cls, param_name, alternatives, default_keys)


class ConditionalStateRegistryTest(unittest.TestCase):

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_validate_state_affecting_params_recmetrics(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        for metric_cls in all_metrics:
            try:
                default_keys = set(extract_state_dict_keys(metric_cls))
            except (TypeError, ValueError, KeyError, RecMetricException):
                continue
            for param_name, param in _get_metric_specific_params(metric_cls).items():
                alternatives = _generate_alternatives(param)
                if not alternatives:
                    continue
                for alt_value in alternatives:
                    with self.subTest(metric=metric_cls.__name__, param=param_name):
                        try:
                            variant_keys = set(
                                extract_state_dict_keys(
                                    metric_cls, **{param_name: alt_value}
                                )
                            )
                        except (TypeError, ValueError, KeyError, RecMetricException):
                            continue

                        if default_keys != variant_keys:
                            self.assertIn(
                                (metric_cls.__name__, param_name),
                                KNOWN_CONDITIONAL_STATE,
                                f"{metric_cls.__name__}.{param_name} affects "
                                f"state_dict keys but is not in "
                                f"KNOWN_CONDITIONAL_STATE. "
                                f"Added: {variant_keys - default_keys}, "
                                f"Removed: {default_keys - variant_keys}",
                            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_all_params_categorized(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        uncategorized = []
        for metric_cls in all_metrics:
            for param_name in _get_metric_specific_params(metric_cls):
                pair = (metric_cls.__name__, param_name)
                if (
                    pair not in KNOWN_CONDITIONAL_STATE
                    and pair not in KNOWN_SAFE_PARAMS
                ):
                    uncategorized.append(pair)

        if uncategorized:
            formatted = "\n".join(
                f'    ("{cls}", "{param}"),' for cls, param in sorted(uncategorized)
            )
            self.fail(
                f"Found {len(uncategorized)} uncategorized metric param(s).\n"
                f"Each param must be in KNOWN_CONDITIONAL_STATE (if it conditionally\n"
                f"registers buffers/state) or KNOWN_SAFE_PARAMS (if it does not).\n"
                f"Add these to the appropriate set:\n{formatted}"
            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_none_default_params_have_test_values(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        missing = []
        for metric_cls in all_metrics:
            for param_name, param in _get_metric_specific_params(metric_cls).items():
                if param.default is None and param_name not in _PARAM_ALTERNATIVES:
                    missing.append((metric_cls.__name__, param_name))

        if missing:
            formatted = "\n".join(
                f'    "{p}",' for _, p in sorted(set(missing), key=lambda x: x[1])
            )
            self.fail(
                f"Found None-default params without _PARAM_ALTERNATIVES entries.\n"
                f"These params can't be auto-probed for conditional state.\n"
                f"Add test values to _PARAM_ALTERNATIVES for:\n{formatted}"
            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_known_safe_params_are_actually_safe(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        metrics_by_name = {cls.__name__: cls for cls in all_metrics}
        default_keys_cache: Dict[str, Set[str]] = {}
        buckets: Dict[str, List[Tuple[str, str]]] = {
            "stale": [],
            "misclassified": [],
            "unprobed": [],
        }
        for cls_name, param_name in sorted(KNOWN_SAFE_PARAMS):
            category = _classify_known_safe_param(
                cls_name, param_name, metrics_by_name, default_keys_cache
            )
            if category is not None:
                buckets[category].append((cls_name, param_name))

        error_messages = {
            "stale": "Stale entries (param not in any signature)",
            "misclassified": (
                "Misclassified (actually affects state_dict keys, "
                "move to KNOWN_CONDITIONAL_STATE)"
            ),
        }
        errors = []
        for key, label in error_messages.items():
            if buckets[key]:
                formatted = "\n".join(f'    ("{c}", "{p}"),' for c, p in buckets[key])
                errors.append(f"{label}:\n{formatted}")
        if errors:
            self.fail("KNOWN_SAFE_PARAMS issues:\n" + "\n\n".join(errors))

    def test_validate_state_affecting_params_throughput(self) -> None:
        default_metric = ThroughputMetric(**_THROUGHPUT_COMMON_KWARGS)
        variant_metric = ThroughputMetric(
            **_THROUGHPUT_COMMON_KWARGS,
            batch_size_stages=_BATCH_SIZE_STAGES_ALTERNATIVE,
        )
        default_keys = set(default_metric.state_dict().keys())
        variant_keys = set(variant_metric.state_dict().keys())

        if default_keys != variant_keys:
            self.assertIn(
                ("ThroughputMetric", "batch_size_stages"),
                KNOWN_CONDITIONAL_STATE,
                f"ThroughputMetric.batch_size_stages affects state_dict "
                f"keys but is not in KNOWN_CONDITIONAL_STATE. "
                f"Added keys: {variant_keys - default_keys}, "
                f"Removed keys: {default_keys - variant_keys}",
            )


class CrossConfigLoadTest(unittest.TestCase):
    """Loading a checkpoint written under a different metric configuration.

    A variant configuration adds state_dict keys the default does not have.
    The two directions are not equally safe, and only one of them is safe.
    """

    def _make_common_kwargs(self) -> Dict[str, Any]:
        return {**_RECMETRIC_COMMON_KWARGS, "tasks": [create_test_task("task1")]}

    def _config_pair(
        self, metric_cls: Type[RecMetric], param_name: str, alt_value: Any
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        common_kwargs = self._make_common_kwargs()
        return (
            metric_cls(**common_kwargs),
            metric_cls(**common_kwargs, **{param_name: alt_value}),
        )

    def _throughput_pair(self) -> Tuple[ThroughputMetric, ThroughputMetric]:
        return (
            ThroughputMetric(**_THROUGHPUT_COMMON_KWARGS),
            ThroughputMetric(
                **_THROUGHPUT_COMMON_KWARGS,
                batch_size_stages=_BATCH_SIZE_STAGES_ALTERNATIVE,
            ),
        )

    def _extra_keys(
        self, default: torch.nn.Module, variant: torch.nn.Module
    ) -> Set[str]:
        extra = set(variant.state_dict()) - set(default.state_dict())
        self.assertTrue(
            extra,
            "the variant adds no keys, so neither direction proves anything",
        )
        return extra

    def _assert_variant_checkpoint_loads(
        self, default: torch.nn.Module, variant: torch.nn.Module
    ) -> None:
        """The variant's extra keys must not break a default-configured module.

        Nothing in the default module claims them, so a pop hook or
        torchmetrics' loader has to absorb them.
        """
        self._extra_keys(default, variant)
        default.load_state_dict(variant.state_dict(), strict=True)

    def test_variant_checkpoint_loads_into_default(self) -> None:
        for (
            metric_cls,
            param_name,
        ), alternatives in RECMETRIC_CONDITIONAL_STATE.items():
            for alt_value in alternatives:
                with self.subTest(metric=metric_cls.__name__, param=param_name):
                    self._assert_variant_checkpoint_loads(
                        *self._config_pair(metric_cls, param_name, alt_value)
                    )

    def test_throughput_variant_checkpoint_loads_into_default(self) -> None:
        self._assert_variant_checkpoint_loads(*self._throughput_pair())

    def test_multi_label_precision_cross_config_load_fails_without_hook(self) -> None:
        variant_kwargs: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 3,
        }
        default_kwargs: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
        }
        variant = MultiLabelPrecisionMetric(**variant_kwargs)
        default = MultiLabelPrecisionMetric(**default_kwargs)

        with self.assertRaises(RuntimeError):
            default.load_state_dict(variant.state_dict(), strict=True)

    def test_multi_label_precision_label_names_cross_config_load_fails(self) -> None:
        kwargs_a: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
            "label_names": ["cat"],
        }
        kwargs_b: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
            "label_names": ["dog"],
        }
        variant_a = MultiLabelPrecisionMetric(**kwargs_a)
        variant_b = MultiLabelPrecisionMetric(**kwargs_b)

        with self.assertRaises(RuntimeError):
            variant_b.load_state_dict(variant_a.state_dict(), strict=True)


def generate_schema_case_entries() -> Dict[str, Dict[str, Any]]:
    """Golden entries for the _SCHEMA_CASES table.

    The comparison path fails rather than writing a missing entry, so this is
    the only way a new case gets one.

    """
    entries: Dict[str, Dict[str, Any]] = {}
    for case in _SCHEMA_CASES:
        if case.key in entries:
            raise ValueError(
                f"Two cases share the key {case.key!r}. One would overwrite "
                "the other's golden entry."
            )
        entries[case.key] = {
            "state_dict_keys": sorted(case.build().state_dict().keys()),
        }
    return entries


def changed_entries(
    old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]
) -> Dict[str, Tuple[List[str], List[str]]]:
    """For entries both snapshots hold, the state keys regeneration would change.

    Returns key -> (added, removed). Entries missing from either side are
    skipped: a brand-new case has no baseline to break, and a dropped one is
    removed_entries' business.
    """
    changed: Dict[str, Tuple[List[str], List[str]]] = {}
    for golden_key, entry in old.items():
        if golden_key not in new:
            continue
        before = set(entry["state_dict_keys"])
        after = set(new[golden_key]["state_dict_keys"])
        if before != after:
            changed[golden_key] = (sorted(after - before), sorted(before - after))
    return changed


def removed_entries(
    old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Golden entries that regeneration would drop.

    A renamed metric moves its entry, and the replacement looks brand new, so
    an added key could ride through on the new-entry exemption. Refusing the
    drop is what closes that.
    """
    return sorted(set(old) - set(new))


def update_golden_snapshot() -> None:
    print("Generating golden snapshot...")
    snapshot = generate_schema_case_entries()

    # Not load_golden_snapshot: an absent file reads as {}, which turns every
    # generated entry into a brand-new one and silences both gates below.
    previous = load_required_golden_snapshot()

    dropped = removed_entries(previous, snapshot)
    if dropped:
        raise ValueError(
            f"Regenerating would remove golden entries: {dropped}.\n"
            "If the removal is right, delete those entries from "
            f"{GOLDEN_SNAPSHOT_PATH.name} by hand and run again, in the same "
            "diff that renames or drops the class. The hand edit is the "
            "acknowledgement, and the diff is the record.\n"
            "If it is not right, a case lost its coverage and the table needs "
            "the row back."
        )

    changed = changed_entries(previous, snapshot)
    if changed:
        detail = "\n".join(
            f"  {key}: added {added}, removed {removed}"
            for key, (added, removed) in sorted(changed.items())
        )
        raise ValueError(
            "Regenerating would change the state_dict keys of entries that "
            f"already exist:\n{detail}\n"
            "An added key makes every older checkpoint unloadable, and no "
            "module hook can repair that: the planner rejects the load before "
            "any hook runs. A removed key is still carried by older "
            "checkpoints, and a checkpoint-derived load reads it back and "
            "finds nobody claiming it.\n"
            f"If the change is right, edit those entries in "
            f"{GOLDEN_SNAPSHOT_PATH.name} by hand and run again. The hand edit "
            "is the acknowledgement, and the diff is the record."
        )

    save_golden_snapshot(snapshot)
    # Resolved: the path runs through buck-out's link tree, which symlinks back
    # to the source. Printing it unresolved suggests the write went to buck-out.
    print(f"Golden snapshot saved to {GOLDEN_SNAPSHOT_PATH.resolve()}")
    print(f"Total metrics captured: {len(snapshot)}")
    for key in sorted(snapshot.keys()):
        info = snapshot[key]
        print(f"  - {key}: {len(info['state_dict_keys'])} state_dict keys")


if __name__ == "__main__":
    if "--update-golden" in sys.argv or os.environ.get("UPDATE_GOLDEN_SNAPSHOT"):
        if "--update-golden" in sys.argv:
            sys.argv.remove("--update-golden")
        update_golden_snapshot()
    else:
        unittest.main()
