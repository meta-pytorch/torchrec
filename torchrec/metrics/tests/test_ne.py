#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial, update_wrapper
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Type
from unittest.mock import patch, PropertyMock

import torch
import torchrec.metrics.rec_metric as rec_metric
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.metrics_config import DefaultTaskInfo, RecTaskInfo
from torchrec.metrics.ne import (
    compute_cross_entropy,
    compute_logloss,
    compute_ne,
    NEMetric,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecMetricComputation
from torchrec.metrics.test_utils import (
    metric_test_helper,
    rec_metric_accuracy_test_helper,
    rec_metric_gpu_sync_test_launcher,
    rec_metric_value_test_launcher,
    sync_test_helper,
    TestMetric,
)

try:
    from pyjk import PyPatchJustKnobs as _PyPatchJustKnobs
except ImportError:
    _PyPatchJustKnobs = None


WORLD_SIZE = 4


@contextmanager
def _single_rank_gloo() -> Iterator[torch.distributed.ProcessGroup]:
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{os.path.join(tmpdir, 'rdzv')}",
            world_size=1,
            rank=0,
        )
        process_group = cast(
            torch.distributed.ProcessGroup, torch.distributed.group.WORLD
        )
        try:
            yield process_group
        finally:
            torch.distributed.destroy_process_group()


@contextmanager
def _fixed_shape_sync_patch(enabled: bool) -> Iterator[None]:
    if _PyPatchJustKnobs is None:
        raise RuntimeError("PyPatchJustKnobs is unavailable")
    with _PyPatchJustKnobs().patch(
        "pytorch/torchrec:enable_fixed_shape_metric_sync", enabled
    ):
        yield


def _fixed_shape_sync_enablement_broadcast_test() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )
    try:
        with (
            _fixed_shape_sync_patch(rank == 0),
            patch.object(
                rec_metric,
                "_default_sync_supports_fixed_shape",
                return_value=rank == 0,
            ),
        ):
            process_group = cast(
                torch.distributed.ProcessGroup, torch.distributed.group.WORLD
            )
            ne = NEMetric(
                world_size=world_size,
                my_rank=rank,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            if not computation._fixed_shape_sync_is_enabled:
                raise AssertionError("rank-zero enablement was not broadcast")
    finally:
        torch.distributed.destroy_process_group()


def _fixed_shape_sync_rank_dependent_shape_test() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )
    try:
        with (
            _fixed_shape_sync_patch(True),
            patch.object(
                rec_metric,
                "_default_sync_supports_fixed_shape",
                return_value=True,
            ),
        ):
            process_group = cast(
                torch.distributed.ProcessGroup, torch.distributed.group.WORLD
            )
            ne = NEMetric(
                world_size=world_size,
                my_rank=rank,
                batch_size=1,
                tasks=[RecTaskInfo(name=f"task_{index}") for index in range(rank + 1)],
                compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            try:
                computation.sync(distributed_available=lambda: True)
            except rec_metric.RecMetricException as error:
                if "fixed-shape states differ across ranks" not in str(error):
                    raise
            else:
                raise AssertionError("rank-dependent shapes were accepted")
    finally:
        torch.distributed.destroy_process_group()


def _fixed_shape_sync_late_rank_dependent_shape_test() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )
    try:
        with _fixed_shape_sync_patch(True):
            process_group = cast(
                torch.distributed.ProcessGroup, torch.distributed.group.WORLD
            )
            ne = NEMetric(
                world_size=world_size,
                my_rank=rank,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)
            computation.unsync()
            if rank == 1:
                state_name = next(iter(cast(Any, computation)._defaults))
                state = cast(torch.Tensor, getattr(computation, state_name))
                setattr(computation, state_name, state.new_zeros((2, *state.shape)))
            try:
                computation.sync(distributed_available=lambda: True)
            except rec_metric.RecMetricException as error:
                if "fixed-shape states differ across ranks" not in str(error):
                    raise
            else:
                raise AssertionError("late rank-dependent shapes were accepted")
    finally:
        torch.distributed.destroy_process_group()


class TestNEMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestNEMetric.eta
        )
        cross_entropy_sum = torch.sum(cross_entropy)
        weighted_num_samples = torch.sum(weights)
        pos_labels = torch.sum(weights * labels)
        neg_labels = torch.sum(weights * (1.0 - labels))
        return {
            "cross_entropy_sum": cross_entropy_sum,
            "weighted_num_samples": weighted_num_samples,
            "pos_labels": pos_labels,
            "neg_labels": neg_labels,
            "num_samples": torch.tensor(labels.size()).long(),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        allow_missing_label_with_zero_weight = False
        if not states["weighted_num_samples"].all():
            allow_missing_label_with_zero_weight = True

        return compute_ne(
            states["cross_entropy_sum"],
            states["weighted_num_samples"],
            pos_labels=states["pos_labels"],
            neg_labels=states["neg_labels"],
            eta=TestNEMetric.eta,
            allow_missing_label_with_zero_weight=allow_missing_label_with_zero_weight,
        )


class TestLoglossMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        required_inputs_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestNEMetric.eta
        )
        cross_entropy_sum = torch.sum(cross_entropy)
        pos_labels = torch.sum(weights * labels, dim=-1)
        neg_labels = torch.sum(weights * (1.0 - labels), dim=-1)
        return {
            "cross_entropy_sum": cross_entropy_sum,
            "pos_labels": pos_labels,
            "neg_labels": neg_labels,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_logloss(
            states["cross_entropy_sum"],
            pos_labels=states["pos_labels"],
            neg_labels=states["neg_labels"],
            eta=TestLoglossMetric.eta,
        )


_logloss_metric_test_helper: Callable[..., None] = partial(
    metric_test_helper, include_logloss=True
)
update_wrapper(_logloss_metric_test_helper, metric_test_helper)


class NEMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = NEMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "ne"

    @staticmethod
    def _set_distinct_sync_states(
        computation: RecMetricComputation,
    ) -> Dict[str, torch.Tensor]:
        defaults = cast(Dict[str, Any], cast(Any, computation)._defaults)
        expected: Dict[str, torch.Tensor] = {}
        for value, name in enumerate(defaults, start=1):
            state = cast(torch.Tensor, getattr(computation, name))
            state.fill_(value)
            expected[name] = state.clone()
        return expected

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_path(self) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(True),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather") as all_gather,
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)

        barrier.assert_not_called()
        self.assertGreater(all_gather.call_count, 0)

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_validates_rank_shapes_once(self) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(True),
            patch.object(
                torch.distributed,
                "all_gather_object",
                wraps=torch.distributed.all_gather_object,
            ) as all_gather_object,
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)
            computation.unsync()
            computation.sync(distributed_available=lambda: True)

        all_gather_object.assert_called_once()

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_disabled_uses_variable_shape_path(self) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(False),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather"),
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)

        barrier.assert_called()

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_uses_variable_path_for_different_group(self) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(True),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather"),
        ):
            alternate_process_group = torch.distributed.new_group(ranks=[0])
            try:
                ne = NEMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=1,
                    tasks=[DefaultTaskInfo],
                    process_group=process_group,
                )
                computation = cast(RecMetricComputation, ne._metrics_computations[0])
                computation.sync(
                    process_group=alternate_process_group,
                    distributed_available=lambda: True,
                )
                self.assertIn("variable_shape", computation._seen_dist_sync_paths)
            finally:
                torch.distributed.destroy_process_group(alternate_process_group)

        barrier.assert_called()

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_uses_variable_path_for_changed_state_shape(
        self,
    ) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(True),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather"),
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)
            computation.unsync()
            state_name = next(iter(cast(Any, computation)._defaults))
            state = cast(torch.Tensor, getattr(computation, state_name))
            setattr(computation, state_name, state.new_zeros((2, *state.shape)))

            computation.sync(distributed_available=lambda: True)

        barrier.assert_called()
        self.assertIn("variable_shape", computation._seen_dist_sync_paths)

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_enablement_is_broadcast_from_rank_zero(self) -> None:
        rec_metric_accuracy_test_helper(
            world_size=2,
            entry_point=_fixed_shape_sync_enablement_broadcast_test,
        )

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_rejects_rank_dependent_shapes(self) -> None:
        rec_metric_accuracy_test_helper(
            world_size=2,
            entry_point=_fixed_shape_sync_rank_dependent_shape_test,
        )

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_fixed_shape_sync_rejects_late_rank_dependent_shapes(self) -> None:
        rec_metric_accuracy_test_helper(
            world_size=2,
            entry_point=_fixed_shape_sync_late_rank_dependent_shape_test,
        )

    def test_missing_default_process_group_skips_enablement_broadcast(self) -> None:
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(
                type(torch.distributed.group),
                "WORLD",
                new_callable=PropertyMock,
                return_value=None,
            ),
            patch.object(
                rec_metric, "invoke_on_rank_and_broadcast_result"
            ) as broadcast,
        ):
            NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
            )

        broadcast.assert_not_called()

    def test_metric_without_fixed_shape_sync_skips_enablement_broadcast(
        self,
    ) -> None:
        with patch.object(
            rec_metric, "invoke_on_rank_and_broadcast_result"
        ) as broadcast:
            AUCMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
            )

        broadcast.assert_not_called()

    @unittest.skipIf(_PyPatchJustKnobs is None, "PyPatchJustKnobs is unavailable")
    def test_empty_state_defaults_are_trivially_fixed_shape(self) -> None:
        with (
            _single_rank_gloo() as process_group,
            _fixed_shape_sync_patch(True),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather"),
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            defaults = cast(Any, computation)._defaults
            cast(Any, computation)._defaults = {}
            try:
                computation.sync(distributed_available=lambda: True)
                barrier.assert_not_called()
                self.assertIn("fixed_shape", computation._seen_dist_sync_paths)
            finally:
                cast(Any, computation)._defaults = defaults

    def test_upstream_gather_without_fixed_shape_support_uses_variable_path(
        self,
    ) -> None:
        with (
            _single_rank_gloo() as process_group,
            patch.object(
                rec_metric, "_default_sync_supports_fixed_shape", return_value=False
            ),
            patch.object(torch.distributed, "barrier") as barrier,
            patch.object(torch.distributed, "all_gather"),
        ):
            ne = NEMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, ne._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)

        barrier.assert_called()
        self.assertIn("variable_shape", computation._seen_dist_sync_paths)

    def test_default_sync_support_tracks_rebound_gather(self) -> None:
        def legacy_gather(
            result: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            return [result]

        def fixed_shape_gather(
            result: torch.Tensor,
            group: Optional[Any] = None,
            *,
            assume_same_shape: bool = False,
        ) -> List[torch.Tensor]:
            return [result]

        with patch.object(rec_metric, "gather_all_tensors", legacy_gather):
            self.assertFalse(rec_metric._default_sync_supports_fixed_shape())
        with patch.object(rec_metric, "gather_all_tensors", fixed_shape_gather):
            self.assertTrue(rec_metric._default_sync_supports_fixed_shape())

    def test_default_sync_classification_survives_rebound_gather(self) -> None:
        def rebound_gather(
            result: torch.Tensor,
            group: Optional[Any] = None,
            *,
            assume_same_shape: bool = False,
        ) -> List[torch.Tensor]:
            return [result]

        with (
            _single_rank_gloo() as process_group,
            patch.object(rec_metric, "gather_all_tensors", rebound_gather),
        ):
            metric = AUCMetric(
                world_size=1,
                my_rank=0,
                batch_size=1,
                tasks=[DefaultTaskInfo],
                process_group=process_group,
            )
            computation = cast(RecMetricComputation, metric._metrics_computations[0])
            computation.sync(distributed_available=lambda: True)

        self.assertIn("variable_shape", computation._seen_dist_sync_paths)

    def test_custom_sync_path_is_unchanged(self) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=1,
            tasks=[DefaultTaskInfo],
        )
        computation = cast(RecMetricComputation, ne._metrics_computations[0])
        expected = self._set_distinct_sync_states(computation)
        synced_tensors: List[torch.Tensor] = []

        def custom_sync(
            tensor: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            synced_tensors.append(tensor)
            return [tensor]

        computation.sync(
            dist_sync_fn=custom_sync,
            distributed_available=lambda: True,
        )

        self.assertEqual(len(synced_tensors), len(expected))
        for actual, expected_tensor in zip(synced_tensors, expected.values()):
            torch.testing.assert_close(actual, expected_tensor)
        for name, expected_tensor in expected.items():
            torch.testing.assert_close(getattr(computation, name), expected_tensor)

    def test_sync_path_is_recorded_after_success(self) -> None:
        ne = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=1,
            tasks=[DefaultTaskInfo],
        )
        computation = cast(RecMetricComputation, ne._metrics_computations[0])

        def failing_sync(
            tensor: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            raise RuntimeError("sync failed")

        def successful_sync(
            tensor: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            return [tensor]

        with patch.object(rec_metric.logger, "info") as log_info:
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                computation._sync_dist(failing_sync)
            self.assertNotIn("custom", computation._seen_dist_sync_paths)
            log_info.assert_not_called()

            computation._sync_dist(successful_sync)
            self.assertIn("custom", computation._seen_dist_sync_paths)
            log_info.assert_called_once_with(
                "RecMetric distributed sync path: metric=%s path=%s states=%d",
                type(computation).__name__,
                "custom",
                len(cast(Any, computation)._defaults),
            )

            computation._sync_dist(successful_sync)
            log_info.assert_called_once()

    def test_ne_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

    def test_ne_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            batch_window_size=10,
        )

    def test_ne_zero_weights(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEMetricTest.task_name,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=metric_test_helper,
            zero_weights=True,
        )

    def test_ne_allow_missing_label_with_zero_weight(self) -> None:
        eta = 1e-12
        ne = compute_ne(
            ce_sum=torch.rand(3),
            weighted_num_samples=torch.tensor([3, 0, 2]),
            pos_labels=torch.tensor([1, 0, 2]),
            neg_labels=torch.tensor([2, 0, 0]),
            eta=eta,
            allow_missing_label_with_zero_weight=True,
        )
        self.assertFalse(ne.isinf().any().item())
        self.assertFalse(ne.isnan().any().item())
        torch.testing.assert_close(
            ne.eq(eta), torch.tensor([False, True, False]), rtol=0, atol=0
        )

    def test_logloss_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_logloss_metric_test_helper,
        )

    def test_logloss_fused_tasks(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_logloss_metric_test_helper,
        )

    def test_logloss_fused_tasks_and_states(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_logloss_metric_test_helper,
        )

    def test_logloss_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_logloss_metric_test_helper,
        )

        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            metric_name="logloss",
            test_clazz=TestLoglossMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=_logloss_metric_test_helper,
            batch_window_size=10,
        )


class NEGPUSyncTest(unittest.TestCase):
    clazz: Type[RecMetric] = NEMetric
    task_name: str = "ne"

    def test_sync_ne(self) -> None:
        rec_metric_gpu_sync_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            metric_name=NEGPUSyncTest.task_name,
            task_names=["t1"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=2,
            batch_size=5,
            batch_window_size=20,
            entry_point=sync_test_helper,
        )
