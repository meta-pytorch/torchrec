#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any

import torch
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.rec_metric import RecMetricException
from torchrec.metrics.session_pairwise_auc import (
    _compute_average_per_batch,
    _compute_pairwise_auc,
    _compute_ratio,
    _compute_tied_pair_rate,
    _get_session_pairwise_auc_states,
    CORRECT_PAIR_WEIGHT,
    DEGENERATE_SESSION_COUNT,
    EFFECTIVE_EXAMPLE_COUNT,
    EXAMPLE_COUNT,
    GROUP_SIZE_MAX_SUM,
    GROUP_SIZE_SUM,
    SAME_SESSION_PAIR_COUNT,
    SESSION_COUNT,
    SessionPairwiseAUCMetric,
    SessionPairwiseAUCMetricComputation,
    SINGLETON_SESSION_COUNT,
    TOTAL_PAIR_WEIGHT,
    VALID_PAIR_COUNT,
)


class SessionPairwiseAUCTest(unittest.TestCase):
    def _metric(
        self,
        *,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        fused_update_limit: int = 0,
    ) -> SessionPairwiseAUCMetric:
        return SessionPairwiseAUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=4,
            tasks=[
                RecTaskInfo(
                    name="task",
                    label_name="label",
                    prediction_name="prediction",
                    weight_name="weight",
                )
            ],
            compute_mode=compute_mode,
            fused_update_limit=fused_update_limit,
        )

    def _make_computation(
        self, *, report_batch_coverage: bool
    ) -> SessionPairwiseAUCMetricComputation:
        return SessionPairwiseAUCMetricComputation(
            my_rank=0,
            batch_size=4,
            n_tasks=1,
            window_size=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            process_group=None,
            report_batch_coverage=report_batch_coverage,
        )

    def _compute(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        session_ids: torch.Tensor,
        weights: torch.Tensor,
        **kwargs: bool,
    ) -> torch.Tensor:
        states = _get_session_pairwise_auc_states(
            predictions=predictions,
            labels=labels,
            session_ids=session_ids,
            example_weights=weights,
            **kwargs,
        )
        return _compute_pairwise_auc(
            correct_pair_weight=states[CORRECT_PAIR_WEIGHT],
            total_pair_weight=states[TOTAL_PAIR_WEIGHT],
        )

    def test_only_pairs_within_the_same_session(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.9, 0.2, 0.1, 0.8]]),
            labels=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            session_ids=torch.tensor([10, 20, 10, 20]),
            weights=torch.ones(1, 4),
        )
        torch.testing.assert_close(actual, torch.tensor([0.5], dtype=torch.double))

    def test_label_ties_and_zero_weight_examples_are_removed(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.9, 0.8, 0.1, 0.0]]),
            labels=torch.tensor([[2.0, 2.0, 1.0, 0.0]]),
            session_ids=torch.tensor([1, 1, 1, 1]),
            weights=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        )
        torch.testing.assert_close(actual, torch.tensor([1.0], dtype=torch.double))

    def test_pair_weight_matches_ranknet_sum_of_example_weights(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.9, 0.1, 0.8]]),
            labels=torch.tensor([[2.0, 1.0, 0.0]]),
            session_ids=torch.tensor([1, 1, 1]),
            weights=torch.tensor([[3.0, 1.0, 1.0]]),
        )
        # Correct pairs have weights 4 and 4; the incorrect pair has weight 2.
        torch.testing.assert_close(actual, torch.tensor([0.8], dtype=torch.double))

    def test_prediction_tie_gets_half_credit(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.5, 0.5]]),
            labels=torch.tensor([[1.0, 0.0]]),
            session_ids=torch.tensor([1, 1]),
            weights=torch.ones(1, 2),
        )
        torch.testing.assert_close(actual, torch.tensor([0.5], dtype=torch.double))

    def test_rank_order_label_reverses_preference(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.9, 0.1]]),
            labels=torch.tensor([[1.0, 2.0]]),
            session_ids=torch.tensor([1, 1]),
            weights=torch.ones(1, 2),
            rank_order_label=True,
        )
        torch.testing.assert_close(actual, torch.tensor([1.0], dtype=torch.double))

    def test_no_eligible_pairs_returns_chance(self) -> None:
        actual = self._compute(
            predictions=torch.tensor([[0.9, 0.1]]),
            labels=torch.tensor([[1.0, 1.0]]),
            session_ids=torch.tensor([1, 1]),
            weights=torch.ones(1, 2),
        )
        torch.testing.assert_close(actual, torch.tensor([0.5], dtype=torch.double))

    def test_all_singleton_sessions_return_chance_and_zero_pair_weight(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor(
                [
                    [0.91, 0.13, 0.72, 0.44, 0.25],
                    [0.02, 0.88, 0.31, 0.67, 0.55],
                ]
            ),
            labels=torch.tensor(
                [
                    [4.0, 0.0, 3.0, 1.0, 2.0],
                    [0.0, 4.0, 1.0, 3.0, 2.0],
                ]
            ),
            session_ids=torch.tensor([101, 205, 309, 413, 517]),
            example_weights=torch.ones(2, 5),
        )

        torch.testing.assert_close(
            _compute_pairwise_auc(
                correct_pair_weight=states[CORRECT_PAIR_WEIGHT],
                total_pair_weight=states[TOTAL_PAIR_WEIGHT],
            ),
            torch.tensor([0.5, 0.5], dtype=torch.double),
        )
        for state_name in (CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT):
            torch.testing.assert_close(
                states[state_name], torch.zeros(2, dtype=torch.double)
            )

    def test_interleaved_uneven_sessions_multitask_weighted_auc(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor(
                [
                    [0.7, 0.1, 0.9, 0.6, 0.4, 0.5, 0.4, 0.2],
                    [0.2, 0.6, 0.8, 0.1, 0.3, 0.5, 0.9, 0.4],
                ]
            ),
            labels=torch.tensor(
                [
                    [3.0, 0.0, 1.0, 7.0, 2.0, 2.0, 1.0, 9.0],
                    [1.0, 3.0, 1.0, 6.0, 2.0, 0.0, 0.0, 8.0],
                ]
            ),
            session_ids=torch.tensor([10, 20, 10, 30, 20, 10, 20, 40]),
            example_weights=torch.tensor(
                [
                    [2.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                ]
            ),
        )

        # Task 0: session 10 contributes 5/12 correct pair weight; session 20
        # contributes 6.5/8 (including half-credit for one prediction tie).
        # Task 1: session 10 contributes 2/4; after removing its zero-weight
        # row, session 20 contributes 0/3. Singleton sessions add no pairs.
        torch.testing.assert_close(
            states[CORRECT_PAIR_WEIGHT],
            torch.tensor([11.5, 2.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            states[TOTAL_PAIR_WEIGHT],
            torch.tensor([20.0, 7.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            _compute_pairwise_auc(
                correct_pair_weight=states[CORRECT_PAIR_WEIGHT],
                total_pair_weight=states[TOTAL_PAIR_WEIGHT],
            ),
            torch.tensor([0.575, 2.0 / 7.0], dtype=torch.double),
        )

    def test_default_unfused_configuration_is_supported(self) -> None:
        metric = self._metric()
        self.assertEqual(metric._compute_mode, RecComputeMode.UNFUSED_TASKS_COMPUTATION)
        self.assertFalse(metric.supports_cpu_offloaded_metric_module)

    def _assert_outputs_match(
        self,
        reference: dict[str, torch.Tensor],
        actual: dict[str, torch.Tensor],
    ) -> None:
        self.assertEqual(reference.keys(), actual.keys())
        for key in reference:
            torch.testing.assert_close(
                actual[key].squeeze(),
                reference[key].squeeze(),
                equal_nan=True,
                msg=(
                    f"Metric output mismatch for {key}: "
                    f"reference={reference[key]}, actual={actual[key]}"
                ),
            )

    def test_fused_computation_matches_unfused_for_dpa_auxiliary_inputs(
        self,
    ) -> None:
        outputs: dict[RecComputeMode, dict[str, torch.Tensor]] = {}
        batches = [
            (
                torch.tensor([0.9, 0.1, 0.3, 0.3]),
                torch.tensor([2.0, 0.0, 1.0, 1.0]),
                torch.tensor([10, 10, 20, 20]),
                torch.tensor([1.0, 1.0, 1.0, 1.0]),
            ),
            (
                torch.tensor([0.4, 0.8, 0.2, 0.7]),
                torch.tensor([2.0, 1.0, 0.0, 5.0]),
                torch.tensor([30, 30, 30, 40]),
                torch.tensor([1.0, 0.0, 2.0, 1.0]),
            ),
        ]
        for compute_mode in RecComputeMode:
            metric_kwargs: dict[str, Any] = {
                "session_key": "session",
                "score_key": "score",
                "pairwise_weight_key": "pairwise_weight",
            }
            metric = SessionPairwiseAUCMetric(
                world_size=1,
                my_rank=0,
                batch_size=4,
                tasks=[
                    RecTaskInfo(
                        name="task",
                        label_name="label",
                        prediction_name="prediction",
                        weight_name="weight",
                    )
                ],
                compute_mode=compute_mode,
                **metric_kwargs,
            )
            for scores, labels, session_ids, pairwise_weights in batches:
                metric.update(
                    predictions={"task": torch.zeros_like(scores)},
                    labels={"task": labels},
                    weights={"task": torch.ones_like(pairwise_weights)},
                    required_inputs={
                        "session": session_ids,
                        "score": scores,
                        "pairwise_weight": pairwise_weights,
                    },
                )
            outputs[compute_mode] = metric.compute()

        reference = outputs[RecComputeMode.UNFUSED_TASKS_COMPUTATION]
        for compute_mode in (
            RecComputeMode.FUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                self._assert_outputs_match(reference, outputs[compute_mode])

    def test_fused_computation_matches_unfused_for_multiple_tasks(self) -> None:
        tasks = [
            RecTaskInfo(
                name="task_0",
                label_name="label_0",
                prediction_name="prediction_0",
                weight_name="weight_0",
            ),
            RecTaskInfo(
                name="task_1",
                label_name="label_1",
                prediction_name="prediction_1",
                weight_name="weight_1",
            ),
        ]
        outputs: dict[RecComputeMode, dict[str, torch.Tensor]] = {}
        for compute_mode in RecComputeMode:
            metric_kwargs: dict[str, Any] = {
                "session_key": "session",
            }
            metric = SessionPairwiseAUCMetric(
                world_size=1,
                my_rank=0,
                batch_size=4,
                tasks=tasks,
                compute_mode=compute_mode,
                **metric_kwargs,
            )
            metric.update(
                predictions={
                    "task_0": torch.tensor([0.9, 0.2, 0.6, 0.1]),
                    "task_1": torch.tensor([0.1, 0.8, 0.4, 0.7]),
                },
                labels={
                    "task_0": torch.tensor([2.0, 0.0, 1.0, 0.0]),
                    "task_1": torch.tensor([0.0, 2.0, 1.0, 1.0]),
                },
                weights={
                    "task_0": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                    "task_1": torch.tensor([1.0, 2.0, 1.0, 1.0]),
                },
                required_inputs={"session": torch.tensor([10, 10, 20, 20])},
            )
            outputs[compute_mode] = metric.compute()

        reference = outputs[RecComputeMode.UNFUSED_TASKS_COMPUTATION]
        for compute_mode in (
            RecComputeMode.FUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                self._assert_outputs_match(reference, outputs[compute_mode])

    def test_fused_update_is_rejected(self) -> None:
        for fused_update_limit in (1, 10):
            with self.subTest(fused_update_limit=fused_update_limit):
                with self.assertRaisesRegex(
                    RecMetricException,
                    "Fused update is not supported for session pairwise AUC",
                ):
                    self._metric(fused_update_limit=fused_update_limit)

    def test_valid_pairs_and_effective_examples(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor([[0.9, 0.8, 0.1, 0.0]]),
            labels=torch.tensor([[2.0, 2.0, 1.0, 0.0]]),
            session_ids=torch.tensor([1, 1, 1, 1]),
            example_weights=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            report_batch_coverage=True,
        )
        # The two rank-2 rows can each pair with the rank-1 row. The zero-weight
        # row is excluded, leaving two unordered pairs and three participating
        # examples.
        torch.testing.assert_close(
            states[VALID_PAIR_COUNT], torch.tensor([2.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            states[EFFECTIVE_EXAMPLE_COUNT],
            torch.tensor([3.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            states[SAME_SESSION_PAIR_COUNT],
            torch.tensor([3.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            states[EXAMPLE_COUNT], torch.tensor([4.0], dtype=torch.double)
        )

    def test_group_health_states(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor([[0.9, 0.8, 0.1, 0.0]]),
            labels=torch.tensor([[1.0, 1.0, 5.0, 7.0]]),
            session_ids=torch.tensor([1, 1, 2, 3]),
            example_weights=torch.ones(1, 4),
            report_batch_coverage=True,
        )
        # Session 1 is a non-singleton with all labels tied. Sessions 2 and 3
        # are reported separately as singletons rather than as degenerate.
        torch.testing.assert_close(
            states[DEGENERATE_SESSION_COUNT],
            torch.tensor([1.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            states[SESSION_COUNT], torch.tensor([3.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            states[GROUP_SIZE_SUM], torch.tensor([4.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            states[GROUP_SIZE_MAX_SUM], torch.tensor([2.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            states[SINGLETON_SESSION_COUNT],
            torch.tensor([2.0], dtype=torch.double),
        )

    def test_no_pairs_has_no_effective_examples(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor([[0.9, 0.1]]),
            labels=torch.tensor([[1.0, 1.0]]),
            session_ids=torch.tensor([1, 1]),
            example_weights=torch.ones(1, 2),
            report_batch_coverage=True,
        )
        torch.testing.assert_close(
            states[VALID_PAIR_COUNT], torch.tensor([0.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            states[EFFECTIVE_EXAMPLE_COUNT],
            torch.tensor([0.0], dtype=torch.double),
        )

    def test_batch_coverage_states_are_not_computed_by_default(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.tensor([[0.9, 0.1]]),
            labels=torch.tensor([[1.0, 0.0]]),
            session_ids=torch.tensor([1, 1]),
            example_weights=torch.ones(1, 2),
        )

        self.assertEqual(
            set(states),
            {CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT},
        )

    def test_average_per_batch(self) -> None:
        actual = _compute_average_per_batch(
            value_sum=torch.tensor([12.0, 0.0], dtype=torch.double),
            batch_count=torch.tensor([3.0, 0.0], dtype=torch.double),
        )
        torch.testing.assert_close(actual, torch.tensor([4.0, 0.0], dtype=torch.double))

    def test_coverage_ratios(self) -> None:
        tied_rate = _compute_tied_pair_rate(
            valid_pair_count=torch.tensor([2.0, 0.0], dtype=torch.double),
            same_session_pair_count=torch.tensor([3.0, 0.0], dtype=torch.double),
        )
        effective_rate = _compute_ratio(
            numerator=torch.tensor([3.0, 0.0], dtype=torch.double),
            denominator=torch.tensor([4.0, 0.0], dtype=torch.double),
        )
        torch.testing.assert_close(
            tied_rate, torch.tensor([1.0 / 3.0, 0.0], dtype=torch.double)
        )
        torch.testing.assert_close(
            effective_rate, torch.tensor([0.75, 0.0], dtype=torch.double)
        )

    def test_batch_coverage_metrics_are_opt_in(self) -> None:
        disabled = self._make_computation(report_batch_coverage=False)
        enabled = self._make_computation(report_batch_coverage=True)

        self.assertFalse(hasattr(disabled, VALID_PAIR_COUNT))
        self.assertFalse(hasattr(disabled, EFFECTIVE_EXAMPLE_COUNT))
        self.assertEqual(len(disabled._compute()), 2)
        self.assertTrue(hasattr(enabled, VALID_PAIR_COUNT))
        self.assertTrue(hasattr(enabled, SAME_SESSION_PAIR_COUNT))
        self.assertTrue(hasattr(enabled, EFFECTIVE_EXAMPLE_COUNT))
        self.assertTrue(hasattr(enabled, DEGENERATE_SESSION_COUNT))
        self.assertTrue(hasattr(enabled, SINGLETON_SESSION_COUNT))
        self.assertEqual(len(enabled._compute()), 18)
