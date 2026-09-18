#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, cast

import torch
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.rec_metric import RecMetricException
from torchrec.metrics.session_pairwise_auc import (
    _as_task_matrix,
    _compute_pairwise_auc,
    _get_session_pairwise_auc_states,
    CORRECT_PAIR_WEIGHT,
    SessionPairwiseAUCMetric,
    SessionPairwiseAUCMetricComputation,
    TOTAL_PAIR_WEIGHT,
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

    def test_rejects_invalid_input_shapes(self) -> None:
        with self.assertRaisesRegex(RecMetricException, "must be positive"):
            _get_session_pairwise_auc_states(
                predictions=torch.ones(1, 2),
                labels=torch.ones(1, 2),
                session_ids=torch.ones(2),
                example_weights=torch.ones(1, 2),
                pair_chunk_size=0,
            )

        with self.assertRaisesRegex(RecMetricException, "expected 2 or 4"):
            _as_task_matrix(torch.ones(3), n_tasks=2, batch_size=2, input_name="score")

        invalid_inputs = [
            (
                torch.ones(1, 2),
                torch.ones(1, 1),
                torch.ones(1, 2),
                torch.ones(2),
                "must have identical shapes",
            ),
            (
                torch.ones(2),
                torch.ones(2),
                torch.ones(2),
                torch.ones(2),
                "must have shape \\[n_tasks, batch_size\\]",
            ),
            (
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.ones(1, 2),
                torch.ones(1, 2),
                "session_ids must have shape \\[batch_size\\]",
            ),
        ]
        for predictions, labels, weights, session_ids, expected_error in invalid_inputs:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(RecMetricException, expected_error):
                    _get_session_pairwise_auc_states(
                        predictions=predictions,
                        labels=labels,
                        session_ids=session_ids,
                        example_weights=weights,
                    )

    def test_empty_batch_returns_zero_pair_weights(self) -> None:
        states = _get_session_pairwise_auc_states(
            predictions=torch.empty(1, 0),
            labels=torch.empty(1, 0),
            session_ids=torch.empty(0, dtype=torch.long),
            example_weights=torch.empty(1, 0),
        )

        for state_name in (CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT):
            torch.testing.assert_close(
                states[state_name], torch.zeros(1, dtype=torch.double)
            )

    def test_update_requires_session_scores_and_pairwise_weights(self) -> None:
        computation = cast(
            SessionPairwiseAUCMetricComputation,
            self._metric()._metrics_computations[0],
        )
        labels = torch.tensor([[1.0, 0.0]])
        scores = torch.tensor([[0.9, 0.1]])
        weights = torch.ones(1, 2)

        with self.assertRaisesRegex(
            RecMetricException, "Input 'session_id' is required"
        ):
            computation.update(
                predictions=scores,
                labels=labels,
                weights=weights,
                required_inputs={},
            )

        with self.assertRaisesRegex(
            RecMetricException, "requires scores and pairwise example weights"
        ):
            computation.update(
                predictions=None,
                labels=labels,
                weights=weights,
                required_inputs={"session_id": torch.tensor([1, 1])},
            )

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
            pair_chunk_size=2,
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

    def test_chunked_computation_supports_session_length_1500(self) -> None:
        session_length = 1500
        labels = torch.arange(session_length, dtype=torch.float32).unsqueeze(0)
        states = _get_session_pairwise_auc_states(
            predictions=labels.clone(),
            labels=labels,
            session_ids=torch.zeros(session_length, dtype=torch.long),
            example_weights=torch.ones_like(labels),
            pair_chunk_size=64,
        )
        expected_pair_weight = float(session_length * (session_length - 1))
        for state_name in (CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT):
            torch.testing.assert_close(
                states[state_name],
                torch.tensor([expected_pair_weight], dtype=torch.double),
                rtol=0,
                atol=0,
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
