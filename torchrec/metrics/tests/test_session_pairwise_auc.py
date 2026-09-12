#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.rec_metric import RecMetricException
from torchrec.metrics.session_pairwise_auc import (
    _compute_pairwise_auc,
    _get_session_pairwise_auc_states,
    CORRECT_PAIR_WEIGHT,
    SessionPairwiseAUCMetric,
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

    def test_fused_computation_is_rejected(self) -> None:
        for compute_mode in (
            RecComputeMode.FUSED_TASKS_COMPUTATION,
            RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
        ):
            with self.subTest(compute_mode=compute_mode):
                with self.assertRaisesRegex(
                    RecMetricException,
                    "Fused computation is not supported for session pairwise AUC",
                ):
                    self._metric(compute_mode=compute_mode)

    def test_fused_update_is_rejected(self) -> None:
        for fused_update_limit in (1, 10):
            with self.subTest(fused_update_limit=fused_update_limit):
                with self.assertRaisesRegex(
                    RecMetricException,
                    "Fused update is not supported for session pairwise AUC",
                ):
                    self._metric(fused_update_limit=fused_update_limit)
