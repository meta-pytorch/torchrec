#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.metrics.session_pairwise_auc import (
    _compute_average_per_batch,
    _compute_pairwise_auc,
    _get_session_pairwise_auc_states,
    CORRECT_PAIR_WEIGHT,
    EFFECTIVE_EXAMPLE_COUNT,
    SessionPairwiseAUCMetricComputation,
    TOTAL_PAIR_WEIGHT,
    VALID_PAIR_COUNT,
)


class SessionPairwiseAUCTest(unittest.TestCase):
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

    def test_batch_coverage_metrics_are_opt_in(self) -> None:
        disabled = self._make_computation(report_batch_coverage=False)
        enabled = self._make_computation(report_batch_coverage=True)

        self.assertFalse(hasattr(disabled, VALID_PAIR_COUNT))
        self.assertFalse(hasattr(disabled, EFFECTIVE_EXAMPLE_COUNT))
        self.assertEqual(len(disabled._compute()), 2)
        self.assertTrue(hasattr(enabled, VALID_PAIR_COUNT))
        self.assertTrue(hasattr(enabled, EFFECTIVE_EXAMPLE_COUNT))
        self.assertEqual(len(enabled._compute()), 6)
