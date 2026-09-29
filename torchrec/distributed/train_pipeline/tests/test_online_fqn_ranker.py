#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import deque
from typing import Deque, Iterable

import torch
from torch import nn
from torchrec.distributed.train_pipeline.online_fqn_ranker import (
    build_cross_rank_consensus,
    build_ems_auto_tuning_workflow_result,
    FQNFinalCandidate,
    FQNRankerFinalResult,
    FQNRankingResult,
    OnlineFQNRanker,
)


class _SequenceClock:
    def __init__(self, values: Iterable[int]) -> None:
        self._values: Deque[int] = deque(values)
        self._last = 0

    def __call__(self) -> int:
        if self._values:
            self._last = self._values.popleft()
        else:
            self._last += 100_000
        return self._last


class _BlockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.ReLU(),
            nn.Linear(4, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x).sum()


class _DepthModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top = nn.Linear(4, 4, bias=False)
        self.deep = nn.Sequential(nn.Sequential(nn.Linear(4, 4, bias=False)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep(self.top(x)).sum()


class OnlineFQNRankerTest(unittest.TestCase):
    def _make_candidate(
        self,
        hook_position: float = 1.0,
        score_p10: float = 70.0,
        param_fqn: str = "block.2.weight",
    ) -> FQNFinalCandidate:
        return FQNFinalCandidate(
            fqn="block",
            hook_position=hook_position,
            param_fqn=param_fqn,
            score_p50=score_p10 + 10.0,
            score_p10=score_p10,
            confidence_p50=1.0,
            safe_count=3,
            observed_count=3,
            scored_iters=3,
            safety_margin_us_p10=100.0,
            safety_margin_us_cv=0.01,
            overlap_window_us_p10=500.0,
            eligible=True,
            noise_reason=None,
        )

    def _make_final_result(
        self,
        rank: int,
        top_candidates: list[FQNFinalCandidate],
        world_size: int = 2,
        stable: bool = True,
    ) -> FQNRankerFinalResult:
        return FQNRankerFinalResult(
            event="online_fqn_ranker_final",
            rank=rank,
            world_size=world_size,
            stable=stable,
            status="stable" if stable else "unstable",
            scored_iters=3,
            top_candidates=top_candidates,
            selected_candidate=top_candidates[0] if stable else None,
            hydra_overrides=(),
            apex_tuning_input={},
        )

    def _make_final_results(self) -> list[FQNRankerFinalResult]:
        candidate = self._make_candidate()
        return [self._make_final_result(rank, [candidate]) for rank in range(2)]

    def _make_ranker(
        self,
        model: nn.Module,
        clock: _SequenceClock,
        warmup_iters: int = 0,
        stability_window: int = 2,
        callback_latency_us: float = 1.0,
        max_scored_iters: int | None = None,
    ) -> OnlineFQNRanker:
        ranker = OnlineFQNRanker(
            model=model,
            hook_positions=(0.01, 1.0),
            warmup_iters=warmup_iters,
            stability_window=stability_window,
            top_n=1,
            max_scored_iters=max_scored_iters,
            steady_state_window=1,
            max_depth=1,
            callback_latency_us=callback_latency_us,
            fqn_allowlist={"block"},
            clock_ns=clock,
        )
        ranker.setup_hooks()
        return ranker

    def _run_iter(
        self,
        model: nn.Module,
        ranker: OnlineFQNRanker,
        iter_idx: int,
        peak_time_ns: int,
        peak_bytes: int = 100,
    ) -> FQNRankingResult | None:
        model.zero_grad()
        ranker.on_iter_start(iter_idx)
        ranker.on_backward_start(iter_idx)
        model(torch.ones(2, 4)).backward()
        ranker.record_peak(peak_time_ns, peak_bytes)
        return ranker.on_backward_end(iter_idx)

    def test_hooks_fire_only_during_backward(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock(
            [
                0,
                300_000,
                400_000,
                500_000,
                1_000_000,
                1_010_000,
                1_100_000,
                1_200_000,
                1_300_000,
                1_400_000,
            ]
        )
        ranker = self._make_ranker(model, clock)

        ranker.on_iter_start(0)
        model(torch.ones(2, 4))
        ranker.on_backward_start(0)
        ranker.record_peak(350_000, 100)
        result = ranker.on_backward_end(0)

        assert result is not None
        self.assertEqual(result.recommendations, [])
        self.assertEqual(
            [evaluation.status for evaluation in result.candidate_evaluations],
            ["skipped_by_compiler", "skipped_by_compiler"],
        )

        result = self._run_iter(model, ranker, iter_idx=1, peak_time_ns=1_050_000)

        assert result is not None
        self.assertEqual(result.recommendations[0].fqn, "block")
        self.assertEqual(result.recommendations[0].hook_position, 1.0)

    def test_pre_peak_position_is_rejected_and_after_peak_position_wins(
        self,
    ) -> None:
        model = _BlockModel()
        clock = _SequenceClock([0, 50_000, 100_000, 200_000, 300_000])
        ranker = self._make_ranker(model, clock, callback_latency_us=0.0)

        result = self._run_iter(model, ranker, iter_idx=0, peak_time_ns=150_000)

        assert result is not None
        self.assertEqual(result.recommendations[0].fqn, "block")
        self.assertEqual(result.recommendations[0].hook_position, 0.01)
        evaluations_by_position = {
            evaluation.hook_position: evaluation
            for evaluation in result.candidate_evaluations
        }
        self.assertEqual(evaluations_by_position[1.0].status, "rejected")
        self.assertLess(evaluations_by_position[1.0].safety_margin_us, 0.0)
        self.assertEqual(evaluations_by_position[0.01].status, "safe")
        self.assertGreater(evaluations_by_position[0.01].score, 0.0)

    def test_multiple_hook_positions_choose_closest_safe_position(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock([0, 10_000, 100_000, 200_000, 300_000])
        ranker = self._make_ranker(model, clock, callback_latency_us=1.0)

        result = self._run_iter(model, ranker, iter_idx=0, peak_time_ns=50_000)

        assert result is not None
        self.assertEqual(result.recommendations[0].fqn, "block")
        self.assertEqual(result.recommendations[0].hook_position, 1.0)
        self.assertEqual(ranker._fire_times["block"][1.0], 100_000)
        self.assertEqual(ranker._fire_times["block"][0.01], 200_000)

    def test_warmup_iters_are_ignored(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock([0, 10_000, 100_000, 200_000, 300_000, 400_000])
        ranker = self._make_ranker(model, clock, warmup_iters=2)

        self.assertIsNone(
            self._run_iter(model, ranker, iter_idx=0, peak_time_ns=50_000)
        )
        self.assertIsNone(
            self._run_iter(model, ranker, iter_idx=1, peak_time_ns=60_000)
        )
        result = self._run_iter(model, ranker, iter_idx=2, peak_time_ns=550_000)

        assert result is not None
        self.assertEqual(result.recommendations[0].fqn, "block")

    def test_stability_detection_locks_recommendation(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock(
            [
                0,
                10_000,
                100_000,
                200_000,
                300_000,
                400_000,
                1_000_000,
                1_010_000,
                1_100_000,
                1_200_000,
                1_300_000,
                1_400_000,
                2_000_000,
                2_010_000,
                2_100_000,
                2_200_000,
                2_300_000,
                2_400_000,
            ]
        )
        ranker = self._make_ranker(model, clock, stability_window=3)

        self._run_iter(model, ranker, iter_idx=0, peak_time_ns=50_000)
        self._run_iter(model, ranker, iter_idx=1, peak_time_ns=1_050_000)
        self.assertFalse(ranker.is_stable())
        self._run_iter(model, ranker, iter_idx=2, peak_time_ns=2_050_000)

        self.assertTrue(ranker.is_stable())
        recommendation = ranker.get_recommendation()
        assert recommendation is not None
        self.assertEqual(recommendation.fqn, "block")
        self.assertEqual(recommendation.hook_position, 1.0)

    def test_same_clock_assertion_rejects_misaligned_peak(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock([100_000, 200_000])
        ranker = self._make_ranker(model, clock)

        ranker.on_iter_start(0)
        ranker.on_backward_start(0)

        with self.assertRaises(AssertionError):
            ranker.record_peak(0, 100)

    def test_max_depth_filters_nested_modules(self) -> None:
        model = _DepthModel()
        ranker = OnlineFQNRanker(
            model=model,
            hook_positions=(0.01,),
            max_depth=1,
            steady_state_window=1,
            callback_latency_us=0.0,
            clock_ns=_SequenceClock([0]),
        )

        ranker.setup_hooks()

        self.assertEqual(
            {candidate.fqn for candidate in ranker._candidates},
            {"top", "deep"},
        )

    def test_duplicate_param_fqns_are_registered_once(self) -> None:
        model = _DepthModel()
        ranker = OnlineFQNRanker(
            model=model,
            hook_positions=(0.01, 1.0),
            max_depth=2,
            steady_state_window=1,
            callback_latency_us=0.0,
            clock_ns=_SequenceClock([0]),
        )

        ranker.setup_hooks()

        param_fqns = [candidate.param_fqn for candidate in ranker._candidates]
        self.assertEqual(len(param_fqns), len(set(param_fqns)))

    def test_final_result_emits_once_and_stops_measurement(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock(
            [
                0,
                10_000,
                100_000,
                200_000,
                300_000,
                400_000,
                1_000_000,
                1_010_000,
                1_100_000,
                1_200_000,
                1_300_000,
                1_400_000,
                2_000_000,
                2_010_000,
            ]
        )
        ranker = self._make_ranker(model, clock, stability_window=2)

        self._run_iter(model, ranker, iter_idx=0, peak_time_ns=50_000)
        self.assertIsNone(ranker.get_final_result())
        self._run_iter(model, ranker, iter_idx=1, peak_time_ns=1_050_000)

        final_result = ranker.get_final_result()
        assert final_result is not None
        self.assertEqual(final_result.event, "online_fqn_ranker_final")
        self.assertTrue(final_result.stable)
        self.assertEqual(final_result.status, "stable")
        selected_candidate = final_result.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(selected_candidate.fqn, "block")

        ranker.on_iter_start(2)
        ranker.on_backward_start(2)
        self.assertFalse(ranker._measurement_active)

    def test_unstable_probe_budget_emits_final_without_selected_candidate(self) -> None:
        model = _BlockModel()
        clock = _SequenceClock([0, 10_000, 100_000, 200_000])
        ranker = self._make_ranker(
            model,
            clock,
            stability_window=3,
            max_scored_iters=1,
        )

        self._run_iter(model, ranker, iter_idx=0, peak_time_ns=50_000)

        final_result = ranker.get_final_result()
        assert final_result is not None
        self.assertFalse(final_result.stable)
        self.assertEqual(final_result.status, "unstable")
        self.assertIsNone(final_result.selected_candidate)

    def test_cross_rank_consensus_uses_quorum_and_apply_payload(self) -> None:
        final_results = self._make_final_results()

        consensus = build_cross_rank_consensus(final_results, quorum_fraction=1.0)

        selected_candidate = consensus.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(consensus.status, "selected")
        self.assertEqual(selected_candidate.fqn, "block")
        self.assertIn(
            "training.memory_bouncer.auto_tuning.enable_online_ranking=false",
            consensus.hydra_overrides,
        )

    def test_cross_rank_consensus_admits_ranks_that_never_converged(self) -> None:
        # `stable` records whether a rank's ordering repeated, not whether its
        # measurements are sound. A fleet where nothing converged still has a
        # usable answer; the non-convergence is reported, not used as a gate.
        candidate = self._make_candidate()
        final_results = [
            self._make_final_result(rank, [candidate], world_size=8, stable=False)
            for rank in range(8)
        ]

        consensus = build_cross_rank_consensus(final_results)

        selected_candidate = consensus.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(consensus.status, "selected")
        self.assertEqual(consensus.stable_rank_count, 0)
        self.assertEqual(consensus.rank_count, 8)

    def test_cross_rank_consensus_can_still_require_stable_ranks(self) -> None:
        candidate = self._make_candidate()
        final_results = [
            self._make_final_result(rank, [candidate], world_size=8, stable=False)
            for rank in range(8)
        ]

        consensus = build_cross_rank_consensus(final_results, min_stable_fraction=0.5)

        self.assertEqual(consensus.status, "no_consensus")
        self.assertEqual(consensus.reason, "insufficient_stable_ranks")
        self.assertIsNone(consensus.selected_candidate)

    def test_cross_rank_consensus_reports_a_near_dead_heat_as_tied(self) -> None:
        # Two candidates within tie_epsilon: the winner is decided by how many
        # ranks ranked it first, not by which one noise nudged ahead.
        first = self._make_candidate(hook_position=1.0, param_fqn="block.2.weight")
        second = self._make_candidate(hook_position=0.5, param_fqn="block.0.weight")
        final_results = []
        for rank in range(3):
            leader = first if rank < 2 else second
            trailer = second if rank < 2 else first
            final_results.append(
                self._make_final_result(
                    rank,
                    [
                        self._make_candidate(
                            hook_position=leader.hook_position,
                            score_p10=100.0,
                            param_fqn=leader.param_fqn or "",
                        ),
                        self._make_candidate(
                            hook_position=trailer.hook_position,
                            score_p10=99.9,
                            param_fqn=trailer.param_fqn or "",
                        ),
                    ],
                    world_size=3,
                )
            )

        consensus = build_cross_rank_consensus(final_results, quorum_fraction=1.0)

        selected_candidate = consensus.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(consensus.status, "tied")
        self.assertEqual(consensus.reason, "scores_within_tie_epsilon")
        self.assertLess(consensus.runner_up_margin, consensus.tie_epsilon)
        # hook 1.0 led on 2 of 3 ranks.
        self.assertEqual(selected_candidate.hook_position, 1.0)

    def test_cross_rank_consensus_reports_a_clear_winner_as_selected(self) -> None:
        strong = self._make_candidate(hook_position=1.0, score_p10=90.0)
        weak = self._make_candidate(
            hook_position=0.5, score_p10=70.0, param_fqn="block.0.weight"
        )
        final_results = [
            self._make_final_result(rank, [strong, weak], world_size=3)
            for rank in range(3)
        ]

        consensus = build_cross_rank_consensus(final_results, quorum_fraction=1.0)

        selected_candidate = consensus.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(consensus.status, "selected")
        self.assertGreater(consensus.runner_up_margin, consensus.tie_epsilon)
        self.assertEqual(selected_candidate.hook_position, 1.0)

    def test_cross_rank_consensus_scores_candidates_on_a_common_rank_set(self) -> None:
        # "wide" is observed everywhere including a slow rank; "narrow" is missing
        # from that slow rank and would win a naive per-subset min().
        wide_scores = {0: 70.0, 1: 70.0, 2: 70.0, 3: 50.0}
        narrow_scores = {0: 65.0, 1: 65.0, 2: 65.0}
        final_results = []
        for rank in range(4):
            top_candidates = [
                self._make_candidate(
                    hook_position=1.0,
                    score_p10=wide_scores[rank],
                    param_fqn="block.2.weight",
                )
            ]
            if rank in narrow_scores:
                top_candidates.append(
                    self._make_candidate(
                        hook_position=0.5,
                        score_p10=narrow_scores[rank],
                        param_fqn="block.0.weight",
                    )
                )
            final_results.append(
                self._make_final_result(rank, top_candidates, world_size=4)
            )

        consensus = build_cross_rank_consensus(final_results, quorum_fraction=0.75)

        selected_candidate = consensus.selected_candidate
        assert selected_candidate is not None
        self.assertEqual(consensus.status, "selected")
        self.assertEqual(selected_candidate.hook_position, 1.0)
        self.assertEqual(selected_candidate.consensus_score, 70.0)
        self.assertEqual(selected_candidate.rank_count, 4)
        self.assertEqual(selected_candidate.scored_rank_count, 3)

    def test_auto_tuning_workflow_result_without_auto_apply_reports_consensus(
        self,
    ) -> None:
        result = build_ems_auto_tuning_workflow_result(
            final_results=self._make_final_results(),
            auto_apply=False,
            quorum_fraction=1.0,
            apply_num_batches=100,
        )

        self.assertEqual(result.status, "consensus_ready")
        self.assertFalse(result.auto_apply)
        self.assertIsNone(result.apply_job_request)
        self.assertEqual(result.consensus.status, "selected")

    def test_auto_tuning_workflow_result_with_auto_apply_requests_mast_job(
        self,
    ) -> None:
        result = build_ems_auto_tuning_workflow_result(
            final_results=self._make_final_results(),
            auto_apply=True,
            quorum_fraction=1.0,
            apply_num_batches=100,
        )

        apply_job_request = result.apply_job_request
        assert apply_job_request is not None
        self.assertEqual(result.status, "apply_job_requested")
        self.assertTrue(result.auto_apply)
        self.assertEqual(apply_job_request.job_type, "mast")
        self.assertEqual(apply_job_request.apply_num_batches, 100)
        self.assertIn(
            "data_loader.dataset.num_batches=100",
            apply_job_request.hydra_overrides,
        )
        self.assertEqual(
            apply_job_request.apex_tuning_input["data_loader"],
            {"dataset": {"num_batches": 100}},
        )
        # The apply job reuses the probe's model entity, so it must not inherit
        # the probe's dataloader cursor -- otherwise it resumes at the end of
        # the data the probe already consumed.
        self.assertIn(
            "checkpoint.skip_loading_data_checkpoint_on_start=true",
            apply_job_request.hydra_overrides,
        )
        self.assertEqual(
            apply_job_request.apex_tuning_input["checkpoint"],
            {"skip_loading_data_checkpoint_on_start": True},
        )
