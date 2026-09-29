#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Online FQN ranking for Memory Bouncer restore-hook selection."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Deque, Iterator, Protocol

import torch
from torch import nn
from torchrec.distributed.train_pipeline.backward_injection import (
    _position_to_index,
    _walk_outward,
    will_hook_fire,
)


logger: logging.Logger = logging.getLogger(__name__)

GB_IN_BYTES: float = 1024.0**3
FQNKey = tuple[str, float]


@dataclass(frozen=True)
class FQNScoreDetails:
    """Auditable score details for one FQN probe candidate."""

    peak_timing_score: float
    restore_overlap_score: float
    memory_benefit_score: float
    stability_score: float
    coverage_score: float
    missing_marker_penalty: float
    early_restore_penalty: float
    late_restore_penalty: float
    median_peak_clearance_us: float
    p10_peak_clearance_us: float
    observed_marker_count: int
    expected_marker_count: int


@dataclass(frozen=True)
class FQNRecommendation:
    """Ranked recommendation for an EMS restore hook."""

    fqn: str
    hook_position: float
    score: float
    score_details: FQNScoreDetails
    marker_count: int
    median_fire_time_us: float
    safety_margin_us: float
    overlap_window_us: float
    confidence: float
    param_fqn: str | None
    hydra_overrides: tuple[str, ...]
    apex_tuning_input: dict[str, object]


@dataclass(frozen=True)
class FQNCandidateEvaluation:
    """Score details for one observed online candidate."""

    fqn: str
    hook_position: float
    status: str
    score_details: FQNScoreDetails
    marker_count: int
    median_fire_time_us: float
    safe_time_us: float
    safety_margin_us: float
    overlap_window_us: float
    confidence: float
    score: float
    param_fqn: str | None


@dataclass(frozen=True)
class FQNTuningResult:
    """Complete result for a memory-bouncer FQN recommendation run."""

    workload: str
    peak_time_us: int
    peak_gb: float
    recommendations: list[FQNRecommendation]
    candidate_evaluations: list[FQNCandidateEvaluation]
    needs_probe_run: bool
    status_message: str


FQNRankingResult = FQNTuningResult


@dataclass(frozen=True)
class FQNFinalCandidate:
    """Per-rank final summary for one online FQN candidate."""

    fqn: str
    hook_position: float
    param_fqn: str | None
    score_p50: float
    score_p10: float
    confidence_p50: float
    safe_count: int
    observed_count: int
    scored_iters: int
    safety_margin_us_p10: float
    safety_margin_us_cv: float
    overlap_window_us_p10: float
    eligible: bool
    noise_reason: str | None

    @property
    def candidate_id(self) -> tuple[str, float, str | None]:
        return (self.fqn, self.hook_position, self.param_fqn)


@dataclass(frozen=True)
class FQNRankerFinalResult:
    """One final online FQN ranking event emitted by one rank."""

    event: str
    rank: int
    world_size: int
    stable: bool
    status: str
    scored_iters: int
    top_candidates: list[FQNFinalCandidate]
    selected_candidate: FQNFinalCandidate | None
    hydra_overrides: tuple[str, ...]
    apex_tuning_input: dict[str, object]


@dataclass(frozen=True)
class FQNConsensusCandidate:
    """Cross-rank aggregate for a candidate selected by external consensus."""

    fqn: str
    hook_position: float
    param_fqn: str | None
    rank_count: int
    scored_rank_count: int
    expected_rank_count: int
    consensus_score: float
    score_p50: float
    safety_margin_us_p10: float
    overlap_window_us_p10: float
    confidence_p50: float


@dataclass(frozen=True)
class FQNCrossRankConsensus:
    """External aggregator result for online FQN ranking final events."""

    status: str
    reason: str
    selected_candidate: FQNConsensusCandidate | None
    rank_count: int
    stable_rank_count: int
    expected_rank_count: int
    required_rank_count: int
    quorum_fraction: float
    min_stable_fraction: float
    tie_epsilon: float
    runner_up_margin: float
    hydra_overrides: tuple[str, ...]
    apex_tuning_input: dict[str, object]


@dataclass(frozen=True)
class FQNApplyJobRequest:
    """Apply-phase MAST relaunch request produced by the Flow aggregator."""

    job_type: str
    hydra_overrides: tuple[str, ...]
    apex_tuning_input: dict[str, object]
    apply_num_batches: int | None


@dataclass(frozen=True)
class FQNAutoTuningWorkflowResult:
    """Flow-level action after cross-rank online FQN consensus."""

    status: str
    reason: str
    auto_apply: bool
    consensus: FQNCrossRankConsensus
    apply_job_request: FQNApplyJobRequest | None


class IterPeakDetector(Protocol):
    """Per-iteration peak detector backend used by ``OnlineFQNRanker``."""

    def on_iter_start(self, ranker: "OnlineFQNRanker") -> None:
        """Start measuring the current iteration."""

    def on_iter_end(self) -> None:
        """Stop measuring the current iteration."""


@dataclass(frozen=True)
class _HookCandidate:
    fqn: str
    hook_position: float
    param_fqn: str
    param: nn.Parameter
    param_index: int
    num_params: int

    @property
    def key(self) -> FQNKey:
        return (self.fqn, self.hook_position)


@dataclass(frozen=True)
class _PeakRecord:
    iter_idx: int
    rank: int
    time_ns: int
    peak_bytes: int


class CUDAMemoryPeakPoller:
    """Fallback peak detector that samples ``torch.cuda.memory_allocated``."""

    def __init__(
        self,
        device: torch.device | None = None,
        poll_interval_s: float = 0.0001,
        memory_reader: Callable[[], int] | None = None,
    ) -> None:
        self._device = device
        self._poll_interval_s = poll_interval_s
        self._memory_reader = memory_reader
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def on_iter_start(self, ranker: "OnlineFQNRanker") -> None:
        if self._memory_reader is None and not self._can_sample_cuda():
            return
        try:
            if self._memory_reader is None:
                torch.cuda.reset_peak_memory_stats(device=self._device)
        except RuntimeError:
            logger.debug("Unable to reset CUDA peak memory stats.", exc_info=True)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(ranker, self._stop_event),
            daemon=True,
            name="online_fqn_ranker_peak_poller",
        )
        self._thread.start()

    def on_iter_end(self) -> None:
        stop_event = self._stop_event
        thread = self._thread
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)
        self._stop_event = None
        self._thread = None

    def _can_sample_cuda(self) -> bool:
        return torch.cuda.is_available() and (
            self._device is None or self._device.type == "cuda"
        )

    def _read_memory(self) -> int:
        if self._memory_reader is not None:
            return self._memory_reader()
        return int(torch.cuda.memory_allocated(device=self._device))

    def _poll_loop(
        self,
        ranker: "OnlineFQNRanker",
        stop_event: threading.Event,
    ) -> None:
        max_bytes = -1
        while not stop_event.is_set():
            time_ns = ranker.clock_ns()
            current_bytes = self._read_memory()
            if current_bytes > max_bytes:
                max_bytes = current_bytes
                ranker.record_peak(time_ns, current_bytes)
            time.sleep(self._poll_interval_s)


class OnlineFQNRanker:
    """Ranks FQNs online, per iteration, using a single clock source."""

    def __init__(
        self,
        model: nn.Module,
        hook_positions: tuple[float, ...] = (0.01,),
        warmup_iters: int = 50,
        measure_every_n: int = 1,
        stability_window: int = 10,
        top_n: int = 5,
        stability_top_n: int | None = None,
        max_scored_iters: int | None = None,
        steady_state_window: int | None = None,
        steady_state_peak_cv_threshold: float = 0.05,
        max_depth: int = 1,
        callback_latency_us: float = 1000.0,
        min_observation_fraction: float = 0.8,
        max_safety_margin_cv: float = 0.5,
        log_per_iter_results: bool = False,
        fqn_allowlist: set[str] | None = None,
        fqn_blocklist: set[str] | None = None,
        max_candidates: int | None = None,
        peak_detector: IterPeakDetector | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        rank: int | None = None,
        world_size: int | None = None,
        workload: str = "memory_bouncer",
    ) -> None:
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative.")
        if measure_every_n <= 0:
            raise ValueError("measure_every_n must be positive.")
        if stability_window <= 0:
            raise ValueError("stability_window must be positive.")
        if top_n <= 0:
            raise ValueError("top_n must be positive.")
        if max_scored_iters is not None and max_scored_iters <= 0:
            raise ValueError("max_scored_iters must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if callback_latency_us < 0.0:
            raise ValueError("callback_latency_us must be non-negative.")
        if not 0.0 <= min_observation_fraction <= 1.0:
            raise ValueError("min_observation_fraction must be in [0.0, 1.0].")
        if max_safety_margin_cv < 0.0:
            raise ValueError("max_safety_margin_cv must be non-negative.")
        if max_candidates is not None and max_candidates < 0:
            raise ValueError("max_candidates must be non-negative.")

        self._model = _unwrap_model(model)
        self._hook_positions = hook_positions
        self._warmup_iters = warmup_iters
        self._measure_every_n = measure_every_n
        self._stability_window = stability_window
        self._top_n = top_n
        # How many ranked candidates must agree for ``is_stable``. Separate from
        # ``top_n`` (the report length) because widening the report to give
        # cross-rank quorum more candidates would otherwise make convergence
        # strictly harder -- a longer tuple has more ways to differ.
        self._stability_top_n = stability_top_n or top_n
        self._max_scored_iters = max_scored_iters or stability_window * 3
        self._steady_state_window = steady_state_window or stability_window
        self._steady_state_peak_cv_threshold = steady_state_peak_cv_threshold
        self._max_depth = max_depth
        self._callback_latency_us = callback_latency_us
        self._min_observation_fraction = min_observation_fraction
        self._max_safety_margin_cv = max_safety_margin_cv
        self._log_per_iter_results = log_per_iter_results
        self._fqn_allowlist = fqn_allowlist
        self._fqn_blocklist = fqn_blocklist
        self._max_candidates = max_candidates
        self._peak_detector = peak_detector
        self._clock_ns = clock_ns
        self._rank = _get_rank() if rank is None else rank
        self._world_size = _get_world_size() if world_size is None else world_size
        self._workload = workload

        self._candidates: list[_HookCandidate] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._current_iter: int | None = None
        self._iter_start_time_ns: int | None = None
        self._backward_start_time_ns: int | None = None
        self._measurement_active: bool = False
        self._peak: _PeakRecord | None = None
        self._fire_times: dict[str, dict[float, int]] = {}
        self._peak_bytes_history: Deque[int] = deque(maxlen=self._steady_state_window)
        self._safety_margin_history: dict[FQNKey, Deque[float]] = {}
        self._evaluation_history: dict[FQNKey, Deque[FQNCandidateEvaluation]] = {}
        self._topk_history: Deque[tuple[FQNKey, ...]] = deque(
            maxlen=self._stability_window
        )
        self._last_result: FQNRankingResult | None = None
        self._locked_recommendation: FQNRecommendation | None = None
        self._measured_iter_count: int = 0
        self._scored_iter_count: int = 0
        self._terminal: bool = False
        self._final_result: FQNRankerFinalResult | None = None

    def setup_hooks(self) -> None:
        """Register backward hooks on all candidate FQNs."""
        if self._terminal:
            return
        if self._handles:
            return
        self._candidates = _dedupe_hook_candidates(
            list(
                _iter_hook_candidates(
                    model=self._model,
                    hook_positions=self._hook_positions,
                    max_depth=self._max_depth,
                    fqn_allowlist=self._fqn_allowlist,
                    fqn_blocklist=self._fqn_blocklist,
                    max_candidates=self._max_candidates,
                )
            )
        )
        for candidate in self._candidates:
            self._handles.append(
                candidate.param.register_post_accumulate_grad_hook(
                    self._make_hook(candidate)
                )
            )

    def remove_hooks(self) -> None:
        """Unregister all hooks created by ``setup_hooks``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def on_iter_start(self, iter_idx: int) -> None:
        """Called at training iter start. Resets per-iter state only."""
        self._current_iter = iter_idx
        self._iter_start_time_ns = self.clock_ns()
        self._backward_start_time_ns = None
        self._peak = None
        self._fire_times = {}
        self._measurement_active = False

    def on_backward_start(self, iter_idx: int) -> None:
        """Called immediately before backward to start online measurement."""
        if self._terminal:
            return
        assert self._current_iter == iter_idx
        self._backward_start_time_ns = self.clock_ns()
        self._measurement_active = self._should_measure(iter_idx)
        if self._measurement_active and self._peak_detector is not None:
            self._peak_detector.on_iter_start(self)

    def record_peak(self, peak_time_ns: int, peak_bytes: int) -> None:
        """Record the current backward peak using the ranker's clock epoch."""
        if not self._measurement_active:
            return
        iter_idx = self._require_current_iter()
        backward_start_time_ns = self._require_backward_start_time()
        now_ns = self.clock_ns()
        assert backward_start_time_ns <= peak_time_ns <= now_ns, (
            "peak_time_ns must come from the same perf_counter_ns clock epoch "
            "and current backward window."
        )
        if self._peak is None or peak_bytes > self._peak.peak_bytes:
            self._peak = _PeakRecord(
                iter_idx=iter_idx,
                rank=self._rank,
                time_ns=peak_time_ns,
                peak_bytes=peak_bytes,
            )

    def on_iter_end(self, iter_idx: int) -> FQNRankingResult | None:
        """Compatibility wrapper for callers that end measurement after backward."""
        return self.on_backward_end(iter_idx)

    def on_backward_end(self, iter_idx: int) -> FQNRankingResult | None:
        """Called after backward. Scores FQNs if measurement is active."""
        if not self._measurement_active:
            return None
        backward_end_time_ns = self.clock_ns()
        self._measurement_active = False
        if self._peak_detector is not None:
            self._peak_detector.on_iter_end()
        assert self._current_iter == iter_idx
        peak = self._peak
        if peak is None:
            logger.debug("OnlineFQNRanker skipped iter %d: no peak recorded.", iter_idx)
            return None
        assert peak.iter_idx == iter_idx
        assert peak.rank == self._rank
        self._peak_bytes_history.append(peak.peak_bytes)
        self._measured_iter_count += 1
        if not self._is_steady_peak(peak.peak_bytes):
            if self._measured_iter_count >= self._max_scored_iters:
                self._finalize("unstable")
            return None
        result = self._build_result(peak, backward_end_time_ns)
        self._scored_iter_count += 1
        self._last_result = result
        self._record_history(result)
        if self._log_per_iter_results:
            self._emit_structured_log(iter_idx, peak, result)
        if self.is_stable():
            self._finalize("stable")
        elif self._scored_iter_count >= self._max_scored_iters:
            self._finalize("unstable")
        return result

    def get_recommendation(self) -> FQNRecommendation | None:
        """Returns the current best FQN with confidence score."""
        if self._locked_recommendation is not None:
            return self._locked_recommendation
        if self._last_result is None or not self._last_result.recommendations:
            return None
        return self._last_result.recommendations[0]

    def get_final_result(self) -> FQNRankerFinalResult | None:
        """Returns the terminal per-rank final result, if one has been emitted."""
        return self._final_result

    def is_stable(self) -> bool:
        """True if ranking has converged over ``stability_window`` iters."""
        if len(self._topk_history) < self._stability_window:
            return False
        latest = self._topk_history[-1]
        return bool(latest) and all(topk == latest for topk in self._topk_history)

    def clock_ns(self) -> int:
        """Return the ranker's single timing source."""
        return self._clock_ns()

    def _make_hook(self, candidate: _HookCandidate) -> Callable[[torch.Tensor], None]:
        def _hook(_param: torch.Tensor) -> None:
            if not self._measurement_active:
                return
            self._require_current_iter()
            self._fire_times.setdefault(candidate.fqn, {})[
                candidate.hook_position
            ] = self.clock_ns()

        return _hook

    def _build_result(
        self,
        peak: _PeakRecord,
        backward_end_time_ns: int,
    ) -> FQNRankingResult:
        assert self._current_iter == peak.iter_idx
        evaluations = [
            self._evaluate_candidate(candidate, peak, backward_end_time_ns)
            for candidate in self._candidates
        ]
        recommendations = [
            _recommendation_from_evaluation(evaluation)
            for evaluation in evaluations
            if evaluation.status == "safe"
        ]
        recommendations.sort(
            key=lambda recommendation: (
                -recommendation.score,
                recommendation.median_fire_time_us,
                recommendation.fqn,
            )
        )
        evaluations.sort(
            key=lambda evaluation: (
                evaluation.status != "safe",
                -evaluation.score,
                evaluation.median_fire_time_us,
                evaluation.fqn,
            )
        )
        return FQNTuningResult(
            workload=self._workload,
            peak_time_us=int(peak.time_ns / 1000),
            peak_gb=peak.peak_bytes / GB_IN_BYTES,
            recommendations=recommendations[: self._top_n],
            candidate_evaluations=evaluations,
            needs_probe_run=not recommendations,
            status_message=(
                "Recommendations ranked successfully."
                if recommendations
                else "No safe online FQN candidates fired after peak memory."
            ),
        )

    def _evaluate_candidate(
        self,
        candidate: _HookCandidate,
        peak: _PeakRecord,
        backward_end_time_ns: int,
    ) -> FQNCandidateEvaluation:
        fire_time_ns = self._fire_times.get(candidate.fqn, {}).get(
            candidate.hook_position
        )
        if fire_time_ns is None:
            return _missing_evaluation(candidate, peak, "skipped_by_compiler")
        safety_margin_us = (fire_time_ns - peak.time_ns) / 1000.0
        overlap_window_us = max(0.0, (backward_end_time_ns - fire_time_ns) / 1000.0)
        if safety_margin_us <= 0.0:
            return _rejected_evaluation(
                candidate,
                peak,
                fire_time_ns,
                safety_margin_us,
                overlap_window_us,
                early_restore_penalty=100.0,
                late_restore_penalty=0.0,
            )
        if overlap_window_us < self._callback_latency_us:
            return _rejected_evaluation(
                candidate,
                peak,
                fire_time_ns,
                safety_margin_us,
                overlap_window_us,
                early_restore_penalty=0.0,
                late_restore_penalty=100.0,
            )
        confidence, coverage_score, stability_score = self._confidence(
            candidate.key,
            safety_margin_us,
        )
        timing_window_us = max(1.0, (backward_end_time_ns - peak.time_ns) / 1000.0)
        total_backward_us = max(
            1.0,
            (
                backward_end_time_ns
                - min(
                    fire_time
                    for fqn_fire_times in self._fire_times.values()
                    for fire_time in fqn_fire_times.values()
                )
            )
            / 1000.0,
        )
        timing_score = _clamp(1.0 - safety_margin_us / timing_window_us, 0.0, 1.0)
        headroom_score = _clamp(
            (overlap_window_us - self._callback_latency_us) / total_backward_us,
            0.0,
            1.0,
        )
        score = 100.0 * (
            0.45 * timing_score + 0.35 * headroom_score + 0.20 * confidence
        )
        return FQNCandidateEvaluation(
            fqn=candidate.fqn,
            hook_position=candidate.hook_position,
            status="safe",
            score_details=FQNScoreDetails(
                peak_timing_score=timing_score,
                restore_overlap_score=headroom_score,
                memory_benefit_score=1.0,
                stability_score=stability_score,
                coverage_score=coverage_score,
                missing_marker_penalty=0.0,
                early_restore_penalty=0.0,
                late_restore_penalty=0.0,
                median_peak_clearance_us=safety_margin_us,
                p10_peak_clearance_us=safety_margin_us,
                observed_marker_count=1,
                expected_marker_count=self._stability_window,
            ),
            marker_count=1,
            median_fire_time_us=fire_time_ns / 1000.0,
            safe_time_us=peak.time_ns / 1000.0,
            safety_margin_us=safety_margin_us,
            overlap_window_us=overlap_window_us,
            confidence=confidence,
            score=score,
            param_fqn=candidate.param_fqn,
        )

    def _confidence(
        self, key: FQNKey, safety_margin_us: float
    ) -> tuple[float, float, float]:
        history = list(self._safety_margin_history.get(key, ())) + [safety_margin_us]
        coverage_score = _clamp(len(history) / self._stability_window, 0.0, 1.0)
        stability_score = 1.0 / (1.0 + _coefficient_of_variation(history))
        return coverage_score * stability_score, coverage_score, stability_score

    def _record_history(self, result: FQNRankingResult) -> None:
        for evaluation in result.candidate_evaluations:
            key = (evaluation.fqn, evaluation.hook_position)
            self._evaluation_history.setdefault(
                key,
                deque(maxlen=self._stability_window),
            ).append(evaluation)
            if evaluation.status != "safe":
                continue
            self._safety_margin_history.setdefault(
                key,
                deque(maxlen=self._stability_window),
            ).append(evaluation.safety_margin_us)
        topk = tuple(
            (recommendation.fqn, recommendation.hook_position)
            for recommendation in result.recommendations[: self._stability_top_n]
        )
        self._topk_history.append(topk)
        if self.is_stable() and result.recommendations:
            self._locked_recommendation = result.recommendations[0]

    def _finalize(self, status: str) -> None:
        if self._terminal:
            return
        stable = status == "stable"
        top_candidates = self._final_top_candidates()
        selected_candidate = (
            top_candidates[0]
            if stable and top_candidates and top_candidates[0].eligible
            else None
        )
        hydra_overrides: tuple[str, ...] = ()
        apex_tuning_input: dict[str, object] = {}
        if selected_candidate is not None:
            hydra_overrides = build_ems_apply_hydra_overrides(
                selected_candidate.fqn,
                selected_candidate.hook_position,
            )
            apex_tuning_input = build_ems_apply_tuning_payload(
                selected_candidate.fqn,
                selected_candidate.hook_position,
            )
        final_result = FQNRankerFinalResult(
            event="online_fqn_ranker_final",
            rank=self._rank,
            world_size=self._world_size,
            stable=stable,
            status=status,
            scored_iters=self._scored_iter_count,
            top_candidates=top_candidates,
            selected_candidate=selected_candidate,
            hydra_overrides=hydra_overrides,
            apex_tuning_input=apex_tuning_input,
        )
        self._final_result = final_result
        self._terminal = True
        self.remove_hooks()
        self._emit_final_log(final_result)

    def _final_top_candidates(self) -> list[FQNFinalCandidate]:
        candidates = [
            self._summarize_final_candidate(evaluations)
            for evaluations in self._evaluation_history.values()
        ]
        candidates.sort(
            key=lambda candidate: (
                not candidate.eligible,
                -candidate.score_p10,
                -candidate.score_p50,
                candidate.safety_margin_us_cv,
                candidate.fqn,
                candidate.hook_position,
            )
        )
        return candidates[: self._top_n]

    def _summarize_final_candidate(
        self,
        evaluations: Deque[FQNCandidateEvaluation],
    ) -> FQNFinalCandidate:
        evaluation_list = list(evaluations)
        assert evaluation_list
        first = evaluation_list[0]
        scored_iters = len(evaluation_list)
        safe_evaluations = [
            evaluation for evaluation in evaluation_list if evaluation.status == "safe"
        ]
        observed_count = sum(
            1 for evaluation in evaluation_list if evaluation.marker_count > 0
        )
        scores = [evaluation.score for evaluation in safe_evaluations]
        confidences = [evaluation.confidence for evaluation in safe_evaluations]
        safety_margins = [
            evaluation.safety_margin_us for evaluation in safe_evaluations
        ]
        overlap_windows = [
            evaluation.overlap_window_us for evaluation in safe_evaluations
        ]
        safety_margin_cv = _coefficient_of_variation(safety_margins)
        noise_reason = self._candidate_noise_reason(
            scored_iters=scored_iters,
            safe_count=len(safe_evaluations),
            observed_count=observed_count,
            safety_margin_us_p10=_percentile(safety_margins, 0.10),
            overlap_window_us_p10=_percentile(overlap_windows, 0.10),
            safety_margin_cv=safety_margin_cv,
        )
        return FQNFinalCandidate(
            fqn=first.fqn,
            hook_position=first.hook_position,
            param_fqn=first.param_fqn,
            score_p50=_percentile(scores, 0.50),
            score_p10=_percentile(scores, 0.10),
            confidence_p50=_percentile(confidences, 0.50),
            safe_count=len(safe_evaluations),
            observed_count=observed_count,
            scored_iters=scored_iters,
            safety_margin_us_p10=_percentile(safety_margins, 0.10),
            safety_margin_us_cv=safety_margin_cv,
            overlap_window_us_p10=_percentile(overlap_windows, 0.10),
            eligible=noise_reason is None,
            noise_reason=noise_reason,
        )

    def _candidate_noise_reason(
        self,
        scored_iters: int,
        safe_count: int,
        observed_count: int,
        safety_margin_us_p10: float,
        overlap_window_us_p10: float,
        safety_margin_cv: float,
    ) -> str | None:
        if scored_iters == 0:
            return "not_scored"
        if observed_count / scored_iters < self._min_observation_fraction:
            return "low_observation"
        if safe_count < scored_iters:
            return "unsafe_or_missing"
        if safety_margin_us_p10 <= 0.0:
            return "pre_peak"
        if overlap_window_us_p10 < self._callback_latency_us:
            return "insufficient_overlap"
        if safety_margin_cv > self._max_safety_margin_cv:
            return "noisy_safety_margin"
        return None

    def _emit_structured_log(
        self,
        iter_idx: int,
        peak: _PeakRecord,
        result: FQNRankingResult,
    ) -> None:
        selected = result.recommendations[0] if result.recommendations else None
        payload = {
            "iter": iter_idx,
            "rank": self._rank,
            "peak_time_ns": peak.time_ns,
            "peak_bytes": peak.peak_bytes,
            "fqn_rankings": [
                asdict(evaluation) for evaluation in result.candidate_evaluations
            ],
            "selected_fqn": selected.fqn if selected is not None else None,
            "selected_hook_position": (
                selected.hook_position if selected is not None else None
            ),
            "confidence": selected.confidence if selected is not None else 0.0,
            "fqn_backward_durations_us": self._fqn_backward_durations_us(),
        }
        logger.info("online_fqn_ranker_result %s", json.dumps(payload, sort_keys=True))

    def _emit_final_log(self, final_result: FQNRankerFinalResult) -> None:
        logger.info(
            "online_fqn_ranker_final %s",
            json.dumps(asdict(final_result), sort_keys=True),
        )

    def _fqn_backward_durations_us(self) -> dict[str, float]:
        durations: dict[str, float] = {}
        for fqn, fire_times in self._fire_times.items():
            early_position_time = fire_times.get(0.01)
            late_position_time = fire_times.get(1.0)
            if early_position_time is not None and late_position_time is not None:
                durations[fqn] = (early_position_time - late_position_time) / 1000.0
        return durations

    def _is_steady_peak(self, peak_bytes: int) -> bool:
        if self._steady_state_window <= 1:
            return True
        if len(self._peak_bytes_history) < self._steady_state_window:
            return False
        values = [float(value) for value in self._peak_bytes_history]
        mean = sum(values) / len(values)
        if mean == 0.0:
            return peak_bytes == 0
        cv = _coefficient_of_variation(values)
        relative_delta = abs(float(peak_bytes) - mean) / mean
        return (
            cv <= self._steady_state_peak_cv_threshold
            and relative_delta <= self._steady_state_peak_cv_threshold
        )

    def _should_measure(self, iter_idx: int) -> bool:
        return iter_idx >= self._warmup_iters and (
            (iter_idx - self._warmup_iters) % self._measure_every_n == 0
        )

    def _require_current_iter(self) -> int:
        assert self._current_iter is not None
        return self._current_iter

    def _require_iter_start_time(self) -> int:
        assert self._iter_start_time_ns is not None
        return self._iter_start_time_ns

    def _require_backward_start_time(self) -> int:
        assert self._backward_start_time_ns is not None
        return self._backward_start_time_ns


def build_ems_hydra_overrides(fqn: str, hook_position: float) -> tuple[str, ...]:
    """Return Hydra overrides that apply an EMS FQN recommendation."""
    return build_ems_apply_hydra_overrides(fqn, hook_position)


def build_ems_apply_hydra_overrides(
    fqn: str,
    hook_position: float,
    apply_num_batches: int | None = None,
    num_batches_override: str = "data_loader.dataset.num_batches",
) -> tuple[str, ...]:
    """Return Hydra overrides for the apply-phase relaunch job.

    ``data_loader`` and ``checkpoint`` are siblings of ``training`` at the
    launcher config root, so they take no ``training.`` prefix — Hydra runs in
    struct mode and rejects the prefixed form outright.

    The apply job reuses the probe's model entity, so without
    ``skip_loading_data_checkpoint_on_start`` it resumes the probe's dataloader
    cursor and starts where the probe stopped reading — leaving it a handful of
    batches before the dataset is exhausted. Skipping only *on start* keeps a
    MAST restart of the apply job resuming normally.

    ``skip_loading_data_checkpoint_on_start`` covers the dataloader cursor but
    *not* the batch counter: that is restored from the trainer state in the
    model checkpoint, and ``DataLoaderWrapper.set_training_state`` then serves
    only ``apply_num_batches - restored_count`` batches. So
    ``apply_num_batches`` is the **lineage total across probe and apply**, not
    the apply job's own budget — pass the job's configured ``num_batches`` and
    keep the probe's cap well under it. If the probe consumes the whole total
    the apply job trains zero batches and still exits COMPLETE. A non-positive
    value (``-1``) means unlimited and is exempt: the dataloader maps it to no
    limit and skips the subtraction entirely.
    """
    overrides = (
        "training.pipeline_type=mb-customized-order-sparse-dist",
        "training.memory_bouncer.embedding_memory_stashing.enable=true",
        f"training.memory_bouncer.embedding_memory_stashing.restore_fqn={fqn}",
        "training.memory_bouncer.embedding_memory_stashing."
        f"hook_position={hook_position}",
        "training.memory_bouncer.auto_tuning.enable=false",
        "training.memory_bouncer.auto_tuning.enable_online_ranking=false",
        "training.memory_bouncer.auto_tuning.auto_apply=false",
        "checkpoint.skip_loading_data_checkpoint_on_start=true",
    )
    if apply_num_batches is None:
        return overrides
    return overrides + (f"{num_batches_override}={apply_num_batches}",)


def build_ems_apex_tuning_payload(
    fqn: str,
    hook_position: float,
) -> dict[str, object]:
    """Return the ApexTuningInput-shaped payload for an EMS recommendation."""
    return build_ems_apply_tuning_payload(fqn, hook_position)


def build_ems_apply_tuning_payload(
    fqn: str,
    hook_position: float,
    apply_num_batches: int | None = None,
) -> dict[str, object]:
    """Return the ApexTuningInput-shaped payload for an apply-phase relaunch."""
    payload: dict[str, object] = {
        "pipeline_type": "mb-customized-order-sparse-dist",
        "memory_bouncer": {
            "embedding_memory_stashing": {
                "enable": True,
                "restore_fqn": fqn,
                "hook_position": hook_position,
            },
            "auto_tuning": {
                "enable": False,
                "enable_online_ranking": False,
                "auto_apply": False,
            },
        },
        "checkpoint": {"skip_loading_data_checkpoint_on_start": True},
    }
    if apply_num_batches is not None:
        payload["data_loader"] = {"dataset": {"num_batches": apply_num_batches}}
    return payload


def build_ems_auto_tuning_workflow_result(
    final_results: list[FQNRankerFinalResult],
    auto_apply: bool,
    quorum_fraction: float = 0.95,
    min_rank_count: int | None = None,
    apply_num_batches: int | None = None,
    num_batches_override: str = "data_loader.dataset.num_batches",
    min_stable_fraction: float = 0.0,
    tie_epsilon: float = 0.01,
) -> FQNAutoTuningWorkflowResult:
    """Return the Flow action for an online Memory Bouncer probe result.

    A ``tied`` consensus still selects and applies: the tie-break is
    deterministic, so the alternative would be to discard a usable answer
    because two candidates are equally good.
    """
    consensus = build_cross_rank_consensus(
        final_results=final_results,
        quorum_fraction=quorum_fraction,
        min_rank_count=min_rank_count,
        min_stable_fraction=min_stable_fraction,
        tie_epsilon=tie_epsilon,
    )
    if consensus.selected_candidate is None:
        return FQNAutoTuningWorkflowResult(
            status="no_consensus",
            reason=consensus.reason,
            auto_apply=auto_apply,
            consensus=consensus,
            apply_job_request=None,
        )
    if not auto_apply:
        return FQNAutoTuningWorkflowResult(
            status="consensus_ready",
            reason="auto_apply_disabled",
            auto_apply=False,
            consensus=consensus,
            apply_job_request=None,
        )

    selected_candidate = consensus.selected_candidate
    return FQNAutoTuningWorkflowResult(
        status="apply_job_requested",
        reason="auto_apply_enabled",
        auto_apply=True,
        consensus=consensus,
        apply_job_request=FQNApplyJobRequest(
            job_type="mast",
            hydra_overrides=build_ems_apply_hydra_overrides(
                selected_candidate.fqn,
                selected_candidate.hook_position,
                apply_num_batches=apply_num_batches,
                num_batches_override=num_batches_override,
            ),
            apex_tuning_input=build_ems_apply_tuning_payload(
                selected_candidate.fqn,
                selected_candidate.hook_position,
                apply_num_batches=apply_num_batches,
            ),
            apply_num_batches=apply_num_batches,
        ),
    )


def build_cross_rank_consensus(
    final_results: list[FQNRankerFinalResult],
    quorum_fraction: float = 0.95,
    min_rank_count: int | None = None,
    min_stable_fraction: float = 0.0,
    tie_epsilon: float = 0.01,
) -> FQNCrossRankConsensus:
    """Select a global EMS recommendation from per-rank final events.

    This is intended to run outside the training job after logs have shipped,
    so it does not add collectives or perturb the measured training loop.

    Every rank contributes its ``eligible`` candidates. ``result.stable`` only
    records whether a rank's *ordering* repeated for ``stability_window``
    consecutive scored iterations; when the top candidates sit within a fraction
    of a percent of each other that ordering is noise, so gating admission on it
    discards ranks whose measurements are perfectly sound. It is reported as
    ``stable_rank_count`` telemetry instead, and only enforced when the caller
    sets ``min_stable_fraction`` above zero.

    Candidates are scored on the set of ranks that observed *every*
    quorum-passing candidate, because ``consensus_score`` is a min over ranks
    and a candidate missing from the slowest ranks would otherwise win on an
    easier subset.

    When the winner does not beat the runner-up by ``tie_epsilon`` in relative
    terms the result is reported as ``tied`` and resolved by fleet plurality of
    the per-rank first choice, so a near-dead-heat produces a reproducible pick
    rather than whichever candidate noise happened to favour.
    """
    if not 0.0 < quorum_fraction <= 1.0:
        raise ValueError("quorum_fraction must be in (0.0, 1.0].")
    if not 0.0 <= min_stable_fraction <= 1.0:
        raise ValueError("min_stable_fraction must be in [0.0, 1.0].")
    if tie_epsilon < 0.0:
        raise ValueError("tie_epsilon must be non-negative.")
    if min_rank_count is not None and min_rank_count <= 0:
        raise ValueError("min_rank_count must be positive.")
    if not final_results:
        return _empty_consensus(
            reason="no_final_results",
            quorum_fraction=quorum_fraction,
            min_stable_fraction=min_stable_fraction,
        )

    expected_rank_count = max(
        max(result.world_size for result in final_results),
        len(final_results),
    )
    stable_results = [result for result in final_results if result.stable]
    if min_stable_fraction > 0.0:
        required_stable_rank_count = max(
            1,
            math.ceil(min_stable_fraction * expected_rank_count),
        )
        if len(stable_results) < required_stable_rank_count:
            return _no_consensus(
                reason="insufficient_stable_ranks",
                rank_count=len(final_results),
                stable_rank_count=len(stable_results),
                expected_rank_count=expected_rank_count,
                required_rank_count=required_stable_rank_count,
                quorum_fraction=quorum_fraction,
                min_stable_fraction=min_stable_fraction,
            )

    required_rank_count = min_rank_count or max(
        1,
        math.ceil(quorum_fraction * len(final_results)),
    )
    grouped_candidates: dict[
        tuple[str, float, str | None], dict[int, FQNFinalCandidate]
    ] = {}
    for result in final_results:
        for candidate in result.top_candidates:
            if not candidate.eligible:
                continue
            observations = grouped_candidates.setdefault(candidate.candidate_id, {})
            observations.setdefault(result.rank, candidate)

    quorum_candidates = {
        candidate_id: observations
        for candidate_id, observations in grouped_candidates.items()
        if len(observations) >= required_rank_count
    }
    if not quorum_candidates:
        return _no_consensus(
            reason="no_candidate_met_quorum",
            rank_count=len(final_results),
            stable_rank_count=len(stable_results),
            expected_rank_count=expected_rank_count,
            required_rank_count=required_rank_count,
            quorum_fraction=quorum_fraction,
            min_stable_fraction=min_stable_fraction,
        )

    rank_sets = [frozenset(observations) for observations in quorum_candidates.values()]
    scored_ranks = rank_sets[0].intersection(*rank_sets[1:])
    if not scored_ranks:
        return _no_consensus(
            reason="no_common_rank_set",
            rank_count=len(final_results),
            stable_rank_count=len(stable_results),
            expected_rank_count=expected_rank_count,
            required_rank_count=required_rank_count,
            quorum_fraction=quorum_fraction,
            min_stable_fraction=min_stable_fraction,
        )

    consensus_candidates = [
        _build_consensus_candidate(
            candidate_id,
            observations,
            scored_ranks,
            expected_rank_count,
        )
        for candidate_id, observations in quorum_candidates.items()
    ]
    consensus_candidates.sort(
        key=lambda candidate: (
            -candidate.consensus_score,
            -candidate.score_p50,
            -candidate.overlap_window_us_p10,
            candidate.safety_margin_us_p10,
            candidate.fqn,
            candidate.hook_position,
        )
    )
    selected_candidate = consensus_candidates[0]
    runner_up_margin = _relative_margin(consensus_candidates)
    status, reason = "selected", "candidate_met_quorum"
    if runner_up_margin < tie_epsilon:
        status, reason = "tied", "scores_within_tie_epsilon"
        selected_candidate = _resolve_tie_by_plurality(
            consensus_candidates,
            final_results,
        )
    return FQNCrossRankConsensus(
        status=status,
        reason=reason,
        selected_candidate=selected_candidate,
        rank_count=selected_candidate.rank_count,
        stable_rank_count=len(stable_results),
        expected_rank_count=expected_rank_count,
        required_rank_count=required_rank_count,
        quorum_fraction=quorum_fraction,
        min_stable_fraction=min_stable_fraction,
        tie_epsilon=tie_epsilon,
        runner_up_margin=runner_up_margin,
        hydra_overrides=build_ems_apply_hydra_overrides(
            selected_candidate.fqn,
            selected_candidate.hook_position,
        ),
        apex_tuning_input=build_ems_apply_tuning_payload(
            selected_candidate.fqn,
            selected_candidate.hook_position,
        ),
    )


def _relative_margin(consensus_candidates: list[FQNConsensusCandidate]) -> float:
    """Relative gap between the best and second-best consensus scores."""
    if len(consensus_candidates) < 2:
        return math.inf
    best = consensus_candidates[0].consensus_score
    runner_up = consensus_candidates[1].consensus_score
    if best <= 0.0:
        return 0.0
    return (best - runner_up) / best


def _resolve_tie_by_plurality(
    consensus_candidates: list[FQNConsensusCandidate],
    final_results: list[FQNRankerFinalResult],
) -> FQNConsensusCandidate:
    """Break a near-dead-heat by how many ranks ranked each candidate first.

    Falls back to the highest ``hook_position`` — the earliest restore, which
    leaves the widest window for the copy to land before the weights are needed.
    """
    candidates_by_id = {
        (c.fqn, c.hook_position, c.param_fqn): c for c in consensus_candidates
    }
    votes: dict[tuple[str, float, str | None], int] = {}
    for result in final_results:
        for candidate in result.top_candidates:
            if not candidate.eligible:
                continue
            if candidate.candidate_id in candidates_by_id:
                votes[candidate.candidate_id] = votes.get(candidate.candidate_id, 0) + 1
            break
    return max(
        consensus_candidates,
        key=lambda c: (
            votes.get((c.fqn, c.hook_position, c.param_fqn), 0),
            c.hook_position,
            c.consensus_score,
        ),
    )


def _empty_consensus(
    reason: str,
    quorum_fraction: float,
    min_stable_fraction: float,
) -> FQNCrossRankConsensus:
    return _no_consensus(
        reason=reason,
        rank_count=0,
        stable_rank_count=0,
        expected_rank_count=0,
        required_rank_count=0,
        quorum_fraction=quorum_fraction,
        min_stable_fraction=min_stable_fraction,
    )


def _no_consensus(
    reason: str,
    rank_count: int,
    stable_rank_count: int,
    expected_rank_count: int,
    required_rank_count: int,
    quorum_fraction: float,
    min_stable_fraction: float,
) -> FQNCrossRankConsensus:
    return FQNCrossRankConsensus(
        status="no_consensus",
        reason=reason,
        selected_candidate=None,
        rank_count=rank_count,
        stable_rank_count=stable_rank_count,
        expected_rank_count=expected_rank_count,
        required_rank_count=required_rank_count,
        quorum_fraction=quorum_fraction,
        min_stable_fraction=min_stable_fraction,
        tie_epsilon=0.0,
        runner_up_margin=0.0,
        hydra_overrides=(),
        apex_tuning_input={},
    )


def _build_consensus_candidate(
    candidate_id: tuple[str, float, str | None],
    observations: dict[int, FQNFinalCandidate],
    scored_ranks: frozenset[int],
    expected_rank_count: int,
) -> FQNConsensusCandidate:
    fqn, hook_position, param_fqn = candidate_id
    candidates = [observations[rank] for rank in sorted(scored_ranks)]
    return FQNConsensusCandidate(
        fqn=fqn,
        hook_position=hook_position,
        param_fqn=param_fqn,
        rank_count=len(observations),
        scored_rank_count=len(candidates),
        expected_rank_count=expected_rank_count,
        consensus_score=min(candidate.score_p10 for candidate in candidates),
        score_p50=_percentile(
            [candidate.score_p50 for candidate in candidates],
            0.50,
        ),
        safety_margin_us_p10=_percentile(
            [candidate.safety_margin_us_p10 for candidate in candidates],
            0.10,
        ),
        overlap_window_us_p10=_percentile(
            [candidate.overlap_window_us_p10 for candidate in candidates],
            0.10,
        ),
        confidence_p50=_percentile(
            [candidate.confidence_p50 for candidate in candidates],
            0.50,
        ),
    )


def _iter_hook_candidates(
    model: nn.Module,
    hook_positions: tuple[float, ...],
    max_depth: int,
    fqn_allowlist: set[str] | None,
    fqn_blocklist: set[str] | None,
    max_candidates: int | None,
) -> Iterator[_HookCandidate]:
    emitted = 0
    for fqn, module in model.named_modules():
        if (
            not fqn
            or _fqn_depth(fqn) > max_depth
            or _should_skip_fqn(fqn, fqn_allowlist, fqn_blocklist)
        ):
            continue
        named_params = list(module.named_parameters())
        if not named_params:
            continue
        for hook_position in hook_positions:
            candidate = _candidate_for_position(fqn, named_params, hook_position)
            if candidate is None:
                continue
            if max_candidates is not None and emitted >= max_candidates:
                return
            emitted += 1
            yield candidate


def _candidate_for_position(
    fqn: str,
    named_params: list[tuple[str, nn.Parameter]],
    hook_position: float,
) -> _HookCandidate | None:
    num_params = len(named_params)
    target_idx = _position_to_index(hook_position, num_params)
    chosen_idx = next(
        (
            i
            for i in _walk_outward(target_idx, num_params)
            if will_hook_fire(named_params[i][1])
        ),
        None,
    )
    if chosen_idx is None:
        return None
    param_name, param = named_params[chosen_idx]
    return _HookCandidate(
        fqn=fqn,
        hook_position=hook_position,
        param_fqn=f"{fqn}.{param_name}",
        param=param,
        param_index=chosen_idx,
        num_params=num_params,
    )


def _dedupe_hook_candidates(
    candidates: list[_HookCandidate],
) -> list[_HookCandidate]:
    candidates_by_param_fqn: dict[str, _HookCandidate] = {}
    for candidate in candidates:
        current = candidates_by_param_fqn.get(candidate.param_fqn)
        if current is None or _candidate_preference_key(
            candidate
        ) < _candidate_preference_key(current):
            candidates_by_param_fqn[candidate.param_fqn] = candidate
    return sorted(
        candidates_by_param_fqn.values(),
        key=lambda candidate: (candidate.fqn, candidate.hook_position),
    )


def _candidate_preference_key(
    candidate: _HookCandidate,
) -> tuple[int, int, float, str]:
    return (
        _fqn_depth(candidate.fqn),
        candidate.param_index,
        candidate.hook_position,
        candidate.fqn,
    )


def _recommendation_from_evaluation(
    evaluation: FQNCandidateEvaluation,
) -> FQNRecommendation:
    return FQNRecommendation(
        fqn=evaluation.fqn,
        hook_position=evaluation.hook_position,
        score=evaluation.score,
        score_details=evaluation.score_details,
        marker_count=evaluation.marker_count,
        median_fire_time_us=evaluation.median_fire_time_us,
        safety_margin_us=evaluation.safety_margin_us,
        overlap_window_us=evaluation.overlap_window_us,
        confidence=evaluation.confidence,
        param_fqn=evaluation.param_fqn,
        hydra_overrides=build_ems_hydra_overrides(
            evaluation.fqn,
            evaluation.hook_position,
        ),
        apex_tuning_input=build_ems_apex_tuning_payload(
            evaluation.fqn,
            evaluation.hook_position,
        ),
    )


def _missing_evaluation(
    candidate: _HookCandidate,
    peak: _PeakRecord,
    status: str,
) -> FQNCandidateEvaluation:
    return FQNCandidateEvaluation(
        fqn=candidate.fqn,
        hook_position=candidate.hook_position,
        status=status,
        score_details=_zero_score_details(missing_marker_penalty=100.0),
        marker_count=0,
        median_fire_time_us=0.0,
        safe_time_us=peak.time_ns / 1000.0,
        safety_margin_us=0.0,
        overlap_window_us=0.0,
        confidence=0.0,
        score=0.0,
        param_fqn=candidate.param_fqn,
    )


def _rejected_evaluation(
    candidate: _HookCandidate,
    peak: _PeakRecord,
    fire_time_ns: int,
    safety_margin_us: float,
    overlap_window_us: float,
    early_restore_penalty: float,
    late_restore_penalty: float,
) -> FQNCandidateEvaluation:
    return FQNCandidateEvaluation(
        fqn=candidate.fqn,
        hook_position=candidate.hook_position,
        status="rejected",
        score_details=_zero_score_details(
            early_restore_penalty=early_restore_penalty,
            late_restore_penalty=late_restore_penalty,
        ),
        marker_count=1,
        median_fire_time_us=fire_time_ns / 1000.0,
        safe_time_us=peak.time_ns / 1000.0,
        safety_margin_us=safety_margin_us,
        overlap_window_us=overlap_window_us,
        confidence=0.0,
        score=0.0,
        param_fqn=candidate.param_fqn,
    )


def _zero_score_details(
    missing_marker_penalty: float = 0.0,
    early_restore_penalty: float = 0.0,
    late_restore_penalty: float = 0.0,
) -> FQNScoreDetails:
    return FQNScoreDetails(
        peak_timing_score=0.0,
        restore_overlap_score=0.0,
        memory_benefit_score=0.0,
        stability_score=0.0,
        coverage_score=0.0,
        missing_marker_penalty=missing_marker_penalty,
        early_restore_penalty=early_restore_penalty,
        late_restore_penalty=late_restore_penalty,
        median_peak_clearance_us=0.0,
        p10_peak_clearance_us=0.0,
        observed_marker_count=0,
        expected_marker_count=1,
    )


def _should_skip_fqn(
    fqn: str,
    fqn_allowlist: set[str] | None,
    fqn_blocklist: set[str] | None,
) -> bool:
    if fqn_allowlist is not None and fqn not in fqn_allowlist:
        return True
    return fqn_blocklist is not None and fqn in fqn_blocklist


def _fqn_depth(fqn: str) -> int:
    return fqn.count(".") + 1


def _unwrap_model(model: nn.Module) -> nn.Module:
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, nn.Module):
        return wrapped
    return model


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / abs(mean)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        max(math.ceil(percentile * len(sorted_values)) - 1, 0),
        len(sorted_values) - 1,
    )
    return sorted_values[index]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
