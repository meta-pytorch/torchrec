#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torch import distributed as dist
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


CORRECT_PAIR_WEIGHT = "correct_pair_weight"
TOTAL_PAIR_WEIGHT = "total_pair_weight"
VALID_PAIR_COUNT = "valid_pair_count"
EFFECTIVE_EXAMPLE_COUNT = "effective_example_count"
BATCH_COUNT = "batch_count"
REQUIRED_INPUTS = "required_inputs"
DEFAULT_SESSION_KEY = "session_id"
DEFAULT_PAIR_CHUNK_SIZE = 256


def _as_task_matrix(
    value: torch.Tensor,
    *,
    n_tasks: int,
    batch_size: int,
    input_name: str,
) -> torch.Tensor:
    """Normalize a per-example or per-task tensor to [n_tasks, batch_size]."""
    if value.numel() == batch_size:
        return value.reshape(1, batch_size).expand(n_tasks, -1)
    if value.numel() == n_tasks * batch_size:
        return value.reshape(n_tasks, batch_size)
    raise RecMetricException(
        f"Input '{input_name}' has {value.numel()} elements; expected "
        f"{batch_size} or {n_tasks * batch_size}."
    )


def _get_session_pairwise_auc_states(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
    example_weights: torch.Tensor,
    rank_order_label: bool = False,
    remove_zero_weight_from_pair: bool = True,
    weight_pairs: bool = True,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
    report_batch_coverage: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute batch-local pairwise concordance in bounded-memory chunks.

    Pair eligibility matches hard-label SessionAwareRankNetLoss: only examples
    from the same session with unequal labels are paired, and (when enabled) a
    pair is removed if either example has zero weight. Each unordered pair is
    evaluated once; prediction ties receive 0.5, as in AUC.

    The implementation compares a fixed-size block of left examples with the
    full batch, masks cross-session and lower-triangle entries, then immediately
    reduces that block. It is exact and does not require padding sessions or
    reading a data-dependent CUDA scalar on the host. Runtime is O(tasks * B^2)
    and temporary memory is O(tasks * pair_chunk_size * B).

    When ``report_batch_coverage`` is enabled, ``valid_pair_count`` is the raw,
    unweighted number of eligible unordered pairs in this batch::

        sum(1[same_session(i, j)] * 1[i < j] *
            1[abs(label_i - label_j) >= 1e-6] *
            1[weight_i * weight_j > 0])

    The prediction values and the magnitude of the pair weight do not affect
    this count. Coverage states are not computed or returned when the option is
    disabled.
    """
    if predictions.shape != labels.shape or predictions.shape != example_weights.shape:
        raise RecMetricException(
            "predictions, labels, and example_weights must have identical shapes"
        )
    if predictions.dim() != 2:
        raise RecMetricException("metric inputs must have shape [n_tasks, batch_size]")
    if session_ids.dim() != 1 or session_ids.numel() != predictions.shape[1]:
        raise RecMetricException("session_ids must have shape [batch_size]")
    if pair_chunk_size <= 0:
        raise RecMetricException("pair_chunk_size must be positive")

    n_tasks = predictions.shape[0]
    batch_size = predictions.shape[1]
    correct_pair_weight = torch.zeros(
        n_tasks, dtype=torch.double, device=predictions.device
    )
    total_pair_weight = torch.zeros_like(correct_pair_weight)
    coverage_states: Dict[str, torch.Tensor] = {}
    effective_example_mask: Optional[torch.Tensor] = None
    if report_batch_coverage:
        coverage_states = {
            VALID_PAIR_COUNT: torch.zeros_like(correct_pair_weight),
            EFFECTIVE_EXAMPLE_COUNT: torch.zeros_like(correct_pair_weight),
        }
        effective_example_mask = torch.zeros_like(example_weights, dtype=torch.bool)

    right_indices = torch.arange(batch_size, device=predictions.device)
    right_sessions = session_ids.unsqueeze(0)
    right_labels = labels.unsqueeze(1)
    right_predictions = predictions.unsqueeze(1)
    right_weights = example_weights.unsqueeze(1)

    for left_start in range(0, batch_size, pair_chunk_size):
        left_stop = min(left_start + pair_chunk_size, batch_size)
        left_indices = right_indices[left_start:left_stop]
        same_session = session_ids[left_start:left_stop].unsqueeze(-1) == right_sessions
        upper_triangle = left_indices.unsqueeze(-1) < right_indices.unsqueeze(0)
        valid_pair = (same_session & upper_triangle).unsqueeze(0)

        label_diff = labels[:, left_start:left_stop].unsqueeze(-1) - right_labels
        if rank_order_label:
            label_diff = -label_diff
        prediction_diff = (
            predictions[:, left_start:left_stop].unsqueeze(-1) - right_predictions
        )
        left_weight = example_weights[:, left_start:left_stop].unsqueeze(-1)

        valid_pair = valid_pair & (torch.abs(label_diff) >= 1e-6)
        if remove_zero_weight_from_pair:
            valid_pair = valid_pair & ((left_weight * right_weights) > 0)

        if report_batch_coverage:
            assert effective_example_mask is not None
            coverage_states[VALID_PAIR_COUNT] += torch.sum(
                valid_pair, dim=(1, 2), dtype=torch.double
            )
            effective_example_mask[:, left_start:left_stop] |= valid_pair.any(dim=-1)
            effective_example_mask |= valid_pair.any(dim=-2)

        pair_weight = (
            left_weight + right_weights if weight_pairs else torch.ones_like(label_diff)
        )
        effective_pair_weight = pair_weight * valid_pair
        concordance = (prediction_diff * label_diff > 0).to(pair_weight.dtype)
        concordance = concordance + 0.5 * (prediction_diff == 0).to(pair_weight.dtype)

        correct_pair_weight += torch.sum(
            concordance * effective_pair_weight,
            dim=(1, 2),
            dtype=torch.double,
        )
        total_pair_weight += torch.sum(
            effective_pair_weight,
            dim=(1, 2),
            dtype=torch.double,
        )

    if report_batch_coverage:
        assert effective_example_mask is not None
        coverage_states[EFFECTIVE_EXAMPLE_COUNT] += torch.sum(
            effective_example_mask, dim=1, dtype=torch.double
        )

    return {
        CORRECT_PAIR_WEIGHT: correct_pair_weight,
        TOTAL_PAIR_WEIGHT: total_pair_weight,
        **coverage_states,
    }


def _compute_pairwise_auc(
    *, correct_pair_weight: torch.Tensor, total_pair_weight: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        total_pair_weight > 0,
        correct_pair_weight / total_pair_weight,
        torch.full_like(total_pair_weight, 0.5),
    )


def _compute_average_per_batch(
    *, value_sum: torch.Tensor, batch_count: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        batch_count > 0,
        value_sum / batch_count,
        torch.zeros_like(value_sum),
    )


class SessionPairwiseAUCMetricComputation(RecMetricComputation):
    def __init__(
        self,
        *args: Any,
        session_key: str = DEFAULT_SESSION_KEY,
        score_key: Optional[str] = None,
        pairwise_weight_key: Optional[str] = None,
        rank_order_label: bool = False,
        remove_zero_weight_from_pair: bool = True,
        weight_pairs: bool = True,
        report_batch_coverage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._session_key = session_key
        self._score_key = score_key
        self._pairwise_weight_key = pairwise_weight_key
        self._rank_order_label = rank_order_label
        self._remove_zero_weight_from_pair = remove_zero_weight_from_pair
        self._weight_pairs = weight_pairs
        self._report_batch_coverage = report_batch_coverage
        state_names = [CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT]
        if self._report_batch_coverage:
            state_names.extend([VALID_PAIR_COUNT, EFFECTIVE_EXAMPLE_COUNT, BATCH_COUNT])
        for state_name in state_names:
            self._add_state(
                state_name,
                torch.zeros(self._n_tasks, dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx="sum",
                # Pairwise AUC is a diagnostic metric, like AUC/RAUC. Keeping
                # its accumulation buffers out of state_dict lets an older
                # metric checkpoint load strictly after this metric is added.
                persistent=False,
            )

    @torch.compiler.disable
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        required_inputs = kwargs.get(REQUIRED_INPUTS, {})
        session_ids = required_inputs.get(self._session_key)
        if session_ids is None:
            raise RecMetricException(
                f"Input '{self._session_key}' is required for SessionPairwiseAUC"
            )

        scores = (
            required_inputs.get(self._score_key)
            if self._score_key is not None
            else predictions
        )
        pairwise_weights = (
            required_inputs.get(self._pairwise_weight_key)
            if self._pairwise_weight_key is not None
            else weights
        )
        if scores is None or pairwise_weights is None:
            raise RecMetricException(
                "SessionPairwiseAUC requires scores and pairwise example weights"
            )

        batch_size = labels.shape[-1]
        labels = labels.reshape(self._n_tasks, batch_size).float()
        scores = _as_task_matrix(
            scores,
            n_tasks=self._n_tasks,
            batch_size=batch_size,
            input_name=self._score_key or "predictions",
        ).float()
        pairwise_weights = _as_task_matrix(
            pairwise_weights,
            n_tasks=self._n_tasks,
            batch_size=batch_size,
            input_name=self._pairwise_weight_key or "weights",
        ).float()
        session_ids = session_ids.reshape(-1)

        states = _get_session_pairwise_auc_states(
            predictions=scores,
            labels=labels,
            session_ids=session_ids,
            example_weights=pairwise_weights,
            rank_order_label=self._rank_order_label,
            remove_zero_weight_from_pair=self._remove_zero_weight_from_pair,
            weight_pairs=self._weight_pairs,
            report_batch_coverage=self._report_batch_coverage,
        )
        if self._report_batch_coverage:
            states[BATCH_COUNT] = torch.ones(
                self._n_tasks, dtype=torch.double, device=labels.device
            )
        for state_name, state_value in states.items():
            state = getattr(self, state_name).to(labels.device)
            state += state_value
            self._aggregate_window_state(state_name, state_value, batch_size)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.SESSION_PAIRWISE_AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=_compute_pairwise_auc(
                    correct_pair_weight=cast(
                        torch.Tensor, getattr(self, CORRECT_PAIR_WEIGHT)
                    ),
                    total_pair_weight=cast(
                        torch.Tensor, getattr(self, TOTAL_PAIR_WEIGHT)
                    ),
                ),
            ),
            MetricComputationReport(
                name=MetricName.SESSION_PAIRWISE_AUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=_compute_pairwise_auc(
                    correct_pair_weight=self.get_window_state(CORRECT_PAIR_WEIGHT),
                    total_pair_weight=self.get_window_state(TOTAL_PAIR_WEIGHT),
                ),
            ),
        ]
        if not self._report_batch_coverage:
            return reports

        reports.extend(
            [
                # These are averages per metric update (and therefore per
                # rank-local batch after distributed state reduction), not a
                # count summed across the global training step.
                MetricComputationReport(
                    name=MetricName.VALID_PAIRS_PER_BATCH,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=_compute_average_per_batch(
                        value_sum=cast(torch.Tensor, getattr(self, VALID_PAIR_COUNT)),
                        batch_count=cast(torch.Tensor, getattr(self, BATCH_COUNT)),
                    ),
                ),
                MetricComputationReport(
                    name=MetricName.VALID_PAIRS_PER_BATCH,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=_compute_average_per_batch(
                        value_sum=self.get_window_state(VALID_PAIR_COUNT),
                        batch_count=self.get_window_state(BATCH_COUNT),
                    ),
                ),
                MetricComputationReport(
                    name=MetricName.EFFECTIVE_TRAINING_EXAMPLES_PER_BATCH,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=_compute_average_per_batch(
                        value_sum=cast(
                            torch.Tensor, getattr(self, EFFECTIVE_EXAMPLE_COUNT)
                        ),
                        batch_count=cast(torch.Tensor, getattr(self, BATCH_COUNT)),
                    ),
                ),
                MetricComputationReport(
                    name=MetricName.EFFECTIVE_TRAINING_EXAMPLES_PER_BATCH,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=_compute_average_per_batch(
                        value_sum=self.get_window_state(EFFECTIVE_EXAMPLE_COUNT),
                        batch_count=self.get_window_state(BATCH_COUNT),
                    ),
                ),
            ]
        )
        return reports


class SessionPairwiseAUCMetric(RecMetric):
    # Pair construction and per-batch coverage depend on logical batch
    # boundaries. ZORM currently merges those boundaries before metric update.
    supports_cpu_offloaded_metric_module: bool = False

    # pyre-ignore[15]: The base metric namespace is intentionally specialized.
    _namespace: MetricNamespace = MetricNamespace.SESSION_PAIRWISE_AUC
    _computation_class: Type[RecMetricComputation] = SessionPairwiseAUCMetricComputation

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        if fused_update_limit > 0:
            raise RecMetricException(
                "Fused update is not supported for session pairwise AUC"
            )
        super().__init__(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            process_group=process_group,
            **kwargs,
        )
        self._required_inputs.add(
            cast(str, kwargs.get("session_key", DEFAULT_SESSION_KEY))
        )
        if (score_key := kwargs.get("score_key")) is not None:
            self._required_inputs.add(cast(str, score_key))
        if (pairwise_weight_key := kwargs.get("pairwise_weight_key")) is not None:
            self._required_inputs.add(cast(str, pairwise_weight_key))
