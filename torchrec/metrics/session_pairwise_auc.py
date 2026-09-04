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
REQUIRED_INPUTS = "required_inputs"
DEFAULT_SESSION_KEY = "session_id"


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
) -> Dict[str, torch.Tensor]:
    """Compute batch-local pairwise concordance within each session.

    Pair eligibility matches hard-label SessionAwareRankNetLoss: only examples
    from the same session with unequal labels are paired, and (when enabled) a
    pair is removed if either example has zero weight. Each unordered pair is
    evaluated once; prediction ties receive 0.5, as in AUC.
    """
    if predictions.shape != labels.shape or predictions.shape != example_weights.shape:
        raise RecMetricException(
            "predictions, labels, and example_weights must have identical shapes"
        )
    if predictions.dim() != 2:
        raise RecMetricException("metric inputs must have shape [n_tasks, batch_size]")
    if session_ids.dim() != 1 or session_ids.numel() != predictions.shape[1]:
        raise RecMetricException("session_ids must have shape [batch_size]")

    n_tasks = predictions.shape[0]
    correct_pair_weight = torch.zeros(
        n_tasks, dtype=torch.double, device=predictions.device
    )
    total_pair_weight = torch.zeros_like(correct_pair_weight)

    order = torch.argsort(session_ids)
    sorted_session_ids = session_ids[order]
    sorted_predictions = predictions[:, order]
    sorted_labels = labels[:, order]
    sorted_weights = example_weights[:, order]
    _, session_lengths = torch.unique_consecutive(
        sorted_session_ids, return_counts=True
    )

    num_sessions = session_lengths.numel()
    if num_sessions == 0:
        return {
            CORRECT_PAIR_WEIGHT: correct_pair_weight,
            TOTAL_PAIR_WEIGHT: total_pair_weight,
        }

    # Pack the sorted examples into [task, session, max_session_length]. This
    # computes all within-session pairs in one vectorized operation while never
    # constructing the much larger [batch_size, batch_size] matrix.
    max_session_length = int(torch.max(session_lengths))
    session_index = torch.repeat_interleave(
        torch.arange(num_sessions, device=predictions.device), session_lengths
    )
    session_starts = torch.cumsum(session_lengths, dim=0) - session_lengths
    position_in_session = torch.arange(
        predictions.shape[1], device=predictions.device
    ) - torch.repeat_interleave(session_starts, session_lengths)

    packed_shape = (n_tasks, num_sessions, max_session_length)
    packed_predictions = torch.zeros(
        packed_shape, dtype=predictions.dtype, device=predictions.device
    )
    packed_labels = torch.zeros_like(packed_predictions)
    packed_weights = torch.zeros_like(packed_predictions)
    packed_predictions[:, session_index, position_in_session] = sorted_predictions
    packed_labels[:, session_index, position_in_session] = sorted_labels
    packed_weights[:, session_index, position_in_session] = sorted_weights

    valid_example = torch.zeros(
        (num_sessions, max_session_length),
        dtype=torch.bool,
        device=predictions.device,
    )
    valid_example[session_index, position_in_session] = True
    upper_triangle = torch.triu(
        torch.ones(
            (max_session_length, max_session_length),
            dtype=torch.bool,
            device=predictions.device,
        ),
        diagonal=1,
    )
    valid_pair = (
        valid_example.unsqueeze(-1)
        & valid_example.unsqueeze(-2)
        & upper_triangle.unsqueeze(0)
    ).unsqueeze(0)

    label_diff = packed_labels.unsqueeze(-1) - packed_labels.unsqueeze(-2)
    if rank_order_label:
        label_diff = -label_diff
    prediction_diff = packed_predictions.unsqueeze(-1) - packed_predictions.unsqueeze(
        -2
    )
    left_weight = packed_weights.unsqueeze(-1)
    right_weight = packed_weights.unsqueeze(-2)

    valid_pair = valid_pair & (torch.abs(label_diff) >= 1e-6)
    if remove_zero_weight_from_pair:
        valid_pair = valid_pair & ((left_weight * right_weight) > 0)

    pair_weight = (
        left_weight + right_weight if weight_pairs else torch.ones_like(label_diff)
    )
    effective_pair_weight = pair_weight * valid_pair
    concordance = (prediction_diff * label_diff > 0).to(pair_weight.dtype)
    concordance = concordance + 0.5 * (prediction_diff == 0).to(pair_weight.dtype)

    correct_pair_weight += torch.sum(
        concordance * effective_pair_weight,
        dim=(1, 2, 3),
        dtype=torch.double,
    )
    total_pair_weight += torch.sum(
        effective_pair_weight,
        dim=(1, 2, 3),
        dtype=torch.double,
    )

    return {
        CORRECT_PAIR_WEIGHT: correct_pair_weight,
        TOTAL_PAIR_WEIGHT: total_pair_weight,
    }


def _compute_pairwise_auc(
    *, correct_pair_weight: torch.Tensor, total_pair_weight: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        total_pair_weight > 0,
        correct_pair_weight / total_pair_weight,
        torch.full_like(total_pair_weight, 0.5),
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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._session_key = session_key
        self._score_key = score_key
        self._pairwise_weight_key = pairwise_weight_key
        self._rank_order_label = rank_order_label
        self._remove_zero_weight_from_pair = remove_zero_weight_from_pair
        self._weight_pairs = weight_pairs
        for state_name in (CORRECT_PAIR_WEIGHT, TOTAL_PAIR_WEIGHT):
            self._add_state(
                state_name,
                torch.zeros(self._n_tasks, dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx="sum",
                persistent=True,
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
        )
        for state_name, state_value in states.items():
            state = getattr(self, state_name).to(labels.device)
            state += state_value
            self._aggregate_window_state(state_name, state_value, batch_size)

    def _compute(self) -> List[MetricComputationReport]:
        return [
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


class SessionPairwiseAUCMetric(RecMetric):
    # pyrefly: ignore[bad-override]
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
