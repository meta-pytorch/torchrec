#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, cast, Dict, List, Optional, Type

import torch
from torch.autograd.profiler import record_function
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)

logger: logging.Logger = logging.getLogger(__name__)


def _segment_cumsum(values: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Cumulative sum within each segment, resetting at segment boundaries.

    Args:
        values: [N] flat tensor of values.
        segment_ids: [N] integer segment IDs (must be sorted/grouped).

    Returns:
        [N] cumulative sum within each segment.
    """
    cumsum = torch.cumsum(values, dim=0)
    boundaries = torch.cat(
        [
            torch.tensor([True], device=values.device),
            segment_ids[1:] != segment_ids[:-1],
        ]
    )
    boundary_indices = torch.where(boundaries)[0]

    # Offset to subtract: cumsum just before each segment's first element
    segment_offsets = cumsum[boundary_indices] - values[boundary_indices]

    # Map each element to its segment index, then look up the offset
    segment_labels = torch.zeros_like(values, dtype=torch.long)
    segment_labels[boundary_indices] = 1
    segment_labels = torch.cumsum(segment_labels, dim=0) - 1
    per_element_offset = segment_offsets[segment_labels]

    return cumsum - per_element_offset


@torch.fx.wrap
def compute_uauc(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute per-user AUC and aggregate, fully vectorized.

    Uses a sort-and-segment approach: creates composite (task, user) group keys,
    sorts globally by (group_key, prediction), then uses segment cumsum and
    scatter-add to compute the AUC per group.

    Args:
        predictions: [n_task, n_sample] model predictions.
        labels: [n_task, n_sample] binary labels.
        weights: [n_task, n_sample] sample weights.
        grouping_keys: [n_sample] user/viewer IDs.

    Returns:
        Dict with keys ``auc_sum``, ``weighted_auc_sum``, ``num_users``,
        ``total_weight`` (all shaped [n_task]).
    """
    n_task, n_sample = predictions.size()
    device = predictions.device

    with record_function("## uauc_vectorized ##"):
        # Map grouping_keys to contiguous 0..n_users-1
        _, user_indices = torch.unique(grouping_keys, return_inverse=True)
        n_users = user_indices.max().item() + 1

        # --- Per-(task, user) aggregations using scatter_add on the UNSORTED data ---
        # Composite group key: task_idx * n_users + user_idx
        task_idx = torch.arange(n_task, device=device)[:, None].expand_as(predictions)
        group_keys = (task_idx * n_users + user_indices[None, :]).long()
        n_groups = n_task * n_users

        flat_labels = labels.reshape(-1).double()
        flat_weights = weights.reshape(-1).double()
        flat_group_keys = group_keys.reshape(-1)

        # Per-group positive/negative weights and sample counts
        w_pos = torch.zeros(n_groups, dtype=torch.double, device=device)
        w_neg = torch.zeros(n_groups, dtype=torch.double, device=device)
        group_counts = torch.zeros(n_groups, dtype=torch.long, device=device)

        w_pos.scatter_add_(0, flat_group_keys, flat_weights * flat_labels)
        w_neg.scatter_add_(0, flat_group_keys, flat_weights * (1 - flat_labels))
        group_counts.scatter_add_(0, flat_group_keys, torch.ones_like(flat_group_keys))

        # --- Identical predictions check ---
        # For each (task, user) group, check if max_pred == min_pred
        # (ignoring zeros, matching original behavior)
        flat_preds = predictions.reshape(-1).double()
        # Replace zeros with NaN for min/max so they're ignored
        preds_for_minmax = flat_preds.clone()
        preds_for_minmax[preds_for_minmax == 0] = float("nan")
        big_val = torch.tensor(float("inf"), dtype=torch.double, device=device)
        small_val = torch.tensor(float("-inf"), dtype=torch.double, device=device)

        group_max_pred = small_val.expand(n_groups).clone()
        group_min_pred = big_val.expand(n_groups).clone()
        group_max_pred.scatter_reduce_(
            0, flat_group_keys, preds_for_minmax, reduce="amax", include_self=False
        )
        group_min_pred.scatter_reduce_(
            0, flat_group_keys, preds_for_minmax, reduce="amin", include_self=False
        )
        # Identical if max == min (or all were NaN/zero)
        identical_preds = (group_max_pred <= group_min_pred) | (
            group_max_pred == small_val
        )

        # --- Validity mask per group ---
        valid_mask = (
            (w_pos > 0) & (w_neg > 0) & (group_counts >= 2) & (~identical_preds)
        )

        # Early exit: no valid groups
        if not valid_mask.any():
            z = torch.zeros(n_task, dtype=torch.double, device=device)
            return {
                "auc_sum": z,
                "weighted_auc_sum": z.clone(),
                "num_users": z.clone(),
                "total_weight": z.clone(),
            }

        # --- Sort by (group_key, prediction) for segment cumsum ---
        flat_preds_raw = predictions.reshape(-1).double()

        # Stable sort by group_key first, then by prediction within group
        # Use a combined sort key: group_key (primary), prediction (secondary)
        # To avoid float precision issues, sort by group_key then sub-sort
        sort_order = torch.argsort(
            flat_group_keys.double() * 1e12 + flat_preds_raw, stable=True
        )

        sorted_group_keys = flat_group_keys[sort_order]
        sorted_labels = flat_labels[sort_order]
        sorted_weights = flat_weights[sort_order]

        pos_mask = sorted_labels
        neg_mask = 1 - sorted_labels

        # Segment cumsum of negative weights within each group
        neg_weight_vals = sorted_weights * neg_mask
        seg_cum_neg = _segment_cumsum(neg_weight_vals, sorted_group_keys)

        # AUC numerator contribution per element: pos * weight * cum_neg_weight
        contrib = pos_mask * sorted_weights * seg_cum_neg

        # Aggregate numerator per group
        numerator = torch.zeros(n_groups, dtype=torch.double, device=device)
        numerator.scatter_add_(0, sorted_group_keys, contrib)

        # --- Compute per-group AUC ---
        denominator = w_pos * w_neg
        auc_per_group = numerator / (denominator + 1e-10)

        # Zero out invalid groups
        auc_per_group = auc_per_group * valid_mask.double()

        # --- Aggregate per task ---
        # Reshape to [n_task, n_users]
        auc_per_group = auc_per_group.view(n_task, n_users)
        valid_mask_2d = valid_mask.view(n_task, n_users).double()
        user_weight_2d = (w_pos + w_neg).view(n_task, n_users)

        auc_sum = auc_per_group.sum(dim=1)  # [n_task]
        weighted_auc_sum = (auc_per_group * user_weight_2d).sum(dim=1)
        num_valid_users = valid_mask_2d.sum(dim=1)
        total_weight = (user_weight_2d * valid_mask_2d).sum(dim=1)

    return {
        "auc_sum": auc_sum,
        "weighted_auc_sum": weighted_auc_sum,
        "num_users": num_valid_users,
        "total_weight": total_weight,
    }


@torch.fx.wrap
def compute_window_uauc(
    auc_sum: torch.Tensor,
    weighted_auc_sum: torch.Tensor,
    num_users: torch.Tensor,
    total_weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return {
        "uauc": auc_sum / (num_users + 1e-10),
        "wuauc": weighted_auc_sum / (total_weight + 1e-10),
        "num_users": num_users,
    }


class UAUCMetricComputation(RecMetricComputation):
    r"""RecMetricComputation for uAUC (User-level AUC).

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._add_state(
            "auc_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "weighted_auc_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "num_users",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "total_weight",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    # pyrefly: ignore[bad-override]
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        grouping_keys: torch.Tensor,
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for UAUCMetricComputation update"
            )

        states = compute_uauc(predictions, labels, weights, grouping_keys)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        result = compute_window_uauc(
            cast(torch.Tensor, self.auc_sum),
            cast(torch.Tensor, self.weighted_auc_sum),
            cast(torch.Tensor, self.num_users),
            cast(torch.Tensor, self.total_weight),
        )
        window_result = compute_window_uauc(
            self.get_window_state("auc_sum"),
            self.get_window_state("weighted_auc_sum"),
            self.get_window_state("num_users"),
            self.get_window_state("total_weight"),
        )

        return [
            MetricComputationReport(
                name=MetricName.UAUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=result["uauc"],
            ),
            MetricComputationReport(
                name=MetricName.UAUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_result["uauc"],
            ),
            MetricComputationReport(
                name=MetricName.WUAUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=result["wuauc"],
            ),
            MetricComputationReport(
                name=MetricName.WUAUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_result["wuauc"],
            ),
            MetricComputationReport(
                name=MetricName.UAUC_NUM_USERS,
                metric_prefix=MetricPrefix.LIFETIME,
                value=result["num_users"],
            ),
            MetricComputationReport(
                name=MetricName.UAUC_NUM_USERS,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_result["num_users"],
            ),
        ]


class UAUCMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.UAUC
    _computation_class: Type[RecMetricComputation] = UAUCMetricComputation
