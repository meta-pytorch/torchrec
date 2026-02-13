#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from functools import partial
from typing import Any, cast, Dict, List, Optional, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


logger: logging.Logger = logging.getLogger(__name__)

PREDICTIONS = "predictions"
LABELS = "labels"
WEIGHTS = "weights"
GROUPING_KEYS = "grouping_keys"
REQUIRED_INPUTS = "required_inputs"


def _riemann_integral(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Riemann integral approximates the area of each cell with a rectangle positioned at the egde.
    It is conventionally used rather than trapezoid approximation, which uses a rectangle positioned in the
    center"""
    return -torch.sum((x[1:] - x[:-1]) * y[:-1])


def _compute_auprc_helper(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    sorted_indices = torch.argsort(predictions, descending=True, dim=-1)

    threshold = torch.index_select(predictions, dim=0, index=sorted_indices)

    sorted_labels = torch.index_select(labels, dim=0, index=sorted_indices)

    sorted_weights = torch.index_select(weights, dim=0, index=sorted_indices)

    mask = F.pad(threshold.diff(dim=0) != 0, [0, 1], value=1.0)
    num_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)[mask]
    num_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)[mask]

    precision = (num_tp / (num_tp + num_fp)).flip(0)
    recall = (num_tp / num_tp[-1]).flip(0)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)])
    recall = torch.cat([recall, recall.new_zeros(1)])

    # nan happens with 0.0 / 0.0. For recall's case, this could happen from its right side:
    # num_fp is a cumsum and thus 0.0 starts from its left side. But given recall has a flip,
    # then those 0.0 goes to right side and thus nan.
    # If recalls are NaNs, set NaNs to 0.0s, as append a 0.0 on its right side above.
    recall = torch.nan_to_num(recall, 0.0)

    # similar as recall, precision's nan would happen from its right side.
    # since we append 1.0 on its right side above, we replace nan by 1.0.
    # If any element in precision is Nan, _riemann_integral will return NaN.
    precision = torch.nan_to_num(precision, 1.0)

    auprc = _riemann_integral(recall, precision)
    return auprc


def compute_auprc(
    n_tasks: int,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Computes AUPRC (Area Under the Curve) for binary classification.

    Args:
        n_tasks (int): number of tasks.
        predictions (torch.Tensor): tensor of size (n_tasks, n_examples).
        labels (torch.Tensor): tensor of size (n_tasks, n_examples).
        weights (torch.Tensor): tensor of size (n_tasks, n_examples).
    """
    auprcs = []
    for predictions_i, labels_i, weights_i in zip(predictions, labels, weights):
        auprc = _compute_auprc_helper(predictions_i, labels_i, weights_i)
        auprcs.append(auprc.view(1))
    return torch.cat(auprcs)


def compute_auprc_per_group(
    n_tasks: int,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Computes AUPRC (Area Under the Curve) for binary classification for groups of predictions/labels.
    Args:
        n_tasks (int): number of tasks
        predictions (torch.Tensor): tensor of size (n_tasks, n_examples)
        labels (torch.Tensor): tensor of size (n_tasks, n_examples)
        weights (torch.Tensor): tensor of size (n_tasks, n_examples)
        grouping_keys (torch.Tensor): tensor of size (n_examples,)

    Returns:
        torch.Tensor: tensor of size (n_tasks,), average of AUPRCs per group.
    """
    auprcs = []
    if grouping_keys.numel() != 0 and grouping_keys[0] == -1:
        # we added padding  as the first elements during init to avoid floating point exception in sync()
        # removing the paddings to avoid numerical errors.
        grouping_keys = grouping_keys[1:]
        predictions = predictions[:, 1:]
        labels = labels[:, 1:]
        weights = weights[:, 1:]

    # get unique group indices
    group_indices = torch.unique(grouping_keys)

    for predictions_i, labels_i, weights_i in zip(predictions, labels, weights):
        # Loop over each group
        auprc_groups_sum = torch.tensor([0], dtype=torch.float32)
        for group_idx in group_indices:
            # get predictions, labels, and weights for this group
            group_mask = grouping_keys == group_idx
            grouped_predictions = predictions_i[group_mask]
            grouped_labels = labels_i[group_mask]
            grouped_weights = weights_i[group_mask]

            auprc = _compute_auprc_helper(
                grouped_predictions, grouped_labels, grouped_weights
            )
            auprc_groups_sum = auprc_groups_sum.to(auprc.device)
            auprc_groups_sum += auprc.view(1)
        avg_auprc = (
            auprc_groups_sum / len(group_indices)
            if len(group_indices) > 0
            else torch.tensor([0.5], dtype=torch.float32)
        )
        auprcs.append(avg_auprc)
    return torch.cat(auprcs)


def _state_reduction(state: List[torch.Tensor], dim: int = 1) -> List[torch.Tensor]:
    return [torch.cat(state, dim=dim)]


_grouping_keys_state_reduction = partial(_state_reduction, dim=0)


LIFETIME_WEIGHTED_AUPRC = "lifetime_weighted_auprc"
LIFETIME_WEIGHT = "lifetime_weight"


class AUPRCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for AUPRC, i.e. Area Under the Curve.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    Args:
        grouped_auprc (bool): If True, computes AUPRC per group and returns average AUPRC across all groups.
            The `grouping_keys` is provided during state updates along with predictions, labels, weights.
            This feature is currently not enabled for `fused_update_limit`.
    """

    def __init__(
        self,
        *args: Any,
        grouped_auprc: bool = False,
        fused_update_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grouped_auprc and fused_update_limit > 0:
            raise RecMetricException(
                "Grouped AUPRC and Fused Update Limit cannot be enabled together yet."
            )

        self._grouped_auprc: bool = grouped_auprc
        self._add_state(
            PREDICTIONS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        self._add_state(
            LABELS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        self._add_state(
            WEIGHTS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        if self._grouped_auprc:
            self._add_state(
                GROUPING_KEYS,
                [],
                add_window_state=False,
                dist_reduce_fx=_grouping_keys_state_reduction,
                persistent=False,
            )
        # Lifetime accumulators for weighted averaging of window AUPRC values
        self._lifetime_weighted_auprc: torch.Tensor = torch.tensor(
            0.0, dtype=torch.float64
        )
        self._lifetime_weight: torch.Tensor = torch.tensor(0.0, dtype=torch.float64)
        self._init_states()

    # The states values are set to empty lists in __init__() and reset(), and then we
    # add a size (self._n_tasks, 1) tensor to each of the list as the initial values
    # This is to bypass the limitation of state aggregation in TorchMetrics sync() when
    # we try to checkpoint the states before update()
    # The reason for using lists here is to avoid automatically stacking the tensors from
    # all the trainers into one tensor in sync()
    # The reason for using non-empty tensors as the first elements is to avoid the
    # floating point exception thrown in sync() for aggregating empty tensors
    def _init_states(self) -> None:
        if len(getattr(self, PREDICTIONS)) > 0:
            return

        getattr(self, PREDICTIONS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        getattr(self, LABELS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        getattr(self, WEIGHTS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        if self._grouped_auprc:
            getattr(self, GROUPING_KEYS).append(torch.tensor([-1], device=self.device))

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Args:
            predictions (torch.Tensor): tensor of size (n_task, n_examples)
            labels (torch.Tensor): tensor of size (n_task, n_examples)
            weights (torch.Tensor): tensor of size (n_task, n_examples)
            grouping_key (torch.Tensor): Optional tensor of size (1, n_examples) that specifies the groups of
                    predictions/labels per batch. If provided, the PR AUC metric also
                    computes PR AUC per group and returns the average PR AUC across all groups.
        """
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for AUPRCMetricComputation update"
            )
        predictions = predictions.float()
        labels = labels.float()
        weights = weights.float()
        num_samples = getattr(self, PREDICTIONS)[0].size(-1)
        batch_size = predictions.size(-1)
        start_index = max(num_samples + batch_size - self._window_size, 0)
        # Using `self.predictions =` will cause Pyre errors.
        getattr(self, PREDICTIONS)[0] = torch.cat(
            [
                cast(torch.Tensor, getattr(self, PREDICTIONS)[0])[:, start_index:],
                predictions,
            ],
            dim=-1,
        )
        getattr(self, LABELS)[0] = torch.cat(
            [cast(torch.Tensor, getattr(self, LABELS)[0])[:, start_index:], labels],
            dim=-1,
        )
        getattr(self, WEIGHTS)[0] = torch.cat(
            [cast(torch.Tensor, getattr(self, WEIGHTS)[0])[:, start_index:], weights],
            dim=-1,
        )
        if self._grouped_auprc:
            if REQUIRED_INPUTS not in kwargs or (
                (grouping_keys := kwargs[REQUIRED_INPUTS].get(GROUPING_KEYS)) is None
            ):
                raise RecMetricException(
                    f"Input '{GROUPING_KEYS}' are required for AUPRCMetricComputation grouped update"
                )
            getattr(self, GROUPING_KEYS)[0] = torch.cat(
                [
                    cast(torch.Tensor, getattr(self, GROUPING_KEYS)[0])[start_index:],
                    grouping_keys.squeeze(),
                ],
                dim=0,
            )

    def _compute(self) -> List[MetricComputationReport]:
        # Compute window AUPRC
        window_auprc = compute_auprc(
            self._n_tasks,
            cast(torch.Tensor, getattr(self, PREDICTIONS)[0]),
            cast(torch.Tensor, getattr(self, LABELS)[0]),
            cast(torch.Tensor, getattr(self, WEIGHTS)[0]),
        )

        # Get total weight for this window
        window_weight = cast(torch.Tensor, getattr(self, WEIGHTS)[0]).sum()

        # Accumulate lifetime stats using weighted averaging
        # Move tensors to same device if needed
        device = window_auprc.device
        self._lifetime_weighted_auprc = self._lifetime_weighted_auprc.to(device)
        self._lifetime_weight = self._lifetime_weight.to(device)

        # For multi-task, compute weighted average across tasks
        weighted_auprc_sum = (window_auprc * window_weight).sum()
        self._lifetime_weighted_auprc = (
            self._lifetime_weighted_auprc + weighted_auprc_sum
        )
        self._lifetime_weight = self._lifetime_weight + window_weight * self._n_tasks

        # Compute lifetime AUPRC
        lifetime_auprc = torch.where(
            self._lifetime_weight > 0,
            self._lifetime_weighted_auprc / self._lifetime_weight,
            torch.tensor(0.5, device=device, dtype=torch.float64),
        )
        # Expand to match n_tasks shape
        lifetime_auprc_per_task = lifetime_auprc.expand(self._n_tasks)

        reports = [
            MetricComputationReport(
                name=MetricName.AUPRC,
                metric_prefix=MetricPrefix.WINDOW,
                value=window_auprc,
            ),
            MetricComputationReport(
                name=MetricName.AUPRC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=lifetime_auprc_per_task,
            ),
        ]
        if self._grouped_auprc:
            reports.append(
                MetricComputationReport(
                    name=MetricName.GROUPED_AUPRC,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_auprc_per_group(
                        self._n_tasks,
                        cast(torch.Tensor, getattr(self, PREDICTIONS)[0]),
                        cast(torch.Tensor, getattr(self, LABELS)[0]),
                        cast(torch.Tensor, getattr(self, WEIGHTS)[0]),
                        cast(torch.Tensor, getattr(self, GROUPING_KEYS)[0]),
                    ),
                )
            )
        return reports

    def reset(self) -> None:
        super().reset()
        self._init_states()


class AUPRCMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.AUPRC
    _computation_class: Type[RecMetricComputation] = AUPRCMetricComputation

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
        if kwargs.get("grouped_auprc"):
            self._required_inputs.add(GROUPING_KEYS)
        if self._compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION:
            logging.warning(
                f"compute_mode FUSED_TASKS_AND_STATES_COMPUTATION can't support {self._namespace} yet "
                "because its states are not 1D Tensors. Only FUSED_TASKS_COMPUTATION will take effect."
            )
