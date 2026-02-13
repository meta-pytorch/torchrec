#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from functools import partial
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torchmetrics.utilities.distributed import gather_all_tensors
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


def _concat_if_needed(
    predictions: List[torch.Tensor],
    labels: List[torch.Tensor],
    weights: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This check exists because of how the state is organized due to quirks in RecMetrics.
    Since we do not do tensor concatenatation in the compute or update call, there are cases (in non-distributed settings)
    where the tensors from updates are not concatted into a single tensor. Which is determined by the length of the list.
    """
    preds_t, labels_t, weights_t = None, None, None
    if len(predictions) > 1:
        preds_t = torch.cat(predictions, dim=-1)
        labels_t = torch.cat(labels, dim=-1)
        weights_t = torch.cat(weights, dim=-1)
    else:
        preds_t = predictions[0]
        labels_t = labels[0]
        weights_t = weights[0]

    return preds_t, labels_t, weights_t


def _compute_auc_helper(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    apply_bin: bool = False,
) -> torch.Tensor:
    sorted_indices = torch.argsort(predictions, descending=True, dim=-1)
    sorted_labels = torch.index_select(labels, dim=0, index=sorted_indices)
    if apply_bin:
        # TODO - [add flag to set bining dyamically] for use with soft labels, >=0.039 --> 1, <0.039 --> 0
        sorted_labels = torch.ge(sorted_labels, 0.039).to(dtype=sorted_labels.dtype)
    sorted_weights = torch.index_select(weights, dim=0, index=sorted_indices)
    if sorted_weights.numel() > 0:
        cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
        cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
        auc = torch.where(
            cum_fp[-1] * cum_tp[-1] == 0,
            0.5,  # 0.5 is the no-signal default value for auc.
            torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
        )
    else:
        # if empty predictions, default value
        auc = torch.tensor(0.5, device=sorted_weights.device)
    return auc


def compute_auc(
    n_tasks: int,
    predictions: List[torch.Tensor],
    labels: List[torch.Tensor],
    weights: List[torch.Tensor],
    apply_bin: bool = False,
) -> torch.Tensor:
    """
    Computes AUC (Area Under the Curve) for binary classification.

    Args:
        n_tasks (int): number of tasks.
        predictions (List[torch.Tensor]): tensor of size (n_tasks, n_examples).
        labels (List[torch.Tensor]): tensor of size (n_tasks, n_examples).
        weights (List[torch.Tensor]): tensor of size (n_tasks, n_examples).
    """
    preds_t, labels_t, weights_t = _concat_if_needed(predictions, labels, weights)
    aucs = [
        _compute_auc_helper(predictions_i, labels_i, weights_i, apply_bin).view(1)
        for predictions_i, labels_i, weights_i in zip(preds_t, labels_t, weights_t)
    ]
    return torch.cat(aucs)


def compute_auc_per_group(
    n_tasks: int,
    predictions: List[torch.Tensor],
    labels: List[torch.Tensor],
    weights: List[torch.Tensor],
    grouping_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Computes AUC (Area Under the Curve) for binary classification for groups of predictions/labels.
    Args:
        n_tasks (int): number of tasks
        predictions (List[torch.Tensor]): tensor of size (n_tasks, n_examples)
        labels (List[torch.Tensor]: tensor of size (n_tasks, n_examples)
        weights (List[torch.Tensor]): tensor of size (n_tasks, n_examples)
        grouping_keys (torch.Tensor): tensor of size (n_examples,)

    Returns:
        torch.Tensor: tensor of size (n_tasks,), average of AUCs per group.
    """
    preds_t, labels_t, weights_t = _concat_if_needed(predictions, labels, weights)
    aucs = []
    if grouping_keys.numel() != 0 and grouping_keys[0] == -1:
        # we added padding  as the first elements during init to avoid floating point exception in sync()
        # removing the paddings to avoid numerical errors.
        grouping_keys = grouping_keys[1:]

    # get unique group indices
    group_indices = torch.unique(grouping_keys)

    for predictions_i, labels_i, weights_i in zip(preds_t, labels_t, weights_t):
        # Loop over each group
        auc_groups_sum = torch.tensor([0], dtype=torch.float32)
        for group_idx in group_indices:
            # get predictions, labels, and weights for this group
            group_mask = grouping_keys == group_idx
            grouped_predictions = predictions_i[group_mask]
            grouped_labels = labels_i[group_mask]
            grouped_weights = weights_i[group_mask]

            auc = _compute_auc_helper(
                grouped_predictions, grouped_labels, grouped_weights
            )
            auc_groups_sum = auc_groups_sum.to(auc.device)
            auc_groups_sum += auc.view(1)
        avg_auc = (
            auc_groups_sum / len(group_indices)
            if len(group_indices) > 0
            else torch.tensor([0.5], dtype=torch.float32)
        )
        aucs.append(avg_auc)
    return torch.cat(aucs)


def _state_reduction(state: List[torch.Tensor], dim: int = 1) -> List[torch.Tensor]:
    return [torch.cat(state, dim=dim)]


_grouping_keys_state_reduction = partial(_state_reduction, dim=0)


class AUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for AUC, i.e. Area Under the Curve.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    Args:
        grouped_auc (bool): If True, computes AUC per group and returns average AUC across all groups.
            The `grouping_keys` is provided during state updates along with predictions, labels, weights.
            This feature is currently not enabled for `fused_update_limit`.
    """

    def __init__(
        self,
        *args: Any,
        grouped_auc: bool = False,
        apply_bin: bool = False,
        fused_update_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grouped_auc and fused_update_limit > 0:
            raise RecMetricException(
                "Grouped AUC and Fused Update Limit cannot be enabled together yet."
            )

        self._grouped_auc: bool = grouped_auc
        self._apply_bin: bool = apply_bin
        self._num_samples: int = 0
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
        if self._grouped_auc:
            self._add_state(
                GROUPING_KEYS,
                [],
                add_window_state=False,
                dist_reduce_fx=_grouping_keys_state_reduction,
                persistent=False,
            )
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
        self._num_samples = 0
        getattr(self, PREDICTIONS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        getattr(self, LABELS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        getattr(self, WEIGHTS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.float, device=self.device)
        )
        if self._grouped_auc:
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
                    predictions/labels per batch. If provided, the AUC metric also
                    computes AUC per group and returns the average AUC across all groups.
        """
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for AUCMetricComputation update"
            )
        predictions = predictions.float()
        labels = labels.float()
        weights = weights.float()
        batch_size = predictions.size(-1)
        start_index = max(self._num_samples + batch_size - self._window_size, 0)

        # Using `self.predictions =` will cause Pyre errors.
        w_preds = getattr(self, PREDICTIONS)
        w_labels = getattr(self, LABELS)
        w_weights = getattr(self, WEIGHTS)

        # remove init states
        if self._num_samples == 0:
            for lst in [w_preds, w_labels, w_weights]:
                lst.pop(0)

        w_preds.append(predictions)
        w_labels.append(labels)
        w_weights.append(weights)

        self._num_samples += batch_size

        while self._num_samples > self._window_size:
            diff = self._num_samples - self._window_size
            if diff > w_preds[0].size(-1):
                self._num_samples -= w_preds[0].size(-1)
                # Remove the first element from predictions, labels, and weights
                for lst in [w_preds, w_labels, w_weights]:
                    lst.pop(0)
            else:
                # Update the first element of predictions, labels, and weights
                # Off by one potentially - keeping legacy behaviour
                for lst in [w_preds, w_labels, w_weights]:
                    lst[0] = lst[0][:, diff:]
                    # if empty tensor, remove it
                    if torch.numel(lst[0]) == 0:
                        lst.pop(0)
                self._num_samples -= diff

        if self._grouped_auc:
            if REQUIRED_INPUTS not in kwargs or (
                (grouping_keys := kwargs[REQUIRED_INPUTS].get(GROUPING_KEYS)) is None
            ):
                raise RecMetricException(
                    f"Input '{GROUPING_KEYS}' are required for AUCMetricComputation grouped update"
                )
            getattr(self, GROUPING_KEYS)[0] = torch.cat(
                [
                    cast(torch.Tensor, getattr(self, GROUPING_KEYS)[0])[start_index:],
                    grouping_keys.squeeze(),
                ],
                dim=0,
            )

    def _compute(self) -> List[MetricComputationReport]:
        reports = []
        reports.append(
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_auc(
                    self._n_tasks,
                    cast(List[torch.Tensor], getattr(self, PREDICTIONS)),
                    cast(List[torch.Tensor], getattr(self, LABELS)),
                    cast(List[torch.Tensor], getattr(self, WEIGHTS)),
                    self._apply_bin,
                ),
            )
        )

        if self._grouped_auc:
            reports.append(
                MetricComputationReport(
                    name=MetricName.GROUPED_AUC,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_auc_per_group(
                        self._n_tasks,
                        cast(List[torch.Tensor], getattr(self, PREDICTIONS)),
                        cast(List[torch.Tensor], getattr(self, LABELS)),
                        cast(List[torch.Tensor], getattr(self, WEIGHTS)),
                        cast(torch.Tensor, getattr(self, GROUPING_KEYS))[0],
                    ),
                )
            )
        return reports

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        """
        This function is overridden from torchmetric.Metric, since for AUC we want to concat the tensors
        right before the allgather collective is called. It directly changes the attributes/states, which
        is ok because end of function sets the attributes to reduced values
        """
        for attr in self._reductions:  # pragma: no cover
            val = getattr(self, attr)
            if isinstance(val, list) and len(val) > 1:
                setattr(self, attr, [torch.cat(val, dim=-1)])
        super()._sync_dist(dist_sync_fn, process_group)

    def reset(self) -> None:
        super().reset()
        self._init_states()


class AUCMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.AUC
    _computation_class: Type[RecMetricComputation] = AUCMetricComputation

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
        if kwargs.get("grouped_auc"):
            self._required_inputs.add(GROUPING_KEYS)
        if self._compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION:
            logging.warning(
                f"compute_mode FUSED_TASKS_AND_STATES_COMPUTATION can't support {self._namespace} yet "
                "because its states are not 1D Tensors. Only FUSED_TASKS_COMPUTATION will take effect."
            )
