#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

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


SUM_NDCG = "sum_ndcg"
NUM_SESSIONS = "num_sessions"
REQUIRED_INPUTS = "required_inputs"
SESSION_KEY = "session_id"


def _validate_model_outputs(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    session_ids: torch.Tensor,
) -> None:
    # Sanity check dimensions.
    assert predictions.shape == labels.shape == weights.shape == session_ids.shape
    assert (
        predictions.dim() == 2 and predictions.shape[0] > 0 and predictions.shape[1] > 0
    )
    assert (session_ids[0] == session_ids).all()


def _get_adjusted_ndcg_inputs(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    session_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove all single-length sessions from all variables.
    """
    # Get unique session IDs and their corresponding indices to put them in range (O, N].
    _, converted_session_ids, session_lengths = session_ids[0].unique(
        return_inverse=True, return_counts=True
    )

    example_to_length = torch.gather(
        session_lengths,
        dim=-1,
        index=converted_session_ids.type(torch.int64),
    )
    example_corresponds_to_session_with_length_greater_than_one = example_to_length > 1

    # Remove all single-length sessions.
    return (
        predictions[:, example_corresponds_to_session_with_length_greater_than_one],
        labels[:, example_corresponds_to_session_with_length_greater_than_one],
        weights[:, example_corresponds_to_session_with_length_greater_than_one],
        converted_session_ids[
            example_corresponds_to_session_with_length_greater_than_one
        ],
    )


def _get_ndcg_states(
    *,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    session_ids: torch.Tensor,
    exponential_gain: bool,
    k: int = -1,  # In case we want to support NDCG @ K in the future.
    report_ndcg_as_decreasing_curve: bool = True,
    remove_single_length_sessions: bool = False,
    scale_by_weights_tensor: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Normalized Discounted Cumulative Gain (NDCG) @ k.

    TODO(@venkatrsrinivas): Refactor into smaller helper functions :)
    """
    # Remove all single-length sessions from all variables.
    if remove_single_length_sessions:
        (
            adjusted_predictions,
            adjusted_labels,
            adjusted_weights,
            adjusted_session_ids,
        ) = _get_adjusted_ndcg_inputs(
            predictions=predictions,
            labels=labels,
            weights=weights,
            session_ids=session_ids,
        )
    else:
        (
            adjusted_predictions,
            adjusted_labels,
            adjusted_weights,
            adjusted_session_ids,
        ) = (predictions, labels, weights, session_ids[0])

    # If we are scaling by weights, then do that before NDCG computation.
    if scale_by_weights_tensor:
        adjusted_labels = adjusted_weights * adjusted_labels
        adjusted_predictions = adjusted_weights * adjusted_predictions

    # Helper variables for all reshaping below.
    num_tasks, batch_size = adjusted_labels.shape

    # Get unique session IDs and their corresponding indices to put them in range (O, N].
    (
        unique_session_ids,
        converted_session_ids,
        session_lengths,
    ) = adjusted_session_ids.unique(return_inverse=True, return_counts=True)

    # Healthy assertion that we are trimming sessions correctly.
    if remove_single_length_sessions:
        assert (session_lengths > 1).all()

    num_sessions = unique_session_ids.shape[0]

    # Return early => no state update if there are no sessions.
    if num_sessions == 0:
        return {}

    max_session_length = torch.max(session_lengths)
    max_session_length = (
        max_session_length
        if k == -1
        else torch.min(torch.tensor(k), max_session_length)
    )

    # Convert session IDs to [num_tasks, num_sessions] from [num_sessions,].
    expanded_session_ids = converted_session_ids.expand(num_tasks, -1)

    # Sort labels by themselves and also by predictions.
    sorted_labels_by_labels, sorted_labels_indices = adjusted_labels.sort(
        descending=True, dim=-1
    )
    _, sorted_predictions_indices = adjusted_predictions.sort(descending=True, dim=-1)
    sorted_labels_by_predictions = torch.gather(
        adjusted_labels,
        dim=-1,
        index=sorted_predictions_indices,
    )

    # Expand these to be [num_task, num_sessions, batch_size] for masking to handle later.
    expanded_sorted_labels_by_labels = sorted_labels_by_labels.unsqueeze(1).expand(
        (num_tasks, num_sessions, batch_size)
    )
    expanded_sorted_labels_by_predictions = sorted_labels_by_predictions.unsqueeze(
        1
    ).expand((num_tasks, num_sessions, batch_size))

    # Make sure to correspondingly sort session IDs according to how we sorted labels above.
    session_ids_by_sorted_labels = torch.gather(
        expanded_session_ids,
        dim=-1,
        index=sorted_labels_indices,
    )
    session_ids_by_sorted_predictions = torch.gather(
        expanded_session_ids,
        dim=-1,
        index=sorted_predictions_indices,
    )

    # Helper variable to track every session ID's examples for every task.
    task_to_session_to_examples = (
        torch.arange(num_sessions)
        .view(1, num_sessions, 1)
        .expand(num_tasks, -1, batch_size)
    ).to(device=labels.device)

    # Figure out after sorting which example indices belong to which session.
    sorted_session_ids_by_labels_mask = (
        task_to_session_to_examples == session_ids_by_sorted_labels.unsqueeze(1)
    ).long()
    sorted_session_ids_by_predictions_mask = (
        task_to_session_to_examples == session_ids_by_sorted_predictions.unsqueeze(1)
    ).long()

    # Get the ranks (1, N] for each example in each session for every task.
    label_by_label_ranks = (sorted_session_ids_by_labels_mask).cumsum(dim=-1)
    label_by_prediction_ranks = (sorted_session_ids_by_predictions_mask).cumsum(dim=-1)

    # Compute coresponding discount factors (according to sorting).
    (
        discounts_for_label_by_label,
        discounts_for_label_by_prediction,
    ) = torch.reciprocal(torch.log2(label_by_label_ranks + 1)), torch.reciprocal(
        torch.log2(label_by_prediction_ranks + 1)
    )

    # Account for edge cases and when we want to compute NDCG @ K.
    (
        discounts_for_label_by_label[label_by_label_ranks <= 0],
        discounts_for_label_by_prediction[label_by_prediction_ranks <= 0],
    ) = (
        0.0,
        0.0,
    )
    (
        discounts_for_label_by_label[label_by_label_ranks > max_session_length],
        discounts_for_label_by_prediction[
            label_by_prediction_ranks > max_session_length
        ],
    ) = (
        0.0,
        0.0,
    )

    # Apply mask => to correctly compute ideal and observed gains before applying discounts.
    ideal_gains = expanded_sorted_labels_by_labels * sorted_session_ids_by_labels_mask
    observed_gains = (
        expanded_sorted_labels_by_predictions * sorted_session_ids_by_predictions_mask
    )

    # Apply exponential gain if applicable.
    ideal_gains = torch.exp2(ideal_gains) - 1.0 if exponential_gain else ideal_gains
    observed_gains = (
        torch.exp2(observed_gains) - 1.0 if exponential_gain else observed_gains
    )

    # Apply discounts and sum.
    ideal_dcg = torch.sum(ideal_gains * discounts_for_label_by_label, dim=-1)
    ideal_dcg[ideal_dcg == 0] = 1e-6  # Avoid division by 0.

    observed_dcg = torch.sum(
        observed_gains * discounts_for_label_by_prediction,
        dim=-1,
    )
    ndcg = observed_dcg / ideal_dcg

    max_weights = (
        torch.zeros((num_tasks, num_sessions), dtype=weights.dtype)
        .to(device=adjusted_weights.device)
        .scatter_reduce_(
            dim=-1,
            index=expanded_session_ids,
            src=adjusted_weights,  # [num_tasks, batch_size]
            reduce="amax",
        )
    )

    # Scale NDCG by max weight per session.
    ndcg_report = (1 - ndcg) if report_ndcg_as_decreasing_curve else ndcg
    ndcg_report = ndcg_report.to(device=labels.device)

    # If we aren't scaling gains by weight tensor,
    # just scale by max_weight per session to match weird production logic.
    if not scale_by_weights_tensor:
        ndcg_report *= max_weights

    final_ndcg_report = torch.sum(
        ndcg_report, dim=-1
    )  # Sum over num_sessions for losses => [num_tasks]

    return {
        SUM_NDCG: final_ndcg_report,
        NUM_SESSIONS: torch.full((num_tasks,), fill_value=num_sessions).to(
            device=converted_session_ids.device
        ),
    }


def _compute_ndcg(
    *, sum_ndcg: torch.Tensor, num_sessions: torch.Tensor
) -> torch.Tensor:
    return sum_ndcg / num_sessions


class NDCGComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for NDCG @ K
    (i.e., Normalized Discounted Cumulative Gain @ K).

    Specially this reports (1 - NDCG) so that TensorBoard
    can capture a decreasing "loss" as opposed to an increasing "gain"
    to visualize similarly to normalized entropy (NE) / pointwise measures.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self,
        *args: Any,
        exponential_gain: bool = False,
        session_key: str = SESSION_KEY,
        k: int = -1,
        report_ndcg_as_decreasing_curve: bool = True,
        remove_single_length_sessions: bool = False,
        scale_by_weights_tensor: bool = False,
        is_negative_task_mask: Optional[List[bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._exponential_gain: bool = exponential_gain
        self._session_key: str = session_key
        self._k: int = k
        self._remove_single_length_sessions: bool = remove_single_length_sessions
        self._is_negative_task_mask: Optional[List[bool]] = is_negative_task_mask
        self._report_ndcg_as_decreasing_curve: bool = report_ndcg_as_decreasing_curve
        self._scale_by_weights_tensor: bool = scale_by_weights_tensor
        self._add_state(
            SUM_NDCG,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            NUM_SESSIONS,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Arguments:
            predictions: Tensor of size (n_task, n_examples)
            labels: Tensor of size (n_task, n_examples)
            weights: Tensor of size (n_task, n_examples)
        Returns:
            Nothing => updates state.
        """
        if (
            REQUIRED_INPUTS not in kwargs
            or self._session_key not in kwargs[REQUIRED_INPUTS]
        ):
            raise RecMetricException(
                f"{self._session_key=} should be in {kwargs=} as input. It is required to calculate NDCG loss."
            )

        session_ids = kwargs[REQUIRED_INPUTS][self._session_key]

        if predictions is None or weights is None or session_ids is None:
            raise RecMetricException(
                "Inputs 'predictions', 'weights' and 'session_ids' should not be None for NDCGMetricComputation update"
            )

        # Apply negative scaling to predictions so that
        # we can accurately compute NDCG for negative tasks
        # (e.g., NDCG_p(skip) prefers to have label => 0
        # towards the top of the relevant ranked list > label = 1)
        # Or maybe, we want to compute NDCG p(skip) the same way as NDCG p(like).
        # In either case, this mask gives us the full control.
        if self._is_negative_task_mask is not None:
            predictions[self._is_negative_task_mask, :] = (
                1 - predictions[self._is_negative_task_mask, :]
            )
            labels[self._is_negative_task_mask, :] = (
                1 - labels[self._is_negative_task_mask, :]
            )

        _validate_model_outputs(
            predictions=predictions,
            labels=labels,
            weights=weights,
            session_ids=session_ids,
        )

        predictions = predictions.double()
        labels = labels.double()
        weights = weights.double()

        # Calculate NDCG loss at current iterations.
        states = _get_ndcg_states(
            labels=labels,
            predictions=predictions,
            weights=weights,
            session_ids=session_ids,
            exponential_gain=self._exponential_gain,
            remove_single_length_sessions=self._remove_single_length_sessions,
            report_ndcg_as_decreasing_curve=self._report_ndcg_as_decreasing_curve,
            k=self._k,
            scale_by_weights_tensor=self._scale_by_weights_tensor,
        )

        # Update based on the new states.
        for state_name, state_value in states.items():
            state = getattr(self, state_name).to(labels.device)
            state += state_value
            self._aggregate_window_state(state_name, state_value, predictions.shape[-1])

    def _compute(self) -> List[MetricComputationReport]:

        return [
            MetricComputationReport(
                name=MetricName.NDCG,
                metric_prefix=MetricPrefix.LIFETIME,
                value=_compute_ndcg(
                    sum_ndcg=cast(torch.Tensor, getattr(self, SUM_NDCG)),
                    num_sessions=cast(torch.Tensor, getattr(self, NUM_SESSIONS)),
                ),
            ),
            MetricComputationReport(
                name=MetricName.NDCG,
                metric_prefix=MetricPrefix.WINDOW,
                value=_compute_ndcg(
                    sum_ndcg=self.get_window_state(SUM_NDCG),
                    num_sessions=self.get_window_state(NUM_SESSIONS),
                ),
            ),
        ]


class NDCGMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.NDCG
    _computation_class: Type[RecMetricComputation] = NDCGComputation

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
        # This is the required metadata to be enriched with
        # the session ID information by loss wrappers, etc.
        # This is set through the front-end configurations,
        # => fallback back to "session_id" if not specified.
        if "session_key" not in kwargs:
            self._required_inputs.add(SESSION_KEY)
        else:
            self._required_inputs.add(kwargs["session_key"])

    def _get_task_kwargs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Dict[str, Any]:
        all_task_info = (
            [task_config] if isinstance(task_config, RecTaskInfo) else task_config
        )

        # Just sanity in weird case if we have no tasks (should never happen).
        if len(all_task_info) == 0:
            return {}
        return {
            "is_negative_task_mask": [
                task_info.is_negative_task for task_info in all_task_info
            ]
        }
