#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
)


def compute_multi_label_precision(
    num_true_positives: torch.Tensor, num_false_positives: torch.Tensor
) -> torch.Tensor:
    total = num_true_positives + num_false_positives
    return torch.where(
        total == 0.0,
        torch.zeros_like(total),
        num_true_positives / total,
    )


class MultiLabelPrecisionMetricComputation(RecMetricComputation):
    r"""Multi-label precision metric computation class.

    This class computes precision for multi-label classification tasks where each
    sample can belong to multiple classes simultaneously. Unlike binary or
    multi-class classification, multi-label classification allows for multiple
    positive labels per sample.

    **Input Format:**
        Both predictions and labels must be integer-encoded tensors where each
        integer represents a binary vector encoded using LSB-first bit ordering.
        For example, with num_labels=3:
            - Integer 5 (binary 101) decodes to [1, 0, 1] (labels 0 and 2 are positive)
            - Integer 3 (binary 011) decodes to [1, 1, 0] (labels 0 and 1 are positive)

    **Precision Formula:**
        Precision = TP / (TP + FP)

    Where:
        - TP (True Positives): Decoded prediction == 1 AND decoded label == 1
        - FP (False Positives): Decoded prediction == 1 AND decoded label == 0

    Args:
        num_labels: Number of labels in the multi-label classification task.
            This determines how many bits to decode from each integer. Default: 1
        label_names: Optional list of human-readable names for each label.
            If not provided, defaults to ["label_0", "label_1", ..., "label_{n-1}"]
        *args: Additional positional arguments passed to RecMetricComputation
        **kwargs: Additional keyword arguments passed to RecMetricComputation

    Example:
        >>> # predictions and labels are integer-encoded
        >>> # e.g., prediction=5 means labels [1,0,1], label=3 means [1,1,0]
        >>> metric = MultiLabelPrecisionMetric(
        ...     world_size=1,
        ...     my_rank=0,
        ...     batch_size=32,
        ...     tasks=[task],
        ...     num_labels=3,
        ...     label_names=["cat", "dog", "horse"],
        ... )
        >>> metric.update(predictions=int_preds, labels=int_labels, weights=None)
        >>> results = metric.compute()  # Returns per-label precision
    """

    def __init__(
        self,
        *args: Any,
        num_labels: int = 1,
        label_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._num_labels: Optional[int] = num_labels  # Store num_labels from config
        self._label_names: List[str] = (
            # pyre-fixme[6]: For 1st argument expected `SupportsIndex` but got
            #  `Optional[int]`.
            [f"label_{i}" for i in range(self._num_labels)]
            if label_names is None
            else label_names
        )

        self._per_label_states_initialized = False

        for label_name in self._label_names:
            self._add_state(
                f"true_pos_sum_{label_name}",
                torch.zeros(1, dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx="sum",
                persistent=True,
            )
            self._add_state(
                f"false_pos_sum_{label_name}",
                torch.zeros(1, dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx="sum",
                persistent=True,
            )

    def _decode_integer_to_labels(
        self,
        tensor: torch.Tensor,
        num_labels: int,
    ) -> torch.Tensor:
        """Convert integer-encoded tensors to binary label vectors (LSB first).

        Each integer is decoded to a binary vector where bit i represents label i.
        Uses LSB-first ordering: the least significant bit corresponds to label 0.
        """
        device = tensor.device
        dtype = tensor.dtype
        powers_of_two = (2 ** torch.arange(num_labels, device=device)).to(dtype)
        tensor_expanded = tensor.reshape(-1, 1)
        label_tensor = (tensor_expanded // powers_of_two) % 2
        return label_tensor.to(torch.int32)

    def _prepare_inputs(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode integer-encoded inputs and prepare for precision computation.

        Converts integer-encoded predictions and labels to binary multi-label
        format and expands weights to match the decoded tensor shape.
        """
        # predictions and labels are integer-encoded [batch_size] or [batch_size, 1]
        predictions = self._decode_integer_to_labels(
            predictions.flatten(),
            # pyre-fixme[6]: For 2nd argument expected `int` but got `Optional[int]`.
            self._num_labels,
        )
        # pyre-fixme[6]: For 2nd argument expected `int` but got `Optional[int]`.
        labels = self._decode_integer_to_labels(labels.flatten(), self._num_labels)
        # Expand weights to match
        if weights is not None:
            # pyre-fixme[6]: For 2nd argument expected `Union[int, SymInt]` but got
            #  `Optional[int]`.
            weights = weights.flatten().unsqueeze(-1).expand(-1, self._num_labels)
        else:
            weights = torch.ones_like(predictions, dtype=torch.float32)
        return predictions, labels, weights

    def _update_label_states(
        self,
        label_idx: int,
        label_name: str,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Compute and update TP/FP states for a single label."""
        pred_i = predictions[:, label_idx]
        target_i = labels[:, label_idx]
        weight_i = weights[:, label_idx]

        # Compute TP and FP for this label (predictions are already binary 0/1)
        true_pos = torch.sum(weight_i * (pred_i * target_i))
        false_pos = torch.sum(weight_i * (pred_i * (1 - target_i)))

        # Update lifetime states
        tp_state = getattr(self, f"true_pos_sum_{label_name}")
        fp_state = getattr(self, f"false_pos_sum_{label_name}")
        tp_state.add_(true_pos)
        fp_state.add_(false_pos)

        # Update window states
        batch_size = predictions.shape[0]
        self._aggregate_window_state(f"true_pos_sum_{label_name}", true_pos, batch_size)
        self._aggregate_window_state(
            f"false_pos_sum_{label_name}", false_pos, batch_size
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        """Update metric states with a new batch of integer-encoded data."""
        # pyre-fixme[16]: `Optional` has no attribute `device`.
        device = predictions.device
        self.to(device)

        predictions, labels, weights = self._prepare_inputs(
            # pyre-fixme[6]: For 1st argument expected `Tensor` but got
            #  `Optional[Tensor]`.
            predictions,
            labels,
            weights,
        )

        for i, label_name in enumerate(self._label_names):
            self._update_label_states(i, label_name, predictions, labels, weights)

    def _compute(self) -> List[MetricComputationReport]:
        reports = []

        # Per-label precision reports
        for label_name in self._label_names:
            tp_state = getattr(self, f"true_pos_sum_{label_name}")
            fp_state = getattr(self, f"false_pos_sum_{label_name}")

            # Lifetime precision for this label
            reports.append(
                MetricComputationReport(
                    name=MetricName.MULTI_LABEL_PRECISION,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=compute_multi_label_precision(tp_state, fp_state),
                    description=f"{label_name}",
                )
            )

            # Window precision for this label
            reports.append(
                MetricComputationReport(
                    name=MetricName.MULTI_LABEL_PRECISION,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_multi_label_precision(
                        self.get_window_state(f"true_pos_sum_{label_name}"),
                        self.get_window_state(f"false_pos_sum_{label_name}"),
                    ),
                    description=f"{label_name}",
                )
            )
        return reports


class MultiLabelPrecisionMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.MULTI_LABEL_PRECISION
    _computation_class: Type[RecMetricComputation] = (
        MultiLabelPrecisionMetricComputation
    )
