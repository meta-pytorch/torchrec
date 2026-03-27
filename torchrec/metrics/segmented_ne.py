#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Type, Union

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


logger: logging.Logger = logging.getLogger(__name__)

PREDICTIONS = "predictions"
LABELS = "labels"
WEIGHTS = "weights"


def compute_cross_entropy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    predictions = predictions.double()
    predictions.clamp_(min=eta, max=1 - eta)
    cross_entropy = -weights * labels * torch.log2(predictions) - weights * (
        1.0 - labels
    ) * torch.log2(1.0 - predictions)
    return cross_entropy


def _compute_cross_entropy_norm(
    mean_label: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    mean_label = mean_label.double()
    mean_label.clamp_(min=eta, max=1 - eta)
    return -pos_labels * torch.log2(mean_label) - neg_labels * torch.log2(
        1.0 - mean_label
    )


def compute_ne_helper(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    mean_label = pos_labels / weighted_num_samples
    ce_norm = _compute_cross_entropy_norm(mean_label, pos_labels, neg_labels, eta)
    return ce_sum / ce_norm


def compute_logloss(
    ce_sum: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    # we utilize tensor broadcasting for operations
    labels_sum = pos_labels + neg_labels
    labels_sum.clamp_(min=eta)
    return ce_sum / labels_sum


def compute_ne(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    num_groups: int,
    eta: float,
) -> torch.Tensor:
    # size should be (num_groups)
    result_ne = torch.zeros(num_groups)
    for group in range(num_groups):
        mean_label = pos_labels[group] / weighted_num_samples[group]
        ce_norm = _compute_cross_entropy_norm(
            mean_label, pos_labels[group], neg_labels[group], eta
        )
        ne = ce_sum[group] / ce_norm
        result_ne[group] = ne

    # ne indexed by group - tensor size (num_groups)
    return result_ne


def compute_ne_fused(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    num_groups: int,
    n_tasks: int,
    eta: float,
) -> torch.Tensor:
    # size should be (n_tasks, num_groups)
    result_ne = torch.zeros([n_tasks, num_groups])
    for group in range(num_groups):
        mean_label = pos_labels[:, group] / weighted_num_samples[:, group]
        ce_norm = _compute_cross_entropy_norm(
            mean_label, pos_labels[:, group], neg_labels[:, group], eta
        )
        ne = ce_sum[:, group] / ce_norm
        result_ne[:, group] = ne

    # ne indexed by group - tensor size (num_groups)
    return result_ne


def get_segemented_ne_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
    eta: float,
    num_groups: int,
) -> Dict[str, torch.Tensor]:
    groups = torch.unique(grouping_keys)
    buffer = torch.zeros((4, num_groups), device=labels.device)
    cross_entropy, weighted_num_samples, pos_labels, neg_labels = buffer.unbind(0)
    for group in groups:
        group_mask = grouping_keys == group

        group_labels = labels[group_mask]
        group_predictions = predictions[group_mask]
        group_weights = weights[group_mask]

        ce_sum_group = torch.sum(
            compute_cross_entropy(
                labels=group_labels,
                predictions=group_predictions,
                weights=group_weights,
                eta=eta,
            ),
            dim=-1,
        )

        weighted_num_samples_group = torch.sum(group_weights, dim=-1)
        pos_labels_group = torch.sum(group_weights * group_labels, dim=-1)
        neg_labels_group = torch.sum(group_weights * (1.0 - group_labels), dim=-1)

        cross_entropy[group] = ce_sum_group.item()
        weighted_num_samples[group] = weighted_num_samples_group.item()
        pos_labels[group] = pos_labels_group.item()
        neg_labels[group] = neg_labels_group.item()

    # tensor size for each value is (num_groups)
    return {
        "cross_entropy_sum": cross_entropy,
        "weighted_num_samples": weighted_num_samples,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels,
    }


def get_segemented_ne_states_fused(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
    eta: float,
    num_groups: int,
    n_tasks: int,
) -> Dict[str, torch.Tensor]:
    # labels, predictions, weights: (n_tasks, num_samples)
    # grouping_keys: (num_samples,) with integer values in [0, num_groups)

    # Compute per-sample cross-entropy across all groups at once
    ce = compute_cross_entropy(labels, predictions, weights, eta)

    # Build scatter index: (n_tasks, num_samples), same group index across tasks
    group_idx = grouping_keys.long().unsqueeze(0).expand(n_tasks, -1)

    # Accumulate per-group sums via scatter_add_ (no GPU sync points)
    def _scatter_sum(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_tasks, num_groups, dtype=torch.double, device=values.device)
        out.scatter_add_(1, group_idx, values.double())
        return out

    cross_entropy = _scatter_sum(ce)
    weighted_num_samples = _scatter_sum(weights)
    pos_labels = _scatter_sum(weights * labels)
    neg_labels = _scatter_sum(weights * (1.0 - labels))

    return {
        "cross_entropy_sum": cross_entropy,
        "weighted_num_samples": weighted_num_samples,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels,
    }


def _state_reduction_sum(state: torch.Tensor) -> torch.Tensor:
    return state.sum(dim=0)


@dataclass
class GroupingKeyConfig:
    """Configuration for a single grouping key.

    Args:
        name: The name of the tensor containing the grouping key values.
        num_groups: Number of groups for this grouping key.
        cast_keys_to_int: Whether to cast the grouping key values to int64.
    """

    name: str
    num_groups: int = 1
    cast_keys_to_int: bool = False


def _normalize_grouping_keys_config(
    grouping_keys: Union[str, List[str], List[Dict[str, Any]], List[GroupingKeyConfig]],
    num_groups: int,
    cast_keys_to_int: bool,
) -> List[GroupingKeyConfig]:
    """Normalize grouping_keys input to a list of GroupingKeyConfig objects.

    Args:
        grouping_keys: Can be:
            - A string (single key name) - uses num_groups and cast_keys_to_int
            - A list of strings (multiple key names) - each uses num_groups and cast_keys_to_int
            - A list of dicts with keys: name, num_groups (optional), cast_keys_to_int (optional)
            - A list of GroupingKeyConfig objects
        num_groups: Default number of groups (used when grouping_keys is a string or list of strings)
        cast_keys_to_int: Default cast setting (used when grouping_keys is a string or list of strings)

    Returns:
        List of GroupingKeyConfig objects
    """
    if isinstance(grouping_keys, str):
        return [
            GroupingKeyConfig(
                name=grouping_keys,
                num_groups=num_groups,
                cast_keys_to_int=cast_keys_to_int,
            )
        ]

    configs = []
    for item in grouping_keys:
        if isinstance(item, str):
            configs.append(
                GroupingKeyConfig(
                    name=item,
                    num_groups=num_groups,
                    cast_keys_to_int=cast_keys_to_int,
                )
            )
        elif isinstance(item, dict):
            configs.append(
                GroupingKeyConfig(
                    name=item["name"],
                    num_groups=item.get("num_groups", num_groups),
                    cast_keys_to_int=item.get("cast_keys_to_int", cast_keys_to_int),
                )
            )
        elif isinstance(item, GroupingKeyConfig):
            configs.append(item)
        else:
            raise ValueError(
                f"Invalid grouping_keys item type: {type(item)}. "
                "Expected str, dict, or GroupingKeyConfig."
            )
    return configs


class SegmentedNEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Segmented NE, i.e. Normalized Entropy - for boolean labels.

    Only binary labels are currently supported (0s, 1s), NE is computed for each label, NE across the whole model output
    can be done through the normal NE metric.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.

    Args:
        include_logloss (bool): return vanilla logloss as one of metrics results, on top of segmented NE.
        num_groups (int): number of groups to segment NE by. This is the default for all grouping_keys.
        grouping_keys (Union[str, List[str], List[Dict], List[GroupingKeyConfig]]): Specifies the grouping key(s).
            Can be:
            - A string (single key name)
            - A list of strings (multiple key names, each using the same num_groups)
            - A list of dicts with keys: "name" (required), "num_groups" (optional), "cast_keys_to_int" (optional)
            - A list of GroupingKeyConfig objects
        cast_keys_to_int (bool): whether to cast grouping_keys to torch.int64. Only works if grouping_keys is of type torch.float32.
    """

    def __init__(
        self,
        *args: Any,
        include_logloss: bool = False,
        num_groups: int = 1,
        grouping_keys: Union[
            str, List[str], List[Dict[str, Any]], List[GroupingKeyConfig]
        ] = "grouping_keys",
        cast_keys_to_int: bool = False,
        **kwargs: Any,
    ) -> None:
        self._include_logloss: bool = include_logloss
        super().__init__(*args, **kwargs)

        # Normalize grouping_keys to a list of GroupingKeyConfig
        self._grouping_key_configs = _normalize_grouping_keys_config(
            grouping_keys, num_groups, cast_keys_to_int
        )

        # Track whether we're in single-default-key mode for backward compat.
        # Use no prefix/suffix when:
        #   - Original input was a plain string (not wrapped in a list), OR
        #   - There is exactly one config with the default name "grouping_keys"
        self._is_single_default_key: bool = isinstance(grouping_keys, str) or (
            len(self._grouping_key_configs) == 1
            and self._grouping_key_configs[0].name == "grouping_keys"
        )

        # For backward compatibility only: expose _num_groups and _grouping_keys
        # as instance attributes. These are ONLY used by legacy code that might
        # access them directly. The actual metric computation uses each config's
        # own num_groups, so this has no impact on correctness.
        # We use the first config's values since legacy code expected a single key.
        self._num_groups = self._grouping_key_configs[0].num_groups
        self._grouping_keys = self._grouping_key_configs[0].name
        self._cast_keys_to_int = cast_keys_to_int

        self.eta = 1e-12

        # Create states for each grouping key config
        for config in self._grouping_key_configs:
            state_prefix = self._get_state_prefix(config.name)
            self._add_state(
                f"{state_prefix}cross_entropy_sum",
                torch.zeros((self._n_tasks, config.num_groups), dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx=_state_reduction_sum,
                persistent=True,
            )
            self._add_state(
                f"{state_prefix}weighted_num_samples",
                torch.zeros((self._n_tasks, config.num_groups), dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx=_state_reduction_sum,
                persistent=True,
            )
            self._add_state(
                f"{state_prefix}pos_labels",
                torch.zeros((self._n_tasks, config.num_groups), dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx=_state_reduction_sum,
                persistent=True,
            )
            self._add_state(
                f"{state_prefix}neg_labels",
                torch.zeros((self._n_tasks, config.num_groups), dtype=torch.double),
                add_window_state=True,
                dist_reduce_fx=_state_reduction_sum,
                persistent=True,
            )

    def _get_state_prefix(self, key_name: str) -> str:
        """Get the state prefix for a grouping key.

        For backward compatibility, if the metric was created with a single
        default key (string input or single "grouping_keys"), use no prefix.
        """
        if self._is_single_default_key:
            return ""
        return f"{key_name}_"

    def _get_description_suffix(self, key_name: str) -> str:
        """Get the description suffix for metric reports.

        For backward compatibility, if the metric was created with a single
        default key, don't add a suffix.
        """
        if self._is_single_default_key:
            return ""
        return f"@{key_name}"

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for SegmentedNEMetricComputation update"
            )

        if "required_inputs" not in kwargs:
            raise RecMetricException(
                f"Required inputs for SegmentedNEMetricComputation update should be provided, got kwargs: {kwargs}"
            )

        required_inputs = kwargs["required_inputs"]
        num_samples = predictions.shape[-1]

        # Process each grouping key configuration
        for config in self._grouping_key_configs:
            key_name = config.name
            state_prefix = self._get_state_prefix(key_name)

            if required_inputs.get(key_name) is None:
                raise RecMetricException(
                    f"Required inputs for SegmentedNEMetricComputation update should contain '{key_name}', got keys: {list(required_inputs.keys())}"
                )

            grouping_keys_tensor = required_inputs[key_name]

            # Validate and cast dtype
            if grouping_keys_tensor.dtype != torch.int64:
                if config.cast_keys_to_int and grouping_keys_tensor.dtype in (
                    torch.float32,
                    torch.float64,
                ):
                    grouping_keys_tensor = grouping_keys_tensor.to(torch.int64)
                else:
                    raise RecMetricException(
                        f"Grouping key '{key_name}' expected to have type torch.int64 or torch.float32/torch.float64 with cast_keys_to_int set to true, got {grouping_keys_tensor.dtype}."
                    )

            # Compute states for this grouping key
            if (
                self._compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
                or self._compute_mode
                == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
            ):
                states = get_segemented_ne_states_fused(
                    labels,
                    predictions,
                    weights,
                    grouping_keys_tensor,
                    eta=self.eta,
                    num_groups=config.num_groups,
                    n_tasks=self._n_tasks,
                )
            else:
                states = get_segemented_ne_states(
                    labels,
                    predictions,
                    weights,
                    grouping_keys_tensor,
                    eta=self.eta,
                    num_groups=config.num_groups,
                )

            # Update states with proper prefix
            for state_name, state_value in states.items():
                full_state_name = f"{state_prefix}{state_name}"
                state = getattr(self, full_state_name)
                state += state_value
                self._aggregate_window_state(full_state_name, state_value, num_samples)

    def _compute_fused(self) -> List[MetricComputationReport]:
        reports = []

        for config in self._grouping_key_configs:
            key_name = config.name
            state_prefix = self._get_state_prefix(key_name)
            description_suffix = self._get_description_suffix(key_name)

            # Compute lifetime NE
            computed_ne = compute_ne_fused(
                getattr(self, f"{state_prefix}cross_entropy_sum"),
                getattr(self, f"{state_prefix}weighted_num_samples"),
                getattr(self, f"{state_prefix}pos_labels"),
                getattr(self, f"{state_prefix}neg_labels"),
                num_groups=config.num_groups,
                n_tasks=self._n_tasks,
                eta=self.eta,
            )
            # Compute window NE
            window_computed_ne = compute_ne_fused(
                self.get_window_state(f"{state_prefix}cross_entropy_sum"),
                self.get_window_state(f"{state_prefix}weighted_num_samples"),
                self.get_window_state(f"{state_prefix}pos_labels"),
                self.get_window_state(f"{state_prefix}neg_labels"),
                num_groups=config.num_groups,
                n_tasks=self._n_tasks,
                eta=self.eta,
            )
            for group in range(config.num_groups):
                reports.append(
                    MetricComputationReport(
                        name=MetricName.SEGMENTED_NE,
                        metric_prefix=MetricPrefix.LIFETIME,
                        value=computed_ne[:, group],
                        description=f"_{group}{description_suffix}",
                    ),
                )
                reports.append(
                    MetricComputationReport(
                        name=MetricName.SEGMENTED_NE,
                        metric_prefix=MetricPrefix.WINDOW,
                        value=window_computed_ne[:, group],
                        description=f"_{group}{description_suffix}",
                    ),
                )

            if self._include_logloss:
                log_loss_groups = compute_logloss(
                    getattr(self, f"{state_prefix}cross_entropy_sum"),
                    getattr(self, f"{state_prefix}pos_labels"),
                    getattr(self, f"{state_prefix}neg_labels"),
                    eta=self.eta,
                )
                window_log_loss_groups = compute_logloss(
                    self.get_window_state(f"{state_prefix}cross_entropy_sum"),
                    self.get_window_state(f"{state_prefix}pos_labels"),
                    self.get_window_state(f"{state_prefix}neg_labels"),
                    eta=self.eta,
                )
                for group in range(config.num_groups):
                    reports.append(
                        MetricComputationReport(
                            name=MetricName.LOG_LOSS,
                            metric_prefix=MetricPrefix.LIFETIME,
                            value=log_loss_groups[:, group],
                            description=f"_{group}{description_suffix}",
                        )
                    )
                    reports.append(
                        MetricComputationReport(
                            name=MetricName.LOG_LOSS,
                            metric_prefix=MetricPrefix.WINDOW,
                            value=window_log_loss_groups[:, group],
                            description=f"_{group}{description_suffix}",
                        )
                    )

        return reports

    def _compute(self) -> List[MetricComputationReport]:
        reports = []
        if (
            self._compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
            or self._compute_mode == RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
        ):
            return self._compute_fused()

        # For non-fused mode, iterate over all grouping key configs
        for config in self._grouping_key_configs:
            key_name = config.name
            state_prefix = self._get_state_prefix(key_name)
            description_suffix = self._get_description_suffix(key_name)

            # Compute lifetime NE
            computed_ne = compute_ne(
                getattr(self, f"{state_prefix}cross_entropy_sum")[0],
                getattr(self, f"{state_prefix}weighted_num_samples")[0],
                getattr(self, f"{state_prefix}pos_labels")[0],
                getattr(self, f"{state_prefix}neg_labels")[0],
                num_groups=config.num_groups,
                eta=self.eta,
            )

            # Compute window NE
            window_computed_ne = compute_ne(
                self.get_window_state(f"{state_prefix}cross_entropy_sum")[0],
                self.get_window_state(f"{state_prefix}weighted_num_samples")[0],
                self.get_window_state(f"{state_prefix}pos_labels")[0],
                self.get_window_state(f"{state_prefix}neg_labels")[0],
                num_groups=config.num_groups,
                eta=self.eta,
            )

            for group in range(config.num_groups):
                reports.append(
                    MetricComputationReport(
                        name=MetricName.SEGMENTED_NE,
                        metric_prefix=MetricPrefix.LIFETIME,
                        value=computed_ne[group],
                        description=f"_{group}{description_suffix}",
                    ),
                )
                reports.append(
                    MetricComputationReport(
                        name=MetricName.SEGMENTED_NE,
                        metric_prefix=MetricPrefix.WINDOW,
                        value=window_computed_ne[group],
                        description=f"_{group}{description_suffix}",
                    ),
                )

            if self._include_logloss:
                log_loss_groups = compute_logloss(
                    getattr(self, f"{state_prefix}cross_entropy_sum")[0],
                    getattr(self, f"{state_prefix}pos_labels")[0],
                    getattr(self, f"{state_prefix}neg_labels")[0],
                    eta=self.eta,
                )

                window_log_loss_groups = compute_logloss(
                    self.get_window_state(f"{state_prefix}cross_entropy_sum")[0],
                    self.get_window_state(f"{state_prefix}pos_labels")[0],
                    self.get_window_state(f"{state_prefix}neg_labels")[0],
                    eta=self.eta,
                )

                for group in range(config.num_groups):
                    reports.append(
                        MetricComputationReport(
                            name=MetricName.LOG_LOSS,
                            metric_prefix=MetricPrefix.LIFETIME,
                            value=log_loss_groups[group],
                            description=f"_{group}{description_suffix}",
                        )
                    )
                    reports.append(
                        MetricComputationReport(
                            name=MetricName.LOG_LOSS,
                            metric_prefix=MetricPrefix.WINDOW,
                            value=window_log_loss_groups[group],
                            description=f"_{group}{description_suffix}",
                        )
                    )

        return reports


class SegmentedNEMetric(RecMetric):
    # pyrefly: ignore[bad-override]
    _namespace: MetricNamespace = MetricNamespace.SEGMENTED_NE
    _computation_class: Type[RecMetricComputation] = SegmentedNEMetricComputation

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
        # Handle required inputs for grouping_keys
        grouping_keys_value = kwargs.get("grouping_keys", "grouping_keys")
        grouping_keys: Union[
            str, List[str], List[Dict[str, Any]], List[GroupingKeyConfig]
        ] = cast(
            Union[str, List[str], List[Dict[str, Any]], List[GroupingKeyConfig]],
            grouping_keys_value,
        )
        num_groups: int = cast(int, kwargs.get("num_groups", 1))
        cast_keys_to_int: bool = cast(bool, kwargs.get("cast_keys_to_int", False))

        # Normalize to list of configs to extract all required input names
        configs = _normalize_grouping_keys_config(
            grouping_keys, num_groups, cast_keys_to_int
        )
        for config in configs:
            self._required_inputs.add(config.name)
