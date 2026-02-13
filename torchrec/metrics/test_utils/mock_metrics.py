#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, cast, Dict, List, Optional, Union
from unittest.mock import MagicMock

import torch
from torchrec.metrics.metrics_namespace import MetricNamespaceBase
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecComputeMode,
    RecMetric,
    RecMetricComputation,
    RecModelOutput,
    RecTaskInfo,
)


class MockRecMetricComputation(RecMetricComputation):
    """
    A mock RecMetricComputation that provides controllable behavior for testing.

    State tensors are in the form of:
        - torch.Tensor: a single tensor (used for most RecMetrics such as NE)
        - List[torch.Tensor]: a list of tensors (i.e. AUC)
    """

    def __init__(
        self,
        initial_states: Optional[Dict[str, Any]] = None,
        reduction_fn: Union[
            str, Callable[[List[torch.Tensor], int], List[torch.Tensor]]
        ] = "sum",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            initial_states (Dict[str, Any]): initial states of the mock metric.
            reduction_fn (str): reduction function of computation to be applied
                before and after distributed syncs.
            kwargs: other arguments to pass to RecMetricComputation.
        """
        super().__init__(**kwargs)

        if initial_states:
            for state_name, initial_value in initial_states.items():
                super().add_state(
                    name=state_name,
                    default=initial_value,
                    dist_reduce_fx=reduction_fn,
                    persistent=True,
                )

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _compute(self) -> List[MetricComputationReport]:
        return []


class MockRecMetric(RecMetric):
    """
    A mock RecMetric that uses MockRecMetricComputation for testing. This class
    focuses primarily on creating mock RecMetricComputation and controlling
    the internal state of RecMetrics.
    """

    _computation_class = MockRecMetricComputation
    _namespace: MetricNamespaceBase = MagicMock()

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        reduction_fn: Union[
            str, Callable[[List[torch.Tensor], int], List[torch.Tensor]]
        ] = "sum",
        initial_states: Union[Dict[str, Any], None] = None,
        is_tensor_list: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Same args as RecMetric except for:
            - initial_states (Dict[str, Any]): initial states of the mock metric.
            - reduction_fn (str): reduction function of computation to be applied.
            - is_tensor_list (bool): whether the mock metric's state tensors should be tensor lists.
        """

        # If initial_states is not provided, create a default set of states.
        initial_states = initial_states or create_tensor_states(
            ["state_1", "state_2", "state_3"]
        )

        # torchmetric.Metric's add_state() enforces that tensor lists must be initialized
        # with an empty list. Set it to the initial states after construction.
        default_states = (
            {state_name: [] for state_name in initial_states.keys()}
            if is_tensor_list
            else initial_states
        )

        super().__init__(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=compute_mode,
            # pyrefly: ignore[bad-argument-type]
            **{
                **kwargs,
                "initial_states": default_states,
                "reduction_fn": reduction_fn,
            },
        )

        if is_tensor_list:
            self.set_computation_states(initial_states)

        self.update_called_count = 0
        self.predictions_update_calls: List[RecModelOutput] = []
        self.labels_update_calls: List[RecModelOutput] = []
        self.weights_update_calls: List[Optional[RecModelOutput]] = []
        self._compute_called = False

    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
        **kwargs: Dict[str, Any],
    ) -> None:
        self.update_called_count += 1
        self.predictions_update_calls.append(predictions)
        self.labels_update_calls.append(labels)
        self.weights_update_calls.append(weights)

    def update_called(self) -> bool:
        return self.update_called_count > 0

    def compute(self) -> Dict[str, torch.Tensor]:
        self._compute_called = True
        return {}

    def compute_called(self) -> bool:
        return self._compute_called

    def reset(self) -> None:
        self.update_called_count = 0
        self.predictions_update_calls = []
        self.labels_update_calls = []
        self.weights_update_calls = []

        self._compute_called = False

    def get_computation_states(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: a dictionary of computation states.
        """
        states = {}
        for computation in self._metrics_computations:
            computation = cast(RecMetricComputation, computation)
            for reduction in computation._reductions:
                if hasattr(computation, reduction):
                    states[reduction] = getattr(computation, reduction)
        return states

    def set_computation_states(self, states: Dict[str, Any]) -> None:
        """
        Args:
            states (Dict[str, Any]): a dictionary of computation states to set.
        """
        for computation in self._metrics_computations:
            computation = cast(RecMetricComputation, computation)
            for state_name, value in states.items():
                if state_name in computation._reductions:
                    setattr(computation, state_name, value)

    def add_to_computation_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Add tensors to existing computation states.
        Args:
            states (Dict[str, Any]): a dictionary of computation states to add.
        """
        for computation in self._metrics_computations:
            computation = cast(RecMetricComputation, computation)
            for state_name, value in states.items():
                if state_name in computation._reductions:
                    original_tensor = getattr(computation, state_name)
                    original_tensor += value

    def append_to_computation_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Append computation states to the end of the state tensor list.
        Args:
            states (Dict[str, Any]): a dictionary of computation states to append.
        """
        for computation in self._metrics_computations:
            computation = cast(RecMetricComputation, computation)
            for state_name, value in states.items():
                if state_name in computation._reductions:
                    getattr(computation, state_name).append(value)

    def verify_sync_disabled(self) -> bool:
        """
        Verify that sync is disabled for all computations. torchmetrics.Metric
        uses _to_sync and process_group to control sync.
        """
        for computation in self._metrics_computations:
            computation = cast(RecMetricComputation, computation)
            if computation._to_sync or computation.process_group:
                return False

        return True


def create_metric_states_dict(
    metric_prefix: str,
    computation_name: str,
    metric_states: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a dictionary of metric states with the given prefix and computation name.

    Args:
        metric_prefix (str): prefix of the metric, either a RecTask name or RecComputationMode.
        computation_name (str): name of the RecMetricComputation.
        metric_states (Dict[str, Any]): expected states of the metric.

    Returns:
        Dict[str, Any]: a dictionary of metric states.
    """
    states = {}
    for state_name, value in metric_states.items():
        key = f"{metric_prefix}_{computation_name}_{state_name}"
        states[key] = value
    return states


def assert_tensor_dict_equals(
    actual_states: Dict[str, Any],
    expected_states: Dict[str, Any],
) -> None:
    """
    Verify that the actual states are equal to the expected states.

    Args:
        test_case (unittest.TestCase): test case object.
        actual_states (Dict[str, Any]): actual states of the metric.
        expected_states (Dict[str, Any]): expected states of the metric.
    """
    assert set(actual_states.keys()) == set(
        expected_states.keys()
    ), f"Keys mismatch. Expected {set(expected_states.keys())}, got {set(actual_states.keys())}"

    for key in expected_states:
        actual = actual_states[key]
        expected = expected_states[key]

        if isinstance(expected, torch.Tensor):
            device = expected.device
            actual = actual.to(device)
            torch.testing.assert_close(
                actual,
                expected,
                msg=f"Mismatch for key {key}. Expected {expected}, got {actual}",
            )
        elif isinstance(expected, list):
            assert len(actual) == len(
                expected
            ), f"Length mismatch for key {key}. Expected {len(expected)}, got {len(actual)}"

            for i, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                if isinstance(expected_item, torch.Tensor):
                    device = expected_item.device
                    actual_item = actual_item.to(device)
                    torch.testing.assert_close(
                        actual_item, expected_item, msg=f"Mismatch for key {key}[{i}]"
                    )
                else:
                    assert (
                        actual_item == expected_item
                    ), f"Mismatch for key {key}[{i}]. Expected {expected_item}, got {actual_item}"
        else:
            assert (
                actual == expected
            ), f"Mismatch for key {key}. Expected {expected}, got {actual}"


def create_tensor_states(keys: List[str], n_tasks: int = 1) -> Dict[str, Any]:
    """Create random tensor states for testing (like NE metrics)."""
    return {key: torch.rand(n_tasks) for key in keys}


def create_tensor_list_states(keys: List[str]) -> Dict[str, Any]:
    """Create simple tensor list states for testing (like AUC metrics)."""
    return {key: [torch.rand(1, 2)] for key in keys}
