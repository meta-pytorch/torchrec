#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Iterable, Union

import torch
from torch import no_grad
from torchrec.metrics.metrics_config import RecComputeMode
from torchrec.metrics.rec_metric import RecMetricException, RecTaskInfo
from torchrec.metrics.segmented_ne import (
    _normalize_grouping_keys_config,
    GroupingKeyConfig,
    SegmentedNEMetric,
)


class SegementedNEValueTest(unittest.TestCase):
    """
    This set of tests verify the computation logic of AUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    @no_grad()
    def _test_segemented_ne_helper(
        self,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        expected_ne: torch.Tensor,
        grouping_keys: torch.Tensor,
        grouping_key_tensor_name: str = "grouping_keys",
        cast_keys_to_int: bool = False,
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        check_window_metrics: bool = False,
    ) -> None:
        num_task = labels.shape[0]
        batch_size = labels.shape[0]
        task_list = []
        inputs: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]] = {
            "predictions": {},
            "labels": {},
            "weights": {},
        }
        if grouping_keys is not None:
            inputs["required_inputs"] = {grouping_key_tensor_name: grouping_keys}
        for i in range(num_task):
            task_info = RecTaskInfo(
                name=f"Task:{i}",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
            task_list.append(task_info)
            # pyrefly: ignore[unsupported-operation]
            inputs["predictions"][task_info.name] = predictions[i]
            # pyrefly: ignore[unsupported-operation]
            inputs["labels"][task_info.name] = labels[i]
            # pyrefly: ignore[unsupported-operation]
            inputs["weights"][task_info.name] = weights[i]

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=max(2, torch.unique(grouping_keys)[-1].item() + 1),
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=grouping_key_tensor_name,
            # pyrefly: ignore[bad-argument-type]
            cast_keys_to_int=cast_keys_to_int,
            compute_mode=compute_mode,
            window_size=100,
        )
        ne.update(**inputs)
        actual_ne = ne.compute()

        for task_id, task in enumerate(task_list):
            for label in [0, 1]:
                # Check lifetime metrics
                cur_actual_ne = actual_ne[
                    f"segmented_ne-{task.name}|lifetime_segmented_ne_{label}"
                ]
                cur_expected_ne = expected_ne[task_id][label]

                torch.testing.assert_close(
                    cur_actual_ne,
                    cur_expected_ne,
                    atol=1e-4,
                    rtol=1e-4,
                    check_dtype=False,
                    equal_nan=True,
                    msg=f"Lifetime Actual: {cur_actual_ne}, Expected: {cur_expected_ne}",
                )

                # Check window metrics exist and have reasonable values
                if check_window_metrics:
                    window_key = f"segmented_ne-{task.name}|window_segmented_ne_{label}"
                    self.assertIn(
                        window_key,
                        actual_ne,
                        f"Window metric {window_key} should exist in output",
                    )
                    # Window NE should equal lifetime NE for a single batch
                    cur_window_ne = actual_ne[window_key]
                    torch.testing.assert_close(
                        cur_window_ne,
                        cur_expected_ne,
                        atol=1e-4,
                        rtol=1e-4,
                        check_dtype=False,
                        equal_nan=True,
                        msg=f"Window Actual: {cur_window_ne}, Expected: {cur_expected_ne}",
                    )

    def test_grouped_ne(self) -> None:
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in UNFUSED_TASKS_COMPUTATION mode",
                    inputs,
                )
                raise

            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in FUSED_TASKS_COMPUTATION mode",
                    inputs,
                )
                raise

            try:
                self._test_segemented_ne_helper(
                    **inputs,
                    compute_mode=RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
                )
            except AssertionError:
                print(
                    "Assertion error caught with data set in FUSED_TASKS_AND_STATES_COMPUTATION mode",
                    inputs,
                )
                raise

    def test_window_metrics(self) -> None:
        """Test that window metrics are computed correctly for SegmentedNE."""
        test_data = generate_model_outputs_cases()
        for inputs in test_data:
            # Test window metrics in all compute modes
            for compute_mode in [
                RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_COMPUTATION,
                RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION,
            ]:
                try:
                    self._test_segemented_ne_helper(
                        **inputs,
                        compute_mode=compute_mode,
                        check_window_metrics=True,
                    )
                except AssertionError:
                    print(
                        f"Assertion error caught with data set in {compute_mode} mode (window metrics)",
                        inputs,
                    )
                    raise


def generate_model_outputs_cases() -> Iterable[Dict[str, Any]]:
    return [
        # base condition
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
        },
        # one sided, edge case 1s
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([1, 1, 1, 1, 1]),
            "expected_ne": torch.tensor([[torch.nan, 1.3936]]),
        },
        # one sided, edge case 0s
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 0, 0, 0, 0]),
            "expected_ne": torch.tensor([[1.3936, torch.nan]]),
        },
        # three labels,
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1, 0]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9, 0.4]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75, 0.4]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 2, 2]),
            "expected_ne": torch.tensor([[3.1615, 1.8311, 0.3814]]),
        },
        # two tasks
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1], [1, 0, 0, 1, 1]]),
            "predictions": torch.tensor(
                [
                    [0.2, 0.6, 0.8, 0.4, 0.9],
                    [0.6, 0.2, 0.4, 0.8, 0.9],
                ]
            ),
            "weights": torch.tensor(
                [
                    [0.13, 0.2, 0.5, 0.8, 0.75],
                    [0.13, 0.2, 0.5, 0.8, 0.75],
                ]
            ),
            "grouping_keys": torch.tensor(
                [0, 1, 0, 1, 1]
            ),  # for this case, both tasks have same groupings
            "expected_ne": torch.tensor([[3.1615, 1.6004], [1.0034, 0.4859]]),
        },
        # Custom grouping key tensor name
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor([0, 1, 0, 1, 1]),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
            "grouping_key_tensor_name": "custom_key",
        },
        # Cast grouping keys to int from float32
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor(
                [0.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float32
            ),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
            "grouping_key_tensor_name": "custom_key",
            "cast_keys_to_int": True,
        },
        # Cast grouping keys to int from float64 (double)
        # This is needed because DPO (Data Pipeline Options) produces labels as float64
        {
            "labels": torch.tensor([[1, 0, 0, 1, 1]]),
            "predictions": torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]]),
            "weights": torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]]),
            "grouping_keys": torch.tensor(
                [0.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64
            ),
            "expected_ne": torch.tensor([[3.1615, 1.6004]]),
            "grouping_key_tensor_name": "custom_key",
            "cast_keys_to_int": True,
        },
    ]


class MultipleGroupingKeysTest(unittest.TestCase):
    """Test cases for multiple grouping keys feature."""

    def test_normalize_grouping_keys_config_string(self) -> None:
        """Test normalizing a single string grouping key."""
        configs = _normalize_grouping_keys_config(
            "my_key", num_groups=3, cast_keys_to_int=True
        )
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "my_key")
        self.assertEqual(configs[0].num_groups, 3)
        self.assertTrue(configs[0].cast_keys_to_int)

    def test_normalize_grouping_keys_config_list_of_strings(self) -> None:
        """Test normalizing a list of string grouping keys."""
        configs = _normalize_grouping_keys_config(
            ["key1", "key2"], num_groups=2, cast_keys_to_int=False
        )
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "key1")
        self.assertEqual(configs[0].num_groups, 2)
        self.assertFalse(configs[0].cast_keys_to_int)
        self.assertEqual(configs[1].name, "key2")
        self.assertEqual(configs[1].num_groups, 2)
        self.assertFalse(configs[1].cast_keys_to_int)

    def test_normalize_grouping_keys_config_list_of_dicts(self) -> None:
        """Test normalizing a list of dict configurations."""
        configs = _normalize_grouping_keys_config(
            [
                {"name": "key1", "num_groups": 3, "cast_keys_to_int": True},
                {"name": "key2"},  # Uses defaults
            ],
            num_groups=2,
            cast_keys_to_int=False,
        )
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "key1")
        self.assertEqual(configs[0].num_groups, 3)
        self.assertTrue(configs[0].cast_keys_to_int)
        self.assertEqual(configs[1].name, "key2")
        self.assertEqual(configs[1].num_groups, 2)  # Uses default
        self.assertFalse(configs[1].cast_keys_to_int)  # Uses default

    def test_normalize_grouping_keys_config_list_of_config_objects(self) -> None:
        """Test normalizing a list of GroupingKeyConfig objects."""
        configs = _normalize_grouping_keys_config(
            [
                GroupingKeyConfig(name="key1", num_groups=5, cast_keys_to_int=True),
                GroupingKeyConfig(name="key2", num_groups=3, cast_keys_to_int=False),
            ],
            num_groups=2,
            cast_keys_to_int=False,
        )
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0].name, "key1")
        self.assertEqual(configs[0].num_groups, 5)
        self.assertTrue(configs[0].cast_keys_to_int)
        self.assertEqual(configs[1].name, "key2")
        self.assertEqual(configs[1].num_groups, 3)
        self.assertFalse(configs[1].cast_keys_to_int)

    @no_grad()
    def test_multiple_grouping_keys_unfused(self) -> None:
        """Test SegmentedNE with multiple grouping keys in UNFUSED mode."""
        self._test_multiple_grouping_keys_helper(
            RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )

    @no_grad()
    def test_multiple_grouping_keys_fused(self) -> None:
        """Test SegmentedNE with multiple grouping keys in FUSED mode."""
        self._test_multiple_grouping_keys_helper(RecComputeMode.FUSED_TASKS_COMPUTATION)

    @no_grad()
    def test_multiple_grouping_keys_fused_and_states(self) -> None:
        """Test SegmentedNE with multiple grouping keys in FUSED_TASKS_AND_STATES mode."""
        self._test_multiple_grouping_keys_helper(
            RecComputeMode.FUSED_TASKS_AND_STATES_COMPUTATION
        )

    def _test_multiple_grouping_keys_helper(
        self,
        compute_mode: RecComputeMode,
    ) -> None:
        """Helper to test multiple grouping keys with a given compute mode."""
        # Setup test data
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])

        # Two different grouping keys with different group assignments
        grouping_key1 = torch.tensor(
            [0, 1, 0, 1, 1]
        )  # Groups: 0 -> [0, 2], 1 -> [1, 3, 4]
        grouping_key2 = torch.tensor(
            [0, 0, 1, 1, 0]
        )  # Groups: 0 -> [0, 1, 4], 1 -> [2, 3]

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "traffic_type": grouping_key1,
                "country": grouping_key2,
            },
        }

        # Create metric with multiple grouping keys using list of dicts
        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,  # Default num_groups
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=[
                {"name": "traffic_type", "num_groups": 2},
                {"name": "country", "num_groups": 2},
            ],
            compute_mode=compute_mode,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Verify we get metrics for both grouping keys
        # With multiple keys, metric names include @key_name suffix
        traffic_type_keys = [k for k in actual_ne.keys() if "@traffic_type" in k]
        country_keys = [k for k in actual_ne.keys() if "@country" in k]

        # We should have 4 metrics per grouping key (2 groups x 2 prefixes: lifetime + window)
        self.assertEqual(
            len(traffic_type_keys),
            4,
            f"Expected 4 traffic_type metrics, got {len(traffic_type_keys)}: {traffic_type_keys}",
        )
        self.assertEqual(
            len(country_keys),
            4,
            f"Expected 4 country metrics, got {len(country_keys)}: {country_keys}",
        )

        # Verify the expected metric names exist
        for group in [0, 1]:
            for prefix in ["lifetime", "window"]:
                traffic_key = (
                    f"segmented_ne-Task:0|{prefix}_segmented_ne_{group}@traffic_type"
                )
                country_key = (
                    f"segmented_ne-Task:0|{prefix}_segmented_ne_{group}@country"
                )
                self.assertIn(traffic_key, actual_ne, f"Missing metric: {traffic_key}")
                self.assertIn(country_key, actual_ne, f"Missing metric: {country_key}")

        # Verify the NE values are different for different grouping keys
        # (since they have different group assignments)
        traffic_ne_0 = actual_ne[
            "segmented_ne-Task:0|lifetime_segmented_ne_0@traffic_type"
        ]
        country_ne_0 = actual_ne["segmented_ne-Task:0|lifetime_segmented_ne_0@country"]
        # These should be different because the groupings are different
        self.assertFalse(
            torch.allclose(traffic_ne_0, country_ne_0, equal_nan=True),
            f"Expected different NE for different groupings, got traffic={traffic_ne_0}, country={country_ne_0}",
        )

    @no_grad()
    def test_multiple_grouping_keys_with_list_of_strings(self) -> None:
        """Test SegmentedNE with multiple grouping keys specified as list of strings."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])

        grouping_key1 = torch.tensor([0, 1, 0, 1, 1])
        grouping_key2 = torch.tensor([0, 0, 1, 1, 0])

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "key1": grouping_key1,
                "key2": grouping_key2,
            },
        }

        # Create metric with list of strings
        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=["key1", "key2"],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Verify metrics for both keys exist
        key1_metrics = [k for k in actual_ne.keys() if "@key1" in k]
        key2_metrics = [k for k in actual_ne.keys() if "@key2" in k]
        self.assertEqual(len(key1_metrics), 4)
        self.assertEqual(len(key2_metrics), 4)

    @no_grad()
    def test_multiple_grouping_keys_different_num_groups(self) -> None:
        """Test SegmentedNE with multiple grouping keys with different num_groups."""
        labels = torch.tensor([[1, 0, 0, 1, 1, 0]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9, 0.3]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75, 0.4]])

        # key1 has 2 groups, key2 has 3 groups
        grouping_key1 = torch.tensor([0, 1, 0, 1, 1, 0])
        grouping_key2 = torch.tensor([0, 1, 2, 0, 1, 2])

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "binary_split": grouping_key1,
                "ternary_split": grouping_key2,
            },
        }

        # Create metric with different num_groups per key
        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=6,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,  # Default
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=[
                {"name": "binary_split", "num_groups": 2},
                {"name": "ternary_split", "num_groups": 3},
            ],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Verify correct number of metrics per key
        binary_metrics = [k for k in actual_ne.keys() if "@binary_split" in k]
        ternary_metrics = [k for k in actual_ne.keys() if "@ternary_split" in k]

        # binary: 2 groups x 2 prefixes = 4 metrics
        self.assertEqual(len(binary_metrics), 4)
        # ternary: 3 groups x 2 prefixes = 6 metrics
        self.assertEqual(len(ternary_metrics), 6)

        # Verify specific group metrics exist
        for group in [0, 1]:
            self.assertIn(
                f"segmented_ne-Task:0|lifetime_segmented_ne_{group}@binary_split",
                actual_ne,
            )
        for group in [0, 1, 2]:
            self.assertIn(
                f"segmented_ne-Task:0|lifetime_segmented_ne_{group}@ternary_split",
                actual_ne,
            )

    def test_normalize_grouping_keys_config_invalid_type(self) -> None:
        """Test that invalid item types raise ValueError."""
        with self.assertRaises(ValueError):
            _normalize_grouping_keys_config(
                [123],  # pyre-ignore[6]: intentionally wrong type for test
                num_groups=2,
                cast_keys_to_int=False,
            )

    @no_grad()
    def test_single_key_backward_compat_no_prefix_no_suffix(self) -> None:
        """Test that a single default key produces the same metric names as before
        (no prefix on state names, no @suffix on description)."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])
        grouping_keys = torch.tensor([0, 1, 0, 1, 1])

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "grouping_keys": grouping_keys,
            },
        }

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys="grouping_keys",
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Metric names should NOT have @suffix (backward compat)
        for key in actual_ne.keys():
            self.assertNotIn("@", key, f"Unexpected @suffix in metric name: {key}")

        # Should have standard metric names: lifetime and window for 2 groups
        for group in [0, 1]:
            self.assertIn(
                f"segmented_ne-Task:0|lifetime_segmented_ne_{group}",
                actual_ne,
            )
            self.assertIn(
                f"segmented_ne-Task:0|window_segmented_ne_{group}",
                actual_ne,
            )

    @no_grad()
    def test_single_key_as_list_gets_prefix(self) -> None:
        """Test that passing a single non-default key as a list gets @suffix
        (not silently treated as backward-compat mode)."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])
        grouping_keys = torch.tensor([0, 1, 0, 1, 1])

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "my_custom_key": grouping_keys,
            },
        }

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=["my_custom_key"],
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Should have @my_custom_key suffix since it was passed as a list
        custom_key_metrics = [k for k in actual_ne.keys() if "@my_custom_key" in k]
        self.assertEqual(len(custom_key_metrics), 4)  # 2 groups x 2 (lifetime+window)

    @no_grad()
    def test_float64_grouping_keys_cast(self) -> None:
        """Test that float64 grouping keys can be cast to int64."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])
        grouping_keys = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "grouping_keys": grouping_keys,
            },
        }

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys="grouping_keys",
            # pyrefly: ignore[bad-argument-type]
            cast_keys_to_int=True,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        # Should not raise - float64 is now accepted with cast_keys_to_int=True
        ne.update(**inputs)
        actual_ne = ne.compute()

        # Verify we get valid NE values (same as float32 cast test)
        for group in [0, 1]:
            key = f"segmented_ne-Task:0|lifetime_segmented_ne_{group}"
            self.assertIn(key, actual_ne)
            self.assertFalse(torch.isnan(actual_ne[key]).all())

    @no_grad()
    def test_float64_without_cast_raises(self) -> None:
        """Test that float64 grouping keys without cast_keys_to_int raises an error."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])
        grouping_keys = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "grouping_keys": grouping_keys,
            },
        }

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys="grouping_keys",
            # pyrefly: ignore[bad-argument-type]
            cast_keys_to_int=False,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        with self.assertRaises(RecMetricException):
            ne.update(**inputs)

    @no_grad()
    def test_multiple_grouping_keys_with_logloss(self) -> None:
        """Test SegmentedNE with multiple grouping keys and include_logloss=True."""
        labels = torch.tensor([[1, 0, 0, 1, 1]])
        predictions = torch.tensor([[0.2, 0.6, 0.8, 0.4, 0.9]])
        weights = torch.tensor([[0.13, 0.2, 0.5, 0.8, 0.75]])

        grouping_key1 = torch.tensor([0, 1, 0, 1, 1])
        grouping_key2 = torch.tensor([0, 0, 1, 1, 0])

        task_list = [
            RecTaskInfo(
                name="Task:0",
                label_name="label",
                prediction_name="prediction",
                weight_name="weight",
            )
        ]

        inputs: Dict[str, Any] = {
            "predictions": {"Task:0": predictions[0]},
            "labels": {"Task:0": labels[0]},
            "weights": {"Task:0": weights[0]},
            "required_inputs": {
                "traffic_type": grouping_key1,
                "country": grouping_key2,
            },
        }

        ne = SegmentedNEMetric(
            world_size=1,
            my_rank=0,
            batch_size=5,
            tasks=task_list,
            # pyrefly: ignore[bad-argument-type]
            num_groups=2,
            # pyrefly: ignore[bad-argument-type]
            grouping_keys=[
                {"name": "traffic_type", "num_groups": 2},
                {"name": "country", "num_groups": 2},
            ],
            # pyrefly: ignore[bad-argument-type]
            include_logloss=True,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )

        ne.update(**inputs)
        actual_ne = ne.compute()

        # Should have both NE and logloss metrics for each key
        for key_name in ["traffic_type", "country"]:
            for group in [0, 1]:
                for prefix in ["lifetime", "window"]:
                    ne_key = (
                        f"segmented_ne-Task:0|{prefix}_segmented_ne_{group}@{key_name}"
                    )
                    logloss_key = (
                        f"segmented_ne-Task:0|{prefix}_logloss_{group}@{key_name}"
                    )
                    self.assertIn(ne_key, actual_ne, f"Missing NE metric: {ne_key}")
                    self.assertIn(
                        logloss_key, actual_ne, f"Missing logloss metric: {logloss_key}"
                    )
