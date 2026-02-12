#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import Mock

import torch
from torchrec.optim.keyed import KeyedOptimizer
from torchrec.optim.semi_sync import SemisyncOptimizer


class TestSemisyncOptimizer(unittest.TestCase):
    """Test cases for SemisyncOptimizer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

        # Create test parameters with same shape to avoid tensor comparison issues
        self.dense_param = torch.nn.Parameter(torch.randn(5, 5))
        self.dense_param2 = torch.nn.Parameter(torch.randn(5, 5))
        self.global_params = [self.dense_param, self.dense_param2]

        # Create mock local optimizer
        self.mock_local_optimizer = Mock(spec=KeyedOptimizer)
        self.mock_local_optimizer.params = {
            "dense": self.dense_param,
            "dense2": self.dense_param2,
        }
        self.mock_local_optimizer.param_groups = [
            {"params": self.global_params, "lr": 0.01}
        ]
        self.mock_local_optimizer.state = {}

        # Create mock global optimizer
        self.mock_global_optimizer = Mock(spec=KeyedOptimizer)
        self.mock_global_optimizer.params = {
            "dense": self.dense_param,
            "dense2": self.dense_param2,
        }
        self.mock_global_optimizer.param_groups = [
            {"params": self.global_params, "lr": 0.1}
        ]
        self.mock_global_optimizer.state = {}

        # Mock the _optimizer attribute for global step calls
        self.mock_global_optimizer._optimizer = Mock()
        self.mock_global_optimizer._optimizer.global_step = Mock()

        self.optimizer = SemisyncOptimizer(
            global_params=self.global_params,
            optimizer=self.mock_local_optimizer,
            global_optimizer=self.mock_global_optimizer,
        )

    def test_initialization_with_mocks(self) -> None:
        """Test SemisyncOptimizer initialization with mock optimizers."""

        optimizer = SemisyncOptimizer(
            global_params=self.global_params,
            optimizer=self.mock_local_optimizer,
            global_optimizer=self.mock_global_optimizer,
            num_local_steps=12,
        )

        # Check that optimizers are stored
        self.assertEqual(optimizer._optimizer, self.mock_local_optimizer)
        self.assertEqual(optimizer._global_optimizer, self.mock_global_optimizer)
        self.assertEqual(optimizer._num_local_steps, 12)

        # Check step counters are initialized
        self.assertEqual(optimizer._local_step_counter.item(), 0)
        self.assertEqual(optimizer._global_step_counter.item(), 0)

    def test_param_groups_property(self) -> None:
        """Test that param_groups property combines both optimizers."""
        optimizer = self.optimizer
        param_groups = optimizer.param_groups

        # Should have exactly 2 param groups from both optimizers
        self.assertEqual(len(param_groups), 2)

        # Check both optimizer param groups
        expected_groups = [
            {"index": 0, "lr": 0.01, "name": "local"},
            {"index": 1, "lr": 0.1, "name": "global"},
        ]
        param_groups_list = list(param_groups)
        for expected in expected_groups:
            with self.subTest(optimizer=expected["name"]):
                index = expected["index"]
                group = param_groups_list[index]
                self.assertIn("params", group)
                self.assertEqual(group["lr"], expected["lr"])

                # Check for parameter identity (not equality) to avoid tensor comparison issues
                param_ids = {id(p) for p in group["params"]}
                self.assertIn(id(self.dense_param), param_ids)
                self.assertIn(id(self.dense_param2), param_ids)

    def test_params_property(self) -> None:
        """Test that params property combines both optimizers with exact structure."""
        optimizer = self.optimizer
        params = optimizer.params

        # Check exact parameter keys - should contain local, global, and step counter params
        expected_keys = {
            "dense",
            "dense2",
            "__semisync_global_step_counter__",
            "__semisync_local_step_counter__",
        }
        actual_keys = set(params.keys())
        self.assertEqual(actual_keys, expected_keys)

        # Verify parameter identities from local optimizer
        self.assertTrue(params["dense"] is self.dense_param)
        self.assertTrue(params["dense2"] is self.dense_param2)

        # Verify step counters are tensors with correct properties
        global_counter = params["__semisync_global_step_counter__"]
        local_counter = params["__semisync_local_step_counter__"]

        # Test both counters have identical properties
        for counter_name, counter in [
            ("global", global_counter),
            ("local", local_counter),
        ]:
            with self.subTest(counter=counter_name):
                self.assertIsInstance(counter, torch.Tensor)
                self.assertEqual(counter.item(), 0)
                self.assertEqual(counter.dtype, torch.int64)
                self.assertEqual(counter.device.type, "cpu")

        # Verify step counters are different tensor objects
        self.assertIsNot(global_counter, local_counter)

        # Verify the params dict has exactly the expected number of items
        self.assertEqual(len(params), 4)

    def test_state_property(self) -> None:
        """Test state property with exact structure verification."""
        # Set up precise mock states
        local_momentum = torch.tensor(1.5)
        global_momentum = torch.tensor(2.5)
        self.mock_local_optimizer.state = {
            self.dense_param: {"momentum": local_momentum}
        }
        self.mock_global_optimizer.state = {
            self.dense_param: {"momentum": global_momentum}
        }

        optimizer = self.optimizer

        state = optimizer.state
        param_state = state[self.dense_param]

        # Verify exact state structure
        expected_keys = {
            "semi_sync_local_momentum",
            "semi_sync_global_momentum",
            "__semisync_global_step_counter__",
            "__semisync_local_step_counter__",
        }
        self.assertEqual(set(param_state.keys()), expected_keys)

        # Verify exact tensor values
        self.assertTrue(
            torch.equal(param_state["semi_sync_local_momentum"], local_momentum)
        )
        self.assertTrue(
            torch.equal(param_state["semi_sync_global_momentum"], global_momentum)
        )

        # Verify step counters
        self.assertEqual(param_state["__semisync_global_step_counter__"].item(), 0)
        self.assertEqual(param_state["__semisync_local_step_counter__"].item(), 0)

    def test_step_delegation(self) -> None:
        """Test that step() delegates to local optimizer."""
        optimizer = SemisyncOptimizer(
            global_params=self.global_params,
            optimizer=self.mock_local_optimizer,
            global_optimizer=self.mock_global_optimizer,
            num_local_steps=4,
        )

        # Call step multiple times
        for i in range(3):
            optimizer.step()

            # Local optimizer should be called each time
            self.assertEqual(self.mock_local_optimizer.step.call_count, i + 1)

            # Global optimizer should not be called yet
            self.mock_global_optimizer._optimizer.global_step.assert_not_called()

            # Local step counter should increment
            self.assertEqual(optimizer._local_step_counter.item(), i + 1)

    def test_global_step_triggering(self) -> None:
        """Test that global step is triggered after num_local_steps."""
        optimizer = SemisyncOptimizer(
            global_params=self.global_params,
            optimizer=self.mock_local_optimizer,
            global_optimizer=self.mock_global_optimizer,
            num_local_steps=2,  # Trigger global step every 2 local steps
        )

        # First step - no global step
        optimizer.step()
        self.mock_global_optimizer._optimizer.global_step.assert_not_called()
        self.assertEqual(optimizer._global_step_counter.item(), 0)

        # Second step - should trigger global step
        optimizer.step()
        self.mock_global_optimizer._optimizer.global_step.assert_called_once()
        self.assertEqual(optimizer._global_step_counter.item(), 1)

        # Third step - no global step
        self.mock_global_optimizer._optimizer.global_step.reset_mock()
        optimizer.step()
        self.mock_global_optimizer._optimizer.global_step.assert_not_called()

        # Fourth step - should trigger global step again
        optimizer.step()
        self.mock_global_optimizer._optimizer.global_step.assert_called_once()
        self.assertEqual(optimizer._global_step_counter.item(), 2)

    def test_zero_grad_delegation(self) -> None:
        """Test zero_grad() with exact parameter verification."""
        optimizer = self.optimizer

        # Test both set_to_none values
        for set_to_none in [False, True]:
            with self.subTest(set_to_none=set_to_none):
                self.mock_local_optimizer.zero_grad.reset_mock()
                self.mock_global_optimizer.zero_grad.reset_mock()

                optimizer.zero_grad(set_to_none=set_to_none)

                self.mock_local_optimizer.zero_grad.assert_called_once_with(
                    set_to_none=set_to_none
                )
                self.mock_global_optimizer.zero_grad.assert_called_once_with(
                    set_to_none=set_to_none
                )

    def test_save_param_groups_delegation(self) -> None:
        """Test save_param_groups() with exact state verification."""
        optimizer = self.optimizer

        # Test with both True and False values
        for save_value in [True, False]:
            with self.subTest(save_value=save_value):
                self.mock_local_optimizer.save_param_groups.reset_mock()
                self.mock_global_optimizer.save_param_groups.reset_mock()

                optimizer.save_param_groups(save_value)

                self.mock_local_optimizer.save_param_groups.assert_called_once_with(
                    save_value
                )
                self.mock_global_optimizer.save_param_groups.assert_called_once_with(
                    save_value
                )
                self.assertEqual(optimizer.defaults["_save_param_groups"], save_value)

    def test_repr_string(self) -> None:
        """Test repr() returns exact expected format."""
        optimizer = self.optimizer

        repr_str = repr(optimizer)

        # Verify exact structure and content
        self.assertIsInstance(repr_str, str)

        # Check for required components in string representation
        required_components = ["semi_sync_local_optim", "semi_sync_global_optim"]
        for component in required_components:
            with self.subTest(component=component):
                self.assertTrue(
                    component in repr_str, f"Missing component: {component}"
                )

    def test_extract_prefixed_state_static_method(self) -> None:
        """Test _extract_prefixed_state static method."""
        param_state = {
            "semi_sync_local_momentum": torch.tensor(1.5),
            "semi_sync_local_step": torch.tensor(10),
            "other_key": torch.tensor(2.7),
            "semi_sync_global_momentum": torch.tensor(3.2),
        }

        test_cases = [
            {"prefix": "semi_sync_local_", "expected_keys": {"momentum", "step"}},
            {"prefix": "semi_sync_global_", "expected_keys": {"momentum"}},
            {"prefix": "nonexistent_", "expected_keys": set()},
        ]

        for case in test_cases:
            with self.subTest(prefix=case["prefix"]):
                result = SemisyncOptimizer._extract_prefixed_state(
                    param_state, case["prefix"]
                )
                self.assertEqual(set(result.keys()), case["expected_keys"])

                # Verify tensor values for non-empty results
                if case["expected_keys"]:
                    for key in case["expected_keys"]:
                        original_key = case["prefix"] + key
                        self.assertTrue(
                            torch.equal(result[key], param_state[original_key])
                        )


class TestSemisyncOptimizerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for SemisyncOptimizer."""

    def test_empty_params_error(self) -> None:
        """Test SemisyncOptimizer with empty parameters raises IndexError."""
        mock_local = Mock(spec=KeyedOptimizer)
        mock_local.params = {}
        mock_local.param_groups = []
        mock_local.state = {}

        mock_global = Mock(spec=KeyedOptimizer)
        mock_global.params = {}
        mock_global.param_groups = []
        mock_global.state = {}
        mock_global._optimizer = Mock()
        mock_global._optimizer.global_step = Mock()

        with self.assertRaises(IndexError) as cm:
            SemisyncOptimizer(
                global_params=[],
                optimizer=mock_local,
                global_optimizer=mock_global,
            )
        self.assertIsInstance(cm.exception, IndexError)

    def test_num_local_steps_configurations(self) -> None:
        """Test num_local_steps trigger behavior."""
        param = torch.nn.Parameter(torch.randn(3, 3))
        mock_local = Mock(spec=KeyedOptimizer)
        mock_local.params = {"param": param}
        mock_local.param_groups = [{"params": [param]}]
        mock_local.state = {}

        mock_global = Mock(spec=KeyedOptimizer)
        mock_global.params = {"param": param}
        mock_global.param_groups = [{"params": [param]}]
        mock_global.state = {}
        mock_global._optimizer = Mock()
        mock_global._optimizer.global_step = Mock()

        for num_steps in [1, 3, 5]:
            with self.subTest(num_local_steps=num_steps):
                optimizer = SemisyncOptimizer(
                    global_params=[param],
                    optimizer=mock_local,
                    global_optimizer=mock_global,
                    num_local_steps=num_steps,
                )

                mock_global._optimizer.global_step.reset_mock()

                # Test exact trigger: should call global step after num_steps local steps
                for _ in range(num_steps):
                    optimizer.step()
                self.assertEqual(mock_global._optimizer.global_step.call_count, 1)

    def test_step_counter_exact_values(self) -> None:
        """Test step counter progression."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_local = Mock(spec=KeyedOptimizer)
        mock_local.params = {"param": param}
        mock_local.param_groups = [{"params": [param]}]
        mock_local.state = {}

        mock_global = Mock(spec=KeyedOptimizer)
        mock_global.params = {"param": param}
        mock_global.param_groups = [{"params": [param]}]
        mock_global.state = {}
        mock_global._optimizer = Mock()
        mock_global._optimizer.global_step = Mock()

        optimizer = SemisyncOptimizer(
            global_params=[param],
            optimizer=mock_local,
            global_optimizer=mock_global,
            num_local_steps=3,
        )

        # Test progression: global step triggers every 3 local steps
        test_progression = [(1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 2)]

        for expected_local, expected_global in test_progression:
            optimizer.step()
            self.assertEqual(optimizer._local_step_counter.item(), expected_local)
            self.assertEqual(optimizer._global_step_counter.item(), expected_global)

    def test_step_counter_tensor_properties(self) -> None:
        """Test step counter tensor properties."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        mock_local = Mock(spec=KeyedOptimizer)
        mock_local.params = {"param": param}
        mock_local.param_groups = [{"params": [param]}]
        mock_local.state = {}

        mock_global = Mock(spec=KeyedOptimizer)
        mock_global.params = {"param": param}
        mock_global.param_groups = [{"params": [param]}]
        mock_global.state = {}
        mock_global._optimizer = Mock()

        optimizer = SemisyncOptimizer(
            global_params=[param],
            optimizer=mock_local,
            global_optimizer=mock_global,
        )

        # Test both counters have identical properties
        counters = [optimizer._local_step_counter, optimizer._global_step_counter]

        for counter in counters:
            self.assertIsInstance(counter, torch.Tensor)
            self.assertEqual(counter.dtype, torch.int64)
            self.assertEqual(counter.device.type, "cpu")
            self.assertEqual(counter.numel(), 1)
            self.assertEqual(counter.dim(), 0)
            self.assertEqual(counter.item(), 0)

        # Verify they are different objects
        self.assertIsNot(counters[0], counters[1])
