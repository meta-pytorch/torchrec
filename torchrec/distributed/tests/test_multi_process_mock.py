#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessMock,
    MultiProcessTestBase,
)


# Test target functions that will be mocked
def expensive_operation() -> str:
    """Simulates an expensive operation that we want to mock in tests."""
    return "real_value"


def another_operation(value: int) -> int:
    """Another function to mock for testing multiple mocks."""
    return value * 2


# Side effect functions for testing (defined at module level for pickling)
def mock_side_effect_for_test() -> str:
    """Side effect function that can be pickled for multiprocessing."""
    return "side_effect_value"


class MultiProcessMockTest(unittest.TestCase):
    """Test suite for MultiProcessMock class."""

    def test_add_mock_stores_configuration(self) -> None:
        """Test that add_mock correctly stores mock configuration."""
        # Setup: Create a MultiProcessMock instance
        mock_manager = MultiProcessMock()

        # Execute: Add a mock
        mock_manager.add_mock(
            target="test_module.test_function",
            return_value="mocked",
        )

        # Assert: Verify the mock was stored
        self.assertEqual(len(mock_manager.mocks), 1)
        self.assertEqual(mock_manager.mocks[0]["target"], "test_module.test_function")
        self.assertEqual(mock_manager.mocks[0]["return_value"], "mocked")

    def test_add_multiple_mocks(self) -> None:
        """Test that multiple mocks can be added."""
        # Setup: Create a MultiProcessMock instance
        mock_manager = MultiProcessMock()

        # Execute: Add multiple mocks
        mock_manager.add_mock(target="module1.func1", return_value="mock1")
        mock_manager.add_mock(target="module2.func2", return_value="mock2")

        # Assert: Verify all mocks were stored
        self.assertEqual(len(mock_manager.mocks), 2)
        self.assertEqual(mock_manager.mocks[0]["return_value"], "mock1")
        self.assertEqual(mock_manager.mocks[1]["return_value"], "mock2")

    def test_add_mock_with_side_effect(self) -> None:
        """Test that add_mock correctly stores side_effect configuration."""
        # Setup: Create a MultiProcessMock instance and a side effect function
        mock_manager = MultiProcessMock()

        def side_effect_func() -> str:
            return "side_effect_value"

        # Execute: Add a mock with side effect
        mock_manager.add_mock(
            target="test_module.test_function",
            side_effect=side_effect_func,
        )

        # Assert: Verify the side effect was stored
        self.assertEqual(len(mock_manager.mocks), 1)
        self.assertEqual(mock_manager.mocks[0]["side_effect"], side_effect_func)

    def test_clear_mocks(self) -> None:
        """Test that clear_mocks removes all registered mocks."""
        # Setup: Create a MultiProcessMock instance and add mocks
        mock_manager = MultiProcessMock()
        mock_manager.add_mock(target="module1.func1", return_value="mock1")
        mock_manager.add_mock(target="module2.func2", return_value="mock2")

        # Execute: Clear all mocks
        mock_manager.clear_mocks()

        # Assert: Verify all mocks were removed
        self.assertEqual(len(mock_manager.mocks), 0)

    def test_apply_mocks_returns_patches(self) -> None:
        """Test that apply_mocks returns active patch objects."""
        # Setup: Create a MultiProcessMock instance and add a mock
        mock_manager = MultiProcessMock()
        mock_manager.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            return_value="mocked_value",
        )

        # Execute: Apply mocks
        active_patches = mock_manager.apply_mocks()

        try:
            # Assert: Verify patches were created and function is mocked
            self.assertEqual(len(active_patches), 1)
            self.assertEqual(expensive_operation(), "mocked_value")
        finally:
            # Cleanup: Exit all patches
            for patcher, _ in active_patches:
                patcher.__exit__(None, None, None)

    def test_apply_mocks_with_side_effect(self) -> None:
        """Test that apply_mocks correctly handles side_effect."""
        # Setup: Create a MultiProcessMock instance and add a mock with side effect
        mock_manager = MultiProcessMock()

        def side_effect_func() -> str:
            return "side_effect_result"

        mock_manager.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            side_effect=side_effect_func,
        )

        # Execute: Apply mocks
        active_patches = mock_manager.apply_mocks()

        try:
            # Assert: Verify side effect is applied
            self.assertEqual(expensive_operation(), "side_effect_result")
        finally:
            # Cleanup: Exit all patches
            for patcher, _ in active_patches:
                patcher.__exit__(None, None, None)


def test_function_without_mock(rank: int, world_size: int, backend: str) -> None:
    """
    Baseline test function that doesn't use any mocks.
    Used to verify backward compatibility - ensures MultiProcessMock
    doesn't impact existing tests that don't set up mocks.
    """
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend):
        # Call the expensive operation WITHOUT any mocking
        result = expensive_operation()

        # Verify the original function behavior is preserved
        assert result == "real_value", f"Expected 'real_value', got '{result}'"


def test_function_with_mock(rank: int, world_size: int, backend: str) -> None:
    """
    Test function that uses a mocked expensive operation.
    Used by integration tests to verify cross-process mocking.
    """
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend):
        # Call the expensive operation (which should be mocked)
        result = expensive_operation()

        # Verify the mock was applied
        assert result == "mocked_value", f"Expected 'mocked_value', got '{result}'"


def test_function_with_multiple_mocks(rank: int, world_size: int, backend: str) -> None:
    """Test function that uses multiple mocked operations."""
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend):
        # Call both mocked functions
        result1 = expensive_operation()
        result2 = another_operation(10)

        # Verify both mocks were applied
        assert result1 == "first_mock", f"Expected 'first_mock', got '{result1}'"
        assert result2 == 100, f"Expected 100, got '{result2}'"


def test_function_with_side_effect(rank: int, world_size: int, backend: str) -> None:
    """Test function that uses a mocked operation with side effect."""
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend):
        # Call the mocked function
        result = expensive_operation()

        # Verify the side effect was applied
        assert (
            result == "side_effect_value"
        ), f"Expected 'side_effect_value', got '{result}'"


class MultiProcessTestBaseIntegrationTest(MultiProcessTestBase):
    """Integration tests for MultiProcessTestBase with mocking functionality."""

    def test_baseline_without_mocks(self) -> None:
        """
        Baseline test that verifies MultiProcessMock doesn't impact existing tests.

        This test ensures backward compatibility by running a multi-process test
        WITHOUT setting up any mocks, confirming that the original function
        behavior is preserved.
        """
        # Execute: Run the test function across multiple processes WITHOUT adding mocks
        # This verifies that the mock infrastructure doesn't interfere with normal operation
        self._run_multi_process_test(
            callable=test_function_without_mock,
            world_size=2,
            backend="gloo",
        )

        # Assert: If we reach here, all processes executed successfully
        # with the original (unmocked) function behavior

    def test_baseline_without_mocks_per_rank(self) -> None:
        """
        Baseline test for _run_multi_process_test_per_rank without mocks.

        This test ensures backward compatibility for the per-rank variant by
        running without any mocks, confirming original function behavior.
        """
        # Setup: Define per-rank kwargs without any mocks
        kwargs_per_rank = [
            {"backend": "gloo"},
            {"backend": "gloo"},
        ]

        # Execute: Run the test with per-rank configuration WITHOUT adding mocks
        self._run_multi_process_test_per_rank(
            callable=test_function_without_mock,
            world_size=2,
            kwargs_per_rank=kwargs_per_rank,
        )

        # Assert: If we reach here, all processes executed successfully
        # with the original (unmocked) function behavior

    def test_cross_process_mock_with_return_value(self) -> None:
        """Test that mocks are applied across multiple processes with return_value."""
        # Setup: Add a mock that will be applied in all child processes
        self.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            return_value="mocked_value",
        )

        # Execute: Run the test function across multiple processes
        # The mock will be automatically applied in each process
        self._run_multi_process_test(
            callable=test_function_with_mock,
            world_size=2,
            backend="gloo",
        )

        # Assert: If we reach here, all processes verified the mock successfully

    def test_cross_process_multiple_mocks(self) -> None:
        """Test that multiple mocks are applied across processes."""
        # Setup: Add multiple mocks
        self.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            return_value="first_mock",
        )
        self.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.another_operation",
            return_value=100,
        )

        # Execute: Run the test function across multiple processes
        self._run_multi_process_test(
            callable=test_function_with_multiple_mocks,
            world_size=2,
            backend="gloo",
        )

        # Assert: If we reach here, all processes verified both mocks successfully

    def test_cross_process_mock_with_side_effect(self) -> None:
        """Test that mocks with side_effect are applied across processes."""

        # Setup: Add a mock with side effect using module-level function
        self.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            side_effect=mock_side_effect_for_test,
        )

        # Execute: Run the test function across multiple processes
        self._run_multi_process_test(
            callable=test_function_with_side_effect,
            world_size=2,
            backend="gloo",
        )

        # Assert: If we reach here, all processes verified the side effect successfully

    def test_run_multi_process_test_per_rank_with_mocks(self) -> None:
        """Test that mocks work with _run_multi_process_test_per_rank."""

        # Setup: Add a mock
        self.add_mock(
            target="torchrec.distributed.tests.test_multi_process_mock.expensive_operation",
            return_value="mocked_value",
        )

        # Setup: Define per-rank kwargs
        kwargs_per_rank = [
            {"backend": "gloo"},
            {"backend": "gloo"},
        ]

        # Execute: Run the test with per-rank configuration
        self._run_multi_process_test_per_rank(
            callable=test_function_with_mock,
            world_size=2,
            kwargs_per_rank=kwargs_per_rank,
        )

        # Assert: If we reach here, all processes verified the mock successfully
