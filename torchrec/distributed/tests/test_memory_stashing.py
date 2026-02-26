#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import torch
from torch import nn
from torchrec.distributed.memory_stashing import MemoryStashingManager


class TestStashTensors(unittest.TestCase):
    """Tests for MemoryStashingManager._stash_tensors."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def test_basic_stash_and_restore(self) -> None:
        """Test basic stash and restore with a single tensor."""
        tensor = torch.randn(100, 64, device=self.device)
        original = tensor.clone()

        await_restore, restore = MemoryStashingManager._stash_tensors([tensor])

        # Verify HBM is freed
        self.assertEqual(tensor.untyped_storage().size(), 0)

        # Restore
        restore(None)
        await_restore(None)

        # Verify values are correct
        self.assertGreater(tensor.untyped_storage().size(), 0)
        self.assertTrue(torch.allclose(tensor, original))

    def test_multiple_tensors(self) -> None:
        """Test stash and restore with multiple tensors."""
        t1 = torch.randn(50, 32, device=self.device)
        t2 = torch.ones(80, 64, device=self.device) * 2
        originals = [t1.clone(), t2.clone()]

        await_restore, restore = MemoryStashingManager._stash_tensors([t1, t2])

        # All freed
        self.assertEqual(t1.untyped_storage().size(), 0)
        self.assertEqual(t2.untyped_storage().size(), 0)

        # Restore
        restore(None)
        await_restore(None)

        # All restored correctly
        self.assertTrue(torch.allclose(t1, originals[0]))
        self.assertTrue(torch.allclose(t2, originals[1]))

    def test_empty_list(self) -> None:
        """Test that an empty tensor list returns no-op callbacks."""
        await_restore, restore = MemoryStashingManager._stash_tensors([])
        # Should not raise
        restore(None)
        await_restore(None)

    def test_preserves_autograd_version(self) -> None:
        """Test that restore does not increment the tensor version counter."""
        tensor = torch.randn(10, 5, device=self.device, requires_grad=True)
        version_before = tensor._version

        await_restore, restore = MemoryStashingManager._stash_tensors([tensor])
        restore(None)
        await_restore(None)

        self.assertEqual(tensor._version, version_before)

    def test_callbacks_accept_grad_argument(self) -> None:
        """Test that callbacks work as backward hooks (accept a grad tensor)."""
        tensor = torch.randn(10, 5, device=self.device)
        original = tensor.clone()

        await_restore, restore = MemoryStashingManager._stash_tensors([tensor])

        dummy_grad = torch.tensor([1.0])
        restore(dummy_grad)
        await_restore(dummy_grad)

        self.assertTrue(torch.allclose(tensor, original))


class TestStashEmbeddingWeights(unittest.TestCase):
    """Tests for stash_embedding_weights function."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def _create_mock_lookup(self, weights_list: List[torch.Tensor]) -> Mock:
        """Helper to create a mock lookup with multiple embedding modules."""
        emb_modules = []
        for weights in weights_list:
            inner = Mock()
            inner.weights_dev = weights
            emb_module = Mock()
            emb_module._emb_module = inner
            emb_modules.append(emb_module)

        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = emb_modules
        return lookup

    def test_basic_stash_and_restore(self) -> None:
        """Test basic stash and restore functionality with the two-callback API."""
        original_weights = torch.ones((100, 64), device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Verify HBM is freed
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        # Restore weights
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify HBM is restored and values are correct
        self.assertGreater(original_weights.untyped_storage().size(), 0)
        self.assertTrue(torch.allclose(original_weights, original_values))

    def test_multiple_emb_modules_stashed(self) -> None:
        """Test that multiple embedding modules are all stashed and restored."""
        weights_1 = torch.ones((50, 32), device=self.device)
        weights_2 = torch.ones((80, 64), device=self.device) * 2
        weights_3 = torch.ones((100, 128), device=self.device) * 3

        original_values_1 = weights_1.clone()
        original_values_2 = weights_2.clone()
        original_values_3 = weights_3.clone()

        lookup = self._create_mock_lookup([weights_1, weights_2, weights_3])

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Verify all are stashed
        self.assertEqual(weights_1.untyped_storage().size(), 0)
        self.assertEqual(weights_2.untyped_storage().size(), 0)
        self.assertEqual(weights_3.untyped_storage().size(), 0)

        # Restore all
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify all are restored correctly
        self.assertTrue(torch.allclose(weights_1, original_values_1))
        self.assertTrue(torch.allclose(weights_2, original_values_2))
        self.assertTrue(torch.allclose(weights_3, original_values_3))

    def test_custom_d2h_stream(self) -> None:
        """Test stash and restore with custom D2H CUDA stream."""
        custom_stream = torch.cuda.Stream(device=self.device)
        MemoryStashingManager.set_streams(
            host_to_device_stream=MemoryStashingManager.h2d_stream(),
            device_to_host_stream=custom_stream,
        )

        original_weights = torch.randn(50, 32, device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Verify stash worked
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify restoration
        self.assertTrue(torch.allclose(original_weights, original_values))

    def test_restore_does_not_break_autograd(self) -> None:
        """Test that restore doesn't break autograd for backward pass."""
        weights = torch.randn(10, 5, device=self.device, requires_grad=True)
        initial_version = weights._version

        lookup = self._create_mock_lookup([weights])

        # Forward pass
        x = torch.randn(3, 5, device=self.device)
        output = torch.matmul(x, weights.t())

        # Stash and restore
        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Version should not have changed
        self.assertEqual(weights._version, initial_version)

        # Backward should work without errors
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(weights.grad)
        self.assertGreater(weights.grad.abs().sum().item(), 0)

    def test_skip_non_cuda_weights(self) -> None:
        """Test that non-CUDA weights are skipped."""
        cuda_weights = torch.randn(50, 32, device=self.device)
        cpu_weights = torch.randn(50, 32, device="cpu")

        cuda_original = cuda_weights.clone()

        # Create mock with both CUDA and CPU weights
        emb_modules = []

        inner_cuda = Mock()
        inner_cuda.weights_dev = cuda_weights
        emb_cuda = Mock()
        emb_cuda._emb_module = inner_cuda
        emb_modules.append(emb_cuda)

        inner_cpu = Mock()
        inner_cpu.weights_dev = cpu_weights
        emb_cpu = Mock()
        emb_cpu._emb_module = inner_cpu
        emb_modules.append(emb_cpu)

        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = emb_modules

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Only CUDA weights should be stashed
        self.assertEqual(cuda_weights.untyped_storage().size(), 0)
        self.assertGreater(cpu_weights.untyped_storage().size(), 0)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        self.assertTrue(torch.allclose(cuda_weights, cuda_original))

    def test_skip_none_weights(self) -> None:
        """Test that None weights are handled gracefully."""
        valid_weights = torch.randn(50, 32, device=self.device)
        valid_original = valid_weights.clone()

        emb_modules = []

        # Module with valid weights
        inner_valid = Mock()
        inner_valid.weights_dev = valid_weights
        emb_valid = Mock()
        emb_valid._emb_module = inner_valid
        emb_modules.append(emb_valid)

        # Module with None weights
        inner_none = Mock()
        inner_none.weights_dev = None
        emb_none = Mock()
        emb_none._emb_module = inner_none
        emb_modules.append(emb_none)

        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = emb_modules

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Valid weights should be stashed
        self.assertEqual(valid_weights.untyped_storage().size(), 0)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        self.assertTrue(torch.allclose(valid_weights, valid_original))

    def test_callback_signature_compatibility_with_register_hook(self) -> None:
        """Test that await_restore can be used as backward hook."""
        weights = torch.randn(10, 5, device=self.device, requires_grad=True)
        original_values = weights.clone()

        lookup = self._create_mock_lookup([weights])

        # Create a tensor that we'll register hooks on
        x = torch.randn(3, 5, device=self.device, requires_grad=True)
        output = torch.matmul(x, weights.t())

        await_restore, _restore = MemoryStashingManager.stash_embedding_weights(lookup)

        # Register restore via class method and await_restore as backward hook
        output.register_hook(
            lambda _grad: MemoryStashingManager.restore_embedding_weights()
        )
        output.register_hook(await_restore)

        # Backward pass should trigger the hooks
        loss = output.sum()
        loss.backward()

        # Weights should be restored after backward
        self.assertGreater(weights.untyped_storage().size(), 0)
        self.assertTrue(torch.allclose(weights, original_values))


class TestStashOptimizerState(unittest.TestCase):
    """Tests for MemoryStashingManager.stash_optimizer_state method."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))
        # Use a large tensor size to exceed the 1MB threshold
        self.large_size = (512, 512)  # 512*512*4 = 1MB for float32

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def test_basic_adam_optimizer_stash_and_restore(self) -> None:
        """Test basic stash and restore with Adam optimizer."""
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run a step to populate optimizer state
        x = torch.randn(32, 512, device=self.device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Get original state values
        original_states: Dict[Any, Dict[str, torch.Tensor]] = {}
        for param, state in optimizer.state.items():
            if isinstance(state, dict):
                original_states[param] = {
                    k: v.clone()
                    for k, v in state.items()
                    if isinstance(v, torch.Tensor)
                }

        # Stash optimizer state
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)

        # Verify state tensors are freed (storage size should be 0)
        for _param, state in optimizer.state.items():
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        # Only large tensors (>= 1MB) should be stashed
                        tensor_size = value.numel() * value.element_size()
                        if tensor_size >= 1024 * 1024:
                            self.assertEqual(
                                value.untyped_storage().size(),
                                0,
                                f"Tensor {key} should be stashed",
                            )

        # Restore
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)

        # Verify restored values match original
        for param, state in optimizer.state.items():
            if param in original_states and isinstance(state, dict):
                for key, value in state.items():
                    if key in original_states[param]:
                        self.assertTrue(
                            torch.allclose(value, original_states[param][key]),
                            f"State {key} not restored correctly",
                        )

    def test_sgd_with_momentum_stash_and_restore(self) -> None:
        """Test stash and restore with SGD optimizer with momentum."""
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Run a step to populate momentum buffers
        x = torch.randn(32, 512, device=self.device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Get original momentum buffer values
        original_momentum: Dict[Any, torch.Tensor] = {}
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and "momentum_buffer" in state:
                original_momentum[param] = state["momentum_buffer"].clone()

        # Stash optimizer state
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)

        # Restore
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)

        # Verify momentum buffers are restored correctly
        for param, state in optimizer.state.items():
            if param in original_momentum and isinstance(state, dict):
                self.assertTrue(
                    torch.allclose(state["momentum_buffer"], original_momentum[param]),
                    "Momentum buffer not restored correctly",
                )

    def test_optimizer_step_works_after_restore(self) -> None:
        """Test that optimizer.step() works correctly after restore."""
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Initial training step
        x = torch.randn(32, 512, device=self.device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store weights before stash
        weights_before = model.weight.clone()

        # Stash, restore
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)

        # Another training step after restore
        x = torch.randn(32, 512, device=self.device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed (optimizer step worked)
        self.assertFalse(
            torch.allclose(model.weight, weights_before),
            "Weights should change after optimizer step",
        )

    def test_skip_small_tensors(self) -> None:
        """Test that small tensors (< 1MB) are not stashed."""
        # Create a small model with small optimizer state
        model = nn.Linear(10, 10).to(self.device)  # Very small
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run a step to populate optimizer state
        x = torch.randn(5, 10, device=self.device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Stash optimizer state
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)

        # Small tensors should NOT be stashed (storage size > 0)
        for param, state in optimizer.state.items():
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        tensor_size = value.numel() * value.element_size()
                        if tensor_size < 1024 * 1024:
                            self.assertGreater(
                                value.untyped_storage().size(),
                                0,
                                f"Small tensor {key} should NOT be stashed",
                            )

    def test_nested_dataclass_state(self) -> None:
        """Test stash and restore with nested dataclass-like optimizer state."""

        @dataclass
        class MockKroneckerFactors:
            """Mock class similar to ShampooKroneckerFactors."""

            factor_matrices: Tuple[torch.Tensor, ...]
            inv_factor_matrices: Tuple[torch.Tensor, ...]

        # Create a mock optimizer with nested state
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Manually inject nested dataclass state (simulating Shampoo)
        for param in model.parameters():
            factor_mat = torch.randn(512, 512, device=self.device)
            inv_factor_mat = torch.randn(512, 512, device=self.device)
            optimizer.state[param] = {
                "step": torch.tensor(1),
                "shampoo": MockKroneckerFactors(
                    factor_matrices=(factor_mat,),
                    inv_factor_matrices=(inv_factor_mat,),
                ),
            }

        # Store original values
        original_factors: List[torch.Tensor] = []
        original_inv_factors: List[torch.Tensor] = []
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and "shampoo" in state:
                shampoo_state = state["shampoo"]
                for t in shampoo_state.factor_matrices:
                    original_factors.append(t.clone())
                for t in shampoo_state.inv_factor_matrices:
                    original_inv_factors.append(t.clone())

        # Stash
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)

        # Verify nested tensors are stashed
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and "shampoo" in state:
                shampoo_state = state["shampoo"]
                for t in shampoo_state.factor_matrices:
                    if t.numel() * t.element_size() >= 1024 * 1024:
                        self.assertEqual(
                            t.untyped_storage().size(),
                            0,
                            "Factor matrix should be stashed",
                        )
                for t in shampoo_state.inv_factor_matrices:
                    if t.numel() * t.element_size() >= 1024 * 1024:
                        self.assertEqual(
                            t.untyped_storage().size(),
                            0,
                            "Inv factor matrix should be stashed",
                        )

        # Restore
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)

        # Verify values are restored correctly
        idx = 0
        inv_idx = 0
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and "shampoo" in state:
                shampoo_state = state["shampoo"]
                for t in shampoo_state.factor_matrices:
                    self.assertTrue(
                        torch.allclose(t, original_factors[idx]),
                        "Factor matrix not restored correctly",
                    )
                    idx += 1
                for t in shampoo_state.inv_factor_matrices:
                    self.assertTrue(
                        torch.allclose(t, original_inv_factors[inv_idx]),
                        "Inv factor matrix not restored correctly",
                    )
                    inv_idx += 1

    def test_callback_signature_compatibility_with_register_hook(self) -> None:
        """Test that await_restore can be used as backward hook."""
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run a step to populate optimizer state
        x = torch.randn(32, 512, device=self.device)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Get original state values
        original_states: Dict[Any, Dict[str, torch.Tensor]] = {}
        for param, state in optimizer.state.items():
            if isinstance(state, dict):
                original_states[param] = {
                    k: v.clone()
                    for k, v in state.items()
                    if isinstance(v, torch.Tensor)
                }

        # Stash and register hooks
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(optimizer)

        # New forward pass with hooks registered
        x = torch.randn(32, 512, device=self.device)
        output = model(x)
        output.register_hook(
            lambda _grad: MemoryStashingManager.restore_optimizer_state()
        )
        output.register_hook(await_restore)

        # Backward pass should trigger the hooks and restore state
        loss = output.sum()
        loss.backward()

        # Verify state is restored
        for param, state in optimizer.state.items():
            if param in original_states and isinstance(state, dict):
                for key, value in state.items():
                    if key in original_states[param]:
                        self.assertGreater(
                            value.untyped_storage().size(),
                            0,
                            f"State {key} should be restored",
                        )


if __name__ == "__main__":
    unittest.main()
