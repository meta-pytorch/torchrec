#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torch import distributed as dist, nn
from torch.distributed._shard.sharded_tensor import init_from_local_shards, Shard
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.memory_stashing import (
    _collect_cuda_tensors_from_value,
    _partition_tensors_into_slices,
    chunked_copy_,
    MemoryStashingManager,
)
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection


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

        await_restore, restore, _execute_stash = MemoryStashingManager._stash_tensors(
            [tensor]
        )

        # Verify tensor is on CPU (HBM freed, data readable for checkpoint)
        self.assertFalse(tensor.is_cuda)

        # Restore
        restore(None)
        await_restore(None)

        # Verify tensor is back on CUDA with correct values
        self.assertTrue(tensor.is_cuda)
        torch.testing.assert_close(tensor, original, rtol=1e-05, atol=1e-08)

    def test_multiple_tensors(self) -> None:
        """Test stash and restore with multiple tensors."""
        t1 = torch.randn(50, 32, device=self.device)
        t2 = torch.ones(80, 64, device=self.device) * 2
        originals = [t1.clone(), t2.clone()]

        await_restore, restore, _execute_stash = MemoryStashingManager._stash_tensors(
            [t1, t2]
        )

        # All on CPU (HBM freed)
        self.assertFalse(t1.is_cuda)
        self.assertFalse(t2.is_cuda)

        # Restore
        restore(None)
        await_restore(None)

        # All restored correctly
        torch.testing.assert_close(t1, originals[0], rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(t2, originals[1], rtol=1e-05, atol=1e-08)

    def test_empty_list(self) -> None:
        """Test that an empty tensor list returns no-op callbacks."""
        await_restore, restore, _execute_stash = MemoryStashingManager._stash_tensors(
            []
        )
        # No-op callbacks must be callable and must not raise.
        self.assertTrue(callable(restore))
        self.assertTrue(callable(await_restore))
        restore(None)
        await_restore(None)

    def test_preserves_autograd_version(self) -> None:
        """Test that restore does not increment the tensor version counter."""
        tensor = torch.randn(10, 5, device=self.device, requires_grad=True)
        version_before = tensor._version

        await_restore, restore, _execute_stash = MemoryStashingManager._stash_tensors(
            [tensor]
        )
        restore(None)
        await_restore(None)

        self.assertEqual(tensor._version, version_before)

    def test_callbacks_accept_grad_argument(self) -> None:
        """Test that callbacks work as backward hooks (accept a grad tensor)."""
        tensor = torch.randn(10, 5, device=self.device)
        original = tensor.clone()

        await_restore, restore, _execute_stash = MemoryStashingManager._stash_tensors(
            [tensor]
        )

        dummy_grad = torch.tensor([1.0])
        restore(dummy_grad)
        await_restore(dummy_grad)

        torch.testing.assert_close(tensor, original, rtol=1e-05, atol=1e-08)


class TestStashEmbeddingWeights(unittest.TestCase):
    """Tests for stash_embedding_weights function."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def _create_mock_lookup(
        self,
        weights_list: List[torch.Tensor],
        stash_weights_list: Optional[List[bool]] = None,
    ) -> Mock:
        """Helper to create a mock lookup with multiple embedding modules.

        Args:
            weights_list: List of weight tensors, one per TBE group.
            stash_weights_list: If provided, sets _config to a
                GroupedEmbeddingConfig with a single ShardedEmbeddingTable
                per group whose stash_weights matches this list. If None,
                no _config is set (backward-compatible: stash everything).
        """
        emb_modules = []
        for i, weights in enumerate(weights_list):
            inner = Mock()
            inner.weights_dev = weights
            emb_module = Mock()
            emb_module._emb_module = inner
            if stash_weights_list is not None:
                emb_module._config = GroupedEmbeddingConfig(
                    data_type=DataType.FP32,
                    pooling=PoolingType.SUM,
                    is_weighted=False,
                    has_feature_processor=False,
                    compute_kernel=EmbeddingComputeKernel.FUSED,
                    embedding_tables=[
                        ShardedEmbeddingTable(
                            num_embeddings=weights.shape[0],
                            embedding_dim=weights.shape[1],
                            name=f"table_{i}",
                            feature_names=[f"feature_{i}"],
                            pooling=PoolingType.SUM,
                            is_weighted=False,
                            has_feature_processor=False,
                            compute_kernel=EmbeddingComputeKernel.FUSED,
                            local_rows=weights.shape[0],
                            local_cols=weights.shape[1],
                            stash_weights=stash_weights_list[i],
                        ),
                    ],
                )
            emb_modules.append(emb_module)

        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = emb_modules
        return lookup

    def test_basic_stash_and_restore(self) -> None:
        """Test basic stash and restore functionality with the two-callback API."""
        original_weights = torch.ones((100, 64), device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Verify tensor is on CPU (HBM freed, data readable for checkpoint)
        self.assertFalse(original_weights.is_cuda)

        # Restore weights
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify tensor is back on CUDA with correct values
        self.assertTrue(original_weights.is_cuda)
        torch.testing.assert_close(
            original_weights, original_values, rtol=1e-05, atol=1e-08
        )

    def test_multiple_emb_modules_stashed(self) -> None:
        """Test that multiple embedding modules are all stashed and restored."""
        weights_1 = torch.ones((50, 32), device=self.device)
        weights_2 = torch.ones((80, 64), device=self.device) * 2
        weights_3 = torch.ones((100, 128), device=self.device) * 3

        original_values_1 = weights_1.clone()
        original_values_2 = weights_2.clone()
        original_values_3 = weights_3.clone()

        lookup = self._create_mock_lookup([weights_1, weights_2, weights_3])

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Verify all are on CPU (HBM freed)
        self.assertFalse(weights_1.is_cuda)
        self.assertFalse(weights_2.is_cuda)
        self.assertFalse(weights_3.is_cuda)

        # Restore all
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify all are restored correctly
        torch.testing.assert_close(weights_1, original_values_1, rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(weights_2, original_values_2, rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(weights_3, original_values_3, rtol=1e-05, atol=1e-08)

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

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Verify stash worked (tensor on CPU)
        self.assertFalse(original_weights.is_cuda)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Verify restoration
        torch.testing.assert_close(
            original_weights, original_values, rtol=1e-05, atol=1e-08
        )

    def test_restore_does_not_break_autograd(self) -> None:
        """Test that restore doesn't break autograd for backward pass."""
        weights = torch.randn(10, 5, device=self.device, requires_grad=True)
        initial_version = weights._version

        lookup = self._create_mock_lookup([weights])

        # Forward pass
        x = torch.randn(3, 5, device=self.device)
        output = torch.matmul(x, weights.t())

        # Stash and restore
        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

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

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Only CUDA weights should be stashed (moved to CPU)
        self.assertFalse(cuda_weights.is_cuda)
        self.assertGreater(cpu_weights.untyped_storage().size(), 0)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        torch.testing.assert_close(cuda_weights, cuda_original, rtol=1e-05, atol=1e-08)

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

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Valid weights should be stashed (moved to CPU)
        self.assertFalse(valid_weights.is_cuda)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        torch.testing.assert_close(
            valid_weights, valid_original, rtol=1e-05, atol=1e-08
        )

    def test_callback_signature_compatibility_with_register_hook(self) -> None:
        """Test that await_restore can be used as backward hook."""
        weights = torch.randn(10, 5, device=self.device, requires_grad=True)
        original_values = weights.clone()

        lookup = self._create_mock_lookup([weights])

        # Create a tensor that we'll register hooks on
        x = torch.randn(3, 5, device=self.device, requires_grad=True)
        output = torch.matmul(x, weights.t())

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Register restore via class method
        output.register_hook(
            lambda _grad: MemoryStashingManager.restore_embedding_weights()
        )
        output.register_hook(await_restore)

        # Backward pass should trigger the hooks
        loss = output.sum()
        loss.backward()

        # Weights should be restored after backward
        self.assertGreater(weights.untyped_storage().size(), 0)
        torch.testing.assert_close(weights, original_values, rtol=1e-05, atol=1e-08)

    def test_stash_weights_config_filters_tbe_groups(self) -> None:
        """Test that only TBE groups with stash_weights=True are stashed."""
        stash_weights = torch.ones((50, 32), device=self.device)
        no_stash_weights = torch.ones((80, 64), device=self.device) * 2

        stash_original = stash_weights.clone()
        no_stash_original = no_stash_weights.clone()

        lookup = self._create_mock_lookup(
            [stash_weights, no_stash_weights],
            stash_weights_list=[True, False],
        )

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Only the stash_weights=True group should be stashed (moved to CPU)
        self.assertFalse(stash_weights.is_cuda)
        # The stash_weights=False group should NOT be stashed
        self.assertTrue(no_stash_weights.is_cuda)
        self.assertTrue(torch.allclose(no_stash_weights, no_stash_original))

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        # Stashed weights should be restored correctly
        self.assertTrue(torch.allclose(stash_weights, stash_original))
        # Non-stashed weights should remain unchanged
        self.assertTrue(torch.allclose(no_stash_weights, no_stash_original))

    def test_stash_weights_all_false_returns_none(self) -> None:
        """Test that stash_embedding_weights returns None when all tables have stash_weights=False."""
        weights_1 = torch.ones((50, 32), device=self.device)
        weights_2 = torch.ones((80, 64), device=self.device)

        lookup = self._create_mock_lookup(
            [weights_1, weights_2],
            stash_weights_list=[False, False],
        )

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNone(result)

        # No weights should be stashed
        self.assertGreater(weights_1.untyped_storage().size(), 0)
        self.assertGreater(weights_2.untyped_storage().size(), 0)

    def test_stash_weights_all_true_stashes_all(self) -> None:
        """Test that all TBE groups are stashed when all have stash_weights=True."""
        weights_1 = torch.ones((50, 32), device=self.device)
        weights_2 = torch.ones((80, 64), device=self.device) * 2

        original_1 = weights_1.clone()
        original_2 = weights_2.clone()

        lookup = self._create_mock_lookup(
            [weights_1, weights_2],
            stash_weights_list=[True, True],
        )

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)
        await_restore, _restore, _execute_stash = result

        # Both should be stashed (moved to CPU)
        self.assertFalse(weights_1.is_cuda)
        self.assertFalse(weights_2.is_cuda)

        # Restore
        MemoryStashingManager.restore_embedding_weights()
        await_restore(None)

        self.assertTrue(torch.allclose(weights_1, original_1))
        self.assertTrue(torch.allclose(weights_2, original_2))

    def test_stash_weights_no_config_stashes_all(self) -> None:
        """Test backward compat: without _config, all TBE groups are stashed."""
        weights_1 = torch.ones((50, 32), device=self.device)
        weights_2 = torch.ones((80, 64), device=self.device)

        lookup = self._create_mock_lookup(
            [weights_1, weights_2],
            stash_weights_list=None,  # No config set
        )

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)

        # Both should be stashed (no config = stash everything)
        self.assertFalse(weights_1.is_cuda)
        self.assertFalse(weights_2.is_cuda)

    def test_is_enabled(self) -> None:
        """Test is_enabled reflects stream initialization state."""
        self.assertTrue(MemoryStashingManager.is_enabled())
        MemoryStashingManager.reset()
        self.assertFalse(MemoryStashingManager.is_enabled())


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

        # Verify large state tensors are stashed to CPU
        for _param, state in optimizer.state.items():
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        tensor_size = value.numel() * value.element_size()
                        if tensor_size >= 1024 * 1024:
                            self.assertFalse(
                                value.is_cuda,
                                f"Tensor {key} should be stashed to CPU",
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

        # Verify nested tensors are stashed to CPU
        for param, state in optimizer.state.items():
            if isinstance(state, dict) and "shampoo" in state:
                shampoo_state = state["shampoo"]
                for t in shampoo_state.factor_matrices:
                    if t.numel() * t.element_size() >= 1024 * 1024:
                        self.assertFalse(
                            t.is_cuda,
                            "Factor matrix should be stashed to CPU",
                        )
                for t in shampoo_state.inv_factor_matrices:
                    if t.numel() * t.element_size() >= 1024 * 1024:
                        self.assertFalse(
                            t.is_cuda,
                            "Inv factor matrix should be stashed to CPU",
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


class TestEmsConfigWiring(unittest.TestCase):
    """Tests that EMS config is correctly wired from EmbeddingBagConfig through to MemoryStashingManager."""

    def test_ebc_stash_weights_propagates_to_stashing_manager(self) -> None:
        """When stash_weights=True on EmbeddingBagConfig, MemoryStashingManager stashes that TBE group."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=device))

        stash_weights = torch.ones((50, 32), device=device)
        no_stash_weights = torch.ones((80, 64), device=device) * 2

        stash_original = stash_weights.clone()
        no_stash_original = no_stash_weights.clone()

        # Build mock lookup where TBE groups have ShardedEmbeddingTable configs
        # with stash_weights derived from the EmbeddingBagConfig value
        emb_modules = []
        for i, (weights, should_stash) in enumerate(
            [(stash_weights, True), (no_stash_weights, False)]
        ):
            inner = Mock()
            inner.weights_dev = weights
            emb_module = Mock()
            emb_module._emb_module = inner
            emb_module._config = GroupedEmbeddingConfig(
                data_type=DataType.FP32,
                pooling=PoolingType.SUM,
                is_weighted=False,
                has_feature_processor=False,
                compute_kernel=EmbeddingComputeKernel.FUSED,
                embedding_tables=[
                    ShardedEmbeddingTable(
                        num_embeddings=weights.shape[0],
                        embedding_dim=weights.shape[1],
                        name=f"table_{i}",
                        feature_names=[f"feature_{i}"],
                        pooling=PoolingType.SUM,
                        is_weighted=False,
                        has_feature_processor=False,
                        compute_kernel=EmbeddingComputeKernel.FUSED,
                        local_rows=weights.shape[0],
                        local_cols=weights.shape[1],
                        stash_weights=should_stash,
                    ),
                ],
            )
            emb_modules.append(emb_module)

        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = emb_modules

        result = MemoryStashingManager.stash_embedding_weights(lookup)
        self.assertIsNotNone(result)

        # Only the stash_weights=True TBE group should be stashed (moved to CPU)
        self.assertFalse(stash_weights.is_cuda)
        # The stash_weights=False TBE group should NOT be stashed
        self.assertTrue(no_stash_weights.is_cuda)
        self.assertTrue(torch.allclose(no_stash_weights, no_stash_original))

        # Restore and verify
        MemoryStashingManager.restore_embedding_weights()
        result[0](None)  # await_restore
        self.assertTrue(torch.allclose(stash_weights, stash_original))

        MemoryStashingManager.reset()

    def test_ebc_model_stash_weights_mutation(self) -> None:
        """Simulates the factory bridge: setting stash_weights=True on EBC configs in a model."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name=f"table_{i}",
                feature_names=[f"feat_{i}"],
            )
            for i in range(3)
        ]

        # Verify default is False
        for t in tables:
            self.assertFalse(t.stash_weights)

        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        model = nn.Module()
        model.ebc = ebc

        # Apply the same bridge logic as ads_rec_train_factory
        for module in model.modules():
            if isinstance(module, EmbeddingBagCollection):
                for eb_config in module.embedding_bag_configs():
                    eb_config.stash_weights = True

        # Verify stash_weights is now True on all configs
        for eb_config in ebc.embedding_bag_configs():
            self.assertTrue(
                eb_config.stash_weights,
                f"{eb_config.name} should have stash_weights=True",
            )

    def test_ebc_model_stash_weights_not_set_when_disabled(self) -> None:
        """When EMS is disabled, stash_weights remains False on all EBC configs."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name=f"table_{i}",
                feature_names=[f"feat_{i}"],
            )
            for i in range(3)
        ]

        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        model = nn.Module()
        model.ebc = ebc

        # Do NOT apply the bridge logic (EMS disabled)
        for eb_config in ebc.embedding_bag_configs():
            self.assertFalse(
                eb_config.stash_weights,
                f"{eb_config.name} should have stash_weights=False when EMS disabled",
            )


def _expected_num_chunks(numel: int, element_size: int, chunk_size_bytes: int) -> int:
    """Mirror chunked_copy_'s chunk arithmetic to predict the per-chunk op count."""
    chunk_elems = max(1, chunk_size_bytes // element_size)
    return math.ceil(numel / chunk_elems)


def _filled(
    shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create a deterministically-filled tensor of the given dtype/device."""
    if dtype.is_floating_point:
        return torch.randn(shape, device=device).to(dtype)
    return torch.randint(-1000, 1000, shape, dtype=dtype, device=device)


class ChunkedCopyTest(unittest.TestCase):
    """Tests for chunked_copy_ exercising real cross-device (H2D / D2H) transfers.

    ``chunked_copy_`` exists to chunk host<->device copies, so every test moves
    data between CPU and CUDA (src and dst on different devices) rather than
    CPU->CPU. Requires a GPU; skipped otherwise.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")

    def _src_dst(
        self, shape: Tuple[int, ...], dtype: torch.dtype, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (src, dst) on opposite devices for the given direction."""
        host = _filled(shape, dtype, torch.device("cpu"))
        if direction == "h2d":  # CPU src -> CUDA dst
            return host.pin_memory(), torch.zeros(
                shape, dtype=dtype, device=self.device
            )
        # d2h: CUDA src -> CPU dst
        return host.to(self.device), torch.zeros(shape, dtype=dtype).pin_memory()

    @given(
        direction=st.sampled_from(["h2d", "d2h"]),
        dtype=st.sampled_from(
            [torch.float32, torch.float16, torch.float64, torch.int32]
        ),
        dims=st.lists(st.integers(min_value=1, max_value=64), min_size=1, max_size=3),
        chunk_size_bytes=st.sampled_from([256, 1024, 65536, 1024**2]),
    )
    @settings(max_examples=100, deadline=None)
    def test_numerical_correctness_h2d_and_d2h(
        self,
        direction: str,
        dtype: torch.dtype,
        dims: List[int],
        chunk_size_bytes: int,
    ) -> None:
        """Chunked H2D/D2H copy reproduces the source bit-for-bit across shapes."""
        src, dst = self._src_dst(tuple(dims), dtype, direction)
        chunked_copy_(dst, src, chunk_size_bytes=chunk_size_bytes)
        torch.cuda.synchronize()
        # Exact match: copy_ between same dtype is lossless.
        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_default_chunk_size_copies_correctly(self) -> None:
        """Calling without chunk_size_bytes uses the default and copies correctly."""
        src, dst = self._src_dst((10000,), torch.float32, "h2d")
        chunked_copy_(dst, src)  # no chunk_size_bytes -> use default
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_h2d_in_place_and_location(self) -> None:
        """H2D writes dst in place and keeps it on the GPU (no realloc)."""
        src, dst = self._src_dst((50000,), torch.float32, "h2d")
        ptr_before = dst.data_ptr()

        # 64 KiB chunks -> many chunks, exercising the loop + dummy compute.
        chunked_copy_(dst, src, chunk_size_bytes=64 * 1024, dummy_compute=True)
        torch.cuda.synchronize()

        self.assertTrue(dst.is_cuda)
        self.assertEqual(dst.data_ptr(), ptr_before)
        self.assertEqual(dst.shape, src.shape)
        self.assertEqual(dst.dtype, src.dtype)
        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_d2h_in_place_and_location(self) -> None:
        """D2H writes dst in place and keeps it on the host."""
        src, dst = self._src_dst((50000,), torch.float32, "d2h")
        ptr_before = dst.data_ptr()

        chunked_copy_(dst, src, chunk_size_bytes=64 * 1024, dummy_compute=True)
        torch.cuda.synchronize()

        self.assertFalse(dst.is_cuda)
        self.assertEqual(dst.data_ptr(), ptr_before)
        self.assertTrue(torch.equal(dst, src.cpu()))

    def test_source_is_not_mutated(self) -> None:
        """Copying does not modify the source tensor (including with dummy_compute)."""
        src, dst = self._src_dst((1000,), torch.float32, "h2d")
        src_snapshot = src.clone()
        chunked_copy_(dst, src, chunk_size_bytes=1024, dummy_compute=True)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(src, src_snapshot))

    def test_size_mismatch_raises(self) -> None:
        """Mismatched element counts raise ValueError."""
        dst = torch.zeros(100, device=self.device)
        src = torch.randn(99)
        with self.assertRaises(ValueError):
            chunked_copy_(dst, src, chunk_size_bytes=1024)

    def test_zero_and_negative_chunk_size_copies_correctly(self) -> None:
        """chunk_size_bytes <= 0 disables chunking but still copies correctly."""
        for chunk_size_bytes in (0, -1):
            with self.subTest(nbytes=chunk_size_bytes):
                src, dst = self._src_dst((1000,), torch.float32, "h2d")
                chunked_copy_(dst, src, chunk_size_bytes=chunk_size_bytes)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(dst.cpu(), src.cpu()))

    def test_empty_tensor_is_noop(self) -> None:
        """Empty tensors copy without error and stay empty."""
        src = torch.randn(0, device=self.device)
        dst = torch.zeros(0)
        chunked_copy_(dst, src, chunk_size_bytes=1024)
        self.assertEqual(dst.numel(), 0)

    def test_non_contiguous_stays_correct(self) -> None:
        """Non-contiguous dst/src (fallback path) still copy correctly across devices."""
        # Non-contiguous CPU source (transposed view) -> contiguous CUDA dst.
        src = torch.randn(20, 10).t()
        dst = torch.zeros(10, 20, device=self.device)
        self.assertFalse(src.is_contiguous())
        chunked_copy_(dst, src, chunk_size_bytes=256)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dst.cpu(), src))

        # Non-contiguous CUDA destination (transposed view) <- contiguous CPU src.
        src2 = torch.randn(10, 20)
        dst2 = torch.zeros(20, 10, device=self.device).t()
        self.assertFalse(dst2.is_contiguous())
        chunked_copy_(dst2, src2, chunk_size_bytes=256)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dst2.cpu(), src2))

    def test_dummy_compute_count_matches_chunks(self) -> None:
        """With dummy_compute, exactly (num_chunks - 1) add_ ops are enqueued."""
        numel = 50000
        src, dst = self._src_dst((numel,), torch.float32, "h2d")
        chunk_size_bytes = 64 * 1024
        expected_chunks = _expected_num_chunks(
            numel, dst.element_size(), chunk_size_bytes
        )
        self.assertGreater(expected_chunks, 1)

        real_add = torch.Tensor.add_
        add_calls: List[int] = []

        def counting_add(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            add_calls.append(1)
            return real_add(self, *args, **kwargs)

        with patch.object(torch.Tensor, "add_", counting_add):
            chunked_copy_(
                dst, src, chunk_size_bytes=chunk_size_bytes, dummy_compute=True
            )
        torch.cuda.synchronize()

        # A dummy op sits between consecutive chunks: one fewer than chunk count.
        self.assertEqual(len(add_calls), expected_chunks - 1)
        self.assertTrue(torch.equal(dst.cpu(), src.cpu()))


class TestCollectCudaTensorsSharded(unittest.TestCase):
    """``_collect_cuda_tensors_from_value`` must unwrap ShardedTensor / DTensor
    optimizer state into their local CUDA shard tensors instead of crashing on
    ``.is_cuda`` (which routes through their ``__torch_function__`` and raises).
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not dist.is_available():
            self.skipTest("torch.distributed not available")
        self.device = torch.device("cuda:0")
        self._created_pg = False
        if not dist.is_initialized():
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                rank=0,
                world_size=1,
                init_method=f"file:///tmp/trec_memstash_pg_{os.getpid()}",
            )
            self._created_pg = True

    def tearDown(self) -> None:
        if self._created_pg and dist.is_initialized():
            dist.destroy_process_group()

    def test_collect_unwraps_sharded_tensor(self) -> None:
        # 1024 * 512 * 4 bytes = 2MB, above the 1MB stash threshold.
        local = torch.randn(1024, 512, device=self.device)
        shard = Shard.from_tensor_and_offsets(local, shard_offsets=[0, 0], rank=0)
        st = init_from_local_shards([shard], 1024, 512)

        collected = _collect_cuda_tensors_from_value(st)

        self.assertEqual(len(collected), 1)
        self.assertTrue(collected[0].is_cuda)
        self.assertEqual(collected[0].data_ptr(), local.data_ptr())

    def test_collect_sharded_tensor_in_optimizer_state_dict(self) -> None:
        # Mirrors the real failure: a sharded optimizer-state tensor nested in
        # the per-param state dict, as iterated by ``stash_optimizer_state``.
        local = torch.randn(1024, 512, device=self.device)
        shard = Shard.from_tensor_and_offsets(local, shard_offsets=[0, 0], rank=0)
        st = init_from_local_shards([shard], 1024, 512)
        state_value = {"exp_avg": st, "step": torch.tensor(1)}

        collected = _collect_cuda_tensors_from_value(state_value)

        # exp_avg (sharded, 2MB) is collected; step (tiny CPU scalar) is skipped.
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].data_ptr(), local.data_ptr())

    def test_collect_unwraps_dtensor(self) -> None:
        mesh = DeviceMesh("cuda", [0])
        # 2MB, above the 1MB stash threshold.
        local = torch.randn(1024, 512, device=self.device)
        # Wrap the already-local tensor as a Replicate DTensor without a
        # broadcast collective; distribute_tensor's broadcast needs an NCCL
        # comm that cannot bootstrap in the single-host test sandbox.
        dt = DTensor.from_local(local, mesh, [Replicate()], run_check=False)

        collected = _collect_cuda_tensors_from_value(dt)

        self.assertEqual(len(collected), 1)
        self.assertTrue(collected[0].is_cuda)


class TestPartitionTensorsIntoSlices(unittest.TestCase):
    """Tests for the byte-balanced slice partitioning helper (CUDA-free)."""

    def test_single_slice_returns_all_tensors(self) -> None:
        tensors = [torch.empty(100), torch.empty(200)]
        slices = _partition_tensors_into_slices(tensors, num_slices=1)
        self.assertEqual(len(slices), 1)
        # num_slices <= 1 returns the original list unmodified.
        self.assertIs(slices[0], tensors)

    def test_nonpositive_slices_returns_all_tensors(self) -> None:
        tensors = [torch.empty(100)]
        slices = _partition_tensors_into_slices(tensors, num_slices=0)
        self.assertEqual(len(slices), 1)
        self.assertIs(slices[0], tensors)

    def test_empty_tensor_list_returns_empty(self) -> None:
        self.assertEqual(_partition_tensors_into_slices([], num_slices=4), [])

    def test_partition_covers_every_tensor_exactly_once(self) -> None:
        sizes = [100, 200, 300, 400, 500, 600, 700, 800]
        tensors = [torch.empty(s, dtype=torch.float32) for s in sizes]
        slices = _partition_tensors_into_slices(tensors, num_slices=4)
        self.assertEqual(len(slices), 4)
        flat = [t for one_slice in slices for t in one_slice]
        self.assertCountEqual(
            [t.data_ptr() for t in flat],
            [t.data_ptr() for t in tensors],
        )

    def test_partition_is_byte_balanced(self) -> None:
        sizes = [100, 200, 300, 400, 500, 600, 700, 800]
        tensors = [torch.empty(s, dtype=torch.float32) for s in sizes]
        slices = _partition_tensors_into_slices(tensors, num_slices=4)
        bin_bytes = [
            sum(t.numel() * t.element_size() for t in one_slice) for one_slice in slices
        ]
        # Greedy LPT keeps bins within a tensor's worth of each other.
        self.assertLessEqual(max(bin_bytes) - min(bin_bytes), 800 * 4)

    def test_shared_storage_tensors_stay_in_same_slice(self) -> None:
        # Two views of one storage must never be split across slices (a
        # resize_(0)/resize_(size) pair would otherwise corrupt the buffer).
        base = torch.empty(1000, dtype=torch.float32)
        view_a = base[:500]
        view_b = base[500:]
        other = torch.empty(4000, dtype=torch.float32)
        tensors = [view_a, other, view_b]
        slices = _partition_tensors_into_slices(tensors, num_slices=2)
        slice_of_a = next(
            i for i, s in enumerate(slices) if any(t is view_a for t in s)
        )
        slice_of_b = next(
            i for i, s in enumerate(slices) if any(t is view_b for t in s)
        )
        self.assertEqual(slice_of_a, slice_of_b)

    def test_fewer_groups_than_requested_slices(self) -> None:
        tensors = [torch.empty(100), torch.empty(200)]
        slices = _partition_tensors_into_slices(tensors, num_slices=5)
        # Bounded by the number of distinct storage groups.
        self.assertEqual(len(slices), 2)


class TestStashOptimizerStateSliced(unittest.TestCase):
    """Tests for gradual (sliced) optimizer-state stash/restore."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def _adam_with_state(self) -> torch.optim.Optimizer:
        # nn.Linear(512, 512): weight is exactly 1MB so Adam keeps two large
        # state tensors (exp_avg, exp_avg_sq) -> the state forms 2 slices.
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True)
        x = torch.randn(32, 512, device=self.device)
        model(x).sum().backward()
        optimizer.step()
        return optimizer

    def _clone_state(
        self, optimizer: torch.optim.Optimizer
    ) -> Dict[Any, Dict[str, torch.Tensor]]:
        original: Dict[Any, Dict[str, torch.Tensor]] = {}
        for param, state in optimizer.state.items():
            if isinstance(state, dict):
                original[param] = {
                    k: v.clone()
                    for k, v in state.items()
                    if isinstance(v, torch.Tensor)
                }
        return original

    def _assert_restored(
        self,
        optimizer: torch.optim.Optimizer,
        original: Dict[Any, Dict[str, torch.Tensor]],
    ) -> None:
        for param, state in optimizer.state.items():
            if param in original and isinstance(state, dict):
                for key, value in state.items():
                    if key in original[param]:
                        self.assertTrue(
                            torch.allclose(value, original[param][key]),
                            f"State {key} not restored correctly",
                        )

    def test_sliced_stash_registers_one_callback_per_slice(self) -> None:
        optimizer = self._adam_with_state()
        MemoryStashingManager.stash_optimizer_state(optimizer, num_slices=2)
        self.assertEqual(
            len(MemoryStashingManager._optimizer_state_restore_callbacks), 2
        )

    def test_restore_optimizer_state_next_pops_one_slice(self) -> None:
        optimizer = self._adam_with_state()
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(
            optimizer, num_slices=2
        )
        callbacks = MemoryStashingManager._optimizer_state_restore_callbacks
        self.assertEqual(len(callbacks), 2)
        MemoryStashingManager.restore_optimizer_state_next()
        self.assertEqual(len(callbacks), 1)
        MemoryStashingManager.restore_optimizer_state_next()
        self.assertEqual(len(callbacks), 0)
        # Popping again with an empty stack is a safe no-op.
        MemoryStashingManager.restore_optimizer_state_next()
        self.assertEqual(len(callbacks), 0)
        await_restore(None)

    def test_pop_all_restores_remaining_slices(self) -> None:
        optimizer = self._adam_with_state()
        original = self._clone_state(optimizer)
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(
            optimizer, num_slices=2
        )
        # Drive one slice via the per-hook path, the rest via the pop-all guard.
        MemoryStashingManager.restore_optimizer_state_next()
        self.assertEqual(
            len(MemoryStashingManager._optimizer_state_restore_callbacks), 1
        )
        MemoryStashingManager.restore_optimizer_state()
        self.assertEqual(
            len(MemoryStashingManager._optimizer_state_restore_callbacks), 0
        )
        await_restore(None)
        torch.cuda.synchronize()
        self._assert_restored(optimizer, original)

    def test_sliced_round_trip_matches_original(self) -> None:
        optimizer = self._adam_with_state()
        original = self._clone_state(optimizer)
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(
            optimizer, num_slices=2
        )
        # All large state tensors should be freed after the sliced stash.
        for _param, state in optimizer.state.items():
            if isinstance(state, dict):
                for value in state.values():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.is_cuda
                        and value.numel() * value.element_size() >= 1024 * 1024
                    ):
                        self.assertEqual(value.untyped_storage().size(), 0)
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)
        torch.cuda.synchronize()
        self._assert_restored(optimizer, original)

    def test_sliced_optimizer_step_works_after_restore(self) -> None:
        model = nn.Linear(512, 512).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=True)
        x = torch.randn(32, 512, device=self.device)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        weights_before = model.weight.detach().clone()
        await_restore, _restore = MemoryStashingManager.stash_optimizer_state(
            optimizer, num_slices=2
        )
        MemoryStashingManager.restore_optimizer_state()
        await_restore(None)
        # Another training step after the sliced restore must update weights.
        model(torch.randn(32, 512, device=self.device)).sum().backward()
        optimizer.step()
        self.assertFalse(
            torch.allclose(model.weight, weights_before),
            "Weights should change after optimizer step",
        )


class TestRestoreStashedSyncTensors(unittest.TestCase):
    """Tests for DMPCollection._restore_stashed_sync_tensors (the 2D-sync IMA fix).

    The helper restores any memory-stashed TBE weight / optimizer tensors back
    to HBM before DMPCollection.sync()'s allreduce, so the collective never
    reads freed memory (cudaErrorIllegalAddress). It is gated to be a no-op when
    stashing is disabled or the tensors are already resident.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        MemoryStashingManager.set_streams(torch.cuda.Stream(device=self.device))

    def tearDown(self) -> None:
        MemoryStashingManager.reset()

    def _call_helper(
        self,
        ctx: object,
        include_optimizer_state: bool = True,
    ) -> None:
        # The method does not use `self`, so None is fine for this unit test.
        # pyre-ignore[6]: None self (unused) + SimpleNamespace ctx are test stubs.
        DMPCollection._restore_stashed_sync_tensors(None, ctx, include_optimizer_state)

    def _stash(self, tensor: torch.Tensor) -> None:
        """Stash a tensor via the embedding path (registers global restore)."""
        inner = Mock()
        inner.weights_dev = tensor
        emb = Mock()
        emb._emb_module = inner
        lookup = Mock(spec=["_emb_modules"])
        lookup._emb_modules = [emb]
        self.assertIsNotNone(MemoryStashingManager.stash_embedding_weights(lookup))

    def test_restores_stashed_weight_before_sync(self) -> None:
        """A stashed sync weight view is restored bit-exact before the allreduce."""
        # ``weights_dev`` is the TBE weight slab that EMS stashes; ``sync_view``
        # mirrors the separate per-table tensor DMPCollection caches from
        # ``split_embedding_weights()`` -- a view sharing weights_dev's storage.
        weights_dev = torch.randn(256, 128, device=self.device)
        original = weights_dev.clone()
        sync_view = weights_dev.detach().view(-1)

        self._stash(weights_dev)
        # EMS re-points weights_dev to CPU, but the sync view keeps the original
        # (now freed) CUDA storage -- exactly the signal the helper detects.
        self.assertTrue(sync_view.is_cuda)
        self.assertEqual(sync_view.untyped_storage().size(), 0)

        ctx = SimpleNamespace(
            weights_by_dtype={sync_view.dtype: [sync_view]},
            optimizer_tensors_by_dtype={},
        )
        self._call_helper(ctx)
        torch.cuda.synchronize()

        # Restored to HBM and bit-exact, so a subsequent allreduce is safe.
        self.assertGreater(sync_view.untyped_storage().size(), 0)
        torch.testing.assert_close(
            sync_view.view(256, 128), original, rtol=1e-05, atol=1e-08
        )

    def test_restores_stashed_optimizer_tensor(self) -> None:
        """A stashed optimizer sync tensor view is also restored."""
        # ``momentum_dev`` is the fused-optimizer slab that EMS stashes;
        # ``sync_view`` mirrors the per-table tensor DMPCollection caches from
        # ``get_optimizer_state()["sum"]`` -- a view sharing momentum_dev's
        # storage. Stash through _stash_tensors and register on the optimizer
        # callback stack directly to mirror optimizer stashing.
        momentum_dev = torch.randn(512, 512, device=self.device)
        original = momentum_dev.clone()
        sync_view = momentum_dev.detach().view(-1)

        _await, restore, _exec = MemoryStashingManager._stash_tensors([momentum_dev])
        MemoryStashingManager._optimizer_state_restore_callbacks.append(restore)
        # The sync view keeps the original (now freed) CUDA storage.
        self.assertTrue(sync_view.is_cuda)
        self.assertEqual(sync_view.untyped_storage().size(), 0)

        ctx = SimpleNamespace(
            weights_by_dtype={},
            optimizer_tensors_by_dtype={sync_view.dtype: [sync_view]},
        )
        self._call_helper(ctx)
        torch.cuda.synchronize()

        self.assertGreater(sync_view.untyped_storage().size(), 0)
        torch.testing.assert_close(
            sync_view.view(512, 512), original, rtol=1e-05, atol=1e-08
        )

    def test_noop_when_tensors_resident(self) -> None:
        """Steady state (tensors resident): no-op, data untouched, no extra IO."""
        weight = torch.randn(64, 64, device=self.device)
        original = weight.clone()
        ctx = SimpleNamespace(
            weights_by_dtype={weight.dtype: [weight]},
            optimizer_tensors_by_dtype={},
        )
        self._call_helper(ctx)
        torch.cuda.synchronize()
        self.assertGreater(weight.untyped_storage().size(), 0)
        torch.testing.assert_close(weight, original, rtol=1e-05, atol=1e-08)

    def test_noop_when_stashing_disabled(self) -> None:
        """When stashing is disabled the helper returns before touching streams."""
        MemoryStashingManager.reset()
        self.assertFalse(MemoryStashingManager.is_enabled())
        weight = torch.randn(32, 32, device=self.device)
        ctx = SimpleNamespace(
            weights_by_dtype={weight.dtype: [weight]},
            optimizer_tensors_by_dtype={},
        )
        # Must not raise (e.g. from h2d_stream() asserting an unset stream).
        self._call_helper(ctx)


if __name__ == "__main__":
    unittest.main()
