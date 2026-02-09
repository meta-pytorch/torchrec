#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List
from unittest.mock import Mock

import torch
from torchrec.distributed.embeddingbag import stash_embedding_weights


class TestStashEmbeddingWeights(unittest.TestCase):
    """Tests for stash_embedding_weights function."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")

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
        """Test basic stash and restore functionality with the three-callback API."""
        original_weights = torch.ones((100, 64), device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        free_hbm, restore, await_restore = stash_embedding_weights(lookup)

        # Free HBM after stash copy completes
        free_hbm()

        # Verify HBM is freed
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        # Restore weights
        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

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

        free_hbm, restore, await_restore = stash_embedding_weights(lookup)

        # Free HBM
        free_hbm()

        # Verify all are stashed
        self.assertEqual(weights_1.untyped_storage().size(), 0)
        self.assertEqual(weights_2.untyped_storage().size(), 0)
        self.assertEqual(weights_3.untyped_storage().size(), 0)

        # Restore all
        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

        # Verify all are restored correctly
        self.assertTrue(torch.allclose(weights_1, original_values_1))
        self.assertTrue(torch.allclose(weights_2, original_values_2))
        self.assertTrue(torch.allclose(weights_3, original_values_3))

    def test_custom_stash_stream(self) -> None:
        """Test stash and restore with custom CUDA stream."""
        custom_stream = torch.cuda.Stream(device=self.device)

        original_weights = torch.randn(50, 32, device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        free_hbm, restore, await_restore = stash_embedding_weights(
            lookup, stash_stream=custom_stream
        )

        # Free HBM
        free_hbm()

        # Verify stash worked
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        # Restore
        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

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
        free_hbm, restore, await_restore = stash_embedding_weights(lookup)
        free_hbm()

        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

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

        free_hbm, restore, await_restore = stash_embedding_weights(lookup)
        free_hbm()

        # Only CUDA weights should be stashed
        self.assertEqual(cuda_weights.untyped_storage().size(), 0)
        self.assertGreater(cpu_weights.untyped_storage().size(), 0)

        # Restore
        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

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

        free_hbm, restore, await_restore = stash_embedding_weights(lookup)
        free_hbm()

        # Valid weights should be stashed
        self.assertEqual(valid_weights.untyped_storage().size(), 0)

        # Restore
        dummy_grad = torch.tensor([])
        restore(dummy_grad)
        await_restore(dummy_grad)

        self.assertTrue(torch.allclose(valid_weights, valid_original))

    def test_callback_signature_compatibility_with_register_hook(self) -> None:
        """Test that restore and await_restore can be used as backward hooks."""
        weights = torch.randn(10, 5, device=self.device, requires_grad=True)
        original_values = weights.clone()

        lookup = self._create_mock_lookup([weights])

        # Create a tensor that we'll register hooks on
        x = torch.randn(3, 5, device=self.device, requires_grad=True)
        output = torch.matmul(x, weights.t())

        free_hbm, restore, await_restore = stash_embedding_weights(lookup)
        free_hbm()

        # Register callbacks as backward hooks (this is how they're used in practice)
        output.register_hook(restore)
        output.register_hook(await_restore)

        # Backward pass should trigger the hooks
        loss = output.sum()
        loss.backward()

        # Weights should be restored after backward
        self.assertGreater(weights.untyped_storage().size(), 0)
        self.assertTrue(torch.allclose(weights, original_values))


if __name__ == "__main__":
    unittest.main()
