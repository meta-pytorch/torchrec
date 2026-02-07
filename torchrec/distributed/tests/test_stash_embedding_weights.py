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
        """Test basic stash and restore functionality."""
        original_weights = torch.ones((100, 64), device=self.device)
        original_values = original_weights.clone()

        lookup = self._create_mock_lookup([original_weights])

        restore_fn = stash_embedding_weights(lookup)

        # Verify HBM is freed
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        restore_fn(torch.tensor([]))

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

        restore_fn = stash_embedding_weights(lookup)

        # Verify all are stashed
        self.assertEqual(weights_1.untyped_storage().size(), 0)
        self.assertEqual(weights_2.untyped_storage().size(), 0)
        self.assertEqual(weights_3.untyped_storage().size(), 0)

        restore_fn(torch.tensor([]))

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

        restore_fn = stash_embedding_weights(lookup, stash_stream=custom_stream)

        # Verify stash worked
        self.assertEqual(original_weights.untyped_storage().size(), 0)

        restore_fn(torch.tensor([]))

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
        restore_fn = stash_embedding_weights(lookup)
        restore_fn(torch.tensor([]))

        # Version should not have changed
        self.assertEqual(weights._version, initial_version)

        # Backward should work without errors
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(weights.grad)
        self.assertGreater(weights.grad.abs().sum().item(), 0)
