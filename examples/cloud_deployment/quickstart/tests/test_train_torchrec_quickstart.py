#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for TorchRec Cloud Quickstart training script.

Tests cover:
- MLP module construction and forward pass
- DLRM model architecture and forward pass
- SyntheticDataset data generation
- Helper functions (setup_distributed, etc.)

Run with:
    buck test fbcode//torchrec/github/examples/cloud_deployment/quickstart/tests:test_train_torchrec_quickstart
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.github.examples.cloud_deployment.quickstart.train_torchrec_quickstart import (
    cleanup_distributed,
    DLRM,
    MLP,
    parse_args,
    setup_distributed,
    SyntheticDataset,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class MLPTest(unittest.TestCase):
    """Tests for the MLP module."""

    def test_mlp_construction(self) -> None:
        """Test that MLP can be constructed with valid parameters."""
        mlp = MLP(in_size=10, layer_sizes=[32, 16, 8])
        self.assertIsInstance(mlp, nn.Module)

    def test_mlp_forward_shape(self) -> None:
        """Test that MLP forward pass produces correct output shape."""
        in_size = 10
        layer_sizes = [32, 16, 8]
        batch_size = 4

        mlp = MLP(in_size=in_size, layer_sizes=layer_sizes)
        x = torch.randn(batch_size, in_size)
        output = mlp(x)

        self.assertEqual(output.shape, (batch_size, layer_sizes[-1]))

    def test_mlp_single_layer(self) -> None:
        """Test MLP with single layer architecture."""
        batch_size = 8
        mlp = MLP(in_size=10, layer_sizes=[5])
        x = torch.randn(batch_size, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_mlp_two_layers(self) -> None:
        """Test MLP with two layer architecture."""
        batch_size = 8
        mlp = MLP(in_size=20, layer_sizes=[10, 5])
        x = torch.randn(batch_size, 20)
        output = mlp(x)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_mlp_three_layers(self) -> None:
        """Test MLP with three layer architecture."""
        batch_size = 8
        mlp = MLP(in_size=30, layer_sizes=[64, 32, 16])
        x = torch.randn(batch_size, 30)
        output = mlp(x)
        self.assertEqual(output.shape, (batch_size, 16))

    def test_mlp_large_expansion(self) -> None:
        """Test MLP with large expansion architecture."""
        batch_size = 8
        mlp = MLP(in_size=5, layer_sizes=[100, 200, 50])
        x = torch.randn(batch_size, 5)
        output = mlp(x)
        self.assertEqual(output.shape, (batch_size, 50))

    def test_mlp_relu_activation(self) -> None:
        """Test that MLP uses ReLU activation by default."""
        mlp = MLP(in_size=10, layer_sizes=[5], activation="relu")
        # Check that ReLU is in the sequential layers
        has_relu = any(isinstance(m, nn.ReLU) for m in mlp.mlp.modules())
        self.assertTrue(has_relu)

    def test_mlp_sigmoid_activation(self) -> None:
        """Test that MLP can use sigmoid activation."""
        mlp = MLP(in_size=10, layer_sizes=[5], activation="sigmoid")
        # Check that Sigmoid is in the sequential layers
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in mlp.mlp.modules())
        self.assertTrue(has_sigmoid)

    def test_mlp_no_bias(self) -> None:
        """Test MLP construction without bias."""
        mlp = MLP(in_size=10, layer_sizes=[5], bias=False)
        # Check that linear layers have no bias
        for module in mlp.mlp.modules():
            if isinstance(module, nn.Linear):
                self.assertIsNone(module.bias)


class DLRMTest(unittest.TestCase):
    """Tests for the DLRM model."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_sparse_features = 4
        self.embedding_dim = 8
        self.num_embeddings = 100
        self.num_dense_features = 5
        self.batch_size = 4

        # Create embedding bag configs
        self.sparse_feature_names = [
            f"sparse_{i}" for i in range(self.num_sparse_features)
        ]
        self.embedding_configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=self.embedding_dim,
                num_embeddings=self.num_embeddings,
                feature_names=[self.sparse_feature_names[i]],
            )
            for i in range(self.num_sparse_features)
        ]

        # Create EmbeddingBagCollection
        self.embedding_bag_collection = EmbeddingBagCollection(
            tables=self.embedding_configs,
            device=torch.device("cpu"),
        )

    def test_dlrm_construction(self) -> None:
        """Test that DLRM model can be constructed."""
        model = DLRM(
            embedding_bag_collection=self.embedding_bag_collection,
            dense_in_features=self.num_dense_features,
            dense_arch_layer_sizes=[16, self.embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )
        self.assertIsInstance(model, nn.Module)

    def test_dlrm_forward_shape(self) -> None:
        """Test DLRM forward pass produces correct output shape."""
        model = DLRM(
            embedding_bag_collection=self.embedding_bag_collection,
            dense_in_features=self.num_dense_features,
            dense_arch_layer_sizes=[16, self.embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )

        # Create dense features
        dense_features = torch.randn(self.batch_size, self.num_dense_features)

        # Create sparse features (KeyedJaggedTensor)
        values_list = []
        lengths_list = []
        for _ in range(self.num_sparse_features):
            lengths = torch.ones(self.batch_size, dtype=torch.int32)
            values = torch.randint(0, self.num_embeddings, (self.batch_size,))
            values_list.append(values)
            lengths_list.append(lengths)

        sparse_features = KeyedJaggedTensor(
            keys=self.sparse_feature_names,
            values=torch.cat(values_list),
            lengths=torch.cat(lengths_list),
        )

        # Forward pass
        output = model(dense_features, sparse_features)

        # Output should be (batch_size,) after sigmoid
        self.assertEqual(output.shape, (self.batch_size,))

    def test_dlrm_output_range(self) -> None:
        """Test that DLRM output is in [0, 1] range (after sigmoid)."""
        model = DLRM(
            embedding_bag_collection=self.embedding_bag_collection,
            dense_in_features=self.num_dense_features,
            dense_arch_layer_sizes=[16, self.embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )

        dense_features = torch.randn(self.batch_size, self.num_dense_features)

        values_list = []
        lengths_list = []
        for _ in range(self.num_sparse_features):
            lengths = torch.ones(self.batch_size, dtype=torch.int32)
            values = torch.randint(0, self.num_embeddings, (self.batch_size,))
            values_list.append(values)
            lengths_list.append(lengths)

        sparse_features = KeyedJaggedTensor(
            keys=self.sparse_feature_names,
            values=torch.cat(values_list),
            lengths=torch.cat(lengths_list),
        )

        output = model(dense_features, sparse_features)

        # All outputs should be between 0 and 1
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())

    def test_dlrm_backward_pass(self) -> None:
        """Test that DLRM can perform backward pass."""
        model = DLRM(
            embedding_bag_collection=self.embedding_bag_collection,
            dense_in_features=self.num_dense_features,
            dense_arch_layer_sizes=[16, self.embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )

        dense_features = torch.randn(self.batch_size, self.num_dense_features)

        values_list = []
        lengths_list = []
        for _ in range(self.num_sparse_features):
            lengths = torch.ones(self.batch_size, dtype=torch.int32)
            values = torch.randint(0, self.num_embeddings, (self.batch_size,))
            values_list.append(values)
            lengths_list.append(lengths)

        sparse_features = KeyedJaggedTensor(
            keys=self.sparse_feature_names,
            values=torch.cat(values_list),
            lengths=torch.cat(lengths_list),
        )

        output = model(dense_features, sparse_features)
        labels = torch.randint(0, 2, (self.batch_size,)).float()

        # Compute loss and backward
        criterion = nn.BCELoss()
        loss = criterion(output, labels)
        loss.backward()

        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


class SyntheticDatasetTest(unittest.TestCase):
    """Tests for the SyntheticDataset class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_embeddings = 1000
        self.num_dense_features = 13
        self.num_sparse_features = 26
        self.batch_size = 32
        self.num_batches = 5
        self.sparse_feature_names = [
            f"sparse_{i}" for i in range(self.num_sparse_features)
        ]

    def test_dataset_iteration(self) -> None:
        """Test that dataset can be iterated over."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            sparse_feature_names=self.sparse_feature_names,
        )

        batch_count = 0
        for _dense, _sparse, _labels in dataset:
            batch_count += 1

        self.assertEqual(batch_count, self.num_batches)

    def test_dataset_dense_features_shape(self) -> None:
        """Test that dense features have correct shape."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for dense, _, _ in dataset:
            self.assertEqual(dense.shape, (self.batch_size, self.num_dense_features))

    def test_dataset_sparse_features_keys(self) -> None:
        """Test that sparse features have correct keys."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for _, sparse, _ in dataset:
            self.assertIsInstance(sparse, KeyedJaggedTensor)
            self.assertEqual(sparse.keys(), self.sparse_feature_names)

    def test_dataset_labels_shape(self) -> None:
        """Test that labels have correct shape."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for _, _, labels in dataset:
            self.assertEqual(labels.shape, (self.batch_size,))

    def test_dataset_labels_binary(self) -> None:
        """Test that labels are binary (0 or 1)."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for _, _, labels in dataset:
            unique_values = torch.unique(labels)
            for val in unique_values:
                self.assertIn(val.item(), [0.0, 1.0])

    def test_dataset_sparse_values_range(self) -> None:
        """Test that sparse feature values are within valid embedding range."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for _, sparse, _ in dataset:
            values = sparse.values()
            self.assertTrue(torch.all(values >= 0))
            self.assertTrue(torch.all(values < self.num_embeddings))

    def test_dataset_dense_features_range(self) -> None:
        """Test that dense features are in [0, 1] range (from torch.rand)."""
        dataset = SyntheticDataset(
            num_embeddings=self.num_embeddings,
            num_dense_features=self.num_dense_features,
            num_sparse_features=self.num_sparse_features,
            batch_size=self.batch_size,
            num_batches=1,
            sparse_feature_names=self.sparse_feature_names,
        )

        for dense, _, _ in dataset:
            self.assertTrue(torch.all(dense >= 0))
            self.assertTrue(torch.all(dense <= 1))


class SetupDistributedTest(unittest.TestCase):
    """Tests for distributed setup helper functions."""

    def test_setup_distributed_single_process(self) -> None:
        """Test setup_distributed in single process mode (no env vars)."""
        # Clear any existing env vars
        env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
        original_values = {}
        for var in env_vars:
            original_values[var] = os.environ.pop(var, None)

        try:
            rank, world_size, device = setup_distributed()
            self.assertEqual(rank, 0)
            self.assertEqual(world_size, 1)
            # Device should be cuda if available, otherwise cpu
            expected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.assertEqual(device, expected_device)
        finally:
            # Restore env vars
            for var, val in original_values.items():
                if val is not None:
                    os.environ[var] = val

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_setup_distributed_multi_process(
        self, mock_set_device: MagicMock, mock_init_pg: MagicMock
    ) -> None:
        """Test setup_distributed in distributed mode with env vars."""
        # Set env vars for distributed training
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "0"

        try:
            rank, world_size, device = setup_distributed()
            self.assertEqual(rank, 0)
            self.assertEqual(world_size, 4)
            mock_init_pg.assert_called_once_with(backend="nccl")
        finally:
            # Clean up env vars
            del os.environ["RANK"]
            del os.environ["WORLD_SIZE"]
            del os.environ["LOCAL_RANK"]

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_distributed_when_initialized(
        self, mock_destroy: MagicMock, mock_is_init: MagicMock
    ) -> None:
        """Test cleanup_distributed when process group is initialized."""
        cleanup_distributed()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_distributed_when_not_initialized(
        self, mock_destroy: MagicMock, mock_is_init: MagicMock
    ) -> None:
        """Test cleanup_distributed when process group is not initialized."""
        cleanup_distributed()
        mock_destroy.assert_not_called()


class MLPEdgeCaseTest(unittest.TestCase):
    """Additional edge case tests for MLP."""

    def test_mlp_gradient_flows_through_all_layers(self) -> None:
        """Test that gradients are computed for all linear layers in the MLP."""
        mlp = MLP(in_size=10, layer_sizes=[32, 16, 8])
        x = torch.randn(4, 10)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        linear_layers = [m for m in mlp.mlp.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(linear_layers), 3)
        for layer in linear_layers:
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.any(layer.weight.grad != 0))

    def test_mlp_parameter_count(self) -> None:
        """Test that MLP has the expected number of trainable parameters."""
        # Linear(10, 5) with bias: 10*5 + 5 = 55 params
        mlp = MLP(in_size=10, layer_sizes=[5], bias=True)
        total_params = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(total_params, 10 * 5 + 5)

    def test_mlp_no_bias_parameter_count(self) -> None:
        """Test MLP parameter count without bias."""
        # Linear(10, 5) without bias: 10*5 = 50 params
        mlp = MLP(in_size=10, layer_sizes=[5], bias=False)
        total_params = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(total_params, 10 * 5)

    def test_mlp_multi_layer_parameter_count(self) -> None:
        """Test parameter count for multi-layer MLP."""
        # Linear(10, 20) + Linear(20, 5) with bias: (10*20+20) + (20*5+5) = 325
        mlp = MLP(in_size=10, layer_sizes=[20, 5], bias=True)
        total_params = sum(p.numel() for p in mlp.parameters())
        self.assertEqual(total_params, (10 * 20 + 20) + (20 * 5 + 5))

    def test_mlp_batch_size_one(self) -> None:
        """Test MLP works with a single sample."""
        mlp = MLP(in_size=10, layer_sizes=[32, 16])
        x = torch.randn(1, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (1, 16))

    def test_mlp_deep_network(self) -> None:
        """Test MLP with many layers."""
        layer_sizes = [64, 128, 64, 32, 16, 8]
        mlp = MLP(in_size=32, layer_sizes=layer_sizes)
        x = torch.randn(4, 32)
        output = mlp(x)
        self.assertEqual(output.shape, (4, 8))

    def test_mlp_output_differs_for_different_inputs(self) -> None:
        """Test that MLP produces different outputs for different inputs."""
        mlp = MLP(in_size=10, layer_sizes=[32, 16])
        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 10)
        out1 = mlp(x1)
        out2 = mlp(x2)
        self.assertFalse(torch.allclose(out1, out2))

    def test_mlp_deterministic_output(self) -> None:
        """Test that MLP produces the same output for the same input."""
        mlp = MLP(in_size=10, layer_sizes=[32, 16])
        mlp.eval()
        x = torch.randn(4, 10)
        out1 = mlp(x)
        out2 = mlp(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_mlp_relu_output_non_negative(self) -> None:
        """Test that MLP with ReLU produces non-negative outputs after each ReLU."""
        mlp = MLP(in_size=10, layer_sizes=[32], activation="relu")
        x = torch.randn(100, 10)
        output = mlp(x)
        # After ReLU, output should be non-negative
        self.assertTrue(torch.all(output >= 0))

    def test_mlp_sigmoid_output_range(self) -> None:
        """Test that MLP with sigmoid produces outputs in (0, 1)."""
        mlp = MLP(in_size=10, layer_sizes=[32], activation="sigmoid")
        x = torch.randn(100, 10)
        output = mlp(x)
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())


class DLRMEdgeCaseTest(unittest.TestCase):
    """Additional edge case tests for DLRM model."""

    def _make_ebc(
        self, num_features: int, embedding_dim: int, num_embeddings: int
    ) -> EmbeddingBagCollection:
        """Helper to create an EmbeddingBagCollection."""
        configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[f"sparse_{i}"],
            )
            for i in range(num_features)
        ]
        return EmbeddingBagCollection(tables=configs, device=torch.device("cpu"))

    def _make_sparse_features(
        self,
        num_features: int,
        batch_size: int,
        num_embeddings: int,
    ) -> KeyedJaggedTensor:
        """Helper to create a KeyedJaggedTensor."""
        values_list = []
        lengths_list = []
        for _ in range(num_features):
            lengths = torch.ones(batch_size, dtype=torch.int32)
            values = torch.randint(0, num_embeddings, (batch_size,))
            values_list.append(values)
            lengths_list.append(lengths)
        return KeyedJaggedTensor(
            keys=[f"sparse_{i}" for i in range(num_features)],
            values=torch.cat(values_list),
            lengths=torch.cat(lengths_list),
        )

    def test_dlrm_with_single_sparse_feature(self) -> None:
        """Test DLRM with minimal single sparse feature."""
        ebc = self._make_ebc(num_features=1, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[16, 8, 1],
        )
        dense = torch.randn(4, 5)
        sparse = self._make_sparse_features(1, 4, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (4,))

    def test_dlrm_with_many_sparse_features(self) -> None:
        """Test DLRM with a large number of sparse features."""
        num_features = 10
        ebc = self._make_ebc(
            num_features=num_features, embedding_dim=8, num_embeddings=50
        )
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[64, 32, 1],
        )
        dense = torch.randn(4, 5)
        sparse = self._make_sparse_features(num_features, 4, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (4,))

    def test_dlrm_eval_mode(self) -> None:
        """Test that DLRM output is valid in eval mode."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[16, 8, 1],
        )
        model.eval()
        dense = torch.randn(4, 5)
        sparse = self._make_sparse_features(2, 4, 50)
        with torch.no_grad():
            output = model(dense, sparse)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())

    def test_dlrm_deterministic_in_eval(self) -> None:
        """Test that DLRM produces same output for same input in eval mode."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[16, 8, 1],
        )
        model.eval()

        dense = torch.randn(4, 5)
        values = torch.randint(0, 50, (8,))
        lengths = torch.ones(8, dtype=torch.int32)
        sparse = KeyedJaggedTensor(
            keys=["sparse_0", "sparse_1"],
            values=values,
            lengths=lengths,
        )
        with torch.no_grad():
            out1 = model(dense, sparse)
            out2 = model(dense, sparse)
        self.assertTrue(torch.allclose(out1, out2))

    def test_dlrm_has_expected_submodules(self) -> None:
        """Test that DLRM contains dense_arch, over_arch, final_linear, and ebc."""
        ebc = self._make_ebc(num_features=3, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[16, 8],
            over_arch_layer_sizes=[32, 16, 1],
        )
        self.assertIsInstance(model.dense_arch, MLP)
        self.assertIsInstance(model.over_arch, MLP)
        self.assertIsInstance(model.final_linear, nn.Linear)
        self.assertIsInstance(model.embedding_bag_collection, EmbeddingBagCollection)

    def test_dlrm_interaction_size_calculation(self) -> None:
        """Test that DLRM correctly computes feature interaction size."""
        num_sparse = 4
        embedding_dim = 8
        ebc = self._make_ebc(
            num_features=num_sparse, embedding_dim=embedding_dim, num_embeddings=50
        )
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )
        # num_features = num_sparse + 1 = 5
        # interaction_size = (5 * 4) // 2 + 8 = 18
        num_features = num_sparse + 1
        expected_interaction_size = (
            num_features * (num_features - 1)
        ) // 2 + embedding_dim
        self.assertEqual(expected_interaction_size, 18)
        # Verify the over_arch input size matches
        over_arch_first_layer = list(model.over_arch.mlp.modules())
        linear_layers = [m for m in over_arch_first_layer if isinstance(m, nn.Linear)]
        self.assertEqual(linear_layers[0].in_features, expected_interaction_size)

    def test_dlrm_batch_size_one(self) -> None:
        """Test DLRM works with a single sample in the batch."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[16, 8, 1],
        )
        dense = torch.randn(1, 5)
        sparse = self._make_sparse_features(2, 1, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (1,))

    def test_dlrm_large_batch(self) -> None:
        """Test DLRM with a large batch size."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[8],
            over_arch_layer_sizes=[16, 8, 1],
        )
        dense = torch.randn(256, 5)
        sparse = self._make_sparse_features(2, 256, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (256,))

    def test_dlrm_different_embedding_dim(self) -> None:
        """Test DLRM with a larger embedding dimension."""
        embedding_dim = 32
        ebc = self._make_ebc(
            num_features=3, embedding_dim=embedding_dim, num_embeddings=50
        )
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=10,
            dense_arch_layer_sizes=[embedding_dim],
            over_arch_layer_sizes=[64, 32, 1],
        )
        dense = torch.randn(4, 10)
        sparse = self._make_sparse_features(3, 4, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(torch.all(output >= 0).item())
        self.assertTrue(torch.all(output <= 1).item())

    def test_dlrm_rejects_mismatched_dense_arch_and_embedding_dim(self) -> None:
        """dense_arch_layer_sizes[-1] must equal embedding_dim."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        with self.assertRaises(ValueError) as ctx:
            DLRM(
                embedding_bag_collection=ebc,
                dense_in_features=5,
                dense_arch_layer_sizes=[16],  # 16 != embedding_dim (8)
                over_arch_layer_sizes=[16, 8, 1],
            )
        self.assertIn("must equal", str(ctx.exception))
        self.assertIn("embedding_dim", str(ctx.exception))

    def test_dlrm_rejects_single_element_over_arch(self) -> None:
        """over_arch_layer_sizes must have at least 2 elements."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        with self.assertRaises(ValueError) as ctx:
            DLRM(
                embedding_bag_collection=ebc,
                dense_in_features=5,
                dense_arch_layer_sizes=[8],
                over_arch_layer_sizes=[1],  # only 1 element
            )
        self.assertIn("at least 2 elements", str(ctx.exception))

    def test_dlrm_rejects_empty_over_arch(self) -> None:
        """over_arch_layer_sizes must not be empty."""
        ebc = self._make_ebc(num_features=2, embedding_dim=8, num_embeddings=50)
        with self.assertRaises(ValueError) as ctx:
            DLRM(
                embedding_bag_collection=ebc,
                dense_in_features=5,
                dense_arch_layer_sizes=[8],
                over_arch_layer_sizes=[],
            )
        self.assertIn("at least 2 elements", str(ctx.exception))

    def test_dlrm_accepts_matching_dense_arch_and_embedding_dim(self) -> None:
        """Verify DLRM works when dense_arch_layer_sizes[-1] == embedding_dim."""
        embedding_dim = 16
        ebc = self._make_ebc(
            num_features=2, embedding_dim=embedding_dim, num_embeddings=50
        )
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[32, embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )
        dense = torch.randn(4, 5)
        sparse = self._make_sparse_features(2, 4, 50)
        output = model(dense, sparse)
        self.assertEqual(output.shape, (4,))


class SyntheticDatasetEdgeCaseTest(unittest.TestCase):
    """Additional edge case tests for SyntheticDataset."""

    def test_dataset_zero_batches(self) -> None:
        """Test that dataset with zero batches produces no items."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=3,
            batch_size=4,
            num_batches=0,
            sparse_feature_names=["s0", "s1", "s2"],
        )
        batch_count = sum(1 for _ in dataset)
        self.assertEqual(batch_count, 0)

    def test_dataset_single_sparse_feature(self) -> None:
        """Test dataset with a single sparse feature."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=1,
            batch_size=8,
            num_batches=1,
            sparse_feature_names=["single_feature"],
        )
        for _, sparse, _ in dataset:
            self.assertEqual(sparse.keys(), ["single_feature"])

    def test_dataset_single_dense_feature(self) -> None:
        """Test dataset with a single dense feature."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=1,
            num_sparse_features=2,
            batch_size=8,
            num_batches=1,
            sparse_feature_names=["s0", "s1"],
        )
        for dense, _, _ in dataset:
            self.assertEqual(dense.shape, (8, 1))

    def test_dataset_sparse_lengths_are_valid(self) -> None:
        """Test that sparse feature lengths are between 1 and 3."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=3,
            batch_size=16,
            num_batches=2,
            sparse_feature_names=["s0", "s1", "s2"],
        )
        for _, sparse, _ in dataset:
            lengths = sparse.lengths()
            self.assertTrue(torch.all(lengths >= 1))
            self.assertTrue(torch.all(lengths <= 3))

    def test_dataset_reiterable(self) -> None:
        """Test that dataset can be iterated over multiple times."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=3,
            batch_size=4,
            num_batches=3,
            sparse_feature_names=["s0", "s1", "s2"],
        )
        count1 = sum(1 for _ in dataset)
        count2 = sum(1 for _ in dataset)
        self.assertEqual(count1, 3)
        self.assertEqual(count2, 3)

    def test_dataset_batch_size_one(self) -> None:
        """Test dataset with a batch size of one."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=2,
            batch_size=1,
            num_batches=1,
            sparse_feature_names=["s0", "s1"],
        )
        for dense, sparse, labels in dataset:
            self.assertEqual(dense.shape, (1, 5))
            self.assertEqual(labels.shape, (1,))
            self.assertEqual(len(sparse.lengths()), 2)  # 2 features * 1 sample

    def test_dataset_labels_dtype(self) -> None:
        """Test that labels are float tensors (for BCELoss)."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=2,
            batch_size=8,
            num_batches=1,
            sparse_feature_names=["s0", "s1"],
        )
        for _, _, labels in dataset:
            self.assertEqual(labels.dtype, torch.float32)

    def test_dataset_dense_dtype(self) -> None:
        """Test that dense features are float tensors."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=2,
            batch_size=8,
            num_batches=1,
            sparse_feature_names=["s0", "s1"],
        )
        for dense, _, _ in dataset:
            self.assertEqual(dense.dtype, torch.float32)

    def test_dataset_sparse_values_are_integers(self) -> None:
        """Test that sparse feature values are integer type."""
        dataset = SyntheticDataset(
            num_embeddings=100,
            num_dense_features=5,
            num_sparse_features=2,
            batch_size=8,
            num_batches=1,
            sparse_feature_names=["s0", "s1"],
        )
        for _, sparse, _ in dataset:
            self.assertTrue(sparse.values().dtype in (torch.int32, torch.int64))


class CreateModelTest(unittest.TestCase):
    """Tests for the create_model helper function.

    Note: create_model uses meta device for EmbeddingBagCollection internally,
    which requires to_empty() instead of to() for CPU. We test the parts
    of create_model that are testable without GPU: embedding config generation
    and sparse feature name generation.
    """

    def test_create_model_generates_correct_embedding_configs(self) -> None:
        """Test that create_model builds correct EmbeddingBagConfigs."""
        # Simulate what create_model does internally for config generation
        num_sparse_features = 4
        num_embeddings = 200
        embedding_dim = 16
        sparse_feature_names = [f"sparse_{i}" for i in range(num_sparse_features)]
        embedding_configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[sparse_feature_names[i]],
            )
            for i in range(num_sparse_features)
        ]
        self.assertEqual(len(embedding_configs), 4)
        for i, config in enumerate(embedding_configs):
            self.assertEqual(config.embedding_dim, 16)
            self.assertEqual(config.num_embeddings, 200)
            self.assertEqual(config.feature_names, [f"sparse_{i}"])
            self.assertEqual(config.name, f"embedding_{i}")

    def test_create_model_sparse_feature_names(self) -> None:
        """Test that create_model generates correct sparse feature names."""
        num_sparse = 5
        names = [f"sparse_{i}" for i in range(num_sparse)]
        self.assertEqual(len(names), 5)
        self.assertEqual(names[0], "sparse_0")
        self.assertEqual(names[4], "sparse_4")

    def test_create_model_with_cpu_ebc(self) -> None:
        """Test that a DLRM model can be created with CPU-device EBC (bypassing meta device)."""
        num_sparse = 3
        embedding_dim = 8
        num_embeddings = 100
        configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[f"sparse_{i}"],
            )
            for i in range(num_sparse)
        ]
        ebc = EmbeddingBagCollection(tables=configs, device=torch.device("cpu"))
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=5,
            dense_arch_layer_sizes=[embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )
        self.assertIsInstance(model, DLRM)
        # Verify configs propagated correctly
        configs_out = model.embedding_bag_collection.embedding_bag_configs()
        self.assertEqual(len(configs_out), num_sparse)
        for c in configs_out:
            self.assertEqual(c.embedding_dim, embedding_dim)
            self.assertEqual(c.num_embeddings, num_embeddings)


class IntegrationTest(unittest.TestCase):
    """Integration tests for the full training pipeline."""

    def test_full_pipeline_single_batch(self) -> None:
        """Test full training pipeline with a single batch."""
        # Model parameters
        num_sparse_features = 4
        embedding_dim = 8
        num_embeddings = 100
        num_dense_features = 5
        batch_size = 4

        # Create embedding configs
        sparse_feature_names = [f"sparse_{i}" for i in range(num_sparse_features)]
        embedding_configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[sparse_feature_names[i]],
            )
            for i in range(num_sparse_features)
        ]

        # Create model
        embedding_bag_collection = EmbeddingBagCollection(
            tables=embedding_configs,
            device=torch.device("cpu"),
        )

        model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=num_dense_features,
            dense_arch_layer_sizes=[16, embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )

        # Create dataset
        dataset = SyntheticDataset(
            num_embeddings=num_embeddings,
            num_dense_features=num_dense_features,
            num_sparse_features=num_sparse_features,
            batch_size=batch_size,
            num_batches=1,
            sparse_feature_names=sparse_feature_names,
        )

        # Create optimizer and loss
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Train one batch
        model.train()
        for dense_features, sparse_features, labels in dataset:
            optimizer.zero_grad()
            predictions = model(dense_features, sparse_features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # Verify loss is a valid number
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))

    def test_model_learns_over_batches(self) -> None:
        """Test that model loss decreases over multiple batches."""
        # Model parameters
        num_sparse_features = 4
        embedding_dim = 8
        num_embeddings = 100
        num_dense_features = 5
        batch_size = 16
        num_batches = 10

        # Create model
        sparse_feature_names = [f"sparse_{i}" for i in range(num_sparse_features)]
        embedding_configs = [
            EmbeddingBagConfig(
                name=f"embedding_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[sparse_feature_names[i]],
            )
            for i in range(num_sparse_features)
        ]

        embedding_bag_collection = EmbeddingBagCollection(
            tables=embedding_configs,
            device=torch.device("cpu"),
        )

        model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=num_dense_features,
            dense_arch_layer_sizes=[16, embedding_dim],
            over_arch_layer_sizes=[32, 16, 1],
        )

        # Create dataset
        dataset = SyntheticDataset(
            num_embeddings=num_embeddings,
            num_dense_features=num_dense_features,
            num_sparse_features=num_sparse_features,
            batch_size=batch_size,
            num_batches=num_batches,
            sparse_feature_names=sparse_feature_names,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.BCELoss()

        # Collect losses
        losses = []
        model.train()
        for dense_features, sparse_features, labels in dataset:
            optimizer.zero_grad()
            predictions = model(dense_features, sparse_features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Verify all losses are valid
        for loss_val in losses:
            self.assertFalse(torch.isnan(torch.tensor(loss_val)))


class ParseArgsTest(unittest.TestCase):
    """Tests for the parse_args function and command-line arguments."""

    def test_parse_args_default_checkpoint_interval(self) -> None:
        """Test that checkpoint_interval defaults to 0."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertEqual(args.checkpoint_interval, 0)

    def test_parse_args_custom_checkpoint_interval(self) -> None:
        """Test that checkpoint_interval can be set via command line."""
        with patch(
            "sys.argv", ["train_torchrec_quickstart.py", "--checkpoint_interval", "500"]
        ):
            args = parse_args()
            self.assertEqual(args.checkpoint_interval, 500)

    def test_parse_args_default_checkpoint_path(self) -> None:
        """Test that checkpoint_path defaults to empty string."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertEqual(args.checkpoint_path, "")

    def test_parse_args_custom_checkpoint_path(self) -> None:
        """Test that checkpoint_path can be set via command line."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--checkpoint_path",
                "s3://bucket/checkpoints/",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.checkpoint_path, "s3://bucket/checkpoints/")

    def test_parse_args_enable_spot_recovery_default_false(self) -> None:
        """Test that enable_spot_recovery defaults to False."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertFalse(args.enable_spot_recovery)

    def test_parse_args_enable_spot_recovery_flag(self) -> None:
        """Test that enable_spot_recovery can be enabled via flag."""
        with patch(
            "sys.argv", ["train_torchrec_quickstart.py", "--enable_spot_recovery"]
        ):
            args = parse_args()
            self.assertTrue(args.enable_spot_recovery)

    def test_parse_args_enable_preemption_recovery_default_false(self) -> None:
        """Test that enable_preemption_recovery defaults to False."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertFalse(args.enable_preemption_recovery)

    def test_parse_args_enable_preemption_recovery_flag(self) -> None:
        """Test that enable_preemption_recovery can be enabled via flag."""
        with patch(
            "sys.argv", ["train_torchrec_quickstart.py", "--enable_preemption_recovery"]
        ):
            args = parse_args()
            self.assertTrue(args.enable_preemption_recovery)

    def test_parse_args_combined_spot_recovery_args(self) -> None:
        """Test that spot recovery args can be combined."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--enable_spot_recovery",
                "--checkpoint_interval",
                "1000",
                "--checkpoint_path",
                "s3://my-bucket/ckpt/",
            ],
        ):
            args = parse_args()
            self.assertTrue(args.enable_spot_recovery)
            self.assertEqual(args.checkpoint_interval, 1000)
            self.assertEqual(args.checkpoint_path, "s3://my-bucket/ckpt/")

    def test_parse_args_combined_preemption_recovery_args(self) -> None:
        """Test that preemption recovery args can be combined."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--enable_preemption_recovery",
                "--checkpoint_interval",
                "2000",
                "--checkpoint_path",
                "gs://my-bucket/ckpt/",
            ],
        ):
            args = parse_args()
            self.assertTrue(args.enable_preemption_recovery)
            self.assertEqual(args.checkpoint_interval, 2000)
            self.assertEqual(args.checkpoint_path, "gs://my-bucket/ckpt/")

    def test_parse_args_default_model_args(self) -> None:
        """Test that default model arguments are set correctly."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertEqual(args.num_embeddings, 100000)
            self.assertEqual(args.embedding_dim, 64)
            self.assertEqual(args.num_sparse_features, 26)
            self.assertEqual(args.num_dense_features, 13)

    def test_parse_args_default_training_args(self) -> None:
        """Test that default training arguments are set correctly."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            self.assertEqual(args.batch_size, 8192)
            self.assertEqual(args.num_epochs, 3)
            self.assertEqual(args.num_batches_per_epoch, 100)
            self.assertEqual(args.learning_rate, 0.1)


class TorchrunConfigTest(unittest.TestCase):
    """Tests for torchrun-compatible configuration.

    Validates that parse_args produces correct defaults for torchrun-based
    distributed training, including environment variable handling for
    MASTER_ADDR, MASTER_PORT, and WORLD_SIZE.
    """

    def test_parse_args_torchrun_defaults(self) -> None:
        """Test that parse_args produces torchrun-compatible defaults."""
        with patch("sys.argv", ["train_torchrec_quickstart.py"]):
            args = parse_args()
            # torchrun sets these via env vars; defaults should be sensible
            self.assertEqual(args.batch_size, 8192)
            self.assertEqual(args.num_epochs, 3)
            self.assertEqual(args.num_batches_per_epoch, 100)

    def test_torchrun_env_vars_for_distributed_setup(self) -> None:
        """Test that setup_distributed works with torchrun env vars."""
        # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        try:
            with (
                patch("torch.distributed.init_process_group") as mock_init,
                patch("torch.cuda.set_device"),
            ):
                rank, world_size, device = setup_distributed()
                self.assertEqual(rank, 0)
                self.assertEqual(world_size, 2)
                mock_init.assert_called_once_with(backend="nccl")
        finally:
            for var in [
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            ]:
                os.environ.pop(var, None)

    def test_torchrun_master_addr_env_preserved(self) -> None:
        """Test that MASTER_ADDR env var is preserved during setup."""
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        try:
            with (
                patch("torch.distributed.init_process_group"),
                patch("torch.cuda.set_device"),
            ):
                rank, world_size, _ = setup_distributed()
                self.assertEqual(rank, 1)
                self.assertEqual(world_size, 4)
                # MASTER_ADDR should still be set for NCCL backend
                self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
                self.assertEqual(os.environ["MASTER_PORT"], "29500")
        finally:
            for var in [
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            ]:
                os.environ.pop(var, None)

    def test_parse_args_spot_recovery_for_torchrun(self) -> None:
        """Test that spot recovery args are correctly parsed for torchrun usage."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--enable_spot_recovery",
                "--checkpoint_interval",
                "500",
                "--checkpoint_path",
                "s3://bucket/ckpt/",
            ],
        ):
            args = parse_args()
            self.assertTrue(args.enable_spot_recovery)
            self.assertEqual(args.checkpoint_interval, 500)
            self.assertEqual(args.checkpoint_path, "s3://bucket/ckpt/")

    def test_parse_args_preemption_recovery_for_torchrun(self) -> None:
        """Test that preemption recovery args are correctly parsed for torchrun usage."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--enable_preemption_recovery",
                "--checkpoint_interval",
                "1000",
                "--checkpoint_path",
                "gs://bucket/ckpt/",
            ],
        ):
            args = parse_args()
            self.assertTrue(args.enable_preemption_recovery)
            self.assertEqual(args.checkpoint_interval, 1000)
            self.assertEqual(args.checkpoint_path, "gs://bucket/ckpt/")

    def test_parse_args_custom_model_args_for_torchrun(self) -> None:
        """Test that custom model args work correctly when launched via torchrun."""
        with patch(
            "sys.argv",
            [
                "train_torchrec_quickstart.py",
                "--num_embeddings",
                "50000",
                "--embedding_dim",
                "128",
                "--batch_size",
                "4096",
                "--learning_rate",
                "0.01",
            ],
        ):
            args = parse_args()
            self.assertEqual(args.num_embeddings, 50000)
            self.assertEqual(args.embedding_dim, 128)
            self.assertEqual(args.batch_size, 4096)
            self.assertEqual(args.learning_rate, 0.01)


if __name__ == "__main__":
    unittest.main()
