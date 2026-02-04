#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Unit tests for Triton TBE integration with TritonBatchedFusedEmbeddingBag.

This module tests the integration of TritonTableBatchedEmbeddingBags with TorchRec's
TritonBatchedFusedEmbeddingBag class, verifying that the Triton-based embedding lookup
works correctly for forward pass, backward pass, optimizer updates, and weight management.
"""

import unittest
from typing import List

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch.distributed._shard.sharded_tensor.metadata import ShardMetadata
from torchrec.distributed.batched_embedding_kernel import (
    TritonBatchedFusedEmbeddingBag,
    TritonEmbeddingFusedOptimizer,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import DataType, PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _create_embedding_table_config(
    name: str,
    num_embeddings: int,
    embedding_dim: int,
    feature_names: List[str],
    data_type: DataType = DataType.FP32,
    pooling: PoolingType = PoolingType.SUM,
) -> ShardedEmbeddingTable:
    """Create a ShardedEmbeddingTable configuration for testing."""
    return ShardedEmbeddingTable(
        name=name,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        feature_names=feature_names,
        pooling=pooling,
        data_type=data_type,
        has_feature_processor=False,
        local_rows=num_embeddings,
        local_cols=embedding_dim,
        compute_kernel=EmbeddingComputeKernel.FUSED_TRITON,
    )


def _create_grouped_embedding_config(
    tables: List[ShardedEmbeddingTable],
    data_type: DataType = DataType.FP32,
    pooling: PoolingType = PoolingType.SUM,
    optimizer: OptimType = OptimType.EXACT_SGD,
    learning_rate: float = 0.01,
) -> GroupedEmbeddingConfig:
    """Create a GroupedEmbeddingConfig for testing."""
    return GroupedEmbeddingConfig(
        data_type=data_type,
        pooling=pooling,
        is_weighted=False,
        has_feature_processor=False,
        compute_kernel=EmbeddingComputeKernel.FUSED_TRITON,
        embedding_tables=tables,
        fused_params={
            "optimizer": optimizer,
            "learning_rate": learning_rate,
        },
    )


def _create_kjt(
    keys: List[str],
    values: List[int],
    lengths: List[int],
    device: torch.device,
) -> KeyedJaggedTensor:
    """Create a KeyedJaggedTensor for testing."""
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.tensor(values, dtype=torch.long, device=device),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )


@skip_if_asan_class
class TritonTBEIntegrationTest(unittest.TestCase):
    """Tests for Triton TBE integration with TritonBatchedFusedEmbeddingBag."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_forward_single_table(self) -> None:
        """Test forward pass with a single embedding table."""
        device = torch.device("cuda:0")

        # Setup: Create a single embedding table with dim divisible by 4
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,  # Must be divisible by 4
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table])

        # Execute: Create TritonBatchedFusedEmbeddingBag and run forward pass
        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Create input KeyedJaggedTensor
        kjt = _create_kjt(
            keys=["feature_0"],
            values=[0, 1, 2, 3, 4],  # indices
            lengths=[2, 3],  # 2 indices for first sample, 3 for second
            device=device,
        )

        output = emb_bag(kjt)

        # Assert: Check output shape
        self.assertEqual(output.shape, (2, 64))  # batch_size=2, dim=64
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_forward_multiple_tables(self) -> None:
        """Test forward pass with multiple embedding tables."""
        device = torch.device("cuda:0")

        # Setup: Create multiple embedding tables
        table_0 = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        table_1 = _create_embedding_table_config(
            name="table_1",
            num_embeddings=200,
            embedding_dim=128,
            feature_names=["feature_1"],
        )
        config = _create_grouped_embedding_config([table_0, table_1])

        # Execute: Create TritonBatchedFusedEmbeddingBag and run forward pass
        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Create input for both features
        kjt = _create_kjt(
            keys=["feature_0", "feature_1"],
            values=[0, 1, 2, 10, 20, 30, 40],
            lengths=[2, 1, 2, 2],  # 2 samples, 2 features
            device=device,
        )

        output = emb_bag(kjt)

        # Assert: Output dimension should be sum of embedding dims
        self.assertEqual(output.shape, (2, 64 + 128))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    @unittest.skip(
        "Skip due to Triton PTX codegen error with clusterlaunchcontrol instructions on current CUDA toolchain"
    )
    def test_backward_pass(self) -> None:
        """Test backward pass and gradient computation."""
        device = torch.device("cuda:0")

        # Setup: Create embedding table
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table])

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Create input
        kjt = _create_kjt(
            keys=["feature_0"],
            values=[0, 1, 2],
            lengths=[2, 1],
            device=device,
        )

        # Execute: Forward and backward pass
        output = emb_bag(kjt)
        loss = output.sum()
        loss.backward()

        # Assert: Backward pass should complete without error
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_split_embedding_weights(self) -> None:
        """Test that split_embedding_weights returns correct weight tensors."""
        device = torch.device("cuda:0")

        # Setup: Create multiple tables with different dimensions
        table_0 = _create_embedding_table_config(
            name="table_0",
            num_embeddings=50,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        table_1 = _create_embedding_table_config(
            name="table_1",
            num_embeddings=100,
            embedding_dim=128,
            feature_names=["feature_1"],
        )
        config = _create_grouped_embedding_config([table_0, table_1])

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Execute: Get split embedding weights
        weights = emb_bag.split_embedding_weights()

        # Assert: Check that we get correct number of weight tensors with correct shapes
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].shape, (50, 64))
        self.assertEqual(weights[1].shape, (100, 128))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_fused_optimizer_learning_rate(self) -> None:
        """Test that fused optimizer correctly updates learning rate."""
        device = torch.device("cuda:0")

        # Setup: Create embedding with specific learning rate
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config(
            [table],
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.05,
        )

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Execute: Get optimizer and check learning rate
        optimizer = emb_bag.fused_optimizer

        # Assert: Optimizer should have correct learning rate
        self.assertIsInstance(optimizer, TritonEmbeddingFusedOptimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.05)

        # Update learning rate through optimizer
        optimizer.param_groups[0]["lr"] = 0.1
        optimizer.step()

        # Assert: Learning rate should be updated on the underlying module
        self.assertEqual(emb_bag._emb_module.learning_rate, 0.1)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_flush_and_purge(self) -> None:
        """Test flush() and purge() methods (no-ops for Triton TBE)."""
        device = torch.device("cuda:0")

        # Setup: Create embedding
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table])

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Execute: Call flush and purge (should not raise exceptions)
        emb_bag.flush()
        emb_bag.purge()

        # Assert: Methods should complete without error (they are no-ops for Triton TBE)
        self.assertIsNotNone(emb_bag)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_named_parameters(self) -> None:
        """Test that named_parameters returns proper parameter dict."""
        device = torch.device("cuda:0")

        # Setup: Create embedding with known table names
        table_0 = _create_embedding_table_config(
            name="my_table",
            num_embeddings=50,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table_0])

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Execute: Get named parameters
        named_params = dict(emb_bag.named_parameters())

        # Assert: Should contain weight parameter
        self.assertIn("my_table.weight", named_params)
        self.assertEqual(named_params["my_table.weight"].shape, (50, 64))

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_cuda_device_requirement(self) -> None:
        """Test that TritonBatchedFusedEmbeddingBag requires CUDA device."""
        # Setup: Create config
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table])

        # Execute & Assert: Should raise assertion error for non-CUDA device
        with self.assertRaises(AssertionError):
            TritonBatchedFusedEmbeddingBag(
                config=config,
                pg=None,
                device=torch.device("cpu"),
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_embedding_dim_divisible_by_4_requirement(self) -> None:
        """Test that embedding dimension must be divisible by 4."""
        device = torch.device("cuda:0")

        # Setup: Create config with embedding dim NOT divisible by 4
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=65,  # Not divisible by 4
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table])

        # Execute & Assert: Should raise assertion error
        with self.assertRaises(AssertionError):
            TritonBatchedFusedEmbeddingBag(
                config=config,
                pg=None,
                device=device,
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    @unittest.skip(
        "Skip due to Triton PTX codegen error with clusterlaunchcontrol instructions on current CUDA toolchain"
    )
    def test_sgd_optimizer_weight_update(self) -> None:
        """Test that SGD optimizer updates weights correctly."""
        device = torch.device("cuda:0")

        # Setup: Create embedding with SGD optimizer
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config(
            [table],
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.1,
        )

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        # Get initial weights for comparison
        initial_weights = emb_bag.split_embedding_weights()[0].clone()

        # Create input and run forward + backward multiple times to accumulate updates
        kjt = _create_kjt(
            keys=["feature_0"],
            values=[0, 1, 2],
            lengths=[2, 1],
            device=device,
        )

        # Run forward/backward pass
        output = emb_bag(kjt)
        loss = output.sum()
        loss.backward()

        # Assert: At least some weights should be updated
        updated_weights = emb_bag.split_embedding_weights()[0]

        # Check that at least one row that was accessed has changed
        any_updated = False
        for idx in [0, 1, 2]:
            if not torch.allclose(initial_weights[idx], updated_weights[idx]):
                any_updated = True
                break

        self.assertTrue(
            any_updated,
            "At least one accessed weight row should have been updated",
        )


@skip_if_asan_class
class TritonEmbeddingFusedOptimizerTest(unittest.TestCase):
    """Tests for TritonEmbeddingFusedOptimizer class."""

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_zero_grad(self) -> None:
        """Test that zero_grad updates learning rate on module."""
        device = torch.device("cuda:0")

        # Setup: Create embedding
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table], learning_rate=0.01)

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        optimizer = emb_bag.fused_optimizer

        # Execute: Change learning rate and call zero_grad
        optimizer.param_groups[0]["lr"] = 0.05
        optimizer.zero_grad()

        # Assert: Learning rate should be propagated to module
        self.assertEqual(emb_bag._emb_module.learning_rate, 0.05)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_step(self) -> None:
        """Test that step updates learning rate on module."""
        device = torch.device("cuda:0")

        # Setup: Create embedding
        table = _create_embedding_table_config(
            name="table_0",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table], learning_rate=0.01)

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        optimizer = emb_bag.fused_optimizer

        # Execute: Change learning rate and call step
        optimizer.param_groups[0]["lr"] = 0.1
        optimizer.step()

        # Assert: Learning rate should be propagated to module
        self.assertEqual(emb_bag._emb_module.learning_rate, 0.1)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_shard_aware_param_keys_without_metadata(self) -> None:
        """Test that optimizer creates standard param keys when no shard metadata."""
        device = torch.device("cuda:0")

        # Setup: Create embedding without shard metadata (single-GPU case)
        table = _create_embedding_table_config(
            name="my_table",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
        )
        config = _create_grouped_embedding_config([table], learning_rate=0.01)

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        optimizer = emb_bag.fused_optimizer

        # Assert: Params should use standard naming (no shard offset suffix)
        param_keys = list(optimizer.params.keys())
        self.assertEqual(param_keys, ["my_table.weight"])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_shard_aware_param_keys_with_metadata(self) -> None:
        """Test that optimizer creates shard-aware param keys when shard metadata is present."""
        device = torch.device("cuda:0")

        # Setup: Create embedding table with shard metadata (simulating column-wise sharding)
        shard_metadata = ShardMetadata(
            shard_offsets=[0, 32],  # row_offset=0, col_offset=32
            shard_sizes=[100, 32],
            placement="rank:0/cuda:0",
        )
        table = ShardedEmbeddingTable(
            name="sharded_table",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
            pooling=PoolingType.SUM,
            data_type=DataType.FP32,
            has_feature_processor=False,
            local_rows=100,
            local_cols=32,  # Sharded column dimension (half of 64)
            compute_kernel=EmbeddingComputeKernel.FUSED_TRITON,
            local_metadata=shard_metadata,
        )
        config = _create_grouped_embedding_config([table], learning_rate=0.01)

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        optimizer = emb_bag.fused_optimizer

        # Assert: Params should use shard-aware naming with offset suffix
        param_keys = list(optimizer.params.keys())
        self.assertEqual(param_keys, ["sharded_table.weight.0_32"])

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for Triton TBE tests",
    )
    def test_multiple_shards_no_duplicate_keys(self) -> None:
        """Test that multiple shards from the same table get unique param keys."""
        device = torch.device("cuda:0")

        # Setup: Create two shard configs for the same table (simulating column-wise sharding)
        shard_metadata_0 = ShardMetadata(
            shard_offsets=[0, 0],  # First shard: row_offset=0, col_offset=0
            shard_sizes=[100, 32],
            placement="rank:0/cuda:0",
        )
        shard_metadata_1 = ShardMetadata(
            shard_offsets=[0, 32],  # Second shard: row_offset=0, col_offset=32
            shard_sizes=[100, 32],
            placement="rank:0/cuda:0",
        )

        table_shard_0 = ShardedEmbeddingTable(
            name="same_table",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
            pooling=PoolingType.SUM,
            data_type=DataType.FP32,
            has_feature_processor=False,
            local_rows=100,
            local_cols=32,
            compute_kernel=EmbeddingComputeKernel.FUSED_TRITON,
            local_metadata=shard_metadata_0,
        )
        table_shard_1 = ShardedEmbeddingTable(
            name="same_table",
            num_embeddings=100,
            embedding_dim=64,
            feature_names=["feature_0"],
            pooling=PoolingType.SUM,
            data_type=DataType.FP32,
            has_feature_processor=False,
            local_rows=100,
            local_cols=32,
            compute_kernel=EmbeddingComputeKernel.FUSED_TRITON,
            local_metadata=shard_metadata_1,
        )

        config = _create_grouped_embedding_config(
            [table_shard_0, table_shard_1], learning_rate=0.01
        )

        emb_bag = TritonBatchedFusedEmbeddingBag(
            config=config,
            pg=None,
            device=device,
        )

        optimizer = emb_bag.fused_optimizer

        # Assert: Both shards should have unique param keys (no duplicates)
        param_keys = list(optimizer.params.keys())
        self.assertEqual(len(param_keys), 2)
        self.assertIn("same_table.weight.0_0", param_keys)
        self.assertIn("same_table.weight.0_32", param_keys)


if __name__ == "__main__":
    unittest.main()
