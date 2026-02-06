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

import os
import unittest
from typing import List, Tuple

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from hypothesis import given, Phase, settings, strategies as st, Verbosity
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
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import ShardingType
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
    def test_learning_rate_propagation(self) -> None:
        """Test that learning rate changes are propagated to module via zero_grad and step."""
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

        # Test zero_grad propagates learning rate
        optimizer.param_groups[0]["lr"] = 0.05
        optimizer.zero_grad()
        self.assertEqual(emb_bag._emb_module.learning_rate, 0.05)

        # Test step propagates learning rate
        optimizer.param_groups[0]["lr"] = 0.1
        optimizer.step()
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


@skip_if_asan_class
class TritonTBEColumnWiseShardingTest(ModelParallelTestShared):
    """
    Tests for Triton TBE with column-wise sharding.

    Mimics test_sharding_nccl_twrw from ModelParallelHierarchicalTest but focuses on
    column-wise sharding with EXACT_ROWWISE_ADAGRAD optimizer for Triton TBE.

    NOTE:
        Requires at least 2 GPUs to test.
    """

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.just(ShardingType.COLUMN_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED_TRITON.value,
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=2,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    @unittest.skip(
        "Skip due to Triton PTX codegen error with clusterlaunchcontrol instructions on current CUDA toolchain"
    )
    def test_sharding_cw_triton_tbe(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        pooling: PoolingType,
    ) -> None:
        """
        Test column-wise sharding with Triton TBE and EXACT_ROWWISE_ADAGRAD optimizer.

        This test validates that:
        1. Column-wise sharding works correctly with Triton TBE
        2. The shard-aware parameter keys prevent CombinedOptimizer conflicts
        3. Forward/backward passes produce correct results
        """
        # Enable detailed distributed debug for non-even collectives
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

        world_size = 2

        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    fused_params={
                        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD,
                        "learning_rate": 0.01,
                        "eps": 0.1,
                    },
                    device=torch.device("cuda"),
                ),
            ],
            backend="nccl",
            world_size=world_size,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            pooling=pooling,
        )


@skip_if_asan_class
class TritonCUDANumericAlignmentTest(unittest.TestCase):
    """
    Tests for verifying numeric alignment between Triton TBE and CUDA TBE.

    These tests ensure that the Triton-based TBE implementation produces
    numerically equivalent results to the CUDA-based SplitTableBatchedEmbeddingBagsCodegen
    for both forward and backward passes.
    """

    @staticmethod
    def _create_cuda_tbe(
        embedding_specs: List[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        optimizer: OptimType = OptimType.EXACT_SGD,
        learning_rate: float = 0.01,
        eps: float = 0.1,
    ) -> torch.nn.Module:
        """Create a CUDA TBE with the given specs."""
        from fbgemm_gpu.split_embedding_configs import SparseType
        from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
            EmbeddingLocation,
            PoolingMode,
        )
        from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
            ComputeDevice,
            SplitTableBatchedEmbeddingBagsCodegen,
        )

        emb = SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for (E, D) in embedding_specs
            ],
            weights_precision=SparseType.from_dtype(dtype),
            output_dtype=SparseType.from_dtype(dtype),
            stochastic_rounding=False,
            optimizer=optimizer,
            learning_rate=learning_rate,
            eps=eps,
            pooling_mode=PoolingMode.SUM,
        )
        return emb

    @staticmethod
    def _create_triton_tbe(
        embedding_specs: List[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        learning_rate: float = 0.01,
        eps: float = 0.1,
        optimizer: OptimType = OptimType.EXACT_SGD,
    ) -> torch.nn.Module:
        """Create a Triton TBE with the given specs."""
        from deeplearning.fbgemm.fbgemm_gpu.fb.triton.triton_table_batched_embeddings import (
            TritonTableBatchedEmbeddingBags,
        )

        emb = TritonTableBatchedEmbeddingBags(
            embedding_specs=embedding_specs,
            weights_precision=dtype,
            learning_rate=learning_rate,
            eps=eps,
            optimizer=optimizer,
        )
        return emb

    @staticmethod
    def _generate_inputs(
        hash_sizes: List[int],
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random indices and offsets for TBE forward."""
        import random

        T = len(hash_sizes)
        offsets = [0]
        indices_per_table = []

        for t in range(T):
            len_sum = 0
            for _ in range(batch_size):
                length = random.randint(0, max_len)
                len_sum += length
                offsets.append(offsets[-1] + length)

            n_rows = hash_sizes[t]
            indices_per_table.append(
                torch.randint(n_rows, [len_sum], dtype=torch.int64, device=device)
            )

        indices = torch.cat(indices_per_table, dim=0)
        offsets_tensor = torch.tensor(offsets, dtype=torch.int64, device=device)

        return indices, offsets_tensor

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for numeric alignment tests",
    )
    # pyre-ignore[56]: Invalid decoration
    @given(
        bag_size=st.integers(3, 15),
        batch_size=st.integers(5, 10),
        num_tables=st.integers(2, 5),
        dtype=st.sampled_from([torch.float32]),
        iters=st.integers(3, 5),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_forward_numeric_alignment_fp32(
        self,
        bag_size: int,
        batch_size: int,
        num_tables: int,
        dtype: torch.dtype,
        iters: int,
    ) -> None:
        """Test that Triton TBE forward pass produces same results as CUDA TBE (FP32)."""
        import random

        device = torch.device("cuda:0")

        # Create embedding specs with dims divisible by 4
        embedding_specs = [
            (random.randint(50, 200), random.randrange(16, 128, 4))
            for _ in range(num_tables)
        ]

        cuda_emb = self._create_cuda_tbe(embedding_specs, dtype=dtype)
        triton_emb = self._create_triton_tbe(embedding_specs, dtype=dtype)

        # Initialize weights uniformly
        cuda_emb.init_embedding_weights_uniform(-1.0, 1.0)
        # Copy CUDA weights to Triton TBE
        triton_emb.weight.data.copy_(cuda_emb.weights_dev)

        hash_sizes = [spec[0] for spec in embedding_specs]

        for _ in range(iters):
            indices, offsets = self._generate_inputs(
                hash_sizes, batch_size, bag_size, device
            )

            cuda_output = cuda_emb(indices, offsets)
            triton_output = triton_emb(indices, offsets)

            self.assertTrue(
                torch.allclose(triton_output, cuda_output, rtol=1e-4, atol=1e-4),
                f"Forward pass mismatch: max diff = {(triton_output - cuda_output).abs().max().item()}",
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for numeric alignment tests",
    )
    # pyre-ignore[56]: Invalid decoration
    @given(
        bag_size=st.integers(3, 15),
        batch_size=st.integers(5, 10),
        num_tables=st.integers(2, 5),
        iters=st.integers(3, 5),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_forward_numeric_alignment_fp16(
        self,
        bag_size: int,
        batch_size: int,
        num_tables: int,
        iters: int,
    ) -> None:
        """Test that Triton TBE forward pass produces same results as CUDA TBE (FP16)."""
        import random

        device = torch.device("cuda:0")
        dtype = torch.float16

        embedding_specs = [
            (random.randint(50, 200), random.randrange(16, 128, 4))
            for _ in range(num_tables)
        ]

        cuda_emb = self._create_cuda_tbe(embedding_specs, dtype=dtype)
        triton_emb = self._create_triton_tbe(embedding_specs, dtype=dtype)

        cuda_emb.init_embedding_weights_uniform(-1.0, 1.0)
        triton_emb.weight.data.copy_(cuda_emb.weights_dev)

        hash_sizes = [spec[0] for spec in embedding_specs]

        for _ in range(iters):
            indices, offsets = self._generate_inputs(
                hash_sizes, batch_size, bag_size, device
            )

            cuda_output = cuda_emb(indices, offsets)
            triton_output = triton_emb(indices, offsets)

            # FP16 has lower precision, so use larger tolerance
            self.assertTrue(
                torch.allclose(triton_output, cuda_output, rtol=1e-2, atol=1e-2),
                f"Forward pass mismatch (FP16): max diff = {(triton_output - cuda_output).abs().max().item()}",
            )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for numeric alignment tests",
    )
    # pyre-ignore[56]: Invalid decoration
    @given(
        bag_size=st.integers(3, 10),
        batch_size=st.integers(5, 8),
        num_tables=st.integers(2, 3),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=5,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    @unittest.skip(
        "Skip due to Triton PTX codegen error with clusterlaunchcontrol instructions on current CUDA toolchain"
    )
    def test_backward_rowwise_adagrad_numeric_alignment(
        self,
        bag_size: int,
        batch_size: int,
        num_tables: int,
    ) -> None:
        """Test that backward pass with EXACT_ROWWISE_ADAGRAD produces aligned weight updates.

        Note: Triton TBE and CUDA TBE use different implementations of rowwise adagrad.
        The momentum accumulation and adaptive learning rate calculations can lead to
        numeric differences. We test a single iteration to verify the implementations
        are behaviorally aligned before weights diverge too much.
        """
        import random

        device = torch.device("cuda:0")
        dtype = torch.float32
        lr = 0.1
        eps = 0.1

        embedding_specs = [
            (random.randint(30, 80), random.randrange(16, 64, 4))
            for _ in range(num_tables)
        ]

        cuda_emb = self._create_cuda_tbe(
            embedding_specs,
            dtype=dtype,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=lr,
            eps=eps,
        )
        triton_emb = self._create_triton_tbe(
            embedding_specs,
            dtype=dtype,
            learning_rate=lr,
            eps=eps,
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        )

        cuda_emb.init_embedding_weights_uniform(-1.0, 1.0)
        triton_emb.weight.data.copy_(cuda_emb.weights_dev)

        B = batch_size
        Ds = [spec[1] for spec in embedding_specs]
        hash_sizes = [spec[0] for spec in embedding_specs]

        fwd_tol = 1e-3
        # Backward tolerance: After fixing the signed/unsigned right-shift bug,
        # Triton and CUDA TBE should produce very similar results. Small differences
        # may still occur due to floating-point accumulation order, but should be
        # within 1e-5 tolerance for float32 precision.
        bwd_tol = 1e-5

        # Only run a single iteration to check numeric alignment before drift accumulates
        indices, offsets = self._generate_inputs(
            hash_sizes, batch_size, bag_size, device
        )

        cuda_output = cuda_emb(indices, offsets)
        triton_output = triton_emb(indices, offsets)

        # Check forward alignment
        self.assertTrue(
            torch.allclose(triton_output, cuda_output, rtol=fwd_tol, atol=fwd_tol),
            f"Forward mismatch (Adagrad): max diff = {(triton_output - cuda_output).abs().max().item()}",
        )

        # Run backward
        grad_output = torch.randn(B, sum(Ds), device=device, dtype=dtype)

        cuda_output.backward(grad_output)
        triton_output.backward(grad_output)

        # Check weight alignment after optimizer update with lenient tolerance
        max_diff = (triton_emb.weight - cuda_emb.weights_dev).abs().max().item()
        self.assertTrue(
            max_diff < bwd_tol,
            f"Weight mismatch after Adagrad backward: max diff = {max_diff} (tolerance = {bwd_tol})",
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "CUDA is required for numeric alignment tests",
    )
    def test_forward_edge_cases(self) -> None:
        """Test forward pass with various edge cases: empty bags, repeated indices, large batch."""
        device = torch.device("cuda:0")
        dtype = torch.float32

        # Test 1: Empty bags (zero-length lookups)
        embedding_specs = [(100, 64)]
        cuda_emb = self._create_cuda_tbe(embedding_specs, dtype=dtype)
        triton_emb = self._create_triton_tbe(embedding_specs, dtype=dtype)

        torch.manual_seed(42)
        weights = torch.randn(100 * 64, dtype=dtype, device=device)
        cuda_emb.weights_dev.copy_(weights)
        triton_emb.weight.data.copy_(weights)

        # Indices and offsets with empty bags (samples 1 and 3 have no indices)
        indices = torch.tensor([0, 5, 10, 15], dtype=torch.int64, device=device)
        offsets = torch.tensor([0, 2, 2, 4, 4], dtype=torch.int64, device=device)

        cuda_output = cuda_emb(indices, offsets)
        triton_output = triton_emb(indices, offsets)

        self.assertTrue(
            torch.allclose(triton_output, cuda_output, rtol=1e-5, atol=1e-5),
            f"Empty bags forward mismatch: max diff = {(triton_output - cuda_output).abs().max().item()}",
        )
        # Verify empty bags are zeros
        self.assertTrue(
            torch.allclose(cuda_output[1], torch.zeros(64, device=device, dtype=dtype)),
            "Empty bag should produce zeros",
        )

        # Test 2: Repeated indices (same row accessed multiple times)
        indices = torch.tensor(
            [5, 5, 5, 10, 10, 15, 20], dtype=torch.int64, device=device
        )
        offsets = torch.tensor([0, 3, 5, 7], dtype=torch.int64, device=device)

        cuda_output = cuda_emb(indices, offsets)
        triton_output = triton_emb(indices, offsets)

        self.assertTrue(
            torch.allclose(triton_output, cuda_output, rtol=1e-5, atol=1e-5),
            f"Repeated indices forward mismatch: max diff = {(triton_output - cuda_output).abs().max().item()}",
        )

        # Test 3: Large batch with multiple tables
        embedding_specs = [(1000, 64), (500, 128)]
        cuda_emb = self._create_cuda_tbe(embedding_specs, dtype=dtype)
        triton_emb = self._create_triton_tbe(embedding_specs, dtype=dtype)

        cuda_emb.init_embedding_weights_uniform(-1.0, 1.0)
        triton_emb.weight.data.copy_(cuda_emb.weights_dev)

        hash_sizes = [spec[0] for spec in embedding_specs]
        indices, offsets = self._generate_inputs(hash_sizes, 64, 20, device)

        cuda_output = cuda_emb(indices, offsets)
        triton_output = triton_emb(indices, offsets)

        self.assertTrue(
            torch.allclose(triton_output, cuda_output, rtol=1e-4, atol=1e-4),
            f"Large batch forward mismatch: max diff = {(triton_output - cuda_output).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()
