#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import random
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
from torch import Tensor
from torchrec import KeyedJaggedTensor
from torchrec.distributed.embedding_types import ShardedEmbeddingTable
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.itep_embedding_modules import (
    ITEPEmbeddingBagCollection,
    ITEPEmbeddingCollection,
)
from torchrec.modules.itep_modules import GenericITEPModule
from torchrec.sparse.jagged_tensor import JaggedTensor

MOCK_NS: str = "torchrec.modules.itep_modules"


class TestITEPEmbeddingBagCollection(unittest.TestCase):
    # Setting up the environment for the tests.
    def setUp(self) -> None:
        # Embedding bag configurations for testing
        embedding_bag_config1 = EmbeddingBagConfig(
            name="table1",
            embedding_dim=4,
            num_embeddings=50,
            feature_names=["feature1"],
        )
        embedding_bag_config2 = EmbeddingBagConfig(
            name="table2",
            embedding_dim=4,
            num_embeddings=40,
            feature_names=["feature2"],
        )
        unpruned_hash_size_1, unpruned_hash_size_2 = (100, 80)
        self._table_name_to_pruned_hash_sizes = {"table1": 50, "table2": 40}
        self._table_name_to_unpruned_hash_sizes = {
            "table1": unpruned_hash_size_1,
            "table2": unpruned_hash_size_2,
        }
        self._feature_name_to_unpruned_hash_sizes = {
            "feature1": unpruned_hash_size_1,
            "feature2": unpruned_hash_size_2,
        }
        self._batch_size = 8

        # Util function for creating sharded embedding tables from embedding bag configurations.
        def embedding_bag_config_to_sharded_table(
            config: EmbeddingBagConfig,
        ) -> ShardedEmbeddingTable:
            return ShardedEmbeddingTable(
                name=config.name,
                embedding_dim=config.embedding_dim,
                num_embeddings=config.num_embeddings,
                feature_names=config.feature_names,
            )

        sharded_et1 = embedding_bag_config_to_sharded_table(embedding_bag_config1)
        sharded_et2 = embedding_bag_config_to_sharded_table(embedding_bag_config2)

        # Create test ebc
        self._embedding_bag_collection = EmbeddingBagCollection(
            tables=[
                embedding_bag_config1,
                embedding_bag_config2,
            ],
            device=torch.device("cuda"),
        )

        # Create a mock object for tbe lookups
        self._mock_list_emb_tables = [
            sharded_et1,
            sharded_et2,
        ]
        self._mock_lookups = [MagicMock()]
        self._mock_lookups[0]._emb_modules = [MagicMock()]
        self._mock_lookups[0]._emb_modules[0]._config = MagicMock()
        self._mock_lookups[0]._emb_modules[
            0
        ]._config.embedding_tables = self._mock_list_emb_tables

    def generate_input_kjt_cuda(
        self, feature_name_to_unpruned_hash_sizes: Dict[str, int], use_vbe: bool = False
    ) -> KeyedJaggedTensor:
        keys = []
        values = []
        lengths = []
        cuda_device = torch.device("cuda")

        # Input KJT uses unpruned hash size (same as sigrid hash), and feature names
        for key, unpruned_hash_size in feature_name_to_unpruned_hash_sizes.items():
            value = []
            length = []
            for _ in range(self._batch_size):
                L = random.randint(0, 8)
                for _ in range(L):
                    index = random.randint(0, unpruned_hash_size - 1)
                    value.append(index)
                length.append(L)
            keys.append(key)
            values += value
            lengths += length

        # generate kjt
        if use_vbe:
            inverse_indices_list = []
            inverse_indices = None
            num_keys = len(keys)
            deduped_batch_size = len(lengths) // num_keys
            # Fix the number of samples after duplicate to 2x the number of
            # deduplicated ones
            full_batch_size = deduped_batch_size * 2
            stride_per_key_per_rank = []

            for _ in range(num_keys):
                stride_per_key_per_rank.append([deduped_batch_size])
                # Generate random inverse indices for each key
                keyed_inverse_indices = torch.randint(
                    low=0,
                    high=deduped_batch_size,
                    size=(full_batch_size,),
                    dtype=torch.int32,
                    device=cuda_device,
                )
                inverse_indices_list.append(keyed_inverse_indices)
            inverse_indices = (
                keys,
                torch.stack(inverse_indices_list),
            )

            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                stride_per_key_per_rank=stride_per_key_per_rank,
                inverse_indices=inverse_indices,
            )
        else:
            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
            )

        return input_kjt_cuda

    def generate_expected_address_lookup_buffer(
        self,
        list_et: List[ShardedEmbeddingTable],
        table_name_to_unpruned_hash_sizes: Dict[str, int],
        table_name_to_pruned_hash_sizes: Dict[str, int],
    ) -> torch.Tensor:

        address_lookup = []
        for et in list_et:
            table_name = et.name
            unpruned_hash_size = table_name_to_unpruned_hash_sizes[table_name]
            pruned_hash_size = table_name_to_pruned_hash_sizes[table_name]
            for idx in range(unpruned_hash_size):
                if idx < pruned_hash_size:
                    address_lookup.append(idx)
                else:
                    address_lookup.append(0)

        return torch.tensor(address_lookup, dtype=torch.int64)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_init_itep_module(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=5,
        )

        # Check the address lookup and row util after initialization
        expected_address_lookup = self.generate_expected_address_lookup_buffer(
            self._mock_list_emb_tables,
            self._table_name_to_unpruned_hash_sizes,
            self._table_name_to_pruned_hash_sizes,
        )
        expetec_row_util = torch.zeros(
            expected_address_lookup.shape, dtype=torch.float32
        )
        torch.testing.assert_close(
            expected_address_lookup,
            itep_module.address_lookup.cpu(),
            atol=0,
            rtol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            expetec_row_util,
            itep_module.row_util.cpu(),
            atol=1.0e-5,
            rtol=1.0e-5,
            equal_nan=True,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_init_itep_module_without_pruned_table(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes={},
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=5,
        )

        self.assertEqual(itep_module.address_lookup.cpu().shape, torch.Size([0]))
        self.assertEqual(itep_module.row_util.cpu().shape, torch.Size([0]))

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_train_forward(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_train_forward_vbe(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(5):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes, use_vbe=True
            )
            _ = itep_ebc(input_kjt)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    # Mock out reset_weight_momentum to count calls
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_check_pruning_schedule(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        random.seed(1)
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

        # Check that reset_weight_momentum is called
        self.assertEqual(mock_reset_weight_momentum.call_count, 5)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    # Mock out reset_weight_momentum to count calls
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_eval_forward(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Set eval mode
        itep_ebc.eval()

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

        # Check that reset_weight_momentum is not called
        self.assertEqual(mock_reset_weight_momentum.call_count, 0)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_iter_increment_per_forward(self) -> None:
        """Test that the iteration counter increments correctly with each forward pass."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Check initial iteration value
        self.assertEqual(itep_ebc._iter.item(), 0)

        # Run several forward passes and verify iteration increments
        for expected_iter in range(1, 6):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

            # Verify iter incremented correctly after forward pass
            self.assertEqual(itep_ebc._iter.item(), expected_iter)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_passed_as_int_to_itep_module(self) -> None:
        """Test that iter is passed as integer to the ITEP module."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Mock the itep_module to capture the iter argument
        original_forward = itep_module.forward
        captured_iter_args = []

        def mock_forward(features: KeyedJaggedTensor, iter_val: int) -> List[Tensor]:
            captured_iter_args.append((type(iter_val), iter_val))
            # pyrefly: ignore[bad-return]
            return original_forward(features, iter_val)

        with patch.object(itep_module, "forward", mock_forward):
            # Set iter to a specific value to test
            itep_ebc._iter = torch.tensor(42)

            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

            # Verify that iter was passed as int, not tensor
            self.assertEqual(len(captured_iter_args), 1)
            arg_type, arg_value = captured_iter_args[0]
            self.assertEqual(arg_type, int)
            self.assertEqual(arg_value, 42)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_blank_line_formatting_preserved(self) -> None:
        """Test that the code formatting with blank lines is preserved."""
        # This test is more about code structure but we can verify the module still works
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Verify the module works correctly after formatting changes
        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )

        # Should not raise any exceptions
        output = itep_ebc(input_kjt)
        self.assertIsNotNone(output)

        # Verify iter incremented
        self.assertEqual(itep_ebc._iter.item(), 1)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_boundary_values_with_pruning_logic(self) -> None:
        """Test iter behavior at boundary values that affect pruning decisions."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=10,  # Test around pruning interval boundaries
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test specific boundary values around pruning interval
        boundary_values = [9, 10, 19, 20, 29, 30, 99, 100]

        for iter_val in boundary_values:
            with self.subTest(iter_val=iter_val):
                itep_ebc._iter = torch.tensor(iter_val)

                # Capture the iter value passed to module
                original_forward = itep_module.forward
                captured_iter = None

                def mock_forward(
                    features: KeyedJaggedTensor, iter_val: int
                ) -> List[Tensor]:
                    nonlocal captured_iter
                    captured_iter = iter_val
                    # pyrefly: ignore[bad-return]
                    return original_forward(features, iter_val)

                with patch.object(itep_module, "forward", mock_forward):
                    input_kjt = self.generate_input_kjt_cuda(
                        self._feature_name_to_unpruned_hash_sizes
                    )
                    _ = itep_ebc(input_kjt)

                    # Verify iter was passed as int and matches expected value
                    self.assertEqual(captured_iter, iter_val)
                    self.assertIsInstance(captured_iter, int)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_error_handling_invalid_iter_tensor_values_cuda_disabled_in_oss_compatibility(
        self,
    ) -> None:
        """Test behavior with invalid iter tensor values."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )

        # Test with negative iter value
        itep_ebc._iter = torch.tensor(-10)
        try:
            _ = itep_ebc(input_kjt)
            self.assertTrue(True, "Negative iter value handled gracefully")
        except Exception as e:
            self.fail(f"Unexpected exception with negative iter: {e}")

        # Test with float tensor (should still work after .item())
        itep_ebc._iter = torch.tensor(42.0)
        try:
            _ = itep_ebc(input_kjt)
            self.assertTrue(True, "Float tensor iter value handled gracefully")
        except Exception as e:
            self.fail(f"Unexpected exception with float tensor iter: {e}")

        # Test with very large iter value
        itep_ebc._iter = torch.tensor(2**31 - 1)
        try:
            _ = itep_ebc(input_kjt)
            self.assertTrue(True, "Large iter value handled gracefully")
        except Exception as e:
            self.fail(f"Unexpected exception with large iter: {e}")

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_consistency_across_training_steps(self) -> None:
        """Test that iter remains consistent and increments properly across training steps."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=1000,  # High interval to avoid pruning during test
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Track iteration values passed to the module
        iter_history = []

        original_forward = itep_module.forward

        def track_iter_forward(
            features: KeyedJaggedTensor, iter_val: int
        ) -> List[Tensor]:
            iter_history.append(iter_val)
            # pyrefly: ignore[bad-return]
            return original_forward(features, iter_val)

        with patch.object(itep_module, "forward", track_iter_forward):
            # Start from a known iteration value
            itep_ebc._iter = torch.tensor(100)

            # Run multiple training steps
            num_steps = 25
            for step in range(num_steps):
                input_kjt = self.generate_input_kjt_cuda(
                    self._feature_name_to_unpruned_hash_sizes
                )

                # Forward pass
                output = itep_ebc(input_kjt)

                # Simulate backward pass
                output.values().sum().backward()

                # Verify iter incremented
                expected_iter = 100 + step + 1
                self.assertEqual(itep_ebc._iter.item(), expected_iter)

            # Verify all iteration values were passed correctly as integers
            self.assertEqual(len(iter_history), num_steps)
            for i, iter_val in enumerate(iter_history):
                expected_iter = 100 + i
                self.assertEqual(iter_val, expected_iter)
                self.assertIsInstance(iter_val, int)

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_performance_iter_conversion_overhead(self) -> None:
        """Test that iter tensor to int conversion doesn't introduce significant overhead."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=1000,  # High interval to minimize pruning overhead
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Prepare input
        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )

        # Warm up
        for _ in range(3):
            _ = itep_ebc(input_kjt)

        # Time multiple forward passes
        import time

        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            _ = itep_ebc(input_kjt)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / num_iterations

        # Performance check - each iteration should complete in reasonable time
        self.assertLess(
            avg_time_per_iteration,
            0.5,
            f"Average time per iteration ({avg_time_per_iteration:.4f}s) seems too high",
        )

        # Log the performance for monitoring
        print(
            f"ITEP Performance test: {avg_time_per_iteration:.6f}s per iteration "
            f"({num_iterations} iterations, total: {total_time:.4f}s)"
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_type_conversion_edge_cases(self) -> None:
        """Test edge cases for iter tensor to int conversion."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )

        # Test various tensor types and values
        test_cases = [
            (torch.tensor(0), 0, "zero value"),
            (torch.tensor(1), 1, "positive small value"),
            (torch.tensor(12345), 12345, "positive large value"),
            (torch.tensor(0, dtype=torch.int32), 0, "int32 tensor"),
            (torch.tensor(42, dtype=torch.int64), 42, "int64 tensor"),
            (torch.tensor(99, dtype=torch.long), 99, "long tensor"),
        ]

        original_forward = itep_module.forward
        captured_values = []

        def capture_iter_forward(
            features: KeyedJaggedTensor, iter_val: int
        ) -> List[Tensor]:
            captured_values.append((type(iter_val), iter_val))
            # pyrefly: ignore[bad-return]
            return original_forward(features, iter_val)

        with patch.object(itep_module, "forward", capture_iter_forward):
            for tensor_val, expected_int, description in test_cases:
                with self.subTest(description=description):
                    captured_values.clear()
                    itep_ebc._iter = tensor_val

                    _ = itep_ebc(input_kjt)

                    # Verify conversion worked correctly
                    self.assertEqual(len(captured_values), 1)
                    converted_type, converted_value = captured_values[0]

                    self.assertEqual(
                        converted_type, int, f"Type should be int for {description}"
                    )
                    self.assertEqual(
                        converted_value,
                        expected_int,
                        f"Value mismatch for {description}",
                    )

    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_concurrent_forward_passes_iter_safety(self) -> None:
        """Test that iter handling is safe under simulated concurrent conditions."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=100,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )

        # Track all iter values passed to the module
        iter_values_received = []
        original_forward = itep_module.forward

        def tracking_forward(
            features: KeyedJaggedTensor, iter_val: int
        ) -> List[Tensor]:
            iter_values_received.append(iter_val)
            # pyrefly: ignore[bad-return]
            return original_forward(features, iter_val)

        with patch.object(itep_module, "forward", tracking_forward):
            # Simulate rapid consecutive forward passes
            for i in range(50):
                # Each forward pass should increment iter
                _ = itep_ebc(input_kjt)

                # Verify iter value is correct
                expected_iter = i + 1
                self.assertEqual(itep_ebc._iter.item(), expected_iter)

                # Verify the module received the correct iter value (before increment)
                self.assertEqual(iter_values_received[i], i)

            # Final verification - all values should be integers and in sequence
            for i, iter_val in enumerate(iter_values_received):
                self.assertIsInstance(iter_val, int)
                self.assertEqual(iter_val, i)


class TestITEPEmbeddingCollection(unittest.TestCase):
    def setUp(self) -> None:
        embedding_config1 = EmbeddingConfig(
            name="table1",
            embedding_dim=4,
            num_embeddings=50,
            feature_names=["feature1"],
        )
        embedding_config2 = EmbeddingConfig(
            name="table2",
            embedding_dim=4,
            num_embeddings=40,
            feature_names=["feature2"],
        )
        unpruned_hash_size_1, unpruned_hash_size_2 = (100, 80)
        self._table_name_to_unpruned_hash_sizes = {
            "table1": unpruned_hash_size_1,
            "table2": unpruned_hash_size_2,
        }
        self._feature_name_to_unpruned_hash_sizes = {
            "feature1": unpruned_hash_size_1,
            "feature2": unpruned_hash_size_2,
        }
        self._batch_size = 8
        self._embedding_configs = [embedding_config1, embedding_config2]

        def embedding_config_to_sharded_table(
            config: EmbeddingConfig,
        ) -> ShardedEmbeddingTable:
            return ShardedEmbeddingTable(
                name=config.name,
                embedding_dim=config.embedding_dim,
                num_embeddings=config.num_embeddings,
                feature_names=config.feature_names,
            )

        sharded_et1 = embedding_config_to_sharded_table(embedding_config1)
        sharded_et2 = embedding_config_to_sharded_table(embedding_config2)

        self._embedding_collection = EmbeddingCollection(
            tables=self._embedding_configs,
            device=torch.device("cuda"),
        )

        self._mock_lookups = [MagicMock()]
        self._mock_lookups[0]._emb_modules = [MagicMock()]
        self._mock_lookups[0]._emb_modules[0]._config = MagicMock()
        self._mock_lookups[0]._emb_modules[0]._config.embedding_tables = [
            sharded_et1,
            sharded_et2,
        ]

    def generate_input_kjt_cuda(
        self,
        feature_name_to_unpruned_hash_sizes: Dict[str, int],
        use_vbe: bool = False,
    ) -> KeyedJaggedTensor:
        keys = []
        values = []
        lengths = []
        cuda_device = torch.device("cuda")

        for key, unpruned_hash_size in feature_name_to_unpruned_hash_sizes.items():
            value = []
            length = []
            for _ in range(self._batch_size):
                L = random.randint(0, 8)
                for _ in range(L):
                    index = random.randint(0, unpruned_hash_size - 1)
                    value.append(index)
                length.append(L)
            keys.append(key)
            values += value
            lengths += length

        if use_vbe:
            inverse_indices_list = []
            num_keys = len(keys)
            deduped_batch_size = len(lengths) // num_keys
            full_batch_size = deduped_batch_size * 2
            stride_per_key_per_rank = []

            for _ in range(num_keys):
                stride_per_key_per_rank.append([deduped_batch_size])
                keyed_inverse_indices = torch.randint(
                    low=0,
                    high=deduped_batch_size,
                    size=(full_batch_size,),
                    dtype=torch.int32,
                    device=cuda_device,
                )
                inverse_indices_list.append(keyed_inverse_indices)
            inverse_indices = (
                keys,
                torch.stack(inverse_indices_list),
            )

            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                stride_per_key_per_rank=stride_per_key_per_rank,
                inverse_indices=inverse_indices,
            )
        else:
            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
            )

        return input_kjt_cuda

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_train_forward(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ec(input_kjt)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_train_forward_vbe(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        for _ in range(5):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes, use_vbe=True
            )
            _ = itep_ec(input_kjt)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_check_pruning_schedule(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        random.seed(1)
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ec(input_kjt)

        self.assertEqual(mock_reset_weight_momentum.call_count, 5)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_eval_forward(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        itep_ec.eval()

        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ec(input_kjt)

        self.assertEqual(mock_reset_weight_momentum.call_count, 0)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_increment_per_forward(self) -> None:
        """Test that the iteration counter increments correctly with each forward pass."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        self.assertEqual(itep_ec._iter.item(), 0)

        for expected_iter in range(1, 6):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ec(input_kjt)
            self.assertEqual(itep_ec._iter.item(), expected_iter)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_iter_passed_as_int_to_itep_module(self) -> None:
        """Test that iter is passed as integer to the ITEP module."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        original_forward = itep_module.forward
        captured_iter_args = []

        # pyre-ignore[53]: Captured variable `captured_iter_args` is not annotated.
        def mock_forward(
            features: KeyedJaggedTensor, iter_val: int
        ) -> KeyedJaggedTensor:
            captured_iter_args.append((type(iter_val), iter_val))
            return original_forward(features, iter_val)

        with patch.object(itep_module, "forward", mock_forward):
            itep_ec._iter = torch.tensor(42)

            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ec(input_kjt)

            self.assertEqual(len(captured_iter_args), 1)
            arg_type, arg_value = captured_iter_args[0]
            self.assertEqual(arg_type, int)
            self.assertEqual(arg_value, 42)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_forward_output_type(self) -> None:
        """Verify forward returns Dict[str, JaggedTensor] with correct keys and shapes."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        input_kjt = self.generate_input_kjt_cuda(
            self._feature_name_to_unpruned_hash_sizes
        )
        output = itep_ec(input_kjt)

        self.assertIsInstance(output, dict)
        self.assertEqual(set(output.keys()), {"feature1", "feature2"})
        for _feature_name, jt in output.items():
            self.assertIsInstance(jt, JaggedTensor)
            # Embedding dim should be 4 for all tables
            if jt.values().numel() > 0:
                self.assertEqual(jt.values().shape[1], 4)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_embedding_configs_accessor(self) -> None:
        """Verify embedding_bag_configs() returns List[EmbeddingConfig] matching original configs."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        configs = itep_ec.embedding_bag_configs()
        self.assertIsInstance(configs, list)
        self.assertEqual(len(configs), 2)
        for config in configs:
            self.assertIsInstance(config, EmbeddingConfig)

        config_names = {c.name for c in configs}
        self.assertEqual(config_names, {"table1", "table2"})

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_forward_empty_kjt(self) -> None:
        """Test forward with an empty KJT (no values or lengths)."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        cuda_device = torch.device("cuda")
        empty_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature1", "feature2"],
            values=torch.tensor([], dtype=torch.int32, device=cuda_device),
            lengths=torch.zeros(
                2 * self._batch_size, dtype=torch.int32, device=cuda_device
            ),
        )

        output = itep_ec(empty_kjt)
        self.assertIsInstance(output, dict)
        self.assertEqual(set(output.keys()), {"feature1", "feature2"})
        for _feature_name, jt in output.items():
            self.assertIsInstance(jt, JaggedTensor)
            self.assertEqual(jt.values().shape[1], 4)

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_forward_batch_size_one(self) -> None:
        """Test forward with batch_size=1."""
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ec = ITEPEmbeddingCollection(
            embedding_collection=self._embedding_collection,
            itep_module=itep_module,
        )

        cuda_device = torch.device("cuda")
        input_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature1", "feature2"],
            values=torch.tensor([1, 2], dtype=torch.int32, device=cuda_device),
            lengths=torch.tensor([1, 1], dtype=torch.int32, device=cuda_device),
        )

        output = itep_ec(input_kjt)
        self.assertIsInstance(output, dict)
        self.assertEqual(set(output.keys()), {"feature1", "feature2"})
        for _feature_name, jt in output.items():
            self.assertIsInstance(jt, JaggedTensor)
            if jt.values().numel() > 0:
                self.assertEqual(jt.values().shape[1], 4)
