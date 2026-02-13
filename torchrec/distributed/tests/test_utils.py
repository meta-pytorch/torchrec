#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import math
import os
import random
import unittest
from typing import cast, List, Optional, Tuple
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_training import SparseType
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_sharding import bucketize_kjt_before_all2all
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    ModuleSharder,
    MultiPassPrefetchConfig,
    ParameterSharding,
    ShardingBucketMetadata,
    ShardMetadata,
)
from torchrec.distributed.utils import (
    _convert_weights,
    add_params_from_parameter_sharding,
    convert_to_fbgemm_types,
    EmbeddingQuantizationUtils,
    get_bucket_metadata_from_shard_metadata,
    get_unsharded_module_names,
    merge_fused_params,
    weights_bytes_in_emb_kernel,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.test_utils import keyed_jagged_tensor_equals
from torchrec.test_utils import get_free_port


class UtilsTest(unittest.TestCase):
    def test_get_unsharded_module_names(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        device = torch.device("cpu")
        backend = "gloo"
        dist.init_process_group(backend=backend)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(2)
        ]
        m = TestSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=device,
            sparse_device=device,
        )

        dmp = DistributedModelParallel(
            module=m,
            init_data_parallel=False,
            device=device,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder()),
            ],
        )

        self.assertListEqual(
            sorted(get_unsharded_module_names(dmp)),
            sorted(["_dmp_wrapped_module.over", "_dmp_wrapped_module.dense"]),
        )
        dist.destroy_process_group()


class EmbeddingQuantizationUtilsTest(unittest.TestCase):
    def test_init(self) -> None:
        """Test EmbeddingQuantizationUtils initialization."""
        utils = EmbeddingQuantizationUtils()
        self.assertEqual(utils._emb_kernel_to_sparse_dtype, {})

    def test_quantize_embedding_modules(self) -> None:
        """Test quantize_embedding_modules correctly converts embedding weight tensors."""
        utils = EmbeddingQuantizationUtils()

        # Create a mock embedding module
        mock_emb = Mock()
        mock_emb.weights_dev = torch.tensor([1.0, 2.0, 3.0])
        mock_emb.weights_host = torch.tensor([4.0, 5.0])
        mock_emb.weights_uvm = torch.tensor([6.0])
        mock_emb.weights_precision = SparseType.FP32

        # Create a module
        module = torch.nn.Module()

        # Mock functions
        with patch("torchrec.distributed.utils._group_sharded_modules") as mock_group:
            mock_group.return_value = [mock_emb]

            with patch(
                "torchrec.distributed.utils.data_type_to_sparse_type"
            ) as mock_convert:
                mock_sparse_type = Mock()
                mock_sparse_type.as_dtype.return_value = torch.float16
                mock_convert.return_value = mock_sparse_type

                with patch(
                    "torchrec.distributed.utils._convert_weights"
                ) as mock_convert_weights:
                    mock_converted_tensor = torch.tensor([1.0], dtype=torch.float16)
                    mock_convert_weights.return_value = mock_converted_tensor

                    with patch(
                        "torchrec.distributed.utils.weights_bytes_in_emb_kernel"
                    ) as mock_bytes:
                        mock_bytes.return_value = 100

                        # Call the function
                        utils.quantize_embedding_modules(module, DataType.FP16)

                        # Verify the weights were converted
                        self.assertEqual(mock_convert_weights.call_count, 3)

                        # Verify weights_precision was updated
                        self.assertEqual(mock_emb.weights_precision, mock_sparse_type)

                        # Verify the sparse type was stored
                        self.assertIn(mock_emb, utils._emb_kernel_to_sparse_dtype)

    def test_recreate_embedding_modules(self) -> None:
        """Test recreate_embedding_modules restores original precision."""
        utils = EmbeddingQuantizationUtils()

        # Create a mock embedding module
        mock_emb = Mock()
        mock_emb.weights_dev = torch.tensor([1.0, 2.0])
        mock_emb.weights_host = torch.tensor([3.0])
        mock_emb.weights_uvm = torch.tensor([4.0])

        # Set up the stored sparse type
        original_sparse_type = SparseType.FP32
        utils._emb_kernel_to_sparse_dtype[mock_emb] = original_sparse_type

        module = torch.nn.Module()

        # Mock functions
        with patch("torchrec.distributed.utils._group_sharded_modules") as mock_group:
            mock_group.return_value = [mock_emb]

            with patch(
                "torchrec.distributed.utils._convert_weights"
            ) as mock_convert_weights:
                mock_converted_tensor = torch.tensor([1.0])
                mock_convert_weights.return_value = mock_converted_tensor

                with patch("torchrec.distributed.utils.weights_bytes_in_emb_kernel"):
                    with patch.object(utils, "_recalculate_torch_state") as mock_recalc:
                        # Call the function
                        utils.recreate_embedding_modules(module)

                        # Verify weights were converted back
                        self.assertEqual(mock_convert_weights.call_count, 3)

                        # Verify _recalculate_torch_state was called
                        mock_recalc.assert_called_once_with(module)

    def test_recalculate_torch_state(self) -> None:
        """Test _recalculate_torch_state calls _initialize_torch_state on modules with lookups."""
        utils = EmbeddingQuantizationUtils()

        # Create a mock module with _lookups attribute
        mock_module_with_lookups = Mock()
        mock_module_with_lookups._initialize_torch_state = Mock()
        mock_module_with_lookups._lookups = Mock()  # Has _lookups attribute

        # Create a mock module with _lookup attribute (singular)
        mock_module_with_lookup = Mock()
        mock_module_with_lookup._initialize_torch_state = Mock()
        mock_module_with_lookup._lookup = Mock()  # Has _lookup attribute

        # Create a mock module without lookup attributes (should be processed recursively)
        mock_module_without_lookups = Mock()
        mock_module_without_lookups.named_children.return_value = (
            []
        )  # No children to recurse into
        # Ensure it doesn't have lookup attributes
        if hasattr(mock_module_without_lookups, "_lookups"):
            delattr(mock_module_without_lookups, "_lookups")
        if hasattr(mock_module_without_lookups, "_lookup"):
            delattr(mock_module_without_lookups, "_lookup")

        # Create parent module that doesn't have _lookups or _lookup but has children
        mock_parent = Mock()
        # Ensure parent doesn't have the attributes
        if hasattr(mock_parent, "_lookups"):
            delattr(mock_parent, "_lookups")
        if hasattr(mock_parent, "_lookup"):
            delattr(mock_parent, "_lookup")
        mock_parent.named_children.return_value = [
            ("child1", mock_module_with_lookups),
            ("child2", mock_module_with_lookup),
            ("child3", mock_module_without_lookups),
        ]

        # Call the function on the parent
        utils._recalculate_torch_state(mock_parent)

        # Verify _initialize_torch_state was called on both child modules with lookups
        mock_module_with_lookups._initialize_torch_state.assert_called_once_with(
            skip_registering=True
        )
        mock_module_with_lookup._initialize_torch_state.assert_called_once_with(
            skip_registering=True
        )


class ConvertWeightsTest(unittest.TestCase):
    def test_convert_weights(self) -> None:
        """Test _convert_weights function."""
        # Create a test tensor
        weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        # Create a mock sparse type
        mock_sparse_type = Mock()
        mock_sparse_type.as_dtype.return_value = torch.float16

        # Call the function
        new_weights = _convert_weights(weights, mock_sparse_type)

        # Verify the tensor was converted to the correct dtype
        self.assertEqual(new_weights.dtype, torch.float16)

        # Verify the original storage was resized to 0
        self.assertEqual(weights.untyped_storage().size(), 0)

        # Verify the values are correct (approximately, due to precision change)
        expected_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        torch.testing.assert_close(new_weights, expected_values)


class WeightsBytesTest(unittest.TestCase):
    def test_weights_bytes_in_emb_kernel(self) -> None:
        """Test weights_bytes_in_emb_kernel function."""
        # Create a mock embedding module
        mock_emb = Mock()

        # Mock weights tensors with specific sizes and element sizes
        mock_weights_dev = Mock()
        mock_weights_dev.element_size.return_value = 4  # 4 bytes per element (float32)
        mock_weights_dev.numel.return_value = 10  # 10 elements

        mock_weights_host = Mock()
        mock_weights_host.element_size.return_value = 2  # 2 bytes per element (float16)
        mock_weights_host.numel.return_value = 5  # 5 elements

        mock_weights_uvm = Mock()
        mock_weights_uvm.element_size.return_value = 1  # 1 byte per element (int8)
        mock_weights_uvm.numel.return_value = 8  # 8 elements

        mock_emb.weights_dev = mock_weights_dev
        mock_emb.weights_host = mock_weights_host
        mock_emb.weights_uvm = mock_weights_uvm

        # Call the function
        total_bytes = weights_bytes_in_emb_kernel(mock_emb)

        # Expected: (4 * 10) + (2 * 5) + (1 * 8) = 40 + 10 + 8 = 58
        self.assertEqual(total_bytes, 58)


def _compute_translated_lengths(
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    trainers_size: int,
    block_sizes: List[int],
) -> List[int]:
    translated_lengths = [0] * trainers_size * lengths_size

    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        for row_idx in row_indices[start_offset:end_offset]:
            # compute the owner of this row
            trainer_offset = int(row_idx / block_sizes[feature_offset])
            # we do not have enough trainers to handle this row
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # increment the corresponding length in the trainer
            translated_lengths[trainer_lengths_offset + local_lengths_offset] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_lengths


def _compute_translated_indices_with_weights(
    translated_lengths: List[int],
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    weights: Optional[List[int]],
    trainers_size: int,
    block_sizes: List[int],
) -> List[Tuple[int, int]]:
    translated_indices_with_weights = [(0, 0)] * len(row_indices)

    translated_indices_offsets = list(itertools.accumulate([0] + translated_lengths))
    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        # and assign the translated row index to the corresponding offset in output
        for current_offset in range(start_offset, end_offset):
            row_idx = row_indices[current_offset]
            feature_block_size = block_sizes[feature_offset]
            # compute the owner of this row
            trainer_offset = int(row_idx / feature_block_size)
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # since we know the number of rows belonging to each trainer,
            # we can figure out the corresponding offset in the translated indices list
            # for the current translated index
            translated_indices_offset = translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ]
            translated_indices_with_weights[translated_indices_offset] = (
                row_idx % feature_block_size,
                weights[current_offset] if weights else 0,
            )
            # the next row that goes to this trainer for this feature and batch
            # combination goes to the next offset
            translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_indices_with_weights


def block_bucketize_ref(
    keyed_jagged_tensor: KeyedJaggedTensor,
    trainers_size: int,
    block_sizes: torch.Tensor,
    device: str = "cuda",
) -> KeyedJaggedTensor:
    lengths_list = keyed_jagged_tensor.lengths().view(-1).tolist()
    indices_list = keyed_jagged_tensor.values().view(-1).tolist()
    weights_list = (
        keyed_jagged_tensor.weights().view(-1).tolist()
        if keyed_jagged_tensor.weights() is not None
        else None
    )
    block_sizes_list = block_sizes.view(-1).tolist()
    lengths_size = len(lengths_list)

    """
    each element in indices_offsets signifies both the starting offset, in indices_list,
    that corresponds to all rows in a particular feature and batch iteration,
    and the ending offset of the previous feature/batch iteration

    For example:
    given that features_size = 2 and batch_size = 2, an indices_offsets of
    [0,1,4,6,6] signifies that:

    elements in indices_list[0:1] belongs to feature 0 batch 0
    elements in indices_list[1:4] belongs to feature 0 batch 1
    elements in indices_list[4:6] belongs to feature 1 batch 0
    elements in indices_list[6:6] belongs to feature 1 batch 1
    """
    indices_offsets = list(itertools.accumulate([0] + lengths_list))

    translated_lengths = _compute_translated_lengths(
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )
    translated_indices_with_weights = _compute_translated_indices_with_weights(
        translated_lengths=translated_lengths,
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        weights=weights_list,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )

    translated_indices = [
        translated_index for translated_index, _ in translated_indices_with_weights
    ]

    translated_weights = [
        translated_weight for _, translated_weight in translated_indices_with_weights
    ]

    expected_keys = [
        key for index in range(trainers_size) for key in keyed_jagged_tensor.keys()
    ]
    if device == "cuda":
        return KeyedJaggedTensor(
            keys=expected_keys,
            lengths=torch.tensor(
                translated_lengths, dtype=keyed_jagged_tensor.lengths().dtype
            )
            .view(-1)
            .cuda(),
            values=torch.tensor(
                translated_indices, dtype=keyed_jagged_tensor.values().dtype
            ).cuda(),
            weights=(
                torch.tensor(translated_weights).float().cuda()
                if weights_list
                else None
            ),
        )
    else:
        return KeyedJaggedTensor(
            keys=expected_keys,
            lengths=torch.tensor(
                translated_lengths, dtype=keyed_jagged_tensor.lengths().dtype
            ).view(-1),
            values=torch.tensor(
                translated_indices, dtype=keyed_jagged_tensor.values().dtype
            ),
            weights=torch.tensor(translated_weights).float() if weights_list else None,
        )


class KJTBucketizeTest(unittest.TestCase):
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        offset_type=st.sampled_from([torch.int, torch.long]),
        world_size=st.integers(1, 129),
        num_features=st.integers(1, 15),
        batch_size=st.integers(1, 15),
        variable_bucket_pos=st.booleans(),
        device=st.sampled_from(
            ["cpu"] + (["cuda"] if torch.cuda.device_count() > 0 else [])
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=50, deadline=None)
    def test_kjt_bucketize_before_all2all(
        self,
        index_type: torch.dtype,
        offset_type: torch.dtype,
        world_size: int,
        num_features: int,
        batch_size: int,
        variable_bucket_pos: bool,
        device: str,
    ) -> None:
        MAX_BATCH_SIZE = 15
        MAX_LENGTH = 10
        # max number of rows needed for a given feature to have unique row index
        MAX_ROW_COUNT = MAX_LENGTH * MAX_BATCH_SIZE

        lengths_list = [
            random.randrange(MAX_LENGTH + 1) for _ in range(num_features * batch_size)
        ]
        keys_list = [f"feature_{i}" for i in range(num_features)]
        # for each feature, generate unrepeated row indices
        indices_lists = [
            random.sample(
                range(MAX_ROW_COUNT),
                # number of indices needed is the length sum of all batches for a feature
                sum(
                    lengths_list[
                        feature_offset * batch_size : (feature_offset + 1) * batch_size
                    ]
                ),
            )
            for feature_offset in range(num_features)
        ]
        indices_list = list(itertools.chain(*indices_lists))

        weights_list = [random.randint(1, 100) for _ in range(len(indices_list))]

        # for each feature, calculate the minimum block size needed to
        # distribute all rows to the available trainers
        block_sizes_list = [
            (
                math.ceil((max(feature_indices_list) + 1) / world_size)
                if feature_indices_list
                else 1
            )
            for feature_indices_list in indices_lists
        ]
        block_bucketize_row_pos = [] if variable_bucket_pos else None
        if variable_bucket_pos:
            for block_size in block_sizes_list:
                # pyrefly: ignore[missing-attribute]
                block_bucketize_row_pos.append(
                    torch.tensor(
                        [w * block_size for w in range(world_size + 1)],
                        dtype=index_type,
                    )
                )

        kjt = KeyedJaggedTensor(
            keys=keys_list,
            lengths=torch.tensor(lengths_list, dtype=offset_type, device=device).view(
                num_features * batch_size
            ),
            values=torch.tensor(indices_list, dtype=index_type, device=device),
            weights=torch.tensor(weights_list, dtype=torch.float, device=device),
        )
        """
        each entry in block_sizes identifies how many hashes for each feature goes
        to every rank; we have three featues in `self.features`
        """
        block_sizes = torch.tensor(block_sizes_list, dtype=index_type, device=device)
        block_bucketized_kjt, _ = bucketize_kjt_before_all2all(
            kjt=kjt,
            num_buckets=world_size,
            block_sizes=block_sizes,
            block_bucketize_row_pos=block_bucketize_row_pos,
        )

        expected_block_bucketized_kjt = block_bucketize_ref(
            kjt,
            world_size,
            block_sizes,
            device,
        )

        self.assertTrue(
            keyed_jagged_tensor_equals(
                block_bucketized_kjt,
                expected_block_bucketized_kjt,
                is_pooled_features=True,
            )
        )


class MergeFusedParamsTest(unittest.TestCase):
    def test_merge_fused_params(self) -> None:
        # Case fused_params is None, change it to be an empty dict
        # and set cache_precision to be the same as weights_precision
        fused_params = None
        configured_fused_params = merge_fused_params(fused_params=fused_params)
        self.assertFalse(configured_fused_params is None)
        self.assertEqual(configured_fused_params, {})

    def test_merge_fused_params_update(self) -> None:
        # Case fused_params is None, change it to be an empty dict
        # and set cache_precision to be the same as weights_precision
        fused_params = None
        configured_fused_params = merge_fused_params(
            fused_params=fused_params, param_fused_params={"learning_rate": 0.0}
        )
        self.assertFalse(configured_fused_params is None)
        self.assertEqual(configured_fused_params, {"learning_rate": 0.0})


class AddParamsFromParameterShardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parameter_sharding = ParameterSharding(
            sharding_type="data_parallel",
            compute_kernel="dense",
            ranks=[0, 1],
            sharding_spec=None,
            cache_params=CacheParams(
                algorithm=CacheAlgorithm.LFU,
                reserved_memory=1.0,
                prefetch_pipeline=False,
                multipass_prefetch_config=MultiPassPrefetchConfig(num_passes=2),
            ),
            enforce_hbm=False,
            stochastic_rounding=True,
            bounds_check_mode=BoundsCheckMode.WARNING,
        )

    def test_add_params_from_parameter_sharding(self) -> None:
        fused_params = None
        fused_params = add_params_from_parameter_sharding(
            fused_params, self.parameter_sharding
        )
        expected_fused_params = {
            "cache_algorithm": CacheAlgorithm.LFU,
            "cache_reserved_memory": 1.0,
            "prefetch_pipeline": False,
            "enforce_hbm": False,
            "stochastic_rounding": True,
            "bounds_check_mode": BoundsCheckMode.WARNING,
            "multipass_prefetch_config": MultiPassPrefetchConfig(num_passes=2),
        }
        self.assertEqual(fused_params, expected_fused_params)

    def test_add_params_from_parameter_sharding_override(self) -> None:
        fused_params = {
            "learning_rate": 0.1,
            "cache_algorithm": CacheAlgorithm.LRU,
            "stochastic_rounding": False,
            "prefetch_pipeline": True,
            "multipass_prefetch_config": MultiPassPrefetchConfig(num_passes=5),
        }
        fused_params = add_params_from_parameter_sharding(
            fused_params, self.parameter_sharding
        )
        expected_fused_params = {
            "learning_rate": 0.1,
            "cache_algorithm": CacheAlgorithm.LFU,
            "cache_reserved_memory": 1.0,
            "prefetch_pipeline": False,
            "enforce_hbm": False,
            "stochastic_rounding": True,
            "bounds_check_mode": BoundsCheckMode.WARNING,
            "multipass_prefetch_config": MultiPassPrefetchConfig(num_passes=2),
        }
        self.assertEqual(fused_params, expected_fused_params)


class ConvertFusedParamsTest(unittest.TestCase):
    def test_convert_to_fbgemm_types(self) -> None:
        per_table_fused_params = {
            "cache_precision": DataType.FP32,
            "weights_precision": DataType.FP32,
            "output_dtype": DataType.FP32,
        }
        self.assertTrue(isinstance(per_table_fused_params["cache_precision"], DataType))
        self.assertTrue(
            isinstance(per_table_fused_params["weights_precision"], DataType)
        )
        self.assertTrue(isinstance(per_table_fused_params["output_dtype"], DataType))

        per_table_fused_params = convert_to_fbgemm_types(per_table_fused_params)
        self.assertFalse(
            isinstance(per_table_fused_params["cache_precision"], DataType)
        )
        self.assertFalse(
            isinstance(per_table_fused_params["weights_precision"], DataType)
        )
        self.assertFalse(isinstance(per_table_fused_params["output_dtype"], DataType))


class TestBucketMetadata(unittest.TestCase):
    def test_bucket_metadata(self) -> None:
        # Given no shards
        # When we get bucket metadata from get_bucket_metadata_from_shard_metadata
        # Then an error should be raised
        self.assertRaisesRegex(
            AssertionError,
            "Shards cannot be empty",
            get_bucket_metadata_from_shard_metadata,
            [],
            num_buckets=4,
        )

        # Given 1 shard and 5 buckets
        shards = [
            ShardMetadata(shard_offsets=[0], shard_sizes=[5], placement="rank:0/cuda:0")
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        bucket_metadata = get_bucket_metadata_from_shard_metadata(shards, num_buckets=5)
        # Then we should get 1 offset with value 0
        expected_metadata = ShardingBucketMetadata(
            num_buckets_per_shard=[5], bucket_offsets_per_shard=[0], bucket_size=1
        )
        self.assertEqual(bucket_metadata, expected_metadata)

        # Given 2 shards of size 5 and 4 buckets
        shards = [
            ShardMetadata(
                shard_offsets=[0], shard_sizes=[5], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[5], shard_sizes=[5], placement="rank:0/cuda:0"
            ),
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        # Then an error should be raised
        self.assertRaisesRegex(
            AssertionError,
            "Table size '10' must be divisible by num_buckets '4'",
            get_bucket_metadata_from_shard_metadata,
            shards,
            num_buckets=4,
        )

        # Given 2 shards of size 2 and 5 buckets
        shards = [
            ShardMetadata(
                shard_offsets=[0], shard_sizes=[2], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[2], shard_sizes=[2], placement="rank:0/cuda:0"
            ),
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        # Then an error should be raised
        self.assertRaisesRegex(
            AssertionError,
            "Table size '4' must be divisible by num_buckets '5'",
            get_bucket_metadata_from_shard_metadata,
            shards,
            num_buckets=5,
        )

        # Given 2 shards sharded by column
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0], shard_sizes=[20, 5], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[0, 5], shard_sizes=[20, 5], placement="rank:0/cuda:0"
            ),
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        # Then an error should be raised
        self.assertRaisesRegex(
            AssertionError,
            r"Shard shard_offsets\[1\] '5' is not 0. Table should be only row-wise sharded for bucketization",
            get_bucket_metadata_from_shard_metadata,
            shards,
            num_buckets=2,
        )

        # Given 2 shards of size 10 and 5 buckets
        shards = [
            ShardMetadata(
                shard_offsets=[0], shard_sizes=[10], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[10], shard_sizes=[10], placement="rank:0/cuda:0"
            ),
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        # Then an error should be raised
        self.assertRaisesRegex(
            AssertionError,
            r"Shard size\[0\] '10' is not divisible by bucket size '4'",
            get_bucket_metadata_from_shard_metadata,
            shards,
            num_buckets=5,
        )

        # Given 2 shards of size 20 and 10 buckets
        shards = [
            ShardMetadata(
                shard_offsets=[0], shard_sizes=[20], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[20], shard_sizes=[20], placement="rank:0/cuda:0"
            ),
        ]
        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        bucket_metadata = get_bucket_metadata_from_shard_metadata(
            shards,
            num_buckets=10,
        )
        # Then num_buckets_per_shard should be set to [5, 5]
        self.assertEqual(
            bucket_metadata,
            ShardingBucketMetadata(
                num_buckets_per_shard=[5, 5],
                bucket_offsets_per_shard=[0, 5],
                bucket_size=4,
            ),
        )

        # Given 3 uneven shards of sizes 12, 16 and 20 and 12 buckets
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0], shard_sizes=[12, 0], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[12, 0], shard_sizes=[16, 0], placement="rank:0/cuda:0"
            ),
            ShardMetadata(
                shard_offsets=[28, 0], shard_sizes=[20, 0], placement="rank:0/cuda:0"
            ),
        ]

        # When we get bucket offsets from get_bucket_metadata_from_shard_metadata
        bucket_metadata = get_bucket_metadata_from_shard_metadata(
            shards,
            num_buckets=12,
        )
        # Then num_buckets_per_shard should be set to [3, 4, 5]
        self.assertEqual(
            bucket_metadata,
            ShardingBucketMetadata(
                num_buckets_per_shard=[3, 4, 5],
                bucket_offsets_per_shard=[0, 3, 7],
                bucket_size=4,
            ),
        )
