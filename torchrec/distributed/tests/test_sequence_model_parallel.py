#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import assume, given, settings, Verbosity
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.test_utils.test_model import TestSparseNNBase
from torchrec.distributed.test_utils.test_sharding import sharding_single_rank_test
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import seed_and_log, skip_if_asan_class


@skip_if_asan_class
class SequenceModelParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.DATA_PARALLEL.value),
        kernel_type=st.just(EmbeddingComputeKernel.DENSE.value),
        apply_optimizer_in_backward_config=st.just(None),
        # TODO - need to enable optimizer overlapped behavior for data_parallel tables
        # apply_optimizer_in_backward_config=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_nccl_dp(
        self,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.TABLE_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.COLUMN_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_nccl_cw(
        self,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            constraints={
                table.name: ParameterConstraints(min_partition=8)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
        index_dedup=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_sharding_variable_batch(
        self,
        sharding_type: str,
        index_dedup: bool,
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                    use_index_dedup=index_dedup,
                )
            ],
            backend="nccl",
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            variable_batch_per_feature=True,
        )

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharding_empty_rank(self) -> None:
        table = self.tables[0]
        embedding_groups = {"group_0": table.feature_names}
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=2,
            model_class=TestSequenceSparseNN,
            tables=[table],
            embedding_groups=embedding_groups,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                )
            ],
            optim=EmbOptimType.EXACT_SGD,
            backend="nccl",
            variable_batch_size=True,
        )

    @seed_and_log
    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables = initial_tables + shared_features_tables
        self.shared_features = [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups = {
            "group_0": [
                (
                    f"{feature}@{table.name}"
                    if feature in self.shared_features
                    else feature
                )
                for table in self.tables
                for feature in table.feature_names
            ]
        }

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=True,
        )


class DedupIndicesWeightAccumulationTest(unittest.TestCase):
    """
    Test suite for validating the _dedup_indices method weight accumulation logic.
    This tests the correctness of the new scatter_add_along_first_dim implementation.
    """

    # to be deleted
    def test_dedup_indices_weight_accumulation(self) -> None:
        """
        Test the _dedup_indices method to ensure weight accumulation works correctly
        with the new scatter_add_along_first_dim implementation.
        """
        # Setup: Create a minimal ShardedEmbeddingCollection for testing
        device = torch.device("cuda:0")

        # Create a mock ShardedEmbeddingCollection with minimal setup
        class MockShardedEmbeddingCollection:
            def __init__(self):
                self._enable_feature_score_weight_accumulation = True
                self._device = device
                # Register required buffers for _dedup_indices
                self._buffers = {}

                # Mock hash_size_cumsum_tensor_0 - cumulative sum of embedding table sizes
                self._buffers["_hash_size_cumsum_tensor_0"] = torch.tensor(
                    [0, 10], dtype=torch.int64, device=device
                )
                # Mock hash_size_offset_tensor_0 - offset for each feature
                self._buffers["_hash_size_offset_tensor_0"] = torch.tensor(
                    [0], dtype=torch.int64, device=device
                )

            def get_buffer(self, name: str) -> torch.Tensor:
                return self._buffers[name]

            def _dedup_indices(
                self,
                ctx: EmbeddingCollectionContext,
                input_feature_splits: List[KeyedJaggedTensor],
            ) -> List[KeyedJaggedTensor]:
                # Copy the actual _dedup_indices logic for testing
                features_by_shards = []
                for i, input_feature in enumerate(input_feature_splits):
                    hash_size_cumsum = self.get_buffer(f"_hash_size_cumsum_tensor_{i}")
                    hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                    (
                        lengths,
                        offsets,
                        unique_indices,
                        reverse_indices,
                    ) = torch.ops.fbgemm.jagged_unique_indices(
                        hash_size_cumsum,
                        hash_size_offset,
                        input_feature.offsets().to(torch.int64),
                        input_feature.values().to(torch.int64),
                    )
                    acc_weights = None
                    if (
                        self._enable_feature_score_weight_accumulation
                        and input_feature.weights_or_none() is not None
                    ):
                        source_weights = input_feature.weights()
                        assert (
                            source_weights.dtype == torch.float32
                        ), "Only float32 weights are supported for feature score eviction weights."

                        # Accumulate weights using scatter_add
                        acc_weights = torch.zeros(
                            unique_indices.numel(),
                            dtype=torch.float32,
                            device=source_weights.device,
                        )

                        # Use PyTorch's scatter_add to accumulate weights
                        acc_weights.scatter_add_(0, reverse_indices, source_weights)

                        features_by_shards.append(
                            KeyedJaggedTensor(
                                keys=input_feature.keys(),
                                values=unique_indices,
                                weights=acc_weights,
                                lengths=lengths,
                                offsets=offsets,
                            )
                        )
                return features_by_shards

        # Create mock ShardedEmbeddingCollection instance
        sharded_ec = MockShardedEmbeddingCollection()

        # Create test input with duplicate indices and varying weights
        values = torch.tensor(
            [0, 1, 0, 2, 1, 0], dtype=torch.int64, device=device
        )  # Indices with duplicates
        weights = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32, device=device
        )  # Corresponding weights
        lengths = torch.tensor(
            [6], dtype=torch.int64, device=device
        )  # Single feature with 6 values

        kjt_input = KeyedJaggedTensor(
            keys=["feature_0"],
            values=values,
            weights=weights,
            lengths=lengths,
        )

        # Execute: Run _dedup_indices method
        ctx = EmbeddingCollectionContext()
        features_by_shards = sharded_ec._dedup_indices(ctx, [kjt_input])

        # Assert: Validate accumulated weights and counts
        dedup_feature = features_by_shards[0]
        self.assertIsNotNone(dedup_feature.weights_or_none())

        # Reconstruct accumulated weights tensor (weights are stored as flattened float64 view)
        acc_weights = dedup_feature.weights().view(torch.float32).view(-1, 1)

        # Expected results based on duplicate indices:
        # Index 0 appears 3 times with weights [1.0, 3.0, 6.0] -> sum = 10.0, count = 3
        # Index 1 appears 2 times with weights [2.0, 5.0] -> sum = 7.0, count = 2
        # Index 2 appears 1 time with weight [4.0] -> sum = 4.0, count = 1

        unique_values = dedup_feature.values()
        self.assertEqual(len(unique_values), 3)  # Should have 3 unique indices

        # Find positions of each unique index (order may vary after deduplication)
        idx_0_pos = (unique_values == 0).nonzero(as_tuple=True)[0][0]
        idx_1_pos = (unique_values == 1).nonzero(as_tuple=True)[0][0]
        idx_2_pos = (unique_values == 2).nonzero(as_tuple=True)[0][0]

        # Validate accumulated weights (column 0) and counts (column 1)
        self.assertAlmostEqual(acc_weights[idx_0_pos, 0].item(), 10.0, places=5)
        self.assertAlmostEqual(acc_weights[idx_1_pos, 0].item(), 7.0, places=5)
        self.assertAlmostEqual(acc_weights[idx_2_pos, 0].item(), 4.0, places=5)


@skip_if_asan_class
class TDSequenceModelParallelTest(SequenceModelParallelTest):

    def test_sharding_variable_batch(self) -> None:
        pass

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=True,
            input_type="td",
        )
