#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import random
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torchrec.distributed.batched_embedding_kernel import ZeroCollisionKeyValueEmbedding
from torchrec.distributed.embedding import EmbeddingCollectionContext
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.embedding_sharding import (
    _get_compute_kernel_type,
    _get_grouping_fused_params,
    _get_weighted_avg_cache_load_factor,
    _prefetch_and_cached,
    _set_sharding_context_post_a2a,
    group_tables,
)
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import ShardedTensorMetadata, ShardMetadata
from torchrec.modules.embedding_configs import DataType, NoEvictionPolicy, PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE = 2


class TestGetWeightedAverageCacheLoadFactor(unittest.TestCase):
    def test_get_avg_cache_load_factor_hbm(self) -> None:
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=1000,
                embedding_dim=MagicMock(),
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor in cache_load_factors
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        self.assertIsNone(weighted_avg_cache_load_factor)

    def test_get_avg_cache_load_factor(self) -> None:
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=1000,
                embedding_dim=MagicMock(),
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor in cache_load_factors
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        expected_avg = sum(cache_load_factors) / len(cache_load_factors)
        self.assertIsNotNone(weighted_avg_cache_load_factor)
        self.assertAlmostEqual(weighted_avg_cache_load_factor, expected_avg)

    def test_get_weighted_avg_cache_load_factor(self) -> None:
        hash_sizes = [random.randint(100, 1000) for _ in range(5)]
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=hash_size,
                embedding_dim=MagicMock(),
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor, hash_size in zip(cache_load_factors, hash_sizes)
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        expected_weighted_avg = sum(
            cache_load_factor * hash_size
            for cache_load_factor, hash_size in zip(cache_load_factors, hash_sizes)
        ) / sum(hash_sizes)

        self.assertIsNotNone(weighted_avg_cache_load_factor)
        self.assertAlmostEqual(weighted_avg_cache_load_factor, expected_weighted_avg)


class TestGetGroupingFusedParams(unittest.TestCase):
    def test_get_grouping_fused_params(self) -> None:
        fused_params_groups = [
            None,
            {},
            {"stochastic_rounding": False},
            {"stochastic_rounding": False, "cache_load_factor": 0.4},
        ]
        grouping_fused_params_groups = [
            _get_grouping_fused_params(fused_params, "table_1")
            for fused_params in fused_params_groups
        ]
        expected_grouping_fused_params_groups = [
            None,
            {},
            {"stochastic_rounding": False},
            {"stochastic_rounding": False},
        ]

        self.assertEqual(
            grouping_fused_params_groups, expected_grouping_fused_params_groups
        )


class TestPerTBECacheLoadFactor(unittest.TestCase):
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        embedding_dim=st.sampled_from(list(range(160, 320, 40))),
        pooling_type=st.sampled_from(list(PoolingType)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_per_tbe_clf_weighted_average(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        embedding_dim: int,
        pooling_type: PoolingType,
    ) -> None:
        compute_kernels = [
            EmbeddingComputeKernel.FUSED_UVM_CACHING,
            EmbeddingComputeKernel.FUSED_UVM_CACHING,
            EmbeddingComputeKernel.FUSED,
            EmbeddingComputeKernel.FUSED_UVM,
        ]
        fused_params_groups = [
            {"cache_load_factor": 0.5},
            {"cache_load_factor": 0.3},
            {"cache_load_factor": 0.9},  # hbm table, would have no effect
            {"cache_load_factor": 0.4},  # uvm table, would have no effect
        ]
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_groups[i],
                compute_kernel=compute_kernels[i],
                embedding_dim=embedding_dim,
                num_embeddings=10000 * (2 * i + 1),  # 10000 and 30000
            )
            for i in range(4)
        ]

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]

        # assert that they are grouped together
        self.assertEqual(len(table_groups), 1)

        table_group = table_groups[0]
        self.assertIsNotNone(table_group.fused_params)
        self.assertEqual(table_group.fused_params.get("cache_load_factor"), 0.35)


def _get_table_names_by_groups(
    embedding_tables: List[ShardedEmbeddingTable],
) -> List[List[str]]:
    # since we don't have access to _group_tables_per_rank
    tables_per_rank: List[List[ShardedEmbeddingTable]] = [embedding_tables]

    # taking only the list for the first rank
    table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]
    return [table_group.table_names() for table_group in table_groups]


class TestGroupTablesPerRank(unittest.TestCase):
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        fused_params_group=st.sampled_from(
            [
                {
                    "cache_load_factor": 0.5,
                    "prefetch_pipeline": False,
                },
                {
                    "cache_load_factor": 0.3,
                    "prefetch_pipeline": True,
                },
            ]
        ),
        local_dim=st.sampled_from(list(range(160, 320, 40))),
        num_cw_shards=st.sampled_from([1, 2]),
        pooling_type=st.sampled_from(list(PoolingType)),
        compute_kernel=st.sampled_from(list(EmbeddingComputeKernel)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_should_group_together(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        fused_params_group: Dict[str, Any],
        local_dim: int,
        num_cw_shards: int,
        pooling_type: PoolingType,
        compute_kernel: EmbeddingComputeKernel,
    ) -> None:
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_group,
                compute_kernel=compute_kernel,
                embedding_dim=local_dim * num_cw_shards,
                local_cols=local_dim,
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        expected_table_names_by_groups = [["table_0", "table_1"]]
        self.assertEqual(
            _get_table_names_by_groups(tables),
            expected_table_names_by_groups,
        )

    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        local_dim=st.sampled_from(list(range(160, 320, 40))),
        num_cw_shards=st.sampled_from([1, 2]),
        pooling_type=st.sampled_from(list(PoolingType)),
        compute_kernel=st.sampled_from(list(EmbeddingComputeKernel)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_should_group_together_with_prefetch(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        local_dim: int,
        num_cw_shards: int,
        pooling_type: PoolingType,
        compute_kernel: EmbeddingComputeKernel,
    ) -> None:
        fused_params_groups = [
            {
                "cache_load_factor": 0.3,
                "prefetch_pipeline": True,
            },
            {
                "cache_load_factor": 0.5,
                "prefetch_pipeline": True,
            },
        ]
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_groups[i],
                compute_kernel=compute_kernel,
                embedding_dim=local_dim * num_cw_shards,
                local_cols=local_dim,
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        expected_table_names_by_groups = [["table_0", "table_1"]]
        self.assertEqual(
            _get_table_names_by_groups(tables),
            expected_table_names_by_groups,
        )

    @given(
        data_types=st.lists(
            st.sampled_from([DataType.FP16, DataType.FP32]),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        has_feature_processors=st.lists(
            st.sampled_from([False, True]), min_size=2, max_size=2, unique=True
        ),
        fused_params_group=st.sampled_from(
            [
                {
                    "cache_load_factor": 0.5,
                    "prefetch_pipeline": True,
                },
                {
                    "cache_load_factor": 0.3,
                    "prefetch_pipeline": True,
                },
            ],
        ),
        local_dim=st.lists(
            st.sampled_from([4, 10, 40]),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        embedding_dims=st.lists(
            st.sampled_from(list(range(160, 320, 40))),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        pooling_types=st.lists(
            st.sampled_from(list(PoolingType)), min_size=2, max_size=2, unique=True
        ),
        compute_kernels=st.lists(
            st.sampled_from(list(EmbeddingComputeKernel)),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        distinct_key=st.sampled_from(
            [
                "data_type",
                "has_feature_processor",
                "embedding_dim",
                "local_dim",
                "pooling_type",
                "compute_kernel",
            ]
        ),
    )
    @settings(max_examples=100, deadline=10000)
    def test_should_not_group_together(
        self,
        data_types: List[DataType],
        has_feature_processors: List[bool],
        fused_params_group: Dict[str, Any],
        local_dim: List[int],
        embedding_dims: List[int],
        pooling_types: List[PoolingType],
        compute_kernels: List[EmbeddingComputeKernel],
        distinct_key: str,
    ) -> None:
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=(
                    data_types[i] if distinct_key == "data_type" else data_types[0]
                ),
                pooling=(
                    pooling_types[i]
                    if distinct_key == "pooling_type"
                    else pooling_types[0]
                ),
                has_feature_processor=(
                    has_feature_processors[i]
                    if distinct_key == "has_feature_processor"
                    else has_feature_processors[0]
                ),
                fused_params=fused_params_group,  # can't hash dicts
                compute_kernel=(
                    compute_kernels[i]
                    if distinct_key == "compute_kernel"
                    else compute_kernels[0]
                ),
                embedding_dim=(
                    embedding_dims[i]
                    if distinct_key == "embedding_dim"
                    else embedding_dims[0]
                ),
                local_cols=(
                    local_dim[i] if distinct_key == "local_dim" else local_dim[0]
                ),
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        if distinct_key == "compute_kernel" and _get_compute_kernel_type(
            compute_kernels[0]
        ) == _get_compute_kernel_type(compute_kernels[1]):
            # Typically, a table with same group of kernel (e.g. FUSED vs FUSED_UVM)
            # would be grouped together. But if one of them are related to CACHE,
            # we'll group them separately because we don't want to add the burden of
            # prefetch()
            if _prefetch_and_cached(tables[0]) != _prefetch_and_cached(tables[1]):
                self.assertEqual(
                    sorted(_get_table_names_by_groups(tables)),
                    [["table_0"], ["table_1"]],
                )
            else:
                self.assertEqual(
                    _get_table_names_by_groups(tables),
                    [["table_0", "table_1"]],
                )
            return

        # emb dim bucketizier only in use when computer kernel is caching. Otherwise
        # they shall be grouped into the same bucket even with different dimensions
        if distinct_key == "local_dim" and not _prefetch_and_cached(tables[0]):
            self.assertEqual(
                _get_table_names_by_groups(tables),
                [["table_0", "table_1"]],
            )
            return

        # We never separate-group by embedding dim. So we don't care if it's distinct
        # or not. Just group them together.
        if distinct_key == "embedding_dim":
            self.assertEqual(
                _get_table_names_by_groups(tables),
                [["table_0", "table_1"]],
            )
            return

        # If both kernels are quantized, we assume this is inference which we no longer split by data_type
        # So if other attributes are the same between the two tables (regardless of data type), we combine them
        if (
            tables[0].compute_kernel == EmbeddingComputeKernel.QUANT
            and tables[1].compute_kernel == EmbeddingComputeKernel.QUANT
            and tables[0].pooling == tables[1].pooling
            and tables[0].has_feature_processor == tables[1].has_feature_processor
        ):

            self.assertEqual(
                sorted(_get_table_names_by_groups(tables)),
                [["table_0", "table_1"]],
            )
            return

        self.assertEqual(
            sorted(_get_table_names_by_groups(tables)),
            [["table_0"], ["table_1"]],
        )

    def test_use_one_tbe_per_table(
        self,
    ) -> None:

        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=DataType.FP16,
                pooling=PoolingType.SUM,
                has_feature_processor=False,
                fused_params={"use_one_tbe_per_table": i % 2 != 0},
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                embedding_dim=10,
                local_cols=5,
                num_embeddings=10000,
            )
            for i in range(5)
        ]

        # table_1 has two shards in the rank
        tables.append(
            ShardedEmbeddingTable(
                name="table_1",
                data_type=DataType.FP16,
                pooling=PoolingType.SUM,
                has_feature_processor=False,
                fused_params={"use_one_tbe_per_table": True},
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                embedding_dim=10,
                local_cols=3,
                num_embeddings=10000,
            )
        )

        # Even tables should all be grouped in a single TBE, odd tables should be in
        # their own TBEs.
        self.assertEqual(
            _get_table_names_by_groups(tables),
            [["table_0", "table_2", "table_4"], ["table_1", "table_1"], ["table_3"]],
        )

    def test_set_sharding_context_post_a2a(self) -> None:
        kjts = [
            KeyedJaggedTensor(
                keys=["dummy_id", "video_id", "owner_id", "xray_concepts", "dummy_id2"],
                values=torch.IntTensor([1] * 10),
                lengths=torch.IntTensor([1] * 10),
                stride_per_key_per_rank=[
                    [1, 2],
                    [1, 2],
                    [2, 3],
                    [5, 7],
                    [3, 4],
                ],
            ),
            KeyedJaggedTensor(
                keys=["dummy_id", "video_id", "owner_id", "xray_concepts", "dummy_id2"],
                values=torch.IntTensor([1] * 10),
                lengths=torch.IntTensor([1] * 10),
                stride_per_key_per_rank=[[3, 1], [5, 2], [7, 3], [1, 2], [6, 8]],
            ),
        ]
        for kjt in kjts:
            kjt._variable_stride_per_key = True

        ctx = EmbeddingCollectionContext(
            sharding_contexts=[
                SequenceShardingContext(batch_size_per_rank_per_feature=[]),
                SequenceShardingContext(batch_size_per_rank_per_feature=[]),
            ]
        )
        results = [
            [[1, 1, 2, 5, 3], [2, 2, 3, 7, 4]],
            [[3, 5, 7, 1, 6], [1, 2, 3, 2, 8]],
        ]
        _set_sharding_context_post_a2a(kjts, ctx)
        for context, result in zip(ctx.sharding_contexts, results):
            self.assertEqual(context.batch_size_per_rank_per_feature, result)


class TestECBucketMetadata(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        embedding_dim=st.sampled_from(list(range(160, 320, 40))),
        total_bucket=st.sampled_from([14, 20, 32, 40]),
        my_rank=st.integers(min_value=0, max_value=WORLD_SIZE - 1),
    )
    @settings(max_examples=10, deadline=10000)
    def test_bucket_metadata_calculation_util_disabled_in_oss_compatibility(
        self, data_type: DataType, embedding_dim: int, total_bucket: int, my_rank: int
    ) -> None:
        compute_kernels = [EmbeddingComputeKernel.SSD_VIRTUAL_TABLE] * WORLD_SIZE
        fused_params_groups = [
            {"cache_load_factor": 0.5},
        ] * WORLD_SIZE
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=PoolingType.NONE,
                has_feature_processor=False,
                fused_params=fused_params_groups[i],
                feature_names=[f"feature_{i}"],
                compute_kernel=compute_kernels[i],
                embedding_dim=embedding_dim,
                local_rows=10000 * (2 * i + 1) // WORLD_SIZE,
                local_cols=embedding_dim,
                num_embeddings=10000 * (2 * i + 1),
                total_num_buckets=total_bucket,
                use_virtual_table=True,
                local_metadata=ShardMetadata(
                    shard_offsets=[i * (10000 * (2 * i + 1) // WORLD_SIZE), 0],
                    shard_sizes=[10000 * (2 * i + 1) // WORLD_SIZE, embedding_dim],
                    placement=f"rank:{i}/cuda:{i}",
                ),
                global_metadata=ShardedTensorMetadata(
                    shards_metadata=[
                        ShardMetadata(
                            shard_offsets=[j * (10000 * (2 * i + 1) // WORLD_SIZE), 0],
                            shard_sizes=[
                                10000 * (2 * i + 1) // WORLD_SIZE,
                                embedding_dim,
                            ],
                            placement=f"rank:{j}/cuda:{j}",
                        )
                        for j in range(WORLD_SIZE)
                    ],
                    size=torch.Size([10000 * (2 * i + 1), embedding_dim]),
                ),
                virtual_table_eviction_policy=NoEvictionPolicy(),
            )
            for i in range(WORLD_SIZE)
        ]

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]

        # assert that they are grouped together
        self.assertEqual(len(table_groups), 1)

        table_group = table_groups[0]

        expect_failure = False
        for table in tables:
            if (
                table.num_embeddings % total_bucket != 0
                or total_bucket % WORLD_SIZE != 0
            ):
                expect_failure = True
                break

        with patch(
            "torchrec.distributed.batched_embedding_kernel.dist.get_world_size",
            return_value=WORLD_SIZE,
        ), patch(
            "torchrec.distributed.batched_embedding_kernel.dist.get_rank",
            return_value=my_rank,
        ):
            if expect_failure:
                with self.assertRaises(AssertionError):
                    zch_kv_emb = ZeroCollisionKeyValueEmbedding(
                        table_group, None, torch.device("cpu")
                    )
            else:
                zch_kv_emb = ZeroCollisionKeyValueEmbedding(
                    table_group, None, torch.device("cpu")
                )
                expected_tuple = [
                    (
                        total_bucket / WORLD_SIZE * my_rank,
                        total_bucket / WORLD_SIZE * (my_rank + 1),
                        table.num_embeddings / total_bucket,
                    )
                    for table in tables
                ]
                self.assertEqual(zch_kv_emb._bucket_spec, expected_tuple)
