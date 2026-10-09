#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
import math
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from torchrec.distributed.embedding_sharding import bucketize_kjt_before_all2all
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.sharding.twrw_sharding import (
    _build_twrw_dist_layouts,
    _rank_order_indices,
    _shard_count_groups,
    TwRwEmbeddingDistLayout,
    TwRwFeatureDistLayout,
    TwRwPooledEmbeddingDist,
    TwRwSparseFeaturesDist,
)
from torchrec.distributed.types import ShardMetadata
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import PoolingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.types import DataType


def _all_feature_shards(num_shards_per_feature: list[int]) -> list[tuple[int, int]]:
    return [
        (feature, shard)
        for feature, num_shards in enumerate(num_shards_per_feature)
        for shard in range(num_shards)
    ]


def _bucket_values(kjt: KeyedJaggedTensor) -> list[list[int]]:
    """Values in each key of a one-sample-per-key bucketize output."""
    values = kjt.values().tolist()
    out: list[list[int]] = []
    start = 0
    for length in kjt.lengths().tolist():
        out.append(values[start : start + length])
        start += length
    return out


def _feature_names(table: str, num_features: int) -> list[str]:
    return (
        [table] if num_features == 1 else [f"{table}f{i}" for i in range(num_features)]
    )


def _layout(
    tables_per_rank: list[list[str]],
    dim_by_table: dict[str, int],
    local_size: int = 1,
    features_by_table: dict[str, int] | None = None,
    placement_ranks_by_table: dict[str, list[int]] | None = None,
) -> tuple[TwRwFeatureDistLayout, TwRwEmbeddingDistLayout, list[Any]]:
    """Build a layout without a distributed environment."""
    table_placement_ranks: dict[str, list[int]] = placement_ranks_by_table or {}
    if not table_placement_ranks:
        for rank, tables in enumerate(tables_per_rank):
            for table in tables:
                table_placement_ranks.setdefault(table, []).append(rank)
    names_by_table = {
        table: _feature_names(table, (features_by_table or {}).get(table, 1))
        for tables in tables_per_rank
        for table in tables
    }
    return _build_twrw_dist_layouts(
        grouped_configs_per_rank=[
            [
                GroupedEmbeddingConfig(
                    data_type=DataType.FP32,
                    pooling=PoolingType.SUM,
                    is_weighted=False,
                    has_feature_processor=False,
                    compute_kernel=EmbeddingComputeKernel.DENSE,
                    embedding_tables=[
                        ShardedEmbeddingTable(
                            name=table,
                            num_embeddings=16,
                            embedding_dim=dim_by_table[table],
                            local_cols=dim_by_table[table],
                            feature_names=names_by_table[table],
                            embedding_names=names_by_table[table],
                            local_metadata=ShardMetadata(
                                shard_sizes=[1, dim_by_table[table]],
                                shard_offsets=[
                                    table_placement_ranks[table].index(rank),
                                    0,
                                ],
                                placement=f"rank:{rank}/cpu",
                            ),
                        )
                        for table in tables
                    ],
                )
            ]
            for rank, tables in enumerate(tables_per_rank)
        ],
        table_placement_ranks=table_placement_ranks,
        local_size=local_size,
    )


def _total_dim(tables_per_group: list[list[str]], dim_by_table: dict[str, int]) -> int:
    return sum(dim_by_table[table] for tables in tables_per_group for table in tables)


class TwRwSparseFeaturesDistTest(unittest.TestCase):
    """Keep each feature's bucket count, including for out-of-range ids."""

    # One feature lives on a 2-rank group; the other spans both groups.
    HASH_SIZE: int = 16
    SINGLE_GROUP_BUCKETS: int = 2
    MULTI_GROUP_BUCKETS: int = 4
    # 20 and 21 take FBGEMM's out-of-range path.
    SINGLE_GROUP_IDS: list[int] = [0, 9, 20, 21]
    MULTI_GROUP_IDS: list[int] = [1, 5, 11, 15]

    def test_legacy_constructor_derives_uniform_routing(self) -> None:
        pg = MagicMock()
        pg.size.return_value = 4
        with patch(
            "torchrec.distributed.sharding.twrw_sharding.KJTAllToAll"
        ) as all_to_all:
            input_dist = TwRwSparseFeaturesDist(
                pg,
                2,
                [2, 2, 1, 1],
                [12, 24, 30],
                torch.device("cpu"),
            )

        self.assertEqual(input_dist._rank_order_indices, [0, 1, 3, 4, 2, 5])
        torch.testing.assert_close(
            input_dist._feature_block_sizes_tensor,
            torch.tensor([6, 12, 15], dtype=torch.int32),
        )
        all_to_all.assert_called_once_with(pg=pg, splits=[2, 2, 1, 1], stagger=2)

    def _kjt(self, keys: list[str], ids_per_key: list[list[int]]) -> KeyedJaggedTensor:
        values: list[int] = [i for ids in ids_per_key for i in ids]
        return KeyedJaggedTensor(
            keys=keys,
            values=torch.tensor(values, dtype=torch.int64),
            lengths=torch.tensor([len(ids) for ids in ids_per_key], dtype=torch.int32),
        )

    def _fbgemm_block_sizes(self, num_buckets_per_feature: list[int]) -> torch.Tensor:
        return torch.tensor(
            [
                math.ceil(self.HASH_SIZE / buckets)
                for buckets in num_buckets_per_feature
            ],
            dtype=torch.int64,
        )

    def _grouped(
        self, num_shards_per_feature: list[int], ids_per_key: list[list[int]]
    ) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
        """Run production grouping by shard count over a mixed KJT."""
        feature_shards = _all_feature_shards(num_shards_per_feature)
        shard_counts = tuple(num_shards_per_feature)
        rank_order_indices = _rank_order_indices(shard_counts, feature_shards)
        index_by_feature_shard = dict(zip(feature_shards, rank_order_indices))
        shard_count_groups = _shard_count_groups(shard_counts)
        group_feature_indices = [
            feature for _, features in shard_count_groups for feature in features
        ]
        bucketizer = TwRwSparseFeaturesDist.__new__(TwRwSparseFeaturesDist)
        torch.nn.Module.__init__(bucketizer)
        bucketizer._shard_count_groups = shard_count_groups
        bucketizer._shard_count_group_feature_indices_tensor = torch.tensor(
            group_feature_indices, dtype=torch.int32
        )
        bucketizer._feature_block_sizes_tensor = self._fbgemm_block_sizes(
            [num_shards_per_feature[feature] for feature in group_feature_indices]
        )
        out = bucketizer._bucketize(
            self._kjt(
                [f"f{i}" for i in range(len(num_shards_per_feature))], ids_per_key
            ),
            bucketize_pos=False,
        )
        return _bucket_values(out), index_by_feature_shard

    def _alone(
        self,
        num_buckets: int,
        ids: list[int],
        block_size_bucket_count: int | None = None,
    ) -> list[list[int]]:
        """Bucketize one feature, optionally reusing another count's block size."""
        out = bucketize_kjt_before_all2all(
            self._kjt(["f"], [ids]),
            num_buckets=num_buckets,
            block_sizes=self._fbgemm_block_sizes(
                [
                    (
                        num_buckets
                        if block_size_bucket_count is None
                        else block_size_bucket_count
                    )
                ]
            ),
            output_permute=False,
            bucketize_pos=False,
        )[0]
        return _bucket_values(out)

    def test_in_range_ids_ignore_the_bucket_count(self) -> None:
        in_range = [0, 9]
        self.assertEqual(self._alone(self.SINGLE_GROUP_BUCKETS, in_range), [[0], [1]])
        self.assertEqual(
            self._alone(
                self.MULTI_GROUP_BUCKETS,
                in_range,
                block_size_bucket_count=self.SINGLE_GROUP_BUCKETS,
            )[: self.SINGLE_GROUP_BUCKETS],
            [[0], [1]],
        )

    def test_shared_max_bucket_count_moves_out_of_range_ids(self) -> None:
        at_own_count = self._alone(self.SINGLE_GROUP_BUCKETS, self.SINGLE_GROUP_IDS)
        at_shared_max = self._alone(
            self.MULTI_GROUP_BUCKETS,
            self.SINGLE_GROUP_IDS,
            block_size_bucket_count=self.SINGLE_GROUP_BUCKETS,
        )[: self.SINGLE_GROUP_BUCKETS]
        self.assertNotEqual(
            at_own_count,
            at_shared_max,
            "if these matched, the bucket count would not affect routing and "
            "the per-count grouping would be unnecessary",
        )

    def test_grouping_leaves_the_single_group_table_untouched(self) -> None:
        bucket_values, index_by_feature_shard = self._grouped(
            [self.SINGLE_GROUP_BUCKETS, self.MULTI_GROUP_BUCKETS],
            [self.SINGLE_GROUP_IDS, self.MULTI_GROUP_IDS],
        )

        single_group = [
            bucket_values[index_by_feature_shard[(0, shard)]]
            for shard in range(self.SINGLE_GROUP_BUCKETS)
        ]
        self.assertEqual(
            single_group,
            self._alone(self.SINGLE_GROUP_BUCKETS, self.SINGLE_GROUP_IDS),
        )

        multi_group = [
            bucket_values[index_by_feature_shard[(1, shard)]]
            for shard in range(self.MULTI_GROUP_BUCKETS)
        ]
        self.assertEqual(
            multi_group,
            self._alone(self.MULTI_GROUP_BUCKETS, self.MULTI_GROUP_IDS),
        )
        self.assertEqual(
            sum(len(bucket) for bucket in single_group + multi_group),
            len(self.SINGLE_GROUP_IDS) + len(self.MULTI_GROUP_IDS),
        )

    def test_single_group_matches_the_ungrouped_call(self) -> None:
        ids = [self.SINGLE_GROUP_IDS, [2, 7, 30]]
        bucket_values, _ = self._grouped([self.SINGLE_GROUP_BUCKETS] * 2, ids)

        ungrouped = _bucket_values(
            bucketize_kjt_before_all2all(
                self._kjt(["f0", "f1"], ids),
                num_buckets=self.SINGLE_GROUP_BUCKETS,
                block_sizes=self._fbgemm_block_sizes([self.SINGLE_GROUP_BUCKETS] * 2),
                output_permute=False,
                bucketize_pos=False,
            )[0]
        )
        self.assertEqual(bucket_values, ungrouped)


class TwRwDistLayoutsTest(unittest.TestCase):
    # (local_size, features of each table on each group). No table spans groups,
    # so every case must match the legacy uniform routing used by GRID_SHARD.
    SINGLE_GROUP_TOPOLOGIES: list[tuple[int, list[list[int]]]] = [
        (1, [[1]]),  # one rank holding one table
        (2, [[1, 1]]),  # one group of two ranks
        (1, [[1], [1]]),  # two groups of one rank
        (2, [[1, 1], [1]]),  # uneven table counts per group
        (2, [[1, 1, 1], []]),  # a group holding nothing
        (2, [[2], [1]]),  # a table with two features
        (3, [[3, 1], [2], [1]]),  # mixed feature counts, three groups
        (2, [[], []]),  # no tables at all
    ]

    def test_single_group_plans_match_the_uniform_plan(self) -> None:
        for local_size, features_per_table_by_group in self.SINGLE_GROUP_TOPOLOGIES:
            with self.subTest(
                local_size=local_size, groups=features_per_table_by_group
            ):
                features_by_table = {
                    f"g{group}t{table}": num_features
                    for group, tables in enumerate(features_per_table_by_group)
                    for table, num_features in enumerate(tables)
                }
                tables_per_rank = [
                    [f"g{group}t{table}" for table in range(len(tables))]
                    for group, tables in enumerate(features_per_table_by_group)
                    for _ in range(local_size)
                ]
                features_per_rank = [
                    sum(features_by_table[table] for table in tables)
                    for tables in tables_per_rank
                ]

                feature_layout, _, _ = _layout(
                    tables_per_rank,
                    dict.fromkeys(features_by_table, 4),
                    local_size,
                    features_by_table,
                )
                self.assertEqual(
                    feature_layout.shard_counts,
                    (local_size,) * sum(features_by_table.values()),
                )
                group_offsets = [0] + list(
                    itertools.accumulate(features_per_rank[::local_size])
                )
                self.assertEqual(
                    feature_layout.rank_order_indices,
                    tuple(
                        shard * sum(features_by_table.values()) + feature
                        for group in range(len(features_per_rank) // local_size)
                        for shard in range(local_size)
                        for feature in range(
                            group_offsets[group], group_offsets[group + 1]
                        )
                    ),
                )

    def test_derives_the_feature_list_and_both_orderings(self) -> None:
        feature_layout, embedding_layout, features = _layout(
            [["a", "b"], ["a", "b"], ["b"], ["b"]],
            {"a": 4, "b": 6},
            local_size=2,
        )

        # Each feature appears once, ordered by its first destination-rank
        # occurrence, with one routed input partition per shard.
        self.assertEqual(
            [
                (
                    feature.feature_name,
                    shard_count,
                    embedding_dim,
                )
                for feature, shard_count, embedding_dim in zip(
                    features,
                    feature_layout.shard_counts,
                    embedding_layout.embedding_dims,
                )
            ],
            [("a", 2, 4), ("b", 4, 6)],
        )
        # Group 0 emits a and b; group 1 emits another partial for b.
        self.assertEqual(embedding_layout.feature_indices_by_twrw_group, ((0, 1), (1,)))
        # One (feature, shard) pair per rank per table it holds, in rank order:
        # (a, 0) (b, 0) | (a, 1) (b, 1) | (b, 2) | (b, 3), resolved against a
        # bucketize output of a's 2 buckets then b's 4.
        self.assertEqual(
            feature_layout.rank_order_indices,
            (0, 2, 1, 3, 4, 5),
        )

    def test_single_group_layout_preserves_feature_metadata(self) -> None:
        feature_layout, embedding_layout, features = _layout(
            [["a", "b"], ["a", "b"], ["c"], ["c"]],
            {"a": 3, "b": 5, "c": 7},
            local_size=2,
            features_by_table={"a": 2, "b": 1, "c": 2},
        )

        self.assertEqual(
            [
                (
                    feature.feature_name,
                    feature.embedding_name,
                    hash_size,
                    embedding_dim,
                    none_throws(feature.shard_metadata).shard_offsets,
                    none_throws(feature.shard_metadata).shard_sizes,
                    none_throws(none_throws(feature.shard_metadata).placement).rank(),
                )
                for feature, hash_size, embedding_dim in zip(
                    features,
                    feature_layout.hash_sizes,
                    embedding_layout.embedding_dims,
                )
            ],
            [
                ("af0", "af0", 16, 3, [0, 0], [1, 3], 0),
                ("af1", "af1", 16, 3, [0, 0], [1, 3], 0),
                ("b", "b", 16, 5, [0, 0], [1, 5], 0),
                ("cf0", "cf0", 16, 7, [0, 0], [1, 7], 2),
                ("cf1", "cf1", 16, 7, [0, 0], [1, 7], 2),
            ],
        )
        self.assertEqual(feature_layout.hash_sizes, (16, 16, 16, 16, 16))
        self.assertEqual(embedding_layout.embedding_dims, (3, 3, 5, 7, 7))

    def test_nonascending_placement_keeps_shard_zero_metadata(self) -> None:
        feature_layout, embedding_layout, features = _layout(
            [["a"], ["a"], ["a"], ["a"]],
            {"a": 4},
            local_size=2,
            placement_ranks_by_table={"a": [2, 3, 0, 1]},
        )

        self.assertEqual(feature_layout.rank_order_indices, (2, 3, 0, 1))
        self.assertEqual(embedding_layout.feature_indices_by_twrw_group, ((0,), (0,)))
        metadata = none_throws(features[0].shard_metadata)
        self.assertEqual(metadata.shard_offsets, [0, 0])
        self.assertEqual(none_throws(metadata.placement).rank(), 2)


class TwRwPooledEmbeddingDistTest(unittest.TestCase):
    """Compare optimized slice sums with one addition per physical partial."""

    # (tables each group holds, embedding dim per table), one rank per group.
    GROUP_PLACEMENTS: list[tuple[list[list[str]], dict[str, int]]] = [
        ([["a"]], {"a": 4}),  # nothing repeats
        ([["a"], ["a"]], {"a": 4}),  # one table on both groups
        ([["a", "b"], ["b"]], {"a": 3, "b": 2}),  # repeat at the end
        ([["a", "b"], ["a"]], {"a": 3, "b": 2}),  # repeat at the start
        ([["a", "b", "c"], ["b"]], {"a": 4, "b": 6, "c": 5}),  # repeat in the middle
        ([["a", "b"], ["a", "b"]], {"a": 3, "b": 2}),  # a whole group repeats
        ([["a", "b"], ["b", "a"]], {"a": 3, "b": 2}),  # reordered, defeats coalescing
        ([["a"], ["a"], ["a"]], {"a": 2}),  # three groups
        ([["a", "b"], [], ["b"]], {"a": 1, "b": 1}),  # a group holding nothing
        ([["a", "b", "c"], ["a", "c"]], {"a": 2, "b": 3, "c": 1}),  # a subset repeats
    ]

    @staticmethod
    def _reference(
        tables_per_group: list[list[str]],
        dim_by_table: dict[str, int],
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Adds one partial at a time into the final feature order."""
        destination_by_table: dict[str, int] = {}
        offset = 0
        for table in dict.fromkeys(
            table for tables in tables_per_group for table in tables
        ):
            destination_by_table[table] = offset
            offset += dim_by_table[table]
        expected = torch.zeros(tensor.shape[0], offset)
        source = 0
        for tables in tables_per_group:
            for table in tables:
                destination = destination_by_table[table]
                embedding_dim = dim_by_table[table]
                expected[:, destination : destination + embedding_dim] += tensor[
                    :, source : source + embedding_dim
                ]
                source += embedding_dim
        return expected

    @staticmethod
    def _output_dist(layout: TwRwEmbeddingDistLayout) -> TwRwPooledEmbeddingDist:
        return TwRwPooledEmbeddingDist(
            rank=0,
            cross_pg=MagicMock(),
            intra_pg=MagicMock(),
            layout=layout,
        )

    def test_legacy_constructor_derives_identity_layout(self) -> None:
        output_dist = TwRwPooledEmbeddingDist(
            0,
            MagicMock(),
            MagicMock(),
            [8, 7],
            [[3, 5], [7]],
            torch.device("cpu"),
        )

        self.assertEqual(output_dist._layout.embedding_dims, (3, 5, 7))
        self.assertEqual(
            output_dist._layout.feature_indices_by_twrw_group, ((0, 1), (2,))
        )
        self.assertEqual(output_dist._dim_sums_by_twrw_group, [8, 7])

    @classmethod
    def _combine(
        cls, layout: TwRwEmbeddingDistLayout, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Applies the combine, which is skipped when no feature repeats."""
        callback = cls._output_dist(layout)._combine_callback(
            list(layout.embedding_dims)
        )
        return tensor if callback is None else callback(tensor)

    def _assert_matches_reference(
        self, tables_per_group: list[list[str]], dim_by_table: dict[str, int]
    ) -> None:
        _, embedding_layout, _ = _layout(tables_per_group, dim_by_table)
        total_dim = _total_dim(tables_per_group, dim_by_table)
        tensor = torch.arange(3 * total_dim, dtype=torch.float32).view(3, total_dim)
        torch.testing.assert_close(
            self._combine(embedding_layout, tensor),
            self._reference(tables_per_group, dim_by_table, tensor),
        )

    def test_variable_batch_combine_flattens_per_feature_batches(self) -> None:
        _, embedding_layout, _ = _layout(
            [["a", "b", "c"], ["b"]], {"a": 2, "b": 3, "c": 1}
        )

        batch_size_per_partial, callback = self._output_dist(
            embedding_layout
        )._variable_batch_combine([2, 1, 3])
        self.assertEqual(batch_size_per_partial, [2, 1, 3, 1])

        tensor = torch.arange(13, dtype=torch.float32)
        expected = torch.cat((tensor[:4], tensor[4:7] + tensor[10:13], tensor[7:10]))
        torch.testing.assert_close(none_throws(callback)(tensor), expected)

    def test_variable_batch_rejects_a_wrong_length(self) -> None:
        _, embedding_layout, _ = _layout([["a", "b"]], {"a": 3, "b": 5})

        with self.assertRaisesRegex(ValueError, r"expected 2 feature batch sizes"):
            self._output_dist(embedding_layout)._variable_batch_combine([1, 2, 3])

    def test_gradient_reaches_every_partial(self) -> None:
        tables_per_group = [["a", "b"], ["b"]]
        dim_by_table = {"a": 3, "b": 2}
        _, embedding_layout, _ = _layout(tables_per_group, dim_by_table)

        total_dim = _total_dim(tables_per_group, dim_by_table)
        # `requires_grad` after the view, so `tensor` is a leaf and keeps a grad.
        tensor = torch.arange(2 * total_dim, dtype=torch.float32).view(2, total_dim)
        tensor.requires_grad_(True)
        self._combine(embedding_layout, tensor).sum().backward()
        torch.testing.assert_close(none_throws(tensor.grad), torch.ones(2, total_dim))

    def test_matches_reference_across_group_placements(self) -> None:
        for tables_per_group, dim_by_table in self.GROUP_PLACEMENTS:
            with self.subTest(groups=tables_per_group):
                self._assert_matches_reference(tables_per_group, dim_by_table)

    def test_single_group_layout_is_a_pass_through(self) -> None:
        _, embedding_layout, _ = _layout([["a", "b"]], {"a": 3, "b": 5})
        output_dist = self._output_dist(embedding_layout)

        # Nothing to sum, so no callback is attached to either awaitable and
        # the variable batch sizes pass through.
        self.assertIsNone(
            output_dist._combine_callback(list(embedding_layout.embedding_dims))
        )
        batch_size_per_partial, callback = output_dist._variable_batch_combine([2, 3])
        self.assertEqual(batch_size_per_partial, [2, 3])
        self.assertIsNone(callback)
