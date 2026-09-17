#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List, Tuple

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensorMetadata
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torchrec.distributed.batched_embedding_kernel import (
    _get_grid_shard_cw_count,
    KeyValueEmbeddingFusedOptimizer,
)
from torchrec.distributed.sharding_plan import _calculate_grid_shard_sizes_and_offsets


def _grid_shards(
    row_sizes: List[int], num_cw_blocks: int, block_size: int
) -> List[ShardMetadata]:
    """Lay out `row_sizes` row shards within each of `num_cw_blocks` column blocks."""
    shards = []
    for block in range(num_cw_blocks):
        row_offset = 0
        for row_size in row_sizes:
            shards.append(
                ShardMetadata(
                    shard_sizes=[row_size, block_size],
                    shard_offsets=[row_offset, block * block_size],
                    placement=f"rank:{len(shards)}/cuda:0",
                )
            )
            row_offset += row_size
    return shards


class GridShardCwCountTest(unittest.TestCase):
    def test_counts_column_blocks_not_row_shards(self) -> None:
        # A pod of 8 ranks: 2 column blocks x 8 row shards of a 300M x 256 table.
        shards = _grid_shards([37_500_000] * 8, num_cw_blocks=2, block_size=128)
        self.assertEqual(len(shards), 16)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)

    def test_independent_of_row_shard_count(self) -> None:
        # Same table, same column split, a narrower pod: the answer must not move.
        shards = _grid_shards([150_000_000] * 2, num_cw_blocks=2, block_size=128)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)

    def test_uneven_rows_with_empty_trailing_shard(self) -> None:
        shards = _grid_shards([4, 4, 2, 0], num_cw_blocks=4, block_size=16)
        self.assertEqual(_get_grid_shard_cw_count(shards), 4)

    def test_single_column_block(self) -> None:
        shards = _grid_shards([8, 8], num_cw_blocks=1, block_size=64)
        self.assertEqual(_get_grid_shard_cw_count(shards), 1)


class GridShardOptimizerMetadataTest(unittest.TestCase):
    """Covers the sizing that raised 'Total volume of shards does not match tensor volume'."""

    def _rowwise_metadata(
        self, rows_per_shard: int, shards_per_block: int, num_cw_blocks: int
    ) -> Tuple[ShardedTensorMetadata, int]:
        # Rows are derived so the row shards tile the table exactly, which is what
        # the planner emits and what the volume check demands.
        rows = rows_per_shard * shards_per_block
        shards = _grid_shards(
            [rows_per_shard] * shards_per_block,
            num_cw_blocks=num_cw_blocks,
            block_size=128,
        )
        table_md = ShardedTensorMetadata(
            shards_metadata=shards,
            size=torch.Size([rows, 128 * num_cw_blocks]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )
        # Unbound: the method never touches `self`.
        _, rowwise_md = (
            KeyValueEmbeddingFusedOptimizer.get_optimizer_rowwise_shard_metadata_and_global_metadata(
                cast(KeyValueEmbeddingFusedOptimizer, None),
                table_md,
                torch.empty(0),
                1,
                True,
            )
        )
        return rowwise_md, rows

    def test_size_covers_shards_exactly(self) -> None:
        # The runtime invariant checked by check_tensor(): declared size must equal
        # the summed shard volume, or ShardedTensor construction raises.
        md, rows = self._rowwise_metadata(
            rows_per_shard=37, shards_per_block=8, num_cw_blocks=2
        )
        self.assertEqual(md.size[0], sum(s.shard_sizes[0] for s in md.shards_metadata))
        self.assertEqual(md.size, torch.Size([rows * 2]))

    def test_size_independent_of_row_shards_per_block(self) -> None:
        # Same table width and column split, a narrower pod: size must not move.
        md, rows = self._rowwise_metadata(
            rows_per_shard=148, shards_per_block=2, num_cw_blocks=2
        )
        self.assertEqual(md.size[0], sum(s.shard_sizes[0] for s in md.shards_metadata))
        self.assertEqual(md.size, torch.Size([rows * 2]))


class GridShardPlannerLayoutTest(unittest.TestCase):
    """Drives the planner's own shard generator, so a layout change breaks these."""

    # 4 hosts x 2 GPUs per host. The planner spreads row shards over the whole pod
    # (Topology.intra_group_size), so per-host GPU count is not the row-shard count.
    HOSTS_PER_POD = 4
    GPUS_PER_HOST = 2
    POD_SIZE = HOSTS_PER_POD * GPUS_PER_HOST

    def _planner_shards(
        self, rows: int, columns: int, col_wise_shard_dim: int
    ) -> List[ShardMetadata]:
        shard_sizes, shard_offsets = _calculate_grid_shard_sizes_and_offsets(
            rows, self.POD_SIZE, columns, col_wise_shard_dim
        )
        return [
            ShardMetadata(
                shard_sizes=sizes,
                shard_offsets=offsets,
                placement=f"rank:{rank}/cuda:0",
            )
            for rank, (sizes, offsets) in enumerate(zip(shard_sizes, shard_offsets))
        ]

    def test_cw_count_on_multi_host_pod(self) -> None:
        shards = self._planner_shards(
            rows=300_000_000, columns=256, col_wise_shard_dim=128
        )
        self.assertEqual(len(shards), 2 * self.POD_SIZE)
        self.assertEqual(_get_grid_shard_cw_count(shards), 2)
        # Must not track the per-host GPU count, which would give 8 here.
        self.assertNotEqual(
            _get_grid_shard_cw_count(shards), len(shards) // self.GPUS_PER_HOST
        )

    def test_optimizer_size_covers_shards_on_multi_host_pod(self) -> None:
        rows, columns = 300_000_000, 256
        shards = self._planner_shards(rows, columns, col_wise_shard_dim=128)
        table_md = ShardedTensorMetadata(
            shards_metadata=shards,
            size=torch.Size([rows, columns]),
            tensor_properties=TensorProperties(dtype=torch.float32),
        )
        _, rowwise_md = (
            KeyValueEmbeddingFusedOptimizer.get_optimizer_rowwise_shard_metadata_and_global_metadata(
                cast(KeyValueEmbeddingFusedOptimizer, None),
                table_md,
                torch.empty(0),
                1,
                True,
            )
        )
        self.assertEqual(
            rowwise_md.size[0],
            sum(s.shard_sizes[0] for s in rowwise_md.shards_metadata),
        )
        self.assertEqual(rowwise_md.size, torch.Size([rows * 2]))
