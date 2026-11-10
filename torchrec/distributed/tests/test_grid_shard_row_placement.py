#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRID_SHARD lookups must return the rows the sharding plan says each rank holds.

Four ranks, two per TWRW group, hold two 256-wide tables as 2 row blocks x 2
column blocks: `t0` with 8 rows and `t1` with 7 (uneven row blocks). Row `g`
of a table is `1000 * g + column`, so a wrong lookup shows which row it read.
"""

import dataclasses
import unittest
from typing import Dict, List

import torch
from torch.distributed.remote_device import _remote_device
from torchrec.distributed.embedding_types import ShardedEmbeddingTable
from torchrec.distributed.sharding.grid_sharding import _column_block_buckets
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    get_module_to_default_sharders,
    grid_shard,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE, LOCAL_SIZE = 4, 2
DIM, COL_BLOCK = 256, 128
ROWS: Dict[str, int] = {"t0": 8, "t1": 7}


def _lookup_rows(t0_rank_order: List[int], rank: int, world_size: int) -> None:
    """Shard with `t0`'s column-0 row blocks on `t0_rank_order` and compare lookups."""
    with MultiProcessContext(rank, world_size, "gloo", LOCAL_SIZE) as ctx:
        tables = [
            EmbeddingBagConfig(
                name=name,
                embedding_dim=DIM,
                num_embeddings=rows,
                feature_names=[f"f_{name}"],
            )
            for name, rows in ROWS.items()
        ]
        weights = {
            f"embedding_bags.{name}.weight": (
                torch.arange(rows, dtype=torch.float32, device=ctx.device).unsqueeze(1)
                * 1000
                + torch.arange(DIM, dtype=torch.float32, device=ctx.device)
            )
            for name, rows in ROWS.items()
        }
        unsharded = EmbeddingBagCollection(tables=tables, device=ctx.device)
        unsharded.load_state_dict(weights)

        plan = construct_module_sharding_plan(
            unsharded,
            {name: grid_shard(host_indexes=[0, 1]) for name in ROWS},
            local_size=LOCAL_SIZE,
            world_size=world_size,
            device_type=ctx.device.type,
        )
        sharding = plan["t0"]
        spec = sharding.sharding_spec
        assert isinstance(spec, EnumerableShardingSpec)
        # Shards 0 and 1 are column block 0, row blocks 0 and 1.
        ranks = t0_rank_order + list(none_throws(sharding.ranks)[LOCAL_SIZE:])
        sharding.ranks = ranks
        spec.shards = [
            dataclasses.replace(
                s,
                placement=_remote_device(
                    f"rank:{r}/" + (f"cuda:{r}" if ctx.device.type == "cuda" else "cpu")
                ),
            )
            for s, r in zip(spec.shards, ranks)
        ]

        model = EmbeddingBagCollection(tables=tables, device=ctx.device)
        sharder = get_module_to_default_sharders()[type(model)]
        sharded = sharder.shard(
            module=model,
            params=plan,
            env=ShardingEnv.from_process_group(none_throws(ctx.pg)),
            device=ctx.device,
        )
        copy_state_dict(sharded.state_dict(), weights)

        # One id per sample, so each pooled output is a single row.
        batch_size = max(ROWS.values())
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[f"f_{name}" for name in ROWS],
            values=torch.cat(
                [torch.arange(batch_size, device=ctx.device) % r for r in ROWS.values()]
            ),
            lengths=torch.ones(
                len(ROWS) * batch_size, dtype=torch.int32, device=ctx.device
            ),
        )
        got, want = sharded(kjt).to_dict(), unsharded(kjt).to_dict()
        for key in want:
            rows_per_block = [
                (got[key][:, c] - c).div(1000).round().long().tolist()
                for c in range(0, DIM, COL_BLOCK)
            ]
            assert torch.equal(
                got[key], want[key]
            ), f"rank {rank}, {key}: column blocks returned rows {rows_per_block}"


class GridShardRowPlacementTest(MultiProcessTestBase):
    def test_rank_ordered_plan_matches_unsharded(self) -> None:
        self._run_multi_process_test(
            callable=_lookup_rows, world_size=WORLD_SIZE, t0_rank_order=[0, 1]
        )

    def test_out_of_order_plan_matches_unsharded(self) -> None:
        self._run_multi_process_test(
            callable=_lookup_rows, world_size=WORLD_SIZE, t0_rank_order=[1, 0]
        )


def _column_block(
    sizes_and_ranks: List[tuple[int, int]], local_rank: int
) -> ShardedEmbeddingTable:
    """One 4-wide column block of a table with the given (rows, rank) row shards."""
    shards, offset = [], 0
    for rows, rank in sizes_and_ranks:
        shards.append(
            ShardMetadata(
                shard_offsets=[offset, 0],
                shard_sizes=[rows, 4],
                placement=_remote_device(f"rank:{rank}/cpu"),
            )
        )
        offset += rows
    local = next(s for s in shards if s.placement.rank() == local_rank)
    return ShardedEmbeddingTable(
        num_embeddings=offset,
        embedding_dim=4,
        name="t",
        local_rows=local.shard_sizes[0],
        local_cols=4,
        local_metadata=local,
        global_metadata=ShardedTensorMetadata(
            shards_metadata=shards, size=torch.Size([offset, 4])
        ),
    )


class GridShardBucketTest(unittest.TestCase):
    def test_zero_row_shards_tie_by_rank(self) -> None:
        # 2 rows over 4 ranks: planner sizes [1, 1, 0, 0], offsets [0, 1, 2, 2].
        table = _column_block([(1, 2), (1, 0), (0, 3), (0, 1)], local_rank=0)
        self.assertEqual(
            _column_block_buckets(table, local_size=4),
            {(0, 2): 0, (1, 0): 1, (2, 1): 2, (2, 3): 3},
        )

    def test_offsets_off_the_bucket_grid_raise(self) -> None:
        # Bucket 1 of 8 rows over 2 ranks starts at row 4, not 2.
        table = _column_block([(2, 0), (6, 1)], local_rank=0)
        with self.assertRaisesRegex(ValueError, "do not match the 2 TWRW buckets"):
            _column_block_buckets(table, local_size=2)

    def test_more_row_shards_than_ranks_raise(self) -> None:
        table = _column_block([(4, 0), (4, 1), (0, 2)], local_rank=0)
        with self.assertRaisesRegex(ValueError, "do not match the 2 TWRW buckets"):
            _column_block_buckets(table, local_size=2)
