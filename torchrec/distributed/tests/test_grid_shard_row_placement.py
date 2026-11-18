#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRID_SHARD lookups must return the rows the sharding plan says each rank holds.

Four ranks, two per TWRW group. `t0` (8 rows) and `t1` (7 rows, uneven row
blocks) are 256 wide, two column blocks, one per node. `t2` (6 rows, two
features) is 128 wide and lives on node 1 only, so the nodes hold different
feature counts. Row `g` of a table is `1000 * g + column`, so a wrong lookup
shows which row it read. The 2D tests run two replica groups of that layout.
"""

import dataclasses
import unittest
from typing import cast, Dict, List, NamedTuple, Tuple
from unittest import mock

import torch
from torch import nn
from torch.distributed.remote_device import _remote_device
from torchrec.distributed.embedding_types import ShardedEmbeddingTable
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DMPCollection
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
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardingPlan,
    ShardMetadata,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

WORLD_SIZE, LOCAL_SIZE = 4, 2
COL_BLOCK = 128


class _Table(NamedTuple):
    rows: int
    dim: int
    features: int
    hosts: List[int]


TABLES: Dict[str, _Table] = {
    "t0": _Table(rows=8, dim=256, features=1, hosts=[0, 1]),
    "t1": _Table(rows=7, dim=256, features=1, hosts=[0, 1]),
    "t2": _Table(rows=6, dim=128, features=2, hosts=[1]),
}
# Tables whose first column block gets its two row blocks swapped between ranks.
SWAPPED = ("t0", "t2")
# What uniform routing returns for `t0` under the swapped plan: column block 0 reads
# the other row block, column block 1 (not swapped) is right.
MIXED_ROWS = (
    "f_t0_0: column blocks returned rows "
    "[[4, 5, 6, 7, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]"
)


def _features(name: str) -> List[str]:
    return [f"f_{name}_{i}" for i in range(TABLES[name].features)]


def _model_and_plan(ctx: MultiProcessContext, out_of_order: bool) -> Tuple[
    List[EmbeddingBagConfig],
    Dict[str, torch.Tensor],
    EmbeddingBagCollection,
    EmbeddingModuleShardingPlan,
]:
    """Tables, their weights, the unsharded model, and a GRID plan over `WORLD_SIZE` ranks."""
    tables = [
        EmbeddingBagConfig(
            name=name,
            embedding_dim=t.dim,
            num_embeddings=t.rows,
            feature_names=_features(name),
        )
        for name, t in TABLES.items()
    ]
    weights = {
        f"embedding_bags.{name}.weight": (
            torch.arange(t.rows, dtype=torch.float32, device=ctx.device).unsqueeze(1)
            * 1000
            + torch.arange(t.dim, dtype=torch.float32, device=ctx.device)
        )
        for name, t in TABLES.items()
    }
    unsharded = EmbeddingBagCollection(tables=tables, device=ctx.device)
    unsharded.load_state_dict(weights)

    plan = construct_module_sharding_plan(
        unsharded,
        {name: grid_shard(host_indexes=t.hosts) for name, t in TABLES.items()},
        local_size=LOCAL_SIZE,
        world_size=WORLD_SIZE,
        device_type=ctx.device.type,
    )
    for name, sharding in plan.items():
        spec = sharding.sharding_spec
        assert isinstance(spec, EnumerableShardingSpec)
        ranks = list(none_throws(sharding.ranks))
        if out_of_order and name in SWAPPED:
            # Shards 0 and 1 are the first column block's row blocks 0 and 1.
            ranks[0], ranks[1] = ranks[1], ranks[0]
            sharding.ranks = ranks
        # The simulated nodes share one host, whose workers use cuda:<rank>.
        spec.shards = [
            dataclasses.replace(
                s,
                placement=_remote_device(
                    f"rank:{r}/" + (f"cuda:{r}" if ctx.device.type == "cuda" else "cpu")
                ),
            )
            for s, r in zip(spec.shards, ranks)
        ]
    return tables, weights, unsharded, plan


def _assert_lookups_match(
    ctx: MultiProcessContext, sharded: nn.Module, unsharded: nn.Module, rank: int
) -> None:
    """Look up every row of every feature, one id per sample, and compare."""
    batch_size = max(t.rows for t in TABLES.values())
    keys = [f for name in TABLES for f in _features(name)]
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.cat(
            [
                torch.arange(batch_size, device=ctx.device) % TABLES[name].rows
                for name in TABLES
                for _ in _features(name)
            ]
        ),
        lengths=torch.ones(
            len(keys) * batch_size, dtype=torch.int32, device=ctx.device
        ),
    )
    got, want = sharded(kjt).to_dict(), unsharded(kjt).to_dict()
    for key in want:
        if not torch.equal(got[key], want[key]):
            rows_per_block = [
                (got[key][:, c] - c).div(1000).round().long().tolist()
                for c in range(0, got[key].shape[1], COL_BLOCK)
            ]
            raise AssertionError(
                f"rank {rank}, {key}: column blocks returned rows {rows_per_block}"
            )


def _lookup_rows(
    out_of_order: bool, two_d: bool, routing: bool, rank: int, world_size: int
) -> None:
    """Shard (1D, or 2D with replica groups of `WORLD_SIZE` ranks) and compare lookups.

    With plan routing off, an out-of-order plan must give the mixed rows of the old
    uniform routing; every other combination must match the unsharded tables.
    """
    with mock.patch(
        "torchrec.distributed.utils.is_grid_shard_plan_row_routing_enabled",
        return_value=routing,
    ), MultiProcessContext(rank, world_size, "gloo", LOCAL_SIZE) as ctx:
        tables, weights, unsharded, plan = _model_and_plan(ctx, out_of_order)
        model = EmbeddingBagCollection(tables=tables, device=ctx.device)
        if two_d:
            sharded = DMPCollection(
                module=model,
                device=ctx.device,
                plan=ShardingPlan({"": plan}),
                world_size=world_size,
                sharding_group_size=WORLD_SIZE,
                global_pg=none_throws(ctx.pg),
                node_group_size=LOCAL_SIZE,
                sharders=[
                    cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder())
                ],
            )
        else:
            sharded = get_module_to_default_sharders()[type(model)].shard(
                module=model,
                params=plan,
                env=ShardingEnv.from_process_group(none_throws(ctx.pg)),
                device=ctx.device,
            )
        copy_state_dict(sharded.state_dict(), weights)
        if routing or not out_of_order:
            _assert_lookups_match(ctx, sharded, unsharded, rank)
            return
        try:
            _assert_lookups_match(ctx, sharded, unsharded, rank)
        except AssertionError as e:
            if MIXED_ROWS not in str(e):
                raise
        else:
            raise AssertionError(f"rank {rank}: expected {MIXED_ROWS}")


class GridShardRowPlacementTest(MultiProcessTestBase):
    def _run(self, out_of_order: bool, two_d: bool, routing: bool) -> None:
        self._run_multi_process_test(
            callable=_lookup_rows,
            world_size=2 * WORLD_SIZE if two_d else WORLD_SIZE,
            out_of_order=out_of_order,
            two_d=two_d,
            routing=routing,
        )

    def test_rank_ordered_plan_matches_unsharded(self) -> None:
        self._run(out_of_order=False, two_d=False, routing=True)

    def test_out_of_order_plan_matches_unsharded(self) -> None:
        self._run(out_of_order=True, two_d=False, routing=True)

    def test_out_of_order_plan_matches_unsharded_2d(self) -> None:
        self._run(out_of_order=True, two_d=True, routing=True)

    def test_routing_off_rank_ordered_plan_matches_unsharded(self) -> None:
        self._run(out_of_order=False, two_d=False, routing=False)

    def test_routing_off_mixes_rows_of_out_of_order_plan(self) -> None:
        self._run(out_of_order=True, two_d=False, routing=False)

    def test_routing_off_mixes_rows_of_out_of_order_plan_2d(self) -> None:
        self._run(out_of_order=True, two_d=True, routing=False)


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

    def test_wrong_row_shard_count_raises(self) -> None:
        for sizes_and_ranks in ([(4, 0), (4, 1), (0, 2)], [(8, 0)]):
            with self.subTest(sizes_and_ranks=sizes_and_ranks):
                table = _column_block(sizes_and_ranks, local_rank=0)
                with self.assertRaisesRegex(
                    ValueError, "do not match the 2 TWRW buckets"
                ):
                    _column_block_buckets(table, local_size=2)

    def test_shard_without_rank_placement_raises(self) -> None:
        for placement in (None, _remote_device("cpu")):
            with self.subTest(placement=placement):
                table = _column_block([(4, 0), (4, 1)], local_rank=0)
                shards = none_throws(table.global_metadata).shards_metadata
                shards[1] = dataclasses.replace(shards[1], placement=placement)
                with self.assertRaisesRegex(
                    ValueError,
                    "table t, column offset 0: every row shard needs a rank placement",
                ):
                    _column_block_buckets(table, local_size=2)
