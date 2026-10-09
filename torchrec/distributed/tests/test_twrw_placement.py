#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest.mock import patch

import torch
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.sharding.twrw_sharding import BaseTwRwEmbeddingSharding
from torchrec.distributed.types import ParameterSharding, ShardingType, ShardMetadata
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingTableConfig

_LOCAL_SIZE: int = 4


class _PlacementResolver:
    """Placement resolver without the live distributed environment."""

    _resolve_placement_ranks: Any = BaseTwRwEmbeddingSharding._resolve_placement_ranks

    def __init__(
        self,
        local_size: int,
        is_2D_parallel: bool = False,
    ) -> None:
        self._local_size = local_size
        self._is_2D_parallel = is_2D_parallel


def _resolve(
    num_twrw_groups: int,
    plan_ranks: list[int],
    num_shards: int | None = None,
    table_group: int = 0,
    local_size: int = _LOCAL_SIZE,
    is_2D_parallel: bool = False,
) -> list[int]:
    resolver = _PlacementResolver(local_size, is_2D_parallel)
    return resolver._resolve_placement_ranks(
        table_name="table_0",
        num_twrw_groups=num_twrw_groups,
        plan_ranks=plan_ranks,
        num_shards=len(plan_ranks) if num_shards is None else num_shards,
        table_group=table_group,
    )


class ResolvePlacementRanksTest(unittest.TestCase):
    def setUp(self) -> None:
        gate = patch(
            "torchrec.distributed.utils.is_twrw_multi_group_enabled",
            return_value=True,
        )
        gate.start()
        self.addCleanup(gate.stop)

    def test_shard_and_rank_counts_must_match(self) -> None:
        for placed, num_shards in ((8, 7), (0, 8)):
            with self.subTest(placed=placed):
                with self.assertRaisesRegex(
                    ValueError, rf"{placed} placement ranks for {num_shards} shards"
                ):
                    _resolve(
                        num_twrw_groups=2,
                        plan_ranks=list(range(placed)),
                        num_shards=num_shards,
                    )

    def test_single_group_shard_shortfall_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"2 shards for a TWRW group of 4 ranks"
        ):
            _resolve(num_twrw_groups=1, plan_ranks=[0, 1], num_shards=2)

    def test_single_group_shard_excess_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"needs exactly 4 shards"):
            _resolve(num_twrw_groups=1, plan_ranks=[0], num_shards=6)

    def test_num_twrw_groups_below_one_is_rejected(self) -> None:
        for num_twrw_groups in (0, -1):
            with self.subTest(num_twrw_groups=num_twrw_groups):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"num_twrw_groups={num_twrw_groups} must be >= 1",
                ):
                    _resolve(
                        num_twrw_groups=num_twrw_groups,
                        plan_ranks=[0, 1, 2, 3],
                    )

    def test_multi_group_under_2d_parallelism_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"num_twrw_groups=2 is not supported under 2D parallelism"
        ):
            _resolve(
                num_twrw_groups=2,
                plan_ranks=list(range(8)),
                is_2D_parallel=True,
            )

    def test_killswitch_rejects_multi_group_only(self) -> None:
        with patch(
            "torchrec.distributed.utils.is_twrw_multi_group_enabled",
            return_value=False,
        ):
            with self.assertRaisesRegex(NotImplementedError, "disabled by killswitch"):
                _resolve(num_twrw_groups=2, plan_ranks=list(range(8)))

        with patch(
            "torchrec.distributed.utils.is_twrw_multi_group_enabled",
            side_effect=AssertionError(
                "single-group placement must not read the multi-group killswitch"
            ),
        ):
            self.assertEqual(
                _resolve(num_twrw_groups=1, plan_ranks=list(range(4))),
                list(range(4)),
            )

    def test_single_group_placed_wider_than_a_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"the plan places it on 8 ranks, more than the 4 in a TWRW group",
        ):
            _resolve(num_twrw_groups=1, plan_ranks=list(range(8)))

    def test_rank_count_must_match_num_twrw_groups_times_local_size(self) -> None:
        for placed in (6, 12):
            with self.subTest(placed=placed):
                with self.assertRaisesRegex(
                    ValueError,
                    r"num_twrw_groups=2 needs 8 ranks at a TWRW group width "
                    r"of 4, but the "
                    rf"plan places it on {placed}",
                ):
                    _resolve(num_twrw_groups=2, plan_ranks=list(range(placed)))

    def test_repeated_rank_is_rejected(self) -> None:
        for num_twrw_groups, plan_ranks in (
            (2, [0, 1, 2, 2, 4, 5, 6, 7]),
            (2, [0, 1, 2, 3, 0, 1, 2, 3]),
            (3, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]),
        ):
            with self.subTest(plan_ranks=plan_ranks):
                with self.assertRaisesRegex(ValueError, r"repeat a rank"):
                    _resolve(
                        num_twrw_groups=num_twrw_groups,
                        plan_ranks=plan_ranks,
                    )

    def test_span_wider_than_num_twrw_groups_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"8 ranks span 3 groups"):
            _resolve(num_twrw_groups=2, plan_ranks=[0, 1, 2, 3, 4, 5, 8, 9])

    def test_single_group_derives_the_whole_group(self) -> None:
        self.assertEqual(
            _resolve(num_twrw_groups=1, plan_ranks=[8, 9, 10, 11], table_group=2),
            [8, 9, 10, 11],
        )
        self.assertEqual(
            _resolve(num_twrw_groups=1, plan_ranks=[0, 1, 2, 3], table_group=0),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            _resolve(num_twrw_groups=1, plan_ranks=[11, 10, 9, 8], table_group=2),
            [8, 9, 10, 11],
        )

    def test_single_group_accepts_a_first_rank_only_plan(self) -> None:
        self.assertEqual(
            _resolve(
                num_twrw_groups=1,
                plan_ranks=[8],
                num_shards=4,
                table_group=2,
            ),
            [8, 9, 10, 11],
        )

    def test_single_group_wider_than_a_group_is_allowed_under_2d(self) -> None:
        self.assertEqual(
            _resolve(
                num_twrw_groups=1,
                plan_ranks=list(range(8)),
                table_group=1,
                is_2D_parallel=True,
            ),
            [4, 5, 6, 7],
        )

    def test_multi_group_preserves_plan_order(self) -> None:
        self.assertEqual(
            _resolve(
                num_twrw_groups=2,
                plan_ranks=[2, 3, 0, 1],
                table_group=1,
                local_size=2,
            ),
            [2, 3, 0, 1],
        )
        self.assertEqual(
            _resolve(
                num_twrw_groups=3,
                plan_ranks=[4, 5, 0, 1, 2, 3],
                local_size=2,
            ),
            [4, 5, 0, 1, 2, 3],
        )

        self.assertEqual(
            _resolve(
                num_twrw_groups=2,
                plan_ranks=[0, 1, 2, 3],
                local_size=2,
            ),
            [0, 1, 2, 3],
        )


class _Sharder:
    """`_shard` with only the attributes it reads; see `_PlacementResolver`."""

    _resolve_placement_ranks: Any = BaseTwRwEmbeddingSharding._resolve_placement_ranks
    _shard: Any = BaseTwRwEmbeddingSharding._shard

    def __init__(self, world_size: int, local_size: int) -> None:
        self._world_size = world_size
        self._local_size = local_size
        self._is_2D_parallel = False
        self._pg = None
        self._env: Any = type("_Env", (), {"output_dtensor": False})()


def _sharding_info(
    name: str,
    rows: int,
    dim: int,
    ranks: list[int],
    num_shards: int,
    num_twrw_groups: int | None = None,
) -> EmbeddingShardingInfo:
    rows_per_shard = math.ceil(rows / num_shards)
    return EmbeddingShardingInfo(
        embedding_config=EmbeddingTableConfig(
            num_embeddings=rows,
            embedding_dim=dim,
            name=name,
            feature_names=[f"f_{name}"],
            embedding_names=[f"f_{name}"],
        ),
        param_sharding=ParameterSharding(
            sharding_type=ShardingType.TABLE_ROW_WISE.value,
            compute_kernel="dense",
            ranks=ranks,
            sharding_spec=EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=[
                            min(rows_per_shard, rows - min(i * rows_per_shard, rows)),
                            dim,
                        ],
                        shard_offsets=[min(i * rows_per_shard, rows), 0],
                        placement=f"rank:{i}/cpu",
                    )
                    for i in range(num_shards)
                ]
            ),
            num_twrw_groups=num_twrw_groups,
        ),
        param=torch.empty(rows, dim, device="meta"),
    )


class ShardPlacementTest(unittest.TestCase):
    def setUp(self) -> None:
        gate = patch(
            "torchrec.distributed.utils.is_twrw_multi_group_enabled",
            return_value=True,
        )
        gate.start()
        self.addCleanup(gate.stop)

    def test_single_group_pairs_shards_with_the_derived_group(self) -> None:
        sharder = _Sharder(world_size=8, local_size=4)
        info = _sharding_info("t", rows=400, dim=8, ranks=[4], num_shards=4)
        per_rank, placement_ranks = sharder._shard([info])

        self.assertEqual(placement_ranks["t"], [4, 5, 6, 7])
        placed = {r: t for r, t in enumerate(per_rank) if t}
        self.assertEqual(sorted(placed), [4, 5, 6, 7])
        self.assertEqual(
            [
                none_throws(placed[r][0].local_metadata).shard_offsets[0]
                for r in sorted(placed)
            ],
            [0, 100, 200, 300],
        )

    def test_multi_group_pairs_shards_with_the_plan_order(self) -> None:
        sharder = _Sharder(world_size=8, local_size=4)
        # Deliberately not ascending: group 1 before group 0.
        ranks = [4, 5, 6, 7, 0, 1, 2, 3]
        info = _sharding_info(
            "t",
            rows=800,
            dim=8,
            ranks=ranks,
            num_shards=8,
            num_twrw_groups=2,
        )
        with patch(
            "torchrec.distributed.sharding.twrw_sharding.is_2d_pod_size_enabled",
            side_effect=AssertionError(
                "a valid multi-group placement must not read the width killswitch"
            ),
        ):
            per_rank, placement_ranks = sharder._shard([info])

        self.assertEqual(placement_ranks["t"], ranks)
        # `ranks[i]` holds `shards[i]`, so rank 4 holds shard 0 and rank 0 holds
        # shard 4. Sorting the placement would swap them.
        offsets = {
            r: none_throws(t[0].local_metadata).shard_offsets[0]
            for r, t in enumerate(per_rank)
            if t
        }
        self.assertEqual(offsets[4], 0)
        self.assertEqual(offsets[0], 400)
        self.assertEqual(sorted(offsets.values()), [i * 100 for i in range(8)])

    def test_multi_group_shard_rank_mismatch_raises(self) -> None:
        sharder = _Sharder(world_size=8, local_size=4)
        info = _sharding_info(
            "t",
            rows=800,
            dim=8,
            ranks=list(range(8)),
            num_shards=8,
            num_twrw_groups=2,
        )
        info.param_sharding.ranks = list(range(7))
        with self.assertRaisesRegex(ValueError, r"7 placement ranks for 8 shards"):
            sharder._shard([info])

    def test_multi_group_accepts_empty_trailing_shards(self) -> None:
        sharder = _Sharder(world_size=8, local_size=4)
        info = _sharding_info(
            "t",
            rows=5,
            dim=8,
            ranks=list(range(8)),
            num_shards=8,
            num_twrw_groups=2,
        )

        per_rank, _ = sharder._shard([info])

        self.assertEqual(
            [
                none_throws(tables[0].local_metadata).shard_offsets[0]
                for tables in per_rank
            ],
            [0, 1, 2, 3, 4, 5, 5, 5],
        )
