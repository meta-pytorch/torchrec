#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grid-aware RES ship: owners stream full-width rows, peers stream no grid rows.

Four gloo ranks hold one 20 x 8 table, grid sharded into 2 row blocks x 2
column blocks of width 4, behind a fake TBE that records what it streams.
"""

from dataclasses import dataclass
from itertools import accumulate
from typing import Any, cast, Dict, List, Optional, Tuple
from unittest import mock

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    EmbeddingLocation,
    PrefetchedInfo,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch.distributed._shard.metadata import ShardMetadata
from torchrec.distributed import batched_embedding_kernel
from torchrec.distributed.batched_embedding_kernel import (
    _maybe_install_res_grid_assembly,
    _res_grid_ship_hbm_rows,
    _res_grid_tables,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase

ROWS, FULL_DIM, LOCAL_DIM, ROW_BLOCK = 20, 8, 4, 10
WORLD_SIZE = 4
# rank -> (row offset, column offset) of the grid table shard it holds.
PLACEMENT: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (ROW_BLOCK, 0),
    2: (0, LOCAL_DIM),
    3: (ROW_BLOCK, LOCAL_DIM),
}
OTHER_ROWS, OTHER_DIM = 6, 3


def _global_table() -> torch.Tensor:
    return torch.arange(ROWS * FULL_DIM, dtype=torch.float32).view(ROWS, FULL_DIM)


def _other_table() -> torch.Tensor:
    return -1.0 - torch.arange(OTHER_ROWS * OTHER_DIM, dtype=torch.float32).view(
        OTHER_ROWS, OTHER_DIM
    )


@dataclass
class _FakeGlobalMetadata:
    shards_metadata: List[ShardMetadata]


@dataclass
class _FakeTable:
    name: str
    local_rows: int
    local_cols: int
    num_embeddings: int
    embedding_dim: int
    local_metadata: Optional[ShardMetadata]
    global_metadata: Optional[_FakeGlobalMetadata]


@dataclass
class _FakeConfig:
    embedding_tables: List[_FakeTable]


def _grid_shards() -> List[ShardMetadata]:
    return [
        ShardMetadata(
            shard_offsets=[row_offset, col_offset],
            shard_sizes=[ROW_BLOCK, LOCAL_DIM],
            placement=f"rank:{rank}/cpu",
        )
        for rank, (row_offset, col_offset) in PLACEMENT.items()
    ]


def _config(rank: int, with_other_table: bool) -> _FakeConfig:
    shards = _grid_shards()
    tables = [
        _FakeTable(
            "grid_t",
            ROW_BLOCK,
            LOCAL_DIM,
            ROWS,
            FULL_DIM,
            shards[rank],
            _FakeGlobalMetadata(shards),
        )
    ]
    if with_other_table:
        tables.insert(
            0,
            _FakeTable(
                "other_t", OTHER_ROWS, OTHER_DIM, OTHER_ROWS, OTHER_DIM, None, None
            ),
        )
    return _FakeConfig(tables)


class _FakeStreamer:
    def __init__(self) -> None:
        self.stream_calls: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def join_hbm_dispatch_and_workers(self) -> None:
        pass

    def stream(self, indices: torch.Tensor, weights: torch.Tensor, **_: Any) -> None:
        self.stream_calls.append((indices.clone(), weights.clone()))


class _FakeResParams:
    res_use_copy_done_token: bool = False


class _FakeTBE:
    """Just the TBE surface the grid install and ship read."""

    def __init__(self, rank: int, with_other_table: bool) -> None:
        row_offset, col_offset = PLACEMENT[rank]
        grid_shard = _global_table()[
            row_offset : row_offset + ROW_BLOCK, col_offset : col_offset + LOCAL_DIM
        ]
        tables = ([_other_table()] if with_other_table else []) + [grid_shard]
        self.embedding_specs: List[Tuple[int, int, EmbeddingLocation, Any]] = [
            (table.size(0), table.size(1), EmbeddingLocation.DEVICE, None)
            for table in tables
        ]
        self.feature_table_map: List[int] = list(range(len(tables)))
        self.weights_dev: torch.Tensor = torch.cat(
            [table.flatten() for table in tables]
        )
        dims = [table.size(1) for table in tables]
        self._res_D_offsets_cpu = torch.tensor([0, *accumulate(dims)])
        self._res_weights_offsets_cpu = torch.tensor(
            [0, *accumulate(table.numel() for table in tables)][:-1]
        )
        self._res_rows_per_table_cpu = torch.tensor([table.size(0) for table in tables])
        self._row_starts: List[int] = [
            0,
            *accumulate(table.size(0) for table in tables),
        ]
        self.max_D: int = max(dims)
        self.current_device = torch.device("cpu")
        self.timestep = 0
        self.uuid = f"rank{rank}"
        self.enable_raw_embedding_streaming = True
        self._res_hbm_streaming = True
        self._res_hbm_row_identities: Optional[torch.Tensor] = None
        self._raw_embedding_streamer = _FakeStreamer()
        self._res_hbm_count_buf: torch.Tensor = torch.zeros(1, dtype=torch.long)
        self._res_hbm_copy_done: torch.Tensor = torch.zeros(1, dtype=torch.int32)
        self._res_require_copy = True
        self.res_params = _FakeResParams()
        self._tables = tables

    def log(self, msg: str) -> None:
        pass

    def _grown_res_host_buf_if_needed(
        self,
        buf: Optional[torch.Tensor],
        num_rows: int,
        row_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if buf is not None and buf.shape[0] >= num_rows:
            return buf
        # Garbage, not zeros: every byte the ship reads must have been written.
        return torch.full((max(num_rows, 1), *row_shape), 7777, dtype=dtype)

    def _gather_res_rows(
        self, res_indices: torch.Tensor, res_weights: torch.Tensor, weights: Any
    ) -> None:
        res_weights.zero_()
        for out_row, linear_row in enumerate(res_indices.tolist()):
            table_idx = max(
                idx
                for idx, row_start in enumerate(self._row_starts[:-1])
                if row_start <= linear_row
            )
            row = self._tables[table_idx][linear_row - self._row_starts[table_idx]]
            res_weights[out_row, : row.numel()] = row


class _FakePrefetchedInfo:
    def __init__(self, drained_rows: Optional[List[int]]) -> None:
        self.res_hbm_indices: Optional[torch.Tensor] = (
            None
            if drained_rows is None
            else torch.tensor(drained_rows, dtype=torch.long)
        )


StreamCalls = List[Tuple[List[int], torch.Tensor]]


def _padded(rows: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.pad(rows, (0, width - rows.size(1)))


def _ship_and_check(
    rank: int,
    world_size: int,
    with_other_table: bool,
    drained_rows_by_rank: Dict[int, Optional[List[int]]],
    expected_stream_calls_by_rank: Dict[int, StreamCalls],
    chunk_rows: int,
) -> None:
    batched_embedding_kernel._RES_GRID_CHUNK_ROWS = chunk_rows
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        tbe = _FakeTBE(rank, with_other_table)
        _maybe_install_res_grid_assembly(
            cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
            cast(GroupedEmbeddingConfig, _config(rank, with_other_table)),
            True,
        )
        # Through the installed method, so the install is covered too.
        cast(SplitTableBatchedEmbeddingBagsCodegen, tbe)._ship_hbm_rows(
            cast(PrefetchedInfo, _FakePrefetchedInfo(drained_rows_by_rank[rank]))
        )
        stream_calls = tbe._raw_embedding_streamer.stream_calls
        expected_stream_calls = expected_stream_calls_by_rank[rank]
        assert len(stream_calls) == len(expected_stream_calls), (
            f"rank {rank}: {len(stream_calls)} stream calls, "
            f"expected {len(expected_stream_calls)}"
        )
        for (indices, weights), (expected_indices, expected_weights) in zip(
            stream_calls, expected_stream_calls
        ):
            assert (
                indices.tolist() == expected_indices
            ), f"rank {rank}: indices {indices.tolist()} != {expected_indices}"
            assert torch.equal(
                weights, expected_weights
            ), f"rank {rank}: weights\n{weights}\n!=\n{expected_weights}"
    finally:
        dist.destroy_process_group()


class ResGridAssemblyTest(MultiProcessTestBase):
    def _ship(
        self,
        with_other_table: bool,
        drained_rows_by_rank: Dict[int, Optional[List[int]]],
        expected_stream_calls_by_rank: Dict[int, StreamCalls],
        chunk_rows: int = batched_embedding_kernel._RES_GRID_CHUNK_ROWS,
    ) -> None:
        self._run_multi_process_test(
            callable=_ship_and_check,
            world_size=WORLD_SIZE,
            with_other_table=with_other_table,
            drained_rows_by_rank=drained_rows_by_rank,
            expected_stream_calls_by_rank=expected_stream_calls_by_rank,
            chunk_rows=chunk_rows,
        )

    def test_owner_alternates_by_row_block(self) -> None:
        role_by_rank = {}
        for rank in range(WORLD_SIZE):
            [grid_table] = _res_grid_tables(
                cast(GroupedEmbeddingConfig, _config(rank, False)),
                cast(SplitTableBatchedEmbeddingBagsCodegen, _FakeTBE(rank, False)),
            )
            role_by_rank[rank] = (
                grid_table.is_owner,
                grid_table.owner_rank,
                [peer_rank for _, _, peer_rank in grid_table.peers],
            )
        # Row block 0 is owned by column block 0 (rank 0), row block 1 by
        # column block 1 (rank 3).
        self.assertEqual(
            role_by_rank,
            {
                0: (True, 0, [2]),
                1: (False, 3, []),
                2: (False, 0, []),
                3: (True, 3, [1]),
            },
        )

    def test_owners_stream_full_rows_peers_stream_nothing(self) -> None:
        # Two-row chunks: each peer's three rows go over in two hand-offs.
        drained_rows = [0, 3, 9]
        table = _global_table()
        self._ship(
            False,
            {rank: drained_rows for rank in range(WORLD_SIZE)},
            {
                0: [(drained_rows, table[drained_rows])],
                1: [],
                2: [],
                3: [(drained_rows, table[[ROW_BLOCK + row for row in drained_rows]])],
            },
            chunk_rows=2,
        )

    def test_nothing_drained_streams_nothing(self) -> None:
        self._ship(
            False,
            dict.fromkeys(range(WORLD_SIZE)),
            {rank: [] for rank in range(WORLD_SIZE)},
        )

    def test_no_grid_rows_drained_skips_hand_off(self) -> None:
        # Ranks 0, 1 drained only other_t; peers 2, 3 drained nothing: no hand-off.
        other = _other_table()
        self._ship(
            True,
            {0: [1], 1: [4], 2: None, 3: None},
            {
                0: [([1], _padded(other[[1]], FULL_DIM))],
                1: [([4], _padded(other[[4]], LOCAL_DIM))],
                2: [],
                3: [],
            },
        )

    def test_other_tables_stream_unchanged_and_padded(self) -> None:
        # Owners pad other_t (rows 0-5) to FULL_DIM; non-owners pad it to
        # LOCAL_DIM, their max_D.
        other, table = _other_table(), _global_table()
        grid_linear_row = OTHER_ROWS + 2
        owner_other = _padded(other[[1, 4]], FULL_DIM)
        peer_other = _padded(other[[1, 4]], LOCAL_DIM)
        self._ship(
            True,
            {rank: [1, 4, grid_linear_row] for rank in range(WORLD_SIZE)},
            {
                0: [([1, 4, grid_linear_row], torch.cat([owner_other, table[[2]]]))],
                1: [([1, 4], peer_other)],
                2: [([1, 4], peer_other)],
                3: [
                    (
                        [1, 4, grid_linear_row],
                        torch.cat([owner_other, table[[ROW_BLOCK + 2]]]),
                    )
                ],
            },
        )

    def test_mpzch_identities_raise(self) -> None:
        # The grid ship forwards no identities, so a TBE holding them fails loudly.
        tbe = _FakeTBE(0, False)
        tbe._res_hbm_row_identities = torch.full((ROW_BLOCK,), -1)
        with self.assertRaisesRegex(RuntimeError, "MPZCH"):
            _res_grid_ship_hbm_rows(
                cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
                _FakePrefetchedInfo([0]),
            )

    def test_no_install_without_process_group(self) -> None:
        tbe = _FakeTBE(0, False)
        _maybe_install_res_grid_assembly(
            cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
            cast(GroupedEmbeddingConfig, _config(0, False)),
            True,
        )
        self.assertFalse(hasattr(tbe, "_res_grid_tables"))

    def test_grid_assembly_off_leaves_tbe_alone(self) -> None:
        # Negative control: every install condition holds except the fused param.
        tbe = _FakeTBE(0, False)
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            _maybe_install_res_grid_assembly(
                cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
                cast(GroupedEmbeddingConfig, _config(0, False)),
                False,
            )
        finally:
            dist.destroy_process_group()
        self.assertNotIn("_res_grid_tables", vars(tbe))
        self.assertNotIn("_ship_hbm_rows", vars(tbe))

    def test_hbm_lane_off_warns_and_leaves_tbe_alone(self) -> None:
        # Fused param on and RES on for this TBE, but this rank's HBM lane is off.
        tbe = _FakeTBE(0, False)
        tbe._res_hbm_streaming = False
        with self.assertLogs(batched_embedding_kernel.logger, "WARNING") as logs:
            _maybe_install_res_grid_assembly(
                cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
                cast(GroupedEmbeddingConfig, _config(0, False)),
                True,
            )
        self.assertIn("RES HBM lane is off on this rank", logs.output[0])
        self.assertNotIn("_ship_hbm_rows", vars(tbe))

    def test_plan_row_routing_off_warns_and_leaves_tbe_alone(self) -> None:
        # Every install condition holds except GRID_SHARD plan row routing.
        tbe = _FakeTBE(0, False)
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            with mock.patch(
                "torchrec.distributed.utils.is_grid_shard_plan_row_routing_enabled",
                return_value=False,
            ), self.assertLogs(batched_embedding_kernel.logger, "WARNING") as logs:
                _maybe_install_res_grid_assembly(
                    cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
                    cast(GroupedEmbeddingConfig, _config(0, False)),
                    True,
                )
        finally:
            dist.destroy_process_group()
        self.assertIn("plan row routing is off", logs.output[0])
        self.assertNotIn("_ship_hbm_rows", vars(tbe))

    def test_res_off_tbe_does_not_warn(self) -> None:
        # Fused param on, but RES is off for this TBE (offline steps, TBE groups
        # without a RES table): no warning, no install.
        tbe = _FakeTBE(0, False)
        tbe.enable_raw_embedding_streaming = False
        tbe._res_hbm_streaming = False
        with self.assertNoLogs(batched_embedding_kernel.logger, "WARNING"):
            _maybe_install_res_grid_assembly(
                cast(SplitTableBatchedEmbeddingBagsCodegen, tbe),
                cast(GroupedEmbeddingConfig, _config(0, False)),
                True,
            )
        self.assertNotIn("_ship_hbm_rows", vars(tbe))
