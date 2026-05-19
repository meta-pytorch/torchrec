#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import copy
import unittest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.pec_embedding import (
    BackwardPartitionContext,
    ForwardPartitionContext,
    PECEmbeddingCollectionSharder,
    ShardedPECEmbeddingCollection,
)
from torchrec.distributed.shard import shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.pec_embedding_modules import PECEmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

EMBEDDING_TABLES: List[EmbeddingConfig] = [
    EmbeddingConfig(
        name="table_0",
        feature_names=["feature_0"],
        embedding_dim=8,
        num_embeddings=16,
    ),
    EmbeddingConfig(
        name="table_1",
        feature_names=["feature_1"],
        embedding_dim=8,
        num_embeddings=8,
    ),
]


class PECEmbeddingCollectionTest(unittest.TestCase):
    """Unit tests for PECEmbeddingCollection (unsharded) — no distributed setup."""

    def test_forward(self) -> None:
        ec = EmbeddingCollection(tables=EMBEDDING_TABLES, device=torch.device("cpu"))
        pec = PECEmbeddingCollection(ec)

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 1, 2, 3, 4, 5]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        )
        out: Dict[str, JaggedTensor] = pec(kjt)

        self.assertIn("feature_0", out)
        self.assertIn("feature_1", out)
        self.assertEqual(out["feature_0"].values().shape[1], 8)
        self.assertEqual(out["feature_1"].values().shape[1], 8)


class PECEmbeddingCollectionSharderTest(unittest.TestCase):
    """Unit tests for PECEmbeddingCollectionSharder — no distributed setup."""

    def test_sharding_types_rw_only(self) -> None:
        sharder = PECEmbeddingCollectionSharder()
        types = sharder.sharding_types(compute_device_type="cuda")
        self.assertEqual(types, [ShardingType.ROW_WISE.value])

    def test_shardable_parameters(self) -> None:
        ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES[:1], device=torch.device("cpu")
        )
        pec = PECEmbeddingCollection(ec)

        sharder = PECEmbeddingCollectionSharder()
        params = sharder.shardable_parameters(pec)

        self.assertIn("table_0", params)
        self.assertEqual(params["table_0"].shape, (16, 8))


class PECSparseArch(nn.Module):
    """Simple model wrapping PECEmbeddingCollection for sharded testing."""

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
    ) -> None:
        super().__init__()
        self._pec_ec: PECEmbeddingCollection = PECEmbeddingCollection(
            EmbeddingCollection(tables=tables, device=device),
        )

    def forward(
        self, kjt: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, Dict[str, JaggedTensor]]:
        ec_out = self._pec_ec(kjt)
        pred = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_1"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, ec_out


def _shard_pec(
    tables: List[EmbeddingConfig],
    ctx: MultiProcessContext,
    sharder: ModuleSharder[nn.Module],
    local_size: Optional[int] = None,
) -> PECSparseArch:
    sparse_arch = PECSparseArch(tables, torch.device("meta"))

    module_sharding_plan = construct_module_sharding_plan(
        sparse_arch._pec_ec,
        per_param_sharding={table.name: row_wise() for table in tables},
        local_size=local_size,
        world_size=ctx.world_size,
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        sharder=sharder,
    )

    return shard_modules(  # pyre-ignore[7]
        module=copy.deepcopy(sparse_arch),
        plan=ShardingPlan({"_pec_ec": module_sharding_plan}),
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=ctx.device,
    )


def _test_sharding(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)

        assert isinstance(sharded_sparse_arch._pec_ec, ShardedPECEmbeddingCollection)
        assert isinstance(
            sharded_sparse_arch._pec_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )

        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        for _ in range(2):
            loss, _ = sharded_sparse_arch(kjt_input)
            loss.backward()


def _verify_forward_permute(
    forward_ctx: ForwardPartitionContext | None,
    expected: List[int],
) -> None:
    assert forward_ctx is not None
    assert forward_ctx.permute.tolist() == expected


def _verify_backward_permutes(
    backward_ctx: BackwardPartitionContext | None,
    expected_ol_permute: List[int] | None,
    expected_nol_permute: List[int] | None,
) -> None:
    if expected_ol_permute is None:
        assert backward_ctx is None
    else:
        assert backward_ctx is not None
        assert backward_ctx.ol_permute.tolist() == expected_ol_permute
        assert expected_nol_permute is not None
        assert backward_ctx.nol_permute.tolist() == expected_nol_permute


def _verify_forward_splits(
    forward_ctx: ForwardPartitionContext,
    expected_input_splits: Tuple[List[int], List[int]],
    expected_output_splits: Tuple[List[int], List[int]],
) -> None:
    assert forward_ctx.splits.input_splits == expected_input_splits
    assert forward_ctx.splits.output_splits == expected_output_splits


def _verify_partition_embeddings(
    embs: torch.Tensor,
    expected_specs: List[Tuple[str, int]],
    ref_ec: EmbeddingCollection,
    device: torch.device,
) -> None:
    assert embs.shape[0] == len(expected_specs)
    if len(expected_specs) > 0:
        expected_tensor = torch.stack(
            [
                # pyre-ignore[16]: embeddings[tname] is nn.Embedding
                ref_ec.embeddings[tname].weight[oid]
                for tname, oid in expected_specs
            ]
        ).to(device)
        torch.testing.assert_close(embs, expected_tensor)


def _verify_merged_output(
    jt_dict: Dict[str, JaggedTensor],
    kjt_input: KeyedJaggedTensor,
    original_values: torch.Tensor,
    original_lengths: torch.Tensor,
    tables: List[EmbeddingConfig],
    ref_ec: EmbeddingCollection,
    device: torch.device,
) -> None:
    feature_to_table = {}
    for table in tables:
        for fname in table.feature_names:
            feature_to_table[fname] = table.name

    assert set(jt_dict.keys()) == set(feature_to_table.keys())

    stride = kjt_input.stride()
    offset = 0
    for feat_idx, fname in enumerate(kjt_input.keys()):
        tname = feature_to_table[fname]
        feat_lengths = original_lengths[feat_idx * stride : (feat_idx + 1) * stride]
        num_values = feat_lengths.sum().item()
        feat_ids = original_values[offset : offset + num_values]

        actual_jt = jt_dict[fname]
        torch.testing.assert_close(actual_jt.lengths().cpu(), feat_lengths)
        if num_values > 0:
            expected_embs = torch.stack(
                [
                    # pyre-ignore[16]: embeddings[tname] is nn.Embedding
                    ref_ec.embeddings[tname].weight[vid.item()]
                    for vid in feat_ids
                ]
            ).to(device)
            torch.testing.assert_close(actual_jt.values(), expected_embs)
        offset += num_values


def _test_pec_forward_stages(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[List[KeyedJaggedTensor]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    expected_forward_permutes_per_rank: List[List[List[int]]] | None = None,
    expected_backward_ol_permutes_per_rank: List[List[List[int] | None]] | None = None,
    expected_backward_nol_permutes_per_rank: List[List[List[int] | None]] | None = None,
    ref_ec: EmbeddingCollection | None = None,
    expected_ol_per_rank: Dict[int, List[List[Tuple[str, int]]]] | None = None,
    expected_nol_per_rank: Dict[int, List[List[Tuple[str, int]]]] | None = None,
    verify_merge: bool = False,
    local_size: Optional[int] = None,
) -> None:
    """Runs PEC forward pipeline using overlap_dist and verifies results.

    Pipeline: input_dist → overlap_dist → compute_and_output_dist_in_partition
    (OL + NOL) → optionally merge_partitioned_embeddings.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)
        sharded_pec = sharded_sparse_arch._pec_ec
        assert isinstance(sharded_pec, ShardedPECEmbeddingCollection)
        assert len(sharded_pec._overlap_handlers) > 0

        if ref_ec is not None:
            copy_state_dict(
                sharded_pec._embedding_collection.state_dict(),
                ref_ec.state_dict(),
            )

        batches = kjt_input_per_rank[rank]
        prev_ctx = None
        prev_features_list = None

        for batch_idx, kjt_input in enumerate(batches):
            original_values = kjt_input.values()
            original_lengths = kjt_input.lengths()
            kjt_input = kjt_input.to(ctx.device)

            # Stage 1: input_dist
            pec_ctx = sharded_pec.create_context()
            dist_input = sharded_pec.input_dist(pec_ctx, kjt_input).wait().wait()

            # Stage 2: overlap_dist (fused detect + split + mask dist + compute)
            results = sharded_pec.overlap_dist(
                ctx=pec_ctx,
                dist_input=dist_input,
                prev_ctx=prev_ctx,
                prev_dist_input=prev_features_list,
            )
            forward_ctx, backward_ctx = results[0].wait()

            if expected_forward_permutes_per_rank is not None:
                _verify_forward_permute(
                    forward_ctx,
                    expected_forward_permutes_per_rank[rank][batch_idx],
                )

            if expected_backward_ol_permutes_per_rank is not None:
                assert expected_backward_nol_permutes_per_rank is not None
                _verify_backward_permutes(
                    backward_ctx,
                    expected_backward_ol_permutes_per_rank[rank][batch_idx],
                    expected_backward_nol_permutes_per_rank[rank][batch_idx],
                )

            # Stage 3: compute_and_output_dist_in_partition
            assert forward_ctx is not None
            ol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
                pec_ctx,
                forward_ctx.ol_features,
                forward_ctx.splits,
                is_overlapped=True,
            )
            assert len(ol_awaitables) == 1

            nol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
                pec_ctx,
                forward_ctx.nol_features,
                forward_ctx.splits,
                is_overlapped=False,
            )
            assert len(nol_awaitables) == 1

            if ref_ec is not None and expected_ol_per_rank is not None:
                _verify_partition_embeddings(
                    ol_awaitables[0].wait(),
                    expected_ol_per_rank[rank][batch_idx],
                    ref_ec,
                    ctx.device,
                )
            if ref_ec is not None and expected_nol_per_rank is not None:
                _verify_partition_embeddings(
                    nol_awaitables[0].wait(),
                    expected_nol_per_rank[rank][batch_idx],
                    ref_ec,
                    ctx.device,
                )

            # Stage 4: merge_partitioned_embeddings
            if verify_merge:
                assert ref_ec is not None
                lazy_result = sharded_pec.merge_partitioned_embeddings(
                    pec_ctx,
                    ol_awaitables,
                    nol_awaitables,
                    [forward_ctx],
                )
                jt_dict = lazy_result.wait()
                _verify_merged_output(
                    jt_dict,
                    kjt_input,
                    original_values,
                    original_lengths,
                    tables,
                    ref_ec,
                    ctx.device,
                )

            prev_ctx = pec_ctx
            prev_features_list = dist_input


# Cross-shard data for uneven tables (table_0: 16 rows, table_1: 8 rows).
# With world_size=2 RW sharding:
#   table_0: block_size=8, shard 0 rows 0-7, shard 1 rows 8-15
#   table_1: block_size=4, shard 0 rows 0-3, shard 1 rows 4-7
# stride=2 (batch_size=2 per rank), 2 batches per rank.
CROSS_SHARD_KJT_INPUT_PER_RANK = [
    [
        # Rank 0 batch 0
        # f0: [0,8,2] → shard0:[0,2], shard1:[8]
        # f1: [1,5,3] → shard0:[1,3], shard1:[5]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 8, 2, 1, 5, 3]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 0 batch 1 (overlaps with batch 0: f0: 0, f1: 1)
        # f0: [0,9,3] → shard0:[0,3], shard1:[9]
        # f1: [1,6,2] → shard0:[1,2], shard1:[6]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 9, 3, 1, 6, 2]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
    [
        # Rank 1 batch 0
        # f0: [1,10,5] → shard0:[1,5], shard1:[10]
        # f1: [4,2,7] → shard0:[2], shard1:[4,7]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 10, 5, 4, 2, 7]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        # Rank 1 batch 1 (overlaps with batch 0: f0: 1, f1: 4)
        # f0: [1,11,7] → shard0:[1,7], shard1:[11]
        # f1: [4,3,6] → shard0:[3], shard1:[4,6]
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 11, 7, 4, 3, 6]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
]


@skip_if_asan_class
class ShardedPECEmbeddingCollectionTest(MultiProcessTestBase):
    """Multi-process tests for ShardedPECEmbeddingCollection."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharding_rw(self) -> None:
        WORLD_SIZE = 2

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=[
                batches[0] for batches in CROSS_SHARD_KJT_INPUT_PER_RANK
            ],
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_overlap_dist_forward_permute(self) -> None:
        """Verifies exact forward_permute values from overlap_dist."""
        WORLD_SIZE = 2

        # Batch 0: no overlap → forward_permute = upt
        # Batch 1: partial overlap → forward_permute derived from post-dist mask + upt
        expected_forward_permutes_per_rank = [
            [
                [0, 4, 1, 2, 5, 3],
                [0, 4, 3, 1, 5, 2],
            ],
            [
                [0, 3, 1, 4, 2, 5],
                [0, 4, 3, 2, 1, 5],
            ],
        ]

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            expected_forward_permutes_per_rank=expected_forward_permutes_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_overlap_dist_backward_permutes(self) -> None:
        """Verifies backward ol/nol permutes from overlap_dist."""
        WORLD_SIZE = 2

        # Batch 0: no prev batch → backward permutes are None
        # Batch 1: backward permutes computed from backward mask
        expected_backward_ol_permutes_per_rank = [
            [None, [0, 3, 5]],
            [None, [0, 4, 3]],
        ]
        expected_backward_nol_permutes_per_rank = [
            [None, [2, 1, 4]],
            [None, [2, 1, 5]],
        ]

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            expected_backward_ol_permutes_per_rank=expected_backward_ol_permutes_per_rank,
            expected_backward_nol_permutes_per_rank=expected_backward_nol_permutes_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_compute_and_output_dist_in_partition(self) -> None:
        """Tests OL/NOL embeddings match expected weight rows.

        Uses uneven cross-shard data. table_0: 16 rows (block_size=8),
        table_1: 8 rows (block_size=4). world_size=2.

        Expected values traced through:
        1. block_bucketize → which values go to which shard
        2. KJTAllToAll + recat [0,2,1,3] → feature-major on shard
        3. remap with per-table offsets (shard 0: t0 offset=0, t1 offset=8)
        4. overlap mask (batch 1 vs batch 0 remapped values)
        5. split → lookup → embedding AllToAll → rank receives
        6. output order: [from_shard0, from_shard1]

        Batch 0: all NOL (first batch, no overlap).
        Batch 1 shard 0 mask=[T,F,T,F,T,T,T], shard 1 mask=[T,F,T,T,F].
        """
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        T0 = "table_0"
        T1 = "table_1"
        expected_ol: Dict[int, List[List[Tuple[str, int]]]] = {
            0: [
                [],  # batch 0: no overlap
                [(T0, 0), (T1, 1), (T1, 2)],
            ],
            1: [
                [],
                [(T0, 1), (T1, 3), (T1, 4)],
            ],
        }
        expected_nol: Dict[int, List[List[Tuple[str, int]]]] = {
            0: [
                [(T0, 0), (T0, 2), (T1, 1), (T1, 3), (T0, 8), (T1, 5)],
                [(T0, 3), (T0, 9), (T1, 6)],
            ],
            1: [
                [(T0, 1), (T0, 5), (T1, 2), (T0, 10), (T1, 4), (T1, 7)],
                [(T0, 7), (T0, 11), (T1, 6)],
            ],
        }

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            expected_ol_per_rank=expected_ol,
            expected_nol_per_rank=expected_nol,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_merge_partitioned_embeddings(self) -> None:
        """Tests merged output matches expected embeddings from ref EC."""
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            verify_merge=True,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_single_shard_features(self) -> None:
        """Tests overlap_dist when one shard receives no features.

        All values in shard 1's range (table_0: 8-15, table_1: 4-7),
        so shard 0 gets nothing after input_dist. Verifies mask_dist,
        splits, permutes, and merge handle empty KJTs on the empty shard.
        """
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        # table_0 shard1: [8,15], table_1 shard1: [4,7]
        shard1_only_input = [
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([8, 9, 4, 5]),
                    lengths=torch.LongTensor([2, 2]),
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([8, 10, 5, 6]),
                    lengths=torch.LongTensor([2, 2]),
                ),
            ],
            [
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([12, 13, 6, 7]),
                    lengths=torch.LongTensor([2, 2]),
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    keys=["feature_0", "feature_1"],
                    values=torch.LongTensor([12, 14, 7, 4]),
                    lengths=torch.LongTensor([2, 2]),
                ),
            ],
        ]
        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=shard1_only_input,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
            ref_ec=ref_ec,
            verify_merge=True,
        )
