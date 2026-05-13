#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import copy
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.pec_collision_handlers import (
    CollisionPermutation,
    CollisionResult,
    CollisionSplits,
    split_features_by_values_mask,
)
from torchrec.distributed.pec_embedding import (
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
        num_embeddings=16,
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


@dataclass
class ExpectedCollisionResult:
    remapped: List[int]
    forward_mask: List[bool]
    backward_mask: List[bool] | None


@dataclass
class ExpectedSplits:
    input_splits: List[List[int]]
    output_splits: List[List[int]]


def _verify_forward_permute(
    forward_permute: torch.Tensor,
    expected: List[int],
) -> None:
    assert forward_permute.tolist() == expected


def _verify_backward_permute(
    permutation: "CollisionPermutation",
    expected_permute: List[int] | None,
    expected_num_ol: int,
) -> None:
    if expected_permute is None:
        assert permutation.backward_permute is None
    else:
        assert permutation.backward_permute is not None
        assert permutation.backward_permute.tolist() == expected_permute
    assert permutation.backward_num_ol == expected_num_ol


def _verify_collision_result(
    result: CollisionResult,
    expected: ExpectedCollisionResult,
) -> None:
    assert result.remapped_feature_values.tolist() == expected.remapped
    assert result.forward_overlap_mask.tolist() == expected.forward_mask
    if expected.backward_mask is None:
        assert result.backward_overlap_mask is None
    else:
        assert result.backward_overlap_mask is not None
        assert result.backward_overlap_mask.tolist() == expected.backward_mask


def _verify_collision_splits(
    all_splits: List[CollisionSplits],
    expected: ExpectedSplits,
) -> None:
    assert len(all_splits) == 1
    splits = all_splits[0]
    assert splits.input_splits == expected.input_splits
    assert splits.output_splits == expected.output_splits


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
    expected_collisions_per_rank: List[List[ExpectedCollisionResult]] | None = None,
    expected_splits_per_rank: List[List[ExpectedSplits]] | None = None,
    expected_forward_permutes_per_rank: List[List[List[int]]] | None = None,
    expected_backward_permutes_per_rank: List[List[List[int] | None]] | None = None,
    expected_backward_num_ol_per_rank: List[List[int]] | None = None,
    ref_ec: EmbeddingCollection | None = None,
    expected_ol_per_rank: Dict[int, List[List[Tuple[str, int]]]] | None = None,
    expected_nol_per_rank: Dict[int, List[List[Tuple[str, int]]]] | None = None,
    verify_merge: bool = False,
    local_size: Optional[int] = None,
) -> None:
    """Runs PEC forward pipeline stages and verifies against expected results.

    Always runs: input_dist → detect_collisions → split →
        collision_split_dist → permute_dist →
        compute_and_output_dist_in_partition (OL + NOL).
    Optionally runs: merge_partitioned_embeddings (when verify_merge=True).
    Verification for each stage is enabled by providing its expected-output parameter.
    Partition and merge verification require ref_ec for weight comparison.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)
        sharded_pec = sharded_sparse_arch._pec_ec
        assert isinstance(sharded_pec, ShardedPECEmbeddingCollection)
        assert len(sharded_pec._collision_handlers) > 0

        if ref_ec is not None:
            copy_state_dict(
                sharded_pec._embedding_collection.state_dict(),
                ref_ec.state_dict(),
            )

        batches = kjt_input_per_rank[rank]
        prev_remapped = None
        prev_ctx = None
        prev_features_list = None

        for batch_idx, kjt_input in enumerate(batches):
            original_values = kjt_input.values()
            original_lengths = kjt_input.lengths()
            kjt_input = kjt_input.to(ctx.device)

            # Stage 1: input_dist → detect_collisions
            pec_ctx = sharded_pec.create_context()
            pec_ctx.prev_remapped_feature_values = prev_remapped
            dist_input = sharded_pec.input_dist(pec_ctx, kjt_input).wait().wait()
            results = sharded_pec.detect_collisions(pec_ctx, dist_input)

            if expected_collisions_per_rank is not None:
                _verify_collision_result(
                    results[0],
                    expected_collisions_per_rank[rank][batch_idx],
                )

            # Stage 2: split features → collision_split_dist
            ol_features_list = []
            nol_features_list = []
            for result, features in zip(results, dist_input):
                ol, nol = split_features_by_values_mask(
                    features,
                    result.forward_overlap_mask,
                )
                ol_features_list.append(ol)
                nol_features_list.append(nol)

            split_awaitables = sharded_pec.collision_split_dist(
                pec_ctx,
                nol_features_list,
            )
            all_splits = [aw.wait() for aw in split_awaitables]

            if expected_splits_per_rank is not None:
                _verify_collision_splits(
                    all_splits,
                    expected_splits_per_rank[rank][batch_idx],
                )

            # Stage 3: permute_dist
            features_list = list(dist_input)
            permute_awaitables = sharded_pec.permute_dist(
                pec_ctx,
                features_list,
                results,
                prev_ctx=prev_ctx,
                prev_features_per_group=prev_features_list,
            )
            permutations = [aw.wait() for aw in permute_awaitables]

            if expected_forward_permutes_per_rank is not None:
                _verify_forward_permute(
                    permutations[0].forward_permute,
                    expected_forward_permutes_per_rank[rank][batch_idx],
                )

            if expected_backward_permutes_per_rank is not None:
                _verify_backward_permute(
                    permutations[0],
                    expected_backward_permutes_per_rank[rank][batch_idx],
                    (
                        expected_backward_num_ol_per_rank[rank][batch_idx]
                        if expected_backward_num_ol_per_rank is not None
                        else 0
                    ),
                )

            # Stage 4: compute_and_output_dist_in_partition
            ol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
                pec_ctx,
                ol_features_list[0],
                all_splits[0],
                is_overlapped=True,
            )
            assert len(ol_awaitables) == 1

            nol_awaitables = sharded_pec.compute_and_output_dist_in_partition(
                pec_ctx,
                nol_features_list[0],
                all_splits[0],
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

            # Stage 5: merge_partitioned_embeddings
            if verify_merge:
                assert ref_ec is not None
                lazy_result = sharded_pec.merge_partitioned_embeddings(
                    pec_ctx,
                    ol_awaitables,
                    nol_awaitables,
                    permutations,
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

            prev_remapped = [r.remapped_feature_values for r in results]
            prev_ctx = pec_ctx
            prev_features_list = features_list


# Cross-shard data: each rank has values in both shard 0 (0-7) and shard 1 (8-15)
CROSS_SHARD_KJT_INPUT_PER_RANK = [
    [
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 8, 2, 10, 4, 12]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 8, 3, 11, 4, 13]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
    [
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 9, 5, 11, 6, 13]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 9, 7, 14, 6, 15]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
]

TWO_BATCH_KJT_INPUT_PER_RANK = [
    [
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 1, 2, 3, 4, 5]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([1, 3, 5, 7, 4, 6]),
            lengths=torch.LongTensor([2, 1, 1, 2]),
        ),
    ],
    [
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([6, 7, 8, 9, 10, 11]),
            lengths=torch.LongTensor([1, 2, 2, 1]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([8, 10, 9, 11, 13, 15]),
            lengths=torch.LongTensor([1, 2, 2, 1]),
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

        kjt_input_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 3, 4, 5]),
                lengths=torch.LongTensor([2, 1, 1, 2]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([6, 7, 8, 9, 10, 11]),
                lengths=torch.LongTensor([1, 2, 2, 1]),
            ),
        ]

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_detect_collisions(self) -> None:
        """Verifies exact collision masks and remapped values."""
        WORLD_SIZE = 2

        # RW sharding with block_size=8: rank 0 owns IDs 0-7, rank 1 owns 8-15.
        # Remapped = local_value + table_offset (table_0 offset=0, table_1 offset=8).
        expected_collisions_per_rank = [
            [
                ExpectedCollisionResult(
                    remapped=[0, 1, 2, 6, 7, 11, 12, 13],
                    forward_mask=[False] * 8,
                    backward_mask=None,
                ),
                ExpectedCollisionResult(
                    remapped=[1, 3, 5, 15, 12, 14],
                    forward_mask=[True, False, False, False, True, False],
                    backward_mask=[
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                    ],
                ),
            ],
            [
                ExpectedCollisionResult(
                    remapped=[0, 9, 10, 11],
                    forward_mask=[False] * 4,
                    backward_mask=None,
                ),
                ExpectedCollisionResult(
                    remapped=[0, 2, 1, 11, 13, 15],
                    forward_mask=[True, False, False, True, False, False],
                    backward_mask=[True, False, False, True],
                ),
            ],
        ]

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=TWO_BATCH_KJT_INPUT_PER_RANK,
            expected_collisions_per_rank=expected_collisions_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_collision_split_dist(self) -> None:
        """Verifies exact collision split values per rank."""
        WORLD_SIZE = 2

        expected_splits_per_rank = [
            [
                ExpectedSplits(
                    input_splits=[[0, 0], [6, 2]],
                    output_splits=[[0, 0], [6, 0]],
                ),
                ExpectedSplits(
                    input_splits=[[2, 0], [4, 0]],
                    output_splits=[[2, 0], [4, 0]],
                ),
            ],
            [
                ExpectedSplits(
                    input_splits=[[0, 0], [0, 4]],
                    output_splits=[[0, 0], [2, 4]],
                ),
                ExpectedSplits(
                    input_splits=[[0, 2], [0, 4]],
                    output_splits=[[0, 2], [0, 4]],
                ),
            ],
        ]

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=TWO_BATCH_KJT_INPUT_PER_RANK,
            expected_splits_per_rank=expected_splits_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_permute_dist(self) -> None:
        """Verifies exact forward_permute values with cross-shard partial overlap."""
        WORLD_SIZE = 2

        # Batch 0: no overlap → forward_permute = upt = [0,3,1,4,2,5]
        # Batch 1: partial overlap → forward_permute derived from received mask + upt
        expected_forward_permutes_per_rank = [
            [
                [0, 3, 1, 4, 2, 5],
                [0, 2, 5, 3, 1, 4],
            ],
            [
                [0, 3, 1, 4, 2, 5],
                [0, 2, 3, 4, 1, 5],
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
    def test_backward_permute_dist(self) -> None:
        """Verifies backward_permute and backward_num_ol with cross-shard overlap.

        Batch 0: no backward (first batch) → backward_permute=None, num_ol=0.
        Batch 1: backward mask from batch 0's values overlapping with batch 1.
          Rank 0 received bwd mask: [T,F,T, T,F,F] → 3 ol, 3 nol
          Rank 1 received bwd mask: [T,F,T, T,T,T] → 5 ol, 1 nol
          backward_permute = _compute_permute_from_mask(bwd_mask, batch0_upt)
        """
        WORLD_SIZE = 2

        # backward_permute = bucket_to_merged[upt]
        # upt for batch 0 = [0,3,1,4,2,5] (no overlap, same as forward)
        #
        # Rank 0: mask=[T,F,T,T,F,F], bucket_to_merged=[0,3,1,2,4,5]
        #   backward_permute = [0,2,3,4,1,5], num_ol=3
        # Rank 1: mask=[T,F,T,T,T,T], bucket_to_merged=[0,5,1,2,3,4]
        #   backward_permute = [0,2,5,3,1,4], num_ol=5
        expected_backward_permutes_per_rank = [
            [None, [0, 2, 3, 4, 1, 5]],
            [None, [0, 2, 5, 3, 1, 4]],
        ]
        expected_backward_num_ol_per_rank = [
            [0, 3],
            [0, 5],
        ]

        self._run_multi_process_test(
            callable=_test_pec_forward_stages,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=CROSS_SHARD_KJT_INPUT_PER_RANK,
            expected_backward_permutes_per_rank=expected_backward_permutes_per_rank,
            expected_backward_num_ol_per_rank=expected_backward_num_ol_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_compute_and_output_dist_in_partition(self) -> None:
        """Tests OL/NOL embeddings match expected weight rows.

        Uses cross-shard partial overlap data. num_embeddings=16, world_size=2,
        block_size=8. Shard 0 stores original rows 0-7, shard 1 stores 8-15.

        Expected values manually traced through:
        1. block_bucketize → which values go to which shard
        2. KJTAllToAll + recat [0,2,1,3] → feature-major on shard
        3. collision mask (batch 1): shard0=[T,F,T,F,T,T], shard1=[T,T,T,T,F,F]
        4. split → lookup on each shard's TBE
        5. forward_recat to rank-major → embedding AllToAll → trainer
        6. output = [from_shard0, from_shard1]
        """
        WORLD_SIZE = 2

        ref_ec = EmbeddingCollection(
            tables=EMBEDDING_TABLES,
            device=torch.device("cpu"),
        )

        # Expected OL/NOL rows as (table_name, original_id).
        # Output order: [from_shard0, from_shard1], rank-major within each.
        #
        # --- Batch 0: no overlap, all NOL ---
        # Shard 0 recat rank-major:
        #   trainer0 ← [t0[0],t0[2],t1[4]], trainer1 ← [t0[1],t0[5],t1[6]]
        # Shard 1 recat rank-major:
        #   trainer0 ← [t0[8],t1[10],t1[12]], trainer1 ← [t0[9],t1[11],t1[13]]
        #
        # --- Batch 1: shard0 mask=[T,F,T,F,T,T], shard1=[T,T,T,T,F,F] ---
        # Shard 0 OL: trainer0 ← [t0[0],t1[4]], trainer1 ← [t0[1],t1[6]]
        # Shard 0 NOL: trainer0 ← [t0[3]], trainer1 ← [t0[7]]
        # Shard 1 OL: trainer0 ← [t0[8],t1[11],t1[13]], trainer1 ← [t0[9]]
        # Shard 1 NOL: trainer0 ← [], trainer1 ← [t1[14],t1[15]]
        T0 = "table_0"
        T1 = "table_1"
        expected_ol: Dict[int, List[List[Tuple[str, int]]]] = {
            0: [
                [],  # batch 0: no overlap
                [(T0, 0), (T1, 4), (T0, 8), (T1, 11), (T1, 13)],
            ],
            1: [
                [],
                [(T0, 1), (T1, 6), (T0, 9)],
            ],
        }
        expected_nol: Dict[int, List[List[Tuple[str, int]]]] = {
            0: [
                [(T0, 0), (T0, 2), (T1, 4), (T0, 8), (T1, 10), (T1, 12)],
                [(T0, 3)],
            ],
            1: [
                [(T0, 1), (T0, 5), (T1, 6), (T0, 9), (T1, 11), (T1, 13)],
                [(T0, 7), (T1, 14), (T1, 15)],
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
