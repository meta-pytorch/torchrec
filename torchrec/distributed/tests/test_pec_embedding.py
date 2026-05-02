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


def _test_detect_collisions(
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[List[KeyedJaggedTensor]],
    expected_per_rank: List[List[ExpectedCollisionResult]],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Shard the model and get the ShardedPECEmbeddingCollection
        sharded_sparse_arch = _shard_pec(tables, ctx, sharder, local_size)
        sharded_pec = sharded_sparse_arch._pec_ec
        assert isinstance(sharded_pec, ShardedPECEmbeddingCollection)
        assert len(sharded_pec._collision_handlers) > 0

        batches = kjt_input_per_rank[rank]
        expected_batches = expected_per_rank[rank]
        prev_remapped = None

        for batch_idx, kjt_input in enumerate(batches):
            kjt_input = kjt_input.to(ctx.device)

            # Simulate the pipeline: set prev batch's remapped values on ctx,
            # run input_dist to redistribute features, then detect collisions.
            pec_ctx = sharded_pec.create_context()
            pec_ctx.prev_remapped_feature_values = prev_remapped
            dist_input = sharded_pec.input_dist(pec_ctx, kjt_input).wait().wait()
            results = sharded_pec.detect_collisions(pec_ctx, dist_input)

            # Verify remapped values and overlap masks against expected results.
            assert len(results) >= 1
            result = results[0]
            expected = expected_batches[batch_idx]

            assert result.remapped_feature_values.tolist() == expected.remapped
            assert result.forward_overlap_mask.tolist() == expected.forward_mask
            if expected.backward_mask is None:
                assert result.backward_overlap_mask is None
            else:
                assert result.backward_overlap_mask is not None
                assert result.backward_overlap_mask.tolist() == expected.backward_mask

            # Save remapped values for next batch's collision detection
            prev_remapped = [r.remapped_feature_values for r in results]


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
    def test_detect_collisions_two_batches(self) -> None:
        """Two batches with partial overlap — verifies exact masks and remapped values."""
        WORLD_SIZE = 2

        kjt_input_per_rank = [
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

        # RW sharding with block_size=8: rank 0 owns IDs 0-7, rank 1 owns 8-15.
        # After input_dist, values are remapped to local indices.
        # Remapped = local_value + table_offset (table_0 offset=0, table_1 offset=8).
        expected_per_rank = [
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
            callable=_test_detect_collisions,
            world_size=WORLD_SIZE,
            tables=EMBEDDING_TABLES,
            kjt_input_per_rank=kjt_input_per_rank,
            expected_per_rank=expected_per_rank,
            sharder=PECEmbeddingCollectionSharder(),
            backend="nccl",
        )
