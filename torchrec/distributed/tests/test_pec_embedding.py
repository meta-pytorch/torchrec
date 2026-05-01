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
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)
        sparse_arch = PECSparseArch(tables, torch.device("meta"))

        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch._pec_ec,
            per_param_sharding={table.name: row_wise() for table in tables},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            sharder=sharder,
        )

        sharded_sparse_arch: PECSparseArch = shard_modules(  # pyre-ignore[9]
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"_pec_ec": module_sharding_plan}),
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=[sharder],
            device=ctx.device,
        )

        assert isinstance(sharded_sparse_arch._pec_ec, ShardedPECEmbeddingCollection)
        assert isinstance(
            sharded_sparse_arch._pec_ec._embedding_collection,
            ShardedEmbeddingCollection,
        )

        # Forward + backward, two iterations to exercise stateful components
        for _ in range(2):
            loss, _ = sharded_sparse_arch(kjt_input)
            loss.backward()


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
