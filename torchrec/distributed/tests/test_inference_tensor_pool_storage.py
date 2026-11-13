#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast

import torch
from torchrec.distributed.tensor_pool import (
    LocalShardPool,
    ShardedInferenceTensorPool,
    TensorPoolSharder,
)
from torchrec.distributed.types import (
    ObjectPoolShardingPlan,
    ObjectPoolShardingType,
    ShardingEnv,
)
from torchrec.modules.tensor_pool import TensorPool


def _pool(values: torch.Tensor, share_storage: bool) -> TensorPool:
    return TensorPool(
        pool_size=values.size(0),
        dim=values.size(1),
        dtype=values.dtype,
        device=values.device,
        loaded_values=values,
        share_storage_with_shards=share_storage,
    )


def _shards(pool: TensorPool, world_size: int) -> list[torch.Tensor]:
    sharded = TensorPoolSharder().shard(
        module=pool,
        plan=ObjectPoolShardingPlan(
            sharding_type=ObjectPoolShardingType.ROW_WISE, inference=True
        ),
        env=ShardingEnv.from_local(world_size, 0),
        device=torch.device("cpu"),
    )
    assert isinstance(sharded, ShardedInferenceTensorPool)
    return [
        cast(LocalShardPool, shard_pool)._shard
        for shard_pool in sharded._local_shard_pools
    ]


class InferenceTensorPoolStorageTest(unittest.TestCase):
    def test_shards_are_copies_by_default(self) -> None:
        values = torch.arange(12, dtype=torch.int64).view(6, 2)
        (shard,) = _shards(_pool(values, share_storage=False), world_size=1)
        self.assertNotEqual(shard.data_ptr(), values.data_ptr())
        torch.testing.assert_close(shard, values)
        expected = values.clone()
        values.zero_()
        torch.testing.assert_close(shard, expected)

    def test_a_pool_without_the_attribute_shards_by_copy(self) -> None:
        values = torch.arange(12, dtype=torch.int64).view(6, 2)
        pool = _pool(values, share_storage=True)
        del pool.share_storage_with_shards
        (shard,) = _shards(pool, world_size=1)
        self.assertNotEqual(shard.data_ptr(), values.data_ptr())
        torch.testing.assert_close(shard, values)

    def test_an_opted_in_single_shard_is_the_pool(self) -> None:
        values = torch.arange(12, dtype=torch.int64).view(6, 2)
        pool = _pool(values, share_storage=True)
        sharded = TensorPoolSharder().shard(
            module=pool,
            plan=ObjectPoolShardingPlan(
                sharding_type=ObjectPoolShardingType.ROW_WISE, inference=True
            ),
            env=ShardingEnv.from_local(1, 0),
            device=torch.device("cpu"),
        )
        assert isinstance(sharded, ShardedInferenceTensorPool)
        shard_pool = cast(LocalShardPool, sharded._local_shard_pools[0])
        self.assertEqual(shard_pool._shard.data_ptr(), values.data_ptr())
        self.assertEqual(shard_pool._shard.shape, values.shape)
        torch.testing.assert_close(
            shard_pool(torch.tensor([4, 1])), values[torch.tensor([4, 1])]
        )

    def test_opted_in_pools_still_copy_into_several_shards(self) -> None:
        values = torch.arange(12, dtype=torch.int64).view(6, 2)
        shards = _shards(_pool(values, share_storage=True), world_size=2)
        torch.testing.assert_close(torch.cat(shards), values)
        for shard in shards:
            self.assertNotEqual(
                shard.untyped_storage().data_ptr(),
                values.untyped_storage().data_ptr(),
            )

    def test_a_meta_pool_gets_a_real_shard(self) -> None:
        for share_storage in (False, True):
            with self.subTest(share_storage=share_storage):
                pool = TensorPool(
                    pool_size=6,
                    dim=2,
                    dtype=torch.int64,
                    share_storage_with_shards=share_storage,
                )
                (shard,) = _shards(pool, world_size=1)
                self.assertEqual(shard.device, torch.device("cpu"))
                self.assertEqual(shard.shape, (6, 2))
                self.assertEqual(shard.dtype, torch.int64)
