#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from hypothesis import given, settings, strategies as st
from torchrec.distributed.tensor_pool import TensorPoolSharder
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ObjectPoolShardingPlan,
    ObjectPoolShardingType,
    ShardedTensor,
    ShardingEnv,
)
from torchrec.modules.tensor_pool import TensorPool


class TestShardedTensorPool(MultiProcessTestBase):
    @staticmethod
    def _test_sharded_tensor_pool(
        rank: int, world_size: int, enable_uvm: bool = False
    ) -> None:

        pool_size = 5
        dim = 4
        backend = "nccl"
        dtype = torch.float32
        sharding_plan = ObjectPoolShardingPlan(
            sharding_type=ObjectPoolShardingType.ROW_WISE
        )
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            torch.use_deterministic_algorithms(False)
            tensor_pool = TensorPool(
                pool_size=pool_size,
                dim=dim,
                dtype=dtype,
                enable_uvm=enable_uvm,
            )

            sharded_tensor_pool = TensorPoolSharder().shard(
                module=tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  `Optional[ProcessGroup]`.
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            if ctx.rank == 0:
                ids = [4, 1]
                values = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]]

            else:
                ids = [3, 0]
                values = [[0.11, 0.21, 0.31, 1.0], [0.41, 0.51, 0.61, 2.0]]

            ids = torch.tensor(ids, dtype=torch.int, device=ctx.device)
            values = torch.tensor(values, dtype=torch.float, device=ctx.device)

            sharded_tensor_pool.update(
                ids=ids,
                values=values,
            )

            lookup_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=ctx.device)

            values = sharded_tensor_pool.lookup(ids=lookup_ids).wait()
            torch.testing.assert_close(
                values.cpu(),
                torch.tensor(
                    [
                        [0.41, 0.51, 0.61, 2.0],
                        [0.4, 0.5, 0.6, 0.7],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.11, 0.21, 0.31, 1.0],
                    ],
                    device=torch.device("cpu"),
                ),
            )

            state_dict = sharded_tensor_pool.state_dict()
            ut = unittest.TestCase()
            ut.assertIn("_pool", state_dict)
            sharded_pool_state = state_dict["_pool"]
            ut.assertIsInstance(sharded_pool_state, ShardedTensor)
            pool_state = (
                torch.empty(size=sharded_pool_state.size(), device=ctx.device)
                if ctx.rank == 0
                else None
            )
            sharded_pool_state.gather(out=pool_state)
            if ctx.rank == 0:
                torch.testing.assert_close(
                    pool_state,
                    torch.tensor(
                        [
                            [0.41, 0.51, 0.61, 2.0],
                            [0.4, 0.5, 0.6, 0.7],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.11, 0.21, 0.31, 1.0],
                            [0.1000, 0.2000, 0.3000, 0.4000],
                        ],
                        device=ctx.device,
                    ),
                )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        enable_uvm=st.booleans(),
    )
    @settings(deadline=None)
    def test_sharded_tensor_pool(self, enable_uvm: bool) -> None:
        world_size = 2
        self._run_multi_process_test(
            callable=self._test_sharded_tensor_pool,
            world_size=world_size,
            enable_uvm=enable_uvm,
        )

    @staticmethod
    def _test_sharded_tensor_pool_conflict_update(
        rank: int,
        world_size: int,
    ) -> None:

        pool_size = 5
        dim = 3
        backend = "nccl"
        dtype = torch.float32
        sharding_plan = ObjectPoolShardingPlan(
            sharding_type=ObjectPoolShardingType.ROW_WISE
        )
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            torch.use_deterministic_algorithms(False)
            tensor_pool = TensorPool(
                pool_size=pool_size,
                dim=dim,
                dtype=dtype,
            )

            sharded_tensor_pool = TensorPoolSharder().shard(
                module=tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  `Optional[ProcessGroup]`.
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            if ctx.rank == 0:
                ids = [4, 1]
                values = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]

            else:
                ids = [3, 1]
                values = [0.11, 0.21, 0.31], [0.41, 0.51, 0.61]

            ids = torch.tensor(ids, dtype=torch.int, device=ctx.device)
            values = torch.tensor(values, dtype=torch.float, device=ctx.device)

            sharded_tensor_pool.update(
                ids=ids,
                values=values,
            )

            lookup_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=ctx.device)

            values = sharded_tensor_pool(ids=lookup_ids).wait()
            torch.testing.assert_close(
                values.cpu(),
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.41, 0.51, 0.61],
                        [0.0, 0.0, 0.0],
                        [0.11, 0.21, 0.31],
                    ],
                    device=torch.device("cpu"),
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharded_tensor_pool_conflict_update(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(
            callable=self._test_sharded_tensor_pool_conflict_update,
            world_size=world_size,
        )
