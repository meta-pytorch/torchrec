#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, List

import torch
from hypothesis import given, Phase, settings, strategies as st
from torchrec.distributed.keyed_jagged_tensor_pool import (
    KeyedJaggedTensorPoolSharder,
    ShardedInferenceKeyedJaggedTensorPool,
    ShardedKeyedJaggedTensorPool,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ObjectPoolShardingPlan,
    ObjectPoolShardingType,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.modules.keyed_jagged_tensor_pool import KeyedJaggedTensorPool
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestShardedKeyedJaggedTensorPool(MultiProcessTestBase):
    @staticmethod
    def _test_sharded_keyed_jagged_tensor_pool(
        rank: int,
        world_size: int,
        backend: str,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_plan: ObjectPoolShardingPlan,
        input_per_rank: List[torch.Tensor],
        enable_uvm: bool = False,
    ) -> None:
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            input_per_rank = [id.to(ctx.device) for id in input_per_rank]
            keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
                pool_size=pool_size,
                feature_max_lengths=feature_max_lengths,
                values_dtype=values_dtype,
                is_weighted=is_weighted,
                device=torch.device("meta"),
                enable_uvm=enable_uvm,
            )

            sharded_keyed_jagged_tensor_pool: (
                ShardedKeyedJaggedTensorPool
                # pyrefly: ignore[bad-assignment]
            ) = KeyedJaggedTensorPoolSharder().shard(
                keyed_jagged_tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  got `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            # rank 0
            #         0      1
            # "f1"   [1]     [3,3]
            # "f2"   [11]    [13,13,13]

            # rank 1
            #         0       1
            # "f1"   [2,2]    [4]
            # "f2"   [12,12]  [14,14,14,14]
            values = KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.tensor(
                    (
                        [1, 3, 3, 11, 13, 13, 13]
                        if ctx.rank == 0
                        else [2, 2, 4, 12, 12, 14, 14, 14, 14]
                    ),
                    dtype=values_dtype,
                    device=ctx.device,
                ),
                lengths=torch.tensor(
                    [1, 2, 1, 3] if ctx.rank == 0 else [2, 1, 2, 4],
                    dtype=torch.int,
                    device=ctx.device,
                ),
            )

            sharded_keyed_jagged_tensor_pool.update(
                ids=torch.tensor(
                    [2, 0] if ctx.rank == 0 else [1, 3],
                    dtype=torch.int,
                    device=ctx.device,
                ),
                values=values,
            )

            # init global state is
            # 4         8
            # f1       f2
            # [3,3] .  [13,13,13]
            # [2,2] .  [12,12]
            # [1] .    [11]
            # [4]      [14,14,14,14]

            kjt = sharded_keyed_jagged_tensor_pool.lookup(input_per_rank[ctx.rank])

            # expected values
            # rank 0: KeyedJaggedTensor({
            #     "f1": [[1], [3, 3]],
            #     "f2": [[11], [13, 13, 13]]
            # })

            # rank 1: KeyedJaggedTensor({
            #     "f1": [[2, 2], [4], [3, 3], [1]],
            #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
            # })

            torch.testing.assert_close(
                kjt.values().cpu(),
                torch.tensor(
                    (
                        [1, 3, 3, 11, 13, 13, 13]
                        if ctx.rank == 0
                        else [2, 2, 4, 3, 3, 1, 12, 12, 14, 14, 14, 14, 13, 13, 13, 11]
                    ),
                    dtype=values_dtype,
                    device=torch.device("cpu"),
                ),
            )

            torch.testing.assert_close(
                kjt.lengths().cpu(),
                torch.tensor(
                    [1, 2, 1, 3] if ctx.rank == 0 else [2, 1, 2, 1, 2, 4, 3, 1],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        enable_uvm=st.booleans(),
        values_dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(
        max_examples=4,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharded_keyed_jagged_tensor_pool_rw(
        self, enable_uvm: bool, values_dtype: torch.dtype
    ) -> None:
        input_per_rank = [
            torch.tensor([2, 0], dtype=torch.int),
            torch.tensor([1, 3, 0, 2], dtype=torch.int),
        ]

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}

        self._run_multi_process_test(
            callable=self._test_sharded_keyed_jagged_tensor_pool,
            world_size=2,
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=values_dtype,
            is_weighted=False,
            input_per_rank=input_per_rank,
            sharding_plan=ObjectPoolShardingPlan(
                sharding_type=ObjectPoolShardingType.ROW_WISE
            ),
            backend="nccl",
            enable_uvm=enable_uvm,
        )

    @staticmethod
    def _test_input_permute(
        rank: int,
        world_size: int,
        backend: str,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_plan: ObjectPoolShardingPlan,
        input_per_rank: List[torch.Tensor],
    ) -> None:
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            input_per_rank = [id.to(ctx.device) for id in input_per_rank]
            keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
                pool_size=pool_size,
                feature_max_lengths=feature_max_lengths,
                values_dtype=values_dtype,
                is_weighted=is_weighted,
                device=torch.device("meta"),
            )

            sharded_keyed_jagged_tensor_pool: (
                ShardedKeyedJaggedTensorPool
                # pyrefly: ignore[bad-assignment]
            ) = KeyedJaggedTensorPoolSharder().shard(
                keyed_jagged_tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  got `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            sharded_keyed_jagged_tensor_pool.update(
                ids=torch.tensor(
                    [2, 0] if ctx.rank == 0 else [1, 3],
                    dtype=torch.int,
                    device=ctx.device,
                ),
                values=KeyedJaggedTensor.from_lengths_sync(
                    keys=["f3", "f2", "f1"],
                    values=torch.tensor(
                        (
                            [21, 11, 13, 13, 13, 1, 3, 3]
                            if ctx.rank == 0
                            else [22, 22, 24, 12, 12, 14, 14, 14, 14, 2, 2, 4]
                        ),
                        dtype=values_dtype,
                        device=ctx.device,
                    ),
                    lengths=torch.tensor(
                        [1, 0, 1, 3, 1, 2] if ctx.rank == 0 else [2, 1, 2, 4, 2, 1],
                        dtype=torch.int,
                        device=ctx.device,
                    ),
                ),
            )

            # init global state is
            # 4         8
            # f1       f2
            # [3,3] .  [13,13,13]
            # [2,2] .  [12,12]
            # [1] .    [11]
            # [4]      [14,14,14,14]

            kjt = sharded_keyed_jagged_tensor_pool(input_per_rank[ctx.rank])

            # expected values
            # rank 0: KeyedJaggedTensor({
            #     "f1": [[1], [3, 3]],
            #     "f2": [[11], [13, 13, 13]]
            # })

            # rank 1: KeyedJaggedTensor({
            #     "f1": [[2, 2], [4], [3, 3], [1]],
            #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
            # })

            torch.testing.assert_close(
                kjt.values().cpu(),
                torch.tensor(
                    (
                        [1, 3, 3, 11, 13, 13, 13]
                        if ctx.rank == 0
                        else [2, 2, 4, 3, 3, 1, 12, 12, 14, 14, 14, 14, 13, 13, 13, 11]
                    ),
                    dtype=values_dtype,
                    device=torch.device("cpu"),
                ),
            )

            torch.testing.assert_close(
                kjt.lengths().cpu(),
                torch.tensor(
                    [1, 2, 1, 3] if ctx.rank == 0 else [2, 1, 2, 1, 2, 4, 3, 1],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_input_permute(
        self,
    ) -> None:
        input_per_rank = [
            torch.tensor([2, 0], dtype=torch.int),
            torch.tensor([1, 3, 0, 2], dtype=torch.int),
        ]

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}

        self._run_multi_process_test(
            callable=self._test_input_permute,
            world_size=2,
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=torch.int64,
            is_weighted=False,
            input_per_rank=input_per_rank,
            sharding_plan=ObjectPoolShardingPlan(
                sharding_type=ObjectPoolShardingType.ROW_WISE
            ),
            backend="nccl",
        )

    @staticmethod
    def _test_sharded_KJT_pool_input_conflict(
        rank: int,
        world_size: int,
        backend: str,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_plan: ObjectPoolShardingPlan,
        input_per_rank: List[torch.Tensor],
    ) -> None:
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            input_per_rank = [id.to(ctx.device) for id in input_per_rank]
            keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
                pool_size=pool_size,
                feature_max_lengths=feature_max_lengths,
                values_dtype=values_dtype,
                is_weighted=is_weighted,
                device=torch.device("meta"),
            )

            sharded_keyed_jagged_tensor_pool: (
                ShardedKeyedJaggedTensorPool
                # pyrefly: ignore[bad-assignment]
            ) = KeyedJaggedTensorPoolSharder().shard(
                keyed_jagged_tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  got `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            # rank 0 input:
            # ids   f1      f2
            # 2     1       11
            # 1     3, 3    13, 13, 13

            # rank 1 input:
            # ids   f1      f2
            # 1     2, 2    12, 12
            # 3     4       14, 14, 14, 14

            sharded_keyed_jagged_tensor_pool.update(
                ids=torch.tensor(
                    [2, 1] if ctx.rank == 0 else [1, 3],
                    dtype=torch.int,
                    device=ctx.device,
                ),
                values=KeyedJaggedTensor.from_lengths_sync(
                    keys=["f1", "f2"],
                    values=torch.tensor(
                        (
                            [1, 3, 3, 11, 13, 13, 13]
                            if ctx.rank == 0
                            else [2, 2, 4, 12, 12, 14, 14, 14, 14]
                        ),
                        dtype=values_dtype,
                        device=ctx.device,
                    ),
                    lengths=torch.tensor(
                        [1, 2, 1, 3] if ctx.rank == 0 else [2, 1, 2, 4],
                        dtype=torch.int,
                        device=ctx.device,
                    ),
                ),
            )

            kjt = sharded_keyed_jagged_tensor_pool(input_per_rank[ctx.rank])
            # expected values
            # rank 0: KeyedJaggedTensor({
            #     "f1": [[1], [3, 3]],
            #     "f2": [[11], [13, 13, 13]]
            # })

            # rank 1: KeyedJaggedTensor({
            #     "f1": [[2, 2], [4], [3, 3], [1]],
            #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
            # })

            torch.testing.assert_close(
                kjt.values().cpu(),
                torch.tensor(
                    (
                        [1, 11]
                        if ctx.rank == 0
                        else [2, 2, 4, 1, 12, 12, 14, 14, 14, 14, 11]
                    ),
                    dtype=values_dtype,
                    device=torch.device("cpu"),
                ),
            )

            torch.testing.assert_close(
                kjt.lengths().cpu(),
                torch.tensor(
                    [1, 0, 1, 0] if ctx.rank == 0 else [2, 1, 0, 1, 2, 4, 0, 1],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharded_KJT_pool_input_conflict(
        self,
    ) -> None:
        input_per_rank = [
            torch.tensor([2, 0], dtype=torch.int),
            torch.tensor([1, 3, 0, 2], dtype=torch.int),
        ]

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}

        self._run_multi_process_test(
            callable=self._test_sharded_KJT_pool_input_conflict,
            world_size=2,
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=torch.int64,
            is_weighted=False,
            input_per_rank=input_per_rank,
            sharding_plan=ObjectPoolShardingPlan(
                sharding_type=ObjectPoolShardingType.ROW_WISE
            ),
            backend="nccl",
        )

    @staticmethod
    def _test_sharded_KJT_pool_input_empty(
        rank: int,
        world_size: int,
        backend: str,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_plan: ObjectPoolShardingPlan,
        input_per_rank: List[torch.Tensor],
    ) -> None:
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            input_per_rank = [id.to(ctx.device) for id in input_per_rank]
            keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
                pool_size=pool_size,
                feature_max_lengths=feature_max_lengths,
                values_dtype=values_dtype,
                is_weighted=is_weighted,
                device=torch.device("meta"),
            )

            sharded_keyed_jagged_tensor_pool: (
                ShardedKeyedJaggedTensorPool
                # pyrefly: ignore[bad-assignment]
            ) = KeyedJaggedTensorPoolSharder().shard(
                keyed_jagged_tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  got `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            # rank 0 input:
            # ids   f1      f2
            # 2     1       11
            # 1     3, 3    13, 13, 13

            # rank 1 input:
            # ids   f1      f2
            # 1     2, 2    12, 12
            # 3     4       14, 14, 14, 14

            sharded_keyed_jagged_tensor_pool.update(
                ids=torch.tensor(
                    [2, 1] if ctx.rank == 0 else [1, 3],
                    dtype=torch.int,
                    device=ctx.device,
                ),
                values=KeyedJaggedTensor.from_lengths_sync(
                    keys=["f1", "f2"],
                    values=torch.tensor(
                        (
                            [1, 3, 3, 11, 13, 13, 13]
                            if ctx.rank == 0
                            else [2, 2, 4, 12, 12, 14, 14, 14, 14]
                        ),
                        dtype=values_dtype,
                        device=ctx.device,
                    ),
                    lengths=torch.tensor(
                        [1, 2, 1, 3] if ctx.rank == 0 else [2, 1, 2, 4],
                        dtype=torch.int,
                        device=ctx.device,
                    ),
                ),
            )

            kjt = sharded_keyed_jagged_tensor_pool(input_per_rank[ctx.rank])
            # expected values
            # rank 0: KeyedJaggedTensor({
            #     "f1": [[1], [3, 3]],
            #     "f2": [[11], [13, 13, 13]]
            # })

            # rank 1: KeyedJaggedTensor({
            #     "f1": [[2, 2], [4], [3, 3], [1]],
            #     "f2": [[12, 12], [14, 14, 14, 14], [13, 13, 13], [11]]
            # })

            torch.testing.assert_close(
                kjt.values().cpu(),
                torch.tensor(
                    [1, 11] if ctx.rank == 0 else [],
                    dtype=values_dtype,
                    device=torch.device("cpu"),
                ),
            )

            torch.testing.assert_close(
                kjt.lengths().cpu(),
                torch.tensor(
                    [1, 0, 1, 0] if ctx.rank == 0 else [],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharded_KJT_pool_input_empty(self) -> None:
        input_per_rank = [
            torch.tensor([2, 0], dtype=torch.int),
            torch.tensor([], dtype=torch.int),
        ]
        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        self._run_multi_process_test(
            callable=self._test_sharded_KJT_pool_input_empty,
            world_size=2,
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            values_dtype=torch.int64,
            is_weighted=False,
            input_per_rank=input_per_rank,
            sharding_plan=ObjectPoolShardingPlan(
                sharding_type=ObjectPoolShardingType.ROW_WISE
            ),
            backend="nccl",
        )

    @staticmethod
    def _test_sharded_keyed_jagged_tensor_pool_replicated_rw(
        rank: int,
        world_size: int,
        local_world_size: int,
        backend: str,
        pool_size: int,
        feature_max_lengths: Dict[str, int],
        values_dtype: torch.dtype,
        is_weighted: bool,
        sharding_plan: ObjectPoolShardingPlan,
    ) -> None:
        with MultiProcessContext(
            rank, world_size, backend, local_size=local_world_size
        ) as ctx:
            keyed_jagged_tensor_pool = KeyedJaggedTensorPool(
                pool_size=pool_size,
                feature_max_lengths=feature_max_lengths,
                values_dtype=values_dtype,
                is_weighted=is_weighted,
                device=torch.device("meta"),
            )

            sharded_keyed_jagged_tensor_pool: (
                ShardedKeyedJaggedTensorPool
                # pyrefly: ignore[bad-assignment]
            ) = KeyedJaggedTensorPoolSharder().shard(
                keyed_jagged_tensor_pool,
                plan=sharding_plan,
                device=ctx.device,
                #  got `Optional[ProcessGroup]`.
                # pyrefly: ignore[bad-argument-type]
                env=ShardingEnv.from_process_group(ctx.pg),
            )

            # init global state is
            # 4         8
            # f1       f2
            # [3,3] .  [13,13,13]
            # [2,2] .  [12,12]
            # [1] .    [11]
            # [4]      [14,14,14,14]

            ids = [[1], [0], [2], [3]]

            values_and_lengths = [
                ([2, 2, 12, 12], [2, 2]),
                ([3, 3, 13, 13, 13], [2, 3]),
                ([1, 11], [1, 1]),
                ([4, 14, 14, 14, 14], [1, 4]),
            ]

            sharded_keyed_jagged_tensor_pool.update(
                ids=torch.tensor(
                    ids[ctx.rank],
                    dtype=torch.int,
                    device=ctx.device,
                ),
                values=KeyedJaggedTensor.from_lengths_sync(
                    keys=["f1", "f2"],
                    values=torch.tensor(
                        values_and_lengths[ctx.rank][0],
                        dtype=values_dtype,
                        device=ctx.device,
                    ),
                    lengths=torch.tensor(
                        values_and_lengths[ctx.rank][1],
                        dtype=torch.int,
                        device=ctx.device,
                    ),
                ),
            )

            lookup_per_rank = [[0, 1, 2, 3], [0, 2], [3, 1], [0]]

            kjt = sharded_keyed_jagged_tensor_pool.lookup(
                torch.tensor(
                    lookup_per_rank[ctx.rank], device=ctx.device, dtype=torch.int32
                )
            ).wait()

            # expected values
            # rank 0:
            # kjt KeyedJaggedTensor({
            #   "f1": [[3, 3], [2, 2], [1], [4]],
            #   "f2": [[13, 13, 13], [12, 12], [11], [14, 14, 14, 14]]
            # })
            # rank 1:
            # kjt KeyedJaggedTensor({
            #   "f1": [[3, 3], [1]],
            #   "f2": [[13, 13, 13], [11]]
            # })
            # rank 2:
            # kjt KeyedJaggedTensor({
            #   "f1": [[4], [2, 2]],
            #   "f2": [[14, 14, 14, 14], [12, 12]]
            # })
            # rank 3:
            # kjt KeyedJaggedTensor({
            #   "f1": [[3, 3]],
            #   "f2": [[13, 13, 13]]
            # })

            expected_values_and_lengths = [
                (
                    [3, 3, 2, 2, 1, 4, 13, 13, 13, 12, 12, 11, 14, 14, 14, 14],
                    [2, 2, 1, 1, 3, 2, 1, 4],
                ),
                ([3, 3, 1, 13, 13, 13, 11], [2, 1, 3, 1]),
                ([4, 2, 2, 14, 14, 14, 14, 12, 12], [1, 2, 4, 2]),
                ([3, 3, 13, 13, 13], [2, 3]),
            ]

            torch.testing.assert_close(
                kjt.values(),
                torch.tensor(
                    expected_values_and_lengths[ctx.rank][0],
                    dtype=kjt.values().dtype,
                    device=kjt.values().device,
                ),
            )

            torch.testing.assert_close(
                kjt.lengths(),
                torch.tensor(
                    expected_values_and_lengths[ctx.rank][1],
                    dtype=kjt.lengths().dtype,
                    device=kjt.lengths().device,
                ),
            )

            assert list(sharded_keyed_jagged_tensor_pool.state_dict().keys()) == [
                "values",
                "key_lengths",
            ]

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharded_kjt_pool_inference(self) -> None:
        world_size = 2
        pool_size = 4
        device = torch.device("cpu")
        cuda_device = torch.device("cuda:0")

        # init global state is
        # 4         8
        # f1       f2
        # [3,3] .  [13,13,13]
        # [2,2] .  [12,12]
        # [1] .    [11]
        # [4]      [14,14,14,14]

        kjt_pool_orig = KeyedJaggedTensorPool(
            pool_size=pool_size,
            feature_max_lengths={"f1": 2, "f2": 4},
            values_dtype=torch.int,
            is_weighted=False,
            device=torch.device("cpu"),
        )
        kjt_pool_orig.update(
            ids=torch.tensor([0, 1, 2, 3], dtype=torch.int, device=device),
            values=KeyedJaggedTensor.from_lengths_sync(
                keys=["f1", "f2"],
                values=torch.tensor(
                    [3, 3, 2, 2, 1, 4, 13, 13, 13, 12, 12, 11, 14, 14, 14, 14],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
                lengths=torch.tensor(
                    [2, 2, 1, 1, 3, 2, 1, 4],
                    dtype=torch.int,
                    device=torch.device("cpu"),
                ),
            ),
        )

        sharded_inference_kjt_pool = _shard_modules(
            kjt_pool_orig,
            plan=ShardingPlan(
                plan={
                    "": ObjectPoolShardingPlan(
                        ObjectPoolShardingType.ROW_WISE, inference=True
                    ),
                }
            ),
            device=cuda_device,
            env=ShardingEnv.from_local(world_size=world_size, rank=0),
            sharders=[
                cast(ModuleSharder[torch.nn.Module], KeyedJaggedTensorPoolSharder())
            ],
        )
        self.assertIsInstance(
            sharded_inference_kjt_pool, ShardedInferenceKeyedJaggedTensorPool
        )

        self.assertEqual(sharded_inference_kjt_pool.dtype, torch.int)

        from torchrec.fx.tracer import symbolic_trace

        sharded_inference_kjt_pool_gm: torch.fx.GraphModule = symbolic_trace(
            sharded_inference_kjt_pool
        )
        sharded_inference_kjt_pool_gm_script = torch.jit.script(
            sharded_inference_kjt_pool_gm
        )  # noqa

        input_cases = [[0, 1, 2, 3], [0, 2, 1, 3]]
        for input in input_cases:
            input = torch.tensor(input, dtype=torch.int64)
            ref = kjt_pool_orig.lookup(input)
            val = sharded_inference_kjt_pool.lookup(input.to(cuda_device))

            torch.testing.assert_close(ref.values().cpu(), val.values().cpu())
            torch.testing.assert_close(ref.length_per_key(), val.length_per_key())

            val_gm_script = sharded_inference_kjt_pool_gm_script(input.to(cuda_device))
            torch.testing.assert_close(ref.values().cpu(), val_gm_script.values().cpu())
            torch.testing.assert_close(
                ref.length_per_key(), val_gm_script.length_per_key()
            )

        assert hasattr(sharded_inference_kjt_pool_gm_script, "_local_kjt_pool_shards")
        assert hasattr(sharded_inference_kjt_pool_gm_script._local_kjt_pool_shards, "0")
        assert hasattr(sharded_inference_kjt_pool_gm_script._local_kjt_pool_shards, "1")
