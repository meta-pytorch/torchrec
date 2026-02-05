#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.distributed.sharding.rw_tensor_pool_sharding import (
    InferRwTensorPoolSharding,
    TensorPoolRwSharding,
)
from torchrec.distributed.tensor_sharding import TensorPoolRwShardingContext
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingEnv


class TestTensorPoolRwSharding(MultiProcessTestBase):
    @staticmethod
    def _test_update(
        rank: int,
        world_size: int,
    ) -> None:
        backend = "nccl"
        dtype = torch.float32
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            #  `Optional[ProcessGroup]`.
            sharding_env = ShardingEnv.from_process_group(ctx.pg)
            if ctx.rank == 0:
                ids = [4, 1]
                values = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]

            else:
                ids = [3, 0]
                values = [0.11, 0.21, 0.31], [0.41, 0.51, 0.61]

            ids = torch.tensor(ids, dtype=torch.int, device=ctx.device)
            values = torch.tensor(values, dtype=torch.float, device=ctx.device)

            block_size = torch.tensor([3], dtype=torch.int, device=ctx.device)
            update_ctx = TensorPoolRwShardingContext(block_size=block_size)
            rw_sharding = TensorPoolRwSharding(
                env=sharding_env, device=ctx.device, dim=3, pool_size=4
            )
            input_dist = rw_sharding.create_lookup_ids_dist()
            update_values_dist = rw_sharding.create_update_values_dist()
            dist_ids = input_dist(ctx=update_ctx, ids=ids).wait().wait()

            torch.testing.assert_close(
                dist_ids.cpu(),
                torch.tensor(
                    [1, 0],
                    device=torch.device("cpu"),
                    dtype=torch.int,
                ),
            )

            dist_values = update_values_dist(ctx=update_ctx, values=values).wait()
            if rank == 0:
                torch.testing.assert_close(
                    dist_values.cpu(),
                    torch.tensor(
                        [[0.4, 0.5, 0.6], [0.41, 0.51, 0.61]],
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                )
            else:
                torch.testing.assert_close(
                    dist_values.cpu(),
                    torch.tensor(
                        [[0.1, 0.2, 0.3], [0.11, 0.21, 0.31]],
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                )

    @staticmethod
    def _test_lookup(
        rank: int,
        world_size: int,
    ) -> None:
        backend = "nccl"
        dtype = torch.float32
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            #  `Optional[ProcessGroup]`.
            sharding_env = ShardingEnv.from_process_group(ctx.pg)

            block_size = torch.tensor([3], dtype=torch.int, device=ctx.device)
            lookup_ctx = TensorPoolRwShardingContext(block_size=block_size)
            rw_sharding = TensorPoolRwSharding(
                env=sharding_env, device=ctx.device, dim=3, pool_size=5
            )
            input_dist = rw_sharding.create_lookup_ids_dist()
            lookup_values_dist = rw_sharding.create_lookup_values_dist()

            ids = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=ctx.device)
            dist_ids = input_dist(ctx=lookup_ctx, ids=ids).wait().wait()
            if rank == 0:
                torch.testing.assert_close(
                    dist_ids.cpu(),
                    torch.tensor(
                        [0, 1, 2, 0, 1, 2],
                        dtype=torch.int,
                        device=torch.device("cpu"),
                    ),
                )
            else:
                torch.testing.assert_close(
                    dist_ids.cpu(),
                    torch.tensor(
                        [0, 0],
                        dtype=torch.int,
                        device=torch.device("cpu"),
                    ),
                )

            # assume the _local_pool on rank 0 is
            # [
            # [0.41, 0.51, 0.61],
            # [0.4, 0.5, 0.6],
            # [0.0, 0.0, 0.0],
            # ]

            # on rank 1 is
            # [
            # [0.11, 0.21, 0.31],
            # [0.1, 0.2, 0.3],
            # ]

            if rank == 0:
                lookup_values = torch.tensor(
                    [
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0],
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=dtype,
                    device=ctx.device,
                )

            else:
                lookup_values = torch.tensor(
                    [
                        [0.11, 0.21, 0.31],
                        [0.11, 0.21, 0.31],
                    ],
                    dtype=dtype,
                    device=ctx.device,
                )

            dist_output_values = lookup_values_dist(
                ctx=lookup_ctx, values=lookup_values
            ).wait()

            torch.testing.assert_close(
                dist_output_values.cpu(),
                torch.tensor(
                    [
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
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
    def test_update(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(callable=self._test_update, world_size=world_size)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_lookup(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(callable=self._test_lookup, world_size=world_size)


class TestInferRwTensorPoolSharding(unittest.TestCase):
    def test_uneven_sharding_with_memory_capacity_per_rank(self) -> None:
        # Setup: create a sharding configuration with uneven memory capacity per rank
        # Rank 0 gets 60% capacity, Rank 1 gets 20%, Rank 2 gets 20%
        pool_size = 1000
        world_size = 3
        memory_capacity_per_rank = [600, 200, 200]  # Uneven distribution
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        # Execute: create InferRwTensorPoolSharding with memory_capacity_per_rank
        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Assert: verify the local pool size per rank is computed based on memory capacity
        # Expected: rank 0 gets 600 rows, rank 1 gets 200 rows, rank 2 gets 200 rows
        expected_local_pool_size_per_rank = [600, 200, 200]
        self.assertEqual(
            sharding.local_pool_size_per_rank, expected_local_pool_size_per_rank
        )

        # Assert: verify block_bucketize_row_pos is set for uneven sharding
        self.assertIsNotNone(sharding._block_bucketize_row_pos)
        self.assertEqual(len(sharding._block_bucketize_row_pos), 1)

        # Assert: verify the row offsets are correct [0, 600, 800, 1000]
        expected_row_offsets = torch.tensor([0, 600, 800, 1000], device=device)
        torch.testing.assert_close(
            sharding._block_bucketize_row_pos[0],
            expected_row_offsets,
        )

    def test_uneven_sharding_with_different_capacities(self) -> None:
        # Setup: create a sharding configuration with different memory capacities
        # Rank 0 gets 50% capacity, Rank 1 gets 30%, Rank 2 gets 20%
        pool_size = 500
        world_size = 3
        memory_capacity_per_rank = [500, 300, 200]  # Different distribution
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=1)

        # Execute: create InferRwTensorPoolSharding with memory_capacity_per_rank
        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Assert: verify the local pool size per rank is computed proportionally
        # Total capacity = 1000
        # Rank 0: 500/1000 * 500 = 250 rows
        # Rank 1: 300/1000 * 500 = 150 rows
        # Rank 2: remaining = 500 - 250 - 150 = 100 rows
        expected_local_pool_size_per_rank = [250, 150, 100]
        self.assertEqual(
            sharding.local_pool_size_per_rank, expected_local_pool_size_per_rank
        )

        # Assert: verify the row offsets are correct [0, 250, 400, 500]
        expected_row_offsets = torch.tensor([0, 250, 400, 500], device=device)
        torch.testing.assert_close(
            sharding._block_bucketize_row_pos[0],
            expected_row_offsets,
        )

    def test_uneven_sharding_total_rows_equals_pool_size(self) -> None:
        # Setup: verify that the sum of local pool sizes equals the pool size
        pool_size = 1234
        world_size = 4
        memory_capacity_per_rank = [100, 200, 300, 400]
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        # Execute: create InferRwTensorPoolSharding with memory_capacity_per_rank
        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Assert: verify the sum of local pool sizes equals the total pool size
        total_rows = sum(sharding.local_pool_size_per_rank)
        self.assertEqual(total_rows, pool_size)

        # Assert: verify the last row offset equals the pool size
        self.assertEqual(sharding._block_bucketize_row_pos[0][-1].item(), pool_size)

    def test_even_sharding_without_memory_capacity_per_rank(self) -> None:
        # Setup: create a sharding configuration without memory_capacity_per_rank
        # This should result in even sharding
        pool_size = 1000
        world_size = 4
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        # Execute: create InferRwTensorPoolSharding without memory_capacity_per_rank
        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=None,
        )

        # Assert: verify the local pool size per rank is evenly distributed
        # block_size = (1000 + 4 - 1) // 4 = 250
        # Expected: [250, 250, 250, 250]
        expected_local_pool_size_per_rank = [250, 250, 250, 250]
        self.assertEqual(
            sharding.local_pool_size_per_rank, expected_local_pool_size_per_rank
        )

        # Assert: verify block_bucketize_row_pos is None for even sharding
        self.assertIsNone(sharding._block_bucketize_row_pos)

    def test_lookup_ids_dist_uses_block_bucketize_row_pos(self) -> None:
        # Setup: create a sharding configuration with uneven memory capacity
        pool_size = 1000
        world_size = 3
        memory_capacity_per_rank = [600, 200, 200]
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Execute: create the lookup ids distribution module
        lookup_ids_dist = sharding.create_lookup_ids_dist()

        # Assert: verify the lookup_ids_dist has the correct block_bucketize_row_pos
        self.assertIsNotNone(lookup_ids_dist._block_bucketize_row_pos)
        self.assertEqual(len(lookup_ids_dist._block_bucketize_row_pos), 1)

        # Assert: verify the row offsets match the sharding configuration
        expected_row_offsets = torch.tensor([0, 600, 800, 1000], device=device)
        torch.testing.assert_close(
            lookup_ids_dist._block_bucketize_row_pos[0], expected_row_offsets
        )

    def test_lookup_with_uneven_sharding_bucketizes_ids_correctly(self) -> None:
        # Setup: create a sharding configuration with uneven memory capacity
        # Rank 0 gets rows 0-599 (60% capacity)
        # Rank 1 gets rows 600-799 (20% capacity)
        # Rank 2 gets rows 800-999 (20% capacity)
        pool_size = 1000
        world_size = 3
        memory_capacity_per_rank = [600, 200, 200]
        device = torch.device("cpu")

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Execute: create the lookup ids distribution and test with various IDs
        # IDs 0, 100, 599 should go to rank 0
        # IDs 600, 700 should go to rank 1
        # IDs 800, 900, 999 should go to rank 2
        lookup_ids_dist = sharding.create_lookup_ids_dist()
        test_ids = torch.tensor([0, 100, 599, 600, 700, 800, 900, 999], device=device)

        dist_ids, unbucketize_permute, bucket_mapping, bucketized_lengths = (
            lookup_ids_dist(test_ids)
        )

        # Assert: verify the number of IDs per rank matches expected distribution
        # Rank 0 should receive 3 IDs (0, 100, 599)
        # Rank 1 should receive 2 IDs (600, 700)
        # Rank 2 should receive 3 IDs (800, 900, 999)
        expected_lengths = torch.tensor([3, 2, 3], device=device)
        torch.testing.assert_close(bucketized_lengths, expected_lengths)

        # Assert: verify IDs are correctly distributed to each rank
        # Note: IDs are stored as local offsets within each rank's pool
        # Rank 0 IDs: 0, 100, 599 (no offset needed, rank 0 starts at 0)
        torch.testing.assert_close(
            dist_ids[0], torch.tensor([0, 100, 599], device=device)
        )
        # Rank 1 IDs: 600, 700 become 0, 100 (offset by 600, rank 1 starts at 600)
        torch.testing.assert_close(dist_ids[1], torch.tensor([0, 100], device=device))
        # Rank 2 IDs: 800, 900, 999 become 0, 100, 199 (offset by 800, rank 2 starts at 800)
        torch.testing.assert_close(
            dist_ids[2], torch.tensor([0, 100, 199], device=device)
        )

        # Assert: verify bucket_mapping assigns IDs to correct ranks
        # IDs 0, 100, 599 -> rank 0
        # IDs 600, 700 -> rank 1
        # IDs 800, 900, 999 -> rank 2
        expected_bucket_mapping = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], device=device)
        torch.testing.assert_close(bucket_mapping, expected_bucket_mapping)

    def test_lookup_output_with_uneven_sharding(self) -> None:
        # Setup: create a sharding configuration with uneven memory capacity
        # This test validates the complete lookup flow including output distribution
        pool_size = 10
        world_size = 3
        # Rank 0: 60% -> 6 rows (IDs 0-5)
        # Rank 1: 20% -> 2 rows (IDs 6-7)
        # Rank 2: 20% -> 2 rows (IDs 8-9)
        memory_capacity_per_rank = [600, 200, 200]
        device = torch.device("cpu")
        dim = 3

        # Create a mock ShardingEnv
        class MockShardingEnv:
            def __init__(self, world_size: int, rank: int) -> None:
                self.world_size = world_size
                self.rank = rank
                self.process_group = None

        env = MockShardingEnv(world_size=world_size, rank=0)

        sharding = InferRwTensorPoolSharding(
            pool_size=pool_size,
            env=env,
            device=device,
            memory_capacity_per_rank=memory_capacity_per_rank,
        )

        # Execute: test lookup with IDs from different ranks
        lookup_ids_dist = sharding.create_lookup_ids_dist()
        lookup_values_dist = sharding.create_lookup_values_dist()

        # Test IDs: 0, 5 (rank 0), 6, 7 (rank 1), 8 (rank 2)
        test_ids = torch.tensor([0, 5, 6, 7, 8], device=device)

        dist_ids, unbucketize_permute, bucket_mapping, bucketized_lengths = (
            lookup_ids_dist(test_ids)
        )

        # Simulate lookup values from each rank's local pool
        # Rank 0 values for IDs 0, 5
        rank_0_values = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32, device=device
        )
        # Rank 1 values for IDs 6, 7
        rank_1_values = torch.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=torch.float32,
            device=device,
        )
        # Rank 2 values for ID 8
        rank_2_values = torch.tensor(
            [[13.0, 14.0, 15.0]], dtype=torch.float32, device=device
        )

        # Execute: merge the lookup values from all ranks
        lookups = [rank_0_values, rank_1_values, rank_2_values]
        merged_values = lookup_values_dist(lookups)

        # Assert: verify the merged values have the correct shape
        self.assertEqual(merged_values.shape, (5, dim))

        # Assert: verify values are correctly merged
        # The merged tensor should contain values in the order they were bucketized
        # After unbucketize_permute, values should be in original order
        expected_merged = torch.tensor(
            [
                [1.0, 2.0, 3.0],  # ID 0 from rank 0
                [4.0, 5.0, 6.0],  # ID 5 from rank 0
                [7.0, 8.0, 9.0],  # ID 6 from rank 1
                [10.0, 11.0, 12.0],  # ID 7 from rank 1
                [13.0, 14.0, 15.0],  # ID 8 from rank 2
            ],
            dtype=torch.float32,
            device=device,
        )
        torch.testing.assert_close(merged_values, expected_merged)
