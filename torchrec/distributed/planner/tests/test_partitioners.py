#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List
from unittest.mock import MagicMock

from pyre_extensions import none_throws
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
    OrderedDeviceHardware,
)
from torchrec.distributed.planner.types import (
    DeviceHardware,
    ParameterConstraints,
    PartitionByType,
    Perf,
    PlannerError,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import reset_shard_rank
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class RWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWRWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWCWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class HostLevelSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value, ShardingType.TABLE_COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestGreedyPerfPartitioner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.topology = Topology(world_size=2, compute_device=compute_device)
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()

    def get_devices_allocation(
        self, sharding_options: List[ShardingOption]
    ) -> List[DeviceHardware]:
        devices_allocation: List[DeviceHardware] = [
            DeviceHardware(rank, Storage(0, 0), Perf(0, 0, 0, 0, 0))
            for rank in range(self.topology.world_size)
        ]
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                rank = none_throws(shard.rank)
                devices_allocation[rank].storage += none_throws(shard.storage)
                devices_allocation[rank].perf += none_throws(shard.perf)
        return devices_allocation

    def test_tw_balanced_perf_device(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWSharder()],
        )

        for sharding_option in sharding_options:
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
            )
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [0],
            "table_3": [1],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        expected_perf = Perf(
            fwd_compute=80,
            fwd_comms=60,
            bwd_compute=40,
            bwd_comms=20,
        )
        devices_allocation = self.get_devices_allocation(sharding_plan)

        self.assertEqual(devices_allocation[0].perf, expected_perf)
        self.assertEqual(devices_allocation[1].perf, expected_perf)

        self.assertEqual(
            devices_allocation[0].storage,
            Storage(2000, 2000),
        )
        self.assertEqual(
            devices_allocation[1].storage,
            Storage(2000, 2000),
        )

    def test_device_partition_heap_invariant(self) -> None:
        """Validate that _device_partition maintains the minheap invariant."""

        def assert_heap(heap: List[OrderedDeviceHardware]) -> None:
            for i in range(len(heap)):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                self.assertFalse(left_child < len(heap) and heap[i] > heap[left_child])
                self.assertFalse(
                    right_child < len(heap) and heap[i] > heap[right_child]
                )

        def device_heaps_equal(
            heap1: List[OrderedDeviceHardware], heap2: List[OrderedDeviceHardware]
        ) -> None:
            # OrderedDeviceHardware 2-key is a partial-order (equally good items might
            # permute), so we validate that each heap maintains its heap invariant and
            # that device ids are identical between them.
            assert_heap(heap1)
            assert_heap(heap2)
            self.assertEqual(
                sorted([id(x.device) for x in heap1]),
                sorted([id(x.device) for x in heap2]),
            )
            # TODO(damian): with 3-key we have a full total ordering, so we can test
            # equivalence with the simpler below. For now leaving in the more complex
            # verification that works for both 2-key and 3-key, if we decide on 3-key we
            # can delete the more complex equality test.
            #   self.assertEqual([id(x.device) for x in heap1],
            #                    [id(x.device) for x in heap2])

        def perf(x: float) -> Perf:
            return Perf(fwd_compute=x, fwd_comms=0, bwd_compute=0, bwd_comms=0)

        def empty_devices() -> List[DeviceHardware]:
            return [
                DeviceHardware(
                    rank=x, storage=Storage(hbm=1_000_000, ddr=0), perf=perf(0)
                )
                for x in range(6)
            ]

        shards = [
            Shard(storage=Storage(hbm=1000, ddr=0), perf=perf(1), size=[], offset=[])
            for _ in range(30)
        ]

        sharding_option: ShardingOption = ShardingOption(
            name=MagicMock(),
            tensor=MagicMock(),
            module=MagicMock(),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=MagicMock(),
            partition_by=MagicMock(),
            compute_kernel=MagicMock(),
            shards=shards,
        )
        local_world_size: int = 3

        def validate(threshold: float) -> None:
            devices = empty_devices()
            minheap_devices = GreedyPerfPartitioner._establish_minheap(
                devices, local_world_size
            )

            GreedyPerfPartitioner._device_partition(
                sharding_option, minheap_devices, threshold
            )

            want_minheap_devices = GreedyPerfPartitioner._establish_minheap(
                devices, local_world_size
            )
            device_heaps_equal(minheap_devices, want_minheap_devices)

        validate(0)  # force heapify
        validate(1)  # force incremental rebuild

    def test_tw_unbalanced_perf_device(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            perf = (
                Perf(fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10)
                if i > 0
                else Perf(fwd_compute=75, fwd_comms=75, bwd_compute=75, bwd_comms=75)
            )
            sharding_option.shards[0].perf = perf
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [1],
            "table_3": [1],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        expected_perfs = [
            Perf(fwd_compute=75, fwd_comms=75, bwd_compute=75, bwd_comms=75),
            Perf(fwd_compute=120, fwd_comms=90, bwd_compute=60, bwd_comms=30),
        ]

        devices_allocation = self.get_devices_allocation(sharding_plan)

        self.assertEqual(devices_allocation[0].perf, expected_perfs[0])
        self.assertEqual(devices_allocation[1].perf, expected_perfs[1])

        self.assertEqual(
            devices_allocation[0].storage,
            Storage(1000, 1000),
        )
        self.assertEqual(
            devices_allocation[1].storage,
            Storage(3000, 3000),
        )

    def test_tw_balanced_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWRWSharder()],
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [8, 9, 10, 11, 12, 13, 14, 15],
            "table_2": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }

        devices_allocation = self.get_devices_allocation(sharding_plan)

        self.assertEqual(expected_ranks, ranks)

        for i in range(self.topology.world_size):
            self.assertEqual(
                devices_allocation[i].storage,
                # there are two shards allocated to each device
                Storage(2000, 2000),
            )
            expected_perf = Perf(
                fwd_compute=80,
                fwd_comms=60,
                bwd_compute=40,
                bwd_comms=20,
            )
            self.assertEqual(devices_allocation[i].perf, expected_perf)

    def test_rw_unbalanced_perf_uniform(self) -> None:
        self.topology = Topology(world_size=4, compute_device="cuda")
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[RWSharder()],
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.UNIFORM.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [0, 1, 2, 3],
            "table_1": [0, 1, 2, 3],
            "table_2": [0, 1, 2, 3],
            "table_3": [0, 1, 2, 3],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        devices_allocation = self.get_devices_allocation(sharding_plan)

        for i in range(self.topology.world_size):
            self.assertEqual(
                devices_allocation[i].storage,
                Storage(4000, 4000),
            )

    def test_twcw_unbalanced_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(min_partition=4 * 2),
            "table_1": ParameterConstraints(min_partition=4 * 10),
            "table_2": ParameterConstraints(min_partition=4 * 5),
            "table_3": ParameterConstraints(min_partition=4 * 8),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWCWSharder()],
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [8, 9, 10, 11, 12, 13, 14, 15, 8, 9],
            "table_1": [4, 5, 6, 7],
            "table_2": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
            "table_3": [10, 11, 12, 13, 14, 15, 8, 9, 10, 11],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

    def test_twrw_and_twcw_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 8,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 10,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[HostLevelSharder()],
        )

        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [8, 9, 10, 11, 12],
            "table_2": [0, 1, 2, 3, 4, 5],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }

        self.assertEqual(expected_ranks, ranks)

    def test_twrw_and_twcw_cohost(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 8,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 10,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[HostLevelSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value
            if i <= 2:
                sharding_option.dependency = "host_0"
        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [0, 1, 2, 3, 4],
            "table_2": [5, 6, 7, 0, 1, 2],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }

        self.assertEqual(expected_ranks, ranks)

        devices_allocation = self.get_devices_allocation(sharding_plan)

        for i in range(self.topology.world_size):
            total_storage = Storage(0, 0)
            total_perf = Perf(
                fwd_compute=0,
                fwd_comms=0,
                bwd_compute=0,
                bwd_comms=0,
            )
            for sharding_option in sharding_plan:
                for shard in sharding_option.shards:
                    if shard.rank == i:
                        total_storage += cast(Storage, shard.storage)
                        total_perf += cast(Perf, shard.perf)
            self.assertEqual(
                devices_allocation[i].storage,
                total_storage,
            )
            self.assertEqual(devices_allocation[i].perf, total_perf)

    def test_oom(self) -> None:
        self.topology = Topology(
            world_size=2, local_world_size=1, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 4,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 7,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=20 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[HostLevelSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(
                    # pyrefly: ignore[bad-argument-type]
                    hbm=self.topology.devices[0].storage.hbm / 2,
                    # pyrefly: ignore[bad-argument-type]
                    ddr=self.topology.devices[0].storage.ddr / 2,
                )
            sharding_option.partition_by = PartitionByType.HOST.value
            if i <= 2:
                sharding_option.dependency = "host_0"

        with self.assertRaises(PlannerError):
            self.partitioner.partition(
                proposal=sharding_options,
                storage_constraint=self.topology,
            )


class TestMemoryBalancedPartitioner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        self.topology = Topology(
            world_size=2,
            compute_device=compute_device,
            hbm_cap=2000 * 1024**2,
        )
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.greedy_perf_partitioner = GreedyPerfPartitioner()
        self.memory_balanced_partitioner = MemoryBalancedPartitioner(tolerance=100)

    def test_same_sharding_plan(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWSharder()],
        )

        for sharding_option in sharding_options:
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
            )
            sharding_option.shards[0].storage = Storage(
                hbm=1000 * 1024**2, ddr=1000 * 1024**2
            )

        greedy_perf_sharding_plan = self.greedy_perf_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        greedy_perf_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in greedy_perf_sharding_plan
        }

        reset_shard_rank(sharding_options)
        memory_balanced_sharding_plan = self.memory_balanced_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        memory_balanced_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in memory_balanced_sharding_plan
        }
        self.assertEqual(greedy_perf_ranks, memory_balanced_ranks)

    def test_different_sharding_plan(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40 * (i + 1), fwd_comms=0, bwd_compute=0, bwd_comms=0
            )
            sharding_option.shards[0].storage = Storage(
                hbm=(1500 - i * 500) * 1024**2, ddr=1000 * 1024**2
            )

        greedy_perf_sharding_plan = self.greedy_perf_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        greedy_perf_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in greedy_perf_sharding_plan
        }
        greedy_perf_expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [0],
        }
        self.assertEqual(greedy_perf_ranks, greedy_perf_expected_ranks)

        greedy_perf_hbm_uses = [0] * self.topology.world_size
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                if shard.storage and shard.rank is not None:
                    greedy_perf_hbm_uses[
                        #  but got `Optional[int]`.
                        shard.rank
                    ] += shard.storage.hbm

        reset_shard_rank(sharding_options)
        memory_balanced_sharding_plan = self.memory_balanced_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        memory_balanced_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in memory_balanced_sharding_plan
        }
        memory_balanced_expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [0],
        }
        self.assertEqual(memory_balanced_ranks, memory_balanced_expected_ranks)

        memory_balanced_hbm_uses = [0.0] * self.topology.world_size
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                if shard.storage and shard.rank:
                    #  got `Optional[int]`.
                    memory_balanced_hbm_uses[shard.rank] += shard.storage.hbm

        self.assertTrue(max(memory_balanced_hbm_uses) < max(greedy_perf_hbm_uses))


class TestBalanceModules(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=200 + i,
                embedding_dim=8 * (10 + i),
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(3)
        ]
        self.topology = Topology(
            world_size=2,
            compute_device=compute_device,
            hbm_cap=2000 * 1024**2,
        )
        self.model = TestSparseNN(tables=tables, weighted_tables=weighted_tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )

        self.sharding_options = self.enumerator.enumerate(
            module=self.model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[TWSharder()],
        )
        for sharding_option in self.sharding_options:
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
            )
            sharding_option.shards[0].storage = Storage(
                hbm=10 * 1024**2, ddr=1000 * 1024**2
            )

    def test_greedy_partitioner(self) -> None:
        greedy_partitioner = GreedyPerfPartitioner(balance_modules=False)
        balance_modules_greedy_partitioner = GreedyPerfPartitioner(balance_modules=True)

        greedy_sharding_plan = greedy_partitioner.partition(
            proposal=self.sharding_options,
            storage_constraint=self.topology,
        )
        greedy_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in greedy_sharding_plan
        }

        reset_shard_rank(self.sharding_options)

        balance_modules_sharding_plan = balance_modules_greedy_partitioner.partition(
            proposal=self.sharding_options,
            storage_constraint=self.topology,
        )
        balance_modules_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in balance_modules_sharding_plan
        }

        greedy_expected_ranks = {
            "weighted_table_0": [0],
            "weighted_table_1": [1],
            "weighted_table_2": [0],
            "table_0": [1],
        }
        balance_modules_expected_ranks = {
            "weighted_table_0": [1],
            "weighted_table_1": [0],
            "weighted_table_2": [1],
            "table_0": [0],
        }

        self.assertEqual(greedy_expected_ranks, greedy_ranks)
        self.assertEqual(balance_modules_expected_ranks, balance_modules_ranks)

    def test_memory_balanced_partitioner(self) -> None:
        memory_balanced_partitioner = MemoryBalancedPartitioner(
            tolerance=100, balance_modules=False
        )
        balance_modules_memory_balanced_partitioner = MemoryBalancedPartitioner(
            tolerance=100, balance_modules=True
        )

        memory_balanced_plan = memory_balanced_partitioner.partition(
            proposal=self.sharding_options,
            storage_constraint=self.topology,
        )
        memory_balanced_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in memory_balanced_plan
        }

        reset_shard_rank(self.sharding_options)

        balance_modules_sharding_plan = (
            balance_modules_memory_balanced_partitioner.partition(
                proposal=self.sharding_options,
                storage_constraint=self.topology,
            )
        )
        balance_modules_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in balance_modules_sharding_plan
        }

        memory_balanced_expected_ranks = {
            "weighted_table_0": [0],
            "weighted_table_1": [1],
            "weighted_table_2": [0],
            "table_0": [1],
        }
        balance_modules_expected_ranks = {
            "weighted_table_0": [1],
            "weighted_table_1": [0],
            "weighted_table_2": [1],
            "table_0": [0],
        }

        self.assertEqual(memory_balanced_expected_ranks, memory_balanced_ranks)
        self.assertEqual(balance_modules_expected_ranks, balance_modules_ranks)
