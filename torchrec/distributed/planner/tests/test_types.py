#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from copy import deepcopy
from typing import cast, Dict, Optional
from unittest.mock import MagicMock

import torch
from torch import multiprocessing
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.perf_models import NoopPerfModel
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    hash_planner_context_inputs,
    ParameterConstraints,
    Shard,
    ShardingOption,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    KeyValueParams,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionCollection,
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)


class TestShardingOption(unittest.TestCase):
    def test_hash_sharding_option(self) -> None:
        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ebc", MagicMock()),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
            cache_params=CacheParams(
                algorithm=CacheAlgorithm.LRU,
                load_factor=0.5,
                reserved_memory=0.0,
                precision=DataType.FP16,
                prefetch_pipeline=True,
            ),
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode.WARNING,
        )
        self.assertTrue(map(hash, [sharding_option]))

    def test_module_pooled_ebc(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])

        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ebc", ebc),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[
                Shard(size=[10000, 80], offset=offset) for offset in [[0, 0], [0, 80]]
            ],
        )
        self.assertEqual(sharding_option.is_pooled, True)

    def test_module_pooled_mch_ebc(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])
        mc_modules = {
            "table_0": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=10000,
                    device=torch.device("meta"),
                    eviction_interval=1,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=[eb_config],
        )
        mch_ebc = ManagedCollisionEmbeddingBagCollection(ebc, mcc)

        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 80), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("mch_ebc", mch_ebc),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[
                Shard(size=[10000, 80], offset=offset) for offset in [[0, 0], [0, 80]]
            ],
        )
        self.assertEqual(sharding_option.is_pooled, True)

    def test_module_pooled_ec(self) -> None:
        e_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=80,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ec = EmbeddingCollection(tables=[e_config])

        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ec", ec),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
        )
        self.assertEqual(sharding_option.is_pooled, False)

    def test_module_pooled_mch_ec(self) -> None:
        e_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=80,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ec = EmbeddingCollection(tables=[e_config])
        mc_modules = {
            "table_0": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=10000,
                    device=torch.device("meta"),
                    eviction_interval=1,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=[e_config],
        )
        mch_ec = ManagedCollisionEmbeddingCollection(ec, mcc)

        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("mch_ec", mch_ec),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
        )
        self.assertEqual(sharding_option.is_pooled, False)


class TestTopologyHash(unittest.TestCase):
    def test_hash_equality(self) -> None:
        # Create two identical Topology instances
        topology1 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 2,
            local_world_size=2,
        )

        topology2 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 2,
            local_world_size=2,
        )

        # Verify that the hash values are equal
        self.assertEqual(
            topology1._hash(),
            topology2._hash(),
            "Hashes should be equal for identical Topology instances",
        )

    def test_hash_inequality(self) -> None:
        # Create two different Topology instances
        topology1 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 2,
            local_world_size=2,
        )

        topology2 = Topology(
            world_size=4,  # Different world_size
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 2,
            local_world_size=2,
        )

        # Verify that the hash values are different
        self.assertNotEqual(
            topology1._hash(),
            topology2._hash(),
            "Hashes should be different for different Topology instances",
        )


class TestParameterConstraintsHash(unittest.TestCase):

    def test_hash_equality(self) -> None:
        # Create two identical instances
        pc1 = ParameterConstraints(
            sharding_types=["type1", "type2"],
            compute_kernels=["kernel1"],
            min_partition=4,
            pooling_factors=[1.0, 2.0],
            num_poolings=[1.0],
            batch_sizes=[32],
            is_weighted=True,
            cache_params=CacheParams(),
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature1", "feature2"],
            output_dtype=DataType.FP32,
            device_group="cuda",
            key_value_params=KeyValueParams(),
        )

        pc2 = deepcopy(pc1)

        self.assertEqual(
            hash(pc1), hash(pc2), "Hashes should be equal for identical instances"
        )

    def test_hash_inequality(self) -> None:
        # Create two different instances
        pc1 = ParameterConstraints(
            sharding_types=["type1"],
            compute_kernels=["kernel1"],
            min_partition=4,
            pooling_factors=[1.0],
            num_poolings=[1.0],
            batch_sizes=[32],
            is_weighted=True,
            cache_params=CacheParams(),
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature1"],
            output_dtype=DataType.FP32,
            device_group="cuda",
            key_value_params=KeyValueParams(),
        )

        pc2 = ParameterConstraints(
            sharding_types=["type2"],
            compute_kernels=["kernel2"],
            min_partition=8,
            pooling_factors=[2.0],
            num_poolings=[2.0],
            batch_sizes=[64],
            is_weighted=False,
            cache_params=CacheParams(),
            enforce_hbm=False,
            stochastic_rounding=True,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature2"],
            output_dtype=DataType.FP16,
            device_group="cpu",
            key_value_params=KeyValueParams(),
        )

        self.assertNotEqual(
            hash(pc1), hash(pc2), "Hashes should be different for different instances"
        )

    def test_hash_equality_with_non_none_cache_and_key_value_params(self) -> None:
        # Create two identical instances with non-None cache_params and key_value_params
        cache_params1 = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        key_value_params1 = KeyValueParams(
            ssd_storage_directory="/tmp/ssd_storage",
            ssd_rocksdb_write_buffer_size=1024,
            ssd_rocksdb_shards=4,
            l2_cache_size=8,
        )

        pc1 = ParameterConstraints(
            sharding_types=["type1", "type2"],
            compute_kernels=["kernel1"],
            min_partition=4,
            pooling_factors=[1.0, 2.0],
            num_poolings=[1.0],
            batch_sizes=[32],
            is_weighted=True,
            cache_params=cache_params1,
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature1", "feature2"],
            output_dtype=DataType.FP32,
            device_group="cuda",
            key_value_params=key_value_params1,
        )

        cache_params2 = deepcopy(cache_params1)
        key_value_params2 = deepcopy(key_value_params1)

        pc2 = ParameterConstraints(
            sharding_types=["type1", "type2"],
            compute_kernels=["kernel1"],
            min_partition=4,
            pooling_factors=[1.0, 2.0],
            num_poolings=[1.0],
            batch_sizes=[32],
            is_weighted=True,
            cache_params=cache_params2,
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature1", "feature2"],
            output_dtype=DataType.FP32,
            device_group="cuda",
            key_value_params=key_value_params2,
        )

        self.assertEqual(
            hash(pc1),
            hash(pc2),
            "Hashes should be equal for identical instances with non-None cache_params and key_value_params",
        )

    def test_hash_inequality_with_non_none_cache_and_key_value_params(self) -> None:
        # Create two different instances with different non-None cache_params and key_value_params
        cache_params1 = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        key_value_params1 = KeyValueParams(
            ssd_storage_directory="/tmp/ssd_storage",
            ssd_rocksdb_write_buffer_size=1024,
            ssd_rocksdb_shards=4,
            l2_cache_size=8,
        )

        pc1 = ParameterConstraints(
            sharding_types=["type1"],
            compute_kernels=["kernel1"],
            min_partition=4,
            pooling_factors=[1.0],
            num_poolings=[1.0],
            batch_sizes=[32],
            is_weighted=True,
            cache_params=cache_params1,
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature1"],
            output_dtype=DataType.FP32,
            device_group="cuda",
            key_value_params=key_value_params1,
        )

        cache_params2 = CacheParams(
            algorithm=CacheAlgorithm.LFU,
            load_factor=0.8,
            reserved_memory=2048.0,
            precision=DataType.FP32,
            prefetch_pipeline=False,
        )
        key_value_params2 = KeyValueParams(
            ssd_storage_directory="/tmp/different_storage",
            ssd_rocksdb_write_buffer_size=2048,
            ssd_rocksdb_shards=8,
            l2_cache_size=16,
        )

        pc2 = ParameterConstraints(
            sharding_types=["type2"],
            compute_kernels=["kernel2"],
            min_partition=8,
            pooling_factors=[2.0],
            num_poolings=[2.0],
            batch_sizes=[64],
            is_weighted=False,
            cache_params=cache_params2,
            enforce_hbm=False,
            stochastic_rounding=True,
            bounds_check_mode=BoundsCheckMode(1),
            feature_names=["feature2"],
            output_dtype=DataType.FP16,
            device_group="cpu",
            key_value_params=key_value_params2,
        )

        self.assertNotEqual(
            hash(pc1),
            hash(pc2),
            "Hashes should be different for different instances with non-None cache_params and key_value_params",
        )


def _test_hashing_consistency(
    rank: int,
    world_size: int,
    backend: str,
    return_hash_dict: Dict[str, int],
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        topology = Topology(
            local_world_size=8,
            world_size=1,
            compute_device="cuda",
        )
        batch_size = 128
        enumerator = EmbeddingEnumerator(topology=topology, batch_size=batch_size)
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        module = EmbeddingBagCollection(
            tables=[eb_config],
            is_weighted=False,
            device=torch.device(
                "meta"
            ),  # Using meta device for now since only getting search space
        )
        sharders = [EmbeddingBagCollectionSharder()]
        enumerator.enumerate(module, sharders)  # pyre-ignore
        storage_reservation = HeuristicalStorageReservation(percentage=0.15)
        constraints = {"table1": ParameterConstraints()}

        storage_reservation.reserve(
            topology=topology,
            batch_size=batch_size,
            module=module,
            sharders=sharders,  # pyre-ignore
            constraints=constraints,
        )
        perf_model = NoopPerfModel(topology=topology)

        planner1 = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            performance_model=perf_model,
            constraints=constraints,
        )

        h = planner1.hash_planner_context_inputs()
        return_hash_dict[str(rank)] = h


class TestHashPlannerContextInputsRounding(unittest.TestCase):
    """Tests for device memory rounding in hash_planner_context_inputs."""

    def _create_mock_enumerator(self) -> MagicMock:
        """Create a mock enumerator with search space."""
        enumerator = MagicMock()
        enumerator.last_stored_search_space = [
            MagicMock(
                fqn="table_0",
                sharding_type=ShardingType.TABLE_WISE.value,
                compute_kernel=EmbeddingComputeKernel.FUSED.value,
                shards=(),
                cache_params=None,
            )
        ]
        return enumerator

    def _create_mock_storage_reservation(self) -> MagicMock:
        """Create a mock storage reservation."""
        storage_reservation = MagicMock()
        storage_reservation._last_reserved_topology = "mock_topology"
        return storage_reservation

    def test_rounding_produces_same_hash_for_small_memory_differences(self) -> None:
        """Test that small memory differences (within 1% tolerance) produce the same hash."""
        # Setup: create two topologies with slightly different memory values
        hbm_base = 1024 * 1024 * 1024  # 1 GB
        # Small difference that should round to the same value
        hbm_slightly_different = hbm_base + 1000  # 1000 bytes difference

        topology1 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=hbm_base,
            local_world_size=2,
        )

        topology2 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=hbm_slightly_different,
            local_world_size=2,
        )

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        batch_size = 128

        # Execute: compute hashes for both topologies
        hash1 = hash_planner_context_inputs(
            topology=topology1,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        hash2 = hash_planner_context_inputs(
            topology=topology2,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        # Assert: hashes should be equal due to rounding
        self.assertEqual(
            hash1,
            hash2,
            "Hashes should be equal for topologies with small memory differences",
        )

    def test_rounding_produces_different_hash_for_large_memory_differences(
        self,
    ) -> None:
        """Test that large memory differences produce different hashes."""
        # Setup: create two topologies with significantly different memory values
        hbm_base = 1024 * 1024 * 1024  # 1 GB
        hbm_significantly_different = hbm_base * 100

        topology1 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=hbm_base,
            local_world_size=2,
        )

        topology2 = Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=hbm_significantly_different,
            local_world_size=2,
        )

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        batch_size = 128

        # Execute: compute hashes for both topologies
        hash1 = hash_planner_context_inputs(
            topology=topology1,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        hash2 = hash_planner_context_inputs(
            topology=topology2,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        # Assert: hashes should be different due to significant memory difference
        self.assertNotEqual(
            hash1,
            hash2,
            "Hashes should be different for topologies with large memory differences",
        )

    def test_rounding_consistency_across_devices(self) -> None:
        """Test that rounding is applied consistently across multiple devices."""
        # Setup: create topologies with multiple devices having small memory variations
        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        batch_size = 128

        # Create two topologies with slightly different memory for all devices
        topology1 = Topology(
            world_size=4,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 1024,
            local_world_size=4,
        )

        topology2 = Topology(
            world_size=4,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 1024 + 500,  # Small difference
            local_world_size=4,
        )

        # Execute: compute hashes for both topologies
        hash1 = hash_planner_context_inputs(
            topology=topology1,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        hash2 = hash_planner_context_inputs(
            topology=topology2,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )

        # Assert: hashes should be equal due to rounding
        self.assertEqual(
            hash1,
            hash2,
            "Hashes should be equal for multi-device topologies with small memory differences",
        )


class TestHashPlannerContextInputsWithConstraints(unittest.TestCase):
    """Tests for hash_planner_context_inputs with ParameterConstraints."""

    def _create_mock_enumerator(self) -> MagicMock:
        """Create a mock enumerator with search space."""
        enumerator = MagicMock()
        enumerator.last_stored_search_space = [
            MagicMock(
                fqn="table_0",
                sharding_type=ShardingType.TABLE_WISE.value,
                compute_kernel=EmbeddingComputeKernel.FUSED.value,
                shards=(),
                cache_params=None,
                key_value_params=None,
            )
        ]
        return enumerator

    def _create_mock_storage_reservation(self) -> MagicMock:
        """Create a mock storage reservation."""
        storage_reservation = MagicMock()
        storage_reservation._last_reserved_topology = "mock_topology"
        return storage_reservation

    def _create_topology(self) -> Topology:
        """Create a standard topology for tests."""
        return Topology(
            world_size=2,
            compute_device="cuda",
            hbm_cap=1024 * 1024 * 1024,
            local_world_size=2,
        )

    def test_hash_equality_with_identical_constraints(self) -> None:
        """Test that identical constraints produce the same hash."""
        # Setup: create two identical constraints
        cache_params1 = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        key_value_params1 = KeyValueParams(
            ssd_storage_directory="/tmp/ssd_storage",
            ssd_rocksdb_write_buffer_size=1024,
            ssd_rocksdb_shards=4,
            l2_cache_size=8,
        )
        constraints1 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params1,
                key_value_params=key_value_params1,
            )
        }

        cache_params2 = deepcopy(cache_params1)
        key_value_params2 = deepcopy(key_value_params1)
        constraints2 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params2,
                key_value_params=key_value_params2,
            )
        }

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        topology = self._create_topology()
        batch_size = 128

        # Execute: compute hashes with identical constraints
        hash1 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints1,
        )
        hash2 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints2,
        )

        # Assert: hashes should be equal
        self.assertEqual(
            hash1,
            hash2,
            "Hashes should be equal for identical constraints with cache_params and key_value_params",
        )

    def test_hash_inequality_with_different_constraints_cache_params(self) -> None:
        """Test that different cache_params in constraints produce different hashes."""
        # Setup: create two constraints with different cache_params
        cache_params1 = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        constraints1 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params1,
                key_value_params=None,
            )
        }

        cache_params2 = CacheParams(
            algorithm=CacheAlgorithm.LFU,
            load_factor=0.8,
            reserved_memory=2048.0,
            precision=DataType.FP32,
            prefetch_pipeline=False,
        )
        constraints2 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params2,
                key_value_params=None,
            )
        }

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        topology = self._create_topology()
        batch_size = 128

        # Execute: compute hashes with different cache_params
        hash1 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints1,
        )
        hash2 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints2,
        )

        # Assert: hashes should be different
        self.assertNotEqual(
            hash1,
            hash2,
            "Hashes should be different for constraints with different cache_params",
        )

    def test_hash_inequality_with_different_constraints_kv_params(self) -> None:
        """Test that different key_value_params in constraints produce different hashes."""
        # Setup: create two constraints with different key_value_params
        key_value_params1 = KeyValueParams(
            ssd_storage_directory="/tmp/ssd_storage",
            ssd_rocksdb_write_buffer_size=1024,
            ssd_rocksdb_shards=4,
            l2_cache_size=8,
        )
        constraints1 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=None,
                key_value_params=key_value_params1,
            )
        }

        key_value_params2 = KeyValueParams(
            ssd_storage_directory="/tmp/different_storage",
            ssd_rocksdb_write_buffer_size=2048,
            ssd_rocksdb_shards=8,
            l2_cache_size=16,
        )
        constraints2 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=None,
                key_value_params=key_value_params2,
            )
        }

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        topology = self._create_topology()
        batch_size = 128

        # Execute: compute hashes with different key_value_params
        hash1 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints1,
        )
        hash2 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints2,
        )

        # Assert: hashes should be different
        self.assertNotEqual(
            hash1,
            hash2,
            "Hashes should be different for constraints with different key_value_params",
        )

    def test_hash_inequality_constraints_none_vs_non_none(self) -> None:
        """Test that None constraints vs non-None constraints produce different hashes."""
        # Setup: create constraints with non-None values
        cache_params = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        constraints_non_none = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params,
                key_value_params=None,
            )
        }

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        topology = self._create_topology()
        batch_size = 128

        # Execute: compute hashes with None vs non-None constraints
        hash_none = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=None,
        )
        hash_non_none = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints_non_none,
        )

        # Assert: hashes should be different
        self.assertNotEqual(
            hash_none,
            hash_non_none,
            "Hashes should be different for None vs non-None constraints",
        )

    def test_hash_consistency_with_multiple_tables_in_constraints(self) -> None:
        """Test hash consistency when constraints contain multiple tables with various param combinations."""
        # Setup: create constraints with multiple tables
        cache_params = CacheParams(
            algorithm=CacheAlgorithm.LRU,
            load_factor=0.5,
            reserved_memory=1024.0,
            precision=DataType.FP16,
            prefetch_pipeline=True,
        )
        key_value_params = KeyValueParams(
            ssd_storage_directory="/tmp/ssd_storage",
            ssd_rocksdb_write_buffer_size=1024,
            ssd_rocksdb_shards=4,
            l2_cache_size=8,
        )

        constraints1 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params,
                key_value_params=None,
            ),
            "table_1": ParameterConstraints(
                sharding_types=["row_wise"],
                compute_kernels=["fused"],
                cache_params=None,
                key_value_params=key_value_params,
            ),
            "table_2": ParameterConstraints(
                sharding_types=["column_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params,
                key_value_params=key_value_params,
            ),
        }

        # Create identical constraints
        cache_params2 = deepcopy(cache_params)
        key_value_params2 = deepcopy(key_value_params)

        constraints2 = {
            "table_0": ParameterConstraints(
                sharding_types=["table_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params2,
                key_value_params=None,
            ),
            "table_1": ParameterConstraints(
                sharding_types=["row_wise"],
                compute_kernels=["fused"],
                cache_params=None,
                key_value_params=key_value_params2,
            ),
            "table_2": ParameterConstraints(
                sharding_types=["column_wise"],
                compute_kernels=["fused"],
                cache_params=cache_params2,
                key_value_params=key_value_params2,
            ),
        }

        enumerator = self._create_mock_enumerator()
        storage_reservation = self._create_mock_storage_reservation()
        topology = self._create_topology()
        batch_size = 128

        # Execute: compute hashes with identical multi-table constraints
        hash1 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints1,
        )
        hash2 = hash_planner_context_inputs(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints2,
        )

        # Assert: hashes should be equal for identical constraints
        self.assertEqual(
            hash1,
            hash2,
            "Hashes should be equal for identical multi-table constraints",
        )


class TestConsistentHashingBetweenProcesses(MultiProcessTestBase):
    # the proposal order might vary in github action so skip this test
    def test_hash_consistency_disabled_in_oss_compatibility(self) -> None:
        # planner
        world_size = 2
        return_hash_dict = multiprocessing.Manager().dict()
        self._run_multi_process_test(
            callable=_test_hashing_consistency,
            world_size=world_size,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            return_hash_dict=return_hash_dict,
        )
        hashes = return_hash_dict.values()
        assert hashes[0] == hashes[1], "hash values are different."
