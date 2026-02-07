#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List

from torch import nn
from torchrec.distributed.embedding_tower_sharding import (
    EmbeddingTowerCollectionSharder,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BIGINT_DTYPE
from torchrec.distributed.planner.storage_reservations import (
    _get_batch_inputs_and_shardable_parameters,
    _get_kjt_storage,
    _get_module_size,
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import PlannerError, PlannerErrorType, Topology
from torchrec.distributed.test_utils.test_model import TestTowerInteraction
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection


class TestModel(nn.Module):
    def __init__(self, shardable_sparse: nn.Module) -> None:
        super().__init__()
        self.dense_arch = nn.Linear(10, 10)
        self.shardable_sparse = shardable_sparse


class TestHeuristicalStorageReservation(unittest.TestCase):

    def test_validate_storage_reservations_errors(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=1_000_000,
                embedding_dim=1024,
                name="table_0",
                feature_names=["feature_0"],
            ),
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)

        # Reserving 100% of HBM to make sure the heuristic storage reservation fails
        heuristical_storage_reservation = HeuristicalStorageReservation(percentage=1)
        with self.assertRaises(PlannerError) as context:
            heuristical_storage_reservation.reserve(
                topology=Topology(world_size=1, compute_device="cuda"),
                batch_size=1024,
                module=model,
                sharders=cast(
                    List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
                ),
            )

        self.assertEqual(
            context.exception.error_type, PlannerErrorType.INSUFFICIENT_STORAGE
        )

    def test_storage_reservations_ebc(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)

        heuristical_storage_reservation = HeuristicalStorageReservation(percentage=0.0)

        heuristical_storage_reservation.reserve(
            topology=Topology(world_size=2, compute_device="cuda"),
            batch_size=10,
            module=model,
            sharders=cast(
                List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
            ),
        )

        self.assertEqual(
            _get_module_size(model.dense_arch, multiplier=6),
            heuristical_storage_reservation._dense_storage.hbm,
        )

    def test_storage_reservations_tower(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
            )
            for idx in range(3)
        ]

        tower_0 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[0], tables[2]]),
            interaction_module=TestTowerInteraction(tables=[tables[0], tables[2]]),
        )
        tower_1 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[1]]),
            interaction_module=TestTowerInteraction(tables=[tables[1]]),
        )
        tower_arch = EmbeddingTowerCollection(towers=[tower_0, tower_1])

        model = TestModel(shardable_sparse=tower_arch)

        heuristical_storage_reservation = HeuristicalStorageReservation(percentage=0.0)

        heuristical_storage_reservation.reserve(
            topology=Topology(world_size=2, compute_device="cuda"),
            batch_size=10,
            module=model,
            sharders=cast(
                List[ModuleSharder[nn.Module]], [EmbeddingTowerCollectionSharder()]
            ),
        )

        self.assertEqual(
            _get_module_size(model.dense_arch, multiplier=6),
            heuristical_storage_reservation._dense_storage.hbm,
        )

    def test_storage_reservations_tower_nested_sharders(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
            )
            for idx in range(3)
        ]

        tower_0 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[0], tables[2]]),
            interaction_module=TestTowerInteraction(tables=[tables[0], tables[2]]),
        )
        tower_1 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[1]]),
            interaction_module=TestTowerInteraction(tables=[tables[1]]),
        )
        tower_arch = EmbeddingTowerCollection(towers=[tower_0, tower_1])

        model = TestModel(shardable_sparse=tower_arch)

        heuristical_storage_reservation = HeuristicalStorageReservation(percentage=0.0)

        heuristical_storage_reservation.reserve(
            topology=Topology(world_size=2, compute_device="cuda"),
            batch_size=10,
            module=model,
            sharders=cast(
                List[ModuleSharder[nn.Module]],
                [EmbeddingTowerCollectionSharder(), EmbeddingBagCollectionSharder()],
            ),
        )

        self.assertEqual(
            _get_module_size(model.dense_arch, multiplier=6),
            heuristical_storage_reservation._dense_storage.hbm,
        )

    def test_storage_reservations_with_dense_estimation(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)

        dense_tensor_estimate = 1000000
        heuristical_storage_reservation = HeuristicalStorageReservation(
            percentage=0.0, dense_tensor_estimate=dense_tensor_estimate
        )

        heuristical_storage_reservation.reserve(
            topology=Topology(world_size=2, compute_device="cuda"),
            batch_size=10,
            module=model,
            sharders=cast(
                List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
            ),
        )

        self.assertEqual(
            dense_tensor_estimate,
            heuristical_storage_reservation._dense_storage.hbm,
        )


class TestGetKjtStorage(unittest.TestCase):
    def test_get_kjt_storage_cuda(self) -> None:
        """Test _get_kjt_storage with CUDA topology returns storage with HBM."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cuda")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            model, sharders, batch_size=10
        )
        kjt_storage = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=20,
        )

        self.assertGreater(kjt_storage.hbm, 0)
        self.assertEqual(kjt_storage.ddr, 0)

    def test_get_kjt_storage_cpu(self) -> None:
        """Test _get_kjt_storage with CPU topology returns storage with DDR."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cpu")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            model, sharders, batch_size=10
        )
        kjt_storage = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=20,
        )

        self.assertEqual(kjt_storage.hbm, 0)
        self.assertGreater(kjt_storage.ddr, 0)

    def test_get_kjt_storage_mtia(self) -> None:
        """Test _get_kjt_storage with MTIA topology returns storage with HBM."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="mtia")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            model, sharders, batch_size=10
        )
        kjt_storage = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=20,
        )

        self.assertGreater(kjt_storage.hbm, 0)
        self.assertEqual(kjt_storage.ddr, 0)

    def test_get_kjt_storage_multiple_tables(self) -> None:
        """Test _get_kjt_storage with multiple tables computes aggregate size."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
            )
            for idx in range(3)
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cuda")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            model, sharders, batch_size=10
        )
        kjt_storage = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=20,
        )

        self.assertGreater(kjt_storage.hbm, 0)

    def test_get_kjt_storage_custom_multiplier(self) -> None:
        """Test _get_kjt_storage with custom multiplier."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cuda")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            model, sharders, batch_size=10
        )

        kjt_storage_default = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=20,
        )

        kjt_storage_half = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=10,
        )

        self.assertEqual(kjt_storage_default.hbm, kjt_storage_half.hbm * 2)


class TestFixedPercentageStorageReservation(unittest.TestCase):
    def test_fixed_percentage_reserves_kjt_storage(self) -> None:
        """Test that FixedPercentageStorageReservation computes and saves _kjt_storage."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cuda")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        fixed_storage_reservation = FixedPercentageStorageReservation(percentage=0.25)

        self.assertIsNone(fixed_storage_reservation._kjt_storage)

        fixed_storage_reservation.reserve(
            topology=topology,
            batch_size=10,
            module=model,
            sharders=sharders,
        )

        self.assertIsNotNone(fixed_storage_reservation._kjt_storage)
        self.assertGreater(fixed_storage_reservation._kjt_storage.hbm, 0)
        self.assertEqual(fixed_storage_reservation._kjt_storage.ddr, 0)

    def test_fixed_percentage_reserves_kjt_storage_cpu(self) -> None:
        """Test that FixedPercentageStorageReservation saves _kjt_storage for CPU."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(world_size=2, compute_device="cpu")
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        fixed_storage_reservation = FixedPercentageStorageReservation(percentage=0.25)

        fixed_storage_reservation.reserve(
            topology=topology,
            batch_size=10,
            module=model,
            sharders=sharders,
        )

        self.assertIsNotNone(fixed_storage_reservation._kjt_storage)
        self.assertEqual(fixed_storage_reservation._kjt_storage.hbm, 0)
        self.assertGreater(fixed_storage_reservation._kjt_storage.ddr, 0)

    def test_fixed_percentage_kjt_storage_not_reserved_from_topology(self) -> None:
        """Test that _kjt_storage is saved but NOT reserved from the topology."""
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]

        ebc = EmbeddingBagCollection(tables)
        model = TestModel(shardable_sparse=ebc)
        topology = Topology(
            world_size=2, compute_device="cuda", hbm_cap=10 * 1024 * 1024 * 1024
        )
        sharders = cast(
            List[ModuleSharder[nn.Module]], [EmbeddingBagCollectionSharder()]
        )

        fixed_storage_reservation = FixedPercentageStorageReservation(percentage=0.25)

        reserved_topology = fixed_storage_reservation.reserve(
            topology=topology,
            batch_size=10,
            module=model,
            sharders=sharders,
        )

        expected_hbm = int((1 - 0.25) * topology.devices[0].storage.hbm)
        self.assertEqual(reserved_topology.devices[0].storage.hbm, expected_hbm)
