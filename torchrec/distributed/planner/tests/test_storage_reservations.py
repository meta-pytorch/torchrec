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
from torchrec.distributed.planner.storage_reservations import (
    _get_module_size,
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
            # pyre-ignore
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
            # pyre-ignore
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
            # pyre-ignore
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
            # pyre-ignore
            heuristical_storage_reservation._dense_storage.hbm,
        )
