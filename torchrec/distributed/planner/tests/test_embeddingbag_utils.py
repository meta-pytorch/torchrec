#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest

from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class CreateShardingInfoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=4,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=4,
                num_embeddings=4,
            ),
        ]

        self.constraints = {
            "table_0": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LRU,
                    load_factor=0.1,
                    reserved_memory=8.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=True,
                stochastic_rounding=False,
                bounds_check_mode=BoundsCheckMode.IGNORE,
            ),
            "table_1": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.2,
                    reserved_memory=0.0,
                    precision=DataType.FP16,
                ),
                enforce_hbm=True,
                stochastic_rounding=False,
                bounds_check_mode=BoundsCheckMode.NONE,
            ),
        }

        self.model = EmbeddingBagCollection(tables=self.tables)
        self.sharder = EmbeddingBagCollectionSharder()
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=1, compute_device="cpu"),
            constraints=self.constraints,
        )
        self.expected_plan = planner.plan(self.model, [self.sharder])

        self.expected_sharding_infos = (
            ShardedEmbeddingBagCollection.create_grouped_sharding_infos(
                self.model,
                self.expected_plan.get_plan_for_module(""),
                prefix="embedding_bags.",
                fused_params=None,
            )
        )

    def test_create_sharding_infos_by_group_override(self) -> None:
        """
        Test that fused_params from sharders get overridden.
        """

        # with sharder fused params that will get overridden
        sharder_fused_params = {"enforce_hbm": False}
        overriden_sharding_infos = (
            ShardedEmbeddingBagCollection.create_grouped_sharding_infos(
                self.model,
                self.expected_plan.get_plan_for_module(""),
                prefix="embedding_bags.",
                fused_params=sharder_fused_params,
            )
        )
        for sharding_type, overriden_sharding_info in overriden_sharding_infos.items():
            expected_sharding_info = self.expected_sharding_infos[sharding_type]
            for a, b in zip(expected_sharding_info, overriden_sharding_info):
                self.assertEqual(a.fused_params, b.fused_params)

        # with sharder fused params that won't get overridden
        sharder_fused_params = {"ABC": True}
        not_overriden_sharding_infos = (
            ShardedEmbeddingBagCollection.create_grouped_sharding_infos(
                self.model,
                self.expected_plan.get_plan_for_module(""),
                prefix="embedding_bags.",
                fused_params=sharder_fused_params,
            )
        )
        for (
            sharding_type,
            not_overriden_sharding_info,
        ) in not_overriden_sharding_infos.items():
            expected_sharding_info = self.expected_sharding_infos[sharding_type]
            for a, b in zip(expected_sharding_info, not_overriden_sharding_info):
                self.assertNotEqual(a.fused_params, b.fused_params)

    def test_create_sharding_infos_by_group_combine(self) -> None:
        """
        Test that fused_params can get info from both sharder and constraints.
        """

        new_constraints = copy.deepcopy(self.constraints)

        # remove two fused_params from constraints
        for _, parameter_constraints in new_constraints.items():
            parameter_constraints.enforce_hbm = None
            parameter_constraints.stochastic_rounding = None

        new_planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=1, compute_device="cpu"),
            constraints=new_constraints,
        )
        new_plan = new_planner.plan(self.model, [self.sharder])

        # provide that two fused params from sharder
        sharder_fused_params = {"enforce_hbm": True, "stochastic_rounding": False}

        combined_sharding_infos = (
            ShardedEmbeddingBagCollection.create_grouped_sharding_infos(
                self.model,
                new_plan.get_plan_for_module(""),
                prefix="embedding_bags.",
                fused_params=sharder_fused_params,
            )
        )

        # directly assertion won't work, since sharding_infos also have parameter_sharding
        for sharding_type, combined_sharding_info in combined_sharding_infos.items():
            expected_sharding_info = self.expected_sharding_infos[sharding_type]
            for a, b in zip(expected_sharding_info, combined_sharding_info):
                self.assertEqual(a.fused_params, b.fused_params)

        # provide that two fused params from sharder wrongly
        sharder_fused_params = {"enforce_hbm": True, "stochastic_rounding": True}
        wrong_combined_sharding_infos = (
            ShardedEmbeddingBagCollection.create_grouped_sharding_infos(
                self.model,
                new_plan.get_plan_for_module(""),
                prefix="embedding_bags.",
                fused_params=sharder_fused_params,
            )
        )
        for (
            sharding_type,
            wrong_combined_sharding_info,
        ) in wrong_combined_sharding_infos.items():
            expected_sharding_info = self.expected_sharding_infos[sharding_type]
            for a, b in zip(expected_sharding_info, wrong_combined_sharding_info):
                self.assertNotEqual(a.fused_params, b.fused_params)
