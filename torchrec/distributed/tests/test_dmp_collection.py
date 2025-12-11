#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from dataclasses import fields, is_dataclass
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torchrec.distributed.types import (
    DMPCollectionConfig,
    DMPCollectionContext,
    ShardingPlan,
    ShardingStrategy,
)


class MockModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.linear(x)


class TestDMPCollectionConfig(unittest.TestCase):

    def test_is_dataclass(self) -> None:
        self.assertTrue(is_dataclass(DMPCollectionConfig))

    def test_dataclass_fields(self) -> None:
        field_names = {f.name for f in fields(DMPCollectionConfig)}
        expected_fields = {
            "module",
            "plan",
            "sharding_group_size",
            "node_group_size",
            "use_inter_host_allreduce",
            "sharding_strategy",
        }
        self.assertEqual(field_names, expected_fields)

    def test_construction_with_required_args(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        config = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        self.assertEqual(config.module, MockModule)
        self.assertEqual(config.plan, mock_plan)
        self.assertEqual(config.sharding_group_size, 4)
        self.assertIsNone(config.node_group_size)
        self.assertFalse(config.use_inter_host_allreduce)
        self.assertEqual(config.sharding_strategy, ShardingStrategy.DEFAULT)

    def test_construction_with_all_args(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        config = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=8,
            node_group_size=4,
            use_inter_host_allreduce=True,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

        self.assertEqual(config.module, MockModule)
        self.assertEqual(config.sharding_group_size, 8)
        self.assertEqual(config.node_group_size, 4)
        self.assertTrue(config.use_inter_host_allreduce)
        self.assertEqual(config.sharding_strategy, ShardingStrategy.FULLY_SHARDED)

    def test_repr_excludes_plan(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        config = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        repr_str = repr(config)
        self.assertIn("MockModule", repr_str)
        self.assertIn("sharding_group_size=4", repr_str)
        self.assertNotIn("plan=", repr_str)

    def test_equality(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        config1 = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )
        config2 = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        self.assertEqual(config1, config2)


class TestDMPCollectionContext(unittest.TestCase):

    def test_is_dataclass(self) -> None:
        self.assertTrue(is_dataclass(DMPCollectionContext))

    def test_inherits_from_config(self) -> None:
        self.assertTrue(issubclass(DMPCollectionContext, DMPCollectionConfig))

    def test_dataclass_fields(self) -> None:
        field_names = {f.name for f in fields(DMPCollectionContext)}
        expected_fields = {
            # Inherited from DMPCollectionConfig
            "module",
            "plan",
            "sharding_group_size",
            "node_group_size",
            "use_inter_host_allreduce",
            "sharding_strategy",
            # New in DMPCollectionContext
            "device_mesh",
            "sharding_pg",
            "replica_pg",
            "modules_to_sync",
            "sharded_module",
        }
        self.assertEqual(field_names, expected_fields)

    def test_construction_preserves_parent_signature(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
            node_group_size=2,
            use_inter_host_allreduce=True,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

        self.assertEqual(context.module, MockModule)
        self.assertEqual(context.sharding_group_size, 4)
        self.assertEqual(context.node_group_size, 2)
        self.assertTrue(context.use_inter_host_allreduce)
        self.assertEqual(context.sharding_strategy, ShardingStrategy.FULLY_SHARDED)

    def test_init_false_fields_not_in_constructor(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        with self.assertRaises(TypeError):
            DMPCollectionContext(
                module=MockModule,
                plan=mock_plan,
                sharding_group_size=4,
                device_mesh=MagicMock(),  # Should not be accepted
            )

    def test_init_false_fields_have_defaults(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        self.assertEqual(context.modules_to_sync, [])
        self.assertIsNone(context.sharded_module)

    def test_init_false_fields_can_be_set_after_construction(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)
        mock_device_mesh = MagicMock()
        mock_pg = MagicMock()

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        context.device_mesh = mock_device_mesh
        context.sharding_pg = mock_pg
        context.replica_pg = mock_pg

        self.assertEqual(context.device_mesh, mock_device_mesh)
        self.assertEqual(context.sharding_pg, mock_pg)
        self.assertEqual(context.replica_pg, mock_pg)

    def test_repr_excludes_runtime_fields(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        repr_str = repr(context)
        self.assertNotIn("device_mesh=", repr_str)
        self.assertNotIn("sharding_pg=", repr_str)
        self.assertNotIn("replica_pg=", repr_str)
        self.assertNotIn("modules_to_sync=", repr_str)
        self.assertNotIn("sharded_module=", repr_str)


class TestShardingStrategy(unittest.TestCase):

    def test_strategy_values(self) -> None:
        self.assertEqual(ShardingStrategy.DEFAULT.value, "default")
        self.assertEqual(ShardingStrategy.PER_MODULE.value, "per_module")
        self.assertEqual(ShardingStrategy.FULLY_SHARDED.value, "fully_sharded")


if __name__ == "__main__":
    unittest.main()
