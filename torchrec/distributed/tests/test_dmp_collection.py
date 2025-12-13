#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
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

    def test_inequality(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        config1 = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )
        config2 = DMPCollectionConfig(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=8,  # Different sharding_group_size
        )

        self.assertNotEqual(config1, config2)


class TestDMPCollectionContext(unittest.TestCase):

    def test_inherits_from_config(self) -> None:
        self.assertTrue(issubclass(DMPCollectionContext, DMPCollectionConfig))

    def test_construction_with_none_module(self) -> None:
        """Test that module=None is allowed for default context in DMPCollection."""
        mock_plan = MagicMock(spec=ShardingPlan)

        # This is used in model_parallel.py for default context
        context = DMPCollectionContext(
            module=None,  # type: ignore[arg-type]
            plan=mock_plan,
            sharding_group_size=4,
        )

        self.assertIsNone(context.module)
        self.assertEqual(context.sharding_group_size, 4)

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

    def test_runtime_fields_have_defaults(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
        )

        self.assertEqual(context.modules_to_sync, [])
        self.assertIsNone(context.sharded_module)
        self.assertIsNone(context.device_mesh)
        self.assertIsNone(context.sharding_pg)
        self.assertIsNone(context.replica_pg)

    def test_modules_to_sync_can_be_passed_to_constructor(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)
        mock_module1 = MagicMock(spec=nn.Module)
        mock_module2 = MagicMock(spec=nn.Module)
        modules_to_sync = [(mock_module1, mock_module2)]

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
            modules_to_sync=modules_to_sync,
        )

        self.assertEqual(context.modules_to_sync, modules_to_sync)

    def test_sharded_module_can_be_passed_to_constructor(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)
        mock_sharded = MagicMock(spec=nn.Module)

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
            sharded_module=mock_sharded,
        )

        self.assertEqual(context.sharded_module, mock_sharded)

    def test_all_runtime_fields_can_be_passed_to_constructor(self) -> None:
        mock_plan = MagicMock(spec=ShardingPlan)
        mock_device_mesh = MagicMock()
        mock_sharding_pg = MagicMock()
        mock_replica_pg = MagicMock()
        mock_sharded = MagicMock(spec=nn.Module)
        modules_to_sync = [(MagicMock(spec=nn.Module), MagicMock(spec=nn.Module))]

        context = DMPCollectionContext(
            module=MockModule,
            plan=mock_plan,
            sharding_group_size=4,
            modules_to_sync=modules_to_sync,
            sharded_module=mock_sharded,
            device_mesh=mock_device_mesh,
            sharding_pg=mock_sharding_pg,
            replica_pg=mock_replica_pg,
        )

        self.assertEqual(context.modules_to_sync, modules_to_sync)
        self.assertEqual(context.sharded_module, mock_sharded)
        self.assertEqual(context.device_mesh, mock_device_mesh)
        self.assertEqual(context.sharding_pg, mock_sharding_pg)
        self.assertEqual(context.replica_pg, mock_replica_pg)

    def test_runtime_fields_can_be_set_after_construction(self) -> None:
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
