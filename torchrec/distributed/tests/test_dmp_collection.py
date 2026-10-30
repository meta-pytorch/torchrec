#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DTensor
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    table_wise,
)
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    DMPCollectionConfig,
    DMPCollectionContext,
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ParameterSharding,
    ShardingPlan,
    ShardingStrategy,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class MockModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.linear(x)


class MockSyncableEmbeddingKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._is_triton_tbe = True
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.optimizer_state = torch.ones(2)

    def split_embedding_weights(self) -> list[torch.Tensor]:
        return [self.weight]

    def get_optimizer_state(self) -> list[dict[str, torch.Tensor]]:
        return [{"sum": self.optimizer_state}]


class MockLookupContainer(nn.Module):
    def __init__(self, kernel: MockSyncableEmbeddingKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self._lookups = [kernel]


class MockTwoEmbeddingModel(nn.Module):
    def __init__(
        self,
        ebc1_kernel: MockSyncableEmbeddingKernel,
        ebc2_kernel: MockSyncableEmbeddingKernel,
    ) -> None:
        super().__init__()
        self.ebc1 = MockLookupContainer(ebc1_kernel)
        self.ebc2 = MockLookupContainer(ebc2_kernel)


class MockChunkAwareLookupContainer(MockLookupContainer):
    def __init__(self, kernel: MockSyncableEmbeddingKernel) -> None:
        super().__init__(kernel)
        self.sync_weight_chunks = [torch.ones(4), torch.ones(2)]

    def get_sync_weight_tensors(self) -> list[torch.Tensor]:
        return self.sync_weight_chunks


class MockExactSgdEmbeddingKernel(MockSyncableEmbeddingKernel):
    def get_optimizer_state(self) -> list[dict[str, torch.Tensor]]:
        return [{}]


class MockCounterOptimizerEmbeddingKernel(MockSyncableEmbeddingKernel):
    def __init__(self) -> None:
        super().__init__()
        self.second_optimizer_state = torch.ones(3)
        self.prev_iter = torch.ones(2, dtype=torch.int64)
        self.shared_iter = torch.tensor(7, dtype=torch.int64)

    def get_optimizer_state(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "sum": self.optimizer_state,
                "prev_iter": self.prev_iter,
                "iter": self.shared_iter,
            },
            {
                "sum": self.second_optimizer_state,
                "iter": self.shared_iter,
            },
        ]


class MockManagedCollisionKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._table_to_tbe_and_index = {"table_0": (MagicMock(), 0)}


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
        self.assertEqual(context.weights_by_dtype, {})
        self.assertEqual(context.optimizer_tensors_by_dtype, {})

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
        self.assertNotIn("weights_by_dtype=", repr_str)
        self.assertNotIn("optimizer_tensors_by_dtype=", repr_str)
        self.assertNotIn("sharded_module=", repr_str)


class TestDMPCollectionAllReduceHook(unittest.TestCase):
    def _dmp(self) -> DMPCollection:
        dmp = DMPCollection.__new__(DMPCollection)
        dmp._custom_all_reduce = None
        dmp._all_reduce_hook = None
        return dmp

    def test_set_hook_receives_process_group_and_tensors(self) -> None:
        dmp = self._dmp()
        process_group = MagicMock(spec=dist.ProcessGroup)
        context = MagicMock(spec=DMPCollectionContext)
        context.replica_pg = process_group
        tensors = [torch.ones(2)]
        hook = MagicMock()

        dmp.set_all_reduce_hook(hook)
        dmp._allreduce_tensors(
            context,
            {torch.float32: tensors},
            "test_all_reduce",
        )

        hook.assert_called_once_with(process_group, tensors)

    def test_set_hook_overrides_legacy_constructor_hook(self) -> None:
        dmp = self._dmp()
        legacy_hook = MagicMock()
        setter_hook = MagicMock()
        process_group = MagicMock(spec=dist.ProcessGroup)
        context = MagicMock(spec=DMPCollectionContext)
        context.replica_pg = process_group
        tensors = [torch.ones(2)]
        dmp._custom_all_reduce = legacy_hook

        dmp.set_all_reduce_hook(setter_hook)
        dmp._allreduce_tensors(
            context,
            {torch.float32: tensors},
            "test_all_reduce",
        )

        legacy_hook.assert_not_called()
        setter_hook.assert_called_once_with(process_group, tensors)


class TestDMPCollectionSyncTensors(unittest.TestCase):
    def _context(
        self,
        sharding_strategy: ShardingStrategy = ShardingStrategy.DEFAULT,
    ) -> DMPCollectionContext:
        return DMPCollectionContext(
            module=MockModule,
            plan=MagicMock(spec=ShardingPlan),
            sharding_group_size=1,
            sharding_strategy=sharding_strategy,
        )

    def _dmp_shell(self, module: nn.Module) -> DMPCollection:
        dmp = DMPCollection.__new__(DMPCollection)
        nn.Module.__init__(dmp)
        dmp._all_reduce_hook = None
        dmp._dmp_wrapped_module = module
        return dmp

    def test_module_sync_ownership_traverses_hidden_lookups(self) -> None:
        kernel = MockSyncableEmbeddingKernel()
        container = MockLookupContainer(kernel)

        reachable = DMPCollection._reachable_module_ids(container)

        self.assertIn(id(container), reachable)
        self.assertIn(id(kernel), reachable)

    def test_marked_triton_kernel_discovery_caches_sync_tensors_once(self) -> None:
        kernel = MockSyncableEmbeddingKernel()
        dmp = self._dmp_shell(MockLookupContainer(kernel))
        context = self._context()

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(len(context.modules_to_sync), 1)
        self.assertEqual(context.weights_by_dtype, {torch.float32: [kernel.weight]})
        self.assertEqual(
            context.optimizer_tensors_by_dtype,
            {torch.float32: [kernel.optimizer_state]},
        )

    def test_unmarked_structural_kernel_is_ignored(self) -> None:
        kernel = MockSyncableEmbeddingKernel()
        del kernel._is_triton_tbe
        dmp = self._dmp_shell(MockLookupContainer(kernel))
        context = self._context()

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(context.modules_to_sync, [])
        self.assertEqual(context.weights_by_dtype, {})
        self.assertEqual(context.optimizer_tensors_by_dtype, {})

    def test_fully_sharded_context_preserves_legacy_sync_tensors(self) -> None:
        kernel = MockSyncableEmbeddingKernel()
        dmp = self._dmp_shell(MockLookupContainer(kernel))
        context = self._context(ShardingStrategy.FULLY_SHARDED)

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(len(context.modules_to_sync), 1)
        self.assertEqual(context.weights_by_dtype, {torch.float32: [kernel.weight]})
        self.assertEqual(
            context.optimizer_tensors_by_dtype,
            {torch.float32: [kernel.optimizer_state]},
        )

    def test_fully_sharded_context_caches_chunk_aware_sync_tensors(self) -> None:
        kernel = MockSyncableEmbeddingKernel()
        container = MockChunkAwareLookupContainer(kernel)
        dmp = self._dmp_shell(container)
        context = self._context(ShardingStrategy.FULLY_SHARDED)

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(
            context.weights_by_dtype,
            {torch.float32: container.sync_weight_chunks},
        )
        self.assertEqual(
            context.optimizer_tensors_by_dtype,
            {torch.float32: [kernel.optimizer_state]},
        )

    def test_exact_sgd_empty_optimizer_state_is_not_cached(self) -> None:
        kernel = MockExactSgdEmbeddingKernel()
        dmp = self._dmp_shell(MockLookupContainer(kernel))
        context = self._context(ShardingStrategy.FULLY_SHARDED)

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(context.weights_by_dtype, {torch.float32: [kernel.weight]})
        self.assertEqual(context.optimizer_tensors_by_dtype, {})

    def test_non_sum_optimizer_state_is_not_cached(self) -> None:
        kernel = MockCounterOptimizerEmbeddingKernel()
        dmp = self._dmp_shell(MockLookupContainer(kernel))
        context = self._context(ShardingStrategy.FULLY_SHARDED)

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])

        self.assertEqual(
            context.optimizer_tensors_by_dtype,
            {
                torch.float32: [
                    kernel.optimizer_state,
                    kernel.second_optimizer_state,
                ]
            },
        )
        self.assertNotIn(torch.int64, context.optimizer_tensors_by_dtype)

    def test_fully_sharded_context_retains_managed_collision_tables(self) -> None:
        kernel = MockManagedCollisionKernel()
        dmp = self._dmp_shell(kernel)
        context = self._context(ShardingStrategy.FULLY_SHARDED)
        context.modules_to_sync = [(kernel, kernel)]

        with patch(
            "torchrec.distributed.model_parallel.BaseShardedManagedCollisionEmbeddingCollection",
            MockManagedCollisionKernel,
        ):
            dmp._cache_sync_tensors([context])

        self.assertEqual(context.weights_by_dtype, {})
        self.assertEqual(context.optimizer_tensors_by_dtype, {})
        self.assertEqual(context.hash_zch_modules, [(kernel, "table_0")])

    def test_sync_skips_default_allreduce_for_single_replica(self) -> None:
        dmp = self._dmp_shell(MockModule())
        dmp._custom_all_reduce = None
        dmp._allreduce_tensors = MagicMock()
        context = self._context()
        kernel = MockSyncableEmbeddingKernel()
        context.modules_to_sync = [(kernel, kernel)]
        context.replica_pg = MagicMock()
        context.replica_pg.size.return_value = 1

        dmp._sync(context)

        dmp._allreduce_tensors.assert_not_called()

    def test_module_sync_preserves_fqn_ownership(self) -> None:
        ebc1_kernel = MockSyncableEmbeddingKernel()
        ebc2_kernel = MockSyncableEmbeddingKernel()
        model = MockTwoEmbeddingModel(ebc1_kernel, ebc2_kernel)
        dmp = self._dmp_shell(model)
        context = self._context()
        process_group = MagicMock(spec=dist.ProcessGroup)
        process_group.size.return_value = 2
        context.replica_pg = process_group
        dmp._ctxs = [context]
        dmp._plan = ShardingPlan(
            {
                "ebc1": MagicMock(spec=EmbeddingModuleShardingPlan),
                "ebc2": MagicMock(spec=EmbeddingModuleShardingPlan),
            }
        )
        dmp._custom_all_reduce = None
        dmp._pending_module_syncs = {}
        dmp._restore_stashed_sync_tensors = MagicMock()
        weight_work = MagicMock(spec=dist.Work)
        optimizer_work = MagicMock(spec=dist.Work)
        process_group.allreduce_coalesced.side_effect = [
            weight_work,
            optimizer_work,
        ]

        dmp._group_sharded_modules([context])
        dmp._cache_sync_tensors([context])
        dmp._build_module_sync_units()

        self.assertEqual(dmp.module_sync_fqns, ("ebc1", "ebc2"))
        dmp.start_module_sync("ebc1")
        calls = process_group.allreduce_coalesced.call_args_list
        self.assertEqual(calls[0].args[0], [ebc1_kernel.weight])
        self.assertEqual(calls[1].args[0], [ebc1_kernel.optimizer_state])

        dmp.wait_all_module_syncs()
        weight_work.wait.assert_called_once_with()
        optimizer_work.wait.assert_called_once_with()
        self.assertNotIn("ebc1", dmp._pending_module_syncs)

    def test_sync_skips_default_allreduce_for_empty_modules(self) -> None:
        dmp = self._dmp_shell(MockModule())
        dmp._custom_all_reduce = None
        dmp._restore_stashed_sync_tensors = MagicMock()
        dmp._allreduce_tensors = MagicMock()
        context = self._context()
        context.replica_pg = MagicMock()
        context.replica_pg.size.return_value = 1

        dmp._sync(context)

        dmp._restore_stashed_sync_tensors.assert_not_called()
        dmp._allreduce_tensors.assert_not_called()

    def test_sync_keeps_default_allreduce_for_non_triton_module(self) -> None:
        dmp = self._dmp_shell(MockModule())
        dmp._custom_all_reduce = None
        dmp._restore_stashed_sync_tensors = MagicMock()
        dmp._allreduce_tensors = MagicMock()
        context = self._context()
        module = MockModule()
        context.modules_to_sync = [(module, module)]
        context.replica_pg = MagicMock()
        context.replica_pg.size.return_value = 1

        dmp._sync(context)

        dmp._allreduce_tensors.assert_called_once()


class TestShardingStrategy(unittest.TestCase):

    def test_strategy_values(self) -> None:
        self.assertEqual(ShardingStrategy.DEFAULT.value, "default")
        self.assertEqual(ShardingStrategy.PER_MODULE.value, "per_module")
        self.assertEqual(ShardingStrategy.FULLY_SHARDED.value, "fully_sharded")


class TestEnsureReduceScatterComplete(unittest.TestCase):
    """Tests for the ensure_reduce_scatter_complete() method."""

    def test_ensure_reduce_scatter_complete_with_no_awaitable(self) -> None:
        """Test that ensure_reduce_scatter_complete is a no-op when _rs_awaitable is None."""
        mock_module = MagicMock()
        mock_module._rs_awaitable = None

        if mock_module._rs_awaitable is not None:
            mock_module._rs_awaitable.wait()
            mock_module._rs_awaitable = None

        self.assertIsNone(mock_module._rs_awaitable)

    def test_ensure_reduce_scatter_complete_with_awaitable(self) -> None:
        """Test that ensure_reduce_scatter_complete waits and clears the awaitable."""
        mock_awaitable = MagicMock()
        mock_module = MagicMock()
        mock_module._rs_awaitable = mock_awaitable

        if mock_module._rs_awaitable is not None:
            mock_module._rs_awaitable.wait()
            mock_module._rs_awaitable = None

        mock_awaitable.wait.assert_called_once()
        self.assertIsNone(mock_module._rs_awaitable)

    def test_dmp_collection_ensure_reduce_scatter_complete_iterates_modules(
        self,
    ) -> None:
        mock_child1 = MagicMock()
        mock_child1.ensure_reduce_scatter_complete = MagicMock()

        mock_child2 = MagicMock()
        mock_child2.ensure_reduce_scatter_complete = MagicMock()

        mock_child3 = MagicMock(spec=["forward"])

        mock_dmp = MagicMock()
        mock_dmp.modules = MagicMock(
            return_value=[mock_dmp, mock_child1, mock_child2, mock_child3]
        )

        for module in mock_dmp.modules():
            if (
                hasattr(module, "ensure_reduce_scatter_complete")
                and module is not mock_dmp
            ):
                module.ensure_reduce_scatter_complete()

        mock_child1.ensure_reduce_scatter_complete.assert_called_once()
        mock_child2.ensure_reduce_scatter_complete.assert_called_once()

    def test_ensure_reduce_scatter_complete_is_idempotent(self) -> None:
        """Test that calling ensure_reduce_scatter_complete multiple times is safe."""
        mock_awaitable = MagicMock()
        mock_module = MagicMock()
        mock_module._rs_awaitable = mock_awaitable

        if mock_module._rs_awaitable is not None:
            mock_module._rs_awaitable.wait()
            mock_module._rs_awaitable = None

        mock_awaitable.wait.assert_called_once()

        if mock_module._rs_awaitable is not None:
            mock_module._rs_awaitable.wait()
            mock_module._rs_awaitable = None

        mock_awaitable.wait.assert_called_once()


def _sharding_plan(device_str: str) -> ShardingPlan:
    return ShardingPlan(
        {
            "ebc": EmbeddingModuleShardingPlan(
                {
                    "table_0": ParameterSharding(
                        sharding_type="column_wise",
                        compute_kernel="dense",
                        ranks=[0, 1],
                        sharding_spec=EnumerableShardingSpec(
                            shards=[
                                ShardMetadata(
                                    shard_offsets=[0, 0],
                                    shard_sizes=[64, 32],
                                    placement=f"rank:0/{device_str}",
                                ),
                                ShardMetadata(
                                    shard_offsets=[0, 32],
                                    shard_sizes=[64, 32],
                                    placement=f"rank:1/{device_str}",
                                ),
                            ]
                        ),
                    )
                }
            )
        }
    )


class TestDMPCollectionRemapPlacement(unittest.TestCase):
    """Device index in the placements `_remap_sharding_plan` rewrites."""

    def _dmp(self, device: torch.device) -> DMPCollection:
        dmp = DMPCollection.__new__(DMPCollection)
        dmp._device = device
        return dmp

    def _remap(self, device_str: str) -> ShardingPlan:
        # rank 1 with step 2 puts group_start at 1, so both shards remap onto odd
        # global ranks, where `rank % local_size` is 1 rather than 0.
        plan = _sharding_plan(device_str)
        with patch(
            "torchrec.distributed.model_parallel.get_local_size", return_value=2
        ):
            self._dmp(torch.device(device_str))._remap_sharding_plan(
                plan=plan, rank=1, step=2, sharding_group_size=2
            )
        return plan

    def _table(self, plan: ShardingPlan) -> ParameterSharding:
        return cast(EmbeddingModuleShardingPlan, plan.plan["ebc"])["table_0"]

    def _placements(self, plan: ShardingPlan) -> list[_remote_device]:
        spec = self._table(plan).sharding_spec
        assert isinstance(spec, EnumerableShardingSpec)
        placements = []
        for shard in spec.shards:
            assert shard.placement is not None
            placements.append(shard.placement)
        return placements

    def _devices(self, plan: ShardingPlan) -> list[torch.device]:
        return [p.device() for p in self._placements(plan)]

    def _ranks(self, plan: ShardingPlan) -> list[int | None]:
        return [p.rank() for p in self._placements(plan)]

    def test_cpu_placement_carries_no_device_index(self) -> None:
        plan = self._remap("cpu")

        # Shard.__post_init__ compares placement.device() against the local
        # tensor's device with `!=`, and torch.device("cpu:0") is not equal to
        # torch.device("cpu"), so any index here fails every shard built from
        # this plan at ShardedTensor construction.
        self.assertEqual(
            self._devices(plan), [torch.device("cpu"), torch.device("cpu")]
        )
        self.assertEqual(self._ranks(plan), [1, 3])

    def test_accelerator_placements_keep_their_device_index(self) -> None:
        for device_str in ("cuda", "mtia"):
            with self.subTest(device=device_str):
                plan = self._remap(device_str)

                self.assertEqual(
                    self._devices(plan),
                    [torch.device(device_str, 1), torch.device(device_str, 1)],
                )
                self.assertEqual(self._ranks(plan), [1, 3])

    def test_ranks_are_remapped_regardless_of_device(self) -> None:
        for device_str in ("cpu", "cuda"):
            with self.subTest(device=device_str):
                plan = self._remap(device_str)
                self.assertEqual(self._table(plan).ranks, [1, 3])


def _test_cpu_construction(
    rank: int,
    world_size: int,
    sharding_group_size: int,
) -> None:
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo") as ctx:
        assert ctx.pg is not None
        # Not ctx.device: it resolves to CUDA whenever the host has GPUs, which is
        # precisely the CPU-job-on-a-GPU-host case that must keep working.
        dmp = DMPCollection(
            module=MockModule(),
            device=torch.device("cpu"),
            plan=ShardingPlan({}),
            world_size=world_size,
            sharding_group_size=sharding_group_size,
            global_pg=ctx.pg,
            init_parameters=False,
            init_data_parallel=False,
        )

        mesh = dmp.device_mesh
        assert mesh.device_type == "cpu", f"expected a cpu mesh, got {mesh.device_type}"
        expected_shape = (world_size // sharding_group_size, sharding_group_size)
        assert (
            tuple(mesh.mesh.shape) == expected_shape
        ), f"expected mesh {expected_shape}, got {tuple(mesh.mesh.shape)}"


class _TableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=["f0"],
                )
            ],
            device=torch.device("meta"),
        )


def _test_cpu_sharded_table_sync(rank: int, world_size: int) -> None:
    with MultiProcessContext(rank=rank, world_size=world_size, backend="gloo") as ctx:
        assert ctx.pg is not None
        model = _TableModel()
        # A plan for a one-rank sharding group; DMPCollection remaps it onto
        # each replica's rank.
        plan = ShardingPlan(
            {
                "ebc": construct_module_sharding_plan(
                    model.ebc,
                    {
                        "table_0": table_wise(
                            rank=0, compute_kernel=EmbeddingComputeKernel.FUSED.value
                        )
                    },
                    local_size=1,
                    world_size=1,
                    device_type="cpu",
                )
            }
        )
        dmp = DMPCollection(
            module=model,
            device=torch.device("cpu"),
            plan=plan,
            world_size=world_size,
            sharding_group_size=1,
            global_pg=ctx.pg,
            init_data_parallel=False,
        )

        weight = dmp.state_dict()["ebc.embedding_bags.table_0.weight"]
        assert isinstance(weight, DTensor), f"got {type(weight)}"
        local = weight.to_local()
        assert isinstance(local, LocalShardsWrapper), f"got {type(local)}"
        shard = local.local_shards()[0]
        assert shard.device == torch.device("cpu"), f"got {shard.device}"

        # Each replica holds a full copy of the table, and sync() averages the
        # copies over the replica group.
        shard.fill_(float(rank))
        dmp.sync()
        expected = sum(range(world_size)) / world_size
        assert torch.all(shard == expected), f"expected {expected}, got {shard}"


class TestDMPCollectionCPUConstruction(MultiProcessTestBase):
    """DMPCollection must build and sync on a CPU topology, not just CUDA and MTIA."""

    def test_cpu_two_replicas(self) -> None:
        self._run_multi_process_test(
            callable=_test_cpu_construction,
            world_size=2,
            sharding_group_size=1,
        )

    def test_cpu_single_replica(self) -> None:
        # num_parallel_worlds=1 in APF, so sharding_group_size == world_size and the
        # replica dimension collapses to 1. This is the shape Ads CPU transfer
        # learning runs, and the rank remap degenerates to identity.
        self._run_multi_process_test(
            callable=_test_cpu_construction,
            world_size=2,
            sharding_group_size=2,
        )

    def test_cpu_sharded_table_syncs_across_replicas(self) -> None:
        self._run_multi_process_test(
            callable=_test_cpu_sharded_table_sync,
            world_size=2,
        )


if __name__ == "__main__":
    unittest.main()
