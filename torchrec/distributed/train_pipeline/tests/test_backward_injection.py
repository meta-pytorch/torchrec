#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Callable, cast, List
from unittest.mock import patch

import torch
from hypothesis import given, settings, strategies as st
from torch import nn, optim
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.comm_ops import All2All_Pooled_Req, ReduceScatterBase_Req
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.emb_sharder import TestEBCSharder
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.train_pipeline.backward_injection import (
    BackwardHookRegistry,
    InjectionSite,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

# TestCase instance for assertions in test runner functions
tc = unittest.TestCase()
tc.maxDiff = None


class BackwardHookRegistryTest(unittest.TestCase):
    """Unit tests for BackwardHookRegistry."""

    def test_add_hook_and_work(self) -> None:
        """Verifies add_hook stores hook and work returns it."""
        registry = BackwardHookRegistry()
        site = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE)

        def work_fn(_p: TrainPipelineSparseDist) -> None:
            pass

        registry.add_hook(site, work_fn)  # pyrefly: ignore[bad-argument-type]

        self.assertEqual(registry.work(site), [work_fn])

    def test_multiple_hooks_same_site(self) -> None:
        """Verifies multiple hooks at same site are returned in order."""
        registry = BackwardHookRegistry()
        site = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE)

        def work_fn_1(_p: TrainPipelineSparseDist) -> None:
            pass

        def work_fn_2(_p: TrainPipelineSparseDist) -> None:
            pass

        registry.add_hook(site, work_fn_1)  # pyrefly: ignore[bad-argument-type]
        registry.add_hook(site, work_fn_2)  # pyrefly: ignore[bad-argument-type]

        self.assertEqual(registry.work(site), [work_fn_1, work_fn_2])

    def test_different_sites_isolated(self) -> None:
        """Verifies hooks at different sites are isolated."""
        registry = BackwardHookRegistry()
        site_1 = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE)
        site_2 = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.ROW_WISE)

        def work_fn_1(_p: TrainPipelineSparseDist) -> None:
            pass

        def work_fn_2(_p: TrainPipelineSparseDist) -> None:
            pass

        registry.add_hook(site_1, work_fn_1)  # pyrefly: ignore[bad-argument-type]
        registry.add_hook(site_2, work_fn_2)  # pyrefly: ignore[bad-argument-type]

        self.assertEqual(registry.work(site_1), [work_fn_1])
        self.assertEqual(registry.work(site_2), [work_fn_2])

    def test_work_unknown_site_returns_empty(self) -> None:
        """Verifies work returns empty list for unregistered site."""
        registry = BackwardHookRegistry()
        site = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE)

        self.assertEqual(registry.work(site), [])

    def test_add_same_hook_to_multiple_sites(self) -> None:
        """Verifies same hook can be added to multiple sites."""
        registry = BackwardHookRegistry()
        site_1 = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE)
        site_2 = InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.ROW_WISE)

        def work_fn(_p: TrainPipelineSparseDist) -> None:
            pass

        registry.add_hook(site_1, work_fn)  # pyrefly: ignore[bad-argument-type]
        registry.add_hook(site_2, work_fn)  # pyrefly: ignore[bad-argument-type]

        self.assertEqual(registry.work(site_1), [work_fn])
        self.assertEqual(registry.work(site_2), [work_fn])

    def test_hooks_dict_empty_by_default(self) -> None:
        """Verifies hooks dict is empty when registry is created."""
        registry = BackwardHookRegistry()
        self.assertEqual(registry.hooks, {})


def _create_pipeline(
    ctx: MultiProcessContext,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    sharding_type: str = ShardingType.TABLE_WISE.value,
) -> TrainPipelineSparseDist:
    """Helper to create a TrainPipelineSparseDist for testing."""
    unsharded_model = TestSparseNN(
        tables=tables,
        weighted_tables=weighted_tables,
        dense_device=ctx.device,
        sparse_device=torch.device("meta"),
    )

    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=EmbeddingComputeKernel.FUSED.value,
    )

    sharded_model = DistributedModelParallel(
        module=copy.deepcopy(unsharded_model),
        env=ShardingEnv.from_process_group(
            ctx.pg  # pyrefly: ignore[bad-argument-type]
        ),
        device=ctx.device,
        sharders=[cast(ModuleSharder[nn.Module], sharder)],
    )

    optimizer = optim.SGD(sharded_model.parameters(), lr=0.1)

    return TrainPipelineSparseDist(
        model=sharded_model,
        optimizer=optimizer,
        device=ctx.device,
    )


def _run_backward_hook_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[ModelInput],
    sharding_type: str,
    site_fqn: str,
    backend: str = "nccl",
    local_size: int | None = None,
    num_hooks: int = 1,
) -> None:
    """Test runner for backward hook execution with multi-GPU setup."""
    operation_order: List[str] = []

    # Different sharding types use different backward comm ops
    if sharding_type in [ShardingType.TABLE_WISE.value, ShardingType.COLUMN_WISE.value]:
        original_backward = All2All_Pooled_Req.backward
        backward_cls = All2All_Pooled_Req
    else:  # ROW_WISE uses reduce-scatter
        original_backward = ReduceScatterBase_Req.backward
        backward_cls = ReduceScatterBase_Req

    def wrapper_func(*args, **kwargs):
        operation_order.append("comm_op")
        return original_backward(*args, **kwargs)

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        tc.assertIsNotNone(ctx.pg)
        pipeline = _create_pipeline(ctx, tables, weighted_tables, sharding_type)

        # Register hook(s)
        def make_work_fn(idx: int) -> Callable[[TrainPipelineSparseDist], None]:
            def work_fn(_p: TrainPipelineSparseDist) -> None:
                operation_order.append(f"hook_{idx}")

            return work_fn

        for i in range(num_hooks):
            pipeline.register_backward_hook(
                site=InjectionSite(
                    fqn=site_fqn, sharding_type=ShardingType(sharding_type)
                ),
                work=make_work_fn(i),  # pyrefly: ignore[bad-argument-type]
            )

        dataloader = iter(data)
        with patch.object(backward_cls, "backward", side_effect=wrapper_func):
            pipeline.progress(dataloader)

        # Verify hooks were executed in order
        hook_entries = [f"hook_{i}" for i in range(num_hooks)]
        if site_fqn == "sparse.ebc":
            expected = ["comm_op"] + hook_entries + ["comm_op"]
        elif site_fqn == "sparse.weighted_ebc":
            expected = hook_entries + ["comm_op", "comm_op"]
        else:
            raise ValueError(f"Unknown site_fqn: {site_fqn}")

        tc.assertEqual(operation_order, expected)


def _run_nonexistent_site_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[ModelInput],
    backend: str = "nccl",
    local_size: int | None = None,
) -> None:
    """Test runner to verify warning is logged for non-existent site."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        tc.assertIsNotNone(ctx.pg)
        pipeline = _create_pipeline(ctx, tables, weighted_tables)

        hook_executed: List[bool] = []
        pipeline.register_backward_hook(
            site=InjectionSite(
                fqn="nonexistent.module", sharding_type=ShardingType.TABLE_WISE
            ),
            work=lambda _p: hook_executed.append(True),
        )

        dataloader = iter(data)
        with tc.assertLogs(
            "torchrec.distributed.train_pipeline.backward_injection", level="WARNING"
        ) as cm:
            pipeline.progress(dataloader)

        tc.assertTrue(
            any("nonexistent.module" in msg for msg in cm.output),
            f"Expected warning about non-existent site. Log output: {cm.output}",
        )
        tc.assertEqual(
            len(hook_executed), 0, "Hook should not execute for non-existent site"
        )


def _run_repeated_calls_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[ModelInput],
    backend: str = "nccl",
    local_size: int | None = None,
    num_progress_calls: int = 3,
) -> None:
    """Test runner to verify hooks work across multiple pipeline.progress() calls."""
    hook_call_count: List[int] = [0]

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        tc.assertIsNotNone(ctx.pg)
        pipeline = _create_pipeline(ctx, tables, weighted_tables)

        pipeline.register_backward_hook(
            site=InjectionSite(fqn="sparse.ebc", sharding_type=ShardingType.TABLE_WISE),
            work=lambda _p: hook_call_count.__setitem__(0, hook_call_count[0] + 1),
        )

        dataloader = iter(data)
        for _ in range(num_progress_calls):
            pipeline.progress(dataloader)

        tc.assertEqual(
            hook_call_count[0],
            num_progress_calls,
            f"Hook should be called {num_progress_calls} times, "
            f"but was called {hook_call_count[0]} times",
        )


class RegisterHooksTest(MultiProcessTestBase):
    """Integration tests for register_hooks with real pipeline."""

    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        num_weighted_features = 2
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

    def _generate_data(
        self,
        num_batches: int = 5,
        batch_size: int = 1,
    ) -> List[ModelInput]:
        return [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=batch_size,
                num_float_features=10,
            )
            for _ in range(num_batches)
        ]

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    @settings(max_examples=6, deadline=None)
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        site_fqn=st.sampled_from(["sparse.ebc", "sparse.weighted_ebc"]),
    )
    def test_backward_hook_executed_during_backward(
        self,
        sharding_type: str,
        site_fqn: str,
    ) -> None:
        """Verifies registered hooks are executed during backward pass."""
        data = self._generate_data(num_batches=5, batch_size=2)
        self._run_multi_process_test(
            callable=_run_backward_hook_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=sharding_type,
            site_fqn=site_fqn,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_multiple_hooks_executed_in_order(self) -> None:
        """Verifies multiple hooks at the same site are executed in registration order."""
        data = self._generate_data(num_batches=5, batch_size=2)
        self._run_multi_process_test(
            callable=_run_backward_hook_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=ShardingType.TABLE_WISE.value,
            site_fqn="sparse.ebc",
            num_hooks=3,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_hook_on_nonexistent_site_logs_warning(self) -> None:
        """Verifies warning is logged when hook registered for site that doesn't exist."""
        data = self._generate_data(num_batches=5, batch_size=2)
        self._run_multi_process_test(
            callable=_run_nonexistent_site_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least 2 GPUs",
    )
    def test_hooks_work_across_repeated_progress_calls(self) -> None:
        """Verifies hooks continue to work correctly across multiple pipeline.progress() calls."""
        data = self._generate_data(num_batches=10, batch_size=2)
        self._run_multi_process_test(
            callable=_run_repeated_calls_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            num_progress_calls=5,
        )
