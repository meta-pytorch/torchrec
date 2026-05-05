#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import cast, List, Optional, Tuple

import torch
from hypothesis import given, settings, strategies as st
from torch import nn
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.emb_sharder import TestEBCSharder
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.train_pipeline.backward_injection import (
    _position_to_index,
    FirstGradTensorFinder,
    InjectionSite,
    InjectionTargetType,
    OutputDistTensorFinder,
    register_backward_hook,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

tc = unittest.TestCase()
tc.maxDiff = None


class SimpleModel(nn.Module):
    """Simple nested model for testing InjectionSite."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_a = nn.Linear(4, 4)
        self.layer_b = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_b(self.layer_a(x))


class InjectionSiteTest(unittest.TestCase):
    """Unit tests for InjectionSite and FirstGradTensorFinder (no GPU/distributed required)."""

    def test_first_grad_tensor_finder_nested(self) -> None:
        finder = FirstGradTensorFinder()
        grad = torch.tensor([1.0], requires_grad=True)
        nested = (torch.tensor([0.0]), {"k": [torch.tensor([0.0]), grad]})
        self.assertIs(finder(None, nested), grad)
        self.assertIsNone(finder(None, torch.tensor([1.0])))

    def test_first_grad_tensor_finder_use_input(self) -> None:
        finder = FirstGradTensorFinder(use_input=True)
        grad = torch.tensor([1.0], requires_grad=True)
        self.assertIs(finder(grad, torch.tensor([0.0])), grad)
        self.assertIsNone(finder(torch.tensor([0.0]), grad))

    def test_first_grad_tensor_finder_kjt_weights(self) -> None:
        """KJT/JT carry the grad-tracking tensor in the optional weights field
        (e.g. the output of PositionWeightedModuleCollection)."""
        finder = FirstGradTensorFinder()

        weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        kjt = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int64),
            weights=weights,
        )
        self.assertIs(finder(None, kjt), weights)

        # Also matches when nested inside containers.
        nested = (torch.tensor([0.0]), {"k": [kjt]})
        self.assertIs(finder(None, nested), weights)

        # JaggedTensor with grad-requiring weights also resolves.
        jt_weights = torch.tensor([0.5, 0.6], requires_grad=True)
        jt = JaggedTensor(
            values=torch.tensor([1, 2], dtype=torch.int64),
            lengths=torch.tensor([1, 1], dtype=torch.int64),
            weights=jt_weights,
        )
        self.assertIs(finder(None, jt), jt_weights)

    def test_first_grad_tensor_finder_kjt_no_weights(self) -> None:
        """KJT/JT with no weights or non-grad weights yield no match."""
        finder = FirstGradTensorFinder()

        kjt_no_weights = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int64),
        )
        self.assertIsNone(finder(None, kjt_no_weights))

        kjt_inert_weights = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.tensor([1, 2, 3], dtype=torch.int64),
            lengths=torch.tensor([2, 1], dtype=torch.int64),
            weights=torch.tensor([1.0, 2.0, 3.0]),  # requires_grad=False
        )
        self.assertIsNone(finder(None, kjt_inert_weights))

    def test_register_hook_nonexistent_raises(self) -> None:
        site = InjectionSite(
            fqn="nonexistent.module",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
        )
        with self.assertRaises(ValueError):
            register_backward_hook(site, SimpleModel(), lambda grad: None)

    def test_position_to_index(self) -> None:
        self.assertEqual(_position_to_index(0.0, 5), 0)
        self.assertEqual(_position_to_index(1.0, 5), 4)
        self.assertEqual(_position_to_index(0.5, 4), 2)
        self.assertEqual(_position_to_index(0.0, 1), 0)
        self.assertEqual(_position_to_index(1.0, 1), 0)
        # out-of-range values are clamped
        self.assertEqual(_position_to_index(-0.5, 5), 0)
        self.assertEqual(_position_to_index(1.5, 5), 4)

    def test_hook_position_selects_parameter(self) -> None:
        """hook_position=0.0 hooks weight, hook_position=1.0 hooks bias."""
        model = SimpleModel()
        grad_shapes: List[torch.Size] = []

        # layer_a is nn.Linear(4, 4): params are [weight(4,4), bias(4)]
        site_first = InjectionSite(
            fqn="layer_a",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
            hook_position=0.0,
        )
        handle = register_backward_hook(
            site_first,
            model,
            lambda grad: grad_shapes.append(grad.shape),
        )
        model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(grad_shapes[-1], torch.Size([4, 4]))
        handle.remove()

        site_last = InjectionSite(
            fqn="layer_a",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
            hook_position=1.0,
        )
        handle = register_backward_hook(
            site_last,
            model,
            lambda grad: grad_shapes.append(grad.shape),
        )
        model.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(grad_shapes[-1], torch.Size([4]))
        handle.remove()

    def test_hook_position_no_trainable_params_raises(self) -> None:
        """PARAM_GRAD on a module with no trainable params raises ValueError."""
        model = SimpleModel()
        # layer_b contains ReLU at index 1 which has no parameters
        site = InjectionSite(
            fqn="layer_b.1",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
            hook_position=0.5,
        )
        with self.assertRaises(ValueError):
            register_backward_hook(site, model, lambda grad: None)

    def test_register_hook_persists_and_removable(self) -> None:
        """Hook fires every iteration; removing it stops firing."""
        model = SimpleModel()
        site = InjectionSite(
            fqn="layer_a",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
        )
        call_count: List[int] = [0]

        handle = register_backward_hook(
            site,
            model,
            lambda grad: call_count.__setitem__(0, call_count[0] + 1),
        )

        for _ in range(3):
            model.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(call_count[0], 3)

        handle.remove()
        model.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        self.assertEqual(call_count[0], 3)


def _create_sharded_model(
    ctx: MultiProcessContext,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    sharding_type: str,
) -> DistributedModelParallel:
    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=EmbeddingComputeKernel.FUSED.value,
    )
    model = TestSparseNN(
        tables=tables,
        weighted_tables=weighted_tables,
        dense_device=ctx.device,
        sparse_device=torch.device("meta"),
    )
    assert ctx.pg is not None
    return DistributedModelParallel(
        module=copy.deepcopy(model),
        env=ShardingEnv.from_process_group(ctx.pg),
        init_data_parallel=False,
        device=ctx.device,
        sharders=[cast(ModuleSharder[nn.Module], sharder)],
    )


def _run_output_dist_backward_hook_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[Tuple[ModelInput, List[ModelInput]]],
    sharding_type: str,
    backend: str = "nccl",
    local_size: Optional[int] = None,
) -> None:
    """register_backward_hook with OutputDistTensorFinder fires during backward."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        dmp = _create_sharded_model(ctx, tables, weighted_tables, sharding_type)

        call_count: List[int] = [0]
        site = InjectionSite(
            fqn="sparse.ebc",
            tensor_finder=OutputDistTensorFinder(
                sharding_type=ShardingType(sharding_type)
            ),
            target_type=InjectionTargetType.ACTIVATION,
        )
        register_backward_hook(
            site,
            dmp.module,
            lambda grad: call_count.__setitem__(0, call_count[0] + 1),
        )

        batch = data[0][1][ctx.rank].to(ctx.device)
        loss, _pred = dmp(batch)
        loss.backward()

        tc.assertEqual(call_count[0], 1, "Backward hook must fire exactly once")


def _run_output_dist_backward_order_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[Tuple[ModelInput, List[ModelInput]]],
    sharding_type: str,
    backend: str = "nccl",
    local_size: Optional[int] = None,
) -> None:
    """Dense backward hook fires before the EBC backward hook.

    The model computes: dense -> sparse(ebc) -> over.  The dummy_tensor
    lives deep inside the EBC's all-to-all op, so its backward hook fires
    after the dense output hook.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        dmp = _create_sharded_model(ctx, tables, weighted_tables, sharding_type)

        order: List[str] = []

        ebc_site = InjectionSite(
            fqn="sparse.ebc",
            tensor_finder=OutputDistTensorFinder(
                sharding_type=ShardingType(sharding_type)
            ),
            target_type=InjectionTargetType.ACTIVATION,
        )
        register_backward_hook(ebc_site, dmp.module, lambda grad: order.append("ebc"))

        dense_site = InjectionSite(
            fqn="dense",
            tensor_finder=FirstGradTensorFinder(),
            target_type=InjectionTargetType.PARAM_GRAD,
        )
        register_backward_hook(
            dense_site, dmp.module, lambda grad: order.append("dense")
        )

        batch = data[0][1][ctx.rank].to(ctx.device)
        loss, _pred = dmp(batch)
        loss.backward()

        tc.assertEqual(order, ["dense", "ebc"])


def _run_output_dist_multiple_hooks_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[Tuple[ModelInput, List[ModelInput]]],
    sharding_type: str,
    backend: str = "nccl",
    local_size: Optional[int] = None,
) -> None:
    """Multiple hooks on the same site all fire in registration order."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        dmp = _create_sharded_model(ctx, tables, weighted_tables, sharding_type)

        order: List[str] = []
        site = InjectionSite(
            fqn="sparse.ebc",
            tensor_finder=OutputDistTensorFinder(
                sharding_type=ShardingType(sharding_type)
            ),
            target_type=InjectionTargetType.ACTIVATION,
        )
        register_backward_hook(site, dmp.module, lambda grad: order.append("hook_0"))
        register_backward_hook(site, dmp.module, lambda grad: order.append("hook_1"))
        register_backward_hook(site, dmp.module, lambda grad: order.append("hook_2"))

        batch = data[0][1][ctx.rank].to(ctx.device)
        loss, _pred = dmp(batch)
        loss.backward()

        tc.assertEqual(order, ["hook_0", "hook_1", "hook_2"])


def _run_output_dist_finder_mismatch_test(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    data: List[Tuple[ModelInput, List[ModelInput]]],
    sharding_type: str,
    mismatched_sharding_type: str,
    backend: str = "nccl",
    local_size: Optional[int] = None,
) -> None:
    """Mismatched sharding type raises RuntimeError during forward."""
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        dmp = _create_sharded_model(ctx, tables, weighted_tables, sharding_type)

        site = InjectionSite(
            fqn="sparse.ebc",
            tensor_finder=OutputDistTensorFinder(
                sharding_type=ShardingType(mismatched_sharding_type)
            ),
            target_type=InjectionTargetType.ACTIVATION,
        )
        register_backward_hook(site, dmp.module, lambda grad: None)

        batch = data[0][1][ctx.rank].to(ctx.device)
        with tc.assertRaises(RuntimeError):
            dmp(batch)


class OutputDistTensorFinderTest(MultiProcessTestBase):
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
    ) -> List[Tuple[ModelInput, List[ModelInput]]]:
        return [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=batch_size,
                world_size=2,
                num_float_features=10,
            )
            for _ in range(num_batches)
        ]

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs for distributed test",
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
    )
    def test_output_dist_backward_hook(self, sharding_type: str) -> None:
        data = self._generate_data()
        self._run_multi_process_test(
            callable=_run_output_dist_backward_hook_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=sharding_type,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs for distributed test",
    )
    def test_output_dist_backward_order(self) -> None:
        data = self._generate_data()
        self._run_multi_process_test(
            callable=_run_output_dist_backward_order_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=ShardingType.TABLE_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs for distributed test",
    )
    def test_output_dist_multiple_hooks(self) -> None:
        data = self._generate_data()
        self._run_multi_process_test(
            callable=_run_output_dist_multiple_hooks_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=ShardingType.TABLE_WISE.value,
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Need at least 2 GPUs for distributed test",
    )
    def test_output_dist_finder_sharding_type_mismatch(self) -> None:
        data = self._generate_data()
        self._run_multi_process_test(
            callable=_run_output_dist_finder_mismatch_test,
            world_size=2,
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            data=data,
            sharding_type=ShardingType.TABLE_WISE.value,
            mismatched_sharding_type=ShardingType.COLUMN_WISE.value,
        )
