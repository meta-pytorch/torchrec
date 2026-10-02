#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import cast, Dict, Tuple
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import distributed as dist
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    data_parallel,
    row_wise,
)
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSparseOnlyBase,
    ModelParallelStateDictBase,
)
from torchrec.distributed.types import ModuleSharder, ShardedModule, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ModelParallelStateDictTestNccl(ModelParallelStateDictBase):
    backend = "nccl"


class SparseArch(nn.Module):
    def __init__(
        self,
        ebc: EmbeddingBagCollection,
        ec: EmbeddingCollection,
    ) -> None:
        super().__init__()
        self.ebc = ebc
        self.ec = ec

    def forward(self, features: KeyedJaggedTensor) -> tuple[torch.Tensor, torch.Tensor]:
        ebc_out = self.ebc(features)
        ec_out = self.ec(features)
        return ebc_out.values(), ec_out.values()


# Create a model with two sparse architectures sharing the same modules
class TwoSparseArchModel(nn.Module):
    def __init__(
        self,
        sparse1: SparseArch,
        sparse2: SparseArch,
    ) -> None:
        super().__init__()
        # Both architectures share the same EBC and EC instances
        self.sparse1 = sparse1
        self.sparse2 = sparse2

    def forward(
        self, features: KeyedJaggedTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ebc1_out, ec1_out = self.sparse1(features)
        ebc2_out, ec2_out = self.sparse2(features)

        return ebc1_out, ec1_out, ebc2_out, ec2_out


class ModelParallelSparseOnlyTestNccl(ModelParallelSparseOnlyBase):
    backend = "nccl"

    def test_dp_lookup_respects_defer_flag(self) -> None:
        module = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="row_wise_table",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=["row_wise_feature"],
                ),
                EmbeddingBagConfig(
                    name="data_parallel_table",
                    embedding_dim=4,
                    num_embeddings=8,
                    feature_names=["data_parallel_feature"],
                ),
            ],
            device=self.device,
        )
        sharder = EmbeddingBagCollectionSharder()
        plan = construct_module_sharding_plan(
            module,
            per_param_sharding={
                "row_wise_table": row_wise(
                    compute_kernel=EmbeddingComputeKernel.FUSED.value
                ),
                "data_parallel_table": data_parallel(),
            },
            local_size=1,
            world_size=1,
            device_type=self.device.type,
            sharder=cast(ModuleSharder[torch.nn.Module], sharder),
        )
        process_group = dist.GroupMember.WORLD
        assert process_group is not None
        sharded_module = sharder.shard(
            module=module,
            params=plan,
            env=ShardingEnv.from_process_group(process_group),
            device=self.device,
        )
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["row_wise_feature", "data_parallel_feature"],
            values=torch.tensor([0, 1], device=self.device),
            lengths=torch.tensor([1, 1], device=self.device),
        )

        lookup_features: list[str] = []

        def record_lookup(
            lookup_input: KeyedJaggedTensor,
            _embeddings: torch.Tensor,
            _module: torch.nn.Module | None,
            _optimizer_state: torch.Tensor | None,
        ) -> None:
            lookup_features.extend(lookup_input.keys())

        sharded_module.register_post_lookup_tracker_fn(record_lookup)

        eager_context = sharded_module.create_context()
        self.assertFalse(eager_context.defer_dp_lookup)
        eager_input = sharded_module.input_dist(eager_context, features).wait().wait()
        eager_request = sharded_module.compute_and_output_dist(
            eager_context, eager_input
        )

        self.assertEqual(["row_wise_feature", "data_parallel_feature"], lookup_features)

        eager_output = eager_request.wait()
        lookup_features.clear()

        deferred_context = sharded_module.create_context()
        deferred_context.defer_dp_lookup = True
        deferred_input = (
            sharded_module.input_dist(deferred_context, features).wait().wait()
        )
        deferred_output = sharded_module.compute_and_output_dist(
            deferred_context, deferred_input
        )

        self.assertEqual(["row_wise_feature"], lookup_features)

        resolved_output = deferred_output.wait()

        self.assertEqual(["row_wise_feature", "data_parallel_feature"], lookup_features)
        self.assertEqual(eager_output.keys(), resolved_output.keys())
        torch.testing.assert_close(eager_output.values(), resolved_output.values())

    def test_shared_sparse_module_in_multiple_parents(self) -> None:
        """
        Test that the module ID cache correctly handles the same sparse module
        being used in multiple parent modules. This tests the caching behavior
        when a single EmbeddingBagCollection and EmbeddingCollection are shared
        across two different parent sparse architectures.
        """

        def mock_init_dmp(
            self_dmp: DistributedModelParallel, module: nn.Module
        ) -> nn.Module:
            """Override _init_dmp to always set module_id_cache to None"""
            # Call _shard_modules_impl with module_id_cache=None (caching disabled)
            module_id_cache: Dict[int, ShardedModule] = {}
            # pyrefly: ignore[bad-argument-type]
            return self_dmp._shard_modules_impl(module, module_id_cache=module_id_cache)

        # Setup: Create shared embedding modules that will be reused
        ebc = EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="ebc_table",
                    embedding_dim=64,
                    num_embeddings=100,
                    feature_names=["ebc_feature"],
                ),
            ],
        )
        ec = EmbeddingCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingConfig(
                    name="ec_table",
                    embedding_dim=32,
                    num_embeddings=50,
                    feature_names=["ec_feature"],
                ),
            ],
        )

        # Create the model with shared modules
        sparse1 = SparseArch(ebc, ec)
        sparse2 = SparseArch(ebc, ec)
        model = TwoSparseArchModel(sparse1, sparse2)

        # Execute: Shard the model with DistributedModelParallel
        with patch.object(
            DistributedModelParallel,
            "_init_dmp",
            mock_init_dmp,
        ):
            dmp = DistributedModelParallel(model, device=self.device)

        # Assert: Verify that the shared modules are properly handled
        self.assertIsNotNone(dmp.module)

        # Verify that the same module instances are reused (cached behavior)
        wrapped_module = dmp.module
        self.assertIs(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse1.ebc,
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ebc,
            "ebc1 and ebc2 should be the same sharded instance",
        )
        self.assertIs(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse1.ec,
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ec,
            "ec1 and ec2 should be the same sharded instance",
        )
        self.assertIsInstance(
            wrapped_module.sparse1.ebc,
            ShardedModule,
            "ebc1 should be sharded",
        )
        self.assertIsInstance(
            wrapped_module.sparse1.ec,
            ShardedModule,
            "ec1 should be sharded",
        )

    def test_shared_sparse_module_in_multiple_parents_negative(self) -> None:
        """
        Test that when module ID caching is disabled (module_id_cache=None),
        the same module instance gets sharded multiple times, resulting in
        different sharded instances. This validates the behavior without caching.
        """

        def mock_init_dmp(
            self_dmp: DistributedModelParallel, module: nn.Module
        ) -> nn.Module:
            """Override _init_dmp to always set module_id_cache to None"""
            # Call _shard_modules_impl with module_id_cache=None (caching disabled)
            return self_dmp._shard_modules_impl(module, module_id_cache=None)

        # Setup: Create shared embedding modules that will be reused
        ebc = EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="ebc_table",
                    embedding_dim=64,
                    num_embeddings=100,
                    feature_names=["ebc_feature"],
                ),
            ],
        )
        ec = EmbeddingCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingConfig(
                    name="ec_table",
                    embedding_dim=32,
                    num_embeddings=50,
                    feature_names=["ec_feature"],
                ),
            ],
        )

        # Create the model with shared modules
        sparse1 = SparseArch(ebc, ec)
        sparse2 = SparseArch(ebc, ec)
        model = TwoSparseArchModel(sparse1, sparse2)

        # Execute: Mock _init_dmp to disable caching, then shard the model
        with patch.object(
            DistributedModelParallel,
            "_init_dmp",
            mock_init_dmp,
        ):
            dmp = DistributedModelParallel(model, device=self.device)

        # Assert: Verify that modules are NOT cached (different instances)
        self.assertIsNotNone(dmp.module)
        wrapped_module = dmp.module

        # Without caching, the same module should be sharded twice,
        # resulting in different sharded instances
        self.assertIsNot(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse1.ebc,
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ebc,
            "Without caching, ebc1 and ebc2 should be different sharded instances",
        )
        self.assertIsNot(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse1.ec,
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ec,
            "Without caching, ec1 and ec2 should be different sharded instances",
        )

        # Both should still be properly sharded, just not cached
        self.assertIsInstance(
            wrapped_module.sparse1.ebc,
            ShardedModule,
            "ebc1 should be sharded",
        )
        self.assertIsInstance(
            wrapped_module.sparse1.ec,
            ShardedModule,
            "ec1 should be sharded",
        )
        self.assertIsInstance(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ebc,
            ShardedModule,
            "ebc2 should be sharded",
        )
        self.assertIsInstance(
            # pyrefly: ignore[missing-attribute]
            wrapped_module.sparse2.ec,
            ShardedModule,
            "ec2 should be sharded",
        )

    def test_shared_sparse_module_optimizer_dedup(self) -> None:
        """
        Test that the module ID cache in _fused_optim_impl correctly handles
        the same sparse module being used in multiple parent modules.

        This validates that:
        1. The optimizer is only collected once for shared modules
        2. No duplicate param keys exist in the CombinedOptimizer
        """

        # Setup: Create shared embedding modules that will be reused
        ebc = EmbeddingBagCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingBagConfig(
                    name="ebc_table",
                    embedding_dim=64,
                    num_embeddings=100,
                    feature_names=["ebc_feature"],
                ),
            ],
        )
        ec = EmbeddingCollection(
            device=torch.device("meta"),
            tables=[
                EmbeddingConfig(
                    name="ec_table",
                    embedding_dim=32,
                    num_embeddings=50,
                    feature_names=["ec_feature"],
                ),
            ],
        )

        # Create the model with shared modules
        sparse1 = SparseArch(ebc, ec)
        sparse2 = SparseArch(ebc, ec)
        model = TwoSparseArchModel(sparse1, sparse2)

        # Execute: Shard the model with DistributedModelParallel
        # This should NOT raise ValueError due to duplicate param keys
        dmp = DistributedModelParallel(model, device=self.device)

        # Assert: Verify optimizer has no duplicate params
        fused_optim = dmp.fused_optimizer
        params = fused_optim.params

        # Check that param tensors are unique (no duplicates)
        param_ids = [id(p) for p in params.values()]
        self.assertEqual(
            len(param_ids),
            len(set(param_ids)),
            "fused_optimizer.params should not have duplicate parameter objects",
        )
