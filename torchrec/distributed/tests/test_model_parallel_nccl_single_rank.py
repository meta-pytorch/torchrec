#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

import torch
import torch.nn as nn
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSparseOnlyBase,
    ModelParallelStateDictBase,
)
from torchrec.distributed.types import ShardedModule
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ebc1_out, ec1_out = self.sparse1(features)
        ebc2_out, ec2_out = self.sparse2(features)

        return ebc1_out, ec1_out, ebc2_out, ec2_out


class ModelParallelSparseOnlyTestNccl(ModelParallelSparseOnlyBase):
    backend = "nccl"

    def test_shared_sparse_module_in_multiple_parents(self) -> None:
        """
        Test that the module ID cache correctly handles the same sparse module
        being used in multiple parent modules. This tests the caching behavior
        when a single EmbeddingBagCollection and EmbeddingCollection are shared
        across two different parent sparse architectures.
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
        dmp = DistributedModelParallel(model, device=self.device)

        # Assert: Verify that the shared modules are properly handled
        self.assertIsNotNone(dmp.module)

        # Verify that the same module instances are reused (cached behavior)
        wrapped_module = dmp.module
        self.assertIs(
            wrapped_module.sparse1.ebc,
            wrapped_module.sparse2.ebc,
            "ebc1 and ebc2 should be the same sharded instance",
        )
        self.assertIs(
            wrapped_module.sparse1.ec,
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
            wrapped_module.sparse1.ebc,
            wrapped_module.sparse2.ebc,
            "Without caching, ebc1 and ebc2 should be different sharded instances",
        )
        self.assertIsNot(
            wrapped_module.sparse1.ec,
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
            wrapped_module.sparse2.ebc,
            ShardedModule,
            "ebc2 should be sharded",
        )
        self.assertIsInstance(
            wrapped_module.sparse2.ec,
            ShardedModule,
            "ec2 should be sharded",
        )
