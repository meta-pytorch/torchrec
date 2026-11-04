#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.test_utils.process_runner import run_local_multi_process_func
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _check_shared_feature_eviction(ctx, rank: int, world_size: int) -> None:
    assert rank == 0 and world_size == 1
    device = ctx.device
    tables = [
        EmbeddingConfig(
            name="tag",
            embedding_dim=8,
            num_embeddings=16,
            feature_names=["item_tag", "user_tag"],
            init_fn=torch.nn.init.ones_,
        ),
        EmbeddingConfig(
            name="item",
            embedding_dim=8,
            num_embeddings=16,
            feature_names=["item_id"],
            init_fn=torch.nn.init.ones_,
        ),
    ]
    embeddings = EmbeddingCollection(tables=tables, device=device)
    collisions = ManagedCollisionCollection(
        managed_collision_modules={
            table.name: MCHManagedCollisionModule(
                zch_size=table.num_embeddings,
                device=device,
                eviction_interval=1,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )
            for table in tables
        },
        embedding_configs=tables,
    )
    module = ManagedCollisionEmbeddingCollection(
        embeddings, collisions, return_remapped_features=True
    )
    apply_optimizer_in_backward(RowWiseAdagrad, module.parameters(), {"lr": 0.1})
    plan = EmbeddingShardingPlanner().collective_plan(module)
    sharded_module = DistributedModelParallel(module, device=device, plan=plan)

    features = KeyedJaggedTensor(
        keys=["item_tag", "user_tag", "item_id"],
        values=torch.tensor([101, 202, 303], device=device),
        lengths=torch.tensor([1, 1, 1], dtype=torch.int32, device=device),
    )
    output, _ = sharded_module(features)
    output = output.wait()
    for feature in features.keys():
        torch.testing.assert_close(
            output[feature].values(),
            torch.ones((1, 8), device=device),
            rtol=0,
            atol=0,
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class ManagedCollisionSharedFeaturesEvictionTest(unittest.TestCase):
    def test_rowwise_adagrad_keeps_all_shared_feature_embeddings(self) -> None:
        run_local_multi_process_func(
            _check_shared_feature_eviction, world_size=1, backend="nccl"
        )


if __name__ == "__main__":
    unittest.main()
