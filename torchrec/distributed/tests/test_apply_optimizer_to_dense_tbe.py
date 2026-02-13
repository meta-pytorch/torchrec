#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import unittest

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import distributed as trec_dist
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSingleRankBase,
)
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad

logger: logging.Logger = logging.getLogger(__name__)


class ApplyOptmizerDenseTBETest(ModelParallelSingleRankBase):
    def _create_tables(self) -> None:
        num_features = 4

        self.tables += [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        fused_learning_rate=st.sampled_from([0, 0.1]),
        non_fused_learning_rate=st.sampled_from([0, 0.1]),
        dense_learning_rate=st.sampled_from([0, 0.1]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_apply_optimizer_to_dense_tbe(
        self,
        fused_learning_rate: float,
        non_fused_learning_rate: float,
        dense_learning_rate: float,
    ) -> None:
        unsharded_model = self._create_model()

        # torchrec sharding
        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[
                    (
                        ShardingType.TABLE_WISE.value
                        if i % 2
                        else ShardingType.DATA_PARALLEL.value
                    )
                ],
                compute_kernels=[
                    (
                        EmbeddingComputeKernel.FUSED.value
                        if i % 2
                        else EmbeddingComputeKernel.DENSE.value
                    )
                ],
            )
            for i, table in enumerate(self.tables)
        }
        sharders = get_default_sharders()
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=trec_dist.comm.get_local_size(env.world_size),
                world_size=env.world_size,
                compute_device=self.device.type,
            ),
            constraints=constraints,
        )
        # pyrefly: ignore[bad-argument-type, missing-argument]
        plan = planner.plan(unsharded_model, sharders)

        ### apply Rowwise Adagrad optimizer to fused TBEs ###
        # fused TBE optimizer needs to be configured before initializing
        #  `named_parameters`.
        # pyrefly: ignore[missing-attribute]
        for _, param in unsharded_model.sparse.named_parameters():
            apply_optimizer_in_backward(
                RowWiseAdagrad,
                [param],
                {"lr": fused_learning_rate},
            )

        # shard model
        sharded_model = DistributedModelParallel(
            module=unsharded_model,
            init_data_parallel=True,
            device=self.device,
            sharders=sharders,
            plan=plan,
        )

        ### apply Rowwise Adagrad optimizer to Data Parallel tables ###
        # Optimizer for non fused tables need to be configured after initializing
        non_fused_tables_optimizer = KeyedOptimizerWrapper(
            dict(
                in_backward_optimizer_filter(
                    #  attribute `named_parameters`.
                    # pyrefly: ignore[missing-attribute]
                    sharded_model.module.sparse.named_parameters()
                )
            ),
            lambda params: RowWiseAdagrad(
                # pyrefly: ignore[bad-argument-type]
                params,
                lr=non_fused_learning_rate,
                eps=1e-8,  # to match with FBGEMM
            ),
        )

        ### apply SGD to dense arch + over arch ###
        dense_params = [
            param
            for name, param in sharded_model.named_parameters()
            if "sparse" not in name
        ]
        dense_opt = torch.optim.Adagrad(
            dense_params,
            lr=dense_learning_rate,
        )

        # create input
        _, local_batch = ModelInput.generate(
            batch_size=self.batch_size,
            world_size=env.world_size,
            num_float_features=self.num_float_features,
            tables=self.tables,
            weighted_tables=[],
        )
        batch = local_batch[0].to(self.device)

        # record signatures
        #  `dhn_arch`.
        # pyrefly: ignore[missing-attribute]
        dense_signature = sharded_model.module.over.dhn_arch.linear0.weight.sum().item()
        non_fused_table_signature = (
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.embedding_bags.table_0.weight.sum().item()
        )
        fused_table_signature = (
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.embedding_bags.table_1.weight.sum().item()
        )

        ### training ###
        # zero grad
        non_fused_tables_optimizer.zero_grad()
        dense_opt.zero_grad()

        # forward and backward
        loss, pred = sharded_model(batch)
        loss.backward()

        # apply gradients
        non_fused_tables_optimizer.step()
        dense_opt.step()

        self.assertEqual(
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.embedding_bags.table_1.weight.sum().item()
            != fused_table_signature,
            bool(fused_learning_rate),
        )
        self.assertEqual(
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.sparse.ebc.embedding_bags.table_0.weight.sum().item()
            != non_fused_table_signature,
            bool(non_fused_learning_rate),
        )
        self.assertEqual(
            #  `dhn_arch`.
            # pyrefly: ignore[missing-attribute]
            sharded_model.module.over.dhn_arch.linear0.weight.sum().item()
            != dense_signature,
            bool(dense_learning_rate),
        )
