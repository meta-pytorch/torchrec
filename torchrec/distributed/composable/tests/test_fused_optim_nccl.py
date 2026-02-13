#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from torchrec.distributed.shard import shard
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.optimizers import PartialRowWiseAdam
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad


class ShardedFusedOptimizerStateDictTest(MultiProcessTestBase):
    @staticmethod
    def _test_sharded_fused_optimizer_state_dict(
        tables: List[EmbeddingBagConfig],
        rank: int,
        local_size: int,
        world_size: int,
        backend: str,
    ) -> None:
        with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
            ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
            apply_optimizer_in_backward(
                RowWiseAdagrad,
                #  got `Iterable[Union[Tensor, Module]]`.
                # pyrefly: ignore[bad-argument-type]
                [
                    ebc.embedding_bags["table_0"].weight,
                    ebc.embedding_bags["table_1"].weight,
                ],
                {"lr": 0.01},
            )
            apply_optimizer_in_backward(
                PartialRowWiseAdam,
                #  got `Iterable[Union[Tensor, Module]]`.
                # pyrefly: ignore[bad-argument-type]
                [
                    ebc.embedding_bags["table_2"].weight,
                    ebc.embedding_bags["table_3"].weight,
                ],
                {"lr": 0.02},
            )

            parameter_sharding_plan = construct_module_sharding_plan(
                ebc,
                per_param_sharding={
                    "table_0": column_wise(ranks=[0, 0, 1, 1]),
                    "table_1": row_wise(),
                    "table_2": column_wise(ranks=[0, 1, 0, 1]),
                    "table_3": row_wise(),
                },
                world_size=ctx.world_size,
                local_size=ctx.local_size,
                device_type=ctx.device.type,
            )

            # pyrefly: ignore[invalid-param-spec]
            ebc = shard(
                module=ebc,
                plan=parameter_sharding_plan,
                device=ctx.device,
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_0"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_0.momentum1"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # sharded column, each shard will have rowwise state
                    else torch.empty((4 * tables[0].num_embeddings,), device=ctx.device)
                ),
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_1"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_1.momentum1"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # sharded rowwise
                    else torch.empty((tables[1].num_embeddings,), device=ctx.device)
                ),
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_2"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_2.momentum1"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # Column wise - with partial rowwise adam, first state is point wise
                    else torch.empty(
                        (tables[2].num_embeddings, tables[2].embedding_dim),
                        device=ctx.device,
                    )
                ),
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_2"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_2.exp_avg_sq"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # Column wise - with partial rowwise adam, first state is point wise
                    else torch.empty((4 * tables[2].num_embeddings,), device=ctx.device)
                ),
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_3"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_3.momentum1"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # Row wise - with partial rowwise adam, first state is point wise
                    else torch.empty(
                        (tables[3].num_embeddings, tables[3].embedding_dim),
                        device=ctx.device,
                    )
                ),
            )

            # pyrefly: ignore[bad-index, missing-attribute]
            ebc.embedding_bags["table_3"].weight._in_backward_optimizers[
                0
            ].state_dict()["state"][""]["table_3.exp_avg_sq"].gather(
                dst=0,
                out=(
                    None
                    if ctx.rank != 0
                    # Column wise - with partial rowwise adam, first state is point wise
                    else torch.empty((tables[2].num_embeddings,), device=ctx.device)
                ),
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharded_fused_optimizer_state_dict(self) -> None:
        WORLD_SIZE = 2
        LOCAL_SIZE = 2
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                # test different optimizer table datatypes to ensure optimizer dtype is consistent
                data_type=DataType.FP16 if i > 1 else DataType.FP32,
            )
            for i in range(4)
        ]

        self._run_multi_process_test(
            callable=self._test_sharded_fused_optimizer_state_dict,
            tables=tables,
            backend="nccl",
            local_size=LOCAL_SIZE,
            world_size=WORLD_SIZE,
        )
