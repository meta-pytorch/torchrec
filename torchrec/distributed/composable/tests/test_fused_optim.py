#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest

import torch
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.shard import shard
from torchrec.distributed.sharding_plan import (
    apply_to_all,
    construct_module_sharding_plan,
    table_wise,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage
from torchrec.test_utils import get_free_port


class TestFusedOptim(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        if torch.cuda.is_available():
            self.curr_device = torch.device("cuda:0")
            torch.cuda.set_device(self.curr_device)
            backend = "nccl"
        else:
            self.curr_device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)

    def tearDown(self) -> None:
        dist.destroy_process_group()

    def test_opt_state_correct(self) -> None:
        num_features = 4

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            ebc.parameters(),
            {"lr": 0.01},
        )
        plan = construct_module_sharding_plan(
            ebc,
            apply_to_all(ebc, table_wise(rank=0)),
        )
        # pyrefly: ignore[invalid-param-spec]
        ebc = shard(
            module=ebc,
            plan=plan,
            device=self.curr_device,
        )
        for name, param in ebc.named_parameters():
            table_name = name[len("embedding_bags.") : -len("weight") - 1]
            self.assertEqual(
                # pyrefly: ignore[missing-attribute]
                param._in_backward_optimizers[0]
                .state_dict()["state"][""][f"{table_name}.momentum1"]
                .local_tensor()
                .data_ptr(),
                # pyrefly: ignore[missing-attribute]
                ebc._optim.state_dict()["state"][f"embedding_bags.{table_name}.weight"][
                    f"{table_name}.momentum1"
                ]
                .local_tensor()
                .data_ptr(),
            )

    def test_set_learning_rate(self) -> None:
        num_features = 1

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            ebc.parameters(),
            {"lr": 0.01},
        )
        plan = construct_module_sharding_plan(
            ebc,
            apply_to_all(ebc, table_wise(rank=0)),
        )
        # pyrefly: ignore[invalid-param-spec]
        ebc = shard(
            module=ebc,
            plan=plan,
            device=self.curr_device,
        )
        for param in ebc.parameters():
            # pyrefly: ignore[missing-attribute]
            param._in_backward_optimizers = [
                WarmupOptimizer(
                    param._in_backward_optimizers[0],
                    [
                        WarmupStage(
                            policy=WarmupPolicy.LINEAR,
                            max_iters=10000,
                            value=0.5,
                            lr_scale=1.0,
                        )
                    ],
                    param_name="__warmup_state",
                )
            ]
            # pyrefly: ignore[missing-attribute]
            param._in_backward_optimizers[0].step()
            # pyrefly: ignore[missing-attribute]
            param._in_backward_optimizers[0].step()
            # pyrefly: ignore[missing-attribute]
            warmup_state = param._in_backward_optimizers[0].state_dict()["state"][
                "__warmup_state"
            ]
            _iter, _ = warmup_state["warmup"]
            self.assertEqual(_iter, 2)
            self.assertEqual(
                # pyrefly: ignore[missing-attribute]
                param._in_backward_optimizers[0].param_groups[0]["lr"],
                0.05001,
            )
