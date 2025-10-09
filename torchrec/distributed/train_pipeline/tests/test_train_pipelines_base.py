#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import os
import unittest
from typing import Any, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestEBCSharder,
    TestEBCSharderMCH,
    TestSparseNN,
)
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.types import ModuleSharder, ShardingEnv
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.test_utils import get_free_port, init_distributed_single_host


class TrainPipelineSparseDistTestBase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        backend = "gloo"
        if torch.cuda.is_available():
            backend = "nccl"
        self.pg = init_distributed_single_host(backend=backend, rank=0, world_size=1)

        num_features = 4
        num_weighted_features = 4
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

        self.device = torch.device("cuda:0")
        self.pipeline_class = TrainPipelineSparseDist

    def tearDown(self) -> None:
        super().tearDown()
        dist.destroy_process_group(self.pg)

    def _generate_data(
        self,
        num_batches: int = 5,
        batch_size: int = 1,
        max_feature_lengths: Optional[List[int]] = None,
    ) -> List[ModelInput]:
        return [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=batch_size,
                world_size=1,
                num_float_features=10,
                max_feature_lengths=max_feature_lengths,
            )[0]
            for i in range(num_batches)
        ]

    def _set_table_weights_precision(self, dtype: DataType) -> None:
        for i in range(len(self.tables)):
            self.tables[i].data_type = dtype

        for i in range(len(self.weighted_tables)):
            self.weighted_tables[i].data_type = dtype

    def _setup_model(
        self,
        model_type: Type[nn.Module] = TestSparseNN,
        enable_fsdp: bool = False,
        postproc_module: Optional[nn.Module] = None,
        zch: bool = False,
    ) -> nn.Module:
        unsharded_model = model_type(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
            postproc_module=postproc_module,
            zch=zch,
        )
        if enable_fsdp:
            unsharded_model.over.dhn_arch.linear0 = FSDP(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `dhn_arch`.
                unsharded_model.over.dhn_arch.linear0
            )
            unsharded_model.over.dhn_arch.linear1 = FSDP(
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `dhn_arch`.
                unsharded_model.over.dhn_arch.linear1
            )
            # pyre-fixme[16]: `Module` has no attribute `dhn_arch`.
            # pyre-fixme[16]: `Tensor` has no attribute `dhn_arch`.
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `dhn_arch`.
            unsharded_model.over.dhn_arch = FSDP(unsharded_model.over.dhn_arch)

        return unsharded_model

    def _generate_sharded_model_and_optimizer(
        self,
        model: nn.Module,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Optimizer]:
        sharder = TestEBCSharder(
            sharding_type=sharding_type,
            kernel_type=kernel_type,
            fused_params=fused_params,
        )
        mc_sharder = TestEBCSharderMCH(
            sharding_type=sharding_type,
            kernel_type=kernel_type,
            fused_params=fused_params,
        )
        sharded_model = DistributedModelParallel(
            module=copy.deepcopy(model),
            env=ShardingEnv.from_process_group(self.pg),
            init_data_parallel=False,
            device=self.device,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    sharder,
                ),
                cast(
                    ModuleSharder[nn.Module],
                    mc_sharder,
                ),
            ],
        )
        # default fused optimizer is SGD w/ lr=0.1; we need to drop params
        fused_named_parameters: List[str] = [
            x for x in DistributedModelParallel._sharded_parameter_names(sharded_model)
        ]
        optimizer = optim.SGD(
            [
                y
                for x, y in sharded_model.named_parameters()
                if x not in fused_named_parameters
            ],
            lr=0.1,
        )
        return sharded_model, optimizer
