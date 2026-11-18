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
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.model_parallel import DataParallelWrapper
from torchrec.distributed.test_utils.emb_sharder import (
    TestEBCSharder,
    TestEBCSharderMCH,
)
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.train_pipelines import TrainPipelineSparseDist
from torchrec.distributed.types import ModuleSharder, ShardingEnv
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.test_utils import init_distributed_single_host


class TrainPipelineSparseDistTestBase(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        backend = "gloo"
        if torch.cuda.is_available():
            backend = "nccl"
        self.pg = init_distributed_single_host(backend=backend, rank=0, world_size=1)

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
        max_feature_lengths: Optional[Dict[str, int]] = None,
        zch: bool = False,
    ) -> nn.Module:
        # Create ZCH kwargs arguments if provided
        zch_kwargs = None
        if zch:
            zch_kwargs = {
                table.name: {
                    "mc_type": "sort-zch",
                    "input_hash_size": 4000,
                }
                for table in self.tables
            }
        unsharded_model = model_type(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
            postproc_module=postproc_module,
            max_feature_lengths=max_feature_lengths,
            zch_kwargs=zch_kwargs,
        )
        if enable_fsdp:
            # pyrefly: ignore[missing-attribute]
            unsharded_model.over.dhn_arch.linear0 = FSDP(
                #  attribute `dhn_arch`.
                # pyrefly: ignore[missing-attribute]
                unsharded_model.over.dhn_arch.linear0
            )
            # pyrefly: ignore[missing-attribute]
            unsharded_model.over.dhn_arch.linear1 = FSDP(
                #  attribute `dhn_arch`.
                # pyrefly: ignore[missing-attribute]
                unsharded_model.over.dhn_arch.linear1
            )
            #  `dhn_arch`.
            # pyrefly: ignore[missing-attribute]
            unsharded_model.over.dhn_arch = FSDP(unsharded_model.over.dhn_arch)

        return unsharded_model

    def _generate_sharded_model_and_optimizer(
        self,
        model: nn.Module,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
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
            data_parallel_wrapper=data_parallel_wrapper,
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

    def _setup_ddp_model_and_optimizer(
        self,
        model: nn.Module,
        sharding_type: str,
        kernel_type: str,
        combined_optimizer: bool = True,
    ) -> Tuple[DistributedModelParallel, Optimizer]:
        """
        Shards ``model`` with DDP enabled and returns it with an Adam optimizer over
        the remaining (dense) parameters. With ``combined_optimizer``, the Adam
        optimizer is combined with the model's fused optimizer.
        """
        sharded_model, _ = self._generate_sharded_model_and_optimizer(
            model, sharding_type, kernel_type
        )
        sharded_model = cast(DistributedModelParallel, sharded_model)
        # The helper above skips DDP, and async_optimizer falls back without it.
        sharded_model.init_data_parallel()
        dense_params = dict(
            in_backward_optimizer_filter(sharded_model.named_parameters())
        )
        if not combined_optimizer:
            # Citrine C2: use the multi-tensor optimizer implementation.
            return sharded_model, optim.Adam(
                dense_params.values(), lr=0.01, foreach=True
            )
        dense_optimizer = KeyedOptimizerWrapper(
            dense_params,
            # Citrine C2: use the multi-tensor optimizer implementation.
            lambda params: optim.Adam(params, lr=0.01, foreach=True),
        )
        return sharded_model, CombinedOptimizer(
            [sharded_model.fused_optimizer, dense_optimizer]
        )

    def _run_async_optimizer_parity(
        self,
        pipeline_class: Type[TrainPipelineSparseDist],
        sharding_type: str,
        kernel_type: str,
        num_batches: int = 10,
        combined_optimizer: bool = True,
        optimizer_factory: Optional[Callable[[CombinedOptimizer], Optimizer]] = None,
        expect_async: bool = True,
        **pipeline_kwargs: Any,
    ) -> Tuple[TrainPipelineSparseDist, TrainPipelineSparseDist]:
        """
        Trains the same model with and without ``async_optimizer`` and checks
        that every step's output and the final parameters match. With
        ``optimizer_factory``, both pipelines get the optimizer it builds from the
        combined optimizer. With ``expect_async=False``, checks that the async
        pipeline fell back to the regular step. Returns the sync and async pipelines.
        """
        model = self._setup_model()
        data = self._generate_data(num_batches=num_batches, batch_size=32)
        sync_model, sync_optimizer = self._setup_ddp_model_and_optimizer(
            model, sharding_type, kernel_type, combined_optimizer
        )
        async_model, async_optimizer = self._setup_ddp_model_and_optimizer(
            model, sharding_type, kernel_type, combined_optimizer
        )
        if optimizer_factory is not None:
            sync_optimizer = optimizer_factory(cast(CombinedOptimizer, sync_optimizer))
            async_optimizer = optimizer_factory(
                cast(CombinedOptimizer, async_optimizer)
            )
        copy_state_dict(async_model.state_dict(), sync_model.state_dict())

        sync_pipeline = pipeline_class(
            model=sync_model,
            optimizer=sync_optimizer,
            device=self.device,
            **pipeline_kwargs,
        )
        async_pipeline = pipeline_class(
            model=async_model,
            optimizer=async_optimizer,
            device=self.device,
            async_optimizer=True,
            **pipeline_kwargs,
        )
        sync_iter, async_iter = iter(data), iter(data)
        for _ in range(num_batches):
            torch.testing.assert_close(
                async_pipeline.progress(async_iter), sync_pipeline.progress(sync_iter)
            )

        if expect_async:
            self.assertGreater(len(async_pipeline._async_optimizer_ddp_modules), 0)
        else:
            self.assertEqual(async_pipeline._async_optimizer_ddp_modules, [])
        self.assertEqual(sync_pipeline._async_optimizer_ddp_modules, [])
        async_params = dict(async_model.named_parameters())
        for name, param in sync_model.named_parameters():
            torch.testing.assert_close(async_params[name], param)
        return sync_pipeline, async_pipeline
