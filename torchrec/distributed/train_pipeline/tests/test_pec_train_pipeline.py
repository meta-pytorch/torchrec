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
import torch.nn as nn
import torch.optim as optim
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.pec_embedding import (
    PECEmbeddingCollectionSharder,
    ShardedPECEmbeddingCollection,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestPECMixedEmbeddingSparseArch,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline.pec_train_pipeline import TrainPipelinePEC
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.test_utils import skip_if_asan_class


def _test_pec_pipeline_numerical(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    ec_tables: List[EmbeddingConfig],
    data: List[Tuple[ModelInput, List[ModelInput]]],
    backend: str = "nccl",
    local_size: Optional[int] = None,
) -> None:
    """Compares PEC pipeline output with a non-pipelined forward.

    Creates two identical sharded models with the same weights:
    - no-pipeline: standard forward + backward + optimizer step
    - pipeline: TrainPipelinePEC.progress()

    The predictions must match exactly -- PEC's NOL deferral is index-safe.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        all_tables: List[EmbeddingBagConfig | EmbeddingConfig] = []
        all_tables.extend(tables)
        all_tables.extend(ec_tables)

        # Meta-device arch for sharding; a real-device copy holds the ground-truth
        # weights copied into both sharded models.
        sparse_arch = TestPECMixedEmbeddingSparseArch(
            # pyrefly: ignore [bad-argument-type]
            tables=all_tables,
            dense_device=ctx.device,
            sparse_device=torch.device("meta"),
        )
        reference_arch = TestPECMixedEmbeddingSparseArch(
            # pyrefly: ignore [bad-argument-type]
            tables=all_tables,
            dense_device=ctx.device,
        )

        ebc_sharder = cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder())
        pec_sharder = cast(ModuleSharder[nn.Module], PECEmbeddingCollectionSharder())
        sharders = [ebc_sharder, pec_sharder]

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=world_size,
                compute_device=ctx.device.type,
                local_world_size=local_size,
            ),
        )
        plan: ShardingPlan = planner.collective_plan(sparse_arch, sharders, ctx.pg)

        sharded_arch_pipeline = DistributedModelParallel(
            module=copy.deepcopy(sparse_arch),
            plan=plan,
            # pyrefly: ignore [bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=sharders,
            device=ctx.device,
            init_data_parallel=True,
        ).to(ctx.device)

        sharded_arch_no_pipeline = DistributedModelParallel(
            module=copy.deepcopy(sparse_arch),
            plan=plan,
            # pyrefly: ignore [bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=sharders,
            device=ctx.device,
            init_data_parallel=True,
        ).to(ctx.device)

        # Copy identical ground-truth weights into both sharded models.
        copy_state_dict(
            sharded_arch_no_pipeline.state_dict(), reference_arch.state_dict()
        )
        copy_state_dict(sharded_arch_pipeline.state_dict(), reference_arch.state_dict())

        batches = [d[1][ctx.rank].to(ctx.device) for d in data]
        dataloader = iter(batches)

        # Citrine C2: foreach=True for multi-tensor optimizer execution. Both
        # optimizers use the same setting so the numerical comparison below
        # stays exact.
        optimizer_no_pipeline = optim.SGD(
            sharded_arch_no_pipeline.parameters(), lr=0.1, foreach=True
        )
        optimizer_pipeline = optim.SGD(
            sharded_arch_pipeline.parameters(), lr=0.1, foreach=True
        )

        pipeline = TrainPipelinePEC(
            sharded_arch_pipeline,
            optimizer_pipeline,
            ctx.device,
        )

        # Skip the last 2 batches: the pipeline consumes them for its 2-ahead fill.
        for batch in batches[:-2]:
            optimizer_no_pipeline.zero_grad()
            loss, pred = sharded_arch_no_pipeline(batch)
            loss.backward()
            optimizer_no_pipeline.step()

            pred_pipeline = pipeline.progress(dataloader)

            # pyrefly: ignore [missing-attribute]
            torch.testing.assert_close(pred_pipeline.cpu(), pred.cpu())

        # _pipeline_model (run during fill) must have discovered every sharded
        # PEC module in the model -- the pipeline's PEC index should match the
        # PEC modules actually present in the sharded model.
        num_model_pec_modules = sum(
            isinstance(module, ShardedPECEmbeddingCollection)
            for module in sharded_arch_pipeline.modules()
        )
        assert num_model_pec_modules > 0, "expected at least one sharded PEC module"
        assert len(pipeline._pec_modules) == num_model_pec_modules, (
            f"pipeline indexed {len(pipeline._pec_modules)} PEC modules but the "
            f"model has {num_model_pec_modules}"
        )


@skip_if_asan_class
class TrainPipelinePECTest(MultiProcessTestBase):
    """Multi-process tests for TrainPipelinePEC."""

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_pipeline_numerical(self) -> None:
        """Verifies PEC pipeline predictions match the non-pipelined forward."""
        WORLD_SIZE = 2
        D = 8

        ebc_tables = [
            EmbeddingBagConfig(
                name="ebc_table_0",
                feature_names=["ebc_feature_0"],
                embedding_dim=D,
                num_embeddings=100,
            ),
        ]

        ec_tables = [
            EmbeddingConfig(
                name="ec_table_0",
                feature_names=["ec_feature_0"],
                embedding_dim=D,
                num_embeddings=100,
            ),
            EmbeddingConfig(
                name="ec_table_1",
                feature_names=["ec_feature_1"],
                embedding_dim=D,
                num_embeddings=100,
            ),
        ]

        all_tables: List[EmbeddingBagConfig | EmbeddingConfig] = []
        all_tables.extend(ebc_tables)
        all_tables.extend(ec_tables)

        data = [
            ModelInput.generate(
                # pyrefly: ignore [bad-argument-type]
                tables=all_tables,
                weighted_tables=[],
                batch_size=10,
                world_size=WORLD_SIZE,
                num_float_features=10,
            )
            for _ in range(7)
        ]

        self._run_multi_process_test(
            callable=_test_pec_pipeline_numerical,
            world_size=WORLD_SIZE,
            tables=ebc_tables,
            ec_tables=ec_tables,
            data=data,
        )
