#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from parameterized import parameterized
from torch import nn
from torch.distributed.checkpoint import CheckpointException
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan_class


def _create_model(
    tables: list[EmbeddingBagConfig],
    embedding_groups: dict[str, list[str]],
    batch: ModelInput,
    device: torch.device,
    process_group: dist.ProcessGroup,
    world_size: int,
    optimizer: EmbOptimType,
    shard_dim: int,
) -> DistributedModelParallel:
    torch.manual_seed(0)
    model = TestSparseNN(
        tables=tables,
        weighted_tables=[],
        embedding_groups=embedding_groups,
        dense_device=device,
        sparse_device=torch.device("meta"),
        num_float_features=16,
    )
    sharders = [
        cast(
            ModuleSharder[nn.Module],
            create_test_sharder(
                SharderType.EMBEDDING_BAG_COLLECTION.value,
                ShardingType.COLUMN_WISE.value,
                EmbeddingComputeKernel.FUSED.value,
                fused_params={"optimizer": optimizer},
                device=device,
            ),
        )
    ]
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=world_size, compute_device=device.type),
        constraints={
            table.name: ParameterConstraints(min_partition=shard_dim)
            for table in tables
        },
    )
    plan = planner.collective_plan(model, sharders, process_group)
    module_plan = cast(EmbeddingModuleShardingPlan, plan.plan["sparse.ebc"])
    for table in tables:
        sharding_spec = cast(
            EnumerableShardingSpec,
            none_throws(module_plan[table.name].sharding_spec),
        )
        widths = {shard.shard_sizes[1] for shard in sharding_spec.shards}
        if widths != {shard_dim}:
            raise AssertionError(
                f"{table.name} has column-wise shard widths {widths}, not {shard_dim}"
            )

    dmp = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
        init_data_parallel=False,
    )
    with torch.no_grad():
        dmp(batch)
        dmp.init_data_parallel()
    return dmp


def _train(model: DistributedModelParallel, batch: ModelInput) -> None:
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _ = cast(tuple[torch.Tensor, torch.Tensor], model(batch))
    loss.backward()


def _prediction(model: DistributedModelParallel, batch: ModelInput) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return cast(torch.Tensor, model(batch))


def _test_checkpoint(
    rank: int,
    world_size: int,
    tables: list[EmbeddingBagConfig],
    embedding_groups: dict[str, list[str]],
    checkpoint_path: str,
    optimizer: EmbOptimType,
    source_shard_dim: int,
    destination_shard_dim: int,
    expected_error: str | None,
) -> None:
    with MultiProcessContext(rank, world_size, "nccl") as ctx:
        process_group = none_throws(ctx.pg)
        _, local_batches = ModelInput.generate(
            batch_size=4,
            world_size=world_size,
            num_float_features=16,
            tables=tables,
            weighted_tables=[],
            random_seed=0,
        )
        batch = local_batches[rank].to(ctx.device)

        def create_model(shard_dim: int) -> DistributedModelParallel:
            return _create_model(
                tables,
                embedding_groups,
                batch,
                ctx.device,
                process_group,
                world_size,
                optimizer,
                shard_dim,
            )

        checkpoint_process_group = cast(
            dist.ProcessGroup, dist.new_group(backend="gloo")
        )
        try:
            source = create_model(source_shard_dim)
            _train(source, batch)
            dcp.save(
                {"module": source, "optimizer": source.fused_optimizer},
                checkpoint_id=checkpoint_path,
                process_group=checkpoint_process_group,
            )
            restored = create_model(destination_shard_dim)

            def load() -> None:
                dcp.load(
                    {"module": restored, "optimizer": restored.fused_optimizer},
                    checkpoint_id=checkpoint_path,
                    process_group=checkpoint_process_group,
                )

            if expected_error is not None:
                with unittest.TestCase().assertRaisesRegex(
                    CheckpointException, expected_error
                ):
                    load()
                return
            load()
            torch.testing.assert_close(
                _prediction(source, batch), _prediction(restored, batch)
            )
            _train(source, batch)
            _train(restored, batch)
            torch.testing.assert_close(
                _prediction(source, batch), _prediction(restored, batch)
            )
        finally:
            dist.destroy_process_group(checkpoint_process_group)


# A width change re-chunks each weight but keeps its global shape, which plain
# DCP handles. Rowwise optimizer state for a column-wise table is one tensor of
# rows * number_of_shards, so a width change resizes it under the same key, and
# plain DCP rejects that while planning, before reading.
_MOMENTUM_MISMATCH = r"Size mismatch between saved .* for .*momentum1"
_SGD = EmbOptimType.EXACT_SGD
_ROWWISE_ADAGRAD = EmbOptimType.EXACT_ROWWISE_ADAGRAD


@skip_if_asan_class
class ColumnWiseCheckpointTest(ModelParallelTestShared):
    """Loads across a column-wise shard width change at the same world size."""

    @parameterized.expand(
        [
            ("sgd_8_to_4", _SGD, 8, 4, None),
            ("sgd_4_to_8", _SGD, 4, 8, None),
            ("rowwise_adagrad_8_to_8", _ROWWISE_ADAGRAD, 8, 8, None),
            ("rowwise_adagrad_8_to_4", _ROWWISE_ADAGRAD, 8, 4, _MOMENTUM_MISMATCH),
            ("rowwise_adagrad_4_to_8", _ROWWISE_ADAGRAD, 4, 8, _MOMENTUM_MISMATCH),
        ]
    )
    @unittest.skipIf(torch.cuda.device_count() <= 1, "Requires two GPUs")
    def test_width_change(
        self,
        _name: str,
        optimizer: EmbOptimType,
        source_shard_dim: int,
        destination_shard_dim: int,
        expected_error: str | None,
    ) -> None:
        self._build_tables_and_groups()
        with TemporaryDirectory() as checkpoint_path:
            self._run_multi_process_test(
                callable=_test_checkpoint,
                world_size=2,
                tables=self.tables,
                embedding_groups=self.embedding_groups,
                checkpoint_path=checkpoint_path,
                optimizer=optimizer,
                source_shard_dim=source_shard_dim,
                destination_shard_dim=destination_shard_dim,
                expected_error=expected_error,
            )

    # parameterized.expand leaves this attribute as None; tpx --regex fails on it.
    del test_width_change
