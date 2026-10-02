#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import unittest
from tempfile import TemporaryDirectory
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
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
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import skip_if_asan_class


def _place_shards_on_local_test_devices(plan: ShardingPlan) -> None:
    """Map simulated multi-group ranks onto this host's four test GPUs."""
    module_plan = cast(EmbeddingModuleShardingPlan, plan.plan["sparse.ebc"])
    for parameter_sharding in module_plan.values():
        sharding_spec = cast(
            EnumerableShardingSpec, none_throws(parameter_sharding.sharding_spec)
        )
        for shard in sharding_spec.shards:
            rank = none_throws(none_throws(shard.placement).rank())
            shard.placement = torch.distributed._remote_device(
                f"rank:{rank}/cuda:{rank}"
            )


def _create_model(
    tables: list[EmbeddingBagConfig],
    embedding_groups: dict[str, list[str]],
    batch: ModelInput,
    device: torch.device,
    process_group: dist.ProcessGroup,
    world_size: int,
    local_size: int,
    num_twrw_groups: int,
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
                ShardingType.TABLE_ROW_WISE.value,
                EmbeddingComputeKernel.FUSED.value,
                fused_params={"optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD},
                device=device,
            ),
        )
    ]
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=world_size,
            local_world_size=local_size,
            compute_device=device.type,
            ssd_cap=2 * 1024**4,
        ),
        constraints={
            table.name: ParameterConstraints(num_twrw_groups=num_twrw_groups)
            for table in tables
        },
    )
    plan = planner.collective_plan(model, sharders, process_group)
    _place_shards_on_local_test_devices(plan)
    module_plan = cast(EmbeddingModuleShardingPlan, plan.plan["sparse.ebc"])
    expected_num_twrw_groups = num_twrw_groups if num_twrw_groups > 1 else None
    for table in tables:
        if module_plan[table.name].num_twrw_groups != expected_num_twrw_groups:
            raise AssertionError(
                f"{table.name} did not use num_twrw_groups={num_twrw_groups}"
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


def _run_checkpoint_case(
    tables: list[EmbeddingBagConfig],
    embedding_groups: dict[str, list[str]],
    batch: ModelInput,
    device: torch.device,
    process_group: dist.ProcessGroup,
    checkpoint_process_group: dist.ProcessGroup,
    checkpoint_root: str,
    world_size: int,
    local_size: int,
    source_num_twrw_groups: int,
    destination_num_twrw_groups: int,
) -> None:
    source = _create_model(
        tables,
        embedding_groups,
        batch,
        device,
        process_group,
        world_size,
        local_size,
        source_num_twrw_groups,
    )
    _train(source, batch)
    checkpoint_path = os.path.join(
        checkpoint_root,
        f"{source_num_twrw_groups}_to_{destination_num_twrw_groups}",
    )
    dcp.save(
        {"module": source, "optimizer": source.fused_optimizer},
        checkpoint_id=checkpoint_path,
        process_group=checkpoint_process_group,
    )
    restored = _create_model(
        tables,
        embedding_groups,
        batch,
        device,
        process_group,
        world_size,
        local_size,
        destination_num_twrw_groups,
    )
    dcp.load(
        {"module": restored, "optimizer": restored.fused_optimizer},
        checkpoint_id=checkpoint_path,
        process_group=checkpoint_process_group,
    )
    torch.testing.assert_close(_prediction(source, batch), _prediction(restored, batch))
    _train(source, batch)
    _train(restored, batch)
    torch.testing.assert_close(_prediction(source, batch), _prediction(restored, batch))


def _test_checkpoint(
    rank: int,
    world_size: int,
    tables: list[EmbeddingBagConfig],
    embedding_groups: dict[str, list[str]],
    checkpoint_root: str,
    local_size: int,
    source_num_twrw_groups: int,
    destination_num_twrw_groups: int,
) -> None:
    with MultiProcessContext(rank, world_size, "nccl", local_size) as ctx:
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
        checkpoint_process_group = cast(
            dist.ProcessGroup, dist.new_group(backend="gloo")
        )
        try:
            _run_checkpoint_case(
                tables,
                embedding_groups,
                batch,
                ctx.device,
                process_group,
                checkpoint_process_group,
                checkpoint_root,
                world_size,
                local_size,
                source_num_twrw_groups,
                destination_num_twrw_groups,
            )
        finally:
            dist.destroy_process_group(checkpoint_process_group)


@skip_if_asan_class
class TWRWCheckpointTest(ModelParallelTestShared):
    def _test_checkpoint(
        self,
        source_num_twrw_groups: int,
        destination_num_twrw_groups: int,
    ) -> None:
        self._build_tables_and_groups()
        with TemporaryDirectory() as checkpoint_root:
            self._run_multi_process_test(
                callable=_test_checkpoint,
                world_size=4,
                local_size=2,
                tables=self.tables,
                embedding_groups=self.embedding_groups,
                checkpoint_root=checkpoint_root,
                source_num_twrw_groups=source_num_twrw_groups,
                destination_num_twrw_groups=destination_num_twrw_groups,
            )

    @unittest.skipIf(torch.cuda.device_count() <= 3, "Requires four GPUs")
    def test_num_twrw_groups_transitions(self) -> None:
        for source_num_twrw_groups, destination_num_twrw_groups in (
            (1, 1),
            (2, 2),
            (1, 2),
            (2, 1),
        ):
            with self.subTest(
                source_num_twrw_groups=source_num_twrw_groups,
                destination_num_twrw_groups=destination_num_twrw_groups,
            ):
                self._test_checkpoint(
                    source_num_twrw_groups, destination_num_twrw_groups
                )
