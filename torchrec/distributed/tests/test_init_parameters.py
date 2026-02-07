#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, Dict, List, Optional, Union

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torch import nn
from torch.distributed._tensor import DTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    ParameterShardingGenerator,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.test_utils import skip_if_asan_class


def initialize_and_test_parameters(
    rank: int,
    world_size: int,
    backend: str,
    table: Union[EmbeddingCollection, EmbeddingBagCollection],
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    sharding_type: str,
    sharders: List[ModuleSharder[nn.Module]],
    table_name: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Initialize embedding bag on non-meta device
        if isinstance(table, EmbeddingBagConfig):
            embedding_tables = EmbeddingBagCollection(
                tables=[table],
                device=device,
            )
        elif isinstance(table, EmbeddingConfig):
            embedding_tables = EmbeddingCollection(
                tables=[table],
                device=device,
            )
        else:
            raise RuntimeError(f"unknown table type {type(table)}")
        embedding_tables.load_state_dict(state_dict)
        module_sharding_plan = construct_module_sharding_plan(
            embedding_tables,
            per_param_sharding={
                table_name: _select_sharding_type(sharding_type),
            },
            local_size=ctx.local_size,
            world_size=ctx.world_size,
            device_type=ctx.device.type,
        )

        model = DistributedModelParallel(
            module=embedding_tables,
            plan=ShardingPlan({"": module_sharding_plan}),
            env=ShardingEnv.from_process_group(ctx.pg),
            sharders=sharders,
            device=ctx.device,
        )

        key = (
            f"embeddings.{table_name}.weight"
            if isinstance(embedding_tables, EmbeddingCollection)
            else f"embedding_bags.{table_name}.weight"
        )

        if isinstance(model.state_dict()[key], DTensor):
            if ctx.rank == 0:
                gathered_tensor = torch.empty(model.state_dict()[key].size())
            else:
                gathered_tensor = None

            gathered_tensor = model.state_dict()[key].full_tensor()
            if ctx.rank == 0:
                torch.testing.assert_close(
                    gathered_tensor,
                    embedding_tables.state_dict()[key],
                )
        elif isinstance(model.state_dict()[key], ShardedTensor):
            if ctx.rank == 0:
                gathered_tensor = torch.empty_like(
                    embedding_tables.state_dict()[key], device=ctx.device
                )
            else:
                gathered_tensor = None

            model.state_dict()[key].gather(dst=0, out=gathered_tensor)

            if ctx.rank == 0:
                torch.testing.assert_close(
                    none_throws(gathered_tensor).cpu(),
                    embedding_tables.state_dict()[key].cpu(),
                )
        elif isinstance(model.state_dict()[key], torch.Tensor):
            torch.testing.assert_close(
                embedding_tables.state_dict()[key].cpu(),
                model.state_dict()[key].cpu(),
            )
        else:
            raise AssertionError(
                f"Model state dict contains unsupported type for key: {key}"
            )


def _select_sharding_type(sharding_type: str) -> ParameterShardingGenerator:
    if sharding_type == "table_wise":
        return table_wise(rank=0)
    elif sharding_type == "column_wise":
        return column_wise(ranks=[0, 1])
    elif sharding_type == "row_wise":
        return row_wise()
    elif sharding_type == "data_parallel":
        return data_parallel()
    else:
        raise AssertionError(f"Invalid sharding type specified: {sharding_type}")


@skip_if_asan_class
class ParameterInitializationTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_WISE.value,
            ]
        ),
        device=st.sampled_from(
            [
                torch.device("cuda"),
                torch.device("cpu"),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_initialize_parameters_ec(
        self, sharding_type: str, device: torch.device
    ) -> None:
        world_size = 2
        backend = "nccl"
        table_name = "free_parameters"

        # Initialize embedding table on non-meta device, in this case cuda:0
        table = EmbeddingConfig(
            name=table_name,
            embedding_dim=64,
            num_embeddings=10,
            data_type=DataType.FP32,
        )

        state_dict = {f"embeddings.{table_name}.weight": torch.randn(10, 64)}

        self._run_multi_process_test(
            callable=initialize_and_test_parameters,
            table=table,
            state_dict=state_dict,
            device=device,
            sharding_type=sharding_type,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder())
            ],
            world_size=world_size,
            backend=backend,
            table_name=table_name,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.DATA_PARALLEL.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.TABLE_WISE.value,
            ]
        ),
        device=st.sampled_from(
            [
                torch.device("cuda"),
                # torch.device("cpu"),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_initialize_parameters_ebc(
        self, sharding_type: str, device: torch.device
    ) -> None:
        world_size = 2
        backend = "nccl"
        table_name = "free_parameters"
        table = EmbeddingBagConfig(
            name=table_name,
            embedding_dim=64,
            num_embeddings=10,
            data_type=DataType.FP32,
        )

        state_dict = {f"embedding_bags.{table_name}.weight": torch.randn(10, 64)}

        self._run_multi_process_test(
            callable=initialize_and_test_parameters,
            table=table,
            state_dict=state_dict,
            device=device,
            sharding_type=sharding_type,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
            world_size=world_size,
            backend=backend,
            table_name=table_name,
        )
