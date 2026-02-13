#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings, Verbosity
from torch.distributed._tensor.api import DTensor
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec import distributed as trec_dist
from torchrec.distributed.embedding import (
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardedTensor, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _test_sharding(  # noqa C901
    tables: List[EmbeddingConfig],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    local_size: Optional[int] = None,
    use_apply_optimizer_in_backward: bool = False,
    use_index_dedup: bool = False,
) -> None:
    # pyrefly: ignore[implicit-import]
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sharder = EmbeddingCollectionSharder(use_index_dedup=use_index_dedup)
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]

        unsharded_model = EmbeddingCollection(
            tables=tables,
            device=ctx.device,
            need_indices=True,
        )

        # syncs model across ranks
        torch.manual_seed(0)
        for param in unsharded_model.parameters():
            nn.init.uniform_(param, -1, 1)
        torch.manual_seed(0)

        if use_apply_optimizer_in_backward:
            apply_optimizer_in_backward(
                torch.optim.SGD,
                unsharded_model.embeddings.parameters(),
                {"lr": 1.0},
            )
        else:
            unsharded_model_optimizer = torch.optim.SGD(
                unsharded_model.parameters(), lr=1.0
            )

        module_sharding_plan = construct_module_sharding_plan(
            unsharded_model,
            per_param_sharding={
                "table_0": table_wise(rank=0),
                "table_1": row_wise(),
                "table_2": column_wise(ranks=[0, 1]),
            },
            local_size=local_size,
            world_size=world_size,
            device_type=ctx.device.type,
            # pyrefly: ignore[bad-argument-type]
            sharder=sharder,
        )

        sharded_model = _shard_modules(
            module=unsharded_model,
            plan=ShardingPlan({"": module_sharding_plan}),
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            #  `Optional[List[ModuleSharder[Module]]]` but got
            #  `List[EmbeddingCollectionSharder]`.
            # pyrefly: ignore[bad-argument-type]
            sharders=[sharder],
            device=ctx.device,
        )

        if not use_apply_optimizer_in_backward:
            sharded_model_optimizer = torch.optim.SGD(
                sharded_model.parameters(), lr=1.0
            )

        assert isinstance(sharded_model, ShardedEmbeddingCollection)

        if not use_apply_optimizer_in_backward:
            # pyrefly: ignore[unbound-name]
            unsharded_model_optimizer.zero_grad()
            # pyrefly: ignore[unbound-name]
            sharded_model_optimizer.zero_grad()

        unsharded_model_pred_jt_dict = []
        for unsharded_rank in range(ctx.world_size):
            # simulate the unsharded model run on the entire batch
            unsharded_model_pred_jt_dict.append(
                unsharded_model(kjt_input_per_rank[unsharded_rank])
            )

        # sharded model
        # each rank gets a subbatch
        sharded_model_pred_jts_dict: Dict[str, JaggedTensor] = sharded_model(
            kjt_input_per_rank[ctx.rank]
        )

        unsharded_model_pred_jt_dict_this_rank: Dict[str, JaggedTensor] = (
            unsharded_model_pred_jt_dict[ctx.rank]
        )

        embedding_names = unsharded_model_pred_jt_dict_this_rank.keys()
        assert set(unsharded_model_pred_jt_dict_this_rank.keys()) == set(
            sharded_model_pred_jts_dict.keys()
        )

        unsharded_loss = []
        sharded_loss = []
        for embedding_name in embedding_names:
            unsharded_jt = unsharded_model_pred_jt_dict_this_rank[embedding_name]
            sharded_jt = sharded_model_pred_jts_dict[embedding_name]

            torch.testing.assert_close(unsharded_jt.values(), sharded_jt.values())
            torch.testing.assert_close(unsharded_jt.lengths(), sharded_jt.lengths())
            torch.testing.assert_close(unsharded_jt.offsets(), sharded_jt.offsets())
            torch.testing.assert_close(
                unsharded_jt.weights_or_none(), sharded_jt.weights_or_none()
            )

            sharded_loss.append(sharded_jt.values().view(-1))

        for rank in range(ctx.world_size):
            for embedding_name in embedding_names:
                unsharded_loss.append(
                    unsharded_model_pred_jt_dict[rank][embedding_name].values().view(-1)
                )

        torch.cat(sharded_loss).sum().backward()
        torch.cat(unsharded_loss).sum().backward()

        if not use_apply_optimizer_in_backward:
            # pyrefly: ignore[unbound-name]
            unsharded_model_optimizer.step()
            # pyrefly: ignore[unbound-name]
            sharded_model_optimizer.step()

        for fqn in unsharded_model.state_dict():
            unsharded_state = unsharded_model.state_dict()[fqn]
            sharded_state = sharded_model.state_dict()[fqn]

            sharded_param = (
                torch.zeros(size=unsharded_state.shape, device=ctx.device)
                if ctx.rank == 0
                else None
            )
            if isinstance(sharded_state, ShardedTensor):
                sharded_state.gather(out=sharded_param)
            elif isinstance(sharded_state, DTensor):
                sharded_param = sharded_state.full_tensor()
            else:
                sharded_param = sharded_state

            if ctx.rank == 0:
                torch.testing.assert_close(
                    unsharded_state,
                    sharded_param,
                    msg=f"Did not match for {fqn=} after backward",
                )


@skip_if_asan_class
class ShardedEmbeddingCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    @given(
        use_apply_optimizer_in_backward=st.booleans(),
        use_index_dedup=st.booleans(),
    )
    def test_sharding_ebc(
        self,
        use_apply_optimizer_in_backward: bool,
        use_index_dedup: bool,
    ) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=4,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_0", "feature_1"],
                embedding_dim=8,
                num_embeddings=4,
            ),
            EmbeddingConfig(
                name="table_2",
                feature_names=["feature_0", "feature_1"],
                embedding_dim=8,
                num_embeddings=4,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
        # "feature_1"   [2, 3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
        ]
        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl",
            use_apply_optimizer_in_backward=use_apply_optimizer_in_backward,
            use_index_dedup=use_index_dedup,
        )
