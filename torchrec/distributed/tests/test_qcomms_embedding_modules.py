#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest
from typing import Any, Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
import torchrec.distributed as trec_dist
from hypothesis import given, settings, Verbosity
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    ParameterShardingGenerator,
    row_wise,
    table_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import ModuleSharder, ParameterSharding, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _test_sharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    parameter_sharding_plan: Dict[str, ParameterSharding],
    sharder: ModuleSharder[nn.Module],
    local_size: Optional[int] = None,
) -> None:
    # pyrefly: ignore[implicit-import]
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model.parameters(),
            {"lr": 1.0},
        )

        unsharded_model = model
        sharded_model = sharder.shard(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            params=parameter_sharding_plan,
            #  `Optional[ProcessGroup]`.
            # pyrefly: ignore[bad-argument-type]
            env=ShardingEnv.from_process_group(ctx.pg),
            device=ctx.device,
        )

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        # each rank gets a subbatch
        sharded_model_pred_kt = sharded_model(kjt_input_per_rank[ctx.rank]).to_dict()
        _sharded_model_pred = torch.stack(  # noqa
            [sharded_model_pred_kt[feature] for feature in feature_keys]
        )

        for _it in range(1):
            unsharded_model_pred_kt = []
            for rank in range(ctx.world_size):
                # simulate the unsharded model run on the entire batch
                unsharded_model_pred_kt.append(
                    unsharded_model(kjt_input_per_rank[rank])
                )

            all_unsharded_preds = []
            for rank in range(ctx.world_size):
                unsharded_model_pred_kt_mini_batch = unsharded_model_pred_kt[
                    rank
                ].to_dict()

                all_unsharded_preds.extend(
                    [
                        unsharded_model_pred_kt_mini_batch[feature]
                        for feature in feature_keys
                    ]
                )
                if rank == ctx.rank:
                    unsharded_model_pred = torch.stack(
                        [
                            unsharded_model_pred_kt_mini_batch[feature]
                            for feature in feature_keys
                        ]
                    )

            # sharded model
            # each rank gets a subbatch
            sharded_model_pred_kt = sharded_model(
                kjt_input_per_rank[ctx.rank]
            ).to_dict()
            sharded_model_pred = torch.stack(
                [sharded_model_pred_kt[feature] for feature in feature_keys]
            )

            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.
            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(
                sharded_model_pred.cpu(),
                # pyrefly: ignore[unbound-name]
                unsharded_model_pred.cpu(),
            )

            sharded_model_pred.sum().backward()

            all_unsharded_preds = torch.stack(all_unsharded_preds)
            all_unsharded_preds.sum().backward()


@skip_if_asan_class
class ConstructParameterShardingTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "0": table_wise(rank=0),
                    "1": row_wise(),
                    "2": column_wise(ranks=[0, 1]),
                },
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.FP32
                ),
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_parameter_sharding_ebc(
        self,
        per_param_sharding: Dict[str, ParameterShardingGenerator],
        qcomms_config: QCommsConfig,
    ) -> None:

        WORLD_SIZE = 2
        EMBEDDING_DIM = 8
        NUM_EMBEDDINGS = 4

        embedding_bag_config = [
            EmbeddingBagConfig(
                name=str(idx),
                feature_names=[f"feature_{idx}"],
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=NUM_EMBEDDINGS,
            )
            for idx in per_param_sharding
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       []        [2]
        # "feature_1"   [2]       [2,3]     []
        # "feature_2"   [0,1,2,3]       [0,2]        [2,3]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]        [0, 1,2,3]
        # "feature_1"   [2,3]       None        [2]
        # "feature_2"   [0, 1]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor([0, 1, 2, 2, 2, 3, 0, 1, 2, 3, 0, 2, 2, 3]),
                lengths=torch.LongTensor([2, 0, 1, 1, 2, 0, 4, 2, 2]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1", "feature_2"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1, 2, 0, 1]),
            ),
        ]

        sharder = EmbeddingBagCollectionSharder(
            qcomm_codecs_registry=(
                get_qcomm_codecs_registry(qcomms_config)
                if qcomms_config is not None
                else None
            )
        )

        ebc = EmbeddingBagCollection(tables=embedding_bag_config)
        apply_optimizer_in_backward(
            torch.optim.SGD,
            ebc.parameters(),
            {"lr": 1.0},
        )

        parameter_sharding_plan = construct_module_sharding_plan(
            module=ebc,
            per_param_sharding=per_param_sharding,
            local_size=2,
            world_size=2,
            # pyrefly: ignore[bad-argument-type]
            sharder=sharder,
        )

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.0.weight": torch.Tensor(
                    [[1] * EMBEDDING_DIM for val in range(NUM_EMBEDDINGS)]
                ),
                "embedding_bags.1.weight": torch.Tensor(
                    [[2] * EMBEDDING_DIM for val in range(NUM_EMBEDDINGS)]
                ),
                "embedding_bags.2.weight": torch.Tensor(
                    [[3] * EMBEDDING_DIM for val in range(NUM_EMBEDDINGS)]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            backend=(
                "nccl"
                if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
                else "gloo"
            ),
            sharder=sharder,
            parameter_sharding_plan=parameter_sharding_plan,
        )
