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
from hypothesis import given, settings, Verbosity
from torchrec import distributed as trec_dist, EmbeddingConfig
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    ModuleSharder,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _test_sharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model.embedding_bags["table_0"].parameters(),
            {"lr": 1.0},
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model.embedding_bags["table_1"].parameters(),
            {"lr": 4.0},
        )

        unsharded_model = model

        plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)
        sharded_model = DistributedModelParallel(
            module=model,
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=[sharder],
            device=ctx.device,
        )

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        for _it in range(5):
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
                sharded_model_pred.cpu(), unsharded_model_pred.cpu()
            )

            sharded_model_pred.sum().backward()

            all_unsharded_preds = torch.stack(all_unsharded_preds)
            all_unsharded_preds.sum().backward()


class TestEmbeddingBagCollectionSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.FUSED.value,
        ]


@skip_if_asan_class
class ShardedEmbeddingBagCollectionApplyOptimPerParamTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_ebc_per_parameter_optimizer(
        self,
        sharding_type: str,
    ) -> None:

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=4,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=4,
                num_embeddings=4,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]        [0, 1,2,3]
        # "feature_1"   [2,3]       None        [2]

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
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.table_0.weight": torch.Tensor(
                    [
                        [1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [4, 4, 4, 4],
                        [8, 8, 8, 8],
                    ]
                ),
                "embedding_bags.table_1.weight": torch.Tensor(
                    [
                        [101, 101, 101, 101],
                        [102, 102, 102, 102],
                        [104, 104, 104, 104],
                        [108, 108, 108, 108],
                    ]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=TestEmbeddingBagCollectionSharder(sharding_type=sharding_type),
            backend="nccl",
        )


def _test_sharding_ec(
    tables: List[EmbeddingConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        model = EmbeddingCollection(
            tables=tables,
            device=ctx.device,
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model.embeddings["table_0"].parameters(),
            {"lr": 1.0},
        )

        apply_optimizer_in_backward(
            torch.optim.SGD,
            model.embeddings["table_1"].parameters(),
            {"lr": 4.0},
        )

        unsharded_model = model

        plan: ShardingPlan = planner.collective_plan(model, [sharder], ctx.pg)
        sharded_model = DistributedModelParallel(
            module=model,
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=[sharder],
            device=ctx.device,
        )

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        for _it in range(5):
            unsharded_model_preds = []
            for rank in range(ctx.world_size):
                unsharded_pred = unsharded_model(kjt_input_per_rank[rank])
                unsharded_pred = torch.cat(
                    [
                        unsharded_pred[feature].values().view(-1)
                        for feature in feature_keys
                    ]
                )
                # simulate the unsharded model run on the entire batch
                unsharded_model_preds.append(unsharded_pred)

            # sharded model
            # each rank gets a subbatch
            sharded_model_pred_kt = sharded_model(kjt_input_per_rank[ctx.rank])
            sharded_model_pred = torch.cat(
                [
                    sharded_model_pred_kt[feature].values().view(-1)
                    for feature in feature_keys
                ]
            )

            # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
            # in normal author modelling code this won't be an issue because each rank would individually create
            # their model. output from sharded_pred is correctly on the correct device.

            # Compare predictions of sharded vs unsharded models.
            torch.testing.assert_close(
                sharded_model_pred.cpu(), unsharded_model_preds[ctx.rank].cpu()
            )

            sharded_model_pred.sum().backward()

            all_unsharded_preds = torch.cat(
                [pred.view(-1) for pred in unsharded_model_preds]
            )
            all_unsharded_preds.sum().backward()


class TestEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.FUSED.value,
        ]


@skip_if_asan_class
class ShardedEmbeddingCollectionApplyOptimPerParamTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_ec_per_parameter_optimizer(
        self,
        sharding_type: str,
    ) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=4,
                num_embeddings=4,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=4,
                num_embeddings=4,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]        [0, 1,2,3]
        # "feature_1"   [2,3]       None        [2]

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
            callable=_test_sharding_ec,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            initial_state_dict={
                "embeddings.table_0.weight": torch.Tensor(
                    [
                        [1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [4, 4, 4, 4],
                        [8, 8, 8, 8],
                    ]
                ),
                "embeddings.table_1.weight": torch.Tensor(
                    [
                        [101, 101, 101, 101],
                        [102, 102, 102, 102],
                        [104, 104, 104, 104],
                        [108, 108, 108, 108],
                    ]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            sharder=TestEmbeddingCollectionSharder(sharding_type=sharding_type),
            backend="nccl",
        )
