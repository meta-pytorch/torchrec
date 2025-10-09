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
from hypothesis import given, settings, Verbosity
from torchrec import distributed as trec_dist
from torchrec.distributed.quant_embedding import (
    QuantManagedCollisionEmbeddingCollectionSharder,
)
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    EmbeddingBagCollectionSharder,
    EmbeddingCollectionSharder,
    FeatureProcessedEmbeddingBagCollectionSharder,
    FusedEmbeddingBagCollectionSharder,
    get_module_to_default_sharders,
    grid_shard,
    ManagedCollisionEmbeddingBagCollectionSharder,
    ManagedCollisionEmbeddingCollectionSharder,
    ParameterShardingGenerator,
    QuantEmbeddingBagCollectionSharder,
    QuantEmbeddingCollectionSharder,
    row_wise,
    table_row_wise,
    table_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ParameterSharding,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import data_type_to_dtype, EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
    QuantManagedCollisionEmbeddingCollection,
)

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class
from torchrec.types import DataType


def _test_sharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    module_sharding_plan: EmbeddingModuleShardingPlan,
    local_size: Optional[int] = None,
) -> None:
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

        sharder = get_module_to_default_sharders()[type(model)]

        unsharded_model = model
        sharded_model = sharder.shard(
            module=model,
            params=module_sharding_plan,
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
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


@skip_if_asan_class
class ConstructParameterShardingAndShardTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "table_0": data_parallel(),
                    "table_1": data_parallel(),
                },
                {
                    "table_0": table_wise(rank=0),
                    "table_1": table_wise(rank=1),
                },
                {
                    "table_0": row_wise(),
                    "table_1": row_wise(),
                },
                {
                    "table_0": column_wise(ranks=[0, 1]),
                    "table_1": column_wise(ranks=[0, 1]),
                },
            ]
        ),
        data_type=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_parameter_sharding_ebc(
        self,
        per_param_sharding: Dict[str, ParameterShardingGenerator],
        data_type: DataType,
    ) -> None:

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=16,
                num_embeddings=4,
                data_type=data_type,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=16,
                num_embeddings=4,
                data_type=data_type,
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

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Row-wise not supported on gloo
        if (
            not torch.cuda.is_available()
            and module_sharding_plan["table_0"].sharding_type
            == ShardingType.ROW_WISE.value
        ):
            return

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.table_0.weight": torch.tensor(
                    [
                        [1] * 16,
                        [2] * 16,
                        [3] * 16,
                        [4] * 16,
                    ],
                    dtype=data_type_to_dtype(data_type),
                ),
                "embedding_bags.table_1.weight": torch.tensor(
                    [
                        [101] * 16,
                        [102] * 16,
                        [103] * 16,
                        [104] * 16,
                    ],
                    dtype=data_type_to_dtype(data_type),
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl" if torch.cuda.is_available() else "gloo",
            module_sharding_plan=module_sharding_plan,
        )


class ConstructParameterShardingTest(unittest.TestCase):
    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_construct_module_sharding_plan(self, data_type: DataType) -> None:

        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=256,
                num_embeddings=32 * 32,
                data_type=data_type,
            )
            for idx in range(6)
        ]

        expected = {
            "table_0": ParameterSharding(
                sharding_type="data_parallel",
                compute_kernel="dense",
                ranks=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ],
                sharding_spec=None,
            ),
            "table_1": ParameterSharding(
                sharding_type="table_wise",
                compute_kernel="dense",
                ranks=[1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 256],
                            placement="rank:1/cuda:1",
                        )
                    ]
                ),
            ),
            "table_2": ParameterSharding(
                sharding_type="row_wise",
                compute_kernel="dense",
                ranks=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[32, 256],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[32, 0],
                            shard_sizes=[32, 256],
                            placement="rank:1/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[64, 0],
                            shard_sizes=[32, 256],
                            placement="rank:2/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[96, 0],
                            shard_sizes=[32, 256],
                            placement="rank:3/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 0],
                            shard_sizes=[32, 256],
                            placement="rank:4/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[160, 0],
                            shard_sizes=[32, 256],
                            placement="rank:5/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[192, 0],
                            shard_sizes=[32, 256],
                            placement="rank:6/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[224, 0],
                            shard_sizes=[32, 256],
                            placement="rank:7/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 0],
                            shard_sizes=[32, 256],
                            placement="rank:8/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[288, 0],
                            shard_sizes=[32, 256],
                            placement="rank:9/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[320, 0],
                            shard_sizes=[32, 256],
                            placement="rank:10/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[352, 0],
                            shard_sizes=[32, 256],
                            placement="rank:11/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 0],
                            shard_sizes=[32, 256],
                            placement="rank:12/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[416, 0],
                            shard_sizes=[32, 256],
                            placement="rank:13/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[448, 0],
                            shard_sizes=[32, 256],
                            placement="rank:14/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[480, 0],
                            shard_sizes=[32, 256],
                            placement="rank:15/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 0],
                            shard_sizes=[32, 256],
                            placement="rank:16/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[544, 0],
                            shard_sizes=[32, 256],
                            placement="rank:17/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[576, 0],
                            shard_sizes=[32, 256],
                            placement="rank:18/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[608, 0],
                            shard_sizes=[32, 256],
                            placement="rank:19/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 0],
                            shard_sizes=[32, 256],
                            placement="rank:20/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[672, 0],
                            shard_sizes=[32, 256],
                            placement="rank:21/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[704, 0],
                            shard_sizes=[32, 256],
                            placement="rank:22/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[736, 0],
                            shard_sizes=[32, 256],
                            placement="rank:23/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 0],
                            shard_sizes=[32, 256],
                            placement="rank:24/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[800, 0],
                            shard_sizes=[32, 256],
                            placement="rank:25/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[832, 0],
                            shard_sizes=[32, 256],
                            placement="rank:26/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[864, 0],
                            shard_sizes=[32, 256],
                            placement="rank:27/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 0],
                            shard_sizes=[32, 256],
                            placement="rank:28/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[928, 0],
                            shard_sizes=[32, 256],
                            placement="rank:29/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[960, 0],
                            shard_sizes=[32, 256],
                            placement="rank:30/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[992, 0],
                            shard_sizes=[32, 256],
                            placement="rank:31/cuda:7",
                        ),
                    ]
                ),
            ),
            "table_3": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[8, 9],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 128],
                            placement="rank:8/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 128],
                            shard_sizes=[1024, 128],
                            placement="rank:9/cuda:1",
                        ),
                    ]
                ),
            ),
            "table_4": ParameterSharding(
                sharding_type="table_row_wise",
                compute_kernel="dense",
                ranks=[24, 25, 26, 27, 28, 29, 30, 31],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[128, 256],
                            placement="rank:24/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 0],
                            shard_sizes=[128, 256],
                            placement="rank:25/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 0],
                            shard_sizes=[128, 256],
                            placement="rank:26/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 0],
                            shard_sizes=[128, 256],
                            placement="rank:27/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 0],
                            shard_sizes=[128, 256],
                            placement="rank:28/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 0],
                            shard_sizes=[128, 256],
                            placement="rank:29/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 0],
                            shard_sizes=[128, 256],
                            placement="rank:30/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 0],
                            shard_sizes=[128, 256],
                            placement="rank:31/cuda:7",
                        ),
                    ]
                ),
            ),
            "table_5": ParameterSharding(
                sharding_type="grid_shard",
                compute_kernel="dense",
                ranks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[128, 128],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 0],
                            shard_sizes=[128, 128],
                            placement="rank:1/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 0],
                            shard_sizes=[128, 128],
                            placement="rank:2/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 0],
                            shard_sizes=[128, 128],
                            placement="rank:3/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 0],
                            shard_sizes=[128, 128],
                            placement="rank:4/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 0],
                            shard_sizes=[128, 128],
                            placement="rank:5/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 0],
                            shard_sizes=[128, 128],
                            placement="rank:6/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 0],
                            shard_sizes=[128, 128],
                            placement="rank:7/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 128],
                            shard_sizes=[128, 128],
                            placement="rank:8/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 128],
                            shard_sizes=[128, 128],
                            placement="rank:9/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 128],
                            shard_sizes=[128, 128],
                            placement="rank:10/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 128],
                            shard_sizes=[128, 128],
                            placement="rank:11/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 128],
                            shard_sizes=[128, 128],
                            placement="rank:12/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 128],
                            shard_sizes=[128, 128],
                            placement="rank:13/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 128],
                            shard_sizes=[128, 128],
                            placement="rank:14/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 128],
                            shard_sizes=[128, 128],
                            placement="rank:15/cuda:7",
                        ),
                    ]
                ),
            ),
        }

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": data_parallel(),
                "table_1": table_wise(rank=1),
                "table_2": row_wise(),
                "table_3": column_wise(ranks=[8, 9]),
                "table_4": table_row_wise(host_index=3),
                "table_5": grid_shard(host_indexes=[0, 1]),
            },
            local_size=8,
            world_size=32,
            device_type="cuda",
        )
        self.assertDictEqual(expected, module_sharding_plan)

    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_table_wise_set_device(self, data_type: DataType) -> None:

        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=64,
                num_embeddings=4096,
                data_type=data_type,
            )
            for idx in range(2)
        ]
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": table_wise(rank=0, device="cpu"),
                "table_1": table_wise(rank=1, device="cpu"),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )

        # Make sure per_param_sharding setting override the default device_type
        self.assertEqual(
            # pyre-ignore[16]
            module_sharding_plan["table_0"]
            .sharding_spec.shards[0]
            .placement.device()
            .type,
            "cpu",
        )

        self.assertEqual(
            module_sharding_plan["table_1"]
            .sharding_spec.shards[0]
            .placement.device()
            .type,
            "cpu",
        )

    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_row_wise_set_heterogenous_device(self, data_type: DataType) -> None:

        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=64,
                num_embeddings=4096,
                data_type=data_type,
            )
            for idx in range(2)
        ]
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": row_wise(
                    sizes_placement=(
                        [2048, 1024, 1024],
                        ["cpu", "cuda", "cuda"],
                    )
                ),
                "table_1": row_wise(
                    sizes_placement=([2048, 1024, 1024], ["cpu", "cpu", "cpu"])
                ),
            },
            local_size=1,
            world_size=2,
            device_type="cuda",
        )

        # Make sure per_param_sharding setting override the default device_type
        device_table_0_shard_0 = (
            # pyre-ignore[16]
            module_sharding_plan["table_0"]
            .sharding_spec.shards[0]
            .placement
        )
        self.assertEqual(
            device_table_0_shard_0.device().type,
            "cpu",
        )
        # cpu always has rank 0
        self.assertEqual(
            device_table_0_shard_0.rank(),
            0,
        )
        for i in range(1, 3):
            device_table_0_shard_i = (
                module_sharding_plan["table_0"].sharding_spec.shards[i].placement
            )
            self.assertEqual(
                device_table_0_shard_i.device().type,
                "cuda",
            )
            # first rank is assigned to cpu so index = rank - 1
            self.assertEqual(
                device_table_0_shard_i.device().index,
                i - 1,
            )
            self.assertEqual(
                device_table_0_shard_i.rank(),
                i,
            )
        for i in range(3):
            device_table_1_shard_i = (
                module_sharding_plan["table_1"].sharding_spec.shards[i].placement
            )
            self.assertEqual(
                device_table_1_shard_i.device().type,
                "cpu",
            )
            # cpu always has rank 0
            self.assertEqual(
                device_table_1_shard_i.rank(),
                0,
            )

    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_column_wise(self, data_type: DataType) -> None:

        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=64,
                num_embeddings=4096,
                data_type=data_type,
            )
            for idx in range(2)
        ]
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(ranks=[0, 1]),
                "table_1": column_wise(ranks=[0, 1]),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )
        expected = {
            "table_0": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[0, 1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[4096, 32],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 32],
                            shard_sizes=[4096, 32],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            ),
            "table_1": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[0, 1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[4096, 32],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 32],
                            shard_sizes=[4096, 32],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            ),
        }
        self.assertDictEqual(expected, module_sharding_plan)

    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_column_wise_size_per_rank(self, data_type: DataType) -> None:
        """Test column_wise sharding with custom size_per_rank parameter."""

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=100,  # Total columns that will be split as [30, 40, 30]
                num_embeddings=1024,
                data_type=data_type,
            )
        ]

        # Test uneven column distribution: rank 0 gets 30 cols, rank 1 gets 40 cols, rank 2 gets 30 cols
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(size_per_rank=[30, 40, 30]),
            },
            local_size=3,
            world_size=3,
            device_type="cuda",
        )

        expected = {
            "table_0": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[0, 1, 2],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 30],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 30],
                            shard_sizes=[1024, 40],
                            placement="rank:1/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 70],
                            shard_sizes=[1024, 30],
                            placement="rank:2/cuda:2",
                        ),
                    ]
                ),
            ),
        }
        self.assertDictEqual(expected, module_sharding_plan)

    # pyre-fixme[56]
    @given(data_type=st.sampled_from([DataType.FP32, DataType.FP16]))
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_column_wise_device_types(self, data_type: DataType) -> None:
        """Test column_wise sharding with mixed device types."""

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=64,
                num_embeddings=1024,
                data_type=data_type,
            )
        ]

        # Test mixed device types: cpu, cuda, cpu, cuda
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(
                    ranks=[0, 1, 2, 3],
                    device_types=["cpu", "cuda", "cpu", "cuda"],
                ),
            },
            local_size=4,
            world_size=4,
            device_type="cuda",
        )

        expected = {
            "table_0": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[0, 1, 2, 3],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 16],
                            placement="rank:0/cpu",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 16],
                            shard_sizes=[1024, 16],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 32],
                            shard_sizes=[1024, 16],
                            placement="rank:0/cpu",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 48],
                            shard_sizes=[1024, 16],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            ),
        }
        self.assertDictEqual(expected, module_sharding_plan)

    def test_column_wise_size_per_rank_insufficient_columns(self) -> None:
        """Test that column_wise raises error when size_per_rank doesn't cover all columns."""

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=100,
                num_embeddings=1024,
                data_type=DataType.FP32,
            )
        ]

        with self.assertRaises(ValueError) as context:
            construct_module_sharding_plan(
                EmbeddingBagCollection(tables=embedding_bag_config),
                per_param_sharding={
                    "table_0": column_wise(
                        size_per_rank=[30, 40]
                    ),  # Only covers 70/100 columns
                },
                local_size=2,
                world_size=2,
                device_type="cuda",
            )

        self.assertIn(
            "Cannot fit tensor of (1024, 100) into sizes_ranks_placements = [30, 40]",
            str(context.exception),
        )

    def test_column_wise_size_per_rank_with_device_types(self) -> None:
        """Test column_wise sharding with both size_per_rank and device_types parameters."""

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=80,  # Total columns that will be split as [20, 30, 30]
                num_embeddings=512,
                data_type=DataType.FP32,
            )
        ]

        # Test combining custom column sizes with mixed device types
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(
                    size_per_rank=[20, 30, 30],
                    device_types=["cpu", "cuda", "cpu"],
                ),
            },
            local_size=3,
            world_size=3,
            device_type="cuda",
        )

        expected = {
            "table_0": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="dense",
                ranks=[0, 1, 2],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[512, 20],
                            placement="rank:0/cpu",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 20],
                            shard_sizes=[512, 30],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 50],
                            shard_sizes=[512, 30],
                            placement="rank:0/cpu",
                        ),
                    ]
                ),
            ),
        }
        self.assertDictEqual(expected, module_sharding_plan)

    def test_column_wise_uneven_division_error(self) -> None:
        """Test that column_wise raises error when columns can't be evenly divided across ranks."""

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=65,  # Cannot be evenly divided by 2
                num_embeddings=1024,
                data_type=DataType.FP32,
            )
        ]

        with self.assertRaises(ValueError) as context:
            construct_module_sharding_plan(
                EmbeddingBagCollection(tables=embedding_bag_config),
                per_param_sharding={
                    "table_0": column_wise(
                        ranks=[0, 1]
                    ),  # 65 columns cannot be evenly divided by 2 ranks
                },
                local_size=2,
                world_size=2,
                device_type="cuda",
            )

        self.assertIn(
            "column dim of 65 cannot be evenly divided across [0, 1]",
            str(context.exception),
        )


class ShardingPlanTest(unittest.TestCase):
    def test_str(self) -> None:
        plan = ShardingPlan(
            {
                "ebc": EmbeddingModuleShardingPlan(
                    {
                        "user_id": ParameterSharding(
                            sharding_type="table_wise",
                            compute_kernel="dense",
                            ranks=[0],
                            sharding_spec=EnumerableShardingSpec(
                                [
                                    ShardMetadata(
                                        shard_offsets=[0, 0],
                                        shard_sizes=[4096, 32],
                                        placement="rank:0/cuda:0",
                                    ),
                                ]
                            ),
                        ),
                        "movie_id": ParameterSharding(
                            sharding_type="row_wise",
                            compute_kernel="dense",
                            ranks=[0, 1],
                            sharding_spec=EnumerableShardingSpec(
                                [
                                    ShardMetadata(
                                        shard_offsets=[0, 0],
                                        shard_sizes=[2048, 32],
                                        placement="rank:0/cuda:0",
                                    ),
                                    ShardMetadata(
                                        shard_offsets=[2048, 0],
                                        shard_sizes=[2048, 32],
                                        placement="rank:0/cuda:1",
                                    ),
                                ]
                            ),
                        ),
                    }
                )
            }
        )
        expected = """module: ebc

 param   | sharding type | compute kernel | ranks
-------- | ------------- | -------------- | ------
user_id  | table_wise    | dense          | [0]
movie_id | row_wise      | dense          | [0, 1]

 param   | shard offsets | shard sizes |   placement
-------- | ------------- | ----------- | -------------
user_id  | [0, 0]        | [4096, 32]  | rank:0/cuda:0
movie_id | [0, 0]        | [2048, 32]  | rank:0/cuda:0
movie_id | [2048, 0]     | [2048, 32]  | rank:0/cuda:1
"""
        self.maxDiff = None
        for i in range(len(expected.splitlines())):
            self.assertEqual(
                expected.splitlines()[i].strip(), str(plan).splitlines()[i].strip()
            )

    def test_module_to_default_sharders(self) -> None:
        default_sharder_map = get_module_to_default_sharders()
        self.assertCountEqual(
            default_sharder_map,
            [
                EmbeddingBagCollection,
                FeatureProcessedEmbeddingBagCollection,
                EmbeddingCollection,
                FusedEmbeddingBagCollection,
                QuantEmbeddingBagCollection,
                QuantEmbeddingCollection,
                ManagedCollisionEmbeddingBagCollection,
                ManagedCollisionEmbeddingCollection,
                QuantManagedCollisionEmbeddingCollection,
            ],
        )
        self.assertIsInstance(
            default_sharder_map[EmbeddingBagCollection], EmbeddingBagCollectionSharder
        )
        self.assertIsInstance(
            default_sharder_map[FeatureProcessedEmbeddingBagCollection],
            FeatureProcessedEmbeddingBagCollectionSharder,
        )
        self.assertIsInstance(
            default_sharder_map[EmbeddingCollection], EmbeddingCollectionSharder
        )
        self.assertIsInstance(
            default_sharder_map[FusedEmbeddingBagCollection],
            FusedEmbeddingBagCollectionSharder,
        )
        self.assertIsInstance(
            default_sharder_map[QuantEmbeddingBagCollection],
            QuantEmbeddingBagCollectionSharder,
        )
        self.assertIsInstance(
            default_sharder_map[QuantEmbeddingCollection],
            QuantEmbeddingCollectionSharder,
        )
        self.assertIsInstance(
            default_sharder_map[ManagedCollisionEmbeddingBagCollection],
            ManagedCollisionEmbeddingBagCollectionSharder,
        )

        self.assertIsInstance(
            default_sharder_map[ManagedCollisionEmbeddingCollection],
            ManagedCollisionEmbeddingCollectionSharder,
        )

        self.assertIsInstance(
            default_sharder_map[QuantManagedCollisionEmbeddingCollection],
            QuantManagedCollisionEmbeddingCollectionSharder,
        )
