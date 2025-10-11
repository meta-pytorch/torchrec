#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from dataclasses import dataclass, field
from typing import cast, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
import torchrec
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

from parameterized import parameterized
from torch import nn
from torchrec import KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import ModuleSharder, ShardingType
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_tracker.tests.utils import (
    EmbeddingTableProps,
    generate_planner_constraints,
    TestEBCModel,
    TestECModel,
)
from torchrec.distributed.model_tracker.types import ModelTrackerConfig, TrackingMode

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.test_utils import skip_if_asan

NUM_EMBEDDINGS: int = 16
EMBEDDING_DIM: int = 256


def generate_test_models(
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
    tables: Iterable[EmbeddingTableProps],
) -> nn.Module:
    return (
        TestECModel(
            tables=[
                cast(EmbeddingConfig, table.embedding_table_config) for table in tables
            ]
        )
        if embedding_config_type == EmbeddingConfig
        else TestEBCModel(
            tables=[
                cast(EmbeddingBagConfig, table.embedding_table_config)
                for table in tables
            ]
        )
    )


# input parameters for KJT generation
@dataclass
class ModelInput:
    keys: List[str]
    values: torch.Tensor
    offsets: torch.Tensor


@dataclass
class ModelDeltaTrackerInputTestParams:
    # input parameters
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]]
    model_tracker_config: ModelTrackerConfig
    embedding_tables: List[EmbeddingTableProps]
    model_inputs: List[ModelInput] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)


@dataclass
class FqnToFeatureNamesOutputTestParams:
    # expected output parameters
    expected_fqn_to_feature_names: Dict[str, List[str]]


@dataclass
class TrackerNotInitOutputTestParams:
    # DMP tracker Attribute
    dmp_tracker_atter: str


@dataclass
class EmbeddingModeOutputTestParams:
    # assert string
    assert_str: Optional[str]


@dataclass
class MultipleOutputTestParams:
    # Expected output for each iteration
    expected_outputs: List[Dict[str, Dict[int, torch.Tensor]]]
    consumer_access: List[str] = field(default_factory=list)


def model_input_generator(
    model_inputs: List[ModelInput], rank: int
) -> List[KeyedJaggedTensor]:
    return [
        KeyedJaggedTensor.from_offsets_sync(
            model_input.keys, model_input.values + rank, model_input.offsets
        ).to(torch.device(f"cuda:{rank}"))
        for model_input in model_inputs
    ]


# Helper function to create a model
def get_models(
    rank: int,
    world_size: int,
    ctx: MultiProcessContext,
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
    tables: Iterable[EmbeddingTableProps],
    optimizer_type: OptimType = OptimType.ADAM,
    config: Optional[ModelTrackerConfig] = None,
) -> Tuple[DistributedModelParallel, DistributedModelParallel]:
    # Create the model
    torch.manual_seed(0)
    test_model = generate_test_models(embedding_config_type, tables)

    # Set up device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Create planner and sharders
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size, "cuda"),
        constraints=generate_planner_constraints(tables),
    )
    sharders = [
        cast(
            ModuleSharder[nn.Module],
            EmbeddingCollectionSharder(
                fused_params={
                    "optimizer": optimizer_type,
                    "beta1": 0.9,
                    "beta2": 0.99,
                }
            ),
        ),
        cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder(fused_params={"optimizer": optimizer_type}),
        ),
    ]

    # Create plan
    plan = planner.collective_plan(test_model, sharders, ctx.pg)

    # Create DMP
    if ctx.pg is None:
        raise ValueError("Process group cannot be None")

    dt_dmp = DistributedModelParallel(
        module=test_model,
        device=device,
        env=torchrec.distributed.ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
        model_tracker_config=config,
    )

    torch.manual_seed(0)
    baseline_module = generate_test_models(embedding_config_type, tables)
    baseline_dmp = DistributedModelParallel(
        module=baseline_module,
        device=device,
        # pyre-ignore[6]
        env=torchrec.distributed.ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
    )

    return dt_dmp, baseline_dmp


class ModelDeltaTrackerTest(MultiProcessTestBase):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)
        self.world_size = 2

    @parameterized.expand(
        [
            (
                "EC_model_test",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f4", "f5", "f6"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ec.embeddings.sparse_table_1": ["f1", "f2", "f3"],
                        "ec.embeddings.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "EBC_model_test",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f4", "f5", "f6"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ebc.embedding_bags.sparse_table_1": ["f1", "f2", "f3"],
                        "ebc.embedding_bags.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "EC_model_test_with_duplicate_feature_names",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f3", "f4", "f5"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ec.embeddings.sparse_table_1": ["f1", "f2", "f3"],
                        "ec.embeddings.sparse_table_2": ["f3", "f4", "f5"],
                    },
                ),
            ),
            (
                "fqns_to_skip_table_name",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f4", "f5", "f6"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        fqns_to_skip=["sparse_table_1"]
                    ),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ebc.embedding_bags.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "fqns_to_skip_mid_fqn",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f4", "f5", "f6"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        fqns_to_skip=["embedding_bags"]
                    ),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={},
                ),
            ),
            (
                "fqns_to_skip_parent_fqn",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f3", "f4", "f5"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(fqns_to_skip=["ec"]),
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={},
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 1, "test requires 1+ GPUs")
    def test_fqn_to_feature_names(
        self,
        _test_name: str,
        input_params: ModelDeltaTrackerInputTestParams,
        output_params: FqnToFeatureNamesOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_fqn_to_feature_names,
            world_size=self.world_size,
            input_params=input_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "get_model_tracker",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="table_fqn_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(),
                ),
                TrackerNotInitOutputTestParams(
                    dmp_tracker_atter="get_model_tracker",
                ),
            ),
            (
                "get_delta",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="table_fqn_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2", "f3"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(),
                ),
                TrackerNotInitOutputTestParams(
                    dmp_tracker_atter="get_delta",
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 1, "test requires 1+ GPUs")
    def test_tracker_not_initialized(
        self,
        _test_name: str,
        input_params: ModelDeltaTrackerInputTestParams,
        output_params: TrackerNotInitOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_tracker_init,
            world_size=self.world_size,
            input_params=input_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "test_dup_with_EC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
            ),
            (
                "test_dup_with_EBC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
            ),
            (
                "test_multi_feature_per_table_EC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f3", "f4"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
            ),
            (
                "test_multi_feature_per_table_EBC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f3", "f4"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Distributed test requires at least 2 GPUs",
    )
    def test_tracker_id_mode(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_id_mode,
            world_size=self.world_size,
            test_params=test_params,
        )

    @parameterized.expand(
        [
            (
                "test_dup_with_EC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(assert_str=None),
            ),
            (
                "test_multi_feature_per_table_EC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f3", "f4"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(assert_str=None),
            ),
            # We don't support tracking of raw ids for EBC yet. This test validates that.
            (
                "assert_on_validation",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(
                    assert_str="EBC's lookup returns pooled embeddings and currently, we do not support tracking raw embeddings."
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Distributed test requires at least 2 GPUs",
    )
    def test_tracker_embedding_mode(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
        output_params: EmbeddingModeOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_embedding_mode,
            world_size=self.world_size,
            test_params=test_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "multi_get_with_EC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14]
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6]
                        {
                            # Rank 0: sparse_table_2 gets f2 IDs [0,2,4,6], sparse_table_1 is empty
                            # Rank 1: sparse_table_1 gets f1 IDs [8,10,12,14], sparse_table_2 is empty
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7]
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,1,2,3], sparse_table_2 gets f2 IDs [4,5,6,7]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor([]),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(4, 8)),
                                1: torch.tensor([]),
                            },
                        },
                    ]
                ),
            ),
            (
                "multi_get_with_EBC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14]
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6]
                        {
                            # Rank 0: sparse_table_2 gets f2 IDs [0,2,4,6], sparse_table_1 is empty
                            # Rank 1: sparse_table_1 gets f1 IDs [8,10,12,14], sparse_table_2 is empty
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7]
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,1,2,3], sparse_table_2 gets f2 IDs [4,5,6,7]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor([]),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(4, 8)),
                                1: torch.tensor([]),
                            },
                        },
                    ]
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    def test_multiple_get(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
        output_params: MultipleOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_multiple_get,
            world_size=self.world_size,
            test_params=test_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "EC_and_delete_on_read_true",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                        consumers=["A", "B"],
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    consumer_access=["A", "B", "A"],
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14] - Consumer A access
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6] - Consumer B access
                        {
                            # Consumer B gets all accumulated data since last access (both inputs)
                            # Rank 0: Both tables have accumulated IDs from both inputs
                            # Rank 1: Both tables have accumulated IDs from both inputs
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7] - Consumer A access
                        {
                            # Consumer A gets delta since last access (inputs 2 and 3)
                            # Rank 0: sparse_table_1 gets new f1 IDs [0,1,2,3], sparse_table_2 gets accumulated IDs
                            # Rank 1: sparse_table_1 gets accumulated IDs, sparse_table_2 gets new f2 ID [0]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor(range(8)),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([0]),
                            },
                        },
                    ],
                ),
            ),
            (
                "EC_and_delete_on_read_false",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=False,
                        consumers=["A", "B"],
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    consumer_access=["A", "B", "A"],
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14] - Consumer A access
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6] - Consumer B access
                        {
                            # Consumer B gets all accumulated data since last access (both inputs)
                            # Rank 0: Both tables have accumulated IDs from both inputs
                            # Rank 1: Both tables have accumulated IDs from both inputs
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7] - Consumer A access
                        {
                            # Consumer A gets delta since last access (inputs 2 and 3)
                            # Rank 0: sparse_table_1 gets new f1 IDs [0,1,2,3], sparse_table_2 gets accumulated IDs
                            # Rank 1: sparse_table_1 gets accumulated IDs, sparse_table_2 gets new f2 ID [0]
                            "ec.embeddings.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor(range(8)),
                            },
                            "ec.embeddings.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([0]),
                            },
                        },
                    ],
                ),
            ),
            (
                "EBC_and_delete_on_read_true",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                        consumers=["A", "B"],
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    consumer_access=["A", "B", "A"],
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14] - Consumer A access
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6] - Consumer B access
                        {
                            # Consumer B gets all accumulated data since last access (both inputs)
                            # Rank 0: Both tables have accumulated IDs from both inputs
                            # Rank 1: Both tables have accumulated IDs from both inputs
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7] - Consumer A access
                        {
                            # Consumer A gets delta since last access (inputs 2 and 3)
                            # Rank 0: sparse_table_1 gets new f1 IDs [0,1,2,3], sparse_table_2 gets accumulated IDs
                            # Rank 1: sparse_table_1 gets accumulated IDs, sparse_table_2 gets new f2 ID [0]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor(range(8)),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([0]),
                            },
                        },
                    ],
                ),
            ),
            (
                "EBC_and_delete_on_read_false",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_2",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f2"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=False,
                        consumers=["A", "B"],
                    ),
                    model_inputs=[
                        # First input: f1 has values 0,2,4,6 and f2 has values 8,10,12,14
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        # Second input: f1 has values 8,10,12,14 and f2 has values 0,2,4,6
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        # Third input: f1 has values 0,1,2,3 and f2 has values 4,5,6,7
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
                MultipleOutputTestParams(
                    consumer_access=["A", "B", "A"],
                    expected_outputs=[
                        # Expected output after first input: f1=[0,2,4,6] f2=[8,10,12,14] - Consumer A access
                        {
                            # Rank 0: sparse_table_1 gets f1 IDs [0,2,4,6], sparse_table_2 is empty
                            # Rank 1: sparse_table_1 is empty, sparse_table_2 gets f2 IDs [8,10,12,14]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([]),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor([]),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after second input: f1=[8,10,12,14] f2=[0,2,4,6] - Consumer B access
                        {
                            # Consumer B gets all accumulated data since last access (both inputs)
                            # Rank 0: Both tables have accumulated IDs from both inputs
                            # Rank 1: Both tables have accumulated IDs from both inputs
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor(range(8)),
                            },
                        },
                        # Expected output after third input: f1=[0,1,2,3] f2=[4,5,6,7] - Consumer A access
                        {
                            # Consumer A gets delta since last access (inputs 2 and 3)
                            # Rank 0: sparse_table_1 gets new f1 IDs [0,1,2,3], sparse_table_2 gets accumulated IDs
                            # Rank 1: sparse_table_1 gets accumulated IDs, sparse_table_2 gets new f2 ID [0]
                            "ebc.embedding_bags.sparse_table_1": {
                                0: torch.tensor(range(5)),
                                1: torch.tensor(range(8)),
                            },
                            "ebc.embedding_bags.sparse_table_2": {
                                0: torch.tensor(range(8)),
                                1: torch.tensor([0]),
                            },
                        },
                    ],
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    def test_multiple_consumers(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
        output_params: MultipleOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_multiple_consumer,
            world_size=self.world_size,
            test_params=test_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "EC_and_single_feature",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.MOMENTUM_LAST,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
            ),
            (
                "EBC_and_multiple_feature",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.MOMENTUM_LAST,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    def test_duplication_with_momentum(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_duplication_with_momentum,
            world_size=self.world_size,
            test_params=test_params,
        )

    @parameterized.expand(
        [
            (
                "EC_and_single_feature",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1"],
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.MOMENTUM_DIFF,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
            ),
            (
                "EBC_and_multiple_feature",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables=[
                        EmbeddingTableProps(
                            embedding_table_config=EmbeddingBagConfig(
                                name="sparse_table_1",
                                num_embeddings=NUM_EMBEDDINGS,
                                embedding_dim=EMBEDDING_DIM,
                                feature_names=["f1", "f2"],
                                pooling=PoolingType.SUM,
                            ),
                            sharding=ShardingType.ROW_WISE,
                        ),
                    ],
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ROWWISE_ADAGRAD,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 7, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 0, 0, 4, 4, 4, 8]),
                        ),
                    ],
                ),
            ),
        ]
    )
    @skip_if_asan
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    def test_duplication_with_rowwise_adagrad(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_duplication_with_rowwise_adagrad,
            world_size=self.world_size,
            test_params=test_params,
        )


def _test_fqn_to_feature_names(
    rank: int,
    world_size: int,
    input_params: ModelDeltaTrackerInputTestParams,
    output_params: FqnToFeatureNamesOutputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        dt_model, _ = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=input_params.embedding_config_type,
            tables=input_params.embedding_tables,
            config=input_params.model_tracker_config,
        )

        dt = dt_model.get_model_tracker()
        unittest.TestCase().assertEqual(
            dt.fqn_to_feature_names(), output_params.expected_fqn_to_feature_names
        )


def _test_tracker_init(
    rank: int,
    world_size: int,
    input_params: ModelDeltaTrackerInputTestParams,
    output_params: TrackerNotInitOutputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        dt_model, _ = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=input_params.embedding_config_type,
            tables=input_params.embedding_tables,
            config=None,
        )
        with unittest.TestCase().assertRaisesRegex(
            AssertionError,
            "Model tracker is not initialized. Add ModelTrackerConfig at DistributedModelParallel init.",
        ):
            getattr(dt_model, output_params.dmp_tracker_atter)()


def _test_id_mode(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
        )
        features_list = model_input_generator(test_params.model_inputs, rank)
        dt = dt_model.get_model_tracker()
        for features in features_list:
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()

        delta_ids = dt.get_delta_ids()

        table_fqns = dt.fqn_to_feature_names().keys()

        # Check if any table has multiple features
        has_multi_feature_tables = any(
            len(dt.fqn_to_feature_names()[fqn]) > 1 for fqn in table_fqns
        )

        table_fqns_list = list(table_fqns)

        if has_multi_feature_tables:
            # For multi-feature tables, each rank is responsible for one table
            # Rank 0 handles the first table, Rank 1 handles the second table
            if rank == 0:
                # Rank 0: First table has IDs, second table is empty
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqns_list[0]].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    )
                )
                unittest.TestCase().assertEqual(
                    0, delta_ids[table_fqns_list[1]].numel()
                )
            elif rank == 1:
                # Rank 1: Second table has IDs, first table is empty
                unittest.TestCase().assertEqual(
                    0, delta_ids[table_fqns_list[0]].numel()
                )
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqns_list[1]].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    )
                )
        else:
            # For single-feature tables, all tables have IDs on all ranks
            for table_fqn in table_fqns:
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqn].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    ),
                    f"Table {table_fqn} on rank {rank} should have IDs",
                )


def _test_embedding_mode(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
    output_params: EmbeddingModeOutputTestParams,
) -> None:

    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Initialize variables to None
        dt_model = None
        baseline_model = None
        if output_params.assert_str is not None:
            with unittest.TestCase().assertRaisesRegex(
                AssertionError,
                # pyre-ignore[6]
                output_params.assert_str,
            ):
                dt_model, baseline_model = get_models(
                    rank=rank,
                    world_size=world_size,
                    ctx=ctx,
                    embedding_config_type=test_params.embedding_config_type,
                    tables=test_params.embedding_tables,
                    config=test_params.model_tracker_config,
                )
        else:
            dt_model, baseline_model = get_models(
                rank=rank,
                world_size=world_size,
                ctx=ctx,
                embedding_config_type=test_params.embedding_config_type,
                tables=test_params.embedding_tables,
                config=test_params.model_tracker_config,
            )

            # Only proceed with the rest of the test if models were created successfully
            features_list = model_input_generator(test_params.model_inputs, rank)
            dt = dt_model.get_model_tracker()

            orig_emb1 = (
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_1.weight.detach().clone()
            )
            orig_emb2 = (
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_2.weight.detach().clone()
            )

            for features in features_list:
                tracked_out = dt_model(features)
                baseline_out = baseline_model(features)
                unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
                tracked_out.sum().backward()
                baseline_out.sum().backward()

            delta_rows = dt.get_delta()

            table_fqns = dt.fqn_to_feature_names().keys()
            table_fqns_list = list(table_fqns)

            # Check if any table has multiple features
            has_multi_feature_tables = any(
                len(dt.fqn_to_feature_names()[fqn]) > 1 for fqn in table_fqns
            )
            if has_multi_feature_tables:
                if rank == 0:
                    # Rank 0: First table has IDs and embeddings, second table is empty
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[0]].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqns_list[0]].states).allclose(
                            orig_emb1[expected_ids]
                        )
                    )
                    # Second table should be empty
                    unittest.TestCase().assertEqual(
                        0, delta_rows[table_fqns_list[1]].ids.numel()
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[1]].states is not None
                        # pyre-ignore[16]:
                        and delta_rows[table_fqns_list[1]].states.numel() == 0,
                    )
                elif rank == 1:
                    # Rank 1: Second table has IDs and embeddings, first table is empty
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[1]].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqns_list[1]].states).allclose(
                            orig_emb2[expected_ids]
                        )
                    )
                    # First table should be empty
                    unittest.TestCase().assertEqual(
                        0, delta_rows[table_fqns_list[0]].ids.numel()
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[0]].states is not None
                        and delta_rows[table_fqns_list[0]].states.numel() == 0,
                    )

            else:
                # For single-feature tables, all tables have IDs and embeddings on all ranks
                for table_fqn, orig_emb in zip(table_fqns, [orig_emb1, orig_emb2]):
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqn].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqn].states).allclose(
                            orig_emb[expected_ids]
                        )
                    )


def _test_multiple_get(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
    output_params: MultipleOutputTestParams,
) -> None:
    """
    Test that verifies the behavior of getting delta_rows multiple times with different inputs.
    This test processes multiple inputs and verifies that the delta tracker correctly
    accumulates delta_rows across multiple calls.
    """
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
        )
        features_list = model_input_generator(test_params.model_inputs, rank)
        dt = dt_model.get_model_tracker()
        table_fqns = dt.fqn_to_feature_names().keys()
        table_fqns_list = list(table_fqns)
        expected_emb1 = torch.tensor([])
        expected_emb2 = torch.tensor([])
        # Process each input and verify the unique IDs after each one
        for i, (features, expected_output) in enumerate(
            zip(features_list, output_params.expected_outputs)
        ):
            if test_params.embedding_config_type == EmbeddingConfig:
                # Embedding mode is only supported for EmbeddingCollection
                expected_emb1 = (
                    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                    dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_1.weight.detach().clone()
                )
                expected_emb2 = (
                    # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                    dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_2.weight.detach().clone()
                )

            # Process the input
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()
            delta_rows = dt.get_delta()

            # Verify that the current batch index is correct
            unittest.TestCase().assertTrue(dt.curr_batch_idx, i + 1)

            for table_fqn, expected_emb in zip(
                table_fqns_list, [expected_emb1, expected_emb2]
            ):
                expected_ids = torch.tensor(
                    expected_output[table_fqn][rank].detach().clone(),
                    dtype=torch.long,
                    device=torch.device(f"cuda:{rank}"),
                )
                # Verify that the delta rows match the expected output
                unittest.TestCase().assertTrue(
                    delta_rows[table_fqn].ids.allclose(expected_ids)
                )
                if test_params.embedding_config_type == EmbeddingConfig:
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqn].states).allclose(
                            expected_emb[expected_ids]
                        )
                    )


def _test_multiple_consumer(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
    output_params: MultipleOutputTestParams,
) -> None:
    """
    Test accessing delta rows with multiple consumers.

    This test verifies that multiple consumers can independently track and retrieve
    delta embedding using delta tracker. Each consumer maintains its own batch index
    and retrieval state, allowing them to get different delta data based on when they
    last accessed the tracker.

    """
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
        )
        features_list = model_input_generator(test_params.model_inputs, rank)
        dt = dt_model.get_model_tracker()
        table_fqns = dt.fqn_to_feature_names().keys()
        table_fqns_list = list(table_fqns)

        for i, (features, expected_output, consumer) in enumerate(
            zip(
                features_list,
                output_params.expected_outputs,
                output_params.consumer_access,
            )
        ):
            # Process the input
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()
            delta_rows = dt.get_delta_ids(consumer=consumer)

            # Verify that the current batch index is correct
            unittest.TestCase().assertTrue(dt.curr_batch_idx, i + 1)

            for table_fqn in table_fqns_list:
                expected_ids = torch.tensor(
                    expected_output[table_fqn][rank].detach().clone(),
                    dtype=torch.long,
                    device=torch.device(f"cuda:{rank}"),
                )
                returned = delta_rows[table_fqn]
                unittest.TestCase().assertTrue(
                    returned.shape == expected_ids.shape
                    and returned.allclose(expected_ids),
                    f"{i=}, {table_fqn=}, mismatch {returned=} vs {expected_ids=}",
                )


def _test_duplication_with_momentum(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
) -> None:
    """
    Test momentum tracking functionality in model delta tracker.

    Validates that the tracker correctly captures and stores momentum values from
    optimizer states when using TrackingMode.MOMENTUM_LAST mode.
    """
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
        )
        dt_model_opt = torch.optim.Adam(dt_model.parameters(), lr=0.1)
        baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.1)
        features_list = model_input_generator(test_params.model_inputs, rank)
        dt = dt_model.get_model_tracker()
        table_fqns = dt.fqn_to_feature_names().keys()
        table_fqns_list = list(table_fqns)
        for features in features_list:
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()
            dt_model_opt.step()
            baseline_opt.step()

        delta_rows = dt.get_delta()
        for table_fqn in table_fqns_list:
            ids = delta_rows[table_fqn].ids
            states = none_throws(delta_rows[table_fqn].states)

            unittest.TestCase().assertTrue(states is not None)
            unittest.TestCase().assertTrue(ids.numel() == states.numel())
            unittest.TestCase().assertTrue(bool((states != 0).all().item()))


def _test_duplication_with_rowwise_adagrad(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
            optimizer_type=OptimType.EXACT_ROWWISE_ADAGRAD,
        )

        # read momemtum directly from the table
        tbe: SplitTableBatchedEmbeddingBagsCodegen = (
            (
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no
                #  attribute `ec`.
                dt_model._dmp_wrapped_module.module.ec._lookups[0]
                ._emb_modules[0]
                .emb_module
            )
            if test_params.embedding_config_type == EmbeddingConfig
            else (
                dt_model._dmp_wrapped_module.module.ebc._lookups[0]  # pyre-ignore
                ._emb_modules[0]
                .emb_module
            )
        )
        assert isinstance(tbe, SplitTableBatchedEmbeddingBagsCodegen)
        start_momentums = tbe.split_optimizer_states()[0][0].detach().clone()

        dt_model_opt = torch.optim.Adam(dt_model.parameters(), lr=0.1)
        baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.1)
        features_list = model_input_generator(test_params.model_inputs, rank)

        dt = dt_model.get_model_tracker()
        table_fqns = dt.fqn_to_feature_names().keys()
        table_fqns_list = list(table_fqns)

        for features in features_list:
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()

            dt_model_opt.step()
            baseline_opt.step()

        end_momentums = tbe.split_optimizer_states()[0][0].detach().clone()

        delta_rows = dt.get_delta()
        table_fqn = table_fqns_list[0]

        ids = delta_rows[table_fqn].ids
        tracked_momentum = none_throws(delta_rows[table_fqn].states)
        unittest.TestCase().assertTrue(tracked_momentum is not None)
        unittest.TestCase().assertTrue(ids.numel() == tracked_momentum.numel())
        unittest.TestCase().assertTrue(bool((tracked_momentum != 0).all().item()))

        expected_momentum = end_momentums[ids] - start_momentums[ids]
        unittest.TestCase().assertTrue(tracked_momentum.allclose(expected_momentum))
