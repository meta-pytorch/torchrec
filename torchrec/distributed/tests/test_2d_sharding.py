#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import assume, given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.test_utils.test_model import (
    TestMixedEmbeddingSparseArch,
    TestSparseNNBase,
)
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
    sharding_single_rank_test,
)
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import (
    DMPCollectionConfig,
    ModuleSharder,
    ShardingStrategy,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class TestEmbeddingBagCollection2DParallel(ModelParallelTestShared):
    """
    Tests for 2D parallelism of embeddingbagcollection tables
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self, backend: str = "nccl") -> None:
        super().setUp(backend=backend)

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                # QCommsConfig(
                #     forward_precision=CommType.FP16, backward_precision=CommType.BF16
                # ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                # None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_cw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.COLUMN_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_tw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.TABLE_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE_2D // 2,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=2)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                # QCommsConfig(
                #     forward_precision=CommType.FP16, backward_precision=CommType.BF16
                # ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                # None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_grid_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.GRID_SHARD.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE // 4,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                "table_0": ParameterConstraints(
                    min_partition=8, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_1": ParameterConstraints(
                    min_partition=12, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_2": ParameterConstraints(
                    min_partition=8, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_3": ParameterConstraints(
                    min_partition=10, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_4": ParameterConstraints(
                    min_partition=4, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_5": ParameterConstraints(
                    min_partition=6, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "weighted_table_0": ParameterConstraints(
                    min_partition=2, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "weighted_table_1": ParameterConstraints(
                    min_partition=3, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_rw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if self.backend == "gloo":
            self.skipTest(
                "Gloo reduce_scatter_base fallback not supported with async_op=True"
            )

        sharding_type = ShardingType.ROW_WISE.value
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_twrw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.TABLE_ROW_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE // 4,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=2)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
        )


@skip_if_asan_class
class TestEmbeddingCollection2DParallel(MultiProcessTestBase):
    """
    Tests for 2D parallelism of embeddingcollection tables
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables = initial_tables + shared_features_tables
        self.shared_features = [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups = {
            "group_0": [
                (
                    f"{feature}@{table.name}"
                    if feature in self.shared_features
                    else feature
                )
                for table in self.tables
                for feature in table.feature_names
            ]
        }

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_ec_rw_2D(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.COLUMN_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_ec_cw_2D(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.TABLE_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_ec_tw_2D(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=2)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        world_size_2D: int = 1,
        local_size: Optional[int] = None,
        node_group_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            world_size_2D=world_size_2D,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=True,
        )


class TestDynamic2DParallel(MultiProcessTestBase):
    """
    Tests for dynamic 2D parallelism
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self) -> None:
        super().setUp()

        num_ec_features = 2
        num_ebc_features = 2
        num_features = num_ec_features + num_ebc_features

        ec_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_ec_features)
        ]

        ebc_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 8,
                name="table_" + str(i + num_ec_features),
                feature_names=["feature_" + str(i)],
                data_type=DataType.FP32,
            )
            for i in range(num_features)
        ]

        self.tables = ec_tables + ebc_tables

        self.embedding_groups = {
            "group_0": [
                (feature) for table in self.tables for feature in table.feature_names
            ]
        }

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_dynamic_2D(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )

        # add sharding plan for embedding collection later
        ec_submodule_config = DMPCollectionConfig(
            module=EmbeddingCollection,
            sharding_group_size=2,
            plan=None,  # pyre-ignore[6]
        )

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[  # pyre-ignore[6]
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        SharderType.EMBEDDING_BAG_COLLECTION.value,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=torch.device("cuda"),
                    ),
                ),
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=2)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            submodule_configs=[ec_submodule_config],
        )

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        world_size_2D: int = 1,
        local_size: Optional[int] = None,
        node_group_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestMixedEmbeddingSparseArch,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
        submodule_configs: Optional[List[DMPCollectionConfig]] = None,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            world_size_2D=world_size_2D,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=True,
            submodule_configs=submodule_configs,
        )


class TestFullySharded2DEBCParallel(ModelParallelTestShared):
    """
    Tests for hybrid sharded 2D parallelism
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self, backend: str = "nccl") -> None:
        super().setUp(backend=backend)

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_cw(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.COLUMN_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_rw(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.ROW_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_twrw(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.TABLE_ROW_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE // 4,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
        use_inter_host_allreduce=st.booleans(),
        custom_all_reduce=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_tw(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
        use_inter_host_allreduce: bool,
        custom_all_reduce: bool,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.TABLE_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE_2D // 2,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
            use_inter_host_allreduce=use_inter_host_allreduce,
            custom_all_reduce=custom_all_reduce,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )


@skip_if_asan_class
class TestFullySharded2DECParallel(MultiProcessTestBase):
    """
    Tests for fully sharded 2D parallelism of embeddingcollection tables
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables = initial_tables + shared_features_tables
        self.shared_features = [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups = {
            "group_0": [
                (
                    f"{feature}@{table.name}"
                    if feature in self.shared_features
                    else feature
                )
                for table in self.tables
                for feature in table.feature_names
            ]
        }

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.COLUMN_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_sequence_cw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_sequence_rw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    @given(
        sharding_type=st.just(ShardingType.TABLE_WISE.value),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_fully_sharded_sequence_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            sharding_strategy=ShardingStrategy.FULLY_SHARDED,
        )

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        world_size_2D: int = 1,
        local_size: Optional[int] = None,
        node_group_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
        sharding_strategy: Optional[ShardingStrategy] = None,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            world_size_2D=world_size_2D,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            global_constant_batch=True,
            sharding_strategy=sharding_strategy,
        )
