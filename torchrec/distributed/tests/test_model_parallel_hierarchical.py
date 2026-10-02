#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from typing import Any, cast, Dict, Optional, Tuple, Type

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import assume, given, Phase, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
    sharding_single_rank_test,
)
from torchrec.distributed.types import EmbeddingModuleShardingPlan, ShardingType
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import PoolingType
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class ModelParallelHierarchicalTest(ModelParallelTestShared):
    """
    Testing hierarchical sharding types.

    NOTE:
        Requires at least 4 GPUs to test.
    """

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        sharder_type=st.sampled_from(
            [
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.just(ShardingType.TABLE_ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        topology_domain=st.sampled_from([None, 1]),
        local_size=st.sampled_from([2]),
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
        pooling=st.sampled_from([PoolingType.SUM, PoolingType.MEAN]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=3,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharding_nccl_twrw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        topology_domain: int,
        local_size: int,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
    ) -> None:
        # Dense kernels do not have overlapped optimizer behavior yet
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )
        # Make sure detail debug will work with non-even collective
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        world_size = 4
        if topology_domain:
            # Need this to test topology group for TWRW
            os.environ["TOPOLOGY_DOMAIN_MULTIPLE"] = str(topology_domain)

        try:
            self._test_sharding(
                # pyrefly: ignore[bad-argument-type]
                sharders=[
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=torch.device("cuda"),
                    ),
                ],
                pod_size=topology_domain,
                backend="nccl",
                world_size=world_size,
                local_size=local_size,
                qcomms_config=qcomms_config,
                apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
                variable_batch_size=variable_batch_size,
                pooling=pooling,
            )
        finally:
            # Clean up to avoid leaking debug mode to subsequent tests,
            # which adds 3 extra Gloo collectives per NCCL collective and
            # causes timeouts.
            os.environ.pop("TORCH_DISTRIBUTED_DEBUG", None)
            os.environ.pop("TOPOLOGY_DOMAIN_MULTIPLE", None)

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        sharder_type=st.sampled_from(
            [
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        local_size=st.sampled_from([2]),
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
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=3,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharding_nccl_twcw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        local_size: int,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        # Dense kernels do not have overlapped optimizer behavior yet
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )
        world_size = 4
        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=world_size,
            local_size=local_size,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least three GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
            ]
        ),
        variable_batch_per_feature=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=2,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharding_empty_rank(
        self, sharding_type: str, variable_batch_per_feature: bool
    ) -> None:
        self._build_tables_and_groups()
        table = self.tables[0]
        embedding_groups = {"group_0": table.feature_names}
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=4,
            local_size=2,
            model_class=TestSparseNN,
            tables=[table],
            embedding_groups=embedding_groups,
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_BAG_COLLECTION.value,
                    sharding_type,
                    EmbeddingComputeKernel.FUSED.value,
                    device=torch.device("cuda"),
                )
            ],
            optim=EmbOptimType.EXACT_SGD,
            backend="nccl",
            constraints={table.name: ParameterConstraints(min_partition=4)},
            variable_batch_size=True,
            variable_batch_per_feature=variable_batch_per_feature,
            weighted_tables=None,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
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
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=1,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_embedding_tower_nccl(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        print(f"phases: {settings().phases}")
        # Dense kernels do not have overlapped optimizer behavior yet
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_TOWER.value,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            model_class=TestTowerSparseNN,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            atol=1e-4,
            rtol=1e-4,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
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
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=4,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_embedding_tower_collection_nccl(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )

        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_TOWER_COLLECTION.value,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            model_class=TestTowerCollectionSparseNN,
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        local_size=st.sampled_from([2]),
        global_constant_batch=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM, PoolingType.MEAN]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=2,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharding_variable_batch(
        self,
        sharding_type: str,
        local_size: int,
        global_constant_batch: bool,
        pooling: PoolingType,
    ) -> None:
        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_BAG_COLLECTION.value,
                    sharding_type,
                    EmbeddingComputeKernel.FUSED.value,
                    device=torch.device("cuda"),
                ),
            ],
            backend="nccl",
            world_size=4,
            local_size=local_size,
            variable_batch_per_feature=True,
            has_weighted_tables=False,
            global_constant_batch=global_constant_batch,
            pooling=pooling,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    @given(
        variable_batch_per_feature=st.booleans(),
        global_constant_batch=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=2,
        deadline=None,
        phases=[Phase.explicit, Phase.generate, Phase.target],
    )
    def test_sharding_grid_variable_batch(
        self,
        variable_batch_per_feature: bool,
        global_constant_batch: bool,
    ) -> None:
        # GRID_SHARD requires explicit per-table constraints; min_partition is
        # half of each table's embedding_dim (16, 24, 32, 40, 16, 24) so it
        # spans both of the 2 node-groups (world_size=4, local_size=2).
        constraints = {
            name: ParameterConstraints(
                min_partition=min_partition,
                sharding_types=[ShardingType.GRID_SHARD.value],
            )
            for name, min_partition in zip(self.table_names, [8, 12, 16, 20, 8, 12])
        }
        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_BAG_COLLECTION.value,
                    ShardingType.GRID_SHARD.value,
                    EmbeddingComputeKernel.FUSED.value,
                    device=torch.device("cuda"),
                ),
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            constraints=constraints,
            variable_batch_per_feature=variable_batch_per_feature,
            has_weighted_tables=False,
            global_constant_batch=global_constant_batch,
        )

    MIXED_NUM_TWRW_GROUPS: Dict[str, int] = {"table_0": 2, "table_3": 2}

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_twrw_num_twrw_groups_constraints_really_split(self) -> None:
        """Verify the constraints produce mixed single- and multi-group plans."""
        self._build_tables_and_groups()
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            embedding_groups=self.embedding_groups,
            sparse_device=torch.device("meta"),
            num_float_features=16,
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size=4,
                local_world_size=2,
                compute_device="cuda",
                ssd_cap=2 * 1024**4,
            ),
            constraints={
                name: ParameterConstraints(num_twrw_groups=num_twrw_groups)
                for name, num_twrw_groups in self.MIXED_NUM_TWRW_GROUPS.items()
            },
        )
        plan = planner.plan(
            module=model,
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_BAG_COLLECTION.value,
                    ShardingType.TABLE_ROW_WISE.value,
                    EmbeddingComputeKernel.FUSED.value,
                    device=torch.device("cuda"),
                ),
            ],
        )
        ebc_plan = cast(EmbeddingModuleShardingPlan, plan.plan["sparse.ebc"])

        for name in self.MIXED_NUM_TWRW_GROUPS:
            parameter_sharding = ebc_plan[name]
            self.assertEqual(
                parameter_sharding.num_twrw_groups,
                2,
                f"{name} is not multi-group",
            )
            ranks = none_throws(parameter_sharding.ranks)
            self.assertEqual(len(ranks), 4, f"{name} ranks: {ranks}")
            self.assertEqual({rank // 2 for rank in ranks}, {0, 1})

        for name in self.table_names:
            if name in self.MIXED_NUM_TWRW_GROUPS:
                continue
            parameter_sharding = ebc_plan[name]
            self.assertIsNone(parameter_sharding.num_twrw_groups, f"{name} opted in")
            ranks = none_throws(parameter_sharding.ranks)
            self.assertEqual(len({rank // 2 for rank in ranks}), 1, f"{name}: {ranks}")

    def _test_twrw_num_twrw_groups(
        self,
        num_twrw_groups_by_table: Dict[str, int],
        pooling: PoolingType = PoolingType.SUM,
        variable_batch_size: bool = False,
        variable_batch_per_feature: bool = False,
    ) -> None:
        """Compare a two-group TWRW model with an unsharded reference."""
        self._test_sharding(
            # pyrefly: ignore[bad-argument-type]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_BAG_COLLECTION.value,
                    ShardingType.TABLE_ROW_WISE.value,
                    EmbeddingComputeKernel.FUSED.value,
                    device=torch.device("cuda"),
                ),
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            constraints={
                name: ParameterConstraints(num_twrw_groups=num_twrw_groups)
                for name, num_twrw_groups in num_twrw_groups_by_table.items()
            },
            variable_batch_size=variable_batch_size,
            variable_batch_per_feature=variable_batch_per_feature,
            has_weighted_tables=not variable_batch_per_feature,
            pooling=pooling,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharding_nccl_twrw_mixed_num_twrw_groups(self) -> None:
        # Includes a feature shared by single- and multi-group tables.
        self._test_twrw_num_twrw_groups(self.MIXED_NUM_TWRW_GROUPS)

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharding_nccl_twrw_all_num_twrw_groups(self) -> None:
        self._test_twrw_num_twrw_groups(dict.fromkeys(self.table_names, 2))

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharding_nccl_twrw_mixed_num_twrw_groups_mean_pooling(self) -> None:
        self._test_twrw_num_twrw_groups(
            self.MIXED_NUM_TWRW_GROUPS, pooling=PoolingType.MEAN
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharding_nccl_twrw_mixed_num_twrw_groups_variable_batch(self) -> None:
        self._test_twrw_num_twrw_groups(
            self.MIXED_NUM_TWRW_GROUPS, variable_batch_size=True
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    def test_sharding_nccl_twrw_mixed_num_twrw_groups_variable_batch_per_feature(
        self,
    ) -> None:
        self._test_twrw_num_twrw_groups(
            self.MIXED_NUM_TWRW_GROUPS,
            variable_batch_size=True,
            variable_batch_per_feature=True,
        )
