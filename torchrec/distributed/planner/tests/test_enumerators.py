#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import cast, List
from unittest.mock import MagicMock, patch

import torch
from torchrec.distributed.embedding_tower_sharding import (
    EmbeddingTowerCollectionSharder,
    EmbeddingTowerSharder,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.mc_embeddingbag import (
    ManagedCollisionEmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.constants import BIGINT_DTYPE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    _calculate_dp_shard_io_sizes,
    _calculate_tw_shard_io_sizes,
)
from torchrec.distributed.planner.types import ParameterConstraints, Storage, Topology
from torchrec.distributed.planner.utils import prod
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

EXPECTED_RW_SHARD_SIZES = [
    [[13, 20], [13, 20], [13, 20], [13, 20], [13, 20], [13, 20], [13, 20], [9, 20]],
    [[14, 40], [14, 40], [14, 40], [14, 40], [14, 40], [14, 40], [14, 40], [12, 40]],
    [[15, 60], [15, 60], [15, 60], [15, 60], [15, 60], [15, 60], [15, 60], [15, 60]],
    [[17, 80], [17, 80], [17, 80], [17, 80], [17, 80], [17, 80], [17, 80], [11, 80]],
]

EXPECTED_RW_SHARD_SIZES_WITH_BUCKETS = [
    [[20, 20], [20, 20], [10, 20], [10, 20], [10, 20], [10, 20], [10, 20], [10, 20]],
    [[22, 40], [22, 40], [11, 40], [11, 40], [11, 40], [11, 40], [11, 40], [11, 40]],
    [[24, 60], [24, 60], [12, 60], [12, 60], [12, 60], [12, 60], [12, 60], [12, 60]],
    [[26, 80], [26, 80], [13, 80], [13, 80], [13, 80], [13, 80], [13, 80], [13, 80]],
]

EXPECTED_RW_SHARD_OFFSETS = [
    [[0, 0], [13, 0], [26, 0], [39, 0], [52, 0], [65, 0], [78, 0], [91, 0]],
    [[0, 0], [14, 0], [28, 0], [42, 0], [56, 0], [70, 0], [84, 0], [98, 0]],
    [[0, 0], [15, 0], [30, 0], [45, 0], [60, 0], [75, 0], [90, 0], [105, 0]],
    [[0, 0], [17, 0], [34, 0], [51, 0], [68, 0], [85, 0], [102, 0], [119, 0]],
]

EXPECTED_RW_SHARD_OFFSETS_WITH_BUCKETS = [
    [[0, 0], [20, 0], [40, 0], [50, 0], [60, 0], [70, 0], [80, 0], [90, 0]],
    [[0, 0], [22, 0], [44, 0], [55, 0], [66, 0], [77, 0], [88, 0], [99, 0]],
    [[0, 0], [24, 0], [48, 0], [60, 0], [72, 0], [84, 0], [96, 0], [108, 0]],
    [[0, 0], [26, 0], [52, 0], [65, 0], [78, 0], [91, 0], [104, 0], [117, 0]],
]


def get_expected_cache_aux_size(rows: int) -> int:
    # 0.2 is the hardcoded cache load factor assumed in this test
    return math.ceil(rows * (4 + 0.2 * 16))


EXPECTED_RW_SHARD_STORAGE = [
    [
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166928, ddr=0),
        Storage(hbm=166608, ddr=0),
    ],
    [
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003712, ddr=0),
        Storage(hbm=1003392, ddr=0),
    ],
    [
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
        Storage(hbm=1007120, ddr=0),
    ],
    [
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2653504, ddr=0),
        Storage(hbm=2651584, ddr=0),
    ],
]

EXPECTED_VIRTUAL_TABLE_RW_SHARD_STORAGE_WITH_BUCKETS = [
    [
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
        Storage(hbm=165888, ddr=0),
    ],
    [
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
        Storage(hbm=1001472, ddr=0),
    ],
    [
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
        Storage(hbm=1003520, ddr=0),
    ],
    [
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
        Storage(hbm=2648064, ddr=0),
    ],
]

EXPECTED_UVM_CACHING_RW_SHARD_STORAGE = [
    [
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166096 + get_expected_cache_aux_size(13), ddr=1040),
        Storage(hbm=166032 + get_expected_cache_aux_size(9), ddr=720),
    ],
    [
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001920 + get_expected_cache_aux_size(14), ddr=2240),
        Storage(hbm=1001856 + get_expected_cache_aux_size(12), ddr=1920),
    ],
    [
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
        Storage(hbm=1004240 + get_expected_cache_aux_size(15), ddr=3600),
    ],
    [
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2649152 + get_expected_cache_aux_size(17), ddr=5440),
        Storage(hbm=2648768 + get_expected_cache_aux_size(11), ddr=3520),
    ],
]

EXPECTED_UVM_CACHING_RW_SHARD_STORAGE_WITH_BUCKETS = [
    [
        Storage(hbm=166352, ddr=1600),
        Storage(hbm=166352, ddr=1600),
        Storage(hbm=166120, ddr=800),
        Storage(hbm=166120, ddr=800),
        Storage(hbm=166120, ddr=800),
        Storage(hbm=166120, ddr=800),
        Storage(hbm=166120, ddr=800),
        Storage(hbm=166120, ddr=800),
    ],
    [
        Storage(hbm=1002335, ddr=3520),
        Storage(hbm=1002335, ddr=3520),
        Storage(hbm=1001904, ddr=1760),
        Storage(hbm=1001904, ddr=1760),
        Storage(hbm=1001904, ddr=1760),
        Storage(hbm=1001904, ddr=1760),
        Storage(hbm=1001904, ddr=1760),
        Storage(hbm=1001904, ddr=1760),
    ],
    [
        Storage(hbm=1004845, ddr=5760),
        Storage(hbm=1004845, ddr=5760),
        Storage(hbm=1004183, ddr=2880),
        Storage(hbm=1004183, ddr=2880),
        Storage(hbm=1004183, ddr=2880),
        Storage(hbm=1004183, ddr=2880),
        Storage(hbm=1004183, ddr=2880),
        Storage(hbm=1004183, ddr=2880),
    ],
    [
        Storage(hbm=2649916, ddr=8320),
        Storage(hbm=2649916, ddr=8320),
        Storage(hbm=2648990, ddr=4160),
        Storage(hbm=2648990, ddr=4160),
        Storage(hbm=2648990, ddr=4160),
        Storage(hbm=2648990, ddr=4160),
        Storage(hbm=2648990, ddr=4160),
        Storage(hbm=2648990, ddr=4160),
    ],
]

EXPECTED_TWRW_SHARD_SIZES = [
    [[25, 20], [25, 20], [25, 20], [25, 20]],
    [[28, 40], [28, 40], [28, 40], [26, 40]],
    [[30, 60], [30, 60], [30, 60], [30, 60]],
    [[33, 80], [33, 80], [33, 80], [31, 80]],
]

EXPECTED_TWRW_SHARD_OFFSETS = [
    [[0, 0], [25, 0], [50, 0], [75, 0]],
    [[0, 0], [28, 0], [56, 0], [84, 0]],
    [[0, 0], [30, 0], [60, 0], [90, 0]],
    [[0, 0], [33, 0], [66, 0], [99, 0]],
]

EXPECTED_TWRW_SHARD_STORAGE = [
    [
        Storage(hbm=169936, ddr=0),
        Storage(hbm=169936, ddr=0),
        Storage(hbm=169936, ddr=0),
        Storage(hbm=169936, ddr=0),
    ],
    [
        Storage(hbm=1024384, ddr=0),
        Storage(hbm=1024384, ddr=0),
        Storage(hbm=1024384, ddr=0),
        Storage(hbm=1024064, ddr=0),
    ],
    [
        Storage(hbm=1031200, ddr=0),
        Storage(hbm=1031200, ddr=0),
        Storage(hbm=1031200, ddr=0),
        Storage(hbm=1031200, ddr=0),
    ],
    [
        Storage(hbm=2685248, ddr=0),
        Storage(hbm=2685248, ddr=0),
        Storage(hbm=2685248, ddr=0),
        Storage(hbm=2684608, ddr=0),
    ],
]

EXPECTED_CW_SHARD_SIZES = [
    [[100, 20]],
    [[110, 20], [110, 20]],
    [[120, 20], [120, 20], [120, 20]],
    [[130, 40], [130, 40]],
]

EXPECTED_CW_SHARD_OFFSETS = [
    [[0, 0]],
    [[0, 0], [0, 20]],
    [[0, 0], [0, 20], [0, 40]],
    [[0, 0], [0, 40]],
]

EXPECTED_CW_SHARD_STORAGE = [
    [Storage(hbm=188224, ddr=0)],
    [Storage(hbm=647776, ddr=0), Storage(hbm=647776, ddr=0)],
    [
        Storage(hbm=501120, ddr=0),
        Storage(hbm=501120, ddr=0),
        Storage(hbm=501120, ddr=0),
    ],
    [Storage(hbm=1544512, ddr=0), Storage(hbm=1544512, ddr=0)],
]

EXPECTED_TWCW_SHARD_SIZES: List[List[List[int]]] = EXPECTED_CW_SHARD_SIZES

EXPECTED_TWCW_SHARD_OFFSETS: List[List[List[int]]] = EXPECTED_CW_SHARD_OFFSETS

EXPECTED_TWCW_SHARD_STORAGE = [
    [Storage(hbm=188224, ddr=0)],
    [Storage(hbm=647776, ddr=0), Storage(hbm=647776, ddr=0)],
    [
        Storage(hbm=501120, ddr=0),
        Storage(hbm=501120, ddr=0),
        Storage(hbm=501120, ddr=0),
    ],
    [Storage(hbm=1544512, ddr=0), Storage(hbm=1544512, ddr=0)],
]


class TWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class RWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class VirtualTableRWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value]


class UVMCachingRWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED_UVM_CACHING.value]


class TWRWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class CWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TWCWSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class DPSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.DATA_PARALLEL.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class AllTypesSharder(EmbeddingBagCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.DENSE.value,
            EmbeddingComputeKernel.FUSED.value,
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            EmbeddingComputeKernel.QUANT.value,
        ]


class TowerTWRWSharder(EmbeddingTowerSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TowerCollectionTWRWSharder(EmbeddingTowerCollectionSharder):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestEnumerators(unittest.TestCase):
    def setUp(self) -> None:
        self.compute_device = "cuda"
        self.batch_size = 256
        self.world_size = 8
        self.local_world_size = 4
        self.constraints = {
            "table_0": ParameterConstraints(min_partition=20),
            "table_1": ParameterConstraints(
                min_partition=20, pooling_factors=[1, 3, 5]
            ),
            "table_2": ParameterConstraints(min_partition=20, pooling_factors=[8, 2]),
            "table_3": ParameterConstraints(
                min_partition=40, pooling_factors=[2, 1, 3, 7]
            ),
        }
        self._virtual_table_constraints = {
            "table_0": ParameterConstraints(
                min_partition=20,
                compute_kernels=[EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value],
            ),
            "table_1": ParameterConstraints(
                min_partition=20,
                pooling_factors=[1, 3, 5],
                compute_kernels=[EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value],
            ),
            "table_2": ParameterConstraints(
                min_partition=20,
                pooling_factors=[8, 2],
                compute_kernels=[EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value],
            ),
            "table_3": ParameterConstraints(
                min_partition=40,
                pooling_factors=[2, 1, 3, 7],
                compute_kernels=[EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value],
            ),
        }
        self.num_tables = 4
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i * 10,
                embedding_dim=20 + i * 20,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(self.num_tables)
        ]
        tables_with_buckets = [
            EmbeddingBagConfig(
                num_embeddings=100 + i * 10,
                embedding_dim=20 + i * 20,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                total_num_buckets=10,
                use_virtual_table=True,
            )
            for i in range(self.num_tables)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.model_with_buckets = EmbeddingBagCollection(
            tables=tables_with_buckets,
        )
        self.enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            batch_size=self.batch_size,
            constraints=self.constraints,
        )
        self.virtual_table_enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            batch_size=self.batch_size,
            constraints=self._virtual_table_constraints,
        )
        self.tower_model = TestTowerSparseNN(
            tables=tables, weighted_tables=weighted_tables
        )
        self.tower_collection_model = TestTowerCollectionSparseNN(
            tables=tables, weighted_tables=weighted_tables
        )
        _get_optimizer_multipler_patcher = patch(
            "torchrec.distributed.planner.shard_estimators._get_optimizer_multipler",
            return_value=0,
        )
        self.addCleanup(_get_optimizer_multipler_patcher.stop)
        self._get_optimizer_multipler_mock = _get_optimizer_multipler_patcher.start()

    def test_dp_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], DPSharder())]
        )

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.DATA_PARALLEL.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                [list(sharding_option.tensor.shape)] * self.world_size,
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                [[0, 0]] * self.world_size,
            )

            input_data_type_size = BIGINT_DTYPE
            output_data_type_size = sharding_option.tensor.element_size()

            input_sizes, output_sizes = _calculate_dp_shard_io_sizes(
                batch_sizes=[self.batch_size] * sharding_option.num_inputs,
                input_lengths=self.constraints[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                num_shards=self.world_size,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                is_pooled=sharding_option.is_pooled,
                num_poolings=[1.0] * sharding_option.num_inputs,
            )

            tensor_sizes = [
                prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            ] * self.world_size

            storage_sizes = [
                input_size + tensor_size + output_size
                for input_size, tensor_size, output_size in zip(
                    input_sizes,
                    tensor_sizes,
                    output_sizes,
                )
            ]

            expected_storage = [
                Storage(hbm=storage_size, ddr=0) for storage_size in storage_sizes
            ]
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards], expected_storage
            )

    def test_tw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], TWSharder())]
        )

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_WISE.value
            )
            self.assertEqual(
                sharding_option.shards[0].size, list(sharding_option.tensor.shape)
            )
            self.assertEqual(sharding_option.shards[0].offset, [0, 0])

            input_data_type_size = BIGINT_DTYPE
            output_data_type_size = sharding_option.tensor.element_size()

            input_sizes, output_sizes = _calculate_tw_shard_io_sizes(
                batch_sizes=[self.batch_size] * sharding_option.num_inputs,
                world_size=self.world_size,
                input_lengths=self.constraints[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                is_pooled=sharding_option.is_pooled,
                num_poolings=[1.0] * sharding_option.num_inputs,
            )

            tensor_size = (
                prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            )
            optimizer_size = 0

            storage_size = (
                input_sizes[0] + output_sizes[0] + tensor_size + optimizer_size
            )

            self.assertEqual(
                sharding_option.shards[0].storage, Storage(hbm=storage_size, ddr=0)
            )

    def test_rw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], RWSharder())]
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_STORAGE[i],
            )

    def test_virtual_table_rw_sharding_with_buckets(self) -> None:
        sharding_options = self.virtual_table_enumerator.enumerate(
            self.model_with_buckets,
            [cast(ModuleSharder[torch.nn.Module], VirtualTableRWSharder())],
        )
        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES_WITH_BUCKETS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS_WITH_BUCKETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_VIRTUAL_TABLE_RW_SHARD_STORAGE_WITH_BUCKETS[i],
            )

    def test_uvm_caching_rw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model,
            [cast(ModuleSharder[torch.nn.Module], UVMCachingRWSharder())],
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_UVM_CACHING_RW_SHARD_STORAGE[i],
            )

    def test_uvm_caching_rw_sharding_with_buckets(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model_with_buckets,
            [cast(ModuleSharder[torch.nn.Module], UVMCachingRWSharder())],
        )
        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES_WITH_BUCKETS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS_WITH_BUCKETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_UVM_CACHING_RW_SHARD_STORAGE_WITH_BUCKETS[i],
            )

    def test_twrw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], TWRWSharder())]
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_ROW_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_STORAGE[i],
            )

    def test_cw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], CWSharder())]
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.COLUMN_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_STORAGE[i],
            )

    def test_twcw_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.model, [cast(ModuleSharder[torch.nn.Module], TWCWSharder())]
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_COLUMN_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_STORAGE[i],
            )

    def test_filtering(self) -> None:
        constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
            compute_kernels=[
                EmbeddingComputeKernel.FUSED_UVM.value,
                EmbeddingComputeKernel.DENSE.value,
            ],
        )
        constraints = {
            "table_0": constraint,
            "table_1": constraint,
            "table_2": constraint,
            "table_3": constraint,
        }

        enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            batch_size=self.batch_size,
            constraints=constraints,
        )
        sharder = cast(ModuleSharder[torch.nn.Module], AllTypesSharder())

        sharding_options = enumerator.enumerate(self.model, [sharder])

        expected_sharding_types = {
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        }
        expected_compute_kernels = {
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.DENSE.value,
        }
        unexpected_sharding_types = (
            set(sharder.sharding_types(self.compute_device)) - expected_sharding_types
        )
        unexpected_compute_kernels = (
            set(sharder.compute_kernels("", "")) - expected_compute_kernels
        )

        self.assertEqual(
            len(sharding_options),
            self.num_tables
            * len(expected_sharding_types)
            * len(expected_compute_kernels),
        )

        for sharding_option in sharding_options:
            self.assertIn(sharding_option.sharding_type, expected_sharding_types)
            self.assertNotIn(sharding_option.sharding_type, unexpected_sharding_types)
            self.assertIn(sharding_option.compute_kernel, expected_compute_kernels)
            self.assertNotIn(sharding_option.compute_kernel, unexpected_compute_kernels)

    def test_filter_sharding_types_ebc(self) -> None:
        constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.TABLE_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = EmbeddingBagCollectionSharder()
        allowed_sharding_types = enumerator._filter_sharding_types(
            "table_0", sharder.sharding_types("cuda")
        )

        self.assertEqual(
            set(allowed_sharding_types),
            {
                ShardingType.TABLE_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            },
        )

    def test_filter_sharding_types_mch_ebc(self) -> None:
        constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = ManagedCollisionEmbeddingBagCollectionSharder()
        allowed_sharding_types = enumerator._filter_sharding_types(
            "table_0", sharder.sharding_types("cuda")
        )

        self.assertEqual(
            set(allowed_sharding_types),
            {
                ShardingType.ROW_WISE.value,
            },
        )

    def test_filter_sharding_types_mch_ebc_no_available(self) -> None:
        constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = ManagedCollisionEmbeddingBagCollectionSharder()
        with self.assertLogs(level="WARNING"):
            allowed_sharding_types = enumerator._filter_sharding_types(
                "table_0", sharder.sharding_types("cuda")
            )

        self.assertEqual(allowed_sharding_types, [])

    def test_filter_compute_kernels_ebc(self) -> None:
        constraint = ParameterConstraints(
            compute_kernels=[
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = EmbeddingBagCollectionSharder()
        sharding_type = ShardingType.ROW_WISE.value
        allowed_compute_kernels = enumerator._filter_compute_kernels(
            "table_0", sharder.compute_kernels(sharding_type, "cuda"), sharding_type
        )

        self.assertEqual(
            set(allowed_compute_kernels),
            {
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            },
        )

    def test_filter_compute_kernels_mch_ebc(self) -> None:
        constraint = ParameterConstraints(
            compute_kernels=[
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = ManagedCollisionEmbeddingBagCollectionSharder()
        sharding_type = ShardingType.ROW_WISE.value
        allowed_compute_kernels = enumerator._filter_compute_kernels(
            "table_0", sharder.compute_kernels(sharding_type, "cuda"), sharding_type
        )

        self.assertEqual(
            set(allowed_compute_kernels),
            {
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            },
        )

    def test_filter_compute_kernels_mch_ebc_no_available(self) -> None:
        constraint = ParameterConstraints(
            compute_kernels=[
                EmbeddingComputeKernel.DENSE.value,
            ],
        )
        constraints = {"table_0": constraint}
        enumerator = EmbeddingEnumerator(
            topology=MagicMock(),
            batch_size=MagicMock(),
            constraints=constraints,
        )

        sharder = ManagedCollisionEmbeddingBagCollectionSharder()
        sharding_type = ShardingType.ROW_WISE.value
        with self.assertLogs(level="WARNING"):
            allowed_compute_kernels = enumerator._filter_compute_kernels(
                "table_0", sharder.compute_kernels(sharding_type, "cuda"), sharding_type
            )

        self.assertEqual(allowed_compute_kernels, [])

    def test_tower_sharding(self) -> None:
        # five tables
        # tower_0: tables[2], tables[3]
        # tower_1: tables[0]
        # sparse_arch:
        #    ebc:
        #      tables[1]
        #      weighted_tables[0]
        sharding_options = self.enumerator.enumerate(
            self.tower_model,
            [
                cast(ModuleSharder[torch.nn.Module], TWRWSharder()),
                cast(ModuleSharder[torch.nn.Module], TowerTWRWSharder()),
            ],
        )
        self.assertEqual(len(sharding_options), 5)

        self.assertEqual(sharding_options[0].dependency, None)
        self.assertEqual(sharding_options[0].module[0], "sparse_arch.weighted_ebc")
        self.assertEqual(sharding_options[1].dependency, None)
        self.assertEqual(sharding_options[1].module[0], "sparse_arch.ebc")
        self.assertEqual(sharding_options[2].dependency, "tower_1")
        self.assertEqual(sharding_options[2].module[0], "tower_1")
        self.assertEqual(sharding_options[3].dependency, "tower_0")
        self.assertEqual(sharding_options[3].module[0], "tower_0")
        self.assertEqual(sharding_options[4].dependency, "tower_0")
        self.assertEqual(sharding_options[4].module[0], "tower_0")

    def test_tower_collection_sharding(self) -> None:
        sharding_options = self.enumerator.enumerate(
            self.tower_collection_model,
            [
                cast(ModuleSharder[torch.nn.Module], TowerCollectionTWRWSharder()),
                cast(ModuleSharder[torch.nn.Module], TowerTWRWSharder()),
            ],
        )
        self.assertEqual(len(sharding_options), 4)

        # table_0
        self.assertEqual(sharding_options[0].dependency, "tower_arch.tower_0")
        self.assertEqual(sharding_options[0].module[0], "tower_arch")
        # table_2
        self.assertEqual(sharding_options[1].dependency, "tower_arch.tower_0")
        self.assertEqual(sharding_options[1].module[0], "tower_arch")
        # table_1
        self.assertEqual(sharding_options[2].dependency, "tower_arch.tower_1")
        self.assertEqual(sharding_options[2].module[0], "tower_arch")
        # weighted_table_0
        self.assertEqual(sharding_options[3].dependency, "tower_arch.tower_2")
        self.assertEqual(sharding_options[3].module[0], "tower_arch")

    def test_empty(self) -> None:
        sharding_options = self.enumerator.enumerate(self.model, sharders=[])
        self.assertFalse(sharding_options)

    def test_throw_ex_no_sharding_option_for_table(self) -> None:
        cw_constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.COLUMN_WISE.value,
            ],
            compute_kernels=[
                EmbeddingComputeKernel.FUSED.value,
            ],
        )

        rw_constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.TABLE_ROW_WISE.value,
            ],
            compute_kernels=[
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
            ],
        )

        constraints = {
            "table_0": cw_constraint,
            "table_1": rw_constraint,
            "table_2": cw_constraint,
            "table_3": cw_constraint,
        }

        enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            batch_size=self.batch_size,
            constraints=constraints,
        )

        sharder = cast(ModuleSharder[torch.nn.Module], CWSharder())

        with self.assertRaises(Exception) as context:
            _ = enumerator.enumerate(self.model, [sharder])

        self.assertEqual(
            str(context.exception),
            "No available sharding type and compute kernel combination after applying user provided constraints for table_1. "
            "Module: torchrec.modules.embedding_modules.EmbeddingBagCollection, sharder: CWSharder, compute device: cuda. "
            "To debug, search above for warning logs about no available sharding types/compute kernels for table: table_1",
        )
