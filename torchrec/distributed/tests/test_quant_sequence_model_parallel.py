#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import cast, List, Optional, Type

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.cache.kv_embedding_ops_inference import KVEmbeddingInference
from hypothesis import given, settings, Verbosity
from torch import nn, quantization as quant
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.quant_embedding import QuantEmbeddingCollectionSharder
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNNBase
from torchrec.distributed.test_utils.test_model_parallel_base import (
    InferenceModelParallelTestBase,
)
from torchrec.distributed.tests.test_sequence_model import TestSequenceSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig, NoEvictionPolicy
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    quant_prep_enable_quant_state_dict_split_scale_bias_for_types,
)
from torchrec.test_utils import skip_if_asan_class


def _quantize(
    module: nn.Module, quant_state_dict_split_scale_bias: bool = False
) -> nn.Module:
    if quant_state_dict_split_scale_bias:
        quant_prep_enable_quant_state_dict_split_scale_bias_for_types(
            module, [EmbeddingCollection]
        )
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            EmbeddingCollection: qconfig,
        },
        mapping={
            EmbeddingCollection: QuantEmbeddingCollection,
        },
        inplace=True,
    )


class TestQuantECSharder(QuantEmbeddingCollectionSharder):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        super().__init__()
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


@skip_if_asan_class
class QuantSequenceModelParallelTest(InferenceModelParallelTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.QUANT.value,
            ]
        ),
        quant_state_dict_split_scale_bias=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        quant_state_dict_split_scale_bias: bool,
    ) -> None:
        self._test_sharding(
            sharders=[
                TestQuantECSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

    def setUp(self) -> None:
        super().setUp()

        num_features = 4

        self.tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.embedding_groups = {
            "group_0": ["feature_" + str(i) for i in range(num_features)]
        }

    def _test_sharding(
        self,
        sharders: List[TestQuantECSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        quant_state_dict_split_scale_bias: bool = False,
    ) -> None:
        self._test_sharded_forward(
            world_size=world_size,
            model_class=cast(TestSparseNNBase, TestSequenceSparseNN),
            embedding_groups=self.embedding_groups,
            tables=self.tables,
            sharders=sharders,
            quantize_callable=_quantize,
            quantize_callable_kwargs={
                "quant_state_dict_split_scale_bias": quant_state_dict_split_scale_bias
            },
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    @given(
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        quant_state_dict_split_scale_bias=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_quant_pred_shard(
        self, output_type: torch.dtype, quant_state_dict_split_scale_bias: bool
    ) -> None:
        device = torch.device("cuda:0")

        # wrap in sequential because _quantize only applies to submodules...
        model = nn.Sequential(EmbeddingCollection(tables=self.tables, device=device))

        quant_model = _quantize(model)

        sharded_quant_model = _shard_modules(
            module=quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantECSharder(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        sharded_quant_model.load_state_dict(sharded_quant_model.state_dict())

        local_batch, _ = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=[],
            indices_dtype=torch.int32,
            lengths_dtype=torch.int32,
        )
        local_batch = local_batch.to(device)
        sharded_quant_model(local_batch.idlist_features)

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_sharded_quant_kv_zch(self) -> None:
        device = torch.device("cuda:0")
        num_features = 4

        tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                use_virtual_table=True if i % 2 == 0 else False,
                virtual_table_eviction_policy=(
                    NoEvictionPolicy() if i % 2 == 0 else None
                ),
            )
            for i in range(num_features)
        ]
        # wrap in sequential because _quantize only applies to submodules...
        model = nn.Sequential(EmbeddingCollection(tables=tables, device=device))

        quant_model = _quantize(model, quant_state_dict_split_scale_bias=True)

        sharded_quant_model = _shard_modules(
            module=quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    TestQuantECSharder(
                        sharding_type=ShardingType.ROW_WISE.value,
                        kernel_type=EmbeddingComputeKernel.QUANT.value,
                    ),
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        sharded_quant_model.load_state_dict(sharded_quant_model.state_dict())

        local_batch, _ = ModelInput.generate(
            batch_size=16,
            world_size=1,
            num_float_features=10,
            tables=self.tables,
            weighted_tables=[],
            indices_dtype=torch.int32,
            lengths_dtype=torch.int32,
        )
        local_batch = local_batch.to(device)
        sharded_quant_model(local_batch.idlist_features)
        self.assertIsInstance(
            sharded_quant_model[0]
            ._lookups[0]
            ._embedding_lookups_per_rank[0]
            ._emb_modules[0]
            ._emb_module,
            KVEmbeddingInference,
        )
        self.assertIsInstance(
            sharded_quant_model[0]
            ._lookups[0]
            ._embedding_lookups_per_rank[0]
            ._emb_modules[1]
            ._emb_module,
            IntNBitTableBatchedEmbeddingBagsCodegen,
        )
        self.assertEqual(
            sharded_quant_model[0]
            ._lookups[0]
            ._embedding_lookups_per_rank[0]
            ._emb_modules[0]
            ._emb_module.table_sharding_offset,
            [0, 0],
        )
        self.assertEqual(
            sharded_quant_model[0]
            ._lookups[0]
            ._embedding_lookups_per_rank[1]
            ._emb_modules[0]
            ._emb_module.table_sharding_offset,
            [6, 17],
        )
