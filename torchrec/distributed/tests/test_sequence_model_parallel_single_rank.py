#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import cast, OrderedDict

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSingleRankBase,
)
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig


class SequenceModelParallelStateDictTest(ModelParallelSingleRankBase):
    def setUp(self, backend: str = "nccl") -> None:
        self.shared_features = []
        self.embedding_groups = {}

        super().setUp(backend=backend)

    def _create_tables(self) -> None:
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

        self.tables += initial_tables + shared_features_tables
        self.shared_features += [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups["group_0"] = [
            (f"{feature}@{table.name}" if feature in self.shared_features else feature)
            for table in self.tables
            for feature in table.feature_names
        ]

    def _create_model(self) -> nn.Module:
        return TestSequenceSparseNN(
            tables=self.tables,
            num_float_features=self.num_float_features,
            embedding_groups=self.embedding_groups,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
        is_training=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_load_state_dict(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
    ) -> None:
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                ),
            ),
        ]
        models, batch = self._generate_dmps_and_batch(sharders)
        m1, m2 = models

        # load the second's (m2's) with the first (m1's) state_dict
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))

        # validate the models are equivalent
        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch)
        self._compare_models(m1, m2)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_numerical_equivalence_between_kernel_types(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        self._set_table_weights_precision(dtype)
        fused_params = {
            "stochastic_rounding": stochastic_rounding,
            "cache_precision": dtype,
        }

        fused_sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                    fused_params=fused_params,
                ),
            ),
        ]
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    fused_params=fused_params,
                ),
            ),
        ]
        (fused_model, _), _ = self._generate_dmps_and_batch(fused_sharders)
        (model, _), batch = self._generate_dmps_and_batch(sharders)

        # load the baseline model's state_dict onto the new model
        model.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", fused_model.state_dict())
        )

        if is_training:
            for _ in range(4):
                self._train_models(fused_model, model, batch)
        # the problem here is output is FP32, but weights are FP16
        # so we should actually use FP16 atol and rtol to check close
        if not is_training or not stochastic_rounding:
            self._eval_models(
                fused_model, model, batch, is_deterministic=not stochastic_rounding
            )
        self._compare_models(
            fused_model, model, is_deterministic=not stochastic_rounding
        )
