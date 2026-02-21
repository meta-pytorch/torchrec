#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torch import nn
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardingEnv
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.quant.utils import meta_to_cpu_placement
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class TestSparseArch(nn.Module):
    def __init__(self, ebc: QuantEmbeddingBagCollection) -> None:
        super().__init__()
        self.ebc = ebc

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        return self.ebc(features)


class QuantUtilsTest(unittest.TestCase):
    def _test_meta_to_cpu(
        self,
        tables: List[EmbeddingBagConfig],
        features: KeyedJaggedTensor,
        quant_type: torch.dtype = torch.qint8,
        output_type: torch.dtype = torch.float,
    ) -> None:
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

        # test forward
        # pyrefly: ignore[bad-argument-type]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        model = TestSparseArch(qebc)
        dmp = DistributedModelParallel(
            model,
            env=ShardingEnv.from_local(1, 0),
            sharders=[],
            init_data_parallel=False,
        )
        meta_to_cpu_placement(dmp)
        # assert device is cpu from meta
        self.assertEqual(dmp.module.ebc.device.type, "cpu")
        # successful cpu execution
        _ = dmp(features)

    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        permute_order=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_meta_to_cpu(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
        permute_order: bool,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=data_type,
        )
        features = (
            KeyedJaggedTensor(
                keys=["f1", "f2"],
                values=torch.as_tensor([0, 1]),
                lengths=torch.as_tensor([1, 1]),
            )
            if not permute_order
            else KeyedJaggedTensor(
                keys=["f2", "f1"],
                values=torch.as_tensor([1, 0]),
                lengths=torch.as_tensor([1, 1]),
            )
        )
        self._test_meta_to_cpu(
            [eb1_config, eb2_config], features, quant_type, output_type
        )
