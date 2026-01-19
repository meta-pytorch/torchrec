#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.test_utils.test_model import TestSparseNNBase
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
    sharding_single_rank_test,
)
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
    TestSequenceTowerSparseNN,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class SequenceModelParallelHierarchicalTest(MultiProcessTestBase):
    """
    Testing hierarchical sharding types.

    NOTE:
        Requires at least 4 GPUs to test.
    """

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-ignore [56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
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
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_seq_emb_tower_nccl(
        self,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        self._test_sharding(
            # pyre-ignore [6]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_TOWER.value, sharding_type, kernel_type
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            model_class=TestSequenceTowerSparseNN,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    # TODO: consolidate the following methods with https://fburl.com/code/62zg0kel
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
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )
