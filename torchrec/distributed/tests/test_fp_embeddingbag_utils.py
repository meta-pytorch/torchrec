#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Any, cast, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.distributed.fp_embeddingbag import (
    FeatureProcessedEmbeddingBagCollectionSharder,
)
from torchrec.distributed.test_utils.test_model import TestEBCSharder
from torchrec.distributed.types import QuantizedCommCodecs
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    FeatureProcessor,
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

DEFAULT_MAX_FEATURE_LENGTH = 12


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        use_fp_collection: bool,
        device: torch.device,
        max_feature_lengths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        feature_names = [
            feature_name for table in tables for feature_name in table.feature_names
        ]

        if max_feature_lengths is None:
            max_feature_lengths = [DEFAULT_MAX_FEATURE_LENGTH] * len(feature_names)

        assert len(max_feature_lengths) == len(
            feature_names
        ), "Expect max_feature_lengths to have the same number of items as feature_names"

        self._fp_ebc: FeatureProcessedEmbeddingBagCollection = (
            FeatureProcessedEmbeddingBagCollection(
                EmbeddingBagCollection(
                    tables=tables,
                    device=device,
                    is_weighted=True,
                ),
                (
                    cast(
                        Dict[str, FeatureProcessor],
                        {
                            feature_name: PositionWeightedModule(
                                max_feature_length=max_feature_length
                            )
                            for feature_name, max_feature_length in zip(
                                feature_names, max_feature_lengths
                            )
                        },
                    )
                    if not use_fp_collection
                    else PositionWeightedModuleCollection(
                        max_feature_lengths=dict(
                            zip(feature_names, max_feature_lengths)
                        ),
                    )
                ),
            ).to(device)
        )

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fp_ebc_out = self._fp_ebc(kjt)
        pred = torch.cat(
            [
                fp_ebc_out[key]
                for key in [
                    "feature_0",
                    "feature_1",
                    "feature_2",
                    "feature_3",
                ]
            ],
            dim=1,
        )
        loss = pred.mean()
        return loss, pred


def create_module_and_freeze(
    tables: List[EmbeddingBagConfig],
    use_fp_collection: bool,
    device: torch.device,
    max_feature_lengths: Optional[List[int]] = None,
) -> SparseArch:

    sparse_arch = SparseArch(tables, use_fp_collection, device, max_feature_lengths)

    torch.manual_seed(0)
    for param in sparse_arch.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    torch.manual_seed(0)

    return sparse_arch


class TestFPEBCSharder(FeatureProcessedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

        ebc_sharder = TestEBCSharder(
            self._sharding_type,
            self._kernel_type,
            fused_params,
            qcomm_codecs_registry,
        )
        super().__init__(ebc_sharder, qcomm_codecs_registry)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        Restricts sharding to single type only.
        """
        return (
            [self._sharding_type]
            if self._sharding_type
            in super().sharding_types(compute_device_type=compute_device_type)
            else []
        )

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        """
        Restricts to single impl.
        """
        return [self._kernel_type]


def get_configs() -> List[EmbeddingBagConfig]:
    dims = [3 * 16, 8, 8, 3 * 16]
    return [
        EmbeddingBagConfig(
            name=f"table_{i}",
            feature_names=[f"feature_{i}"],
            embedding_dim=dim,
            num_embeddings=16,
        )
        for i, dim in enumerate(dims)
    ]


def get_kjt_inputs() -> List[KeyedJaggedTensor]:
    # Rank 0
    #             instance 0   instance 1  instance 2
    # "feature_0"   [0, 1]       None        [2]
    # "feature_1"   [0, 1]       None        [2]
    # "feature_2"   [3, 1]       [4,1]        [5]
    # "feature_3"   [1]       [6,1,8]        [0,3,3]

    # Rank 1

    #             instance 0   instance 1  instance 2
    # "feature_0"   [3, 2]       [1,2]       [0,1,2,3]
    # "feature_1"   [2, 3]       None        [2]
    # "feature_2"   [2, 7]       [1,8,2]        [8,1]
    # "feature_3"   [9]       [8]        [7]

    kjt_input_per_rank = [  # noqa
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1", "feature_2", "feature_3"],
            values=torch.LongTensor(
                [0, 1, 2, 0, 1, 2, 3, 1, 4, 1, 5, 1, 6, 1, 8, 0, 3, 3]
            ),
            lengths=torch.LongTensor(
                [
                    2,
                    0,
                    1,
                    2,
                    0,
                    1,
                    2,
                    2,
                    1,
                    1,
                    3,
                    3,
                ]
            ),
            weights=torch.FloatTensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1", "feature_2", "feature_3"],
            values=torch.LongTensor(
                [3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2, 2, 7, 1, 8, 2, 8, 1, 9, 8, 7]
            ),
            lengths=torch.LongTensor([2, 2, 4, 2, 0, 1, 2, 3, 2, 1, 1, 1]),
            weights=torch.FloatTensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ),
        ),
    ]
    return kjt_input_per_rank
