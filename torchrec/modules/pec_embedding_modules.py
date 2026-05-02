#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from enum import Enum
from typing import Dict, List

import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class OverlappingCheckerType(Enum):
    """Type of overlap checker used for collision detection."""

    BOOLEAN = "boolean"


class PECEmbeddingCollection(nn.Module):
    """Wraps an EmbeddingCollection with Prioritized Embedding Communication (PEC) config.

    PEC optimizes embedding communication by detecting overlapping IDs between
    consecutive batches. Overlapped embeddings are sent first (prioritized),
    allowing the trainer to start computation earlier.

    This unsharded module simply wraps an EmbeddingCollection and carries PEC
    configuration. The actual PEC logic lives in the sharded version
    (ShardedPECEmbeddingCollection).

    Args:
        embedding_collection: The EmbeddingCollection to wrap.
        checker_type: Type of overlap checker. BOOLEAN uses a full boolean mask.

    Example::

        import torch
        from torchrec.modules.embedding_configs import EmbeddingConfig
        from torchrec.modules.embedding_modules import EmbeddingCollection
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        tables = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
        ]
        ec = EmbeddingCollection(tables=tables, device=torch.device("cpu"))
        pec = PECEmbeddingCollection(ec, checker_type=OverlappingCheckerType.BOOLEAN)
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0"],
            values=torch.LongTensor([0, 1, 2]),
            lengths=torch.LongTensor([2, 1]),
        )
        output = pec(kjt)
        print(output["feature_0"].values().shape)
    """

    def __init__(
        self,
        embedding_collection: EmbeddingCollection,
        checker_type: OverlappingCheckerType = OverlappingCheckerType.BOOLEAN,
    ) -> None:
        super().__init__()
        self._embedding_collection = embedding_collection
        self._checker_type = checker_type

    @property
    def embedding_collection(self) -> EmbeddingCollection:
        return self._embedding_collection

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """Delegates to the inner EmbeddingCollection.

        Args:
            features: KJT of sparse features.

        Returns:
            Dict mapping feature names to JaggedTensors of embeddings.
        """
        return self._embedding_collection(features)

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_collection.embedding_configs()

    def embedding_dim(self) -> int:
        return self._embedding_collection.embedding_dim()

    def need_indices(self) -> bool:
        return self._embedding_collection.need_indices()
