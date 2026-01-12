#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Dict, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.schema.utils import is_signature_compatible
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class StableEmbeddingBagCollectionInterface:
    """
    Stable Interface for `EmbeddingBagCollection`.
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        pass

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        return KeyedTensor(
            keys=[],
            length_per_key=[],
            values=torch.empty(0),
        )

    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return []

    def is_weighted(self) -> bool:
        return False


class StableEmbeddingCollectionInterface:
    """
    Stable Interface for `EmbeddingBagCollection`.
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
        need_indices: bool = False,
    ) -> None:
        return

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        return {}

    def embedding_configs(
        self,
    ) -> List[EmbeddingConfig]:
        return []

    def need_indices(self) -> bool:
        return False

    def embedding_dim(self) -> int:
        return 0

    def embedding_names_by_table(self) -> List[List[str]]:
        return []


class TestEmbeddingModuleSchema(unittest.TestCase):
    def test_embedding_bag_collection(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingBagCollectionInterface.__init__),
                inspect.signature(EmbeddingBagCollection.__init__),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingBagCollectionInterface.forward),
                inspect.signature(EmbeddingBagCollection.forward),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(
                    StableEmbeddingBagCollectionInterface.embedding_bag_configs
                ),
                inspect.signature(EmbeddingBagCollection.embedding_bag_configs),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingBagCollectionInterface.is_weighted),
                inspect.signature(EmbeddingBagCollection.is_weighted),
            )
        )

    def test_embedding_collection(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingCollectionInterface.__init__),
                inspect.signature(EmbeddingCollection.__init__),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingCollectionInterface.forward),
                inspect.signature(EmbeddingCollection.forward),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingCollectionInterface.embedding_configs),
                inspect.signature(EmbeddingCollection.embedding_configs),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingCollectionInterface.embedding_dim),
                inspect.signature(EmbeddingCollection.embedding_dim),
            )
        )

        self.assertTrue(
            is_signature_compatible(
                inspect.signature(
                    StableEmbeddingCollectionInterface.embedding_names_by_table
                ),
                inspect.signature(EmbeddingCollection.embedding_names_by_table),
            )
        )
