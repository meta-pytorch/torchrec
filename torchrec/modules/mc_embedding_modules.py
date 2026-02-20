#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import cast, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


def evict(
    evictions: Dict[str, Optional[torch.Tensor]],
    ebc: nn.Module,
) -> None:
    # TODO: write function
    return


class BaseManagedCollisionEmbeddingCollection(nn.Module):
    """
    BaseManagedCollisionEmbeddingCollection represents a EC/EBC module and a set of managed collision modules.
    The inputs into the MC-EC/EBC will first be modified by the managed collision module before being passed into the embedding collection.

    Args:
        embedding_module: EmbeddingCollection to lookup embeddings
        managed_collision_modules: Dict of managed collision modules
        return_remapped_features (bool): whether to return remapped input features
            in addition to embeddings
        allow_in_place_embed_weight_update(bool): Enables in-place update of embedding
            weights on eviction. When enabled, this flag allows updates to embedding
            weights without modifying the autograd graph.

    """

    def __init__(
        self,
        embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection],
        managed_collision_collection: ManagedCollisionCollection,
        return_remapped_features: bool = False,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        super().__init__()
        self._managed_collision_collection = managed_collision_collection
        self._return_remapped_features = return_remapped_features
        self._allow_in_place_embed_weight_update = allow_in_place_embed_weight_update
        self._embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection] = (
            embedding_module
        )

        if isinstance(embedding_module, EmbeddingBagCollection):
            assert (
                #  list[EmbeddingBagConfig], Module, Tensor]` is not a function.
                # pyrefly: ignore[not-callable]
                self._embedding_module.embedding_bag_configs()
                == self._managed_collision_collection.embedding_configs()
            ), "Embedding Bag Collection and Managed Collision Collection must contain the Embedding Configs"

        else:
            assert (
                #  list[EmbeddingConfig], Module, Tensor]` is not a function.
                # pyrefly: ignore[not-callable]
                self._embedding_module.embedding_configs()
                == self._managed_collision_collection.embedding_configs()
            ), "Embedding Collection and Managed Collision Collection must contain the Embedding Configs"

    def forward(
        self,
        features: KeyedJaggedTensor,
        mutate_miss_lengths: bool = True,
    ) -> Tuple[
        Union[KeyedTensor, Dict[str, JaggedTensor]], Optional[KeyedJaggedTensor]
    ]:

        features = self._managed_collision_collection(
            features,
            mutate_miss_lengths=mutate_miss_lengths,
        )

        embedding_res = self._embedding_module(features)

        evict(self._managed_collision_collection.evict(), self._embedding_module)

        if not self._return_remapped_features:
            return embedding_res, None
        return embedding_res, features

    def lookup_remapped_lengths_mask(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        features = self._managed_collision_collection(
            features,
            mutate_miss_lengths=True,
        )
        remapped_lengths = return_remapped_lengths_as_mask(features)
        return remapped_lengths


@torch.fx.wrap
def return_remapped_lengths_as_mask(features: KeyedJaggedTensor) -> torch.Tensor:
    return features.lengths().to(torch.bool)


class ManagedCollisionEmbeddingCollection(BaseManagedCollisionEmbeddingCollection):
    """
    ManagedCollisionEmbeddingCollection represents a EmbeddingCollection module and a set of managed collision modules.
    The inputs into the MC-EC will first be modified by the managed collision module before being passed into the embedding collection.

    For details of input and output types, see EmbeddingCollection

    Args:
        embedding_collection: EmbeddingCollection to lookup embeddings
        managed_collision_collection: Dict of managed collision modules
        return_remapped_features (bool): whether to return remapped input features
            in addition to embeddings
        allow_in_place_embed_weight_update(bool): enable in place update of embedding
            weights on evict. This flag when enabled will allow update embedding
            weights without modifying of autograd graph.

    """

    def __init__(
        self,
        embedding_collection: EmbeddingCollection,
        managed_collision_collection: ManagedCollisionCollection,
        return_remapped_features: bool = False,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        super().__init__(
            embedding_collection,
            managed_collision_collection,
            return_remapped_features,
            allow_in_place_embed_weight_update,
        )

    # For consistency with embedding bag collection
    @property
    def _embedding_collection(self) -> EmbeddingCollection:
        return cast(EmbeddingCollection, self._embedding_module)


class ManagedCollisionEmbeddingBagCollection(BaseManagedCollisionEmbeddingCollection):
    """
    ManagedCollisionEmbeddingBagCollection represents a EmbeddingBagCollection module and a set of managed collision modules.
    The inputs into the MC-EBC will first be modified by the managed collision module before being passed into the embedding bag collection.

    For details of input and output types, see EmbeddingBagCollection

    Args:
        embedding_module: EmbeddingBagCollection to lookup embeddings
        managed_collision_modules: Dict of managed collision modules
        return_remapped_features (bool): whether to return remapped input features
            in addition to embeddings
        allow_in_place_embed_weight_update(bool): Enables in-place update of embedding
            weights on eviction. When enabled, this flag allows updates to embedding
            weights without modifying the autograd graph.


    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        managed_collision_collection: ManagedCollisionCollection,
        return_remapped_features: bool = False,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        super().__init__(
            embedding_bag_collection,
            managed_collision_collection,
            return_remapped_features,
            allow_in_place_embed_weight_update,
        )

    # For backwards compat, as references existed in tests
    @property
    def _embedding_bag_collection(self) -> EmbeddingBagCollection:
        return cast(EmbeddingBagCollection, self._embedding_module)
