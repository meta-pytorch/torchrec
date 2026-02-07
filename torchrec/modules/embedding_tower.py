#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def tower_input_params(module: nn.Module) -> Tuple[bool, bool]:
    """
    Utilty to compute the mapping of tower KJT args to pass to the embedding modules.

    Args:
        module (nn.Module):
    Returns:
        Tuple[bool, bool]: tuple of 2 booleans representing if KJT and weighted KJT are required, respectively.
    """
    if isinstance(module, EmbeddingCollection):
        return True, False
    elif isinstance(module, EmbeddingBagCollection):
        return not module.is_weighted(), module.is_weighted()
    # default to assuming both kjt and weight_kjt required
    return True, True


class EmbeddingTower(nn.Module):
    """
    Logical "Tower" of embeddings directly passed to provided interaction.
    All TorchRec shardable embedding modules are supported.

    Args:
        embedding_module (nn.Module):
        interaction_module (nn.Module):
        device (Optional[torch.device]):

    Example::

        ebc, interaction = EmbeddingBagCollection(), MyInteractionModule()
        tower = EmbeddingTower(ebc, interaction, device)
        kjt = KeyedJaggedTensor()
        output = tower(kjt)
    """

    def __init__(
        self,
        embedding_module: nn.Module,
        interaction_module: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding_module
        self.interaction = interaction_module

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Executes the embedding module and interaction module.

        Args:
            *args (Any): user provided positional arguments.
            **kwargs (Any): user provided keyword arguments.

        Returns:
            torch.Tensor: 2-D tensor of shape of `N X B`, where `B` is local batch size.
        """
        embeddings = self.embedding(*args, **kwargs)
        return self.interaction(embeddings)


class EmbeddingTowerCollection(nn.Module):
    """
    Collection of EmbeddingTowers.

    Args:
        towers (List[EmbeddingTower]): list of embedding towers that make up the
            collection.
        device (Optional[torch.device]): default compute device.

    Example::

        ebc, ebc_interaction = EmbeddingBagCollection(), MyEBCInteractionModule()
        eb, eb_interaction = EmbeddingCollection(), MyECInteractionModule()
        tower_0 = EmbeddingTower(ebc, ebc_interaction, device)
        tower_1 = EmbeddingTower(eb, eb_interaction, device)
        tower_collection = EmbeddingTowerCollection([tower_0, tower_1])
        kjt = KeyedJaggedTensor()
        output = tower_collection(kjt)
    """

    def __init__(
        self,
        towers: List[EmbeddingTower],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.towers = nn.ModuleList(towers)
        self._input_params: List[Tuple[bool, bool]] = []
        for tower in towers:
            self._input_params.append(tower_input_params(tower.embedding))

    def forward(
        self,
        features: Optional[KeyedJaggedTensor] = None,
        weighted_features: Optional[KeyedJaggedTensor] = None,
    ) -> torch.Tensor:
        """
        Executes the collection of towers.

        Features and/or weighted features must be provided as required by the
        underlying embedding modules.

        Args:
            features (Optional[KeyedJaggedTensor]):
            weighted_features (Optional[KeyedJaggedTensor]):

        Returns:
            torch.Tensor: 2-D tensor of shape `M X B`, where `M = sum(N_i)` for tower output i, and `B` is local batch size.
        """

        tower_outputs = []
        for tower, input_params in zip(self.towers, self._input_params):
            has_kjt_param, has_wkjt_param = input_params
            if has_kjt_param and has_wkjt_param:
                assert features is not None
                assert weighted_features is not None
                tower_outputs.append(tower(features, weighted_features))
            elif has_wkjt_param:
                assert weighted_features is not None
                tower_outputs.append(tower(weighted_features))
            else:
                assert features is not None
                tower_outputs.append(tower(features))

        return torch.cat(tower_outputs, dim=1)
