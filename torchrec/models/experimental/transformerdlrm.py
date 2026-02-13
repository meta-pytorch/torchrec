#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch
from torch import nn
from torchrec.models.dlrm import DLRM, OverArch
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class InteractionTransformerArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the output of the nn.transformerencoder,
    that takes the combined values of both sparse features and the output of the dense layer,
    and the dense layer itself (i.e. concat(dense layer output, transformer encoder output).
    Note: This model is for benchmarking purposes only, i.e. to measure the performance of transformer + embeddings using the dlrm models.
    It is not intended to increase model convergence metrics.
    Implemented TE as described here:
    https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html?highlight=transformer+encoder#torch.nn.TransformerEncoder
    BERT Transformer Paper: https://arxiv.org/abs/1810.04805
    Attention is All you Need: https://arxiv.org/abs/1706.03762


    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.
    Args:
        num_sparse_features (int): F.
        embedding_dim: int,
        nhead: int, #number of attention heads
        ntransformer_layers: int, #number of transformer layers.
    Example::
        D = 8   #must divisible by number of transformer heads
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionTransormerArch(num_sparse_features=len(keys))
        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))
        #  B X (D * (F + 1))
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        num_sparse_features: int,
        embedding_dim: int,
        nhead: int = 8,
        ntransformer_layers: int = 4,
    ) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.nhead = nhead
        self.ntransformer_layers = ntransformer_layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self.nhead,
        )
        self.interarch_TE = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=self.ntransformer_layers
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.
        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape
        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )
        # Transformer for Interactions
        transformer_interactions = self.interarch_TE(combined_values)
        interactions_flat = torch.reshape(transformer_interactions, (B, -1))
        return interactions_flat


class DLRM_Transformer(DLRM):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. On the interaction layer,
    the relationship between dense features and sparse features is learned through a transformer encoder layer
    https://arxiv.org/abs/1706.03762.
    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).
    The following notation is used throughout the documentation for the models:
    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features
    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        nhead: int: Number of multi-attention heads
        ntransformer_layers: int: Number of transformer encoder layers
        dense_device (Optional[torch.device]): default compute device.
    Example::
        B = 2
        D = 8
        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM_Transformer(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
        features = torch.rand((B, 100))
        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )
        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        nhead: int = 8,
        ntransformer_layers: int = 4,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dense_device,
        )
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)
        # pyrefly: ignore[bad-assignment]
        self.inter_arch = InteractionTransformerArch(
            num_sparse_features=num_sparse_features,
            embedding_dim=embedding_dim,
            nhead=nhead,
            ntransformer_layers=ntransformer_layers,
        )
        over_in_features: int = (num_sparse_features + 1) * embedding_dim
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )
