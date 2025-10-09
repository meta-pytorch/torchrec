#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.modules.deepfm import DeepFM, FactorizationMachine
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class SparseArch(nn.Module):
    """
    Processes the sparse features of the DeepFMNN model. Does embedding lookups for all
    EmbeddingBag and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor):

        Returns:
            KeyedJaggedTensor: an output KJT of size F * D X B.
        """
        return self.embedding_bag_collection(features)


class DenseArch(nn.Module):
    """
    Processes the dense features of DeepFMNN model. Output layer is sized to
    the embedding_dimension of the EmbeddingBagCollection embeddings.

    Args:
        in_features (int): dimensionality of the dense input features.
        hidden_layer_size (int): sizes of the hidden layers in the DenseArch.
        embedding_dim (int): the same size of the embedding_dimension of sparseArch.
        device (torch.device): default compute device.

    Example::

        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(
            in_features=in_features, hidden_layer_size=10, embedding_dim=D
        )

        dense_arch_input = torch.rand((B, in_features))
        dense_embedded = dense_arch(dense_arch_input)
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): size B X `num_features`.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class FMInteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features) and apply the general DeepFM interaction according to the
    external source of DeepFM paper: https://arxiv.org/pdf/1703.04247.pdf

    The output dimension is expected to be a cat of `dense_features`, D.

    Args:
        fm_in_features (int): the input dimension of `dense_module` in DeepFM. For
            example, if the input embeddings is [randn(3, 2, 3), randn(3, 4, 5)], then
            the `fm_in_features` should be: 2 * 3 + 4 * 5.
        sparse_feature_names (List[str]): length of F.
        deep_fm_dimension (int): output of the deep interaction (DI) in the DeepFM arch.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        fm_inter_arch = FMInteractionArch(sparse_feature_names=keys)
        dense_features = torch.rand((B, D))
        sparse_features = KeyedTensor(
            keys=keys,
            length_per_key=[D, D],
            values=torch.rand((B, D * F)),
        )
        cat_fm_output = fm_inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        fm_in_features: int,
        sparse_feature_names: List[str],
        deep_fm_dimension: int,
    ) -> None:
        super().__init__()
        self.sparse_feature_names: List[str] = sparse_feature_names
        self.deep_fm = DeepFM(
            dense_module=nn.Sequential(
                nn.Linear(fm_in_features, deep_fm_dimension),
                nn.ReLU(),
            )
        )
        self.fm = FactorizationMachine()

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): tensor of size B X D.
            sparse_features (KeyedJaggedTensor): KJT of size F * D X B.

        Returns:
            torch.Tensor: an output tensor of size B X (D + DI + 1).
        """
        if len(self.sparse_feature_names) == 0:
            return dense_features

        tensor_list: List[torch.Tensor] = [dense_features]
        # dense/sparse interaction
        # size B X F
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])

        deep_interaction = self.deep_fm(tensor_list)
        fm_interaction = self.fm(tensor_list)

        return torch.cat([dense_features, deep_interaction, fm_interaction], dim=1)


class OverArch(nn.Module):
    """
    Final Arch - simple MLP. The output is just one target.

    Args:
        in_features (int): the output dimension of the interaction arch.

    Example::

        B = 20
        over_arch = OverArch()
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: an output tensor of size B X 1.
        """
        return self.model(features)


class SimpleDeepFMNN(nn.Module):
    """
    Basic recsys module with DeepFM arch. Processes sparse features by
    learning pooled embeddings for each feature. Learns the relationship between
    dense features and sparse features by projecting dense features into the same
    embedding space. Learns the interaction among those dense and sparse features
    by deep_fm proposed in this paper: https://arxiv.org/pdf/1703.04247.pdf

    The module assumes all sparse features have the same embedding dimension
    (i.e, each `EmbeddingBagConfig` uses the same embedding_dim)

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        num_dense_features (int): the number of input dense features.
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        hidden_layer_size (int): the hidden layer size used in dense module.
        deep_fm_dimension (int): the output layer size used in `deep_fm`'s deep
            interaction module.

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
        sparse_nn = SimpleDeepFMNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
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

        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        num_dense_features: int,
        embedding_bag_collection: EmbeddingBagCollection,
        hidden_layer_size: int,
        deep_fm_dimension: int,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs()) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs())):
            conf_prev = embedding_bag_collection.embedding_bag_configs()[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs()[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim

        feature_names = []

        fm_in_features = embedding_dim
        for conf in embedding_bag_collection.embedding_bag_configs():
            for feat in conf.feature_names:
                feature_names.append(feat)
                fm_in_features += conf.embedding_dim

        self.sparse_arch = SparseArch(embedding_bag_collection)
        self.dense_arch = DenseArch(
            in_features=num_dense_features,
            hidden_layer_size=hidden_layer_size,
            embedding_dim=embedding_dim,
        )
        self.inter_arch = FMInteractionArch(
            fm_in_features=fm_in_features,
            sparse_feature_names=feature_names,
            deep_fm_dimension=deep_fm_dimension,
        )
        over_in_features = embedding_dim + deep_fm_dimension + 1
        self.over_arch = OverArch(over_in_features)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits with size B X 1.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits


class SimpleDeepFMNNWrapper(SimpleDeepFMNN):
    # pyre-ignore[14, 15]
    def forward(
        self, model_input: ModelInput
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the SimpleDeepFMNNWrapper.

        Args:
            model_input (ModelInput): Contains dense and sparse features.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            If training, returns (loss, prediction). Otherwise, returns prediction.
        """
        pred = super().forward(
            dense_features=model_input.float_features,
            sparse_features=model_input.idlist_features,  # pyre-ignore[6]
        )

        if self.training:
            # Calculate loss and return both loss and prediction
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred.squeeze(), model_input.label
            )
            return (loss, pred)
        else:
            # Return just the prediction
            return pred
