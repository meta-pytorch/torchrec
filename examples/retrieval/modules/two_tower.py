#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, List, Mapping, Optional, OrderedDict, Tuple, Union

import faiss  # @manual=//faiss/python:pyfaiss_gpu
import faiss.contrib.torch_utils  # @manual=//faiss/contrib:faiss_contrib_gpu
import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def convert_TwoTower_to_TwoTowerRetrieval(
    sd: OrderedDict[str, torch.Tensor],
    query_tables: List[str],
    candidate_tables: List[str],
) -> OrderedDict[str, torch.Tensor]:
    for query_table in query_tables:
        sd[f"query_ebc.embedding_bags.{query_table}.weight"] = sd.pop(
            f"two_tower.ebc.embedding_bags.{query_table}.weight"
        )
    for candidate_table in candidate_tables:
        sd[f"candidate_ebc.embedding_bags.{candidate_table}.weight"] = sd.pop(
            f"two_tower.ebc.embedding_bags.{candidate_table}.weight"
        )
    return sd


class TwoTower(nn.Module):
    """
    Simple TwoTower (UV) Model. Embeds two different entities into the same space.
    A simplified version of the `A Dual Augmented Two-tower Model for Online Large-scale Recommendation
    <https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf>`_ model.
    Used to train the retrieval model

    Embeddings trained with this model will be indexed and queried in the retrieval example.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): embedding_bag_collection with two EmbeddingBags
        layer_sizes (List[int]): list of the layer_sizes for the MLP
        device (Optional[torch.device])

    Example::

        m = TwoTower(ebc, [16, 8], device)
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # If running this example on Torcherc < v0.2.0,
        # please use embedding_bag_configs as a property, not a function
        assert (
            len(embedding_bag_collection.embedding_bag_configs()) == 2
        ), "Expected two EmbeddingBags in the two tower model"
        assert (
            embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
            == embedding_bag_collection.embedding_bag_configs()[1].embedding_dim
        ), "Both EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        self._feature_names_query: List[str] = (
            embedding_bag_collection.embedding_bag_configs()[0].feature_names
        )
        self._candidate_feature_names: List[str] = (
            embedding_bag_collection.embedding_bag_configs()[1].feature_names
        )
        self.ebc = embedding_bag_collection
        self.query_proj = MLP(
            in_size=embedding_dim, layer_sizes=layer_sizes, device=device
        )
        self.candidate_proj = MLP(
            in_size=embedding_dim, layer_sizes=layer_sizes, device=device
        )

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            kjt (KeyedJaggedTensor): KJT containing query_ids and candidate_ids to query

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing embeddings for each tower
        """
        pooled_embeddings = self.ebc(kjt)
        query_embedding: torch.Tensor = self.query_proj(
            torch.cat(
                [pooled_embeddings[feature] for feature in self._feature_names_query],
                dim=1,
            )
        )
        candidate_embedding: torch.Tensor = self.candidate_proj(
            torch.cat(
                [
                    pooled_embeddings[feature]
                    for feature in self._candidate_feature_names
                ],
                dim=1,
            )
        )
        return query_embedding, candidate_embedding


class TwoTowerTrainTask(nn.Module):
    """
    Train Task for the TwoTower model. Adds BinaryCrossEntropy Loss.  to use with train_pipeline

    Args:
        two_tower (TwoTower): two tower model

    Example::

        m = TwoTowerTrainTask(two_tower_model)
    """

    def __init__(self, two_tower: TwoTower) -> None:
        super().__init__()
        self.two_tower = two_tower
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch (Batch): batch from torchrec.datasets

        Returns:
            Tuple[loss, Tuple[loss, logits, labels]]: each of shape B x 1
        """
        query_embedding, candidate_embedding = self.two_tower(batch.sparse_features)
        logits = (query_embedding * candidate_embedding).sum(dim=1).squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())


class TwoTowerRetrieval(nn.Module):
    """
    Simple TwoTower (UV) Model. Embeds two different entities (query and candidate) into the same space.
    Similar to the TwoTower model above, but is meant for retrieval. Specifically, this module
    also contiains a FAISS index, used to KNN search the K closest entities of tower 2. It separates query
    and candidate into separate EmbeddingBagCollections

    Args:
        faiss_index (Union[faiss.GpuIndexIVFPQ, faiss.IndexIVFPQ]): faiss index to search candidate
        query_ebc (EmbeddingBagCollection): embedding_bag_collection with one EmbeddingBag
        candidate_ebc (EmbeddingBagCollection): embedding_bag_collection with one EmbeddingBag
        layer_sizes (List[int]): list of the layer_sizes for the MLP
        k (int): number of tower 2 nearest neighbors to score
        device (Optional[torch.device])

    Example::

        m = TwoTowerRetrieval(index, query_ebc, candidate_ebc, [16, 8], 100, device)
    """

    def __init__(
        self,
        faiss_index: Union[faiss.GpuIndexIVFPQ, faiss.IndexIVFPQ],
        query_ebc: EmbeddingBagCollection,
        candidate_ebc: EmbeddingBagCollection,
        layer_sizes: List[int],
        k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.embedding_dim: int = query_ebc.embedding_bag_configs()[0].embedding_dim
        assert (
            candidate_ebc.embedding_bag_configs()[0].embedding_dim == self.embedding_dim
        ), "Both EmbeddingBagCollections must have the same dimension"
        self.candidate_feature_names: List[str] = candidate_ebc.embedding_bag_configs()[
            0
        ].feature_names
        self.query_ebc = query_ebc
        self.candidate_ebc = candidate_ebc
        self.query_proj = MLP(
            in_size=self.embedding_dim,
            layer_sizes=layer_sizes,
            device=device,
            dtype=dtype,
        )
        self.candidate_proj = MLP(
            in_size=self.embedding_dim,
            layer_sizes=layer_sizes,
            device=device,
            dtype=dtype,
        )
        self.faiss_index: Union[faiss.GpuIndexIVFPQ, faiss.IndexIVFPQ] = faiss_index
        self.k = k
        self.device = device

    def forward(self, query_kjt: KeyedJaggedTensor) -> torch.Tensor:
        """
        Args:
            query_kjt (KeyedJaggedTensor): KJT containing query_ids to query

        Returns:
            torch.Tensor: logits
        """
        batch_size = query_kjt.stride()
        # tower 1 lookup
        query_embedding = self.query_proj(self.query_ebc(query_kjt).values())

        # KNN lookup
        distances = torch.empty((batch_size, self.k), device=self.device)
        candidates = torch.empty(
            (batch_size, self.k), device=self.device, dtype=torch.int64
        )
        query_embedding = query_embedding.to(torch.float32)  # required by faiss
        self.faiss_index.search(query_embedding, self.k, distances, candidates)

        # candidate lookup
        candidate_kjt = KeyedJaggedTensor(
            keys=self.candidate_feature_names,
            values=candidates.reshape(-1),
            lengths=torch.tensor([self.k] * batch_size),
        )
        candidate_embedding = self.candidate_proj(
            self.candidate_ebc(candidate_kjt).values()
        )

        # return logit (dot product)
        return (query_embedding * candidate_embedding).sum(dim=1).squeeze()

    #  inconsistently.
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool) -> None:
        super().load_state_dict(state_dict, strict)
