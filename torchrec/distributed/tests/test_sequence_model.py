#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestDenseArch,
    TestSparseNNBase,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.embedding_tower import EmbeddingTower
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def _post_sparsenn_forward(
    padded_embeddings: List[torch.Tensor],
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if batch_size is None or padded_embeddings[0].size(0) == batch_size:
        return torch.cat(
            padded_embeddings,
            dim=1,
        )
    else:
        seq_emb = torch.cat(padded_embeddings, dim=1)
        ec_values = torch.zeros(
            batch_size,
            seq_emb.size(1),
            dtype=seq_emb.dtype,
            device=seq_emb.device,
        )
        ec_values[: seq_emb.size(0), :] = seq_emb
        return ec_values


class TestSequenceSparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        embedding_names: List[str],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.ec: EmbeddingCollection = EmbeddingCollection(
            tables=tables, device=device, need_indices=True
        )
        self.embedding_names = embedding_names
        self.embedding_dim: int = self.ec.embedding_dim()

    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        jt_dict = self.ec(id_list_features)
        padded_embeddings = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=jt_dict[e].values(),
                offsets=jt_dict[e].offsets(),
                max_sequence_length=20,
            ).view(-1, 20 * self.embedding_dim)
            for e in self.embedding_names
        ]

        return _post_sparsenn_forward(padded_embeddings, batch_size)


class TestSequenceTowerInteraction(nn.Module):
    def __init__(
        self,
        embedding_names: List[str],
        embedding_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.embedding_names = embedding_names
        self.embedding_dim: int = embedding_dim
        self.max_sequence_length = 20
        self.linear = nn.Linear(
            in_features=self.max_sequence_length
            * self.embedding_dim
            * len(embedding_names),
            out_features=8,
            device=device,
        )

    def forward(
        self,
        sequence_emb: Dict[str, JaggedTensor],
    ) -> torch.Tensor:
        padded_embeddings = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=sequence_emb[e].values(),
                offsets=sequence_emb[e].offsets(),
                max_sequence_length=self.max_sequence_length,
            ).view(-1, self.max_sequence_length * self.embedding_dim)
            for e in self.embedding_names
        ]
        cat_embeddings = torch.cat(padded_embeddings, dim=1)
        return self.linear(cat_embeddings)


class TestSequenceTowerSparseNN(TestSparseNNBase):
    """
    Simple version of a sequence tower embedding model.

    Args:
        tables: List[EmbeddingBagConfig],
        num_float_features: int,
        weighted_tables: Optional[List[EmbeddingConfig]],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example:
        >>> TestSequenceTowerInteraction()
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )

        self.dense = TestDenseArch(num_float_features, dense_device)
        # current planner put table_0 and table_3 on the same node
        # while table_1 and table_2 are on the other node
        # TODO: after adding planner support for tower_module, we can random assign
        # tables to towers
        t0_tables = [tables[0], tables[2]]
        t0_emb_names = []
        for table in t0_tables:
            t0_emb_names += table.feature_names
        embedding_dim = tables[0].embedding_dim

        t1_tables = [tables[1], tables[3]]
        t1_emb_names = []
        for table in t1_tables:
            t1_emb_names += table.feature_names

        self.tower_0 = EmbeddingTower(
            embedding_module=EmbeddingCollection(tables=t0_tables),
            interaction_module=TestSequenceTowerInteraction(
                embedding_names=t0_emb_names,
                embedding_dim=embedding_dim,
            ),
        )
        self.tower_1 = EmbeddingTower(
            embedding_module=EmbeddingCollection(tables=t1_tables),
            interaction_module=TestSequenceTowerInteraction(
                embedding_names=t1_emb_names,
                embedding_dim=embedding_dim,
            ),
        )
        self.over = nn.Linear(
            in_features=8
            #  `out_features`.
            # pyrefly: ignore[missing-attribute]
            + self.tower_0.interaction.linear.out_features
            #  `out_features`.
            # pyrefly: ignore[missing-attribute]
            + self.tower_1.interaction.linear.out_features,
            out_features=16,
            device=dense_device,
        )

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        tower_0_r = self.tower_0(input.idlist_features)
        tower_1_r = self.tower_1(input.idlist_features)

        sparse_r = torch.cat([tower_0_r, tower_1_r], dim=1)
        over_r = self.over(torch.cat([dense_r, sparse_r], dim=1))
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


class TestSequenceOverArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        in_features = 8 + sum(
            [table.embedding_dim * len(table.feature_names) * 20 for table in tables]
        )
        self.linear: nn.modules.Linear = nn.Linear(
            in_features=in_features, out_features=16, device=device
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(torch.cat([dense, sparse], dim=1))


class TestSequenceSparseNN(TestSparseNNBase):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        weighted_tables: Optional[List[EmbeddingConfig]] = None,
        num_float_features: int = 10,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if embedding_groups is None:
            embedding_groups = {}

        self.dense = TestDenseArch(
            device=dense_device, num_float_features=num_float_features
        )
        self.sparse = TestSequenceSparseArch(
            tables,
            list(embedding_groups.values())[0] if embedding_groups.values() else [],
            device=sparse_device,
        )
        self.over = TestSequenceOverArch(tables=tables, device=dense_device)

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        # multiply the sparse output by 10 since the model output is not sensitive to the
        # embedding output. It won't catch the unexpected embedding output without this
        sparse_r = 10 * self.sparse(input.idlist_features, input.float_features.size(0))
        over_r = self.over(dense_r, sparse_r)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


class TestEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

        qcomm_codecs_registry = {}
        if qcomms_config is not None:
            qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config)

        if fused_params is None:
            fused_params = {}
        if "learning_rate" not in fused_params:
            fused_params["learning_rate"] = 0.1

        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=use_index_dedup,
        )

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]
