#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class EmbeddingTableProps:
    """
    Properties of an embedding table.

    Args:
        name (str): name of the table
        num_embeddings (int): number of embeddings in the table
        embedding_dim (int): dimension of each embedding
        sharding_type (ShardingType): sharding type of the table
        feature_names (List[str]): list of feature names associated with the table
        pooling (PoolingType): pooling type of the table
        data_type (DataType): data type of the table
        weight_type (WeightedType): weight
    """

    num_embeddings: int
    embedding_dim: int
    sharding: ShardingType
    feature_names: List[str]
    pooling: PoolingType
    data_type: DataType = DataType.FP32
    is_weighted: bool = False


class TestECModel(nn.Module):
    """
    Test model with EmbeddingCollection and Linear layers.

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables
        device (Optional[torch.device]): device on which buffers will be initialized

    Example:
        TestECModel(tables=[EmbeddingConfig(...)])
    """

    def __init__(
        self, tables: List[EmbeddingConfig], device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.ec: EmbeddingCollection = EmbeddingCollection(
            tables=tables,
            device=device if device else torch.device("meta"),
        )

        embedding_dim = tables[0].embedding_dim

        self.seq: nn.Sequential = nn.Sequential(
            *[nn.Linear(embedding_dim, embedding_dim) for _ in range(3)]
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        """
        Forward pass of the TestECModel.

        Args:
            features (KeyedJaggedTensor): Input features for the model.

        Returns:
            torch.Tensor: Output tensor after processing through the model.
        """

        lookup_result = self.ec(features)
        return self.seq(torch.cat([jt.values() for _, jt in lookup_result.items()]))


class TestEBCModel(nn.Module):
    """
    Test model with EmbeddingBagCollection and Linear layers.

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables
        device (Optional[torch.device]): device on which buffers will be initialized

    Example:
        TestEBCModel(tables=[EmbeddingBagConfig(...)])
    """

    def __init__(
        self, tables: List[EmbeddingBagConfig], device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.ebc: EmbeddingBagCollection
        self.ebc = EmbeddingBagCollection(
            tables=tables,
            device=device if device else torch.device("meta"),
        )

        embedding_dim = tables[0].embedding_dim
        self.seq: nn.Sequential = nn.Sequential(
            *[nn.Linear(embedding_dim, embedding_dim) for _ in range(3)]
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        """
        Forward pass of the TestEBCModel.

        Args:
            features (KeyedJaggedTensor): Input features for the model.

        Returns:
            torch.Tensor: Output tensor after processing through the model.
        """

        lookup_result = self.ebc(features).to_dict()
        return self.seq(torch.cat(tuple(lookup_result.values())))


def create_ec_model(
    tables: Dict[str, EmbeddingTableProps],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create an EmbeddingCollection model with the given tables.

    Args:
        tables (List[EmbeddingTableProps]): list of embedding tables
        device (Optional[torch.device]): device on which buffers will be initialized

    Returns:
        nn.Module: EmbeddingCollection model
    """
    return TestECModel(
        tables=[
            EmbeddingConfig(
                name=name,
                embedding_dim=table.embedding_dim,
                num_embeddings=table.num_embeddings,
                feature_names=table.feature_names,
                data_type=table.data_type,
            )
            for name, table in tables.items()
        ],
        device=device,
    )


def create_ebc_model(
    tables: Dict[str, EmbeddingTableProps],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create an EmbeddinBagCollection model with the given tables.

    Args:
        tables (List[EmbeddingTableProps]): list of embedding tables
        device (Optional[torch.device]): device on which buffers will be initialized

    Returns:
        nn.Module: EmbeddingCollection model
    """
    return TestEBCModel(
        tables=[
            EmbeddingBagConfig(
                name=name,
                embedding_dim=table.embedding_dim,
                num_embeddings=table.num_embeddings,
                feature_names=table.feature_names,
                data_type=table.data_type,
                pooling=table.pooling,
            )
            for name, table in tables.items()
        ],
        device=device,
    )


def generate_planner_constraints(
    tables: Dict[str, EmbeddingTableProps],
) -> dict[str, ParameterConstraints]:
    """
    Generate planner constraints for the given tables.

    Args:
        tables (List[EmbeddingTableProps]): list of embedding tables

    Returns:
        Dict[str, ParameterConstraints]: planner constraints
    """
    constraints: Dict[str, ParameterConstraints] = {}
    for name, table in tables.items():
        sharding_types = [table.sharding.value]
        constraints[name] = ParameterConstraints(
            sharding_types=sharding_types,
            compute_kernels=[EmbeddingComputeKernel.FUSED.value],
            feature_names=table.feature_names,
            pooling_factors=[1.0],
        )
    return constraints
