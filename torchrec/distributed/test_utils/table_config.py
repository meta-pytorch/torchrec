#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Any, Dict, List

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.types import DataType


@dataclass
class EmbeddingTablesConfig:
    """
    Configuration for generating embedding tables for test and benchmark

    This class defines the parameters for generating embedding tables with both weighted
    and unweighted features.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
            Default is 100.
        num_weighted_features (int): Number of weighted features to generate.
            Default is 100.
        embedding_feature_dim (int): Dimension of the embedding vectors.
            Default is 128.
        additional_tables (List[List[Dict[str, Any]]]): Additional tables to include in the configuration.
            Default is an empty list.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128
    table_data_type: DataType = DataType.FP32
    additional_tables: List[List[Dict[str, Any]]] = field(default_factory=list)

    def convert_to_ebconf(self, kwargs: Dict[str, Any]) -> EmbeddingBagConfig:
        if "data_type" in kwargs:
            kwargs["data_type"] = DataType[kwargs["data_type"]]
        else:
            kwargs["data_type"] = self.table_data_type
        return EmbeddingBagConfig(**kwargs)

    def generate_tables(
        self,
    ) -> List[List[EmbeddingBagConfig]]:
        """
        Generate embedding bag configurations for both unweighted and weighted features.

        This function creates two lists of EmbeddingBagConfig objects:
        1. Unweighted tables: Named as "table_{i}" with feature names "feature_{i}"
        2. Weighted tables: Named as "weighted_table_{i}" with feature names "weighted_feature_{i}"

        For both types, the number of embeddings scales with the feature index,
        calculated as max(i + 1, 100) * 1000.

        Args:
            num_unweighted_features (int): Number of unweighted features to generate.
            num_weighted_features (int): Number of weighted features to generate.
            embedding_feature_dim (int): Dimension of the embedding vectors.

        Returns:
            Tuple[List[EmbeddingBagConfig], List[EmbeddingBagConfig]]: A tuple containing
            two lists - the first for unweighted embedding tables and the second for
            weighted embedding tables.
        """
        unweighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=max(i + 1, 100) * 2000,
                embedding_dim=self.embedding_feature_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=self.table_data_type,
            )
            for i in range(self.num_unweighted_features)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=max(i + 1, 100) * 2000,
                embedding_dim=self.embedding_feature_dim,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
                data_type=self.table_data_type,
            )
            for i in range(self.num_weighted_features)
        ]
        tables_list = []
        for idx, adts in enumerate(self.additional_tables):
            if idx == 0:
                tables = unweighted_tables
            elif idx == 1:
                tables = weighted_tables
            else:
                tables = []
            for adt in adts:
                tables.append(self.convert_to_ebconf(adt))

        if len(tables_list) == 0:
            tables_list.append(unweighted_tables)
            tables_list.append(weighted_tables)
        elif len(tables_list) == 1:
            tables_list.append(weighted_tables)
        return tables_list
