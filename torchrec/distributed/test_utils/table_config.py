#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from torchrec.modules.embedding_configs import (
    CountBasedEvictionPolicy,
    CountTimestampMixedEvictionPolicy,
    EmbeddingBagConfig,
    FeatureScoreBasedEvictionPolicy,
    NoEvictionPolicy,
    TimestampBasedEvictionPolicy,
    VirtualTableEvictionPolicy,
)

from torchrec.types import DataType


def _return_correct_eviction_policy(
    eviction_str: str,
) -> Type[VirtualTableEvictionPolicy]:
    if eviction_str == "CountBasedEvictionPolicy":
        return CountBasedEvictionPolicy
    if eviction_str == "TimestampBasedEvictionPolicy":
        return TimestampBasedEvictionPolicy
    if eviction_str == "CountTimestampMixedEvictionPolicy":
        return CountTimestampMixedEvictionPolicy
    if eviction_str == "FeatureScoreBasedEvictionPolicy":
        return FeatureScoreBasedEvictionPolicy
    raise ValueError(f"Could not recognize eviction_str in yaml file: {eviction_str}")


def _process_virtual_table_config(config_dict: Dict[str, Any]) -> None:
    """Converts YAML virtual table fields (location, eviction-policy) to EBC format."""
    if "location" in config_dict:
        # config_dict["location"] should match LocationType
        config_dict["use_virtual_table"] = config_dict["location"] in [
            "DRAM_VIRTUAL_TABLE",
            "SSD_VIRTUAL_TABLE",
        ]

        if config_dict["use_virtual_table"]:
            assert (
                config_dict["total_num_buckets"] > 0
            ), "Should be larger 0  when using SSD_VIRTUAL_TABLE or DRAM_VIRTUAL_TABLE"

            assert (
                config_dict["num_embeddings"] % config_dict["total_num_buckets"] == 0
            ), (
                f"num_embeddings ({config_dict['num_embeddings']}) must be divisible by "
                f"total_num_buckets ({config_dict['total_num_buckets']})"
            )

            if "virtual_table_eviction_policy" in config_dict:
                # Obtain what eviction strategy was chosen
                eviction = config_dict["virtual_table_eviction_policy"]
                policy_class_name = next(iter(eviction.keys()))
                policy_params = eviction[policy_class_name]
                eviction = _return_correct_eviction_policy(policy_class_name)(
                    **policy_params
                )
            else:
                # Choose standard no eviction policy
                eviction = NoEvictionPolicy()

            # Initialize the eviction policy
            data_type = config_dict["data_type"]
            embedding_dim = config_dict["embedding_dim"]
            eviction.init_metaheader_config(data_type, embedding_dim)

            config_dict["virtual_table_eviction_policy"] = eviction


@dataclass
class ManagedCollisionConfig:
    """Configuration for ManagedCollision (MC) module parameters, e.g. MP-ZCH, Sort-ZCH."""

    mc_type: str = "mp-zch"
    input_hash_size: int = 0

    # MP-ZCH (HashZCH class) parameters
    max_probe: int = 128
    total_num_buckets: Optional[int] = None
    output_segments: Optional[List[int]] = (None,)
    eviction_policy_name: Optional[str] = None
    eviction_config: Optional[Dict[str, int]] = None
    opt_in_prob: int = -1
    percent_reserved_slots: float = 0.0
    disable_fallback: bool = False
    tb_logging_frequency: int = 0
    start_bucket: int = 0
    end_bucket: Optional[int] = None

    # Sort ZCH (MCH class) parameters
    eviction_policy: Optional[Any] = "DistanceLFU"
    eviction_interval: int = 1


@dataclass
class TableExtendedConfigs:
    """
    Container for table-related configurations outside of EmbeddingBagConfig.

    Holds additional configs and used to pass extra table configurations
    through the benchmark pipeline.
    """

    mc_configs: Dict[str, ManagedCollisionConfig] = field(default_factory=dict)


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
        mc_config (Dict[str, ManagedCollisionConfig]): Maps table to its ManagedCollision configuration.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128
    base_row_size: int = 100_000
    table_data_type: DataType = DataType.FP32
    total_num_buckets: Optional[int] = None
    additional_tables: List[List[Dict[str, Any]]] = field(default_factory=list)
    # ManagedCollision configs for all tables
    mc_config: Optional[ManagedCollisionConfig] = None  # Default for all tables
    mc_configs_per_table: Dict[str, ManagedCollisionConfig] = field(
        default_factory=dict
    )  # MC configs for all tables

    def _get_mc_config_to_table(
        self, table_name: str, mc_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Get MC config for a table. Per-table config takes priority over default."""
        if mc_config is not None:
            # Per-table config from additional_tables
            self.mc_configs_per_table[table_name] = ManagedCollisionConfig(**mc_config)
        elif self.mc_config is not None:
            # Use global default for tables
            self.mc_configs_per_table[table_name] = ManagedCollisionConfig(
                **self.mc_config
            )

    def convert_to_ebconf(self, kwargs: Dict[str, Any]) -> EmbeddingBagConfig:
        if "data_type" in kwargs:
            kwargs["data_type"] = DataType[kwargs["data_type"]]
        else:
            kwargs["data_type"] = self.table_data_type

        # Process configs for KV-ZCH/ZCH v.Next
        _process_virtual_table_config(kwargs)

        # Process configs for MP-ZCH
        mc_config = kwargs.pop("mc_config", None)
        self._get_mc_config_to_table(kwargs["name"], mc_config)

        # Remove all keys that are not part of EmbeddingBagConfig
        kwargs.pop("location", None)

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
                num_embeddings=max(i + 1, 100) * self.base_row_size // 100,
                embedding_dim=self.embedding_feature_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=self.table_data_type,
            )
            for i in range(self.num_unweighted_features)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=max(i + 1, 100) * self.base_row_size // 100,
                embedding_dim=self.embedding_feature_dim,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
                data_type=self.table_data_type,
            )
            for i in range(self.num_weighted_features)
        ]

        # Get default ManagedCollision configs for all tables, if provided
        for table in unweighted_tables + weighted_tables:
            self._get_mc_config_to_table(table.name)

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
            tables_list.append(tables)

        if len(tables_list) == 0:
            tables_list.append(unweighted_tables)
            tables_list.append(weighted_tables)
        elif len(tables_list) == 1:
            tables_list.append(weighted_tables)

        return tables_list
