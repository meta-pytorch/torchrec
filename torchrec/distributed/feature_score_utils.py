#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from typing import Dict, List, Sequence, Tuple

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import ShardingType
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    FeatureScoreBasedEvictionPolicy,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def create_sharding_type_to_feature_score_mapping(
    embedding_configs: Sequence[EmbeddingConfig],
    sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]],
) -> Tuple[bool, bool, Dict[str, Dict[str, float]]]:
    enable_feature_score_weight_accumulation = False
    enabled_feature_score_auto_collection = False

    # Validation for virtual table configurations
    virtual_tables = [
        config for config in embedding_configs if config.use_virtual_table
    ]
    if virtual_tables:
        virtual_tables_with_eviction = [
            config
            for config in virtual_tables
            if config.virtual_table_eviction_policy is not None
        ]
        if virtual_tables_with_eviction:
            # Check if any virtual table uses FeatureScoreBasedEvictionPolicy
            tables_with_feature_score_policy = [
                config
                for config in virtual_tables_with_eviction
                if isinstance(
                    config.virtual_table_eviction_policy,
                    FeatureScoreBasedEvictionPolicy,
                )
            ]

            # If any virtual table uses FeatureScoreBasedEvictionPolicy,
            # then ALL virtual tables with eviction policies must use FeatureScoreBasedEvictionPolicy
            if tables_with_feature_score_policy:
                assert all(
                    isinstance(
                        config.virtual_table_eviction_policy,
                        FeatureScoreBasedEvictionPolicy,
                    )
                    for config in virtual_tables_with_eviction
                ), "If any virtual table uses FeatureScoreBasedEvictionPolicy, all virtual tables with eviction policies must use FeatureScoreBasedEvictionPolicy"
                enable_feature_score_weight_accumulation = True

                # Check if any table has enable_auto_feature_score_collection=True
                tables_with_auto_collection = [
                    config
                    for config in tables_with_feature_score_policy
                    if config.virtual_table_eviction_policy is not None
                    and isinstance(
                        config.virtual_table_eviction_policy,
                        FeatureScoreBasedEvictionPolicy,
                    )
                    and config.virtual_table_eviction_policy.enable_auto_feature_score_collection
                ]
                if tables_with_auto_collection:
                    # All virtual tables with FeatureScoreBasedEvictionPolicy must have enable_auto_feature_score_collection=True
                    assert all(
                        config.virtual_table_eviction_policy is not None
                        and isinstance(
                            config.virtual_table_eviction_policy,
                            FeatureScoreBasedEvictionPolicy,
                        )
                        and config.virtual_table_eviction_policy.enable_auto_feature_score_collection
                        for config in tables_with_feature_score_policy
                    ), "If any virtual table has enable_auto_feature_score_collection=True, all virtual tables with FeatureScoreBasedEvictionPolicy must have enable_auto_feature_score_collection=True"
                    enabled_feature_score_auto_collection = True

    sharding_type_feature_score_mapping: Dict[str, Dict[str, float]] = {}
    if enabled_feature_score_auto_collection:
        for (
            sharding_type,
            sharding_info,
        ) in sharding_type_to_sharding_infos.items():
            feature_score_mapping: Dict[str, float] = {}
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                sharding_type_feature_score_mapping[sharding_type] = (
                    feature_score_mapping
                )
                continue
            for config in sharding_info:
                vtep = config.embedding_config.virtual_table_eviction_policy
                if vtep is not None and isinstance(
                    vtep, FeatureScoreBasedEvictionPolicy
                ):
                    if vtep.eviction_ttl_mins > 0:
                        logger.info(
                            f"Virtual table eviction policy enabled for table {config.embedding_config.name} {sharding_type} with eviction TTL {vtep.eviction_ttl_mins} mins."
                        )
                        feature_score_mapping.update(
                            dict.fromkeys(config.embedding_config.feature_names, 0.0)
                        )
                        continue
                    for k in config.embedding_config.feature_names:
                        if (
                            k
                            # pyrefly: ignore[missing-attribute]
                            in config.embedding_config.virtual_table_eviction_policy.feature_score_mapping
                        ):
                            feature_score_mapping[k] = (
                                # pyrefly: ignore[missing-attribute]
                                config.embedding_config.virtual_table_eviction_policy.feature_score_mapping[
                                    k
                                ]
                            )
                        else:
                            assert (
                                # pyrefly: ignore[missing-attribute]
                                config.embedding_config.virtual_table_eviction_policy.feature_score_default_value
                                is not None
                            ), f"Table {config.embedding_config.name} eviction policy feature_score_default_value is not set but feature {k} is not in feature_score_mapping."
                            feature_score_mapping[k] = (
                                # pyrefly: ignore[unsupported-operation]
                                config.embedding_config.virtual_table_eviction_policy.feature_score_default_value
                            )
            sharding_type_feature_score_mapping[sharding_type] = feature_score_mapping
    return (
        enable_feature_score_weight_accumulation,
        enabled_feature_score_auto_collection,
        sharding_type_feature_score_mapping,
    )


@torch.fx.wrap
def may_collect_feature_scores(
    input_feature_splits: List[KeyedJaggedTensor],
    enabled_feature_score_auto_collection: bool,
    sharding_type_feature_score_mapping: Dict[str, Dict[str, float]],
) -> List[KeyedJaggedTensor]:
    if not enabled_feature_score_auto_collection:
        return input_feature_splits
    with record_function("## collect_feature_score ##"):
        for features, mapping in zip(
            input_feature_splits, sharding_type_feature_score_mapping.values()
        ):
            assert (
                features.weights_or_none() is None
            ), f"Auto feature collection: {features.keys()=} has non empty weights"
            if (
                mapping is None or len(mapping) == 0
            ):  # collection is disabled fir this sharding type
                continue
            feature_score_weights = []
            device = features.device()
            for f in features.keys():
                # input dist includes multiple lookups input including both virtual table and non-virtual table features.
                # We needs to attach weights for all features due to KJT weights requirements, so set 0.0 score for non virtual table features
                score = mapping[f] if f in mapping else 0.0
                feature_score_weights.append(
                    torch.ones_like(
                        features[f].values(),
                        dtype=torch.float32,
                        device=device,
                    )
                    * score
                )
            features._weights = (
                torch.cat(feature_score_weights, dim=0)
                if feature_score_weights
                else None
            )
    return input_feature_splits
