#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Feature Processing Utilities for Two-Tower Retrieval.

This module provides utilities for preprocessing and transforming features
for use with two-tower recommendation models.

Key Components:
    - FeatureConfig: Configuration for feature processing
    - FeatureProcessor: Transforms raw features into model inputs
    - BatchBuilder: Creates KeyedJaggedTensor batches from raw data
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclass
class FeatureConfig:
    """
    Configuration for a categorical feature.

    Attributes:
        name: Feature name (must match embedding table name).
        num_embeddings: Vocabulary size for this feature.
        default_value: Value to use for missing/OOV entries.
        is_query_feature: Whether this is a query (user) feature.
        is_candidate_feature: Whether this is a candidate (item) feature.
    """

    name: str
    num_embeddings: int
    default_value: int = 0
    is_query_feature: bool = False
    is_candidate_feature: bool = False


@dataclass
class FeatureProcessorConfig:
    """
    Configuration for the feature processor.

    Attributes:
        features: List of feature configurations.
        query_features: Names of query (user) features.
        candidate_features: Names of candidate (item) features.
    """

    features: List[FeatureConfig] = field(default_factory=list)

    @property
    def query_features(self) -> List[str]:
        """Get names of query features."""
        return [f.name for f in self.features if f.is_query_feature]

    @property
    def candidate_features(self) -> List[str]:
        """Get names of candidate features."""
        return [f.name for f in self.features if f.is_candidate_feature]


class FeatureProcessor:
    """
    Processes raw features into model-ready format.

    Handles vocabulary mapping, missing values, and OOV handling.

    Args:
        config: FeatureProcessorConfig with feature definitions.
        device: Device to place output tensors on.

    Example::

        >>> config = FeatureProcessorConfig(
        ...     features=[
        ...         FeatureConfig("user_id", num_embeddings=1000000, is_query_feature=True),
        ...         FeatureConfig("item_id", num_embeddings=10000000, is_candidate_feature=True),
        ...     ]
        ... )
        >>> processor = FeatureProcessor(config)
        >>> kjt = processor.process_query({"user_id": [123, 456]})
    """

    def __init__(
        self,
        config: FeatureProcessorConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")

        self._feature_map: Dict[str, FeatureConfig] = {
            f.name: f for f in config.features
        }

    def _validate_and_clip(
        self,
        feature_name: str,
        values: List[int],
    ) -> torch.Tensor:
        """
        Validate feature values and clip to valid range.

        Args:
            feature_name: Name of the feature.
            values: Raw feature values.

        Returns:
            Tensor of validated values.
        """
        config = self._feature_map[feature_name]

        tensor = torch.tensor(values, dtype=torch.long, device=self.device)

        mask = (tensor < 0) | (tensor >= config.num_embeddings)
        tensor[mask] = config.default_value

        return tensor

    def process_query(
        self,
        raw_features: Dict[str, List[int]],
    ) -> KeyedJaggedTensor:
        """
        Process query (user) features into a KeyedJaggedTensor.

        Args:
            raw_features: Dictionary mapping feature names to values.
                Each value list should have the same length (batch size).

        Returns:
            KeyedJaggedTensor suitable for the query tower.

        Example::

            >>> raw = {"user_id": [1, 2, 3], "user_segment": [0, 1, 0]}
            >>> kjt = processor.process_query(raw)
        """
        return self._process_features(
            raw_features,
            self.config.query_features,
        )

    def process_candidate(
        self,
        raw_features: Dict[str, List[int]],
    ) -> KeyedJaggedTensor:
        """
        Process candidate (item) features into a KeyedJaggedTensor.

        Args:
            raw_features: Dictionary mapping feature names to values.

        Returns:
            KeyedJaggedTensor suitable for the candidate tower.
        """
        return self._process_features(
            raw_features,
            self.config.candidate_features,
        )

    def _process_features(
        self,
        raw_features: Dict[str, List[int]],
        feature_names: List[str],
    ) -> KeyedJaggedTensor:
        """
        Process a set of features into a KeyedJaggedTensor.

        Args:
            raw_features: Dictionary mapping feature names to values.
            feature_names: List of feature names to process.

        Returns:
            KeyedJaggedTensor with the processed features.
        """
        if not feature_names:
            raise ValueError("No features to process")

        batch_size = len(raw_features[feature_names[0]])

        all_values = []
        all_lengths = []

        for feature_name in feature_names:
            if feature_name not in raw_features:
                raise ValueError(f"Missing feature: {feature_name}")

            values = raw_features[feature_name]
            if len(values) != batch_size:
                raise ValueError(
                    f"Feature {feature_name} has {len(values)} values, "
                    f"expected {batch_size}"
                )

            validated = self._validate_and_clip(feature_name, values)
            all_values.append(validated)

            lengths = torch.ones(batch_size, dtype=torch.int32, device=self.device)
            all_lengths.append(lengths)

        return KeyedJaggedTensor(
            keys=feature_names,
            values=torch.cat(all_values),
            lengths=torch.cat(all_lengths),
        )


class BatchBuilder:
    """
    Builds training batches for two-tower models.

    Creates paired query/candidate batches with optional labels.

    Args:
        processor: FeatureProcessor for feature transformation.

    Example::

        >>> builder = BatchBuilder(processor)
        >>> query_kjt, candidate_kjt, labels = builder.build_batch(
        ...     query_features={"user_id": [1, 2]},
        ...     candidate_features={"item_id": [100, 200]},
        ...     labels=[1.0, 0.0],
        ... )
    """

    def __init__(self, processor: FeatureProcessor) -> None:
        self.processor = processor

    def build_batch(
        self,
        query_features: Dict[str, List[int]],
        candidate_features: Dict[str, List[int]],
        labels: Optional[List[float]] = None,
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor, Optional[torch.Tensor]]:
        """
        Build a training batch from raw features.

        Args:
            query_features: Raw query (user) features.
            candidate_features: Raw candidate (item) features.
            labels: Optional labels (e.g., clicks, purchases).

        Returns:
            Tuple of (query_kjt, candidate_kjt, labels_tensor).
        """
        query_kjt = self.processor.process_query(query_features)
        candidate_kjt = self.processor.process_candidate(candidate_features)

        labels_tensor = None
        if labels is not None:
            labels_tensor = torch.tensor(
                labels,
                dtype=torch.float32,
                device=self.processor.device,
            )

        return query_kjt, candidate_kjt, labels_tensor

    def build_inference_batch(
        self,
        query_features: Dict[str, List[int]],
    ) -> KeyedJaggedTensor:
        """
        Build an inference batch (query only).

        Args:
            query_features: Raw query (user) features.

        Returns:
            KeyedJaggedTensor for the query tower.
        """
        return self.processor.process_query(query_features)


def create_default_processor(
    num_users: int = 1000000,
    num_items: int = 10000000,
    device: Optional[torch.device] = None,
) -> FeatureProcessor:
    """
    Create a feature processor with default MovieLens-style configuration.

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        device: Device to place tensors on.

    Returns:
        Configured FeatureProcessor.
    """
    config = FeatureProcessorConfig(
        features=[
            FeatureConfig(
                name="userId",
                num_embeddings=num_users,
                is_query_feature=True,
            ),
            FeatureConfig(
                name="movieId",
                num_embeddings=num_items,
                is_candidate_feature=True,
            ),
        ]
    )
    return FeatureProcessor(config, device)
