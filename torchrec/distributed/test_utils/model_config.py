#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Utilities for benchmarking training pipelines with different model configurations.

To support a new model in pipeline benchmark:
    1. Create config class inheriting from BaseModelConfig with generate_model() method
    2. Add the model to model_configs dict in create_model_config()
    3. Add model-specific params to ModelSelectionConfig and create_model_config's arguments in benchmark_train_pipeline.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import nn
from torchrec.distributed.test_utils.table_config import ManagedCollisionConfig
from torchrec.distributed.test_utils.test_model import (
    TestMixedEmbeddingSparseArch,
    TestMixedSequenceOverArch,
    TestMixedSequenceOverArchLargeActivation,
    TestOverArch,
    TestOverArchLarge,
    TestOverArchRegroupModule,
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.models.deepfm import SimpleDeepFMNNWrapper
from torchrec.models.dlrm import DLRMWrapper
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


@dataclass
class BaseModelConfig(ABC):
    """
    Abstract base class for model configurations.

    This class defines the common parameters shared across all model types
    and requires each concrete implementation to provide its own generate_model method.
    """

    ## Common parameters for all model types, please do not set default values here
    # we assume all model arch has a single dense feature layer
    num_float_features: int

    @abstractmethod
    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Generate a model instance based on the configuration.

        Args:
            tables: List of unweighted embedding tables
            weighted_tables: List of weighted embedding tables
            dense_device: Device to place dense layers on

        Returns:
            A neural network module instance
        """
        pass


OVER_ARCH_CLASSES: Dict[str, Type[nn.Module]] = {
    "default": TestOverArch,
    "large": TestOverArchLarge,
    "regroup_module": TestOverArchRegroupModule,
}

MIXED_OVER_ARCH_CLASSES: Dict[str, Type[nn.Module]] = {
    "default": TestMixedSequenceOverArch,
    "large_activation": TestMixedSequenceOverArchLargeActivation,
}


@dataclass
class TestSparseNNConfig(BaseModelConfig):
    """Configuration for TestSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None
    max_feature_lengths: Optional[Dict[str, int]] = None
    over_arch_clazz: Type[nn.Module] = TestOverArchLarge
    postproc_module: Optional[nn.Module] = None
    mc_configs: Optional[Dict[str, ManagedCollisionConfig]] = None
    submodule_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.over_arch_clazz, str):
            if self.over_arch_clazz not in OVER_ARCH_CLASSES:
                raise ValueError(
                    f"Unknown over_arch_clazz: {self.over_arch_clazz}. "
                    f"Available: {list(OVER_ARCH_CLASSES.keys())}"
                )
            self.over_arch_clazz = OVER_ARCH_CLASSES[self.over_arch_clazz]

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        return TestSparseNN(
            tables=tables,
            num_float_features=self.num_float_features,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            max_feature_lengths=self.max_feature_lengths,
            feature_processor_modules=self.feature_processor_modules,
            over_arch_clazz=self.over_arch_clazz,
            postproc_module=self.postproc_module,
            embedding_groups=self.embedding_groups,
            zch_kwargs=kwargs["mc_configs"] if kwargs["mc_configs"] else None,
            submodule_kwargs=self.submodule_kwargs,
        )


@dataclass
class TestTowerSparseNNConfig(BaseModelConfig):
    """Configuration for TestTowerSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        return TestTowerSparseNN(
            num_float_features=self.num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            embedding_groups=self.embedding_groups,
            feature_processor_modules=self.feature_processor_modules,
        )


@dataclass
class TestTowerCollectionSparseNNConfig(BaseModelConfig):
    """Configuration for TestTowerCollectionSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        return TestTowerCollectionSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            num_float_features=self.num_float_features,
            embedding_groups=self.embedding_groups,
            feature_processor_modules=self.feature_processor_modules,
        )


@dataclass
class DeepFMConfig(BaseModelConfig):
    """Configuration for DeepFM model."""

    hidden_layer_size: int = 20
    deep_fm_dimension: int = 5

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        # DeepFM only uses unweighted tables
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

        # Create and return SimpleDeepFMNN model
        return SimpleDeepFMNNWrapper(
            num_dense_features=self.num_float_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=self.hidden_layer_size,
            deep_fm_dimension=self.deep_fm_dimension,
        )


@dataclass
class DLRMConfig(BaseModelConfig):
    """Configuration for DLRM model."""

    dense_arch_layer_sizes: List[int] = field(default_factory=lambda: [20, 128])
    over_arch_layer_sizes: List[int] = field(default_factory=lambda: [5, 1])

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
        **kwargs: Any,
    ) -> nn.Module:
        # DLRM only uses unweighted tables
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

        return DLRMWrapper(
            embedding_bag_collection=ebc,
            dense_in_features=self.num_float_features,
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dense_device=dense_device,
        )


@dataclass
class MixedEmbeddingConfig(BaseModelConfig):
    """Configuration for mixed EBC+EC model using TestMixedEmbeddingSparseArch."""

    embedding_groups: Optional[Dict[str, List[str]]] = None
    over_arch_clazz: Type[nn.Module] = TestMixedSequenceOverArch
    enable_activation_stashing: bool = False
    dense_arch_hidden_sizes: Optional[List[int]] = None
    over_arch_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.over_arch_clazz, str):
            if self.over_arch_clazz not in MIXED_OVER_ARCH_CLASSES:
                raise ValueError(
                    f"Unknown mixed over_arch_clazz: {self.over_arch_clazz}. "
                    f"Available: {list(MIXED_OVER_ARCH_CLASSES.keys())}"
                )
            self.over_arch_clazz = MIXED_OVER_ARCH_CLASSES[self.over_arch_clazz]

    def generate_model(
        self,
        tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        dense_device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> nn.Module:
        return TestMixedEmbeddingSparseArch(
            tables=tables,
            num_float_features=self.num_float_features,
            weighted_tables=weighted_tables,
            embedding_groups=self.embedding_groups,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            over_arch_clazz=self.over_arch_clazz,
            enable_activation_stashing=self.enable_activation_stashing,
            dense_arch_hidden_sizes=self.dense_arch_hidden_sizes,
            over_arch_kwargs=self.over_arch_kwargs,
        )


def create_model_config(model_name: str, **kwargs: Any) -> BaseModelConfig:
    """
    deprecated function, please use ModelSelectionConfig.create_model_config instead
    """
    model_configs = {
        "test_sparse_nn": TestSparseNNConfig,
        "test_tower_sparse_nn": TestTowerSparseNNConfig,
        "test_tower_collection_sparse_nn": TestTowerCollectionSparseNNConfig,
        "deepfm": DeepFMConfig,
        "dlrm": DLRMConfig,
        "mixed_embedding": MixedEmbeddingConfig,
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    # Filter kwargs to only include valid parameters for the specific model config class
    model_class = model_configs[model_name]
    valid_field_names = {field.name for field in fields(model_class)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}

    return model_class(**filtered_kwargs)


@dataclass
class ModelSelectionConfig:
    model_name: str = "test_sparse_nn"
    model_config: Dict[str, Any] = field(
        default_factory=lambda: {"num_float_features": 10}
    )

    def get_model_config_class(self) -> Type[BaseModelConfig]:
        match self.model_name:
            case "test_sparse_nn":
                return TestSparseNNConfig
            case "test_tower_sparse_nn":
                return TestTowerSparseNNConfig
            case "test_tower_collection_sparse_nn":
                return TestTowerCollectionSparseNNConfig
            case "deepfm":
                return DeepFMConfig
            case "dlrm":
                return DLRMConfig
            case "mixed_embedding":
                return MixedEmbeddingConfig
            case _:
                raise ValueError(f"Unknown model name: {self.model_name}")

    def create_model_config(self) -> BaseModelConfig:
        config_class = self.get_model_config_class()
        valid_field_names = {field.name for field in fields(config_class)}
        filtered_kwargs = {
            k: v for k, v in self.model_config.items() if k in valid_field_names
        }
        return config_class(**filtered_kwargs)

    def create_test_model(self, **kwargs: Any) -> nn.Module:
        model_config = self.create_model_config()
        return model_config.generate_model(**kwargs)
