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

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist

from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.types import ShardingEnv
from torchrec.models.deepfm import SimpleDeepFMNNWrapper
from torchrec.models.dlrm import DLRMWrapper
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


@dataclass
class BaseModelConfig(ABC):
    """
    Abstract base class for model configurations.

    This class defines the common parameters shared across all model types
    and requires each concrete implementation to provide its own generate_model method.
    """

    # Common parameters for all model types
    num_float_features: int  # we assume all model arch has a single dense feature layer

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


@dataclass
class TestSparseNNConfig(BaseModelConfig):
    """Configuration for TestSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]
    max_feature_lengths: Optional[Dict[str, int]]
    over_arch_clazz: Type[nn.Module]
    postproc_module: Optional[nn.Module]
    zch: bool

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
            zch=self.zch,
        )


@dataclass
class TestTowerSparseNNConfig(BaseModelConfig):
    """Configuration for TestTowerSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]

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

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]

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

    hidden_layer_size: int
    deep_fm_dimension: int

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

    dense_arch_layer_sizes: List[int]
    over_arch_layer_sizes: List[int]

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


# pyre-ignore[2]: Missing parameter annotation
def create_model_config(model_name: str, **kwargs) -> BaseModelConfig:

    model_configs = {
        "test_sparse_nn": TestSparseNNConfig,
        "test_tower_sparse_nn": TestTowerSparseNNConfig,
        "test_tower_collection_sparse_nn": TestTowerCollectionSparseNNConfig,
        "deepfm": DeepFMConfig,
        "dlrm": DLRMConfig,
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    # Filter kwargs to only include valid parameters for the specific model config class
    model_class = model_configs[model_name]
    valid_field_names = {field.name for field in fields(model_class)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}

    return model_class(**filtered_kwargs)


def generate_sharded_model_and_optimizer(
    model: nn.Module,
    pg: dist.ProcessGroup,
    device: torch.device,
    fused_params: Dict[str, Any],
    dense_optimizer: str = "SGD",
    dense_lr: float = 0.1,
    dense_momentum: Optional[float] = None,
    dense_weight_decay: Optional[float] = None,
    planner: Optional[
        Union[
            EmbeddingShardingPlanner,
            HeteroEmbeddingShardingPlanner,
        ]
    ] = None,
) -> Tuple[nn.Module, Optimizer]:
    """
    Generate a sharded model and optimizer for distributed training.

    Args:
        model: The model to be sharded
        sharding_type: Type of sharding strategy
        kernel_type: Type of compute kernel
        pg: Process group for distributed training
        device: Device to place the model on
        fused_params: Parameters for the fused optimizer
        dense_optimizer: Optimizer type for dense parameters
        dense_lr: Learning rate for dense parameters
        dense_momentum: Momentum for dense parameters (optional)
        dense_weight_decay: Weight decay for dense parameters (optional)
        planner: Optional planner for sharding strategy

    Returns:
        Tuple of sharded model and optimizer
    """
    sharders = get_default_sharders()

    # Use planner if provided
    plan = None
    if planner is not None:
        if pg is not None:
            plan = planner.collective_plan(model, sharders, pg)
        else:
            plan = planner.plan(model, sharders)

    sharded_model = DistributedModelParallel(
        module=copy.deepcopy(model),
        env=ShardingEnv.from_process_group(pg),
        init_data_parallel=True,
        device=device,
        sharders=sharders,
        plan=plan,
    ).to(device)

    # Get dense parameters
    dense_params = [
        param
        for name, param in sharded_model.named_parameters()
        if "sparse" not in name
    ]

    # Create optimizer based on the specified type
    optimizer_class = getattr(optim, dense_optimizer)

    # Create optimizer with momentum and/or weight_decay if provided
    optimizer_kwargs = {"lr": dense_lr}

    if dense_momentum is not None:
        optimizer_kwargs["momentum"] = dense_momentum

    if dense_weight_decay is not None:
        optimizer_kwargs["weight_decay"] = dense_weight_decay

    optimizer = optimizer_class(dense_params, **optimizer_kwargs)

    return sharded_model, optimizer
