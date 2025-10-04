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
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist

from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import NUM_POOLINGS, POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import (
    TestEBCSharder,
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
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
    batch_size: int
    batch_sizes: Optional[List[int]]
    num_float_features: int
    feature_pooling_avg: int
    use_offsets: bool
    dev_str: str
    long_kjt_indices: bool
    long_kjt_offsets: bool
    long_kjt_lengths: bool
    pin_memory: bool

    @abstractmethod
    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
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


def generate_data(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    model_config: BaseModelConfig,
    batch_sizes: List[int],
) -> List[ModelInput]:
    """
    Generate model input data for benchmarking.

    Args:
        tables: List of unweighted embedding tables
        weighted_tables: List of weighted embedding tables
        model_config: Configuration for model generation
        num_batches: Number of batches to generate

    Returns:
        A list of ModelInput objects representing the generated batches
    """
    device = torch.device(model_config.dev_str) if model_config.dev_str else None

    return [
        ModelInput.generate(
            batch_size=batch_size,
            tables=tables,
            weighted_tables=weighted_tables,
            num_float_features=model_config.num_float_features,
            pooling_avg=model_config.feature_pooling_avg,
            use_offsets=model_config.use_offsets,
            device=device,
            indices_dtype=(
                torch.int64 if model_config.long_kjt_indices else torch.int32
            ),
            offsets_dtype=(
                torch.int64 if model_config.long_kjt_offsets else torch.int32
            ),
            lengths_dtype=(
                torch.int64 if model_config.long_kjt_lengths else torch.int32
            ),
            pin_memory=model_config.pin_memory,
        )
        for batch_size in batch_sizes
    ]


def generate_planner(
    planner_type: str,
    topology: Topology,
    tables: Optional[List[EmbeddingBagConfig]],
    weighted_tables: Optional[List[EmbeddingBagConfig]],
    sharding_type: ShardingType,
    compute_kernel: EmbeddingComputeKernel,
    batch_sizes: List[int],
    pooling_factors: Optional[List[float]] = None,
    num_poolings: Optional[List[float]] = None,
) -> Union[EmbeddingShardingPlanner, HeteroEmbeddingShardingPlanner]:
    """
    Generate an embedding sharding planner based on the specified configuration.

    Args:
        planner_type: Type of planner to use ("embedding" or "hetero")
        topology: Network topology for distributed training
        tables: List of unweighted embedding tables
        weighted_tables: List of weighted embedding tables
        sharding_type: Strategy for sharding embedding tables
        compute_kernel: Compute kernel to use for embedding tables
        batch_sizes: Sizes of each batch
        pooling_factors: Pooling factors for each feature of the table
        num_poolings: Number of poolings for each feature of the table

    Returns:
        An instance of EmbeddingShardingPlanner or HeteroEmbeddingShardingPlanner

    Raises:
        RuntimeError: If an unknown planner type is specified
    """
    # Create parameter constraints for tables
    constraints = {}
    num_batches = len(batch_sizes)

    if pooling_factors is None:
        pooling_factors = [POOLING_FACTOR] * num_batches

    if num_poolings is None:
        num_poolings = [NUM_POOLINGS] * num_batches

    assert (
        len(pooling_factors) == num_batches and len(num_poolings) == num_batches
    ), "The length of pooling_factors and num_poolings must match the number of batches."

    if tables is not None:
        for table in tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
            )

    if weighted_tables is not None:
        for table in weighted_tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
                is_weighted=True,
            )

    if planner_type == "embedding":
        return EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints if constraints else None,
        )
    elif planner_type == "hetero":
        topology_groups = {"cuda": topology}
        return HeteroEmbeddingShardingPlanner(
            topology_groups=topology_groups,
            constraints=constraints if constraints else None,
        )
    else:
        raise RuntimeError(f"Unknown planner type: {planner_type}")


def generate_sharded_model_and_optimizer(
    model: nn.Module,
    sharding_type: str,
    kernel_type: str,
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
    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=kernel_type,
        fused_params=fused_params,
    )
    sharders = [cast(ModuleSharder[nn.Module], sharder)]

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
