#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import KVZCHTBEConfig
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import HybridEvalDMP
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import CacheParams, ParameterConstraints
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import (
    KeyValueParams,
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingPlanner,
    ShardingType,
)
from torchrec.fb.distributed.planner.lp_planner import LinearProgrammingPlanner
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig

logger: logging.Logger = logging.getLogger(__name__)


def _detect_hbm_cap(compute_device: str) -> Optional[int]:
    """Auto-detect HBM capacity from the current CUDA device.

    Returns the total GPU memory in bytes, or None if detection is not
    possible (e.g., no CUDA device available or compute_device is not cuda).
    """
    if compute_device != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(torch.device("cuda")).total_memory
    except Exception:
        return None


@dataclass
class PlannerConfig:
    planner_type: str = "embedding"
    world_size: int = 2
    device_group: str = "cuda"
    batch_size: int = 512  # Must match RunOptions.batch_size for accurate planner stats
    pooling_factors: List[float] = field(default_factory=lambda: [POOLING_FACTOR])
    num_poolings: Optional[List[float]] = None
    batch_sizes: Optional[List[int]] = None
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.FUSED
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    additional_constraints: Dict[str, Any] = field(default_factory=dict)
    # Storage reservation percentage (0.0 to 1.0) for planner memory estimation
    storage_reservation_percentage: float = 0.15
    # Hardware configuration for topology (dict with keys: hbm_cap, ddr_cap, etc.)
    hardware: Optional[Dict[str, Any]] = None

    def _apply_hardware_config(
        self,
        topology_kwargs: Dict[str, Any],
        device_type: str,
    ) -> None:
        """Apply hardware config to topology kwargs, with hbm_cap fact-checking."""
        if self.hardware is None:
            return

        # Passthrough keys from hardware config to topology kwargs
        _HARDWARE_KEYS = [
            "hbm_cap",
            "ddr_cap",
            "hbm_mem_bw",
            "ddr_mem_bw",
            "hbm_to_ddr_mem_bw",
            "intra_host_bw",
            "inter_host_bw",
            "pod_size",
            "local_world_size",
        ]
        for key in _HARDWARE_KEYS:
            if key in self.hardware:
                topology_kwargs[key] = self.hardware[key]

        # Fact-check user-provided hbm_cap against detected value
        if "hbm_cap" in self.hardware:
            detected_hbm = _detect_hbm_cap(device_type)
            if detected_hbm is not None:
                user_hbm = self.hardware["hbm_cap"]
                ratio = user_hbm / detected_hbm if detected_hbm else 0.0
                if ratio < 0.5 or ratio > 2.0:
                    logger.warning(
                        "Hardware config hbm_cap=%d (%.1f GB) differs significantly "
                        "from detected GPU memory %d (%.1f GB), ratio=%.2fx. "
                        "Using configured value.",
                        user_hbm,
                        user_hbm / (1024**3),
                        detected_hbm,
                        detected_hbm / (1024**3),
                        ratio,
                    )

    def generate_topology(self, device_type: str) -> Topology:
        """
        Generate a topology for distributed training.

        Supports GB200 NVLink domain topology via pod_size parameter:
        - pod_size: Number of hosts per NVLink domain (topology_domain_multiple)
        - local_world_size: Number of GPUs per host (overrides auto-detection)

        The intra_group_size is calculated as: pod_size * local_world_size
        This represents the total number of GPUs connected via high-bandwidth
        NVLink within a single domain.

        Returns:
            A Topology object representing the network topology for distributed training
        """
        local_world_size = get_local_size(self.world_size)

        # Build topology kwargs from hardware config
        topology_kwargs: Dict[str, Any] = {
            "world_size": self.world_size,
            "local_world_size": local_world_size,
            "compute_device": device_type,
        }

        self._apply_hardware_config(topology_kwargs, device_type)

        # Auto-detect hbm_cap when not provided in hardware config
        if "hbm_cap" not in topology_kwargs:
            detected_hbm = _detect_hbm_cap(device_type)
            if detected_hbm is not None:
                topology_kwargs["hbm_cap"] = detected_hbm
                logger.info(
                    "Auto-detected hbm_cap=%d (%.1f GB) from GPU.",
                    detected_hbm,
                    detected_hbm / (1024**3),
                )

        return Topology(**topology_kwargs)

    def table_to_constraint(
        self,
        table: Union[EmbeddingConfig, EmbeddingBagConfig],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ParameterConstraints]:
        default_kwargs = {
            "sharding_types": [self.sharding_type.value],
            "compute_kernels": [self.compute_kernel.value],
            "device_group": self.device_group,
            "pooling_factors": self.pooling_factors,
            "num_poolings": self.num_poolings,
            "batch_sizes": self.batch_sizes,
        }
        if kwargs is None:
            kwargs = default_kwargs
        else:
            kwargs = default_kwargs | kwargs

        # (KVZCH) Convert key_value_params dict to KeyValueParams object if present
        if "key_value_params" in kwargs:
            key_value_params = kwargs["key_value_params"]
            # If eviction policy is set then construct object
            if (
                isinstance(key_value_params, dict)
                and "kvzch_tbe_config" in key_value_params
            ):
                key_value_params["kvzch_tbe_config"] = KVZCHTBEConfig(
                    **key_value_params["kvzch_tbe_config"]
                )
            # pyrefly: ignore[bad-unpacking, unsupported-operation]
            kwargs["key_value_params"] = KeyValueParams(**key_value_params)

        # Convert cache_params dict to CacheParams object if present
        if "cache_params" in kwargs:
            # pyrefly: ignore[bad-unpacking, unsupported-operation]
            kwargs["cache_params"] = CacheParams(**kwargs["cache_params"])

        # pyrefly: ignore[bad-argument-type]
        constraint = ParameterConstraints(**kwargs)
        return table.name, constraint

    def generate_planner(
        self,
        tables: List[EmbeddingBagConfig],
    ) -> ShardingPlanner:
        """
        Generate an embedding sharding planner based on the specified configuration.

        Args:
            tables: List of unweighted embedding tables

        Returns:
            A ShardingPlanner instance (EmbeddingShardingPlanner, LinearProgrammingPlanner, or HeteroEmbeddingShardingPlanner)

        Raises:
            RuntimeError: If an unknown planner type is specified
        """
        # Create parameter constraints for tables
        constraints = {}

        topology = self.generate_topology(self.device_group)

        for table in tables:
            name, cons = self.table_to_constraint(
                table, self.additional_constraints.get(table.name, None)
            )
            constraints[name] = cons

        # Create storage reservation if percentage > 0
        storage_reservation = (
            HeuristicalStorageReservation(
                percentage=self.storage_reservation_percentage
            )
            if self.storage_reservation_percentage > 0
            else None
        )

        if self.planner_type == "embedding":
            return EmbeddingShardingPlanner(
                topology=topology,
                batch_size=self.batch_size,
                constraints=constraints if constraints else None,
                storage_reservation=storage_reservation,
            )
        elif self.planner_type == "lp":
            return LinearProgrammingPlanner(
                topology=topology,
                batch_size=self.batch_size,
                constraints=constraints if constraints else None,
                storage_reservation=storage_reservation,
            )
        elif self.planner_type == "hetero":
            topology_groups = {self.device_group: topology}
            return HeteroEmbeddingShardingPlanner(
                topology_groups=topology_groups,
                batch_size=self.batch_size,
                constraints=constraints if constraints else None,
                storage_reservation=storage_reservation,  # pyrefly: ignore[unexpected-keyword]
            )
        else:
            raise RuntimeError(f"Unknown planner type: {self.planner_type}")


def _get_sharders_with_fused_params(
    fused_params: Optional[Dict[str, Any]] = None,
) -> List[ModuleSharder[nn.Module]]:
    """
    Get EBC and EC sharders configured with fused parameters for sparse optimizers.

    This function creates sharders that will use the specified fused_params
    (including optimizer type like Shampoo) for embedding operations.

    Args:
        fused_params: Dictionary of fused parameters including optimizer settings.
            Example: {"optimizer": EmbOptimType.SHAMPOO, "learning_rate": 0.01}

    Returns:
        List of sharders configured with the fused parameters
    """
    if not fused_params:
        return get_default_sharders()
    return [
        cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ),
        cast(
            ModuleSharder[nn.Module],
            EmbeddingCollectionSharder(fused_params=fused_params),
        ),
    ]


@dataclass
class ShardingConfig:
    """
    Configuration for generating a sharded model and optimizer for distributed training.

    This dataclass encapsulates all the parameters needed to create a sharded model
    with proper optimizer configuration for both sparse and dense parameters.

    Args:
        fused_params: Parameters for the fused sparse optimizer. This includes
            optimizer settings like {"optimizer": "EXACT_ROWWISE_ADAGRAD", "learning_rate": 0.01}.
            The optimizer can be specified as a string (e.g., "EXACT_ROWWISE_ADAGRAD")
            or as an EmbOptimType enum.
            Supported optimizers include EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, ADAM, etc.
        dense_optimizer: Optimizer type for dense parameters. Supported values include
            standard torch.optim optimizers (SGD, Adam, AdamW, etc.) and "Shampoo"
            for the distributed Shampoo optimizer.
        dense_lr: Learning rate for dense parameters.
        dense_momentum: Momentum for dense parameters (optional).
        dense_weight_decay: Weight decay for dense parameters (optional).
        dense_optimizer_kwargs: Additional keyword arguments forwarded to the dense
            optimizer constructor. For Shampoo, this can include
            precondition_frequency, start_preconditioning_step,
            max_preconditioner_dim, etc.
        init_data_parallel: Whether to wrap in DDP. Set to False for eval
            workflows that don't need gradient synchronization.
        embedding_device: If set, DMP uses this device for embeddings (e.g. "cpu").
            When set, the .to(device) call is skipped to preserve split-device
            placement (CPU embeddings + GPU dense).
        skip_dense_optimizer: If True, return a dummy optimizer instead of creating
            a real one for dense parameters. Useful for eval-only workflows.
        deepcopy_model: If True (default), deepcopy the model before passing to DMP.
            Set to False to save memory when the original model is not needed.
    """

    fused_params: Dict[str, Any] = field(default_factory=dict)
    dense_optimizer: str = "SGD"
    dense_lr: float = 0.1
    dense_momentum: Optional[float] = None
    dense_weight_decay: Optional[float] = None
    dense_optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    init_data_parallel: bool = True
    embedding_device: Optional[str] = None
    skip_dense_optimizer: bool = False
    deepcopy_model: bool = True

    def _convert_fused_params(self) -> Optional[Dict[str, Any]]:
        """
        Convert fused_params optimizer string to EmbOptimType enum if needed.

        Returns:
            Converted fused_params dict with EmbOptimType enum, or None if empty.
        """
        if not self.fused_params:
            return None

        fused_params = dict(self.fused_params)

        # Convert optimizer string to EmbOptimType enum if it's a string
        if "optimizer" in fused_params and isinstance(fused_params["optimizer"], str):
            optimizer_str = fused_params["optimizer"].upper()
            try:
                fused_params["optimizer"] = EmbOptimType[optimizer_str]
            except KeyError:
                raise ValueError(
                    f"Unknown optimizer type: {optimizer_str}. "
                    f"Valid options: {[e.name for e in EmbOptimType]}"
                )

        return fused_params

    def _plan_and_sharders(
        self,
        model: nn.Module,
        pg: dist.ProcessGroup,
        planner: Optional[ShardingPlanner] = None,
    ) -> Tuple[List[ModuleSharder[nn.Module]], Optional[ShardingPlan]]:
        """
        Convert fused params, create sharders, and run the planner.

        Returns:
            Tuple of (sharders, plan)
        """
        converted_fused_params = self._convert_fused_params()
        sharders = _get_sharders_with_fused_params(converted_fused_params)

        plan = None
        if planner is not None:
            if pg is not None:
                plan = planner.collective_plan(model, sharders, pg)  # pyre-ignore[28]
            else:
                # pyrefly: ignore[bad-argument-type, missing-argument]
                plan = planner.plan(model, sharders)

        return sharders, plan

    def generate_dmp_model(
        self,
        model: nn.Module,
        pg: dist.ProcessGroup,
        device: torch.device,
        planner: Optional[ShardingPlanner] = None,
    ) -> DistributedModelParallel:
        """
        Generate a standard DistributedModelParallel model.

        All modules are placed on the same device.
        """
        sharders, plan = self._plan_and_sharders(model, pg, planner)

        return DistributedModelParallel(
            module=copy.deepcopy(model) if self.deepcopy_model else model,
            env=ShardingEnv.from_process_group(pg),
            init_data_parallel=self.init_data_parallel,
            device=device,
            sharders=sharders,
            plan=plan,
        ).to(device)

    def generate_hybrid_dmp_model(
        self,
        model: nn.Module,
        pg: dist.ProcessGroup,
        device: torch.device,
        planner: Optional[ShardingPlanner] = None,
    ) -> HybridEvalDMP:
        """
        Generate a HybridEvalDMP model for split-device placement.

        Embeddings are placed on ``embedding_device`` (e.g. CPU),
        while dense modules are moved to ``device`` (e.g. CUDA).
        CPU embedding storage is shared across ranks via /dev/shm.
        """
        assert (
            self.embedding_device is not None
        ), "embedding_device must be set for generate_hybrid_dmp_model"
        sharders, plan = self._plan_and_sharders(model, pg, planner)

        model = HybridEvalDMP(
            module=copy.deepcopy(model) if self.deepcopy_model else model,
            env=ShardingEnv.from_process_group(pg),
            device=torch.device(self.embedding_device),
            sharders=sharders,
            plan=plan,
        ).to(device)
        model.share_embedding_memory(pg)
        return model

    def generate_dense_optimizer(
        self,
        sharded_model: nn.Module,
    ) -> Optimizer:
        """
        Generate an optimizer for dense (non-sparse) parameters.

        If skip_dense_optimizer is True, returns a dummy optimizer.
        """
        if self.skip_dense_optimizer:
            return torch.optim.SGD([torch.zeros(1)], lr=0.0)

        dense_params = [
            param
            for name, param in sharded_model.named_parameters()
            if "sparse" not in name
        ]

        optimizer_kwargs: Dict[str, Any] = {"lr": self.dense_lr}
        if self.dense_momentum is not None:
            optimizer_kwargs["momentum"] = self.dense_momentum
        if self.dense_weight_decay is not None:
            optimizer_kwargs["weight_decay"] = self.dense_weight_decay

        if self.dense_optimizer.lower() == "shampoo":
            from torchrec.distributed.test_utils.test_modules import DistributedShampoo

            optimizer_class = DistributedShampoo
            if "precondition_frequency" not in self.dense_optimizer_kwargs:
                optimizer_kwargs["precondition_frequency"] = 100
        else:
            optimizer_class = getattr(optim, self.dense_optimizer)

        optimizer_kwargs.update(self.dense_optimizer_kwargs)
        return optimizer_class(dense_params, **optimizer_kwargs)

    def generate_sharded_model_and_optimizer(
        self,
        model: nn.Module,
        pg: dist.ProcessGroup,
        device: torch.device,
        planner: Optional[ShardingPlanner] = None,
    ) -> Tuple[nn.Module, Optimizer]:
        """
        Generate a sharded model and optimizer for distributed training.

        Dispatches to generate_dmp_model or generate_hybrid_dmp_model based
        on whether embedding_device is set, then creates the dense optimizer.
        """
        if self.embedding_device:
            sharded_model = self.generate_hybrid_dmp_model(model, pg, device, planner)
        else:
            sharded_model = self.generate_dmp_model(model, pg, device, planner)

        optimizer = self.generate_dense_optimizer(sharded_model)
        return sharded_model, optimizer
