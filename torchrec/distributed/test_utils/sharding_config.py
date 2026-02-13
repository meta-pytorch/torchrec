#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from fbgemm_gpu.split_table_batched_embeddings_ops_common import KVZCHTBEConfig
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import CacheParams, ParameterConstraints
from torchrec.distributed.types import KeyValueParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig


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

    def generate_topology(self, device_type: str) -> Topology:
        """
        Generate a topology for distributed training.

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

        if self.hardware is not None:
            if "hbm_cap" in self.hardware:
                topology_kwargs["hbm_cap"] = self.hardware["hbm_cap"]
            if "ddr_cap" in self.hardware:
                topology_kwargs["ddr_cap"] = self.hardware["ddr_cap"]
            if "hbm_mem_bw" in self.hardware:
                topology_kwargs["hbm_mem_bw"] = self.hardware["hbm_mem_bw"]
            if "ddr_mem_bw" in self.hardware:
                topology_kwargs["ddr_mem_bw"] = self.hardware["ddr_mem_bw"]
            if "hbm_to_ddr_mem_bw" in self.hardware:
                topology_kwargs["hbm_to_ddr_mem_bw"] = self.hardware[
                    "hbm_to_ddr_mem_bw"
                ]
            if "intra_host_bw" in self.hardware:
                topology_kwargs["intra_host_bw"] = self.hardware["intra_host_bw"]
            if "inter_host_bw" in self.hardware:
                topology_kwargs["inter_host_bw"] = self.hardware["inter_host_bw"]

        return Topology(**topology_kwargs)

    def table_to_constraint(
        self,
        table: Union[EmbeddingConfig, EmbeddingBagConfig],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, ParameterConstraints]:
        default_kwargs = dict(
            sharding_types=[self.sharding_type.value],
            compute_kernels=[self.compute_kernel.value],
            device_group=self.device_group,
            pooling_factors=self.pooling_factors,
            num_poolings=self.num_poolings,
            batch_sizes=self.batch_sizes,
        )
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
    ) -> Union[EmbeddingShardingPlanner, HeteroEmbeddingShardingPlanner]:
        """
        Generate an embedding sharding planner based on the specified configuration.

        Args:
            tables: List of unweighted embedding tables

        Returns:
            An instance of EmbeddingShardingPlanner or HeteroEmbeddingShardingPlanner

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
