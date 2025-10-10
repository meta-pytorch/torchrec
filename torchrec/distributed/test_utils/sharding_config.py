#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from torchrec.distributed.comm import get_local_size

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig


@dataclass
class PlannerConfig:
    planner_type: str = "embedding"
    world_size: int = 2
    device_group: str = "cuda"
    pooling_factors: List[float] = field(default_factory=lambda: [POOLING_FACTOR])
    num_poolings: Optional[List[float]] = None
    batch_sizes: Optional[List[int]] = None
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.FUSED
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    additional_constraints: Dict[str, Any] = field(default_factory=dict)

    def generate_topology(self, device_type: str) -> Topology:
        """
        Generate a topology for distributed training.

        Returns:
            A Topology object representing the network topology for distributed training
        """
        local_world_size = get_local_size(self.world_size)
        return Topology(
            world_size=self.world_size,
            local_world_size=local_world_size,
            compute_device=device_type,
        )

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

        constraint = ParameterConstraints(**kwargs)  # pyre-ignore [6]
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

        if self.planner_type == "embedding":
            return EmbeddingShardingPlanner(
                topology=topology,
                constraints=constraints if constraints else None,
            )
        elif self.planner_type == "hetero":
            topology_groups = {self.device_group: topology}
            return HeteroEmbeddingShardingPlanner(
                topology_groups=topology_groups,
                constraints=constraints if constraints else None,
            )
        else:
            raise RuntimeError(f"Unknown planner type: {self.planner_type}")
