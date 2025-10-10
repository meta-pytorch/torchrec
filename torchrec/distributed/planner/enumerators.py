#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    PartitionByType,
    Shard,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.sharding_plan import calculate_shard_sizes_and_offsets
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import DataType
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection


logger: logging.Logger = logging.getLogger(__name__)

# compute kernels that should only be used if users specified them
GUARDED_COMPUTE_KERNELS: Set[EmbeddingComputeKernel] = {
    EmbeddingComputeKernel.KEY_VALUE,
    EmbeddingComputeKernel.SSD_VIRTUAL_TABLE,
    EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE,
}


class EmbeddingEnumerator(Enumerator):
    """
    Generates embedding sharding options for given `nn.Module`, considering user provided
    constraints.

    Args:
        topology (Topology): device topology.
        batch_size (int): batch size.
        constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter names
            to provided ParameterConstraints.
        estimator (Optional[Union[ShardEstimator, List[ShardEstimator]]]): shard performance estimators.
        use_exact_enumerate_order (bool): whether to enumerate shardable parameters in the exact name_children enumeration order
    """

    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
    ) -> None:
        self._compute_device: str = topology.compute_device
        self._world_size: int = topology.world_size
        self._local_world_size: int = topology.local_world_size
        self._batch_size: int = batch_size
        self._constraints = constraints
        self._sharder_map: Dict[str, ModuleSharder[nn.Module]] = {}
        self._use_exact_enumerate_order: bool = (
            use_exact_enumerate_order if use_exact_enumerate_order else False
        )
        memory_type = (
            "hbm_cap" if topology.compute_device in {"cuda", "mtia"} else "ddr_cap"
        )
        self._device_memory_sizes: Optional[
            List[int]
        ] = (  # only used with custom topology where memory is different within a topology
            topology._custom_topology_data.get_data(memory_type)
            if topology._custom_topology_data
            and topology._custom_topology_data.has_data(memory_type)
            else None
        )

        if estimator:
            self._estimators: List[ShardEstimator] = (
                [estimator] if not isinstance(estimator, list) else estimator
            )
        else:
            self._estimators: List[ShardEstimator] = [
                EmbeddingPerfEstimator(topology=topology, constraints=constraints),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ]

        # Initializing caching for enumerate
        self._last_stored_search_space: Optional[List[ShardingOption]] = None
        self._last_stored_module: Optional[nn.Module] = None
        self._last_stored_sharders: Optional[List[ModuleSharder[nn.Module]]] = None

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """
        Generates relevant sharding options given module and sharders.

        Args:
            module (nn.Module): module to be sharded.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            List[ShardingOption]: valid sharding options with values populated.
        """

        if (
            self._last_stored_module == module
            and self._last_stored_sharders == sharders
        ):
            return copy.deepcopy(self._last_stored_search_space)  # pyre-ignore

        self._sharder_map = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        sharding_options: List[ShardingOption] = []

        named_modules_queue = [("", module)]
        while named_modules_queue:
            if not self._use_exact_enumerate_order:
                child_path, child_module = named_modules_queue.pop()
            else:
                child_path, child_module = named_modules_queue.pop(0)
            sharder_key = sharder_name(type(child_module))
            sharder = self._sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
                continue

            # Determine the pooling state for all sharding_options using this
            # (child_module, child_path). With this optimization, we change enumerate()
            # from being O(N^2) with respect to the number of tables to O(N). The
            # previous quadratic behavior is because in populate_estimates() invoked below, each
            # sharding_option needs to determine its pooling state, which is does via
            # an expensive O(N) walk through the list of embedding tables. With this
            # change sharding_option.is_pooled becomes O(1).
            is_pooled = ShardingOption.module_pooled(child_module, child_path)

            for name, param in sharder.shardable_parameters(child_module).items():
                (
                    input_lengths,
                    col_wise_shard_dim,
                    cache_params,
                    enforce_hbm,
                    stochastic_rounding,
                    bounds_check_mode,
                    feature_names,
                    output_dtype,
                    device_group,
                    key_value_params,
                ) = _extract_constraints_for_param(self._constraints, name)

                # skip for other device groups
                if device_group and device_group != self._compute_device:
                    continue

                sharding_options_per_table: List[ShardingOption] = []

                for sharding_type in self._filter_sharding_types(
                    name, sharder.sharding_types(self._compute_device)
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                        sharding_type,
                    ):
                        try:
                            (
                                shard_sizes,
                                shard_offsets,
                            ) = calculate_shard_sizes_and_offsets(
                                tensor=param,
                                world_size=self._world_size,
                                local_world_size=self._local_world_size,
                                sharding_type=sharding_type,
                                col_wise_shard_dim=col_wise_shard_dim,
                                device_memory_sizes=self._device_memory_sizes,
                            )
                        except ZeroDivisionError as e:
                            # Re-raise with additional context about the table and module
                            raise ValueError(
                                f"Failed to calculate sharding plan: {str(e)} "
                                f"Context: table_name='{name}', module_path='{child_path}', "
                                f"module_type='{type(child_module).__name__}', "
                                f"sharder='{sharder.__class__.__name__}', "
                                f"tensor.shape={param.shape}, sharding_type='{sharding_type}', "
                                f"compute_kernel='{compute_kernel}', world_size={self._world_size}, "
                                f"local_world_size={self._local_world_size}, "
                                f"col_wise_shard_dim={col_wise_shard_dim}, "
                                f"compute_device='{self._compute_device}'"
                            ) from e
                        dependency = None
                        if isinstance(child_module, EmbeddingTower):
                            dependency = child_path
                        elif isinstance(child_module, EmbeddingTowerCollection):
                            tower_index = _get_tower_index(name, child_module)
                            dependency = child_path + ".tower_" + str(tower_index)
                        sharding_options_per_table.append(
                            ShardingOption(
                                name=name,
                                tensor=param,
                                module=(child_path, child_module),
                                input_lengths=input_lengths,
                                batch_size=self._batch_size,
                                compute_kernel=compute_kernel,
                                sharding_type=sharding_type,
                                partition_by=get_partition_by_type(sharding_type),
                                shards=[
                                    Shard(size=size, offset=offset)
                                    for size, offset in zip(shard_sizes, shard_offsets)
                                ],
                                cache_params=cache_params,
                                enforce_hbm=enforce_hbm,
                                stochastic_rounding=stochastic_rounding,
                                bounds_check_mode=bounds_check_mode,
                                dependency=dependency,
                                is_pooled=is_pooled,
                                feature_names=feature_names,
                                output_dtype=output_dtype,
                                key_value_params=key_value_params,
                            )
                        )
                if not sharding_options_per_table:
                    raise RuntimeError(
                        "No available sharding type and compute kernel combination "
                        f"after applying user provided constraints for {name}. "
                        f"Module: {sharder_key}, sharder: {sharder.__class__.__name__}, compute device: {self._compute_device}. "
                        f"To debug, search above for warning logs about no available sharding types/compute kernels for table: {name}"
                    )

                sharding_options.extend(sharding_options_per_table)

        self.populate_estimates(sharding_options)

        self._last_stored_module = module
        self._last_stored_sharders = sharders

        # Caching the search space with a copy of sharding options, to avoid unexpected modifications to list
        self._last_stored_search_space = copy.deepcopy(sharding_options)
        return sharding_options

    @property
    def last_stored_search_space(self) -> Optional[List[ShardingOption]]:
        # NOTE: This is the last search space stored by enumerate(...), do not use
        # this field in place of actually calling enumerate(...) as it will varie for each
        # module/sharders passed in.
        return self._last_stored_search_space

    def populate_estimates(self, sharding_options: List[ShardingOption]) -> None:
        for estimator in self._estimators:
            estimator.estimate(sharding_options, self._sharder_map)

    def _filter_sharding_types(
        self, name: str, allowed_sharding_types: List[str]
    ) -> List[str]:
        # GRID_SHARD is only supported if specified by user in parameter constraints
        if not self._constraints or not self._constraints.get(name):
            return [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
        constraints: ParameterConstraints = self._constraints[name]
        if not constraints.sharding_types:
            return [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
        constrained_sharding_types: List[str] = constraints.sharding_types

        filtered_sharding_types = list(
            set(constrained_sharding_types) & set(allowed_sharding_types)
        )
        if not filtered_sharding_types:
            logger.warning(
                "No available sharding types after applying user provided "
                f"constraints for {name}. Constrained sharding types: "
                f"{constrained_sharding_types}, allowed sharding types: "
                f"{allowed_sharding_types}, filtered sharding types: "
                f"{filtered_sharding_types}. Please check if the constrained "
                "sharding types are too restrictive, if the sharder allows the "
                "sharding types, or if non-strings are passed in."
            )
        return filtered_sharding_types

    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
    ) -> List[str]:
        # setup constrained_compute_kernels
        if (
            self._constraints
            and self._constraints.get(name)
            and self._constraints[name].compute_kernels
        ):
            # pyre-ignore
            constrained_compute_kernels: List[str] = self._constraints[
                name
            ].compute_kernels
        else:
            constrained_compute_kernels: List[str] = [
                compute_kernel.value
                for compute_kernel in EmbeddingComputeKernel
                if compute_kernel not in GUARDED_COMPUTE_KERNELS
            ]

        # setup filtered_compute_kernels
        filtered_compute_kernels = list(
            set(constrained_compute_kernels) & set(allowed_compute_kernels)
        )

        # special rules
        if EmbeddingComputeKernel.DENSE.value in filtered_compute_kernels:
            if (
                EmbeddingComputeKernel.FUSED.value in filtered_compute_kernels
            ):  # always false for data_parallel
                filtered_compute_kernels.remove(EmbeddingComputeKernel.DENSE.value)

        if not filtered_compute_kernels:
            logger.warning(
                "No available compute kernels after applying user provided "
                f"constraints for {name}. Constrained compute kernels: "
                f"{constrained_compute_kernels}, allowed compute kernels: "
                f"{allowed_compute_kernels}, filtered compute kernels: "
                f"{filtered_compute_kernels}, sharding type: {sharding_type}. Please check if the constrained "
                "compute kernels are too restrictive, if the sharder allows the "
                "compute kernels, or if non-strings are passed in."
            )
        return filtered_compute_kernels


def _extract_constraints_for_param(
    constraints: Optional[Dict[str, ParameterConstraints]], name: str
) -> Tuple[
    List[float],
    Optional[int],
    Optional[CacheParams],
    Optional[bool],
    Optional[bool],
    Optional[BoundsCheckMode],
    Optional[List[str]],
    Optional[DataType],
    Optional[str],
    Optional[KeyValueParams],
]:
    input_lengths = [POOLING_FACTOR]
    col_wise_shard_dim = None
    cache_params = None
    enforce_hbm = None
    stochastic_rounding = None
    bounds_check_mode = None
    feature_names = None
    output_dtype = None
    device_group = None
    key_value_params = None

    if constraints and constraints.get(name):
        # For nested fields return a deep copy instead
        input_lengths = constraints[name].pooling_factors
        col_wise_shard_dim = constraints[name].min_partition
        cache_params = deepcopy(constraints[name].cache_params)
        enforce_hbm = constraints[name].enforce_hbm
        stochastic_rounding = constraints[name].stochastic_rounding
        bounds_check_mode = constraints[name].bounds_check_mode
        feature_names = constraints[name].feature_names
        output_dtype = constraints[name].output_dtype
        device_group = constraints[name].device_group
        key_value_params = deepcopy(constraints[name].key_value_params)

    return (
        input_lengths,
        col_wise_shard_dim,
        cache_params,
        enforce_hbm,
        stochastic_rounding,
        bounds_check_mode,
        feature_names,
        output_dtype,
        device_group,
        key_value_params,
    )


def get_partition_by_type(sharding_type: str) -> str:
    """
    Gets corresponding partition by type for provided sharding type.

    Args:
        sharding_type (str): sharding type string.

    Returns:
        str: the corresponding `PartitionByType` value.
    """

    device_sharding_types = {
        ShardingType.TABLE_WISE.value,
        ShardingType.COLUMN_WISE.value,
    }
    host_sharding_types = {
        ShardingType.TABLE_ROW_WISE.value,
        ShardingType.TABLE_COLUMN_WISE.value,
    }
    uniform_sharding_types = {
        ShardingType.ROW_WISE.value,
        ShardingType.DATA_PARALLEL.value,
    }
    multi_host_sharding_types = {ShardingType.GRID_SHARD.value}

    if sharding_type in device_sharding_types:
        return PartitionByType.DEVICE.value
    elif sharding_type in host_sharding_types:
        return PartitionByType.HOST.value
    elif sharding_type in uniform_sharding_types:
        return PartitionByType.UNIFORM.value
    elif sharding_type in multi_host_sharding_types:
        return PartitionByType.MULTI_HOST.value

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


def _get_tower_index(name: str, child_module: EmbeddingTowerCollection) -> int:
    for i, tower in enumerate(child_module.towers):
        for n, m in tower.named_modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.EmbeddingBag):
                table_name = n.split(".")[-1]
                if name == table_name:
                    return i
    raise RuntimeError(
        f"couldn't get the tower index for table {name}, tower collection: {child_module}"
    )
