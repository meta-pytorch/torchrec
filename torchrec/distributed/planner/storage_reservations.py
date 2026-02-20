#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

from torch import nn
from torchrec.distributed.planner.constants import BIGINT_DTYPE, POOLING_FACTOR
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PlannerError,
    PlannerErrorType,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name, storage_repr_in_gb
from torchrec.distributed.types import get_tensor_size_bytes, ModuleSharder


logger: logging.Logger = logging.getLogger(__name__)


def _get_module_size(module: nn.Module, multiplier: float) -> int:
    parameters_size = sum(
        [
            multiplier * get_tensor_size_bytes(parameter)
            for parameter in module.parameters()
        ]
    )

    buffers_size = sum([get_tensor_size_bytes(buffer) for buffer in module.buffers()])

    return round(parameters_size + buffers_size)


def _get_dense_tensor_size(
    module: nn.Module,
    shardable_modules: Set[nn.Module],
    multiplier: float = 6.0,
) -> int:
    dense_tensor_size = _get_module_size(module, multiplier) - sum(
        [
            _get_module_size(shardable_module, multiplier)
            for shardable_module in shardable_modules
        ]
    )
    return dense_tensor_size


def _reserve_dense_storage(
    topology: Topology,
    module: nn.Module,
    shardable_modules: Set[nn.Module],
    multiplier: float,
    dense_tensor_estimate: Optional[int] = None,
) -> Storage:

    dense_tensor_size = _get_dense_tensor_size(module, shardable_modules, multiplier)
    if dense_tensor_estimate:
        logger.info(
            f"We override default dense tensor estimate ({dense_tensor_size} bytes) "
            f"with user-provided dense tensor estimate ({dense_tensor_estimate} bytes)."
        )
        dense_tensor_size = dense_tensor_estimate
    else:
        logger.warning(
            "There is a known issue with TorchRec's dense tensor size calculation in dry run scenarios, where tensors are not materialized. Consider passing in a dense_tensor_estimate to planner input if you are running in dry run environment."
        )

    dense_tensor_storage = Storage(
        hbm=dense_tensor_size if topology.compute_device in {"cuda", "mtia"} else 0,
        ddr=dense_tensor_size if topology.compute_device == "cpu" else 0,
        ssd=0,
    )

    for device in topology.devices:
        device.storage -= dense_tensor_storage

    return dense_tensor_storage


def _get_kjt_storage(
    topology: Topology,
    batch_inputs: List[float],
    input_data_type_size: int,
    multiplier: int,
) -> Storage:
    kjt_size = math.ceil(sum(batch_inputs) * float(input_data_type_size)) * multiplier
    return Storage(
        hbm=kjt_size if topology.compute_device in {"cuda", "mtia"} else 0,
        ddr=kjt_size if topology.compute_device == "cpu" else 0,
        ssd=0,
    )


def _reserve_kjt_storage(
    topology: Topology,
    batch_size: int,
    batch_inputs: List[float],
    input_data_type_size: int,
    multiplier: int,
) -> Storage:
    kjt_storage = _get_kjt_storage(
        topology, batch_inputs, input_data_type_size, multiplier
    )

    for device in topology.devices:
        device.storage -= kjt_storage

    return kjt_storage


def _reserve_storage_percentage(topology: Topology, percent: float) -> None:
    for device in topology.devices:
        device.storage.hbm = int((1 - percent) * device.storage.hbm)


def _get_batch_inputs_and_shardable_parameters(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
    batch_size: int,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
) -> Tuple[List[float], Set[nn.Module]]:
    sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
        sharder_name(sharder.module_type): sharder for sharder in sharders
    }
    input_lengths: List[float] = []
    batch_sizes: List[int] = []
    shardable_modules: Set[nn.Module] = set()

    def populate_shardable_modules(
        module: nn.Module,
    ) -> None:
        sharder_key = sharder_name(type(module))
        sharder = sharder_map.get(sharder_key)

        if not sharder:
            for _child_name, child in module.named_children():
                populate_shardable_modules(child)
        else:
            names = sharder.shardable_parameters(module).keys()
            shardable_modules.add(module)

            for name in names:
                pooling_factors = (
                    constraints[name].pooling_factors
                    if constraints and constraints.get(name)
                    else [POOLING_FACTOR]
                )
                input_lengths.extend(pooling_factors)
                batch_sizes.extend(
                    # pyrefly: ignore[bad-argument-type]
                    constraints[name].batch_sizes
                    if constraints
                    and constraints.get(name)
                    and constraints[name].batch_sizes
                    else [batch_size] * len(pooling_factors)
                )

    populate_shardable_modules(module)

    batch_inputs: List[float] = [
        input_length * batch_size
        for input_length, batch_size in zip(input_lengths, batch_sizes)
    ]

    return batch_inputs, shardable_modules


class FixedPercentageStorageReservation(StorageReservation):
    def __init__(self, percentage: float) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage
        self._last_reserved_topology: Optional[Topology] = None
        self._kjt_storage: Optional[Storage] = None

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)
        _reserve_storage_percentage(reserved_topology, self._percentage)
        self._last_reserved_topology = reserved_topology
        # save the estimated kjt size (no memory reservation)
        batch_inputs, _ = _get_batch_inputs_and_shardable_parameters(
            module, sharders, batch_size, constraints
        )
        self._kjt_storage = _get_kjt_storage(
            topology=topology,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            # 2 pipelined batches each with 10 internal copies
            multiplier=20,
        )
        return reserved_topology

    @property
    def last_reserved_topology(self) -> Optional[Topology]:
        "Returns a copy of the cached value of the most recent output from the reserve() method."
        return copy.deepcopy(self._last_reserved_topology)


class HeuristicalStorageReservation(StorageReservation):
    """
    Reserves storage for model to be sharded with heuristical calculation. The storage
    reservation is comprised of dense tensor storage, KJT storage, and an extra
    percentage of total storage.

    Args:
        percentage (float): extra storage percent to reserve that acts as a margin of
            error beyond heuristic calculation of storage.
        parameter_multiplier (float): heuristic multiplier for total parameter storage.
        dense_tensor_estimate (Optional[int]): storage estimate for dense tensors, uses
            default heuristic estimate if not provided.
    """

    def __init__(
        self,
        percentage: float,
        # heuristic: 6 * dense parameter size
        # parameter + optimizer (~2x parameter) + ddp (~3x parameter)
        parameter_multiplier: float = 6.0,
        dense_tensor_estimate: Optional[int] = None,
    ) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage
        self._parameter_multiplier = parameter_multiplier
        self._dense_tensor_estimate = dense_tensor_estimate

        self._dense_storage: Optional[Storage] = None
        self._kjt_storage: Optional[Storage] = None
        self._last_reserved_topology: Optional[Topology] = None

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        # TODO: enable proper caching of topology values through _last_reserved_topology
        reserved_topology = copy.deepcopy(topology)

        batch_inputs, shardable_modules = _get_batch_inputs_and_shardable_parameters(
            module, sharders, batch_size, constraints
        )

        _reserve_storage_percentage(reserved_topology, self._percentage)

        self._dense_storage = _reserve_dense_storage(
            topology=reserved_topology,
            module=module,
            shardable_modules=shardable_modules,
            multiplier=self._parameter_multiplier,
            dense_tensor_estimate=self._dense_tensor_estimate,
        )

        self._kjt_storage = _reserve_kjt_storage(
            topology=reserved_topology,
            batch_size=batch_size,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            # 2 pipelined batches each with 10 internal copies
            multiplier=20,
        )

        if reserved_topology.devices[0].storage.hbm < 0:
            negative_storage_solution = (
                f"The reserved topology ({storage_repr_in_gb(reserved_topology.devices[0].storage)}) "
                "has negative available hbm storage, "
                "after taking into account of the reserved hbm percentage, "
                "the storage for dense modules, and the kjt storages. Hence "
                "it is not possible to find a valid sharding plan. "
                "\n \n Note: There is a known issue with dense storage estimation in dry run scenario where tensors are not fully materialized."
                f"If the dense storage ({storage_repr_in_gb(self._dense_storage)}) looks higher than expected, consider passing in dense_storage_estimate as a Planner input to override torchrec dense tensor size calculation while we resolve this issue."
                "\n \n Other Possible solutions:"
                "\n  1) If FSDP is used, consider switching to FixedPercentageStorageReservation, since "
                f"HeuristicalStorageReservation would not be able to calculate the "
                f"dense storage ({storage_repr_in_gb(self._dense_storage)}) correctly. "
                f"\n  2) Reduce local batch size ({batch_size}), which can help "
                f"reduce the per rank kjt storage ({storage_repr_in_gb(self._kjt_storage)}). "
                f"\n  3) Decrease the reserved hbm percentage ({self._percentage}). "
                "\n  4) Use hardware with a higher hbm cap (current hardware has "
                f"{storage_repr_in_gb(topology.devices[0].storage)} per rank). "
            )
            raise PlannerError(
                error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                message=negative_storage_solution,
            )

        self._last_reserved_topology = copy.deepcopy(reserved_topology)
        return reserved_topology

    @property
    def last_reserved_topology(self) -> Optional[Topology]:
        "Cached value of the most recent output from the reserve() method."
        return self._last_reserved_topology


class InferenceStorageReservation(StorageReservation):
    """
    Reserves storage for model to be sharded for inference. The storage reservation
    is comprised of dense tensor storage, KJT storage, and an extra percentage of total
    storage. Note that when estimating for storage, dense modules are assumed to be on
    GPUs and replicated across ranks. If this is not the case, please override the
    estimates with dense_tensor_estimate.

    Args:
        percentage (float): extra storage percentage to reserve that acts as a margin of
            error beyond storage calculation.
        dense_tensor_estimate (Optional[int]): storage estimate for dense tensors, use
            default heuristic estimate if not provided.
    """

    def __init__(
        self,
        percentage: float,
        dense_tensor_estimate: Optional[int] = None,
    ) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage
        self._dense_tensor_estimate = dense_tensor_estimate

        self._dense_storage: Optional[Storage] = None
        self._kjt_storage: Optional[Storage] = None
        self._last_reserved_topology: Optional[Topology] = None

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)

        batch_inputs, shardable_modules = _get_batch_inputs_and_shardable_parameters(
            module, sharders, batch_size, constraints
        )

        _reserve_storage_percentage(reserved_topology, self._percentage)

        self._dense_storage = _reserve_dense_storage(
            topology=reserved_topology,
            module=module,
            shardable_modules=shardable_modules,
            multiplier=1,
            dense_tensor_estimate=self._dense_tensor_estimate,
        )

        self._kjt_storage = _reserve_kjt_storage(
            topology=reserved_topology,
            batch_size=batch_size,
            batch_inputs=batch_inputs,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=1,
        )

        self._last_reserved_topology = copy.deepcopy(reserved_topology)

        return reserved_topology

    # pyrefly: ignore[bad-override]
    def last_reserved_topology(self) -> Optional[Topology]:
        return copy.deepcopy(self._last_reserved_topology)
