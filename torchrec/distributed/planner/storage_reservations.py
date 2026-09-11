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
import warnings
from enum import Enum, unique
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
from torchrec.distributed.planner.utils import (
    gb_to_bytes,
    sharder_name,
    storage_repr_in_gb,
)
from torchrec.distributed.types import get_tensor_size_bytes, ModuleSharder


logger: logging.Logger = logging.getLogger(__name__)


@unique
class StorageReservationType(str, Enum):
    HEURISTIC = "heuristic"
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_ABSOLUTE = "fixed_absolute"
    SKU_AWARE = "sku_aware"


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


def _reserve_storage_absolute(topology: Topology, hbm_bytes: int) -> None:
    for device in topology.devices:
        device.storage.hbm = max(0, device.storage.hbm - hbm_bytes)


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
        assert (
            percentage >= 0 and percentage <= 1
        ), f"reserved dense storage percentage must be between 0 and 1, got {percentage}"
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


class FixedAbsoluteStorageReservation(FixedPercentageStorageReservation):
    """
    Reserves a fixed absolute amount of HBM storage on each device, rather
    than a percentage of total HBM. Useful when the non-sharded memory footprint is
    known in advance and does not scale with device capacity.

    Args:
        hbm_reserved_bytes (int): the amount of HBM to reserve per device, in bytes.
    """

    def __init__(self, hbm_reserved_bytes: int) -> None:
        assert hbm_reserved_bytes >= 0
        super().__init__(percentage=0.0)
        self._hbm_reserved_bytes: int = hbm_reserved_bytes

    @classmethod
    def from_gb(cls, hbm_reserved_gb: float) -> "FixedAbsoluteStorageReservation":
        return cls(hbm_reserved_bytes=gb_to_bytes(hbm_reserved_gb))

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = super().reserve(
            topology, batch_size, module, sharders, constraints
        )
        _reserve_storage_absolute(reserved_topology, self._hbm_reserved_bytes)
        self._last_reserved_topology = reserved_topology
        return reserved_topology


class SKUAwareStorageReservation(StorageReservation):
    """
    SKU-correct replacement for HeuristicalStorageReservation. Reserves, per rank:

        reserved = model_base_bytes (STATIC: dense + margin)
                 + kjt              (DYNAMIC, recomputed live from the local batch)
                 + runtime_overhead_bytes

    The STATIC base has two forms:
      - computed (default): ``margin_bytes`` (``percentage * HBM[home]``, anchored to a FIXED
        home SKU so it does not scale with the landing SKU) + ``dense`` (``params * multiplier``,
        computed from the module, optionally overridden by ``dense_tensor_estimate``);
      - provided: an explicit ``model_base_bytes`` (a user value, or a KR 2.9 measured static
        base) that REPLACES the computed margin + dense.

    The DYNAMIC term (``kjt`` today; a measured activation term later) is ALWAYS recomputed live
    from the current local batch, so the reservation stays correct as the batch / world_size
    changes (variable trainers, checkpoint-eval).

    HeuristicalStorageReservation reserves ``dense + kjt + percentage * HBM[current]``; the last
    term is the only one that scales with the device the job lands on, a dominant cause of
    cross-SKU planner OOM/mis-scaling under hardware fungibility. With the computed static base
    and ``runtime_overhead_bytes == 0``, this class is byte-identical to Heuristical on the home
    SKU (an exact no-op) while being SKU-invariant. ``margin_bytes`` and ``runtime_overhead_bytes``
    are resolved by the framework-side factory (the OSS planner cannot read the internal hardware
    registry).

    Args:
        margin_bytes (int): home-anchored percentage margin (``percentage * HBM[home]``); used
            only when ``model_base_bytes`` is not provided.
        runtime_overhead_bytes (int): per-SKU runtime tax (driver/context, NCCL buffers,
            allocator fragmentation); the slot for exact measured overhead (KR 2.9). Defaults
            to 0 (exact Heuristical parity).
        parameter_multiplier (float): dense-footprint multiplier for the computed dense
            (parameter + optimizer + DDP); used only when ``model_base_bytes`` is not provided.
        model_base_bytes (Optional[int]): explicit STATIC base (dense + margin) that replaces the
            computed margin + dense when set (user-provided or a KR 2.9 measured static base).
        dense_tensor_estimate (Optional[int]): explicit DENSE-only override for the computed path
            (keeps the margin); used only when ``model_base_bytes`` is None - e.g. for FSDP /
            dry-run, or a measured-dense value that should still get the margin on top.
    """

    def __init__(
        self,
        margin_bytes: int,
        runtime_overhead_bytes: int = 0,
        parameter_multiplier: float = 6.0,
        model_base_bytes: Optional[int] = None,
        dense_tensor_estimate: Optional[int] = None,
    ) -> None:
        assert margin_bytes >= 0
        assert runtime_overhead_bytes >= 0
        assert model_base_bytes is None or model_base_bytes >= 0
        self._margin_bytes: int = margin_bytes
        self._runtime_overhead_bytes: int = runtime_overhead_bytes
        self._parameter_multiplier: float = parameter_multiplier
        self._model_base_bytes: Optional[int] = model_base_bytes
        self._dense_tensor_estimate: Optional[int] = dense_tensor_estimate
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

        # Static base as a flooring absolute, applied BEFORE the (non-flooring)
        # module-derived terms so an over-reservation stays visible to the <= 0 guard
        # (_reserve_storage_absolute floors at max(0, ...)). The static base is either
        # an explicit model_base_bytes (dense + margin, provided/measured) or the
        # home-anchored margin (with dense computed below). Overhead is also a flooring
        # absolute, applied here for the same ordering reason.
        _reserve_storage_absolute(
            reserved_topology,
            (
                self._model_base_bytes
                if self._model_base_bytes is not None
                else self._margin_bytes
            ),
        )
        _reserve_storage_absolute(reserved_topology, self._runtime_overhead_bytes)

        # Computed dense (only when the static base is not explicitly provided) plus the
        # live dynamic (kjt) term, recomputed from the current local batch every plan so
        # the reservation stays correct across batch / world_size changes (VT, eval).
        if self._model_base_bytes is None:
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

        # <= 0 (not < 0): the static base + overhead are flooring absolutes, so at the
        # exact-capacity boundary (they consume all hbm and the module terms add 0) the
        # remaining hbm lands at exactly 0, which leaves no room for sharded tables and
        # is still infeasible. Catching it here surfaces an actionable reservation error
        # instead of a confusing downstream "could not place tables" failure.
        if reserved_topology.devices[0].storage.hbm <= 0:
            if self._model_base_bytes is not None:
                static_desc = (
                    f"model_base_bytes {self._model_base_bytes / (1024**3):.2f} GB"
                )
                # An explicit static base was supplied, so there is no analytic dense to
                # replace; the fix is to lower the provided value or the overhead.
                reduce_static_solution = (
                    "\n  1) Reduce model_base_bytes (the provided static base) or "
                    "runtime_overhead_bytes. "
                )
            else:
                static_desc = (
                    f"home-anchored margin {self._margin_bytes / (1024**3):.2f} GB + "
                    f"dense {storage_repr_in_gb(self._dense_storage)}"
                )
                reduce_static_solution = (
                    "\n  1) Supply a measured model_base_bytes / dense_tensor_estimate "
                    "(the analytic dense estimate over-counts under FSDP/dry-run). "
                )
            overhead_gb = self._runtime_overhead_bytes / (1024**3)
            insufficient_storage_solution = (
                f"The SKU-aware storage reservation ({static_desc} + kjt "
                f"{storage_repr_in_gb(self._kjt_storage)} + overhead {overhead_gb:.2f} GB) "
                f"meets or exceeds the available hbm storage per rank "
                f"({storage_repr_in_gb(topology.devices[0].storage)}), leaving no hbm "
                "for sharded embedding tables, so it is not possible to find a valid "
                "sharding plan. "
                "\n \n Possible solutions:"
                + reduce_static_solution
                + f"\n  2) Reduce the local batch size ({batch_size}) to lower the kjt "
                "storage. "
                "\n  3) Use hardware with a higher hbm cap. "
            )
            raise PlannerError(
                error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                message=insufficient_storage_solution,
            )

        self._last_reserved_topology = copy.deepcopy(reserved_topology)
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
        warnings.warn(
            "HeuristicalStorageReservation is known to cause issues, particularly "
            "during dry run where it over-estimates dense tensor size. We recommend "
            "passing a dense_tensor_estimate, or migrating to "
            "FixedPercentageStorageReservation. This policy is over-tuned to specific "
            "use cases.",
            FutureWarning,
            stacklevel=2,
        )
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

    @property
    def last_reserved_topology(self) -> Optional[Topology]:
        return copy.deepcopy(self._last_reserved_topology)
