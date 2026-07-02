#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import hashlib
import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.logger import (
    one_time_logger,
    one_time_rank0_logger,
    static_logger,
)
from torchrec.distributed.planner.constants import (
    BATCH_SIZE,
    BWD_COMPUTE_MULTIPLIER,
    CROSS_NODE_BANDWIDTH,
    DDR_CAP,
    DDR_MEM_BW,
    HBM_CAP,
    HBM_MEM_BW,
    HBM_TO_DDR_MEM_BW,
    HUNDRED_GB,
    INTRA_NODE_BANDWIDTH,
    POOLING_FACTOR,
    SSD_CAP,
    SSD_MEM_BW,
    WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    ShardingPlan,
    StorageUsageType,
)
from torchrec.modules.embedding_configs import DataType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection


# Fractional gap above which a TrainerConfig capacity is treated as a deliberate
# override of the detected HardwareConfig value. Below this, the difference is
# attributable to expected noise -- rounding, or the per-rank vs per-host DDR
# basis (some callers divide the detected per-host DDR by local_world_size) --
# and is not worth flagging. Above it, the trainer value is almost certainly a
# static model config diverging from detected hardware, which is the common
# cause of planner OOMs when a job lands on a different SKU than assumed.
_CAP_OVERRIDE_THRESHOLD: float = 0.05


# ---- Perf ---- #
@dataclass(repr=True, eq=True)
class Perf:
    """
    Representation of the breakdown of the perf estimate a single shard of an
    embedding table.
    """

    fwd_compute: float
    fwd_comms: float
    bwd_compute: float
    bwd_comms: float
    input_dist_comms: float = 0.0
    prefetch_compute: float = 0.0

    @property
    def total(self) -> float:
        # When using embedding offload, there is a prefetch compute component. This
        # prefetch can overlap with fwd_compute + fwd_comm and dense fwd (some of it
        # overlaps with fwd_compute) and dense bwd. (fwd_compute and bwd_compute are
        # embedding fwd/bwd, nothing to do with dense). Only when prefetch is longer
        # than fwd_compute + dense_fwd + dense_bwd it will block bwd_compute. However,
        # we don't have an effective way to estimate dense fwd/bwd at this point, so our
        # cost model is too simplistic.  Instead prefetch is always considered blocking.
        #
        # Also note, measuring prefetch blocking can only be done after partitioning,
        # here are only have the per shard estimates.
        #
        # However adding a per-shard prefetch component to the cost model does have the
        # benefit that 1) it enables the ScaleupProposer to explore the trade off
        # between increasing cache sizes vs more difficult bin-packing constraints. 2)
        # it helps balance the prefetch compute across the ranks.
        return (
            self.fwd_compute
            + self.bwd_compute
            + self.fwd_comms
            + self.bwd_comms
            + self.prefetch_compute
        )

    def __add__(self, other: "Perf") -> "Perf":
        return Perf(
            fwd_compute=self.fwd_compute + other.fwd_compute,
            fwd_comms=self.fwd_comms + other.fwd_comms,
            bwd_compute=self.bwd_compute + other.bwd_compute,
            bwd_comms=self.bwd_comms + other.bwd_comms,
            input_dist_comms=self.input_dist_comms + other.input_dist_comms,
            prefetch_compute=self.prefetch_compute + other.prefetch_compute,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.fwd_compute,
                self.fwd_comms,
                self.bwd_compute,
                self.bwd_comms,
                self.input_dist_comms,
                self.prefetch_compute,
            )
        )


# ---- TOPOLOGY ---- #


@dataclass(repr=True, order=True, eq=True)
class Storage:
    """
    Representation of the storage capacities of a hardware used in training.
    """

    hbm: int
    ddr: int
    ssd: int = 0

    def __add__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm + other.hbm,
            ddr=self.ddr + other.ddr,
            ssd=self.ssd + other.ssd,
        )

    def __sub__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm - other.hbm,
            ddr=self.ddr - other.ddr,
            ssd=self.ssd - other.ssd,
        )

    def __hash__(self) -> int:
        return hash((self.hbm, self.ddr, self.ssd))

    def fits_in(self, other: "Storage") -> bool:
        return self.hbm <= other.hbm and self.ddr <= other.ddr and self.ssd <= other.ssd


@dataclass
class DeviceHardware:
    """
    Representation of a device in a process group. 'perf' is an estimation of network,
    CPU, and storage usages.
    """

    rank: int
    storage: Storage
    perf: Perf

    def __hash__(self) -> int:
        return hash((self.rank, self.storage, self.perf))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceHardware):
            return False
        return (
            self.rank == other.rank
            and self.storage == other.storage
            and self.perf == other.perf
        )


class CustomTopologyData:
    """
    Custom device data for individual device in a topology.
    """

    supported_fields = ["ddr_cap", "hbm_cap", "ssd_cap"]

    def __init__(
        self,
        data: Dict[str, List[int]],
        world_size: int,
    ) -> None:
        assert all(
            key in self.supported_fields for key in data.keys()
        ), f"{data.keys()} not supported in CustomTopologyData"
        assert all(
            len(v) == world_size for v in data.values()
        ), f"{data.values()} must be positive"
        self._data = data
        self._world_size = world_size

    def get_data(self, key: str) -> List[int]:
        assert (
            key in self.supported_fields
        ), f"{key} not supported in CustomTopologyData"
        return self._data[key]

    def has_data(self, key: str) -> bool:
        return key in self._data


class CollectiveType(Enum):
    ALL_TO_ALL = "all_to_all"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    ALL_REDUCE = "all_reduce"


class GeneralizedCommsBandwidth(abc.ABC):
    @abc.abstractmethod
    def get_bw(
        self,
        local_world_size: int,
        world_size: int,
        collective_type: CollectiveType,
    ) -> float:
        """
        Get Bandwidth Corresponding to a collective communication where involving world_size ranks
            spread equally across world_size / local_world_size nodes
        """
        pass

    @property
    @abc.abstractmethod
    def intra_host_bw(self) -> float:
        """this must be implemented for backward compatibility"""
        pass

    @property
    @abc.abstractmethod
    def inter_host_bw(self) -> float:
        """this must be implemented for backward compatibility"""
        pass


class BasicCommsBandwidths(GeneralizedCommsBandwidth):
    def __init__(
        self,
        inter_host_bw: float = CROSS_NODE_BANDWIDTH,
        intra_host_bw: float = INTRA_NODE_BANDWIDTH,
    ) -> None:
        self.name = "BasicCommsBandwidths"
        self._inter_host_bw = inter_host_bw
        self._intra_host_bw = intra_host_bw

    def __str__(self) -> str:
        return (
            self.name
            + f": inter_host_bw={self.inter_host_bw}, intra_host_bw={self.intra_host_bw}"
        )

    @property
    def inter_host_bw(self) -> float:
        return self._inter_host_bw

    @property
    def intra_host_bw(self) -> float:
        return self._intra_host_bw

    def get_bw(
        self,
        local_world_size: int,
        world_size: int,
        collective_type: CollectiveType,
    ) -> float:
        if collective_type == CollectiveType.ALL_REDUCE:
            return self.inter_host_bw * local_world_size  # 1 NIC per GPU
        if world_size <= local_world_size:
            return self.intra_host_bw
        else:
            return self.inter_host_bw

    def __hash__(self) -> int:
        return hash((self._inter_host_bw, self._intra_host_bw))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasicCommsBandwidths):
            return False
        return (
            self._inter_host_bw == other._inter_host_bw
            and self._intra_host_bw == other._intra_host_bw
        )


# ============================================================================
# Topology Configuration Classes
# ============================================================================
#
# These configuration classes provide a structured way to pass parameters
# to the Topology class. They support a precedence-based resolution:
#   1. TrainerConfig (explicit user overrides) - HIGHEST PRIORITY
#   2. HardwareConfig (detected/mapped values)
#   3. KernelConfig (compute kernel specific)
#   4. Default constants - LOWEST PRIORITY
#
# Usage:
#   hardware_config = HardwareConfig(...)
#   trainer_config = TrainerConfig(...)
#   kernel_config = KernelConfig(...)
#   topology = TopologyConfig.create_topology(
#       world_size=8,
#       hardware_config=hardware_config,
#       trainer_config=trainer_config,
#       kernel_config=kernel_config,
#   )
# ============================================================================


class TopologyConfigBase(abc.ABC):
    """
    Abstract base class for all topology configuration classes.

    Provides a common interface for configuration validation, serialization,
    and extensible key-value storage via additional_params.

    All topology-related configs should inherit from this class and include
    an `additional_params: Dict[str, Any]` field for extensibility.

    The `additional_params` field provides a generalized mechanism for passing
    custom data without modifying the schema. This can be used for:
    - Custom per-device topology data (e.g., CustomTopologyData for heterogeneous setups)
    - Framework-specific parameters
    - Hardware-specific metadata (e.g., training_hardware type, LLST info)
    - Experimental or deprecated parameters during migration

    Attributes:
        additional_params: Key-value store for framework-specific or extensible data.
            Allows passing custom configuration without modifying the schema.

    Example Usage:
        # Pass CustomTopologyData via additional_params
        trainer_config = TrainerConfig(
            hbm_cap_bytes=80 * 1024**3,
            additional_params={
                "custom_topology_data": CustomTopologyData(
                    data={"hbm_cap": [40*1024**3, 80*1024**3]},
                    world_size=2,
                ),
            }
        )

        # Access custom data using helper methods
        custom_data = trainer_config.get_param("custom_topology_data")
        if trainer_config.has_param("custom_topology_data"):
            # Use custom per-device capacities
            pass
    """

    # Subclasses must define: additional_params: Dict[str, Any] = field(default_factory=dict)
    # Note: We declare this as a class attribute rather than an abstract property
    # because frozen dataclasses define fields as class attributes, not properties.
    # Using @property @abstractmethod would make the dataclass remain abstract.
    additional_params: Dict[str, Any]

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a value from additional_params with optional default.

        Args:
            key: The parameter key to look up.
            default: Value to return if key is not found.

        Returns:
            The value associated with the key, or default if not found.
        """
        return self.additional_params.get(key, default)

    def has_param(self, key: str) -> bool:
        """
        Check if a key exists in additional_params.

        Args:
            key: The parameter key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self.additional_params

    @abc.abstractmethod
    def validate(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        pass


@dataclass(frozen=True)
class HardwareConfig(TopologyConfigBase):
    """
    Hardware-related configuration for Topology creation.

    Contains parameters that are typically detected from the hardware environment
    or mapped from hardware type specifications. These values represent the
    physical capabilities of the training infrastructure.

    This is a base class that can be extended for specific hardware types
    (e.g., GrandTeton, ZionEX, MTIA) in framework-specific code to provide
    hardware-specific defaults and capabilities.

    Attributes:
        hbm_cap_bytes: HBM (High Bandwidth Memory) capacity per device in bytes.
            Typically detected via torch.cuda.get_device_properties() or
            torch.mtia.get_device_properties().
        ddr_cap_bytes: DDR (host memory) capacity per rank in bytes.
            Typically detected via psutil.virtual_memory() divided by local_world_size.
        ssd_cap_bytes: SSD storage capacity per rank in bytes.
        intra_host_bw: Intra-node communication bandwidth in bytes/ms.
            High bandwidth interconnect (e.g., NVLink, NVSwitch).
        inter_host_bw: Inter-node communication bandwidth in bytes/ms.
            Network bandwidth between nodes (e.g., InfiniBand, RoCE).
        hbm_mem_bw: HBM memory bandwidth in bytes/ms.
        ddr_mem_bw: DDR memory bandwidth in bytes/ms.
        hbm_to_ddr_mem_bw: HBM to DDR transfer bandwidth in bytes/ms (for UVM).
        ssd_mem_bw: SSD memory bandwidth in bytes/ms.
        additional_params: Inherited from TopologyConfigBase. Key-value store for
            framework-specific or extensible data.

    Example Extension (in FB code):
        @dataclass(frozen=True)
        class GrandTetonHardwareConfig(HardwareConfig):
            '''Hardware config with GrandTeton-specific defaults.'''
            hbm_cap_bytes: int = 80 * 1024**3  # 80GB HBM
            pod_size: int = 8  # Hardware-specific pod size
            intra_host_bw: float = 900 * 1024**3 / 1000  # NVSwitch bandwidth
    """

    # Memory Capacities (detected from hardware APIs)
    hbm_cap_bytes: Optional[int] = None
    ddr_cap_bytes: Optional[int] = None
    ssd_cap_bytes: Optional[int] = None

    # Communication Bandwidths (from hardware type mapping)
    intra_host_bw: Optional[float] = None
    inter_host_bw: Optional[float] = None

    # Memory Bandwidths (from hardware type mapping)
    hbm_mem_bw: Optional[float] = None
    ddr_mem_bw: Optional[float] = None
    hbm_to_ddr_mem_bw: Optional[float] = None
    ssd_mem_bw: Optional[float] = None

    # Extensible Key-Value Store (implements TopologyConfigBase.additional_params)
    # pyrefly: ignore[bad-override]
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def get_validation_issues(self, compute_device: Optional[str] = None) -> List[str]:
        """Return human-readable validation issues (pure: no logging, no raise).

        Flags values that are definitively invalid -- detection failures
        (non-positive capacities/bandwidths) and physical-invariant violations. A
        detected HardwareConfig value may be legitimately overridden by TrainerConfig
        precedence, so callers treat these as warnings, not errors; the
        resolved/effective value is the one worth failing on, and is validated
        separately post-precedence.

        Exposed as a pure method (vs only logging) so a caller that evaluates
        many configs in one process (building one topology per candidate
        config) can report per-config without depending on logger
        rate-limiting.

        Args:
            compute_device: optional device hint. The HBM check runs only for
                HBM-bearing accelerators ("cuda"/"mtia"), where a non-positive
                HBM (incl. 0) is a detection failure. It is skipped for "cpu"
                (no HBM device), "meta", None, or any other device, where a
                0/None HBM is not a failure. (Using an allowlist rather than
                "!= cpu" avoids false positives on meta/unknown devices.)
        """
        issues: List[str] = []

        # HBM only exists on accelerators that have it (cuda/mtia), so a
        # non-positive value there is a detection failure. Skipped for cpu
        # (no HBM device), meta, None, or any other device.
        if (
            self.hbm_cap_bytes is not None
            and compute_device in ("cuda", "mtia")
            and self.hbm_cap_bytes <= 0
        ):
            issues.append(f"hbm_cap_bytes={self.hbm_cap_bytes} is non-positive")

        # DDR: every host has DDR, so a non-positive value is a detection or
        # per-rank-division failure.
        if self.ddr_cap_bytes is not None and self.ddr_cap_bytes <= 0:
            issues.append(f"ddr_cap_bytes={self.ddr_cap_bytes} is non-positive")

        # SSD: 0 is legitimate (a host may have no SSD tier); only a negative
        # value is invalid.
        if self.ssd_cap_bytes is not None and self.ssd_cap_bytes < 0:
            issues.append(f"ssd_cap_bytes={self.ssd_cap_bytes} is negative")

        # Bandwidths are divisors in perf estimation; a set non-positive value
        # is invalid (and is not replaced by a default downstream).
        for name, value in (
            ("intra_host_bw", self.intra_host_bw),
            ("inter_host_bw", self.inter_host_bw),
            ("hbm_mem_bw", self.hbm_mem_bw),
            ("ddr_mem_bw", self.ddr_mem_bw),
            ("hbm_to_ddr_mem_bw", self.hbm_to_ddr_mem_bw),
            ("ssd_mem_bw", self.ssd_mem_bw),
        ):
            if value is not None and value <= 0:
                issues.append(f"{name}={value} is non-positive")

        # Physical invariant: intra-node bandwidth should be >= inter-node.
        if (
            self.intra_host_bw is not None
            and self.inter_host_bw is not None
            and self.intra_host_bw < self.inter_host_bw
        ):
            issues.append(
                f"intra_host_bw ({self.intra_host_bw:.0f}) < "
                f"inter_host_bw ({self.inter_host_bw:.0f}); intra-node "
                f"bandwidth is typically much higher than inter-node"
            )

        return issues

    def validate(self, compute_device: Optional[str] = None) -> None:
        """Validate hardware configuration parameters (warning-only).

        Thin shell over get_validation_issues(): emits a single combined
        warning and never raises, to avoid breaking existing flows.

        Logs via static_logger (rank 0, uncapped) rather than a per-location
        rate-limited logger so that a caller building many topologies in one
        process (one validate() call per candidate config) surfaces a warning
        for every config instead of only the first.

        Args:
            compute_device: optional device hint forwarded to
                get_validation_issues(); see that method.
        """
        issues = self.get_validation_issues(compute_device)
        if issues:
            static_logger.warning("HardwareConfig validation: " + "; ".join(issues))


@dataclass(frozen=True)
class TrainerConfig(TopologyConfigBase):
    """
    Trainer-specified configuration overrides for Topology creation.

    Contains parameters that users explicitly configure through their training
    framework (e.g., planner_config, dry_run_config). These values have the
    highest precedence and override hardware-detected values.

    Attributes:
        world_size: Total number of devices (ranks) in distributed training.
            Required parameter for Topology creation.
        local_world_size: Number of devices (GPUs) per node.
            Typically from LOCAL_WORLD_SIZE environment variable or explicit config.
        hbm_cap_bytes: User-specified HBM capacity override in bytes.
        ddr_cap_bytes: User-specified DDR capacity override in bytes.
        ssd_cap_bytes: User-specified SSD capacity override in bytes.
        is_dry_run: Whether this is a dry-run/planning mode execution.
            When True, dry_run_* values take precedence over detected values.
        dry_run_hbm_bytes: HBM capacity to use during dry-run in bytes.
        dry_run_ddr_bytes: DDR capacity to use during dry-run in bytes.
        pod_size: User-specified pod size override. Number of nodes per
            NVLink domain, used to calculate intra_group_size.
        additional_params: Inherited from TopologyConfigBase. Key-value store for
            trainer-specific or extensible data. Can be used to pass
            CustomTopologyData for heterogeneous topologies via:
            `additional_params={"custom_topology_data": CustomTopologyData(...)}`
    """

    # Distributed Training Topology (required for Topology creation)
    world_size: Optional[int] = None
    local_world_size: Optional[int] = None

    # User-specified Memory Overrides (highest priority)
    hbm_cap_bytes: Optional[int] = None
    ddr_cap_bytes: Optional[int] = None
    ssd_cap_bytes: Optional[int] = None

    # Dry-run Mode Configuration
    is_dry_run: bool = False
    dry_run_hbm_bytes: Optional[int] = None
    dry_run_ddr_bytes: Optional[int] = None

    # Topology Overrides
    pod_size: Optional[int] = None

    # Extensible Key-Value Store (implements TopologyConfigBase.additional_params)
    # pyrefly: ignore[bad-override]
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate trainer configuration parameters."""
        # world_size is required for Topology creation
        if self.world_size is None:
            raise ValueError("world_size must be provided in TrainerConfig")

        # Match Topology class validation: pod_size cannot exceed world_size
        if self.pod_size is not None and self.pod_size > self.world_size:
            raise ValueError(
                f"pod_size ({self.pod_size}) cannot be greater than "
                f"world_size ({self.world_size})"
            )


@dataclass(frozen=True)
class KernelConfig(TopologyConfigBase):
    """
    Compute kernel-specific configuration for Topology creation.

    Contains parameters related to the compute device and kernel performance
    characteristics. These affect how the planner estimates performance for
    different sharding strategies.

    This is a base class that can be extended for specific kernel/device types
    (e.g., CUDAKernelConfig, MTIAKernelConfig) in framework-specific code to
    provide device-specific performance multipliers and communication patterns.

    Attributes:
        compute_device: The compute device type ("cuda", "mtia", or "cpu").
        bwd_compute_multiplier: Multiplier for backward compute estimation.
            Accounts for the additional compute in backward pass vs forward.
        weighted_feature_bwd_compute_multiplier: Multiplier for weighted feature
            backward compute estimation.
        uneven_sharding_perf_multiplier: Performance penalty multiplier for
            uneven sharding distributions.
        use_hardware_based_bandwidth: If True, TopologyFactory will compute
            generalized_comms_bandwidths from detected hardware capability.
            If False, uses TorchRec defaults (BasicCommsBandwidths).
        generalized_comms_bandwidths: Custom communication bandwidth model.
            If provided, overrides both use_hardware_based_bandwidth and
            intra_host_bw/inter_host_bw from HardwareConfig.
        additional_params: Inherited from TopologyConfigBase. Key-value store for
            kernel-specific or extensible data.

    Example Extension (in FB code):
        @dataclass(frozen=True)
        class MTIAKernelConfig(KernelConfig):
            '''Kernel config with MTIA-specific defaults.'''
            compute_device: str = "mtia"
            bwd_compute_multiplier: float = 2.5  # MTIA-specific
            custom_mtia_param: float = 1.0  # Device-specific parameter

        @dataclass(frozen=True)
        class CUDAFusedKernelConfig(KernelConfig):
            '''Kernel config optimized for CUDA fused kernels.'''
            compute_device: str = "cuda"
            fused_kernel_efficiency: float = 0.95
    """

    # Compute Device
    compute_device: str = "cuda"

    # Performance Multipliers
    bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER
    weighted_feature_bwd_compute_multiplier: float = (
        WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER
    )
    uneven_sharding_perf_multiplier: float = 1.0

    # Hardware-based Bandwidth Configuration
    # If True, TopologyFactory computes bandwidths from detected hardware
    use_hardware_based_bandwidth: bool = False

    # Custom Communication Bandwidth Model (overrides use_hardware_based_bandwidth)
    generalized_comms_bandwidths: Optional[GeneralizedCommsBandwidth] = None

    # Extensible Key-Value Store (implements TopologyConfigBase.additional_params)
    # pyrefly: ignore[bad-override]
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate kernel configuration parameters."""
        # Match Topology class validation: compute_device must be valid
        valid_devices = {"cuda", "mtia", "cpu"}
        if self.compute_device not in valid_devices:
            raise ValueError(
                f"compute_device must be one of {valid_devices}, got '{self.compute_device}'"
            )


class TopologyFactory:
    """
    Factory for creating Topology instances with precedence-based resolution.

    Resolves parameters in the following order (highest to lowest priority):
        1. TrainerConfig (explicit user overrides)
        2. HardwareConfig (detected/mapped values)
        3. Default constants

    Usage:
        hardware_config = HardwareConfig(hbm_cap_bytes=80 * 1024**3, ...)
        trainer_config = TrainerConfig(world_size=8, local_world_size=8, ...)
        kernel_config = KernelConfig(compute_device="cuda", ...)

        topology = TopologyFactory.create_topology(
            trainer_config=trainer_config,
            hardware_config=hardware_config,
            kernel_config=kernel_config,
        )
    """

    @staticmethod
    def create_topology(
        trainer_config: TrainerConfig,
        hardware_config: Optional[HardwareConfig] = None,
        kernel_config: Optional[KernelConfig] = None,
    ) -> "Topology":
        """
        Create a Topology instance using precedence-based parameter resolution.

        Args:
            trainer_config: User-specified overrides (required, must have world_size).
            hardware_config: Hardware-detected values.
            kernel_config: Compute kernel specific parameters.

        Returns:
            A configured Topology instance.

        Raises:
            ValueError: If validation fails on any config.
        """
        try:
            hardware = hardware_config or HardwareConfig()
            kernel = kernel_config or KernelConfig()

            # Validate configs
            trainer_config.validate()
            hardware.validate(compute_device=kernel.compute_device)
            kernel.validate()

            # Build topology kwargs with precedence resolution
            topology_kwargs: Dict[str, Any] = {
                "world_size": trainer_config.world_size,
                "compute_device": kernel.compute_device,
                "bwd_compute_multiplier": kernel.bwd_compute_multiplier,
                "weighted_feature_bwd_compute_multiplier": kernel.weighted_feature_bwd_compute_multiplier,
                "uneven_sharding_perf_multiplier": kernel.uneven_sharding_perf_multiplier,
            }

            # Add optional parameters from configs
            TopologyFactory._add_trainer_params(
                topology_kwargs, trainer_config, hardware
            )
            TopologyFactory._add_hardware_params(topology_kwargs, hardware, kernel)
            TopologyFactory._add_comms_params(topology_kwargs, hardware, kernel)

            one_time_rank0_logger.info("TopologyFactory.create_topology called.")
            topology_kwargs["created_by_factory"] = True
            return Topology(**topology_kwargs)
        except Exception as e:
            one_time_logger.error(f"TopologyFactory.create_topology failed: {e}")
            raise

    @staticmethod
    def _add_trainer_params(
        kwargs: Dict[str, Any],
        trainer: TrainerConfig,
        hardware: HardwareConfig,
    ) -> None:
        """Add trainer config parameters with precedence over hardware."""
        if trainer.local_world_size is not None:
            kwargs["local_world_size"] = trainer.local_world_size
        if trainer.pod_size is not None:
            kwargs["pod_size"] = trainer.pod_size

        # Memory capacities: trainer > hardware > defaults
        hbm_cap = (
            trainer.hbm_cap_bytes
            if trainer.hbm_cap_bytes is not None
            else hardware.hbm_cap_bytes
        )
        ddr_cap = (
            trainer.ddr_cap_bytes
            if trainer.ddr_cap_bytes is not None
            else hardware.ddr_cap_bytes
        )
        ssd_cap = (
            trainer.ssd_cap_bytes
            if trainer.ssd_cap_bytes is not None
            else hardware.ssd_cap_bytes
        )

        # Warn when a TrainerConfig capacity overrides the detected HardwareConfig
        # value by more than the noise threshold. Skipped in dry-run, where the
        # effective caps come from the dry_run_* overrides below, not the
        # trainer/hardware pair.
        if not trainer.is_dry_run:
            TopologyFactory._warn_if_cap_override(
                "hbm_cap",
                trainer.hbm_cap_bytes,
                hardware.hbm_cap_bytes,
                "This may indicate a static model config overriding detected "
                "hardware.",
            )
            # Assumes both ddr values share the same per-rank/per-host basis;
            # a mismatched basis (e.g. per-host trainer vs per-rank hardware)
            # can inflate the reported delta.
            # TODO: revisit once the per-rank/per-host DDR basis is unified so
            # this can't false-positive.
            TopologyFactory._warn_if_cap_override(
                "ddr_cap",
                trainer.ddr_cap_bytes,
                hardware.ddr_cap_bytes,
            )

        # Handle dry-run mode overrides
        if trainer.is_dry_run:
            if trainer.dry_run_hbm_bytes is not None:
                hbm_cap = trainer.dry_run_hbm_bytes
            if trainer.dry_run_ddr_bytes is not None:
                ddr_cap = trainer.dry_run_ddr_bytes

        if hbm_cap is not None:
            kwargs["hbm_cap"] = hbm_cap
        if ddr_cap is not None:
            kwargs["ddr_cap"] = ddr_cap
        if ssd_cap is not None:
            kwargs["ssd_cap"] = ssd_cap

        # Custom topology data from additional_params
        custom_topology_data = trainer.get_param("custom_topology_data")
        if custom_topology_data is not None:
            kwargs["custom_topology_data"] = custom_topology_data

    @staticmethod
    def _warn_if_cap_override(
        name: str,
        trainer_value: Optional[int],
        hardware_value: Optional[int],
        hint: str = "",
    ) -> None:
        """Warn when a trainer-supplied capacity deviates from the detected
        hardware value by more than _CAP_OVERRIDE_THRESHOLD.

        The trainer value takes precedence in topology construction; a large gap
        usually means a stale static model config overriding detected hardware.
        No-op when either value is missing or the hardware value is non-positive.

        Logs via static_logger (rank 0, uncapped) for the same reason as
        HardwareConfig.validate(): a caller building many topologies in one
        process would otherwise have a per-location rate-limited logger
        suppress every warning after the first.
        """
        if trainer_value is None or hardware_value is None or hardware_value <= 0:
            return
        ratio = abs(trainer_value - hardware_value) / hardware_value
        if ratio > _CAP_OVERRIDE_THRESHOLD:
            message = (
                f"TopologyFactory: TrainerConfig {name} "
                f"({trainer_value / 1024**3:.1f} GiB) differs from "
                f"HardwareConfig {name} "
                f"({hardware_value / 1024**3:.1f} GiB) by "
                f"{ratio:.0%}. Using TrainerConfig value."
            )
            if hint:
                message = f"{message} {hint}"
            static_logger.warning(message)

    @staticmethod
    def _add_hardware_params(
        kwargs: Dict[str, Any], hardware: HardwareConfig, kernel: KernelConfig
    ) -> None:
        """Add hardware config parameters (memory bandwidths).

        When use_hardware_based_bandwidth=True: use hardware-detected values
        (falls back to TorchRec defaults if hardware values are None)

        When use_hardware_based_bandwidth=False: use TorchRec defaults
        (matches old legacy path which doesn't set these, so Topology uses defaults)
        """
        if kernel.use_hardware_based_bandwidth:
            # Hardware-based path: use hardware values, fall back to TorchRec defaults
            kwargs["hbm_mem_bw"] = (
                hardware.hbm_mem_bw if hardware.hbm_mem_bw is not None else HBM_MEM_BW
            )
            kwargs["ddr_mem_bw"] = (
                hardware.ddr_mem_bw if hardware.ddr_mem_bw is not None else DDR_MEM_BW
            )
            kwargs["ssd_mem_bw"] = (
                hardware.ssd_mem_bw if hardware.ssd_mem_bw is not None else SSD_MEM_BW
            )
            kwargs["hbm_to_ddr_mem_bw"] = (
                hardware.hbm_to_ddr_mem_bw
                if hardware.hbm_to_ddr_mem_bw is not None
                else HBM_TO_DDR_MEM_BW
            )
        else:
            # Default path: match Topology defaults (old legacy path doesn't set these)
            kwargs["hbm_mem_bw"] = HBM_MEM_BW
            kwargs["ddr_mem_bw"] = DDR_MEM_BW
            kwargs["ssd_mem_bw"] = SSD_MEM_BW
            kwargs["hbm_to_ddr_mem_bw"] = HBM_TO_DDR_MEM_BW

    @staticmethod
    def _add_comms_params(
        kwargs: Dict[str, Any],
        hardware: HardwareConfig,
        kernel: KernelConfig,
    ) -> None:
        """Add communication bandwidth parameters.

        When generalized_comms_bandwidths is provided (from get_bw_info_for_curr_capability()
        when use_hardware_based_bandwidth=True): use it directly.

        When use_hardware_based_bandwidth=True but no generalized_comms_bandwidths:
        use hardware-detected intra/inter bandwidth values.

        When use_hardware_based_bandwidth=False: use TorchRec defaults
        (matches old legacy path which creates BasicCommsBandwidths() with defaults)
        """
        if kernel.generalized_comms_bandwidths is not None:
            # Hardware-based path with generalized bandwidths from
            # get_bw_info_for_curr_capability()
            kwargs["generalized_comms_bandwidths"] = kernel.generalized_comms_bandwidths
        elif kernel.use_hardware_based_bandwidth:
            # Hardware-based path: use hardware values, fall back to TorchRec defaults
            kwargs["intra_host_bw"] = (
                hardware.intra_host_bw
                if hardware.intra_host_bw is not None
                else INTRA_NODE_BANDWIDTH
            )
            kwargs["inter_host_bw"] = (
                hardware.inter_host_bw
                if hardware.inter_host_bw is not None
                else CROSS_NODE_BANDWIDTH
            )
        else:
            # Default path: match Topology defaults (old legacy path)
            kwargs["intra_host_bw"] = INTRA_NODE_BANDWIDTH
            kwargs["inter_host_bw"] = CROSS_NODE_BANDWIDTH


class Topology:
    """
    Representation of a network of devices in a cluster.
    """

    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: Optional[int] = None,
        ssd_cap: Optional[int] = None,
        local_world_size: Optional[int] = None,
        pod_size: Optional[int] = None,
        hbm_mem_bw: float = HBM_MEM_BW,
        ddr_mem_bw: float = DDR_MEM_BW,
        ssd_mem_bw: float = SSD_MEM_BW,
        hbm_to_ddr_mem_bw: float = HBM_TO_DDR_MEM_BW,
        intra_host_bw: float = INTRA_NODE_BANDWIDTH,
        inter_host_bw: float = CROSS_NODE_BANDWIDTH,
        bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER,
        custom_topology_data: Optional[CustomTopologyData] = None,
        weighted_feature_bwd_compute_multiplier: float = WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,
        uneven_sharding_perf_multiplier: float = 1.0,
        generalized_comms_bandwidths: Optional[GeneralizedCommsBandwidth] = None,
        created_by_factory: bool = False,
    ) -> None:
        """
        Representation of a network of devices in a cluster.

        If a GeneralizedCommsBandwidth is passed to generalized_comms_bandwidths, this object will
            take precedence over the formulation using only intra_host_bw and inter_host_bw.
            If it's not passed, we will create a BasicCommsBandwidths object with the provided bandwidths.
        """
        # validate input
        assert compute_device in [
            "cpu",
            "cuda",
            "mtia",
        ], f"unsupported compute device {compute_device}"
        if pod_size and pod_size > world_size:
            raise ValueError(
                f"pod_size={pod_size} cannot be greater than world_size={world_size}"
            )

        self._compute_device = compute_device
        self._world_size = world_size

        hbm_per_device = [0] * world_size
        if self._compute_device == "cuda" or self._compute_device == "mtia":
            hbm_per_device = [hbm_cap if hbm_cap is not None else HBM_CAP] * world_size
        ddr_cap_per_rank = [ddr_cap if ddr_cap is not None else DDR_CAP] * world_size
        ssd_cap_per_rank = [ssd_cap if ssd_cap is not None else SSD_CAP] * world_size

        if custom_topology_data:
            if custom_topology_data.has_data("hbm_cap"):
                hbm_per_device = custom_topology_data.get_data("hbm_cap")
                assert (
                    len(hbm_per_device) == world_size
                ), "Must provide individual hbm_cap for each device"
            if custom_topology_data.has_data("ddr_cap"):
                ddr_cap_per_rank = custom_topology_data.get_data("ddr_cap")
                assert (
                    len(ddr_cap_per_rank) == world_size
                ), "Must provide individual ddr_cap for each device"
            if custom_topology_data.has_data("ssd_cap"):
                ssd_cap_per_rank = custom_topology_data.get_data("ssd_cap")
                assert (
                    len(ssd_cap_per_rank) == world_size
                ), "Must provide individual ssd_cap for each device"

        self._devices: List[DeviceHardware] = []
        for rank in range(world_size):
            self._devices.append(
                DeviceHardware(
                    rank=rank,
                    storage=Storage(
                        hbm=hbm_per_device[rank],
                        ddr=ddr_cap_per_rank[rank],
                        ssd=ssd_cap_per_rank[rank],
                    ),
                    perf=Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0),
                )
            )

        # Local world size is the number of devices (GPUs) in a single node
        self._local_world_size: int = (
            local_world_size if local_world_size else world_size
        )
        self._pod_size: Optional[int] = pod_size
        # Maximum numb of devices with high bandwidth interconnect (e.g. NVLink)
        #  if pod_size isn't given, then assumes local_world_size is maximum group size
        self._intra_group_size: int = (
            pod_size * self._local_world_size
            if pod_size is not None
            else self._local_world_size
        )

        self._hbm_mem_bw = hbm_mem_bw
        self._ddr_mem_bw = ddr_mem_bw
        self._ssd_mem_bw = ssd_mem_bw
        self._hbm_to_ddr_mem_bw = hbm_to_ddr_mem_bw

        self._comms_bandwidths: GeneralizedCommsBandwidth = (
            generalized_comms_bandwidths
            if generalized_comms_bandwidths is not None
            else BasicCommsBandwidths(
                intra_host_bw=intra_host_bw, inter_host_bw=inter_host_bw
            )
        )

        self._bwd_compute_multiplier = bwd_compute_multiplier
        self._custom_topology_data = custom_topology_data
        self._weighted_feature_bwd_compute_multiplier = (
            weighted_feature_bwd_compute_multiplier
        )
        self._uneven_sharding_perf_multiplier = uneven_sharding_perf_multiplier
        self._created_by_factory: bool = created_by_factory

        if not self._created_by_factory:
            logging.getLogger(__name__).warning(
                "The topology was constructed directly rather than via TopologyFactory.create_topology(). "
                "Please use TopologyFactory to ensure proper hardware configuration resolution and validation; "
                "otherwise, the job will fail."
            )

    @property
    def compute_device(self) -> str:
        return self._compute_device

    @property
    def devices(self) -> List[DeviceHardware]:
        return self._devices

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    def intra_group_size(self) -> int:
        # The largest set of nodes connected with high intra-node bandwidth (e.g. NVLink)
        return self._intra_group_size

    @property
    def hbm_mem_bw(self) -> float:
        return self._hbm_mem_bw

    @property
    def ddr_mem_bw(self) -> float:
        return self._ddr_mem_bw

    @property
    def ssd_mem_bw(self) -> float:
        return self._ssd_mem_bw

    @property
    def hbm_to_ddr_mem_bw(self) -> float:
        return self._hbm_to_ddr_mem_bw

    @property
    def intra_host_bw(self) -> float:
        return self._comms_bandwidths.intra_host_bw

    @property
    def inter_host_bw(self) -> float:
        return self._comms_bandwidths.inter_host_bw

    @property
    def comms_bandwidths(self) -> GeneralizedCommsBandwidth:
        return self._comms_bandwidths

    @property
    def bwd_compute_multiplier(self) -> float:
        return self._bwd_compute_multiplier

    @property
    def weighted_feature_bwd_compute_multiplier(self) -> float:
        return self._weighted_feature_bwd_compute_multiplier

    @property
    def uneven_sharding_perf_multiplier(self) -> float:
        return self._uneven_sharding_perf_multiplier

    def __repr__(self) -> str:
        topology_repr: str = f"world_size={self._world_size} \n"
        topology_repr += f"compute_device={self._compute_device}\n"
        topology_repr += "devices=\n"
        for idx, device in enumerate(self._devices):
            topology_repr += f"\tdevice {idx} {device}\n"
        topology_repr += f"local_world_size={self._local_world_size} \n"
        topology_repr += f"intra_group_size={self._intra_group_size} \n"
        topology_repr += str(self._comms_bandwidths) + "\n"
        return topology_repr

    def _hash(self) -> int:
        """
        Compute a consistent hash value for this Topology instance.

        Returns:
            str: A hash value for this Topology instance.

        NOTE: Not overriding the __hash__ method here to account for other
        potential variables that may be unchecked by the following list
        """

        # Compute hbms and ddrs from the decives
        hbms = [device.storage.hbm for device in self._devices]
        ddrs = [device.storage.ddr for device in self._devices]
        ssds = [device.storage.ssd for device in self._devices]

        # Combine all attributes into a hashable tuple
        hashable_list = [
            self._world_size,
            self._compute_device,
            hbms,
            ddrs,
            ssds,
            self._local_world_size,
            self._intra_group_size,
            self._hbm_mem_bw,
            self._ddr_mem_bw,
            self._ssd_mem_bw,
            self._hbm_to_ddr_mem_bw,
            self._comms_bandwidths.intra_host_bw,
            self._comms_bandwidths.inter_host_bw,
            self._bwd_compute_multiplier,
            self._weighted_feature_bwd_compute_multiplier,
            self._uneven_sharding_perf_multiplier,
        ]

        return hash_sha256_to_int(hashable_list)


# ---- INPUT / OUTPUT ----- #


@dataclass
class Shard:
    """
    Representation of a subset of an embedding table. 'size' and 'offset' fully
    determine the tensors in the shard. 'storage' is an estimation of how much it takes
    to store the shard with an estimation 'perf'.
    """

    size: List[int]
    offset: List[int]
    storage: Optional[Storage] = None
    perf: Optional[Perf] = None
    rank: Optional[int] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.size),
                tuple(self.offset),
                self.storage,
                self.perf,
                self.rank,
            )
        )

    def __str__(self) -> str:
        return f"Shard size: {tuple(self.size)}, offset: {tuple(self.offset)}, storage: {str(self.storage)}, perf: {str(self.perf)}, rank: {self.rank}"


class ShardingOption:
    """
    One way of sharding an embedding table. In the enumerator, we generate
    multiple sharding options per table, but in the planner output, there
    should only be one sharding option per table.

    Attributes:
        name (str): name of the sharding option.
        tensor (torch.Tensor): tensor of the sharding option. Usually on meta
            device.
        module (Tuple[str, nn.Module]): module and its fqn that contains the
            table.
        input_lengths (List[float]): list of pooling factors of the feature for
            the table.
        batch_size (int): batch size of training / eval job.
        sharding_type (str): sharding type of the table. Value of enum ShardingType.
        compute_kernel (str): compute kernel of the table. Value of enum
            EmbeddingComputeKernel.
        shards (List[Shard]): list of shards of the table.
        cache_params (Optional[CacheParams]): cache parameters to be used by this table.
            These are passed to FBGEMM's Split TBE kernel.
        enforce_hbm (Optional[bool]): whether to place all weights/momentums in HBM when
            using cache.
        stochastic_rounding (Optional[bool]): whether to do stochastic rounding. This is
            passed to FBGEMM's Split TBE kernel. Stochastic rounding is
            non-deterministic, but important to maintain accuracy in longer
            term with FP16 embedding tables.
        bounds_check_mode (Optional[BoundsCheckMode]): bounds check mode to be used by
            FBGEMM's Split TBE kernel. Bounds check means checking if values
            (i.e. row id) is within the table size. If row id exceeds table
            size, it will be set to 0.
        dependency (Optional[str]): dependency of the table. Related to
            Embedding tower.
        is_pooled (Optional[bool]): whether the table is pooled. Pooling can be
            sum pooling or mean pooling. Unpooled tables are also known as
            sequence embeddings.
        feature_names (Optional[List[str]]): list of feature names for this table.
        output_dtype (Optional[DataType]): output dtype to be used by this table.
            The default is FP32. If not None, the output dtype will also be used
            by the planner to produce a more balanced plan.
        key_value_params (Optional[KeyValueParams]): Params for SSD TBE, either
            for SSD or PS.
    """

    def __init__(
        self,
        name: str,
        tensor: torch.Tensor,
        module: Tuple[str, nn.Module],
        input_lengths: List[float],
        batch_size: int,
        sharding_type: str,
        partition_by: str,
        compute_kernel: str,
        shards: List[Shard],
        cache_params: Optional[CacheParams] = None,
        enforce_hbm: Optional[bool] = None,
        stochastic_rounding: Optional[bool] = None,
        bounds_check_mode: Optional[BoundsCheckMode] = None,
        dependency: Optional[str] = None,
        is_pooled: Optional[bool] = None,
        feature_names: Optional[List[str]] = None,
        output_dtype: Optional[DataType] = None,
        key_value_params: Optional[KeyValueParams] = None,
        num_poolings: Optional[List[float]] = None,
    ) -> None:
        self.name = name
        self._tensor = tensor
        self._module = module
        self.input_lengths = input_lengths
        self.batch_size = batch_size
        self.sharding_type = sharding_type
        self.partition_by = partition_by
        self.compute_kernel = compute_kernel
        # relevant to planner output, must be populated if sharding option
        # part of final solution
        self.shards = shards
        self.cache_params = cache_params
        self.enforce_hbm = enforce_hbm
        self.stochastic_rounding = stochastic_rounding
        self.bounds_check_mode = bounds_check_mode
        self.dependency = dependency
        self._is_pooled: bool = (
            is_pooled
            if is_pooled is not None
            else ShardingOption.module_pooled(module[1], name)
        )
        self.is_weighted: Optional[bool] = None
        self.feature_names: Optional[List[str]] = feature_names
        self.output_dtype: Optional[DataType] = output_dtype
        self.key_value_params: Optional[KeyValueParams] = key_value_params
        self.num_poolings: Optional[List[float]] = num_poolings

        child_module = module[1]
        self._module_type_key: str = (
            type(child_module).__module__ + "." + type(child_module).__name__
        )
        _module_has_fp = (
            hasattr(child_module, "_feature_processor")
            and hasattr(
                child_module._feature_processor,
                "feature_processor_modules",
            )
            and isinstance(
                # pyrefly: ignore[missing-attribute]: `Module` has no attribute `_feature_processor`
                child_module._feature_processor.feature_processor_modules,
                nn.ModuleDict,
            )
        )
        self._has_feature_processor: bool = (
            _module_has_fp
            and name
            # pyrefly: ignore[missing-attribute]: `Module` has no attribute `_feature_processor`
            in child_module._feature_processor.feature_processor_modules.keys()
        )
        if hasattr(child_module, "is_weighted") and callable(child_module.is_weighted):
            if isinstance(child_module, EmbeddingBagCollectionInterface):
                # pyrefly: ignore[not-callable]: `Module` has no attribute `is_weighted`
                self.is_weighted = child_module.is_weighted()

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def module(self) -> Tuple[str, nn.Module]:
        return self._module

    @property
    def fqn(self) -> str:
        return self.module[0] + "." + self.name

    @property
    def cache_load_factor(self) -> Optional[float]:
        if self.cache_params is not None:
            return self.cache_params.load_factor
        return None

    @property
    def path(self) -> str:
        return self.module[0]

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def num_inputs(self) -> int:
        return len(self.input_lengths)

    @property
    def total_storage(self) -> Storage:
        storage: Storage = Storage(hbm=0, ddr=0, ssd=0)
        for shard in self.shards:
            storage += cast(Storage, shard.storage)
        return storage

    @property
    def total_perf(self) -> float:
        perf: float = 0
        for shard in self.shards:
            # pyrefly: ignore[missing-attribute]
            perf += shard.perf.total
        return perf

    @property
    def is_pooled(self) -> bool:
        return self._is_pooled

    @staticmethod
    def module_pooled(module: nn.Module, sharding_option_name: str) -> bool:
        """Determine if module pools output (e.g. EmbeddingBag) or uses unpooled/sequential output."""
        if isinstance(module, EmbeddingCollectionInterface) or isinstance(
            module, ManagedCollisionEmbeddingCollection
        ):
            return False

        for submodule in module.modules():
            if isinstance(submodule, EmbeddingCollectionInterface) or isinstance(
                submodule, ManagedCollisionEmbeddingCollection
            ):
                for name, _ in submodule.named_parameters():
                    if sharding_option_name in name:
                        return False

        return True

    @property
    def module_type_key(self) -> str:
        return self._module_type_key

    @property
    def has_feature_processor(self) -> bool:
        return self._has_feature_processor

    def get_shards_assignment(self) -> List[Optional[int]]:
        return [shard.rank for shard in self.shards]

    def __hash__(self) -> int:
        return hash(
            (
                self.fqn,
                self.sharding_type,
                self.compute_kernel,
                tuple(self.shards),
                self.cache_params,
            )
        )

    def storage_hash(self) -> int:
        """
        Hash needed to preserve sharding option uniquely based on input before
        planning. This is needed to restore sharding option from the loaded plan.
        Hash is computed based on the following attributes:
            - fqn
            - sharding_type
            - compute_kernel
            - column_wise_shard_dim
        """
        # Use BLAKE2b for deterministic hashing, constrained to 64-bit signed int range
        hash_str = f"{self.fqn}|{self.sharding_type}|{self.compute_kernel}|{self.cache_load_factor}|{self.num_shards}"
        hash_bytes = hashlib.blake2b(hash_str.encode("utf-8"), digest_size=7).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder="big")
        return hash_int

    def __deepcopy__(
        self, memo: Optional[Dict[int, "ShardingOption"]]
    ) -> "ShardingOption":
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in ["_tensor", "_module"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __str__(self) -> str:
        str_obj: str = ""
        str_obj += f"name: {self.name}"
        str_obj += f"\nsharding type: {self.sharding_type}"
        str_obj += f"\ncompute kernel: {self.compute_kernel}"
        str_obj += f"\nnum shards: {len(self.shards)}"
        for shard in self.shards:
            str_obj += f"\n\t{str(shard)}"

        return str_obj


class PartitionByType(Enum):
    """
    Well-known partition types.
    """

    # Partitioning based on device
    DEVICE = "device"
    # Partitioning based on host
    HOST = "host"
    # Uniform, (ie. fixed layout)
    UNIFORM = "uniform"
    # Partitioning based on multiple hosts
    MULTI_HOST = "multi_host"


@dataclass
class SharderData:
    """Picklable snapshot of sharder data needed by estimators.

    Captures fused_params, quantized comm codec dtype sizes, and storage
    usage dispatch info so estimators can work without live sharder objects.
    """

    fused_params: Dict[str, Any]
    qcomm_dtype_sizes: Dict[str, Tuple[float, float]]
    storage_usage_type: StorageUsageType


SharderDataMap = Dict[str, SharderData]


@dataclass
class ParameterConstraints:
    """
    Stores user provided constraints around the sharding plan.

    If provided, `pooling_factors`, `num_poolings`, and `batch_sizes` must match in
    length, as per sample.

    Attributes:
        sharding_types (Optional[List[str]]): sharding types allowed for the table.
            Values of enum ShardingType.
        compute_kernels (Optional[List[str]]): compute kernels allowed for the table.
            Values of enum EmbeddingComputeKernel.
        min_partition (Optional[int]): lower bound for dimension of column wise shards.
            Planner will search for the column wise shard dimension in the
            range of [min_partition, embedding_dim], as long as the column wise
            shard dimension divides embedding_dim and is divisible by 4. Used
            for column wise sharding only.
        pooling_factors (Optional[List[float]]): pooling factors for each feature of the
            table. This is the average number of values each sample has for
            the feature. Length of pooling_factors should match the number of
            features.
        num_poolings (OptionalList[float]]): number of poolings for each feature of the
            table. Length of num_poolings should match the number of features.
        batch_sizes (Optional[List[int]]): batch sizes for each feature of the table. Length
            of batch_sizes should match the number of features.
        is_weighted (Optional[bool]): whether the table is weighted.
        cache_params (Optional[CacheParams]): cache parameters to be used by this table.
            These are passed to FBGEMM's Split TBE kernel.
        enforce_hbm (Optional[bool]): whether to place all weights/momentums in HBM when
            using cache.
        stochastic_rounding (Optional[bool]): whether to do stochastic rounding. This is
            passed to FBGEMM's Split TBE kernel. Stochastic rounding is
            non-deterministic, but important to maintain accuracy in longer
            term with FP16 embedding tables.
        bounds_check_mode (Optional[BoundsCheckMode]): bounds check mode to be used by
            FBGEMM's Split TBE kernel. Bounds check means checking if values
            (i.e. row id) is within the table size. If row id exceeds table
            size, it will be set to 0.
        feature_names (Optional[List[str]]): list of feature names for this table.
        output_dtype (Optional[DataType]): output dtype to be used by this table.
            The default is FP32. If not None, the output dtype will also be used
            by the planner to produce a more balanced plan.
        device_group (Optional[str]): device group to be used by this table. It can be cpu
            or cuda. This specifies if the table should be placed on a cpu device
            or a gpu device.
        key_value_params (Optional[KeyValueParams]): key value params for SSD TBE, either for
            SSD or PS.
        use_virtual_table (bool): is virtual table enabled for this table.
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None  # CW sharding, min CW dim to shard
    pooling_factors: List[float] = field(
        default_factory=lambda: [POOLING_FACTOR]
    )  # average number of embedding lookups required per sample
    num_poolings: Optional[List[float]] = None  # number of poolings per sample in batch
    batch_sizes: Optional[List[int]] = None  # batch size per input feature
    is_weighted: bool = False
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None
    feature_names: Optional[List[str]] = None
    output_dtype: Optional[DataType] = None
    device_group: Optional[str] = None
    key_value_params: Optional[KeyValueParams] = None
    use_virtual_table: bool = False

    def __hash__(self) -> int:
        hashable_list = [
            tuple(self.sharding_types) if self.sharding_types else None,
            tuple(self.compute_kernels) if self.compute_kernels else None,
            self.min_partition,
            tuple(self.pooling_factors),
            tuple(self.num_poolings) if self.num_poolings else None,
            tuple(self.batch_sizes) if self.batch_sizes else None,
            self.is_weighted,
            self.cache_params,
            self.enforce_hbm,
            self.stochastic_rounding,
            self.bounds_check_mode,
            tuple(self.feature_names) if self.feature_names else None,
            self.output_dtype,
            self.device_group,
            self.key_value_params,
            self.use_virtual_table,
        ]

        return hash_sha256_to_int(hashable_list)


class PlannerErrorType(Enum):
    """
    Classify PlannerError based on the following cases.
    """

    INSUFFICIENT_STORAGE = "insufficient_storage"
    STRICT_CONSTRAINTS = "strict_constraints"
    PARTITION = "partition"
    OTHER = "other"
    PLANNER_INPUT_CONTEXT_MISMATCH = "planner_input_context_mismatch"
    PLAN_LOADING_FAILED = "plan_loading_failed"
    INVALID_RANK_ASSIGNMENT = "invalid_rank_assignment"
    INPUT_VALIDATION = "input_validation"
    MISSING_MODULE_IN_PLAN = "missing_module_in_plan"
    INVALID_COMPUTE_KERNEL = "invalid_compute_kernel"


class PlannerError(Exception):
    def __init__(
        self,
        message: str,
        error_type: PlannerErrorType = PlannerErrorType.OTHER,
    ) -> None:
        self.error_type = error_type
        super().__init__(message)


# ---- PLANNER COMPONENTS ---- #


class StorageReservation(abc.ABC):
    """
    Reserves storage space for non-sharded parts of the model.
    """

    @abc.abstractmethod
    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology: ...

    @property
    @abc.abstractmethod
    def last_reserved_topology(self) -> Optional[Topology]: ...


class PerfModel(abc.ABC):
    @abc.abstractmethod
    def rate(self, plan: List[ShardingOption]) -> float: ...


class ShardEstimator(abc.ABC):
    """
    Estimates shard perf or storage, requires fully specified sharding options.
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None: ...

    @abc.abstractmethod
    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_data_map: SharderDataMap,
    ) -> None:
        # update sharding_options with per shard estimate in-place
        ...


class Enumerator(abc.ABC):
    """
    Generates all relevant sharding options for given topology, constraints, nn.Module,
    and sharders.
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        batch_size: int = BATCH_SIZE,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
    ) -> None: ...

    @abc.abstractmethod
    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """
        See class description.
        """
        ...

    @abc.abstractmethod
    def populate_estimates(self, sharding_options: List[ShardingOption]) -> None:
        """
        See class description.
        """
        ...


class Proposer(abc.ABC):
    """
    Prosposes complete lists of sharding options which can be parititioned to generate a
    plan.
    """

    @abc.abstractmethod
    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        """
        Load search space into proposer.

        Args:
            search_space (List[ShardingOption]): search space to load.
            enumerator (Enumerator): enumerator used to generate search space.
        """
        ...

    @abc.abstractmethod
    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        """
        Provide feedback to proposer.

        Args:
            partitionable (bool): whether the plan is partitionable.
            plan (Optional[List[ShardingOption]]): plan to provide feedback on.
            perf_rating (Optional[float]): performance rating of the plan.
            storage_constraint (Optional[Topology]): storage constraint of the plan.
        """
        ...

    @abc.abstractmethod
    def propose(self) -> Optional[List[ShardingOption]]:
        """
        Propose a sharding plan.

        Returns:
            Optional[List[ShardingOption]]: proposed plan.
        """
        ...


class Partitioner(abc.ABC):
    """
    Partitions shards.

    Today we have multiple strategies ie. (Greedy, BLDM, Linear).
    """

    @abc.abstractmethod
    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        # modifies sharding_options and topology in-place
        ...


@dataclass
class PlanDebugStats:
    """
    Representation of debug stats associated with a sharding plan, used for logging.
    """

    planner_type: str
    timeout_seconds: Optional[int]


class Stats(abc.ABC):
    """
    Logs statistics related to the sharding plan.
    """

    @abc.abstractmethod
    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        batch_size: int,
        storage_reservation: StorageReservation,
        num_proposals: int,
        num_plans: int,
        run_time: float,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        enumerator: Optional[Enumerator] = None,
        debug: bool = False,
        debug_stats: Optional[PlanDebugStats] = None,
    ) -> None:
        """
        See class description
        """
        ...


class PlanLoader(abc.ABC):
    """
    Retrieves a pre-computed sharding plan from its stored location. This is useful in two scenarios:
        1. To utilize a specific sharding plan that was previously computed and stored, saving the cost of re-generating the plan
        2. To use a sharding plan from previous runs as a starting point for the next run, allowing for improvement over time.
    """

    @abc.abstractmethod
    def load(
        self,
    ) -> Optional[Dict[int, ShardingOption]]:
        """
        Load sharding plan from its stored location.

        Returns:
            Dict[int, ShardingOption]: loaded sharding plan. key is hash of sharding option to map to sharding option with enumerated sharding option.
        """
        ...

    @abc.abstractmethod
    def plan_context_hash(
        self,
    ) -> Optional[str]:
        """
        Input context hash of a sharding plan.

        Returns:
            str: hash of sharding plan context.
        """
        ...

    ...


@dataclass
class CriticalPathEstimate:
    comms_estimate: float
    comp_estimate: float

    def total(self) -> float:
        return self.comms_estimate + self.comp_estimate


class TrainingFramework(str, Enum):
    """Training framework that consumes the sharding plan.

    Selects framework-specific topology behavior (DDR division logic,
    JustKnob-gated detection, framework-specific adjustments) in downstream
    consumers. ``UNSET`` is the sentinel for "not set", which yields
    framework-agnostic defaults.

    Defined here (rather than imported) because this open-source module cannot
    import the equivalent framework enum that lives in internal code; the string
    values are the contract that bridges the two.

    Subclasses ``str`` so members behave as their string value: they JSON
    serialize to the bare string, compare equal to it, and ``UNSET`` ("") is
    falsy — keeping downstream string-based consumers (fingerprints, configs)
    working without special-casing the enum.
    """

    UNSET = ""
    APF = "apf"
    PYPER = "pyper"
    MVAI = "mvai"
    UNIFIED = "unified"


class PlannerVariant(str, Enum):
    """Planner backend that produces the plan.

    Why an enum rather than a bare string: the frameworks (Pyper/MVAI/APF) pass
    the backend as a free-form config value today, but the set of backends is
    small, closed, and fully owned here in OSS. Normalizing that config string
    into an enum at the request boundary gives (1) a single source of truth for
    the valid backends, (2) typo-safety and autocomplete at call sites, and (3)
    exhaustive, checkable dispatch in the executor (which switches on this to
    pick the planner). Contrast launcher_hardware, kept a ``str`` because its
    value set is large, still growing, and resolved fb-side — the same rule that
    keeps TrainingFramework an enum but launcher_hardware a string.

    Where the string -> enum conversion (and validation) happens: at the framework
    request boundary, via the enum constructor — ``PlannerVariant(cfg_str)`` both
    converts and raises ``ValueError`` on an unknown value in one call. The field
    is typed as the enum and ``PlannerConfig`` does no coercion, so every reader
    sees a ``PlannerVariant`` without a redundant normalization step.

    Subclasses ``str`` so a member serializes/compares as its bare value, keeping
    it stable in the request content hash and in string-based configs.

    ``UNSET`` ("") is the "not specified" sentinel and the default: the executor
    treats it as the OSS backend, so callers that do not care get sensible
    behavior without naming a backend.
    """

    UNSET = ""
    OSS = "oss"
    LINEAR_PROGRAMMING = "linear_programming"
    MANIFOLD = "manifold"


class StorageReservationPolicy(str, Enum):
    """Policy for reserving non-sharded (dense/overhead) memory before planning.

    An enum for the same reasons as ``PlannerVariant``: a small, closed,
    OSS-owned set that the frameworks pass as config strings, normalized here for
    one source of truth, typo-safety, and an exhaustive switch in the
    storage-reservation resolver (which maps each value to a StorageReservation
    implementation). As with ``PlannerVariant``, the string -> enum conversion and
    validation happen at the framework boundary via the enum constructor
    (``StorageReservationPolicy(cfg_str)``); the field is typed as the enum and
    ``PlannerConfig`` does no coercion.

    Subclasses ``str`` for the same serialization reasons as ``PlannerVariant``.

    ``UNSET`` ("") is the "not specified" sentinel and the default: the
    storage-reservation resolver treats it as its default policy (heuristical).

    Only policies with a backing StorageReservation are listed. HEURISTICAL,
    FIXED_PERCENTAGE, and INFERENCE map to Heuristical/FixedPercentage/Inference
    StorageReservation respectively; SKU_AWARE is planned (SKUAwareStorageReservation
    is designed but not yet implemented). "memory_balanced" is intentionally absent
    — it is a partitioner strategy (MemoryBalancedPartitioner), not a reservation.
    """

    UNSET = ""
    HEURISTICAL = "heuristical"
    FIXED_PERCENTAGE = "fixed_percentage"
    INFERENCE = "inference"
    # Planned: SKUAwareStorageReservation (design done, not yet implemented).
    SKU_AWARE = "sku_aware"


@dataclass(frozen=True)
class LpPlannerConfig:
    """OSS-safe scalar knobs for the LINEAR_PROGRAMMING planner variant.

    Carried as plain data so it stays serializable/hashable (part of the request
    content hash) — the fb LinearProgrammingPlanner accepts these scalars (e.g.
    MVAI already passes ``objective``/``shard_solver_type`` as plain strings). Each
    field is Optional; None means "use the LP planner's own default", so the fb
    builder only forwards the ones that are set. The fb-typed forms
    (OptimObjective/ShardSolverType/EMOConfig) are never referenced from OSS.
    """

    # Optimization objective, e.g. "max_total_perf" (LP OptimObjective by value)
    objective: Optional[str] = None
    # Shard solver, e.g. "greedy" (LP ShardSolverType by value)
    shard_solver_type: Optional[str] = None
    # Whether to tune column dimensions
    tune_col_dims: Optional[bool] = None
    # Hybrid solver percentage (0-100)
    hybrid_percentage: Optional[int] = None
    # Allowed solution-quality worsening percentage (>= 0.0)
    allowed_worsening_percentage: Optional[float] = None
    # Whether to apply per-stage timeouts
    stagewise_timeout: Optional[bool] = None
    # Whether plan caching is enabled
    caching_enabled: Optional[bool] = None
    # Number of local-search cycles
    num_cycles: Optional[int] = None
    # Whether to auto-derive column dimensions
    auto_col_dims: Optional[bool] = None


@dataclass(frozen=True)
class PlannerConfig:
    """Plan-affecting knobs the trainer expresses per request.

    These are *data* (serializable, hashable) that select how the planner runs;
    the concrete PlannerExecutor maps them to enumerator/estimator/proposer/
    partitioner/planner instances. Object-valued, framework-specific behavior
    (custom proposer instances, stats sinks) is injected into the API instance as
    a per-framework profile, not carried here — so this stays part of the
    request's content hash and cache key.
    """

    # Planner backend to run; UNSET (default) resolves to the OSS backend
    planner_variant: PlannerVariant = PlannerVariant.UNSET
    # How non-sharded memory is reserved; UNSET (default) resolves to heuristical
    storage_reservation_policy: StorageReservationPolicy = (
        StorageReservationPolicy.UNSET
    )
    # Fraction (0.0-1.0) to reserve for the chosen policy; None = policy default
    storage_reservation_percentage: Optional[float] = None
    # Proposer selector (e.g. "greedy", "grid_search", "dynamic_col_dim"); None =
    # executor default. Free-form: the proposer set is extensible and custom
    # proposer instances are injected via the API profile, so it isn't validated.
    proposer_type: Optional[str] = None
    # Partitioner selector (e.g. "greedy_perf", "memory_balanced"); None =
    # executor default. Free-form for the same reason as proposer_type.
    partitioner_type: Optional[str] = None
    # Use hardware-capability-based compute estimates instead of the default model
    use_hardware_based_compute: bool = False
    # Use hardware-capability-based bandwidths instead of default topology values
    use_hardware_based_bandwidth: bool = False
    # Backward-pass compute multiplier for the perf estimate; None = planner default
    bwd_compute_multiplier: Optional[float] = None
    # Manifold path to a pre-computed sharding plan, consumed only by the MANIFOLD
    # planner_variant (ManifoldPlanner loads the plan from here instead of solving).
    # Required when planner_variant is MANIFOLD; ignored by other variants.
    manifold_path: Optional[str] = None
    # Enable planner debug mode (extra logging/validation). Forwarded to the planner.
    debug: bool = False
    # Solver timeout in seconds; None = planner default. Forwarded to the planner.
    timeout_seconds: Optional[int] = None
    # LINEAR_PROGRAMMING scalar knobs; None = LP planner defaults. Consumed only by
    # the LP variant's fb builder (ignored by other variants).
    lp_config: Optional[LpPlannerConfig] = None

    def __post_init__(self) -> None:
        # planner_variant / storage_reservation_policy are typed as enums, so
        # callers pass a member (framework builders convert a config string with
        # PlannerVariant(cfg_str), which validates). No coercion is done here.
        if (
            self.storage_reservation_percentage is not None
            and not 0.0 <= self.storage_reservation_percentage <= 1.0
        ):
            raise ValueError(
                "storage_reservation_percentage must be between 0.0 and 1.0, got "
                f"{self.storage_reservation_percentage}"
            )
        if self.bwd_compute_multiplier is not None and self.bwd_compute_multiplier < 0:
            raise ValueError(
                "bwd_compute_multiplier must be non-negative, got "
                f"{self.bwd_compute_multiplier}"
            )


@dataclass(frozen=True)
class ShardingPlanRequest:
    """Request for sharding plan generation.

    Encapsulates the model, sharders, cluster topology parameters, and
    optional overrides needed to produce a sharding plan. Base type for
    both runtime and dry-run planning flows.

    Immutability note: ``frozen=True`` only prevents rebinding the fields
    themselves; it does not deep-freeze their contents. The ``sharders``
    list and the ``constraints`` dict remain mutable in place (e.g.
    ``request.sharders.append(...)``). They are kept as ``List``/``Dict``
    because downstream planner APIs consume those concrete types; callers
    must treat them as read-only by convention rather than relying on an
    enforced guarantee.
    """

    # User-provided nn.Module or a factory for lazy construction;
    # consumers should use isinstance(model, nn.Module) to distinguish
    model: Union[nn.Module, Callable[[], nn.Module]]
    # Code-derived from model architecture + training config via get_default_sharders()
    sharders: List[ModuleSharder[nn.Module]]
    # Total number of devices in the cluster
    world_size: int
    # Devices per host — determines intra-host vs inter-host communication split
    local_world_size: int
    # Batch size affects per-device memory estimates from the perf model
    batch_size: int

    # Pod size for multi-pod topologies (None = single pod)
    pod_size: Optional[int] = None
    # Derived from planner.storage_reservation_policy config
    storage_reservation: Optional[StorageReservation] = None
    # Code-derived from embedding tables + fused params + UVM cache stats
    constraints: Optional[Dict[str, ParameterConstraints]] = None
    # Training framework that consumes the plan. TrainingFramework.UNSET (the
    # default) means "not set" and yields framework-agnostic defaults. Typed as
    # an enum so downstream gets a checked value; a plain string (e.g. from
    # config) is coerced to the enum in __post_init__ and an unknown value
    # raises ValueError.
    training_framework: TrainingFramework = TrainingFramework.UNSET
    # Override HBM capacity (GB); None = auto-detect from CUDA or hardware registry
    hbm_gb: Optional[float] = None
    # Override DDR capacity (GB); None = auto-detect
    ddr_gb: Optional[float] = None
    # Launcher hardware identifier for topology creation (e.g. "ZIONEX",
    # "TC_ANY"); None = auto-detect from the launch environment. Kept free-form:
    # the recognized values mirror the hardware-type identifiers resolved by
    # downstream consumers, and that set grows as new accelerators are onboarded,
    # so it is intentionally not validated here.
    launcher_hardware: Optional[str] = None
    # Plan-affecting knobs (planner backend, reservation policy, proposer/
    # partitioner selection, hardware-based flags). Data-only so it participates
    # in request_hash; object-valued behavior is injected into the API instance.
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    # Unique per-instance id for this request, used to correlate it with the
    # ShardingPlanResult(s) it produces (see ShardingPlanResult.request_id).
    # Complements request_hash: request_hash is a *content* hash shared by
    # identical requests (cache/dedup key), whereas request_id is unique per
    # request object — use it to tell two otherwise-identical requests apart.
    # Auto-generated; override to thread an externally-supplied id.
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        # Accept either a TrainingFramework or its string value (e.g. read from
        # config) and normalize to the enum, so downstream always reads a
        # TrainingFramework. Unknown strings raise ValueError.
        if not isinstance(self.training_framework, TrainingFramework):
            try:
                object.__setattr__(
                    self,
                    "training_framework",
                    TrainingFramework(self.training_framework),
                )
            except ValueError as e:
                valid = [f.value for f in TrainingFramework]
                raise ValueError(
                    f"training_framework must be a TrainingFramework or one of "
                    f"{valid}, got {self.training_framework!r}"
                ) from e
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if self.local_world_size <= 0:
            raise ValueError(
                f"local_world_size must be positive, got {self.local_world_size}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.pod_size is not None and self.pod_size <= 0:
            raise ValueError(f"pod_size must be positive, got {self.pod_size}")
        if self.hbm_gb is not None and self.hbm_gb < 0:
            raise ValueError(f"hbm_gb must be non-negative, got {self.hbm_gb}")
        if self.ddr_gb is not None and self.ddr_gb < 0:
            raise ValueError(f"ddr_gb must be non-negative, got {self.ddr_gb}")
        if self.local_world_size > self.world_size:
            raise ValueError(
                f"local_world_size ({self.local_world_size}) must not exceed "
                f"world_size ({self.world_size})"
            )
        if self.world_size % self.local_world_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"local_world_size ({self.local_world_size})"
            )

    @property
    def request_hash(self) -> str:
        """Stable content hash identifying this request.

        Deterministic over the planner-affecting parameters, so two requests
        with the same parameters share a hash. Used to correlate the request
        with its ShardingPlanResult(s) and PlannerSessionContext, and as a
        cache key. Excludes `model` and `sharders` (not stably hashable), so
        callers needing model-level uniqueness should scope by the context's
        model — mirroring DryRunRequest.fingerprint().
        """
        return format(
            hash_sha256_to_int(
                [
                    self.world_size,
                    self.local_world_size,
                    self.batch_size,
                    self.pod_size,
                    self.hbm_gb,
                    self.ddr_gb,
                    self.training_framework.value,
                    self.launcher_hardware,
                    self.constraints,
                    self.planner_config.planner_variant.value,
                    self.planner_config.storage_reservation_policy.value,
                    self.planner_config.storage_reservation_percentage,
                    self.planner_config.proposer_type,
                    self.planner_config.partitioner_type,
                    self.planner_config.use_hardware_based_compute,
                    self.planner_config.use_hardware_based_bandwidth,
                    self.planner_config.bwd_compute_multiplier,
                    self.planner_config.manifold_path,
                    self.planner_config.debug,
                    self.planner_config.timeout_seconds,
                    (
                        (
                            lp.objective,
                            lp.shard_solver_type,
                            lp.tune_col_dims,
                            lp.hybrid_percentage,
                            lp.allowed_worsening_percentage,
                            lp.stagewise_timeout,
                            lp.caching_enabled,
                            lp.num_cycles,
                            lp.auto_col_dims,
                        )
                        if (lp := self.planner_config.lp_config) is not None
                        else None
                    ),
                ]
            ),
            "x",
        )[:16]


# ---- Types Utils ---- #
def hash_sha256_to_int(hashable_list: List[Any]) -> int:
    """
    Hashes the given data using SHA256 and returns the hash as an integer
    """
    serialized_list = str(hashable_list).encode("utf-8")
    hash_object = hashlib.sha256(serialized_list)
    hash_digest = hash_object.hexdigest()
    return int(hash_digest, 16)


def hash_sha256_str(hashable_list: List[Any]) -> str:
    """
    Hashes the given data using SHA256 and returns the hash as an string
    """
    serialized_list = str(hashable_list).encode("utf-8")
    hash_object = hashlib.sha256(serialized_list)
    hash_digest = hash_object.hexdigest()
    return hash_digest


def _topology_hash_components(
    topology: Topology,
    round_unit: int = HUNDRED_GB,
) -> List[Any]:
    """Extract hash-stable components from a Topology with storage rounding.

    Device memory (HBM/DDR/SSD) is rounded to the nearest ``round_unit``
    (default 100 GB) so that minor driver/OS differences across machines
    do not change the hash.  Every other field is included verbatim.

    This helper is the *single* place where topology normalisation happens.
    Both ``hash_planner_context_inputs`` and ``hash_planner_context_inputs_str``
    must use it for every Topology they include in the hash (raw topology,
    ``_last_reserved_topology``, etc.).
    """
    rounded_devices = []
    for device in topology.devices:
        rounded_devices.append(
            (
                device.rank,
                round_to_nearest(device.storage.hbm, round_unit),
                round_to_nearest(device.storage.ddr, round_unit),
                round_to_nearest(device.storage.ssd, round_unit),
            )
        )
    return [
        topology.world_size,
        topology.compute_device,
        rounded_devices,
        topology.local_world_size,
        topology.intra_group_size,
        topology.hbm_mem_bw,
        topology.ddr_mem_bw,
        topology.ssd_mem_bw,
        topology.hbm_to_ddr_mem_bw,
        topology.comms_bandwidths.intra_host_bw,
        topology.comms_bandwidths.inter_host_bw,
        topology.bwd_compute_multiplier,
        topology.weighted_feature_bwd_compute_multiplier,
        topology.uneven_sharding_perf_multiplier,
    ]


def _shard_hash_components(shard: "Shard") -> tuple:
    """Extract hash-stable components from a Shard.

    Uses explicit field extraction rather than ``__repr__()`` so the hash
    is not affected by new fields added to ``Shard`` or by
    machine-specific values leaking through ``Storage``/``Perf`` reprs.
    """
    return (
        tuple(shard.size),
        tuple(shard.offset),
        shard.rank,
        (
            (shard.storage.hbm, shard.storage.ddr, shard.storage.ssd)
            if shard.storage
            else None
        ),
    )


def _build_hashable_list(
    topology: Topology,
    batch_size: int,
    enumerator: Enumerator,
    storage_reservation: StorageReservation,
    constraints: Optional[Dict[str, ParameterConstraints]],
) -> List[Any]:
    """Build the canonical hashable list for planner context inputs.

    Shared by both ``hash_planner_context_inputs`` (int hash) and
    ``hash_planner_context_inputs_str`` (str hash) so the two can never
    drift apart.
    """
    assert hasattr(
        enumerator, "last_stored_search_space"
    ), "This enumerator is not compatible with hashing"
    assert (
        enumerator.last_stored_search_space is not None
    ), "Unable to hash planner context without an enumerator that has a precomputed search space"

    reserved_topology = storage_reservation.last_reserved_topology
    assert (
        reserved_topology is not None
    ), "Unable to hash planner context without a storage reservation that has a precomputed topology"

    search_space = enumerator.last_stored_search_space
    storage_reservation_policy = type(storage_reservation).__name__

    # Hash topology components with storage rounding applied uniformly.
    # Previously _last_reserved_topology was included as a raw Topology
    # object, whose __repr__ embedded unrounded device DDR values that
    # vary across machines — causing cache misses on MAST job restarts.
    hashed_topology = hash_sha256_to_int(_topology_hash_components(topology))
    hashed_reserved_topology = hash_sha256_to_int(
        _topology_hash_components(reserved_topology)
    )

    return [
        hashed_topology,
        batch_size,
        [
            [
                shard_option.fqn,
                shard_option.sharding_type,
                shard_option.compute_kernel,
                tuple(_shard_hash_components(shard) for shard in shard_option.shards),
                shard_option.cache_params,
            ]
            for shard_option in search_space
        ],
        storage_reservation_policy,
        hashed_reserved_topology,
        (
            tuple((k, v.__hash__()) for k, v in sorted(constraints.items()))
            if constraints
            else None
        ),
    ]


def hash_planner_context_inputs(
    topology: Topology,
    batch_size: int,
    enumerator: Enumerator,
    storage_reservation: StorageReservation,
    constraints: Optional[Dict[str, ParameterConstraints]],
    hash_function: Callable[[List[Any]], int] = hash_sha256_to_int,
) -> int:
    hashable_list = _build_hashable_list(
        topology, batch_size, enumerator, storage_reservation, constraints
    )
    return hash_function(hashable_list)


def hash_planner_context_inputs_str(
    topology: Topology,
    batch_size: int,
    enumerator: Enumerator,
    storage_reservation: StorageReservation,
    constraints: Optional[Dict[str, ParameterConstraints]],
    hash_function: Callable[[List[Any]], str] = hash_sha256_str,
) -> str:
    hashable_list = _build_hashable_list(
        topology, batch_size, enumerator, storage_reservation, constraints
    )
    return hash_function(hashable_list)


def round_to_nearest(x: int, unit: int) -> int:
    """Round to nearest unit (e.g., 100GB)."""
    return round(x / unit) * unit
