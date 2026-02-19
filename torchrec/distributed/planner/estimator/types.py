#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


"""
Performance Estimator Types.

This module contains the core dataclasses  performance estimator:
- Coefficient classes: PerfCoefficient, EstimatorPerfCoefficients, PerfCoefficientConfig
- Context class: ShardPerfContext
- Config class: HardwarePerfConfig

Evaluator classes and factory are in estimator.py.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    BIGINT_DTYPE,
    BWD_COMPUTE_MULTIPLIER,
    DDR_MEM_BW,
    HBM_MEM_BW,
    kernel_bw_lookup,
    SSD_MEM_BW,
    WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,
    WEIGHTED_KERNEL_MULTIPLIER,
)
from torchrec.distributed.planner.estimator.annotations import (
    get_bwd_coefficient,
    get_fwd_coefficient,
    get_prefetch_coefficient,
)
from torchrec.distributed.planner.types import (
    GeneralizedCommsBandwidth,
    ParameterConstraints,
    Perf,
    PlannerError,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import (
    extract_comm_data_type_size,
    get_num_poolings,
    is_prefetch_pipelined,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Coefficient Dataclasses
# =============================================================================


@dataclass(frozen=True)
class PerfCoefficient:
    # Multiplier for number of indices to lookup.
    input_read_size_multiplier: float = 1.0

    # Multiplier for lookup size.
    lookup_size_multiplier: float = 1.0

    # Multiplier after pooled operation,basically the size of the output.
    embedding_output_multiplier: float = 1.0

    # Multiplier for table size
    hash_size_multiplier: float = 0.0


@dataclass(frozen=True)
class EstimatorPerfCoefficients:
    # Coefficients for forward pass
    fwd: PerfCoefficient = field(default_factory=PerfCoefficient)
    # Coefficients for backward pass (None = use fwd_compute * bwd_compute_multiplier)
    bwd: Optional[PerfCoefficient] = None


@dataclass(frozen=True)
class PrefetchCoefficients:
    """
    Prefetch pipeline per estimation coefficients.
    """

    # Multiplier for number of lookups
    expected_num_lookups_coefficient: float = 0.0
    # Multipliers for number of unique lookups
    expected_num_unique_lookups_coefficient: float = 0.0
    # Multiplier for number of cache fetches
    expected_size_cache_fetches_coefficient: float = 0.0


@dataclass
class PerfCoefficientConfig:
    """
    Each sharding type has a set of coefficients for forward and backward passes., If these coefficients are need to be specified for
    a specific sharding type,for a particular compute kernel, then they can be specified in the by_kernel dictionary.
    Priority order:
    1. by_kernel[(sharding_type, compute_kernel)] if compute_kernel provided
    2. Sharding-type specific attribute
    3. default
    """

    table_wise: Optional[EstimatorPerfCoefficients] = None
    row_wise: Optional[EstimatorPerfCoefficients] = None
    table_row_wise: Optional[EstimatorPerfCoefficients] = None
    data_parallel: Optional[EstimatorPerfCoefficients] = None
    column_wise: Optional[EstimatorPerfCoefficients] = None

    # Kernel-specific overrides: (sharding_type, compute_kernel) -> coefficients
    by_kernel: Dict[Tuple[str, str], EstimatorPerfCoefficients] = field(
        default_factory=dict
    )

    # Default fallback
    default: EstimatorPerfCoefficients = field(
        default_factory=lambda: EstimatorPerfCoefficients()
    )

    # Compute multipliers (imported from constants)
    bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER
    weighted_kernel_multiplier: float = WEIGHTED_KERNEL_MULTIPLIER
    weighted_feature_bwd_compute_multiplier: float = (
        WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER
    )

    # Prefetch coefficients (optional, for FB hardware estimators)
    # When set and use_linear_regression=True, enables extended prefetch formula
    prefetch: Optional[PrefetchCoefficients] = None

    def get_coefficients(
        self, sharding_type: str, compute_kernel: Optional[str] = None
    ) -> EstimatorPerfCoefficients:
        """
        Get coefficients for a sharding type and optional compute kernel.

        Priority order:
        1. by_kernel[(sharding_type, compute_kernel)] if compute_kernel provided
        2. Sharding-type specific attribute
        3. default
        """
        # Priority 1: Check kernel-specific override
        if compute_kernel:
            key = (sharding_type.lower(), compute_kernel.lower())
            if key in self.by_kernel:
                return self.by_kernel[key]

        # Priority 2: Check sharding-type specific
        sharding_type_lower = sharding_type.lower()

        if sharding_type_lower == ShardingType.TABLE_WISE.value:
            coeff = self.table_wise
        elif sharding_type_lower == ShardingType.ROW_WISE.value:
            coeff = self.row_wise
        elif sharding_type_lower == ShardingType.TABLE_ROW_WISE.value:
            coeff = self.table_row_wise
        elif sharding_type_lower == ShardingType.DATA_PARALLEL.value:
            coeff = self.data_parallel
        elif sharding_type_lower == ShardingType.COLUMN_WISE.value:
            coeff = self.column_wise
        elif sharding_type_lower == ShardingType.TABLE_COLUMN_WISE.value:
            coeff = self.column_wise
        elif sharding_type_lower == ShardingType.GRID_SHARD.value:
            coeff = self.table_row_wise
        else:
            coeff = None

        if coeff is not None:
            return coeff

        # Priority 3: Default
        return self.default


# =============================================================================
# HardwarePerfConfig
# =============================================================================


class HardwarePerfConfig:
    """
    Hardware-specific performance configuration.

    This class contains hardware-specific coefficients and compute methods.
    It is used by the EmbeddingPerfEstimator to estimate performance for a given hardware.

    Args:
        name: Name of the hardware (e.g. "H100", "A100", "V100").
        coefficients: Coefficients for each sharding type.
    """

    # Internal storage for coefficients (populated lazily from annotations)
    _coefficients: Optional[PerfCoefficientConfig] = None

    name: str = "default"

    @property
    def coefficients(self) -> PerfCoefficientConfig:
        """
        Get coefficients config, populated from annotations.

        This property dynamically builds the PerfCoefficientConfig from
        annotation-based coefficients (@fwd_coefficient, @bwd_coefficient, @prefetch_coefficient).
        The result is cached for subsequent accesses.

        Returns:
            PerfCoefficientConfig with coefficients populated from annotations.
        """
        if self._coefficients is not None:
            return self._coefficients

        # Build coefficients from annotations for each sharding type
        sharding_types = [
            ShardingType.TABLE_WISE,
            ShardingType.ROW_WISE,
            ShardingType.TABLE_ROW_WISE,
            ShardingType.COLUMN_WISE,
            ShardingType.DATA_PARALLEL,
        ]

        config_kwargs: Dict[str, Optional[EstimatorPerfCoefficients]] = {}

        for sharding_type in sharding_types:
            fwd_coeff = get_fwd_coefficient(self, sharding_type.value)
            bwd_coeff = get_bwd_coefficient(self, sharding_type.value)

            if fwd_coeff is not None or bwd_coeff is not None:
                config_kwargs[sharding_type.value] = EstimatorPerfCoefficients(
                    fwd=fwd_coeff if fwd_coeff is not None else PerfCoefficient(),
                    bwd=bwd_coeff,  # Keep as None if @bwd_coefficient not defined
                )

        # Get prefetch coefficients
        prefetch = get_prefetch_coefficient(self)

        self._coefficients = PerfCoefficientConfig(
            table_wise=config_kwargs.get(ShardingType.TABLE_WISE.value),
            row_wise=config_kwargs.get(ShardingType.ROW_WISE.value),
            table_row_wise=config_kwargs.get(ShardingType.TABLE_ROW_WISE.value),
            column_wise=config_kwargs.get(ShardingType.COLUMN_WISE.value),
            data_parallel=config_kwargs.get(ShardingType.DATA_PARALLEL.value),
            prefetch=prefetch,
        )

        return self._coefficients

    # Hardware bandwidth defaults (can be overridden by decorators)
    hbm_mem_bw: float = HBM_MEM_BW
    ddr_mem_bw: float = DDR_MEM_BW
    ssd_mem_bw: float = SSD_MEM_BW

    # Optional bandwidth overrides (None = use ctx/topology values)
    device_bw: Optional[float] = None
    hbm_to_ddr_mem_bw: Optional[float] = None
    intra_host_bw: Optional[float] = None
    inter_host_bw: Optional[float] = None

    kernel_device_bandwidths: Dict[Tuple[str, str], float] = {}

    # Supported sharding types (None = all supported, set via @supported_sharding_types)
    _supported_sharding_types: Optional[FrozenSet[str]] = None

    # Configurable data type defaults (can be overridden by subclasses)
    # FB legacy uses INT_DTYPE (4.0) as default output_data_type_size when output_dtype is None
    # OSS uses tensor.element_size() (can be 2.0 for FP16)
    # Set to 4.0 in FBHardwarePerfConfig to match FB legacy behavior, None for OSS default
    _default_output_data_type_size: Optional[float] = None

    # Optional multiplier overrides (None = use ctx/topology values)
    # These allow configs to override the default multipliers from topology
    _bwd_compute_multiplier: Optional[float] = None
    _weighted_feature_bwd_compute_multiplier: Optional[float] = None

    # Linear regression prefetch estimation flag
    # When True: use linear regression coefficients (requires @prefetch_coefficient)
    # When False: use bandwidth-based formula (prefetch_bytes / hbm_to_ddr_mem_bw)
    # FB hardware estimators set this to True when tuned coefficients are available
    _use_linear_regression_prefetch_estimate: bool = False

    def get_bwd_compute_multiplier(self, ctx_multiplier: float) -> float:
        """
        Get backward compute multiplier with priority-based lookup.

        Priority:
        1. Config-defined _bwd_compute_multiplier (if explicitly set)
        2. ctx_multiplier (from topology)

        Args:
            ctx_multiplier: The multiplier from ShardPerfContext (sourced from topology)

        Returns:
            The backward compute multiplier to use
        """
        if self._bwd_compute_multiplier is not None:
            return self._bwd_compute_multiplier
        return ctx_multiplier

    def get_weighted_feature_bwd_compute_multiplier(
        self, ctx_multiplier: float
    ) -> float:
        """
        Get weighted feature backward compute multiplier with priority-based lookup.

        Priority:
        1. Config-defined _weighted_feature_bwd_compute_multiplier (if explicitly set)
        2. ctx_multiplier (from topology)

        Args:
            ctx_multiplier: The multiplier from ShardPerfContext (sourced from topology)

        Returns:
            The weighted feature backward compute multiplier to use
        """
        if self._weighted_feature_bwd_compute_multiplier is not None:
            return self._weighted_feature_bwd_compute_multiplier
        return ctx_multiplier

    def get_coefficients_for_sharding(
        self, sharding_type: str, compute_kernel: Optional[str] = None
    ) -> EstimatorPerfCoefficients:
        """
        Get coefficients for a specific sharding type.

        Priority order:
        1. Annotation-based coefficients (@fwd_coefficient, @bwd_coefficient)
        2. Explicit self.coefficients attribute (PerfCoefficientConfig)
        3. Default coefficients

        Args:
            sharding_type: The sharding type to get coefficients for
            compute_kernel: Optional compute kernel for kernel-specific overrides

        Returns:
            EstimatorPerfCoefficients with fwd and bwd coefficients
        """
        # Priority 1: Check for annotation-based coefficients
        fwd_coeff = get_fwd_coefficient(self, sharding_type)
        bwd_coeff = get_bwd_coefficient(self, sharding_type)

        if fwd_coeff is not None or bwd_coeff is not None:
            return EstimatorPerfCoefficients(
                fwd=fwd_coeff if fwd_coeff is not None else PerfCoefficient(),
                bwd=bwd_coeff,  # Keep as None if @bwd_coefficient not defined
            )

        # Priority 2 & 3: Use explicit coefficients attribute (falls back to default)
        return self.coefficients.get_coefficients(sharding_type, compute_kernel)

    def is_sharding_type_supported(self, sharding_type: str) -> bool:
        """
        Check if a sharding type is supported by this config.

        If _supported_sharding_types is None, all types are supported.
        If set (via @supported_sharding_types decorator), only listed types are supported.

        Args:
            sharding_type: The sharding type to check

        Returns:
            True if supported, False otherwise
        """
        if self._supported_sharding_types is None:
            return True  # All supported by default
        return sharding_type.lower() in self._supported_sharding_types

    def validate_sharding_type(self, sharding_type: str) -> None:
        """
        Validate that a sharding type is supported, raise ValueError if not.

        Args:
            sharding_type: The sharding type to validate

        Raises:
            ValueError: If sharding type is not supported
        """
        if not self.is_sharding_type_supported(sharding_type):
            supported = self._supported_sharding_types or "all"
            raise ValueError(
                f"Sharding type '{sharding_type}' is not supported by {self.__class__.__name__}. "
                f"Supported types: {supported}"
            )

    def post_process_perfs(
        self,
        shard_perfs: List[Perf],
        shard_sizes: List[List[int]],
        sharding_type: str,
        uneven_sharding_perf_multiplier: float = 1.0,
    ) -> List[Perf]:
        """
        Optional post-processing hook for adjusting shard perfs after computation.

        This method is called after all shards have been computed individually.
        Override in subclasses for custom cross-shard adjustments (e.g., uneven sharding).

        Default implementation: returns shard_perfs unchanged.

        Args:
            shard_perfs: List of computed Perf objects, one per shard
            shard_sizes: List of [hash_size, emb_dim] for each shard
            sharding_type: The sharding type being evaluated
            uneven_sharding_perf_multiplier: Multiplier for uneven sharding adjustment

        Returns:
            List of (potentially adjusted) Perf objects
        """
        return shard_perfs

    def get_device_bw(
        self,
        compute_device: str,
        compute_kernel: str,
        hbm_mem_bw: float,
        ddr_mem_bw: float,
        ssd_mem_bw: float,
        hbm_to_ddr_mem_bw: float,
        caching_ratio: Optional[float] = None,
        prefetch_pipeline: bool = False,
    ) -> Optional[float]:
        """
        Get device bandwidth with priority-based lookup.

        Priority order:
        1. kernel_device_bandwidths (specific device+kernel overrides)
        2. device_bw (general device bandwidth override)
        3. kernel_bw_lookup() (computed from memory bandwidth)

        Args:
            compute_device: The compute device (e.g., "cuda", "cpu", "mtia")
            compute_kernel: The embedding compute kernel (e.g., "fused", "dense")
            hbm_mem_bw: HBM memory bandwidth
            ddr_mem_bw: DDR memory bandwidth
            ssd_mem_bw: SSD memory bandwidth
            hbm_to_ddr_mem_bw: HBM to DDR bandwidth
            caching_ratio: Optional caching ratio for UVM caching
            prefetch_pipeline: Whether prefetch pipeline is enabled

        Returns:
            The device bandwidth in bytes/ms, or None if not found
        """
        # Following same logic as kernel_bw_lookup
        effective_compute_kernel = compute_kernel
        if (
            prefetch_pipeline
            and compute_device.lower() == "cuda"
            and compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        ):
            effective_compute_kernel = EmbeddingComputeKernel.FUSED.value

        # Priority 1: Check kernel-specific overrides (case-insensitive lookup)
        key = (compute_device.lower(), effective_compute_kernel.lower())
        if key in self.kernel_device_bandwidths:
            return self.kernel_device_bandwidths[key]

        # Priority 2: Check general device_bw override
        if self.device_bw is not None:
            return self.device_bw

        # Priority 3: Use kernel_bw_lookup (returns None for invalid kernels)
        bw = kernel_bw_lookup(
            compute_device=compute_device,
            compute_kernel=compute_kernel,
            hbm_mem_bw=hbm_mem_bw,
            ddr_mem_bw=ddr_mem_bw,
            hbm_to_ddr_mem_bw=hbm_to_ddr_mem_bw,
            caching_ratio=caching_ratio,
            prefetch_pipeline=prefetch_pipeline,
            ssd_mem_bw=ssd_mem_bw,
        )
        if bw is None:
            raise ValueError(
                f"Unrecognized or unsupported compute kernel: {compute_kernel} for hardware aware perf estimators."
            )
        return bw


# =============================================================================
# ShardPerfContext
# =============================================================================


@dataclass
class ShardPerfContext:
    """
    Raw data container for shard performance estimation.

    This is a pure data container - all computation logic is in the evaluator classes.
    Evaluators use the raw data fields to compute sizes and performance metrics.
    """

    # Identifiers
    sharding_type: str = ""
    compute_kernel: str = ""

    # Shard dimensions
    hash_size: int = 0
    emb_dim: int = 0

    # Raw inputs
    batch_sizes: List[int] = field(default_factory=list)
    num_poolings: List[float] = field(default_factory=list)
    input_lengths: List[float] = field(default_factory=list)

    # Topology info
    world_size: int = 1
    local_world_size: int = 1

    # Data type sizes
    input_data_type_size: float = BIGINT_DTYPE
    table_data_type_size: float = 4.0
    output_data_type_size: float = 4.0
    fwd_a2a_comm_data_type_size: float = 4.0
    bwd_a2a_comm_data_type_size: float = 4.0
    fwd_sr_comm_data_type_size: float = 4.0
    bwd_sr_comm_data_type_size: float = 4.0

    # Bandwidth values
    device_bw: float = HBM_MEM_BW
    hbm_to_ddr_mem_bw: float = HBM_MEM_BW
    intra_host_bw: float = 0.0
    inter_host_bw: float = 0.0

    # Communication bandwidths
    comms_bandwidths: Optional[GeneralizedCommsBandwidth] = None

    # Flags
    is_inference: bool = False
    is_weighted: bool = False
    is_pooled: bool = True
    has_feature_processor: bool = False

    # Multipliers from topology
    bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER
    weighted_feature_bwd_compute_multiplier: float = (
        WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER
    )

    # =========================================================================
    # Raw prefetch data - passed to _default_prefetch_comp for computation
    # =========================================================================
    # Prefetch computation happens in _default_prefetch_comp, not in build_shard_perf_contexts.
    # This allows prefetch logic to be customized via config annotations.

    # Flags controlling prefetch computation (passed from config/estimator)
    use_linear_regression_prefetch_estimate: bool = False
    use_batch_inputs_for_expected_cache_fetches: bool = False

    # Raw cache data for prefetch computation
    caching_ratio: Optional[float] = None
    table_name: str = ""
    hash_size_for_clamping: int = 0  # sharding_option.tensor.shape[0] for clamping

    # Raw cache stats expected_lookups (from cache_stats.expected_lookups)
    cache_stats_expected_lookups: Optional[float] = None

    # Computed expected_miss_rate (computed once from cache_stats in build_shard_perf_contexts)
    expected_miss_rate: Optional[float] = None

    # =========================================================================
    # Class method to build from ShardingOption
    # =========================================================================

    @classmethod
    def build_shard_perf_contexts(
        cls,
        config: HardwarePerfConfig,
        shard_sizes: List[List[int]],
        sharding_option: ShardingOption,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]],
        sharder: ModuleSharder[nn.Module],
        is_inference: bool = False,
        use_batch_inputs_for_expected_cache_fetches: bool = False,
        use_linear_regression_prefetch_estimate: bool = False,
    ) -> List["ShardPerfContext"]:
        """
        Build list of ShardPerfContexts from ShardingOption and Topology.

        This follows the exact logic from OSS EmbeddingPerfEstimator.estimate.

        Args:
            config: Hardware performance configuration
            shard_sizes: List of [hash_size, emb_dim] for each shard
            sharding_option: The sharding option being evaluated
            topology: Device topology with bandwidth and world size info
            constraints: Optional parameter constraints
            sharder: Module sharder for this option
            is_inference: Whether this is for inference
            use_batch_inputs_for_expected_cache_fetches: If True, expected_cache_fetches
                is computed as expected_miss_rate * batch_inputs (total lookups per batch).
                If False (default), uses expected_miss_rate * expected_unique_lookups.
            use_linear_regression_prefetch_estimate: If True, clamps num_unique_lookups
                to min(num_unique_lookups, batch_inputs, hash_size) before computing
                prefetch time.

        Returns:
            List of ShardPerfContext instances, one per shard.
        """
        # Get caching ratio
        caching_ratio = sharding_option.cache_load_factor
        if caching_ratio is None:
            caching_ratio = (
                sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                if hasattr(sharder, "fused_params") and sharder.fused_params
                else None
            )

        # Get num_poolings and batch_sizes
        num_poolings = get_num_poolings(constraints, sharding_option)
        batch_sizes = (
            list(constraints[sharding_option.name].batch_sizes or [])
            if constraints
            and constraints.get(sharding_option.name)
            and constraints[sharding_option.name].batch_sizes
            else [sharding_option.batch_size] * sharding_option.num_inputs
        )

        assert (
            len(sharding_option.input_lengths) == len(num_poolings) == len(batch_sizes)
        ), "Provided `pooling_factors`, `num_poolings`, and `batch_sizes` constraints must match."

        # Check for feature processor
        module = sharding_option.module[1]
        has_feature_processor = False
        if (
            hasattr(module, "_feature_processor")
            and hasattr(module._feature_processor, "feature_processor_modules")
            and isinstance(
                module._feature_processor.feature_processor_modules,  # pyre-ignore[16]
                nn.ModuleDict,
            )
            and sharding_option.name
            in module._feature_processor.feature_processor_modules.keys()  # pyre-ignore[16]
        ):
            has_feature_processor = True

        # Determine is_weighted
        if isinstance(module, EmbeddingBagCollectionInterface):
            is_weighted = module.is_weighted()
        elif (
            constraints
            and constraints.get(sharding_option.name)
            and constraints[sharding_option.name].is_weighted
        ):
            is_weighted = constraints[sharding_option.name].is_weighted
        else:
            is_weighted = False

        is_weighted = is_weighted or has_feature_processor

        # Get data type sizes
        table_data_type_size = sharding_option.tensor.element_size()
        (
            fwd_a2a_comm_data_type_size,
            bwd_a2a_comm_data_type_size,
            fwd_sr_comm_data_type_size,
            bwd_sr_comm_data_type_size,
        ) = extract_comm_data_type_size(sharder, sharding_option)

        # Check prefetch pipeline
        prefetch_pipeline = is_prefetch_pipelined(sharding_option, sharder)

        # Get input_data_type_size from config annotation (if set) or use default BIGINT_DTYPE
        input_data_type_size = getattr(config, "_input_data_type_size", BIGINT_DTYPE)

        # Output data type size - fetch default from config if output_dtype not specified
        # FB legacy uses INT_DTYPE (4.0), OSS uses tensor.element_size()
        default_output_data_type_size = getattr(
            config, "_default_output_data_type_size", None
        )
        output_data_type_size: float = (
            DATA_TYPE_NUM_BITS[sharding_option.output_dtype] / 8
            if sharding_option.output_dtype
            else (
                default_output_data_type_size
                if default_output_data_type_size is not None
                else sharding_option.tensor.element_size()
            )
        )

        # =========================================================================
        # Raw prefetch data - passed to _default_prefetch_comp for computation
        # =========================================================================
        # Prefetch computation happens in _default_prefetch_comp, not here.
        # This allows prefetch logic to be customized via config annotations.

        # Extract raw cache stats for prefetch computation
        cache_stats_expected_lookups: Optional[float] = None
        expected_miss_rate: Optional[float] = None

        if (
            caching_ratio is not None
            and sharding_option.cache_params is not None
            and sharding_option.cache_params.stats is not None
            and sharding_option.compute_kernel
            == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        ):
            _stats = sharding_option.cache_params.stats
            cache_stats_expected_lookups = _stats.expected_lookups
            expected_miss_rate = _stats.expected_miss_rate(caching_ratio)

        # Get device bandwidth
        device_bw = config.get_device_bw(
            topology.compute_device,
            sharding_option.compute_kernel,
            topology.hbm_mem_bw,
            topology.ddr_mem_bw,
            topology.ssd_mem_bw,
            topology.hbm_to_ddr_mem_bw,
            caching_ratio,
            prefetch_pipeline,
        )

        if device_bw is None:
            raise PlannerError(
                f"No kernel bandwidth for compute device: {topology.compute_device}, "
                f"compute kernel: {sharding_option.compute_kernel}"
            )

        # Build contexts
        contexts = []
        for hash_size, emb_dim in shard_sizes:
            ctx = cls(
                sharding_type=sharding_option.sharding_type,
                compute_kernel=sharding_option.compute_kernel,
                hash_size=hash_size,
                emb_dim=emb_dim,
                batch_sizes=batch_sizes,
                num_poolings=num_poolings,
                input_lengths=sharding_option.input_lengths,
                world_size=topology.world_size,
                local_world_size=topology.local_world_size,
                input_data_type_size=input_data_type_size,
                table_data_type_size=table_data_type_size,
                output_data_type_size=output_data_type_size,
                fwd_a2a_comm_data_type_size=fwd_a2a_comm_data_type_size,
                bwd_a2a_comm_data_type_size=bwd_a2a_comm_data_type_size,
                fwd_sr_comm_data_type_size=fwd_sr_comm_data_type_size,
                bwd_sr_comm_data_type_size=bwd_sr_comm_data_type_size,
                device_bw=device_bw,
                hbm_to_ddr_mem_bw=topology.hbm_to_ddr_mem_bw,
                comms_bandwidths=topology.comms_bandwidths,
                is_inference=is_inference,
                is_weighted=is_weighted,
                is_pooled=sharding_option.is_pooled,
                has_feature_processor=has_feature_processor,
                bwd_compute_multiplier=topology.bwd_compute_multiplier,
                weighted_feature_bwd_compute_multiplier=topology.weighted_feature_bwd_compute_multiplier,
                # Raw prefetch data - computation happens in _default_prefetch_comp
                use_linear_regression_prefetch_estimate=use_linear_regression_prefetch_estimate,
                use_batch_inputs_for_expected_cache_fetches=use_batch_inputs_for_expected_cache_fetches,
                caching_ratio=caching_ratio,
                table_name=sharding_option.name,
                hash_size_for_clamping=sharding_option.tensor.shape[0],
                cache_stats_expected_lookups=cache_stats_expected_lookups,
                expected_miss_rate=expected_miss_rate,
            )
            contexts.append(ctx)

        return contexts

    # =========================================================================
    # Computed properties - raw derived values only
    # =========================================================================

    @property
    def num_hosts(self) -> int:
        """Number of hosts in the topology."""
        return max(1, self.world_size // self.local_world_size)

    @property
    def batch_inputs(self) -> float:
        """Raw batch inputs: sum of input_lengths * num_poolings * batch_sizes."""
        return sum(
            x * y * z
            for x, y, z in zip(self.input_lengths, self.num_poolings, self.batch_sizes)
        )

    @property
    def batch_outputs(self) -> float:
        """
        Raw batch outputs.

        For pooled: sum of num_poolings * batch_sizes.
        For unpooled: same as batch_inputs.
        """
        if self.is_pooled:
            return sum(x * y for x, y in zip(self.num_poolings, self.batch_sizes))
        return self.batch_inputs

    @property
    def is_uvm_caching(self) -> bool:
        """Check if using UVM caching."""
        return self.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
