#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Embedding Performance Estimator with Evaluator Pattern.
This module contains
- EmbeddingShardingPerfEvaluator: Base class with the strategy pattern
- Evaluator implementations for each sharding type
- EmbeddingPerfEstimatorFactory: Factory for creating estimators
- EmbeddingPerfEstimator: Main estimator class
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

from torch import nn
from torchrec.distributed.planner.constants import (
    BATCHED_COPY_PERF_FACTOR,
    DP_ELEMENTWISE_KERNELS_PERF_FACTOR,
    FULL_BLOCK_EMB_DIM,
    HALF_BLOCK_PENALTY,
    QUARTER_BLOCK_PENALTY,
)
from torchrec.distributed.planner.estimator.annotations import get_output_write_size
from torchrec.distributed.planner.estimator.types import (
    HardwarePerfConfig,
    ShardPerfContext,
)
from torchrec.distributed.planner.types import (
    CollectiveType,
    ParameterConstraints,
    Perf,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder, ShardingType

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# EmbeddingPerfShardingEvaluator Base Class
# =============================================================================


class EmbeddingShardingPerfEvaluator(ABC):
    """
    #TODO: override for inference cases later
    This class is the base class for all sharding type evaluators.
    """

    # =========================================================================
    # Composable Communication Primitives
    # =========================================================================

    def _compute_collective_comms(
        self,
        ctx: ShardPerfContext,
        output_size: float,
        collective_type: CollectiveType,
        world_size: int,
        local_world_size: int,
    ) -> float:
        """
        Compute communication time for a single collective operation.

        This is the fundamental building block for all communication calculations.
        Strategies can compose this to build complex communication patterns.

        Args:
            ctx: Shard performance context
            output_size: Size of data to communicate in bytes
            collective_type: Type of collective (ALL_TO_ALL, REDUCE_SCATTER, ALL_GATHER, ALL_REDUCE)
            world_size: World size for the collective
            local_world_size: Local world size for bandwidth lookup

        Returns:
            Communication time in seconds
        """
        assert ctx.comms_bandwidths is not None

        comms_bw = ctx.comms_bandwidths.get_bw(
            world_size=world_size,
            local_world_size=local_world_size,
            collective_type=collective_type,
        )

        assert comms_bw > 0, f"Invalid comms bw: {comms_bw}"

        return output_size / comms_bw

    def _compute_batched_copy(
        self,
        ctx: ShardPerfContext,
        config: HardwarePerfConfig,
        output_size: float,
    ) -> float:
        """
        Compute batched copy time (used by ROW_WISE and TABLE_ROW_WISE backward).

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration
            output_size: Size of data to copy in bytes

        Returns:
            Batched copy time in seconds
        """
        device_bw = self._get_device_bw(ctx, config)
        assert device_bw > 0, f"Invalid device bw: {device_bw}"
        return output_size * BATCHED_COPY_PERF_FACTOR / device_bw

    def _compute_single_level_comms(
        self,
        ctx: ShardPerfContext,
        output_size: float,
        collective_type: CollectiveType,
    ) -> float:
        """
        Compute communication time for a single-level collective (world-wide).

        This is used by TABLE_WISE (A2A) and ROW_WISE (RS/AG) strategies.

        Args:
            ctx: Shard performance context
            output_size: Size of data to communicate in bytes
            collective_type: Type of collective

        Returns:
            Communication time in seconds
        """
        return self._compute_collective_comms(
            ctx=ctx,
            output_size=output_size,
            collective_type=collective_type,
            world_size=ctx.world_size,
            local_world_size=ctx.local_world_size,
        )

    def _compute_two_level_comms(
        self,
        ctx: ShardPerfContext,
        output_size: float,
        intra_collective: CollectiveType,
        inter_collective: CollectiveType,
    ) -> float:
        """
        Compute communication time for a two-level hierarchical collective.

        This is used by TABLE_ROW_WISE which has:
        - Intra-host collective (within a node)
        - Inter-host collective (across nodes)

        Args:
            ctx: Shard performance context
            output_size: Size of data to communicate in bytes
            intra_collective: Collective type for intra-host communication
            inter_collective: Collective type for inter-host communication

        Returns:
            Total communication time (intra + inter) in seconds
        """

        # Intra-host communication
        intra_comms = self._compute_collective_comms(
            ctx=ctx,
            output_size=output_size,
            collective_type=intra_collective,
            world_size=ctx.local_world_size,
            local_world_size=ctx.local_world_size,
        )

        # Inter-host communication
        inter_comms = self._compute_collective_comms(
            ctx=ctx,
            output_size=output_size,
            collective_type=inter_collective,
            world_size=ctx.num_hosts,
            local_world_size=1,
        )

        return intra_comms + inter_comms

    # =========================================================================
    # Bandwidth Getters
    # =========================================================================

    def _get_device_bw(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Get device bandwidth, preferring config annotation if set.

        Priority:
        1. config.device_bw (if annotated via @device_bw decorator)
        2. ctx.device_bw (computed from topology via kernel_bw_lookup)

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Device bandwidth in bytes/second
        """
        if config.device_bw is not None:
            return config.device_bw
        return ctx.device_bw

    def _get_hbm_to_ddr_mem_bw(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Get HBM-to-DDR bandwidth, preferring config annotation if set.

        Priority:
        1. config.hbm_to_ddr_mem_bw (if annotated via @hbm_to_ddr_mem_bw decorator)
        2. ctx.hbm_to_ddr_mem_bw (from topology)

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            HBM-to-DDR bandwidth in bytes/second
        """
        if config.hbm_to_ddr_mem_bw is not None:
            return config.hbm_to_ddr_mem_bw
        return ctx.hbm_to_ddr_mem_bw

    def _get_intra_host_bw(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Get intra-host bandwidth, preferring config annotation if set.

        Priority:
        1. config.intra_host_bw (if annotated via @intra_host_bw decorator)
        2. ctx.intra_host_bw (from topology)

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Intra-host bandwidth in bytes/second
        """
        if config.intra_host_bw is not None:
            return config.intra_host_bw
        return ctx.intra_host_bw

    def _get_inter_host_bw(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Get inter-host bandwidth, preferring config annotation if set.

        Priority:
        1. config.inter_host_bw (if annotated via @inter_host_bw decorator)
        2. ctx.inter_host_bw (from topology)

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Inter-host bandwidth in bytes/second
        """
        if config.inter_host_bw is not None:
            return config.inter_host_bw
        return ctx.inter_host_bw

    # =========================================================================
    # Custom method invocation helper
    # =========================================================================

    def execute_custom_fn(
        self,
        config: HardwarePerfConfig,
        method_name: str,
        custom_flag_attr: str,
        ctx: ShardPerfContext,
        check_sharding_type: bool = True,
    ) -> Optional[float]:
        """
        This is a generalized helper for checking and invoking custom methods
        decorated with annotations like @compute_fwd, @compute_bwd, @fwd_comms, etc.

        Args:
            config: Hardware performance configuration
            method_name: Name of the method to look for (e.g., "compute_fwd")
            custom_flag_attr: Attribute name that marks this as a custom method
                             (e.g., "_is_custom_forward_compute")
            ctx: Shard performance context
            check_sharding_type: Whether to check if method applies to current sharding type

        Returns:
            Result from custom method if it exists and applies, None otherwise
        """
        compute_method = getattr(config, method_name, None)
        if compute_method and getattr(compute_method, custom_flag_attr, False):
            if check_sharding_type:
                custom_sharding_type = getattr(
                    compute_method, "_custom_sharding_type", None
                )
                if custom_sharding_type is not None:
                    # Handle list/tuple of sharding types
                    if isinstance(custom_sharding_type, (list, tuple)):
                        if ctx.sharding_type.lower() not in [
                            st.lower() for st in custom_sharding_type
                        ]:
                            return None
                    elif custom_sharding_type.lower() != ctx.sharding_type.lower():
                        return None
            return compute_method(ctx)
        return None

    # =========================================================================
    # Batch inputs helpers
    # =========================================================================

    def get_batch_inputs(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Get batch inputs (number of indices to lookup per batch) adjusted for sharding.

        This is computed as: batch_size * pooling_factor / divisor
        where divisor depends on sharding type:
        - TABLE_WISE: 1 (each device handles all lookups for its tables)
        - ROW_WISE: world_size (lookups split across all devices)
        - TABLE_ROW_WISE: local_world_size (lookups split within host)

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration (unused, kept for API consistency)

        Returns:
            Effective batch inputs for this shard
        """
        divisor = self.get_batch_inputs_divisor(ctx)
        assert divisor > 0, f"Invalid divisor: {divisor}"
        return ctx.batch_inputs / divisor

    def get_batch_inputs_divisor(self, ctx: ShardPerfContext) -> int:
        """
        Get the divisor for batch_inputs based on sharding type.

        Override in subclasses that need dynamic divisors (e.g., ROW_WISE uses world_size).
        Returns:
            Divisor value
        """
        return 1

    # =========================================================================
    # Prefetch helpers
    # =========================================================================

    def get_prefetch_divisor(self, ctx: ShardPerfContext) -> int:
        """
        Get divisor for prefetch compute based on sharding type.

        Override in subclasses. Default is 1 (no division).

        Args:
            ctx: Shard performance context (unused in base class, but available for subclass overrides)
        """
        return 1

    # =========================================================================
    # Forward compute
    # =========================================================================

    def compute_fwd_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute forward pass time.

        This checks if the config has a custom forward_compute method (via @forward_compute
        decorator). If so, use it. Otherwise, use the default coefficient-based formula.

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Forward compute time in seconds
        """
        # Check for custom forward compute method via @compute_fwd annotation
        custom_result = self.execute_custom_fn(
            config,
            "compute_fwd",
            "_is_custom_forward_compute",
            ctx,
            check_sharding_type=True,
        )
        if custom_result is not None:
            return custom_result

        return self._default_fwd_comp(ctx, config)

    def _default_fwd_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default forward using coefficient-based formula.

        Formula:
            fwd_compute = (input_read_size * input_read_size_multiplier
                          + embedding_lookup_size * lookup_size_multiplier
                          + fwd_output_write_size * embedding_output_multiplier
                          + hash_size * hash_size_multiplier)
                          * block_usage_penalty / device_bw

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Forward compute time in seconds
        """
        device_bw = self._get_device_bw(ctx, config)

        # Get coefficients for this sharding type
        coefficients = config.get_coefficients_for_sharding(
            ctx.sharding_type, ctx.compute_kernel
        )
        fwd_coeff = coefficients.fwd
        # Compute block usage penalty based on strategy
        block_penalty = (
            compute_block_usage_penalty(ctx.emb_dim)
            if self.use_block_usage_penalty(config)
            else 1.0
        )

        # Compute raw sizes
        input_read_size = self._get_input_read_size(ctx=ctx, config=config)
        embedding_lookup_size = self._get_embedding_lookup_size(
            ctx=ctx, use_min_dim=self.use_min_dim_for_lookup(config)
        )

        # Check for custom output_write_size method from config (via @output_write_size decorator)
        custom_output_write_size_fn = get_output_write_size(config, ctx.sharding_type)
        if custom_output_write_size_fn:
            fwd_output_write_size = custom_output_write_size_fn(ctx, is_fwd=True)
        else:
            fwd_output_write_size = self._get_output_write_size(
                ctx, self._get_comm_data_type_size(ctx, is_fwd=True)
            )

        # Apply coefficients
        compute_size = (
            fwd_coeff.input_read_size_multiplier * input_read_size
            + fwd_coeff.lookup_size_multiplier * embedding_lookup_size
            + fwd_coeff.embedding_output_multiplier * fwd_output_write_size
            + fwd_coeff.hash_size_multiplier * ctx.hash_size
        )

        return compute_size * block_penalty / device_bw

    def use_min_dim_for_lookup(self, config: HardwarePerfConfig) -> bool:
        """
        Whether to use max(emb_dim, 32) for embedding lookup size.

        TABLE_WISE returns True for kernel efficiency. Other strategies return False.
        Can be overridden via @use_min_dim_for_lookup annotation on HardwarePerfConfig.
        """
        return getattr(config, "_use_min_dim_for_lookup", False)

    def use_block_usage_penalty(self, config: HardwarePerfConfig) -> bool:
        """
        Whether to apply block usage penalty to fwd_compute.

        TABLE_WISE returns True per OSS. Other strategies return False.
        Can be overridden via @use_block_usage_penalty annotation on HardwarePerfConfig.
        """
        return getattr(config, "_use_block_usage_penalty", False)

    # =========================================================================
    # Backward compute
    # =========================================================================
    def compute_bwd_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute backward pass compute time.

        This checks if the config has a custom backward_compute method (via @backward_compute
        decorator). If so, use it. Otherwise, use the default formula.

        Default formula:
            bwd_compute = fwd_compute * bwd_compute_multiplier
            if is_weighted: bwd_compute *= weighted_feature_bwd_compute_multiplier

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Backward compute time in seconds (includes bwd_grad_indice_weights_kernel)
        """
        # Check for custom backward compute method via @backward_compute annotation
        custom_result = self.execute_custom_fn(
            config,
            "compute_bwd",
            "_is_custom_backward_compute",
            ctx,
            check_sharding_type=True,
        )
        if custom_result is not None:
            return custom_result

        return self._default_bwd_comp(ctx, config)

    def _default_bwd_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default backward compute - supports both OSS and FB formulas.

        Two approaches based on coefficient configuration:
        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Backward compute time in seconds
        """
        coeffs = config.get_coefficients_for_sharding(
            ctx.sharding_type, ctx.compute_kernel
        )
        bwd_coeff = coeffs.bwd

        # Calculate sizes (same as fwd but may use different coefficients)
        # Use getter methods for sharding-specific data sizes

        input_read_size = self._get_input_read_size(ctx=ctx, config=config)
        embedding_lookup_size = self._get_embedding_lookup_size(ctx=ctx)

        # Check for custom output_write_size method from config (via @output_write_size decorator)
        custom_output_write_size_fn = get_output_write_size(config, ctx.sharding_type)
        if custom_output_write_size_fn:
            output_write_size = custom_output_write_size_fn(ctx, is_fwd=False)
        else:
            output_write_size = self._get_output_write_size(
                ctx, data_type_size=self._get_comm_data_type_size(ctx, is_fwd=False)
            )

        fwd_compute = self._default_fwd_comp(ctx=ctx, config=config)
        if config.name == "default":
            bwd_compute = fwd_compute * config.coefficients.bwd_compute_multiplier
        else:
            device_bw = self._get_device_bw(ctx=ctx, config=config)
            compute_size = (
                bwd_coeff.input_read_size_multiplier * input_read_size
                + bwd_coeff.lookup_size_multiplier * embedding_lookup_size
                + bwd_coeff.embedding_output_multiplier * output_write_size
                + bwd_coeff.hash_size_multiplier * ctx.hash_size
            )
            bwd_compute = compute_size / device_bw

        # Add bwd_grad_indice_weights_kernel for weighted features
        bwd_grad_indice_weights_kernel = self._compute_bwd_grad_indice_weights_kernel(
            fwd_compute, ctx, config
        )
        # Apply weighted feature multiplier if applicable
        if ctx.is_weighted:
            bwd_compute = (
                bwd_compute
                * config.coefficients.weighted_feature_bwd_compute_multiplier
            )

        return bwd_compute + bwd_grad_indice_weights_kernel

    def _get_embedding_lookup_size(
        self, ctx: ShardPerfContext, use_min_dim: bool = False
    ) -> float:
        """
        Get embedding lookup size based on sharding type.

        Args:
            ctx: Shard performance context
            use_min_dim: If True, use max(emb_dim, 32) for kernel alignment (TABLE_WISE)

        Returns:
            Embedding lookup size in bytes.
        """

        emb_dim = max(ctx.emb_dim, 32) if use_min_dim else ctx.emb_dim
        return ctx.batch_inputs * ctx.world_size * emb_dim * ctx.table_data_type_size

    def _get_output_write_size(
        self, ctx: ShardPerfContext, data_type_size: float
    ) -> float:
        """
        Get output write size for the given data type.

        Args:
            ctx: Shard performance context
            data_type_size: Data type size in bytes

        Returns:
            Output write size in bytes.
        """
        return ctx.batch_outputs * ctx.world_size * ctx.emb_dim * data_type_size

    def _get_comm_data_type_size(
        self, ctx: ShardPerfContext, is_fwd: bool = True
    ) -> float:
        """
        Get communication data type size for forward or backward pass.

        Default uses A2A comm data type. ROW_WISE/TABLE_ROW_WISE override
        to use SR comm data type.

        Args:
            ctx: Shard performance context
            is_fwd: If True, return forward comm data type; else backward

        Returns:
            Data type size in bytes.
        """
        if is_fwd:
            return ctx.fwd_a2a_comm_data_type_size
        return ctx.bwd_a2a_comm_data_type_size

    def _get_input_read_size(
        self, ctx: ShardPerfContext, config: Optional[HardwarePerfConfig] = None
    ) -> float:
        """
        Get input read size.

        Default includes world_size multiplier. DATA_PARALLEL overrides
        to NOT use world_size.

        The config can control whether to use bytes or counts via
        @use_bytes_for_input_read_size decorator:
        - True (default/OSS): size = batch_inputs * world_size * input_data_type_size
        - False (FB legacy): size = batch_inputs * world_size (count of indices)
        """
        use_bytes = (
            self.use_bytes_for_input_read_size(config)
            if config is not None
            else True  # Default to OSS behavior (bytes)
        )

        if use_bytes:
            size = math.ceil(
                ctx.batch_inputs * ctx.world_size * ctx.input_data_type_size
            )
        else:
            # FB legacy: use count of indices, not bytes
            size = math.ceil(ctx.batch_inputs * ctx.world_size)

        if ctx.is_weighted:
            size *= 2
        return size

    def use_bytes_for_input_read_size(self, config: HardwarePerfConfig) -> bool:
        """
        Whether to multiply input_read_size by input_data_type_size.

        Default is True (OSS behavior - calculate in bytes).
        Can be overridden via @use_bytes_for_input_read_size(False) annotation on HardwarePerfConfig.
        """
        return getattr(config, "_use_bytes_for_input_read_size", True)

    def _compute_bwd_grad_indice_weights_kernel(
        self, compute_value: float, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute backward gradient indice weights kernel time.

        For weighted features (e.g., id-score lists), there's an additional
        kernel that computes gradients for the indice weights.

        OSS Formula:
            bwd_grad_indice_weights_kernel = fwd_compute * WEIGHTED_KERNEL_MULTIPLIER
            (only if is_weighted is True)

        Args:
            fwd_compute: Forward compute time
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Backward gradient indice weights kernel time in seconds
        """

        if ctx.is_weighted:
            return compute_value * config.coefficients.weighted_kernel_multiplier
        return 0.0

    # =========================================================================
    # Prefetch compute
    # =========================================================================

    def compute_prefetch_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute prefetch/cache loading time.

        This checks if the config has a custom prefetch_compute method (via @prefetch_compute
        decorator). If so, use it. Otherwise, use the default formula.

        OSS Formula:
            prefetch_bytes = expected_cache_fetches * emb_dim * table_data_type_size
            prefetch_compute = prefetch_bytes / hbm_to_ddr_mem_bw

        For ROW_WISE: expected_cache_fetches /= world_size
        For TABLE_ROW_WISE: expected_cache_fetches /= local_world_size

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Prefetch compute time in seconds
        """
        # Check for custom prefetch compute method via @prefetch_compute annotation
        custom_result = self.execute_custom_fn(
            config,
            "compute_prefetch",
            "_is_custom_prefetch_compute",
            ctx,
            check_sharding_type=False,
        )
        if custom_result is not None:
            return custom_result
        # Default implementation
        return self._default_prefetch_comp(ctx, config)

    def _default_prefetch_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default prefetch compute with support for both OSS and FB hardware formulas.
        The prefetch_divisor varies by sharding type:
        - TABLE_WISE: 1 (no division)
        - ROW_WISE: world_size
        - TABLE_ROW_WISE: local_world_size

        For linear regression (FB hardware):
            prefetch_time = (
                expected_num_lookups_coefficient * expected_lookups +
                expected_num_unique_lookups_coefficient * expected_unique_lookups +
                expected_size_cache_fetches_coefficient * prefetch_bytes
            )
            where prefetch_bytes = expected_cache_fetches * emb_dim (NO table_data_type_size!)

        For default (OSS):
            prefetch_time = prefetch_bytes / hbm_to_ddr_mem_bw
            where prefetch_bytes = expected_cache_fetches * emb_dim * table_data_type_size
        """

        # Apply prefetch divisor based on sharding type
        prefetch_divisor = self.get_prefetch_divisor(ctx)

        # Get expected cache fetches with divisor applied
        expected_cache_fetches = ctx.expected_cache_fetches
        if prefetch_divisor > 0:
            expected_cache_fetches = expected_cache_fetches / prefetch_divisor

        # Check if linear regression mode is enabled (FB hardware estimators)
        # If prefetch coefficients are defined and lookup data is available, use linear regression
        prefetch_coeffs = config.coefficients.prefetch
        if (
            prefetch_coeffs
            and ctx.expected_lookups is not None
            and ctx.expected_unique_lookups is not None
        ):
            # Apply divisor to lookups as well
            expected_lookups = ctx.expected_lookups
            expected_unique_lookups = ctx.expected_unique_lookups
            if prefetch_divisor > 0:
                expected_lookups = expected_lookups / prefetch_divisor
                expected_unique_lookups = expected_unique_lookups / prefetch_divisor

            # IMPORTANT: For linear regression, prefetch_bytes does NOT include table_data_type_size
            # This matches the FB hardware estimator implementation (D89929552)
            prefetch_bytes = expected_cache_fetches * ctx.emb_dim

            return (
                prefetch_coeffs.expected_num_lookups_coefficient * expected_lookups
                + prefetch_coeffs.expected_num_unique_lookups_coefficient
                * expected_unique_lookups
                + prefetch_coeffs.expected_size_cache_fetches_coefficient
                * prefetch_bytes
            )
        else:
            # Default OSS formula: prefetch_bytes / hbm_to_ddr_mem_bw
            # For OSS, prefetch_bytes includes table_data_type_size
            prefetch_bytes = (
                expected_cache_fetches * ctx.emb_dim * ctx.table_data_type_size
            )
            hbm_to_ddr_bw = self._get_hbm_to_ddr_mem_bw(ctx, config)
            return prefetch_bytes / hbm_to_ddr_bw if hbm_to_ddr_bw > 0 else 0.0

    # =========================================================================
    # Communication methods
    # Check for annotations, then delegate to evaluator-specific implementation
    # =========================================================================

    def compute_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute forward communication time.

        Checks for @fwd_comms annotation on config first, then uses
        evaluator-specific implementation.

        Each sharding type has different communication patterns:
        - TABLE_WISE: All-to-all
        - ROW_WISE: Reduce-scatter
        - DATA_PARALLEL: No communication

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Forward communication time in seconds
        """
        # Check for custom method via @fwd_comms annotation
        # Pass sharding_type to allow sharding-type-specific custom methods
        custom_result = self.execute_custom_fn(
            config,
            "compute_fwd_comms",
            "_is_custom_fwd_comms",
            ctx,
            check_sharding_type=True,
        )
        if custom_result is not None:
            return custom_result

        return self._default_fwd_comms(ctx, config)

    @abstractmethod
    def _default_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default forward communication implementation.

        Must be implemented by evaluator subclasses.
        """
        pass

    def compute_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute backward communication time.

        Checks for @bwd_comms annotation on config first, then uses
        evaluator-specific implementation.

        Each sharding type has different communication patterns:
        - TABLE_WISE: All-to-all
        - ROW_WISE: All-gather + batched copy
        - DATA_PARALLEL: All-reduce

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Backward communication time in seconds
        """
        # Check for custom method via @bwd_comms annotation
        # Pass sharding_type to allow sharding-type-specific custom methods
        custom_result = self.execute_custom_fn(
            config,
            "compute_bwd_comms",
            "_is_custom_bwd_comms",
            ctx,
            check_sharding_type=True,
        )
        if custom_result is not None:
            return custom_result
        return self._default_bwd_comms(ctx, config)

    @abstractmethod
    def _default_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default backward communication implementation.

        Must be implemented by evaluator subclasses.
        """
        pass

    # =========================================================================
    # Input distribution communication
    # =========================================================================

    def compute_input_dist_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Compute input distribution communication time.

        This checks if the config has a custom input_dist_comms method (via @input_dist_comms
        decorator). If so, use it. Otherwise, use the default formula.

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Input distribution communication time in seconds
        """
        # Check for custom input dist comms method via @input_dist_comms annotation
        # Pass sharding_type to allow sharding-type-specific custom methods
        custom_result = self.execute_custom_fn(
            config,
            "compute_input_dist_comms",
            "_is_custom_input_dist_comms",
            ctx,
            check_sharding_type=True,
        )

        if custom_result is not None:
            return custom_result

        return self._default_input_dist_comms(ctx, config)

    def _default_input_dist_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        """
        Default input distribution communication formula.

        This is the same for TABLE_WISE, ROW_WISE, and TABLE_ROW_WISE sharding.
        Only DATA_PARALLEL overrides this to return 0.0.

        Formula from OSS _input_dist_expected_latency:
        - input_read_size = batch_inputs * world_size * input_data_type_size
        - Return input_read_size / comms_bw (using ALL_TO_ALL collective)

        Note: OSS has the is_weighted logic in the function but callers DON'T
        pass is_weighted, so it defaults to False. We match this behavior by
        NOT applying the weighted multiplier here.

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Input distribution communication time in seconds
        """

        # Calculate input_read_size WITHOUT the weighted multiplier
        # (matches OLD estimator _input_dist_expected_latency which is called
        # without is_weighted parameter, so it defaults to False)
        #
        # NOTE: input_dist_comms always uses BYTES since it's about actual data
        # transfer over the network. The use_bytes_for_input_read_size annotation
        # only affects compute formulas (fwd_compute, bwd_compute).
        input_read_size = math.ceil(
            ctx.batch_inputs * ctx.world_size * ctx.input_data_type_size
        )

        assert ctx.comms_bandwidths is not None

        comms_bw = ctx.comms_bandwidths.get_bw(
            world_size=ctx.world_size,
            local_world_size=ctx.local_world_size,
            collective_type=CollectiveType.ALL_TO_ALL,
        )
        assert comms_bw is not None and comms_bw > 0

        return input_read_size / comms_bw

    # =========================================================================
    # Main compute_perf method
    # =========================================================================

    def compute_perf(self, ctx: ShardPerfContext, config: HardwarePerfConfig) -> Perf:
        """
        Compute the full performance estimate for a shard.

        This is the main entry point that orchestrates all the compute and
        communication calculations.

        Args:
            ctx: Shard performance context
            config: Hardware performance configuration

        Returns:
            Perf object with fwd_compute, fwd_comms, bwd_compute, bwd_comms, prefetch_compute
        """

        fwd_compute = self.compute_fwd_comp(ctx=ctx, config=config)
        fwd_comms = self.compute_fwd_comms(ctx=ctx, config=config)
        bwd_compute = self.compute_bwd_comp(ctx=ctx, config=config)
        bwd_comms = self.compute_bwd_comms(ctx=ctx, config=config)
        prefetch_compute = self.compute_prefetch_comp(ctx=ctx, config=config)

        # OSS always computes input_dist_comms for applicable sharding types
        input_dist_comms = self.compute_input_dist_comms(ctx=ctx, config=config)

        return Perf(
            fwd_compute=fwd_compute,
            fwd_comms=fwd_comms,
            bwd_compute=bwd_compute,
            bwd_comms=bwd_comms,
            prefetch_compute=prefetch_compute,
            input_dist_comms=input_dist_comms,
        )


# =============================================================================
class TableWiseEvaluator(EmbeddingShardingPerfEvaluator):
    """
    Evaluator for TABLE_WISE sharding.

    Differences from other strategies:
    - No batch_inputs division (divisor = 1)
    - Uses max(emb_dim, 32) for embedding lookup (kernel efficiency)
    - Uses A2A comm data type
    - Uses block_usage_penalty
    - Forward/Backward: All-to-all
    - Prefetch divisor = 1
    """

    def use_min_dim_for_lookup(self, config: HardwarePerfConfig) -> bool:
        return getattr(config, "_use_min_dim_for_lookup", True)

    def use_block_usage_penalty(self, config: HardwarePerfConfig) -> bool:
        return getattr(config, "_use_block_usage_penalty", True)

    def _default_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        fwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=True)
        )
        return self._compute_single_level_comms(
            ctx=ctx,
            output_size=fwd_output_write_size,
            collective_type=CollectiveType.ALL_TO_ALL,
        )

    def _default_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        bwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=False)
        )
        return self._compute_single_level_comms(
            ctx=ctx,
            output_size=bwd_output_write_size,
            collective_type=CollectiveType.ALL_TO_ALL,
        )


class RowWiseEvaluator(EmbeddingShardingPerfEvaluator):
    """
    Evaluator for ROW_WISE sharding.

    Differences from base (TABLE_WISE):
    - batch_inputs divided by world_size
    - Uses SR comm data type (if pooled) instead of A2A
    - No block_usage_penalty
    - Forward: Reduce-scatter, Backward: All-gather + batched_copy
    - Prefetch divided by world_size
    """

    def get_batch_inputs_divisor(self, ctx: ShardPerfContext) -> int:
        return ctx.world_size

    def _get_input_read_size(
        self, ctx: ShardPerfContext, config: Optional[HardwarePerfConfig] = None
    ) -> float:
        """
        ROW_WISE: input_read_size does NOT multiply by world_size.

        In OLD estimator, batch_inputs is pre-divided by world_size, then multiplied back.
        The world_size factors cancel out, so input_read_size = raw * input_data_type_size.

        Note: ROW_WISE always uses bytes (input_data_type_size), ignoring the
        use_bytes_for_input_read_size config flag since legacy behavior was consistent.
        """
        size = math.ceil(ctx.batch_inputs * ctx.input_data_type_size)
        if ctx.is_weighted:
            size *= 2
        return size

    def _get_embedding_lookup_size(
        self, ctx: ShardPerfContext, use_min_dim: bool = False
    ) -> float:
        """
        ROW_WISE: embedding_lookup_size does NOT multiply by world_size.

        In OLD estimator, batch_inputs is pre-divided by world_size, then multiplied back.
        The world_size factors cancel out.
        """
        return ctx.batch_inputs * ctx.emb_dim * ctx.table_data_type_size

    def _get_output_write_size(
        self, ctx: ShardPerfContext, data_type_size: float
    ) -> float:
        """
        ROW_WISE: output_write_size calculation.

        For pooled: batch_outputs * world_size * emb_dim * data_type_size
        For non-pooled (sequence): batch_outputs * emb_dim * data_type_size (NO world_size)

        Explanation:
        In OLD estimator, batch_inputs is pre-divided by world_size for ROW_WISE:
          batch_inputs = raw / world_size
          batch_outputs = (pooled) sum(...) or (non-pooled) batch_inputs = raw / world_size
          output_write_size = batch_outputs * world_size * emb_dim * data_type_size

        For pooled: (sum(...)) * world_size * ... (world_size stays)
        For non-pooled: (raw / world_size) * world_size * ... = raw * ... (world_size cancels)

        In NEW estimator, batch_inputs/batch_outputs are RAW values, so we need to
        NOT multiply by world_size for non-pooled to match the cancellation behavior.
        """
        if ctx.is_pooled:
            return ctx.batch_outputs * ctx.world_size * ctx.emb_dim * data_type_size
        else:
            # For non-pooled (sequence), world_size factor cancels out in OLD estimator
            return ctx.batch_outputs * ctx.emb_dim * data_type_size

    def _get_comm_data_type_size(
        self, ctx: ShardPerfContext, is_fwd: bool = True
    ) -> float:
        if is_fwd:
            return (
                ctx.fwd_sr_comm_data_type_size
                if ctx.is_pooled
                else ctx.fwd_a2a_comm_data_type_size
            )
        return (
            ctx.bwd_sr_comm_data_type_size
            if ctx.is_pooled
            else ctx.bwd_a2a_comm_data_type_size
        )

    def _default_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        fwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=True)
        )
        return self._compute_single_level_comms(
            ctx=ctx,
            output_size=fwd_output_write_size,
            collective_type=CollectiveType.REDUCE_SCATTER,
        )

    def _default_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        # All-gather
        bwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=False)
        )
        bwd_comms = self._compute_single_level_comms(
            ctx=ctx,
            output_size=bwd_output_write_size,
            collective_type=CollectiveType.ALL_GATHER,
        )

        # Batched copy (per OSS formula)
        bwd_batched_copy = self._compute_batched_copy(
            ctx=ctx,
            config=config,
            output_size=bwd_output_write_size,
        )

        return bwd_comms + bwd_batched_copy

    def get_prefetch_divisor(self, ctx: ShardPerfContext) -> int:
        return ctx.world_size


class TableRowWiseEvaluator(EmbeddingShardingPerfEvaluator):
    """
    Evaluator for TABLE_ROW_WISE sharding.

    Differences from base (TABLE_WISE):
    - batch_inputs divided by local_world_size
    - Uses SR comm data type (always, unlike ROW_WISE which checks is_pooled)
    - No block_usage_penalty
    - Forward: Reduce-scatter (intra) + All-to-all (inter)
    - Backward: All-to-all (inter) + All-gather (intra) + batched_copy
    - Prefetch divided by local_world_size
    """

    def get_batch_inputs_divisor(self, ctx: ShardPerfContext) -> int:
        return ctx.local_world_size

    def _get_input_read_size(
        self, ctx: ShardPerfContext, config: Optional[HardwarePerfConfig] = None
    ) -> float:
        """
        TABLE_ROW_WISE: input_read_size = (raw / local_world_size) * world_size * input_data_type_size.

        This is different from ROW_WISE where world_size cancels out.

        Note: TABLE_ROW_WISE always uses bytes (input_data_type_size), ignoring the
        use_bytes_for_input_read_size config flag since legacy behavior was consistent.
        """
        effective_batch_inputs = ctx.batch_inputs / ctx.local_world_size
        size = math.ceil(
            effective_batch_inputs * ctx.world_size * ctx.input_data_type_size
        )
        if ctx.is_weighted:
            size *= 2
        return size

    def _get_embedding_lookup_size(
        self, ctx: ShardPerfContext, use_min_dim: bool = False
    ) -> float:
        """
        TABLE_ROW_WISE: embedding_lookup_size = (raw / local_world_size) * world_size * emb_dim * table_data_type_size.
        """
        effective_batch_inputs = ctx.batch_inputs / ctx.local_world_size
        return (
            effective_batch_inputs
            * ctx.world_size
            * ctx.emb_dim
            * ctx.table_data_type_size
        )

    def _get_output_write_size(
        self, ctx: ShardPerfContext, data_type_size: float
    ) -> float:
        """
        TABLE_ROW_WISE: output_write_size uses batch_outputs * world_size * emb_dim.

        This matches the OLD estimator formula exactly.
        """
        return ctx.batch_outputs * ctx.world_size * ctx.emb_dim * data_type_size

    def _get_comm_data_type_size(
        self, ctx: ShardPerfContext, is_fwd: bool = True
    ) -> float:
        return (
            ctx.fwd_sr_comm_data_type_size if is_fwd else ctx.bwd_sr_comm_data_type_size
        )

    def _default_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        if ctx.comms_bandwidths is None:
            return 0.0

        fwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=True)
        )

        # Intra-host: reduce-scatter within the host
        intra_comms = self._compute_collective_comms(
            ctx=ctx,
            output_size=fwd_output_write_size,
            collective_type=CollectiveType.REDUCE_SCATTER,
            world_size=ctx.local_world_size,
            local_world_size=ctx.local_world_size,
        )

        # Inter-host: all-to-all across hosts (only if num_hosts > 1)
        inter_comms = 0.0
        if ctx.num_hosts > 1:
            # Inter-host uses A2A data type and different output size
            inter_host_fwd_output_write_size = (
                ctx.batch_outputs
                * ctx.num_hosts
                * ctx.emb_dim
                * ctx.fwd_a2a_comm_data_type_size
            )
            inter_comms = self._compute_collective_comms(
                ctx=ctx,
                output_size=inter_host_fwd_output_write_size,
                collective_type=CollectiveType.ALL_TO_ALL,
                world_size=ctx.num_hosts,
                local_world_size=1,
            )

        return intra_comms + inter_comms

    def _default_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:

        bwd_output_write_size = self._get_output_write_size(
            ctx, self._get_comm_data_type_size(ctx, is_fwd=False)
        )

        # Inter-host: all-to-all across hosts (only if num_hosts > 1)
        inter_comms = 0.0
        if ctx.num_hosts > 1:

            # Inter-host uses A2A data type and different output size
            inter_host_bwd_output_write_size = (
                ctx.batch_outputs
                * ctx.num_hosts
                * ctx.emb_dim
                * ctx.bwd_a2a_comm_data_type_size
            )
            inter_comms = self._compute_collective_comms(
                ctx=ctx,
                output_size=inter_host_bwd_output_write_size,
                collective_type=CollectiveType.ALL_TO_ALL,
                world_size=ctx.num_hosts,
                local_world_size=1,
            )

        # Intra-host: all-gather within the host
        intra_comms = self._compute_collective_comms(
            ctx=ctx,
            output_size=bwd_output_write_size,
            collective_type=CollectiveType.ALL_GATHER,
            world_size=ctx.local_world_size,
            local_world_size=ctx.local_world_size,
        )

        # Batched copy (same formula as ROW_WISE)
        bwd_batched_copy = self._compute_batched_copy(
            ctx=ctx,
            config=config,
            output_size=bwd_output_write_size,
        )

        return inter_comms + intra_comms + bwd_batched_copy

    def get_prefetch_divisor(self, ctx: ShardPerfContext) -> int:
        """TABLE_ROW_WISE divides prefetch by local_world_size (per OSS)."""
        return ctx.local_world_size


class DataParallelEvaluator(EmbeddingShardingPerfEvaluator):
    """
    Evaluator for DATA_PARALLEL sharding.

    Differences from other strategies:
    - No batch_inputs multiplication by world_size (local batch only)
    - Forward: No communication (each device has full table)
    - Backward: All-reduce of gradients + optimizer kernels
    - No prefetch compute, no input_dist_comms
    """

    # =========================================================================
    # Override size calculations to NOT use world_size (local batch only)
    # This enables reuse of base class _default_fwd_comp() and _default_bwd_comp()
    # =========================================================================

    def _get_input_read_size(
        self, ctx: ShardPerfContext, config: Optional[HardwarePerfConfig] = None
    ) -> float:
        """DATA_PARALLEL: input_read_size without world_size (local batch only)."""
        size = math.ceil(ctx.batch_inputs * ctx.input_data_type_size)
        if ctx.is_weighted:
            size *= 2
        return size

    def _get_embedding_lookup_size(
        self, ctx: ShardPerfContext, use_min_dim: bool = False
    ) -> float:
        return ctx.batch_inputs * ctx.emb_dim * ctx.table_data_type_size

    def _get_output_write_size(
        self, ctx: ShardPerfContext, data_type_size: float
    ) -> float:
        return ctx.batch_outputs * ctx.emb_dim * ctx.table_data_type_size

    def _default_fwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        return 0

    def _default_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        # Table size (gradient size for all-reduce)
        table_size = ctx.hash_size * ctx.emb_dim * ctx.table_data_type_size

        # All-reduce: NCCL ring-reduce formula
        num_nodes = min(ctx.world_size / ctx.local_world_size, 2)
        comms_bw = (
            ctx.comms_bandwidths.get_bw(
                world_size=ctx.world_size,
                local_world_size=ctx.local_world_size,
                collective_type=CollectiveType.ALL_REDUCE,
            )
            if ctx.comms_bandwidths is not None
            else None
        )
        all_reduce = (
            table_size * (2 * num_nodes - 1) / num_nodes / comms_bw
            if comms_bw and comms_bw > 0
            else 0.0
        )

        # Inter-host communication constraint
        if ctx.world_size > 2 * ctx.local_world_size:
            all_reduce *= 2

        # Optimizer kernels (SGD + Fill + Binary)
        device_bw = self._get_device_bw(ctx, config)
        optimizer_kernels = (
            table_size * DP_ELEMENTWISE_KERNELS_PERF_FACTOR / device_bw
            if device_bw > 0
            else 0.0
        )

        return all_reduce + optimizer_kernels

    # =========================================================================
    # Prefetch and Input Dist - not applicable for DATA_PARALLEL
    # =========================================================================

    def compute_prefetch_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        return 0.0

    def compute_input_dist_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        return 0.0


class ColumnWiseEvaluator(TableWiseEvaluator):
    pass


class TableColumnWiseEvaluator(TableWiseEvaluator):

    pass


class GridShardEvaluator(TableRowWiseEvaluator):

    pass


# =============================================================================
# EmbeddingPerfEstimator Factory
# =============================================================================


class EmbeddingPerfEstimatorFactory:
    """
    Factory for creating hardware-specific EmbeddingPerfEstimator instances.

    This factory maintains a registry of hardware configurations and provides
    a simple interface to create estimators for different hardware types.

    Usage:
        # Register a hardware config
        @EmbeddingPerfEstimatorFactory.register("my_hardware")
        class MyHardwarePerfConfig(HardwarePerfConfig):
            ...

        # Create an estimator
        estimator = EmbeddingPerfEstimatorFactory.create(
            "my_hardware",
            is_inference=False,
        )
    """

    _registry: Dict[str, Type[HardwarePerfConfig]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Type[HardwarePerfConfig]], Type[HardwarePerfConfig]]:
        """
        Decorator to register a hardware config class.

        Args:
            name: Name to register the config under (e.g., "grand_teton", "athena")

        Returns:
            Decorator function that registers the class

        Example:
            @EmbeddingPerfEstimatorFactory.register("grand_teton")
            class GrandTetonHardwarePerfConfig(HardwarePerfConfig):
                ...
        """

        def decorator(
            config_cls: Type[HardwarePerfConfig],
        ) -> Type[HardwarePerfConfig]:
            cls._registry[name.lower()] = config_cls
            return config_cls

        return decorator

    @classmethod
    def create(
        cls,
        hardware_name: str,
        is_inference: bool = False,
        topology: Optional[Topology] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        use_batch_inputs_for_expected_cache_fetches: bool = False,
        use_linear_regression_prefetch_estimate: bool = False,
    ) -> "EmbeddingPerfEstimatorV2":
        """
        Create an EmbeddingPerfEstimatorV2 for the specified hardware.

        Args:
            hardware_name: Name of the registered hardware config
            is_inference: Whether this is for inference
            topology: Device topology with bandwidth and world size info
            constraints: Optional parameter constraints
            use_batch_inputs_for_expected_cache_fetches: If True, expected_cache_fetches
                is computed as expected_miss_rate * batch_inputs (total lookups per batch).
                If False (default), uses expected_miss_rate * expected_unique_lookups.
            use_linear_regression_prefetch_estimate: If True, enables linear regression
                based prefetch time estimation using hardware-specific coefficients.

        Returns:
            EmbeddingPerfEstimatorV2 instance configured for the hardware

        Raises:
            ValueError: If hardware_name is not registered
        """
        name_lower = hardware_name.lower()
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown hardware: '{hardware_name}'. "
                f"Available: {available}. "
                f"Use 'default' for OSS defaults."
            )

        config_cls = cls._registry[name_lower]
        config = config_cls()
        logger.info(
            f" EmbeddingPerfEstimatorFactory is creating the Perf Estimator for {hardware_name} "
        )
        if topology is None:
            raise ValueError("topology is required to create EmbeddingPerfEstimatorV2")
        return EmbeddingPerfEstimatorV2(
            topology=topology,
            constraints=constraints,
            is_inference=is_inference,
            config=config,
            use_batch_inputs_for_expected_cache_fetches=use_batch_inputs_for_expected_cache_fetches,
            use_linear_regression_prefetch_estimate=use_linear_regression_prefetch_estimate,
        )

    @classmethod
    def create_with_config(
        cls,
        config: HardwarePerfConfig,
        is_inference: bool = False,
        topology: Optional[Topology] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        use_batch_inputs_for_expected_cache_fetches: bool = False,
        use_linear_regression_prefetch_estimate: bool = False,
    ) -> "EmbeddingPerfEstimatorV2":
        """
        Create an EmbeddingPerfEstimatorV2 with a specific config instance.

        This is useful when you need to customize a config at runtime.

        Args:
            config: HardwarePerfConfig instance
            is_inference: Whether this is for inference
            topology: Device topology with bandwidth and world size info
            constraints: Optional parameter constraints
            use_batch_inputs_for_expected_cache_fetches: If True, expected_cache_fetches
                is computed as expected_miss_rate * batch_inputs (total lookups per batch).
                If False (default), uses expected_miss_rate * expected_unique_lookups.
            use_linear_regression_prefetch_estimate: If True, enables linear regression
                based prefetch time estimation using hardware-specific coefficients.

        Returns:
            EmbeddingPerfEstimatorV2 instance
        """
        if topology is None:
            raise ValueError("topology is required to create EmbeddingPerfEstimatorV2")
        return EmbeddingPerfEstimatorV2(
            topology=topology,
            constraints=constraints,
            is_inference=is_inference,
            config=config,
            use_batch_inputs_for_expected_cache_fetches=use_batch_inputs_for_expected_cache_fetches,
            use_linear_regression_prefetch_estimate=use_linear_regression_prefetch_estimate,
        )

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all registered hardware names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a hardware name is registered."""
        return name.lower() in cls._registry


# =============================================================================
#  InferencePerfEvaluator
# =============================================================================


class InferenceShardingPerfEvaluator(EmbeddingShardingPerfEvaluator):
    """
    Inference sharding performance evaluator.

    Implement common inference behaviour
        - No backward compute
        - No backward comms
        - Inherits forward compute/comms from traning evaluator
    """

    def compute_bwd_comp(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        return 0.0

    def compute_bwd_comms(
        self, ctx: ShardPerfContext, config: HardwarePerfConfig
    ) -> float:
        return 0.0


class TableWiseInferenceEvaluator(InferenceShardingPerfEvaluator, TableWiseEvaluator):
    # TODO inference perf logic for TW should go here in future, if need more customization
    pass


class RowWiseInferenceEvaluator(InferenceShardingPerfEvaluator, RowWiseEvaluator):
    pass


class TableRowWiseInferenceEvaluator(
    InferenceShardingPerfEvaluator, TableRowWiseEvaluator
):
    pass


class ColumnWiseInferenceEvaluator(InferenceShardingPerfEvaluator, ColumnWiseEvaluator):
    pass


class DataParallelInferenceEvaluator(
    InferenceShardingPerfEvaluator, DataParallelEvaluator
):
    pass


class TableColumnWiseInferenceEvaluator(
    InferenceShardingPerfEvaluator, TableColumnWiseEvaluator
):
    pass


class GridShardInferenceEvaluator(InferenceShardingPerfEvaluator, GridShardEvaluator):
    pass


def compute_block_usage_penalty(embedding_dim: int) -> float:
    """
    Compute block usage penalty based on embedding dimension.
    Args:
        embedding_dim: Embedding dimension

    Returns:
        Penalty multiplier (1.0 = no penalty)
    """
    if embedding_dim < FULL_BLOCK_EMB_DIM:
        if embedding_dim >= 64:
            return HALF_BLOCK_PENALTY
        else:
            return QUARTER_BLOCK_PENALTY
    return 1.0


TRAINING_EVALUATORS: Dict[str, EmbeddingShardingPerfEvaluator] = {
    ShardingType.TABLE_WISE.value: TableWiseEvaluator(),
    ShardingType.ROW_WISE.value: RowWiseEvaluator(),
    ShardingType.TABLE_ROW_WISE.value: TableRowWiseEvaluator(),
    ShardingType.COLUMN_WISE.value: ColumnWiseEvaluator(),
    ShardingType.DATA_PARALLEL.value: DataParallelEvaluator(),
    ShardingType.TABLE_COLUMN_WISE.value: TableColumnWiseEvaluator(),
    ShardingType.GRID_SHARD.value: GridShardEvaluator(),
}

INFERENCE_EVALUATORS: Dict[str, EmbeddingShardingPerfEvaluator] = {
    ShardingType.TABLE_WISE.value: TableWiseInferenceEvaluator(),
    ShardingType.ROW_WISE.value: RowWiseInferenceEvaluator(),
    ShardingType.TABLE_ROW_WISE.value: TableRowWiseInferenceEvaluator(),
    ShardingType.COLUMN_WISE.value: ColumnWiseInferenceEvaluator(),
    ShardingType.DATA_PARALLEL.value: DataParallelInferenceEvaluator(),
    ShardingType.TABLE_COLUMN_WISE.value: TableColumnWiseInferenceEvaluator(),
    ShardingType.GRID_SHARD.value: GridShardInferenceEvaluator(),
}


def get_embedding_perf_sharding_evaluator(
    sharding_type: str, is_inference: bool = False
) -> EmbeddingShardingPerfEvaluator:
    """Get the appropriate evaluator for a sharding type.

    Args:
        sharding_type: The sharding type (e.g., "table_wise", "row_wise")
        is_inference: Whether this is for inference mode

    Returns:
        The appropriate evaluator instance
    """
    evaluators = INFERENCE_EVALUATORS if is_inference else TRAINING_EVALUATORS
    return evaluators.get(sharding_type, TableWiseEvaluator())


# =============================================================================
#  EmbeddingPerfEstimatorV2
# =============================================================================


class EmbeddingPerfEstimatorV2(ShardEstimator):  # TODO rename this later
    """
    Embedding Performance Estimator using HardwarePerfConfig.
    This estimator uses :
    - ShardPerfContext: Computed sizes and performance parameters
    - HardwarePerfConfig: Hardware-specific coefficients and compute methods
    - EmbeddingShardingPerfEvaluator: Sharding-type-specific communication patterns

    Implements ShardEstimator interface for drop-in replacement of EmbeddingPerfEstimator.

    Args:
        topology: Device topology with bandwidth and world size info
        constraints: Optional parameter constraints
        is_inference: Whether this is for inference
        config: Hardware-specific performance config
        use_batch_inputs_for_expected_cache_fetches: If True, expected_cache_fetches
            is computed as expected_miss_rate * batch_inputs (total lookups per batch).
            If False (default), uses expected_miss_rate * expected_unique_lookups
            (from CacheStatistics).
        use_linear_regression_prefetch_estimate: If True, enables linear regression
            based prefetch time estimation using hardware-specific coefficients.
            Also clamps num_unique_lookups to min(num_unique_lookups, batch_inputs, hash_size).

    Example:
        # Using factory (recommended)
        estimator = EmbeddingPerfEstimatorFactory.create(
            "grand_teton",
            topology=topology,
        )

        # Using directly with config
        config = GrandTetonHardwarePerfConfig()
        estimator = EmbeddingPerfEstimatorV2(
            topology=topology,
            config=config,
        )

        # Drop-in replacement for EmbeddingPerfEstimator
        estimator = EmbeddingPerfEstimatorFactory.create_default(
            topology=topology,
            constraints=constraints,
        )
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        is_inference: bool = False,
        config: Optional[HardwarePerfConfig] = None,
        use_batch_inputs_for_expected_cache_fetches: bool = False,
        use_linear_regression_prefetch_estimate: bool = False,
    ) -> None:
        # Note: We don't call super().__init__() because ShardEstimator is an ABC
        # with abstract __init__ that defines the interface signature
        self._topology = topology
        self._constraints = constraints
        self._is_inference = is_inference
        self._config = config if config is not None else HardwarePerfConfig()
        self._use_batch_inputs_for_expected_cache_fetches = (
            use_batch_inputs_for_expected_cache_fetches
        )
        self._use_linear_regression_prefetch_estimate = (
            use_linear_regression_prefetch_estimate
        )
        if self._use_linear_regression_prefetch_estimate:
            logger.info("use_linear_regression_prefetch_estimate is enabled.")

    @property
    def config(self) -> HardwarePerfConfig:
        return self._config

    @property
    def is_inference(self) -> bool:
        return self._is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        """
        Estimates the wall time of given sharding options
        Args:
            sharding_options: List of sharding options to estimate
            sharder_map: Map of sharder names to sharder instances
        """

        if not sharder_map:
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return

        assert self._topology is not None, "Topology must be set to use estimate method"

        num_feature_processors = 0
        for sharding_option in sharding_options:
            # Validate sharding type is supported by this config
            # (raises ValueError if not supported)
            self._config.validate_sharding_type(sharding_option.sharding_type)

            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]

            shard_sizes = [shard.size for shard in sharding_option.shards]

            # Build all contexts at once using shard_sizes list
            # This follows OSS pattern: extract common params ONCE per sharding_option
            contexts = ShardPerfContext.build_shard_perf_contexts(
                config=self._config,
                shard_sizes=shard_sizes,
                sharding_option=sharding_option,
                topology=self._topology,
                constraints=self._constraints,
                sharder=sharder,
                is_inference=self._is_inference,
                use_batch_inputs_for_expected_cache_fetches=self._use_batch_inputs_for_expected_cache_fetches,
                use_linear_regression_prefetch_estimate=self._use_linear_regression_prefetch_estimate,
            )

            # Update is_weighted from first context (common across all shards)
            if contexts:
                sharding_option.is_weighted = contexts[0].is_weighted
                # Track feature processors
                if contexts[0].has_feature_processor:
                    num_feature_processors += 1

            # Compute perf for each context and assign to corresponding shard
            for shard, ctx in zip(sharding_option.shards, contexts):
                evaluator = get_embedding_perf_sharding_evaluator(
                    sharding_type=ctx.sharding_type, is_inference=ctx.is_inference
                )
                shard.perf = evaluator.compute_perf(
                    ctx=ctx,
                    config=self._config,
                )

            # Post-process perfs (e.g., uneven sharding adjustment)
            # This allows configs to apply cross-shard adjustments
            shard_perfs = [
                shard.perf for shard in sharding_option.shards if shard.perf is not None
            ]
            shard_perfs = self._config.post_process_perfs(
                shard_perfs=shard_perfs,
                shard_sizes=shard_sizes,
                sharding_type=sharding_option.sharding_type,
                uneven_sharding_perf_multiplier=self._topology.uneven_sharding_perf_multiplier,
            )
            for shard, perf in zip(sharding_option.shards, shard_perfs):
                shard.perf = perf

        logger.info(f"Total {num_feature_processors} feature processor.")
