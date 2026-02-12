#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Hardware Performance Configuration.

This module contains:
- EmbeddingPerfEstimatorConfig: Default  configuration with coefficients
For custom hardware configurations, extend EmbeddingPerfEstimatorConfig or
HardwarePerfConfig and use decorators from annotations.py.
"""

from torchrec.distributed.planner.constants import (
    BWD_COMPUTE_MULTIPLIER,
    DEFAULT_PERF_ESTIMATOR,
    WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,
    WEIGHTED_KERNEL_MULTIPLIER,
)

# Import factory from estimator module to avoid circular import
# Note: This import must be after types imports
from torchrec.distributed.planner.estimator.estimator import (
    EmbeddingPerfEstimatorFactory,
)
from torchrec.distributed.planner.estimator.types import (  # noqa: F401
    EstimatorPerfCoefficients,
    HardwarePerfConfig,
    PerfCoefficient,
    PerfCoefficientConfig,
)


# =============================================================================
# Default  Configuration
# =============================================================================

# Shared coefficients for sharding types with block usage penalty (TABLE_WISE, COLUMN_WISE)
# bwd=None means use fwd_compute * bwd_compute_multiplier (OSS legacy behavior)
_TABLE_WISE_COEFFICIENTS = EstimatorPerfCoefficients(
    fwd=PerfCoefficient(
        input_read_size_multiplier=1.0,
        lookup_size_multiplier=1.0,
        embedding_output_multiplier=1.0,
        hash_size_multiplier=0.0,  #  doesn't use hash_size
    ),
    bwd=None,  # Use fwd_compute * bwd_compute_multiplier for backward compatibility
)

# Shared coefficients for sharding types without block usage penalty (ROW_WISE, TABLE_ROW_WISE)
# bwd=None means use fwd_compute * bwd_compute_multiplier (OSS legacy behavior)
_ROW_WISE_COEFFICIENTS = EstimatorPerfCoefficients(
    fwd=PerfCoefficient(
        input_read_size_multiplier=1.0,
        lookup_size_multiplier=1.0,
        embedding_output_multiplier=1.0,
        hash_size_multiplier=0.0,
    ),
    bwd=None,  # Use fwd_compute * bwd_compute_multiplier for backward compatibility
)


@EmbeddingPerfEstimatorFactory.register(DEFAULT_PERF_ESTIMATOR)
class EmbeddingPerfEstimatorConfig(HardwarePerfConfig):
    """
    Default  EmbeddingPerfEstimator configuration.

    This configuration matches   EmbeddingPerfEstimator formulas:
    - All coefficients are 1.0 (no hardware-specific tuning)
    - Block usage penalty is enabled for TABLE_WISE/COLUMN_WISE sharding
    - Uses topology bandwidth values (HBM, DDR, etc.) from base class

    Block usage penalty ( TABLE_WISE only):
        - emb_dim >= 128: 1.0 (no penalty)
        - emb_dim >= 64:  1.15 (HALF_BLOCK_PENALTY)
        - emb_dim >= 32:  1.75 (QUARTER_BLOCK_PENALTY)

    Usage:
        # Use directly
        config = EmbeddingPerfEstimatorConfig()

        # Or extend for custom hardware with bandwidth override
        @hbm_mem_bw(2400 * 1024 * 1024 * 1024 / 1000)
        class MyHardwareConfig(EmbeddingPerfEstimatorConfig):
            pass
    """

    # =========================================================================
    # Default  Coefficients
    # Bandwidth values are inherited from HardwarePerfConfig base class
    # =========================================================================
    coefficients: PerfCoefficientConfig = PerfCoefficientConfig(
        # TABLE_WISE sharding (includes COLUMN_WISE, TABLE_COLUMN_WISE)
        # uses block_usage_penalty for these sharding types
        table_wise=_TABLE_WISE_COEFFICIENTS,
        column_wise=_TABLE_WISE_COEFFICIENTS,  # Same as TABLE_WISE
        # ROW_WISE sharding
        # does NOT use block_usage_penalty for ROW_WISE
        row_wise=_ROW_WISE_COEFFICIENTS,
        # TABLE_ROW_WISE sharding (includes GRID_SHARD)
        # does NOT use block_usage_penalty for TABLE_ROW_WISE
        table_row_wise=_ROW_WISE_COEFFICIENTS,  # Same as ROW_WISE
        # DATA_PARALLEL sharding
        # does NOT use block_usage_penalty for DATA_PARALLEL
        data_parallel=_ROW_WISE_COEFFICIENTS,  # Same as ROW_WISE
        # Compute multipliers
        bwd_compute_multiplier=BWD_COMPUTE_MULTIPLIER,  # 2.0
        weighted_kernel_multiplier=WEIGHTED_KERNEL_MULTIPLIER,  # 1.1
        weighted_feature_bwd_compute_multiplier=WEIGHTED_FEATURE_BWD_COMPUTE_MULTIPLIER,  # 1.0
    )
