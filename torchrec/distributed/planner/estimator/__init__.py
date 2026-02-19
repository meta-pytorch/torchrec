#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""TorchRec Performance Estimator Module.

This module provides a refactored, extensible architecture for embedding
performance estimation with hardware-specific configurations.

Key components:
    - HardwarePerfConfig: Base class for hardware-specific configurations.
    - EmbeddingPerfEstimatorFactory: Factory for creating estimators.
    - EmbeddingPerfEstimatorV2: Refactored estimator implementation.
"""
from torchrec.distributed.planner.estimator.annotations import (
    backward_compute as backward_compute,  # noqa: F401
    bwd_coefficient as bwd_coefficient,  # noqa: F401
    device_bw as device_bw,  # noqa: F401
    forward_compute as forward_compute,  # noqa: F401
    fwd_coefficient as fwd_coefficient,  # noqa: F401
    output_write_size as output_write_size,  # noqa: F401
    prefetch_coefficient as prefetch_coefficient,  # noqa: F401
    ssd_mem_bw as ssd_mem_bw,  # noqa: F401
    supported_sharding_types as supported_sharding_types,  # noqa: F401
)
from torchrec.distributed.planner.estimator.config import (
    EmbeddingPerfEstimatorConfig as EmbeddingPerfEstimatorConfig,  # noqa: F401
)
from torchrec.distributed.planner.estimator.estimator import (
    EmbeddingPerfEstimatorFactory as EmbeddingPerfEstimatorFactory,  # noqa: F401
    EmbeddingPerfEstimatorV2 as EmbeddingPerfEstimatorV2,  # noqa: F401
)
from torchrec.distributed.planner.estimator.types import (
    EstimatorPerfCoefficients as EstimatorPerfCoefficients,  # noqa: F401
    HardwarePerfConfig as HardwarePerfConfig,  # noqa: F401
    PerfCoefficient as PerfCoefficient,  # noqa: F401
    PerfCoefficientConfig as PerfCoefficientConfig,  # noqa: F401
    PrefetchCoefficients as PrefetchCoefficients,  # noqa: F401
    ShardPerfContext as ShardPerfContext,  # noqa: F401
)
