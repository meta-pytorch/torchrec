#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Comprehensive benchmarks for planner enumerator to analyze performance and scaling behavior.

This module provides benchmarks for the EmbeddingEnumerator component, including:
- Performance with varying table counts
- Performance with varying world sizes
- Memory usage tracking
"""

import argparse
import gc
import logging
import resource
import time
from typing import Dict, List, Tuple, Type

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

# Configure logging to ensure visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
# Force the logger to use the configured level
logger.setLevel(logging.INFO)


class TWSharder(EmbeddingBagCollectionSharder):
    """
    Table-wise sharder for benchmarking.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        # compute_device_type is required by the interface
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        # sharding_type and compute_device_type are required by the interface
        return [EmbeddingComputeKernel.DENSE.value]


class RWSharder(EmbeddingBagCollectionSharder):
    """
    Row-wise sharder for benchmarking.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        # compute_device_type is required by the interface
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        # sharding_type and compute_device_type are required by the interface
        return [EmbeddingComputeKernel.DENSE.value]


class CWSharder(EmbeddingBagCollectionSharder):
    """
    Column-wise sharder for benchmarking.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        # compute_device_type is required by the interface
        return [ShardingType.COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        # sharding_type and compute_device_type are required by the interface
        return [EmbeddingComputeKernel.DENSE.value]


def build_model_and_enumerator(
    world_size: int,
    num_tables: int,
    embedding_dim: int = 128,
    local_world_size: int = 8,
    compute_device: str = "cuda",
) -> Tuple[EmbeddingEnumerator, nn.Module]:
    """
    Build an enumerator and model for benchmarking.

    Args:
        world_size: Number of devices in the topology
        num_tables: Number of embedding tables in the model
        embedding_dim: Dimension of each embedding vector
        local_world_size: Number of devices per node
        compute_device: Device type ("cuda" or "cpu")

    Returns:
        Tuple of (enumerator, model)
    """
    topology = Topology(
        world_size=world_size,
        local_world_size=local_world_size,
        compute_device=compute_device,
    )
    tables = [
        EmbeddingBagConfig(
            num_embeddings=100 + i,
            embedding_dim=embedding_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_tables)
    ]
    model = TestSparseNN(tables=tables, weighted_tables=[])
    enumerator = EmbeddingEnumerator(topology=topology, batch_size=BATCH_SIZE)
    return enumerator, model


def measure_memory_and_time(
    world_size: int,
    num_tables: int,
    embedding_dim: int = 128,
    # pyrefly: ignore[bad-function-definition]
    sharder_class: Type[EmbeddingBagCollectionSharder] = TWSharder,
) -> Dict[str, float]:
    """
    Measure both time and memory usage for the enumerate operation.

    Args:
        world_size: Number of devices in the topology
        num_tables: Number of embedding tables in the model
        embedding_dim: Dimension of each embedding vector
        sharder_class: The sharder class to use

    Returns:
        Dictionary with time and memory metrics
    """
    # Force garbage collection before measurement
    gc.collect()

    # Build model and enumerator
    enumerator, model = build_model_and_enumerator(
        world_size=world_size,
        num_tables=num_tables,
        embedding_dim=embedding_dim,
    )

    # Force garbage collection again after model building
    gc.collect()

    # Get initial memory usage
    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Measure time
    start_time = time.time()
    # pyrefly: ignore[bad-argument-type]
    sharding_options = enumerator.enumerate(module=model, sharders=[sharder_class()])
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Get peak memory usage
    peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Calculate memory used during operation
    memory_used = peak_memory - initial_memory

    # Convert to MB (note: ru_maxrss is in KB on Linux, bytes on macOS)
    # We'll assume Linux here, so divide by 1024 to get MB
    peak_mb = memory_used / 1024

    # Verify the result
    assert len(sharding_options) == num_tables, "Unexpected number of sharding options"

    # Convert time to milliseconds
    elapsed_time_ms = elapsed_time * 1000

    return {
        "time_ms": elapsed_time_ms,
        "memory_mb": peak_mb,
        "options_count": len(sharding_options),
    }


def benchmark_enumerator_comprehensive(
    # pyrefly: ignore[bad-function-definition]
    sharder_class: Type[EmbeddingBagCollectionSharder] = TWSharder,
) -> None:
    """
    Comprehensive benchmark testing all combinations of world sizes and table counts.
    Tests world sizes from 16 to 2048 and table counts from 200 to 6400.

    Args:
        sharder_class: The sharder class to use for benchmarking
    """
    # Define the ranges for world sizes and table counts
    world_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    table_counts = [200, 400, 800, 1600, 3200, 6400]
    # Create a matrix to store results
    results = {}

    sharder_name = sharder_class.__name__
    logger.info(f"Running comprehensive enumerator benchmark with {sharder_name}...")
    logger.info(
        f"Testing {len(world_sizes)} world sizes × {len(table_counts)} table counts = {len(world_sizes) * len(table_counts)} combinations"
    )

    # Track progress
    total_combinations = len(world_sizes) * len(table_counts)
    completed = 0

    # Run benchmarks for all combinations
    for world_size in world_sizes:
        logger.info(f"Starting benchmarks for world_size={world_size}...")
        results[world_size] = {}
        world_size_start_time = time.time()

        # Run all table counts for this world size
        for num_tables in table_counts:
            try:
                metrics = measure_memory_and_time(
                    world_size=world_size,
                    num_tables=num_tables,
                    sharder_class=sharder_class,
                )
                results[world_size][num_tables] = metrics
            except Exception as e:
                results[world_size][num_tables] = {
                    "time_ms": -1,
                    "memory_mb": -1,
                    "options_count": -1,
                    "error": str(e),
                }

            completed += 1

        # Log completion of all table counts for this world size
        world_size_elapsed = time.time() - world_size_start_time
        logger.info(
            f"Completed world_size={world_size} ({len(table_counts)} table counts) "
            f"in {world_size_elapsed:.2f}s ({completed}/{total_combinations} combinations done)"
        )

        # Print intermediate results for this world size
        logger.info(f"Results for world_size={world_size}:")
        logger.info(f"{'Table Count':<12} {'Time (ms)':<10} {'Memory (MB)':<12}")
        logger.info("-" * 35)
        for num_tables in table_counts:
            if results[world_size][num_tables].get("error"):
                logger.info(f"{num_tables:<12} {'ERROR':<10} {'ERROR':<12}")
            else:
                logger.info(
                    f"{num_tables:<12} "
                    f"{results[world_size][num_tables]['time_ms']:<10.2f} "
                    f"{results[world_size][num_tables]['memory_mb']:<12.2f}"
                )

    # Print summary table after all tests are complete
    logger.info(f"\nComprehensive Enumerator Benchmark with {sharder_name} - Results:")

    # Print header row with table counts
    header = "World Size"
    for num_tables in table_counts:
        header += f" | {num_tables:>8}"
    logger.info(header)
    logger.info("-" * len(header))

    # Print time results
    logger.info("\nTime (milliseconds):")
    for world_size in world_sizes:
        row = f"{world_size:>10}"
        for num_tables in table_counts:
            if results[world_size][num_tables].get("error"):
                row += f" | {'ERROR':>8}"
            else:
                row += f" | {results[world_size][num_tables]['time_ms']:>8.2f}"
        logger.info(row)

    # Print memory results
    logger.info("\nMemory (MB):")
    for world_size in world_sizes:
        row = f"{world_size:>10}"
        for num_tables in table_counts:
            if results[world_size][num_tables].get("error"):
                row += f" | {'ERROR':>8}"
            else:
                row += f" | {results[world_size][num_tables]['memory_mb']:>8.2f}"
        logger.info(row)


def main() -> None:
    """
    Main entry point for the benchmark script.

    Provides a command-line interface to run specific benchmarks.
    """
    parser = argparse.ArgumentParser(description="Run planner enumerator benchmarks")
    parser.add_argument(
        "--sharder",
        type=str,
        choices=["tw", "rw", "cw", "all"],
        default="tw",
        help="Sharder type to use: table-wise (tw), row-wise (rw), column-wise (cw), or all",
    )
    logger.warning("Running planner enumerator benchmarks...")

    args = parser.parse_args()

    # Run benchmark with specified sharder(s)
    if args.sharder == "tw" or args.sharder == "all":
        # pyrefly: ignore[bad-argument-type]
        benchmark_enumerator_comprehensive(TWSharder)

    if args.sharder == "rw" or args.sharder == "all":
        # pyrefly: ignore[bad-argument-type]
        benchmark_enumerator_comprehensive(RWSharder)

    if args.sharder == "cw" or args.sharder == "all":
        # pyrefly: ignore[bad-argument-type]
        benchmark_enumerator_comprehensive(CWSharder)


if __name__ == "__main__":
    main()
