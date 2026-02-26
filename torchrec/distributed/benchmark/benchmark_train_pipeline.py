#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- --world_size=2 --pipeline=sparse --batch_size=10

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_train_pipeline --world_size=4 --pipeline=sparse --batch_size=10

To support a new model in pipeline benchmark:
    See benchmark_pipeline_utils.py for step-by-step instructions.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger: logging.Logger = logging.getLogger(__name__)

import torch
from torch import nn
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
    CPUMemoryStats,
    GPUMemoryStats,
)
from torchrec.distributed.test_utils.input_config import ModelInputConfig
from torchrec.distributed.test_utils.model_config import (
    BaseModelConfig,
    ModelSelectionConfig,
)
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.pipeline_config import PipelineConfig
from torchrec.distributed.test_utils.sharding_config import (
    PlannerConfig,
    ShardingConfig,
)
from torchrec.distributed.test_utils.table_config import (
    EmbeddingTablesConfig,
    TableExtendedConfigs,
)
from torchrec.distributed.train_pipeline import (
    GradientAccumulationConfig,
    GradientAccumulationWrapper,
    TrainPipeline,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


@dataclass
class RunOptions(BenchFuncConfig):
    """
    Configuration options for running sparse neural network benchmarks.

    This class defines the parameters that control how the benchmark is executed,
    including distributed training settings, batch configuration, and profiling options.

    Args:
        world_size (int): Number of processes/GPUs to use for distributed training.
            Default is 2.
        batch_size (int): Batch size for training.
            Default is 1024 * 32.
        num_batches (int): Number of batches to process during the benchmark.
            Default is 10.
        input_type (str): Type of input format to use for the model.
            Default is "kjt" (KeyedJaggedTensor).
        num_benchmarks (int): Number of benchmark iterations.
            Default is 5.
        num_profiles (int): Number of profiling iterations.
            Default is 2.
        export_stacks (bool): Whether to export stack traces.
            Default is False.
        debug_mode (bool): Whether to enable debug mode.
            Default is False.
        topology_domain_multiple (Optional[int]): Number of hosts per NVLink domain/pod.
            Used for GB200 topology-aware benchmarking. When set, this determines the
            pod_size for the Topology object, affecting intra_group_size calculations.
            For GB200 with 10 hosts per domain: topology_domain_multiple=10.
            Default is None (falls back to single-node topology).
        topology_domain_max_group_count (Optional[int]): Maximum number of domain groups
            in the job. This is primarily used for MAST scheduling and documentation.
            Default is None.
        local_world_size (Optional[int]): Number of GPUs per host.
            For GB200: local_world_size=2. If not set, defaults to world_size
            (single-node assumption).
            Default is None.
        ga_num_steps (int): Number of gradient accumulation steps. When set to 1
            (default), the pipeline runs without gradient accumulation. When > 1,
            the pipeline is wrapped with GradientAccumulationWrapper to accumulate
            gradients over multiple micro-batches before synchronizing.
    """

    world_size: int = 2
    batch_size: int = 1024 * 32
    num_batches: int = 10
    input_type: str = "kjt"
    num_benchmarks: int = 5
    num_profiles: int = 2
    export_stacks: bool = False
    debug_mode: bool = False
    topology_domain_multiple: Optional[int] = None
    topology_domain_max_group_count: Optional[int] = None
    local_world_size: Optional[int] = None
    ga_num_steps: int = 1


# single-rank runner
def runner(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    run_option: RunOptions,
    model_config: BaseModelConfig,
    pipeline_config: PipelineConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
    sharding_config: ShardingConfig,
    table_related_configs: Optional[TableExtendedConfigs] = None,
    debug_mode: bool = False,
) -> BenchmarkResult:
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    # Set topology domain environment variable if specified
    # This enables TorchRec's comm.py to calculate proper topology group sizes
    if run_option.topology_domain_multiple is not None:
        os.environ["TOPOLOGY_DOMAIN_MULTIPLE"] = str(
            run_option.topology_domain_multiple
        )
        logger.info(
            f"Set TOPOLOGY_DOMAIN_MULTIPLE={run_option.topology_domain_multiple} "
            f"(local_world_size={run_option.local_world_size or 'auto'})"
        )

    # debug mode only works with vscode for now.
    if debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    run_option.set_log_level()
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:
        unsharded_model = model_config.generate_model(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=ctx.device,
            mc_configs=(
                table_related_configs.mc_configs if table_related_configs else None
            ),
        )

        # Create a planner for sharding based on the specified type
        planner = planner_config.generate_planner(
            tables=tables + weighted_tables,
        )

        bench_inputs = input_config.generate_batches(
            tables=tables,
            weighted_tables=weighted_tables,
        )

        sharded_model, optimizer = sharding_config.generate_sharded_model_and_optimizer(
            model=unsharded_model,
            # pyrefly: ignore[bad-argument-type]
            pg=ctx.pg,
            device=ctx.device,
            planner=planner,
        )

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            model: nn.Module,
            pipeline: TrainPipeline,
        ) -> None:
            pipeline.reset()
            dataloader = iter(bench_inputs)
            while True:
                try:
                    pipeline.progress(dataloader)
                except StopIteration:
                    break

        pipeline = pipeline_config.generate_pipeline(
            model=sharded_model,
            opt=optimizer,
            device=ctx.device,
        )

        if run_option.ga_num_steps > 1:
            ga_config = GradientAccumulationConfig(
                num_steps=run_option.ga_num_steps,
            )
            pipeline = GradientAccumulationWrapper(
                pipeline=pipeline,
                optimizer=optimizer,
                model=sharded_model,
                config=ga_config,
            )
        # Commented out due to potential conflict with pipeline.reset()
        # pipeline.progress(iter(bench_inputs))  # warmup

        run_option.name = (
            type(pipeline).__name__ if run_option.name == "" else run_option.name
        )
        result = benchmark_func(
            # pyrefly: ignore[bad-argument-type]
            bench_inputs=bench_inputs,
            # pyrefly: ignore[bad-argument-type]
            prof_inputs=bench_inputs,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={"model": sharded_model, "pipeline": pipeline},
            **run_option.benchmark_func_kwargs(rank=rank),
        )

        if rank == 0:
            logger.setLevel(logging.INFO)
            logger.info(result.prettify())
            logger.info("\nMarkdown format:\n%s", result)

        return result


# a standalone function to run the benchmark in multi-process mode
def run_pipeline(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    pipeline_config: PipelineConfig,
    model_config: BaseModelConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
    sharding_config: ShardingConfig,
) -> BenchmarkResult:
    tables, weighted_tables, *_ = table_config.generate_tables()

    benchmark_res_per_rank = run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        tables=tables,
        weighted_tables=weighted_tables,
        run_option=run_option,
        model_config=model_config,
        pipeline_config=pipeline_config,
        input_config=input_config,
        planner_config=planner_config,
        sharding_config=sharding_config,
    )

    # Combine results from all ranks into a single BenchmarkResult
    # Use timing data from rank 0, combine memory stats from all ranks
    world_size = run_option.world_size

    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        gpu_mem_stats=[
            GPUMemoryStats(rank, 0, 0, 0, 0, 0) for rank in range(world_size)
        ],
        cpu_mem_stats=[CPUMemoryStats(rank, 0) for rank in range(world_size)],
        rank=0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 GPU and 1 CPU memory measurement
        if len(res.gpu_mem_stats) > 0:
            total_benchmark_res.gpu_mem_stats[res.rank] = res.gpu_mem_stats[0]
        if len(res.cpu_mem_stats) > 0:
            total_benchmark_res.cpu_mem_stats[res.rank] = res.cpu_mem_stats[0]

    return total_benchmark_res


# command-line interface
@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    model_selection: ModelSelectionConfig,
    pipeline_config: PipelineConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
    sharding_config: ShardingConfig,
) -> None:
    if run_option.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    tables, weighted_tables, *_ = table_config.generate_tables()
    table_extended_config = TableExtendedConfigs(
        mc_configs=table_config.mc_configs_per_table,
    )
    model_config = model_selection.create_model_config()

    # launch trainers
    run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        tables=tables,
        weighted_tables=weighted_tables,
        run_option=run_option,
        model_config=model_config,
        pipeline_config=pipeline_config,
        input_config=input_config,
        planner_config=planner_config,
        sharding_config=sharding_config,
        table_related_configs=table_extended_config,
        debug_mode=run_option.debug_mode,
    )


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
