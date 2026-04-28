#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Model Lifecycle Benchmark

Benchmarks the full model lifecycle in a single measured function:
  1. Model creation and sharding (DMP construction)
  2. Checkpoint save (state_dict extraction)
  3. Checkpoint load (load_state_dict)
  4. Training pipeline (forward + backward for N batches)
  5. RecMetrics update and compute

Input batches are pre-generated outside the benchmark loop so that only
the model lifecycle work is measured.

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_model_lifecycle -- \
        --world_size=2 --batch_size=4096 --num_batches=5 --pipeline=sparse

    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_model_lifecycle -- \
        --world_size=2 --batch_size=4096 --num_batches=5 --pipeline=sparse \
        --enable_metrics=True --metrics ne --compute_interval=5

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_model_lifecycle \
        --world_size=2 --batch_size=4096 --num_batches=5 --pipeline=sparse
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
)
from torchrec.distributed.test_utils.input_config import ModelInputConfig
from torchrec.distributed.test_utils.metric_config import RecMetricConfig
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
from torchrec.distributed.utils import EmbeddingQuantizationUtils
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.types import DataType


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RunOptions(BenchFuncConfig):
    """
    Configuration for the model lifecycle benchmark.

    Args:
        world_size: Number of processes/GPUs for distributed training.
        batch_size: Batch size for training.
        num_batches: Number of batches per pipeline iteration.
        num_benchmarks: Number of times the full lifecycle is measured.
        num_iters: Total training iterations per lifecycle run. When set,
            the dataloader cycles over num_batches until num_iters is reached.
        output_json: Emit JSON output instead of human-readable table.
        local_world_size: Number of GPUs per host. Defaults to world_size.
    """

    world_size: int = 2
    batch_size: int = 1024 * 32
    num_batches: int = 10
    num_benchmarks: int = 0
    num_profiles: int = 1
    export_stacks: bool = False
    debug_mode: bool = False
    output_json: bool = False
    num_iters: Optional[int] = None
    local_world_size: Optional[int] = None
    workflow: str = "model_init"
    run_forward: bool = True


def _setup(
    run_option: RunOptions,
    input_config: ModelInputConfig,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    rank: int,
) -> List[ModelInput]:
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= run_option.world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    if run_option.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    run_option.set_log_level()

    # --- Pre-generate inputs outside the benchmark loop ---
    bench_inputs = input_config.generate_batches(
        tables=tables,
        weighted_tables=weighted_tables,
    )

    total_bytes = 0
    for i, batch in enumerate(bench_inputs):
        batch_bytes = batch.size_in_bytes()
        total_bytes += batch_bytes
        if batch_bytes >= 1024 * 1024 * 1024:
            batch_size_str = f"{batch_bytes / 1024 / 1024 / 1024:.2f} GB"
        else:
            batch_size_str = f"{batch_bytes / 1024 / 1024:.2f} MB"
        logger.info(f"Rank {rank} batch {i} input size: {batch_size_str}")
    if total_bytes >= 1024 * 1024 * 1024:
        total_size_str = f"{total_bytes / 1024 / 1024 / 1024:.2f} GB"
    else:
        total_size_str = f"{total_bytes / 1024 / 1024:.2f} MB"
    logger.info(
        f"Rank {rank} total input size: {total_size_str} ({len(bench_inputs)} batches)"
    )
    return bench_inputs


def model_init_runner(
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
    metric_config: RecMetricConfig,
    table_related_configs: Optional[TableExtendedConfigs] = None,
) -> BenchmarkResult:

    bench_inputs = _setup(run_option, input_config, tables, weighted_tables, rank)

    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="cpu:gloo,cuda:nccl",
        use_deterministic_algorithms=False,
    ) as ctx:

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
        ) -> None:
            with record_function("## model_creation ##"):
                # unsharded_model is created on meta device (sparse) and CPU (dense)
                unsharded_model = model_config.generate_model(
                    tables=tables,
                    weighted_tables=weighted_tables,
                    dense_device=ctx.device,
                    mc_configs=(
                        table_related_configs.mc_configs
                        if table_related_configs
                        else None
                    ),
                )
                planner = planner_config.generate_planner(
                    tables=tables + weighted_tables,
                )
                sharded_model, optimizer = (
                    sharding_config.generate_sharded_model_and_optimizer(
                        model=unsharded_model,
                        # pyrefly: ignore[bad-argument-type]
                        pg=ctx.pg,
                        device=ctx.device,
                        planner=planner,
                    )
                )

            with record_function("## forward ##"):
                if run_option.run_forward:
                    batch = bench_inputs[0]
                    sharded_model(batch.to(ctx.device))

            with record_function("## checkpoint_save ##"):
                state_dict = sharded_model.state_dict()

            with record_function("## checkpoint_load ##"):
                # pyrefly: ignore[bad-argument-type]
                sharded_model.load_state_dict(dict(state_dict))

            torch.cuda.synchronize()

        result = benchmark_func(
            # pyrefly: ignore[bad-argument-type]
            bench_inputs=bench_inputs,
            # pyrefly: ignore[bad-argument-type]
            prof_inputs=bench_inputs,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={},
            sample_count=0,
            **run_option.benchmark_func_kwargs(rank=rank),
        )

        if rank == 0:
            logger.setLevel(logging.INFO)
            if run_option.output_json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                logger.info(result.prettify())
                logger.info("\nMarkdown format:\n%s", result)

        return result


def quant_model_init1_runner(
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
    metric_config: RecMetricConfig,
    table_related_configs: Optional[TableExtendedConfigs] = None,
) -> BenchmarkResult:
    """Shard a float model first, then quantize TBE kernels in-place."""

    bench_inputs = _setup(run_option, input_config, tables, weighted_tables, rank)

    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="cpu:gloo,cuda:nccl",
        use_deterministic_algorithms=False,
    ) as ctx:

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
        ) -> None:
            with record_function("## model_creation ##"):
                unsharded_model = model_config.generate_model(
                    tables=tables,
                    weighted_tables=weighted_tables,
                    dense_device=ctx.device,
                    mc_configs=(
                        table_related_configs.mc_configs
                        if table_related_configs
                        else None
                    ),
                )

            with record_function("## shard ##"):
                planner = planner_config.generate_planner(
                    tables=tables + weighted_tables,
                )
                sharded_model, optimizer = (
                    sharding_config.generate_sharded_model_and_optimizer(
                        model=unsharded_model,
                        # pyrefly: ignore[bad-argument-type]
                        pg=ctx.pg,
                        device=ctx.device,
                        planner=planner,
                    )
                )

            with record_function("## quantize ##"):
                quant_utils = EmbeddingQuantizationUtils()
                quant_utils.quantize_embedding_modules(
                    sharded_model, converted_dtype=DataType.NFP8
                )

            with record_function("## forward ##"):
                if run_option.run_forward:
                    batch = bench_inputs[0]
                    sharded_model(batch.to(ctx.device))

            # with record_function("## checkpoint_save ##"):
            #     state_dict = sharded_model.state_dict()

            # with record_function("## checkpoint_load ##"):
            #     # pyrefly: ignore[bad-argument-type]
            #     sharded_model.load_state_dict(dict(state_dict))

            torch.cuda.synchronize()

        result = benchmark_func(
            # pyrefly: ignore[bad-argument-type]
            bench_inputs=bench_inputs,
            # pyrefly: ignore[bad-argument-type]
            prof_inputs=bench_inputs,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={},
            sample_count=0,
            **run_option.benchmark_func_kwargs(rank=rank),
        )

        if rank == 0:
            logger.setLevel(logging.INFO)
            if run_option.output_json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                logger.info(result.prettify())
                logger.info("\nMarkdown format:\n%s", result)

        return result


@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    model_selection: ModelSelectionConfig,
    pipeline_config: PipelineConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
    sharding_config: ShardingConfig,
    metric_config: RecMetricConfig,
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
    match run_option.workflow:
        case "model_init":
            runner = model_init_runner
        case "quant_model_init1":
            runner = quant_model_init1_runner
        case _:
            raise ValueError(f"Unknown workflow {run_option.workflow}")

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
        metric_config=metric_config,
        table_related_configs=table_extended_config,
    )


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
