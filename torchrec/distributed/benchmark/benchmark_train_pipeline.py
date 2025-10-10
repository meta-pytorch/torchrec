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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
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
    create_model_config,
    generate_sharded_model_and_optimizer,
)
from torchrec.distributed.test_utils.model_input import ModelInput

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.pipeline_config import PipelineConfig
from torchrec.distributed.test_utils.sharding_config import PlannerConfig
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import TestOverArchLarge
from torchrec.distributed.train_pipeline import TrainPipeline
from torchrec.distributed.types import ShardingType
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
        num_batches (int): Number of batches to process during the benchmark.
            Default is 10.
        sharding_type (ShardingType): Strategy for sharding embedding tables across devices.
            Default is ShardingType.TABLE_WISE (entire tables are placed on single devices).
        compute_kernel (EmbeddingComputeKernel): Compute kernel to use for embedding tables.
            Default is EmbeddingComputeKernel.FUSED.
        input_type (str): Type of input format to use for the model.
            Default is "kjt" (KeyedJaggedTensor).
        profile (str): Directory to save profiling results. If empty, profiling is disabled.
            Default is "" (disabled).
        name (str): Name of the profiling file. Default is pipeline classname.
        planner_type (str): Type of sharding planner to use. Options are:
            - "embedding": EmbeddingShardingPlanner (default)
            - "hetero": HeteroEmbeddingShardingPlanner
        pooling_factors (Optional[List[float]]): Pooling factors for each feature of the table.
            This is the average number of values each sample has for the feature.
        num_poolings (Optional[List[float]]): Number of poolings for each feature of the table.
        dense_optimizer (str): Optimizer to use for dense parameters.
            Default is "SGD".
        dense_lr (float): Learning rate for dense parameters.
            Default is 0.1.
        sparse_optimizer (str): Optimizer to use for sparse parameters.
            Default is "EXACT_ADAGRAD".
        sparse_lr (float): Learning rate for sparse parameters.
            Default is 0.1.
    """

    world_size: int = 2
    num_batches: int = 10
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    input_type: str = "kjt"
    name: str = ""
    profile_dir: str = ""
    num_benchmarks: int = 5
    num_profiles: int = 2
    num_poolings: Optional[List[float]] = None
    dense_optimizer: str = "SGD"
    dense_lr: float = 0.1
    dense_momentum: Optional[float] = None
    dense_weight_decay: Optional[float] = None
    sparse_optimizer: str = "EXACT_ADAGRAD"
    sparse_lr: float = 0.1
    sparse_momentum: Optional[float] = None
    sparse_weight_decay: Optional[float] = None
    export_stacks: bool = False


@dataclass
class ModelSelectionConfig:
    model_name: str = "test_sparse_nn"

    # Common config for all model types
    batch_size: int = 1024 * 32
    batch_sizes: Optional[List[int]] = None
    num_float_features: int = 10
    feature_pooling_avg: int = 10
    use_offsets: bool = False
    dev_str: str = ""
    long_kjt_indices: bool = True
    long_kjt_offsets: bool = True
    long_kjt_lengths: bool = True
    pin_memory: bool = True

    # TestSparseNN specific config
    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None
    max_feature_lengths: Optional[Dict[str, int]] = None
    over_arch_clazz: Type[nn.Module] = TestOverArchLarge
    postproc_module: Optional[nn.Module] = None
    zch: bool = False

    # DeepFM specific config
    hidden_layer_size: int = 20
    deep_fm_dimension: int = 5

    # DLRM specific config
    dense_arch_layer_sizes: List[int] = field(default_factory=lambda: [20, 128])
    over_arch_layer_sizes: List[int] = field(default_factory=lambda: [5, 1])


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
) -> BenchmarkResult:
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    torch.autograd.set_detect_anomaly(True)
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
        )

        # Create a planner for sharding based on the specified type
        planner = planner_config.generate_planner(
            tables=tables + weighted_tables,
        )

        bench_inputs = input_config.generate_batches(
            tables=tables,
            weighted_tables=weighted_tables,
        )

        # Prepare fused_params for sparse optimizer
        fused_params = {
            "optimizer": getattr(EmbOptimType, run_option.sparse_optimizer.upper()),
            "learning_rate": run_option.sparse_lr,
        }

        # Add momentum and weight_decay to fused_params if provided
        if run_option.sparse_momentum is not None:
            fused_params["momentum"] = run_option.sparse_momentum

        if run_option.sparse_weight_decay is not None:
            fused_params["weight_decay"] = run_option.sparse_weight_decay

        sharded_model, optimizer = generate_sharded_model_and_optimizer(
            model=unsharded_model,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params=fused_params,
            dense_optimizer=run_option.dense_optimizer,
            dense_lr=run_option.dense_lr,
            dense_momentum=run_option.dense_momentum,
            dense_weight_decay=run_option.dense_weight_decay,
            planner=planner,
        )

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            model: nn.Module,
            pipeline: TrainPipeline,
        ) -> None:
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
        pipeline.progress(iter(bench_inputs))  # warmup

        run_option.name = (
            type(pipeline).__name__ if run_option.name == "" else run_option.name
        )
        result = benchmark_func(
            bench_inputs=bench_inputs,  # pyre-ignore
            prof_inputs=bench_inputs,  # pyre-ignore
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={"model": sharded_model, "pipeline": pipeline},
            **run_option.benchmark_func_kwargs(rank=rank)
        )

        if rank == 0:
            print(result)

        return result


# a standalone function to run the benchmark in multi-process mode
def run_pipeline(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    pipeline_config: PipelineConfig,
    model_config: BaseModelConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
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
    )

    # Combine results from all ranks into a single BenchmarkResult
    # Use timing data from rank 0, combine memory stats from all ranks
    world_size = run_option.world_size

    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        gpu_mem_stats=[GPUMemoryStats(rank, 0, 0, 0) for rank in range(world_size)],
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
@cmd_conf  # pyre-ignore [56]
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    model_selection: ModelSelectionConfig,
    pipeline_config: PipelineConfig,
    input_config: ModelInputConfig,
    planner_config: PlannerConfig,
    model_config: Optional[BaseModelConfig] = None,
) -> None:
    tables, weighted_tables, *_ = table_config.generate_tables()

    if model_config is None:
        model_config = create_model_config(
            model_name=model_selection.model_name,
            batch_size=model_selection.batch_size,
            batch_sizes=model_selection.batch_sizes,
            num_float_features=model_selection.num_float_features,
            feature_pooling_avg=model_selection.feature_pooling_avg,
            use_offsets=model_selection.use_offsets,
            dev_str=model_selection.dev_str,
            long_kjt_indices=model_selection.long_kjt_indices,
            long_kjt_offsets=model_selection.long_kjt_offsets,
            long_kjt_lengths=model_selection.long_kjt_lengths,
            pin_memory=model_selection.pin_memory,
            embedding_groups=model_selection.embedding_groups,
            feature_processor_modules=model_selection.feature_processor_modules,
            max_feature_lengths=model_selection.max_feature_lengths,
            over_arch_clazz=model_selection.over_arch_clazz,
            postproc_module=model_selection.postproc_module,
            zch=model_selection.zch,
            hidden_layer_size=model_selection.hidden_layer_size,
            deep_fm_dimension=model_selection.deep_fm_dimension,
            dense_arch_layer_sizes=model_selection.dense_arch_layer_sizes,
            over_arch_layer_sizes=model_selection.over_arch_layer_sizes,
        )

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
    )


if __name__ == "__main__":
    main()
