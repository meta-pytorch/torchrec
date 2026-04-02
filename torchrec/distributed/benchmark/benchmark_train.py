#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import argparse
import logging
import os
import time
from functools import partial
from typing import List, Optional, Tuple

import torch
from torchrec.distributed.benchmark.base import (
    BenchmarkResult,
    CompileMode,
    init_argparse_and_args,
    set_embedding_config,
    write_report,
)
from torchrec.distributed.benchmark.embedding_collection_wrappers import (
    benchmark_ebc_module,
    get_tables,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.test_utils.emb_sharder import TestEBCSharder, TestECSharder
from torchrec.distributed.types import DataType
from torchrec.modules.embedding_configs import EmbeddingConfig, NoEvictionPolicy
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger()


BENCH_SHARDING_TYPES: List[ShardingType] = [
    ShardingType.TABLE_WISE,
    ShardingType.ROW_WISE,
    ShardingType.COLUMN_WISE,
]

BENCH_COMPILE_MODES: List[CompileMode] = [
    CompileMode.EAGER,
    # CompileMode.FX_SCRIPT,
]

WRITE_BENCH_SHARDING_TYPES: List[ShardingType] = [
    ShardingType.ROW_WISE,
]


def training_func_to_benchmark(
    model: torch.nn.Module,
    bench_inputs: List[KeyedJaggedTensor],
    optimizer: Optional[torch.optim.Optimizer],
) -> None:
    for bench_input in bench_inputs:
        pooled_embeddings = model(bench_input)
        vals = []
        for _name, param in pooled_embeddings.to_dict().items():
            vals.append(param)
        torch.cat(vals, dim=1).sum().backward()
        if optimizer:
            optimizer.step()
            optimizer.zero_grad()


def benchmark_ebc(
    tables: List[Tuple[int, int]],
    args: argparse.Namespace,
    output_dir: str,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
) -> List[BenchmarkResult]:
    table_configs = get_tables(tables, data_type=DataType.FP32)
    sharder = TestEBCSharder(
        sharding_type="",  # sharding_type gets populated during benchmarking
        kernel_type=EmbeddingComputeKernel.FUSED.value,
    )

    # we initialize the embedding tables using CUDA, because when the table is large,
    # CPU initialization will be prohibitively long. We then copy the module back
    # to CPU because this module will be sent over subprocesses via multiprocessing,
    # and we don't want to create an extra CUDA context on GPU0 for each subprocess.
    # we also need to release the memory in the parent process (empty_cache)
    module = EmbeddingBagCollection(
        # pyrefly: ignore[bad-argument-type]
        tables=table_configs,
        is_weighted=False,
        device=torch.device("cuda"),
    ).cpu()

    torch.cuda.empty_cache()

    IGNORE_ARGNAME = ["output_dir", "embedding_config_json", "max_num_embeddings"]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.02)
    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    if pooling_configs:
        args_kwargs["pooling_configs"] = pooling_configs

    args_kwargs["variable_batch_embeddings"] = variable_batch_embeddings

    return benchmark_ebc_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=table_configs,
        output_dir=output_dir,
        func_to_benchmark=training_func_to_benchmark,
        benchmark_func_kwargs={"optimizer": optimizer},
        **args_kwargs,
    )


def write_func_to_benchmark(
    model: torch.nn.Module,
    bench_inputs: List[KeyedJaggedTensor],
    embedding_dim: int = 64,
) -> None:
    for bench_input in bench_inputs:
        # Construct write input: KJT with weights containing the embedding values.
        # The write path expects a KJT whose keys are the writable feature names,
        # with weights of shape [num_values, embedding_dim] as the new embeddings.
        total_values = int(bench_input.lengths().sum().item())
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=bench_input.keys(),
            values=bench_input.values(),
            lengths=bench_input.lengths(),
            weights=torch.rand(
                total_values,
                embedding_dim,
                dtype=torch.float32,
                device=bench_input.values().device,
            ),
        )

        assert isinstance(model, DistributedModelParallel)
        model.write(write_kjt)


def write_read_func_to_benchmark(
    model: torch.nn.Module,
    bench_inputs: List[KeyedJaggedTensor],
    embedding_dim: int = 64,
) -> None:
    for bench_input in bench_inputs:
        # Write: construct KJT with random embedding weights and write to model.
        total_values = int(bench_input.lengths().sum().item())
        write_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=bench_input.keys(),
            values=bench_input.values(),
            lengths=bench_input.lengths(),
            weights=torch.rand(
                total_values,
                embedding_dim,
                dtype=torch.float32,
                device=bench_input.values().device,
            ),
        )
        assert isinstance(model, DistributedModelParallel)
        model.write(write_kjt)
        # Read: forward pass to read back embeddings.
        model(bench_input)


def benchmark_ec_write(
    tables: List[Tuple[int, int]],
    args: argparse.Namespace,
    output_dir: str,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
) -> List[BenchmarkResult]:
    total_num_buckets = 20
    # Round up num_embeddings to be divisible by total_num_buckets
    aligned_tables = [
        (((num + total_num_buckets - 1) // total_num_buckets) * total_num_buckets, dim)
        for num, dim in tables
    ]
    table_configs = get_tables(aligned_tables, is_pooled=False, data_type=DataType.FP32)
    # Set enable_embedding_update and virtual table configs for write support
    ec_tables = []
    for table in table_configs:
        assert isinstance(table, EmbeddingConfig)
        table.enable_embedding_update = True
        table.use_virtual_table = True
        table.total_num_buckets = total_num_buckets
        table.virtual_table_eviction_policy = NoEvictionPolicy()
        # make type checking happy
        ec_tables.append(table)

    sharder = TestECSharder(
        sharding_type="",  # sharding_type gets populated during benchmarking
        kernel_type=EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value,
    )

    module = EmbeddingCollection(
        tables=ec_tables,
        device=torch.device("cuda"),
    ).cpu()

    torch.cuda.empty_cache()

    IGNORE_ARGNAME = ["output_dir", "embedding_config_json", "max_num_embeddings"]
    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    if pooling_configs:
        args_kwargs["pooling_configs"] = pooling_configs

    args_kwargs["variable_batch_embeddings"] = variable_batch_embeddings

    # DRAM_VIRTUAL_TABLE is a guarded compute kernel - must be explicitly
    # specified via ParameterConstraints or the planner will filter it out.
    constraints = {
        table.name: ParameterConstraints(
            compute_kernels=[EmbeddingComputeKernel.DRAM_VIRTUAL_TABLE.value],
        )
        for table in table_configs
    }

    # Benchmark write-only, then write+read
    results: List[BenchmarkResult] = []
    results.extend(
        benchmark_ebc_module(
            module=module,
            sharder=sharder,
            sharding_types=WRITE_BENCH_SHARDING_TYPES,
            compile_modes=BENCH_COMPILE_MODES,
            tables=table_configs,
            output_dir=output_dir,
            benchmark_func_kwargs={"embedding_dim": table_configs[0].embedding_dim},
            constraints=constraints,
            func_to_benchmark=write_func_to_benchmark,
            **args_kwargs,
        )
    )
    results.extend(
        benchmark_ebc_module(
            module=module,
            sharder=sharder,
            sharding_types=WRITE_BENCH_SHARDING_TYPES,
            compile_modes=BENCH_COMPILE_MODES,
            tables=table_configs,
            output_dir=output_dir,
            benchmark_func_kwargs={"embedding_dim": table_configs[0].embedding_dim},
            constraints=constraints,
            func_to_benchmark=write_read_func_to_benchmark,
            **args_kwargs,
        )
    )
    return results


def main() -> None:
    args: argparse.Namespace = init_argparse_and_args()

    num_requests = args.bench_iters * args.batch_size * args.num_benchmarks
    datetime_sfx: str = time.strftime("%Y%m%dT%H%M%S")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        # Create output directory if not exist
        os.mkdir(output_dir)

    benchmark_results_per_module = []
    write_report_funcs_per_module = []
    shrunk_table_sizes = []

    embedding_configs, pooling_configs = set_embedding_config(
        args.embedding_config_json
    )
    for config in embedding_configs:
        shrunk_table_sizes.append((min(args.max_num_embeddings, config[0]), config[1]))

    for module_name in ["EmbeddingBagCollection"]:
        output_dir = args.output_dir + f"/run_{datetime_sfx}"
        output_dir += "_ebc"
        benchmark_func = benchmark_ebc

        if not os.path.exists(output_dir):
            # Place all outputs under the datetime folder
            os.mkdir(output_dir)

        tables_info = "\nTABLE SIZES:"
        for i, (num, dim) in enumerate(shrunk_table_sizes):
            # FP32 is 4 bytes
            mb = int(float(num * dim) / 1024 / 1024) * 4
            tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] {mb:6}Mb"

        ### Benchmark no VBE
        report: str = (
            f"REPORT BENCHMARK {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size}\n"
        )
        report += f"Module: {module_name}\n"
        report += tables_info
        report += "\n"

        report += f"num_requests:{num_requests:8}\n"
        report_file: str = f"{output_dir}/run.report"

        # Save results to output them once benchmarking is all done
        benchmark_results_per_module.append(
            benchmark_func(shrunk_table_sizes, args, output_dir, pooling_configs)
        )
        write_report_funcs_per_module.append(
            partial(
                write_report,
                report_file=report_file,
                report_str=report,
                num_requests=num_requests,
            )
        )

        ### Benchmark with VBE
        report: str = (
            f"REPORT BENCHMARK (VBE) {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size}\n"
        )
        report += f"Module: {module_name} (VBE)\n"
        report += tables_info
        report += "\n"
        report_file = f"{output_dir}/run_vbe.report"

        benchmark_results_per_module.append(
            benchmark_func(shrunk_table_sizes, args, output_dir, pooling_configs, True)
        )
        write_report_funcs_per_module.append(
            partial(
                write_report,
                report_file=report_file,
                report_str=report,
                num_requests=num_requests,
            )
        )

    ### Benchmark EC write and write+read
    output_dir = args.output_dir + f"/run_{datetime_sfx}"
    output_dir += "_ec_write"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tables_info = "\nTABLE SIZES:"
    for i, (num, dim) in enumerate(shrunk_table_sizes):
        mb = int(float(num * dim) / 1024 / 1024) * 4
        tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] {mb:6}Mb"

    report = f"REPORT BENCHMARK {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size}\n"
    report += "Module: EmbeddingCollection_write\n"
    report += tables_info
    report += "\n"
    report += f"num_requests:{num_requests:8}\n"
    report_file = f"{output_dir}/run.report"

    benchmark_results_per_module.append(
        benchmark_ec_write(shrunk_table_sizes, args, output_dir, pooling_configs)
    )
    write_report_funcs_per_module.append(
        partial(
            write_report,
            report_file=report_file,
            report_str=report,
            num_requests=num_requests,
        )
    )

    for i, write_report_func in enumerate(write_report_funcs_per_module):
        write_report_func(benchmark_results_per_module[i])


def invoke_main() -> None:
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    main()


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
