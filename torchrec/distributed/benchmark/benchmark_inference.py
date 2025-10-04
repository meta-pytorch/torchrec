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
from typing import List, Tuple

import torch

from torchrec.distributed.benchmark.base import (
    BenchmarkResult,
    CompileMode,
    DLRM_NUM_EMBEDDINGS_PER_FEATURE,
    EMBEDDING_DIM,
    init_argparse_and_args,
    write_report,
)
from torchrec.distributed.benchmark.embedding_collection_wrappers import (
    benchmark_ebc_module,
    get_tables,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.test_utils.infer_utils import (
    TestQuantEBCSharder,
    TestQuantECSharder,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)

logger: logging.Logger = logging.getLogger()


BENCH_SHARDING_TYPES: List[ShardingType] = [
    ShardingType.TABLE_WISE,
    ShardingType.ROW_WISE,
    # ShardingType.COLUMN_WISE,
    # TODO: CW with FXJIT takes long time while profiling, doesn't cause an issue with no profiling in automatic benchmark
]

BENCH_COMPILE_MODES: List[CompileMode] = [
    CompileMode.EAGER,
    CompileMode.FX_SCRIPT,
]


TABLE_SIZES: List[Tuple[int, int]] = [
    (num_embeddings, EMBEDDING_DIM)
    for num_embeddings in DLRM_NUM_EMBEDDINGS_PER_FEATURE
]

IGNORE_ARGNAME = ["output_dir", "embedding_config_json", "max_num_embeddings"]


def benchmark_qec(args: argparse.Namespace, output_dir: str) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES, is_pooled=False)
    sharder = TestQuantECSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingCollection(
        # pyre-ignore [6]
        tables=tables,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    return benchmark_ebc_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        **args_kwargs,
    )


def benchmark_qebc(args: argparse.Namespace, output_dir: str) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES)
    sharder = TestQuantEBCSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingBagCollection(
        # pyre-ignore [6]
        tables=tables,
        is_weighted=False,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    return benchmark_ebc_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        **args_kwargs,
    )


def benchmark_qec_unsharded(
    args: argparse.Namespace, output_dir: str
) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES, is_pooled=False)
    sharder = TestQuantECSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingCollection(
        # pyre-ignore [6]
        tables=tables,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    return benchmark_ebc_module(
        module=module,
        sharder=sharder,
        sharding_types=[],
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        benchmark_unsharded=True,  # benchmark unsharded module
        **args_kwargs,
    )


def benchmark_qebc_unsharded(
    args: argparse.Namespace, output_dir: str
) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES)
    sharder = TestQuantEBCSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingBagCollection(
        # pyre-ignore [6]
        tables=tables,
        is_weighted=False,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname not in IGNORE_ARGNAME
    }

    return benchmark_ebc_module(
        module=module,
        sharder=sharder,
        sharding_types=[],
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        benchmark_unsharded=True,  # benchmark unsharded module
        **args_kwargs,
    )


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

    module_names = [
        "QuantEmbeddingBagCollection",
        "QuantEmbeddingCollection",
    ]

    # Only do unsharded QEBC/QEC benchmark when using CPU device
    if args.device_type == "cpu":
        module_names.append("unshardedQuantEmbeddingBagCollection")
        module_names.append("unshardedQuantEmbeddingCollection")

    for module_name in module_names:
        output_dir = args.output_dir + f"/run_{datetime_sfx}"
        if module_name == "QuantEmbeddingBagCollection":
            output_dir += "_qebc"
            benchmark_func = benchmark_qebc
        elif module_name == "QuantEmbeddingCollection":
            output_dir += "_qec"
            benchmark_func = benchmark_qec
        elif module_name == "unshardedQuantEmbeddingBagCollection":
            output_dir += "_uqebc"
            benchmark_func = benchmark_qebc_unsharded
        else:
            output_dir += "_uqec"
            benchmark_func = benchmark_qec_unsharded

        if not os.path.exists(output_dir):
            # Place all outputs under the datetime folder
            os.mkdir(output_dir)

        tables_info = "\nTABLE SIZES QUANT:"
        for i, (num, dim) in enumerate(TABLE_SIZES):
            mb = int(float(num * dim) / 1024 / 1024)
            tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] u8: {mb:6}Mb"

        report: str = (
            f"REPORT BENCHMARK {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size}\n"
        )
        report += f"Module: {module_name}\n"
        report += tables_info
        report += "\n"

        report += f"num_requests:{num_requests:8}\n"
        report_file: str = f"{output_dir}/run.report"

        # Save results to output them once benchmarking is all done
        benchmark_results_per_module.append(benchmark_func(args, output_dir))
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
