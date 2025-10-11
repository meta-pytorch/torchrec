#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16]

"""
Wrapper utilities for EmbeddingCollection (EC), EmbeddingBagCollection (EBC),
QuantEmbeddingCollection (QEC), and QuantEmbeddingBagCollection (QEBC) modules.

This module contains wrapper classes and utility functions for benchmarking EC, EBC, QEC,
and QEBC modules with different sharding strategies and compilation modes.
"""

import contextlib
import copy
import gc
import logging
import time
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import multiprocessing as mp
from torchrec.distributed import DistributedModelParallel

from torchrec.distributed.benchmark.base import (
    benchmark_model_with_warmup,
    BenchmarkResult,
    CompileMode,
    multi_process_benchmark,
)
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import DataType, ModuleSharder, ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

logger: logging.Logger = logging.getLogger()

T = TypeVar("T", bound=torch.nn.Module)


class ECWrapper(torch.nn.Module):
    """
    Wrapper Module for benchmarking EC Modules

    Args:
        module: module to benchmark

    Call Args:
        input: KeyedJaggedTensor KJT input to module

    Returns:
        output: Dict[str, JaggedTensor] output from module


    Example:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        ec.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qec = QuantEmbeddingCollection.from_float(ecc)

        wrapped_module = ECWrapper(qec)
        quantized_embeddings = wrapped_module(features)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """
        return self._module.forward(input)


class EBCWrapper(torch.nn.Module):
    """
    Wrapper Module for benchmarking EBC Modules

    Args:
        module: module to benchmark

    Call Args:
        input: KeyedJaggedTensor KJT input to module

    Returns:
        output: KT output from module

    Example:
        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)

        wrapped_module = EBCWrapper(qebc)
        quantized_embeddings = wrapped_module(features)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """
        return self._module.forward(input)


def _default_func_to_benchmark(
    model: torch.nn.Module, bench_inputs: List[KeyedJaggedTensor]
) -> None:
    with torch.inference_mode():
        for bench_input in bench_inputs:
            model(bench_input)


def get_tables(
    table_sizes: List[Tuple[int, int]],
    is_pooled: bool = True,
    data_type: DataType = DataType.INT8,
) -> Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]:
    if is_pooled:
        tables: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=data_type,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]
    else:
        tables: List[EmbeddingConfig] = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=data_type,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]

    return tables


def _get_inputs(
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    batch_size: int,
    world_size: int,
    num_inputs: int,
    train: bool,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
) -> List[List[KeyedJaggedTensor]]:
    inputs_batch: List[List[KeyedJaggedTensor]] = []

    if variable_batch_embeddings and not train:
        raise RuntimeError("Variable batch size is only supported in training mode")

    for _ in range(num_inputs):
        if variable_batch_embeddings:
            _, model_input_by_rank = ModelInput.generate_variable_batch_input(
                average_batch_size=batch_size,
                world_size=world_size,
                num_float_features=0,
                tables=tables,
            )
        else:
            _, model_input_by_rank = ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=0,
                tables=tables,
                weighted_tables=[],
                tables_pooling=pooling_configs,
                indices_dtype=torch.int32,
                lengths_dtype=torch.int32,
            )

        if train:
            sparse_features_by_rank = [
                model_input.idlist_features
                for model_input in model_input_by_rank
                if isinstance(model_input.idlist_features, KeyedJaggedTensor)
            ]
            inputs_batch.append(sparse_features_by_rank)
        else:
            sparse_features = model_input_by_rank[0].idlist_features
            assert isinstance(sparse_features, KeyedJaggedTensor)
            inputs_batch.append([sparse_features])

    # Transpose if train, as inputs_by_rank is currently in  [B X R] format
    inputs_by_rank = list(zip(*inputs_batch))

    return inputs_by_rank


def _transform_module(
    module: torch.nn.Module,
    device: torch.device,
    inputs: List[KeyedJaggedTensor],
    sharder: ModuleSharder[T],
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    # pyre-fixme[24]: Generic type `ContextManager` expects 1 type parameter.
    ctx: ContextManager,
    benchmark_unsharded_module: bool = False,
) -> torch.nn.Module:
    def fx_script_module(eager_module: torch.nn.Module) -> torch.nn.Module:
        eager_module(inputs[0])
        graph_module = symbolic_trace(
            eager_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
        )
        scripted_module = torch.jit.script(graph_module)
        return scripted_module

    set_propogate_device(True)

    sharded_module = None

    if not benchmark_unsharded_module:
        topology: Topology = Topology(world_size=world_size, compute_device=device.type)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        # Don't want to modify the module outright
        # Since module is on cpu, won't cause cuda oom.
        copied_module = copy.deepcopy(module)
        # pyre-ignore [6]
        plan = planner.plan(copied_module, [sharder])

        if isinstance(ctx, MultiProcessContext):
            sharded_module = DistributedModelParallel(
                copied_module,
                # pyre-ignore[6]
                env=ShardingEnv.from_process_group(ctx.pg),
                plan=plan,
                # pyre-ignore[6]
                sharders=[sharder],
                device=ctx.device,
            )
        else:
            env = ShardingEnv.from_local(world_size=topology.world_size, rank=0)

            sharded_module = _shard_modules(
                module=copied_module,
                # pyre-fixme[6]: For 2nd argument expected
                #  `Optional[List[ModuleSharder[Module]]]` but got
                #  `List[ModuleSharder[Variable[T (bound to Module)]]]`.
                sharders=[sharder],
                device=device,
                plan=plan,
                env=env,
            )

    if compile_mode == CompileMode.FX_SCRIPT:
        return fx_script_module(
            # pyre-fixme[6]: For 1st argument expected `Module` but got
            #  `Optional[Module]`.
            sharded_module
            if not benchmark_unsharded_module
            else module
        )
    else:
        # pyre-fixme[7]: Expected `Module` but got `Optional[Module]`.
        return sharded_module if not benchmark_unsharded_module else module


def _benchmark_type_name(compile_mode: CompileMode, sharding_type: ShardingType) -> str:
    if sharding_type == ShardingType.TABLE_WISE:
        name = "tw-sharded"
    elif sharding_type == ShardingType.ROW_WISE:
        name = "rw-sharded"
    elif sharding_type == ShardingType.COLUMN_WISE:
        name = "cw-sharded"
    else:
        raise Exception(f"Unknown sharding type {sharding_type}")

    if compile_mode == CompileMode.EAGER:
        name += "-eager"
    elif compile_mode == CompileMode.FX_SCRIPT:
        name += "-fxjit"

    return name


def _init_module_and_run_benchmark(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    device: torch.device,
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    warmup_inputs: List[List[KeyedJaggedTensor]],
    bench_inputs: List[List[KeyedJaggedTensor]],
    prof_inputs: List[List[KeyedJaggedTensor]],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int = -1,
    queue: Optional[mp.Queue] = None,
    pooling_configs: Optional[List[int]] = None,
    benchmark_unsharded_module: bool = False,
) -> BenchmarkResult:
    """
    There are a couple of caveats here as to why the module has to be initialized
    here:
    1. Device. To accurately track memory usage, when sharding modules the initial
       placement of the module should be on CPU. This is to avoid double counting
       memory allocations and also to prevent CUDA OOMs.
    2. Garbage Collector. Since torch.fx.GraphModule has circular references,
       garbage collection us funky and can lead to ooms. Since this frame is
       called by the loop through compile modes and sharding types, returning the
       benchmark result will mean that the reference to module is lost instead of
       existing in the loop
    """

    if rank >= 0:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:{rank}"))
            for warmup_input in warmup_inputs[rank]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:{rank}"))
            for bench_input in bench_inputs[rank]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:{rank}"))
            for prof_input in prof_inputs[rank]
        ]
    else:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:0"))
            for warmup_input in warmup_inputs[0]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:0"))
            for bench_input in bench_inputs[0]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:0"))
            for prof_input in prof_inputs[0]
        ]

    with (
        MultiProcessContext(rank, world_size, "nccl", None)
        if rank != -1
        else contextlib.nullcontext()
    ) as ctx:
        module = _transform_module(
            module=module,
            device=device,
            inputs=warmup_inputs_cuda,
            sharder=sharder,
            sharding_type=sharding_type,
            compile_mode=compile_mode,
            world_size=world_size,
            batch_size=batch_size,
            # pyre-ignore[6]
            ctx=ctx,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if benchmark_unsharded_module:
            name = "unsharded" + compile_mode.name
        else:
            name = _benchmark_type_name(compile_mode, sharding_type)

        res = benchmark_model_with_warmup(
            name,
            module,
            warmup_inputs_cuda,
            bench_inputs_cuda,
            prof_inputs_cuda,
            world_size=world_size,
            output_dir=output_dir,
            num_benchmarks=num_benchmarks,
            func_to_benchmark=func_to_benchmark,
            benchmark_func_kwargs=benchmark_func_kwargs,
            rank=rank,
            device_type=device.type,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if queue is not None:
            queue.put(res)

            while not queue.empty():
                time.sleep(1)

    return res


def benchmark_ebc_module(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    sharding_types: List[ShardingType],
    compile_modes: List[CompileMode],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    warmup_iters: int = 20,
    bench_iters: int = 500,
    prof_iters: int = 20,
    batch_size: int = 2048,
    world_size: int = 2,
    num_benchmarks: int = 5,
    output_dir: str = "",
    benchmark_unsharded: bool = False,
    func_to_benchmark: Callable[..., None] = _default_func_to_benchmark,
    benchmark_func_kwargs: Optional[Dict[str, Any]] = None,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
    device_type: str = "cuda",
) -> List[BenchmarkResult]:
    """
    Benchmark EmbeddingBagCollection (EBC) and QuantEmbeddingBagCollection (QEBC) modules.

    Args:
        module: EBC or QEBC module to be benchmarked
        sharder: Module sharder for distributing the module
        sharding_types: Sharding types to be benchmarked
        compile_modes: Compilation modes to be benchmarked
        tables: Embedding table configurations
        warmup_iters: Number of iterations to run before profiling
        bench_iters: Number of iterations to run during profiling
        prof_iters: Number of iterations to run after profiling
        batch_size: Batch size used in the model
        world_size: World size used in distributed training
        num_benchmarks: How many times to run over benchmark inputs for statistics
        output_dir: Directory to output profiler outputs (traces, stacks)
        benchmark_unsharded: Whether to benchmark unsharded version
        func_to_benchmark: Custom function to benchmark, check out default_func_to_benchmark for default
        benchmark_func_kwargs: Custom keyword arguments to pass to func_to_benchmark
        pooling_configs: The pooling factor for the tables (Optional; if not set, we'll use 10 as default)
        variable_batch_embeddings: Whether to use variable batch size embeddings
        device_type: Device type to use for benchmarking

    Returns:
        A list of BenchmarkResults

    Note:
        This function is specifically designed for EmbeddingBagCollection (EBC) and
        QuantEmbeddingBagCollection (QEBC) modules. It automatically detects the module
        type and applies appropriate wrapping and training mode settings.
    """

    logging.info(
        f"Warmup iterations: {warmup_iters}, "
        f"Benchmark iterations: {bench_iters}, "
        f"Profile iterations: {prof_iters}, "
        f"Batch Size: {batch_size}, "
        f"World Size: {world_size}, "
        f"Number of Benchmarks: {num_benchmarks}, "
        f"Output Directory: {output_dir}"
    )

    assert (
        num_benchmarks > 2
    ), "num_benchmarks needs to be greater than 2 for statistical analysis"

    # Determine training mode based on module type
    is_train = not isinstance(
        module, (QuantEmbeddingCollection, QuantEmbeddingBagCollection)
    )

    benchmark_results: List[BenchmarkResult] = []

    # Wrap the module appropriately based on table type
    if isinstance(tables[0], EmbeddingBagConfig):
        wrapped_module = EBCWrapper(module)
    else:
        wrapped_module = ECWrapper(module)

    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters
    inputs = _get_inputs(
        tables,
        batch_size,
        world_size,
        num_inputs_to_gen,
        is_train,
        pooling_configs,
        variable_batch_embeddings,
    )

    warmup_inputs = [rank_inputs[:warmup_iters] for rank_inputs in inputs]
    bench_inputs = [
        rank_inputs[warmup_iters : (warmup_iters + bench_iters)]
        for rank_inputs in inputs
    ]
    prof_inputs = [rank_inputs[-prof_iters:] for rank_inputs in inputs]

    for sharding_type in sharding_types if not benchmark_unsharded else ["Unsharded"]:
        for compile_mode in compile_modes:
            if not benchmark_unsharded:
                # Test sharders should have a singular sharding_type
                sharder._sharding_type = sharding_type.value
                # pyre-ignore [6]
                benchmark_type = _benchmark_type_name(compile_mode, sharding_type)
            else:
                benchmark_type = "unsharded" + compile_mode.name

            logging.info(
                f"\n\n###### Running EBC/QEBC Benchmark Type: {benchmark_type} ######\n"
            )

            if is_train:
                res = multi_process_benchmark(
                    # pyre-ignore[6]
                    callable=_init_module_and_run_benchmark,
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,
                    bench_inputs=bench_inputs,
                    prof_inputs=prof_inputs,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                )
            else:
                res = _init_module_and_run_benchmark(
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    # pyre-ignore
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,
                    bench_inputs=bench_inputs,
                    prof_inputs=prof_inputs,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                    benchmark_unsharded_module=benchmark_unsharded,
                )

            gc.collect()

            benchmark_results.append(res)

    return benchmark_results
