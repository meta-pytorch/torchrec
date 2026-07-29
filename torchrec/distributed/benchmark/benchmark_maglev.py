#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for the Maglev staged pipeline (MVP).

Measures the microbatched 1F1B schedule
(:func:`~torchrec.distributed.maglev.pipeline.run_1f1b`) of a K-stage Maglev
model across per-stage process groups (one hardware scale-up domain, HSD, each),
including the cross-HSD activation / gradient hand-off. Each stage is an
``EmbeddingBagCollection`` feature partition plus dense compute
(:class:`~torchrec.distributed.test_utils.test_model.MaglevTestStage`, shared
with the correctness test). One measured iteration is one full 1F1B pass over
``num_microbatches`` microbatches.

Runs on GPU + nccl by default (the cross-HSD P2P hand-off and the profiler path
are CUDA-only); `all_rank_traces` defaults on so every stage's HSD shows up in
the traces. Pass ``--device_type=cpu`` for the portable gloo repro (no traces on
that path).

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_maglev -- --profile_dir=/tmp/maglev_traces

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_maglev --profile_dir=/tmp/maglev_traces
"""

import json
import logging
from dataclasses import dataclass
from typing import List

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
from torchrec.distributed.maglev.pipeline import MaglevPipeline, run_1f1b
from torchrec.distributed.maglev.stage import (
    build_handoff_process_groups,
    build_stage_process_groups,
    EmbeddingShard,
    Replicated,
    StageWrapper,
)
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import MaglevTestStage
from torchrec.modules.embedding_configs import EmbeddingBagConfig

# Weights are seeded per stage so an HSD's two DP ranks start identical (the
# sharded path builds DMP with init_parameters=False, so it relies on this).
# Inputs are left unseeded -- values don't affect a throughput benchmark.
_WEIGHT_SEED = 100


@dataclass
class RunOptions(BenchFuncConfig):
    """Configuration for the Maglev pipeline benchmark.

    Args:
        name (str): Human-readable benchmark name. Default is "maglev".
        world_size (int): Total number of ranks. Must equal
            ``num_stages * ranks_per_stage``. Default is 8.
        num_benchmarks (int): Number of measured iterations (each one full 1F1B
            pass). Default is 12.
        num_profiles (int): Number of profiling iterations (CUDA only; unused on
            the CPU path). Default is 2.
        num_stages (int): Number of Maglev stages (one HSD each). Default is 4.
        ranks_per_stage (int): Ranks per stage's HSD (intra-HSD data
            parallelism). Default is 2.
        batch_size (int): Per-microbatch batch size ``B``. Default is 8192, which
            with ``stage_dim`` gives a 128 MiB cross-stage activation (past NCCL's
            ~64 MiB P2P buffer cliff).
        num_microbatches (int): Number of microbatches per 1F1B pass.
            Default is 8.
        num_tables (int): Embedding tables per stage (one feature each). Default is 8.
        num_embeddings (int): Rows per embedding table. Default is 1000000.
        emb_dim (int): Embedding dimension ``D``. Default is 256.
        num_float_features (int): Width of each stage's dense/float feature input.
            Default is 64.
        stage_dim (int): Width of the activation carried between stages.
            Default is 4096 (128 MiB activation with the default batch size).
        lr (float): Learning rate for the per-stage SGD optimizer.
            Default is 0.05.
        shard_embeddings (bool): Shard each stage's embedding tables within its
            HSD via DMP scoped to the stage's process group (real intra-HSD
            embedding sharding; the lookup all-to-all stays local to the pg).
            Default is True. Requires ``device_type="cuda"`` (nccl) -- gloo cannot
            P2P the CUDA activations between stages.
        output_json (bool): Print the result as JSON instead of a table.
            Default is False.
        all_rank_traces (bool): Export a profiler trace from every rank (not just
            rank 0) so each stage's HSD is captured. Default is True. Traces are
            only emitted when ``profile_dir`` is set and ``device_type`` is
            "cuda".
    """

    name: str = "maglev"
    world_size: int = 8
    num_benchmarks: int = 12
    num_profiles: int = 2
    num_stages: int = 4
    ranks_per_stage: int = 2
    # Heavy default: batch_size * stage_dim * 4B = 8192 * 4096 * 4 = 128 MiB
    # cross-stage activation, past NCCL's ~64 MiB P2P buffer cliff, so the
    # hand-off exercises the large-payload (rendezvous) path -- see the parity
    # handoff pgs in stage.py and tech-docs/nccl_p2p_execution_order_buffer_size.md.
    batch_size: int = 8192
    num_microbatches: int = 8
    # Heavier embedding tables: 1M rows * 256 dim * 8 tables per stage.
    num_tables: int = 8
    num_embeddings: int = 1_000_000
    emb_dim: int = 256
    num_float_features: int = 64
    stage_dim: int = 4096
    lr: float = 0.05
    # Intra-HSD embedding sharding (DMP per stage), scoped to each stage's pg so
    # the lookup all-to-all stays local to the HSD. Requires nccl (cuda).
    shard_embeddings: bool = True
    output_json: bool = False
    # Capture a trace from every rank so each stage's HSD shows up, not only
    # rank 0's stage (traces are emitted only when profile_dir is set on CUDA).
    all_rank_traces: bool = True
    # Benchmark on GPU by default (nccl P2P for the cross-HSD hand-off); the
    # profiler path is CUDA-only. Pass --device_type=cpu for the portable
    # gloo repro (no traces on that path).
    device_type: str = "cuda"
    profile_dir: str = "."


def _make_tables(
    stage_index: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
) -> List[EmbeddingBagConfig]:
    """This stage's feature partition: ``num_tables`` disjoint 1-feature tables.

    Reuses the shared :class:`EmbeddingTablesConfig` generator, namespaced per
    stage via ``name_prefix`` so each stage's tables/features are disjoint.
    """
    return EmbeddingTablesConfig(
        num_unweighted_features=num_tables,
        num_weighted_features=0,
        embedding_feature_dim=emb_dim,
        base_row_size=num_embeddings,
    ).generate_tables(name_prefix=f"s{stage_index}_")[0]


def _make_input(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    num_float_features: int,
    device: torch.device,
) -> ModelInput:
    """A ModelInput (float + sparse) for the given tables (canonical generator)."""
    return ModelInput.generate(
        batch_size=batch_size,
        tables=tables,
        weighted_tables=[],
        num_float_features=num_float_features,
        device=device,
    )


# single-rank runner
def runner(
    rank: int,
    world_size: int,
    run_option: RunOptions,
) -> BenchmarkResult:
    run_option.set_log_level()

    num_stages = run_option.num_stages
    ranks_per_stage = run_option.ranks_per_stage
    assert world_size == num_stages * ranks_per_stage, (
        f"world_size ({world_size}) must equal num_stages ({num_stages}) * "
        f"ranks_per_stage ({ranks_per_stage})"
    )

    # CUDA uses nccl for the cross-HSD P2P hand-off (falls back to gloo for the
    # CPU repro). The profiler path is CUDA-only, so traces require device_type=cuda.
    backend = "cpu:gloo,cuda:nccl" if run_option.device_type == "cuda" else "gloo"
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend) as ctx:
        device = ctx.device
        criterion = nn.MSELoss()

        # HSD layout: contiguous rank groups, one stage per HSD.
        stage_ranks: List[List[int]] = [
            list(range(s * ranks_per_stage, (s + 1) * ranks_per_stage))
            for s in range(num_stages)
        ]

        # Collective: every rank builds every stage's process group.
        stage_pgs = build_stage_process_groups(stage_ranks)
        # Build hand-off pgs up front (before any DMP sharding) so all new_group
        # collectives stay contiguous and cannot deadlock against sharding comms.
        handoff_pgs = build_handoff_process_groups(stage_ranks)
        my_stage_index = rank // ranks_per_stage

        # This rank's stage (weights seeded by stage index so an HSD's two DP
        # ranks start identical -> grad all-reduce keeps them in lock-step).
        tables = _make_tables(
            my_stage_index,
            run_option.num_tables,
            run_option.num_embeddings,
            run_option.emb_dim,
        )
        torch.manual_seed(_WEIGHT_SEED + my_stage_index)
        module = MaglevTestStage(
            tables=tables,
            stage_dim=run_option.stage_dim,
            is_first=(my_stage_index == 0),
            num_float_features=run_option.num_float_features,
            device=device,
        )
        parallelizer = (
            EmbeddingShard(embedding_lr=run_option.lr)
            if run_option.shard_embeddings
            else Replicated()
        )
        stage = StageWrapper(
            module=module,
            stage_pg=stage_pgs[my_stage_index],
            stage_index=my_stage_index,
            parallelizer=parallelizer,
        )
        pipeline = MaglevPipeline(
            stage=stage,
            stage_ranks=stage_ranks,
            global_rank=rank,
            activation_shape=torch.Size([run_option.batch_size, run_option.stage_dim]),
            device=device,
            handoff_pgs=handoff_pgs,
        )

        # Sharded: CombinedOptimizer(fused embedding + dense SGD). Replicated:
        # a plain SGD. (Citrine C2: foreach=True in the dense/replicated SGD.)
        optimizer = stage.configure_optimizer(run_option.lr)

        # Microbatch inputs for this rank's stage; labels only on the last stage.
        micro_inputs: List[ModelInput] = [
            _make_input(
                tables,
                run_option.batch_size,
                run_option.num_float_features,
                device,
            )
            for _ in range(run_option.num_microbatches)
        ]
        labels: List[torch.Tensor] = []
        if pipeline.is_last:
            labels = [
                torch.randn(run_option.batch_size, run_option.stage_dim, device=device)
                for _ in range(run_option.num_microbatches)
            ]

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            pipeline: MaglevPipeline,
            optimizer: torch.optim.Optimizer,
            micro_inputs: List[ModelInput],
            labels: List[torch.Tensor],
            criterion: nn.Module,
        ) -> None:
            # One measured iteration = one full 1F1B pass over all microbatches
            # (warmup + steady 1F1B + cooldown), including the cross-HSD hand-off
            # and the per-batch DP grad all-reduce + optimizer step.
            run_1f1b(
                pipeline=pipeline,
                microbatch_inputs=micro_inputs,
                optimizer=optimizer,
                labels=labels if pipeline.is_last else None,
                criterion=criterion if pipeline.is_last else None,
            )

        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={
                "pipeline": pipeline,
                "optimizer": optimizer,
                "micro_inputs": micro_inputs,
                "labels": labels,
                "criterion": criterion,
            },
            # One 1F1B pass consumes num_microbatches per-rank batches.
            sample_count=run_option.batch_size * run_option.num_microbatches,
            **run_option.benchmark_func_kwargs(rank=rank),
        )

        return result


# a standalone function to run the benchmark in multi-process mode
def run_maglev(run_option: RunOptions) -> BenchmarkResult:
    benchmark_res_per_rank = run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        run_option=run_option,
    )

    # Combine results from all ranks: timing from rank 0, memory stats per rank.
    world_size = run_option.world_size
    total_benchmark_res = BenchmarkResult(
        short_name=benchmark_res_per_rank[0].short_name,
        gpu_elapsed_time=benchmark_res_per_rank[0].gpu_elapsed_time,
        cpu_elapsed_time=benchmark_res_per_rank[0].cpu_elapsed_time,
        cpu_utilization=benchmark_res_per_rank[0].cpu_utilization,
        normalized_cpu_utilization=benchmark_res_per_rank[0].normalized_cpu_utilization,
        gpu_mem_stats=[
            GPUMemoryStats(rank, 0, 0, 0, 0, 0) for rank in range(world_size)
        ],
        cpu_mem_stats=[CPUMemoryStats(rank, 0) for rank in range(world_size)],
        qps=benchmark_res_per_rank[0].qps,
        rank=0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult carries at most 1 GPU and 1 CPU measurement.
        if len(res.gpu_mem_stats) > 0:
            total_benchmark_res.gpu_mem_stats[res.rank] = res.gpu_mem_stats[0]
        if len(res.cpu_mem_stats) > 0:
            total_benchmark_res.cpu_mem_stats[res.rank] = res.cpu_mem_stats[0]

    return total_benchmark_res


# command-line interface
@cmd_conf
def main(run_option: RunOptions) -> None:
    result = run_maglev(run_option=run_option)

    # Print from the parent process so the result is always visible (rank-0
    # worker logging has no stream handler under the spawn pool).
    if run_option.output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.prettify())
        print(f"\nMarkdown format:\n{result}")


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
