#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for the Maglev staged pipeline (MVP).

Measures a microbatched 1F1B schedule, selected by ``--pipeline``:
``1f1b-recv-ahead`` posts each receive a microbatch ahead of the compute it feeds
(:class:`~torchrec.distributed.maglev.pipeline.Maglev1F1BRecvAhead`, the default)
and ``1f1b`` posts it immediately before its wait
(:class:`~torchrec.distributed.maglev.pipeline.Maglev1F1B`). The model is a
``sum(layers_per_stage)``-layer model authored on ``meta`` and cut across
per-stage process groups (one hardware scale-up domain, HSD, each). One measured
iteration is the input-dist all-to-all plus one full 1F1B pass over
``num_microbatches`` microbatches, including the cross-HSD activation / gradient
hand-off and the DP grad all-reduce + optimizer step.

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

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import cast, Dict, Iterator, List, Optional, Type

logger: logging.Logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
    CPUMemoryStats,
    GPUMemoryStats,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.maglev.module import MaglevLayer
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    Maglev1F1BRecvAhead,
    MaglevPipelineBase,
)
from torchrec.distributed.maglev.stage import remap_plan_to_process_group, StageWrapper
from torchrec.distributed.model_parallel import (
    DefaultDataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.model_input import ModelInput
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.table_config import EmbeddingTablesConfig
from torchrec.distributed.test_utils.test_model import MaglevTestLayer, MaglevTestModel
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.optimizers import in_backward_optimizer_filter

# Weights are seeded per stage, just before the stage materializes, so an HSD's
# DP ranks start identical -- nothing syncs them afterwards.
# Inputs are left unseeded -- values don't affect a throughput benchmark.
_WEIGHT_SEED = 100

# Selectable schedules, by --pipeline. "base" runs one microbatch per pass and
# so is the un-pipelined reference the 1F1B variants are worth measuring against;
# the two 1F1B entries move identical data and differ only in where the receives
# are posted. sample_count is taken from the pipeline's microbatches_per_pass, so
# throughput stays comparable across all three.
_PIPELINE_CLS: Dict[str, Type[MaglevPipelineBase]] = {
    "base": MaglevPipelineBase,
    "1f1b": Maglev1F1B,
    "1f1b-recv-ahead": Maglev1F1BRecvAhead,
}


def _shard_embeddings_in_hsd(
    module: nn.Module,
    stage_pg: dist.ProcessGroup,
    device: torch.device,
    embedding_lr: float,
) -> nn.Module:
    """Shard a stage's ``EmbeddingBagCollection`` within its HSD, via DMP.

    Plain TorchRec, scoped to a sub-process-group -- nothing here is Maglev's, and
    a caller is free to use any other wrapper (FSDP, none) instead. It lives in
    the benchmark rather than the library for exactly that reason.

    The tables are sharded over ``stage_pg``, so the lookup all-to-all stays local
    to the HSD with no global cross-HSD exchange, and the embedding optimizer is
    fused into the TBE backward. Dense params are replicated in DDP over the same
    group and reduced once per pass, the schedule having suppressed DDP's sync on
    every backward but the last.
    """
    sharders: List[ModuleSharder[nn.Module]] = [
        cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder())
    ]
    # Fuse the embedding optimizer into the TBE backward *before* DMP so the
    # sharder builds it into the kernel (sparse params train in backward).
    emb_params = [
        p
        for m in module.modules()
        if isinstance(m, EmbeddingBagCollection)
        for p in m.parameters()
    ]
    apply_optimizer_in_backward(torch.optim.SGD, emb_params, {"lr": embedding_lr})

    env = ShardingEnv.from_process_group(stage_pg)
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=stage_pg.size(), compute_device=device.type)
    )
    # NOTE: not planner.collective_plan(): it broadcasts from group-local rank 0
    # but passes it as a *global* broadcast src, which deadlocks for any stage
    # whose pg does not contain global rank 0 (stages 1..N-1). Compute the plan on
    # the pg's leader and broadcast with the correct global src.
    if stage_pg.rank() == 0:
        plan_holder: List[Optional[ShardingPlan]] = [planner.plan(module, sharders)]
    else:
        plan_holder = [None]
    if stage_pg.size() > 1:
        dist.broadcast_object_list(
            plan_holder, src=dist.get_global_rank(stage_pg, 0), group=stage_pg
        )
    plan = plan_holder[0]
    assert plan is not None
    remap_plan_to_process_group(plan, stage_pg, device)
    return DistributedModelParallel(
        module=module,
        env=env,
        device=device,
        plan=plan,
        sharders=sharders,
        # Dense params go into DDP over the stage pg; the schedule suppresses its
        # sync around every forward but the pass's last, so one all-reduce lands
        # per pass. static_graph is off: it requires the reducer to be armed on
        # the first backward, which cannot hold when the first microbatch's
        # forward is deliberately inside no_sync.
        init_data_parallel=True,
        data_parallel_wrapper=DefaultDataParallelWrapper(static_graph=False),
        init_parameters=True,
    )


@dataclass
class RunOptions(BenchFuncConfig):
    """Configuration for the Maglev pipeline benchmark.

    Args:
        name (str): Human-readable benchmark name. Default is "maglev".
        world_size (int): Total number of ranks. Must equal
            ``num_stages * ranks_per_stage``. Default is 8.
        num_benchmarks (int): Number of measured iterations (each one full pass).
            Default is 0 -- a plain run only profiles, emitting traces and no
            timings. Pass e.g. ``--num_benchmarks=12`` to measure.
        num_profiles (int): Number of profiling iterations (CUDA only; unused on
            the CPU path). Default is 2.
        layers_per_stage (List[int]): How the model's layers are cut across the
            pipeline -- one entry per stage, so this also sets the stage count
            (one HSD each) and the model depth (its sum). Default is
            ``[2, 3, 4, 3]``: 12 layers over 4 stages, deliberately uneven, since
            only stage boundaries cross the network and a real cut is rarely
            uniform.
        ranks_per_stage (int): Ranks per stage's HSD (intra-HSD data
            parallelism). Default is 2.
        batch_size (int): Per-microbatch batch size ``B``. Default is 8192, which
            with ``layer_dim`` gives a 128 MiB cross-stage activation (past NCCL's
            ~64 MiB P2P buffer cliff).
        num_microbatches (int): Number of microbatches per 1F1B pass.
            Default is 8.
        pipeline (str): Which schedule to measure. Options:
            - "base": one microbatch per pass, no pipelining -- the reference the
              1F1B variants are worth measuring against
            - "1f1b": each receive posted immediately before its wait
            - "1f1b-recv-ahead": each receive posted a microbatch ahead of the
              compute it feeds
            The two 1F1B variants move identical data; only the receive placement
            differs. Default is "1f1b-recv-ahead".
        num_tables (int): Embedding tables per layer (one feature each). Default is 8.
        num_embeddings (int): Rows per embedding table. Default is 1000000.
        emb_dim (int): Embedding dimension ``D``. Default is 256.
        num_float_features (int): Width of each stage's dense/float feature input.
            Default is 64.
        layer_dim (int): Width of the activation carried between layers (and so
            between stages). Default is 4096 (128 MiB activation with the default
            batch size).
        lr (float): Learning rate for the per-stage SGD optimizer.
            Default is 0.05.
        shard_embeddings (bool): Shard each stage's embedding tables within its
            HSD via DMP scoped to the stage's process group (real intra-HSD
            embedding sharding; the lookup all-to-all stays local to the pg).
            Default is True. Requires ``device_type="cuda"`` (nccl) -- gloo cannot
            P2P the CUDA activations between stages.
        output_json (bool): Print the result as JSON instead of a table.
            Default is False.
        debug_mode (bool): Attach a debugger before running. Every rank attaches,
            so pair it with a small ``world_size`` unless you want eight
            simultaneous sessions. Default is False.
        all_rank_traces (bool): Export a profiler trace from every rank (not just
            rank 0) so each stage's HSD is captured. Default is True. Traces are
            only emitted when ``profile_dir`` is set and ``device_type`` is
            "cuda".
    """

    name: str = "maglev"
    world_size: int = 8
    num_benchmarks: int = 1
    num_profiles: int = 2
    layers_per_stage: List[int] = field(default_factory=lambda: [2, 3, 4, 3])
    ranks_per_stage: int = 2
    # Heavy default: batch_size * layer_dim * 4B = 8192 * 4096 * 4 = 128 MiB
    # cross-stage activation, past NCCL's ~64 MiB P2P buffer cliff, so the
    # hand-off exercises the large-payload (rendezvous) path -- see the parity
    # handoff pgs in stage.py and tech-docs/nccl_p2p_execution_order_buffer_size.md.
    batch_size: int = 8192
    num_microbatches: int = 8
    # Which schedule to measure; see _PIPELINE_CLS. The two move identical data
    # and differ only in where the receives are posted, so this is the knob that
    # isolates what running the receives ahead is worth.
    pipeline: str = "1f1b-recv-ahead"
    # Heavier embedding tables: 1M rows * 256 dim * 8 tables per layer.
    num_tables: int = 8
    num_embeddings: int = 1_000_000
    emb_dim: int = 256
    num_float_features: int = 64
    layer_dim: int = 4096
    lr: float = 0.05
    # Intra-HSD embedding sharding (DMP per stage), scoped to each stage's pg so
    # the lookup all-to-all stays local to the HSD. Requires nccl (cuda).
    shard_embeddings: bool = True
    output_json: bool = False
    debug_mode: bool = False
    # Capture a trace from every rank so each stage's HSD shows up, not only
    # rank 0's stage (traces are emitted only when profile_dir is set on CUDA).
    all_rank_traces: bool = True
    # Benchmark on GPU by default (nccl P2P for the cross-HSD hand-off); the
    # profiler path is CUDA-only. Pass --device_type=cpu for the portable
    # gloo repro (no traces on that path).
    device_type: str = "cuda"
    profile_dir: str = "."

    def generate_pipeline(
        self, stage: StageWrapper, optimizer: torch.optim.Optimizer
    ) -> MaglevPipelineBase:
        """Build the schedule named by :attr:`pipeline`.

        Raises:
            ValueError: if :attr:`pipeline` is not a known schedule.
        """
        if self.pipeline not in _PIPELINE_CLS:
            raise ValueError(
                f"unknown pipeline {self.pipeline!r}; expected one of "
                f"{sorted(_PIPELINE_CLS)}"
            )
        match self.pipeline:
            case "base":
                # One microbatch per pass; takes no microbatch count.
                return MaglevPipelineBase(stage=stage, optimizer=optimizer)
            case _:
                # Every non-"base" entry is a Maglev1F1B subclass, which is what
                # makes num_microbatches a valid argument.
                Pipeline = cast(Type[Maglev1F1B], _PIPELINE_CLS[self.pipeline])
                return Pipeline(
                    stage=stage,
                    optimizer=optimizer,
                    num_microbatches=self.num_microbatches,
                )


def _make_tables(
    layer_index: int,
    num_tables: int,
    num_embeddings: int,
    emb_dim: int,
) -> List[EmbeddingBagConfig]:
    """This layer's feature partition: ``num_tables`` disjoint 1-feature tables.

    Reuses the shared :class:`EmbeddingTablesConfig` generator, namespaced per
    layer via ``name_prefix`` so each layer's tables/features are disjoint.
    """
    return EmbeddingTablesConfig(
        num_unweighted_features=num_tables,
        num_weighted_features=0,
        embedding_feature_dim=emb_dim,
        base_row_size=num_embeddings,
    ).generate_tables(name_prefix=f"l{layer_index}_")[0]


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
        # Citrine C0: pin host input memory for efficient CPU-to-GPU transfer.
        pin_memory=True,
    )


# single-rank runner
def runner(
    rank: int,
    world_size: int,
    run_option: RunOptions,
) -> BenchmarkResult:
    # debug mode only works with vscode for now.
    if run_option.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

    run_option.set_log_level()

    # The cut is the source of truth: it fixes the stage count and the depth.
    layers_per_stage = list(run_option.layers_per_stage)
    num_stages = len(layers_per_stage)
    num_layers = sum(layers_per_stage)
    ranks_per_stage = run_option.ranks_per_stage
    assert world_size == num_stages * ranks_per_stage, (
        f"world_size ({world_size}) must equal len(layers_per_stage) "
        f"({num_stages}) * ranks_per_stage ({ranks_per_stage})"
    )

    # CUDA uses nccl for the cross-HSD P2P hand-off (falls back to gloo for the
    # CPU repro). The profiler path is CUDA-only, so traces require device_type=cuda.
    backend = "cpu:gloo,cuda:nccl" if run_option.device_type == "cuda" else "gloo"
    with MultiProcessContext(rank=rank, world_size=world_size, backend=backend) as ctx:
        device = ctx.device

        # HSD layout is implicit in stage_size: stage i is the contiguous rank
        # block starting at i * ranks_per_stage. StageWrapper builds every process
        # group (stage / hand-off / cascade) itself, before it shards.
        my_stage_index, position = StageWrapper.locate_rank(ranks_per_stage, rank)

        # All layers' table configs (identical on every rank) so each rank can
        # generate a full per-stage input set for the input-dist all-to-all.
        all_tables = [
            _make_tables(
                l,
                run_option.num_tables,
                run_option.num_embeddings,
                run_option.emb_dim,
            )
            for l in range(num_layers)
        ]
        # Every rank authors the whole model and lets StageWrapper keep its own
        # stage. Layers this rank does not own are built on the ``meta`` device:
        # StageWrapper drops them, and meta tensors have no storage, so they cost
        # nothing -- which matters here because a materialized layer is 1M x 256 x
        # 8 tables. Weights are seeded by layer index so an HSD's two DP ranks
        # start identical -> the grad all-reduce keeps them in lock-step.
        # Author the whole model on meta -- no storage, so holding every layer
        # costs nothing -- then let StageWrapper keep this rank's stage and
        # materialize only those layers on the device.
        meta_device = torch.device("meta")
        layers: List[MaglevLayer] = [
            MaglevTestLayer(
                tables=all_tables[layer_index],
                layer_dim=run_option.layer_dim,
                is_first=(layer_index == 0),
                batch_size=run_option.batch_size,
                num_float_features=run_option.num_float_features,
                device=meta_device,
            )
            for layer_index in range(num_layers)
        ]
        model = MaglevTestModel(layers)
        # Meta construction consumes no randomness, so the weights are drawn when
        # the stage materializes: seed here, per stage, so an HSD's ranks start
        # identical (the sharded path relies on that -- it does not sync weights).
        torch.manual_seed(_WEIGHT_SEED + my_stage_index)
        stage = StageWrapper(
            model=model,
            layers_per_stage=layers_per_stage,
            stage_size=ranks_per_stage,
        )
        # Parallelism is the caller's: shard the embeddings within the HSD, then
        # materialize. Wrapping before to() is what keeps a meta-authored stage
        # from allocating full-size tables the sharder is about to cut up.
        if run_option.shard_embeddings:
            stage.module = _shard_embeddings_in_hsd(
                stage.module, stage.stage_pg, device, run_option.lr
            )
        stage.to(device)
        # Dense params only, as benchmark_train_pipeline does: when the stage is
        # sharded the embeddings train in backward via the fused TBE optimizer, so
        # they neither need nor belong in the outer step.
        # (Citrine C2: foreach=True for multi-tensor execution.)
        optimizer = torch.optim.SGD(
            [
                p
                for _, p in in_backward_optimizer_filter(
                    stage.module.named_parameters()
                )
            ],
            lr=run_option.lr,
            foreach=True,
        )
        pipeline = run_option.generate_pipeline(stage=stage, optimizer=optimizer)

        # One CPU batch, replayed forever: data loading is not what we benchmark,
        # so the dataloader hands back the same pre-built batch every round. A
        # batch here is one ModelInput per layer of the whole model, which is what
        # the model's (passthrough) preproc consumes. Input distribution owns the
        # H2D copy, as it does for a real dataloader.
        cpu_device = torch.device("cpu")
        model_input = [
            _make_input(
                all_tables[l],
                run_option.batch_size,
                run_option.num_float_features,
                cpu_device,
            )
            for l in range(num_layers)
        ]
        dataloader_iter: Iterator[List[ModelInput]] = itertools.repeat(model_input)

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            pipeline: MaglevPipelineBase,
            dataloader_iter: Iterator[List[ModelInput]],
        ) -> None:
            # One measured iteration = the input-dist all-to-all for this pass's
            # microbatches (refilling the queue as needed) + one full pass,
            # including the cross-HSD hand-off and the per-batch DP grad
            # all-reduce + optimizer step.
            pipeline.progress(dataloader_iter)

        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={
                "pipeline": pipeline,
                "dataloader_iter": dataloader_iter,
            },
            # A pass consumes microbatches_per_pass per-rank batches -- which is
            # num_microbatches for 1F1B but 1 for the base schedule, so this has
            # to come from the pipeline, not the config.
            sample_count=run_option.batch_size * pipeline.microbatches_per_pass,
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
    if run_option.debug_mode:
        # pyrefly: ignore[missing-module-attribute]
        from fbvscode import attach_debugger

        attach_debugger()

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
