#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Benchmark for the Maglev staged pipeline (MVP).

Measures the schedule selected by ``--pipeline``. ``1f1b`` is the default;
``rail`` batches each layer's sparse work over the whole pass before running a
dense zero-bubble schedule. ``data_parallel`` is the baseline without a
pipeline: every rank runs the whole model on its share of the same global batch,
with the tables sharded over all ranks. ``--dp_mode`` picks the dense halves'
data parallelism under each: DDP, or FSDP2 ``fully_shard`` without or with
resharding after each forward.

The model is a ``sum(layers_per_stage)``-layer model authored on ``meta``; a
pipeline cuts it across per-stage process groups (one hardware scale-up domain,
HSD, each). One measured iteration is ``num_stages`` passes, so it covers whole
input-distribution rounds whatever the schedule: per pass the cross-HSD
activation and gradient handoff, backward, the gradient reduction and the
optimizer step. Under ``data_parallel`` it is one step on a pass's global batch.
QPS divides an iteration's samples, every data-parallel replica's rows, by rank
0's GPU time, so it compares across ``--pipeline`` values.

Runs on GPU + nccl by default (the cross-HSD P2P hand-off and the profiler path
are CUDA-only); `all_rank_traces` defaults on so every stage's HSD shows up in
the traces. Pass ``--device_type=cpu`` for the portable gloo repro (no traces on
that path).

Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_maglev -- --profile_dir=/tmp/maglev_traces

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_maglev --profile_dir=/tmp/maglev_traces

The data-parallel arms of one model and global batch: ``--pipeline`` ``1f1b`` or
``data_parallel``, each with ``--dp_mode`` ``ddp``, ``zero2`` or ``zero3``
(see ``_DP_MODES``).
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple, Type

logger: logging.Logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
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
from torchrec.distributed.maglev.module import MaglevLayer, StructuredActivationsLayout
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    Maglev1F1BRecvAhead,
    MaglevPipelineBase,
    MaglevRail,
)
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevProcessGroups,
    remap_plan_to_process_group,
    StageWrapper,
)
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
from torchrec.distributed.test_utils.test_model import (
    MaglevTestActivations,
    MaglevTestLayer,
    MaglevTestModel,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.optimizers import in_backward_optimizer_filter

# Weights are seeded per stage, just before the stage materializes, so an HSD's
# DP ranks start identical -- nothing syncs them afterwards.
# Inputs are left unseeded -- values don't affect a throughput benchmark.
_WEIGHT_SEED = 100

# Selectable schedules, by --pipeline. "base" runs one microbatch per pass,
# forwarding all then backwarding all. The recv-ahead name is kept for callers
# of the earlier experimental schedule. --pipeline=data_parallel is the
# baseline without a pipeline (see _PIPELINES).
_PIPELINE_CLS: Dict[str, Type[MaglevPipelineBase]] = {
    "base": MaglevPipelineBase,
    "1f1b": Maglev1F1B,
    "1f1b-recv-ahead": Maglev1F1BRecvAhead,
    "rail": MaglevRail,
}
_PIPELINES: tuple[str, ...] = (*_PIPELINE_CLS, "data_parallel")


# The dense halves' data parallelism, and what each costs per pass:
#
#   ddp    DMP's DDP: 1 AllReduce per bucket, after the pass under a pipeline
#   zero2  fully_shard(reshard_after_forward=False): 1 AllGather + 1
#          ReduceScatter per pass
#   zero3  fully_shard(reshard_after_forward=True): AllGathers per forward and
#          backward, and 1 ReduceScatter per microbatch (ZeRO-3 gradient memory)
#
# MaglevRail reduces FSDP2 only: it refuses ddp, defaults to zero2, and keeps
# its parameters gathered and reduces once per pass under zero3 too.
_DP_MODES: tuple[str, ...] = ("ddp", "zero2", "zero3")
# The schedules that reduce DDP only, as on master; Maglev1F1B and
# data_parallel take every mode, and MaglevRail the FSDP2 modes.
_DDP_ONLY_PIPELINES: tuple[str, ...] = ("base", "1f1b-recv-ahead")


def _shard_and_wrap(
    module: nn.Module,
    pg: dist.ProcessGroup,
    device: torch.device,
    embedding_lr: float,
    dp_mode: str = "ddp",
    pipelined: bool = True,
) -> nn.Module:
    """Shard ``module``'s ``EmbeddingBagCollection`` tables over ``pg``, and data-parallelize its dense halves over it.

    Plain TorchRec, scoped to a process group -- nothing here is Maglev's, and
    a caller is free to use any other wrapper instead. It lives in the benchmark
    rather than the library for exactly that reason.

    Under a pipeline ``pg`` is a stage's HSD, so the lookup all-to-all stays
    local to the HSD; without one (``pipelined=False``) it spans every rank. The
    tables are sharded via DMP, with the embedding optimizer fused into the TBE
    backward. ``dp_mode`` picks the dense halves' data parallelism: DMP's DDP
    for ``ddp``, or FSDP2 ``fully_shard`` for ``zero2`` (without resharding
    after the forward) and ``zero3`` (with). Without a pipeline DDP keeps its
    default, overlapping its bucket all-reduces with the backward.

    The FSDP2 wrap goes on each layer's **dense half**, not the layer: applied to a
    whole layer it would try to shard the table DMP has already sharded, and
    ``aten._embedding_bag`` rejects a DTensor weight against plain indices.
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

    env = ShardingEnv.from_process_group(pg)
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size=pg.size(), compute_device=device.type)
    )
    # NOTE: not planner.collective_plan(): it broadcasts from group-local rank 0
    # but passes it as a *global* broadcast src, which deadlocks for any stage
    # whose pg does not contain global rank 0 (stages 1..N-1). Compute the plan on
    # the pg's leader and broadcast with the correct global src.
    if pg.rank() == 0:
        plan_holder: List[Optional[ShardingPlan]] = [planner.plan(module, sharders)]
    else:
        plan_holder = [None]
    if pg.size() > 1:
        dist.broadcast_object_list(
            plan_holder, src=dist.get_global_rank(pg, 0), group=pg
        )
    plan = plan_holder[0]
    assert plan is not None
    remap_plan_to_process_group(plan, pg, device)
    dmp = DistributedModelParallel(
        module=module,
        env=env,
        device=device,
        plan=plan,
        sharders=sharders,
        # A schedule's one flush per pass reduces through DDP's used-parameter
        # map, which needs dynamic unused-parameter tracking, not a static graph.
        init_data_parallel=dp_mode == "ddp",
        data_parallel_wrapper=(
            DefaultDataParallelWrapper(static_graph=False, find_unused_parameters=True)
            if dp_mode == "ddp" and pipelined
            else None
        ),
        init_parameters=True,
    )
    if dp_mode == "ddp":
        return dmp
    dense_mesh = DeviceMesh.from_group(pg, device.type)
    for layer in dmp.modules():
        if isinstance(layer, MaglevLayer) and layer.dense is not None:
            dense = layer.require_dense()
            fully_shard(
                dense, mesh=dense_mesh, reshard_after_forward=dp_mode == "zero3"
            )
    return dmp


@dataclass
class RunOptions(BenchFuncConfig):
    """Configuration for the Maglev pipeline benchmark.

    Args:
        name (str): Human-readable benchmark name. Default is "maglev".
        world_size (int): Total number of ranks. Must equal
            ``num_stages * ranks_per_stage``. Default is 8.
        num_benchmarks (int): Number of measured iterations (each ``num_stages``
            passes, or one ``data_parallel`` step).
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
            ~64 MiB P2P buffer cliff). Rail constructs one whole-pass input of
            ``B * num_microbatches`` samples and splits it after sparse pooling.
        num_microbatches (int): Number of microbatches per 1F1B pass.
            Default is 8.
        pipeline (str): Which schedule to measure. Options:
            - "base": one microbatch per pass, no pipelining -- the reference the
              1F1B variants are worth measuring against
            - "1f1b": the standard bidirectional 1F1B schedule
            - "1f1b-recv-ahead": split-mode 1F1B with receives posted one
              microbatch ahead; shared mode uses its batched crossover schedule
            - "rail": one whole-pass sparse lookup per layer followed by a
              zero-bubble dense pipeline over ``num_microbatches``
            - "data_parallel": no pipeline; every rank trains the whole model,
              tables sharded over all ranks, on ``1 / world_size`` of the
              pipelines' global batch per step. Requires ``shard_embeddings``.
            Default is "1f1b".
        dp_mode (Optional[str]): The dense halves' data parallelism, when
            ``shard_embeddings``: "ddp" (DMP's DDP), or FSDP2 "zero2"/"zero3"
            (``fully_shard`` without/with resharding after each forward).
            ddp/zero2 reduce once per pass; zero3 reduce-scatters per
            microbatch for ZeRO-3 gradient memory. See
            ``_DP_MODES``. "base" and "1f1b-recv-ahead" take "ddp" only,
            "rail" "zero2"/"zero3" only. Default is None: "ddp", except
            "zero2" under "rail", which has no DDP path.
        handoff_pg_mode (str): Whether activation and gradient traffic uses
            separate process groups ("split", the default) or one shared process
            group with batched steady-state exchanges ("shared").
        num_tables (int): Embedding tables per layer (one feature each). Default is 8.
        num_embeddings (int): Rows per embedding table. Default is 1000000.
        emb_dim (int): Embedding dimension ``D``. Default is 256.
        num_float_features (int): Width of each stage's dense/float feature input.
            Default is 64.
        pooling_avg (int): Average sparse indices per sample and feature.
            Default is 10.
        layer_dim (int): Width of the activation carried between layers (and so
            between stages). Default is 4096 (128 MiB activation with the default
            batch size).
        lr (float): Learning rate for the per-stage SGD optimizer.
            Default is 0.05.
        shard_embeddings (bool): Shard each stage's
            embedding tables within its HSD via DMP scoped to the stage's process
            group (real intra-HSD embedding sharding; the lookup all-to-all stays
            local to the pg).
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
    # hand-off exercises the large-payload (rendezvous) path.
    batch_size: int = 8192
    num_microbatches: int = 8
    # What to measure; see _PIPELINES.
    pipeline: str = "1f1b"
    # Split preserves the established direction-specific communicator topology;
    # shared exercises the batch_isend_irecv handoff path.
    handoff_pg_mode: str = "split"
    # How the dense halves are data-parallelized, when --shard_embeddings.
    # See _DP_MODES.
    dp_mode: Optional[str] = None
    # Heavier embedding tables: 1M rows * 256 dim * 8 tables per layer.
    num_tables: int = 8
    num_embeddings: int = 1_000_000
    emb_dim: int = 256
    num_float_features: int = 64
    pooling_avg: int = 10
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

    def data_parallelism(self) -> str:
        """:attr:`dp_mode`, or the default of :attr:`pipeline` when unset.

        Raises:
            ValueError: if the mode is unknown or :attr:`pipeline` cannot
                reduce it.
        """
        # Rail has no DDP path, so it defaults to an FSDP2 mode.
        dp_mode = self.dp_mode or ("zero2" if self.pipeline == "rail" else "ddp")
        if dp_mode not in _DP_MODES:
            raise ValueError(
                f"unknown dp_mode {dp_mode!r}; expected one of {list(_DP_MODES)}"
            )
        if self.pipeline in _DDP_ONLY_PIPELINES and dp_mode != "ddp":
            raise ValueError(f"--pipeline={self.pipeline} reduces DDP only")
        if self.pipeline == "rail" and dp_mode == "ddp":
            raise ValueError("--pipeline=rail reduces FSDP2 only")
        return dp_mode

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
                # One input-distribution round per pass; takes no configured
                # microbatch count.
                return MaglevPipelineBase(stage=stage, optimizer=optimizer)
            case "rail":
                return MaglevRail(
                    stage=stage,
                    optimizer=optimizer,
                    num_microbatches=self.num_microbatches,
                )
            case _:
                # The remaining entries are Maglev1F1B subclasses.
                Pipeline = cast(Type[Maglev1F1B], _PIPELINE_CLS[self.pipeline])
                return Pipeline(
                    stage=stage,
                    optimizer=optimizer,
                    num_microbatches=self.num_microbatches,
                )


def _rows_per_pass(run_option: "RunOptions") -> int:
    """A pass's rows across every data-parallel replica of a stage."""
    microbatches = (
        len(run_option.layers_per_stage)
        if run_option.pipeline == "base"
        else run_option.num_microbatches
    )
    return run_option.ranks_per_stage * run_option.batch_size * microbatches


def _samples_per_iteration(run_option: "RunOptions") -> int:
    """The rows one measured iteration trains: ``num_stages`` passes, or one
    ``data_parallel`` step on a pass's global batch."""
    if run_option.pipeline == "data_parallel":
        return _rows_per_pass(run_option)
    return len(run_option.layers_per_stage) * _rows_per_pass(run_option)


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
    pooling_avg: int,
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
        pooling_avg=pooling_avg,
    )


def _dense_optimizer(
    module: nn.Module, run_option: RunOptions
) -> torch.optim.Optimizer:
    """The outer optimizer, over the dense parameters only.

    As benchmark_train_pipeline does: when the tables are sharded the embeddings
    train in backward via the fused TBE optimizer, so they neither need nor
    belong in the outer step.
    """
    dense_params = [
        p for _, p in in_backward_optimizer_filter(module.named_parameters())
    ]
    # (Citrine C2: foreach=True for multi-tensor execution.)
    return torch.optim.SGD(dense_params, lr=run_option.lr, foreach=True)


def _replayed_batch(
    all_tables: List[List[EmbeddingBagConfig]], rows: int, run_option: RunOptions
) -> List[ModelInput]:
    """A CPU batch of ``rows`` rows, to replay forever.

    Data loading is not what we benchmark, so every step reads the same
    pre-built batch: one ModelInput per layer of the whole model, which is what
    the model's (passthrough) preproc consumes. The run owns the H2D copy, as it
    does for a real dataloader.
    """
    return [
        _make_input(
            tables,
            rows,
            run_option.num_float_features,
            torch.device("cpu"),
            run_option.pooling_avg,
        )
        for tables in all_tables
    ]


# What benchmark_func measures: the function and its keyword arguments.
_Step = Tuple[Callable[..., None], Dict[str, Any]]


def _pipelined(
    rank: int,
    model: MaglevTestModel,
    all_tables: List[List[EmbeddingBagConfig]],
    device: torch.device,
    run_option: RunOptions,
) -> _Step:
    """Build this rank's stage and the schedule :attr:`RunOptions.pipeline` names."""
    layers_per_stage = list(run_option.layers_per_stage)
    num_stages = len(layers_per_stage)
    process_groups = MaglevProcessGroups.from_scratch(
        stage_size=run_option.ranks_per_stage,
        num_stages=num_stages,
        handoff_pg_mode=HandoffPGMode(run_option.handoff_pg_mode),
    )
    my_stage_index, _ = MaglevProcessGroups.locate_rank(
        process_groups.stage_ranks, rank
    )
    # Per stage: an HSD's ranks must start identical, and nothing syncs them.
    torch.manual_seed(_WEIGHT_SEED + my_stage_index)
    stage = StageWrapper(
        model=model,
        layers_per_stage=layers_per_stage,
        stage_size=run_option.ranks_per_stage,
        loss_only_output=run_option.shard_embeddings and run_option.ranks_per_stage > 1,
        process_groups=process_groups,
        enable_rail=run_option.pipeline == "rail",
    )
    # Shard before to(): a meta-authored stage never allocates full tables.
    if run_option.shard_embeddings:
        stage.module = _shard_and_wrap(
            stage.module,
            stage.stage_pg,
            device,
            run_option.lr,
            run_option.data_parallelism(),
        )
    stage.to(device)
    pipeline = run_option.generate_pipeline(
        stage=stage,
        optimizer=_dense_optimizer(stage.module, run_option),
    )
    # A whole-pass schedule reads one batch per pass; the others read one
    # microbatch per input carrier.
    batch = _replayed_batch(
        all_tables,
        run_option.batch_size
        * (run_option.num_microbatches if pipeline.takes_whole_pass_input else 1),
        run_option,
    )

    def _round(
        bench_inputs: List[ModelInput],
        pipeline: MaglevPipelineBase,
        dataloader_iter: Iterator[List[ModelInput]],
    ) -> None:
        # num_stages passes: whole input-dist rounds for every schedule.
        for _ in range(num_stages):
            pipeline.progress(dataloader_iter)

    return _round, {"pipeline": pipeline, "dataloader_iter": itertools.repeat(batch)}


def _data_parallel(
    model: MaglevTestModel,
    all_tables: List[List[EmbeddingBagConfig]],
    device: torch.device,
    run_option: RunOptions,
    world_size: int,
) -> _Step:
    """Build the whole model on every rank, data-parallel over all of them.

    Each rank trains ``1 / world_size`` of the pipelines' global batch per step,
    so every arm trains the same global batch.

    Raises:
        ValueError: Without ``shard_embeddings``, or if the global batch does
            not split evenly over the ranks.
    """
    if not run_option.shard_embeddings:
        raise ValueError(
            "data_parallel shards the tables over every rank: pass "
            "--shard_embeddings=true"
        )
    global_rows = _rows_per_pass(run_option)
    if global_rows % world_size:
        raise ValueError(
            f"the global batch of {global_rows} rows does not split over "
            f"{world_size} ranks"
        )
    world = dist.group.WORLD
    assert world is not None
    # Still on meta: the sharder cuts the tables before any is allocated.
    torch.manual_seed(_WEIGHT_SEED)
    module = _shard_and_wrap(
        model,
        world,
        device,
        run_option.lr,
        run_option.data_parallelism(),
        pipelined=False,
    )
    module.to(device)
    optimizer = _dense_optimizer(module, run_option)
    batch = _replayed_batch(all_tables, global_rows // world_size, run_option)

    def _step(
        bench_inputs: List[ModelInput],
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: List[ModelInput],
    ) -> None:
        # One step on a pass's global batch.
        losses, _ = module(
            [layer_input.to(device, non_blocking=True) for layer_input in batch]
        )
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

    return _step, {"module": module, "optimizer": optimizer, "batch": batch}


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
    if run_option.pipeline not in _PIPELINES:
        raise ValueError(
            f"unknown pipeline {run_option.pipeline!r}; expected one of "
            f"{list(_PIPELINES)}"
        )
    run_option.data_parallelism()
    # The cut is the source of truth: it fixes the stage count, the depth and,
    # with ranks_per_stage, the global batch every arm trains.
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

        # All layers' table configs (identical on every rank) so each rank can
        # generate a full per-stage input set for the input-dist all-to-all.
        all_tables = [
            _make_tables(
                layer_index,
                run_option.num_tables,
                run_option.num_embeddings,
                run_option.emb_dim,
            )
            for layer_index in range(num_layers)
        ]
        # Author the whole model on meta -- no storage, so holding every layer
        # costs nothing, which matters here because a materialized layer is
        # num_tables tables of num_embeddings x emb_dim. A pipeline keeps this
        # rank's stage and materializes only its layers; data_parallel keeps
        # every layer, and the sharder cuts their tables first.
        meta_device = torch.device("meta")
        activation_layout = StructuredActivationsLayout[MaglevTestActivations]()
        layers: List[MaglevTestLayer] = [
            MaglevTestLayer(
                tables=all_tables[layer_index],
                layer_dim=run_option.layer_dim,
                is_first=(layer_index == 0),
                activation_layout=activation_layout,
                num_float_features=run_option.num_float_features,
                device=meta_device,
            )
            for layer_index in range(num_layers)
        ]
        model = MaglevTestModel(layers, activation_layout)
        if run_option.pipeline == "data_parallel":
            step, step_kwargs = _data_parallel(
                model, all_tables, device, run_option, world_size
            )
        else:
            step, step_kwargs = _pipelined(rank, model, all_tables, device, run_option)
        return benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            func_to_benchmark=step,
            benchmark_func_kwargs=step_kwargs,
            sample_count=_samples_per_iteration(run_option),
            **run_option.benchmark_func_kwargs(rank=rank),
        )


# a standalone function to run the benchmark in multi-process mode
def run_maglev(run_option: RunOptions) -> BenchmarkResult:
    benchmark_res_per_rank = run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        run_option=run_option,
    )

    # Combine results from all ranks: timing from rank 0, memory stats per rank.
    world_size = run_option.world_size
    rank0 = benchmark_res_per_rank[0]
    # The workers' QPS is host wall-clock, which misses GPU work still queued
    # when an iteration returns; on CUDA take it from rank 0's GPU time instead.
    gpu_ms = rank0.gpu_elapsed_time
    qps = rank0.qps
    if run_option.device_type == "cuda":
        qps = torch.where(
            gpu_ms > 0,
            _samples_per_iteration(run_option) * 1e3 / gpu_ms,
            torch.zeros_like(gpu_ms),
        )
    total_benchmark_res = BenchmarkResult(
        short_name=rank0.short_name,
        gpu_elapsed_time=gpu_ms,
        cpu_elapsed_time=rank0.cpu_elapsed_time,
        cpu_utilization=rank0.cpu_utilization,
        normalized_cpu_utilization=rank0.normalized_cpu_utilization,
        gpu_mem_stats=[
            GPUMemoryStats(rank, 0, 0, 0, 0, 0) for rank in range(world_size)
        ],
        cpu_mem_stats=[CPUMemoryStats(rank, 0) for rank in range(world_size)],
        qps=qps,
        wall_elapsed_time=rank0.wall_elapsed_time,
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
