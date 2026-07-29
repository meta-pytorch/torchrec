#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
from typing import Any, cast, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


def build_stage_process_groups(
    stage_ranks: List[List[int]],
) -> List[dist.ProcessGroup]:
    """Create one process group per Maglev stage (one HSD each).

    ``dist.new_group`` is a **collective**: every rank in the job must call it
    for every stage, in the same order, even though a rank ends up owning only
    one of the returned groups. Returns the list of stage process groups,
    index-aligned with ``stage_ranks``.

    Args:
        stage_ranks: ``stage_ranks[i]`` is the list of global ranks that make up
            stage ``i``'s HSD.
    """
    pgs: List[dist.ProcessGroup] = []
    for ranks in stage_ranks:
        # Collective across the whole job; returns a handle only ranks in
        # ``ranks`` can actually communicate over. (``new_group`` is typed
        # ProcessGroup | int | None; on member ranks it is a ProcessGroup.)
        pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
        pgs.append(pg)
    return pgs


def _remap_plan_to_pg_devices(
    plan: ShardingPlan, stage_pg: dist.ProcessGroup, device: torch.device
) -> None:
    """Rewrite each shard's placement device to the sub-pg rank's actual device.

    The planner builds the plan over a size-``N`` topology, so it emits
    *group-local* ranks/devices: ``rank:0/cuda:0``, ``rank:1/cuda:1`` for a 2-rank
    stage. But shard placements are interpreted against the *global* rank space
    (DMP maps ``placement.rank()`` back through the pg, and the shard tensor lives
    on the process's actual device). For a stage whose ranks are global ``{2,3}``
    on ``cuda:{2,3}`` the group-local plan is wrong twice over -- rank 0 is not in
    the pg, and the device is cuda:0 not cuda:2. Remap group-local rank ``r`` to
    global rank ``g = get_global_rank(stage_pg, r)`` on ``cuda:{g % device_count}``
    (single-host assumption). Deterministic given ``(plan, stage_pg)``, so every
    rank in the pg produces the same result.
    """
    if device.type != "cuda":
        return
    device_count = torch.cuda.device_count()
    for module_plan in plan.plan.values():
        # ModuleShardingPlan is dict-like at runtime (param name -> ParameterSharding).
        # pyrefly: ignore[missing-attribute]
        for param_sharding in module_plan.values():
            spec = getattr(param_sharding, "sharding_spec", None)
            if spec is None:
                continue
            for shard in spec.shards:
                g = dist.get_global_rank(stage_pg, shard.placement.rank())
                dev = g % device_count if device_count else 0
                shard.placement = torch.distributed._remote_device(
                    f"rank:{g}/cuda:{dev}"
                )


def build_handoff_process_groups(
    stage_ranks: List[List[int]],
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create the two direction-split P2P communicators for the pipeline hand-off.

    Returns ``(act_pg, grad_pg)`` -- two full-membership NCCL communicators, split
    by what they carry: ``act_pg`` for the forward activations (downstream,
    ``i -> i+1``) and ``grad_pg`` for the backward gradients (upstream,
    ``i+1 -> i``). Splitting the hand-off by direction puts a boundary's two
    directions on *different* communicators, so they run on separate NCCL streams
    instead of serializing on one; and each communicator carries a single flow
    direction (each rank does recv-then-send on it, never send-first), which avoids
    the symmetric send-first deadlock and the >64 MiB rendezvous hang that a shared
    bidirectional communicator hits.

    This mirrors ``torch.distributed.pipelining``'s optional per-direction split
    (``pp_p2p_downstream`` / ``pp_p2p_upstream``); see the study in
    ``tech-docs/nccl_p2p_execution_order_buffer_size.md`` (sections 4-5). Both
    groups are separate from each stage's ``stage_pg`` (the intra-HSD sharded
    all-to-all), so the P2P never interleaves with the sharding collectives.
    ``dist.new_group`` is collective: every rank builds both groups in the same
    order, up front (before any DMP sharding) so the ``new_group`` calls stay
    contiguous.
    """
    all_ranks = sorted({r for ranks in stage_ranks for r in ranks})
    act_pg = cast(dist.ProcessGroup, dist.new_group(ranks=all_ranks))
    grad_pg = cast(dist.ProcessGroup, dist.new_group(ranks=all_ranks))
    return act_pg, grad_pg


def _all_reduce_dense(params: List[nn.Parameter], stage_pg: dist.ProcessGroup) -> None:
    """DP-average the given params' grads across the HSD's ranks (once per step)."""
    world_size = stage_pg.size()
    if world_size == 1:
        return
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, group=stage_pg)
            param.grad /= world_size


class StageParallelizer(abc.ABC):
    """Strategy for mapping a stage module onto its HSD process group.

    Encapsulates the three things that vary by parallelism scheme: how the module
    is transformed (:meth:`parallelize`), how its optimizer is built
    (:meth:`build_optimizer`), and how its gradients are reduced
    (:meth:`reduce_gradients`).

    **Invariant:** implementations must configure any data-parallelism so
    gradients are NOT reduced during backward. The pipeline accumulates across
    microbatches and reduces once, via :meth:`reduce_gradients`, after all
    microbatch backwards (see ``run_1f1b``). This keeps the 1F1B schedule
    decoupled from per-backward DP collectives.
    """

    @abc.abstractmethod
    def parallelize(
        self, module: nn.Module, stage_pg: dist.ProcessGroup, device: torch.device
    ) -> nn.Module:
        """Transform ``module`` for ``stage_pg`` (shard/wrap) and return it."""
        ...

    @abc.abstractmethod
    def build_optimizer(self, module: nn.Module, lr: float) -> torch.optim.Optimizer:
        """Build the optimizer for the parallelized ``module``."""
        ...

    @abc.abstractmethod
    def reduce_gradients(self, module: nn.Module, stage_pg: dist.ProcessGroup) -> None:
        """Complete the once-per-step DP gradient reduction over ``stage_pg``."""
        ...


class Replicated(StageParallelizer):
    """Full replica per rank; grads are DP-averaged over the HSD.

    The numerics-transparent path the correctness test compares against a
    single-process reference.

    Example::

        stage = StageWrapper(module, stage_pg, stage_index, Replicated())
    """

    def parallelize(
        self, module: nn.Module, stage_pg: dist.ProcessGroup, device: torch.device
    ) -> nn.Module:
        return module

    def build_optimizer(self, module: nn.Module, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(module.parameters(), lr=lr, foreach=True)

    def reduce_gradients(self, module: nn.Module, stage_pg: dist.ProcessGroup) -> None:
        _all_reduce_dense(list(module.parameters()), stage_pg)


class EmbeddingShard(StageParallelizer):
    """Shard the stage's ``EmbeddingBagCollection`` within its HSD via DMP.

    The EBC tables are sharded over ``stage_pg`` (the lookup all-to-all stays
    local to the HSD -- no global cross-HSD embedding exchange), and the embedding
    optimizer is fused into the TBE backward (``apply_optimizer_in_backward``).
    Dense params stay replicated and are DP-averaged by :meth:`reduce_gradients`;
    ``init_data_parallel=False`` keeps them out of DDP so the pipeline's manual
    1F1B backward drives them (the pipeline-safe dense-DP mode). See
    :func:`_remap_plan_to_pg_devices` for the sub-pg placement fix.

    Args:
        embedding_lr: learning rate for the fused (in-backward) embedding
            optimizer.

    Example::

        stage = StageWrapper(module, stage_pg, stage_index, EmbeddingShard())
    """

    def __init__(self, embedding_lr: float = 0.05) -> None:
        self._embedding_lr = embedding_lr
        # Dense (non-sharded) params to DP-average once per step.
        self._dense_params: List[nn.Parameter] = []

    def parallelize(
        self, module: nn.Module, stage_pg: dist.ProcessGroup, device: torch.device
    ) -> nn.Module:
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
        if not emb_params:
            raise ValueError(
                "EmbeddingShard requires an EmbeddingBagCollection in the stage"
            )
        apply_optimizer_in_backward(
            torch.optim.SGD, emb_params, {"lr": self._embedding_lr}
        )

        # Plan + shard scoped to this stage's HSD; the embedding all-to-all runs
        # only over ``stage_pg`` (world_size == the HSD's rank count).
        env = ShardingEnv.from_process_group(stage_pg)
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size=stage_pg.size(), compute_device=device.type)
        )
        # NOTE: not planner.collective_plan(): it broadcasts from group-local rank
        # 0 but passes it as a *global* broadcast src, which deadlocks for any
        # stage whose pg does not contain global rank 0 (stages 1..N-1). Compute
        # the plan on the pg's leader and broadcast with the correct global src.
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
        # The planner places shards on cuda:{group-local-rank}; remap to this
        # stage's actual device ordinals (cuda:2/3 for stage 1, etc.).
        _remap_plan_to_pg_devices(plan, stage_pg, device)
        dmp = DistributedModelParallel(
            module=module,
            env=env,
            device=device,
            plan=plan,
            sharders=sharders,
            init_data_parallel=False,
            init_parameters=False,
        )
        self._dense_params = [
            p for _, p in in_backward_optimizer_filter(dmp.named_parameters())
        ]
        return dmp

    def build_optimizer(self, module: nn.Module, lr: float) -> torch.optim.Optimizer:
        dense_optim = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(module.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=lr, foreach=True),
        )
        dmp = cast(DistributedModelParallel, module)
        return CombinedOptimizer([dmp.fused_optimizer, dense_optim])

    def reduce_gradients(self, module: nn.Module, stage_pg: dist.ProcessGroup) -> None:
        # Only the dense params need DP averaging; the sharded embedding grads are
        # handled locally by the fused (in-backward) optimizer.
        _all_reduce_dense(self._dense_params, stage_pg)


class StageWrapper(nn.Module):
    """Binds a Maglev stage module to its HSD process group via a parallelizer.

    The :class:`StageParallelizer` decides how the stage is mapped onto
    ``stage_pg`` -- replicated, embedding-sharded, etc. -- and owns optimizer
    construction and the once-per-step gradient reduction. The wrapper itself is a
    thin delegator, so new parallelism schemes (FSDP/TP on the dense sub-arch) are
    added as new parallelizers without changing the wrapper or the pipeline.

    Args:
        module: the stage module, following the ``forward(stage_input,
            prev_output)`` contract.
        stage_pg: the process group for this stage's HSD.
        stage_index: this stage's position in the pipeline.
        parallelizer: the parallelization strategy. Defaults to
            :class:`Replicated`.

    Example::

        stage = StageWrapper(module, stage_pg, stage_index, EmbeddingShard())
        opt = stage.configure_optimizer(lr=0.05)
    """

    def __init__(
        self,
        module: nn.Module,
        stage_pg: dist.ProcessGroup,
        stage_index: int,
        parallelizer: Optional[StageParallelizer] = None,
    ) -> None:
        super().__init__()
        self._stage_pg: dist.ProcessGroup = stage_pg
        self.stage_index: int = stage_index
        self._parallelizer: StageParallelizer = parallelizer or Replicated()
        device: torch.device = next(module.parameters()).device
        self.module: nn.Module = self._parallelizer.parallelize(
            module, stage_pg, device
        )

    @property
    def stage_pg(self) -> dist.ProcessGroup:
        return self._stage_pg

    def forward(
        self, stage_input: Any, prev_output: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Run the wrapped stage.

        Args:
            stage_input: this stage's own input (its feature partition).
            prev_output: the previous stage's activation, or ``None`` for the
                first stage.

        Returns:
            torch.Tensor: this stage's output activation.
        """
        return self.module(stage_input, prev_output)

    def configure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Build the optimizer matching this stage's parallelizer."""
        return self._parallelizer.build_optimizer(self.module, lr)

    def reduce_gradients(self) -> None:
        """Complete the once-per-step DP gradient reduction for this stage."""
        self._parallelizer.reduce_gradients(self.module, self._stage_pg)
