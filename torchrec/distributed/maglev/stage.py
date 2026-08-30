#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Distributing a Maglev model: cutting it into stages and parallelizing them.

Everything that knows about ranks lives here -- process-group builders, the
layers-to-stages partitioning, the parallelism strategies, and
:class:`StageWrapper`, which binds one stage's layers to its HSD and owns the
wire between stages. The authoring side is in
:mod:`torchrec.distributed.maglev.module`.
"""

from collections import deque
from typing import Any, Callable, cast, Deque, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.profiler import record_function
from torchrec.distributed.maglev.input_dist import InputDistDriver
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    check_layers_chain,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.types import LazyAwaitable, ShardingPlan
from torchrec.distributed.utils import init_parameters


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


def build_cascade_process_groups(
    stage_ranks: List[List[int]],
) -> List[dist.ProcessGroup]:
    """Create one process group per pipeline "cascade" for input distribution.

    A *cascade* is the set of same-position ranks across all stages:
    ``cascade_pgs[c]`` groups ``stage_ranks[0][c], stage_ranks[1][c], ...`` and so
    has one rank per stage (size == the number of stages). It is the group over
    which a rank's full per-stage input set is all-to-all'd so every rank ends up
    holding its own stage's inputs -- one microbatch contributed by each stage's
    rank in the cascade (see the benchmark's input-dist driver).

    Like :func:`build_stage_process_groups` / :func:`build_handoff_process_groups`,
    ``dist.new_group`` is collective: every rank creates every cascade group in the
    same order, and building these up front (before any DMP sharding) keeps all
    ``new_group`` collectives contiguous. The group inherits the job's backend
    (``cpu:gloo,cuda:nccl``), so the same handle drives both the CPU/gloo size
    exchange and the CUDA/nccl data exchange.

    Args:
        stage_ranks: ``stage_ranks[i]`` is the global ranks of stage ``i``'s HSD;
            every stage must have the same number of ranks (cascades).
    """
    num_cascades = len(stage_ranks[0]) if stage_ranks else 0
    pgs: List[dist.ProcessGroup] = []
    for c in range(num_cascades):
        ranks = [stage_ranks[s][c] for s in range(len(stage_ranks))]
        pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
        pgs.append(pg)
    return pgs


def pg_init(stage_size: int, num_stages: int) -> Tuple[
    List[dist.ProcessGroup],
    Tuple[dist.ProcessGroup, dist.ProcessGroup],
    List[dist.ProcessGroup],
]:
    """Build the three sets of process groups a Maglev pipeline needs.

    Returns ``(stage_pgs, handoff_pgs, cascade_pgs)``:

    1. **stage** -- one per stage, joined only by that stage's own ranks; what the
       parallelizer shards over (:func:`build_stage_process_groups`).
    2. **cascade** -- one per position, holding one rank from each stage; what the
       input distribution all-to-alls over
       (:func:`build_cascade_process_groups`).
    3. **hand-off** -- exactly two, full-membership, one carrying forward
       activations and one carrying backward gradients
       (:func:`build_handoff_process_groups`).

    Their memberships overlap -- a cascade contains the very ranks the hand-off
    walks at that position -- but they must stay distinct communicators: the
    cascade carries a collective, the hand-off carries P2P, and mixing the two on
    one communicator is the documented NCCL hang. The hand-off is two groups so a
    boundary's forward and backward run on separate streams rather than
    serializing.

    This function exists for the *ordering*, which is the part that is easy to get
    wrong: ``dist.new_group`` is a collective, so every rank must create every
    group in the same order, and interleaving those calls with DMP's sharding
    collectives deadlocks. Building all three sets here, in a fixed order, before
    any parallelizer runs, makes both properties hold by construction --
    :class:`StageWrapper` calls this before it shards.

    The stage layout is implicit in ``stage_size``: stage ``i`` is the contiguous
    rank block starting at ``i * stage_size``. This is the one place that table is
    materialized; the builders take it explicitly.

    Args:
        stage_size: ranks per stage (one HSD); also the number of cascades.
        num_stages: how many stages the job holds; also each cascade's size.

    Returns:
        The stage groups (indexed by stage), the ``(act_pg, grad_pg)`` hand-off
        pair, and the cascade groups (indexed by position).
    """
    stage_ranks: List[List[int]] = [
        list(range(s * stage_size, (s + 1) * stage_size)) for s in range(num_stages)
    ]
    stage_pgs = build_stage_process_groups(stage_ranks)
    handoff_pgs = build_handoff_process_groups(stage_ranks)
    cascade_pgs = build_cascade_process_groups(stage_ranks)
    return stage_pgs, handoff_pgs, cascade_pgs


def remap_plan_to_process_group(
    plan: ShardingPlan, pg: dist.ProcessGroup, device: torch.device
) -> None:
    """Rewrite a plan's shard placements from group-local to global, in place.

    A planner built over a sub-process-group topology emits *group-local* ranks
    and devices -- ``rank:0/cuda:0``, ``rank:1/cuda:1`` for a 2-rank group. Shard
    placements are interpreted against the *global* rank space, though:
    ``DistributedModelParallel`` maps ``placement.rank()`` back through the
    process group, and the shard tensor lives on the process's actual device. For
    a group whose ranks are global ``{2, 3}`` on ``cuda:{2, 3}`` the group-local
    plan is wrong twice over -- rank 0 is not in the group at all, and the device
    is ``cuda:0`` rather than ``cuda:2``.

    Call this on a plan produced by a planner whose ``Topology`` was sized to
    ``pg`` rather than to the world, before handing it to
    ``DistributedModelParallel``. Group-local rank ``r`` is remapped to global
    rank ``g = get_global_rank(pg, r)``, placed on ``cuda:{g % device_count}``.
    Deterministic given ``(plan, pg)``, so every rank in the group produces the
    same result.

    A no-op for non-CUDA devices, which carry no ordinal to correct.

    Args:
        plan: the sharding plan to rewrite, modified in place.
        pg: the process group the plan was planned over.
        device: the compute device; only ``cuda`` placements are remapped.

    .. note::
        Assumes one host -- global rank ``g`` is taken to be on local device
        ``g % torch.cuda.device_count()``.
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
                g = dist.get_global_rank(pg, shard.placement.rank())
                dev = g % device_count if device_count else 0
                shard.placement = torch.distributed._remote_device(
                    f"rank:{g}/cuda:{dev}"
                )


class _LayerChain(nn.Module):
    """Runs a stage's layers back to back, threading the activation through.

    Not part of the Maglev API -- it exists only because the parallelizers need a
    single callable ``nn.Module`` to own the stage's layers:
    ``DistributedModelParallel`` shards exactly one module, and a bare
    ``nn.ModuleList`` has no ``forward``. :class:`StageWrapper` builds one from
    the layer list it is given and exposes it as :attr:`StageWrapper.module`.

    The stage holding the model's *final* layer is also given the model's
    :meth:`~torchrec.distributed.maglev.module.MaglevModuleList.postproc`, and
    applies it to the last activation, returning ``(losses, output)``. That is
    what keeps the two execution modes equivalent: ``MaglevModuleList.forward``
    ends with ``postproc``, so a pipelined run has to end with it too, or the
    model's head and loss would simply not run. It happens here, inside the
    parallelized module, so it stays in the autograd graph and a head with
    parameters shards with the rest of the stage.

    ``postproc`` is handed the last layer's *input* as well as its activation,
    because that is where the target lives -- which is why the pipeline never has
    to carry labels.

    Args:
        layers: the stage's layers, in execution order.
        postproc: the model's output seam, on the stage that owns the final
            layer; ``None`` on every other stage, which must hand a plain
            activation tuple to the next HSD. Called as
            ``postproc(activations, layer_inputs[-1])``.

    Example::

        chain = _LayerChain([layer0, layer1])
        (out,) = chain([layer_input0, layer_input1])
    """

    def __init__(
        self,
        layers: Sequence[MaglevLayer],
        postproc: Optional[Callable[[Activations, Any], Any]] = None,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(layers)
        # Plain attribute, not a submodule: postproc is a bound method of the
        # authored model, whose parameters are already owned by ``layers``.
        self._postproc = postproc

    def forward(
        self, layer_inputs: Sequence[Any], in_activations: Activations = ()
    ) -> Any:
        """Chain the layers, threading each layer's activation into the next.

        Args:
            layer_inputs: one input per layer, index-aligned with :attr:`layers`.
            in_activations: the activation entering the stage. Default is ``()``.

        Returns:
            Any: a plain ``Activations`` tuple, except on the stage owning the
            final layer, where the model's ``postproc`` has produced
            ``(losses, output)``.

        Raises:
            ValueError: if the input count does not match the layer count.
        """
        if len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        activations = in_activations
        for layer, layer_input in zip(self.layers, layer_inputs):
            activations = layer(layer_input, activations)
        if self._postproc is not None:
            # The last layer's input carries the target postproc scores against.
            return self._postproc(activations, layer_inputs[-1])
        return activations


class StageWrapper(nn.Module):
    """Takes a whole Maglev model and keeps the one stage this rank owns.

    Every rank passes the same ``model`` and ``layers_per_stage``; only
    ``dist.get_rank()`` differs. The wrapper derives which stage this rank owns
    (:meth:`locate_rank` -- HSDs are contiguous blocks of ``stage_size`` ranks, so
    no rank-to-stage table is carried), keeps that contiguous run of layers, and
    drops the rest. The kept layers are the model's *own* modules, not copies, so
    the standalone and pipelined executions share parameters.

    It builds its own process groups: :attr:`stage_pg` (intra-HSD),
    :attr:`handoff_pgs` (the two direction-split P2P communicators), and
    :attr:`cascade_pg` (input distribution). ``dist.new_group`` is a collective,
    so every rank must create every group in the same order; doing it in the
    constructor makes that ordering unmissable.

    **Parallelism is the caller's job.** The wrapper cuts the model, builds the
    groups, and reads the boundary contract; it does not shard. Wrap
    :attr:`module` however you like, assign it back, then :meth:`to` to
    materialize whatever is still on ``meta``::

        stage = StageWrapper(model, layers_per_stage, stage_size)
        stage.module = DistributedModelParallel(
            stage.module, env=ShardingEnv.from_process_group(stage.stage_pg), ...
        )
        stage.to(device)

    Wrap before materializing, so a sharder places the tables rather than
    resharding ones already allocated at full size.

    The stage reduces no gradients. They accumulate across a pass, and the
    schedule runs every backward but the last inside a caller-supplied
    ``no_sync`` context so the DP wrapper's own reducer fires once (see
    :class:`~torchrec.distributed.maglev.pipeline.MaglevPipelineBase`).

    The activation specs are captured in the constructor, before any wrapping,
    since a wrapped module may no longer re-expose the
    :class:`~torchrec.distributed.maglev.module.MaglevLayer` API.

    Args:
        model: the authored model, in full.
        layers_per_stage: how many layers each pipeline stage owns; must sum to
            ``len(model)``, with one entry per stage.
        stage_size: ranks per stage -- the size of one hardware scale-up domain
            (HSD). Stage ``i`` is the contiguous rank block starting at
            ``i * stage_size``.

    Raises:
        ValueError: if ``layers_per_stage`` does not describe ``model``, the
            number of stages implied by ``stage_size`` does not match
            ``layers_per_stage``, or the kept layers disagree on the activation
            they exchange.

    Example::

        # Authored on meta: no storage anywhere yet.
        model = MaglevModuleList([l0, l1, l2, l3])
        # World size 4, so two stages of two ranks. On rank 3: position 1 of
        # stage 1, so layers l2 and l3 -- and only those are materialized.
        stage = StageWrapper(model, [2, 2], stage_size=2)
        stage.to(device)
        (out,) = stage([layer_inputs[i] for i in stage.layer_indices])
    """

    @classmethod
    def count_stages(cls, stage_size: int, world_size: Optional[int] = None) -> int:
        """How many pipeline stages a job of ``world_size`` ranks holds.

        The layout is implicit: stage ``i`` is the contiguous block of
        ``stage_size`` ranks starting at ``i * stage_size``.

        Args:
            stage_size: ranks per stage (one HSD).
            world_size: total ranks in the job. Defaults to
                ``dist.get_world_size()``.

        Returns:
            int: the number of stages.

        Raises:
            ValueError: if ``stage_size`` is not positive, or ``world_size`` is
                not a whole number of stages.

        Example::

            StageWrapper.count_stages(2, world_size=8)   # 4
        """
        if world_size is None:
            world_size = dist.get_world_size()
        if stage_size <= 0:
            raise ValueError(f"stage_size must be positive, got {stage_size}")
        if world_size % stage_size:
            raise ValueError(
                f"world_size {world_size} is not a whole number of stages of "
                f"{stage_size} ranks"
            )
        return world_size // stage_size

    @classmethod
    def locate_rank(cls, stage_size: int, global_rank: int) -> Tuple[int, int]:
        """Find which stage a rank belongs to, and where in that stage's HSD.

        With HSDs laid out as contiguous blocks of ``stage_size`` ranks, a rank's
        whole role is one ``divmod``: the stage it owns, and its *position* within
        that stage, which is what pairs it with the corresponding rank of the
        neighbouring HSDs for the hand-off.

        Args:
            stage_size: ranks per stage.
            global_rank: the rank to locate.

        Returns:
            Tuple[int, int]: ``(stage_index, position)``.

        Raises:
            ValueError: if ``stage_size`` is not positive, or ``global_rank`` is
                negative.

        Example::

            StageWrapper.locate_rank(2, 3)   # (1, 1)
        """
        if stage_size <= 0:
            raise ValueError(f"stage_size must be positive, got {stage_size}")
        if global_rank < 0:
            raise ValueError(f"global_rank must not be negative, got {global_rank}")
        stage_index, position = divmod(global_rank, stage_size)
        return stage_index, position

    def __init__(
        self,
        model: MaglevModuleList,
        layers_per_stage: Sequence[int],
        stage_size: int,  # number of ranks for each stage
    ) -> None:
        super().__init__()
        num_stages = self.count_stages(stage_size)
        if len(layers_per_stage) != num_stages:
            raise ValueError(
                f"layers_per_stage describes {len(layers_per_stage)} stages, but "
                f"stage_size {stage_size} gives {num_stages}"
            )
        if sum(layers_per_stage) != len(model):
            raise ValueError(
                f"layers_per_stage sums to {sum(layers_per_stage)}, but the model "
                f"has {len(model)} layers"
            )
        for s, count in enumerate(layers_per_stage):
            if count <= 0:
                raise ValueError(f"stage {s} must own at least one layer, got {count}")
        # This rank's own role: the wrapper shards collectively, so it can only
        # ever be built for the calling rank.
        stage_index, position = self.locate_rank(stage_size, dist.get_rank())
        self.stage_index: int = stage_index
        self.position: int = position
        self.stage_size: int = stage_size
        self.num_stages: int = num_stages
        # How the model is cut is the source of truth; which layers this stage
        # owns is derived from it (see layer_indices), never stored twice.
        self.layers_per_stage: List[int] = list(layers_per_stage)
        layers: List[MaglevLayer] = [
            cast(MaglevLayer, model[i]) for i in self.layer_indices
        ]
        check_layers_chain(layers, f"stage {stage_index}")
        # Every group the pipeline needs, built in one contiguous run before the
        # parallelizer issues a single sharding collective (see pg_init).
        stage_pgs, handoff_pgs, cascade_pgs = pg_init(stage_size, num_stages)
        self._stage_pg: dist.ProcessGroup = stage_pgs[stage_index]
        self._handoff_pgs: Tuple[dist.ProcessGroup, dist.ProcessGroup] = handoff_pgs
        self._cascade_pg: dist.ProcessGroup = cascade_pgs[position]
        # Posted receives, oldest first: each entry is one transfer's work
        # handles and the buffers landing into them.
        # pyre-ignore[4]: dist work handles have no public type
        self._recv_act: Deque[Tuple[List[Any], List[torch.Tensor]]] = deque()
        # pyre-ignore[4]
        self._recv_grad: Deque[Tuple[List[Any], List[torch.Tensor]]] = deque()
        # The one send in flight per direction: one (work, buffer) per tensor,
        # the buffer keeping the send open.
        # pyre-ignore[4]
        self._send_act: List[Tuple[Any, torch.Tensor]] = []
        # pyre-ignore[4]
        self._send_grad: List[Tuple[Any, torch.Tensor]] = []
        # The model's own input seam. Kept as the bound method, not the model:
        # holding the model would register every other stage's layers as
        # submodules of this wrapper.
        self._preproc: Callable[[Any], List[Any]] = model.preproc
        # Holds this stage's inputs between all-to-all rounds and the schedule
        # asking for microbatches.
        self._input_driver: InputDistDriver[List[Any]] = InputDistDriver(
            pg_gloo=self._cascade_pg,
            pg_nccl=self._cascade_pg,
            self_index=stage_index,
        )
        # Microbatches forwarded but not yet backwarded, oldest first:
        # (incoming activation, this stage's output, microbatch id).
        self._pending: List[Tuple[Activations, Any, int]] = []
        # Read the boundary contract off the authored layers; parallelize() may
        # return a wrapper that hides it.
        self._in_specs: Tuple[ActivationSpec, ...] = layers[0].in_activation_specs()
        self._out_specs: Tuple[ActivationSpec, ...] = layers[-1].out_activation_specs()
        # The stage owning the model's final layer also owns its output seam, so
        # the pipelined run ends exactly where MaglevModuleList.forward does.
        self.is_last_stage: bool = stage_index == num_stages - 1
        # Public and reassignable: wrap it in DMP/FSDP/nothing and assign back.
        self.module: nn.Module = _LayerChain(
            layers, model.postproc if self.is_last_stage else None
        )
        # Set by to(): a meta-authored model has no device to infer, which is the
        # whole point of authoring it there.
        self.device: Optional[torch.device] = None

    @property
    def num_layers(self) -> int:
        """How many layers this stage owns (== how many inputs it takes)."""
        return self.layers_per_stage[self.stage_index]

    @property
    def layer_indices(self) -> range:
        """Which of the model's layers this stage owns, in model order.

        Derived from :attr:`layers_per_stage` rather than stored: this stage's
        layers are the ones following every earlier stage's.

        Example::

            (out,) = stage([layer_inputs[i] for i in stage.layer_indices])
        """
        start = sum(self.layers_per_stage[: self.stage_index])
        return range(start, start + self.num_layers)

    @property
    def handoff_pgs(self) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
        """The ``(act_pg, grad_pg)`` pair the pipeline hands activations over."""
        return self._handoff_pgs

    @property
    def cascade_pg(self) -> dist.ProcessGroup:
        """This rank's input-distribution group (one rank per stage)."""
        return self._cascade_pg

    def neighbor_rank(self, offset: int) -> int:
        """The global rank at this position in the HSD ``offset`` stages away.

        The hand-off is by position: ``neighbor_rank(-1)`` and
        ``neighbor_rank(+1)`` are the ranks this one exchanges activations and
        gradients with.

        Args:
            offset: stages to move, e.g. ``-1`` for the previous HSD.

        Returns:
            int: the neighbouring rank.

        Raises:
            ValueError: if the offset lands outside the pipeline.

        Example::

            prev_rank = stage.neighbor_rank(-1)
        """
        stage_index = self.stage_index + offset
        if not 0 <= stage_index < self.num_stages:
            raise ValueError(
                f"stage {self.stage_index} has no neighbor at offset {offset}: "
                f"the pipeline has {self.num_stages} stages"
            )
        return stage_index * self.stage_size + self.position

    @property
    def stage_pg(self) -> dist.ProcessGroup:
        return self._stage_pg

    # pyre-ignore[14]: narrower than nn.Module.to by design -- this one
    # materializes meta parameters, which nn.Module.to cannot.
    def to(self, device: torch.device) -> "StageWrapper":
        """Place this stage on ``device``, materializing anything still on ``meta``.

        Call after wrapping :attr:`module`, not before: a meta-authored stage
        materialized first allocates full-size embedding tables a sharder is
        about to cut up. Idempotent, and a no-op for parameters a wrapper
        (``DistributedModelParallel``) already placed.

        Args:
            device: where this stage runs. Also where the hand-off allocates its
                receive buffers, so it must be set before the pipeline runs.
        """
        init_parameters(self.module, device)
        self.device = device
        self._input_driver.set_device(device)
        return self

    @property
    def _placed_device(self) -> torch.device:
        """:attr:`device`, or a clear error if :meth:`to` was never called."""
        device = self.device
        if device is None:
            raise ValueError(
                f"stage {self.stage_index}: no device; call stage.to(device) after "
                "wrapping stage.module and before running the pipeline"
            )
        return device

    def in_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation this stage receives from the previous HSD."""
        return self._in_specs

    def out_activation_specs(self) -> Tuple[ActivationSpec, ...]:
        """The activation this stage sends to the next HSD."""
        return self._out_specs

    def forward(
        self, stage_input: Sequence[Any], in_activations: Activations = ()
    ) -> Any:
        """Run this stage's layers.

        Args:
            stage_input: one input per layer this stage owns.
            in_activations: the previous stage's activation, ``()`` for the first
                stage.

        Returns:
            Any: a plain ``Activations`` tuple, except on the last stage, where
            the model's ``postproc`` has produced ``(losses, output)``.

        Raises:
            ValueError: if the input count does not match the layers this stage
                owns.

        .. note::
            The incoming activation is deliberately *not* validated here. On the
            pipeline path it cannot be wrong: the connector allocates each buffer
            as ``torch.empty(spec.shape, dtype=spec.dtype)`` from these very
            specs, so a per-microbatch check would only restate its own premise
            while running on the host's critical path. Call
            :func:`~torchrec.distributed.maglev.module.check_activations`
            explicitly when feeding a stage hand-built activations.
        """
        return self.module(stage_input, in_activations)

    # ---- cross-HSD hand-off ----
    #
    # The stage owns the wire, not just the compute: it knows its neighbours
    # (:meth:`neighbor_rank`), the two direction-split communicators
    # (:attr:`handoff_pgs`), and the specs that fix the wire layout. A schedule
    # (see :mod:`torchrec.distributed.maglev.pipeline`) decides
    # *when* to call these; it does not need to know how a boundary is wired.
    # This mirrors ``torch.distributed.pipelining``, where ``PipelineStage`` owns
    # the send/recv ops and the schedule only orders them.
    #
    # Every transfer is split into a start and a wait, and every one is
    # non-blocking underneath. Issuing is therefore free of ordering constraints,
    # and a schedule chooses how much work to slide between the two halves --
    # posting a receive well before the data is needed is what lets a zero-bubble
    # schedule hide the hand-off behind compute. Each direction keeps its own
    # queue. Receives queue: each start posts another, and the matching wait
    # dequeues in issue order, so a schedule can run several boundaries ahead of
    # the compute. Sends do not queue -- one per direction is in flight at a time,
    # and the schedule must finish it before starting the next. No start ever
    # blocks, so where the wait falls is the schedule's choice, not a side effect
    # buried in the send.

    @property
    def is_first(self) -> bool:
        """Whether this stage starts the pipeline (nothing to receive)."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this stage ends the pipeline (nothing to send)."""
        return self.stage_index == self.num_stages - 1

    def start_recv_act(self) -> None:
        """Ensure a receive is posted for the previous HSD's activation.

        Allocates one buffer per incoming spec and issues the receives in spec
        order, so both sides of the boundary agree without exchanging metadata.
        Collect them with :meth:`wait_for_act`.

        No-op on the first stage. Otherwise each call posts another receive and
        queues it, so a schedule can run several boundaries ahead of the compute;
        :meth:`wait_for_act` dequeues them in issue order.
        """
        if self.is_first:
            return
        act_pg, _ = self._handoff_pgs
        src = self.neighbor_rank(-1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        for spec in self._in_specs:
            tensor = torch.empty(
                spec.shape, device=self._placed_device, dtype=spec.dtype
            )
            works.append(dist.irecv(tensor, src=src, group=act_pg))
            tensors.append(tensor)
        self._recv_act.append((works, tensors))

    def wait_for_act(self) -> Activations:
        """Dequeue the oldest receive posted by :meth:`start_recv_act`.

        Returns:
            Activations: the received activation; ``()`` on the first stage.

        Raises:
            ValueError: if no receive is queued.
        """
        if self.is_first:
            return ()
        if not self._recv_act:
            raise ValueError(
                f"stage {self.stage_index}: wait_for_act() with no activation "
                "receive in flight; call start_recv_act() first"
            )
        works, tensors = self._recv_act.popleft()
        for work in works:
            work.wait()
        for spec, tensor in zip(self._in_specs, tensors):
            if spec.requires_grad:
                # Only now the receive has landed: making it a leaf that requires
                # grad any earlier would make the incoming write an in-place op on
                # a grad-requiring leaf. As a leaf it collects the gradient this
                # stage hands back to the previous one.
                tensor.requires_grad_(True)
        return tuple(tensors)

    def start_send_act(self, outputs: Activations) -> None:
        """Send this stage's activation to the next HSD; no-op if last.

        Never blocks. One activation send is in flight at a time, so the caller
        must :meth:`finish_send_act` the previous one first -- keeping the wait
        where the schedule put it, and bounding the buffers held open to a single
        transfer.

        Args:
            outputs: this stage's output activation.

        Raises:
            ValueError: if an activation send is still in flight.
        """
        if self.is_last:
            return
        if self._send_act:
            raise ValueError(
                f"stage {self.stage_index}: an activation send is still in "
                "flight; call finish_send_act() before starting the next"
            )
        act_pg, _ = self._handoff_pgs
        self._send_act = self._isend(outputs, self.neighbor_rank(1), act_pg)

    def finish_send_act(self) -> None:
        """Complete the activation send in flight, if any."""
        self._drain(self._send_act)

    def start_recv_grad(self) -> None:
        """Ensure a receive is posted for the next HSD's output gradients.

        One receive per grad-carrying output slot. No-op on the last stage, which
        takes its gradient from the loss its own ``postproc`` computed. Otherwise
        each call queues another, as :meth:`start_recv_act`.
        """
        if self.is_last:
            return
        _, grad_pg = self._handoff_pgs
        src = self.neighbor_rank(1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        for spec in self._out_specs:
            if not spec.requires_grad:
                continue
            grad = torch.empty(spec.shape, device=self._placed_device, dtype=spec.dtype)
            works.append(dist.irecv(grad, src=src, group=grad_pg))
            tensors.append(grad)
        self._recv_grad.append((works, tensors))

    def wait_for_grad(self) -> List[torch.Tensor]:
        """Dequeue the oldest receive posted by :meth:`start_recv_grad`.

        Returns:
            List[torch.Tensor]: the gradients, in spec order; empty on the last
            stage.

        Raises:
            ValueError: if no receive is queued.
        """
        if self.is_last:
            return []
        if not self._recv_grad:
            raise ValueError(
                f"stage {self.stage_index}: wait_for_grad() with no gradient "
                "receive in flight; call start_recv_grad() first"
            )
        works, tensors = self._recv_grad.popleft()
        for work in works:
            work.wait()
        return tensors

    def start_send_grad(self, in_activations: Activations) -> None:
        """Send this stage's input gradients upstream; no-op if first.

        A slot unused by the stage's graph has no ``.grad``; zeros are sent so the
        previous stage's receive still matches -- the wire layout is fixed by the
        specs, not by graph connectivity. As with :meth:`start_send_act`, never
        blocks: the previous gradient send must already have been finished.

        Args:
            in_activations: the activation this stage received.

        Raises:
            ValueError: if a gradient send is still in flight.
        """
        if self.is_first:
            return
        if self._send_grad:
            raise ValueError(
                f"stage {self.stage_index}: a gradient send is still in flight; "
                "call finish_send_grad() before starting the next"
            )
        _, grad_pg = self._handoff_pgs
        grads: List[torch.Tensor] = []
        for tensor, spec in zip(in_activations, self._in_specs):
            if not spec.requires_grad:
                continue
            grads.append(
                tensor.grad if tensor.grad is not None else torch.zeros_like(tensor)
            )
        self._send_grad = self._isend(grads, self.neighbor_rank(-1), grad_pg)

    def finish_send_grad(self) -> None:
        """Complete the gradient send in flight, if any."""
        self._drain(self._send_grad)

    def group_by_stage(self, layer_inputs: Sequence[Any]) -> List[List[Any]]:
        """Regroup a full per-layer input list into one carrier per stage.

        The inverse of :attr:`layer_indices`, and the form :meth:`input_dist`
        needs: ``result[s]`` holds the inputs for stage ``s``'s layers, so it is
        what that stage's rank in the cascade should receive.

        Derived from :attr:`layers_per_stage`, so a non-uniform cut like
        ``[1, 3]`` groups correctly -- which hand-slicing by a single
        layers-per-stage number does not.

        Args:
            layer_inputs: one input per layer of the whole model, in model order.

        Returns:
            List[List[Any]]: one carrier per stage, index-aligned with the
            pipeline.

        Raises:
            ValueError: if there is not exactly one input per model layer.

        Example::

            send_set = stage.group_by_stage(layer_inputs)
            microbatches = stage.input_dist(send_set, send_set[stage.stage_index]).wait()
        """
        grouped: List[List[Any]] = []
        offset = 0
        for count in self.layers_per_stage:
            grouped.append(list(layer_inputs[offset : offset + count]))
            offset += count
        return grouped

    def input_dist(self, layer_inputs: Sequence[Any]) -> LazyAwaitable[List[List[Any]]]:
        """All-to-all a full per-layer input set over this rank's cascade.

        A *cascade* holds one rank from every stage (see :attr:`cascade_pg`), so
        this is the exchange that turns "the whole batch, held by every rank" into
        "this stage's inputs, delivered here". The inputs are grouped per stage
        (:meth:`group_by_stage`), each group is sent to that stage's rank in the
        cascade, and what comes back is one group from every stage's rank -- i.e.
        ``num_stages`` microbatches, all of them for *this* stage.

        One round, unwaited. Use :meth:`take_inputs` when the schedule wants a
        microbatch count that does not divide evenly into rounds.

        The cascade group inherits the job's backend (``cpu:gloo,cuda:nccl``), so
        the one handle drives both phases of
        :func:`~torchrec.distributed.maglev.input_dist.input_dist`: the small size
        exchange on CPU/gloo and the bulk tensor exchange on CUDA/nccl.

        Args:
            layer_inputs: one input per layer of the whole model, in model order
                -- what
                :meth:`~torchrec.distributed.maglev.module.MaglevModuleList.preproc`
                produces.

        Returns:
            LazyAwaitable[List[List[Any]]]: ``wait()`` yields ``num_stages``
            microbatches for this stage, each one input per layer this stage
            owns. Returned unwaited so the caller can overlap other work with the
            exchange.

        Raises:
            ValueError: if there is not exactly one input per model layer.

        Example::

            microbatches = stage.input_dist(layer_inputs).wait()
            stage.forward_micro(microbatches[0])
        """
        return self._input_driver.exchange(self.group_by_stage(layer_inputs))

    def send_set(self, model_input: Any) -> List[List[Any]]:
        """Turn one raw batch into the cascade send set: one carrier per stage.

        The model's own :meth:`preproc` seam splits the batch into one input per
        layer (under ``no_grad``, as in
        :meth:`~torchrec.distributed.maglev.module.MaglevModuleList.forward`),
        then :meth:`group_by_stage` regroups those by destination stage.

        Args:
            model_input: the raw batch, as the dataloader yields it.

        Returns:
            List[List[Any]]: ``result[s]`` is the input list destined for stage
            ``s``'s rank in this cascade.
        """
        with torch.no_grad():
            layer_inputs = self._preproc(model_input)
        return self.group_by_stage(layer_inputs)

    def take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Hand the schedule ``n`` microbatches for this stage.

        Pulls raw batches from ``dataloader_iter`` and runs as many
        :meth:`input_dist` rounds as it takes, keeping the remainder queued -- so
        the microbatch count a schedule wants need not equal the ``num_stages`` a
        round produces, and a batch is consumed only when a round actually runs.
        Every rank in the cascade runs the same number of rounds, so every rank
        advances its dataloader in lock-step -- see
        :class:`~torchrec.distributed.maglev.input_dist.InputDistDriver`.

        Args:
            dataloader_iter: yields raw batches, one per round.
            n: how many microbatches the schedule wants.

        Returns:
            List[List[Any]]: ``n`` microbatches, each one input per layer this
            stage owns.
        """
        return self._input_driver.take(lambda: self.send_set(next(dataloader_iter)), n)

    def backward(self, outputs: Activations, grads: Sequence[torch.Tensor]) -> None:
        """Backward through this stage from the gradients its outputs received.

        Args:
            outputs: this stage's output activation.
            grads: the matching gradients, from :meth:`wait_for_grad`.
        """
        grad_carrying = [
            t for t, spec in zip(outputs, self._out_specs) if spec.requires_grad
        ]
        pairs = [
            (out, grad) for out, grad in zip(grad_carrying, grads) if out.requires_grad
        ]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])

    # pyre-ignore[3]: dist work handle has no public type
    def _isend(
        self, tensors: Sequence[torch.Tensor], dst: int, pg: dist.ProcessGroup
    ) -> List[Tuple[Any, torch.Tensor]]:
        """Issue non-blocking sends, in order, for a later drain.

        The detached buffer is returned alongside the work handle so it stays
        alive until the send completes. ``.contiguous()`` is a no-op for an
        already contiguous tensor, so this aliases the activation rather than
        copying it.
        """
        out: List[Tuple[Any, torch.Tensor]] = []
        for tensor in tensors:
            buf = tensor.detach().contiguous()
            out.append((dist.isend(buf, dst=dst, group=pg), buf))
        return out

    # pyre-ignore[2]: dist work handle has no public type
    def _drain(self, sends: List[Tuple[Any, torch.Tensor]]) -> None:
        """Wait every send in ``sends`` and release the buffers holding it open."""
        for work, _buf in sends:
            work.wait()
        sends.clear()

    # ---- one microbatch of this stage's work ----

    def forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int = 0,
    ) -> None:
        """One microbatch forward: recv activation, compute, send activation.

        The result is parked on an internal FIFO for the matching
        :meth:`backward_micro`, so a schedule can run several forwards before the
        first backward without tracking in-flight state itself.

        Does *not* post its own activation receive: the schedule starts that
        ahead of time (:meth:`start_recv_act`) so the transfer overlaps earlier
        work, and this dequeues it.

        Neither receive is posted here -- the schedule owns both (see the
        warning on :meth:`backward_micro` for the constraint the gradient one
        carries).

        Args:
            stage_input: one input per layer this stage owns.
            microbatch_id: tags the profiler ranges (and is carried to the
                matching backward) so a trace shows which microbatch each
                comm/compute belongs to.
        """
        # Posted by the schedule ahead of this call, so the transfer has already
        # been in flight; wait_for_act raises if it was never started.
        with record_function(f"## recv_act mb{microbatch_id} ##"):
            in_activations = self.wait_for_act()

        with record_function(f"## forward mb{microbatch_id} ##"):
            outputs = self.forward(stage_input, in_activations)

        # The previous microbatch's send, drained before this one is issued.
        with record_function(f"## finish_send_act mb{microbatch_id} ##"):
            self.finish_send_act()
        with record_function(f"## send_act mb{microbatch_id} ##"):
            self.start_send_act(outputs)

        self._pending.append((in_activations, outputs, microbatch_id))

    def backward_micro(self) -> Optional[torch.Tensor]:
        """One microbatch backward: recv grad, backward, send input grad.

        Pops the oldest microbatch parked by :meth:`forward_micro`. Gradients
        accumulate across microbatches (no ``zero_grad`` here); the schedule runs
        the DP all-reduce and optimizer step once per batch. Profiler ranges are
        tagged with the ``microbatch_id`` recorded at forward time.

        The last stage takes its gradient from the loss its own ``postproc``
        computed; every other stage collects the receive the schedule posted, and
        :meth:`wait_for_grad` raises if it did not.

        .. warning::
            The schedule must not post that receive during the forward phase. The
            hand-off groups are full-membership, so the *first* P2P on a pair
            lazily creates a 2-rank communicator and **blocks until both ranks
            touch the pair** -- and a peer only touches the gradient direction in
            its own backward. Posted too early, this rank parks mid-wave and the
            pipeline deadlocks: the peer is left waiting for the next activation
            this rank can no longer send. Once that communicator exists, posting
            a microbatch ahead is free, which is what
            :class:`~torchrec.distributed.maglev.pipeline.MaglevPipeline1F1B` does.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage,
            ``None`` on every other stage.
        """
        in_activations, outputs, microbatch_id = self._pending.pop(0)

        loss: Optional[torch.Tensor] = None
        if self.is_last:
            with record_function(f"## backward mb{microbatch_id} ##"):
                loss = outputs[0]
                loss.backward()
        else:
            with record_function(f"## recv_grad mb{microbatch_id} ##"):
                grads = self.wait_for_grad()
            with record_function(f"## backward mb{microbatch_id} ##"):
                self.backward(outputs, grads)

        with record_function(f"## finish_send_grad mb{microbatch_id} ##"):
            self.finish_send_grad()
        with record_function(f"## send_grad mb{microbatch_id} ##"):
            self.start_send_grad(in_activations)

        return loss

    def drain_sends(self) -> None:
        """Complete the in-flight send in each direction, if any.

        ``forward_micro`` / ``backward_micro`` each finish the *previous* send
        before starting the next, so at the end of a pass exactly one send per
        direction is still outstanding. Left unwaited, its work handle leaks and
        its buffer stays pinned while NCCL may still be reading it -- and
        :meth:`start_send_act` refuses to start another while one is in flight.
        A schedule therefore ends every pass here.
        """
        self.finish_send_act()
        self.finish_send_grad()
