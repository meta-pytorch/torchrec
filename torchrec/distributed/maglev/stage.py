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
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast, Deque, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh
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


class HandoffPGMode(Enum):
    """How activation and gradient handoffs share process groups."""

    SHARED = "shared"
    SPLIT = "split"


@dataclass(frozen=True)
class MaglevProcessGroups:
    """The rank-local process groups and global rank layout for a Maglev stage.

    Device-mesh integrations use :meth:`from_device_mesh`. Standalone users such
    as the benchmark use :meth:`from_scratch`, which creates every group in a
    fixed collective order. Keeping the rank layout beside the groups lets
    :class:`StageWrapper` support non-contiguous layouts without independently
    inferring the topology.
    """

    stage_ranks: Tuple[Tuple[int, ...], ...]
    stage_pg: dist.ProcessGroup
    handoff_pgs: Tuple[dist.ProcessGroup, dist.ProcessGroup]
    cascade_pg: dist.ProcessGroup
    cascade_gloo_pg: dist.ProcessGroup
    handoff_pg_mode: HandoffPGMode

    def __post_init__(self) -> None:
        normalized_ranks = self._normalize_stage_ranks(self.stage_ranks)
        object.__setattr__(self, "stage_ranks", normalized_ranks)
        handoff_pgs_are_shared = self.handoff_pgs[0] is self.handoff_pgs[1]
        if (self.handoff_pg_mode is HandoffPGMode.SHARED) != handoff_pgs_are_shared:
            raise ValueError(
                f"{self.handoff_pg_mode.value} handoff mode does not match the "
                "supplied process groups"
            )

    @classmethod
    def build_stage_process_groups(
        cls,
        stage_ranks: Sequence[Sequence[int]],
    ) -> List[dist.ProcessGroup]:
        """Create one data-parallel process group per Maglev stage.

        Every rank must call this collective for every stage in the same order.
        The returned list is index-aligned with ``stage_ranks``.
        """
        return [
            cast(dist.ProcessGroup, dist.new_group(ranks=list(ranks)))
            for ranks in stage_ranks
        ]

    @classmethod
    def build_handoff_process_groups(
        cls,
        stage_ranks: Sequence[Sequence[int]],
        handoff_pg_mode: HandoffPGMode,
    ) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
        """Create the activation and gradient handoff communicators.

        Shared mode returns the same communicator twice. Split mode creates one
        communicator per direction, preserving the original P2P behavior.
        """
        ranks = sorted(rank for stage in stage_ranks for rank in stage)
        activation_pg = cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
        gradient_pg = (
            activation_pg
            if handoff_pg_mode is HandoffPGMode.SHARED
            else cast(dist.ProcessGroup, dist.new_group(ranks=ranks))
        )
        return activation_pg, gradient_pg

    @classmethod
    def build_cascade_process_groups(
        cls,
        stage_ranks: Sequence[Sequence[int]],
        backend: Optional[str] = None,
    ) -> List[dist.ProcessGroup]:
        """Create one same-position input-distribution group across all stages.

        Position ``p`` joins ``stage_ranks[0][p]``, ``stage_ranks[1][p]``, and
        so on. The default backend carries tensors; Gloo carries CPU metadata.
        """
        stage_size = len(stage_ranks[0]) if stage_ranks else 0
        return [
            cast(
                dist.ProcessGroup,
                dist.new_group(
                    ranks=[stage[position] for stage in stage_ranks],
                    backend=backend,
                ),
            )
            for position in range(stage_size)
        ]

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh: DeviceMesh,
        *,
        stage_mesh_dim_name: str,
        pipeline_mesh_dim_name: str,
        handoff_pg_mode: HandoffPGMode = HandoffPGMode.SPLIT,
    ) -> "MaglevProcessGroups":
        """Build Maglev process groups from a two-dimensional device mesh.

        The mesh remains the topology owner. Its stage-dimension group is reused
        for data parallelism, and its pipeline-dimension group is reused for
        accelerator input distribution. Maglev creates only its dedicated P2P
        handoff groups and the Gloo counterpart used for input metadata.

        Every rank must call this method with the same mesh and handoff mode.
        """
        mesh_dim_names = device_mesh.mesh_dim_names
        if device_mesh.ndim != 2 or mesh_dim_names is None:
            raise ValueError("Maglev requires a named two-dimensional device mesh")
        if (
            stage_mesh_dim_name not in mesh_dim_names
            or pipeline_mesh_dim_name not in mesh_dim_names
            or stage_mesh_dim_name == pipeline_mesh_dim_name
        ):
            raise ValueError(
                "stage and pipeline dimensions must be distinct dimensions of "
                f"the device mesh, got {mesh_dim_names}"
            )

        stage_dim = mesh_dim_names.index(stage_mesh_dim_name)
        pipeline_dim = mesh_dim_names.index(pipeline_mesh_dim_name)
        stage_ranks = tuple(
            tuple(int(rank) for rank in ranks)
            for ranks in device_mesh.mesh.permute(pipeline_dim, stage_dim).tolist()
        )
        _, position = cls.locate_rank(stage_ranks, dist.get_rank())
        handoff_pgs = cls.build_handoff_process_groups(stage_ranks, handoff_pg_mode)
        cascade_gloo_pgs = cls.build_cascade_process_groups(stage_ranks, backend="gloo")
        return cls(
            stage_ranks=stage_ranks,
            stage_pg=cast(
                dist.ProcessGroup,
                device_mesh[stage_mesh_dim_name].get_group(),
            ),
            handoff_pgs=handoff_pgs,
            cascade_pg=cast(
                dist.ProcessGroup,
                device_mesh[pipeline_mesh_dim_name].get_group(),
            ),
            cascade_gloo_pg=cascade_gloo_pgs[position],
            handoff_pg_mode=handoff_pg_mode,
        )

    @classmethod
    def from_scratch(
        cls,
        stage_size: int,
        num_stages: int,
        handoff_pg_mode: HandoffPGMode = HandoffPGMode.SPLIT,
    ) -> "MaglevProcessGroups":
        """Create every Maglev process group for a contiguous rank layout.

        Process-group creation is collective. Every rank must call this method
        with identical arguments and before any sharding component creates
        groups. Stage, handoff, tensor-cascade, and Gloo metadata-cascade groups
        are created contiguously in that order.
        """
        if stage_size <= 0:
            raise ValueError(f"stage_size must be positive, got {stage_size}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        stage_ranks = tuple(
            tuple(range(stage * stage_size, (stage + 1) * stage_size))
            for stage in range(num_stages)
        )
        stage_pgs = cls.build_stage_process_groups(stage_ranks)
        handoff_pgs = cls.build_handoff_process_groups(stage_ranks, handoff_pg_mode)
        cascade_pgs = cls.build_cascade_process_groups(stage_ranks)
        cascade_gloo_pgs = cls.build_cascade_process_groups(stage_ranks, backend="gloo")
        stage_index, position = cls.locate_rank(stage_ranks, dist.get_rank())
        return cls(
            stage_ranks=stage_ranks,
            stage_pg=stage_pgs[stage_index],
            handoff_pgs=handoff_pgs,
            cascade_pg=cascade_pgs[position],
            cascade_gloo_pg=cascade_gloo_pgs[position],
            handoff_pg_mode=handoff_pg_mode,
        )

    @classmethod
    def locate_rank(
        cls,
        stage_ranks: Sequence[Sequence[int]],
        global_rank: int,
    ) -> Tuple[int, int]:
        """Return a rank's stage index and position within that stage."""
        for stage_index, ranks in enumerate(stage_ranks):
            if global_rank in ranks:
                return stage_index, ranks.index(global_rank)
        raise ValueError(f"rank {global_rank} is absent from the Maglev rank layout")

    @classmethod
    def _normalize_stage_ranks(
        cls,
        stage_ranks: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[int, ...], ...]:
        normalized = tuple(tuple(ranks) for ranks in stage_ranks)
        if not normalized or not normalized[0]:
            raise ValueError("stage_ranks must contain at least one non-empty stage")
        stage_size = len(normalized[0])
        if any(len(ranks) != stage_size for ranks in normalized):
            raise ValueError("every Maglev stage must contain the same number of ranks")
        flattened = tuple(rank for ranks in normalized for rank in ranks)
        if len(set(flattened)) != len(flattened):
            raise ValueError("each rank must occur exactly once in stage_ranks")
        return normalized

    @property
    def stage_size(self) -> int:
        return len(self.stage_ranks[0])

    @property
    def num_stages(self) -> int:
        return len(self.stage_ranks)


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
    applies it to the last activation. By default its complete result is
    preserved. A caller that wraps the chain in DDP can enable loss-only output,
    keeping auxiliary predictions out of DDP's backward-root traversal.
    ``postproc`` still runs inside the parallelized module, so its head and loss
    remain in the autograd graph and its parameters shard with the rest of the
    stage.

    ``postproc`` is handed the last layer's *input* as well as its activation,
    because that is where the target lives -- which is why the pipeline never has
    to carry labels.

    Args:
        layers: the stage's layers, in execution order.
        postproc: the model's output seam, on the stage that owns the final
            layer; ``None`` on every other stage, which must hand a plain
            activation tuple to the next HSD. Called as
            ``postproc(activations, layer_inputs[-1])``.
        first_layer_index: global index of this stage's first layer, for tracing.
        loss_only_output: whether to expose only the loss from ``postproc``.

    Example::

        chain = _LayerChain([layer0, layer1])
        (out,) = chain([layer_input0, layer_input1])
    """

    def __init__(
        self,
        layers: Sequence[MaglevLayer],
        postproc: Optional[Callable[[Activations, Any], Any]] = None,
        first_layer_index: int = 0,
        loss_only_output: bool = False,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList(layers)
        self._profile_names: List[str] = [
            f"## torchrec_maglev:layer[{first_layer_index + index}] "
            f"{type(layer).__name__.lstrip('_')} ##"
            for index, layer in enumerate(layers)
        ]
        # Plain attribute, not a submodule: postproc is a bound method of the
        # authored model, whose parameters are already owned by ``layers``.
        self._postproc = postproc
        self._loss_only_output = loss_only_output

    def forward(
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations = (),
    ) -> Any:
        """Chain the layers, threading each layer's activation into the next.

        Args:
            layer_inputs: one input per layer, index-aligned with :attr:`layers`.
            in_activations: the activation entering the stage. Default is ``()``.

        Returns:
            Any: a plain ``Activations`` tuple, except on the stage owning the
            final layer, where the model's ``postproc`` result is returned. In
            loss-only mode, that result is reduced to a singleton ``(losses,)``.

        Raises:
            ValueError: if the input count does not match the layer count.
        """
        if len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        activations = in_activations
        for layer, layer_input, profile_name in zip(
            self.layers, layer_inputs, self._profile_names
        ):
            maglev_layer = cast(MaglevLayer, layer)
            with record_function(profile_name):
                activations = maglev_layer(layer_input, activations)
        if self._postproc is not None:
            # The last layer's input carries the target postproc scores against.
            with record_function("## torchrec_maglev:postproc ##"):
                output = self._postproc(activations, layer_inputs[-1])
            if self._loss_only_output:
                # DDP treats every returned tensor as a backward root. Auxiliary
                # predictions must stay out when only the loss is backpropagated.
                losses, _model_output = output
                return (losses,)
            return output
        return activations


class StageWrapper(nn.Module):
    """Takes a whole Maglev model and keeps the one stage this rank owns.

    Every rank passes the same ``model`` and ``layers_per_stage``; only
    ``dist.get_rank()`` differs. The wrapper locates that rank in the process-group
    layout, keeps the corresponding contiguous run of layers, and drops the rest.
    The kept layers are the model's *own* modules, not copies, so the standalone
    and pipelined executions share parameters.

    By default :class:`MaglevProcessGroups` creates a contiguous topology. A
    runtime that already owns process-group construction can pass its bundle via
    ``process_groups`` instead, keeping one source of truth for rank placement.

    **Parallelism is the caller's job.** The wrapper cuts the model, builds the
    process-group bundle, and reads the boundary contract; it does not shard. Wrap
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
            (HSD). Used to build the default contiguous topology and validated
            against ``process_groups`` when one is provided.
        loss_only_output: whether the final stage returns only the loss. Enable
            this before wrapping the stage module in DDP so auxiliary prediction
            tensors are not treated as backward roots.
        process_groups: optional externally constructed Maglev process groups.

    Raises:
        ValueError: if ``layers_per_stage`` does not describe ``model``, the
            process-group layout does not match ``stage_size`` or
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
        loss_only_output: bool = False,
        process_groups: Optional[MaglevProcessGroups] = None,
    ) -> None:
        super().__init__()
        if process_groups is None:
            num_stages = self.count_stages(stage_size)
            process_groups = MaglevProcessGroups.from_scratch(stage_size, num_stages)
        else:
            num_stages = process_groups.num_stages
            if process_groups.stage_size != stage_size:
                raise ValueError(
                    f"stage_size {stage_size} does not match process-group layout "
                    f"size {process_groups.stage_size}"
                )
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
        stage_index, position = MaglevProcessGroups.locate_rank(
            process_groups.stage_ranks, dist.get_rank()
        )
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
        self._process_groups = process_groups
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
        self._get_batch_size: Callable[[Sequence[Any]], int] = model.get_batch_size
        # Holds this stage's inputs between all-to-all rounds and the schedule
        # asking for microbatches.
        self._input_driver: InputDistDriver[List[Any]] = InputDistDriver(
            pg_gloo=process_groups.cascade_gloo_pg,
            pg_nccl=process_groups.cascade_pg,
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
            layers,
            model.postproc if self.is_last_stage else None,
            first_layer_index=self.layer_indices.start,
            loss_only_output=loss_only_output,
        )
        self._loss_only_output = loss_only_output
        # Set by to(): a meta-authored model has no device to infer, which is the
        # whole point of authoring it there.
        self.device: Optional[torch.device] = None

    @property
    def loss_only_output(self) -> bool:
        """Whether the final stage exposes only the postprocessed loss."""
        return self._loss_only_output

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
        """The activation and gradient handoff process groups."""
        return self._process_groups.handoff_pgs

    @property
    def handoff_pg_mode(self) -> HandoffPGMode:
        """Whether activation and gradient handoffs share one communicator."""
        return self._process_groups.handoff_pg_mode

    @property
    def cascade_pg(self) -> dist.ProcessGroup:
        """This rank's input-distribution group (one rank per stage)."""
        return self._process_groups.cascade_pg

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
        return self._process_groups.stage_ranks[stage_index][self.position]

    @property
    def stage_pg(self) -> dist.ProcessGroup:
        return self._process_groups.stage_pg

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

    def activation_batch_size(self, stage_input: Sequence[Any]) -> Optional[int]:
        """Return this microbatch's size when a boundary depends on it."""
        if not any(spec.batch_size_dependent for spec in self._in_specs):
            return None
        batch_size = self._get_batch_size(stage_input)
        if batch_size < 0:
            raise ValueError(f"batch size must be non-negative, got {batch_size}")
        return batch_size

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
            the model's ``postproc`` result follows the configured output mode.

        Raises:
            ValueError: if the input count does not match the layers this stage
                owns.

        .. note::
            The incoming activation is deliberately *not* validated here. On the
            pipeline path it cannot be wrong: the connector allocates each buffer
            from these very specs after materializing any batch-size-dependent
            leading dimension, so a per-microbatch check would only restate its
            own premise while running on the host's critical path. Call
            :func:`~torchrec.distributed.maglev.module.check_activations`
            explicitly when feeding a stage hand-built activations.
        """
        return self.module(stage_input, in_activations)

    # ---- cross-HSD hand-off ----
    #
    # The stage owns the wire, not just the compute: it knows its neighbours
    # (:meth:`neighbor_rank`), the configured communicators
    # (:attr:`handoff_pgs`), and the specs that fix the wire layout. A schedule
    # (see :mod:`torchrec.distributed.maglev.pipeline`) decides
    # *when* to call these; it does not need to know how a boundary is wired.
    # This mirrors ``torch.distributed.pipelining``, where ``PipelineStage`` owns
    # the send/recv ops and the schedule only orders them.
    #
    # SPLIT mode uses independent activation and gradient PGs with the start/wait
    # methods. SHARED mode additionally permits the paired exchange methods:
    # batch_isend_irecv requires every operation in a batch to use one PG.

    @property
    def is_first(self) -> bool:
        """Whether this stage starts the pipeline (nothing to receive)."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this stage ends the pipeline (nothing to send)."""
        return self.stage_index == self.num_stages - 1

    def start_recv_act(self, batch_size: Optional[int] = None) -> None:
        """Ensure a receive is posted for the previous HSD's activation.

        Allocates one buffer per incoming spec and issues the receives in spec
        order, so both sides of the boundary agree without exchanging metadata.
        Collect them with :meth:`wait_for_act`.

        No-op on the first stage. Otherwise each call posts another receive and
        queues it, so a schedule can run several boundaries ahead of the compute;
        :meth:`wait_for_act` dequeues them in issue order.

        Args:
            batch_size: size used to materialize batch-size-dependent specs.
        """
        if self.is_first:
            return
        act_pg, _ = self.handoff_pgs
        src = self.neighbor_rank(-1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        for spec in self._in_specs:
            tensor = torch.empty(
                spec.materialize_shape(batch_size),
                device=self._placed_device,
                dtype=spec.dtype,
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

        One activation send is in flight at a time, so the caller must
        :meth:`finish_send_act` the previous one first, bounding the buffers held
        open to a single transfer. The peer must have posted its matching receive.

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
        act_pg, _ = self.handoff_pgs
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
        _, grad_pg = self.handoff_pgs
        src = self.neighbor_rank(1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        outputs = cast(Activations, self._pending[len(self._recv_grad)][1])
        for output, spec in zip(outputs, self._out_specs):
            if not spec.requires_grad:
                continue
            grad = torch.empty_like(output, memory_format=torch.contiguous_format)
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

    def send_act_recv_grad(self, outputs: Activations) -> List[torch.Tensor]:
        """Exchange a forward activation for its downstream gradient.

        Available only in ``SHARED`` mode because ``batch_isend_irecv`` requires
        every operation in its batch to use the same process group.
        """
        if self.is_last:
            return []
        if self.handoff_pg_mode is not HandoffPGMode.SHARED:
            raise RuntimeError("batched handoff requires handoff_pg_mode=shared")
        peer = self.neighbor_rank(1)
        pg = self.handoff_pgs[0]
        send_buffers = [tensor.detach().contiguous() for tensor in outputs]
        backward_outputs = cast(Activations, self._pending[0][1])
        grad_buffers = [
            torch.empty_like(output, memory_format=torch.contiguous_format)
            for output, spec in zip(backward_outputs, self._out_specs)
            if spec.requires_grad
        ]
        ops = [dist.P2POp(dist.isend, tensor, peer, pg) for tensor in send_buffers]
        ops.extend(dist.P2POp(dist.irecv, tensor, peer, pg) for tensor in grad_buffers)
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()
        return grad_buffers

    def send_grad_recv_act(
        self,
        in_activations: Activations,
        recv_next: bool,
        batch_size: Optional[int] = None,
    ) -> Activations:
        """Exchange an upstream gradient for the next forward activation.

        Available only in ``SHARED`` mode; see :meth:`send_act_recv_grad`.

        Args:
            in_activations: activations whose gradients are sent upstream.
            recv_next: whether to receive the next forward activation.
            batch_size: size of that next activation microbatch.
        """
        if self.is_first:
            return ()
        if self.handoff_pg_mode is not HandoffPGMode.SHARED:
            raise RuntimeError("batched handoff requires handoff_pg_mode=shared")
        peer = self.neighbor_rank(-1)
        pg = self.handoff_pgs[0]
        grad_buffers = [
            (tensor.grad if tensor.grad is not None else torch.zeros_like(tensor))
            for tensor, spec in zip(in_activations, self._in_specs)
            if spec.requires_grad
        ]
        activation_buffers = (
            [
                torch.empty(
                    spec.materialize_shape(batch_size),
                    device=self._placed_device,
                    dtype=spec.dtype,
                )
                for spec in self._in_specs
            ]
            if recv_next
            else []
        )
        ops = [dist.P2POp(dist.isend, tensor, peer, pg) for tensor in grad_buffers]
        ops.extend(
            dist.P2POp(dist.irecv, tensor, peer, pg) for tensor in activation_buffers
        )
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()
        for spec, tensor in zip(self._in_specs, activation_buffers):
            if spec.requires_grad:
                tensor.requires_grad_(True)
        return tuple(activation_buffers)

    def start_send_grad(self, in_activations: Activations) -> None:
        """Send this stage's input gradients upstream; no-op if first.

        A slot unused by the stage's graph has no ``.grad``; zeros are sent so the
        previous stage's receive still matches -- the wire layout is fixed by the
        specs, not by graph connectivity. The previous gradient send must already
        have been finished and the peer must have posted its matching receive.

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
        _, grad_pg = self.handoff_pgs
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

        Two cascade groups with identical membership drive the exchange:
        CPU/Gloo carries the small size metadata, while CUDA/NCCL carries the
        tensor payload. Keeping their collective order independent avoids a
        backend mismatch when inputs contain tensors of several dtypes.

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
        with record_function("## torchrec_maglev:preproc ##"), torch.no_grad():
            layer_inputs = self._preproc(model_input)
        return self.group_by_stage(layer_inputs)

    def take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Hand the schedule ``n`` microbatches for this stage.

        Each raw batch from ``dataloader_iter`` is already one microbatch; this
        method redistributes batches but never chunks them. It runs as many
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
        with record_function("## torchrec_maglev:input_driver ##"):
            return self._input_driver.take(
                lambda: self.send_set(next(dataloader_iter)), n
            )

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
            buffer = tensor.detach().contiguous()
            out.append((dist.isend(buffer, dst=dst, group=pg), buffer))
        return out

    # pyre-ignore[2]: dist work handle has no public type
    def _drain(self, sends: List[Tuple[Any, torch.Tensor]]) -> None:
        """Wait every send in ``sends`` and release the buffers holding it open."""
        for work, _buf in sends:
            work.wait()
        sends.clear()

    # ---- one microbatch of this stage's work ----

    def compute_forward_micro(
        self,
        stage_input: Sequence[Any],
        in_activations: Activations,
        microbatch_id: int,
    ) -> Any:
        """Compute one forward and retain its graph for the matching backward."""
        with record_function(f"## forward mb{microbatch_id} ##"):
            outputs = self.module(stage_input, in_activations)
        self._pending.append((in_activations, outputs, microbatch_id))
        return outputs

    def compute_backward_micro(
        self, grads: Sequence[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Activations]:
        """Compute the oldest pending backward and return its input activation."""
        in_activations, outputs, microbatch_id = self._pending.pop(0)
        loss: Optional[torch.Tensor] = None
        if self.is_last:
            with record_function(f"## backward mb{microbatch_id} ##"):
                loss = outputs[0]
                loss.backward()
        else:
            with record_function(f"## backward mb{microbatch_id} ##"):
                self.backward(outputs, grads)
        return loss, in_activations

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

        outputs = self.compute_forward_micro(stage_input, in_activations, microbatch_id)

        # The previous microbatch's send, drained before this one is issued.
        with record_function(f"## finish_send_act mb{microbatch_id} ##"):
            self.finish_send_act()
        with record_function(f"## send_act mb{microbatch_id} ##"):
            self.start_send_act(outputs)

    def backward_micro(self) -> Optional[torch.Tensor]:
        """One microbatch backward: recv grad, backward, send input grad.

        Pops the oldest microbatch parked by :meth:`forward_micro`. Gradients
        accumulate across microbatches (no ``zero_grad`` here); the schedule runs
        the DP all-reduce and optimizer step once per batch. Profiler ranges are
        tagged with the ``microbatch_id`` recorded at forward time.

        The last stage takes its gradient from the loss its own ``postproc``
        computed; every other stage collects the receive the schedule posted, and
        :meth:`wait_for_grad` raises if it did not.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage,
            ``None`` on every other stage.
        """
        microbatch_id = self._pending[0][2]
        with record_function(f"## recv_grad mb{microbatch_id} ##"):
            grads = self.wait_for_grad()
        loss, in_activations = self.compute_backward_micro(grads)

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
