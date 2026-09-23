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
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining._backward import (
    _get_grad_fn_or_grad_acc,
    stage_backward_input,
    stage_backward_weight,
)
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

    Layers are registered in an ``nn.ModuleDict`` whose keys are their global
    model indices. This preserves checkpoint FQNs after non-local layers are
    discarded: for example, a stage beginning at layer 10 owns parameters under
    ``layers.10`` rather than renumbering them under ``layers.0``. Execution does
    not derive its order by sorting those string keys (which would place ``10``
    before ``2``). ``nn.ModuleDict`` preserves insertion order, the constructor
    inserts the supplied sequence in execution order, and :meth:`forward`
    iterates :meth:`nn.ModuleDict.values` in that same order.

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
        first_layer_index: global index of this stage's first layer, used for
            parameter FQNs and tracing.
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
        self.layers: nn.ModuleDict = nn.ModuleDict(
            {
                str(first_layer_index + index): layer
                for index, layer in enumerate(layers)
            }
        )
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
            self.layers.values(), layer_inputs, self._profile_names
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


@dataclass
class LayerSparseOutput:
    r"""LayerSparseOutput(pooled, seams)

    Store one whole-pass sparse output and its detached microbatch leaves.

    Args:
        pooled (torch.Tensor): Whole-pass output attached to the sparse graph.
        seams (List[torch.Tensor]): Detached leaves consumed by dense forwards.
    """

    pooled: torch.Tensor
    seams: List[torch.Tensor]


@dataclass
class MaglevRailPassState:
    r"""MaglevRailPassState(sparse_outputs, dense_inputs)

    Hold data shared by the sparse and dense phases of one Rail pass.

    Args:
        sparse_outputs (List[Optional[LayerSparseOutput]]): Per-layer sparse
            graphs and seams. ``None`` identifies a dense-only layer.
        dense_inputs (List[List[Any]]): Per-microbatch inputs for local layers.
    """

    sparse_outputs: List[Optional[LayerSparseOutput]]
    dense_inputs: List[List[Any]]

    def seams_for(self, microbatch_id: int) -> List[Optional[torch.Tensor]]:
        r"""seams_for(microbatch_id) -> List[Optional[torch.Tensor]]

        Return one sparse seam per local layer for a dense microbatch.

        Args:
            microbatch_id (int): Dense microbatch index.

        Returns:
            List[Optional[torch.Tensor]]: Per-layer sparse outputs.
        """
        return [
            None if output is None else output.seams[microbatch_id]
            for output in self.sparse_outputs
        ]


_RailLayerForward = Tuple[Optional[torch.Tensor], List[Any]]


def _reject_unhalved_parameters(layers: Sequence[MaglevLayer]) -> None:
    """Require every trainable layer parameter to live in one of the two halves.

    The split backward computes weight gradients for ``dense_parameters()`` only,
    and reaches the sparse side through the seams. A parameter registered on the
    layer body is in neither set: ``torch.autograd.grad`` runs no
    ``AccumulateGrad`` for it, so its ``.grad`` stays ``None`` after
    ``zero_grad()`` and the optimizer skips it. It never trains, with no error --
    which is exactly the mistake a layer migrating to the split API makes by
    leaving one submodule behind.

    Raises:
        ValueError: if a layer owns a trainable parameter outside both halves.
    """
    for index, layer in enumerate(layers):
        halved = {
            id(parameter)
            for half in (layer.sparse, layer.dense)
            if half is not None
            for parameter in half.parameters()
        }
        stray = sorted(
            name
            for name, parameter in layer.named_parameters()
            if parameter.requires_grad and id(parameter) not in halved
        )
        if stray:
            raise ValueError(
                f"{type(layer).__name__} at index {index} owns trainable "
                f"parameters outside its sparse and dense halves: {stray}. "
                "MaglevRail computes weight gradients from the dense half only, "
                "so these would silently never train -- move them into the "
                "sparse or dense submodule"
            )


def _reject_uncovered_weights(
    param_groups: Sequence[Mapping[str, Any]],
    weights: Sequence[nn.Parameter],
) -> None:
    """Refuse a weight half that would silently skip some of its parameters.

    ``get_param_groups`` records a parameter only when its reverse closure
    intersects the closure of the ``input_values`` it was given, and upstream's
    own coverage assertion for that is commented out
    (``pipelining/_backward.py``). A dense parameter whose path to the output
    never merges with a path descended from a received activation or a sparse
    seam therefore produces no weight group, ``stage_backward_weight`` writes no
    ``.grad`` for it, and it never trains -- with no exception anywhere.

    Cheap: a set of ids over the stage's dense parameters, once per microbatch.

    Raises:
        RuntimeError: if a trainable dense parameter has no weight group.
    """
    trainable = [weight for weight in weights if weight.requires_grad]
    # By identity, not by count. ``param_groups`` holds grad-accumulator nodes,
    # the same keys ``stage_backward_weight`` looks weights up by, and a count
    # comparison passes when one node appears in two groups while another
    # parameter is missing entirely -- which is exactly the case this exists to
    # catch.
    covered = {id(node) for group in param_groups for node in group["params"]}
    missing = [
        weight
        for weight in trainable
        if id(_get_grad_fn_or_grad_acc(weight)) not in covered
    ]
    if not missing:
        return
    raise RuntimeError(
        f"the split backward reached {len(trainable) - len(missing)} of "
        f"{len(trainable)} trainable dense parameters, so {len(missing)} would "
        "receive no gradient and never train. A dense parameter is reachable "
        "only if its path to the stage output merges with one descended from an "
        "incoming activation or a sparse seam; one fed solely by "
        "non-differentiable inputs is not"
    )


class _RailLayerChain(_LayerChain):
    r"""A layer chain with separate sparse and dense Rail execution phases."""

    def __init__(
        self,
        layers: Sequence[MaglevLayer],
        postproc: Optional[Callable[[Activations, Any], Any]] = None,
        first_layer_index: int = 0,
        loss_only_output: bool = False,
    ) -> None:
        super().__init__(layers, postproc, first_layer_index, loss_only_output)
        # Author time, before any wrapper obscures which half owns what.
        _reject_unhalved_parameters(layers)

    def forward(
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations = (),
        *,
        sparse_outputs: Optional[Sequence[Optional[torch.Tensor]]] = None,
        batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
    ) -> Any:
        r"""forward(layer_inputs, in_activations=(), *, sparse_outputs=None, batch_size=None, num_microbatches=None) -> Any

        Run the global sparse phase, or dispatch a dense microbatch through the
        wrapper-visible ``forward`` entry point.

        Args:
            layer_inputs (Sequence[Any]): One input per local layer.
            in_activations (Activations): Activation entering the stage.
                Default: ``()``
            sparse_outputs (Sequence[Optional[torch.Tensor]], optional): One
                precomputed sparse output per layer. Supplying this dispatches
                to :meth:`forward_dense`. Default: ``None``
            batch_size (int, optional): Whole-pass batch size. Required when
                ``sparse_outputs`` is ``None``. Default: ``None``
            num_microbatches (int, optional): Number of dense microbatches,
                required when ``sparse_outputs`` is ``None``. Default: ``None``

        Returns:
            Any: Per-layer sparse outputs and dense inputs, or the result of
            :meth:`forward_dense` when ``sparse_outputs`` is supplied.

        Raises:
            ValueError: If the input counts or phase arguments are invalid.
        """
        if len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        if sparse_outputs is not None:
            # Dispatch through forward so an outer DMP/DDP/FSDP wrapper observes
            # the dense call before this implementation-specific method runs.
            if batch_size is not None or num_microbatches is not None:
                raise ValueError(
                    "dense forward does not accept batch_size or num_microbatches"
                )
            return self.forward_dense(
                layer_inputs,
                in_activations,
                sparse_outputs,
            )
        if in_activations:
            raise ValueError("sparse forward does not accept incoming activations")
        if batch_size is None or num_microbatches is None:
            raise ValueError("sparse forward requires batch_size and num_microbatches")

        rail_outputs: List[_RailLayerForward] = []
        for layer, layer_input, profile_name in zip(
            self.layers.values(), layer_inputs, self._profile_names
        ):
            maglev_layer = cast(MaglevLayer, layer)
            with record_function(profile_name):
                sparse = maglev_layer.sparse
                pooled = None if sparse is None else sparse(layer_input)
                if sparse is not None:
                    if not isinstance(pooled, torch.Tensor):
                        raise TypeError(
                            f"{type(maglev_layer).__name__} sparse half returned "
                            f"{type(pooled).__name__}, expected torch.Tensor"
                        )
                    if pooled.ndim == 0 or pooled.shape[0] != batch_size:
                        raise ValueError(
                            f"{type(maglev_layer).__name__} sparse output has shape "
                            f"{tuple(pooled.shape)}, expected leading dimension "
                            f"{batch_size}"
                        )
                    if not pooled.dtype.is_floating_point:
                        raise ValueError(
                            f"{type(maglev_layer).__name__} sparse output must be "
                            f"floating point, got {pooled.dtype}"
                        )
                dense_inputs = maglev_layer.split_dense_input(
                    layer_input,
                    batch_size,
                    num_microbatches,
                )
                if len(dense_inputs) != num_microbatches:
                    raise ValueError(
                        f"{type(maglev_layer).__name__}.split_dense_input() returned "
                        f"{len(dense_inputs)} inputs, expected {num_microbatches}"
                    )
            rail_outputs.append((pooled, dense_inputs))
        return rail_outputs

    def forward_dense(
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations,
        sparse_outputs: Sequence[Optional[torch.Tensor]],
    ) -> Any:
        r"""forward_dense(layer_inputs, in_activations, sparse_outputs) -> Any

        Run one dense microbatch through the stage's layers and postprocessor.

        Args:
            layer_inputs (Sequence[Any]): One dense input per local layer.
            in_activations (Activations): Activation entering the stage.
            sparse_outputs (Sequence[Optional[torch.Tensor]]): One precomputed
                sparse output per local layer.

        Returns:
            Any: The stage activation or final-stage postprocessor result.

        Raises:
            ValueError: If an input count does not match the layer count.
        """
        if len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        if len(sparse_outputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} sparse outputs, got "
                f"{len(sparse_outputs)}"
            )
        activations = in_activations
        for index, (layer, layer_input, profile_name) in enumerate(
            zip(self.layers.values(), layer_inputs, self._profile_names)
        ):
            maglev_layer = cast(MaglevLayer, layer)
            with record_function(profile_name):
                activations = maglev_layer.require_dense()(
                    sparse_outputs[index], layer_input, activations
                )
        if self._postproc is not None:
            with record_function("## torchrec_maglev:postproc ##"):
                output = self._postproc(activations, layer_inputs[-1])
            if self._loss_only_output:
                losses, _model_output = output
                return (losses,)
            return output
        return activations


class _StageLoss(torch.autograd.Function):
    """Expose a stage's ordered cross-rank backward as an autograd edge."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, stage: "StageWrapper") -> torch.Tensor:
        ctx.stage = stage
        return value.clone()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        stage = cast(StageWrapper, ctx.stage)
        stage._backward_round(cast(torch.Tensor, grad_outputs[0]))
        return None, None


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

    The stage reduces no gradients itself. Functional execution combines the
    microbatch graphs into one autograd traversal, while explicit schedules use
    their DP wrapper's synchronization controls; either way, that wrapper's
    reducer fires once per pass (see
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
        enable_rail: whether to build the Rail-specific sparse/dense stage module.
            This must be selected before wrapping :attr:`module`. Default is
            ``False``.

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
        loss = stage(raw_batch)
        loss.backward()
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
        enable_rail: bool = False,
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
        layer_chain = _RailLayerChain if enable_rail else _LayerChain
        self.module: nn.Module = layer_chain(
            layers,
            model.postproc if self.is_last_stage else None,
            first_layer_index=self.layer_indices.start,
            loss_only_output=loss_only_output,
        )
        self._rail_enabled: bool = enable_rail
        # Kept because :attr:`module` is reassignable: once a caller wraps it in
        # DMP/DDP/FSDP the authored layers are no longer reachable from it, and
        # the split backward needs their dense halves by then.
        self._layers: List[MaglevLayer] = list(layers)
        # Rail only. Sparse seams parallel to :attr:`_pending`, so the split
        # backward can list them among its differentiable inputs; and the weight
        # work each ``I`` deferred, oldest first.
        self._rail_seams: List[Optional[Sequence[Optional[torch.Tensor]]]] = []
        self._wdense: List[Tuple[Any, int]] = []
        self._loss_only_output = loss_only_output
        # Set by to(): a meta-authored model has no device to infer, which is the
        # whole point of authoring it there.
        self.device: Optional[torch.device] = None

    @property
    def loss_only_output(self) -> bool:
        """Whether the final stage exposes only the postprocessed loss."""
        return self._loss_only_output

    @property
    def rail_enabled(self) -> bool:
        r"""Return whether this stage was built with Rail phase support."""
        return self._rail_enabled

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

    def get_batch_size(self, stage_input: Sequence[Any]) -> int:
        r"""get_batch_size(stage_input) -> int

        Return and validate the logical batch size of local layer inputs.

        Args:
            stage_input (Sequence[Any]): One input per local layer.

        Returns:
            int: Logical batch size.

        Raises:
            ValueError: If the model reports a negative batch size.
        """
        batch_size = self._get_batch_size(stage_input)
        if batch_size < 0:
            raise ValueError(f"batch size must be non-negative, got {batch_size}")
        return batch_size

    def _activation_batch_size(
        self, stage_input: Optional[Sequence[Any]]
    ) -> Optional[int]:
        """Return this microbatch's size when a boundary depends on it."""
        if not any(spec.batch_size_dependent for spec in self._in_specs):
            return None
        if stage_input is None:
            raise ValueError(
                "stage input is required for a batch-size-dependent activation"
            )
        return self.get_batch_size(stage_input)

    def forward(self, model_input: Any) -> torch.Tensor:
        """Distribute and forward every microbatch produced by one raw batch.

        Args:
            model_input: one raw dataloader batch. The model's ``preproc`` creates
                per-layer inputs, and one input-distribution round exchanges them
                across the cascade. The batch is already a microbatch; it is not
                split.

        Returns:
            torch.Tensor: the summed microbatch loss on the final stage, and a
            zero-valued backward token on earlier stages. Every rank must call
            ``backward`` on this tensor; its autograd edge drains the matching
            cross-stage backwards in order.

        Pipeline schedules use the lower-level ``*_micro`` methods to choose a
        different forward/backward ordering. The returned value is meaningful
        for reporting only on the final stage.
        """
        with record_function("## torchrec_maglev:input_driver ##"):
            microbatch_inputs = self._input_driver.exchange(
                self.send_set(model_input)
            ).wait()

        outputs: List[Any] = []
        for microbatch_id, stage_input in enumerate(microbatch_inputs):
            self.start_recv_act(stage_input)
            outputs.append(self.forward_micro(stage_input, microbatch_id))
        self.finish_send_act()
        if self.is_last:
            value = torch.stack([output[0] for output in outputs]).sum().detach()
        else:
            first_outputs = cast(Activations, outputs[0])
            value = first_outputs[0].new_zeros(())
        value.requires_grad_(True)
        return cast(torch.Tensor, _StageLoss.apply(value, self))

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

    def start_recv_act(self, stage_input: Optional[Sequence[Any]] = None) -> None:
        """Ensure a receive is posted for the previous HSD's activation.

        Allocates one buffer per incoming spec and issues the receives in spec
        order, so both sides of the boundary agree without exchanging metadata.
        Collect them with :meth:`wait_for_act`.

        No-op on the first stage. Otherwise each call posts another receive and
        queues it, so a schedule can run several boundaries ahead of the compute;
        :meth:`wait_for_act` dequeues them in issue order.

        Args:
            stage_input: the local inputs for the activation being received. Only
                required when an incoming spec has a dynamic batch dimension.
        """
        if self.is_first:
            return
        batch_size = self._activation_batch_size(stage_input)
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
        next_stage_input: Optional[Sequence[Any]] = None,
    ) -> Activations:
        """Exchange an upstream gradient for the next forward activation.

        Available only in ``SHARED`` mode; see :meth:`send_act_recv_grad`.

        Args:
            in_activations: activations whose gradients are sent upstream.
            recv_next: whether to receive the next forward activation.
            next_stage_input: local inputs for the next activation microbatch.
                Only required when ``recv_next`` and an incoming spec has a
                dynamic batch dimension.
        """
        if self.is_first:
            return ()
        if self.handoff_pg_mode is not HandoffPGMode.SHARED:
            raise RuntimeError("batched handoff requires handoff_pg_mode=shared")
        peer = self.neighbor_rank(-1)
        pg = self.handoff_pgs[0]
        batch_size = (
            self._activation_batch_size(next_stage_input) if recv_next else None
        )
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
        """Hand the schedule ``n`` redistributed inputs for this stage.

        This method redistributes batches but never chunks them. The schedule
        decides whether each input is a microbatch or a whole pass. It runs as
        many :meth:`input_dist` rounds as needed and keeps the remainder queued,
        so the requested count need not equal the ``num_stages`` a round
        produces, and a batch is consumed only when a round actually runs.
        Every rank in the cascade runs the same number of rounds, so every rank
        advances its dataloader in lock-step -- see
        :class:`~torchrec.distributed.maglev.input_dist.InputDistDriver`.

        Args:
            dataloader_iter: yields raw batches, one per round.
            n: how many redistributed inputs the schedule wants.

        Returns:
            List[List[Any]]: ``n`` inputs, each containing one value per local
            layer.
        """
        with record_function("## torchrec_maglev:input_driver ##"):
            return self._input_driver.take(
                lambda: self.send_set(next(dataloader_iter)), n
            )

    def take_global_inputs(self, dataloader_iter: Iterator[Any]) -> List[Any]:
        r"""take_global_inputs(dataloader_iter) -> List[Any]

        Return one local-stage input whose batch covers an entire Rail pass.

        The input driver does not reshape the batch. A Rail dataloader therefore
        supplies whole-pass batches, while ordinary schedules supply individual
        microbatches through :meth:`take_inputs`.

        Args:
            dataloader_iter (Iterator[Any]): Iterator over whole-pass batches.

        Returns:
            List[Any]: One whole-pass input per local layer.
        """
        with record_function("## torchrec_maglev:global_inputs ##"):
            (global_inputs,) = self.take_inputs(dataloader_iter, 1)
        return global_inputs

    def _backward_activations(
        self, outputs: Activations, grads: Sequence[torch.Tensor]
    ) -> None:
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
                self._backward_activations(outputs, grads)
        return loss, in_activations

    def forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int = 0,
    ) -> Any:
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
        return outputs

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

    def _backward_round(self, loss_grad: torch.Tensor) -> None:
        """Drain one functional forward as a single autograd traversal.

        DDP shares one reducer across all outstanding forwards. Separate
        backwards would therefore reduce the first microbatch before the later
        graphs contribute their gradients; one traversal makes its parameter
        hooks observe the accumulated pass instead.
        """
        pending = self._pending
        if self.is_last:
            losses = [outputs[0] for _inputs, outputs, _id in pending]
            with record_function("## backward round ##"):
                torch.autograd.backward(losses, [loss_grad] * len(losses))
        else:
            for _ in pending:
                self.start_recv_grad()

            backward_outputs: List[torch.Tensor] = []
            backward_grads: List[torch.Tensor] = []
            for _in_activations, outputs, microbatch_id in pending:
                with record_function(f"## recv_grad mb{microbatch_id} ##"):
                    grads = self.wait_for_grad()
                grad_carrying = [
                    output
                    for output, spec in zip(cast(Activations, outputs), self._out_specs)
                    if spec.requires_grad
                ]
                for output, grad in zip(grad_carrying, grads):
                    if output.requires_grad:
                        backward_outputs.append(output)
                        backward_grads.append(grad)
            if backward_outputs:
                with record_function("## backward round ##"):
                    torch.autograd.backward(backward_outputs, backward_grads)

        self._pending = []
        for in_activations, _outputs, microbatch_id in pending:
            with record_function(f"## finish_send_grad mb{microbatch_id} ##"):
                self.finish_send_grad()
            with record_function(f"## send_grad mb{microbatch_id} ##"):
                self.start_send_grad(in_activations)
        self.finish_send_grad()

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

    # ---- MaglevRail's global sparse pass and dense microbatch work ----

    def _prepare_rail_layer(
        self,
        pooled: Optional[torch.Tensor],
        dense_inputs: List[Any],
        num_microbatches: int,
    ) -> Tuple[Optional[LayerSparseOutput], List[Any]]:
        sparse_output: Optional[LayerSparseOutput] = None
        if pooled is not None:
            seams = [
                part.detach().requires_grad_(True)
                for part in torch.tensor_split(pooled, num_microbatches, dim=0)
            ]
            sparse_output = LayerSparseOutput(pooled=pooled, seams=seams)
        return sparse_output, dense_inputs

    def sparse_forward_global(
        self,
        global_inputs: Sequence[Any],
        num_microbatches: int,
    ) -> MaglevRailPassState:
        r"""sparse_forward_global(global_inputs, num_microbatches) -> MaglevRailPassState

        Run each sparse half once and create detached dense microbatch seams.

        Args:
            global_inputs (Sequence[Any]): One whole-pass input per local layer.
            num_microbatches (int): Number of dense microbatches.

        Returns:
            MaglevRailPassState: Sparse graphs, seams, and dense inputs.

        Raises:
            RuntimeError: If this stage was not built with Rail support.
            TypeError: If a sparse half does not return a tensor.
            ValueError: If input counts or batch dimensions are invalid.
        """
        if not self.rail_enabled:
            raise RuntimeError(
                "sparse_forward_global requires StageWrapper(..., enable_rail=True)"
            )
        if len(global_inputs) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} layer inputs, got {len(global_inputs)}"
            )
        if num_microbatches <= 0:
            raise ValueError(
                f"num_microbatches must be positive, got {num_microbatches}"
            )
        batch_size = self.get_batch_size(global_inputs)
        if batch_size < num_microbatches:
            raise ValueError(
                f"batch size {batch_size} cannot produce {num_microbatches} "
                "non-empty microbatches"
            )
        if batch_size % num_microbatches:
            raise ValueError(
                f"batch size {batch_size} must divide evenly into "
                f"{num_microbatches} microbatches"
            )

        sparse_outputs: List[Optional[LayerSparseOutput]] = []
        dense_inputs_by_layer: List[List[Any]] = []
        with record_function("## torchrec_maglev:sparse_forward_global ##"):
            layer_outputs = cast(
                List[_RailLayerForward],
                self.module(
                    global_inputs,
                    batch_size=batch_size,
                    num_microbatches=num_microbatches,
                ),
            )
            if len(layer_outputs) != self.num_layers:
                raise ValueError(
                    f"sparse phase returned {len(layer_outputs)} layer outputs, "
                    f"expected {self.num_layers}"
                )
            for pooled, dense_inputs in layer_outputs:
                sparse_output, dense_inputs = self._prepare_rail_layer(
                    pooled,
                    dense_inputs,
                    num_microbatches,
                )
                sparse_outputs.append(sparse_output)
                dense_inputs_by_layer.append(dense_inputs)

        dense_inputs = [
            [
                dense_inputs_by_layer[layer][microbatch]
                for layer in range(self.num_layers)
            ]
            for microbatch in range(num_microbatches)
        ]
        expected_batch_size = batch_size // num_microbatches
        for microbatch, dense_input in enumerate(dense_inputs):
            actual_batch_size = self.get_batch_size(dense_input)
            if actual_batch_size != expected_batch_size:
                raise ValueError(
                    f"dense microbatch {microbatch} has batch size "
                    f"{actual_batch_size}, expected {expected_batch_size}"
                )
        return MaglevRailPassState(
            sparse_outputs=sparse_outputs,
            dense_inputs=dense_inputs,
        )

    def sparse_backward_global(self, state: MaglevRailPassState) -> None:
        r"""sparse_backward_global(state) -> None

        Resume each whole-pass sparse graph from its dense seam gradients.

        Args:
            state (MaglevRailPassState): State produced by
                :meth:`sparse_forward_global`.
        """
        with record_function("## torchrec_maglev:sparse_backward_global ##"):
            for output in state.sparse_outputs:
                if output is None or not output.pooled.requires_grad:
                    continue
                gradient = torch.cat(
                    [
                        seam.grad if seam.grad is not None else torch.zeros_like(seam)
                        for seam in output.seams
                    ],
                    dim=0,
                )
                torch.autograd.backward(output.pooled, gradient)

    def compute_dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        in_activations: Activations,
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[torch.Tensor]],
    ) -> Any:
        r"""compute_dense_forward_micro(stage_input, in_activations, microbatch_id, sparse_outputs) -> Any

        Compute one dense-only forward and retain its graph for backward.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            in_activations (Activations): Activation entering this stage.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[torch.Tensor]]): Precomputed sparse
                outputs for each local layer.

        Returns:
            Any: Stage outputs retained for backward.

        Raises:
            RuntimeError: If this stage was not built with Rail support.
        """
        if not self.rail_enabled:
            raise RuntimeError(
                "compute_dense_forward_micro requires "
                "StageWrapper(..., enable_rail=True)"
            )
        with record_function(f"## dense_forward mb{microbatch_id} ##"):
            outputs = self.module(
                stage_input,
                in_activations,
                sparse_outputs=sparse_outputs,
            )
        self._pending.append((in_activations, outputs, microbatch_id))
        # A whole backward fills every leaf's ``.grad`` implicitly; the split
        # backward fills only the inputs it is handed, so the seams must survive.
        self._rail_seams.append(sparse_outputs)
        return outputs

    def dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[torch.Tensor]],
    ) -> Any:
        r"""dense_forward_micro(stage_input, microbatch_id, sparse_outputs) -> Any

        Receive, compute, and send one Rail dense microbatch.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[torch.Tensor]]): Precomputed sparse
                outputs for each local layer.

        Returns:
            Any: Stage outputs retained for dense backward.
        """
        with record_function(f"## dense_recv_act mb{microbatch_id} ##"):
            in_activations = self.wait_for_act()
        outputs = self.compute_dense_forward_micro(
            stage_input,
            in_activations,
            microbatch_id,
            sparse_outputs,
        )
        with record_function(f"## dense_finish_send_act mb{microbatch_id} ##"):
            self.finish_send_act()
        with record_function(f"## dense_send_act mb{microbatch_id} ##"):
            self.start_send_act(outputs)
        return outputs

    def reset_pass_state(self) -> None:
        r"""reset_pass_state() -> None

        Drop the per-microbatch state a pass accumulates.

        The queues are FIFOs: appended in forward, popped in backward. A pass
        that raises part-way leaves them populated, and a caller that catches
        the exception and calls :meth:`progress` again then pops the *previous*
        pass's entries, pairing one microbatch's saved activations with
        another's gradients. Clearing them turns that silent mispairing into an
        ordinary empty-queue error.

        This is local cleanup, not recovery. Outstanding point-to-point work is
        abandoned rather than waited on -- waiting deadlocks, because the peer
        that would match a posted recv is typically the rank that already
        failed. After a raised pass the handoff communicators are out of step
        across ranks and the job cannot continue; restart it. What this buys is
        that the wreckage is loud rather than quiet.
        """
        self._send_act = []
        self._send_grad = []
        self._recv_act.clear()
        self._recv_grad.clear()
        self._pending = []
        self._rail_seams = []
        self._wdense = []

    def dense_parameters(self) -> List[nn.Parameter]:
        r"""dense_parameters() -> List[nn.Parameter]

        The parameters the split backward computes weight gradients for.

        A list, not an iterator: the ``I`` and ``W`` halves each need the full
        set, and handing the same exhausted generator to both fails as a
        ``KeyError`` deep inside torch rather than as anything legible.

        Exact by construction rather than by name filtering: whatever the layers'
        dense halves own. A final-stage postprocessor with parameters of its own
        is therefore *not* included -- a Rail model's head must have no
        parameters of its own.

        Returns:
            Iterator[nn.Parameter]: dense parameters, in layer order.
        """
        return [
            parameter
            for layer in self._layers
            for parameter in layer.dense_parameters()
        ]

    @property
    def pending_weight_work(self) -> int:
        """Microbatches whose input gradient is done but weight gradient is not."""
        return len(self._wdense)

    def dense_backward_act_micro(self) -> Optional[torch.Tensor]:
        r"""dense_backward_act_micro() -> Optional[torch.Tensor]

        The ``I`` half: gradients w.r.t. this stage's inputs, and nothing else.

        Computes only what the pipeline is waiting for. The input gradient is what
        the previous stage needs, so it is produced and sent immediately; the
        weight gradient gates nothing until the optimizer step, so it is deferred
        to :meth:`dense_backward_weight_micro`.

        The win is not that ``W`` fills this stage's idle time -- during warmup a
        stage has run forwards and no backwards, so no ``W`` exists yet. It is
        that every hop of the returning gradient wave costs ``I`` instead of
        ``I + W``, so the wave reaches stage 0 sooner and its stall is shorter.

        ``stage_backward_input`` writes ``.grad`` on everything in
        ``input_values``, which is why both classes of input go in together: the
        received activations, whose gradients go upstream, and the sparse seams,
        whose gradients :meth:`sparse_backward_global` resumes from. The seams are
        filled here, by ``I``, never by the weight half.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage, ``None``
            elsewhere. ``stage_backward_input`` detaches its roots, so that loss
            keeps its value but is no longer a graph root.
        """
        in_activations, outputs, microbatch_id = self._pending.pop(0)
        seams = self._rail_seams.pop(0)

        loss: Optional[torch.Tensor] = None
        root_grads: Optional[List[torch.Tensor]] = None
        if self.is_last:
            # postproc produced the loss; it is the graph root and its gradient
            # is implicit, so no output gradients are supplied.
            loss = cast(Activations, outputs)[0]
            roots: List[torch.Tensor] = [loss]
        else:
            with record_function(f"## dense_recv_grad mb{microbatch_id} ##"):
                grads = self.wait_for_grad()
            grad_carrying = [
                tensor
                for tensor, spec in zip(cast(Activations, outputs), self._out_specs)
                if spec.requires_grad
            ]
            if len(grads) != len(grad_carrying):
                # zip would truncate to the shorter side and drop the tail
                # output's gradient without a word -- a spec/wire mismatch is a
                # wiring bug, not something to silently absorb.
                raise ValueError(
                    f"stage {self.stage_index} received {len(grads)} gradients "
                    f"for {len(grad_carrying)} grad-carrying outputs"
                )
            pairs = [
                (output, grad)
                for output, grad in zip(grad_carrying, grads)
                if output.requires_grad
            ]
            roots = [output for output, _ in pairs]
            root_grads = [grad for _, grad in pairs]

        input_values: List[torch.Tensor] = [
            tensor
            for tensor, spec in zip(in_activations, self._in_specs)
            if spec.requires_grad
        ]
        if seams is not None:
            input_values.extend(seam for seam in seams if seam is not None)

        if roots and input_values:
            with record_function(f"## dense_backward_act mb{microbatch_id} ##"):
                _input_grads, param_groups = stage_backward_input(
                    stage_outputs_or_loss=roots,
                    output_grads=root_grads,
                    input_values=input_values,
                    # iter() only to satisfy upstream's Iterator annotation;
                    # both halves traverse the same list, so the weight groups
                    # the I half records line up with what W looks up.
                    weights=iter(self.dense_parameters()),
                )
            _reject_uncovered_weights(param_groups, self.dense_parameters())
            self._wdense.append((param_groups, microbatch_id))
        elif roots:
            # Nothing differentiable coming in: the first stage, holding only
            # layers with no sparse half. ``stage_backward_input`` traces *from*
            # the inputs, so with none it returns empty weight groups and the
            # weight gradients would silently never be computed. There is also
            # nothing worth deferring -- no input gradient means no wave to start,
            # which is the only reason to split -- so run the whole backward.
            with record_function(f"## dense_backward_act mb{microbatch_id} (whole) ##"):
                torch.autograd.backward(roots, root_grads)

        with record_function(f"## dense_finish_send_grad mb{microbatch_id} ##"):
            self.finish_send_grad()
        with record_function(f"## dense_send_grad mb{microbatch_id} ##"):
            self.start_send_grad(in_activations)
        return loss

    def dense_backward_weight_micro(self) -> None:
        r"""dense_backward_weight_micro() -> None

        The ``W`` half: gradients w.r.t. the dense weights, for one microbatch.

        Runs what :meth:`dense_backward_act_micro` deferred, resuming from the
        intermediates it saved. It writes ``.grad`` through ``torch.autograd.grad``
        rather than through an ``AccumulateGrad`` node, so no data-parallel
        wrapper's hook observes it -- which is why a schedule using this must
        either issue the pass's gradient reduction itself or refuse to run under a
        wrapper. The accumulation order also differs from a whole backward, so the
        two dense schedules agree to floating-point tolerance, not bit-exactly.

        Gates nothing: place it wherever the stage would otherwise idle. The queue
        must be drained before the optimizer step.

        Raises:
            ValueError: if no deferred weight work is queued.
        """
        if not self._wdense:
            raise ValueError(
                f"stage {self.stage_index}: dense_backward_weight_micro() with no "
                "deferred weight work; call dense_backward_act_micro() first"
            )
        param_groups, microbatch_id = self._wdense.pop(0)
        with record_function(f"## dense_backward_weight mb{microbatch_id} ##"):
            stage_backward_weight(iter(self.dense_parameters()), param_groups)
