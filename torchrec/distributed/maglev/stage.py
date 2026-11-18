#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Cutting a Maglev model into stages and running one stage's share of a pass.

Everything that knows about ranks lives here: the process groups, the
layers-to-stages cut, and :class:`StageWrapper`, which binds one stage's layers
to its HSD, owns the wire between stages, and runs the stage's half of every
phase of a pass. The authoring side is in :mod:`torchrec.distributed.maglev.module`;
the order of the phases is the schedule's, in :mod:`torchrec.distributed.maglev.pipeline`.
"""

from __future__ import annotations

import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.graph import GradientEdge
from torch.autograd.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining._backward import (
    _get_grad_fn_or_grad_acc,
    stage_backward_input,
    stage_backward_weight,
)
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.checks import (
    check_sparse_output,
    DenseWeightChecks,
    reject_parameters_outside_layers,
    reject_unhalved_parameters,
)
from torchrec.distributed.maglev.data_parallel import fsdp_graph_parameters
from torchrec.distributed.maglev.input_dist import InputDistDriver
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    check_activations,
    check_layers_chain,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.types import LazyAwaitable, ShardingPlan
from torchrec.distributed.utils import init_parameters


# ---- process groups ----


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
    handoff_pg_mode: HandoffPGMode = HandoffPGMode.SPLIT

    def __post_init__(self) -> None:
        normalized_ranks = self._normalize_stage_ranks(self.stage_ranks)
        object.__setattr__(self, "stage_ranks", normalized_ranks)
        handoff_pgs_are_shared = self.handoff_pgs[0] is self.handoff_pgs[1]
        if self.handoff_pg_mode is HandoffPGMode.SHARED:
            if not handoff_pgs_are_shared:
                raise ValueError(
                    "shared handoff mode does not match the supplied process groups"
                )
        elif handoff_pgs_are_shared and self.num_stages > 1:
            raise ValueError(
                "activation and gradient handoffs need separate process groups: "
                "on one communicator the two directions between neighboring "
                "stages deadlock once a message is too large to send eagerly; "
                "pass handoff_pg_mode=HandoffPGMode.SHARED to batch them instead"
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
        handoff_pg_mode: HandoffPGMode = HandoffPGMode.SPLIT,
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


# ---- one stage's share of a pass ----

# What a sparse half returns and a dense half receives: one batch-major tensor,
# or a tuple of them (mixed dtypes, or MaglevLayer.shared_sparse_outputs).
SparseSeam = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


_SparseLayerOutput = Tuple[Optional[SparseSeam], List[Any]]


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


def _seam_tensors(seam: SparseSeam) -> Tuple[torch.Tensor, ...]:
    return (seam,) if isinstance(seam, torch.Tensor) else tuple(seam)


def _cut_seams(
    pooled: SparseSeam, num_microbatches: int, shared: Sequence[int] = ()
) -> List[SparseSeam]:
    """Cut one whole-pass sparse output into detached per-microbatch seams.

    A batch-major tensor is split by rows; a tensor at a ``shared`` position is
    one leaf that every microbatch's seam holds, so the dense backwards'
    gradients accumulate into its ``.grad``.
    """
    leaves = [
        (
            [tensor.detach().requires_grad_(True)] * num_microbatches
            if position in shared
            else [
                part.detach().requires_grad_(True)
                for part in torch.tensor_split(tensor, num_microbatches, dim=0)
            ]
        )
        for position, tensor in enumerate(_seam_tensors(pooled))
    ]
    return [
        (
            leaves[0][index]
            if isinstance(pooled, torch.Tensor)
            else tuple(parts[index] for parts in leaves)
        )
        for index in range(num_microbatches)
    ]


def _accumulate_weight_grads(param_groups: List[Dict[str, Any]]) -> None:
    """Upstream ``stage_backward_weight``, run through each weight's accumulator.

    Upstream assigns ``w.grad`` from Python, so no post-accumulate hook fires,
    and DDP's reducer learns which parameters a pass used only from that hook.
    Passing the edges as ``inputs`` runs the accumulators instead, with the same
    ``.grad``. TODO: upstream an ``accumulate_grad`` option.

    Args:
        param_groups: the weight groups ``stage_backward_input`` recorded.
    """
    for param_group in param_groups:
        edges: List[GradientEdge] = []
        grads: List[torch.Tensor] = []
        for grads_tuple, intermediate in zip(
            param_group["grads"], param_group["intermediates"]
        ):
            for i, grad in enumerate(grads_tuple):
                if grad is not None:
                    edges.append(GradientEdge(intermediate, i))
                    grads.append(grad)
        handles = []
        try:
            if len(param_group["intermediates"]) > 1:
                for grads_tuple, intermediate in zip(
                    param_group["grads"], param_group["intermediates"], strict=True
                ):
                    handles.append(
                        intermediate.register_prehook(
                            lambda grad_outputs, saved=grads_tuple: saved
                        )
                    )
            # Break the hook -> param_group -> intermediates cycle, as upstream does.
            del param_group["intermediates"]
            if edges:
                torch.autograd.backward(
                    edges,
                    grad_tensors=grads,
                    inputs=[GradientEdge(node, 0) for node in param_group["params"]],
                )
                del param_group["grads"]
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class LayerSparseOutput:
    r"""LayerSparseOutput(pooled, seams)

    One layer's whole-pass sparse output and its seams, by microbatch.

    Args:
        pooled (SparseSeam): The sparse half's whole-pass output.
        seams (List[Optional[SparseSeam]]): Per microbatch, the detached slice
            the microbatch's dense forward reads.
    """

    pooled: SparseSeam
    seams: List[Optional[SparseSeam]]


@dataclass
class MaglevPassState:
    r"""MaglevPassState(pooled, seams, dense_inputs)

    What a pass's sparse forward hands its dense microbatches and its sparse
    backward.

    Args:
        pooled (List[Optional[SparseSeam]]): Per local layer, the sparse half's
            whole-pass output; ``None`` for a layer with no sparse half.
        seams (List[List[Optional[SparseSeam]]]): Per microbatch, a detached
            slice of each ``pooled``. A shared position
            (``MaglevLayer.shared_sparse_outputs``) holds one whole-pass leaf.
        dense_inputs (List[List[Any]]): Per microbatch, one dense input per
            local layer.
    """

    pooled: List[Optional[SparseSeam]]
    seams: List[List[Optional[SparseSeam]]]
    dense_inputs: List[List[Any]]

    @property
    def sparse_outputs(self) -> List[Optional[LayerSparseOutput]]:
        r"""sparse_outputs -> List[Optional[LayerSparseOutput]]

        Per local layer, its whole-pass output and its seams by microbatch;
        ``None`` for a layer with no sparse half.
        """
        return [
            (
                None
                if pooled is None
                else LayerSparseOutput(pooled, [seams[index] for seams in self.seams])
            )
            for index, pooled in enumerate(self.pooled)
        ]

    def seams_for(self, microbatch_id: int) -> List[Optional[SparseSeam]]:
        r"""seams_for(microbatch_id) -> List[Optional[SparseSeam]]

        Return microbatch ``microbatch_id``'s seam for each local layer.
        """
        return list(self.seams[microbatch_id])


MaglevRailPassState = MaglevPassState


def _runs_halves(layer: nn.Module) -> bool:
    """Whether ``layer`` runs on its halves: a dense half and the default forward."""
    return (
        isinstance(layer, MaglevLayer)
        and layer.dense is not None
        and type(layer).forward is MaglevLayer.forward
    )


class _LayerChain(nn.Module):
    """A stage's layers as one module, for the parallelizers to wrap.

    Layers are registered under their global model indices, so a stage starting
    at layer 10 keeps the parameter FQNs ``layers.10...``; they run in insertion
    order.

    :meth:`forward` runs one ``Maglev1F1B`` microbatch through every layer, or
    one MaglevRail phase, so a wrapper observes both: given ``sparse_outputs``,
    one dense microbatch; given ``batch_size`` and ``num_microbatches``, every
    sparse half on the pass's whole batch.

    Args:
        layers: the stage's layers, in execution order.
        postproc: the model's output seam on the stage that owns the final
            layer, called as ``postproc(activations, layer_inputs[-1])``;
            ``None`` elsewhere.
        first_layer_index: global index of this stage's first layer.
        loss_only_output: whether the last stage returns only ``postproc``'s
            loss, keeping auxiliary predictions out of a wrapper's output
            traversal.

    Raises:
        ValueError: if a layer owns a trainable parameter outside its halves.
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
        self._sparse_profile_names: List[str] = [
            f"## torchrec_maglev:layer[{first_layer_index + index}] "
            f"{type(layer).__name__.lstrip('_')} sparse ##"
            for index, layer in enumerate(layers)
        ]
        halved = [layer for layer in layers if _runs_halves(layer)]
        reject_unhalved_parameters(halved)
        self._runs_halves: bool = bool(layers) and len(halved) == len(layers)
        # Plain attribute, not a submodule: postproc is a bound method of the
        # authored model, whose parameters are already owned by ``layers``.
        self._postproc = postproc
        self._loss_only_output = loss_only_output

    def forward(
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations = (),
        *,
        sparse_outputs: Optional[Sequence[Optional[SparseSeam]]] = None,
        batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
    ) -> Any:
        r"""forward(layer_inputs, in_activations=(), *, sparse_outputs=None, batch_size=None, num_microbatches=None) -> Any

        Run one microbatch, the sparse phase, or one dense microbatch.

        Args:
            layer_inputs (Sequence[Any]): One input per local layer.
            in_activations (Activations): Activation entering the stage; dense
                phase only. Default: ``()``
            sparse_outputs (Sequence[Optional[SparseSeam]], optional): One
                sparse seam per layer, which selects the dense phase. Default:
                ``None``
            batch_size (int, optional): Whole-pass batch size; sparse phase
                only. Default: ``None``
            num_microbatches (int, optional): Dense microbatches; sparse phase
                only. Default: ``None``

        Returns:
            Any: The sparse phase's ``(sparse output, dense inputs)`` per layer,
            or the dense phase's activation (``postproc``'s result on the last
            stage).

        Raises:
            ValueError: If the input counts or phase arguments are invalid.
        """
        if len(layer_inputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} layer inputs, got {len(layer_inputs)}"
            )
        if sparse_outputs is not None:
            if batch_size is not None or num_microbatches is not None:
                raise ValueError(
                    "dense forward does not accept batch_size or num_microbatches"
                )
            return self._dense_phase(layer_inputs, in_activations, sparse_outputs)
        if batch_size is None and num_microbatches is None:
            return self._whole_microbatch(layer_inputs, in_activations)
        if in_activations:
            raise ValueError("sparse forward does not accept incoming activations")
        if batch_size is None or num_microbatches is None:
            raise ValueError("sparse forward requires batch_size and num_microbatches")
        return self._sparse_phase(layer_inputs, batch_size, num_microbatches)

    def _whole_microbatch(
        self, layer_inputs: Sequence[Any], in_activations: Activations
    ) -> Any:
        if not self._runs_halves:
            activations = in_activations
            for layer, layer_input, profile_name in zip(
                self.layers.values(), layer_inputs, self._profile_names
            ):
                with record_function(profile_name):
                    activations = cast(MaglevLayer, layer)(layer_input, activations)
            return self._finish(activations, layer_inputs)
        sparse_outputs: List[Any] = []
        for layer, layer_input, profile_name in zip(
            self.layers.values(), layer_inputs, self._sparse_profile_names
        ):
            sparse = cast(MaglevLayer, layer).sparse
            with record_function(profile_name):
                sparse_outputs.append(None if sparse is None else sparse(layer_input))
        return self._dense_phase(layer_inputs, in_activations, sparse_outputs)

    def _sparse_phase(
        self, layer_inputs: Sequence[Any], batch_size: int, num_microbatches: int
    ) -> List[_SparseLayerOutput]:
        outputs: List[_SparseLayerOutput] = []
        for layer, layer_input, profile_name in zip(
            self.layers.values(), layer_inputs, self._profile_names
        ):
            maglev_layer = cast(MaglevLayer, layer)
            with record_function(profile_name):
                sparse = maglev_layer.sparse
                pooled = None if sparse is None else sparse(layer_input)
                if sparse is not None:
                    check_sparse_output(
                        maglev_layer,
                        pooled,
                        batch_size,
                        shared=maglev_layer.shared_sparse_outputs(),
                    )
                dense_inputs = maglev_layer.split_dense_input(
                    layer_input, batch_size, num_microbatches
                )
                if len(dense_inputs) != num_microbatches:
                    raise ValueError(
                        f"{type(maglev_layer).__name__}.split_dense_input() returned "
                        f"{len(dense_inputs)} inputs, expected {num_microbatches}"
                    )
            outputs.append((pooled, dense_inputs))
        return outputs

    def _dense_phase(
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations,
        sparse_outputs: Sequence[Optional[SparseSeam]],
    ) -> Any:
        if len(sparse_outputs) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} sparse outputs, got "
                f"{len(sparse_outputs)}"
            )
        activations = in_activations
        for layer, layer_input, sparse_output, profile_name in zip(
            self.layers.values(), layer_inputs, sparse_outputs, self._profile_names
        ):
            with record_function(profile_name):
                activations = cast(MaglevLayer, layer).require_dense()(
                    sparse_output, layer_input, activations
                )
        return self._finish(activations, layer_inputs)

    def _finish(self, activations: Activations, layer_inputs: Sequence[Any]) -> Any:
        if self._postproc is None:
            return activations
        with record_function("## torchrec_maglev:postproc ##"):
            output = self._postproc(activations, layer_inputs[-1])
        if self._loss_only_output:
            # DDP treats every returned tensor as a backward root. Auxiliary
            # predictions must stay out when only the loss is backpropagated.
            losses, _model_output = output
            return (losses,)
        return output


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


@dataclass
class _Microbatch:
    """A dense microbatch forwarded but not yet backwarded."""

    in_activations: Activations
    outputs: Any
    microbatch_id: int
    seams: Sequence[Optional[SparseSeam]]


class StageWrapper(nn.Module):
    """Takes a whole Maglev model and keeps the one stage this rank owns.

    Every rank passes the same ``model`` and ``layers_per_stage``; the wrapper
    keeps this rank's contiguous run of the model's own layers, not copies.

    It runs the stage's share of each phase of a pass, in whatever order a
    schedule (:mod:`~torchrec.distributed.maglev.pipeline`) calls them:

    * :meth:`sparse_forward_global` runs every sparse half once on the pass's
      whole batch and cuts each output into detached per-microbatch seams;
    * :meth:`dense_forward_micro` runs one dense microbatch, between the
      activation handoffs;
    * :meth:`dense_backward_micro` backwards one microbatch whole, or
      :meth:`dense_backward_act_micro` and :meth:`dense_backward_weight_micro`
      split it into its input and weight halves;
    * :meth:`sparse_backward_global` resumes every sparse graph from its
      seams' summed gradients.

    **Parallelism is the caller's job.** The wrapper cuts the model, builds the
    process-group bundle, and reads the boundary contract; it does not shard.
    Wrap :attr:`module`, assign it back, then :meth:`to` to materialize whatever
    is still on ``meta``::

        stage = StageWrapper(model, layers_per_stage, stage_size)
        stage.module = DistributedModelParallel(
            stage.module, env=ShardingEnv.from_process_group(stage.stage_pg), ...
        )
        stage.to(device)

    The activation specs are captured in the constructor, before any wrapping,
    since a wrapped module may no longer expose the
    :class:`~torchrec.distributed.maglev.module.MaglevLayer` API.

    Args:
        model: the authored model, in full.
        layers_per_stage: how many layers each pipeline stage owns; must sum to
            ``len(model)``, with one entry per stage.
        stage_size: ranks per stage -- the size of one hardware scale-up domain
            (HSD). Used to build the default contiguous topology and validated
            against ``process_groups`` when one is provided.
        loss_only_output: whether the final stage returns only the loss.
        process_groups: optional externally constructed Maglev process groups.
        enable_rail: recorded as :attr:`rail_enabled`. Default is ``False``.

    Raises:
        ValueError: if ``layers_per_stage`` does not describe ``model``, the
            process-group layout does not match ``stage_size`` or
            ``layers_per_stage``, the kept layers disagree on the activation
            they exchange, or a layer owns a trainable parameter outside its
            halves.
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
        whole role is one ``divmod``: the stage it owns, and its *position*
        within that stage, which pairs it with the corresponding rank of the
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
        # On every rank, before any collective: a stage that refused alone would
        # leave its peers blocked in group creation.
        reject_unhalved_parameters(
            [cast(MaglevLayer, layer) for layer in model if _runs_halves(layer)]
        )
        self._outside_layers_error: Optional[str] = None
        try:
            reject_parameters_outside_layers(model)
        except ValueError as error:
            self._outside_layers_error = str(error)
        self._rail_enabled: bool = enable_rail
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
        # The wrapper shards collectively, so it is only ever built for the
        # calling rank.
        stage_index, position = MaglevProcessGroups.locate_rank(
            process_groups.stage_ranks, dist.get_rank()
        )
        self.stage_index: int = stage_index
        self.position: int = position
        self.stage_size: int = stage_size
        self.num_stages: int = num_stages
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
        # The model's own input seam, kept as bound methods: holding the model
        # would register every other stage's layers as submodules.
        self._preproc: Callable[[Any], List[Any]] = model.preproc
        self._get_batch_size: Callable[[Sequence[Any]], int] = model.get_batch_size
        self._input_driver: InputDistDriver[List[Any]] = InputDistDriver(
            pg_gloo=process_groups.cascade_gloo_pg,
            pg_nccl=process_groups.cascade_pg,
            self_index=stage_index,
        )
        self._pending: Deque[_Microbatch] = deque()
        # The weight work each input half deferred, oldest first.
        self._wdense: Deque[Tuple[List[Dict[str, Any]], int]] = deque()
        self._in_specs: Tuple[ActivationSpec, ...] = layers[0].in_activation_specs()
        self._out_specs: Tuple[ActivationSpec, ...] = layers[-1].out_activation_specs()
        self.is_last_stage: bool = stage_index == num_stages - 1
        chain = _LayerChain(
            layers,
            model.postproc if self.is_last_stage else None,
            first_layer_index=self.layer_indices.start,
            loss_only_output=loss_only_output,
        )
        # Public and reassignable: wrap it in DMP/DDP/FSDP2 and assign it back.
        self.module: nn.Module = chain
        # The authored chain, reachable after :attr:`module` is wrapped. Not a
        # submodule: registering it twice would duplicate every state_dict key.
        object.__setattr__(self, "_chain", chain)
        self._layers: List[MaglevLayer] = list(layers)
        self._pass_active = False
        self._pass_failed = False
        self._checks = DenseWeightChecks(
            stage_index,
            stage_size,
            process_groups.stage_pg,
            lambda: self._placed_device,
        )
        self._loss_scale: float = 1.0
        # The schedule that configured the last pass. Weak, so a collected
        # schedule's successor still counts as a change.
        self._schedule: Optional[weakref.ReferenceType[object]] = None
        self._pass_losses: List[torch.Tensor] = []
        self._loss_only_output = loss_only_output
        # Set by to(): a meta-authored model has no device to infer.
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

        Example::

            stage_input = [layer_inputs[i] for i in stage.layer_indices]
        """
        start = sum(self.layers_per_stage[: self.stage_index])
        return range(start, start + self.num_layers)

    @property
    def handoff_pgs(self) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
        """The activation and gradient handoff process groups."""
        return self._process_groups.handoff_pgs

    @property
    def handoff_pg_mode(self) -> HandoffPGMode:
        """Whether activation and gradient handoffs share one process group."""
        return self._process_groups.handoff_pg_mode

    @property
    def rail_enabled(self) -> bool:
        """Whether the caller asked for a Rail stage (``enable_rail``)."""
        return self._rail_enabled

    def require_pass_model(self) -> None:
        """Refuse a model MaglevRail cannot train.

        Raises:
            ValueError: if a trainable weight sits outside the layers' halves,
                which are all the pass runs.
        """
        if self._outside_layers_error is not None:
            raise ValueError(self._outside_layers_error)
        reject_unhalved_parameters(self._layers)

    @property
    def cascade_pg(self) -> dist.ProcessGroup:
        """This rank's input-distribution group (one rank per stage)."""
        return self._process_groups.cascade_pg

    @property
    def stage_pg(self) -> dist.ProcessGroup:
        return self._process_groups.stage_pg

    @property
    def is_first(self) -> bool:
        """Whether this stage starts the pipeline (nothing to receive)."""
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        """Whether this stage ends the pipeline (nothing to send)."""
        return self.stage_index == self.num_stages - 1

    def neighbor_rank(self, offset: int) -> int:
        """The global rank at this position in the HSD ``offset`` stages away.

        Args:
            offset: stages to move, e.g. ``-1`` for the previous HSD.

        Returns:
            int: the neighbouring rank.

        Raises:
            ValueError: if the offset lands outside the pipeline.
        """
        stage_index = self.stage_index + offset
        if not 0 <= stage_index < self.num_stages:
            raise ValueError(
                f"stage {self.stage_index} has no neighbor at offset {offset}: "
                f"the pipeline has {self.num_stages} stages"
            )
        return self._process_groups.stage_ranks[stage_index][self.position]

    # pyre-ignore[14]: narrower than nn.Module.to by design -- this one
    # materializes meta parameters, which nn.Module.to cannot.
    def to(self, device: torch.device) -> "StageWrapper":
        """Place this stage on ``device``, materializing anything still on ``meta``.

        Call after wrapping :attr:`module`, so a sharder places the tables rather
        than resharding ones already allocated at full size. Idempotent, and a
        no-op for parameters a wrapper already placed.

        Args:
            device: where this stage runs, and where the handoffs allocate their
                receive buffers.
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

    # ---- one pass ----

    @contextmanager
    def pass_scope(self) -> Iterator[None]:
        """Run one pass, from ``zero_grad`` to the optimizer step.

        A pass that raises poisons the stage (:meth:`abandon_pass`), so a
        later pass is refused rather than run against wreckage.

        Raises:
            RuntimeError: If a previous pass failed or a pass is already running.
        """
        self._refuse_out_of_step()
        self._pass_active = True
        try:
            yield
        except BaseException:
            self.abandon_pass()
            raise
        finally:
            self._pass_active = False
        self._checks.end_first_pass()

    def _refuse_out_of_step(self) -> None:
        """Refuse a new pass, or its inputs, after a failed pass or during one."""
        if self._pass_failed or self._pass_active:
            raise RuntimeError(
                f"stage {self.stage_index}: a previous pass failed or is still "
                "running, so the handoffs are out of step; restart the job"
            )

    def abandon_pass(self) -> None:
        r"""abandon_pass() -> None

        Drop the state a pass accumulates and mark the stage failed.

        The queues are FIFOs, appended in forward and popped in backward, so a
        retried pass would pair one microbatch's saved activations with
        another's gradients. Outstanding point-to-point work is abandoned rather
        than waited on: its peer is typically the rank that already failed.
        The handoffs are then out of step across ranks, so the job must restart;
        :meth:`pass_scope` refuses every later pass.
        """
        self._pass_failed = True
        self._send_act = []
        self._send_grad = []
        self._recv_act.clear()
        self._recv_grad.clear()
        self._pending.clear()
        self._wdense.clear()
        self._pass_losses = []

    def configure_pass(
        self, schedule: object, *, loss_scale: float, allow_unreached: bool
    ) -> None:
        r"""configure_pass(schedule, *, loss_scale, allow_unreached) -> None

        Set what the next pass's backwards need, for the schedule running it.
        The first pass after the schedule changes repeats the first-pass checks
        on every replica: another order builds another graph.

        Args:
            schedule (object): The schedule about to run a pass.
            loss_scale (float): The gradient seeded at each microbatch loss;
                ``1 / m`` over ``m`` mean-loss microbatches trains on the mean
                over the whole batch.
            allow_unreached (bool): Whether a trainable dense weight that no
                output of a microbatch reaches may keep ``grad=None``.
        """
        previous = None if self._schedule is None else self._schedule()
        if previous is not schedule:
            self._schedule = weakref.ref(schedule)
            self._checks.restart()
        self._loss_scale = loss_scale
        self._checks.set_unreached_allowed(allow_unreached)

    def _loss_grads(self, loss: torch.Tensor) -> Optional[List[torch.Tensor]]:
        if self._loss_scale == 1.0:
            return None
        return [torch.full_like(loss, self._loss_scale)]

    def take_pass_losses(self) -> List[torch.Tensor]:
        r"""take_pass_losses() -> List[torch.Tensor]

        The microbatch losses the last stage's dense backwards ran since the last
        call, in order, detached; empty on other stages.
        """
        losses, self._pass_losses = self._pass_losses, []
        return losses

    def only_data_parallel_wraps_layers(self) -> bool:
        r"""only_data_parallel_wraps_layers() -> bool

        Return whether :attr:`module` reaches the authored layer chain through
        ``DistributedModelParallel`` and ``DistributedDataParallel`` wrappers
        only, so calling the chain directly skips nothing but their ``forward``.
        """
        module: object = self.module
        while module is not self._chain:
            wrapped = getattr(module, "_dmp_wrapped_module", None)
            if isinstance(wrapped, nn.Module):
                module = wrapped
            elif isinstance(module, DistributedDataParallel):
                module = module.module
            else:
                return False
        return True

    # ---- cross-HSD hand-off ----
    #
    # The stage owns the wire: its neighbours (neighbor_rank), the handoff
    # communicators, and the specs that fix the wire layout. A schedule decides
    # when to call these, as with torch.distributed.pipelining's PipelineStage.

    def start_recv_act(self, stage_input: Optional[Sequence[Any]] = None) -> None:
        """Post a receive for the previous HSD's activation.

        Allocates one buffer per incoming spec and issues the receives in spec
        order, so both sides of the boundary agree without exchanging metadata.
        No-op on the first stage. Each call queues another receive;
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
                # Only once landed: before that, the incoming write would be an
                # in-place op on a leaf that requires grad.
                tensor.requires_grad_(True)
        return tuple(tensors)

    def start_send_act(self, outputs: Activations) -> None:
        """Send this stage's activation to the next HSD; no-op if last.

        The caller must :meth:`finish_send_act` the previous send first. The send
        is checked against the output specs the next stage receives with: a
        mismatch (a mixed-precision cast, say) would hang or read garbage.

        Args:
            outputs: this stage's output activation.

        Raises:
            ValueError: if an activation send is still in flight, or ``outputs``
                does not match the stage's output specs.
        """
        if self.is_last:
            return
        if self._send_act:
            raise ValueError(
                f"stage {self.stage_index}: an activation send is still in "
                "flight; call finish_send_act() before starting the next"
            )
        check_activations(
            self._out_specs, outputs, f"stage {self.stage_index} output activation"
        )
        act_pg, _ = self.handoff_pgs
        self._send_act = self._isend(outputs, self.neighbor_rank(1), act_pg)

    def finish_send_act(self) -> None:
        """Complete the activation send in flight, if any."""
        self._drain(self._send_act)

    def start_recv_grad(self) -> None:
        """Post a receive for the next HSD's gradient of the oldest awaiting output.

        One receive per grad-carrying output slot. No-op on the last stage, whose
        gradient starts at its own loss. Each call queues another receive.
        """
        if self.is_last:
            return
        _, grad_pg = self.handoff_pgs
        src = self.neighbor_rank(1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        outputs = cast(Activations, self._pending[len(self._recv_grad)].outputs)
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

    def start_send_grad(self, in_activations: Activations) -> None:
        """Send this stage's input gradients upstream; no-op if first.

        A slot unused by the stage's graph has no ``.grad``; zeros are sent so
        the previous stage's receive still matches the specs.

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

    def drain_sends(self) -> None:
        """Complete the send in flight in each direction; a pass ends here."""
        self.finish_send_act()
        self.finish_send_grad()

    # pyre-ignore[3]: dist work handle has no public type
    def _isend(
        self, tensors: Sequence[torch.Tensor], dst: int, pg: dist.ProcessGroup
    ) -> List[Tuple[Any, torch.Tensor]]:
        """Issue non-blocking sends, in order; each buffer stays alive with its work."""
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

    # ---- input distribution ----

    def group_by_stage(self, layer_inputs: Sequence[Any]) -> List[List[Any]]:
        """Regroup a full per-layer input list into one carrier per stage.

        Args:
            layer_inputs: one input per layer of the whole model, in model order.

        Returns:
            List[List[Any]]: ``result[s]`` holds stage ``s``'s inputs.
        """
        grouped: List[List[Any]] = []
        offset = 0
        for count in self.layers_per_stage:
            grouped.append(list(layer_inputs[offset : offset + count]))
            offset += count
        return grouped

    def send_set(self, model_input: Any) -> List[List[Any]]:
        """Turn one raw batch into the cascade send set: one carrier per stage.

        The model's :meth:`preproc` splits the batch, under ``no_grad``.

        Args:
            model_input: the raw batch, as the dataloader yields it.

        Returns:
            List[List[Any]]: ``result[s]`` is the input list for stage ``s``.
        """
        with record_function("## torchrec_maglev:preproc ##"), torch.no_grad():
            layer_inputs = self._preproc(model_input)
        return self.group_by_stage(layer_inputs)

    def take_global_inputs(self, dataloader_iter: Iterator[Any]) -> List[Any]:
        r"""take_global_inputs(dataloader_iter) -> List[Any]

        Return this stage's inputs for one pass: one whole-pass input per layer.

        One input-distribution round yields ``num_stages`` passes' inputs; the
        rest stay queued (:class:`~torchrec.distributed.maglev.input_dist.InputDistDriver`).

        Args:
            dataloader_iter (Iterator[Any]): Iterator over whole-pass batches.

        Returns:
            List[Any]: One whole-pass input per local layer.

        Raises:
            RuntimeError: If a previous pass failed or a pass is running: the
                round is a collective, and the peers it would wait on are out of
                step.
        """
        self._refuse_out_of_step()
        with record_function("## torchrec_maglev:global_inputs ##"):
            (global_inputs,) = self._input_driver.take(
                lambda: self.send_set(next(dataloader_iter)), 1
            )
        return global_inputs

    # ---- Maglev1F1B: one microbatch through the stage ----

    def forward(self, model_input: Any) -> torch.Tensor:
        """Distribute and forward every microbatch produced by one raw batch.

        Args:
            model_input: one raw dataloader batch, already a microbatch. The
                model's ``preproc`` creates per-layer inputs, and one
                input-distribution round exchanges them across the cascade.

        Returns:
            torch.Tensor: the summed microbatch loss on the final stage, and a
            zero-valued backward token on earlier stages. Every rank must call
            ``backward`` on it.
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

    # ---- paired hand-off ----
    #
    # SPLIT mode uses independent activation and gradient PGs with the start/wait
    # methods. SHARED mode additionally permits the paired exchange methods:
    # batch_isend_irecv requires every operation in a batch to use one PG.

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
        backward_outputs = cast(Activations, self._pending[0].outputs)
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

    def compute_forward_micro(
        self,
        stage_input: Sequence[Any],
        in_activations: Activations,
        microbatch_id: int,
    ) -> Any:
        """Compute one forward and retain its graph for the matching backward."""
        with record_function(f"## forward mb{microbatch_id} ##"):
            outputs = self.module(stage_input, in_activations)
        self._pending.append(_Microbatch(in_activations, outputs, microbatch_id, ()))
        return outputs

    def compute_backward_micro(
        self, grads: Sequence[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Activations]:
        """Compute the oldest pending backward and return its input activation."""
        microbatch = self._pending.popleft()
        in_activations = microbatch.in_activations
        outputs = microbatch.outputs
        microbatch_id = microbatch.microbatch_id
        loss: Optional[torch.Tensor] = None
        if self.is_last:
            with record_function(f"## backward mb{microbatch_id} ##"):
                loss = outputs[0]
                # pyrefly: ignore [missing-attribute]
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
        microbatch_id = self._pending[0].microbatch_id
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
        pending = list(self._pending)
        if self.is_last:
            losses = [microbatch.outputs[0] for microbatch in pending]
            with record_function("## backward round ##"):
                torch.autograd.backward(losses, [loss_grad] * len(losses))
        else:
            for _ in pending:
                self.start_recv_grad()

            backward_outputs: List[torch.Tensor] = []
            backward_grads: List[torch.Tensor] = []
            for microbatch in pending:
                outputs, microbatch_id = microbatch.outputs, microbatch.microbatch_id
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

        self._pending.clear()
        for microbatch in pending:
            in_activations, microbatch_id = (
                microbatch.in_activations,
                microbatch.microbatch_id,
            )
            with record_function(f"## finish_send_grad mb{microbatch_id} ##"):
                self.finish_send_grad()
            with record_function(f"## send_grad mb{microbatch_id} ##"):
                self.start_send_grad(in_activations)
        self.finish_send_grad()

    # ---- the pass: the sparse phases and the dense microbatch work ----

    def sparse_forward_global(
        self,
        global_inputs: Sequence[Any],
        num_microbatches: int,
        *,
        unwrapped: bool = False,
    ) -> MaglevPassState:
        r"""sparse_forward_global(global_inputs, num_microbatches, *, unwrapped=False) -> MaglevPassState

        Run each sparse half once and cut its output into detached dense seams.

        Args:
            global_inputs (Sequence[Any]): One whole-pass input per local layer.
            num_microbatches (int): Number of dense microbatches.
            unwrapped (bool): Call the authored layer chain directly, past
                wrappers :meth:`only_data_parallel_wraps_layers` allows. Default:
                ``False``

        Returns:
            MaglevPassState: Sparse graphs, seams, and dense inputs.

        Raises:
            TypeError: If a sparse half returns neither a tensor nor a
                non-empty tuple of tensors.
            ValueError: If input counts, batch dimensions or shared positions
                are invalid.
        """
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

        module = cast(nn.Module, self._chain) if unwrapped else self.module
        with record_function("## torchrec_maglev:sparse_forward_global ##"):
            layer_outputs = cast(
                List[_SparseLayerOutput],
                module(
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
            pooled = [output for output, _dense_inputs in layer_outputs]
            cuts = [
                (
                    None
                    if output is None
                    else _cut_seams(
                        output, num_microbatches, layer.shared_sparse_outputs()
                    )
                )
                for layer, output in zip(self._layers, pooled)
            ]

        # Microbatch-major, as the dense phase consumes them.
        seams = [
            [None if cut is None else cut[microbatch] for cut in cuts]
            for microbatch in range(num_microbatches)
        ]
        dense_inputs = [
            [layer_inputs[microbatch] for _output, layer_inputs in layer_outputs]
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
        return MaglevPassState(pooled=pooled, seams=seams, dense_inputs=dense_inputs)

    def sparse_backward_global(self, state: MaglevPassState) -> None:
        r"""sparse_backward_global(state) -> None

        Resume every whole-pass sparse graph from its seams' gradients, in one
        backward; afterwards every seam leaf's ``.grad`` is ``None``.

        One backward for every layer, not one per layer: a wrapper may join the
        layers' outputs in one autograd node (DDP's output sink does), and layers
        may share a sparse frontend, whose graph a per-layer backward would free.

        Args:
            state (MaglevPassState): State produced by
                :meth:`sparse_forward_global`.
        """
        roots: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []
        for index, (layer, pooled) in enumerate(zip(self._layers, state.pooled)):
            if pooled is None:
                continue
            shared = layer.shared_sparse_outputs()
            layer_seams = [cast(SparseSeam, seams[index]) for seams in state.seams]
            for position, whole in enumerate(_seam_tensors(pooled)):
                leaves = [_seam_tensors(seam)[position] for seam in layer_seams]
                if position in shared:
                    leaves = leaves[:1]
                if not whole.requires_grad:
                    for leaf in leaves:
                        leaf.grad = None
                    continue
                parts: List[torch.Tensor] = []
                for leaf in leaves:
                    parts.append(
                        leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
                    )
                    leaf.grad = None
                roots.append(whole)
                gradients.append(parts[0] if len(parts) == 1 else torch.cat(parts))
                del parts
        if not roots:
            return
        with record_function("## torchrec_maglev:sparse_backward_global ##"):
            torch.autograd.backward(roots, gradients)

    def compute_dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        in_activations: Activations,
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[SparseSeam]],
    ) -> Any:
        r"""compute_dense_forward_micro(stage_input, in_activations, microbatch_id, sparse_outputs) -> Any

        Compute one dense forward, without the handoffs, and keep its graph for
        the backward.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            in_activations (Activations): Activation entering this stage.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[SparseSeam]]): One sparse seam per
                local layer.

        Returns:
            Any: The stage's output, kept for the backward.
        """
        with record_function(f"## dense_forward mb{microbatch_id} ##"):
            outputs = self.module(
                stage_input, in_activations, sparse_outputs=sparse_outputs
            )
        # The seams survive with the microbatch: a split backward fills only the
        # leaves it is handed.
        self._pending.append(
            _Microbatch(in_activations, outputs, microbatch_id, sparse_outputs)
        )
        return outputs

    def dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[SparseSeam]],
    ) -> Any:
        r"""dense_forward_micro(stage_input, microbatch_id, sparse_outputs) -> Any

        Receive, compute and send one dense microbatch. The schedule posts the
        activation receive (:meth:`start_recv_act`) before calling this.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[SparseSeam]]): One sparse seam per
                local layer.

        Returns:
            Any: The stage's output, kept for the backward.
        """
        with record_function(f"## dense_recv_act mb{microbatch_id} ##"):
            in_activations = self.wait_for_act()
        outputs = self.compute_dense_forward_micro(
            stage_input, in_activations, microbatch_id, sparse_outputs
        )
        with record_function(f"## dense_finish_send_act mb{microbatch_id} ##"):
            self.finish_send_act()
        with record_function(f"## dense_send_act mb{microbatch_id} ##"):
            self.start_send_act(outputs)
        return outputs

    def dense_parameters(self) -> List[nn.Parameter]:
        r"""dense_parameters() -> List[nn.Parameter]

        The trainable parameters of the layers' dense halves, in layer order,
        as the modules register them.

        Returns:
            List[nn.Parameter]: Trainable dense parameters.
        """
        return [
            parameter
            for layer in self._layers
            for parameter in layer.dense_parameters()
            if parameter.requires_grad
        ]

    def _graph_dense_parameters(self) -> List[nn.Parameter]:
        """:meth:`dense_parameters` as a backward's graph holds them.

        A ZeRO-3 module re-registers its sharded parameters after its forward,
        while the graph keeps the unsharded ones. Valid only around a backward:
        once FSDP2 reshards, an unsharded parameter's storage is freed.
        """
        graph: Dict[int, nn.Parameter] = {}
        for layer in self._layers:
            graph.update(fsdp_graph_parameters(layer.require_dense()))
        return [
            graph.get(id(parameter), parameter)
            for layer in self._layers
            for parameter in layer.dense_parameters()
            if parameter.requires_grad
        ]

    @property
    def pending_weight_work(self) -> int:
        """Weight halves :meth:`dense_backward_act_micro` queued and none ran yet."""
        return len(self._wdense)

    def dense_backward_micro(self) -> Optional[torch.Tensor]:
        r"""dense_backward_micro() -> Optional[torch.Tensor]

        The oldest pending dense microbatch's whole backward, through the dense
        weights' ``AccumulateGrad`` nodes so a data-parallel wrapper's hooks
        observe them. The schedule posts :meth:`start_recv_grad` first.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage, ``None``
            elsewhere.

        Raises:
            ValueError: if the received gradients do not match this stage's
                grad-carrying outputs.
            RuntimeError: on the first pass, if a trainable dense weight is
                unreached while unreached weights are not allowed.
        """
        microbatch, roots, root_grads, loss = self._pop_roots()
        self._checks.before_whole_backward(roots, self._graph_dense_parameters)
        if roots:
            with record_function(f"## dense_backward mb{microbatch.microbatch_id} ##"):
                torch.autograd.backward(roots, root_grads)
        self._send_input_grads(microbatch)
        return loss

    def dense_backward_act_micro(self) -> Optional[torch.Tensor]:
        r"""dense_backward_act_micro() -> Optional[torch.Tensor]

        The ``I`` half: gradients w.r.t. this stage's inputs (the received
        activations and the sparse seams), sent upstream at once; the weight half
        waits for :meth:`dense_backward_weight_micro`. With no differentiable
        input the backward runs whole and leaves an empty weight half.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage,
            ``None`` elsewhere; detached unless it is a view.

        Raises:
            RuntimeError: if a trainable dense weight would not be trained.
        """
        microbatch, roots, root_grads, loss = self._pop_roots()
        microbatch_id = microbatch.microbatch_id
        input_values: List[torch.Tensor] = [
            tensor
            for tensor, spec in zip(microbatch.in_activations, self._in_specs)
            if spec.requires_grad
        ]
        for seam in microbatch.seams:
            if seam is not None:
                input_values.extend(_seam_tensors(seam))

        if roots and input_values:
            weights = self._graph_dense_parameters()
            # stage_backward_input detaches the roots that are not views in
            # place; the checks walk the graph from their nodes afterwards.
            root_nodes = [_get_grad_fn_or_grad_acc(root) for root in roots]
            with record_function(f"## dense_backward_act mb{microbatch_id} ##"):
                _input_grads, param_groups = stage_backward_input(
                    stage_outputs_or_loss=roots,
                    output_grads=root_grads,
                    input_values=input_values,
                    # One list for both halves, so the groups the I half records
                    # line up with what W looks up.
                    weights=iter(weights),
                )
            self._checks.after_input_half(
                param_groups, weights, root_nodes, input_values
            )
            del root_nodes
            self._wdense.append((param_groups, microbatch_id))
        else:
            self._checks.before_whole_backward(roots, self._graph_dense_parameters)
            if roots:
                with record_function(
                    f"## dense_backward_act mb{microbatch_id} (whole) ##"
                ):
                    torch.autograd.backward(roots, root_grads)
            # An empty weight half, so every input half still pairs with one.
            self._wdense.append(([], microbatch_id))
        self._send_input_grads(microbatch)
        return loss

    def dense_backward_weight_micro(self, *, run_accumulate_grad: bool = False) -> None:
        r"""dense_backward_weight_micro(*, run_accumulate_grad=False) -> None

        The ``W`` half :meth:`dense_backward_act_micro` deferred, for its oldest
        microbatch; the queue must be empty before the optimizer step. It agrees
        with a whole backward to floating-point tolerance, not bit-exactly.

        Args:
            run_accumulate_grad (bool): Run each weight's ``AccumulateGrad`` node,
                so post-accumulate hooks (DDP's used-parameter map) observe the
                gradient. Default: ``False``, which writes ``.grad`` directly.

        Raises:
            ValueError: if no deferred weight work is queued.
        """
        if not self._wdense:
            raise ValueError(
                f"stage {self.stage_index}: dense_backward_weight_micro() with no "
                "deferred weight work; call dense_backward_act_micro() first"
            )
        param_groups, microbatch_id = self._wdense.popleft()
        if not param_groups:
            return
        with record_function(f"## dense_backward_weight mb{microbatch_id} ##"):
            if run_accumulate_grad:
                _accumulate_weight_grads(param_groups)
            else:
                stage_backward_weight(
                    iter(self._graph_dense_parameters()), param_groups
                )

    def _pop_roots(
        self,
    ) -> Tuple[
        _Microbatch,
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Pop the oldest pending microbatch with its backward's roots: the
        loss-scaled ``postproc`` loss on the last stage, else the grad-carrying
        outputs paired with the next stage's gradients.

        Returns:
            The microbatch, the roots, their gradients, and the loss (last stage
            only).

        Raises:
            ValueError: if the received gradients do not match the grad-carrying
                outputs.
        """
        microbatch = self._pending.popleft()
        if self.is_last:
            loss = cast(Activations, microbatch.outputs)[0]
            self._pass_losses.append(loss.detach())
            return microbatch, [loss], self._loss_grads(loss), loss
        with record_function(f"## dense_recv_grad mb{microbatch.microbatch_id} ##"):
            grads = self.wait_for_grad()
        grad_carrying = [
            tensor
            for tensor, spec in zip(
                cast(Activations, microbatch.outputs), self._out_specs
            )
            if spec.requires_grad
        ]
        if len(grads) != len(grad_carrying):
            raise ValueError(
                f"stage {self.stage_index} received {len(grads)} gradients "
                f"for {len(grad_carrying)} grad-carrying outputs"
            )
        pairs = [
            (output, grad)
            for output, grad in zip(grad_carrying, grads)
            if output.requires_grad
        ]
        return microbatch, [o for o, _ in pairs], [g for _, g in pairs], None

    def _send_input_grads(self, microbatch: _Microbatch) -> None:
        with record_function(
            f"## dense_finish_send_grad mb{microbatch.microbatch_id} ##"
        ):
            self.finish_send_grad()
        with record_function(f"## dense_send_grad mb{microbatch.microbatch_id} ##"):
            self.start_send_grad(microbatch.in_activations)
