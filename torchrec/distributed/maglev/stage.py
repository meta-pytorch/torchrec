#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Cutting a Maglev model into stages and running one stage's share of a pass.

Three concepts, one section each, in this order:

* Layer: one ``MaglevLayer``, an optional sparse net and a required dense net
  joined at a seam, which keeps ``MaglevLayer``'s own ``forward``;
  :func:`~torchrec.distributed.maglev.checks.reject_unsplit_layers` refuses any
  other layer at build. :data:`SparseSeam`, the seam helpers, and
  :class:`_LayerChain`, the stage's layers as one module, whose
  ``_run_<net>_<global|micro>`` modes each phase calls.
* Global batch and microbatch: what one pass carries. :class:`MaglevPassState`
  and the per-microbatch split of the sparse nets' global-batch outputs.
* Stage: MaglevPP-infra does two jobs. Allocation, here: :class:`StageWrapper`
  places the model's layers on stages under ``layers_per_stage`` and binds this
  rank's layers to its HSD (cut and bind). The global-/micro-batch schedule, in
  :mod:`torchrec.distributed.maglev.pipeline`, calls the phase methods below.
  Splitting the input batch (feed), owning the wire between stages (connect)
  and running one pass at a time (pass discipline) are the stage's means to
  both. Its phase methods, named ``<net>_<direction>[_<part>]_<global|micro>``,
  follow in pass order: ``StageWrapper.<net>_forward_<granularity>`` calls
  ``_LayerChain._run_<net>_<granularity>``.

The process groups are in :mod:`torchrec.distributed.maglev.process_groups`; the
authoring side in :mod:`torchrec.distributed.maglev.module`; the data-parallel
drivers in :mod:`torchrec.distributed.maglev.data_parallel`; the order of the
phases is the schedule's, in :mod:`torchrec.distributed.maglev.pipeline`.
"""


from __future__ import annotations

import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
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
    TYPE_CHECKING,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.distributed.pipelining._backward import (
    _get_grad_fn_or_grad_acc,
    stage_backward_input,
    stage_backward_weight,
)
from torchrec.distributed.maglev.checks import (
    check_sparse_output,
    DenseWeightChecks,
    reject_parameters_outside_layers,
    reject_parameters_outside_nets,
    reject_unsplit_layers,
)
from torchrec.distributed.maglev.input_dist import InputDistDriver
from torchrec.distributed.maglev.module import (
    Activations,
    ActivationSpec,
    check_activations,
    check_layers_chain,
    MaglevLayer,
    MaglevModuleList,
)
from torchrec.distributed.maglev.process_groups import MaglevProcessGroups
from torchrec.distributed.utils import init_parameters

if TYPE_CHECKING:
    from torchrec.distributed.maglev.data_parallel import PassReduction


# ======== Layer ========

# What a sparse net returns and a dense net receives: one batch-major tensor,
# or a tuple of them (mixed dtypes, or MaglevLayer.shared_sparse_outputs).
SparseSeam = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def _unpack_seam(seam: SparseSeam) -> Tuple[torch.Tensor, ...]:
    return (seam,) if isinstance(seam, torch.Tensor) else tuple(seam)


def _detach_seam(layer: nn.Module, output: Any) -> SparseSeam:
    """Detach one microbatch's sparse output into the leaves its dense net reads.

    A leaf requires grad only where its source does, so an integer output or a
    frozen table's output stays a plain tensor.

    Raises:
        TypeError: If the output is neither a tensor nor a tuple of tensors.
    """
    if isinstance(output, torch.Tensor):
        return output.detach().requires_grad_(output.requires_grad)
    if (
        isinstance(output, tuple)
        and output
        and all(isinstance(t, torch.Tensor) for t in output)
    ):
        return tuple(t.detach().requires_grad_(t.requires_grad) for t in output)
    raise TypeError(
        f"{type(layer).__name__} sparse net returned {type(output).__name__}; "
        "the stage hands its dense net a tensor or a non-empty tuple of "
        "tensors. Return a tensor or a non-empty tuple of tensors"
    )


class _LayerChain(nn.Module):
    """The layers this stage owns under layers_per_stage, as one module: e.g.
    layers_per_stage=[3, 2] gives stage 1 layers 3 and 4, registered as layers.3
    and layers.4.

    Registering the layers under their global model indices keeps the authored
    model's parameter FQNs; the layers run in insertion order. This is the
    module the parallelizers wrap.

    :meth:`forward` runs one microbatch through every layer, or one phase, so
    a wrapper observes every call: given ``sparse_only``, every sparse net on
    one microbatch; given ``sparse_outputs``, one dense microbatch; given
    ``batch_size`` and ``num_microbatches``, every sparse net on the pass's
    whole batch.

    Args:
        layers: the stage's layers, in execution order.
        postproc: the model's output seam on the stage that owns the final
            layer, called as ``postproc(activations, layer_inputs[-1])``;
            ``None`` elsewhere.
        first_layer_index: global index of this stage's first layer.
        loss_only_output: whether the last stage returns only ``postproc``'s
            loss, keeping auxiliary predictions out of a wrapper's output
            traversal.
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
        self._layer_profile_names: List[str] = [
            f"## torchrec_maglev:layer[{first_layer_index + index}] "
            f"{type(layer).__name__.lstrip('_')} ##"
            for index, layer in enumerate(layers)
        ]
        self._sparse_net_profile_names: List[str] = [
            f"## torchrec_maglev:layer[{first_layer_index + index}] "
            f"{type(layer).__name__.lstrip('_')} sparse ##"
            for index, layer in enumerate(layers)
        ]
        # Plain attribute, not a submodule: postproc is a bound method of the
        # authored model, whose parameters are already owned by ``layers``.
        self._postproc = postproc
        self._loss_only_output = loss_only_output

    def forward(  # noqa: C901
        self,
        layer_inputs: Sequence[Any],
        in_activations: Activations = (),
        *,
        sparse_only: bool = False,
        sparse_outputs: Optional[Sequence[Optional[SparseSeam]]] = None,
        batch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
    ) -> Any:
        r"""forward(layer_inputs, in_activations=(), *, sparse_only=False, sparse_outputs=None, batch_size=None, num_microbatches=None) -> Any

        Run one microbatch, one microbatch's sparse nets, the sparse phase, or
        one dense microbatch.

        Args:
            layer_inputs (Sequence[Any]): One input per local layer.
            in_activations (Activations): Activation entering the stage; dense
                phase only. Default: ``()``
            sparse_only (bool): Run every sparse net on this microbatch and
                return their outputs, ``None`` for a layer without one.
                Default: ``False``
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
        if sparse_only:
            if (
                in_activations
                or sparse_outputs is not None
                or batch_size is not None
                or num_microbatches is not None
            ):
                raise ValueError(
                    "a microbatch's sparse nets take no other phase argument"
                )
            return self._run_sparse_micro(layer_inputs)
        if sparse_outputs is not None:
            if batch_size is not None or num_microbatches is not None:
                raise ValueError(
                    "dense forward does not accept batch_size or num_microbatches"
                )
            return self._run_dense_micro(layer_inputs, in_activations, sparse_outputs)
        if batch_size is None and num_microbatches is None:
            return self._run_dense_micro(
                layer_inputs, in_activations, self._run_sparse_micro(layer_inputs)
            )
        if in_activations:
            raise ValueError("sparse forward does not accept incoming activations")
        if batch_size is None or num_microbatches is None:
            raise ValueError("sparse forward requires batch_size and num_microbatches")
        return self._run_sparse_global(layer_inputs, batch_size, num_microbatches)

    def _run_sparse_micro(self, layer_inputs: Sequence[Any]) -> List[Any]:
        """Every layer's sparse net on its input; ``None`` where there is none."""
        sparse_outputs: List[Any] = []
        for layer, layer_input, profile_name in zip(
            self.layers.values(), layer_inputs, self._sparse_net_profile_names
        ):
            sparse = cast(MaglevLayer, layer).sparse
            with record_function(profile_name):
                sparse_outputs.append(None if sparse is None else sparse(layer_input))
        return sparse_outputs

    def _run_sparse_global(
        self, layer_inputs: Sequence[Any], batch_size: int, num_microbatches: int
    ) -> List[_GlobalSparseOutput]:
        outputs: List[_GlobalSparseOutput] = []
        for layer, layer_input, profile_name in zip(
            self.layers.values(), layer_inputs, self._layer_profile_names
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

    def _run_dense_micro(
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
            self.layers.values(),
            layer_inputs,
            sparse_outputs,
            self._layer_profile_names,
        ):
            with record_function(profile_name):
                activations = cast(MaglevLayer, layer).require_dense()(
                    sparse_output, layer_input, activations
                )
        return self._apply_postproc(activations, layer_inputs)

    def _apply_postproc(
        self, activations: Activations, layer_inputs: Sequence[Any]
    ) -> Any:
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


# ======== Global batch and microbatch: the pass's data ========

_GlobalSparseOutput = Tuple[Optional[SparseSeam], List[Any]]


def _split_seams_per_microbatch(
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
        for position, tensor in enumerate(_unpack_seam(pooled))
    ]
    return [
        (
            leaves[0][index]
            if isinstance(pooled, torch.Tensor)
            else tuple(parts[index] for parts in leaves)
        )
        for index in range(num_microbatches)
    ]


@dataclass
class MaglevPassState:
    r"""MaglevPassState(sparse_outputs, seams, dense_inputs)

    What a pass's sparse forward hands its dense microbatches and its sparse
    backward.

    Args:
        sparse_outputs (List[Optional[SparseSeam]]): Per local layer, the
            sparse net's whole-pass output; ``None`` for a layer with no sparse
            net.
        seams (List[List[Optional[SparseSeam]]]): Per microbatch, a detached
            slice of each ``sparse_outputs`` entry. A shared position
            (``MaglevLayer.shared_sparse_outputs``) holds one whole-pass leaf.
        dense_inputs (List[List[Any]]): Per microbatch, one dense input per
            local layer.
    """

    sparse_outputs: List[Optional[SparseSeam]]
    seams: List[List[Optional[SparseSeam]]]
    dense_inputs: List[List[Any]]


@dataclass
class _PendingMicrobatch:
    """A dense microbatch forwarded but not yet backwarded."""

    in_activations: Activations
    outputs: Any
    microbatch_id: int
    seams: Sequence[Optional[SparseSeam]]


# ======== Stage ========


class StageWrapper(nn.Module):
    """Takes a whole Maglev model and keeps the one stage this rank owns.

    Every rank passes the same ``model`` and ``layers_per_stage``; the wrapper
    keeps this rank's contiguous run of the model's own layers, not copies.

    It runs the stage's share of each phase of a pass, in whatever order a
    schedule (:mod:`~torchrec.distributed.maglev.pipeline`) calls them:

    * :meth:`sparse_forward_micro` runs every sparse net on one microbatch
      and hands the dense phase detached seams, or
      :meth:`sparse_forward_global` runs every sparse net once on the pass's
      whole batch and cuts each output into detached per-microbatch seams;
    * :meth:`dense_forward_micro` runs one dense microbatch, between the
      activation handoffs;
    * :meth:`dense_backward_micro` backwards one microbatch whole, or
      :meth:`dense_backward_act_micro` and :meth:`dense_backward_weight_micro`
      split it into its input-gradient and weight-gradient phases;
    * :meth:`sparse_backward_micro` resumes one microbatch's sparse graph from
      its seams' gradients, or :meth:`sparse_backward_global` resumes every
      sparse graph from its seams' summed gradients.

    **Parallelism is the caller's job.** The wrapper cuts the model, builds the
    stage's process groups, and reads the boundary contract; it does not shard.
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

    Raises:
        ValueError: if a layer has no dense net or overrides ``forward``
            (:func:`~torchrec.distributed.maglev.checks.reject_unsplit_layers`),
            a layer owns a trainable parameter outside its sparse and dense
            nets, ``layers_per_stage`` does not describe ``model``, the
            process-group layout does not match ``stage_size`` or
            ``layers_per_stage``, or the kept layers disagree on the activation
            they exchange.
    """

    # ---- Stage: cut and bind ----

    def __init__(
        self,
        model: MaglevModuleList,
        layers_per_stage: Sequence[int],
        stage_size: int,  # number of ranks for each stage
        loss_only_output: bool = False,
        process_groups: Optional[MaglevProcessGroups] = None,
    ) -> None:
        super().__init__()
        # On every rank, before any collective: a stage that refused alone would
        # leave its peers blocked in group creation.
        reject_unsplit_layers(list(model))
        reject_parameters_outside_nets(
            [layer for layer in model if isinstance(layer, MaglevLayer)]
        )
        self._outside_layers_error: Optional[str] = None
        try:
            reject_parameters_outside_layers(model)
        except ValueError as error:
            self._outside_layers_error = str(error)
        if process_groups is None:
            process_groups = MaglevProcessGroups.from_scratch(stage_size)
        else:
            if process_groups.stage_size != stage_size:
                raise ValueError(
                    f"stage_size {stage_size} does not match process-group layout "
                    f"size {process_groups.stage_size}"
                )
        num_stages = process_groups.num_stages
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
            cast(MaglevLayer, model[i]) for i in self._layer_indices
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
        self._pending_microbatches: Deque[_PendingMicrobatch] = deque()
        # The weight work each input-gradient phase deferred, oldest first.
        self._pending_weight_backwards: Deque[Tuple[List[Dict[str, Any]], int]] = (
            deque()
        )
        # Sparse forwards whose backward has not run yet, oldest first: each
        # layer's sparse output, the seams the dense phase read, microbatch id.
        self._pending_sparse_microbatches: Deque[
            Tuple[List[Any], List[Optional[SparseSeam]], int]
        ] = deque()
        self._in_specs: Tuple[ActivationSpec, ...] = layers[0].in_activation_specs()
        self._out_specs: Tuple[ActivationSpec, ...] = layers[-1].out_activation_specs()
        chain = _LayerChain(
            layers,
            model.postproc if stage_index == num_stages - 1 else None,
            first_layer_index=self._layer_indices.start,
            loss_only_output=loss_only_output,
        )
        # Public and reassignable: wrap it in DMP/DDP/FSDP2 and assign it back.
        self.module: nn.Module = chain
        # The authored layers, reachable after :attr:`module` is wrapped. Not a
        # submodule: registering it twice would duplicate every state_dict key.
        object.__setattr__(self, "_layer_chain", chain)
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
        # The configuring schedule's gradient reduction; None until configured.
        self._reduction: Optional[PassReduction] = None
        self._pass_losses: List[torch.Tensor] = []
        # Set by to(): a meta-authored model has no device to infer.
        self.device: Optional[torch.device] = None

    @property
    def _num_layers(self) -> int:
        """How many layers this stage owns (== how many inputs it takes)."""
        return self.layers_per_stage[self.stage_index]

    @property
    def _layer_indices(self) -> range:
        """Which of the model's layers this stage owns, in model order."""
        start = sum(self.layers_per_stage[: self.stage_index])
        return range(start, start + self._num_layers)

    @property
    def _handoff_pgs(self) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
        """The activation and gradient handoff process groups."""
        return self._process_groups.handoff_pgs

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

    def _neighbor_rank(self, offset: int) -> int:
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

    @property
    def layer_chain(self) -> _LayerChain:
        """This stage's layers as authored, even after :attr:`module` is wrapped."""
        return cast(_LayerChain, self._layer_chain)

    def reject_untrained_parameters(self) -> None:
        """Refuse a model MaglevRail cannot train.

        Raises:
            ValueError: if a trainable weight sits outside every layer, where
                the pass never runs it.
        """
        if self._outside_layers_error is not None:
            raise ValueError(self._outside_layers_error)

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

    # ---- Stage: feed (input distribution) ----

    def _batch_size(self, stage_input: Sequence[Any]) -> int:
        r"""_batch_size(stage_input) -> int

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
        return self._batch_size(stage_input)

    def _split_batch_per_stage(self, model_input: Any) -> List[List[Any]]:
        """Turn one raw batch into the cascade send set: ``result[s]`` is the
        input list for stage ``s``, from the model's :meth:`preproc` split of the
        batch, under ``no_grad``."""
        with record_function("## torchrec_maglev:preproc ##"), torch.no_grad():
            layer_inputs = self._preproc(model_input)
        grouped: List[List[Any]] = []
        offset = 0
        for count in self.layers_per_stage:
            grouped.append(list(layer_inputs[offset : offset + count]))
            offset += count
        return grouped

    def take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Hand the schedule ``n`` redistributed inputs for this stage.

        This method redistributes batches but never chunks them. The schedule
        decides whether each input is a microbatch or a whole pass. It runs as
        many input-distribution rounds as needed and keeps the remainder queued,
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

        Raises:
            RuntimeError: If a previous pass failed or a pass is running: the
                round is a collective, and the peers it would wait on are out of
                step.
        """
        self._refuse_out_of_step()
        with record_function("## torchrec_maglev:input_driver ##"):
            return self._input_driver.take(
                lambda: self._split_batch_per_stage(next(dataloader_iter)), n
            )

    # ---- Stage: connect (cross-HSD hand-off) ----
    #
    # The stage owns the wire: its neighbours (_neighbor_rank), the handoff
    # communicators, and the specs that fix the wire layout. A schedule decides
    # when to call these, as with torch.distributed.pipelining's PipelineStage.

    def start_recv_act(self, stage_input: Optional[Sequence[Any]] = None) -> None:
        """Post a receive for the previous HSD's activation.

        Allocates one buffer per incoming spec and issues the receives in spec
        order, so both sides of the boundary agree without exchanging metadata.
        No-op on the first stage. Each call queues another receive;
        :meth:`_wait_for_act` dequeues them in issue order.

        Args:
            stage_input: the local inputs for the activation being received. Only
                required when an incoming spec has a dynamic batch dimension.
        """
        if self.is_first:
            return
        batch_size = self._activation_batch_size(stage_input)
        act_pg, _ = self._handoff_pgs
        src = self._neighbor_rank(-1)
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

    def _wait_for_act(self) -> Activations:
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
                f"stage {self.stage_index}: waiting for an activation with no "
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

    def _start_send_act(self, outputs: Activations) -> None:
        """Send this stage's activation to the next HSD; no-op if last.

        The caller must :meth:`_finish_send_act` the previous send first. The send
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
                "flight; call _finish_send_act() before starting the next"
            )
        check_activations(
            self._out_specs, outputs, f"stage {self.stage_index} output activation"
        )
        act_pg, _ = self._handoff_pgs
        self._send_act = self._isend(outputs, self._neighbor_rank(1), act_pg)

    def _finish_send_act(self) -> None:
        """Complete the activation send in flight, if any."""
        self._drain(self._send_act)

    def start_recv_grad(self) -> None:
        """Post a receive for the next HSD's gradient of the oldest awaiting output.

        One receive per grad-carrying output slot. No-op on the last stage, whose
        gradient starts at its own loss. Each call queues another receive.
        """
        if self.is_last:
            return
        _, grad_pg = self._handoff_pgs
        src = self._neighbor_rank(1)
        works: List[Any] = []
        tensors: List[torch.Tensor] = []
        outputs = cast(
            Activations, self._pending_microbatches[len(self._recv_grad)].outputs
        )
        for output, spec in zip(outputs, self._out_specs):
            if not spec.requires_grad:
                continue
            grad = torch.empty_like(output, memory_format=torch.contiguous_format)
            works.append(dist.irecv(grad, src=src, group=grad_pg))
            tensors.append(grad)
        self._recv_grad.append((works, tensors))

    def _wait_for_grad(self) -> List[torch.Tensor]:
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
                f"stage {self.stage_index}: waiting for a gradient with no "
                "receive in flight; call start_recv_grad() first"
            )
        works, tensors = self._recv_grad.popleft()
        for work in works:
            work.wait()
        return tensors

    def _start_send_grad(self, in_activations: Activations) -> None:
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
                "call _finish_send_grad() before starting the next"
            )
        _, grad_pg = self._handoff_pgs
        grads: List[torch.Tensor] = []
        for tensor, spec in zip(in_activations, self._in_specs):
            if not spec.requires_grad:
                continue
            grads.append(
                tensor.grad if tensor.grad is not None else torch.zeros_like(tensor)
            )
        self._send_grad = self._isend(grads, self._neighbor_rank(-1), grad_pg)

    def _finish_send_grad(self) -> None:
        """Complete the gradient send in flight, if any."""
        self._drain(self._send_grad)

    def drain_sends(self) -> None:
        """Complete the send in flight in each direction; a pass ends here."""
        self._finish_send_act()
        self._finish_send_grad()

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

    # ---- Stage: pass discipline ----

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
        self._pending_microbatches.clear()
        self._pending_weight_backwards.clear()
        self._pending_sparse_microbatches.clear()
        self._pass_losses = []

    def configure_pass(
        self,
        schedule: object,
        *,
        loss_scale: float,
        allow_unreached: bool,
        reduction: Optional[PassReduction] = None,
    ) -> None:
        r"""configure_pass(schedule, *, loss_scale, allow_unreached, reduction=None) -> None

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
            reduction (PassReduction, optional): The schedule's gradient
                reduction, which runs the deferred weight backwards and maps
                the dense parameters to the ones a backward's graph holds
                (:class:`~torchrec.distributed.maglev.data_parallel.PassReduction`).
                ``None`` runs upstream ``stage_backward_weight`` on the
                parameters as registered. Default: ``None``
        """
        previous = None if self._schedule is None else self._schedule()
        if previous is not schedule:
            self._schedule = weakref.ref(schedule)
            self._checks.restart()
        self._loss_scale = loss_scale
        self._checks.set_unreached_allowed(allow_unreached)
        self._reduction = reduction

    def _seed_loss_grads(self, loss: torch.Tensor) -> Optional[List[torch.Tensor]]:
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

    # ---- Global batch and microbatch: phases, in pass order ----

    def sparse_forward_global(
        self,
        global_inputs: Sequence[Any],
        num_microbatches: int,
        *,
        unwrapped: bool = False,
    ) -> MaglevPassState:
        r"""sparse_forward_global(global_inputs, num_microbatches, *, unwrapped=False) -> MaglevPassState

        Run each sparse net once and cut its output into detached dense seams.

        Args:
            global_inputs (Sequence[Any]): One whole-pass input per local layer.
            num_microbatches (int): Number of dense microbatches.
            unwrapped (bool): Call :attr:`layer_chain` directly, past the
                wrappers
                :func:`~torchrec.distributed.maglev.data_parallel.only_data_parallel_wraps`
                allows. Default: ``False``

        Returns:
            MaglevPassState: Sparse graphs, seams, and dense inputs.

        Raises:
            TypeError: If a sparse net returns neither a tensor nor a
                non-empty tuple of tensors.
            ValueError: If input counts, batch dimensions or shared positions
                are invalid.
        """
        if len(global_inputs) != self._num_layers:
            raise ValueError(
                f"expected {self._num_layers} layer inputs, got {len(global_inputs)}"
            )
        if num_microbatches <= 0:
            raise ValueError(
                f"num_microbatches must be positive, got {num_microbatches}"
            )
        batch_size = self._batch_size(global_inputs)
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

        module = cast(nn.Module, self._layer_chain) if unwrapped else self.module
        with record_function("## torchrec_maglev:sparse_forward_global ##"):
            layer_outputs = cast(
                List[_GlobalSparseOutput],
                module(
                    global_inputs,
                    batch_size=batch_size,
                    num_microbatches=num_microbatches,
                ),
            )
            if len(layer_outputs) != self._num_layers:
                raise ValueError(
                    f"sparse phase returned {len(layer_outputs)} layer outputs, "
                    f"expected {self._num_layers}"
                )
            sparse_outputs = [output for output, _dense_inputs in layer_outputs]
            cuts = [
                (
                    None
                    if output is None
                    else _split_seams_per_microbatch(
                        output, num_microbatches, layer.shared_sparse_outputs()
                    )
                )
                for layer, output in zip(self._layers, sparse_outputs)
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
            actual_batch_size = self._batch_size(dense_input)
            if actual_batch_size != expected_batch_size:
                raise ValueError(
                    f"dense microbatch {microbatch} has batch size "
                    f"{actual_batch_size}, expected {expected_batch_size}"
                )
        return MaglevPassState(
            sparse_outputs=sparse_outputs, seams=seams, dense_inputs=dense_inputs
        )

    def sparse_net_buffers(self) -> List[torch.Tensor]:
        """The buffers the local layers' sparse nets hold."""
        return [
            buffer
            for layer in self._layers
            if layer.sparse is not None
            for buffer in layer.sparse.buffers()
        ]

    def sparse_forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int = 0,
        *,
        unwrapped: bool = False,
    ) -> List[Optional[SparseSeam]]:
        r"""sparse_forward_micro(stage_input, microbatch_id=0, *, unwrapped=False) -> List[Optional[SparseSeam]]

        Run every local layer's sparse net on one microbatch and return the
        detached seams its dense phase reads; :meth:`sparse_backward_micro`
        resumes the sparse graph from the seams' gradients.

        A seam requires grad only where its sparse output does.

        Args:
            stage_input (Sequence[Any]): One input per local layer.
            microbatch_id (int): Identifier used in profiler ranges. Default: ``0``
            unwrapped (bool): Call :attr:`layer_chain` directly, past the
                wrappers
                :func:`~torchrec.distributed.maglev.data_parallel.only_data_parallel_wraps`
                allows. Default: ``False``

        Returns:
            List[Optional[SparseSeam]]: Per local layer, its seam, or ``None``
            for a layer without a sparse net.

        Raises:
            TypeError: If a sparse net returns neither a tensor nor a non-empty
                tuple of tensors.
        """
        module = cast(nn.Module, self._layer_chain) if unwrapped else self.module
        with record_function(f"## sparse_forward mb{microbatch_id} ##"):
            outputs = list(module(stage_input, sparse_only=True))
        seams: List[Optional[SparseSeam]] = [
            None if output is None else _detach_seam(layer, output)
            for layer, output in zip(self._layers, outputs)
        ]
        self._pending_sparse_microbatches.append((outputs, seams, microbatch_id))
        return seams

    def _compute_dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        in_activations: Activations,
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[SparseSeam]],
    ) -> Any:
        r"""_compute_dense_forward_micro(stage_input, in_activations, microbatch_id, sparse_outputs) -> Any

        Compute one dense forward, without the handoffs, and keep its graph for
        the backward.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            in_activations (Activations): Activation entering this stage.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[SparseSeam]]): One sparse seam per
                local layer, ``None`` for a layer without a sparse net.

        Returns:
            Any: The stage's output, kept for the backward.
        """
        with record_function(f"## dense_forward mb{microbatch_id} ##"):
            outputs = self.module(
                stage_input, in_activations, sparse_outputs=sparse_outputs
            )
        # The seams survive with the microbatch: a split backward fills only the
        # leaves it is handed.
        self._pending_microbatches.append(
            _PendingMicrobatch(in_activations, outputs, microbatch_id, sparse_outputs)
        )
        return outputs

    def dense_forward_micro(
        self,
        stage_input: Sequence[Any],
        microbatch_id: int,
        sparse_outputs: Sequence[Optional[SparseSeam]],
    ) -> Any:
        r"""dense_forward_micro(stage_input, microbatch_id, sparse_outputs) -> Any

        Receive, compute and send one microbatch's dense phase. The schedule
        posts the activation receive (:meth:`start_recv_act`) before calling
        this.

        Args:
            stage_input (Sequence[Any]): One dense input per local layer.
            microbatch_id (int): Identifier used in profiler ranges.
            sparse_outputs (Sequence[Optional[SparseSeam]]): One sparse seam per
                local layer, ``None`` for a layer without a sparse net.

        Returns:
            Any: The stage's output, kept for the backward.
        """
        with record_function(f"## dense_recv_act mb{microbatch_id} ##"):
            in_activations = self._wait_for_act()
        outputs = self._compute_dense_forward_micro(
            stage_input, in_activations, microbatch_id, sparse_outputs
        )
        with record_function(f"## dense_finish_send_act mb{microbatch_id} ##"):
            self._finish_send_act()
        with record_function(f"## dense_send_act mb{microbatch_id} ##"):
            self._start_send_act(outputs)
        return outputs

    def dense_parameters(self) -> List[nn.Parameter]:
        r"""dense_parameters() -> List[nn.Parameter]

        The trainable parameters of the layers' dense nets, in layer order,
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

    def _dense_parameters_in_graph(self) -> List[nn.Parameter]:
        """:meth:`dense_parameters` as a backward's graph holds them, mapped by
        the configured reduction
        (:meth:`~torchrec.distributed.maglev.data_parallel.PassReduction.graph_parameters`);
        as registered when no schedule configured one.
        """
        parameters = self.dense_parameters()
        if self._reduction is None:
            return parameters
        return self._reduction.graph_parameters(
            [layer.require_dense() for layer in self._layers], parameters
        )

    @property
    def pending_weight_work(self) -> int:
        """``W`` phases :meth:`dense_backward_act_micro` queued and none ran yet."""
        return len(self._pending_weight_backwards)

    def _compute_dense_backward_micro(
        self, grads: Sequence[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Activations]:
        r"""_compute_dense_backward_micro(grads) -> Tuple[Optional[torch.Tensor], Activations]

        The oldest pending dense microbatch's whole backward, without the
        handoffs, through the dense weights' ``AccumulateGrad`` nodes so a
        data-parallel wrapper's hooks observe them: from its loss on the last
        stage, else from ``grads``, the gradients its outputs received. It
        stops at the seams, filling their ``.grad``.

        Args:
            grads (Sequence[torch.Tensor]): The next stage's gradients for this
                microbatch's grad-carrying outputs; ignored on the last stage.

        Returns:
            Tuple[Optional[torch.Tensor], Activations]: The microbatch loss on
            the last stage (else ``None``), and the activation it received, whose
            ``.grad`` goes upstream.

        Raises:
            ValueError: if the gradients do not match this stage's grad-carrying
                outputs.
            RuntimeError: on the first pass, if a trainable dense weight is
                unreached while unreached weights are not allowed.
        """
        microbatch, roots, root_grads, loss = self._pop_backward_roots(grads)
        self._checks.before_whole_backward(roots, self._dense_parameters_in_graph)
        if roots:
            with record_function(f"## dense_backward mb{microbatch.microbatch_id} ##"):
                torch.autograd.backward(roots, root_grads)
        return loss, microbatch.in_activations

    def dense_backward_micro(self) -> Optional[torch.Tensor]:
        r"""dense_backward_micro() -> Optional[torch.Tensor]

        Receive, compute and send the oldest pending dense microbatch's whole
        backward (:meth:`_compute_dense_backward_micro`). The schedule posts
        :meth:`start_recv_grad` first.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage, ``None``
            elsewhere.

        Raises:
            ValueError: if the received gradients do not match this stage's
                grad-carrying outputs.
            RuntimeError: on the first pass, if a trainable dense weight is
                unreached while unreached weights are not allowed.
        """
        microbatch = self._pending_microbatches[0]
        loss, _in_activations = self._compute_dense_backward_micro(
            self._receive_grads(microbatch)
        )
        self._send_input_grads(microbatch)
        return loss

    def dense_backward_act_micro(self) -> Optional[torch.Tensor]:
        r"""dense_backward_act_micro() -> Optional[torch.Tensor]

        The ``I`` phase: gradients w.r.t. this stage's inputs (the received
        activations and the sparse seams), sent upstream at once; the ``W``
        phase waits for :meth:`dense_backward_weight_micro`. With no
        differentiable input the backward runs whole and leaves an empty ``W``.

        Returns:
            Optional[torch.Tensor]: the microbatch loss on the last stage,
            ``None`` elsewhere; detached unless it is a view.

        Raises:
            RuntimeError: if a trainable dense weight would not be trained.
        """
        microbatch, roots, root_grads, loss = self._pop_backward_roots()
        microbatch_id = microbatch.microbatch_id
        input_values: List[torch.Tensor] = [
            tensor
            for tensor, spec in zip(microbatch.in_activations, self._in_specs)
            if spec.requires_grad
        ]
        for seam in microbatch.seams:
            if seam is not None:
                input_values.extend(_unpack_seam(seam))

        if roots and input_values:
            weights = self._dense_parameters_in_graph()
            # stage_backward_input detaches the roots that are not views in
            # place; the checks walk the graph from their nodes afterwards.
            root_nodes = [_get_grad_fn_or_grad_acc(root) for root in roots]
            with record_function(f"## dense_backward_act mb{microbatch_id} ##"):
                _input_grads, param_groups = stage_backward_input(
                    stage_outputs_or_loss=roots,
                    output_grads=root_grads,
                    input_values=input_values,
                    # One list for both phases, so the groups that I records
                    # line up with what W looks up.
                    weights=iter(weights),
                )
            self._checks.after_backward_act(
                param_groups, weights, root_nodes, input_values
            )
            del root_nodes
            self._pending_weight_backwards.append((param_groups, microbatch_id))
        else:
            self._checks.before_whole_backward(roots, self._dense_parameters_in_graph)
            if roots:
                with record_function(
                    f"## dense_backward_act mb{microbatch_id} (whole) ##"
                ):
                    torch.autograd.backward(roots, root_grads)
            # An empty W, so every I still pairs with one.
            self._pending_weight_backwards.append(([], microbatch_id))
        self._send_input_grads(microbatch)
        return loss

    def dense_backward_weight_micro(self) -> None:
        r"""dense_backward_weight_micro() -> None

        The ``W`` phase :meth:`dense_backward_act_micro` deferred, for its oldest
        microbatch; the queue must be empty before the optimizer step. It agrees
        with a whole backward to floating-point tolerance, not bit-exactly. The
        configured reduction runs it
        (:meth:`~torchrec.distributed.maglev.data_parallel.PassReduction.weight_backward`:
        under DDP, through each weight's ``AccumulateGrad``); upstream
        ``stage_backward_weight`` when no schedule configured one.

        Raises:
            ValueError: if no deferred weight work is queued.
        """
        if not self._pending_weight_backwards:
            raise ValueError(
                f"stage {self.stage_index}: dense_backward_weight_micro() with no "
                "deferred weight work; call dense_backward_act_micro() first"
            )
        param_groups, microbatch_id = self._pending_weight_backwards.popleft()
        if not param_groups:
            return
        with record_function(f"## dense_backward_weight mb{microbatch_id} ##"):
            if self._reduction is None:
                stage_backward_weight(
                    iter(self._dense_parameters_in_graph()), param_groups
                )
            else:
                self._reduction.weight_backward(
                    self._dense_parameters_in_graph, param_groups
                )

    def sparse_backward_micro(self) -> None:
        r"""sparse_backward_micro() -> None

        Resume the oldest pending microbatch's sparse graphs from its seams'
        gradients, in one backward for every layer; afterwards the seams hold
        no ``.grad``. A sparse output whose seam received no gradient is left
        out, as a single backward through both nets would leave it (unlike
        :meth:`sparse_backward_global`, which must concatenate a whole-pass
        gradient and zero-fills).
        """
        outputs, seams, microbatch_id = self._pending_sparse_microbatches.popleft()
        roots: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []
        for output, seam in zip(outputs, seams):
            if output is None or seam is None:
                continue
            for whole, leaf in zip(_unpack_seam(output), _unpack_seam(seam)):
                grad = leaf.grad
                leaf.grad = None
                if whole.requires_grad and grad is not None:
                    roots.append(whole)
                    gradients.append(grad)
        if roots:
            with record_function(f"## sparse_backward mb{microbatch_id} ##"):
                torch.autograd.backward(roots, gradients)

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
        for index, (layer, pooled) in enumerate(
            zip(self._layers, state.sparse_outputs)
        ):
            if pooled is None:
                continue
            shared = layer.shared_sparse_outputs()
            layer_seams = [cast(SparseSeam, seams[index]) for seams in state.seams]
            for position, whole in enumerate(_unpack_seam(pooled)):
                leaves = [_unpack_seam(seam)[position] for seam in layer_seams]
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

    # ---- Global batch and microbatch: backward helpers ----

    def _receive_grads(self, microbatch: _PendingMicrobatch) -> List[torch.Tensor]:
        """The next stage's gradients for ``microbatch``; none on the last stage."""
        if self.is_last:
            return []
        with record_function(f"## dense_recv_grad mb{microbatch.microbatch_id} ##"):
            return self._wait_for_grad()

    def _pop_backward_roots(
        self, grads: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[
        _PendingMicrobatch,
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Pop the oldest pending microbatch with its backward's roots: the
        loss-scaled ``postproc`` loss on the last stage, else the grad-carrying
        outputs paired with the next stage's gradients.

        Args:
            grads: the next stage's gradients, already received; received here
                when ``None``.

        Returns:
            The microbatch, the roots, their gradients, and the loss (last stage
            only).

        Raises:
            ValueError: if the received gradients do not match the grad-carrying
                outputs.
        """
        microbatch = self._pending_microbatches.popleft()
        if self.is_last:
            loss = cast(Activations, microbatch.outputs)[0]
            self._pass_losses.append(loss.detach())
            return microbatch, [loss], self._seed_loss_grads(loss), loss
        if grads is None:
            grads = self._receive_grads(microbatch)
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

    def _send_input_grads(self, microbatch: _PendingMicrobatch) -> None:
        with record_function(
            f"## dense_finish_send_grad mb{microbatch.microbatch_id} ##"
        ):
            self._finish_send_grad()
        with record_function(f"## dense_send_grad mb{microbatch.microbatch_id} ##"):
            self._start_send_grad(microbatch.in_activations)
