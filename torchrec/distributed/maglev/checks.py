#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Refusals for what a Maglev pass would run wrong or silently leave untrained.

Detached seams, a split dense backward, and training only a layer's sparse and
dense nets can each leave a weight without a gradient where one whole backward would not;
these checks refuse such a model or graph.
"""

from __future__ import annotations

from typing import Any, Callable, cast, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.graph import Node
from torch.distributed.distributed_c10d import _get_object_coll_device
from torch.distributed.pipelining._backward import _get_grad_fn_or_grad_acc
from torchrec.distributed.maglev.module import MaglevLayer


def check_sparse_output(
    layer: nn.Module,
    pooled: object,
    batch_size: int,
    shared: Sequence[int] = (),
) -> None:
    """Require a sparse net's output to be one floating-point tensor or a
    non-empty tuple of them, batch-major except at the ``shared`` positions.

    Raises:
        TypeError: If the output is not a tensor or a tuple of tensors.
        ValueError: If a shape, dtype or shared position does not fit.
    """
    name = f"{type(layer).__name__} (sparse net)"
    tensors = pooled if isinstance(pooled, tuple) else (pooled,)
    if not tensors or not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError(
            f"{name} returned {type(pooled).__name__}, expected "
            "torch.Tensor or a non-empty tuple of them"
        )
    typed = cast(Tuple[torch.Tensor, ...], tensors)
    outside = sorted(p for p in shared if not 0 <= p < len(typed))
    if outside:
        raise ValueError(
            f"{name} names shared positions {outside}, but returned "
            f"{len(typed)} tensor(s)"
        )
    for p, tensor in enumerate(typed):
        if p not in shared and (tensor.ndim == 0 or tensor.shape[0] != batch_size):
            raise ValueError(
                f"{name} returned shape {tuple(tensor.shape)}, expected "
                f"leading dimension {batch_size}"
            )
    for tensor in typed:
        if not tensor.dtype.is_floating_point:
            raise ValueError(
                f"{name} output must be floating point, got {tensor.dtype}"
            )


def reject_unsplit_layers(layers: Sequence[nn.Module]) -> None:
    """Require every layer to be a ``MaglevLayer`` with a dense net and
    ``MaglevLayer``'s own ``forward``: a schedule calls the sparse and dense
    nets itself and never the layer's ``forward``.

    Raises:
        ValueError: If a layer has no dense net or overrides ``forward``.
    """
    for index, layer in enumerate(layers):
        if not isinstance(layer, MaglevLayer):
            continue  # reject_parameters_outside_layers names it
        if layer.dense is None:
            problem = "has no dense net"
        elif type(layer).forward is not MaglevLayer.forward:
            problem = f"overrides forward ({type(layer).__qualname__}.forward)"
        else:
            continue
        raise ValueError(
            f"{type(layer).__name__} at index {index} {problem}. MaglevPP runs "
            "each layer as its sparse net and its dense net, called separately "
            "by the schedule, and never calls a layer's forward. Build the layer "
            "as MaglevLayer(sparse=..., dense=...) and keep MaglevLayer.forward: "
            "move the work forward did into the dense net, or into the sparse net "
            "if it reads only the layer input. A layer that is a whole stage (one "
            "layer standing for several) should be its layers instead, with "
            "layers_per_stage placing them."
        )


def reject_parameters_outside_nets(layers: Sequence[MaglevLayer]) -> None:
    """Require every trainable layer parameter to live in its sparse or dense
    net, the only parts of a layer a pass runs.

    Raises:
        ValueError: If a layer owns a trainable parameter outside both nets.
    """
    for index, layer in enumerate(layers):
        in_nets = {
            id(parameter)
            for net in (layer.sparse, layer.dense)
            if net is not None
            for parameter in net.parameters()
        }
        stray = sorted(
            name
            for name, parameter in layer.named_parameters()
            if parameter.requires_grad and id(parameter) not in in_nets
        )
        if stray:
            raise ValueError(
                f"{type(layer).__name__} at index {index} owns trainable "
                f"parameters outside its sparse and dense nets: {stray}. "
                "A Maglev pass runs only a layer's sparse and dense nets, so "
                "these would silently never train -- move them into the sparse "
                "or dense submodule"
            )


def reject_parameters_outside_layers(model: nn.ModuleList) -> None:
    """Require every child of the model to be a layer and every trainable model
    parameter to live in one.

    A weight ``postproc`` applied, outside every layer, would never train.

    Raises:
        ValueError: If a child is not a ``MaglevLayer``, or the model owns a
            trainable parameter outside its layers.
    """
    not_layers = [
        f"{index} ({type(child).__name__})"
        for index, child in enumerate(model)
        if not isinstance(child, MaglevLayer)
    ]
    if not_layers:
        raise ValueError(
            f"{type(model).__name__} has children that are not MaglevLayers, at "
            f"{not_layers}: every module registered on the model counts as a "
            "layer -- move a head that postproc applies into the last layer's "
            "dense net"
        )
    owned = {id(parameter) for layer in model for parameter in layer.parameters()}
    stray = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in owned
    )
    if stray:
        raise ValueError(
            f"{type(model).__name__} owns trainable parameters outside its "
            f"layers: {stray}. A Maglev pass runs only the layers, so these would "
            "silently never train -- move a head that postproc applies into the "
            "last layer's dense net, and register a tensor preproc reads as a "
            "buffer or with requires_grad=False"
        )


def collect_graph_nodes(roots: Sequence[Optional[Node]]) -> Set[Node]:
    """Every autograd node reachable from ``roots``, the roots included."""
    pending = [node for node in roots if node is not None]
    visited: Set[Node] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(
            parent
            for parent, _index in node.next_functions
            if parent is not None and parent not in visited
        )
    return visited


def find_uncovered_weights(
    param_groups: Sequence[Mapping[str, Any]],
    weights: Sequence[nn.Parameter],
) -> List[nn.Parameter]:
    """The trainable ``weights`` that no weight group of ``param_groups`` holds.

    By identity: a group holds grad-accumulator nodes, and a count comparison
    passes when one node sits in two groups while another weight is missing.
    """
    covered = {id(node) for group in param_groups for node in group["params"]}
    return [
        weight
        for weight in weights
        if weight.requires_grad and id(_get_grad_fn_or_grad_acc(weight)) not in covered
    ]


def count_weight_only_roots(
    root_nodes: Sequence[Optional[Node]],
    input_values: Sequence[torch.Tensor],
    weights: Sequence[nn.Parameter],
) -> int:
    """How many of the roots reach a trainable weight but no stage input.

    The weight-gradient phase resumes only where a weight's path meets an
    input's, and the input-gradient phase computes only gradients toward the
    inputs, so the gradient such a root receives never reaches its weights.
    """
    inputs = {_get_grad_fn_or_grad_acc(value) for value in input_values}
    trained = {_get_grad_fn_or_grad_acc(weight) for weight in weights}
    count = 0
    for node in root_nodes:
        reached = collect_graph_nodes([node])
        if reached & trained and not reached & inputs:
            count += 1
    return count


class DenseWeightChecks:
    r"""DenseWeightChecks(stage_index, stage_size, stage_pg, device)

    A stage's refusals of trainable dense weights its backward would not train.

    The checks inspect this replica's autograd graph, which can depend on its
    data, and one replica raising alone would leave its peers blocked in the
    next collective on the stage group. Until :meth:`end_first_pass`, every
    replica therefore exchanges a pass/fail flag before any of them raises.
    Later passes check locally, on the premise that the graph's structure does
    not change between passes.

    Args:
        stage_index (int): The stage, for error messages.
        stage_size (int): Replicas of the stage.
        stage_pg (dist.ProcessGroup): The stage's data-parallel group.
        device (Callable[[], torch.device]): The stage's device, for the flag
            exchange when the group has no CPU backend.
    """

    def __init__(
        self,
        stage_index: int,
        stage_size: int,
        stage_pg: dist.ProcessGroup,
        device: Callable[[], torch.device],
    ) -> None:
        self._stage_index = stage_index
        self._stage_size = stage_size
        self._stage_pg = stage_pg
        self._device = device
        self._agreed: bool = False
        self._allow_unreached: bool = True
        # Sets of group-less weights already proven unreached, by id.
        self._unreached_verified: Set[frozenset[int]] = set()

    def set_unreached_allowed(self, allowed: bool) -> None:
        """Whether a trainable dense weight no output reaches may keep ``grad=None``.

        Right when the reduction reconciles replicas that used different weights
        (DDP, FSDP2 with ``reduce_scatter_unused_params``) or there is nothing to
        reconcile; the schedule sets it from its reduction, before every pass.
        """
        if allowed != self._allow_unreached:
            self._allow_unreached = allowed
            self._unreached_verified.clear()

    def end_first_pass(self) -> None:
        """Mark the graph's structure agreed: later checks stay local."""
        self._agreed = True

    def restart(self) -> None:
        """Check the graph's structure again, as on the stage's first pass."""
        self._agreed = False
        self._unreached_verified.clear()

    def after_backward_act(
        self,
        param_groups: Sequence[Mapping[str, Any]],
        weights: Sequence[nn.Parameter],
        root_nodes: Sequence[Optional[Node]],
        input_values: Sequence[torch.Tensor],
    ) -> None:
        """Refuse a weight the deferred weight-gradient phase would not train.

        A weight without a group is refused unless unreached weights are allowed
        and no output reaches it. On the first pass this also refuses an output
        that depends on dense weights but on no input.
        """

        def check() -> None:
            self._check_coverage(param_groups, weights, root_nodes)
            if not self._agreed:
                count = count_weight_only_roots(root_nodes, input_values, weights)
                if count:
                    raise RuntimeError(
                        f"stage {self._stage_index}: {count} output(s) depend on "
                        "dense weights but on no incoming activation or sparse "
                        "seam, so the split backward would drop their gradient "
                        "before it reaches those weights. Compute such an output "
                        "from an input, run its stage's backwards whole "
                        "(a custom order's B ops), or run "
                        "MaglevRail1F1B"
                    )

        self.agree(check)

    def before_whole_backward(
        self,
        roots: Sequence[torch.Tensor],
        weights: Callable[[], Sequence[nn.Parameter]],
    ) -> None:
        """Refuse a weight a whole backward from ``roots`` misses, if not allowed.

        First pass only: the graph walk has no cheap later-pass counterpart.
        """
        if self._allow_unreached:
            return

        def check() -> None:
            if self._agreed:
                return
            trainable = weights()
            visited = collect_graph_nodes(
                [_get_grad_fn_or_grad_acc(root) for root in roots]
            )
            missing = [
                weight
                for weight in trainable
                if _get_grad_fn_or_grad_acc(weight) not in visited
            ]
            if missing:
                raise _make_unreached_error(len(trainable), len(missing))

        self.agree(check)

    def agree(self, check: Callable[[], None]) -> None:
        """Run ``check``; until the first pass ends, fail on every replica together."""
        error: Optional[Exception] = None
        try:
            check()
        except Exception as raised:
            error = raised
        if self._agreed or self._stage_size == 1:
            if error is not None:
                raise error
            return
        group = self._stage_pg
        device = (
            torch.device("cpu")
            if _get_object_coll_device(group) == "cpu"
            else self._device()
        )
        failed = torch.tensor(
            [int(error is not None)], dtype=torch.int32, device=device
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        if error is not None:
            raise error
        # Citrine C6: once per check on the first pass only, so every replica
        # raises together instead of hanging the ones that passed.
        if failed.item():
            raise RuntimeError(
                f"a replica of stage {self._stage_index} failed a structural check "
                "on the first pass; its log has the cause"
            )

    def _check_coverage(
        self,
        param_groups: Sequence[Mapping[str, Any]],
        weights: Sequence[nn.Parameter],
        root_nodes: Sequence[Optional[Node]],
    ) -> None:
        missing = find_uncovered_weights(param_groups, weights)
        if not missing:
            return
        key = frozenset(id(weight) for weight in missing)
        if self._allow_unreached and key in self._unreached_verified:
            return
        visited = collect_graph_nodes(root_nodes)
        reached = [
            weight for weight in missing if _get_grad_fn_or_grad_acc(weight) in visited
        ]
        if reached:
            raise RuntimeError(
                f"the split backward reached {len(weights) - len(reached)} of "
                f"{len(weights)} trainable dense parameters, so {len(reached)} "
                "would receive no gradient and never train. A dense parameter is "
                "reachable only if its path to the stage output merges with one "
                "descended from an incoming activation or a sparse seam; one fed "
                "solely by non-differentiable inputs is not"
            )
        if not self._allow_unreached:
            raise _make_unreached_error(len(weights), len(missing))
        self._unreached_verified.add(key)


def _make_unreached_error(num_trainable: int, num_missing: int) -> RuntimeError:
    return RuntimeError(
        f"{num_missing} of {num_trainable} trainable dense parameters received no "
        "gradient from this microbatch. Under FSDP2 a parameter without a "
        "gradient is left out of the reduce-scatter, so replicas that used "
        "different parameters would issue mismatched collectives. Call "
        "set_reduce_scatter_unused_params(True) on the FSDP2 modules before "
        "building the schedule to allow inactive dense modules, or make every "
        "dense parameter reachable from the stage output"
    )
