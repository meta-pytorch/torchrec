#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""A ``torch.compile`` backend whose backward still splits into ``I`` and ``W``.

The zero-bubble dense schedule splits each microbatch backward with
``torch.distributed.pipelining``'s ``stage_backward_input`` (``I``) and
``stage_backward_weight`` (``W``). In eager the split is free: every aten
backward node computes only the outputs the current graph task consumes, so
``I`` runs the input-gradient matmuls and ``W`` the weight-gradient ones.

``torch.compile`` breaks it. AOTAutograd wraps the whole compiled backward in
one ``autograd.Function``, and a Function only knows the ``needs_input_grad``
recorded at forward time. ``I`` therefore runs the entire backward and discards
the weight gradients, and ``W`` re-enters the same node and runs the entire
backward again -- every compiled dense half pays for its backward twice.

:class:`SplitBackwardBackend` restores the split the way GraphPP does for LLM
pipelines (torchtitan ``graph_trainer/graph_pp/split_di_dw.py``): export the
joint graph, partition it with the min-cut partitioner Inductor uses, cut the
backward graph into an input-gradient graph and a weight-gradient graph, and
compile each separately. The graphs run behind one autograd node that asks the
engine which of its inputs the current graph task needs, so the pipelining
helpers drive it unchanged.
"""

from dataclasses import dataclass
from typing import Any, Callable, cast, List, Optional, Sequence, Set, Tuple

import torch
import torch._inductor
import torch.fx as fx
import torch.nn as nn
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch._functorch.partitioners import (
    _extract_graph_with_inputs_outputs,
    min_cut_rematerialization_partition,
)
from torch.autograd.function import once_differentiable
from torch.fx._lazy_graph_module import _make_graph_module

# Compiles one extracted aten graph against example inputs.
Compiler = Callable[[fx.GraphModule, Sequence[torch.Tensor]], Callable[..., Any]]


def inductor_compiler(
    gm: fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> Callable[..., Any]:
    """Compile one extracted graph with Inductor, specialized to its inputs."""
    with torch.no_grad():
        return torch._inductor.standalone_compile(
            gm, list(example_inputs), dynamic_shapes="from_example_inputs"
        )


def eager_compiler(
    gm: fx.GraphModule, example_inputs: Sequence[torch.Tensor]
) -> Callable[..., Any]:
    """Run one extracted graph as traced, without compiling it."""
    return gm


class _LazyGraph:
    """An extracted graph, compiled on its first call against that call's inputs.

    The backward graphs see their first real inputs only inside the first
    backward, which is also when AOTAutograd compiles its own backward.
    """

    def __init__(self, gm: fx.GraphModule, compiler: Compiler) -> None:
        self.gm = gm
        self._compiler = compiler
        self._compiled: Optional[Callable[..., Any]] = None

    def __call__(self, *args: Any) -> Any:
        compiled = self._compiled
        if compiled is None:
            compiled = self._compiled = self._compiler(self.gm, args)
        return compiled(*args)


@dataclass(frozen=True)
class _BackwardSplit:
    """A backward graph cut into its input-gradient and weight-gradient halves.

    ``input_grads`` takes the backward's inputs and returns the input gradients
    followed by the live-ins ``weight_grads`` takes. Slots index the outputs of
    the uncut backward, which are one gradient per forward input.
    """

    input_grads: _LazyGraph
    weight_grads: _LazyGraph
    input_slots: Tuple[int, ...]
    weight_slots: Tuple[int, ...]


def _extract(
    graph: fx.Graph, inputs: List[fx.Node], outputs: List[fx.Node]
) -> fx.Graph:
    descs: List[Any] = [None] * len(outputs)
    return _extract_graph_with_inputs_outputs(
        graph, inputs, outputs, descs, ignore_must_be_in_fw_bw=True
    )


def _weight_live_ins(
    graph: fx.Graph, input_closure: Set[str], weight_grads: Set[fx.Node]
) -> List[fx.Node]:
    """The values the input-gradient graph computes that the weight graph reads."""
    live_ins: List[fx.Node] = []
    for node in graph.nodes:
        if node.name not in input_closure or node.op == "get_attr":
            continue
        weight_users = [
            user
            for user in node.users
            if user.op != "output" and user.name not in input_closure
        ]
        if not weight_users and node not in weight_grads:
            continue
        if isinstance(node.meta.get("val"), (tuple, list)):
            # A multi-output op is consumed through getitem; hand over the leaves.
            live_ins.extend(weight_users)
        else:
            live_ins.append(node)
    return list(dict.fromkeys(live_ins))


def _split_backward(
    bw: fx.GraphModule, num_weights: int, compiler: Compiler
) -> Optional[_BackwardSplit]:
    """Cut ``bw`` into input- and weight-gradient graphs, or ``None`` if one is empty.

    ``bw`` must return one gradient per forward input with the ``num_weights``
    weight gradients first, the convention GraphPP's ``split_di_dw_graph``
    uses. Everything the weight half needs is either computed by the input half
    or is one of the backward's own inputs; both reach it as outputs of the
    input half, so the weight graph runs on its live-ins alone.
    """
    (output,) = bw.graph.find_nodes(op="output")
    grads = output.args[0]
    weight_slots = tuple(
        slot for slot in range(num_weights) if isinstance(grads[slot], fx.Node)
    )
    input_slots = tuple(
        slot
        for slot in range(num_weights, len(grads))
        if isinstance(grads[slot], fx.Node)
    )
    if not weight_slots or not input_slots:
        return None

    placeholders = list(bw.graph.find_nodes(op="placeholder"))
    input_grads = [grads[slot] for slot in input_slots]
    weight_grads = [grads[slot] for slot in weight_slots]
    closure = _extract(bw.graph, placeholders, input_grads)
    live_ins = _weight_live_ins(
        bw.graph,
        {node.name for node in closure.nodes if node.op != "output"},
        set(weight_grads),
    )
    input_graph = _extract(bw.graph, placeholders, input_grads + live_ins)
    weight_graph = _extract(bw.graph, live_ins, weight_grads)
    return _BackwardSplit(
        input_grads=_LazyGraph(_make_graph_module(bw, input_graph), compiler),
        weight_grads=_LazyGraph(_make_graph_module(bw, weight_graph), compiler),
        input_slots=input_slots,
        weight_slots=weight_slots,
    )


def _engine_visible(primal: torch.Tensor) -> torch.Tensor:
    # A leaf's edge is its AccumulateGrad node, and the engine refuses to say
    # whether it will deliver to a leaf that autograd.grad() captures -- which
    # is what both I and W do. A view puts a non-leaf node on that edge.
    if primal.requires_grad and primal.grad_fn is None:
        return primal.view_as(primal)
    return primal


class _SplitProgram:
    """One dynamo graph as forward, input-gradient and weight-gradient graphs."""

    def __init__(
        self,
        gm: fx.GraphModule,
        example_inputs: Sequence[torch.Tensor],
        compiler: Compiler,
    ) -> None:
        weights = [
            slot
            for slot, value in enumerate(example_inputs)
            if isinstance(value, nn.Parameter) and value.requires_grad
        ]
        others = [slot for slot in range(len(example_inputs)) if slot not in weights]
        # Primal order: weights first, so the backward's gradients are too.
        self.order: Tuple[int, ...] = (*weights, *others)
        (gm_output,) = gm.graph.find_nodes(op="output")
        self.num_outputs: int = len(gm_output.args[0])

        def forward(*primals: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            args: List[Any] = [None] * len(primals)
            for slot, primal in zip(self.order, primals):
                args[slot] = primal
            return tuple(gm(*args))

        joint = cast(
            fx.GraphModule,
            aot_export_joint_simple(
                forward,
                tuple(example_inputs[slot] for slot in self.order),
                trace_joint=True,
            ),
        )
        tangents = [
            node.name
            for node in joint.graph.find_nodes(op="placeholder")
            if node.name.startswith("tangents")
        ]
        if len(tangents) != self.num_outputs:
            raise NotImplementedError(
                "SplitBackwardBackend needs every output to be differentiable; "
                f"{self.num_outputs - len(tangents)} of {self.num_outputs} are not"
            )
        fw, bw = min_cut_rematerialization_partition(
            joint,
            None,
            num_fwd_outputs=self.num_outputs,
            static_lifetime_input_indices=list(range(len(weights))),
        )

        # The partitioner names each backward input after the forward output or
        # tangent it receives, so match them by name rather than by position.
        (fw_output,) = fw.graph.find_nodes(op="output")
        saved = [node.name for node in fw_output.args[0][self.num_outputs :]]
        self.backward_sources: Tuple[Tuple[bool, int], ...] = tuple(
            (
                (True, tangents.index(node.name))
                if node.name in tangents
                else (False, saved.index(node.name))
            )
            for node in bw.graph.find_nodes(op="placeholder")
        )
        self.forward = _LazyGraph(fw, compiler)
        self.backward = _LazyGraph(bw, compiler)
        self.split: Optional[_BackwardSplit] = _split_backward(
            bw, len(weights), compiler
        )

    def backward_args(
        self,
        saved: Sequence[torch.Tensor],
        grad_outputs: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        # Contiguous as AOTAutograd makes them: the compiled backward is
        # specialized to the strides of its first tangents.
        return [
            grad_outputs[index].contiguous() if is_tangent else saved[index]
            for is_tangent, index in self.backward_sources
        ]

    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        primals = [_engine_visible(args[slot]) for slot in self.order]
        return _SplitBackwardFunction.apply(self, *primals)


def _scatter(
    grads: List[Optional[torch.Tensor]],
    wanted: Sequence[bool],
    slots: Sequence[int],
    values: Sequence[torch.Tensor],
) -> None:
    for slot, value in zip(slots, values):
        if wanted[slot]:
            grads[slot] = value


class _SplitBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, program: _SplitProgram, *primals: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        results = program.forward(*primals)
        ctx.program = program
        ctx.weight_live_ins = None
        ctx.save_for_backward(*results[program.num_outputs :])
        return tuple(results[: program.num_outputs])

    @staticmethod
    @once_differentiable
    # pyrefly: ignore[bad-override]
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        program: _SplitProgram = ctx.program
        # The question each eager backward node asks the engine before computing
        # an output, asked here once per input of the whole compiled region.
        wanted = [
            fn is not None and torch._C._will_engine_execute_node(fn)
            for fn, _ in ctx.next_functions
        ]
        grads: List[Optional[torch.Tensor]] = [None] * len(wanted)
        split = program.split
        if split is None:
            results = program.backward(
                *program.backward_args(ctx.saved_tensors, grad_outputs)
            )
            _scatter(grads, wanted, range(len(grads)), results)
            return (None, *grads)

        if any(wanted[slot] for slot in split.input_slots) or (
            ctx.weight_live_ins is None
        ):
            results = split.input_grads(
                *program.backward_args(ctx.saved_tensors, grad_outputs)
            )
            num_input_grads = len(split.input_slots)
            _scatter(grads, wanted, split.input_slots, results[:num_input_grads])
            # Detached: a live-in can be one of this node's own outputs, and
            # holding it would tie the node to itself until W runs.
            ctx.weight_live_ins = [
                value.detach() for value in results[num_input_grads:]
            ]
        if any(wanted[slot] for slot in split.weight_slots):
            live_ins, ctx.weight_live_ins = ctx.weight_live_ins, None
            _scatter(grads, wanted, split.weight_slots, split.weight_grads(*live_ins))
        return (None, *grads)


class SplitBackwardBackend:
    r"""SplitBackwardBackend(compiler=inductor_compiler)

    ``torch.compile`` backend whose backward runs its input and weight halves
    separately.

    A drop-in for the default backend on a Maglev dense half: the forward, the
    input-gradient half and the weight-gradient half are each compiled, and
    ``stage_backward_input`` / ``stage_backward_weight`` run only their own
    half. A plain ``loss.backward()`` runs both, so the monolithic schedules
    are unaffected.

    Weights are the graph inputs that are trainable ``nn.Parameter`` s, which
    is what dynamo lifts a module's parameters to; every other differentiable
    input belongs to the input half.

    Limitations: static shapes only, and the region must neither mutate its
    inputs nor return a view of one -- the constraints of
    ``aot_export_joint_simple``, which traces it.

    Args:
        compiler (Callable): compiles each extracted aten graph against example
            inputs. Default: :func:`inductor_compiler`

    Example::

        >>> layer.dense = torch.compile(layer.dense, backend=SplitBackwardBackend())
    """

    def __init__(self, compiler: Compiler = inductor_compiler) -> None:
        self._compiler = compiler

    def __call__(
        self, gm: fx.GraphModule, example_inputs: Sequence[Any]
    ) -> Callable[..., Any]:
        if not all(isinstance(value, torch.Tensor) for value in example_inputs):
            raise NotImplementedError(
                "SplitBackwardBackend traces static shapes only; dynamo passed a "
                "symbolic input"
            )
        if not (
            torch.is_grad_enabled()
            and any(value.requires_grad for value in example_inputs)
        ):
            return self._compiler(gm, example_inputs)
        return _SplitProgram(gm, example_inputs, self._compiler)
