#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reduce a Maglev stage's gradients under DDP or FSDP2.

DDP and ZeRO-2 (``fully_shard(reshard_after_forward=False)``) reduce once per
pass, the whole-pass sparse backward's gradients included; ZeRO-3
(``reshard_after_forward=True``) reduce-scatters per microbatch, for ZeRO-3
gradient memory. :func:`make_pass_reduction` returns the driver for whichever
wrapper a stage holds. The driver also runs the stage's deferred weight
backwards (:meth:`PassReduction.weight_backward`) and maps its dense parameters
to the ones a backward's graph holds (:meth:`PassReduction.graph_parameters`),
so the stage needs no DDP or FSDP2 type.
"""

from __future__ import annotations

import contextlib
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
from torch.autograd.graph import GradientEdge
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining._backward import stage_backward_weight
from torch.nn.parallel import DistributedDataParallel


class PassReduction:
    """The no-op reduction of an unwrapped stage, and the drivers' interface."""

    #: Whether a trainable dense weight may end a microbatch without a gradient:
    #: true unless the reduction would then issue mismatched collectives.
    allows_unreached_dense: bool = True
    #: The FSDP2 modules that reshard after forward (ZeRO-3), and so
    #: reduce-scatter per microbatch.
    resharded_modules: Tuple[FSDPModule, ...] = ()
    # The DDP and FSDP2 modules make_pass_reduction found, by id, in tree order.
    _data_parallel_module_ids: Tuple[int, ...] = ()

    def no_sync(self) -> ContextManager[None]:
        """Context for every forward of the pass."""
        return contextlib.nullcontext()

    def may_unwrap_sparse_forward(self) -> bool:
        """Whether the next sparse forward may call the authored layer chain
        directly, skipping a wrapper ``forward`` that contributes nothing."""
        return False

    def accumulate(self) -> ContextManager[None]:
        """Context for the whole pass, from the sparse forward to the last backward."""
        return contextlib.nullcontext()

    def reduce(self) -> None:
        """Reduce the pass's accumulated gradients, before the optimizer step."""

    def close(self) -> None:
        """Release whatever the driver registered on the modules."""

    def graph_parameters(
        self, modules: Sequence[nn.Module], params: Sequence[nn.Parameter]
    ) -> List[nn.Parameter]:
        r"""graph_parameters(modules, params) -> List[nn.Parameter]

        Return ``params`` as a backward's graph holds them: a ZeRO-3 module
        re-registers its sharded parameters after its forward, while the graph
        keeps the unsharded ones. Valid only around a backward: once FSDP2
        reshards, an unsharded parameter's storage is freed. Every driver maps
        the FSDP2 modules in ``modules``' trees, whatever wrapper it drives.

        Args:
            modules (Sequence[nn.Module]): The modules whose FSDP2 parameters
                to map, e.g. the stage's dense nets.
            params (Sequence[nn.Parameter]): The parameters, as registered.

        Returns:
            List[nn.Parameter]: ``params``, each FSDP2 parameter in
            ``modules`` replaced by its unsharded tensor.
        """
        graph: Dict[int, nn.Parameter] = {}
        for module in modules:
            graph.update(fsdp_graph_parameters(module))
        return [graph.get(id(param), param) for param in params]

    def weight_backward(
        self,
        graph_parameters: Callable[[], Iterable[nn.Parameter]],
        param_groups: List[Dict[str, Any]],
    ) -> None:
        r"""weight_backward(graph_parameters, param_groups) -> None

        Run one deferred weight backward: upstream ``stage_backward_weight``.

        Args:
            graph_parameters (Callable[[], Iterable[nn.Parameter]]): Returns the
                weights as the backward's graph holds them (:meth:`graph_parameters`);
                called only by a reduction that needs them.
            param_groups (List[Dict[str, Any]]): The weight groups
                ``stage_backward_input`` recorded.
        """
        stage_backward_weight(iter(graph_parameters()), param_groups)

    def reject_changed_wrappers(self, module: nn.Module, schedule: str) -> None:
        r"""reject_changed_wrappers(module, schedule) -> None

        Refuse a stage module whose DDP and FSDP2 modules are not the ones
        :func:`make_pass_reduction` found, e.g. one wrapped after the schedule
        was built: this driver would reduce stale modules. A schedule calls it
        before every pass.

        Args:
            module (nn.Module): The stage module, as the caller wrapped it.
            schedule (str): The schedule's name, for error messages.

        Raises:
            ValueError: If ``module``'s tree holds other DDP or FSDP2 modules.
        """
        if _data_parallel_module_ids(module) != self._data_parallel_module_ids:
            raise ValueError(
                f"{schedule}: the stage module's DDP and FSDP2 modules changed "
                "after the schedule was built, so its gradient reduction would "
                "drive stale ones. Wrap stage.module before building the schedule"
            )


def _data_parallel_module_ids(module: nn.Module) -> Tuple[int, ...]:
    """The ids of the DDP and FSDP2 modules in ``module``'s tree, in tree order."""
    return tuple(
        id(m)
        for m in module.modules()
        if isinstance(m, (FSDPModule, DistributedDataParallel))
    )


def reject_model_trackers(modules: Iterable[nn.Module], schedule: str) -> None:
    """Refuse ``DistributedModelParallel``'s model trackers.

    DMP steps its trackers on every forward call, and every schedule calls the
    stage module more than once per batch: once per phase of a microbatch, or
    once per dense microbatch.

    Raises:
        ValueError: If any of ``modules`` holds a model tracker.
    """
    trackers = [getattr(m, "model_trackers", None) for m in modules]
    if any(isinstance(tracked, dict) and tracked for tracked in trackers):
        raise ValueError(
            f"{schedule} cannot run DistributedModelParallel's model trackers: "
            "they step on every forward, and a stage calls its module more than "
            "once per batch, so one batch would count several times. Build DMP "
            "without model_tracker_configs, and add no tracker later"
        )


def only_data_parallel_wraps(module: nn.Module, layer_chain: nn.Module) -> bool:
    r"""only_data_parallel_wraps(module, layer_chain) -> bool

    Return whether ``module`` reaches ``layer_chain`` through
    ``DistributedModelParallel`` and ``DistributedDataParallel`` wrappers
    only, so calling the layers directly skips nothing but their ``forward``.

    Args:
        module (nn.Module): The stage module, as the caller wrapped it.
        layer_chain (nn.Module): The authored layers it wraps
            (``StageWrapper.layer_chain``).
    """
    current: object = module
    while current is not layer_chain:
        wrapped = getattr(current, "_dmp_wrapped_module", None)
        if isinstance(wrapped, nn.Module):
            current = wrapped
        elif isinstance(current, DistributedDataParallel):
            current = current.module
        else:
            return False
    return True


def make_pass_reduction(
    module: nn.Module,
    schedule: str,
    *,
    layers_unwrappable: bool = False,
    sparse_net_buffers: Iterable[torch.Tensor] = (),
) -> PassReduction:
    r"""make_pass_reduction(module, schedule, *, layers_unwrappable=False, sparse_net_buffers=()) -> PassReduction

    Return the driver for the data-parallel wrappers in ``module``'s tree.

    Args:
        module (nn.Module): The stage module, as the caller wrapped it.
        schedule (str): The schedule's name, for error messages.
        layers_unwrappable (bool): Whether only DMP and DDP wrap the authored
            layer chain, so a sparse forward may call it directly
            (:func:`only_data_parallel_wraps`). Default: ``False``
        sparse_net_buffers (Iterable[torch.Tensor]): The buffers the layers'
            sparse nets hold. Default: ``()``

    Raises:
        ValueError: If a wrapper other than DDP or FSDP2 is present, the stage
            mixes them, or a module is configured in a way the driver cannot
            reduce correctly.
    """
    modules = list(module.modules())
    # Anything else exposing no_sync (FSDP1, a custom wrapper) would train on
    # local gradients forever: the pass reduces only what it can drive by hand.
    unsupported = sorted(
        {
            type(m).__name__
            for m in modules
            if hasattr(m, "no_sync")
            and not isinstance(m, (FSDPModule, DistributedDataParallel))
        }
    )
    if unsupported:
        raise ValueError(
            f"{schedule} cannot reduce gradients for {', '.join(unsupported)}: it "
            "drives DDP's reducer or FSDP2's post-backward by hand. Use "
            "DistributedDataParallel or replicate() for a replicated dense net, "
            "or fully_shard() for a sharded one"
        )
    ddp = [m for m in modules if isinstance(m, DistributedDataParallel)]
    fsdp = [m for m in modules if isinstance(m, FSDPModule)]
    if ddp and fsdp:
        raise ValueError(
            f"{schedule} reduces either FSDP2 or DDP modules in a stage, not both"
        )
    if ddp:
        reduction: PassReduction = DDPPassReduction(
            ddp,
            schedule,
            layers_unwrappable=layers_unwrappable,
            sparse_net_buffers=sparse_net_buffers,
        )
    elif fsdp:
        reduction = FSDP2PassReduction(fsdp)
    else:
        reduction = PassReduction()
    reduction._data_parallel_module_ids = _data_parallel_module_ids(module)
    return reduction


def _flush_ddp_module(module: DistributedDataParallel) -> None:
    """Reduce one DDP module's accumulated gradients outside any backward.

    Every bucket goes through DDP's communication hook after the used-parameter
    map is all-reduced. Finalizes a manually finalized reducer: twice raises.
    """
    module.reducer.prepare_for_backward([])
    module.reducer._delay_all_reduce()
    if getattr(module, "require_manual_backward_finalization", False):
        module.finalize_backward()


def _validate_ddp_module(module: DistributedDataParallel, schedule: str) -> None:
    """Refuse a DDP configuration that a hook-free flush after the pass gets wrong."""
    if module.static_graph:
        raise ValueError(
            f"{schedule} requires DDP static_graph=False: its output sink queues a "
            "reduction from the first backward, mid-pass"
        )
    if not module.find_unused_parameters:
        raise ValueError(
            f"{schedule} requires DDP find_unused_parameters=True: the per-pass "
            "flush reduces through the used-parameter map"
        )
    if getattr(module, "mixed_precision", None) is not None:
        raise ValueError(
            f"{schedule} does not support DDP mixed_precision: the post-backward "
            "callback that copies the reduced gradients into the full-precision "
            "parameters never runs from a flush outside a backward, so they would "
            "not train"
        )
    if any(
        hasattr(parameter, "_in_backward_optimizers")
        for parameter in getattr(module, "_module_parameters", ())
    ):
        raise ValueError(
            f"{schedule} does not support optimizers applied in backward to "
            "DDP-managed parameters: they would step on every microbatch's local, "
            "unreduced gradient"
        )


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


class DDPPassReduction(PassReduction):
    r"""DDPPassReduction(modules, schedule, *, layers_unwrappable=False, sparse_net_buffers=())

    Drive DDP's reducer once per pass, by hand: every forward runs under
    ``no_sync``, and :meth:`reduce` flushes every bucket after the pass. A
    parameter no replica used keeps ``grad=None``.

    Args:
        modules (List[DistributedDataParallel]): Every DDP module in the stage.
        schedule (str): The schedule's name, for error messages.
        layers_unwrappable (bool): Whether only DMP and DDP wrap the authored
            layer chain: DDP's ``forward`` contributes nothing to a sparse
            forward under ``no_sync`` but its output sink. Default: ``False``
        sparse_net_buffers (Iterable[torch.Tensor]): The buffers the layers'
            sparse nets hold. A sparse forward runs through DDP while DDP would
            broadcast one of them first. Default: ``()``

    Raises:
        ValueError: If a module is configured in a way the flush cannot drive.
    """

    def __init__(
        self,
        modules: List[DistributedDataParallel],
        schedule: str,
        *,
        layers_unwrappable: bool = False,
        sparse_net_buffers: Iterable[torch.Tensor] = (),
    ) -> None:
        for module in modules:
            _validate_ddp_module(module, schedule)
        self._modules: Sequence[DistributedDataParallel] = modules
        self._layers_unwrappable = layers_unwrappable
        sparse_buffer_ids = {id(buffer) for buffer in sparse_net_buffers}
        # The modules that broadcast a buffer a sparse net reads.
        self._sparse_buffer_modules: List[DistributedDataParallel] = [
            module
            for module in modules
            if any(
                id(buffer) in sparse_buffer_ids
                for buffer in getattr(module, "modules_buffers", ())
            )
        ]

    def may_unwrap_sparse_forward(self) -> bool:
        r"""may_unwrap_sparse_forward() -> bool

        Return whether the next sparse forward may skip DDP's ``forward``: only
        DMP and DDP wrap the layers, and no DDP module would first broadcast a
        buffer a sparse net reads (a dense net's buffers are broadcast at the
        dense forward, before it reads them).
        """
        return self._layers_unwrappable and not any(
            module.will_sync_module_buffers() for module in self._sparse_buffer_modules
        )

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for module in self._modules:
                stack.enter_context(module.no_sync())
            yield

    def weight_backward(
        self,
        graph_parameters: Callable[[], Iterable[nn.Parameter]],
        param_groups: List[Dict[str, Any]],
    ) -> None:
        r"""weight_backward(graph_parameters, param_groups) -> None

        Run one deferred weight backward through each weight's
        ``AccumulateGrad``, so the reducer observes which weights the pass used
        (:func:`_accumulate_weight_grads`); ``graph_parameters`` is never called.
        """
        _accumulate_weight_grads(param_groups)

    def reduce(self) -> None:
        r"""reduce() -> None

        Flush every DDP module's accumulated gradients; the flush finalizes
        the reducer, so a trainer must not call ``finalize_backward()``.
        """
        for module in self._modules:
            _flush_ddp_module(module)
            # no_sync cleared it; the next pass's first forward syncs buffers.
            module.require_forward_param_sync = True


def fsdp_graph_parameters(module: nn.Module) -> Dict[int, nn.Parameter]:
    """The unsharded parameter a forward's graph holds for each FSDP2 parameter in
    ``module``'s tree, by ``id`` of its sharded parameter: a ZeRO-3 module
    re-registers the sharded ones after its forward. A parameter FSDP2 has not
    yet all-gathered has no entry.
    """
    graph: Dict[int, nn.Parameter] = {}
    for submodule in module.modules():
        if not isinstance(submodule, FSDPModule):
            continue
        for group in submodule._get_fsdp_state()._fsdp_param_groups:
            for fsdp_param in group.fsdp_params:
                unsharded = getattr(fsdp_param, "_unsharded_param", None)
                if unsharded is not None:
                    graph[id(fsdp_param.sharded_param)] = unsharded
    return graph


def _reshards_after_forward(module: FSDPModule) -> bool:
    """Whether ``module`` was wrapped with ``fully_shard(reshard_after_forward=True)``."""
    return any(
        group.post_forward_mesh_info is not None
        for group in module._get_fsdp_state()._fsdp_param_groups
    )


class FSDP2PassReduction(PassReduction):
    r"""FSDP2PassReduction(modules)

    Reduce each FSDP2 module on the schedule its ``fully_shard`` call chose.

    * ZeRO-2 (``reshard_after_forward=False``, and ``replicate``): once per
      pass. :meth:`accumulate` holds the gradient sync for the pass, and
      :meth:`reduce` runs each param group's post-backward and the root's
      final callback by hand (FSDP2 internals).
    * ZeRO-3 (``reshard_after_forward=True``): FSDP2's own reduce-scatter per
      microbatch, never held. Its final callback, with ``is_last_backward``
      left True, waits for each reduce-scatter, so the last one completes
      before the optimizer step.

    Holding the sync sets ``is_last_backward`` on a whole state context, so a
    ZeRO-2 and a ZeRO-3 module may share a stage but not a state context. A
    split backward needs ZeRO-2: held, the reduction waits for the
    weight-gradient phase, and unresharded, that phase finds the parameters its
    graph used.

    Args:
        modules (Sequence[FSDPModule]): Every FSDP2 module in the stage.

    Raises:
        ValueError: If a ZeRO-2 and a ZeRO-3 module share a state context.
    """

    def __init__(self, modules: Sequence[FSDPModule]) -> None:
        self._held_modules: List[FSDPModule] = [
            module for module in modules if not _reshards_after_forward(module)
        ]
        self.resharded_modules = tuple(
            module for module in modules if module not in self._held_modules
        )
        held_contexts = {
            id(module._get_fsdp_state()._state_ctx) for module in self._held_modules
        }
        mixed = [
            type(module).__name__
            for module in modules
            if module not in self._held_modules
            and id(module._get_fsdp_state()._state_ctx) in held_contexts
        ]
        if mixed:
            raise ValueError(
                "FSDP2 modules that reshard after forward (ZeRO-3) share a state "
                f"context with modules that do not (ZeRO-2): {mixed}. ZeRO-2 holds "
                "the gradient sync for the pass and ZeRO-3 reduce-scatters per "
                "microbatch, and the held sync would hold the whole context -- "
                "wrap the stage's nested modules with one reshard_after_forward"
            )
        # A parameter without a gradient is left out of the reduce-scatter
        # unless its group reduce-scatters zeros for it.
        self.allows_unreached_dense: bool = all(
            getattr(group, "reduce_scatter_unused_params", False)
            for module in modules
            for group in module._get_fsdp_state()._fsdp_param_groups
        )

    @contextlib.contextmanager
    def accumulate(self) -> Iterator[None]:
        """Hold the ZeRO-2 modules' reduction for the pass; re-arm it even if the pass raised.

        The whole pass rather than each dense microbatch: the sparse backward is
        a real backward that runs after the dense pipeline.
        """
        self._arm_gradient_sync(False)
        try:
            yield
        finally:
            self._arm_gradient_sync(True)

    def reduce(self) -> None:
        # finalize_backward's wait for the reduce-scatter needs is_last_backward.
        self._arm_gradient_sync(True)
        # fully_shard([a, b]) maps several modules onto one state.
        states = {}
        for module in self._held_modules:
            state = module._get_fsdp_state()
            states.setdefault(id(state), state)
        for state in states.values():
            for param_group in state._fsdp_param_groups:
                param_group.post_backward()
        # Once per state context: the callback walks every state in it, and a
        # second call would re-enter post_backward on groups back at IDLE.
        contexts = {}
        for state in states.values():
            contexts.setdefault(id(state._state_ctx), state)
        for state in contexts.values():
            state._root_post_backward_final_callback()

    def _arm_gradient_sync(self, armed: bool) -> None:
        for module in self._held_modules:
            module.set_is_last_backward(armed)
            module.set_reshard_after_backward(armed, recurse=False)
            module.set_requires_gradient_sync(armed, recurse=False)
