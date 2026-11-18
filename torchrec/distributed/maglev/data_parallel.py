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
wrapper a stage holds.
"""

from __future__ import annotations

import contextlib
from typing import ContextManager, Dict, Iterable, Iterator, List, Sequence, Tuple

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel


class PassReduction:
    """The no-op reduction of an unwrapped stage, and the drivers' interface."""

    #: Whether a trainable dense weight may end a microbatch without a gradient:
    #: true unless the reduction would then issue mismatched collectives.
    allows_unreached_dense: bool = True
    #: Whether a deferred weight backward must run each weight's
    #: ``AccumulateGrad``, so the reducer observes which weights the pass used.
    accumulates_weight_grads: bool = False
    #: The FSDP2 modules that reshard after forward (ZeRO-3), and so
    #: reduce-scatter per microbatch.
    resharded_modules: Tuple[FSDPModule, ...] = ()
    #: Whether the sparse forward may call the authored layer chain directly,
    #: skipping a wrapper ``forward`` that contributes nothing to the reduction.
    sparse_forward_unwrapped: bool = False

    def no_sync(self) -> ContextManager[None]:
        """Context for every forward of the pass."""
        return contextlib.nullcontext()

    def accumulate(self) -> ContextManager[None]:
        """Context for the whole pass, from the sparse forward to the last backward."""
        return contextlib.nullcontext()

    def reduce(self) -> None:
        """Reduce the pass's accumulated gradients, before the optimizer step."""

    def close(self) -> None:
        """Release whatever the driver registered on the modules."""


def reject_model_trackers(modules: Iterable[nn.Module], schedule: str) -> None:
    """Refuse ``DistributedModelParallel``'s model trackers.

    DMP steps its trackers on every forward call, and a pass calls the stage
    module once per dense microbatch.

    Raises:
        ValueError: If any of ``modules`` holds a model tracker.
    """
    trackers = [getattr(m, "model_trackers", None) for m in modules]
    if any(isinstance(tracked, dict) and tracked for tracked in trackers):
        raise ValueError(
            f"{schedule} cannot run DistributedModelParallel's model trackers: they "
            "step on every forward, and a pass calls the stage module once per "
            "dense microbatch, so each microbatch would count as a batch. Build "
            "DMP without model_tracker_configs, and add no tracker later"
        )


def make_pass_reduction(
    module: nn.Module, schedule: str, *, layers_unwrappable: bool = False
) -> PassReduction:
    r"""make_pass_reduction(module, schedule, *, layers_unwrappable=False) -> PassReduction

    Return the driver for the data-parallel wrappers in ``module``'s tree.

    Args:
        module (nn.Module): The stage module, as the caller wrapped it.
        schedule (str): The schedule's name, for error messages.
        layers_unwrappable (bool): Whether only DMP and DDP wrap the authored
            layer chain, so the sparse forward may call it directly. Default:
            ``False``

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
        return DDPPassReduction(
            ddp, schedule, sparse_forward_unwrapped=layers_unwrappable
        )
    if fsdp:
        return FSDP2PassReduction(fsdp)
    return PassReduction()


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


class DDPPassReduction(PassReduction):
    r"""DDPPassReduction(modules, schedule, *, sparse_forward_unwrapped=False)

    Drive DDP's reducer once per pass, by hand: every forward runs under
    ``no_sync``, and :meth:`reduce` flushes every bucket after the pass. A
    parameter no replica used keeps ``grad=None``.

    Args:
        modules (List[DistributedDataParallel]): Every DDP module in the stage.
        schedule (str): The schedule's name, for error messages.
        sparse_forward_unwrapped (bool): Whether the sparse forward calls the
            authored layer chain directly: DDP's ``forward`` contributes nothing
            under ``no_sync``, and its output sink would clone every whole-pass
            sparse output. Default: ``False``

    Raises:
        ValueError: If a module is configured in a way the flush cannot drive.
    """

    accumulates_weight_grads = True

    def __init__(
        self,
        modules: List[DistributedDataParallel],
        schedule: str,
        *,
        sparse_forward_unwrapped: bool = False,
    ) -> None:
        for module in modules:
            _validate_ddp_module(module, schedule)
        self._modules: Sequence[DistributedDataParallel] = modules
        self.sparse_forward_unwrapped: bool = sparse_forward_unwrapped

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        with contextlib.ExitStack() as stack:
            for module in self._modules:
                stack.enter_context(module.no_sync())
            yield

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
    split backward needs ZeRO-2: held, the reduction waits for the weight half,
    and unresharded, the weight half finds the parameters its graph used.

    Args:
        modules (Sequence[FSDPModule]): Every FSDP2 module in the stage.

    Raises:
        ValueError: If a ZeRO-2 and a ZeRO-3 module share a state context.
    """

    def __init__(self, modules: Sequence[FSDPModule]) -> None:
        self._held: List[FSDPModule] = [
            module for module in modules if not _reshards_after_forward(module)
        ]
        self.resharded_modules = tuple(
            module for module in modules if module not in self._held
        )
        held_contexts = {
            id(module._get_fsdp_state()._state_ctx) for module in self._held
        }
        mixed = [
            type(module).__name__
            for module in modules
            if module not in self._held
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
        self._arm(False)
        try:
            yield
        finally:
            self._arm(True)

    def reduce(self) -> None:
        # finalize_backward's wait for the reduce-scatter needs is_last_backward.
        self._arm(True)
        # fully_shard([a, b]) maps several modules onto one state.
        states = {}
        for module in self._held:
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

    def _arm(self, armed: bool) -> None:
        for module in self._held:
            module.set_is_last_backward(armed)
            module.set_reshard_after_backward(armed, recurse=False)
            module.set_requires_gradient_sync(armed, recurse=False)
