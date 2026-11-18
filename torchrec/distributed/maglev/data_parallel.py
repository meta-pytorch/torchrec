#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reduce a Maglev stage's gradients under DDP or FSDP2.

DDP and ZeRO-2 (``fully_shard(reshard_after_forward=False)``) reduce once per
pass; ZeRO-3 (``reshard_after_forward=True``) reduce-scatters per microbatch,
for ZeRO-3 gradient memory. :func:`make_pass_reduction` returns the driver for
whichever wrapper a stage holds.
"""

from __future__ import annotations

import contextlib
from typing import ContextManager, Iterator, List, Sequence

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel


class PassReduction:
    """The no-op reduction of an unwrapped stage, and the drivers' interface."""

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


def make_pass_reduction(module: nn.Module, schedule: str) -> PassReduction:
    r"""make_pass_reduction(module, schedule) -> PassReduction

    Return the driver for the data-parallel wrappers in ``module``'s tree.

    Args:
        module (nn.Module): The stage module, as the caller wrapped it.
        schedule (str): The schedule's name, for error messages.

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
        return DDPPassReduction(ddp, schedule)
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
    r"""DDPPassReduction(modules, schedule)

    Drive DDP's reducer once per pass, by hand: every forward runs under
    ``no_sync``, and :meth:`reduce` flushes every bucket after the pass. A
    parameter no replica used keeps ``grad=None``.

    Args:
        modules (List[DistributedDataParallel]): Every DDP module in the stage.
        schedule (str): The schedule's name, for error messages.

    Raises:
        ValueError: If a module is configured in a way the flush cannot drive.
    """

    def __init__(self, modules: List[DistributedDataParallel], schedule: str) -> None:
        for module in modules:
            _validate_ddp_module(module, schedule)
        self._modules: Sequence[DistributedDataParallel] = modules

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
    ZeRO-2 and a ZeRO-3 module may share a stage but not a state context.

    Args:
        modules (Sequence[FSDPModule]): Every FSDP2 module in the stage.

    Raises:
        ValueError: If a ZeRO-2 and a ZeRO-3 module share a state context.
    """

    def __init__(self, modules: Sequence[FSDPModule]) -> None:
        self._held: List[FSDPModule] = [
            module for module in modules if not _reshards_after_forward(module)
        ]
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

    @contextlib.contextmanager
    def accumulate(self) -> Iterator[None]:
        """Hold the ZeRO-2 modules' reduction for the pass; re-arm it even if the pass raised."""
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
