#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Schedules that drive a staged Maglev model across per-stage HSDs.

A schedule reads raw batches from a dataloader and decides *when* to forward,
backward, and hand off; the :class:`StageWrapper` it holds owns everything about
how (the wire, the process groups, the input distribution).
"""

import contextlib
from typing import Any, Callable, ContextManager, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.stage import StageWrapper


def _no_sync_modules(module: nn.Module) -> List[nn.Module]:
    """The parallel wrappers in ``module``'s tree whose gradient sync to suppress.

    Two-tier, following ``GradientAccumulationWrapper._get_ddp_modules``: the root
    counts if it merely *has* ``no_sync``, which covers DDP, FSDP and custom
    wrappers alike, while descendants must be ``DistributedDataParallel``
    instances. The asymmetry is deliberate -- the root is known to be the parallel
    wrapper, whereas a descendant is an arbitrary submodule and a nested FSDP's
    ``no_sync`` does not mean the same thing.

    Descendants matter because a sharded submodule can hold its own inner DDP for
    data-parallel lookups; suppressing only the outer wrapper leaves those
    all-reducing on every backward.

    ``DistributedModelParallel`` is unwrapped first: it defines no ``no_sync`` of
    its own, so the wrapper worth entering is the module it holds.
    """
    found: List[nn.Module] = []
    root = module
    wrapped = getattr(module, "_dmp_wrapped_module", None)
    if isinstance(wrapped, nn.Module):
        root = wrapped
    if hasattr(root, "no_sync"):
        found.append(root)
    found.extend(
        m
        for m in root.modules()
        if m is not root and isinstance(m, DistributedDataParallel)
    )
    return found


@contextlib.contextmanager
def _no_sync(modules: Sequence[nn.Module]) -> Iterator[None]:
    """Enter every wrapper's ``no_sync`` at once; a no-op when there are none."""
    with contextlib.ExitStack() as stack:
        for module in modules:
            # pyre-ignore[16]: presence of no_sync is what put it in this list
            stack.enter_context(module.no_sync())
        yield


class MaglevPipelineBase:
    """The trivial schedule -- one forward, one backward, one step -- and the base
    every other schedule extends.

    With a single microbatch every stage runs the same forward-then-backward
    wave, so the sends drain before the gradients are reduced and no interleaving
    is needed. Used directly by the correctness test. This is *not* 1F1B with
    ``num_microbatches=1``: that schedule needs at least one microbatch per stage
    to fill (see :class:`Maglev1F1B`).

    Subclasses override :meth:`progress` with their own ordering; what they
    inherit is the stage and optimizer, the boundary-contract check, and
    :meth:`_forward_context`, which every schedule uses to place the single
    gradient sync of a pass.

    Args:
        stage: this rank's :class:`StageWrapper`. Everything about where this
            rank sits -- its stage, its position in that stage's HSD, the
            neighbouring ranks, the hand-off process groups -- is read off the
            wrapper rather than derived a second time here.
        optimizer: the stage's optimizer. Gradients accumulate across a pass and
            are applied with a single step.
        no_sync: returns a context that suppresses the DP wrapper's gradient
            sync, entered around every forward but the pass's last. Defaults to
            suppressing whatever wrappers :func:`_no_sync_modules` finds on
            ``stage.module``, so an unwrapped stage needs nothing and a
            DDP/FSDP/DMP one works unconfigured. Pass your own only if that
            derivation is wrong for your setup.

    Raises:
        ValueError: if a boundary stage declares an activation it cannot have (an
            incoming one on the first stage, or none on any later stage).
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
    ) -> None:
        self.stage = stage
        self.optimizer = optimizer
        # Read once: the module tree is static after wrapping, and the
        # pipeline is built after it.
        modules = _no_sync_modules(stage.module)
        self._no_sync: Callable[[], ContextManager[None]] = no_sync or (
            lambda: _no_sync(modules)
        )

        in_specs = stage.in_activation_specs()
        if stage.is_first and in_specs:
            raise ValueError(
                f"stage 0 declares an incoming activation {in_specs} but has "
                "no previous stage to receive it from"
            )
        if not stage.is_first and not in_specs:
            raise ValueError(
                f"stage {stage.stage_index} declares no incoming activation; only "
                "stage 0 may start the chain"
            )

    @property
    def is_first(self) -> bool:
        return self.stage.is_first

    @property
    def is_last(self) -> bool:
        return self.stage.is_last

    @property
    def microbatches_per_pass(self) -> int:
        """Microbatches one :meth:`progress` call consumes.

        What a caller measuring throughput has to divide by; schedules differ.
        """
        return 1

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run one microbatch and apply the gradients.

        Args:
            dataloader_iter: yields raw batches -- whatever the model's
                :meth:`~torchrec.distributed.maglev.module.MaglevModuleList.preproc`
                consumes. The stage pulls from it, partitions each batch into
                per-layer inputs, and all-to-alls them over its cascade, so a
                rank never has to know which layers it owns to feed it.

        Returns:
            Optional[torch.Tensor]: the loss on the last stage, ``None`` on every
            other stage. The last stage's model scored itself in ``postproc``, so
            no label or criterion is passed in.
        """
        (stage_input,) = self.stage.take_inputs(dataloader_iter, 1)

        self.optimizer.zero_grad()
        self.stage.start_recv_act()
        self.stage.forward_micro(stage_input, microbatch_id=0)
        # Immediately before the backward, never earlier -- see Maglev1F1B.
        self.stage.start_recv_grad()
        loss = self.stage.backward_micro()
        self.stage.drain_sends()
        self.optimizer.step()
        return loss

    def _forward_context(self, fwd_idx: int, num_forwards: int) -> ContextManager[None]:
        """Suppress gradient sync for every microbatch but the pass's last.

        Wraps the **forward**, not the backward. ``DistributedDataParallel``
        reads ``require_backward_grad_sync`` in ``_pre_forward`` /
        ``_post_forward`` -- that is where the reducer is armed for the coming
        backward -- so a ``no_sync`` placed around ``backward()`` alone has no
        effect and every microbatch all-reduces. PyTorch documents the same:
        "The forward pass should be included inside the context manager, or else
        gradients will still be synchronized."

        The last microbatch forwards outside the context, so its backward carries
        the one reduction for the whole pass, over gradients every earlier
        microbatch accumulated locally.

        Args:
            fwd_idx: index of this forward within the pass.
            num_forwards: forwards this pass will run.
        """
        if fwd_idx == num_forwards - 1:
            return contextlib.nullcontext()
        return self._no_sync()


class Maglev1F1B(MaglevPipelineBase):
    """The 1F1B (one-forward-one-backward) schedule, with plain P2P.

    Standard PipeDream-flush: each stage runs ``num_warmup`` forwards, then
    interleaves 1 forward / 1 backward in steady state, then drains the remaining
    backwards in cooldown. ``num_warmup = num_stages - stage_index - 1`` (clamped
    to the microbatch count), so deeper stages warm up less and the send/recv
    pairs line up across ranks.

    Each receive is posted immediately before the operation that consumes it, so
    a transfer is issued and then waited on with nothing in between: the stage
    stalls for the whole latency of every hand-off. That is the straightforward
    ordering, and the baseline that :class:`Maglev1F1BRecvAhead` is measured
    against.

    Sends are still asynchronous -- ``forward_micro`` / ``backward_micro`` start
    one and finish it on the next call -- so only the receive side blocks.

    Args:
        stage: this rank's :class:`StageWrapper`.
        optimizer: the stage's optimizer.
        num_microbatches: microbatches per pass. Fixed for the pipeline's life,
            so the schedule's shape is computed once here rather than per pass.

    Raises:
        ValueError: if there are fewer microbatches than stages. That leaves the
            early stages with no steady phase at all, and it is the steady phase
            that seeds the gradient receives -- so those ranks would fail in
            cooldown while the later ones ran on, hanging the job
            asymmetrically. It is also a pointless schedule: the pipeline never
            fills, so it is all bubble.
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
    ) -> None:
        super().__init__(stage, optimizer, no_sync)
        if num_microbatches < stage.num_stages:
            raise ValueError(
                f"1F1B needs at least one microbatch per stage: got "
                f"{num_microbatches} for {stage.num_stages} stages"
            )
        self.num_microbatches: int = num_microbatches
        self.num_warmup: int = min(
            stage.num_stages - stage.stage_index - 1, num_microbatches
        )
        self.num_steady: int = num_microbatches - self.num_warmup

    @property
    def microbatches_per_pass(self) -> int:
        return self.num_microbatches

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run one 1F1B pass.

        Gradients are accumulated across all microbatches, then DP-averaged and
        applied with a single optimizer step.

        Args:
            dataloader_iter: yields raw batches, as in
                :meth:`MaglevPipelineBase.progress`. One pass consumes as many as
                the input distribution needs to produce ``num_microbatches``.

        Returns:
            Optional[torch.Tensor]: always ``None``. Reading a per-microbatch
            loss means ``loss.item()``, which is a device-to-host sync in the
            middle of the schedule -- it would stall the pipeline it is meant to
            be measuring. Use :class:`MaglevPipelineBase` when a value is
            actually needed.
        """
        stage = self.stage
        microbatch_inputs = stage.take_inputs(dataloader_iter, self.num_microbatches)

        self.optimizer.zero_grad()

        fwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            stage.start_recv_act()
            with self._forward_context(fwd_idx, self.num_microbatches):
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        def _backward() -> None:
            stage.start_recv_grad()
            stage.backward_micro()

        # Warmup: fill the pipeline.
        for _ in range(self.num_warmup):
            _forward()

        # Steady state: one forward, one backward.
        for _ in range(self.num_steady):
            _forward()
            _backward()

        # Cooldown: drain remaining backwards.
        for _ in range(self.num_warmup):
            _backward()

        stage.drain_sends()
        self.optimizer.step()
        return None


class Maglev1F1BRecvAhead(Maglev1F1B):
    """1F1B that posts each receive a microbatch ahead of the compute it feeds.

    Same warmup / steady / cooldown structure and the same number of transfers as
    :class:`Maglev1F1B`; only the *placement* of the receives differs. The
    transfer for microbatch ``k+1`` is posted before ``k`` is computed, so it
    lands during that compute instead of being waited on the moment it is issued.
    That hides the hand-off latency behind the neighbouring stage's work, which is
    the whole point of splitting the P2P into start/wait.

    .. note::
        Gradient receives cannot run as far ahead as activation receives. The
        first one on a rank creates the pair's NCCL communicator and blocks until
        the peer's matching send, and a peer only touches the gradient direction
        in its own backward -- so posting one during the forward phase parks this
        rank mid-wave and deadlocks the pipeline. The first gradient receive
        therefore waits until warmup is over; from then on running a microbatch
        ahead is free.
    """

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run one 1F1B pass, keeping both receive directions one microbatch ahead.

        Returns:
            Optional[torch.Tensor]: always ``None``, as
            :meth:`Maglev1F1B.progress`.
        """
        stage = self.stage
        microbatch_inputs = stage.take_inputs(dataloader_iter, self.num_microbatches)

        self.optimizer.zero_grad()

        fwd_idx = 0

        def _forward() -> None:
            """Run one forward, consuming the activation receive posted before it."""
            nonlocal fwd_idx
            with self._forward_context(fwd_idx, self.num_microbatches):
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        # Two receives outstanding before the first forward: this seed plus the
        # one the first warmup iteration posts.
        stage.start_recv_act()

        # Warmup: fill the pipeline.
        for _ in range(self.num_warmup):
            stage.start_recv_act()
            _forward()

        def _backward() -> None:
            """Run one backward."""
            stage.backward_micro()

        # Steady state: one forward, one backward.
        for i in range(self.num_steady):
            # Warmup is over by now, so the stages below have every activation
            # they need to reach their own first backward -- see the note on the
            # class about why this cannot move earlier.
            stage.start_recv_grad()
            _forward()
            if i < self.num_steady - 1:
                stage.start_recv_act()
            else:
                # No more forwards to run ahead for; keep the gradient side one
                # ahead for cooldown instead.
                stage.start_recv_grad()
            _backward()

        # Cooldown: drain remaining backwards.
        for i in range(self.num_warmup):
            if i < self.num_warmup - 1:
                stage.start_recv_grad()
            _backward()

        stage.drain_sends()
        self.optimizer.step()
        return None
