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
from torch.autograd.profiler import record_function
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.stage import HandoffPGMode, StageWrapper


def _no_sync_modules(module: nn.Module) -> List[nn.Module]:
    """The parallel wrappers in ``module``'s tree whose gradient sync to suppress.

    Two-tier, following ``GradientAccumulationWrapper._get_ddp_modules``: the root
    counts if it merely *has* ``no_sync``, which covers DDP and custom wrappers,
    while descendants must be ``DistributedDataParallel``
    instances. The asymmetry is deliberate -- the root is known to be the parallel
    wrapper, whereas a descendant is an arbitrary submodule. FSDP2 synchronization
    is controlled separately immediately before each backward.

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
    :meth:`_forward_context`, which every schedule uses to arm DDP's single
    gradient sync for a pass. FSDP2 synchronization is configured directly before
    each backward.

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
            ``stage.module``, so an unwrapped stage needs nothing and a DDP/DMP
            one works unconfigured. Pass your own only if that derivation is
            wrong for your setup. FSDP2 modules are discovered independently.

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
        self._fsdp_modules: List[FSDPModule] = [
            module
            for module in stage.module.modules()
            if isinstance(module, FSDPModule)
        ]

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

    def _take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Acquire this stage's inputs; adapters may override the input seam."""
        return self.stage.take_inputs(dataloader_iter, n)

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
        (stage_input,) = self._take_inputs(dataloader_iter, 1)

        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()
        self.stage.start_recv_act()
        self.stage.forward_micro(stage_input, microbatch_id=0)
        # Immediately before the backward, never earlier -- see Maglev1F1B.
        self.stage.start_recv_grad()
        loss = self.stage.backward_micro()
        self.stage.drain_sends()
        with record_function("## torchrec_maglev:optimizer_step ##"):
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

    def _configure_fsdp_backward(
        self,
        backward_idx: int,
        num_backwards: int,
    ) -> None:
        """Reduce FSDP2 gradients on the final microbatch backward."""
        should_sync = backward_idx == num_backwards - 1
        for module in self._fsdp_modules:
            module.set_requires_gradient_sync(should_sync, recurse=False)
            module.set_is_last_backward(should_sync)


class Maglev1F1B(MaglevPipelineBase):
    """The 1F1B (one-forward-one-backward) schedule.

    Standard PipeDream-flush: each stage runs ``num_warmup`` forwards, then
    interleaves 1 forward / 1 backward in steady state, then drains the remaining
    backwards in cooldown. ``num_warmup = num_stages - stage_index - 1`` (clamped
    to the microbatch count), so deeper stages warm up less and the send/recv
    pairs line up across ranks.

    Communication follows the stage's global handoff-PG configuration. ``SPLIT``
    keeps activation and gradient traffic on independent communicators and uses
    the established start/wait ordering. ``SHARED`` batches the opposing
    directions at each steady-state crossover with ``batch_isend_irecv``.

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
        microbatch_inputs = self._take_inputs(dataloader_iter, self.num_microbatches)

        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()

        if stage.handoff_pg_mode is HandoffPGMode.SHARED:
            self._progress_shared(microbatch_inputs)
        else:
            self._progress_split(microbatch_inputs)

        stage.drain_sends()
        with record_function("## torchrec_maglev:optimizer_step ##"):
            self.optimizer.step()
        return None

    def _progress_split(self, microbatch_inputs: List[List[Any]]) -> None:
        """Run 1F1B with independent activation and gradient communicators."""
        stage = self.stage
        fwd_idx = 0
        bwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            stage.start_recv_act()
            with self._forward_context(fwd_idx, self.num_microbatches):
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        def _backward() -> None:
            nonlocal bwd_idx
            self._configure_fsdp_backward(bwd_idx, self.num_microbatches)
            stage.start_recv_grad()
            stage.backward_micro()
            bwd_idx += 1

        for _ in range(self.num_warmup):
            _forward()
        for _ in range(self.num_steady):
            _forward()
            _backward()
        for _ in range(self.num_warmup):
            _backward()

    def _progress_shared(self, microbatch_inputs: List[List[Any]]) -> None:
        """Run 1F1B with both handoff directions batched on one communicator."""
        stage = self.stage

        fwd_idx = 0
        bwd_idx = 0

        def _compute_forward(in_activations: tuple[torch.Tensor, ...]) -> Any:
            nonlocal fwd_idx
            with self._forward_context(fwd_idx, self.num_microbatches):
                outputs = stage.compute_forward_micro(
                    microbatch_inputs[fwd_idx], in_activations, fwd_idx
                )
            fwd_idx += 1
            return outputs

        def _compute_backward(
            grads: Sequence[torch.Tensor],
        ) -> tuple[Optional[torch.Tensor], tuple[torch.Tensor, ...]]:
            nonlocal bwd_idx
            self._configure_fsdp_backward(bwd_idx, self.num_microbatches)
            result = stage.compute_backward_micro(grads)
            bwd_idx += 1
            return result

        # Warmup: fill the pipeline with one-way forward hand-offs.
        if self.num_warmup:
            stage.start_recv_act()
        for i in range(self.num_warmup):
            in_activations = stage.wait_for_act()
            outputs = _compute_forward(in_activations)
            stage.start_send_act(outputs)
            stage.finish_send_act()
            if i < self.num_warmup - 1:
                stage.start_recv_act()

        # Seed the activation consumed by the first steady iteration.
        stage.start_recv_act()
        in_activations = stage.wait_for_act()

        # Steady state: batch the two opposing directions at each boundary.
        for i in range(self.num_steady):
            outputs = _compute_forward(in_activations)
            grads = stage.send_act_recv_grad(outputs)
            _loss, backward_inputs = _compute_backward(grads)
            if i < self.num_steady - 1:
                in_activations = stage.send_grad_recv_act(
                    backward_inputs, recv_next=True
                )
            else:
                stage.start_send_grad(backward_inputs)
                stage.finish_send_grad()

        # Cooldown: drain the backwards left by warmup.
        for _ in range(self.num_warmup):
            stage.start_recv_grad()
            grads = stage.wait_for_grad()
            _loss, backward_inputs = _compute_backward(grads)
            stage.start_send_grad(backward_inputs)
            stage.finish_send_grad()


class Maglev1F1BRecvAhead(Maglev1F1B):
    """1F1B that posts each split-mode receive ahead of its compute."""

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run receive-ahead 1F1B in split mode.

        Shared mode uses :class:`Maglev1F1B` because its opposing handoffs must
        be issued together on the shared communicator.

        Returns:
            Optional[torch.Tensor]: always ``None``, as
            :meth:`Maglev1F1B.progress`.
        """
        stage = self.stage
        if stage.handoff_pg_mode is HandoffPGMode.SHARED:
            return super().progress(dataloader_iter)

        microbatch_inputs = self._take_inputs(dataloader_iter, self.num_microbatches)

        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()

        fwd_idx = 0
        bwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            with self._forward_context(fwd_idx, self.num_microbatches):
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        stage.start_recv_act()

        for _ in range(self.num_warmup):
            stage.start_recv_act()
            _forward()

        def _backward() -> None:
            nonlocal bwd_idx
            self._configure_fsdp_backward(bwd_idx, self.num_microbatches)
            stage.backward_micro()
            bwd_idx += 1

        for i in range(self.num_steady):
            stage.start_recv_grad()
            _forward()
            if i < self.num_steady - 1:
                stage.start_recv_act()
            else:
                stage.start_recv_grad()
            _backward()

        for i in range(self.num_warmup):
            if i < self.num_warmup - 1:
                stage.start_recv_grad()
            _backward()

        stage.drain_sends()
        with record_function("## torchrec_maglev:optimizer_step ##"):
            self.optimizer.step()
        return None
