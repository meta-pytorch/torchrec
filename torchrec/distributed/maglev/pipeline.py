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
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevRailPassState,
    StageWrapper,
)


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
    """The non-interleaved schedule and the base every other schedule extends.

    One raw input batch produces one input-distribution round. The stage forwards
    every microbatch in that round, then backwards all of them, so this schedule
    exercises the complete model without imposing a 1F1B ordering. Used directly
    by the correctness test.

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
        # pipeline is built after it. Kept as an attribute so a subclass that
        # issues the reduction itself reduces over exactly the wrappers whose
        # sync was suppressed, rather than re-deriving a list that could differ.
        self._no_sync_modules: List[nn.Module] = _no_sync_modules(stage.module)
        modules = self._no_sync_modules
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
        return self.stage.num_stages

    @property
    def takes_whole_pass_input(self) -> bool:
        r"""takes_whole_pass_input -> bool

        Return whether one dataloader batch contains the entire pass.
        """
        return False

    def _take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Acquire this stage's inputs; adapters may override the input seam."""
        return self.stage.take_inputs(dataloader_iter, n)

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run one input-distribution round and apply the gradients.

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
        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()
        loss = self.stage(next(dataloader_iter))
        loss.backward()
        with record_function("## torchrec_maglev:optimizer_step ##"):
            self.optimizer.step()
        return loss if self.stage.is_last else None

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
        """Reduce FSDP2 gradients on the final microbatch backward.

        All three flags, as upstream's ``backward_maybe_with_nosync`` does.
        Leaving ``reshard_after_backward`` at its default reshards after every
        microbatch, so the next microbatch's forward has to all-gather again --
        measured at 8x the all-gathers for ``m=8`` before this was set.
        """
        should_sync = backward_idx == num_backwards - 1
        for module in self._fsdp_modules:
            module.set_requires_gradient_sync(should_sync, recurse=False)
            module.set_is_last_backward(should_sync)
            module.set_reshard_after_backward(should_sync)


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

    def _progress_split(
        self,
        microbatch_inputs: List[List[Any]],
    ) -> None:
        """Run 1F1B with independent activation and gradient communicators."""
        stage = self.stage
        fwd_idx = 0
        bwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            stage.start_recv_act(microbatch_inputs[fwd_idx])
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

    def _progress_shared(
        self,
        microbatch_inputs: List[List[Any]],
    ) -> None:
        """Run 1F1B with both handoff directions batched on one communicator."""
        stage = self.stage

        fwd_idx = 0
        bwd_idx = 0

        def _compute_forward(in_activations: tuple[torch.Tensor, ...]) -> Any:
            nonlocal fwd_idx
            with self._forward_context(fwd_idx, self.num_microbatches):
                outputs = stage.compute_forward_micro(
                    microbatch_inputs[fwd_idx],
                    in_activations,
                    fwd_idx,
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
            stage.start_recv_act(microbatch_inputs[fwd_idx])
        for i in range(self.num_warmup):
            in_activations = stage.wait_for_act()
            outputs = _compute_forward(in_activations)
            stage.start_send_act(outputs)
            stage.finish_send_act()
            if i < self.num_warmup - 1:
                stage.start_recv_act(microbatch_inputs[fwd_idx])

        # Seed the activation consumed by the first steady iteration.
        stage.start_recv_act(microbatch_inputs[fwd_idx])
        in_activations = stage.wait_for_act()

        # Steady state: batch the two opposing directions at each boundary.
        for i in range(self.num_steady):
            outputs = _compute_forward(in_activations)
            grads = stage.send_act_recv_grad(outputs)
            _loss, backward_inputs = _compute_backward(grads)
            if i < self.num_steady - 1:
                in_activations = stage.send_grad_recv_act(
                    backward_inputs,
                    recv_next=True,
                    next_stage_input=microbatch_inputs[fwd_idx],
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
        recv_idx = 0

        def _start_recv_act() -> None:
            nonlocal recv_idx
            stage.start_recv_act(microbatch_inputs[recv_idx])
            recv_idx += 1

        def _forward() -> None:
            nonlocal fwd_idx
            with self._forward_context(fwd_idx, self.num_microbatches):
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        _start_recv_act()

        for _ in range(self.num_warmup):
            _start_recv_act()
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
                _start_recv_act()
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


class MaglevRail(MaglevPipelineBase):
    r"""MaglevRail(stage, optimizer, num_microbatches, no_sync=None, w_lag=None)

    Batch each layer's sparse work over a pass, then run a zero-bubble dense
    pipeline against it.

    A dataloader batch covers the whole pass. Each local layer runs its sparse
    half once and its output is cut into detached per-microbatch seams; the dense
    pipeline consumes those seams with its backward split into an input half
    (``I``, sent upstream immediately) and a weight half (``W``, deferred); then
    the seams' gradients resume each whole-pass sparse graph. See
    :meth:`_dense_zero_bubble` for the schedule.

    This is the only dense schedule Rail runs -- use :class:`Maglev1F1B` for the
    monolithic one.

    Args:
        stage (StageWrapper): This rank's pipeline stage, built with
            ``enable_rail=True``.
        optimizer (torch.optim.Optimizer): Optimizer stepped once per pass.
        num_microbatches (int): Number of dense microbatches. Must be at least
            ``stage.num_stages``, and must divide the whole-pass batch evenly.
        no_sync (Callable, optional): Gradient synchronization suppression
            context. Default: ``None``
        w_lag (int, optional): How many microbatches' weight work may stay
            outstanding. Defaults to the stage's own index -- the ZB1P rule, so
            the deepest stage defers longest.

    Requirements this schedule imposes on the model, none of which
    :class:`Maglev1F1B` does:

    * every layer implements ``split_dense_input()``, including one with no
      sparse half;
    * every trainable parameter belongs to a layer's ``sparse`` or ``dense``
      half -- one on the layer body would never receive a gradient;
    * ``MaglevModuleList.get_batch_size()`` is overridden, called every pass;
    * a layer's out-activation is not a view, because the split backward detaches
      its roots in place.

    Data parallelism is FSDP2: ``fully_shard`` for a sharded dense net,
    ``replicate`` for a replicated one. Both produce an ``FSDPModule``, so both
    take the same path. Raw ``DistributedDataParallel`` does not work with a
    zero-bubble schedule anywhere -- upstream skips the combination, see
    pytorch/pytorch#144530 -- because DDP's reducer is driven only by autograd
    hooks and the deferred weight half runs none.

    Raises:
        ValueError: if the stage is not Rail-enabled, or the microbatch count is
            invalid.
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        w_lag: Optional[int] = None,
    ) -> None:
        if not stage.rail_enabled:
            raise ValueError("MaglevRail requires StageWrapper(..., enable_rail=True)")
        super().__init__(stage, optimizer, no_sync)
        if num_microbatches < stage.num_stages:
            raise ValueError(
                "Rail needs at least one microbatch per stage: got "
                f"{num_microbatches} for {stage.num_stages} stages"
            )
        if w_lag is not None and w_lag < 0:
            raise ValueError(f"w_lag must not be negative, got {w_lag}")
        unsupported = sorted(
            {
                type(module).__name__
                for module in self._no_sync_modules
                if not isinstance(module, FSDPModule)
            }
        )
        if unsupported:
            # Any wrapper that is not FSDP2, not just DDP. _no_sync_modules
            # collects anything exposing no_sync -- DDP, FSDP1, a custom
            # wrapper -- and the pass suppresses all of them on every forward,
            # but _reduce_gradients only drives FSDP2. Whatever is left would
            # train on local gradients forever without a word. DDP in
            # particular cannot be driven by hand at all: its reducer runs off
            # autograd hooks, and the split backward fires none.
            raise ValueError(
                f"MaglevRail cannot reduce gradients for {', '.join(unsupported)}: "
                "it drives FSDP2's post-backward by hand, and no other wrapper "
                "exposes an equivalent entry point. Upstream skips "
                "DistributedDataParallel with a zero-bubble schedule for the "
                "same reason (pytorch/pytorch#144530). Use replicate() for a "
                "replicated dense net, or fully_shard() for a sharded one"
            )
        self._refresh_fsdp_modules()
        self.num_microbatches: int = num_microbatches
        self.num_warmup: int = min(
            stage.num_stages - stage.stage_index - 1,
            num_microbatches,
        )
        self.num_steady: int = num_microbatches - self.num_warmup
        # The ZB1P rule: W trails I by the stage's own index, so the deepest
        # stage defers longest and the gradient wave returns fastest.
        self.w_lag: int = stage.stage_index if w_lag is None else w_lag

    @property
    def microbatches_per_pass(self) -> int:
        return self.num_microbatches

    @property
    def takes_whole_pass_input(self) -> bool:
        r"""takes_whole_pass_input -> bool

        Return ``True`` because each input batch contains the entire Rail pass.
        """
        return True

    def _dense_zero_bubble(self, state: MaglevRailPassState) -> None:
        """The dense pipeline with the backward split into ``I`` and ``W``.

        ``I`` computes the gradient w.r.t. this stage's inputs and sends it
        upstream immediately; ``W`` computes the dense weight gradients, which
        gate nothing until the optimizer step, so they are deferred. Each hop of
        the returning wave then costs ``I`` rather than ``I + W``.

        For ``p=4, m=8`` with ``w_lag`` defaulting to the stage index
        (``..`` is idle)::

            s0 | F0 F1 F2 F3 .. .. .. I0 W0 F4 I1 W1 F5 I2 W2 F6 I3 W3 F7 I4 ...
            s1 | .. F0 F1 F2 .. .. I0 F3 I1 W0 F4 I2 W1 F5 I3 W2 F6 I4 W3 F7 I5 ...
            s2 | .. .. F0 F1 .. I0 F2 I1 F3 I2 W0 F4 I3 W1 F5 I4 W2 F6 I5 W3 F7 ...
            s3 | .. .. .. F0 I0 F1 I1 F2 I2 F3 I3 W0 F4 I4 W1 F5 I5 W2 F6 I6 W3 ...

        Each steady slot is one ``F`` then one ``I``, then whatever ``W`` the lag
        permits -- the loop below is ``_forward(); _backward_act()``.

        The deeper the stage, the longer it defers. The gain is not that ``W``
        fills a stage's own idle time -- ``s0`` defers nothing and still gains --
        it is that a deep stage running ``I`` alone forwards its gradient sooner.

        The gradient receive is posted immediately before the ``I`` that consumes
        it, never earlier. Posting it ahead of the weight work looks like free
        overlap and is not: a posted ``irecv`` is a spinning NCCL kernel holding
        SMs, so it competes with the ``W`` it was meant to hide behind. This is
        the same property that makes :class:`Maglev1F1BRecvAhead` slower than
        :class:`Maglev1F1B`. Do not move it without measuring.
        """
        stage = self.stage
        fwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            stage_input = state.dense_inputs[fwd_idx]
            stage.start_recv_act(stage_input)
            with self._no_sync():
                stage.dense_forward_micro(
                    stage_input,
                    fwd_idx,
                    state.seams_for(fwd_idx),
                )
            fwd_idx += 1

        def _drain_weight(keep: int) -> None:
            while stage.pending_weight_work > keep:
                stage.dense_backward_weight_micro()

        def _backward_act() -> None:
            stage.start_recv_grad()
            stage.dense_backward_act_micro()
            # I has already sent this stage's gradient upstream, so the wave is
            # moving before any of the weight work below runs.
            _drain_weight(self.w_lag)

        for _ in range(self.num_warmup):
            _forward()
        for _ in range(self.num_steady):
            _forward()
            _backward_act()
        for _ in range(self.num_warmup):
            _backward_act()
        # Every remaining W. Independent across stages, so this costs no pipeline
        # time; keeping it here rather than before the pass's last I is what holds
        # the deferred queue off the critical path.
        _drain_weight(0)

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        r"""progress(dataloader_iter) -> Optional[torch.Tensor]

        Run one whole-pass sparse phase, dense 1F1B phase, and sparse backward.

        Args:
            dataloader_iter (Iterator[Any]): Iterator over whole-pass batches.

        Returns:
            Optional[torch.Tensor]: Always ``None`` to avoid synchronizing on a
            microbatch loss.
        """
        stage = self.stage
        global_inputs = stage.take_global_inputs(dataloader_iter)
        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()

        try:
            with self._fsdp_accumulate():
                with self._no_sync():
                    state = stage.sparse_forward_global(
                        global_inputs,
                        self.num_microbatches,
                    )
                del global_inputs
                self._dense_zero_bubble(state)

                stage.sparse_backward_global(state)
                # Freed before the reduction and the step, which is where the
                # pass peaks: state still holds every layer's whole-pass pooled
                # output and all m seam gradients.
                del state
                stage.drain_sends()
        except BaseException:
            # A half-finished pass leaves the microbatch queues populated, and
            # the next progress() would pop this pass's entries against the next
            # pass's gradients.
            stage.reset_pass_state()
            raise
        # Outside the context, so a pass that raised does not reduce a partial
        # gradient -- and so the flags are already re-armed when we get here.
        self._reduce_gradients()
        with record_function("## torchrec_maglev:optimizer_step ##"):
            self.optimizer.step()
        return None

    def _refresh_fsdp_modules(self) -> None:
        """Re-read the FSDP2 modules and re-apply the policy they need.

        Re-read rather than trusted from construction because ``stage.module``
        is documented as reassignable: a caller who applies ``fully_shard``
        *after* building the schedule would otherwise get a pass that reduces
        nothing and says nothing.

        ``reshard_after_forward=False`` is required, and it fails silently if
        skipped. A post-forward reshard repoints the module at its sharded
        parameter, so the ``dense_parameters()`` that ``I`` hands to
        ``stage_backward_input`` no longer intersect the graph, which was built
        on the unsharded ones. ``get_param_groups`` then records *empty* weight
        groups and ``stage_backward_weight`` writes no gradient -- no exception,
        just a dense weight that never trains. Verified: with this off, a
        two-pass run ends with a parameter whose ``.grad`` is ``None``.

        It also saves all-gathers, which is the reason torchtitan cites for any
        pipeline schedule -- though the 8x measured against ``Maglev1F1B`` came
        from ``reshard_after_backward``, not this flag. Both are off for the
        pass, and the two cannot be separated by measurement here because
        ``reshard_after_forward=True`` does not produce a correct run to compare.
        """
        self._fsdp_modules = [
            module
            for module in self.stage.module.modules()
            if isinstance(module, FSDPModule)
        ]
        for module in self._fsdp_modules:
            # recurse=False because _fsdp_modules already lists every FSDP
            # module in the tree -- and because recursing would descend from a
            # fully_shard ancestor into a replicate descendant, whose DDPMeshInfo
            # trips set_reshard_after_forward's FSDPMeshInfo assertion.
            # Replicated groups have no post-forward mesh to reshard to, so they
            # are skipped rather than set.
            state = module._get_fsdp_state()
            if all(
                isinstance(group.mesh_info, FSDPMeshInfo)
                for group in state._fsdp_param_groups
            ):
                module.set_reshard_after_forward(False, recurse=False)

    @contextlib.contextmanager
    def _fsdp_accumulate(self) -> Iterator[None]:
        """Hold FSDP2's reduction back for the duration of the pass.

        The whole pass, not each microbatch: ``sparse_backward_global`` is a
        real backward that runs after the dense pipeline, so a dense-scoped
        scheme would let the sparse half reduce on its own while the dense half
        is still accumulating.

        Re-arms on exit whether or not the body raised, so a failed pass cannot
        leave the modules unable to reduce.
        """
        self._refresh_fsdp_modules()
        modules = self._fsdp_modules
        for module in modules:
            module.set_is_last_backward(False)
            module.set_reshard_after_backward(False, recurse=False)
            module.set_requires_gradient_sync(False, recurse=False)
        try:
            yield
        finally:
            for module in modules:
                module.set_is_last_backward(True)
                module.set_reshard_after_backward(True, recurse=False)
                module.set_requires_gradient_sync(True, recurse=False)

    def _reduce_gradients(self) -> None:
        """Drive FSDP2's post-backward by hand, once, for the whole pass.

        The split backward writes ``.grad`` through ``torch.autograd.grad``,
        which runs no ``AccumulateGrad`` node, so FSDP2's post-backward hook
        never fires on its own. This mirrors
        :meth:`~torch.distributed.pipelining.PipelineStage.perform_reduce_grad`,
        which upstream added for the same reason and schedules after a stage's
        last ``W``.

        A no-op when nothing is wrapped.
        """
        if not self._fsdp_modules:
            return
        with record_function("## torchrec_maglev:reduce_gradients ##"):
            # Set here rather than relying on _fsdp_accumulate's exit: the
            # finalize_backward that makes the compute stream wait on the
            # reduce-scatter events is gated on is_last_backward, so moving this
            # call inside the context would silently drop that wait.
            for module in self._fsdp_modules:
                module.set_is_last_backward(True)
                module.set_reshard_after_backward(True, recurse=False)
                module.set_requires_gradient_sync(True, recurse=False)
            # Deduplicated by state identity: fully_shard([a, b]) maps several
            # modules onto one FSDPState, so _get_fsdp_state() returns the same
            # object more than once. (Nested modules do not need this -- their
            # param groups are disjoint, and _validate_no_duplicate_params
            # raises if they ever overlap.)
            states = {}
            for module in self._fsdp_modules:
                state = module._get_fsdp_state()
                states.setdefault(id(state), state)
            for state in states.values():
                for param_group in state._fsdp_param_groups:
                    param_group.post_backward()
            # Strictly after every post_backward: foreach_reduce writes
            # sharded_param.grad on its own stream, and the only place the
            # compute stream waits on those events is finalize_backward,
            # reachable only from here. Without it optimizer.step() can read a
            # gradient whose kernel has not run.
            # Once per state *context*, not per state: the callback already
            # iterates state_ctx.all_states, and a second call would find every
            # group back at IDLE and re-enter post_backward on all of them.
            # Upstream's perform_reduce_grad calls it exactly once for the same
            # reason.
            contexts = {}
            for state in states.values():
                contexts.setdefault(id(state._state_ctx), state)
            for state in contexts.values():
                state._root_post_backward_final_callback()
