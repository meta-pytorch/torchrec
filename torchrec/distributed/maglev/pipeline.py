#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""The Maglev schedules: per-microbatch 1F1B, and MaglevRail on one pass.

* :class:`Maglev1F1B` -- each microbatch runs every local layer's sparse half
  and then its dense half, in 1F1B order; :class:`Maglev1F1BRecvAhead` posts
  each receive ahead of its compute. :class:`MaglevPipelineBase` is the
  non-interleaved schedule they extend.
* :class:`MaglevRail` runs one pass: each sparse half once on the pass's
  whole batch, the dense halves pipelined over per-microbatch seams in 1F1B
  (``dense_schedule="1f1b"``, the default) or zero-bubble
  (``"zero_bubble"``) order, each sparse half's backward once, and one
  optimizer step.

Under DDP and ZeRO-2 FSDP2 every schedule reduces gradients once per pass;
ZeRO-3 FSDP2 reduce-scatters per microbatch
(:mod:`~torchrec.distributed.maglev.data_parallel`).
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, ContextManager, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.autograd.profiler import record_function
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.maglev.data_parallel import (
    make_pass_reduction,
    PassReduction,
    reject_model_trackers,
)
from torchrec.distributed.maglev.stage import (
    HandoffPGMode,
    MaglevPassState,
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
    inherit is the stage and optimizer, the boundary-contract check, and the
    data-parallel wrapper discovery.

    Args:
        stage: this rank's :class:`StageWrapper`. Everything about where this
            rank sits -- its stage, its position in that stage's HSD, the
            neighbouring ranks, the hand-off process groups -- is read off the
            wrapper rather than derived a second time here.
        optimizer: the stage's optimizer. Gradients accumulate across a pass and
            are applied with a single step.
        no_sync: returns a context that suppresses the DP wrapper's gradient
            sync. Defaults to suppressing whatever wrappers
            :func:`_no_sync_modules` finds on ``stage.module``, so an unwrapped
            stage needs nothing and a DDP/DMP one works unconfigured. Pass your
            own only if that derivation is wrong for your setup. DDP and FSDP2
            modules are discovered independently for final-backward handling.

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
        self._ddp_modules: List[DistributedDataParallel] = [
            module for module in modules if isinstance(module, DistributedDataParallel)
        ]
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


class Maglev1F1B(MaglevPipelineBase):
    """The 1F1B (one-forward-one-backward) schedule.

    Standard PipeDream-flush: each stage runs ``num_warmup`` forwards, then
    interleaves 1 forward / 1 backward in steady state, then drains the remaining
    backwards in cooldown. ``num_warmup = num_stages - stage_index - 1`` (clamped
    to the microbatch count), so deeper stages warm up less and the send/recv
    pairs line up across ranks.

    Each microbatch runs every local layer's sparse half, then its dense half,
    so a fused embedding optimizer steps once per microbatch backward. DDP and
    ZeRO-2 FSDP2 reduce once per pass; ZeRO-3 reduce-scatters per microbatch
    (:mod:`~torchrec.distributed.maglev.data_parallel`).

    Communication follows the stage's global handoff-PG configuration. ``SPLIT``
    keeps activation and gradient traffic on independent communicators and uses
    the established start/wait ordering. ``SHARED`` batches the opposing
    directions at each steady-state crossover with ``batch_isend_irecv``.

    Args:
        stage: this rank's :class:`StageWrapper`.
        optimizer: the stage's optimizer.
        num_microbatches: microbatches per pass. Fixed for the pipeline's life,
            so the schedule's shape is computed once here rather than per pass.
        no_sync: an extra context entered around every forward, beside the
            reduction's own ``no_sync``. Default: ``None``

    Raises:
        ValueError: if there are fewer microbatches than stages. That leaves the
            early stages with no steady phase at all, and it is the steady phase
            that seeds the gradient receives -- so those ranks would fail in
            cooldown while the later ones ran on, hanging the job
            asymmetrically. It is also a pointless schedule: the pipeline never
            fills, so it is all bubble. Also if the stage's data-parallel
            wrappers cannot be reduced as above (see
            :func:`~torchrec.distributed.maglev.data_parallel.make_pass_reduction`).
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
        self._extra_no_sync: Optional[Callable[[], ContextManager[None]]] = no_sync
        self._reduction: PassReduction = make_pass_reduction(
            stage.module, type(self).__name__
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

        Gradients are accumulated across all microbatches, then reduced once and
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
        reduction = self._reduction
        with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
            self.optimizer.zero_grad()
        with reduction.accumulate():
            self._run_microbatches(microbatch_inputs)
            stage.drain_sends()
        # Outside the context: a pass that raised reduces nothing.
        with record_function("## torchrec_maglev:reduce_gradients ##"):
            reduction.reduce()
        with record_function("## torchrec_maglev:optimizer_step ##"):
            self.optimizer.step()
        return None

    def close(self) -> None:
        """Release what the gradient reduction registered on the stage's modules."""
        self._reduction.close()

    @contextlib.contextmanager
    def _forward_no_sync(self) -> Iterator[None]:
        """Suppress every wrapper's gradient sync for one microbatch forward."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._reduction.no_sync())
            if self._extra_no_sync is not None:
                stack.enter_context(self._extra_no_sync())
            yield

    def _run_microbatches(self, microbatch_inputs: List[List[Any]]) -> None:
        """Run the pass's forwards and backwards in the handoff mode's order."""
        if self.stage.handoff_pg_mode is HandoffPGMode.SHARED:
            self._progress_shared(microbatch_inputs)
        else:
            self._progress_split(microbatch_inputs)

    def _progress_split(
        self,
        microbatch_inputs: List[List[Any]],
    ) -> None:
        """Run 1F1B with independent activation and gradient communicators."""
        stage = self.stage
        fwd_idx = 0

        def _forward() -> None:
            nonlocal fwd_idx
            stage.start_recv_act(microbatch_inputs[fwd_idx])
            with self._forward_no_sync():
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        def _backward() -> None:
            stage.start_recv_grad()
            stage.backward_micro()

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

        def _compute_forward(in_activations: tuple[torch.Tensor, ...]) -> Any:
            nonlocal fwd_idx
            with self._forward_no_sync():
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
            return stage.compute_backward_micro(grads)

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

    def _run_microbatches(self, microbatch_inputs: List[List[Any]]) -> None:
        """Run receive-ahead 1F1B in split mode.

        Shared mode uses :class:`Maglev1F1B`'s order because its opposing
        handoffs must be issued together on the shared communicator.
        """
        stage = self.stage
        if stage.handoff_pg_mode is HandoffPGMode.SHARED:
            super()._run_microbatches(microbatch_inputs)
            return

        fwd_idx = 0
        recv_idx = 0

        def _start_recv_act() -> None:
            nonlocal recv_idx
            stage.start_recv_act(microbatch_inputs[recv_idx])
            recv_idx += 1

        def _forward() -> None:
            nonlocal fwd_idx
            with self._forward_no_sync():
                stage.forward_micro(microbatch_inputs[fwd_idx], fwd_idx)
            fwd_idx += 1

        _start_recv_act()

        for _ in range(self.num_warmup):
            _start_recv_act()
            _forward()

        for i in range(self.num_steady):
            stage.start_recv_grad()
            _forward()
            if i < self.num_steady - 1:
                _start_recv_act()
            else:
                stage.start_recv_grad()
            stage.backward_micro()

        for i in range(self.num_warmup):
            if i < self.num_warmup - 1:
                stage.start_recv_grad()
            stage.backward_micro()


#: The dense orders :class:`MaglevRail` runs, by ``dense_schedule``.
_DENSE_SCHEDULES: tuple[str, ...] = ("1f1b", "zero_bubble")


class MaglevRail(MaglevPipelineBase):
    r"""MaglevRail(stage, optimizer, num_microbatches, no_sync=None, dense_schedule="1f1b", *, scale_grads=False)

    One Maglev pass. Each local layer's sparse half runs once on the whole batch,
    its output cut into detached per-microbatch seams; the dense halves are
    pipelined over the seams in the order ``dense_schedule`` names; each
    whole-pass sparse graph resumes once from its seams' summed gradients, so a
    fused embedding optimizer (TBE) steps once per pass on the global batch's
    gradient; and one optimizer step applies the pass.

    The dense orders, for ``p=4, m=8``:

    * ``"1f1b"`` (the default): 1F1B, each microbatch's backward whole::

        s0 | F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
        s1 | F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
        s2 | F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
        s3 | F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7

    * ``"zero_bubble"``: each backward is split into an input half (``I``,
      sent upstream immediately) and a weight half (``W``, deferred by
      :attr:`w_lag`, the stage's own index -- the ZB1P rule). Stage 0 has
      nobody waiting, so it runs each ``W`` right after its ``I`` (``..`` is
      idle)::

        s0 | F0 F1 F2 F3 .. .. .. I0 W0 F4 I1 W1 F5 I2 W2 F6 I3 W3 F7 I4 W4 ...
        s1 | .. F0 F1 F2 .. .. I0 F3 I1 W0 F4 I2 W1 F5 I3 W2 F6 I4 W3 F7 I5 ...
        s2 | .. .. F0 F1 .. I0 F2 I1 F3 I2 W0 F4 I3 W1 F5 I4 W2 F6 I5 W3 F7 ...
        s3 | .. .. .. F0 I0 F1 I1 F2 I2 F3 I3 W0 F4 I4 W1 F5 I5 W2 F6 I6 W3 ...

      Every remaining ``W`` runs after the pass's last ``I``. Under DDP, ``W``
      runs through each weight's ``AccumulateGrad`` so the reducer's
      used-parameter map sees it; upstream skips DDP with a zero-bubble
      schedule for want of exactly that (pytorch/pytorch#144530).

    Data parallelism (:mod:`~torchrec.distributed.maglev.data_parallel`): DDP
    and ZeRO-2 ``fully_shard`` reduce once per pass; ZeRO-3 reduce-scatters per
    microbatch, under ``"1f1b"`` only.

    Every layer implements ``split_dense_input()`` and keeps its trainable
    parameters in its halves, and ``MaglevModuleList.get_batch_size()`` is
    overridden. Under FSDP2, every trainable dense parameter must be reached on
    every microbatch unless every module set
    ``set_reduce_scatter_unused_params(True)``.

    Args:
        stage (StageWrapper): This rank's pipeline stage, wrapped as the caller
            wants it.
        optimizer (torch.optim.Optimizer): Optimizer stepped once per pass.
        num_microbatches (int): Number of dense microbatches; at least
            ``stage.num_stages``, and dividing the whole-pass batch evenly.
        no_sync (Callable, optional): An extra context entered around every
            forward. Default: ``None``
        dense_schedule (str): ``"1f1b"`` or ``"zero_bubble"``. Default: ``"1f1b"``
        scale_grads (bool): Seed each microbatch loss's backward with
            ``1 / num_microbatches``, training on the mean over the pass.
            Default: ``False``, which sums them.

    Raises:
        ValueError: On an unknown ``dense_schedule``, ``"zero_bubble"`` with a
            ZeRO-3 module, a weight outside the layers' halves, an invalid
            microbatch count or boundary activation, or data-parallel wrappers
            that cannot be reduced.
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        dense_schedule: str = "1f1b",
        *,
        scale_grads: bool = False,
    ) -> None:
        name = type(self).__name__
        if dense_schedule not in _DENSE_SCHEDULES:
            raise ValueError(
                f"{name} dense_schedule must be one of {list(_DENSE_SCHEDULES)}, "
                f"got {dense_schedule!r}"
            )
        stage.require_pass_model()
        if num_microbatches < stage.num_stages:
            raise ValueError(
                f"{name} needs at least one microbatch per stage: got "
                f"{num_microbatches} for {stage.num_stages} stages"
            )
        super().__init__(stage, optimizer, no_sync)
        self.dense_schedule: str = dense_schedule
        self.num_microbatches: int = num_microbatches
        self.num_warmup: int = min(
            stage.num_stages - stage.stage_index - 1, num_microbatches
        )
        # How many weight halves a stage defers: its own index under
        # zero-bubble; 0 under 1F1B, whose backwards run whole.
        self.w_lag: int = stage.stage_index if dense_schedule == "zero_bubble" else 0
        self._extra_no_sync: Optional[Callable[[], ContextManager[None]]] = no_sync
        self._reduction: PassReduction = make_pass_reduction(
            stage.module,
            name,
            layers_unwrappable=stage.only_data_parallel_wraps_layers(),
        )
        resharded = self._reduction.resharded_modules
        if dense_schedule == "zero_bubble" and resharded:
            # On every stage: a refusal on some stages only would leave the
            # others blocked in the first pass.
            raise ValueError(
                f"{name}(dense_schedule='zero_bubble') cannot split the backward "
                "of an FSDP2 module that reshards after forward (ZeRO-3): "
                f"{sorted({type(m).__name__ for m in resharded})}. The weight "
                "half would find the module repointed at its sharded parameters "
                "and train nothing. Wrap with fully_shard(..., "
                "reshard_after_forward=False) (ZeRO-2), or use "
                "dense_schedule='1f1b'"
            )
        # DMP can gain a model tracker after this (init_torchrec_delta_tracker),
        # so every pass checks again.
        self._tracker_owners: List[nn.Module] = [
            m for m in stage.module.modules() if hasattr(m, "model_trackers")
        ]
        self._loss_scale: float = 1.0 / num_microbatches if scale_grads else 1.0
        self._configure_stage()

    def _configure_stage(self) -> None:
        """Set this schedule's loss scale and unreached-weight policy on the stage."""
        reject_model_trackers(self._tracker_owners, type(self).__name__)
        self.stage.configure_pass(
            self,
            loss_scale=self._loss_scale,
            allow_unreached=self._reduction.allows_unreached_dense,
        )

    @property
    def microbatches_per_pass(self) -> int:
        return self.num_microbatches

    @property
    def takes_whole_pass_input(self) -> bool:
        r"""takes_whole_pass_input -> bool

        Return ``True`` because each input batch contains the entire pass.
        """
        return True

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        r"""progress(dataloader_iter) -> Optional[torch.Tensor]

        Run one pass and step the optimizer.

        Args:
            dataloader_iter (Iterator[Any]): Iterator over whole-pass batches.

        Returns:
            Optional[torch.Tensor]: On the last stage, the pass's microbatch
            losses stacked in order, detached and on the device; ``None`` on
            other stages.
        """
        stage = self.stage
        try:
            global_inputs = stage.take_global_inputs(dataloader_iter)
        except StopIteration:
            raise
        except BaseException:
            stage.abandon_pass()
            raise
        reduction = self._reduction
        with stage.pass_scope():
            self._configure_stage()
            with record_function("## torchrec_maglev:optimizer_zero_grad ##"):
                self.optimizer.zero_grad()
            with reduction.accumulate():
                with self._forward_no_sync():
                    state = stage.sparse_forward_global(
                        global_inputs,
                        self.num_microbatches,
                        unwrapped=reduction.sparse_forward_unwrapped,
                    )
                del global_inputs
                self._dense(state)
                stage.sparse_backward_global(state)
                # The pass peaks at the reduction and the step; free every
                # whole-pass sparse output and seam gradient before them.
                del state
                stage.drain_sends()
            with record_function("## torchrec_maglev:reduce_gradients ##"):
                reduction.reduce()
            with record_function("## torchrec_maglev:optimizer_step ##"):
                self.optimizer.step()
        losses = stage.take_pass_losses()
        return torch.stack(losses) if losses else None

    def close(self) -> None:
        r"""close() -> None

        Release what the gradient reduction registered on the stage's modules.
        """
        self._reduction.close()

    @contextlib.contextmanager
    def _forward_no_sync(self) -> Iterator[None]:
        """Suppress every wrapper's gradient sync for one forward."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._reduction.no_sync())
            if self._extra_no_sync is not None:
                stack.enter_context(self._extra_no_sync())
            yield

    def _dense(self, state: MaglevPassState) -> None:
        """Phase 2, the dense pipeline, in the order ``dense_schedule`` names."""
        stage = self.stage
        run_accumulate_grad = self._reduction.accumulates_weight_grads

        def drain_weight(keep: int) -> None:
            while stage.pending_weight_work > keep:
                stage.dense_backward_weight_micro(
                    run_accumulate_grad=run_accumulate_grad
                )

        def backward() -> None:
            stage.start_recv_grad()
            if self.dense_schedule == "1f1b":
                stage.dense_backward_micro()
                return
            stage.dense_backward_act_micro()
            drain_weight(self.w_lag)

        self._one_forward_one_backward(state, backward)
        drain_weight(0)

    def _dense_forward(self, state: MaglevPassState, fwd_idx: int) -> None:
        """``F``: microbatch ``fwd_idx``'s dense forward, its receive posted just before."""
        stage = self.stage
        stage_input = state.dense_inputs[fwd_idx]
        stage.start_recv_act(stage_input)
        with self._forward_no_sync():
            stage.dense_forward_micro(stage_input, fwd_idx, state.seams[fwd_idx])

    def _one_forward_one_backward(
        self, state: MaglevPassState, backward: Callable[[], None]
    ) -> None:
        """The dense forwards in 1F1B order around ``backward``.

        ``num_warmup`` forwards, then a forward-backward pair for each remaining
        microbatch, then the ``num_warmup`` backwards still owed. ``backward``
        posts its gradient receive immediately before the op that consumes it,
        never earlier: a posted ``irecv`` is a spinning NCCL kernel holding SMs,
        so it competes with the work it was meant to hide behind.
        """
        for fwd_idx in range(self.num_warmup):
            self._dense_forward(state, fwd_idx)
        for fwd_idx in range(self.num_warmup, self.num_microbatches):
            self._dense_forward(state, fwd_idx)
            backward()
        for _ in range(self.num_warmup):
            backward()
