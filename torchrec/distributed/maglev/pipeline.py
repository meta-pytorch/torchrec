#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""The Maglev schedules: per-microbatch 1F1B, and MaglevRail on one pass.

* :class:`Maglev1F1B` -- each microbatch runs every local layer's sparse net
  and then its dense net, in 1F1B order. :class:`MaglevPipelineBase` is the
  base every schedule extends.
* :class:`MaglevRail` is the pass the Rail schedules share: each sparse net
  once on the pass's whole batch, the dense nets pipelined over
  per-microbatch seams, each sparse net's backward once, and one optimizer
  step. :class:`MaglevRail1F1B` runs the dense nets in 1F1B order and
  :class:`MaglevRailZB` in zero-bubble order;
  :class:`~torchrec.distributed.maglev.customized_pipeline.MaglevRailCustom`
  takes the order as ops.

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
from torchrec.distributed.maglev.data_parallel import (
    make_pass_reduction,
    only_data_parallel_wraps,
    PassReduction,
    reject_model_trackers,
)
from torchrec.distributed.maglev.stage import MaglevPassState, StageWrapper


class MaglevPipelineBase:
    """The base every schedule extends; it is not a schedule itself.

    It gives every schedule the stage and optimizer, the boundary-contract
    check, the microbatch count and the 1F1B shape it fixes, the data-parallel
    reduction (:mod:`~torchrec.distributed.maglev.data_parallel`) and its
    :meth:`close`, the ``no_sync`` every forward enters
    (:meth:`_forward_no_sync`), and the 1F1B loop over the schedule's own
    forward and backward ops (:meth:`_run_one_forward_one_backward`). Subclasses
    override :meth:`progress` with their own ordering.

    Args:
        stage: this rank's :class:`StageWrapper`. Everything about where this
            rank sits -- its stage, its position in that stage's HSD, the
            neighbouring ranks, the hand-off process groups -- is read off the
            wrapper rather than derived a second time here.
        optimizer: the stage's optimizer. Gradients accumulate across a pass and
            are applied with a single step.
        no_sync: an extra context (``_extra_no_sync``) entered around every
            forward, beside the reduction's own ``no_sync``. The reduction finds
            the stage's DDP and FSDP2 wrappers itself, so an unwrapped stage
            needs nothing and a wrapped one works unconfigured. Default:
            ``None``
        num_microbatches: microbatches per pass; at least ``stage.num_stages``.
        scale_grads: seed each microbatch loss's backward with
            ``1 / num_microbatches``. Default: ``False``

    Raises:
        TypeError: if built directly rather than through a schedule such as
            :class:`Maglev1F1B`.
        ValueError: if a boundary stage declares an activation it cannot have (an
            incoming one on the first stage, or none on any later stage); if
            there are fewer microbatches than stages; or if the stage's
            data-parallel wrappers cannot be reduced (see
            :func:`~torchrec.distributed.maglev.data_parallel.make_pass_reduction`).
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        *,
        num_microbatches: int,
        scale_grads: bool = False,
    ) -> None:
        if type(self) is MaglevPipelineBase:
            raise TypeError(
                "MaglevPipelineBase is the base of the Maglev schedules, not a "
                "schedule: build Maglev1F1B, a MaglevRail subclass or "
                "MaglevRailCustom"
            )
        self.stage = stage
        self.optimizer = optimizer

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

        name = type(self).__name__
        if num_microbatches < stage.num_stages:
            raise ValueError(
                f"{name} needs at least one microbatch per stage: got "
                f"{num_microbatches} for {stage.num_stages} stages"
            )
        self.num_microbatches: int = num_microbatches
        self._extra_no_sync: Optional[Callable[[], ContextManager[None]]] = no_sync
        self._loss_scale: float = 1.0 / self.num_microbatches if scale_grads else 1.0
        self._reduction: PassReduction = make_pass_reduction(
            stage.module,
            name,
            layers_unwrappable=only_data_parallel_wraps(
                stage.module, stage.layer_chain
            ),
            sparse_net_buffers=stage.sparse_net_buffers(),
        )
        self.num_warmup: int = min(
            stage.num_stages - stage.stage_index - 1, self.num_microbatches
        )
        self.num_steady: int = self.num_microbatches - self.num_warmup

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
        return self.num_microbatches

    @property
    def takes_whole_pass_input(self) -> bool:
        r"""takes_whole_pass_input -> bool

        Return whether one dataloader batch contains the entire pass.
        """
        return False

    def close(self) -> None:
        """Release what the gradient reduction registered on the stage's modules."""
        self._reduction.close()

    def _take_inputs(self, dataloader_iter: Iterator[Any], n: int) -> List[List[Any]]:
        """Acquire this stage's inputs; adapters may override the input seam."""
        return self.stage.take_inputs(dataloader_iter, n)

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        """Run one pass and apply the gradients; every schedule overrides it.

        Args:
            dataloader_iter: yields raw batches -- whatever the model's
                :meth:`~torchrec.distributed.maglev.module.MaglevModuleList.preproc`
                consumes. The stage pulls from it, partitions each batch into
                per-layer inputs, and all-to-alls them over its cascade, so a
                rank never has to know which layers it owns to feed it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define progress")

    @contextlib.contextmanager
    def _forward_no_sync(self) -> Iterator[None]:
        """Suppress every wrapper's gradient sync for one forward."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._reduction.no_sync())
            if self._extra_no_sync is not None:
                stack.enter_context(self._extra_no_sync())
            yield

    def _run_one_forward_one_backward(
        self, forward: Callable[[int], None], backward: Callable[[], None]
    ) -> None:
        """``forward`` and ``backward`` in 1F1B order.

        ``num_warmup`` forwards, then a forward-backward pair for each remaining
        microbatch, then the ``num_warmup`` backwards still owed. ``backward``
        posts its gradient receive immediately before the op that consumes it,
        never earlier: a posted ``irecv`` is a spinning NCCL kernel holding SMs,
        so it competes with the work it was meant to hide behind.
        """
        for fwd_idx in range(self.num_warmup):
            forward(fwd_idx)
        for fwd_idx in range(self.num_warmup, self.num_microbatches):
            forward(fwd_idx)
            backward()
        for _ in range(self.num_warmup):
            backward()


class Maglev1F1B(MaglevPipelineBase):
    """The 1F1B (one-forward-one-backward) schedule.

    Standard PipeDream-flush: each stage runs ``num_warmup`` forwards, then
    interleaves 1 forward / 1 backward in steady state, then drains the remaining
    backwards in cooldown. ``num_warmup = num_stages - stage_index - 1`` (clamped
    to the microbatch count), so deeper stages warm up less and the send/recv
    pairs line up across ranks.

    Each microbatch runs on the stage's phases: every local layer's sparse net
    (:meth:`~StageWrapper.sparse_forward_micro`), handing the dense nets
    detached seams, then the dense nets
    (:meth:`~StageWrapper.dense_forward_micro`); its backward runs the dense
    nets down to the seams (:meth:`~StageWrapper.dense_backward_micro`,
    which sends the input gradient upstream), then the sparse nets from the
    seams' gradients (:meth:`~StageWrapper.sparse_backward_micro`). A fused
    embedding optimizer therefore steps once per microbatch backward. DDP and
    ZeRO-2 FSDP2 reduce once per pass; ZeRO-3 reduce-scatters per microbatch
    (:mod:`~torchrec.distributed.maglev.data_parallel`).

    Args:
        stage: this rank's :class:`StageWrapper`.
        optimizer: the stage's optimizer.
        num_microbatches: microbatches per pass. Fixed for the pipeline's life,
            so the schedule's shape is computed once here rather than per pass.
        no_sync: an extra context entered around every forward, beside the
            reduction's own ``no_sync``. Default: ``None``
        scale_grads: seed each microbatch loss's backward with
            ``1 / num_microbatches``, so a ``postproc`` returning its
            microbatch's mean loss trains on the mean over the pass's whole
            batch. A fused embedding optimizer still steps once per microbatch
            backward, each step on ``1 / num_microbatches`` of that
            microbatch's mean. The seed takes the loss's dtype, so a bf16 loss
            rounds ``1 / num_microbatches`` when it is not a power of two.
            Default: ``False``, which sums the microbatch losses.

    Raises:
        ValueError: if there are fewer microbatches than stages. That leaves the
            early stages with no steady phase at all, and it is the steady phase
            that seeds the gradient receives -- so those ranks would fail in
            cooldown while the later ones ran on, hanging the job
            asymmetrically. It is also a pointless schedule: the pipeline never
            fills, so it is all bubble. Also if the stage's data-parallel
            wrappers cannot be reduced as above (see
            :func:`~torchrec.distributed.maglev.data_parallel.make_pass_reduction`),
            or the stage holds a model tracker.
        RuntimeError: from :meth:`progress`, after a pass on the stage failed:
            the handoffs are out of step across ranks.
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        *,
        scale_grads: bool = False,
    ) -> None:
        super().__init__(
            stage,
            optimizer,
            no_sync,
            num_microbatches=num_microbatches,
            scale_grads=scale_grads,
        )
        # DMP can gain a model tracker after this (init_torchrec_delta_tracker),
        # so every pass checks again.
        self._tracker_owners: List[nn.Module] = [
            m for m in stage.module.modules() if hasattr(m, "model_trackers")
        ]
        self._configure_stage()

    def _configure_stage(self) -> None:
        """Refuse a model tracker, or a stage module rewrapped since the schedule
        was built, and set the stage's backwards for this schedule: the loss
        scale (``1 / num_microbatches`` under ``scale_grads``, else 1), no
        unreached-weight refusal, and this schedule's reduction."""
        name = type(self).__name__
        reject_model_trackers(self._tracker_owners, name)
        self._reduction.reject_changed_wrappers(self.stage.module, name)
        self.stage.configure_pass(
            self,
            loss_scale=self._loss_scale,
            allow_unreached=True,
            reduction=self._reduction,
        )

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
            be measuring.
        """
        stage = self.stage
        try:
            microbatch_inputs = self._take_inputs(
                dataloader_iter, self.num_microbatches
            )
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
                self._run_microbatches(microbatch_inputs)
                stage.drain_sends()
            # Outside the context: a pass that raised reduces nothing.
            with record_function("## torchrec_maglev:reduce_gradients ##"):
                reduction.reduce()
            with record_function("## torchrec_maglev:optimizer_step ##"):
                self.optimizer.step()
        # The last stage's dense backwards kept each microbatch loss.
        stage.take_pass_losses()
        return None

    def _run_microbatches(self, microbatch_inputs: List[List[Any]]) -> None:
        """Run the pass's forwards and backwards in 1F1B order."""
        stage = self.stage

        def forward(fwd_idx: int) -> None:
            stage.start_recv_act(microbatch_inputs[fwd_idx])
            self._forward_micro(microbatch_inputs[fwd_idx], fwd_idx)

        def backward() -> None:
            stage.start_recv_grad()
            self._backward_micro()

        self._run_one_forward_one_backward(forward, backward)

    def _sparse_forward_micro(
        self, stage_input: Sequence[Any], fwd_idx: int
    ) -> List[Any]:
        """Microbatch ``fwd_idx``'s sparse phase; the caller holds ``no_sync``."""
        return self.stage.sparse_forward_micro(
            stage_input,
            fwd_idx,
            unwrapped=self._reduction.may_unwrap_sparse_forward(),
        )

    def _forward_micro(self, stage_input: Sequence[Any], fwd_idx: int) -> None:
        """``F``: the sparse phase, then the dense phase between the handoffs.
        The activation receive is already posted."""
        with self._forward_no_sync():
            seams = self._sparse_forward_micro(stage_input, fwd_idx)
            self.stage.dense_forward_micro(stage_input, fwd_idx, seams)

    def _backward_micro(self) -> None:
        """``B``: the dense backward, which sends the input gradient, then the
        sparse backward from the seams; the gradient receive is already posted."""
        self.stage.dense_backward_micro()
        self.stage.sparse_backward_micro()


class MaglevRail(MaglevPipelineBase):
    r"""MaglevRail(stage, optimizer, num_microbatches, no_sync=None, *, scale_grads=True)

    The pass every Rail schedule shares. Each local layer's sparse net runs
    once on the whole batch, its output cut into detached per-microbatch seams;
    the dense nets are pipelined over the seams in the order the subclass
    runs; each whole-pass sparse graph resumes once from its seams' summed
    gradients, so a fused embedding optimizer (TBE) steps once per pass on the
    global batch's gradient; and one optimizer step applies the pass.

    Construct a subclass, which supplies the dense order: :class:`MaglevRail1F1B`
    (1F1B, whole backwards), :class:`MaglevRailZB` (zero-bubble, split
    backwards) or
    :class:`~torchrec.distributed.maglev.customized_pipeline.MaglevRailCustom`
    (an order given as ops). A subclass implements ``_dense`` and sets
    ``_splits_backward`` if its order splits backwards into input-gradient
    and weight-gradient phases.

    Data parallelism (:mod:`~torchrec.distributed.maglev.data_parallel`): DDP
    and ZeRO-2 ``fully_shard`` reduce once per pass; ZeRO-3 reduce-scatters per
    microbatch, under whole backwards only.

    Every layer implements ``split_dense_input()`` and keeps its trainable
    parameters in its sparse and dense nets, and ``MaglevModuleList.get_batch_size()`` is
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
        scale_grads (bool): Seed each microbatch loss's backward with
            ``1 / num_microbatches``, training on the mean over the pass. The
            seed takes the loss's dtype. Default: ``True``; ``False`` sums
            them.

    Raises:
        TypeError: If the class supplies no dense order (``MaglevRail`` itself,
            or a subclass that does not implement ``_dense``).
        ValueError: On a split-backward order with a ZeRO-3 module, a weight
            outside every layer, an invalid microbatch count or boundary
            activation, or data-parallel wrappers that cannot be reduced.
    """

    #: Whether the dense order splits backwards into input-gradient and
    #: weight-gradient phases, which a ZeRO-3 module cannot run.
    _splits_backward: bool = False

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        *,
        scale_grads: bool = True,
    ) -> None:
        name = type(self).__name__
        if type(self)._dense is MaglevRail._dense:
            raise TypeError(
                f"{name} runs no dense order: construct MaglevRail1F1B, "
                "MaglevRailZB or MaglevRailCustom, or override _dense"
            )
        stage.reject_untrained_parameters()
        super().__init__(
            stage,
            optimizer,
            no_sync,
            num_microbatches=num_microbatches,
            scale_grads=scale_grads,
        )
        resharded = self._reduction.resharded_modules
        if self._splits_backward and resharded:
            # On every stage: a refusal on some stages only would leave the
            # others blocked in the first pass.
            raise ValueError(
                f"{name} cannot split the backward "
                "of an FSDP2 module that reshards after forward (ZeRO-3): "
                f"{sorted({type(m).__name__ for m in resharded})}. The weight "
                "phase would find the module repointed at its sharded parameters "
                "and train nothing. Wrap with fully_shard(..., "
                "reshard_after_forward=False) (ZeRO-2), or use "
                "MaglevRail1F1B"
            )
        # DMP can gain a model tracker after this (init_torchrec_delta_tracker),
        # so every pass checks again.
        self._tracker_owners: List[nn.Module] = [
            m for m in stage.module.modules() if hasattr(m, "model_trackers")
        ]
        self._configure_stage()

    def _configure_stage(self) -> None:
        """Refuse a model tracker, or a stage module rewrapped since the schedule
        was built, and set this schedule's loss scale, unreached-weight policy
        and reduction on the stage."""
        name = type(self).__name__
        reject_model_trackers(self._tracker_owners, name)
        self._reduction.reject_changed_wrappers(self.stage.module, name)
        self.stage.configure_pass(
            self,
            loss_scale=self._loss_scale,
            allow_unreached=self._reduction.allows_unreached_dense,
            reduction=self._reduction,
        )

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
            (global_inputs,) = self._take_inputs(dataloader_iter, 1)
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
                        unwrapped=reduction.may_unwrap_sparse_forward(),
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

    def _dense(self, state: MaglevPassState) -> None:
        """Phase 2, the dense pipeline, in the subclass's order."""
        raise NotImplementedError(f"{type(self).__name__} runs no dense order")

    def _dense_forward_micro(self, state: MaglevPassState, fwd_idx: int) -> None:
        """``F``: microbatch ``fwd_idx``'s dense forward, its receive posted just before."""
        stage = self.stage
        stage_input = state.dense_inputs[fwd_idx]
        stage.start_recv_act(stage_input)
        with self._forward_no_sync():
            stage.dense_forward_micro(stage_input, fwd_idx, state.seams[fwd_idx])


class MaglevRail1F1B(MaglevRail):
    r"""MaglevRail1F1B(stage, optimizer, num_microbatches, no_sync=None, *, scale_grads=True)

    :class:`MaglevRail` with the dense nets in 1F1B order, each microbatch's
    backward whole. For ``p=4, m=8``::

        s0 | F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
        s1 | F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
        s2 | F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
        s3 | F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7

    Runs under DDP, ZeRO-2 and ZeRO-3. Arguments and errors are
    :class:`MaglevRail`'s.
    """

    def _dense(self, state: MaglevPassState) -> None:
        """Phase 2: 1F1B, each backward whole."""
        stage = self.stage

        def backward() -> None:
            stage.start_recv_grad()
            stage.dense_backward_micro()

        self._run_one_forward_one_backward(
            lambda fwd_idx: self._dense_forward_micro(state, fwd_idx), backward
        )


class MaglevRailZB(MaglevRail):
    r"""MaglevRailZB(stage, optimizer, num_microbatches, no_sync=None, *, scale_grads=True)

    :class:`MaglevRail` with the dense nets in zero-bubble (ZB1P) order. Each
    backward is split into an input-gradient phase (``I``, sent upstream
    immediately) and a weight-gradient phase (``W``, deferred by
    :attr:`weight_backward_lag`, the stage's own index).
    Stage 0 has nobody waiting, so it runs each ``W`` right after its ``I``
    (``..`` is idle). For ``p=4, m=8``::

        s0 | F0 F1 F2 F3 .. .. .. I0 W0 F4 I1 W1 F5 I2 W2 F6 I3 W3 F7 I4 W4 ...
        s1 | .. F0 F1 F2 .. .. I0 F3 I1 W0 F4 I2 W1 F5 I3 W2 F6 I4 W3 F7 I5 ...
        s2 | .. .. F0 F1 .. I0 F2 I1 F3 I2 W0 F4 I3 W1 F5 I4 W2 F6 I5 W3 F7 ...
        s3 | .. .. .. F0 I0 F1 I1 F2 I2 F3 I3 W0 F4 I4 W1 F5 I5 W2 F6 I6 W3 ...

    Every remaining ``W`` runs after the pass's last ``I``. Under DDP, ``W``
    runs through each weight's ``AccumulateGrad`` so the reducer's
    used-parameter map sees it; upstream skips DDP with a zero-bubble schedule
    for want of exactly that (pytorch/pytorch#144530).

    Runs under DDP and ZeRO-2; refuses a ZeRO-3 module at construction (the
    split-backward check in :class:`MaglevRail`). Arguments and other errors are
    :class:`MaglevRail`'s.
    """

    _splits_backward: bool = True

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
        *,
        scale_grads: bool = True,
    ) -> None:
        super().__init__(
            stage, optimizer, num_microbatches, no_sync, scale_grads=scale_grads
        )
        #: How many weight-gradient phases this stage defers: its own index.
        self.weight_backward_lag: int = stage.stage_index

    def _dense(self, state: MaglevPassState) -> None:
        """Phase 2: zero-bubble, each backward split, its ``W`` deferred."""
        stage = self.stage

        def drain_weight(keep: int) -> None:
            while stage.pending_weight_work > keep:
                stage.dense_backward_weight_micro()

        def backward() -> None:
            stage.start_recv_grad()
            stage.dense_backward_act_micro()
            drain_weight(self.weight_backward_lag)

        self._run_one_forward_one_backward(
            lambda fwd_idx: self._dense_forward_micro(state, fwd_idx), backward
        )
        drain_weight(0)
