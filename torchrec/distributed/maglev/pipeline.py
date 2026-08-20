#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function
from torchrec.distributed.maglev.stage import build_handoff_process_groups, StageWrapper

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MaglevPipeline:
    """Drives a staged Maglev model across per-stage HSDs.

    This is the pipeline-parallel analogue of the sequential loop in
    ``WukongMaglev.forward`` (legokit/backbones/maglev_prototype.py): instead of
    iterating stages in one process, each stage lives on a disjoint set of ranks
    (its HSD) and the activation is handed between HSDs over the network.

    A rank owns exactly one stage. The hand-off is **by position**: the rank at
    position ``p`` in HSD ``i`` exchanges with the rank at position ``p`` in HSD
    ``i ± 1``. Forward activations flow ``i -> i+1``; backward gradients flow
    ``i+1 -> i``. The two ranks of an HSD are data-parallel lanes; gradients are
    averaged across them (:meth:`StageWrapper.reduce_gradients`).

    The hand-off runs over **two direction-split process groups** (``act_pg`` for
    forward activations, ``grad_pg`` for backward gradients, built by
    :func:`build_handoff_process_groups`) rather than P2P over the default (world)
    group. A boundary's forward activation and backward gradient thus land on
    *different* communicators (separate NCCL streams instead of serializing), and
    each communicator carries a single flow direction (recv-then-send per rank,
    never send-first). Both are separate from each stage's intra-HSD sharded
    all-to-all; interleaving P2P with that on the world group deadlocks once the
    stages are sharded with DMP. See
    ``tech-docs/nccl_p2p_execution_order_buffer_size.md`` (sections 4-5).

    Three drivers are provided:

    * :meth:`step` -- a single-batch forward-then-backward pass (used by the
      correctness test). Blocking send/recv form a matched wave; no deadlock.
    * :meth:`forward_micro` / :meth:`backward_micro` + :func:`run_1f1b` -- a
      microbatched 1F1B schedule (used by the benchmark). Activations are
      received with blocking recv (strict dependency, forms a wave) and sent
      with non-blocking isend whose completion is deferred to
      :meth:`wait_sends`, which keeps the schedule deadlock-free.
    * :meth:`forward_sparse_micro` / :meth:`forward_dense_micro` +
      :func:`run_1f1b_split` -- the split-forward variant of 1F1B. Every
      microbatch's sparse (embedding) forward completes before the first
      backward, so all sparse lookups see the same pre-step embedding weights.
      Warmup runs an initial burst of ``prehoist`` sparses, then each warmup
      dense is followed by the prefetch of the sparse it would otherwise
      "leave for later" -- so every dense in warmup is adjacent to one of the
      trailing sparses. Requires the stage module to satisfy
      :class:`~torchrec.distributed.maglev.stage.MaglevStage`.

    Every comm (activation/gradient send+recv) and the forward/backward compute
    is wrapped in a ``record_function`` range tagged with the microbatch id
    (``## recv_act mb3 ##`` etc.), so profiler traces show which microbatch each
    operation belongs to.

    Args:
        stage: this rank's :class:`StageWrapper`.
        stage_ranks: ``stage_ranks[i]`` is the global ranks of stage ``i``'s HSD.
            Every stage must have the same number of ranks (same position layout).
        global_rank: this process's global rank.
        activation_shape: shape of the activation handed between stages.
        device: device the stage runs on.
        dtype: dtype of the activation / gradient tensors.
    """

    def __init__(
        self,
        stage: StageWrapper,
        stage_ranks: List[List[int]],
        global_rank: int,
        activation_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        handoff_pgs: Optional[Tuple[dist.ProcessGroup, dist.ProcessGroup]] = None,
    ) -> None:
        self.stage = stage
        self.stage_ranks = stage_ranks
        self.global_rank = global_rank
        self.num_stages: int = len(stage_ranks)
        self.activation_shape = activation_shape
        self.device = device
        self.dtype = dtype

        # Locate this rank's stage and its position within that stage's HSD.
        stage_index = -1
        position = -1
        for s_idx, ranks in enumerate(stage_ranks):
            if global_rank in ranks:
                stage_index = s_idx
                position = ranks.index(global_rank)
                break
        if stage_index < 0:
            raise ValueError(
                f"rank {global_rank} not found in stage_ranks {stage_ranks}"
            )
        if stage_index != stage.stage_index:
            raise ValueError(
                f"rank {global_rank} owns stage {stage_index} but the StageWrapper "
                f"is for stage {stage.stage_index}"
            )
        self.stage_index: int = stage_index
        self.position: int = position

        # Two direction-split P2P communicators (see build_handoff_process_groups):
        # forward activations go on ``act_pg``, backward gradients on ``grad_pg``,
        # so a boundary's two directions land on different communicators (separate
        # streams) and each carries a single flow direction (recv-then-send, never
        # send-first). When sharding, pass ``handoff_pgs`` built *before* any DMP so
        # the two ``new_group`` collectives stay contiguous; otherwise built here.
        if handoff_pgs is None:
            handoff_pgs = build_handoff_process_groups(stage_ranks)
        self._act_pg, self._grad_pg = handoff_pgs

        # Per-direction pg for this stage: activations on act_pg, gradients on
        # grad_pg (None where this stage has no neighbor on that side).
        self._recv_act_pg: Optional[dist.ProcessGroup] = (
            None if self.is_first else self._act_pg
        )
        self._send_act_pg: Optional[dist.ProcessGroup] = (
            None if self.is_last else self._act_pg
        )
        self._recv_grad_pg: Optional[dist.ProcessGroup] = (
            None if self.is_last else self._grad_pg
        )
        self._send_grad_pg: Optional[dist.ProcessGroup] = (
            None if self.is_first else self._grad_pg
        )

        # 1F1B state: FIFO of in-flight microbatches (with their microbatch id,
        # for profiler labels) and deferred send handles.
        self._pending: List[
            Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], int]
        ] = []
        # pyre-ignore[4]: dist work handle has no public type
        self._send_work: List[Tuple[Any, torch.Tensor]] = []
        # Split-forward cache: sparse (pooled) output per hoisted microbatch,
        # keyed by microbatch id. Populated by ``forward_sparse_micro`` and
        # consumed (popped) by the matching ``forward_dense_micro``. Empty
        # between :func:`run_1f1b_split` calls.
        self._pooled: Dict[int, torch.Tensor] = {}

    @property
    def is_first(self) -> bool:
        return self.stage_index == 0

    @property
    def is_last(self) -> bool:
        return self.stage_index == self.num_stages - 1

    def _prev_rank(self) -> int:
        """Global rank at this position in the previous HSD."""
        return self.stage_ranks[self.stage_index - 1][self.position]

    def _next_rank(self) -> int:
        """Global rank at this position in the next HSD."""
        return self.stage_ranks[self.stage_index + 1][self.position]

    def _recv(
        self, src: int, pg: dist.ProcessGroup, requires_grad: bool
    ) -> torch.Tensor:
        tensor = torch.empty(
            self.activation_shape, device=self.device, dtype=self.dtype
        )
        dist.recv(tensor, src=src, group=pg)
        if requires_grad:
            # Make it a leaf that requires grad so backward yields its grad to
            # hand back to the previous stage.
            tensor.requires_grad_(True)
        return tensor

    def _isend(self, tensor: torch.Tensor, dst: int, pg: dist.ProcessGroup) -> None:
        # Non-blocking send; the buffer is kept alive until wait_sends().
        buf = tensor.detach().contiguous()
        work = dist.isend(buf, dst=dst, group=pg)
        self._send_work.append((work, buf))

    def wait_sends(self) -> None:
        """Complete all deferred non-blocking sends."""
        for work, _buf in self._send_work:
            work.wait()
        self._send_work.clear()

    # ---- single-batch driver (correctness) ----

    def step(
        self,
        stage_input: Any,
        optimizer: torch.optim.Optimizer,
        label: Optional[torch.Tensor] = None,
        criterion: Optional[LossFn] = None,
    ) -> Optional[torch.Tensor]:
        """Run one forward + backward + optimizer step for a single batch.

        Returns the loss on the last stage, ``None`` on every other stage.
        ``label`` and ``criterion`` are required on (and only used by) the last
        stage.
        """
        optimizer.zero_grad()

        prev_output: Optional[torch.Tensor] = None
        if not self.is_first:
            assert self._recv_act_pg is not None
            with record_function("## recv_act mb0 ##"):
                prev_output = self._recv(
                    self._prev_rank(), self._recv_act_pg, requires_grad=True
                )

        with record_function("## forward mb0 ##"):
            output = self.stage(stage_input, prev_output)

        if not self.is_last:
            assert self._send_act_pg is not None
            with record_function("## send_act mb0 ##"):
                dist.send(
                    output.detach().contiguous(),
                    dst=self._next_rank(),
                    group=self._send_act_pg,
                )

        loss: Optional[torch.Tensor] = None
        if self.is_last:
            if criterion is None or label is None:
                raise ValueError("last stage requires both `criterion` and `label`")
            with record_function("## backward mb0 ##"):
                loss = criterion(output, label)
                loss.backward()
        else:
            assert self._recv_grad_pg is not None
            grad_output = torch.empty(
                output.shape, device=self.device, dtype=self.dtype
            )
            with record_function("## recv_grad mb0 ##"):
                dist.recv(grad_output, src=self._next_rank(), group=self._recv_grad_pg)
            with record_function("## backward mb0 ##"):
                output.backward(grad_output)

        if not self.is_first:
            assert prev_output is not None and prev_output.grad is not None
            assert self._send_grad_pg is not None
            with record_function("## send_grad mb0 ##"):
                dist.send(
                    prev_output.grad.contiguous(),
                    dst=self._prev_rank(),
                    group=self._send_grad_pg,
                )

        self.stage.reduce_gradients()
        optimizer.step()
        return loss

    # ---- microbatched 1F1B driver (benchmark) ----

    def forward_micro(
        self,
        stage_input: Any,
        label: Optional[torch.Tensor] = None,
        microbatch_id: int = 0,
    ) -> None:
        """One microbatch forward: recv activation, compute, send activation.

        ``microbatch_id`` tags the profiler ranges (and is carried on the pending
        FIFO to the matching backward) so a trace shows which microbatch each
        comm/compute belongs to.
        """
        prev_output: Optional[torch.Tensor] = None
        if not self.is_first:
            assert self._recv_act_pg is not None
            with record_function(f"## recv_act mb{microbatch_id} ##"):
                prev_output = self._recv(
                    self._prev_rank(), self._recv_act_pg, requires_grad=True
                )

        with record_function(f"## forward mb{microbatch_id} ##"):
            output = self.stage(stage_input, prev_output)

        if not self.is_last:
            assert self._send_act_pg is not None
            with record_function(f"## send_act mb{microbatch_id} ##"):
                self._isend(output, self._next_rank(), self._send_act_pg)

        self._pending.append((prev_output, output, label, microbatch_id))

    def backward_micro(
        self, criterion: Optional[LossFn] = None
    ) -> Optional[torch.Tensor]:
        """One microbatch backward: recv grad, backward, send input grad.

        Gradients accumulate across microbatches (no ``zero_grad`` here); the
        caller runs the DP all-reduce and optimizer step once per batch. Profiler
        ranges are tagged with the ``microbatch_id`` recorded at forward time.
        """
        prev_output, output, label, microbatch_id = self._pending.pop(0)

        loss: Optional[torch.Tensor] = None
        if self.is_last:
            if criterion is None or label is None:
                raise ValueError("last stage requires both `criterion` and `label`")
            with record_function(f"## backward mb{microbatch_id} ##"):
                loss = criterion(output, label)
                loss.backward()
        else:
            assert self._recv_grad_pg is not None
            grad_output = torch.empty(
                output.shape, device=self.device, dtype=self.dtype
            )
            with record_function(f"## recv_grad mb{microbatch_id} ##"):
                dist.recv(grad_output, src=self._next_rank(), group=self._recv_grad_pg)
            with record_function(f"## backward mb{microbatch_id} ##"):
                output.backward(grad_output)

        if not self.is_first:
            assert prev_output is not None and prev_output.grad is not None
            assert self._send_grad_pg is not None
            with record_function(f"## send_grad mb{microbatch_id} ##"):
                self._isend(prev_output.grad, self._prev_rank(), self._send_grad_pg)

        return loss

    # ---- split-forward 1F1B driver (SGD-exact w/ fused TBE) ----

    def forward_sparse_micro(self, stage_input: Any, microbatch_id: int) -> None:
        """Sparse-only forward: pool this stage's embeddings and cache the result.

        No cross-stage comm: the sparse forward depends only on ``stage_input``
        (per :class:`~torchrec.distributed.maglev.stage.MaglevStage`), so every
        microbatch's sparse pass can run up-front. The pooled tensor is stashed
        in ``_pooled[microbatch_id]`` for the matching
        :meth:`forward_dense_micro`; autograd retains its graph via that reference.

        Raises:
            AssertionError: if ``microbatch_id`` is already in ``_pooled`` (the
                caller invoked ``forward_sparse_micro`` twice for the same
                microbatch without a matching ``forward_dense_micro`` in
                between).
        """
        assert microbatch_id not in self._pooled, (
            f"forward_sparse_micro: microbatch {microbatch_id} already cached; "
            f"duplicate call without a matching forward_dense_micro"
        )
        with record_function(f"## forward_sparse mb{microbatch_id} ##"):
            pooled = self.stage.forward_sparse(stage_input)
        self._pooled[microbatch_id] = pooled

    def forward_dense_micro(
        self,
        stage_input: Any,
        label: Optional[torch.Tensor] = None,
        microbatch_id: int = 0,
    ) -> None:
        """Dense-only forward for a microbatch whose sparse was already hoisted.

        Mirrors :meth:`forward_micro` (same recv/send/pending semantics) but
        substitutes ``stage.forward_dense(stage_input, pooled, prev)`` for the
        monolithic ``stage(...)``. Consumes (pops) ``_pooled[microbatch_id]`` --
        must be preceded by a :meth:`forward_sparse_micro` call for the same
        microbatch id.

        Raises:
            RuntimeError: if ``_pooled[microbatch_id]`` is missing (no matching
                ``forward_sparse_micro`` was called for this microbatch).
        """
        prev_output: Optional[torch.Tensor] = None
        if not self.is_first:
            assert self._recv_act_pg is not None
            with record_function(f"## recv_act mb{microbatch_id} ##"):
                prev_output = self._recv(
                    self._prev_rank(), self._recv_act_pg, requires_grad=True
                )

        try:
            pooled = self._pooled.pop(microbatch_id)
        except KeyError:
            raise RuntimeError(
                f"forward_dense_micro({microbatch_id}): no cached pooled tensor; "
                f"call forward_sparse_micro({microbatch_id}) first"
            ) from None

        with record_function(f"## forward_dense mb{microbatch_id} ##"):
            output = self.stage.forward_dense(stage_input, pooled, prev_output)

        if not self.is_last:
            assert self._send_act_pg is not None
            with record_function(f"## send_act mb{microbatch_id} ##"):
                self._isend(output, self._next_rank(), self._send_act_pg)

        self._pending.append((prev_output, output, label, microbatch_id))


def run_1f1b(
    pipeline: MaglevPipeline,
    microbatch_inputs: List[Any],
    optimizer: torch.optim.Optimizer,
    labels: Optional[List[torch.Tensor]] = None,
    criterion: Optional[LossFn] = None,
) -> List[float]:
    """Run one 1F1B (one-forward-one-backward) pass over ``microbatch_inputs``.

    Standard PipeDream-flush schedule: each stage runs ``num_warmup`` forwards,
    then interleaves 1 forward / 1 backward in steady state, then drains the
    remaining backwards in cooldown. ``num_warmup = num_stages - stage_index - 1``
    (clamped to the microbatch count), so deeper stages warm up less and the
    send/recv pairs line up across ranks.

    Returns the per-microbatch losses observed on the last stage (empty on
    other stages). Gradients are accumulated across all microbatches, then
    DP-averaged and applied with a single optimizer step.
    """
    num_stages = pipeline.num_stages
    num_micro = len(microbatch_inputs)
    stage_index = pipeline.stage_index
    num_warmup = min(num_stages - stage_index - 1, num_micro)
    num_steady = num_micro - num_warmup

    def _label(idx: int) -> Optional[torch.Tensor]:
        return labels[idx] if labels is not None else None

    optimizer.zero_grad()
    losses: List[float] = []

    fwd_idx = 0
    bwd_idx = 0

    # Warmup: fill the pipeline.
    for _ in range(num_warmup):
        pipeline.forward_micro(microbatch_inputs[fwd_idx], _label(fwd_idx), fwd_idx)
        fwd_idx += 1

    # Steady state: one forward, one backward.
    for _ in range(num_steady):
        pipeline.forward_micro(microbatch_inputs[fwd_idx], _label(fwd_idx), fwd_idx)
        fwd_idx += 1
        loss = pipeline.backward_micro(criterion)
        bwd_idx += 1
        if loss is not None:
            losses.append(loss.item())

    # Cooldown: drain remaining backwards.
    for _ in range(num_warmup):
        loss = pipeline.backward_micro(criterion)
        bwd_idx += 1
        if loss is not None:
            losses.append(loss.item())

    pipeline.wait_sends()
    pipeline.stage.reduce_gradients()
    optimizer.step()
    return losses


def run_1f1b_split(
    pipeline: MaglevPipeline,
    microbatch_inputs: List[Any],
    optimizer: torch.optim.Optimizer,
    labels: Optional[List[torch.Tensor]] = None,
    criterion: Optional[LossFn] = None,
) -> List[float]:
    """Split-forward 1F1B with trailing-sparse interleave: SGD-exact under fused TBE.

    Schedule (``w = min(num_stages - stage_index - 1, N)``,
    ``prehoist = min(max(stage_index + 1, N - w), N)``); four phases, each
    wrapped in its own ``record_function`` for profiler visibility:

    * **sparse_hoist** -- ``S_j`` for ``j in [0, prehoist)``. Front-loads the
      sparses that will not be paired with a warmup dense. This is the visible
      cost of split-forward vs monolithic 1F1B.
    * **warmup** -- for ``j in [0, w)``, run ``D_j``; if ``j + prehoist < N``,
      prefetch ``S_{j + prehoist}`` immediately after ``D_j``. Every ``D_j``
      is adjacent to one of the *trailing* sparses. Symmetric with
      ``steady``'s (dense, backward) interleave.
    * **steady** -- for ``_ in [0, N - w)``: one dense forward, one backward.
      The dense forward pops the pooled tensor cached by its matching sparse.
    * **cooldown** -- for ``_ in [0, w)``: drain remaining backwards.

    By the end of ``warmup`` all ``N`` sparses have completed -- the first
    backward in ``steady`` can safely fire the fused TBE optimizer without
    invalidating any subsequent sparse lookup.

    Per-stage sizing (``Sj`` sparse, ``Dj`` dense, ``Bj`` backward, all for
    microbatch ``j``). Read each row across for the full stage timeline.

    **4 stages, N = 4 microbatches** (``prehoist == stage_index + 1``, floor
    dominates):

    ==== === ======== ================ ============================ ============================ ============
    stage w   prehoist sparse_hoist     warmup                       steady                       cooldown
    ==== === ======== ================ ============================ ============================ ============
     0    3    1       ``S0``           ``D0 S1 D1 S2 D2 S3``        ``D3 B0``                    ``B1 B2 B3``
     1    2    2       ``S0 S1``        ``D0 S2 D1 S3``              ``D2 B0 D3 B1``              ``B2 B3``
     2    1    3       ``S0 S1 S2``     ``D0 S3``                    ``D1 B0 D2 B1 D3 B2``        ``B3``
     3    0    4       ``S0 S1 S2 S3``  (empty)                      ``D0 B0 D1 B1 D2 B2 D3 B3``  (empty)
    ==== === ======== ================ ============================ ============================ ============

    **4 stages, N = 8 microbatches** (``prehoist == N - w``, ceiling
    dominates; the initial burst grows so warmup denses interleave with the
    *last* ``w`` sparses):

    ==== === ======== =========================== ===================== ================================================== ============
    stage w   prehoist sparse_hoist                warmup                steady                                             cooldown
    ==== === ======== =========================== ===================== ================================================== ============
     0    3    5       ``S0 S1 S2 S3 S4``          ``D0 S5 D1 S6 D2 S7`` ``D3 B0 D4 B1 D5 B2 D6 B3 D7 B4``                  ``B5 B6 B7``
     1    2    6       ``S0 S1 S2 S3 S4 S5``       ``D0 S6 D1 S7``       ``D2 B0 D3 B1 D4 B2 D5 B3 D6 B4 D7 B5``            ``B6 B7``
     2    1    7       ``S0 S1 S2 S3 S4 S5 S6``    ``D0 S7``             ``D1 B0 D2 B1 D3 B2 D4 B3 D5 B4 D6 B5 D7 B6``      ``B7``
     3    0    8       ``S0 S1 S2 S3 S4 S5 S6 S7`` (empty)               ``D0 B0 D1 B1 D2 B2 D3 B3 D4 B4 D5 B5 D6 B6 D7 B7`` (empty)
    ==== === ======== =========================== ===================== ================================================== ============

    All ``Sj`` complete in ``sparse_hoist + warmup`` (before any ``Bj``);
    ``Dj`` forwards pair with ``Bj`` backwards in steady (except the last
    stage where they collocate). Every dense in warmup interleaves with the
    *last* ``w`` sparses -- there is no separate tail-hoist block, by
    construction of the ``prehoist`` formula.

    The last stage (``w == 0``) hoists all sparses up-front; the first stage
    (``w == N - 1``) hoists only one and interleaves the rest with dense
    forwards. Front-loading everything on every stage would delay stage 0's
    first activation send and push a bubble downstream.

    Why the hoist: under fused TBE (``apply_optimizer_in_backward``) the sparse
    optimizer fires *during* backward. In monolithic 1F1B the sparse forward
    for microbatch N sees embedding weights already updated by the backward of
    some earlier microbatch -- diverging from SGD-exact semantics. Ensuring
    every sparse completes before the first backward makes all microbatches'
    sparse lookups observe the same pre-step weights, matching a single-process
    ``for mb: forward; loss.backward()`` reference.

    Requires the stage module to satisfy
    :class:`~torchrec.distributed.maglev.stage.MaglevStage` (provides
    ``forward_sparse`` and ``forward_dense``, with ``forward`` derived from
    them -- see ``MaglevSplitStage``, or
    :class:`~torchrec.distributed.maglev.stage.SplitForwardMixin` to derive it
    automatically). Falling back to a stage that lacks these methods raises
    ``AttributeError`` on the first sparse call -- fail-fast, no partial
    schedule execution.

    Returns the per-microbatch losses observed on the last stage (empty on
    other stages). Same gradient semantics as :func:`run_1f1b`: gradients
    accumulate across microbatches, DP-averaged once, single optimizer step.
    """
    num_stages = pipeline.num_stages
    num_micro = len(microbatch_inputs)
    stage_index = pipeline.stage_index
    num_warmup = min(num_stages - stage_index - 1, num_micro)
    # prehoist floor is (stage_index + 1) so the first stages have a small
    # initial burst (fast pipeline start); ceiling is (num_micro - num_warmup)
    # so every dense in warmup pairs with a prefetched sparse (no tail-hoist
    # block). For N <= num_stages the two are equal; for N > num_stages the
    # burst grows so the trailing sparses become the ones interleaved with
    # dense forwards.
    prehoist = min(max(stage_index + 1, num_micro - num_warmup), num_micro)
    num_steady = num_micro - num_warmup

    def _label(idx: int) -> Optional[torch.Tensor]:
        return labels[idx] if labels is not None else None

    optimizer.zero_grad()
    losses: List[float] = []

    fwd_idx = 0

    # Sparse hoist: front-load the `prehoist` sparses that will not be paired
    # with a warmup dense. This is the visible cost of split-forward vs
    # monolithic 1F1B -- profiler traces show it as its own span.
    with record_function("## sparse_hoist ##"):
        for j in range(prehoist):
            pipeline.forward_sparse_micro(microbatch_inputs[j], j)

    # Warmup: for each of `num_warmup` denses, run `D_j`, then prefetch the
    # sparse that will be needed `prehoist` microbatches later. Symmetric with
    # `steady`'s (dense, backward) interleave. By the end of warmup, all
    # sparses have completed -- the first backward in steady can safely fire
    # the fused TBE optimizer.
    with record_function("## warmup ##"):
        for j in range(num_warmup):
            pipeline.forward_dense_micro(microbatch_inputs[j], _label(j), j)
            fwd_idx += 1
            next_s = j + prehoist
            if next_s < num_micro:
                pipeline.forward_sparse_micro(microbatch_inputs[next_s], next_s)
        # Tail hoist: with the current `prehoist` formula
        # `max(stage_index + 1, num_micro - num_warmup)`, `prehoist +
        # num_warmup >= num_micro` always, so this loop body never runs. Kept
        # as a safety net: if `prehoist` is ever tuned down (e.g. an opt-in
        # smaller burst to speed up pipeline start), this catches the missing
        # sparses so exactness cannot be silently broken.
        done_sparse_count = min(prehoist + num_warmup, num_micro)
        for j in range(done_sparse_count, num_micro):
            pipeline.forward_sparse_micro(microbatch_inputs[j], j)

    # Steady: 1F1B on the dense halves. All sparses are done; each dense pops
    # its pre-computed pooled tensor from _pooled.
    with record_function("## steady ##"):
        for _ in range(num_steady):
            pipeline.forward_dense_micro(
                microbatch_inputs[fwd_idx], _label(fwd_idx), fwd_idx
            )
            fwd_idx += 1
            loss = pipeline.backward_micro(criterion)
            if loss is not None:
                losses.append(loss.item())

    with record_function("## cooldown ##"):
        for _ in range(num_warmup):
            loss = pipeline.backward_micro(criterion)
            if loss is not None:
                losses.append(loss.item())

    assert not pipeline._pooled, (
        f"run_1f1b_split: {len(pipeline._pooled)} pooled tensors undrained on "
        f"stage {stage_index}: {sorted(pipeline._pooled)}"
    )

    with record_function("## wait_sends ##"):
        pipeline.wait_sends()
    with record_function("## reduce_gradients ##"):
        pipeline.stage.reduce_gradients()
    with record_function("## optimizer_step ##"):
        optimizer.step()
    return losses
