#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, List, Optional, Tuple

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

    Two drivers are provided:

    * :meth:`step` -- a single-batch forward-then-backward pass (used by the
      correctness test). Blocking send/recv form a matched wave; no deadlock.
    * :meth:`forward_micro` / :meth:`backward_micro` + :func:`run_1f1b` -- a
      microbatched 1F1B schedule (used by the benchmark). Activations are
      received with blocking recv (strict dependency, forms a wave) and sent
      with non-blocking isend whose completion is deferred to
      :meth:`wait_sends`, which keeps the schedule deadlock-free.

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
