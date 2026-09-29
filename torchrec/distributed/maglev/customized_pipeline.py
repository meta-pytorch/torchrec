#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The rail pipeline with its dense order given rather than derived.

:class:`MaglevRail` computes phase 2's op order from ``num_warmup`` and
``w_lag``. :class:`MaglevRailCustom` takes it instead, as one explicit row of
ops per stage, so a memory/bubble trade can be swept from config rather than by
editing a loop.

A row for p=4, m=8 is what the derived schedule already runs::

    s0  F0 F1 F2 F3 I0 W0 F4 I1 W1 F5 I2 W2 F6 I3 W3 F7 I4 W4 I5 W5 I6 W6 I7 W7
    s1  F0 F1 F2 I0 F3 I1 W0 F4 I2 W1 F5 I3 W2 F6 I4 W3 F7 I5 W4 I6 W5 I7 W6 W7

Two classes, split on whether they touch the stage:

* :class:`ScheduleValidator` -- everything about an order except running it: the
  grammar, the generator that reproduces the derived shape, the per-row and
  cross-stage checks, and the simulation those checks run on. Pure data; no
  stage, no collectives.
* :class:`MaglevRailCustom` -- the execution. One overridden method.

The ops map one-to-one onto ``StageWrapper`` calls, so a row IS the execution
order. Stage construction, sharding and the layer cut all happen before the
pipeline object exists, so nothing here can change the model; a row only
reorders calls on an already-built stage.
"""

import logging
import re
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, List, Optional, Sequence, Tuple

import torch
from torchrec.distributed.maglev.pipeline import MaglevRail
from torchrec.distributed.maglev.stage import MaglevRailPassState, StageWrapper

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    """One op in a stage's row: an op letter and the microbatch it runs."""

    op: str
    microbatch: int

    def __str__(self) -> str:
        return f"{self.op}{self.microbatch}"


class ScheduleError(ValueError):
    """An order that would run the wrong work, or hang."""


class ScheduleValidator:
    """Everything about a dense order except executing it.

    The grammar, the generator, the checks, and the simulation the cross-stage
    check runs on. Nothing here touches a stage or a collective, which is what
    lets a row be written and checked before a job is ever launched -- the
    failure mode of a bad row is a silent hang, not an exception, so it is not
    something to discover on 8 GPUs.

    The microbatch indices do NOT select a microbatch. Every ``StageWrapper``
    queue is FIFO (``_pending``, ``_recv_act``, ``_recv_grad``, ``_rail_seams``,
    ``_wdense``), so an op's position already fixes which microbatch it runs.
    The indices are carried so a row is self-describing, and
    :meth:`validate_row` rejects a row whose indices disagree with the order it
    would actually execute in.

    There is no monolithic-backward op. Rail's seams are popped by the ``I``
    half and by nothing else, so a ``B`` would leave ``_rail_seams`` out of step
    with ``_pending`` and the whole-pass sparse graph resuming from gradients
    nobody filled -- wrong numbers, silently. Use :class:`Maglev1F1B` for a
    monolithic backward.
    """

    FORWARD: str = "F"
    BACKWARD_INPUT: str = "I"
    BACKWARD_WEIGHT: str = "W"
    OPS: Tuple[str, ...] = (FORWARD, BACKWARD_INPUT, BACKWARD_WEIGHT)

    _TOKEN: "re.Pattern[str]" = re.compile(r"^([FIW])(\d+)$")

    @classmethod
    def parse_row(cls, row: str, stage: int) -> List[Action]:
        """Parse one stage's row, e.g. ``"F0 F1 I0 W0"``."""
        actions = []
        for token in row.split():
            match = cls._TOKEN.match(token)
            if match is None:
                raise ScheduleError(
                    f"stage {stage}: bad action token {token!r}; expected one of "
                    f"{cls.OPS} followed by a microbatch index, e.g. 'F0' or 'W7'"
                )
            actions.append(Action(match.group(1), int(match.group(2))))
        return actions

    @classmethod
    def parse(cls, rows: Sequence[str]) -> List[List[Action]]:
        """Parse one row per stage."""
        return [cls.parse_row(row, stage) for stage, row in enumerate(rows)]

    @classmethod
    def format_rows(cls, grid: Sequence[Sequence[Action]]) -> List[str]:
        """The inverse of :meth:`parse`."""
        return [" ".join(str(a) for a in row) for row in grid]

    @classmethod
    def generate(cls, num_stages: int, num_microbatches: int) -> List[str]:
        """The order :meth:`MaglevRail._dense_zero_bubble` runs, as rows.

        Reproduces it exactly, so handing the result back to
        :class:`MaglevRailCustom` is a no-op. Start from this rather than
        writing a row by hand.
        """
        return [
            " ".join(
                str(a) for a in cls._generate_stage(s, num_stages, num_microbatches)
            )
            for s in range(num_stages)
        ]

    @classmethod
    def _generate_stage(
        cls, stage: int, num_stages: int, num_microbatches: int
    ) -> List[Action]:
        num_warmup = min(num_stages - stage - 1, num_microbatches)
        num_steady = num_microbatches - num_warmup
        actions: List[Action] = []
        counts = dict.fromkeys(cls.OPS, 0)

        def emit(op: str) -> None:
            actions.append(Action(op, counts[op]))
            counts[op] += 1

        def drain(keep: int) -> None:
            while counts[cls.BACKWARD_INPUT] - counts[cls.BACKWARD_WEIGHT] > keep:
                emit(cls.BACKWARD_WEIGHT)

        # The ZB1P rule MaglevRail derives: W trails I by the stage's own index.
        w_lag = stage
        for _ in range(num_warmup):
            emit(cls.FORWARD)
        for _ in range(num_steady):
            emit(cls.FORWARD)
            emit(cls.BACKWARD_INPUT)
            drain(w_lag)
        for _ in range(num_warmup):
            emit(cls.BACKWARD_INPUT)
            drain(w_lag)
        drain(0)
        return actions

    @classmethod
    def validate(
        cls,
        grid: Sequence[Sequence[Action]],
        num_stages: int,
        num_microbatches: int,
    ) -> List[List[Optional[Action]]]:
        """Reject an order that runs the wrong work, or hangs; log what passed.

        Logging rather than accepting silently is the point: a bad order fails
        as a hang, and a hung job leaves no trace file. Having the grid in every
        rank's log makes "what order was it running when it stopped?" answerable
        from the logs alone.

        Returns:
            The simulated timeline, one row per stage, ``None`` where a stage
            idled. Also what the makespan -- the bubble a deeper or shallower
            warmup costs -- is read from.

        Raises:
            ScheduleError: with the offending stage and position named.
        """
        if len(grid) != num_stages:
            raise ScheduleError(
                f"schedule has {len(grid)} rows, expected one per stage "
                f"({num_stages})"
            )
        for stage, row in enumerate(grid):
            cls.validate_row(row, stage, num_microbatches)
        timeline = cls.simulate(grid, num_stages)
        logger.info(
            "custom dense order accepted: %d stages x %d microbatches, %d steps\n%s",
            num_stages,
            num_microbatches,
            len(timeline[0]) if timeline else 0,
            cls.format_grid(timeline),
        )
        return timeline

    @classmethod
    def validate_row(
        cls, row: Sequence[Action], stage: int, num_microbatches: int
    ) -> None:
        """Reject one stage's row, independent of what the other stages do.

        The half of :meth:`validate` a single rank can check on its own.

        Raises:
            ScheduleError: naming the offending position.
        """
        counts = dict.fromkeys(cls.OPS, 0)
        for pos, action in enumerate(row):
            # FIFO: the stage's queues fix which microbatch runs, so an index
            # that disagrees with the running count would silently run
            # something else.
            if action.microbatch != counts[action.op]:
                raise ScheduleError(
                    f"stage {stage} position {pos}: {action} runs out of order "
                    f"-- the queues are FIFO, so this slot is "
                    f"{action.op}{counts[action.op]}"
                )
            counts[action.op] += 1
            if (
                action.op == cls.BACKWARD_INPUT
                and counts[cls.BACKWARD_INPUT] > counts[cls.FORWARD]
            ):
                raise ScheduleError(
                    f"stage {stage} position {pos}: {action} has no forward to "
                    "consume; every backward needs an outstanding forward"
                )
            if (
                action.op == cls.BACKWARD_WEIGHT
                and counts[cls.BACKWARD_WEIGHT] > counts[cls.BACKWARD_INPUT]
            ):
                raise ScheduleError(
                    f"stage {stage} position {pos}: {action} has no deferred "
                    "weight work queued; every W needs a preceding I"
                )
        for op in cls.OPS:
            if counts[op] != num_microbatches:
                raise ScheduleError(
                    f"stage {stage}: has {counts[op]} {op} ops, expected "
                    f"{num_microbatches} (one per microbatch)"
                )

    @classmethod
    def simulate(
        cls, grid: Sequence[Sequence[Action]], num_stages: int
    ) -> List[List[Optional[Action]]]:
        """Step the whole grid to completion, or report the deadlock.

        Models the two couplings that hang a job and that no per-row check can
        see:

        * an ``F`` on stage ``s`` waits for stage ``s-1`` to have produced that
          activation, and an ``I`` waits for stage ``s+1`` to have produced that
          gradient;
        * sends are single-slot (``StageWrapper.start_send_act`` raises on a
          second one), so a forward drains the previous send first, and that
          drain does not complete until the peer has posted its matching
          receive -- which it only does at the top of its own op. A stage can
          run at most one op ahead of its neighbour.

        Ops are atomic here and the drain is charged to the whole op rather than
        its tail, so the timeline is a little more staggered than the idealised
        ZB1P picture. The conservatism is one-sided: it can in principle reject
        an order that runs one op further ahead than modelled, never accept one
        that hangs.

        Raises:
            ScheduleError: naming every blocked stage and what it waited for.
        """
        pcs = [0] * num_stages
        done: List[Dict[str, int]] = [
            dict.fromkeys(cls.OPS, 0) for _ in range(num_stages)
        ]
        timeline: List[List[Optional[Action]]] = [[] for _ in range(num_stages)]

        while any(pcs[s] < len(grid[s]) for s in range(num_stages)):
            ran = False
            reasons: List[str] = []
            step: List[Optional[Action]] = [None] * num_stages
            # Snapshot, never the running counts: an op that only becomes
            # runnable because a neighbour ran in this same step could not have
            # started in it.
            seen = [dict(counts) for counts in done]
            for stage in range(num_stages):
                if pcs[stage] >= len(grid[stage]):
                    continue
                reason = cls._blocked_on(
                    grid[stage][pcs[stage]], stage, num_stages, seen
                )
                if reason is not None:
                    reasons.append(f"  stage {stage} blocked on {reason}")
                    continue
                action = grid[stage][pcs[stage]]
                step[stage] = action
                done[stage][action.op] += 1
                pcs[stage] += 1
                ran = True
            if not ran:
                raise ScheduleError(
                    "schedule deadlocks -- every stage is waiting on another:\n"
                    + "\n".join(reasons)
                    + f"\n\nran to here:\n{cls.format_grid(timeline)}"
                )
            for stage in range(num_stages):
                timeline[stage].append(step[stage])
        return timeline

    @classmethod
    def _blocked_on(
        cls,
        action: Action,
        stage: int,
        num_stages: int,
        seen: Sequence[Dict[str, int]],
    ) -> Optional[str]:
        """Why ``stage`` cannot run ``action`` yet given ``seen``, else None."""
        k = action.microbatch
        if action.op == cls.FORWARD:
            if stage > 0 and seen[stage - 1][cls.FORWARD] <= k:
                return f"activation {k} from stage {stage - 1}"
            if stage < num_stages - 1 and k >= 1 and seen[stage + 1][cls.FORWARD] < k:
                return f"stage {stage + 1} to receive activation {k - 1}"
            return None
        if action.op == cls.BACKWARD_INPUT:
            if stage < num_stages - 1 and seen[stage + 1][cls.BACKWARD_INPUT] <= k:
                return f"gradient {k} from stage {stage + 1}"
            if stage > 0 and k >= 1 and seen[stage - 1][cls.BACKWARD_INPUT] < k:
                return f"stage {stage - 1} to receive gradient {k - 1}"
            return None
        return None  # W is local

    @classmethod
    def format_grid(cls, timeline: Sequence[Sequence[Optional[Action]]]) -> str:
        """Render a simulated timeline as the s0..sN table, '..' for idle."""
        cells = [
            [".." if action is None else str(action) for action in row]
            for row in timeline
        ]
        width = max((len(c) for row in cells for c in row), default=2)
        return "\n".join(
            f"s{stage} | " + " ".join(c.rjust(width) for c in row)
            for stage, row in enumerate(cells)
        )


class MaglevRailCustom(MaglevRail):
    """:class:`MaglevRail` with phase 2's order given rather than derived.

    Same pass as the base class -- same whole-pass sparse phases, same seams,
    same single gradient reduction -- with only the dense order replaced. It is
    its own class so an experimental order cannot perturb the shipped shape:
    everything it changes is in :meth:`_dense`.

    The caller owns the cross-stage consistency that makes an order safe. Rows
    that are each individually legal can still deadlock against each other, and
    a stage stuck in an unmatched NCCL receive does not raise -- it hangs. That
    needs the whole grid, so run :meth:`ScheduleValidator.validate` over it
    before building any of these.

    Args:
        stage (StageWrapper): This rank's pipeline stage, built with
            ``enable_rail=True``.
        optimizer (torch.optim.Optimizer): Optimizer stepped once per pass.
        num_microbatches (int): Number of dense microbatches.
        actions (Sequence[str]): This stage's ops, e.g.
            ``("F0", "F1", "I0", "W0", ...)``.
        no_sync (Callable, optional): As :class:`MaglevRail`. Default: ``None``

    Raises:
        ScheduleError: if this stage's row cannot run.
    """

    def __init__(
        self,
        stage: StageWrapper,
        optimizer: torch.optim.Optimizer,
        num_microbatches: int,
        actions: Sequence[str],
        no_sync: Optional[Callable[[], ContextManager[None]]] = None,
    ) -> None:
        super().__init__(stage, optimizer, num_microbatches, no_sync)
        # Up front, not from inside the loop: a row that only goes wrong at its
        # tenth op would otherwise have issued nine ops' worth of sends first,
        # leaving the neighbouring stages waiting on a pass about to raise.
        ScheduleValidator.validate_row(
            ScheduleValidator.parse_row(" ".join(actions), stage.stage_index),
            stage.stage_index,
            num_microbatches,
        )
        self.actions: Tuple[str, ...] = tuple(actions)
        # Named per rank, not just in the grid validate() logged: a hang is the
        # failure mode, and the first question is which order THIS stage ran.
        logger.info(
            "stage %d custom dense order (%d ops): %s",
            stage.stage_index,
            len(self.actions),
            " ".join(self.actions),
        )

    def _dense(self, state: MaglevRailPassState) -> None:
        """Phase 2, in the order this instance was given.

        One op per entry, each expanding to exactly the calls
        :meth:`MaglevRail._dense_zero_bubble` makes::

            F<k>   start_recv_act + dense_forward_micro
            I<k>   start_recv_grad + dense_backward_act_micro
            W<k>   dense_backward_weight_micro       (local, no communication)

        So a row IS the execution order, and the peer-to-peer traffic follows
        from it with nothing to wire separately: every receive is posted by the
        op that consumes it and every send is issued inside ``*_micro``.

        Args:
            state: the pass state phase 1 produced.
        """
        stage = self.stage
        counts: Dict[str, int] = dict.fromkeys(ScheduleValidator.OPS, 0)
        for token in self.actions:
            # DEBUG, so it costs nothing normally and is there when it is
            # needed: an unmatched receive hangs rather than raising, and this
            # is what says which op each stage stopped on.
            logger.debug("stage %d: %s", stage.stage_index, token)
            op = token[:1]
            fwd_idx = counts[ScheduleValidator.FORWARD]
            counts[op] += 1
            if op == ScheduleValidator.FORWARD:
                stage_input = state.dense_inputs[fwd_idx]
                stage.start_recv_act(stage_input)
                with self._no_sync():
                    stage.dense_forward_micro(
                        stage_input,
                        fwd_idx,
                        state.seams_for(fwd_idx),
                    )
            elif op == ScheduleValidator.BACKWARD_INPUT:
                stage.start_recv_grad()
                stage.dense_backward_act_micro()
            else:
                stage.dense_backward_weight_micro()
