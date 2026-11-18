#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One entry point for running a Maglev model pipeline-parallel.

A model is authored once, for every schedule, as a
:class:`~torchrec.distributed.maglev.module.MaglevModuleList` of layers with
sparse and dense halves. :class:`MaglevPP` builds this rank's stage, lets the
caller parallelize its module, and builds the schedule :class:`MaglevPPConfig`
names, so nothing above it branches on the schedule.

The package is experimental; see :mod:`torchrec.distributed.maglev`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrec.distributed.maglev.customized_pipeline import (
    MaglevRailCustom,
    ScheduleValidator,
)
from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import Maglev1F1B, MaglevRail
from torchrec.distributed.maglev.stage import MaglevProcessGroups, StageWrapper

#: The schedules :meth:`MaglevPP.build` makes.
_Schedule = Union[Maglev1F1B, MaglevRail]
#: MaglevRail's dense orders, by ``MaglevPPConfig.dense_schedule``.
_DENSE_SCHEDULES: Tuple[str, ...] = ("1f1b", "zero_bubble")


@unique
class MaglevSchedule(Enum):
    """The schedules :class:`MaglevPP` routes to, by their config names."""

    #: :class:`~torchrec.distributed.maglev.pipeline.Maglev1F1B`: each
    #: microbatch runs every layer's sparse and dense halves, in 1F1B order.
    ONE_F_ONE_B = "1f1b"
    #: :class:`~torchrec.distributed.maglev.pipeline.MaglevRail`: each sparse
    #: half once per pass, the dense halves in the order ``dense_schedule`` names.
    RAIL = "rail"


@dataclass(frozen=True)
class MaglevPPConfig:
    r"""MaglevPPConfig(schedule, num_microbatches, layers_per_stage, stage_size, schedule_actions=None, loss_only_output=False, scale_grads=True, dense_schedule=None)

    What :class:`MaglevPP` builds. A schedule name and lists, as a config file
    gives them, are converted.

    Under ``1f1b`` a pass trains on the sum of its microbatch losses, as on
    master; under ``rail`` with ``scale_grads`` it trains on their mean. Switching
    schedule therefore rescales the gradients by ``num_microbatches``, which
    matters for the SGD learning rate and for gradient clipping.

    Args:
        schedule (MaglevSchedule): The schedule to run, or its name.
        num_microbatches (int): Dense microbatches per pass, at least one per
            stage. Each pass's batch must divide evenly into them.
        layers_per_stage (Tuple[int, ...]): Layers each stage owns, one entry per
            stage, summing to the model's layer count.
        stage_size (int): Ranks per stage (one hardware scale-up domain).
        schedule_actions (Tuple[str, ...], optional): Under ``RAIL``, the dense
            order given rather than derived, one row per stage, as
            :class:`~torchrec.distributed.maglev.customized_pipeline.ScheduleValidator`
            parses it; runs ``"zero_bubble"``'s split backward. Default: ``None``
        loss_only_output (bool): Whether the last stage returns only the loss.
            Default: ``False``
        scale_grads (bool): Under ``RAIL``, seed each microbatch's loss gradient
            with ``1 / num_microbatches``, so mean-loss microbatches train on the
            mean over the pass's whole batch. No effect under ``ONE_F_ONE_B``,
            whose ``Maglev1F1B`` sums its microbatch losses. Default: ``True``
        dense_schedule (str, optional): Under ``RAIL``, ``"1f1b"`` or
            ``"zero_bubble"``. Default: ``None``, which is ``"1f1b"``

    Raises:
        ValueError: On an unknown name, an empty cut, fewer microbatches than
            stages, or a ``dense_schedule`` or ``schedule_actions`` the schedule
            cannot run.
    """

    schedule: MaglevSchedule
    num_microbatches: int
    layers_per_stage: Tuple[int, ...]
    stage_size: int
    schedule_actions: Optional[Tuple[str, ...]] = None
    loss_only_output: bool = False
    scale_grads: bool = True
    dense_schedule: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", MaglevSchedule(self.schedule))
        object.__setattr__(self, "layers_per_stage", tuple(self.layers_per_stage))
        if self.schedule_actions is not None:
            object.__setattr__(self, "schedule_actions", tuple(self.schedule_actions))
        if not self.layers_per_stage:
            raise ValueError("layers_per_stage needs one entry per stage")
        if self.num_microbatches < self.num_stages:
            raise ValueError(
                f"num_microbatches must be at least the stage count: got "
                f"{self.num_microbatches} for {self.num_stages} stages"
            )
        rail = self.schedule is MaglevSchedule.RAIL
        if self.dense_schedule is not None:
            if not rail:
                raise ValueError(
                    f"dense_schedule applies to schedule 'rail' only, not "
                    f"{self.schedule.value!r}"
                )
            if self.dense_schedule not in _DENSE_SCHEDULES:
                raise ValueError(
                    f"dense_schedule must be one of {list(_DENSE_SCHEDULES)}, "
                    f"got {self.dense_schedule!r}"
                )
        if self.schedule_actions is None:
            return
        if not rail:
            raise ValueError(
                f"schedule_actions give MaglevRail's dense order; schedule "
                f"{self.schedule.value!r} would ignore them"
            )
        if self.dense_schedule == "1f1b":
            raise ValueError(
                "schedule_actions split every backward, as dense_schedule "
                "'zero_bubble' does; they cannot run dense_schedule '1f1b'"
            )
        if len(self.schedule_actions) != self.num_stages:
            raise ValueError(
                f"schedule_actions has {len(self.schedule_actions)} rows for "
                f"{self.num_stages} stages"
            )

    @property
    def num_stages(self) -> int:
        return len(self.layers_per_stage)


class MaglevPP:
    r"""MaglevPP(model, config, process_groups=None)

    Run a Maglev model pipeline-parallel under the schedule ``config`` names.

    Every rank passes the same ``model`` and keeps the stage it owns
    (:attr:`stage`). Parallelism stays the caller's: shard :attr:`module` (DMP)
    and assign it back, place it with :meth:`to`, wrap the dense halves (DDP or
    FSDP2) and assign it back, then call :meth:`build`. The schedule snapshots
    the data-parallel wrappers when built, so :attr:`module` is fixed from then
    on.

    Each pass trains ``num_microbatches`` microbatches and steps the optimizer
    once. ``RAIL`` reads one raw batch per pass, ``1F1B`` one per microbatch;
    :meth:`input_batch_size` gives the rows either way.

    Args:
        model (MaglevModuleList): The authored model, in full, on every rank.
        config (MaglevPPConfig): The schedule and the cut.
        process_groups (MaglevProcessGroups, optional): The Maglev process groups,
            if the caller builds them; by default the stage builds a contiguous
            layout. Default: ``None``

    Raises:
        ScheduleError: If ``config.schedule_actions`` describe an order that runs
            the wrong work or deadlocks.

    Example::

        config = MaglevPPConfig("rail", 4, (3, 3), 8, dense_schedule="zero_bubble")
        pp = MaglevPP(model, config, process_groups)
        pp.module = shard_embeddings(pp.module)  # DMP over the stage's ranks
        pp.to(device)
        pp.module = wrap_dense(pp.module)  # DDP or FSDP2 on the dense halves
        pp.build(optimizer)
        rows = pp.input_batch_size(microbatch_rows)
        for _ in range(passes):
            losses = pp.progress(dataloader_iter)  # last stage: one per microbatch
    """

    def __init__(
        self,
        model: MaglevModuleList,
        config: MaglevPPConfig,
        process_groups: Optional[MaglevProcessGroups] = None,
    ) -> None:
        if config.schedule_actions is not None:
            # The whole grid, before any process group exists: rows that are each
            # legal can still deadlock against each other, which hangs.
            ScheduleValidator.validate(
                ScheduleValidator.parse(config.schedule_actions),
                config.num_stages,
                config.num_microbatches,
            )
        self.config: MaglevPPConfig = config
        self.stage: StageWrapper = StageWrapper(
            model,
            layers_per_stage=config.layers_per_stage,
            stage_size=config.stage_size,
            loss_only_output=config.loss_only_output,
            process_groups=process_groups,
        )
        self._pipeline: Optional[_Schedule] = None

    @property
    def module(self) -> nn.Module:
        """This rank's stage module, for the caller to shard and wrap."""
        return self.stage.module

    @module.setter
    def module(self, module: nn.Module) -> None:
        if self._pipeline is not None:
            raise RuntimeError(
                "wrap the stage module before build(): the schedule found its DDP "
                "and FSDP2 modules when it was built"
            )
        self.stage.module = module

    def to(self, device: torch.device) -> MaglevPP:
        """Place the stage on ``device``, after sharding; return ``self``."""
        self.stage.to(device)
        return self

    def build(self, optimizer: torch.optim.Optimizer) -> None:
        r"""build(optimizer) -> None

        Build the schedule over the stage as wrapped so far.

        Args:
            optimizer (torch.optim.Optimizer): The stage's optimizer, stepped once
                per pass.

        Raises:
            RuntimeError: If the schedule was already built.
            ValueError: If the schedule refuses the stage or its wrappers.
        """
        if self._pipeline is not None:
            raise RuntimeError("MaglevPP.build() runs once")
        config, stage = self.config, self.stage
        num_microbatches = config.num_microbatches
        scale_grads = config.scale_grads
        pipeline: _Schedule
        if config.schedule is MaglevSchedule.ONE_F_ONE_B:
            pipeline = Maglev1F1B(stage, optimizer, num_microbatches)
        elif config.schedule_actions is not None:
            pipeline = MaglevRailCustom(
                stage,
                optimizer,
                num_microbatches,
                actions=config.schedule_actions[stage.stage_index].split(),
                scale_grads=scale_grads,
            )
        else:
            pipeline = MaglevRail(
                stage,
                optimizer,
                num_microbatches,
                dense_schedule=config.dense_schedule or "1f1b",
                scale_grads=scale_grads,
            )
        self._pipeline = pipeline

    @property
    def pipeline(self) -> _Schedule:
        """The schedule :meth:`build` made; drive it through :meth:`progress`."""
        if self._pipeline is None:
            raise RuntimeError("call MaglevPP.build() first")
        return self._pipeline

    def progress(self, dataloader_iter: Iterator[Any]) -> Optional[torch.Tensor]:
        r"""progress(dataloader_iter) -> Optional[torch.Tensor]

        Run one pass and step the optimizer.

        Args:
            dataloader_iter (Iterator[Any]): Raw batches of
                :meth:`input_batch_size` rows.

        Returns:
            Optional[torch.Tensor]: Under ``RAIL``, on the last stage, the pass's
            microbatch losses in order, detached and on the device; ``None``
            elsewhere and under ``1F1B``.
        """
        return self.pipeline.progress(dataloader_iter)

    def close(self) -> None:
        """Release the schedule's reduction hooks; a no-op before :meth:`build`."""
        if self._pipeline is not None:
            self._pipeline.close()

    def input_batch_size(self, microbatch_size: int) -> int:
        """Rows per raw batch for ``microbatch_size``-row microbatches: the whole
        pass's under ``RAIL``, one microbatch's under ``1F1B``."""
        if self.config.schedule is MaglevSchedule.ONE_F_ONE_B:
            return microbatch_size
        return microbatch_size * self.config.num_microbatches
