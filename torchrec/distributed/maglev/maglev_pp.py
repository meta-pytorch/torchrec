#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One entry point for running a Maglev model pipeline-parallel.

A model is authored once, for every schedule, as a
:class:`~torchrec.distributed.maglev.module.MaglevModuleList` of layers with
sparse and dense nets. :class:`MaglevPP` builds this rank's stage, lets the
caller parallelize its module, and builds the schedule :class:`MaglevPPConfig`
names, so nothing above it branches on the schedule.

The package is experimental; see :mod:`torchrec.distributed.maglev`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Iterator, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torchrec.distributed.maglev.customized_pipeline import (
    MaglevRailCustom,
    ScheduleValidator,
)
from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import (
    Maglev1F1B,
    MaglevRail,
    MaglevRail1F1B,
    MaglevRailZB,
)
from torchrec.distributed.maglev.process_groups import MaglevProcessGroups
from torchrec.distributed.maglev.stage import StageWrapper

#: The schedules :meth:`MaglevPP.build` makes.
_Schedule = Union[Maglev1F1B, MaglevRail]


@unique
class MaglevSchedule(Enum):
    """The schedules :class:`MaglevPP` routes to, by their config names."""

    #: :class:`~torchrec.distributed.maglev.pipeline.Maglev1F1B`: each
    #: microbatch runs every layer's sparse and dense nets, in 1F1B order.
    ONE_F_ONE_B = "1f1b"
    #: :class:`~torchrec.distributed.maglev.pipeline.MaglevRail1F1B`: each sparse
    #: net once per pass, the dense nets in 1F1B order with whole backwards.
    RAIL_1F1B = "rail_1f1b"
    #: :class:`~torchrec.distributed.maglev.pipeline.MaglevRailZB`: each sparse
    #: net once per pass, the dense nets in zero-bubble order.
    RAIL_ZB = "rail_zb"
    #: :class:`~torchrec.distributed.maglev.customized_pipeline.MaglevRailCustom`:
    #: each sparse net once per pass, the dense nets in the order
    #: ``schedule_actions`` gives.
    RAIL_CUSTOM = "rail_custom"


def _schedule(name: Union[MaglevSchedule, str]) -> MaglevSchedule:
    """``name`` as a :class:`MaglevSchedule`, naming the valid ones if it is not."""
    try:
        return MaglevSchedule(name)
    except ValueError:
        names = [s.value for s in MaglevSchedule]
        hint = (
            "; 'rail' was split into 'rail_1f1b' (1F1B dense order), "
            "'rail_zb' (zero-bubble) and 'rail_custom' (schedule_actions)"
            if name == "rail"
            else ""
        )
        raise ValueError(
            f"unknown schedule {name!r}; expected one of {names}{hint}"
        ) from None


@dataclass(frozen=True)
class MaglevPPConfig:
    r"""MaglevPPConfig(schedule, num_microbatches, layers_per_stage, stage_size, schedule_actions=None, loss_only_output=False, scale_grads=True)

    What :class:`MaglevPP` builds. A schedule name and lists, as a config file
    gives them, are converted.

    With ``scale_grads`` every schedule trains on the mean of its microbatch
    losses; without it, on their sum. The rule is the same under every
    schedule, so switching schedule does not rescale the gradients.

    Args:
        schedule (MaglevSchedule): The schedule to run, or its name.
        num_microbatches (int): Dense microbatches per pass, at least one per
            stage. Each pass's batch must divide evenly into them.
        layers_per_stage (Tuple[int, ...]): Layers each stage owns, one entry per
            stage, summing to the model's layer count.
        stage_size (int): Ranks per stage (one hardware scale-up domain).
        schedule_actions (Tuple[str, ...], optional): The dense order under
            ``RAIL_CUSTOM``, which requires it: one row per stage, as
            :class:`~torchrec.distributed.maglev.customized_pipeline.ScheduleValidator`
            parses it. Every other schedule derives its order and refuses it.
            Default: ``None``
        loss_only_output (bool): Whether the last stage returns only the loss.
            Default: ``False``
        scale_grads (bool): Under every schedule, seed each microbatch's loss
            gradient with ``1 / num_microbatches``, so mean-loss microbatches
            train on the mean over the pass's whole batch. The seed takes the
            loss's dtype, so a bf16 loss rounds ``1 / num_microbatches`` when it
            is not a power of two. Default: ``True``

    Raises:
        ValueError: On an unknown schedule name, an empty cut, fewer
            microbatches than stages, ``RAIL_CUSTOM`` without
            ``schedule_actions``, or ``schedule_actions`` under another schedule
            or with the wrong number of rows.

    Example::

        zero_bubble = MaglevPPConfig("rail_zb", 4, (3, 3), 8)
        custom = MaglevPPConfig(
            "rail_custom", 4, (3, 3), 8,
            schedule_actions=ScheduleValidator.generate(2, 4),
        )
    """

    schedule: MaglevSchedule
    num_microbatches: int
    layers_per_stage: Tuple[int, ...]
    stage_size: int
    schedule_actions: Optional[Tuple[str, ...]] = None
    loss_only_output: bool = False
    scale_grads: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schedule", _schedule(self.schedule))
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
        if self.schedule_actions is None:
            if self.schedule is MaglevSchedule.RAIL_CUSTOM:
                raise ValueError(
                    "schedule 'rail_custom' runs the dense order schedule_actions "
                    "gives; pass one row per stage"
                )
            return
        if self.schedule is not MaglevSchedule.RAIL_CUSTOM:
            raise ValueError(
                f"schedule_actions give MaglevRailCustom's dense order, under "
                f"schedule 'rail_custom'; schedule {self.schedule.value!r} "
                "derives its own order"
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
    and assign it back, place it with :meth:`to`, wrap the dense nets (DDP or
    FSDP2) and assign it back, then call :meth:`build`. The schedule snapshots
    the data-parallel wrappers when built, so :attr:`module` is fixed from then
    on.

    Each pass trains ``num_microbatches`` microbatches and steps the optimizer
    once. A Rail schedule reads one raw batch per pass, ``1F1B`` one per microbatch;
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
        ValueError: If ``model`` has a layer without a dense net or with its own
            ``forward``, or the stage refuses it otherwise
            (:class:`~torchrec.distributed.maglev.stage.StageWrapper`).

    Example::

        config = MaglevPPConfig("rail_zb", 4, (3, 3), 8)
        pp = MaglevPP(model, config, process_groups)
        pp.module = shard_embeddings(pp.module)  # DMP over the stage's ranks
        pp.to(device)
        pp.module = wrap_dense(pp.module)  # DDP or FSDP2 on the dense nets
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

    def build(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        schedule_cls: Optional[Type[_Schedule]] = None,
        **schedule_kwargs: Any,
    ) -> None:
        r"""build(optimizer, *, schedule_cls=None, **schedule_kwargs) -> None

        Build the schedule over the stage as wrapped so far.

        Args:
            optimizer (torch.optim.Optimizer): The stage's optimizer, stepped once
                per pass.
            schedule_cls (Type, optional): A subclass of the class the config's
                schedule selects, built in its place with the same arguments, so
                a caller can hook the schedule without choosing a different
                one. Default: ``None``
            **schedule_kwargs: Extra keyword arguments for ``schedule_cls``.

        Raises:
            RuntimeError: If the schedule was already built.
            TypeError: If ``schedule_cls`` is not a subclass of the class the
                config's schedule selects, or ``schedule_kwargs`` are given
                without it.
            ValueError: If the schedule refuses the stage or its wrappers.
        """
        if self._pipeline is not None:
            raise RuntimeError("MaglevPP.build() runs once")
        config, stage = self.config, self.stage
        selected: Type[_Schedule]
        kwargs: Dict[str, Any] = {}
        if config.schedule is MaglevSchedule.ONE_F_ONE_B:
            selected = Maglev1F1B
        elif config.schedule_actions is not None:  # RAIL_CUSTOM, which requires them
            selected = MaglevRailCustom
            kwargs["actions"] = config.schedule_actions[stage.stage_index].split()
        elif config.schedule is MaglevSchedule.RAIL_1F1B:
            selected = MaglevRail1F1B
        else:
            selected = MaglevRailZB
        if schedule_cls is None:
            if schedule_kwargs:
                raise TypeError(
                    f"schedule_kwargs {sorted(schedule_kwargs)} need a schedule_cls "
                    "that takes them"
                )
            schedule_cls = selected
        elif not (
            isinstance(schedule_cls, type) and issubclass(schedule_cls, selected)
        ):
            raise TypeError(
                f"schedule_cls must subclass {selected.__name__}, the class "
                f"schedule {config.schedule.value!r} selects; got {schedule_cls!r}"
            )
        self._pipeline = schedule_cls(
            stage,
            optimizer,
            config.num_microbatches,
            scale_grads=config.scale_grads,
            **kwargs,
            **schedule_kwargs,
        )

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
            Optional[torch.Tensor]: Under a Rail schedule, on the last stage, the pass's
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
        pass's under a Rail schedule, one microbatch's under ``1F1B``."""
        if self.config.schedule is MaglevSchedule.ONE_F_ONE_B:
            return microbatch_size
        return microbatch_size * self.config.num_microbatches
