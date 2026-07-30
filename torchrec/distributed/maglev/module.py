#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, List

import torch
import torch.nn as nn


class MaglevModuleList(nn.ModuleList):
    """An ordered ``ModuleList`` of Maglev stages that chains them in ``forward``.

    ``forward`` runs the stages in order, threading each stage's output into the
    next::

        out_i = stage_i(stage_inputs[i], out_{i-1})   with   out_{-1} = None

    This is the single-process reference execution -- the in-process analogue of
    the pipeline-parallel ``MaglevPipeline`` (which runs only the stage a given
    rank owns and transfers the activation across HSD boundaries). Both must
    produce identical numerics; that equivalence is what the correctness test
    checks. Container behavior (``len``, indexing, iteration) is inherited from
    ``nn.ModuleList``.

    Args:
        stages: the ordered stage modules. Each stage's ``forward`` takes
            ``(stage_input, prev_output)`` and returns its output activation
            (``prev_output=None`` for the first stage).

    Example::

        model = MaglevModuleList([stage0, stage1])
        out = model([stage_input0, stage_input1])
    """

    def __init__(self, stages: List[nn.Module]) -> None:
        if len(stages) == 0:
            raise ValueError("MaglevModuleList requires at least one stage")
        super().__init__(stages)

    def preproc(self, model_input: Any) -> List[Any]:
        """Split the raw model input into one input per stage.

        This is the seam where feature partitioning / indexing lives (the Maglev
        Indexer). The MVP is a passthrough: ``model_input`` is already the list of
        per-stage inputs. Runs under ``torch.no_grad()`` (see :meth:`forward`).

        Args:
            model_input: the raw batch to partition across stages.

        Returns:
            List[Any]: one input per stage, index-aligned with the stage list.
        """
        return model_input

    def forward(self, model_input: Any) -> Any:
        """Chain the stages, threading each stage's output into the next.

        :meth:`preproc` (run under ``torch.no_grad()``) splits ``model_input``
        into one input per stage; each stage's output feeds the next as
        ``prev_output`` (``None`` for the first stage).

        Args:
            model_input: the raw batch; ``preproc`` partitions it into one input
                per stage. Inputs may be of any (per-stage) type.

        Returns:
            Any: the last stage's output (e.g. the final prediction). The
            inter-stage carrier type may differ from the final output type, so
            both input and output are typed ``Any``.
        """
        with torch.no_grad():
            stage_inputs = self.preproc(model_input)

        if len(stage_inputs) != len(self):
            raise ValueError(
                f"expected {len(self)} stage inputs, got {len(stage_inputs)}"
            )
        prev_output: Any = None
        for stage, stage_input in zip(self, stage_inputs):
            prev_output = stage(stage_input, prev_output)
        assert prev_output is not None  # guaranteed: at least one stage
        return prev_output
