#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Maglev: staged, HSD-aligned recommender execution for TorchRec (MVP).

This package holds the systems skeleton of Maglev:

* :class:`~torchrec.distributed.maglev.module.MaglevModuleList` -- the
  ``ModuleList``-like API where each entry is a stage.
* :class:`~torchrec.distributed.maglev.stage.StageWrapper` -- binds a stage to
  its process group (one hardware scale-up domain, HSD).
* :class:`~torchrec.distributed.maglev.pipeline.MaglevPipeline` -- hooks the
  stages together across HSDs (forward activation + backward gradient hand-off).

Modeling components (Maglev Indexers / Induct) and the zero-bubble Maglev Rail
schedule are intentionally out of scope for the MVP; the API leaves room for
them to slot in later.
"""

from torchrec.distributed.maglev.module import MaglevModuleList
from torchrec.distributed.maglev.pipeline import MaglevPipeline, run_1f1b
from torchrec.distributed.maglev.stage import (
    build_stage_process_groups,
    EmbeddingShard,
    Replicated,
    StageParallelizer,
    StageWrapper,
)

__all__ = [
    "MaglevModuleList",
    "MaglevPipeline",
    "run_1f1b",
    "StageWrapper",
    "StageParallelizer",
    "Replicated",
    "EmbeddingShard",
    "build_stage_process_groups",
]
