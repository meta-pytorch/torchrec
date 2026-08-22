#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Maglev: staged, HSD-aligned recommender execution for TorchRec (MVP).

A **layer** is the unit of compute; a **stage** is one or more consecutive layers
and the unit of parallelism, one per hardware scale-up domain (HSD). A model is
authored as a :class:`~torchrec.distributed.maglev.module.MaglevModuleList` of
layers and runs either standalone in one process or pipeline-parallel: each rank
hands the whole model to a
:class:`~torchrec.distributed.maglev.stage.StageWrapper`, which keeps the one
stage that rank owns, and a
:class:`~torchrec.distributed.maglev.pipeline.MaglevPipelineBase` schedule drives
it.

Import from the submodules directly (``maglev.module``, ``maglev.stage``,
``maglev.pipeline``); nothing is re-exported here, so there is no second list of
names to keep in step with the code.
"""
