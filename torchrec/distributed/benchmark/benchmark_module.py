#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for module-level ops (placeholder).

``benchmark_runner`` is the per-rank entry point. It is invoked once per rank by the process
runner (``process_runner.run_single_process_func`` /
``process_runner.run_local_multi_process_func``), which owns process group init +
handshake and injects a live ``SingleProcessContext`` (``ctx``) plus this rank's
``rank`` and ``world_size``. The runner must therefore use ``ctx.device`` /
``ctx.pg`` directly rather than creating its own context.

A follow-up launcher binary will call ``runner`` explicitly with options to run on
MAST or locally.
"""

import logging
from typing import Any

from torchrec.distributed.test_utils.process_runner import SingleProcessContext

logger: logging.Logger = logging.getLogger(__name__)


def benchmark_runner(
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> None:
    """Per-rank module benchmark entry point (placeholder -- does nothing yet).

    Args:
        ctx: live single-process context (device + process group) injected by the
            process runner; use ``ctx.device`` / ``ctx.pg`` directly.
        rank: this process' global rank.
        world_size: total number of ranks.
        **kwargs: forwarded benchmark options (none consumed yet).
    """
    # TODO: implement the module benchmark body.
    pass
