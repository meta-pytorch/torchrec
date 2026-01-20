#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict


__all__: list[str] = []

UnfilteredLogger = "UnfilteredLogger"
SingleRankStaticLogger = "SingleRankStaticLogger"
AllRankStaticLogger = "AllRankStaticLogger"
CappedLogger = "CappedLogger"
MethodLogger = "MethodLogger"

_log_handlers: dict[str, logging.Handler] = defaultdict(logging.NullHandler)
