#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for the shared batch-attribution primitives."""

from torchrec.distributed.batch_attribution import (
    attach_exception_attribution,
    AttributedBatch,
    ExceptionAttribution,
    get_exception_attribution,
    unwrap_attributed_batch,
)

__all__ = [
    "attach_exception_attribution",
    "AttributedBatch",
    "ExceptionAttribution",
    "get_exception_attribution",
    "unwrap_attributed_batch",
]
