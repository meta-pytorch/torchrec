#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
OSS test modules for TorchRec distributed testing.

This module provides the OSS (open-source) implementations of test utilities.
For internal Meta usage, the fb/test_modules.py file is used instead,
which imports the internal implementations.

Usage:
    from torchrec.distributed.test_utils.test_modules import DistributedShampoo
"""

# Try to import OSS DistributedShampoo
# This will be available when installed via: pip install distributed-shampoo
from distributed_shampoo import DistributedShampoo


__all__ = ["DistributedShampoo"]
