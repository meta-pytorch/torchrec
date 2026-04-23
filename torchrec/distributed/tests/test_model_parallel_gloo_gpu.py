#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torchrec.distributed.test_utils.test_model_parallel import _ModelParallelBase

# GPU tests for Gloo.


class ModelParallelTestGlooGPU(_ModelParallelBase):
    backend = "gloo"


del _ModelParallelBase
