#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import Any, List, Optional

import torch
from torch import nn
from torchrec.distributed.model_parallel import (
    DataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.schema.utils import is_signature_compatible


def stable_dmp_init(
    # pyre-ignore [2]
    self,
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    init_data_parallel: bool = True,
    init_parameters: bool = True,
    data_parallel_wrapper: Optional[DataParallelWrapper] = None,
) -> None:
    pass


# pyre-ignore [3]
def stable_dmp_forward(
    # pyre-ignore [2]
    self,
    # pyre-ignore [2]
    *args,
    # pyre-ignore [2]
    **kwargs,
) -> Any:
    pass


class TestModelParallelSchema(unittest.TestCase):
    def test_dmp_init(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(stable_dmp_init),
                inspect.signature(DistributedModelParallel.__init__),
            )
        )

    def test_dmp_forward(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(stable_dmp_forward),
                inspect.signature(DistributedModelParallel.forward),
            )
        )
