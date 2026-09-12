#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Torchrec Optimizers

Torchrec contains a special optimizer called KeyedOptimizer. KeyedOptimizer exposes the state_dict with meaningful keys- it enables  loading both
torch.tensor and `ShardedTensor <https://github.com/pytorch/pytorch/issues/55207>`_ in place, and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

It also contains
- several modules wrapping KeyedOptimizer, called CombinedOptimizer and OptimizerWrapper
- Optimizers used in RecSys: e.g. rowwise adagrad/adam/etc
"""
from torchrec.optim.apply_optimizer_in_backward import (  # noqa
    apply_optimizer_in_backward,
)

from torchrec.optim.clipping import GradientClipping, GradientClippingOptimizer  # noqa
from torchrec.optim.ftrl import FTRL  # noqa
from torchrec.optim.fused import FusedOptimizer, FusedOptimizerModule  # noqa
from torchrec.optim.keyed import (  # noqa
    CombinedOptimizer,
    KeyedOptimizer,
    KeyedOptimizerWrapper,
    OptimizerWrapper,
)
from torchrec.optim.optimizers import (  # noqa
    Adagrad,
    Adam,
    LAMB,
    LarsSGD,
    PartialRowWiseAdam,
    PartialRowWiseLAMB,
    SGD,
)
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad  # noqa
from torchrec.optim.semi_sync import SemisyncOptimizer  # noqa
from torchrec.optim.warmup import WarmupOptimizer, WarmupPolicy, WarmupStage  # noqa

from . import (  # noqa  # noqa  # noqa  # noqa
    apply_optimizer_in_backward,
    clipping,
    ftrl,
    fused,
    keyed,
    optimizers,
    rowwise_adagrad,
    warmup,
)
