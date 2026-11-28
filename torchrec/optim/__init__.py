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
from torchrec.optim.apply_optimizer_in_backward import (
    apply_optimizer_in_backward as apply_optimizer_in_backward,
)

from torchrec.optim.clipping import (
    GradientClipping as GradientClipping,
    GradientClippingOptimizer as GradientClippingOptimizer,
)
from torchrec.optim.fused import (
    FusedOptimizer as FusedOptimizer,
    FusedOptimizerModule as FusedOptimizerModule,
)
from torchrec.optim.keyed import (
    CombinedOptimizer as CombinedOptimizer,
    KeyedOptimizer as KeyedOptimizer,
    KeyedOptimizerWrapper as KeyedOptimizerWrapper,
    OptimizerWrapper as OptimizerWrapper,
)
from torchrec.optim.optimizers import (
    Adagrad as Adagrad,
    Adam as Adam,
    LAMB as LAMB,
    LarsSGD as LarsSGD,
    PartialRowWiseAdam as PartialRowWiseAdam,
    PartialRowWiseLAMB as PartialRowWiseLAMB,
    SGD as SGD,
)
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad as RowWiseAdagrad
from torchrec.optim.semi_sync import SemisyncOptimizer as SemisyncOptimizer
from torchrec.optim.warmup import (
    WarmupOptimizer as WarmupOptimizer,
    WarmupPolicy as WarmupPolicy,
    WarmupStage as WarmupStage,
)

from . import (
    clipping as clipping,
    fused as fused,
    keyed as keyed,
    optimizers as optimizers,
    rowwise_adagrad as rowwise_adagrad,
    warmup as warmup,
)
