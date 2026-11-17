#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import ModuleSharder


def shard_model(
    model: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> DistributedModelParallel:
    # shard the model
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If experience OOM, increase the percentage. see
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )

    sharders = get_default_sharders()
    sharders.append(cast(ModuleSharder[nn.Module], ManagedCollisionCollectionSharder()))

    plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
    )
    return model
