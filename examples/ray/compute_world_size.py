#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import os

import torch
import torch.nn.functional as F
from torch.distributed import all_reduce, get_rank, get_world_size, init_process_group


def compute_world_size() -> int:
    "Dummy script to compute world_size. Meant to test if can run Ray + Pytorch DDP"
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    master_port = int(os.getenv("MASTER_PORT"))
    master_addr = os.getenv("MASTER_ADDR")
    backend = "gloo"

    print(f"initializing `{backend}` process group")
    init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    print("successfully initialized process group")

    rank = get_rank()
    world_size = get_world_size()

    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    all_reduce(t)
    computed_world_size = int(torch.sum(t).item())
    print(
        f"rank: {rank}, actual world_size: {world_size}, computed world_size: {computed_world_size}"
    )
    return computed_world_size


def main() -> None:
    compute_world_size()


if __name__ == "__main__":
    main()
