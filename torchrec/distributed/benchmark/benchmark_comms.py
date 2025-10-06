#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_comms -- 

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_comms 

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist

from torch.autograd.profiler import record_function

from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    cmd_conf,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)

_cc = cmd_conf()


@dataclass
class AllToAllSingleRunConfig(BenchFuncConfig):
    name: str = "all_to_all_single"
    world_size: int = 2
    dim: int = 2048
    profile_dir: str = "."
    num_benchmarks: int = 1
    num_profiles: int = 2
    num_mul: int = 10
    num_concat: int = 100


# all_to_all_single with sync and single stream
def a2a_sync_base(
    batch_inputs: List[Dict[str, Any]],
    dim: int,
    num_mul: int,
    num_concat: int,
    ctx: MultiProcessContext,
) -> None:
    with record_function("## pre-comms compute ##"):
        pre_comms = torch.rand(dim, dim, device=ctx.device) - 0.5
        for _ in range(num_mul):
            pre_comms = pre_comms @ pre_comms
            pre_comms = torch.sigmoid(pre_comms - torch.mean(pre_comms))
        pre_comms = torch.sigmoid(pre_comms).reshape(1, dim, dim) + ctx.rank
        pre_comms = torch.concat([pre_comms] * num_concat)

    with record_function("## all_to_all_single ##"):
        post_comms = torch.empty_like(pre_comms)
        req = dist.all_to_all_single(output=post_comms, input=pre_comms, group=ctx.pg)

    with record_function("## comms validation ##"):
        mixed_ranks = post_comms.to(torch.int).reshape(-1)
        N = mixed_ranks.numel() // ctx.world_size
        checks = [
            torch.all(mixed_ranks[i * N : (i + 1) * N] == i)
            for i in range(ctx.world_size)
        ]

    with record_function("## irrelevant compute ##"):
        pre_comms = torch.rand(dim, dim, device=ctx.device) - 0.5
        for _ in range(num_mul):
            pre_comms = pre_comms @ pre_comms
            pre_comms = torch.sigmoid(pre_comms - torch.mean(pre_comms))
        pre_comms = torch.sigmoid(pre_comms) + ctx.rank

    with record_function("## post-comms compute ##"):
        post_comms = post_comms[0]
        for _ in range(num_mul):
            post_comms = post_comms @ post_comms
            post_comms = torch.sigmoid(pre_comms - torch.mean(post_comms))
        post_comms = torch.sigmoid(post_comms) + ctx.rank

    with record_function("## assert ##"):
        assert all(checks)


# single-rank runner
def a2a_single_runner(rank: int, world_size: int, arg: AllToAllSingleRunConfig) -> None:
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:

        if arg.name.startswith("a2a_sync_base"):
            func = a2a_sync_base
        else:
            func = a2a_sync_base

        result = benchmark_func(
            bench_inputs=[],
            prof_inputs=[],
            benchmark_func_kwargs={
                "ctx": ctx,
                "dim": arg.dim,
                "num_mul": arg.num_mul,
                "num_concat": arg.num_concat,
            },
            func_to_benchmark=func,
            rank=rank,
            **arg.benchmark_func_kwargs()
        )

        if rank == 0:
            print(result)


@_cc.register
def a2a_single(arg: AllToAllSingleRunConfig) -> None:
    run_multi_process_func(func=a2a_single_runner, world_size=arg.world_size, arg=arg)


if __name__ == "__main__":
    _cc.main()
