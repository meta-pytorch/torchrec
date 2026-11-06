#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import tempfile
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchcomms import is_backend_built, new_comm
from torchrec.distributed.fsdp_sharded_relay import (
    ShardedRelayAllReduce,
    ShardedRelayReduceScatter,
)


def _adapter(cls, comm, rank, groups, count, dtype):
    return cls(
        comm,
        rank,
        list(range(dist.get_world_size())),
        groups,
        [count] * len(groups),
        dtype,
        torch.device("cuda", rank),
    )


def _time_call(call: Callable[[], None], warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            [start.elapsed_time(end) * 1000], device="cuda", dtype=torch.float64
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.cpu().tolist()[0])
    return statistics.median(samples)


def _collectives(rank, comm, args, dtype):
    rows = []
    size = dist.get_world_size()
    for active in (2, 4):
        if size < 2 * active:
            continue
        groups = [list(range(i, i + active)) for i in range(0, size, active)]
        pgs = [dist.new_group(g) for g in groups]
        pg = pgs[rank // active]
        for kib in args.sizes_kib:
            for name in ("all_reduce", "reduce_scatter"):
                rows.append(
                    _collective_case(rank, comm, args, dtype, groups, pg, kib, name)
                )
    return rows


def _collective_case(rank, comm, args, dtype, groups, pg, kib, name):
    active = pg.size()
    count = kib * 1024 // torch.empty((), dtype=dtype).element_size()
    count -= count % active
    output_count = count if name == "all_reduce" else count // active
    cls = ShardedRelayAllReduce if name == "all_reduce" else ShardedRelayReduceScatter
    relay = _adapter(cls, comm, rank, groups, output_count, dtype)
    source = torch.full((count,), rank + 1, dtype=dtype, device="cuda")
    before = torch.empty(output_count, dtype=dtype, device="cuda")
    after = torch.empty_like(before)

    def run(backend, op=dist.ReduceOp.AVG):
        if name == "all_reduce":
            output = before if backend == "before" else after
            output.copy_(source)
            if backend == "before":
                dist.all_reduce(output, group=pg, op=op)
            else:
                relay(output, pg, op)
        elif backend == "before":
            dist.reduce_scatter_single(before, source, group=pg, op=op)
        else:
            relay(after, source, pg, op)

    for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG, dist.ReduceOp.PREMUL_SUM(0.25)):
        run("before", op)
        run("after", op)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    _check_graph(run, source, before, after)
    result = {"kind": name, "active": active, "input_kib": kib}
    for backend in args.backends:
        result[f"{backend}_us"] = _time_call(
            partial(run, backend), args.warmups, args.iterations
        )
    return result


def _check_graph(run, source, before, after):
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Relay scratch bookkeeping requires the mode used by its C++ suite.
    with torch.cuda.graph(graph, capture_error_mode="relaxed"):
        run("after")
    for _ in range(3):
        source.add_(1)
        run("before")
        graph.replay()
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def _models(rank, comm, args, dtype, shape, hsdp):
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=("replicate", "shard"))
    grid = mesh.mesh.tolist()
    shard_groups = grid
    replica_groups = [list(column) for column in zip(*grid)]
    torch.manual_seed(1234)
    template = torch.nn.Sequential(
        torch.nn.Linear(args.dim, args.dim, bias=False, device="cuda"),
        torch.nn.ReLU(),
        torch.nn.Linear(args.dim, args.dim, bias=False, device="cuda"),
    )
    models = {}
    for backend in ("before", "after"):
        model = copy.deepcopy(template)
        for linear in (model[0], model[2]):
            fully_shard(
                linear,
                mesh=mesh if hsdp else mesh["shard"],
                mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype),
            )
            if backend == "after":
                count = args.dim * args.dim // shape[1]
                linear.set_custom_reduce_scatter(
                    _adapter(
                        ShardedRelayReduceScatter,
                        comm,
                        rank,
                        shard_groups,
                        count,
                        dtype,
                    )
                )
                if hsdp:
                    linear.set_custom_all_reduce(
                        _adapter(
                            ShardedRelayAllReduce,
                            comm,
                            rank,
                            replica_groups,
                            count,
                            dtype,
                        )
                    )
        fully_shard(model, mesh=mesh if hsdp else mesh["shard"])
        models[backend] = model
    return models


def _parity(rank, models, args, hsdp):
    # Citrine C2: use foreach optimizer updates for the sharded parameters.
    optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
        for name, model in models.items()
    }
    generator = torch.Generator(device="cuda").manual_seed(rank + 2026)
    for _ in range(3):
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for microstep in range(2):
            inputs = torch.randn((8, args.dim), generator=generator, device="cuda")
            for model in models.values():
                if hsdp:
                    model.set_requires_all_reduce(microstep == 1)
                else:
                    model.set_requires_gradient_sync(microstep == 1)
                model(inputs).float().square().mean().backward()
        for left, right in zip(
            models["before"].parameters(), models["after"].parameters()
        ):
            torch.testing.assert_close(
                left.grad.to_local(), right.grad.to_local(), rtol=0.02, atol=2e-5
            )
        for optimizer in optimizers.values():
            optimizer.step()
        for left, right in zip(
            models["before"].parameters(), models["after"].parameters()
        ):
            torch.testing.assert_close(
                left.to_local(), right.to_local(), rtol=0.02, atol=2e-5
            )


def _training(rank, comm, args, dtype):
    rows = []
    size = dist.get_world_size()
    shapes = [(size // shard, shard) for shard in (2, 4) if size >= 2 * shard]
    # Each FSDP group is an independent workload; HSDP covers the full node.
    for hsdp, shape in ((hsdp, shape) for hsdp in (False, True) for shape in shapes):
        models = _models(rank, comm, args, dtype, shape, hsdp)
        _parity(rank, models, args, hsdp)
        result = {
            "kind": "hsdp_step" if hsdp else "fsdp_step",
            "mesh": shape,
            "parity": "passed",
        }
        inputs = torch.randn((8, args.dim), device="cuda")
        for backend in args.backends:
            model = models[backend]
            result[f"{backend}_us"] = _time_call(
                partial(_training_step, model, inputs), args.warmups, args.iterations
            )
        rows.append(result)
        del models
    return rows


def _training_step(model, inputs):
    model.zero_grad(set_to_none=True)
    model(inputs).float().square().mean().backward()


def _worker(rank, size, rendezvous, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(minutes=5),
    )
    os.environ["TORCHCOMM_RANK"] = str(rank)
    os.environ["TORCHCOMM_SIZE"] = str(size)
    comm = new_comm(
        "rcclx",
        torch.device("cuda", rank),
        "fsdp_relay_bench",
        store=dist.PrefixStore(
            "fsdp_relay", dist.distributed_c10d._get_default_store()
        ),
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    try:
        rows = _collectives(rank, comm, args, dtype)
        rows.extend(_training(rank, comm, args, dtype))
        if rank == 0:
            report = {"dtype": args.dtype, "gpus": size, "results": rows}
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        comm.finalize()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FSDP/HSDP relay correctness and before/after timings"
    )
    parser.add_argument(
        "--backend", choices=("both", "before", "after"), default="both"
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--sizes-kib", nargs="+", type=int, default=[4, 512, 2048, 73728, 147456]
    )
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="/tmp/fsdp_sharded_relay.json")
    args = parser.parse_args()
    if args.dim <= 0 or args.dim % 4 or args.iterations <= 0 or args.warmups < 0:
        parser.error(
            "dim must be positive and divisible by 4; iterations > 0; warmups >= 0"
        )
    if any(kib <= 0 for kib in args.sizes_kib):
        parser.error("sizes must be positive")
    size = torch.cuda.device_count()
    if size not in (4, 8):
        parser.error("Expose exactly four or eight AMD GPUs on one node")
    if not is_backend_built("rcclx"):
        parser.error(
            "Restore the native RCCLX reference stack and build with "
            "-c fsdp_relay.native_backend=true"
        )
    args.backends = ["before", "after"] if args.backend == "both" else [args.backend]
    with tempfile.TemporaryDirectory(prefix="fsdp_relay_") as scratch:
        mp.spawn(
            _worker,
            args=(size, f"file://{scratch}/store", args),
            nprocs=size,
            join=True,
        )


if __name__ == "__main__":
    main()
