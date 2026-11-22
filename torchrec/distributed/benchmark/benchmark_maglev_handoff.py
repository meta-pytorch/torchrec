#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA microbenchmark for Maglev's paired cross-stage handoff.

The default payload matches the Maglev benchmark's 8192 x 4096 FP32 activation,
split into 26 contiguous tensors. It compares the fused payload handoff with
per-tensor P2P, including packing, receive allocation, and copying received
slots out of the payload.

Run locally on two GPUs with:
    buck2 run @fbcode//mode/opt \
      fbcode//torchrec/distributed/benchmark:benchmark_maglev_handoff -- \
      --num_benchmarks=50
"""

import json
import os
from dataclasses import dataclass
from typing import Any, cast, Literal

import torch
import torch.distributed as dist
from torchrec.distributed.benchmark.base import (
    BenchFuncConfig,
    benchmark_func,
    BenchmarkResult,
    cmd_conf,
)
from torchrec.distributed.maglev.stage import (
    _allocate_handoff_payload,
    _pack_handoff_payload,
    _unpack_handoff_payload,
)
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    run_single_process_func,
    SingleProcessContext,
)

_Transport = Literal["per_tensor", "fused"]
_TRANSPORTS: tuple[_Transport, ...] = ("per_tensor", "fused")


@dataclass
class HandoffOptions(BenchFuncConfig):
    name: str = "maglev_handoff"
    world_size: int = 2
    num_profiles: int = 0
    num_benchmarks: int = 50
    batch_size: int = 8192
    layer_dim: int = 4096
    num_tensors: int = 26
    all_rank_traces: bool = True
    output_json: bool = False


def _tensor_shapes(
    batch_size: int, layer_dim: int, num_tensors: int
) -> list[tuple[int, int]]:
    if batch_size <= 0 or layer_dim <= 0:
        raise ValueError("batch_size and layer_dim must be positive")
    if num_tensors <= 0 or num_tensors > layer_dim:
        raise ValueError("num_tensors must be between 1 and layer_dim")

    width, wider_tensors = divmod(layer_dim, num_tensors)
    return [
        (batch_size, width + int(index < wider_tensors)) for index in range(num_tensors)
    ]


def _exchange(
    _bench_inputs: list[dict[str, Any]],
    *,
    send_tensors: list[torch.Tensor],
    transport: _Transport,
    peer: int,
    pg: dist.ProcessGroup,
) -> list[torch.Tensor]:
    if transport == "per_tensor":
        recv_tensors = [torch.empty_like(tensor) for tensor in send_tensors]
        ops = [dist.P2POp(dist.isend, tensor, peer, pg) for tensor in send_tensors] + [
            dist.P2POp(dist.irecv, tensor, peer, pg) for tensor in recv_tensors
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        return recv_tensors

    send_payload = _pack_handoff_payload(send_tensors)
    recv_payload, recv_views = _allocate_handoff_payload(
        [(tensor.shape, tensor.dtype) for tensor in send_tensors],
        send_tensors[0].device,
    )
    assert send_payload is not None and recv_payload is not None
    ops = [
        dist.P2POp(dist.isend, send_payload, peer, pg),
        dist.P2POp(dist.irecv, recv_payload, peer, pg),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return _unpack_handoff_payload(recv_views)


def _validate_received(
    received: list[torch.Tensor], peer: int, transport: _Transport
) -> None:
    for tensor in received:
        expected = torch.full_like(tensor, float(peer + 1))
        if not torch.equal(tensor, expected):
            raise AssertionError(f"{transport} handoff failed validation")


def _measure_transport(
    run_option: HandoffOptions,
    rank: int,
    world_size: int,
    send_tensors: list[torch.Tensor],
    peer: int,
    transport: _Transport,
    pg: dist.ProcessGroup,
) -> BenchmarkResult:
    return benchmark_func(
        func_to_benchmark=_exchange,
        bench_inputs=[],
        prof_inputs=[],
        benchmark_func_kwargs={
            "send_tensors": send_tensors,
            "transport": transport,
            "peer": peer,
            "pg": pg,
        },
        pg=pg,
        **run_option.benchmark_func_kwargs(
            name=f"{run_option.name}_{transport}", rank=rank, world_size=world_size
        ),
    )


def _rank_runner(
    *,
    ctx: SingleProcessContext,
    rank: int,
    world_size: int,
    run_option: HandoffOptions,
) -> tuple[BenchmarkResult, ...]:
    if world_size != 2 or run_option.world_size != 2:
        raise ValueError("the Maglev handoff microbenchmark requires world_size=2")
    if run_option.num_benchmarks <= 0:
        raise ValueError("num_benchmarks must be positive")
    if run_option.device_type == "cuda" and ctx.device.type != "cuda":
        raise RuntimeError("CUDA mode requires two visible GPUs")
    pg = dist.group.WORLD
    if pg is None:
        raise RuntimeError("default process group was not initialized")

    peer = 1 - rank
    send_tensors = [
        torch.full(shape, float(rank + 1), dtype=torch.float32, device=ctx.device)
        for shape in _tensor_shapes(
            run_option.batch_size, run_option.layer_dim, run_option.num_tensors
        )
    ]
    results: list[BenchmarkResult] = []
    for transport in _TRANSPORTS:
        received = _exchange(
            [],
            send_tensors=send_tensors,
            transport=transport,
            peer=peer,
            pg=pg,
        )
        _validate_received(received, peer, transport)
        results.append(
            _measure_transport(
                run_option, rank, world_size, send_tensors, peer, transport, pg
            )
        )
    return tuple(results)


def _run(run_option: HandoffOptions) -> list[tuple[BenchmarkResult, ...]]:
    if run_option.device_type != "cuda":
        raise ValueError("the Maglev handoff microbenchmark requires CUDA/NCCL")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < local_world_size:
        raise RuntimeError(
            f"CUDA mode needs {local_world_size} local GPUs, found "
            f"{torch.cuda.device_count()}"
        )
    backend = "cpu:gloo,cuda:nccl"
    if "RANK" in os.environ:
        result = run_single_process_func(
            func=_rank_runner,
            backend=backend,
            run_option=run_option,
        )
        return [cast(tuple[BenchmarkResult, ...], result)]

    return cast(
        list[tuple[BenchmarkResult, ...]],
        run_local_multi_process_func(
            func=_rank_runner,
            world_size=run_option.world_size,
            backend=backend,
            run_option=run_option,
        ),
    )


@cmd_conf
def main(run_option: HandoffOptions) -> None:
    rank_results = _run(run_option)
    if "RANK" in os.environ and int(os.environ["RANK"]) != 0:
        return

    results = rank_results[0]
    if run_option.output_json:
        print(
            json.dumps(
                {result.short_name: result.to_dict() for result in results}, indent=2
            )
        )
    else:
        for result in results:
            print(result.prettify())


if __name__ == "__main__":
    # pyrefly: ignore[not-callable]
    main()
