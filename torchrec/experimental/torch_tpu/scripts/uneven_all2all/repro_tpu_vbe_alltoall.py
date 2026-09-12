#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
"""Reproduce the 128-rank TorchTPU uneven all-to-all failure."""

from __future__ import annotations

import importlib.metadata

import torch
import torch.distributed as dist


BASE_ELEMENTS = 8
VARIATIONS = 8


def _run_case(case: str) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if case == "equal":
        # Each rank gets same number of elements
        split_sizes = [BASE_ELEMENTS] * world_size
    else:
        # Different ranks get different number of elements
        split_sizes = [BASE_ELEMENTS + rank % VARIATIONS for rank in range(world_size)]

    local_split = split_sizes[rank]
    input_splits = [local_split] * world_size
    output_splits = split_sizes

    # Create a tensor such that each rank r_i is assigned a certain range [r_i * 1M, (r_i +1) * 1M)
    input_tensor = rank * 1_000_000 + torch.arange(
        world_size * local_split, device="tpu", dtype=torch.int32
    )
    output_tensor = torch.empty(sum(output_splits), device="tpu", dtype=torch.int32)

    print(
        f"rank={rank} case={case} start input_splits={input_splits[:8]} "
        f"output_splits={output_splits[:8]}",
        flush=True,
    )
    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
    )
    print(f"rank={rank} case={case} collective complete", flush=True)
    actual = output_tensor.cpu()

    expected = torch.cat(
        [
            r_i * 1_000_000
            + torch.arange(rank * size, (rank + 1) * size, dtype=torch.int32)
            for r_i, size in enumerate(split_sizes)
        ]
    )
    torch.testing.assert_close(actual, expected)
    print(f"rank={rank} case={case} PASS", flush=True)


def main() -> None:
    import torch_tpu  # noqa: F401  # pyre-ignore[21]

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(
            "torch="
            + str(torch.__version__)
            + "; torch_tpu="
            + importlib.metadata.version("torch-tpu"),
            flush=True,
        )
        print(
            f"world_size={world_size}; equal is the non-VBE control, uneven is "
            "the VBE-like collective",
            flush=True,
        )

    for case in ("equal", "uneven"):
        _run_case(case)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
