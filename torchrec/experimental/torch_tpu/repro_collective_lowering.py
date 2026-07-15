#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal repro for a torch_tpu multi-host all_to_all lowering failure.

A plain `torch.distributed.all_to_all_single` on the `tpu` device fails to lower at
world=32 (multi-host) and aborts every rank:

    loc("distributed.all_to_all_single/..."): error: replica id #0 seen more than once
    LLVM ERROR: Failed to infer result type(s)      (fatal -> SIGABRT on all ranks)

The identical op at world=8 (single host) lowers cleanly, so the failure is specific to
the world=32 cross-host device mesh.

Launch with the accompanying run_repro.sh (see that script for the JobSet prereq):
    ./run_repro.sh            # multi-host (world=32) -- reproduces the failure
    ./run_repro.sh single     # single-host (world=8) -- lowers cleanly
"""

# @noautodeps -- manual repro script; run via run_repro.sh, not a Buck target.

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  (registers the "tpu" device + "tpu_dist" backend)


def _run(name: str, fn) -> None:
    """Run one collective and report per-rank; catchable errors are printed, a fatal
    LLVM lowering error aborts the process (which is itself the repro)."""
    rank = dist.get_rank()
    try:
        fn()
        if rank == 0:
            print(f"[OK] {name}", flush=True)
    except Exception as e:  # noqa: BLE001 -- repro: surface the lowering error verbatim
        print(f"[FAIL rank{rank}] {name}: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("tpu")
    if rank == 0:
        print(f"repro: world_size={world}", flush=True)

    # all_to_all_single fails to lower at world=32 (fatal LLVM infer error).
    def _all_to_all() -> None:
        inp = torch.arange(world * 4, dtype=torch.float32, device=device)
        out = torch.empty_like(inp)
        dist.all_to_all_single(out, inp)
        _ = out.cpu()

    _run("all_to_all_single", _all_to_all)

    dist.barrier()
    if rank == 0:
        print("repro: done (all_to_all lowered)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
