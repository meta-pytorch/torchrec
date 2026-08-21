#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated correctness check for the SparseCore recat gather (`_sc_gather_1d`).

Gathers a known permutation on the SparseCore and compares to a plain torch gather.
Run: ./run_pod.sh run test_sc_gather.py
"""

import torch
import torch.distributed as dist

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
import torch_tpu  # noqa: F401
from torchrec.experimental.torch_tpu.pallas.recat_sc import _sc_gather_1d


def main() -> None:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    dev = torch.device("tpu")

    n = 1024
    gen = torch.Generator().manual_seed(0)
    values_cpu = (torch.arange(n, dtype=torch.int64) * 7) % 997
    idx_cpu = torch.randperm(n, generator=gen)  # a full permutation

    values = values_cpu.to(dev)
    got = _sc_gather_1d(values, idx_cpu.to(torch.int32).to(dev)).cpu()
    expected = values_cpu[idx_cpu]  # reference: values[idx]

    ok = torch.equal(got, expected)
    if rank == 0:
        print(f"SC_GATHER_MATCH: {ok}", flush=True)
        if not ok:
            nmis = int((got != expected).sum())
            print(f"  mismatched {nmis}/{n}", flush=True)
            print(f"  idx[:12]      {idx_cpu[:12].tolist()}", flush=True)
            print(f"  got[:12]      {got[:12].tolist()}", flush=True)
            print(f"  expected[:12] {expected[:12].tolist()}", flush=True)
            print(f"  values[:12]   {values_cpu[:12].tolist()}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
