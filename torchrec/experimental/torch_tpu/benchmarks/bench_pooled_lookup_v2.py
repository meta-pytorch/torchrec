#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @noautodeps -- manual perf harness; run via run_pod.sh, not a Buck target.

"""Perf A/B for the v2 pooled-lookup backward: searchsorted vs precomputed segments.

The backward derives each flat id's bag (`seg`) either by binary search over the
offsets (the default) or by a repeat over the lengths, selected by
`TPU_POOLED_BWD_MODE`. Both feed the same gather + dense scatter-add, so on identical
inputs this isolates the cost of the derivation alone:

  * bwd(old, searchsorted): binary search per id + gather + dense scatter-add
  * bwd(new, gather):       gather + dense scatter-add, with `seg` supplied
  * seg-build:              the repeat derivation on its own

Both derivations produce the same [num_rows, emb_dim] gradient with the same sync
overhead, so (old - new) is the searchsorted cost and seg-build is what replaces it.

Run:  ./run_pod.sh run bench_pooled_lookup_v2.py
"""

import time

import jax
import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401  (registers the "tpu" device + "tpu_dist" backend)
from torch_tpu._internal import pallas
from torchrec.experimental.torch_tpu.pallas.pooled_lookup_offset import (
    embedding_pooled_lookup_bwd_seg_jax,
    embedding_pooled_lookup_segments_jax,
    embedding_pooled_lookup_segments_searchsorted_jax,
)

embedding_pooled_lookup_segments_fn = pallas.jax_op(
    "pallas::_bench_v2_segments", embedding_pooled_lookup_segments_jax
)
embedding_pooled_lookup_bwd_fn = pallas.jax_op(
    "pallas::_bench_v2_bwd_seg", embedding_pooled_lookup_bwd_seg_jax
)


def _bwd_searchsorted_jax(
    grad_out: jax.Array,
    indices: jax.Array,
    offsets: jax.Array,
    num_rows: int,
    emb_dim: int,
) -> jax.Array:
    """The default (searchsorted) derivation, composed from the module's own pieces."""
    seg = embedding_pooled_lookup_segments_searchsorted_jax(offsets, indices.shape[0])
    return embedding_pooled_lookup_bwd_seg_jax(
        grad_out, indices, seg, num_rows, emb_dim
    )


_bwd_searchsorted_fn = pallas.jax_op(
    "pallas::_bench_v2_bwd_searchsorted", _bwd_searchsorted_jax
)


def _sync(x: torch.Tensor) -> float:
    # Reduce + host-copy forces the whole device program to complete before we stop timing.
    return float(x.sum())


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        _sync(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        _sync(fn())
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


def main() -> None:
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    torch.manual_seed(0)

    NUM_ROWS, EMB_DIM = 100_000, 128
    # (batch, fixed pooling factor)
    CASES = [(512, 32), (4096, 32), (4096, 64), (16384, 32)]

    if rank == 0:
        print(
            f"v2 pooled-lookup backward perf  (NUM_ROWS={NUM_ROWS}, EMB_DIM={EMB_DIM})",
            flush=True,
        )

    for batch, pool in CASES:
        total = batch * pool
        indices = torch.randint(0, NUM_ROWS, (total,), dtype=torch.int32, device="tpu")
        lengths = torch.full((batch,), pool, dtype=torch.int32, device="tpu")
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="tpu"),
                lengths.cumsum(0).to(torch.int32),
            ]
        )
        grad_out = torch.randn(batch, EMB_DIM, dtype=torch.float32, device="tpu")
        seg = embedding_pooled_lookup_segments_fn(offsets=offsets, total_ids=total)

        t_seg = _bench(
            lambda offsets=offsets, total=total: embedding_pooled_lookup_segments_fn(
                offsets=offsets, total_ids=total
            )
        )
        t_new = _bench(
            lambda seg=seg, grad_out=grad_out, indices=indices: embedding_pooled_lookup_bwd_fn(
                grad_out=grad_out,
                indices=indices,
                seg=seg,
                num_rows=NUM_ROWS,
                emb_dim=EMB_DIM,
            )
        )
        t_old = _bench(
            lambda grad_out=grad_out, indices=indices, offsets=offsets: _bwd_searchsorted_fn(
                grad_out=grad_out,
                indices=indices,
                offsets=offsets,
                num_rows=NUM_ROWS,
                emb_dim=EMB_DIM,
            )
        )

        if rank == 0:
            print(
                f"B={batch:>6} pool={pool:>3} total_ids={total:>8} | "
                f"seg-build={t_seg:7.3f}ms  bwd(old,searchsorted)={t_old:7.3f}ms  "
                f"bwd(new,gather)={t_new:7.3f}ms  fwd+bwd delta={t_seg + t_new - t_old:+7.3f}ms",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
