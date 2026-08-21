#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Steady-state training-throughput benchmark for the MLPerf DLRM-v2 model on the
torch_tpu stack (torch side only -- no JAX).

Same model as the reference JAX MLPerf DLRM benchmark: torchrec `DLRM_DCN`
(DCNv2 cross net) over a multi-hot `EmbeddingBagCollection` (pool sum = 214),
embedding_dim 128, dense arch [512,256,128], over arch [1024,1024,512,256,1],
per-element Adagrad. Cardinalities are the MLPerf-v2 `shrunk` (sum 30M, any
value >4M capped at 4M) or `canonical` (sum 228M) sets, selectable via
--cardinality. This is the third comparison point for the GPU<->TPU per-chip
gap-closure work: B200 GPU (MAST) vs v7x jte (JAX) vs **v7x torch_tpu (here)**.

The embeddings are ROW_WISE-sharded on the `UNFUSED_TPU` compute kernel under
`DistributedModelParallel` over the `tpu_dist` backend; the dense DCN/MLP path
runs on TPU via torch_tpu. Reports steady-state ms/step and K samples/s/chip
(per_chip_batch / step_time), the same metric as
`mlperf_dlrm_tpu/EXPERIMENT_LOG.md`.

NOTE (faithful-model caveat): the full MLPerf model uses POOLED, MULTI-HOT
embeddings, which produce an UNEVEN row-wise all2all. The torch_tpu RW path was
first brought up for `EmbeddingCollection`, 1-hot, EVEN splits only (see
`dlrm_bench_rw_tpu.py`). Running this faithful config may require torch_tpu
support for pooled multi-hot / uneven all2all that is still landing; that is an
intentional choice (measure the real model, surface the gap).

Run on the TPU pod via run_pod.sh (pushes torchrec, launches via run_dist_file.sh):

    ./run_pod.sh run train_dlrm_mlperf_tpu.py --cardinality shrunk --steps 50
    ./run_pod.sh run train_dlrm_mlperf_tpu.py --cardinality canonical
"""

import argparse
import logging
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
import torch_tpu  # noqa: F401  (registers the "tpu" device + "tpu_dist" backend)
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.types import ShardingEnv, ShardingPlan
from torchrec.models.dlrm import DLRM_DCN
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# ── MLPerf DLRM-v2 config (matches the reference JAX benchmark) ───────────────
EMBEDDING_DIM = 128
NUM_DENSE = 13
NUM_SPARSE = 26
DENSE_ARCH_LAYER_SIZES = [512, 256, 128]  # last must == EMBEDDING_DIM
OVER_ARCH_LAYER_SIZES = [1024, 1024, 512, 256, 1]
DCN_NUM_LAYERS = 3
DCN_LOW_RANK_DIM = 512
LR = 0.01
DENSE_DTYPE: torch.dtype = torch.float32

# Multi-hot pool sizes per feature (sum = 214). MLPerf-v2 `_MHS`.
MULTI_HOT_SIZES = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]

# Shrunk Criteo cardinalities (any value >4M capped at 4M). Sum 30M.
NEPF_SHRUNK = [
    4000000,
    4000000,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    4000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    4000000,
    4000000,
    4000000,
    585935,
    12972,
    108,
    36,
]
# Canonical MLPerf-v2 cardinalities. Sum 228,487,552.
NEPF_CANONICAL = [
    40000000,
    39060192,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    39979771,
    25641295,
    39664984,
    585935,
    12972,
    108,
    36,
]


# torch_tpu materialization (force lazy ops to execute, optionally blocking) --
# same primitive dlrm_bench_rw_tpu.py uses to make wall-clock timing meaningful.
try:
    from torch_tpu._internal import sync as _tpu_sync  # type: ignore[import-not-found]

    def _materialize(tensor: torch.Tensor | None = None) -> None:
        if tensor is None:
            _tpu_sync.synchronize(wait=True)
        else:
            _tpu_sync.synchronize(tensor, wait=True)

except Exception:  # noqa: BLE001
    # Warn loudly: this is a timing harness, so a silent fallback risks publishing
    # numbers that only measure dispatch. The fallback below still blocks, but say so.
    logging.warning(
        "torch_tpu._internal.sync unavailable; falling back to torch.tpu.synchronize() "
        "for step-boundary flushes. Verify reported ms/step before trusting it."
    )

    def _materialize(tensor: torch.Tensor | None = None) -> None:
        # Must still FLUSH on the no-tensor call: run_step()/warmup call _materialize()
        # with no argument purely to drain pending TPU work before the clock is read,
        # so a no-op here would silently report meaningless step times.
        if tensor is None:
            # pyre-ignore[16]: torch.tpu is registered at runtime by torch_tpu, which
            # is a pod-only dependency and so is invisible to the type checker.
            torch.tpu.synchronize()
        else:
            tensor.detach().to("cpu")


def _round_up_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_tables(nepf: list[int], world_size: int) -> list[EmbeddingBagConfig]:
    """One EmbeddingBagConfig per sparse feature. num_embeddings is rounded up to a
    multiple of world_size so the RW shards are even-sized."""
    tables = []
    for i, rows in enumerate(nepf):
        padded = _round_up_div(rows, world_size) * world_size
        tables.append(
            EmbeddingBagConfig(
                name=f"table{i}",
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=padded,
                feature_names=[f"feature{i}"],
            )
        )
    return tables


def _build_sharded(
    tables: list[EmbeddingBagConfig],
    device: torch.device,
    pg: dist.ProcessGroup,
    world_size: int,
    dcn_num_layers: int,
    dcn_low_rank_dim: int,
) -> nn.Module:
    """Faithful MLPerf DLRM_DCN with a RW / UNFUSED_TPU EmbeddingBagCollection under DMP."""
    ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
    model = DLRM_DCN(
        embedding_bag_collection=ebc,
        dense_in_features=NUM_DENSE,
        dense_arch_layer_sizes=DENSE_ARCH_LAYER_SIZES,
        over_arch_layer_sizes=OVER_ARCH_LAYER_SIZES,
        dcn_num_layers=dcn_num_layers,
        dcn_low_rank_dim=dcn_low_rank_dim,
        dense_device=device,
    )
    plan = ShardingPlan(
        {
            # fqn of the EBC inside DLRM_DCN: sparse_arch.embedding_bag_collection
            "sparse_arch.embedding_bag_collection": construct_module_sharding_plan(
                model.sparse_arch.embedding_bag_collection,
                per_param_sharding={
                    t.name: row_wise(
                        compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU.value
                    )
                    for t in tables
                },
                # pyre-ignore[6]: ModuleSharder is invariant; the EBC sharder is
                # the correct sharder for this module type.
                sharder=EmbeddingBagCollectionSharder(),
                local_size=1,
                world_size=world_size,
                device_type="tpu",
            )
        }
    )
    return DistributedModelParallel(
        module=model,
        env=ShardingEnv.from_process_group(pg),
        device=device,
        plan=plan,
        # pyre-ignore[6]: see note above on ModuleSharder invariance.
        sharders=[EmbeddingBagCollectionSharder()],
    )


def make_multihot_kjt(
    features: list[str],
    nepf: list[int],
    per_chip_batch: int,
    generator: torch.Generator,
) -> KeyedJaggedTensor:
    """A realistic multi-hot KJT: feature i has MULTI_HOT_SIZES[i] random ids per sample.

    Values span each table's full cardinality, so the RW bucketize routes them across
    all ranks (an UNEVEN all2all -- the faithful MLPerf regime). Values vary per call
    (seed per step) so the looked-up rows change every step.

    Citrine C3 exception (deliberate): these ids are built on CPU and moved to the TPU by
    the caller, rather than created directly on device. `generator` is a CPU
    `torch.Generator` -- passing it to a device-side `torch.randint` is not allowed, and
    the per-step reseeding is what makes each step look up different rows. This also
    mirrors a real input pipeline, where sparse ids arrive from the host. The resulting
    host->device copy is done OUTSIDE the timed region in `run_step` (before `t0`), so it
    does not inflate the reported ms/step.
    """
    values_list: list[torch.Tensor] = []
    lengths_list: list[torch.Tensor] = []
    for i in range(len(features)):
        pool = MULTI_HOT_SIZES[i]
        n = per_chip_batch * pool
        values_list.append(
            torch.randint(0, nepf[i], (n,), generator=generator, dtype=torch.int32)
        )
        lengths_list.append(torch.full((per_chip_batch,), pool, dtype=torch.int32))
    return KeyedJaggedTensor.from_lengths_sync(
        keys=features,
        values=torch.cat(values_list),
        lengths=torch.cat(lengths_list),
    )


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLPerf DLRM-v2 torch_tpu train-perf benchmark"
    )
    p.add_argument(
        "--cardinality",
        choices=["shrunk", "canonical"],
        default="shrunk",
        help="Embedding table cardinalities: shrunk (sum 30M) or canonical (sum 228M).",
    )
    p.add_argument(
        "--per-chip-batch",
        type=int,
        default=4096,
        help="Local batch per rank/chip. Global batch = world_size * this.",
    )
    p.add_argument("--steps", type=int, default=50, help="Timed steps.")
    p.add_argument(
        "--warmup", type=int, default=10, help="Warmup steps (compile/autotune)."
    )
    p.add_argument("--dcn-num-layers", type=int, default=DCN_NUM_LAYERS)
    p.add_argument("--dcn-low-rank", type=int, default=DCN_LOW_RANK_DIM)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("tpu")
    pg = dist.group.WORLD
    assert pg is not None, "default process group missing after init_process_group"
    assert world_size > 1, "benchmark needs WORLD_SIZE > 1 for RW sharding"
    # Guard the timed loop: with --steps 0 the stats below divide by len(step_ms) == 0.
    assert args.steps > 0, "--steps must be > 0 to report timing statistics"
    assert args.warmup >= 0, "--warmup cannot be negative"

    nepf = NEPF_CANONICAL if args.cardinality == "canonical" else NEPF_SHRUNK
    tables = _build_tables(nepf, world_size)
    padded_nepf = [t.num_embeddings for t in tables]
    features = [t.feature_names[0] for t in tables]
    per_chip_batch = args.per_chip_batch

    sharded_model = _build_sharded(
        tables, device, pg, world_size, args.dcn_num_layers, args.dcn_low_rank
    )

    # Fixed dense inputs + labels (constant across steps); only sparse ids vary.
    dense_x = torch.randn(per_chip_batch, NUM_DENSE, dtype=DENSE_DTYPE, device=device)
    # Citrine C3: create directly on device rather than building on CPU and moving.
    labels = (torch.rand(per_chip_batch, 1, device=device) > 0.5).to(DENSE_DTYPE)

    # Per-element (D=128 momenta/row) Adagrad over all params (dense + unfused
    # embeddings), matching MLPerf's EXACT_ADAGRAD (see EXPERIMENT_LOG A.1).
    # Citrine C2: foreach=True for multi-tensor execution. Safe here -- the UNFUSED_TPU
    # kernel's backward is a dense scatter-add, so no sparse grads are involved.
    opt = torch.optim.Adagrad(sharded_model.parameters(), lr=LR, foreach=True)

    kjt_gen = torch.Generator()

    def run_step(step: int, timed: bool) -> float:
        # The KJT must already be on the TPU: the sharded EBC input_dist bucketizes (RW)
        # and all2alls it over the `tpu_dist` process group, which has no CPU backend
        # ("No backend type associated with device type cpu" if the ids stay on host).
        kjt_gen.manual_seed(args.seed + 1 + step)
        kjt = make_multihot_kjt(features, padded_nepf, per_chip_batch, kjt_gen).to(
            device
        )
        opt.zero_grad()
        t0 = time.perf_counter()
        logits = sharded_model(dense_x, kjt)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
        if timed:
            _materialize()  # flush fwd+bwd+opt for this step
        return time.perf_counter() - t0

    for w in range(args.warmup):
        run_step(w, timed=False)
    _materialize()

    step_ms: list[float] = []
    for s in range(args.steps):
        step_ms.append(run_step(args.warmup + s, timed=True) * 1e3)

    # Aggregate across ranks: MAX = straggler-bound wall-clock, SUM/world = avg rank.
    local_mean = torch.tensor([sum(step_ms) / len(step_ms)], device=device)
    xmax = local_mean.clone()
    dist.all_reduce(xmax, op=dist.ReduceOp.MAX)
    xsum = local_mean.clone()
    dist.all_reduce(xsum, op=dist.ReduceOp.SUM)
    xrank_max = xmax.cpu().item()
    xrank_mean = (xsum / world_size).cpu().item()

    if rank == 0:
        # K samples/s/chip = per_chip_batch / step_time_s / 1000 (straggler-bound).
        kspc = per_chip_batch / (xrank_max / 1e3) / 1e3
        print("=" * 72, flush=True)
        print(
            f"MLPerf DLRM-v2 torch_tpu train-perf  (world={world_size}, "
            f"per_chip_batch={per_chip_batch}, global_bs={world_size * per_chip_batch}, "
            f"cardinality={args.cardinality} sum={sum(nepf):,}, "
            f"tables={NUM_SPARSE}, dim={EMBEDDING_DIM}, pool_sum={sum(MULTI_HOT_SIZES)}, "
            f"DCN[{args.dcn_num_layers}x{args.dcn_low_rank}])",
            flush=True,
        )
        print(
            f"{args.steps} timed steps after {args.warmup} warmup:",
            flush=True,
        )
        print(
            f"  ms/step   xrank-mean={xrank_mean:.3f}  xrank-max={xrank_max:.3f}  "
            f"r0-p50={_median(step_ms):.3f}",
            flush=True,
        )
        print(f"  K samples/s/chip (straggler-bound) = {kspc:.1f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
