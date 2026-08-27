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

The embeddings are COLUMN_WISE-sharded on the `UNFUSED_TPU` compute kernel under
`DistributedModelParallel` over the `tpu_dist` backend; the dense DCN/MLP path
runs on TPU via torch_tpu. The embedding lookup (forward gather) runs on the
SparseCore (`LOOKUP_MODE=v1_sc`, set in `main`); its unfused backward runs on the
TensorCore. Reports steady-state ms/step and K samples/s/chip
(per_chip_batch / step_time), the same metric as
`mlperf_dlrm_tpu/EXPERIMENT_LOG.md`.

WHY COLUMN_WISE (not row_wise): the `tpu_dist` backend only implements EVEN
`all_to_all_single`. Row-wise bucketizes ids across ranks by row, giving
data-dependent, uneven per-partition counts -> uneven all2all (unsupported).
Column-wise reuses the table-wise input dist (`KJTAllToAll`, no bucketization):
each table's `embedding_dim` is split across all ranks, so every rank owns every
feature and each destination receives exactly `per_chip_batch * sum(MULTI_HOT_SIZES)`
ids -> an EVEN all2all, with no id-dropping. Requires `EMBEDDING_DIM % world_size
== 0` AND `(EMBEDDING_DIM // world_size) % 4 == 0` -- CW rounds each shard's column
width up to a multiple of 4, so a narrower split would place fewer, wider shards on a
subset of ranks and the all2all would go uneven again. With dim=128 that caps the
benchmark at world_size <= 32.

Run on the TPU pod via run_pod.sh (pushes torchrec, launches via run_dist_file.sh):

    ./run_pod.sh run train_dlrm_mlperf_tpu.py --cardinality shrunk --steps 50
    ./run_pod.sh run train_dlrm_mlperf_tpu.py --cardinality canonical
"""

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
import torch_tpu  # noqa: F401  (registers the "tpu" device + "tpu_dist" backend)
import torchrec.experimental.torch_tpu.pallas.dispatcher  # noqa: F401  registers fbgemm  TPU kernels
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
)
from torchrec.distributed.types import ShardingEnv, ShardingPlan

# Registers the TPU (SparseCore) impls of torchrec::embedding_lookup{,_backward};
# pallas/ops.py only registers the CPU ones, so the UNFUSED_TPU lookup needs this.
from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401
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


def _build_tables(nepf: list[int]) -> list[EmbeddingBagConfig]:
    """One EmbeddingBagConfig per sparse feature. Column-wise sharding keeps every
    row on every rank (only the embedding_dim is split), so num_embeddings needs no
    world_size padding (unlike row-wise)."""
    tables = []
    for i, rows in enumerate(nepf):
        tables.append(
            EmbeddingBagConfig(
                name=f"table{i}",
                embedding_dim=EMBEDDING_DIM,
                num_embeddings=rows,
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
    """Faithful MLPerf DLRM_DCN with a CW / UNFUSED_TPU EmbeddingBagCollection under DMP."""
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
    # Split each table's embedding_dim across ALL ranks (one column shard per rank),
    # so every rank owns every feature -> the sparse input all2all is EVEN.
    ranks = list(range(world_size))
    plan = ShardingPlan(
        {
            # fqn of the EBC inside DLRM_DCN: sparse_arch.embedding_bag_collection
            "sparse_arch.embedding_bag_collection": construct_module_sharding_plan(
                model.sparse_arch.embedding_bag_collection,
                per_param_sharding={
                    t.name: column_wise(
                        ranks=ranks,
                        compute_kernel=EmbeddingComputeKernel.UNFUSED_TPU.value,
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
    device: torch.device,
) -> KeyedJaggedTensor:
    """A realistic multi-hot KJT: feature i has MULTI_HOT_SIZES[i] random ids per sample.

    Values span each table's full cardinality. Under column-wise sharding the KJT is
    routed feature-wise (no bucketize) and, since every rank owns every feature, the
    all2all is EVEN. Values vary per call (seed per step) so looked-up rows change
    every step.

    Citrine C3 exception (deliberate): these ids are built on CPU and moved to the TPU
    at the end of this function, rather than created directly on device. `generator` is a CPU
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
    ).to(device)


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
    p.add_argument(
        "--profile-dir",
        type=str,
        default="",
        help="If set, capture an xprof xplane trace of the timed steps to this dir ",
    )
    p.add_argument(
        "--pooled-bwd-mode",
        choices=["searchsorted", "repeat"],
        default="searchsorted",
        help="Pooled backward segment derivation: searchsorted (default, no "
        "precondition) or repeat (faster, requires offsets[-1] == len(indices)).",
    )
    p.add_argument("--dcn-num-layers", type=int, default=DCN_NUM_LAYERS)
    p.add_argument("--dcn-low-rank", type=int, default=DCN_LOW_RANK_DIM)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _skip_ddp_shape_verify_on_tpu() -> None:
    """No-op DDP's startup param-shape verification on TPU (multi-host workaround).

    `DistributedDataParallel.__init__` calls `_verify_param_shape_across_processes`,
    which does a device->CPU transfer whose StableHLO lowering fails on the torch_tpu
    multi-host runtime (32 ranks): "transfer to 'cpu' device failed ... StableHLO
    failed". The check is only a sanity guard, and DMP's dense arch is deterministically
    replicated (identical shapes on every rank), so skipping it is safe. Single-host is
    unaffected (the check already passes there). The durable fix belongs in torch_tpu's
    device->CPU lowering; this unblocks the multi-host benchmark run.
    """
    # pyre-ignore[16]: torch.tpu is registered at runtime by torch_tpu, which is a
    # pod-only dependency and so is invisible to the type checker.
    if not (hasattr(torch, "tpu") and torch.tpu.is_available()):
        return
    import torch.nn.parallel.distributed as _ddp

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    # DDP resolves the name in its own module namespace, so patch it there.
    _ddp._verify_param_shape_across_processes = _noop


def main() -> None:
    args = parse_args()
    # Run the embedding lookup (forward gather) on the SparseCore. Set before the
    # first lookup; single_lookup._lookup_mode() reads LOOKUP_MODE at call time.
    # The unfused backward always runs on the TensorCore.
    os.environ["LOOKUP_MODE"] = "v1_sc"
    # How the pooled backward derives each id's bag. pooled_lookup_offset reads this at
    # trace time, so it must be set before the first backward compiles. Defaults to
    # searchsorted, matching the kernel's own default.
    os.environ["TPU_POOLED_BWD_MODE"] = args.pooled_bwd_mode
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("tpu")
    pg = dist.group.WORLD
    assert pg is not None, "default process group missing after init_process_group"
    assert world_size > 1, "benchmark needs WORLD_SIZE > 1 for CW sharding"
    # CW rounds each shard's column width up to a multiple of 4 (sharding_plan
    # _find_base_dim); dim//world_size must be a multiple of 4 so CW places exactly
    # one shard per rank (every rank owns every feature -> even input all2all).
    assert EMBEDDING_DIM % world_size == 0 and (EMBEDDING_DIM // world_size) % 4 == 0, (
        f"column_wise across all ranks needs EMBEDDING_DIM ({EMBEDDING_DIM}) "
        f"// world_size ({world_size}) to be a multiple of 4"
    )
    # Guard the timed loop: with --steps 0 the stats below divide by len(step_ms) == 0.
    assert args.steps > 0, "--steps must be > 0 to report timing statistics"
    assert args.warmup >= 0, "--warmup cannot be negative"

    nepf = NEPF_CANONICAL if args.cardinality == "canonical" else NEPF_SHRUNK
    tables = _build_tables(nepf)
    table_rows = [t.num_embeddings for t in tables]
    features = [t.feature_names[0] for t in tables]
    per_chip_batch = args.per_chip_batch

    # torch_tpu multi-host workaround: DDP's startup shape-verify can't lower its
    # device->CPU transfer at 32 ranks; skip it (dense arch shapes are replicated).
    _skip_ddp_shape_verify_on_tpu()

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
        # KJT is built on host then moved to device; the sharded EBC input_dist
        # all2alls it feature-wise (CW: even). It must already be on device -- the
        # splits all2all inside KJTAllToAll runs on the KJT's device.
        kjt_gen.manual_seed(args.seed + 1 + step)
        kjt = make_multihot_kjt(features, table_rows, per_chip_batch, kjt_gen, device)
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

    # Optional xprof capture of the timed steps. jax.profiler records the TPU device
    # timeline (SC/TC ops) via libtpu regardless of framework; rank 0 captures.
    profiling = bool(args.profile_dir) and rank == 0
    if profiling:
        import jax

        os.makedirs(args.profile_dir, exist_ok=True)
        jax.profiler.start_trace(args.profile_dir)

    step_ms: list[float] = []
    for s in range(args.steps):
        step_ms.append(run_step(args.warmup + s, timed=True) * 1e3)

    if profiling:
        _materialize()  # flush pending TPU work into the trace window
        # pyre-ignore[18]: jax imported above under the same guard
        jax.profiler.stop_trace()
        print(f"xprof trace written to {args.profile_dir}", flush=True)

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
