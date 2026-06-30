#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Benchmark for CUDA Graphs on a single-GPU, unsharded ``DLRM`` / ``DLRM_DCN``.

It builds an eager model, measures ms/step, enables CUDA Graphs on the
dense compute path via ``DLRM.compile_dense_path()`` (Step 5), re-measures, and
reports the speedup, the max abs output diff (numerical parity), and the GPU
memory delta. With ``--trace-dir`` it also exports before/after Kineto traces so
you can confirm ``cudaGraphLaunch`` replaced long ``cudaLaunchKernel`` runs.

CUDA Graphs only engage on a CUDA device; on CPU the script still runs (compile
falls back to no graph capture) so it can be smoke-tested without a GPU.

Example::

    buck2 run @fbcode//mode/opt fbcode//torchrec/models:cuda_graph_benchmark -- \\
        --model dlrm --batch-size 1024 --num-sparse-features 26 \\
        --embedding-dim 64 --dense-in-features 13 --trace-dir /tmp/dlrm_cudagraph
"""

import argparse
import logging
import os
from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch
from torchrec.models.cuda_graph_utils import (
    build_dlrm,
    build_ebc,
    collect_outputs,
    generate_batch,
)
from torchrec.models.dlrm import DLRM, DLRM_DCN
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def _build_model(args: argparse.Namespace, device: torch.device) -> DLRM:
    """Build an unsharded DLRM / DLRM_DCN on ``device`` from the parsed args."""
    if args.model != "dlrm_dcn":
        return build_dlrm(
            device,
            embedding_dim=args.embedding_dim,
            num_sparse_features=args.num_sparse_features,
            num_embeddings=args.num_embeddings,
            dense_in_features=args.dense_in_features,
        )
    ebc = build_ebc(
        device,
        embedding_dim=args.embedding_dim,
        num_sparse_features=args.num_sparse_features,
        num_embeddings=args.num_embeddings,
    )
    model = DLRM_DCN(
        embedding_bag_collection=ebc,
        dense_in_features=args.dense_in_features,
        # Final dense_arch layer size must equal embedding_dim (DLRM constraint).
        dense_arch_layer_sizes=[args.embedding_dim * 2, args.embedding_dim],
        over_arch_layer_sizes=[512, 256, 1],
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim,
        dense_device=device,
    )
    return model.to(device).eval()


def _generate_inputs(
    args: argparse.Namespace, device: torch.device
) -> Tuple[torch.Tensor, KeyedJaggedTensor]:
    """Generate one synthetic (dense, sparse-KJT) batch directly on ``device``."""
    return generate_batch(
        device,
        batch_size=args.batch_size,
        num_sparse_features=args.num_sparse_features,
        num_embeddings=args.num_embeddings,
        dense_in_features=args.dense_in_features,
        pooling_factor=args.pooling_factor,
    )


@contextmanager
def _maybe_autocast(enabled: bool, device: torch.device) -> Iterator[None]:
    """fp16 autocast on CUDA. Applied identically to warmup and measured loops so
    the CUDA graph is captured and replayed under the same context (see the
    runbook "Hang on first replay" failure mode)."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


@torch.inference_mode()
def _measure_ms_per_step(
    model: DLRM,
    dense: torch.Tensor,
    kjt: KeyedJaggedTensor,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    """Warmup then time ``warmup_iters`` + ``bench_iters`` forward passes."""
    with _maybe_autocast(args.amp, device):
        for _ in range(args.warmup_iters):
            torch.compiler.cudagraph_mark_step_begin()
            model(dense_features=dense, sparse_features=kjt)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with _maybe_autocast(args.amp, device):
            for _ in range(args.bench_iters):
                torch.compiler.cudagraph_mark_step_begin()
                model(dense_features=dense, sparse_features=kjt)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.bench_iters

    # CPU fallback (no CUDA graph capture; timing is wall-clock only).
    import time

    t0 = time.perf_counter()
    with _maybe_autocast(args.amp, device):
        for _ in range(args.bench_iters):
            torch.compiler.cudagraph_mark_step_begin()
            model(dense_features=dense, sparse_features=kjt)
    return (time.perf_counter() - t0) * 1000.0 / args.bench_iters


@torch.inference_mode()
def _collect_outputs(
    model: DLRM,
    batches: List[Tuple[torch.Tensor, KeyedJaggedTensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[torch.Tensor]:
    """Collect one output per batch under the requested precision."""
    with _maybe_autocast(args.amp, device):
        return collect_outputs(model, batches)


def _export_trace(
    model: DLRM,
    dense: torch.Tensor,
    kjt: KeyedJaggedTensor,
    path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Export a Kineto/Chrome trace of a few forward passes to ``path``."""
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), _maybe_autocast(args.amp, device):
        with profile(activities=activities) as prof:
            for _ in range(10):
                torch.compiler.cudagraph_mark_step_begin()
                model(dense_features=dense, sparse_features=kjt)
            if device.type == "cuda":
                torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    logger.info("wrote trace: %s", path)


def _run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda":
        logger.warning(
            "device=%s is not CUDA: 'reduce-overhead' will NOT capture a CUDA "
            "graph, so this measures torch.compile overhead only.",
            device,
        )

    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    model = _build_model(args, device)
    timing_input = _generate_inputs(args, device)
    parity_batches = [
        _generate_inputs(args, device) for _ in range(args.parity_batches)
    ]

    # --- Step 4: baseline (eager) ---
    baseline_ms = _measure_ms_per_step(model, *timing_input, args, device)
    eager_outputs = _collect_outputs(model, parity_batches, args, device)
    if args.trace_dir:
        _export_trace(
            model, *timing_input, f"{args.trace_dir}/before.json", args, device
        )

    # --- Step 5: enable Path A (CUDA Graphs via torch.compile) ---
    mem_before = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    model.compile_dense_path()
    compiled_ms = _measure_ms_per_step(model, *timing_input, args, device)
    compiled_outputs = _collect_outputs(model, parity_batches, args, device)
    mem_after = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if args.trace_dir:
        _export_trace(
            model, *timing_input, f"{args.trace_dir}/after.json", args, device
        )

    # --- Step 7: validate ---
    max_abs_diff = max(
        (e - c).abs().max().item() for e, c in zip(eager_outputs, compiled_outputs)
    )
    tol = 1e-3 if args.amp else 1e-5
    speedup_pct = (baseline_ms - compiled_ms) / baseline_ms * 100.0

    print("=" * 60)
    print(f"model:            {args.model}")
    print(f"batch_size:       {args.batch_size}")
    print(f"num_sparse_feats: {args.num_sparse_features}")
    print(f"amp (fp16):       {args.amp}")
    print("-" * 60)
    print(f"baseline (eager): {baseline_ms:.4f} ms/step")
    print(f"cudagraph (A):    {compiled_ms:.4f} ms/step")
    print(f"speedup:          {speedup_pct:+.1f}%  (>=10% is the target)")
    print(f"max abs diff:     {max_abs_diff:.2e}  (tol {tol:.0e})")
    print(f"parity:           {'PASS' if max_abs_diff < tol else 'FAIL'}")
    if device.type == "cuda":
        print(f"max mem delta:    {(mem_after - mem_before) / 1e6:.1f} MB")
    print("=" * 60)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dlrm", "dlrm_dcn"], default="dlrm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dense-in-features", type=int, default=13)
    parser.add_argument("--num-sparse-features", type=int, default=26)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-embeddings", type=int, default=100_000)
    parser.add_argument("--pooling-factor", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--bench-iters", type=int, default=500)
    parser.add_argument("--parity-batches", type=int, default=100)
    parser.add_argument("--dcn-num-layers", type=int, default=2)
    parser.add_argument("--dcn-low-rank-dim", type=int, default=8)
    parser.add_argument(
        "--amp", action="store_true", help="run forward under fp16 autocast"
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="if set, export before.json / after.json Kineto traces here",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
