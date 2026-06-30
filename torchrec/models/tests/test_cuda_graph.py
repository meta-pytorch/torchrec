#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
GPU (RE) test for ``DLRM.compile_dense_path()`` — the CUDA Graphs dense-path
optimization. Runs on a ``gpu-remote-execution`` worker via ``buck2 test`` so the
``reduce-overhead`` path actually captures a CUDA graph (it is a no-op on CPU).

Validates two things on a real GPU:
  * numerical parity — the compiled dense path matches eager outputs, and
  * the optimization is runnable end-to-end (idempotent enable, replay works).

It deliberately does NOT hard-assert a speedup threshold: the win is real but
HW- and queue-dependent, and a fixed bar would make the test flaky. The speedup
is logged for the human reading the test output.
"""

import logging
import time
import unittest

import torch
from torchrec.models.cuda_graph_utils import build_dlrm, collect_outputs, generate_batch
from torchrec.models.dlrm import DLRM
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# Small, fast model/inputs — this is a correctness test, not a benchmark.
_BATCH_SIZE = 256
_NUM_SPARSE_FEATURES = 8
_EMBEDDING_DIM = 32
_NUM_EMBEDDINGS = 10_000
_DENSE_IN_FEATURES = 13
_POOLING_FACTOR = 20
_PARITY_BATCHES = 8


def _build_model(device: torch.device) -> DLRM:
    return build_dlrm(
        device,
        embedding_dim=_EMBEDDING_DIM,
        num_sparse_features=_NUM_SPARSE_FEATURES,
        num_embeddings=_NUM_EMBEDDINGS,
        dense_in_features=_DENSE_IN_FEATURES,
    )


def _gen_batch(device: torch.device) -> tuple[torch.Tensor, KeyedJaggedTensor]:
    return generate_batch(
        device,
        batch_size=_BATCH_SIZE,
        num_sparse_features=_NUM_SPARSE_FEATURES,
        num_embeddings=_NUM_EMBEDDINGS,
        dense_in_features=_DENSE_IN_FEATURES,
        pooling_factor=_POOLING_FACTOR,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA Graphs require a GPU")
class DLRMCudaGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_dense_path_parity(self) -> None:
        """Compiled dense path must match eager outputs within fusion tolerance."""
        device = torch.device("cuda")
        model = _build_model(device)
        batches = [_gen_batch(device) for _ in range(_PARITY_BATCHES)]

        eager = collect_outputs(model, batches)

        model.compile_dense_path()
        # Idempotent: a second call must be a no-op.
        model.compile_dense_path()

        compiled = collect_outputs(model, batches)

        max_abs_diff = max((e - c).abs().max().item() for e, c in zip(eager, compiled))
        logger.info("max abs diff (eager vs cudagraph): %.3e", max_abs_diff)
        # fp32 path; tolerance accounts for Inductor fusion/reduction reordering.
        self.assertLess(max_abs_diff, 1e-3)

    @torch.inference_mode()
    def test_dense_path_runs_and_reports_speedup(self) -> None:
        """End-to-end smoke + informational speedup (no hard threshold)."""
        device = torch.device("cuda")
        model = _build_model(device)
        dense, kjt = _gen_batch(device)

        def _time_ms(iters: int) -> float:
            # Simple CPU wall-clock timing (runbook Step 4: Capture a Baseline).
            # cudagraph_mark_step_begin() is required on the compiled replay path
            # (the runbook's Step 4 snippet omits it because it times eager).
            for _ in range(10):  # warmup
                torch.compiler.cudagraph_mark_step_begin()
                model(dense_features=dense, sparse_features=kjt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                torch.compiler.cudagraph_mark_step_begin()
                model(dense_features=dense, sparse_features=kjt)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0 / iters

        baseline_ms = _time_ms(50)
        model.compile_dense_path()
        compiled_ms = _time_ms(50)

        speedup_pct = (baseline_ms - compiled_ms) / baseline_ms * 100.0
        logger.info(
            "baseline=%.4f ms/step  cudagraph=%.4f ms/step  speedup=%+.1f%%",
            baseline_ms,
            compiled_ms,
            speedup_pct,
        )
        self.assertGreater(baseline_ms, 0.0)
        self.assertGreater(compiled_ms, 0.0)
