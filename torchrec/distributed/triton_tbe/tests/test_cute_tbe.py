#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest import mock

import torch
import torchrec.distributed.triton_tbe.triton_table_batched_embeddings as ttbe
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.tbe.utils import generate_requests_for_grouped_tables


def _forward(Es, Ds, Ls, B, weighted, cute, bad_index=False, vbe=False):
    """Pooled output and bounds-check warning count of one Triton TBE forward,
    with or without the CuTe kernels."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ftm = list(range(len(Es)))
    # VBE: a different batch size per feature.
    Bs = [B - 97 * t for t in range(len(Es))] if vbe else B
    req = generate_requests_for_grouped_tables(
        1,
        Bs,
        len(Es),
        max(Ls),
        max(Es),
        Ls=Ls,
        feature_table_map=ftm,
        Es=Es,
        weighted=weighted,
    )[0]
    m = ttbe.TritonTableBatchedEmbeddingBags(
        list(zip(Es, Ds)),
        feature_table_map=ftm,
        weights_precision=torch.float16,
        output_dtype=torch.float32,
        stochastic_rounding=False,
        learning_rate=0.01,
        eps=0.1,
        optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        device=dev,
        fused_bounds_check=True,
        enable_triton_tbe_optimizations=True,
        cute_kernels=cute,
    )
    torch.manual_seed(1)
    m.weight.data.uniform_(-1.0, 1.0)
    indices = req.indices.clone()
    if bad_index:
        indices[0] = Es[0] + 7  # out of range: repaired to 0 and counted
    out = m(
        indices,
        req.offsets.clone(),
        req.per_sample_weights,
        batch_size_per_feature_per_rank=req.Bs_per_feature_per_rank,
    )
    torch.cuda.synchronize()
    warnings = m.bounds_check_warning.item()
    return out.detach(), indices, warnings


def _train_step(
    Es,
    Ds,
    ftm,
    Ls,
    B,
    cute,
    weighted=False,
    vbe=False,
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
):
    """Weights and momentum after one Triton TBE forward + backward, with or
    without the CuTe kernels."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    req = generate_requests_for_grouped_tables(
        1,
        [B - 97 * t for t in range(len(ftm))] if vbe else B,  # VBE: B per feature
        len(Es),
        max(Ls),
        max(Es),
        Ls=Ls,
        feature_table_map=ftm,
        Es=Es,
        weighted=weighted,
    )[0]
    m = ttbe.TritonTableBatchedEmbeddingBags(
        list(zip(Es, Ds)),
        feature_table_map=ftm,
        weights_precision=torch.float16,
        output_dtype=torch.float32,
        stochastic_rounding=False,
        learning_rate=0.01,
        eps=0.1,
        optimizer=optimizer,
        device=dev,
        enable_triton_tbe_optimizations=True,
        cute_kernels=cute,
    )
    torch.manual_seed(1)
    m.weight.data.uniform_(-1.0, 1.0)
    out = m(
        req.indices.clone(),
        req.offsets.clone(),
        req.per_sample_weights,
        batch_size_per_feature_per_rank=req.Bs_per_feature_per_rank,
    )
    torch.manual_seed(2)
    out.backward(torch.randn_like(out) * 0.01)
    torch.cuda.synchronize()
    return m.weight.detach().clone(), m.momentum.detach().clone()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    "needs a Blackwell GPU",
)
class CuteTbeBackwardTest(unittest.TestCase):
    def test_matches_triton(self):
        # Table 0: ~33 lookups per row (with 1024-sample tiles: rows split over
        # 8 tiles, finalize). Table 1: hot rows (Triton's long-run tier).
        # Table 2: D=128 (second dimension bucket), ~1 lookup per row. Table 3:
        # shared by features 3 and 4, so runs span two features. Table 4: D=112,
        # in the same bucket as D=128 (mixed widths in one launch). Tables 5-6
        # (not with tiling, which needs D % 8 == 0): hot D=8 / D=4 rows, whose
        # long runs read the compact narrow copy.
        Es, Ds = [2000, 50, 30000, 1000, 2500], [256, 256, 128, 256, 112]
        ftm, Ls = [0, 1, 2, 3, 3, 4], [8, 3, 4, 5, 2, 6]
        tiled = (Es, Ds, ftm, Ls, 8192)
        args = (Es + [40, 2], Ds + [8, 4], ftm + [5, 6], Ls + [5, 2], 8192)
        # tile_b 0: run-packed row-order kernel; 1024: batch-tiled kernel
        # (unweighted, non-VBE only: weighted and VBE take the packed kernel).
        cases = [(False, False, "0"), (False, False, "1024"), (True, False, "0")]
        cases += [(False, True, "0"), (True, True, "0")]
        for weighted, vbe, tile_b in cases:
            a = tiled if tile_b != "0" else args
            ref_w, ref_m = _train_step(*a, False, weighted, vbe)
            with (
                self.subTest(weighted=weighted, vbe=vbe, tile_b=tile_b),
                mock.patch.dict(os.environ, {"CUTE_TBE_BWD_TILE_B": tile_b}),
            ):
                got_w, got_m = _train_step(*a, True, weighted, vbe)
                # fp32 sums in another order: weights within 1 fp16 ulp (|w| <= 1).
                torch.testing.assert_close(got_w, ref_w, rtol=0, atol=1e-3)
                torch.testing.assert_close(got_m, ref_m, rtol=1e-4, atol=1e-9)

    def test_unsupported_optimizer_falls_back_to_triton(self):
        # SGD: the CuTe backward launchers hand every launch back to Triton.
        args = ([2000, 50], [256, 8], [0, 1], [8, 5], 8192)
        sgd = OptimType.EXACT_SGD
        ref_w, _ = _train_step(*args, False, optimizer=sgd)
        got_w, _ = _train_step(*args, True, optimizer=sgd)
        torch.testing.assert_close(got_w, ref_w, rtol=0, atol=1e-3)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    "needs a Blackwell GPU",
)
class CuteTbeForwardTest(unittest.TestCase):
    def test_matches_triton(self):
        cases = {
            # Mixed widths: D=4 (8B lanes), 8 (32 rows/instr), 112/128 (lane groups
            # or packed), 256 (whole-warp flat stream); one empty-bag feature.
            "mixed": (
                [1000, 5000, 20000, 20000, 20000],
                [4, 8, 112, 128, 256],
                [1, 5, 9, 0, 3],
            ),
            "long_bags": ([3000, 70000], [112, 256], [130, 40]),
        }
        for name, (Es, Ds, Ls) in cases.items():
            for weighted in (False, True):
                for vbe in (False, True):
                    with self.subTest(case=name, weighted=weighted, vbe=vbe):
                        ref, _, _ = _forward(Es, Ds, Ls, 4096, weighted, False, vbe=vbe)
                        got, _, _ = _forward(Es, Ds, Ls, 4096, weighted, True, vbe=vbe)
                        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)

    def test_bounds_check_repairs_like_triton(self):
        args = ([1000, 20000], [8, 256], [5, 3], 4096, False)
        ref, ref_idx, ref_warn = _forward(*args, cute=False, bad_index=True)
        got, got_idx, got_warn = _forward(*args, cute=True, bad_index=True)
        self.assertEqual(int(got_idx[0]), 0)
        self.assertTrue(torch.equal(got_idx, ref_idx))
        self.assertEqual(got_warn, ref_warn)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
