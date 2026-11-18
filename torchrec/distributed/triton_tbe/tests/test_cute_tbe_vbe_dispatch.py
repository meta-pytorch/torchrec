#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import TYPE_CHECKING
from unittest import mock

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.tbe.utils import generate_requests_for_grouped_tables

# The Triton TBE module needs triton and triton_kernels, and cute_tbe the CuTe
# DSL, which OSS CI does not install. Import them only where the test can run.
_TESTS_SUPPORTED: bool = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)

if TYPE_CHECKING or _TESTS_SUPPORTED:
    import torchrec.distributed.triton_tbe.triton_table_batched_embeddings as ttbe
    from torchrec.distributed.triton_tbe import cute_tbe


def _vbe_forward(cute: bool) -> torch.Tensor:
    """Pooled output of one unweighted VBE forward with the Triton TBE
    optimizations on, with or without the CuTe kernels."""
    torch.manual_seed(0)
    Es, Ds, Ls, B = [1000, 5000, 20000], [8, 128, 256], [1, 5, 3], 4096
    ftm = list(range(len(Es)))
    req = generate_requests_for_grouped_tables(
        1,
        [B - 97 * t for t in ftm],  # VBE: a different batch size per feature
        len(Es),
        max(Ls),
        max(Es),
        Ls=Ls,
        feature_table_map=ftm,
        Es=Es,
        weighted=False,
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
        device=torch.device("cuda"),
        enable_triton_tbe_optimizations=True,
        cute_kernels=cute,
    )
    torch.manual_seed(1)
    m.weight.data.uniform_(-1.0, 1.0)
    out = m(
        req.indices,
        req.offsets,
        None,
        batch_size_per_feature_per_rank=req.Bs_per_feature_per_rank,
    )
    torch.cuda.synchronize()
    return out.detach()


@unittest.skipIf(not _TESTS_SUPPORTED, "needs a Blackwell GPU")
class CuteVbeForwardDispatchTest(unittest.TestCase):
    def test_cute_kernels_select_the_vbe_forward(self) -> None:
        # cute_kernels=False: the grouped Triton VBE forward. cute_kernels=True:
        # the CuTe forward, not the grouped Triton path.
        grouped = mock.Mock(wraps=ttbe._launch_unweighted_vbe_forward)
        cute_forward = mock.Mock(wraps=cute_tbe.tbe_forward)
        with (
            mock.patch.object(ttbe, "_launch_unweighted_vbe_forward", grouped),
            mock.patch.object(cute_tbe, "tbe_forward", cute_forward),
        ):
            ref = _vbe_forward(cute=False)
            grouped.assert_called_once()
            cute_forward.assert_not_called()
            grouped.reset_mock()
            got = _vbe_forward(cute=True)
            grouped.assert_not_called()
            cute_forward.assert_called_once()
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-4)
