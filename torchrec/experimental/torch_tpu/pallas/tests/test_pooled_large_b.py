#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Large-bag-count (B) behavior of the fused pooled kernel.

Stacking many features into one `torchrec::embedding_pooled_lookup_offset` call
makes B huge (~100K bags). These cases exercise the forward and backward at that
scale to show whether the cost is in the forward kernel, the dense backward
scatter, or neither; the timings are printed alongside the assertions.

Requires an attached TPU -- the cases skip off-hardware. On the pod:

    ./run_pod.sh run test_pooled_large_b.py
"""

import time
import unittest

import torch
import torch.distributed as dist

V = 5_000_000  # table rows
D = 16  # embedding dim (SparseCore vector-lane width)
AVG_POOL = 2  # average ids per bag


def tpu_is_available() -> bool:
    """Whether a TPU backend is importable and has a device attached."""
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


def _register_tpu_kernels() -> None:
    """Bind the Pallas kernels to the TPU dispatch key.

    Imported lazily: `impl` pulls in `torch_tpu` and `jax`, which are only present
    on a TPU host, so importing it at module scope would break collection off-device.
    """
    from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401


def _synchronize() -> None:
    """Block until pending TPU work has executed, so the timings mean something."""
    # pyre-ignore[21]: torch_tpu ships in the TPU pod venv, not as a buck dep.
    from torch_tpu._internal import sync as _sync

    _sync.synchronize(wait=True)


class PooledLargeBTest(unittest.TestCase):
    def setUp(self) -> None:
        """Skip off-hardware; bind the Pallas kernels on it.

        The check runs here rather than in a `skipIf` decorator so that `torch_tpu`
        is only imported when a case actually executes.
        """
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        _register_tpu_kernels()
        # The pod launcher runs one process per TensorCore; torch_tpu needs its
        # distributed runtime up before a device is usable in that mode.
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _check_pooled_lookup(self, batch_size: int) -> None:
        dev = torch.device("tpu")
        gen = torch.Generator().manual_seed(0)
        lengths = torch.randint(1, 2 * AVG_POOL, (batch_size,), generator=gen)
        total = int(lengths.sum())
        indices = torch.randint(0, V, (total,), dtype=torch.int32, generator=gen)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)]
        )
        # Host -> device once, before the clock starts. The ids come from a seeded CPU
        # generator (reproducibility), so the copy is inherent to the harness rather
        # than something the kernel pays for, and must not land in the measurement.
        indices_d = indices.to(dev)
        offsets_d = offsets.to(dev)
        weight = torch.randn(V, D, device=dev, requires_grad=True)

        def _forward() -> torch.Tensor:
            return torch.ops.torchrec.embedding_pooled_lookup_offset(
                indices_d, offsets_d, weight, D
            )

        # Warm up first: the initial call compiles the Pallas kernel for this shape, and
        # each batch_size is a distinct shape, so without this the numbers below would be
        # compile time rather than steady state.
        _forward().pow(2).sum().backward()
        _synchronize()
        weight.grad = None

        t0 = time.perf_counter()
        out = _forward()
        _synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1e3
        self.assertEqual(tuple(out.shape), (batch_size, D))

        t0 = time.perf_counter()
        out.pow(2).sum().backward()
        _synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1e3
        grad = weight.grad
        self.assertIsNotNone(grad, "backward produced no gradient for the table")
        self.assertTrue(
            bool(torch.isfinite(grad).all()), "table gradient has non-finite entries"
        )
        print(
            f"[B={batch_size} V={V} D={D}] fwd {fwd_ms:.1f}ms  bwd {bwd_ms:.1f}ms "
            f"(steady state, post-warmup)",
            flush=True,
        )

    def test_small_b(self) -> None:
        """Baseline at one feature's worth of bags."""
        self._check_pooled_lookup(batch_size=4_096)

    def test_large_b(self) -> None:
        """Stacking-scale bag count -- the case feature stacking would produce."""
        self._check_pooled_lookup(batch_size=100_000)
