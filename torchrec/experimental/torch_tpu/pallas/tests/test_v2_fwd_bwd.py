#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward+backward smoke test for the offset-driven ("v2") pooled SparseCore kernel.

Runs one small sum-pooled lookup through `torchrec::embedding_pooled_lookup_offset`
on the "tpu" device, then a backward to the table, and checks both against a
plain-torch CPU `embedding_bag(mode="sum")` reference. The kernel does no
collectives, so each rank checks independently.

Requires an attached TPU -- the cases skip off-hardware. On the pod:

    ./run_pod.sh run test_v2_fwd_bwd.py
"""

import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F

NUM_ROWS = 1024
EMB_DIM = 16  # must be a multiple of 16 (SparseCore vector-lane width)


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


class V2FwdBwdTest(unittest.TestCase):
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

    def test_fwd_bwd_matches_cpu_embedding_bag(self) -> None:
        torch.manual_seed(0)
        # 3 bags (samples), multi-hot lengths [2, 3, 1] over 6 flat ids.
        lengths = torch.tensor([2, 3, 1], dtype=torch.int32)
        indices = torch.tensor([5, 9, 1, 3, 7, 42], dtype=torch.int32)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)]
        )
        weight_cpu = torch.randn(NUM_ROWS, EMB_DIM, dtype=torch.float32)
        grad_seed = torch.randn(len(lengths), EMB_DIM, dtype=torch.float32)

        # Reference: torch CPU embedding_bag (sum), fwd + bwd.
        w_ref = weight_cpu.clone().requires_grad_(True)
        ref_out = F.embedding_bag(
            input=indices.long(), weight=w_ref, offsets=offsets[:-1].long(), mode="sum"
        )
        ref_out.backward(grad_seed)

        # Offset-driven SparseCore kernel on TPU: fwd + bwd.
        w_tpu = weight_cpu.clone().to("tpu").requires_grad_(True)
        out = torch.ops.torchrec.embedding_pooled_lookup_offset(
            indices.to("tpu"), offsets.to("tpu"), w_tpu, EMB_DIM
        )
        out.backward(grad_seed.to("tpu"))  # dense scatter-add backward -> w_tpu.grad

        ref_grad, tpu_grad = w_ref.grad, w_tpu.grad
        self.assertIsNotNone(ref_grad)
        self.assertIsNotNone(tpu_grad, "backward produced no gradient for the table")
        fwd_err = (out.detach().to("cpu") - ref_out.detach()).abs().max().item()
        bwd_err = (tpu_grad.to("cpu") - ref_grad).abs().max().item()
        self.assertTrue(
            torch.allclose(out.detach().to("cpu"), ref_out.detach(), atol=1e-4),
            f"forward mismatch: max abs diff = {fwd_err}",
        )
        self.assertTrue(
            torch.allclose(tpu_grad.to("cpu"), ref_grad, atol=1e-3),
            f"backward mismatch: max abs diff = {bwd_err}",
        )
        print(
            f"[PASS] v2 fwd+bwd  out={tuple(out.shape)}  "
            f"fwd_err={fwd_err:.2e}  bwd_err={bwd_err:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
