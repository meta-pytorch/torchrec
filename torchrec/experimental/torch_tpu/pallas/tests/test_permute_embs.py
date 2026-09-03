#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from itertools import accumulate

import torch
import torch.distributed as dist
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401

_ = ops


EMBEDDING_DIMS = [16, 32, 16, 8, 16]
EMBEDDING_ORDER = [0, 2, 4, 1, 3]
BATCH_SIZE = 10


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401

        _ = torch_tpu
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


def _reference(pooled_embs: torch.Tensor) -> torch.Tensor:
    blocks = torch.split(pooled_embs, EMBEDDING_DIMS, dim=1)
    return torch.cat([blocks[index] for index in EMBEDDING_ORDER], dim=1)


def _metadata(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [0, *accumulate(EMBEDDING_DIMS)], dtype=torch.int32, device=device
        ),
        torch.tensor(EMBEDDING_ORDER, dtype=torch.int32, device=device),
    )


class PermutePooledEmbeddingsTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_tpu(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

    def _check_forward_backward(self, device: torch.device) -> None:
        torch.manual_seed(0)
        width = sum(EMBEDDING_DIMS)
        pooled = torch.randn((BATCH_SIZE, width), dtype=torch.float32)
        grad_seed = torch.randn_like(pooled)

        expected_input = pooled.clone().requires_grad_(True)
        expected = _reference(expected_input)
        expected.backward(grad_seed)

        actual_input = pooled.to(device).requires_grad_(True)
        offset_dim_list, permute_list = _metadata(device)
        actual = torch.ops.torchrec.permute_pooled_embs(
            actual_input, offset_dim_list, permute_list
        )
        actual.backward(grad_seed.to(device))

        self.assertIsNotNone(expected_input.grad)
        self.assertIsNotNone(actual_input.grad)
        assert expected_input.grad is not None
        assert actual_input.grad is not None
        torch.testing.assert_close(actual.detach().cpu(), expected.detach())
        torch.testing.assert_close(actual_input.grad.cpu(), expected_input.grad)

    def test_cpu_forward_backward(self) -> None:
        self._check_forward_backward(torch.device("cpu"))

    def test_tpu_forward_backward(self) -> None:
        self._setup_tpu()
        self._check_forward_backward(torch.device("tpu"))

    def test_sparse_core_forward(self) -> None:
        self._setup_tpu()
        from torchrec.experimental.torch_tpu.pallas import impl

        torch.manual_seed(0)
        device = torch.device("tpu")
        width = sum(EMBEDDING_DIMS)
        pooled = torch.randn((BATCH_SIZE, width), dtype=torch.float32)
        offset_dim_list, permute_list = _metadata(device)
        output_dims = [EMBEDDING_DIMS[index] for index in EMBEDDING_ORDER]
        output_offset_dim_list = torch.tensor(
            [0, *accumulate(output_dims)], dtype=torch.int32, device=device
        )

        actual = impl.permute_pooled_embs_sparse_core_tpu(
            pooled.to(device),
            offset_dim_list,
            permute_list,
            output_offset_dim_list,
            max(EMBEDDING_DIMS),
        )

        torch.testing.assert_close(actual.cpu(), _reference(pooled))


if __name__ == "__main__":
    unittest.main()
