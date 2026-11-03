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
Also checks table-batched backward with empty bags and bucket-padded ids.

Requires an attached TPU -- the cases skip off-hardware. On the pod:

    ./run_pod.sh run test_v2_fwd_bwd.py
"""

import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.experimental.torch_tpu.modules.embedding_modules import (
    PallasTableBatchedEmbeddingBags,
)
from torchrec.experimental.torch_tpu.pallas import ops  # noqa: F401

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


class FusedAdagradCpuTest(unittest.TestCase):
    def test_batched_module_updates_rows_without_a_weight_gradient(self) -> None:
        module = PallasTableBatchedEmbeddingBags(
            embedding_specs=[(5, EMB_DIM)],
            device=torch.device("cpu"),
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
            learning_rate=0.1,
        )
        pooled = module(
            indices=torch.tensor([1, 1, 2], dtype=torch.int32),
            offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
        )
        pooled.sum().backward()

        self.assertIsNone(module.weights.grad)
        self.assertEqual(module.weights.shape, (5 * EMB_DIM,))
        torch.testing.assert_close(
            module.momentum, torch.tensor([0.0, 4.0, 1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            module.weights[EMB_DIM : 2 * EMB_DIM], torch.full((EMB_DIM,), -0.1)
        )
        torch.testing.assert_close(
            module.weights[2 * EMB_DIM : 3 * EMB_DIM],
            torch.full((EMB_DIM,), -0.1),
        )

    def test_unfused_default_and_unsupported_optimizer(self) -> None:
        module = PallasTableBatchedEmbeddingBags(
            embedding_specs=[(5, EMB_DIM)], device=torch.device("cpu")
        )
        self.assertEqual(module.optimizer, OptimType.NONE)
        self.assertIsNone(module.momentum)
        module(
            indices=torch.tensor([1], dtype=torch.int32),
            offsets=torch.tensor([0, 1], dtype=torch.int32),
        ).sum().backward()
        grad = module.weights.grad
        self.assertIsNotNone(grad)
        self.assertEqual(module.weights.shape, (5 * EMB_DIM,))
        self.assertEqual(grad.shape, module.weights.shape)
        torch.testing.assert_close(grad[EMB_DIM : 2 * EMB_DIM], torch.ones(EMB_DIM))

        with self.assertRaisesRegex(ValueError, "NONE or EXACT_ROWWISE_ADAGRAD"):
            PallasTableBatchedEmbeddingBags(
                embedding_specs=[(5, EMB_DIM)],
                device=torch.device("cpu"),
                optimizer=OptimType.EXACT_SGD,
            )

    def test_flat_weights_pool_across_tables_and_accumulate_repeated_ids(self) -> None:
        dim = EMB_DIM
        module = PallasTableBatchedEmbeddingBags(
            embedding_specs=[(3, dim), (2, dim)],
            feature_table_map=[1, 0],
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            module.weights.copy_(torch.arange(5 * dim, dtype=torch.float32))
        indices = torch.tensor([1, 0, 2, 1, 1], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)

        pooled = module(indices, offsets)
        weights = module.weights.detach()
        expected = torch.stack(
            (
                torch.cat((weights[4 * dim : 5 * dim], weights[2 * dim : 3 * dim])),
                torch.cat((weights[3 * dim : 4 * dim], 2 * weights[dim : 2 * dim])),
            )
        )
        torch.testing.assert_close(pooled, expected)
        self.assertEqual(
            [w.shape for w in module.split_embedding_weights()], [(3, dim), (2, dim)]
        )

        pooled.sum().backward()
        grad = module.weights.grad
        self.assertIsNotNone(grad)
        torch.testing.assert_close(grad[:dim], torch.zeros(dim))
        torch.testing.assert_close(grad[dim : 2 * dim], torch.full((dim,), 2.0))
        torch.testing.assert_close(grad[2 * dim :], torch.ones(3 * dim))

    def test_duplicate_rows_empty_bag_and_padding(self) -> None:
        indices = torch.tensor([2, 2, 1, 0, 1, 0, 0, 0], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 2, 3, 5], dtype=torch.int32)
        row_offsets = torch.tensor([0, 0, 3, 3], dtype=torch.int32)
        weights = torch.arange(10, dtype=torch.float32).reshape(5, 2).requires_grad_()
        momentum = torch.zeros(5)
        grad_out = torch.tensor([[1.0, 2.0], [99.0, 99.0], [3.0, 4.0], [5.0, 12.0]])

        pooled = (
            torch.ops.torchrec.embedding_pooled_batched_lookup_offset_fused_adagrad(
                indices, offsets, weights, row_offsets, momentum, 0.1, 0.01, 2
            )
        )
        torch.testing.assert_close(
            pooled, torch.tensor([[8.0, 10.0], [0.0, 0.0], [8.0, 9.0], [14.0, 16.0]])
        )
        pooled.backward(grad_out)

        expected_momentum = torch.tensor([0.0, 0.0, 10.0, 84.5, 160.0])
        expected_weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        expected_weights[2] -= (
            0.1 * torch.tensor([2.0, 4.0]) / (expected_momentum[2].sqrt() + 0.01)
        )
        expected_weights[3] -= (
            0.1 * torch.tensor([5.0, 12.0]) / (expected_momentum[3].sqrt() + 0.01)
        )
        expected_weights[4] -= (
            0.1 * torch.tensor([8.0, 16.0]) / (expected_momentum[4].sqrt() + 0.01)
        )
        torch.testing.assert_close(momentum, expected_momentum)
        torch.testing.assert_close(weights, expected_weights)
        self.assertIsNone(weights.grad)


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

    def test_nonmonotonic_row_offsets_empty_bags_and_padding(self) -> None:
        from torch_tpu._internal import pallas  # pyrefly: ignore[missing-import]
        from torchrec.experimental.torch_tpu.pallas import pooled_lookup_offset

        gradient_map = pallas.jax_op(
            "torchrec_pallas_test::embedding_pooled_batched_lookup_gradient_map",
            pooled_lookup_offset.embedding_pooled_batched_lookup_gradient_map_jax,
        )
        row_runs = pallas.jax_op(
            "torchrec_pallas_test::embedding_pooled_batched_lookup_row_runs",
            pooled_lookup_offset.embedding_pooled_batched_lookup_row_runs_jax,
        )
        indices = torch.tensor(
            [1, 2, 0, 3, 0, 0, 0, 0], dtype=torch.int32, device="tpu"
        )
        offsets = torch.tensor([0, 0, 2, 2, 4, 4], dtype=torch.int32, device="tpu")
        row_offsets = torch.tensor([7, 1, 9, 2, 0], dtype=torch.int32, device="tpu")

        rows, bags, valid = gradient_map(indices, offsets, row_offsets)
        self.assertEqual(rows.cpu().tolist(), [2, 3, 2, 5, 0, 0, 0, 0])
        self.assertEqual(bags.cpu().tolist(), [1, 1, 3, 3, 4, 4, 4, 4])
        self.assertEqual(valid.cpu().tolist(), [True] * 4 + [False] * 4)

        sorted_rows, sorted_bags, run_offsets, num_runs = row_runs(
            indices, offsets, row_offsets, num_rows=16
        )
        self.assertEqual(sorted_rows.cpu().tolist(), [2, 2, 3, 5, 16, 16, 16, 16])
        self.assertEqual(sorted_bags.cpu().tolist()[:4], [1, 3, 1, 3])
        self.assertEqual(run_offsets.cpu().tolist()[:4], [0, 2, 3, 4])
        self.assertEqual(num_runs.item(), 3)

    def test_long_run_matches_cpu_after_two_steps(self) -> None:
        dim = 128
        indices = torch.tensor([2] * 19 + [5] * 5, dtype=torch.int32)
        bag_ids = torch.tensor([0] * 10 + [1] * 9 + [0, 1, 1, 0, 0])
        grad_out = torch.stack(
            [torch.arange(1, dim + 1, dtype=torch.float32), torch.full((dim,), 0.5)]
        )
        weights = torch.arange(8 * dim, dtype=torch.float32).reshape(8, dim) * 0.01
        momentum = torch.arange(8, dtype=torch.float32) * 0.5
        order = bag_ids.argsort(stable=True)
        args = (
            grad_out.to("tpu"),
            indices[order].to("tpu"),
            torch.tensor([0, 13, 24], dtype=torch.int32, device="tpu"),
            torch.zeros(2, dtype=torch.int32, device="tpu"),
        )

        actual_weights = weights.flatten().to("tpu")
        actual_momentum = momentum.to("tpu")
        expected_weights = weights.clone()
        expected_momentum = momentum.clone()
        grad_per_row = torch.zeros_like(weights).index_add_(
            0, indices.long(), grad_out[bag_ids]
        )
        rows = torch.tensor([2, 5])
        row_grads = grad_per_row[rows]
        tpu = vars(torch)["tpu"]
        for iteration, lr in enumerate((0.05, 0.02)):
            updated_momentum = expected_momentum[rows] + row_grads.square().mean(1)
            expected_weights[rows] -= (
                lr * row_grads / (updated_momentum.sqrt().unsqueeze(1) + 0.001)
            )
            expected_momentum[rows] = updated_momentum
            misses_before = tpu._get_cache_misses()
            torch.ops.torchrec.embedding_pooled_batched_lookup_offset_adagrad_update(
                *args, actual_weights, actual_momentum, lr, 0.001
            )
            torch.accelerator.synchronize()
            if iteration > 0:
                self.assertEqual(tpu._get_cache_misses(), misses_before)
            self.assertEqual(actual_weights.shape, (expected_weights.numel(),))
            torch.testing.assert_close(
                actual_weights.cpu().reshape_as(expected_weights),
                expected_weights,
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                actual_momentum.cpu(),
                expected_momentum,
                rtol=1e-4,
                atol=1e-5,
            )

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

    def test_batched_backward_handles_empty_bags_and_padded_ids(self) -> None:
        indices = torch.tensor(
            [2, 2, 1, 0, 1] + [0] * 11, dtype=torch.int32, device="tpu"
        )
        offsets = torch.tensor([0, 2, 2, 3, 5], dtype=torch.int32, device="tpu")
        row_offsets = torch.tensor([0, 0, 3, 3], dtype=torch.int32, device="tpu")
        grad_out = (
            torch.arange(1, 5, dtype=torch.float32, device="tpu")
            .unsqueeze(1)
            .expand(4, EMB_DIM)
            .contiguous()
        )

        grad = torch.ops.torchrec.embedding_pooled_batched_lookup_offset_backward(
            grad_out, indices, offsets, row_offsets, 5, EMB_DIM
        )
        expected = torch.tensor([0, 0, 2, 4, 7], dtype=torch.float32)
        expected = expected.unsqueeze(1).expand(5, EMB_DIM)
        torch.testing.assert_close(grad.cpu(), expected, rtol=0, atol=0)

    def test_flat_table_batched_forward_backward_matches_cpu(self) -> None:
        dim = 32
        indices = torch.tensor([1, 0, 2, 1, 1], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int32)
        row_offsets = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
        weights_cpu = torch.arange(5 * dim, dtype=torch.float32, requires_grad=True)
        weights_tpu = torch.arange(
            5 * dim, dtype=torch.float32, device="tpu", requires_grad=True
        )

        pooled_cpu = torch.ops.torchrec.embedding_pooled_batched_lookup_offset(
            indices, offsets, weights_cpu, row_offsets, dim
        )
        pooled_tpu = torch.ops.torchrec.embedding_pooled_batched_lookup_offset(
            indices.to("tpu"),
            offsets.to("tpu"),
            weights_tpu,
            row_offsets.to("tpu"),
            dim,
        )
        torch.testing.assert_close(pooled_tpu.cpu(), pooled_cpu)
        pooled_cpu.sum().backward()
        pooled_tpu.sum().backward()
        self.assertIsNotNone(weights_cpu.grad)
        self.assertIsNotNone(weights_tpu.grad)
        torch.testing.assert_close(weights_tpu.grad.cpu(), weights_cpu.grad)

    def test_fused_rowwise_adagrad_updates_parameter_and_momentum(self) -> None:
        dim = 128
        indices = torch.tensor([2] * 19 + [1] * 5 + [0] * 8, dtype=torch.int32)
        offsets = torch.tensor([0, 10, 10, 19, 24], dtype=torch.int32)
        row_offsets = torch.tensor([0, 0, 0, 4], dtype=torch.int32)
        grad_out = torch.stack(
            (
                torch.arange(1, dim + 1, dtype=torch.float32),
                torch.full((dim,), 99.0),
                torch.full((dim,), 0.5),
                torch.linspace(0.1, 1.6, dim),
            )
        )
        weights_cpu = torch.zeros((8 * dim,), requires_grad=True)
        momentum_cpu = torch.zeros(8)

        # Citrine C3: allocate TPU weight and optimizer state directly on the device.
        weights_tpu = torch.zeros((8 * dim,), device="tpu", requires_grad=True)
        momentum_tpu = torch.zeros(8, device="tpu")
        indices_tpu = indices.to("tpu")
        offsets_tpu = offsets.to("tpu")
        row_offsets_tpu = row_offsets.to("tpu")
        grad_out_tpu = grad_out.to("tpu")

        for _ in range(2):
            pooled_cpu = (
                torch.ops.torchrec.embedding_pooled_batched_lookup_offset_fused_adagrad(
                    indices,
                    offsets,
                    weights_cpu,
                    row_offsets,
                    momentum_cpu,
                    0.1,
                    0.01,
                    dim,
                )
            )
            pooled_tpu = (
                torch.ops.torchrec.embedding_pooled_batched_lookup_offset_fused_adagrad(
                    indices_tpu,
                    offsets_tpu,
                    weights_tpu,
                    row_offsets_tpu,
                    momentum_tpu,
                    0.1,
                    0.01,
                    dim,
                )
            )
            torch.testing.assert_close(pooled_tpu.cpu(), pooled_cpu)
            pooled_cpu.backward(grad_out)
            pooled_tpu.backward(grad_out_tpu)
            self.assertIsNone(weights_tpu.grad)
            torch.testing.assert_close(
                weights_tpu.detach().cpu(), weights_cpu.detach(), rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(
                momentum_tpu.cpu(), momentum_cpu, rtol=1e-4, atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
