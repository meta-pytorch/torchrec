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
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import (
    generate_vbe_metadata,
)
from torchrec.experimental.torch_tpu.modules.embedding_modules import (
    PallasTableBatchedEmbeddingBags,
)
from torchrec.experimental.torch_tpu.pallas import ops

NUM_ROWS = 1024
EMB_DIM = 16  # must be a multiple of 16 (SparseCore vector-lane width)


def tpu_is_available() -> bool:
    """Whether a TPU backend is importable and has a device attached."""
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401
    except ImportError:
        return False
    return torch.accelerator.is_available()


def _register_tpu_kernels() -> None:
    """Bind the Pallas kernels to the TPU dispatch key.

    Imported lazily: `impl` pulls in `torch_tpu` and `jax`, which are only present
    on a TPU host, so importing it at module scope would break collection off-device.
    """
    from torchrec.experimental.torch_tpu.pallas import impl  # noqa: F401


class FusedAdagradCpuTest(unittest.TestCase):
    def test_rejects_mismatched_vbe_batch_sizes_before_device_work(self) -> None:
        module = PallasTableBatchedEmbeddingBags(
            embedding_specs=[(3, EMB_DIM), (2, EMB_DIM)],
            device=torch.device("cpu"),
        )
        batch_sizes = [[1, 1], [1, 1]]
        indices = torch.tensor([0], dtype=torch.int64)
        for num_bags in (3, 5):
            with self.subTest(num_bags=num_bags):
                offsets = torch.tensor([0] * num_bags + [1], dtype=torch.int64)
                with patch.object(
                    torch.Tensor,
                    "to",
                    side_effect=AssertionError("Unexpected device work"),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"VBE batch sizes sum to 4 bags, but offsets describe {num_bags}",
                    ):
                        module(
                            indices,
                            offsets,
                            batch_size_per_feature_per_rank=batch_sizes,
                        )

    def test_context_saves_precomputed_vbe_offsets(self) -> None:
        indices = torch.tensor([0, 1], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
        row_offsets = torch.zeros(2, dtype=torch.int32)
        weights = torch.zeros(2 * EMB_DIM)
        momentum = torch.zeros(2)
        for row_output_offsets in (None, torch.tensor([EMB_DIM, 0])):
            cases = (
                (
                    ops._setup_pooled_batched_offset_context,
                    (
                        indices,
                        offsets,
                        weights,
                        row_offsets,
                        EMB_DIM,
                        row_output_offsets,
                    ),
                ),
                (
                    ops._setup_fused_adagrad_context,
                    (
                        indices,
                        offsets,
                        weights,
                        row_offsets,
                        momentum,
                        0.1,
                        0.01,
                        EMB_DIM,
                        row_output_offsets,
                    ),
                ),
            )
            for setup_context, inputs in cases:
                with self.subTest(
                    setup=setup_context.__name__, vbe=row_output_offsets is not None
                ):
                    ctx = Mock()
                    with patch.object(
                        ops,
                        "generate_vbe_row_output_offsets",
                        side_effect=AssertionError("VBE offsets must be reused"),
                    ):
                        setup_context(ctx, inputs, output=None)
                    ctx.save_for_backward.assert_called_once()
                    self.assertIs(
                        ctx.save_for_backward.call_args.args[-1], row_output_offsets
                    )

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

    def test_vbe_offsets_for_backward_and_fused_update(self) -> None:
        indices = torch.tensor([2, 2, 1, 0, 1, 0, 0, 0], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 2, 3, 5], dtype=torch.int32)
        row_offsets = torch.tensor([0, 0, 3, 3], dtype=torch.int32)
        row_output_offsets = torch.tensor([0, 4, 2, 6], dtype=torch.int64)
        grad_out = torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0], [5.0, 12.0]])
        expected_grad = torch.tensor(
            [[0.0, 0.0], [0.0, 0.0], [2.0, 4.0], [5.0, 12.0], [8.0, 16.0]]
        )
        for flat_weights in (False, True):
            with self.subTest(flat_weights=flat_weights):
                grad = (
                    torch.ops.torchrec.embedding_pooled_batched_lookup_offset_backward(
                        grad_out,
                        indices,
                        offsets,
                        row_offsets,
                        5,
                        2,
                        flat_weights,
                        row_output_offsets=row_output_offsets,
                    )
                )
                torch.testing.assert_close(
                    grad, expected_grad.flatten() if flat_weights else expected_grad
                )

        weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        initial_weights = weights.clone()
        momentum = torch.zeros(5)
        torch.ops.torchrec.embedding_pooled_batched_lookup_offset_adagrad_update(
            grad_out,
            indices,
            offsets,
            row_offsets,
            weights,
            momentum,
            0.1,
            0.01,
            row_output_offsets=row_output_offsets,
        )
        expected_momentum = expected_grad.square().mean(dim=1)
        torch.testing.assert_close(momentum, expected_momentum)
        torch.testing.assert_close(
            weights,
            initial_weights
            - 0.1 * expected_grad / (expected_momentum.sqrt().unsqueeze(1) + 0.01),
        )

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

    def test_batched_module_uses_generated_vbe_offsets(self) -> None:
        from torchrec.experimental.torch_tpu.pallas import impl

        dim = EMB_DIM
        counts = [[2, 0, 1], [0, 2, 0], [1, 0, 1]]
        indices = torch.tensor([1, 1, 2, 0, 3, 4, 1, 1, 2], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 2, 3, 5, 6, 6, 9], dtype=torch.int32)
        row_offsets = torch.tensor([0, 0, 0, 7, 7, 0, 0], dtype=torch.int32)
        for mode in (PoolingMode.SUM, PoolingMode.MEAN):
            for fused in (False, True):
                with self.subTest(mode=mode, fused=fused):
                    module = PallasTableBatchedEmbeddingBags(
                        embedding_specs=[(7, dim), (5, dim)],
                        feature_table_map=[0, 1, 0],
                        device=torch.device("tpu"),
                        mode=mode,
                        optimizer=(
                            OptimType.EXACT_ROWWISE_ADAGRAD if fused else OptimType.NONE
                        ),
                        learning_rate=0.1,
                        eps=0.01,
                    )
                    weights_cpu = (
                        torch.arange(module.weights.numel(), dtype=torch.float32) * 0.01
                    ).requires_grad_()
                    with torch.no_grad():
                        module.weights.copy_(weights_cpu.detach().to("tpu"))
                    momentum_cpu = torch.zeros(module.total_hash_size)
                    metadata = generate_vbe_metadata(
                        offsets,
                        counts,
                        mode,
                        torch.tensor([dim] * 3),
                        torch.device("cpu"),
                    )
                    element_offsets = ops.generate_vbe_row_output_offsets(
                        metadata.B_offsets,
                        metadata.B_offsets_rank_per_feature,
                        metadata.output_offsets_feature_rank,
                        7,
                        dim,
                    )
                    assert element_offsets is not None
                    if fused:
                        feature_major = torch.ops.torchrec.embedding_pooled_batched_lookup_offset_fused_adagrad(
                            indices,
                            offsets,
                            weights_cpu,
                            row_offsets,
                            momentum_cpu,
                            0.1,
                            0.01,
                            dim,
                        )
                    else:
                        feature_major = (
                            torch.ops.torchrec.embedding_pooled_batched_lookup_offset(
                                indices,
                                offsets,
                                weights_cpu,
                                row_offsets,
                                dim,
                            )
                        )
                    if mode == PoolingMode.MEAN:
                        feature_major = (
                            feature_major / torch.diff(offsets).clamp(min=1)[:, None]
                        )
                    expected = (
                        torch.zeros_like(feature_major)
                        .index_copy(0, (element_offsets // dim).long(), feature_major)
                        .flatten()
                    )
                    with patch.object(
                        impl,
                        "_generate_vbe_metadata",
                        wraps=impl._generate_vbe_metadata,
                    ) as kernel:
                        actual = module(
                            indices.to("tpu"),
                            offsets.to("tpu"),
                            batch_size_per_feature_per_rank=counts,
                        )
                        torch.testing.assert_close(actual.cpu(), expected)
                        gradient = torch.linspace(0.1, 1.0, actual.numel())
                        expected.backward(gradient)
                        actual.backward(gradient.to("tpu"))
                        kernel.assert_called_once()
                    if fused:
                        self.assertIsNone(module.weights.grad)
                        torch.testing.assert_close(
                            module.weights.detach().cpu(),
                            weights_cpu.detach(),
                            rtol=1e-4,
                            atol=1e-5,
                        )
                        assert module.momentum is not None
                        torch.testing.assert_close(
                            module.momentum.cpu(), momentum_cpu, rtol=1e-4, atol=1e-5
                        )
                    else:
                        assert module.weights.grad is not None
                        torch.testing.assert_close(
                            module.weights.grad.cpu(),
                            weights_cpu.grad,
                            rtol=1e-4,
                            atol=1e-5,
                        )

    def test_vbe_int64_offsets_forward_backward_and_fused_update(self) -> None:
        dim = EMB_DIM
        indices = torch.tensor([2, 2, 1, 0, 1] + [0] * 11, dtype=torch.int32)
        offsets = torch.tensor([0, 2, 2, 3, 5], dtype=torch.int32)
        row_offsets = torch.tensor([0, 0, 3, 3], dtype=torch.int32)
        output_order = torch.tensor([0, 2, 1, 3])
        row_output_offsets = torch.tensor([0, 2 * dim, dim, 3 * dim], dtype=torch.int64)
        grad_out = torch.tensor([1.0, 3.0, 99.0, 5.0])[:, None].repeat(1, dim)
        initial_weights = torch.arange(64 * dim, dtype=torch.float32) * 0.01

        for fused in (False, True):
            with self.subTest(fused=fused):
                weights_cpu = initial_weights.clone().requires_grad_()
                weights_tpu = initial_weights.to("tpu").requires_grad_()
                momentum_cpu = torch.zeros(64)
                momentum_tpu = torch.zeros(64, device="tpu")
                if fused:
                    op = (
                        torch.ops.torchrec.embedding_pooled_batched_lookup_offset_fused_adagrad
                    )
                    cpu_args = (momentum_cpu, 0.1, 0.01, dim)
                    tpu_args = (momentum_tpu, 0.1, 0.01, dim)
                else:
                    op = torch.ops.torchrec.embedding_pooled_batched_lookup_offset
                    cpu_args = (dim,)
                    tpu_args = (dim,)

                expected = op(indices, offsets, weights_cpu, row_offsets, *cpu_args)
                actual = op(
                    indices.to("tpu"),
                    offsets.to("tpu"),
                    weights_tpu,
                    row_offsets.to("tpu"),
                    *tpu_args,
                    row_output_offsets=row_output_offsets.to("tpu"),
                )
                torch.testing.assert_close(actual.cpu(), expected[output_order])
                expected.backward(grad_out[output_order])
                actual.backward(grad_out.to("tpu"))
                if fused:
                    self.assertIsNone(weights_tpu.grad)
                    torch.testing.assert_close(
                        weights_tpu.detach().cpu(), weights_cpu.detach()
                    )
                    torch.testing.assert_close(momentum_tpu.cpu(), momentum_cpu)
                else:
                    self.assertIsNotNone(weights_tpu.grad)
                    self.assertIsNotNone(weights_cpu.grad)
                    torch.testing.assert_close(weights_tpu.grad.cpu(), weights_cpu.grad)

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
