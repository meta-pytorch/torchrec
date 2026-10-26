#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401
import numpy as np
import torch
import torch.distributed as dist
from parameterized import parameterized


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyrefly: ignore [missing-import]  # noqa: F401

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


def _setup_tpu() -> None:
    if not tpu_is_available():
        raise unittest.SkipTest("requires an attached TPU")
    from torchrec.experimental.torch_tpu.pallas import dispatcher  # noqa: F401

    _ = dispatcher
    if not dist.is_initialized():
        dist.init_process_group(backend="tpu_dist")


def tearDownModule() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _covering_indices(
    rows: int, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Random inverse_indices for one feature where every bag in [0, rows) is used."""
    extra = torch.randint(0, rows, (batch_size - rows,), generator=generator)
    idx = torch.cat([torch.arange(rows), extra])
    return idx[torch.randperm(batch_size, generator=generator)]


def _make_case(
    batch_size: int, input_rows: list[int], pattern: str, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    per_feature = []
    for rows in input_rows:
        if pattern == "modulo":
            per_feature.append(torch.arange(batch_size) % rows)
        elif pattern == "random":
            per_feature.append(_covering_indices(rows, batch_size, generator))
        elif pattern == "bag_crosses_16":
            # bag 0 has 17 samples, the rest go to the last bag
            idx = torch.full((batch_size,), rows - 1)
            idx[torch.randperm(batch_size, generator=generator)[:17]] = 0
            per_feature.append(idx)
        elif pattern == "unused_bag":
            # bag 1 is never referenced, so its gradient must be zero
            idx = torch.arange(batch_size) % rows
            idx[idx == 1] = 0
            per_feature.append(idx)
        else:
            raise ValueError(pattern)
    return torch.cat(per_feature).to(torch.int64)


# (name, batch_size, input_rows, input_columns, pattern)
# All cases satisfy the SparseCore constraints: B % 16 == 0 and every D % 16 == 0.
CASES = [
    ("small_two_features", 16, [2, 2], [16, 16], "modulo"),
    ("heavy_dedup_one_bag", 64, [1, 1], [16, 32], "modulo"),
    ("no_dedup", 32, [32, 32], [16, 16], "random"),
    ("bag_crosses_16_samples", 32, [2, 3], [16, 32], "bag_crosses_16"),
    ("mixed_dims", 48, [3, 5, 7, 1], [16, 64, 32, 128], "random"),
    ("single_feature", 16, [5], [48], "random"),
    ("unused_bag_gets_zero_grad", 16, [4, 3], [16, 16], "unused_bag"),
    ("large_random", 256, [17, 100, 256], [32, 16, 48], "random"),
]


def _fbgemm_cpu_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    input_rows: list[int],
    input_columns: list[int],
) -> torch.Tensor:
    num_indices = torch.tensor([batch_size] * len(input_rows))
    numel = sum(r * c for r, c in zip(input_rows, input_columns))
    return torch.ops.fbgemm.batch_index_select_dim0_backward_cpu_impl(
        grad_output.contiguous(),
        indices,
        num_indices,
        num_indices,
        torch.tensor(input_rows),
        torch.tensor(input_columns),
        True,
        torch.tensor([numel]),
    )


def _grouped_reference_backward(
    grad_output: np.ndarray,
    grad_rows: np.ndarray,
    bag_offsets: np.ndarray,
    input_rows: list[int],
    input_columns: list[int],
) -> np.ndarray:
    """Backward computed the way the SparseCore kernel does: one bag at a time."""
    batch_size = grad_rows.shape[1]
    grad_output = grad_output.reshape(batch_size, sum(input_columns))
    input_offsets = np.cumsum([0] + [r * c for r, c in zip(input_rows, input_columns)])
    column_offsets = np.cumsum([0] + list(input_columns))
    out = np.zeros(input_offsets[-1], dtype=grad_output.dtype)
    for f, (rows, dim) in enumerate(zip(input_rows, input_columns)):
        for r in range(rows):
            samples = grad_rows[f, bag_offsets[f, r] : bag_offsets[f, r + 1]]
            cols = slice(column_offsets[f], column_offsets[f] + dim)
            start = input_offsets[f] + r * dim
            out[start : start + dim] = grad_output[samples, cols].sum(axis=0)
    return out


class GroupBatchesByUniqueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _setup_tpu()
        from torch_tpu._internal import pallas  # pyrefly: ignore [missing-import]
        from torchrec.experimental.torch_tpu.pallas.batch_index import (
            group_batches_by_unique,
        )

        cls._group_batches_by_unique = staticmethod(
            pallas.jax_op(
                "torchrec_pallas_test::group_batches_by_unique", group_batches_by_unique
            )
        )

    def test_docstring_example(self) -> None:
        indices = torch.tensor(
            [0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.int32, device="tpu"
        )
        grad_rows, bag_offsets = self._group_batches_by_unique(
            indices=indices, num_features=2, batch_size=4
        )
        grad_rows, bag_offsets = grad_rows.cpu().numpy(), bag_offsets.cpu().numpy()
        np.testing.assert_array_equal(grad_rows, [[0, 1, 3, 2], [0, 2, 3, 1]])
        np.testing.assert_array_equal(bag_offsets, [[0, 3, 4, 4, 4], [0, 3, 4, 4, 4]])

    @parameterized.expand(CASES)
    def test_groups_every_sample_into_its_bag(
        self,
        _name: str,
        batch_size: int,
        input_rows: list[int],
        _input_columns: list[int],
        pattern: str,
    ) -> None:
        indices = _make_case(batch_size, input_rows, pattern)
        grad_rows, bag_offsets = self._group_batches_by_unique(
            indices=indices.to(device="tpu", dtype=torch.int32),
            num_features=len(input_rows),
            batch_size=batch_size,
        )
        grad_rows, bag_offsets = grad_rows.cpu().numpy(), bag_offsets.cpu().numpy()
        idx = indices.numpy().reshape(len(input_rows), batch_size)

        self.assertEqual(bag_offsets.shape, (len(input_rows), batch_size + 1))
        for f, rows in enumerate(input_rows):
            self.assertEqual(bag_offsets[f, -1], batch_size)
            for r in range(batch_size):
                members = grad_rows[f, bag_offsets[f, r] : bag_offsets[f, r + 1]]
                expected = np.flatnonzero(idx[f] == r) if r < rows else []
                np.testing.assert_array_equal(np.sort(members), expected)

    @parameterized.expand(CASES)
    def test_grouped_backward_matches_fbgemm_cpu(
        self,
        _name: str,
        batch_size: int,
        input_rows: list[int],
        input_columns: list[int],
        pattern: str,
    ) -> None:
        indices = _make_case(batch_size, input_rows, pattern)
        grad_output = torch.randn(
            batch_size * sum(input_columns), generator=torch.Generator().manual_seed(1)
        )
        expected = _fbgemm_cpu_backward(
            grad_output, indices, batch_size, input_rows, input_columns
        )

        grad_rows, bag_offsets = self._group_batches_by_unique(
            indices=indices.to(device="tpu", dtype=torch.int32),
            num_features=len(input_rows),
            batch_size=batch_size,
        )
        actual = _grouped_reference_backward(
            grad_output.numpy(),
            grad_rows.cpu().numpy(),
            bag_offsets.cpu().numpy(),
            input_rows,
            input_columns,
        )

        torch.testing.assert_close(torch.from_numpy(actual), expected)


class BatchIndexSelectTest(unittest.TestCase):
    def test_cpu_backward_impl_matches_autograd(self) -> None:
        inputs = torch.tensor(
            [10, 11, 20, 21, 100, 200, 300, 1, 2, 3, 4, 5, 6],
            dtype=torch.float32,
            requires_grad=True,
        )
        indices = torch.tensor([1, 0, 1, 2, 0, 1, 0, 1, 0], dtype=torch.int64)
        input_num_indices = [3, 3, 3]
        input_rows = [2, 3, 2]
        input_columns = [2, 1, 3]
        output = torch.ops.fbgemm.batch_index_select_dim0(
            inputs,
            indices,
            input_num_indices,
            input_rows,
            input_columns,
            True,
        )
        grad_output = torch.randn(
            output.shape, generator=torch.Generator().manual_seed(0)
        )
        output.backward(grad_output)
        actual = _fbgemm_cpu_backward(
            grad_output, indices, 3, input_rows, input_columns
        )

        torch.testing.assert_close(actual, inputs.grad)

    @parameterized.expand(
        [(*case, False) for case in CASES]
        + [("compiled_two_features", 16, [2, 2], [16, 16], "modulo", True)]
    )
    def test_sparse_core_matches_fbgemm_cpu(
        self,
        _name: str,
        batch_size: int,
        input_rows: list[int],
        input_columns: list[int],
        pattern: str,
        compile_op: bool,
    ) -> None:
        _setup_tpu()
        input_num_indices = [batch_size] * len(input_columns)
        indices = _make_case(batch_size, input_rows, pattern)
        inputs = torch.randn(
            sum(r * c for r, c in zip(input_rows, input_columns)),
            generator=torch.Generator().manual_seed(2),
        ).requires_grad_()
        expected = torch.ops.fbgemm.batch_index_select_dim0(
            inputs, indices, input_num_indices, input_rows, input_columns, True
        )

        tpu_inputs = inputs.detach().to("tpu").requires_grad_()
        op = torch.ops.fbgemm.batch_index_select_dim0
        if compile_op:
            op = torch.compile(op, backend="aot_eager", fullgraph=True)
        actual = op(
            tpu_inputs,
            indices.to("tpu"),
            input_num_indices,
            input_rows,
            input_columns,
            True,
        )

        torch.testing.assert_close(actual.cpu(), expected)
        grad_output = torch.randn(
            expected.shape, generator=torch.Generator().manual_seed(0)
        )
        expected.backward(grad_output)
        actual.backward(grad_output.to("tpu"))
        assert tpu_inputs.grad is not None
        assert inputs.grad is not None
        torch.testing.assert_close(tpu_inputs.grad.cpu(), inputs.grad)

    @parameterized.expand(
        [
            ("unaligned_batch", 15, [16, 32], True),
            ("unaligned_columns", 16, [18, 32], True),
            ("feature_major", 16, [16, 32], False),
        ]
    )
    def test_cpu_fallback_matches_fbgemm_cpu(
        self,
        _name: str,
        batch_size: int,
        input_columns: list[int],
        permute_output_dim_0_1: bool,
    ) -> None:
        _setup_tpu()
        input_rows = [2, 3]
        indices = _make_case(batch_size, input_rows, "modulo")
        inputs = torch.randn(
            sum(r * c for r, c in zip(input_rows, input_columns)),
            generator=torch.Generator().manual_seed(2),
        ).requires_grad_()
        tpu_inputs = inputs.detach().to("tpu").requires_grad_()
        args = ([batch_size] * len(input_rows), input_rows, input_columns)
        expected = torch.ops.fbgemm.batch_index_select_dim0(
            inputs, indices, *args, permute_output_dim_0_1
        )
        actual = torch.ops.fbgemm.batch_index_select_dim0(
            tpu_inputs, indices.to("tpu"), *args, permute_output_dim_0_1
        )

        torch.testing.assert_close(actual.cpu(), expected)
        grad_output = torch.randn(
            expected.shape, generator=torch.Generator().manual_seed(0)
        )
        expected.backward(grad_output)
        actual.backward(grad_output.to("tpu"))
        self.assertIsNotNone(tpu_inputs.grad)
        self.assertIsNotNone(inputs.grad)
        torch.testing.assert_close(tpu_inputs.grad.cpu(), inputs.grad)

    @parameterized.expand(CASES)
    def test_sparse_core_backward_kernel_matches_fbgemm_cpu(
        self,
        _name: str,
        batch_size: int,
        input_rows: list[int],
        input_columns: list[int],
        pattern: str,
    ) -> None:
        _setup_tpu()
        indices = _make_case(batch_size, input_rows, pattern)
        grad_output = torch.randn(
            batch_size * sum(input_columns), generator=torch.Generator().manual_seed(1)
        )
        expected = _fbgemm_cpu_backward(
            grad_output, indices, batch_size, input_rows, input_columns
        )

        from torchrec.experimental.torch_tpu.pallas import impl

        actual = impl.batch_index_select_dim0_backward_tpu_impl(
            grad_output.to("tpu"),
            indices.to("tpu"),
            input_rows,
            input_columns,
            expected.numel(),
        )

        torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
