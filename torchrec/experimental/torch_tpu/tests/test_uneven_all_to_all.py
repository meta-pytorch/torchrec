#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import unittest
from typing import cast
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torchrec.distributed.comm_ops import AllToAllSingle, pg_name
from torchrec.distributed.test_utils.process_runner import (
    run_local_multi_process_func,
    SingleProcessContext,
)
from torchrec.experimental.torch_tpu.uneven_all_to_all import (
    maybe_all2all_pooled_uneven_tpu,
    maybe_kjt_a2a_uneven_tpu,
    maybe_variable_batch_all2all_pooled_uneven_tpu,
)


class UnevenAllToAllTest(unittest.TestCase):
    def test_variable_batch_rejects_invalid_rank_metadata(self) -> None:
        group = MagicMock()
        group.size.return_value = 2
        group.rank.return_value = 0
        input_embeddings = torch.empty(0)
        valid_batch_sizes = [[0], [0]]
        valid_embedding_dims = [[1], [1]]

        for batch_sizes, embedding_dims in (
            (valid_batch_sizes[:-1], valid_embedding_dims),
            (valid_batch_sizes, valid_embedding_dims[:-1]),
        ):
            with self.subTest(
                batch_size_rank_count=len(batch_sizes),
                embedding_dim_rank_count=len(embedding_dims),
            ):
                with self.assertRaisesRegex(
                    ValueError, "VBE metadata must match the process-group size"
                ):
                    maybe_variable_batch_all2all_pooled_uneven_tpu(
                        group,
                        input_embeddings,
                        batch_sizes,
                        [0, 0],
                        embedding_dims,
                        has_codecs=False,
                        even_all_to_all=MagicMock(),
                    )

    def test_variable_batch_with_codecs_uses_native_path(self) -> None:
        group = MagicMock()
        group.size.return_value = 2
        even_all_to_all = MagicMock()

        self.assertIsNone(
            maybe_variable_batch_all2all_pooled_uneven_tpu(
                group,
                torch.empty(0),
                [[0], [0]],
                [0, 0],
                [[1], [1]],
                has_codecs=True,
                even_all_to_all=even_all_to_all,
            )
        )
        even_all_to_all.assert_not_called()

    @staticmethod
    def _assert_kjt_all_to_all(
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        input_splits_2d = [rank + destination + 1 for destination in range(world_size)]
        output_splits_2d = [source + rank + 1 for source in range(world_size)]
        rows_2d = torch.arange(sum(input_splits_2d), device=device) + rank * 1000
        input_2d = torch.stack([rows_2d * 2, rows_2d * 2 + 1], dim=1)

        input_splits_1d = [
            int(rank != destination) for destination in range(world_size)
        ]
        output_splits_1d = [int(source != rank) for source in range(world_size)]
        input_1d = torch.arange(sum(input_splits_1d), device=device) + rank * 100
        outputs = maybe_kjt_a2a_uneven_tpu(
            group,
            [input_2d, input_1d],
            [input_splits_2d, input_splits_1d],
            [output_splits_2d, output_splits_1d],
            device,
        )
        if outputs is None:
            raise AssertionError("uneven KJT adapter did not handle uneven splits")

        expected_rows = []
        for source in range(world_size):
            source_splits = [
                source + destination + 1 for destination in range(world_size)
            ]
            start = sum(source_splits[:rank])
            expected_rows.append(
                torch.arange(start, start + output_splits_2d[source], device=device)
                + source * 1000
            )
        expected_rows_tensor = torch.cat(expected_rows)
        expected = torch.stack(
            [expected_rows_tensor * 2, expected_rows_tensor * 2 + 1], dim=1
        )
        torch.testing.assert_close(outputs[0], expected)

        expected_1d = torch.tensor(
            [source * 100 for source in range(world_size) if source != rank],
            device=device,
        )
        torch.testing.assert_close(outputs[1], expected_1d)

        even_splits = [[1] * world_size]
        even_output = maybe_kjt_a2a_uneven_tpu(
            group,
            [torch.arange(world_size, device=device)],
            even_splits,
            even_splits,
            device,
        )
        if even_output is not None:
            raise AssertionError("even KJT splits should use the native path")

    @staticmethod
    def _assert_pooled_all_to_all(
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        def even_all_to_all(tensor: torch.Tensor, split_size: int) -> torch.Tensor:
            splits = [split_size] * world_size
            return cast(
                torch.Tensor,
                AllToAllSingle.apply(
                    tensor, splits, splits, pg_name(group), world_size, False
                ),
            )

        batch_sizes = [rank_index + 1 for rank_index in range(world_size)]
        dimensions = [rank_index + 1 for rank_index in range(world_size)]
        local_dim = dimensions[rank]
        pooled_input = (
            torch.arange(sum(batch_sizes) * local_dim, device=device).float()
            + rank * 1000
        ).view(sum(batch_sizes), local_dim)
        pooled_input.requires_grad_()
        output = maybe_all2all_pooled_uneven_tpu(
            group,
            pooled_input,
            batch_sizes,
            dimensions,
            has_codecs=False,
            even_all_to_all=even_all_to_all,
        )
        if output is None:
            raise AssertionError("uneven pooled adapter did not handle uneven splits")

        batch_start = sum(batch_sizes[:rank])
        expected_blocks = []
        for source, source_dim in enumerate(dimensions):
            source_input = (
                torch.arange(sum(batch_sizes) * source_dim, device=device).float()
                + source * 1000
            ).view(sum(batch_sizes), source_dim)
            expected_blocks.append(
                source_input[batch_start : batch_start + batch_sizes[rank]]
            )
        torch.testing.assert_close(output, torch.cat(expected_blocks, dim=1))
        output.sum().backward()
        if pooled_input.grad is None:
            raise AssertionError("padded all-to-all did not propagate gradients")
        torch.testing.assert_close(pooled_input.grad, torch.ones_like(pooled_input))

        even_batch_sizes = [2] * world_size
        even_dimensions = [3] * world_size
        even_input = torch.zeros(
            sum(even_batch_sizes), even_dimensions[rank], device=device
        )
        if (
            maybe_all2all_pooled_uneven_tpu(
                group,
                even_input,
                even_batch_sizes,
                even_dimensions,
                has_codecs=False,
                even_all_to_all=even_all_to_all,
            )
            is not None
        ):
            raise AssertionError("even pooled splits should use the native path")

    @staticmethod
    def _assert_variable_batch_all_to_all(
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        def even_all_to_all(tensor: torch.Tensor, split_size: int) -> torch.Tensor:
            splits = [split_size] * world_size
            return cast(
                torch.Tensor,
                AllToAllSingle.apply(
                    tensor, splits, splits, pg_name(group), world_size, False
                ),
            )

        emb_dim_per_rank_per_feature = [[source + 1] for source in range(world_size)]
        batch_size_per_rank_per_feature = [
            [destination + 1] for destination in range(world_size)
        ]
        batch_size_per_feature_pre_a2a = [rank + 1] * world_size
        input_splits = [
            (rank + 1) * (destination + 1) for destination in range(world_size)
        ]
        output_splits = [(source + 1) * (rank + 1) for source in range(world_size)]
        input_blocks = [
            torch.arange(size, device=device).float() + rank * 1000 + destination * 100
            for destination, size in enumerate(input_splits)
        ]
        variable_input = torch.cat(input_blocks).requires_grad_()
        output = maybe_variable_batch_all2all_pooled_uneven_tpu(
            group,
            variable_input,
            batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a,
            emb_dim_per_rank_per_feature,
            has_codecs=False,
            even_all_to_all=even_all_to_all,
        )
        if output is None:
            raise AssertionError("uneven VBE adapter did not handle uneven splits")

        expected = torch.cat(
            [
                torch.arange(output_splits[source], device=device).float()
                + source * 1000
                + rank * 100
                for source in range(world_size)
            ]
        )
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        if variable_input.grad is None:
            raise AssertionError("padded VBE all-to-all did not propagate gradients")
        torch.testing.assert_close(variable_input.grad, torch.ones_like(variable_input))

    @classmethod
    def _run_uneven_all_to_all(
        cls,
        ctx: SingleProcessContext,
        rank: int,
        world_size: int,
        device_type: str,
        test_kind: str,
    ) -> None:
        group = ctx.pg
        if group is None:
            raise AssertionError("process group is not initialized")
        device = torch.device(device_type)
        if test_kind == "kjt":
            cls._assert_kjt_all_to_all(rank, world_size, group, device)
        elif test_kind == "pooled":
            cls._assert_pooled_all_to_all(rank, world_size, group, device)
        else:
            cls._assert_variable_batch_all_to_all(rank, world_size, group, device)

    def _run_backend(self, backend: str, device_type: str, test_kind: str) -> bool:
        if backend == "gloo":
            run_local_multi_process_func(
                func=self._run_uneven_all_to_all,
                world_size=2,
                backend=backend,
                device_type=device_type,
                test_kind=test_kind,
            )
            return True

        try:
            import torch_tpu  # pyre-ignore[21]  # noqa: F401
        except ImportError:
            self.skipTest("requires torch_tpu")
        configured_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tpu = getattr(torch, "tpu", None)
        if tpu is None or not tpu.is_available() or configured_world_size < 2:
            self.skipTest("requires a multi-rank TPU launch")
        with SingleProcessContext(backend=backend) as ctx:
            self._run_uneven_all_to_all(
                ctx=ctx,
                rank=ctx.rank,
                world_size=ctx.world_size,
                device_type=device_type,
                test_kind=test_kind,
            )
        return True

    @parameterized.expand([("gloo", "cpu"), ("tpu_dist", "tpu")])
    def test_kjt_forward(self, backend: str, device_type: str) -> None:
        self.assertTrue(self._run_backend(backend, device_type, "kjt"))

    @parameterized.expand([("gloo", "cpu"), ("tpu_dist", "tpu")])
    def test_pooled_forward_and_backward(self, backend: str, device_type: str) -> None:
        self.assertTrue(self._run_backend(backend, device_type, "pooled"))

    @parameterized.expand([("gloo", "cpu"), ("tpu_dist", "tpu")])
    def test_variable_batch_forward_and_backward(
        self, backend: str, device_type: str
    ) -> None:
        self.assertTrue(self._run_backend(backend, device_type, "variable_batch"))
