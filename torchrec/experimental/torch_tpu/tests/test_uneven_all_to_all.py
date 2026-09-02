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
)


class UnevenAllToAllTest(unittest.TestCase):
    @staticmethod
    def _assert_kjt_all_to_all(
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
        device: torch.device,
    ) -> None:
        input_splits = [rank + destination + 1 for destination in range(world_size)]
        output_splits = [source + rank + 1 for source in range(world_size)]
        rows = torch.arange(sum(input_splits), device=device) + rank * 1000
        input_2d = torch.stack([rows * 2, rows * 2 + 1], dim=1)
        outputs = maybe_kjt_a2a_uneven_tpu(
            group, [input_2d], [input_splits], [output_splits], device
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
                torch.arange(start, start + output_splits[source], device=device)
                + source * 1000
            )
        expected_rows_tensor = torch.cat(expected_rows)
        expected = torch.stack(
            [expected_rows_tensor * 2, expected_rows_tensor * 2 + 1], dim=1
        )
        torch.testing.assert_close(outputs[0], expected)

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
        else:
            cls._assert_pooled_all_to_all(rank, world_size, group, device)

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
