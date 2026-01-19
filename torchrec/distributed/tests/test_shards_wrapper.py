#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional, Union

import torch
from torch import distributed as dist
from torchrec.distributed.shards_wrapper import LocalShardsWrapper
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.test_utils import skip_if_asan_class


def all_gather_into_tensor(
    rank: int,
    world_size: int,
    backend: str,
    expected_result: Union[torch.Tensor, List[torch.Tensor]],
    shards_wrapper: List[LocalShardsWrapper],
    local_size: Optional[int] = None,
    async_op: bool = False,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        local_shards_wrapper = shards_wrapper[ctx.rank]
        output_tensor = torch.empty((8, 5), device=torch.device(f"cuda:{ctx.rank}"))
        res = dist.all_gather_into_tensor(
            output_tensor, local_shards_wrapper, group=ctx.pg, async_op=async_op
        )
        if async_op:
            res.wait()
        torch.testing.assert_close(
            output_tensor.cpu(),
            expected_result,
        )


def all_gather(
    rank: int,
    world_size: int,
    backend: str,
    expected_result: Union[torch.Tensor, List[torch.Tensor]],
    shards_wrapper: List[LocalShardsWrapper],
    local_size: Optional[int] = None,
    async_op: bool = False,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        local_shards_wrapper = shards_wrapper[ctx.rank]
        tensor_list = [
            torch.zeros((4, 5), dtype=torch.float32, device=f"cuda:{rank}")
            for _ in range(2)
        ]
        res = dist.distributed_c10d.all_gather(
            tensor_list,
            local_shards_wrapper,
            async_op=True,
        )
        if async_op:
            res.wait()
        for tensor, expected in zip(tensor_list, expected_result):
            torch.testing.assert_close(
                tensor.cpu(),
                expected.cpu(),
            )


def all_gather_object(
    rank: int,
    world_size: int,
    backend: str,
    expected_result: Union[torch.Tensor, List[torch.Tensor]],
    shards_wrapper: List[LocalShardsWrapper],
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        local_shards_wrapper = shards_wrapper[ctx.rank]
        output = [None] * world_size
        dist.distributed_c10d.all_gather_object(
            output,
            local_shards_wrapper,
        )
        for i in range(world_size):
            torch.testing.assert_close(
                output[i]._local_shards[0],  # pyre-ignore[16]
                shards_wrapper[i]._local_shards[0],
            )


@skip_if_asan_class
class LocalShardsWrapperDistributedTest(MultiProcessTestBase):
    def setUp(self, backend: str = "nccl") -> None:
        super().setUp()

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @unittest.skip("Need to fix circular import errors with Torch")
    def test_shards_wrapper_all_gather_into_tensor(self) -> None:
        world_size = 2
        backend = "nccl"
        shards_0 = [torch.rand((4, 5), device=torch.device("cuda:0"))]
        shards_1 = [torch.rand((4, 5), device=torch.device("cuda:1"))]
        expected_result = torch.cat(
            [torch.cat(shards_0, dim=0).cpu(), torch.cat(shards_1, dim=0).cpu()], dim=0
        )
        offsets = [(0, 0)]

        # shards wrapper for rank 0 and rank 1, offsets don't matter
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        ls_1 = LocalShardsWrapper(local_shards=shards_1, local_offsets=offsets)

        self._run_multi_process_test(
            callable=all_gather_into_tensor,
            shards_wrapper=[
                ls_0,
                ls_1,
            ],
            expected_result=expected_result,
            world_size=world_size,
            backend=backend,
        )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @unittest.skip("Need to fix circular import errors with Torch")
    def test_shards_wrapper_all_gather(self) -> None:
        world_size = 2
        backend = "nccl"
        shards_0 = [torch.rand((4, 5), device=torch.device("cuda:0"))]
        shards_1 = [torch.zeros((4, 5), device=torch.device("cuda:1"))]
        expected_result = [shards_0[0], shards_1[0]]
        offsets = [(0, 0)]

        # shards wrapper for rank 0 and rank 1, offsets don't matter
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        ls_1 = LocalShardsWrapper(local_shards=shards_1, local_offsets=offsets)

        self._run_multi_process_test(
            callable=all_gather,
            shards_wrapper=[
                ls_0,
                ls_1,
            ],
            expected_result=expected_result,
            world_size=world_size,
            backend=backend,
        )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @unittest.skip("Need to fix circular import errors with Torch")
    def test_shards_wrapper_all_gather_object(self) -> None:
        world_size = 2
        backend = "nccl"
        shards_0 = [torch.rand((4, 5), device=torch.device("cuda:0"))]
        shards_1 = [torch.zeros((4, 5), device=torch.device("cuda:1"))]
        expected_result = [shards_0[0], shards_1[0]]
        offsets = [(0, 0)]

        # shards wrapper for rank 0 and rank 1, offsets don't matter
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        ls_1 = LocalShardsWrapper(local_shards=shards_1, local_offsets=offsets)

        self._run_multi_process_test(
            callable=all_gather_object,
            shards_wrapper=[
                ls_0,
                ls_1,
            ],
            expected_result=expected_result,
            world_size=world_size,
            backend=backend,
        )
