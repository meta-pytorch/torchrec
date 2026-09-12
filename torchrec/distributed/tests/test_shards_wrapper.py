#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from types import SimpleNamespace
from typing import Any, cast, List, Optional, Union
from unittest.mock import patch

import torch
from torch import distributed as dist
from torchrec.distributed.embedding_kernel import get_state_dict
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
        if res is not None:
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
        output: List[Any] = [None] * world_size
        dist.distributed_c10d.all_gather_object(
            output,
            local_shards_wrapper,
        )
        for i in range(world_size):
            torch.testing.assert_close(
                # pyrefly: ignore[missing-attribute]
                output[i]._local_shards[0],
                shards_wrapper[i]._local_shards[0],
            )


class LocalShardsWrapperTest(unittest.TestCase):
    def test_dtensor_state_dict_flattens_fragmented_table(self) -> None:
        fragments = [torch.ones((2, 3)), torch.full((3, 3), 2.0)]
        wrapper = LocalShardsWrapper(
            local_shards=fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=torch.Size([5, 3]),
        )
        embedding_table: Any = SimpleNamespace(
            name="table",
            use_virtual_table=False,
            compute_kernel="fused_triton",
            local_rows=5,
            local_cols=3,
            local_metadata=SimpleNamespace(shard_offsets=[10, 4]),
            dtensor_metadata=SimpleNamespace(
                mesh=object(),
                placements=(),
                size=(20, 12),
                stride=(12, 1),
            ),
            global_metadata=None,
        )

        with patch("torchrec.distributed.embedding_kernel.DTensor") as dtensor:
            state = get_state_dict(
                [embedding_table],
                [wrapper],
                pg=cast(Any, object()),
            )

        local_tensor = dtensor.from_local.call_args.kwargs["local_tensor"]
        self.assertIs(state["table.weight"], dtensor.from_local.return_value)
        self.assertIsInstance(local_tensor, LocalShardsWrapper)
        self.assertEqual(torch.Size([5, 3]), local_tensor.size())
        self.assertEqual(
            [torch.Size([10, 4]), torch.Size([12, 4])],
            local_tensor.local_offsets(),
        )
        self.assertIs(fragments[0], local_tensor.local_shards()[0])
        self.assertIs(fragments[1], local_tensor.local_shards()[1])

    def test_explicit_logical_size_for_row_fragments(self) -> None:
        fragments = [torch.zeros((2, 3)), torch.zeros((3, 3))]
        logical_size = torch.Size([5, 3])

        wrapper = LocalShardsWrapper(
            local_shards=fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=logical_size,
        )

        self.assertEqual(logical_size, wrapper.size())
        self.assertEqual(logical_size, wrapper.storage_metadata().size)
        self.assertEqual(
            [torch.Size([2, 3]), torch.Size([3, 3])], wrapper.local_sizes()
        )

    def test_wrapper_preserves_fragment_tensor_metadata(self) -> None:
        fragment = torch.ones((2, 3), dtype=torch.float16, requires_grad=True)

        wrapper = LocalShardsWrapper(
            local_shards=[fragment],
            local_offsets=[(0, 0)],
        )

        self.assertEqual(torch.float16, wrapper.dtype)
        self.assertEqual(fragment.device, wrapper.device)
        self.assertEqual(fragment.layout, wrapper.layout)
        self.assertTrue(wrapper.requires_grad)
        parameter = torch.nn.Parameter(wrapper, requires_grad=False)
        self.assertFalse(parameter.requires_grad)

    def test_legacy_columnwise_size_inference(self) -> None:
        wrapper = LocalShardsWrapper(
            local_shards=[torch.zeros((2, 2)), torch.zeros((2, 3))],
            local_offsets=[(0, 0), (0, 2)],
        )

        self.assertEqual(torch.Size([2, 5]), wrapper.size())
        self.assertEqual(torch.Size([2, 5]), wrapper.storage_metadata().size)

    def test_copy_from_dense_tensor_to_row_fragments(self) -> None:
        fragments = [torch.zeros((2, 3)), torch.zeros((3, 3))]
        wrapper = LocalShardsWrapper(
            local_shards=fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=torch.Size([5, 3]),
        )
        source = torch.arange(15, dtype=torch.float32).view(5, 3)

        result = wrapper.copy_(source)

        self.assertIs(wrapper, result)
        torch.testing.assert_close(source[:2], fragments[0])
        torch.testing.assert_close(source[2:], fragments[1])

    def test_uniform_initializes_every_row_fragment_in_place(self) -> None:
        fragments = [torch.full((2, 3), -1.0), torch.full((3, 3), -1.0)]
        logical_size = torch.Size([5, 3])
        wrapper = LocalShardsWrapper(
            local_shards=fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=logical_size,
        )

        result = torch.nn.init.uniform_(wrapper, a=0.25, b=0.75)

        self.assertIs(wrapper, result)
        self.assertEqual(logical_size, wrapper.size())
        self.assertEqual(logical_size, wrapper.storage_metadata().size)
        self.assertEqual(
            [torch.Size([0, 0]), torch.Size([2, 0])],
            wrapper.local_offsets(),
        )
        for original, initialized in zip(fragments, wrapper.local_shards()):
            self.assertIs(original, initialized)
            self.assertTrue(torch.all(initialized >= 0.25))
            self.assertTrue(torch.all(initialized <= 0.75))

    def test_copy_from_wrapper_preserves_existing_behavior(self) -> None:
        source_fragments = [torch.ones((2, 3)), torch.full((3, 3), 2.0)]
        source = LocalShardsWrapper(
            local_shards=source_fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=torch.Size([5, 3]),
        )
        destination_fragments = [torch.zeros((2, 3)), torch.zeros((3, 3))]
        destination = LocalShardsWrapper(
            local_shards=destination_fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=torch.Size([5, 3]),
        )

        result = destination.copy_(source)

        self.assertIs(destination, result)
        torch.testing.assert_close(source_fragments[0], destination_fragments[0])
        torch.testing.assert_close(source_fragments[1], destination_fragments[1])

    def test_copy_from_wrapper_with_different_fragment_layout(self) -> None:
        source_fragments = [
            torch.arange(6, dtype=torch.float32).view(2, 3),
            torch.arange(6, 15, dtype=torch.float32).view(3, 3),
        ]
        source = LocalShardsWrapper(
            local_shards=source_fragments,
            local_offsets=[(0, 0), (2, 0)],
            logical_size=torch.Size([5, 3]),
        )
        destination_fragments = [
            torch.zeros((1, 3)),
            torch.zeros((3, 3)),
            torch.zeros((1, 3)),
        ]
        destination = LocalShardsWrapper(
            local_shards=destination_fragments,
            local_offsets=[(0, 0), (1, 0), (4, 0)],
            logical_size=torch.Size([5, 3]),
        )

        destination.copy_(source)

        torch.testing.assert_close(source_fragments[0][:1], destination_fragments[0])
        torch.testing.assert_close(
            torch.cat([source_fragments[0][1:], source_fragments[1][:2]]),
            destination_fragments[1],
        )
        torch.testing.assert_close(source_fragments[1][2:], destination_fragments[2])

    def test_clone_and_detach_preserve_logical_size(self) -> None:
        logical_size = torch.Size([5, 3])
        wrapper = LocalShardsWrapper(
            local_shards=[torch.ones((2, 3)), torch.ones((3, 3))],
            local_offsets=[(0, 0), (2, 0)],
            logical_size=logical_size,
        ).requires_grad_()

        cloned = cast(LocalShardsWrapper, wrapper.clone())
        detached = cast(LocalShardsWrapper, wrapper.detach())

        self.assertEqual(logical_size, cloned.size())
        self.assertEqual(logical_size, cloned.storage_metadata().size)
        self.assertEqual(logical_size, detached.size())
        self.assertEqual(logical_size, detached.storage_metadata().size)
        self.assertTrue(wrapper.requires_grad)
        self.assertFalse(detached.requires_grad)
        self.assertFalse(detached.storage_metadata().properties.requires_grad)
        self.assertTrue(
            all(not shard.requires_grad for shard in detached.local_shards())
        )


@skip_if_asan_class
class LocalShardsWrapperDistributedTest(MultiProcessTestBase):
    def setUp(self, backend: str = "nccl") -> None:
        super().setUp()

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
        # pyrefly: ignore[bad-argument-type]
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        # pyrefly: ignore[bad-argument-type]
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
        # pyrefly: ignore[bad-argument-type]
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        # pyrefly: ignore[bad-argument-type]
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
        # pyrefly: ignore[bad-argument-type]
        ls_0 = LocalShardsWrapper(local_shards=shards_0, local_offsets=offsets)
        # pyrefly: ignore[bad-argument-type]
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
