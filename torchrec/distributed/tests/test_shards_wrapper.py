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
from torchrec.test_utils import seed_and_log, skip_if_asan_class


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


class LocalShardsWrapperPaddingTest(unittest.TestCase):
    """Test cases for constant padding functionality in LocalShardsWrapper."""

    def test_empty_shards_padding(self) -> None:
        """Test padding with empty shards list."""
        lsw = LocalShardsWrapper([], [])
        pad_spec = [1, 2, 3, 4]
        pad_value = 5.0

        self.assertRaises(
            Exception,
            torch.ops.aten.constant_pad_nd.default,
            lsw,
            pad_spec,
            pad_value,
        )

    def test_invalid_1d_rw_padding(self) -> None:
        """Test invalid padding on 1D tensor throws ValueError."""
        shard1 = torch.tensor([1.0, 2.0])
        shard2 = torch.tensor([3.0, 4.0])
        lsw = LocalShardsWrapper([shard1, shard2], [(2, 0)])
        pad_spec = [1]  # invalid padding spec
        pad_value = 5.0

        self.assertRaises(
            ValueError,
            torch.ops.aten.constant_pad_nd.default,
            lsw,
            pad_spec,
            pad_value,
        )

    def test_invalid_2d_cw_padding(self) -> None:
        """Test invalid padding on 2D tensor throws ValueError."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [1, 2, 3]  # invalid padding spec
        pad_value = 5.0

        self.assertRaises(
            ValueError,
            torch.ops.aten.constant_pad_nd.default,
            lsw,
            pad_spec,
            pad_value,
        )

        pad_spec = [1]

        self.assertRaises(
            ValueError,
            torch.ops.aten.constant_pad_nd.default,
            lsw,
            pad_spec,
            pad_value,
        )

    def test_single_shard_padding_2d(self) -> None:
        """Test padding with single 2D shard."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        lsw = LocalShardsWrapper([tensor], [(0, 0)])
        pad_spec = [1, 2, 3, 4]  # [left=1, right=2, top=3, bottom=4]
        pad_value = 0.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        expected = torch.nn.functional.pad(
            tensor, pad_spec, mode="constant", value=pad_value
        )
        self.assertIsInstance(result, LocalShardsWrapper)
        self.assertEqual(len(result.local_shards()), 1)
        torch.testing.assert_close(result.local_shards()[0], expected)

    def test_single_shard_padding_1d(self) -> None:
        """Test padding with single 1D shard."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        lsw = LocalShardsWrapper([tensor], [(0,)])
        pad_spec = [2, 1]  # [top=2, bottom=1]
        pad_value = -1.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        self.assertIsInstance(result, LocalShardsWrapper)
        self.assertEqual(len(result.local_shards()), 1)

        expected = torch.nn.functional.pad(
            tensor, pad_spec, mode="constant", value=pad_value
        )
        torch.testing.assert_close(result.local_shards()[0], expected)

    def test_2d_cw_sharding_top_padding(self) -> None:
        """Test column-wise sharding with top padding (affects all shards)."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [0, 0, 2, 0]  # top=2
        pad_value = 0.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        self.assertEqual(len(result.local_shards()), 2)
        # Both shards should have 2 rows added at top
        expected_shape = (4, 2)
        self.assertEqual(result.local_shards()[0].shape, expected_shape)
        self.assertEqual(result.local_shards()[1].shape, expected_shape)

        torch.testing.assert_close(result.local_shards()[0][:2], torch.zeros(2, 2))
        torch.testing.assert_close(result.local_shards()[1][:2], torch.zeros(2, 2))
        torch.testing.assert_close(result.local_shards()[0][2:], shard1)
        torch.testing.assert_close(result.local_shards()[1][2:], shard2)

    def test_2d_cw_sharding_bottom_padding(self) -> None:
        """Test column-wise sharding with bottom padding (affects all shards)."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [0, 0, 0, 1]  # bottom=1
        pad_value = -1.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        self.assertEqual(len(result.local_shards()), 2)
        expected_shape = (3, 2)
        self.assertEqual(result.local_shards()[0].shape, expected_shape)
        self.assertEqual(result.local_shards()[1].shape, expected_shape)

        torch.testing.assert_close(result.local_shards()[0][:2], shard1)
        torch.testing.assert_close(result.local_shards()[1][:2], shard2)
        torch.testing.assert_close(
            result.local_shards()[0][2:], torch.full((1, 2), -1.0)
        )
        torch.testing.assert_close(
            result.local_shards()[1][2:], torch.full((1, 2), -1.0)
        )

    def test_2d_cw_sharding_left_padding(self) -> None:
        """Test column-wise sharding with left padding (affects first shard only)."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [3, 0, 0, 0]  # left=3
        pad_value = 2.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        self.assertEqual(len(result.local_shards()), 2)
        # First shard should have 3 columns added at left
        self.assertEqual(result.local_shards()[0].shape, (2, 5))
        self.assertEqual(result.local_shards()[1].shape, (2, 2))

        # Check content
        torch.testing.assert_close(
            result.local_shards()[0][:, :3], torch.full((2, 3), 2.0)
        )
        torch.testing.assert_close(result.local_shards()[0][:, 3:], shard1)
        torch.testing.assert_close(result.local_shards()[1], shard2)

    def test_2d_cw_sharding_right_padding(self) -> None:
        """Test column-wise sharding with right padding (affects last shard only)."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [0, 2, 0, 0]  # right=2
        pad_value = 3.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        # Second shard should have 2 columns added at right
        expected_shard_1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        expected_shard_2 = torch.tensor([[3.0, 4.0, 3.0, 3.0], [7.0, 8.0, 3.0, 3.0]])
        self.assertEqual(len(result.local_shards()), 2)
        torch.testing.assert_close(result.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result.local_shards()[1], expected_shard_2)

        # 1D padding on 2D pads the last dimension
        pad_spec_2 = [0, 2]  # right=2
        result_2 = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec_2, pad_value)
        torch.testing.assert_close(result_2.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result_2.local_shards()[1], expected_shard_2)

    def test_2d_cw_sharding_mixed_padding(self) -> None:
        """Test column-wise sharding with mixed padding directions."""
        shard1 = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        shard2 = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        lsw = LocalShardsWrapper([shard1, shard2], [(0, 0), (0, 2)])
        pad_spec = [1, 2, 1, 1]  # [left=1, right=2, top=1, bottom=1]
        pad_value = 0.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        expected_shard_1 = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
        )

        expected_shard_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [7.0, 8.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )

        self.assertEqual(len(result.local_shards()), 2)
        torch.testing.assert_close(result.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result.local_shards()[1], expected_shard_2)

    def test_1d_rw_sharding_top_padding(self) -> None:
        """Test row-wise sharding with top padding (affects first shard only)."""
        shard1 = torch.tensor([1.0, 2.0, 3.0])
        shard2 = torch.tensor([4.0, 5.0, 6.0])
        lsw = LocalShardsWrapper([shard1, shard2], [(0,), (3,)])
        pad_spec = [2, 0]  # top=2
        pad_value = 0.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        expected_shard_1 = torch.tensor(
            [0.0, 0.0, 1.0, 2.0, 3.0],
        )
        expected_shard_2 = torch.tensor(
            [4.0, 5.0, 6.0],
        )

        self.assertEqual(len(result.local_shards()), 2)
        torch.testing.assert_close(result.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result.local_shards()[1], expected_shard_2)

    def test_1d_rw_sharding_bottom_padding(self) -> None:
        """Test row-wise sharding with bottom padding (affects last shard only)."""
        shard1 = torch.tensor([1.0, 2.0, 3.0])
        shard2 = torch.tensor([4.0, 5.0, 6.0])
        lsw = LocalShardsWrapper([shard1, shard2], [(0,), (3,)])
        pad_spec = [0, 1]  # bottom=1
        pad_value = -1.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        expected_shard_1 = torch.tensor(
            [1.0, 2.0, 3.0],
        )
        expected_shard_2 = torch.tensor(
            [4.0, 5.0, 6.0, -1.0],
        )

        self.assertEqual(len(result.local_shards()), 2)
        torch.testing.assert_close(result.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result.local_shards()[1], expected_shard_2)

    def test_1d_rw_sharding_mixed_padding(self) -> None:
        """Test row-wise sharding with mixed top/bottom padding."""
        shard1 = torch.tensor([1.0, 2.0])
        shard2 = torch.tensor([3.0, 4.0])
        lsw = LocalShardsWrapper([shard1, shard2], [(0,), (2,)])
        pad_spec = [1, 2]  # [top=1, bottom=2]
        pad_value = 5.0

        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        expected_shard_1 = torch.tensor(
            [5.0, 1.0, 2.0],
        )
        expected_shard_2 = torch.tensor(
            [3.0, 4.0, 5.0, 5.0],
        )

        self.assertEqual(len(result.local_shards()), 2)
        torch.testing.assert_close(result.local_shards()[0], expected_shard_1)
        torch.testing.assert_close(result.local_shards()[1], expected_shard_2)

    def test_higher_dimensions_not_implemented(self) -> None:
        """Test that higher dimensional tensors raise NotImplementedError."""
        tensor_3d = torch.rand(2, 3, 4)  # 3D tensor
        lsw = LocalShardsWrapper([tensor_3d, tensor_3d], [(0, 0, 0), (2, 0, 0)])
        pad_spec = [1, 1, 1, 1, 1, 1]  # 3D padding spec
        pad_value = 0.0

        with self.assertRaises(NotImplementedError) as cm:
            torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, pad_value)

        self.assertIn("3D tensors is not supported", str(cm.exception))
        self.assertIn(
            "Only 1D and 2D tensors are currently supported", str(cm.exception)
        )

    def test_offsets_and_storage_metadata_after_padding_1d_rw(self) -> None:
        # Test 1D RW sharding with top+bottom padding
        shard1 = torch.tensor([1.0, 2.0])
        shard2 = torch.tensor([3.0, 4.0])
        original_offsets = [(0,), (2,)]
        lsw = LocalShardsWrapper([shard1, shard2], original_offsets)

        # Check original storage metadata
        original_storage = lsw.storage_metadata()
        self.assertEqual(original_storage.size, torch.Size([4]))  # [1,2,3,4]
        self.assertEqual(len(original_storage.chunks), 2)
        self.assertEqual(original_storage.chunks[0].offsets, torch.Size([0]))
        self.assertEqual(original_storage.chunks[0].sizes, torch.Size([2]))
        self.assertEqual(original_storage.chunks[1].offsets, torch.Size([2]))
        self.assertEqual(original_storage.chunks[1].sizes, torch.Size([2]))

        pad_spec = [1, 1]  # add 1 element at top and bottom
        result = torch.ops.aten.constant_pad_nd.default(lsw, pad_spec, 0.0)

        expected_offsets = [
            torch.Size([0]),
            torch.Size([3]),
        ]  # Second shard's offset shifted by 1
        self.assertEqual(result.local_offsets(), expected_offsets)

        result_storage = result.storage_metadata()

        # Global tensor should be: [0, 1, 2, 3, 4, 0] shape=[6]
        expected_global_size = torch.Size([6])
        self.assertEqual(result_storage.size, expected_global_size)

        self.assertEqual(len(result_storage.chunks), 2)

        # First chunk: [3] elements at offset [0] (size increased by top padding)
        # Second chunk: [3] elements at offset [3] (size increased by bottom padding, offset shifted)
        self.assertEqual(result_storage.chunks[0].offsets, torch.Size([0]))
        self.assertEqual(result_storage.chunks[0].sizes, torch.Size([3]))
        self.assertEqual(result_storage.chunks[1].offsets, torch.Size([3]))
        self.assertEqual(result_storage.chunks[1].sizes, torch.Size([3]))

    def test_offsets_and_storage_metadata_after_padding_2d_cw(self) -> None:
        # Test 2D CW sharding with left+right padding
        shard1_2d = torch.tensor([[1.0, 2.0], [5.0, 6.0]])  # [2, 2] columns 0-1
        shard2_2d = torch.tensor([[3.0, 4.0], [7.0, 8.0]])  # [2, 2] columns 2-3
        original_offsets_2d = [(0, 0), (0, 2)]
        lsw_2d = LocalShardsWrapper([shard1_2d, shard2_2d], original_offsets_2d)

        pad_spec_2d = [1, 1, 0, 0]  # [left=1, right=1, top=0, bottom=0]
        result_2d = torch.ops.aten.constant_pad_nd.default(lsw_2d, pad_spec_2d, 0.0)

        expected_offsets_2d = [
            torch.Size([0, 0]),
            torch.Size([0, 3]),
        ]
        self.assertEqual(result_2d.local_offsets(), expected_offsets_2d)

        result_storage_2d = result_2d.storage_metadata()

        # Global tensor should go from [2,4] to [2,6] (add 1 left + 1 right)
        expected_global_size_2d = torch.Size([2, 6])  # [2, 4+1+1]
        self.assertEqual(result_storage_2d.size, expected_global_size_2d)

        # First chunk: [2,3] at offset [0,0] (size increased by left padding)
        # Second chunk: [2,3] at offset [0,3] (size increased by right padding, offset shifted)
        self.assertEqual(result_storage_2d.chunks[0].offsets, torch.Size([0, 0]))
        self.assertEqual(result_storage_2d.chunks[0].sizes, torch.Size([2, 3]))
        self.assertEqual(result_storage_2d.chunks[1].offsets, torch.Size([0, 3]))
        self.assertEqual(result_storage_2d.chunks[1].sizes, torch.Size([2, 3]))


@skip_if_asan_class
class LocalShardsWrapperDistributedTest(MultiProcessTestBase):
    @seed_and_log
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
