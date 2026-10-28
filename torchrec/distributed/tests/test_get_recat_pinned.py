#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List, Optional

import torch
from pyre_extensions import none_throws
from torchrec.distributed.dist_data import _get_recat, _int_tensor_on


class IntTensorOnTest(unittest.TestCase):
    def test_cpu_and_none_match_torch_tensor(self) -> None:
        for device in [None, torch.device("cpu")]:
            for values in [[], [0, 3, 1, 2], [2**31 - 1, -(2**31)]]:
                expected = torch.tensor(values, device=device, dtype=torch.int32)
                actual = _int_tensor_on(values, device)
                self.assertEqual(actual.dtype, torch.int32)
                self.assertEqual(actual.device, expected.device)
                self.assertFalse(actual.is_pinned())
                self.assertTrue(torch.equal(actual, expected))

    def test_cpu_overflow_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            _int_tensor_on([2**31], torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_torch_tensor(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        for values in [[], [7], [0, 3, 1, 2], list(range(1000)), [2**31 - 1]]:
            expected = torch.tensor(values, device=device, dtype=torch.int32)
            actual = _int_tensor_on(values, device)
            self.assertEqual(actual.dtype, torch.int32)
            self.assertEqual(actual.device, expected.device)
            self.assertTrue(torch.equal(actual, expected))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_staging_buffer_not_reused_before_copy(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        values = list(range(4096))
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
            actual = _int_tensor_on(values, device)
            for _ in range(8):
                torch.full((len(values),), -1, dtype=torch.int32).pin_memory()
        stream.synchronize()
        self.assertTrue(
            torch.equal(actual.cpu(), torch.tensor(values, dtype=torch.int32))
        )


class GetRecatPinnedTest(unittest.TestCase):
    def _assert_cuda_matches_cpu(
        self,
        local_split: int,
        num_splits: int,
        stagger: int,
        batch_size_per_rank: Optional[List[int]],
    ) -> None:
        cpu = none_throws(
            _get_recat(
                local_split=local_split,
                num_splits=num_splits,
                stagger=stagger,
                device=torch.device("cpu"),
                batch_size_per_rank=batch_size_per_rank,
            )
        )
        for cached in [False, True]:
            gpu = none_throws(
                _get_recat(
                    local_split=local_split,
                    num_splits=num_splits,
                    stagger=stagger,
                    device=torch.device("cuda"),
                    batch_size_per_rank=batch_size_per_rank,
                    cached=cached,
                )
            )
            self.assertEqual(gpu.device.type, "cuda")
            self.assertEqual(gpu.dtype, cpu.dtype)
            self.assertTrue(torch.equal(gpu.cpu(), cpu))

    def test_cpu_expected_values(self) -> None:
        self.assertEqual(
            none_throws(_get_recat(2, 4, 1, torch.device("cpu"))).tolist(),
            [0, 2, 4, 6, 1, 3, 5, 7],
        )
        self.assertEqual(
            none_throws(_get_recat(2, 4, 2, torch.device("cpu"))).tolist(),
            [0, 4, 2, 6, 1, 5, 3, 7],
        )
        self.assertIsNone(_get_recat(0, 4, 2, torch.device("cpu")))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_expected_values(self) -> None:
        recat = none_throws(_get_recat(2, 4, 2, torch.device("cuda")))
        self.assertEqual(recat.dtype, torch.int32)
        self.assertEqual(recat.tolist(), [0, 4, 2, 6, 1, 5, 3, 7])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_matches_cpu(self) -> None:
        for stagger in [1, 2]:
            self._assert_cuda_matches_cpu(2, 4, stagger, None)
            self._assert_cuda_matches_cpu(3, 4, stagger, [5, 5, 5, 5])

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_variable_batch_matches_cpu(self) -> None:
        for stagger in [1, 2]:
            self._assert_cuda_matches_cpu(2, 4, stagger, [1, 3, 2, 4])
            self._assert_cuda_matches_cpu(3, 4, stagger, [0, 2, 0, 5])
