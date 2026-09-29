#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyre-ignore[21]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


class AssociativeScanTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_complete_cumsum_matches_cpu(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import dispatcher  # noqa: F401

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

        lengths = torch.tensor([3, 0, 7, 1, 16_777_217, 2], dtype=torch.int64)
        actual = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths.to("tpu"))
        expected = torch.cat(
            [torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)]
        )
        torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
