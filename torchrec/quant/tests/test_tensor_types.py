#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest

import torch
from torchrec.tensor_types import UInt2Tensor, UInt4Tensor


class QuantUtilsTest(unittest.TestCase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs available",
    )
    def test_uint42_tensor(self) -> None:
        t_u8 = torch.tensor(
            [
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            ],
            dtype=torch.uint8,
        )
        t_u4 = UInt4Tensor(t_u8)
        t_u4.detach()

        t_u4.to(torch.device("cuda"))
        assert torch.equal(t_u4.view(torch.uint8), t_u8)
        t_u2 = UInt2Tensor(t_u8)
        t_u2.to(torch.device("cuda"))
        assert torch.equal(t_u2.view(torch.uint8), t_u8)

        for t in [t_u4[:, :8], t_u4[:, 8:]]:
            assert t.size(1) == 8
        t_u4[:, :8].copy_(t_u4[:, 8:])

        for t in [t_u2[:, 4:8], t_u2[:, 8:12]]:
            assert t.size(1) == 4

        t_u2[:, 4:8].copy_(t_u2[:, 8:12])
