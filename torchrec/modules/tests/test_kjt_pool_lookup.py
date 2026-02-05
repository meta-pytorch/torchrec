#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
from torchrec.modules.object_pool_lookups import UVMCachingInt64Lookup
from torchrec.sparse.jagged_tensor import JaggedTensor


class KeyedJaggedTensorPoolLookupTest(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_uvm_caching_int64_lookup(
        self,
    ) -> None:
        device = torch.device("cuda:0")

        pool_size, feature_max_lengths = 4, {"f1": 2, "f2": 4}
        lookup = UVMCachingInt64Lookup(
            pool_size=pool_size,
            feature_max_lengths=feature_max_lengths,
            is_weighted=False,
            device=device,
        )
        ids = torch.tensor([0, 1, 2, 3], device=device)
        jt_values = torch.tensor(
            [1, 3, 3, 2, 2, 4, 11, 13, 13, 13, 12, 12, 14, 14, 14, 14],
            dtype=torch.int64,
            device=device,
        )
        jt_lengths = torch.tensor(
            [1, 2, 2, 1, 1, 3, 2, 4], dtype=torch.int, device=device
        )

        lookup.update(
            ids=ids,
            values=JaggedTensor(
                jt_values,
                lengths=jt_lengths,
            ),
        )

        torch.testing.assert_close(lookup.lookup(ids).values(), jt_values)

        INT64_VALUE_SHIFT = int(3e9)
        lookup.update(
            ids=ids,
            values=JaggedTensor(
                jt_values + INT64_VALUE_SHIFT,
                lengths=jt_lengths,
            ),
        )

        torch.testing.assert_close(
            lookup.lookup(ids).values(), jt_values + INT64_VALUE_SHIFT
        )
