#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import unittest

import torch
from torchrec_dynamic_embedding.ps import PS
from utils import register_memory_io


register_memory_io()


class TestPS(unittest.TestCase):
    def testEvictFetch(self):
        cache_ids = [0, 2, 4, 8]
        ids = torch.tensor([[100, 0], [101, 2], [102, 4], [103, 8]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://", 1024)
        ps.evict(ids)
        tensor[:, :] = 0
        ps.fetch(ids, 0).wait()
        torch.testing.assert_close(
            tensor[cache_ids],
            origin_tensor[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )

    def testOS(self):
        cache_ids = [1, 3, 6]
        ids = torch.tensor([[100, 1], [101, 3], [102, 6]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        optim1 = torch.rand((10, 4))
        optim2 = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        origin_optim1 = optim1.clone()
        origin_optim2 = optim2.clone()
        ps = PS("table", [tensor, optim1, optim2], "memory://", 1024)
        ps.evict(ids)
        tensor[:, :] = 0
        optim1[:, :] = 0
        optim2[:, :] = 0
        ps.fetch(ids, 0).wait()
        torch.testing.assert_close(
            tensor[cache_ids],
            origin_tensor[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )
        torch.testing.assert_close(
            optim1[cache_ids],
            origin_optim1[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )
        torch.testing.assert_close(
            optim2[cache_ids],
            origin_optim2[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )

    def testFetchToDifferentCacheID(self):
        cache_ids = [0, 2, 4, 8]
        evict_ids = torch.tensor(
            [[100, 0], [101, 2], [102, 4], [103, 8]], dtype=torch.long
        )
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://", 1024)
        ps.evict(evict_ids)
        tensor[:, :] = 0
        new_cache_ids = [1, 3, 5, 7]
        fetch_ids = torch.tensor(
            [[100, 1], [101, 3], [102, 5], [103, 7]], dtype=torch.long
        )
        ps.fetch(fetch_ids, 0).wait()
        torch.testing.assert_close(
            tensor[new_cache_ids],
            origin_tensor[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )

    def testFetchNonExist(self):
        cache_ids = [0, 2, 4]
        evict_ids = torch.tensor([[100, 0], [101, 2], [102, 4]], dtype=torch.long)
        tensor = torch.rand((10, 4))
        origin_tensor = tensor.clone()
        ps = PS("table", [tensor], "memory://", 1024)
        ps.evict(evict_ids)
        tensor[:, :] = 0
        addition_cache_ids = [3, 9]
        additional_fetch_ids = torch.tensor([[103, 3], [104, 9]], dtype=torch.long)
        ps.fetch(torch.cat([evict_ids, additional_fetch_ids]), 0).wait()
        torch.testing.assert_close(
            tensor[cache_ids],
            origin_tensor[cache_ids],
            rtol=1e-05,
            atol=1e-08,
        )
        torch.testing.assert_close(
            tensor[addition_cache_ids],
            torch.zeros_like(tensor[addition_cache_ids]),
            rtol=1e-05,
            atol=1e-08,
        )


if __name__ == "__main__":
    unittest.main()
