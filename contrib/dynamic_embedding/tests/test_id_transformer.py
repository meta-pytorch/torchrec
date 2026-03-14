#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec_dynamic_embedding.id_transformer import IDTransformer
from torchrec_dynamic_embedding.tensor_list import TensorList


class PythonIdTransformer:
    def __init__(self, num_embedding):
        self._num_embedding = num_embedding
        self.dict = {}
        self.overflow = False

    def transform(self, global_ids: torch.Tensor):
        global_id_list = global_ids.flatten().tolist()
        cache_id_list = [0] * len(global_id_list)
        for i in range(len(global_id_list)):
            gid = global_id_list[i]
            if gid in self.dict:
                cid = self.dict[gid]
            else:
                cid = len(self.dict)
                self.dict[gid] = cid
                if len(self.dict) >= self._num_embedding:
                    self.overflow = True
            cache_id_list[i] = cid
        cache_ids = torch.tensor(cache_id_list, dtype=torch.long)
        return cache_ids


class TestIDTransformer(unittest.TestCase):
    def testSkeleton(self):
        self.assertIsNotNone(IDTransformer(1024))

    def testTransform(self):
        num_embedding = 1024
        shape = (1024,)
        transformer = IDTransformer(
            num_embedding,
            transform_config={
                "type": "naive",
            },
        )
        python_transformer = PythonIdTransformer(num_embedding)
        global_ids = torch.empty(shape, dtype=torch.int64)

        for timestamp in range(10):
            global_ids.random_(0, 512)
            cache_ids = torch.empty_like(global_ids)

            result = transformer.transform(
                TensorList([global_ids]), TensorList([cache_ids]), timestamp
            )
            success, ids_to_fetch = result.success, result.ids_to_fetch
            self.assertTrue(success)

            python_cache_ids = python_transformer.transform(global_ids)
            self.assertTrue(torch.all(cache_ids == python_cache_ids).item())

            if ids_to_fetch is not None:
                ids_map = {
                    global_id: cache_id for global_id, cache_id in ids_to_fetch.tolist()
                }
                for global_id, cache_id in zip(global_ids.tolist(), cache_ids.tolist()):
                    if global_id in ids_map:
                        self.assertEqual(cache_id, ids_map[global_id])

    def testEvict(self):
        num_embedding = 9
        transformer = IDTransformer(
            num_embedding,
            transform_config={
                "type": "naive",
            },
        )
        global_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        cache_ids = torch.empty_like(global_ids)
        result = transformer.transform(
            TensorList([global_ids]), TensorList([cache_ids]), 0
        )
        self.assertTrue(result.success)

        global_ids = torch.tensor([1, 3, 5, 7], dtype=torch.long)
        result = transformer.transform(
            TensorList([global_ids]), TensorList([cache_ids]), 1
        )
        self.assertTrue(result.success)

        num_to_evict = 2
        evicted_tensor = transformer.evict(num_to_evict)
        self.assertEqual(num_to_evict, evicted_tensor.shape[0])
        evicted_ids = sorted(evicted_tensor.tolist())
        self.assertEqual(evicted_ids, [[2, 1], [4, 3]])

    def testAll(self):
        num_embedding = 9
        transformer = IDTransformer(
            num_embedding,
            transform_config={
                "type": "naive",
            },
        )
        global_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        cache_ids = torch.empty_like(global_ids)
        result = transformer.transform(
            TensorList([global_ids]), TensorList([cache_ids]), 0
        )
        self.assertTrue(result.success)

        id_pairs = transformer.save().tolist()
        self.assertEqual(len(id_pairs), 4)
        id_dict = {1: 0, 2: 1, 3: 2, 4: 3}
        for global_id, cache_id in id_pairs:
            self.assertTrue(global_id in id_dict)
            self.assertEqual(cache_id, id_dict[global_id])

        global_ids = torch.tensor([1, 3, 5, 7], dtype=torch.long)
        result = transformer.transform(
            TensorList([global_ids]), TensorList([cache_ids]), 1
        )
        self.assertTrue(result.success)

        id_pairs = transformer.save().tolist()
        self.assertEqual(len(id_pairs), 4)
        id_dict = {1: 0, 3: 2, 5: 4, 7: 5}
        for global_id, cache_id in id_pairs:
            self.assertTrue(global_id in id_dict)
            self.assertEqual(cache_id, id_dict[global_id])
