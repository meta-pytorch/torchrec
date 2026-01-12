#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import unittest

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)


class TestTableBatchedEmbeddingSlice(unittest.TestCase):
    def test_is_view(self) -> None:
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        emb = DenseTableBatchedEmbeddingBagsCodegen(
            [(2, 4), (2, 4)], use_cpu=device == "cpu"
        )
        first_table = TableBatchedEmbeddingSlice(emb.weights, 0, 8, 2, 4)
        self.assertEqual(first_table.data_ptr(), emb.weights.data_ptr())

    def test_copy(self) -> None:
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        emb = DenseTableBatchedEmbeddingBagsCodegen(
            [(2, 4), (2, 4)], use_cpu=device == "cpu"
        )
        first_table = TableBatchedEmbeddingSlice(emb.weights, 0, 8, 2, 4)
        copied = copy.deepcopy(first_table)
        self.assertNotEqual(first_table.data_ptr(), copied.data_ptr())
