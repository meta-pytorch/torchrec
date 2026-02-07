#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# @nolint

import unittest

from torchrec.distributed.test_utils.infer_utils import TorchTypesModelInputWrapper
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.inference.modules import quantize_inference_model, shard_quant_model
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class EagerModelProcessingTests(unittest.TestCase):
    def test_quantize_shard_cuda(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(10)
        ]

        model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=tables,
            )
        )

        quantized_model = quantize_inference_model(model)
        sharded_model, _ = shard_quant_model(quantized_model)

        sharded_qebc = sharded_model._module.sparse.ebc
        self.assertEqual(len(sharded_qebc.tbes), 1)
